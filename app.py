"""Flask app for NWS ingestion, embedding generation, and semantic search."""

import hashlib
import json
import logging
import os

from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

import lakebase
from weather_client import WeatherClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-semantic-search")

app = Flask(__name__)
WEATHER_TABLE = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE_NAME", "weather_embeddings")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search_page():
    return render_template("weather_search.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    lakebase.init_weather_tables()
    body = request.get_json()
    locations = body.get("locations") or []
    limit = max(1, min(int(body.get("limit", 50)), 100))
    if not locations:
        return jsonify({"error": "locations must be a non-empty list"}), 400

    client = WeatherClient()
    synced = 0
    processed = []

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for location in locations:
                try:
                    name = location["name"]
                    lat = float(location["lat"])
                    lon = float(location["lon"])
                except (KeyError, TypeError, ValueError):
                    continue

                documents = client.get_weather_documents(name, lat, lon, limit)
                for document in documents:
                    cur.execute(
                        f"""
                        INSERT INTO {WEATHER_TABLE} (
                            id, location, source_type, headline, narrative_text,
                            issued_at, effective_at, payload, synced_at
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                        ON CONFLICT (id) DO UPDATE SET
                            location=EXCLUDED.location,
                            source_type=EXCLUDED.source_type,
                            headline=EXCLUDED.headline,
                            narrative_text=EXCLUDED.narrative_text,
                            issued_at=EXCLUDED.issued_at,
                            effective_at=EXCLUDED.effective_at,
                            payload=EXCLUDED.payload,
                            synced_at=NOW()
                        """,
                        (
                            document["id"], document["location"],
                            document["source_type"], document.get("headline"),
                            document["narrative_text"], document.get("issued_at"),
                            document.get("effective_at"), json.dumps(document["payload"]),
                        ),
                    )
                    synced += 1
                processed.append(name)
            conn.commit()

    if not processed:
        return jsonify({"error": "No valid locations supplied"}), 400
    return jsonify({"synced": synced, "locations": processed})


@app.route("/weather/generate-embeddings", methods=["POST"])
def generate_embeddings():
    lakebase.init_weather_tables()
    rows = lakebase.run_query(
        f"""
        SELECT d.id, d.narrative_text
        FROM {WEATHER_TABLE} d
        LEFT JOIN {EMBEDDINGS_TABLE} e ON d.id = e.document_id
        WHERE e.document_id IS NULL
        ORDER BY d.synced_at
        """
    )
    if not rows:
        return jsonify({"embedded": 0, "message": "All documents already have embeddings"})

    model = get_model()
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for row in rows:
                vector = model.encode(row["narrative_text"])
                vector_text = "[" + ",".join(str(float(x)) for x in vector) + "]"
                embedding_id = hashlib.sha256(f"{row['id']}:0".encode()).hexdigest()
                cur.execute(
                    f"""
                    INSERT INTO {EMBEDDINGS_TABLE}
                        (id, document_id, chunk_index, chunk_text, embedding, model_name)
                    VALUES (%s,%s,0,%s,%s::vector,%s)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                        chunk_text=EXCLUDED.chunk_text,
                        embedding=EXCLUDED.embedding,
                        model_name=EXCLUDED.model_name,
                        created_at=NOW()
                    """,
                    (embedding_id, row["id"], row["narrative_text"], vector_text, MODEL_NAME),
                )
            conn.commit()
    return jsonify({"embedded": len(rows)})


@app.route("/weather/search", methods=["POST"])
def search_weather():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    body = request.get_json()
    query = str(body.get("query", "")).strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    top_k = max(1, min(int(body.get("top_k", 5)), 20))

    vector = get_model().encode(query)
    vector_text = "[" + ",".join(str(float(x)) for x in vector) + "]"
    results = lakebase.run_query(
        f"""
        SELECT d.location, d.source_type, d.headline,
               e.chunk_index, e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {WEATHER_TABLE} d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        (vector_text, vector_text, top_k),
    )
    return jsonify({"query": query, "top_k": top_k, "results": results})


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_RUN_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_RUN_PORT", 8000)),
        debug=True,
    )

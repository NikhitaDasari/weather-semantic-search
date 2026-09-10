"""Batch pipeline: weather_documents -> chunks -> Sentence Transformer embeddings -> pgvector."""

import hashlib

from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

import lakebase

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 32


def chunk_text(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_SIZE >= len(text):
            break
    return chunks


def main() -> None:
    lakebase.init_weather_tables()
    documents = lakebase.run_query(
        """
        SELECT d.id, d.narrative_text
        FROM weather_documents d
        WHERE d.narrative_text IS NOT NULL
          AND TRIM(d.narrative_text) <> ''
          AND NOT EXISTS (
              SELECT 1 FROM weather_embeddings e
              WHERE e.document_id = d.id
          )
        ORDER BY d.synced_at
        """
    )

    if not documents:
        print("No new weather documents to embed.")
        return

    chunks = []
    for document in documents:
        for index, text in enumerate(chunk_text(document["narrative_text"])):
            chunks.append({
                "id": hashlib.sha256(f"{document['id']}:{index}".encode()).hexdigest(),
                "document_id": document["id"],
                "chunk_index": index,
                "chunk_text": text,
            })

    model = SentenceTransformer(MODEL_NAME)
    rows = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vectors = model.encode([item["chunk_text"] for item in batch])
        for item, vector in zip(batch, vectors):
            vector_text = "[" + ",".join(str(float(x)) for x in vector) + "]"
            rows.append((
                item["id"], item["document_id"], item["chunk_index"],
                item["chunk_text"], vector_text, MODEL_NAME,
            ))

    sql = """
        INSERT INTO weather_embeddings
            (id, document_id, chunk_index, chunk_text, embedding, model_name)
        VALUES %s
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = NOW()
    """
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur, sql, rows,
                template="(%s,%s,%s,%s,%s::vector,%s)",
                page_size=100,
            )
            conn.commit()

    print(f"Embedded {len(rows)} chunks from {len(documents)} weather documents.")


if __name__ == "__main__":
    main()

# Weather Intelligence & Semantic Retrieval Platform

A Databricks-based AI data engineering project that turns narrative weather data into searchable vector representations and exposes natural-language semantic retrieval through a user-facing web application and REST API.

The project demonstrates the data layer behind retrieval-augmented AI systems: **API ingestion → normalized operational data → text chunking → vector embeddings → pgvector similarity search → application/API retrieval**.

**Tech:** Databricks Apps · Lakebase/PostgreSQL · pgvector · Sentence Transformers · Python · Flask · National Weather Service API

---

## What this project demonstrates

- Ingesting unstructured forecast and alert narratives from an external REST API
- Normalizing heterogeneous API responses into a common document model
- Persisting operational data in Databricks Lakebase/PostgreSQL
- Chunking long-form text with overlap to preserve retrieval context
- Generating 384-dimensional embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Storing vectors in PostgreSQL with pgvector
- Building HNSW-indexed cosine-similarity search
- Exposing semantic retrieval through Flask REST endpoints and a browser UI
- Implementing stable document IDs, upserts, and idempotent embedding ingestion
- Managing database connectivity through Databricks secrets rather than source code

---

## Architecture

```text
National Weather Service API
            |
            v
     weather_client.py
            |
            v
      POST /weather/sync
            |
            v
   +---------------------+
   | weather_documents   |
   | Databricks Lakebase |
   +----------+----------+
              |
              v
        text chunking
      800 chars / 100 overlap
              |
              v
   Sentence Transformers
      all-MiniLM-L6-v2
              |
              v
   +---------------------+
   | weather_embeddings  |
   | pgvector VECTOR(384)|
   | HNSW cosine index   |
   +----------+----------+
              |
              v
      POST /weather/search
              |
              v
   Natural-language ranked results
              |
              v
        Flask Web UI
```

---

## Data pipeline

### 1. Weather ingestion

`weather_client.py` integrates with the **National Weather Service API** and retrieves multi-day forecast periods and active weather alerts. Forecast narratives, alert descriptions, and recommended actions are normalized into a shared document structure before being persisted in Lakebase.

### 2. Operational storage

`weather_documents` stores normalized source data and retains the original API payload for provenance and reprocessing.

```text
id
location
source_type
headline
narrative_text
issued_at
effective_at
payload
synced_at
```

Stable IDs and PostgreSQL upserts make repeated synchronization safe without creating duplicate documents.

### 3. Chunking and embeddings

The embedding pipeline splits narrative text using a sliding window:

```text
Chunk size:    800 characters
Chunk overlap: 100 characters
```

Each chunk is embedded with:

```text
sentence-transformers/all-MiniLM-L6-v2
Vector dimension: 384
```

Vectors are persisted in `weather_embeddings` using PostgreSQL's **pgvector** extension. The project also creates an HNSW index using cosine-vector operations for similarity search.

### 4. Semantic retrieval

Natural-language queries are embedded using the same Sentence Transformer model and compared with stored vectors using pgvector cosine distance.

Example:

```text
flash flood risk this weekend
```

The API returns the most semantically relevant forecast or alert passages with location, source type, headline, and similarity score.

---

## Application endpoints

### Sync weather data

```http
POST /weather/sync
```

Example request:

```json
{
  "locations": [
    {
      "name": "Dallas, TX",
      "lat": 32.7767,
      "lon": -96.7970
    }
  ],
  "limit": 20
}
```

### Generate embeddings

```http
POST /weather/generate-embeddings
```

### Semantic search

```http
POST /weather/search
```

Example request:

```json
{
  "query": "chance of severe thunderstorms",
  "top_k": 5
}
```

### Health check

```http
GET /healthz
```

---

## Browser UI

The Databricks App includes two user-facing views:

- **Weather Sync** — load forecasts and active alerts for a supplied location
- **Semantic Search** — ask natural-language weather questions and view ranked vector-search results

---

## Repository structure

```text
.
├── app.py
├── weather_client.py
├── lakebase.py
├── app.yaml
├── requirements.txt
├── setup_secrets.py
├── templates/
│   ├── index.html
│   └── weather_search.html
└── notebooks/
    └── ingest_weather_embeddings.ipynb
```

---

## Background

I built this project while expanding my enterprise data-engineering experience into **AI data engineering**, with a focus on the data and retrieval architecture required by LLM applications: API ingestion, unstructured-data processing, embeddings, vector search, operational storage, and application integration.

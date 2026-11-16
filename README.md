# Enterprise Document Intelligence & RAG API
*(Production-Ready Multi-Tenant Knowledge Base Microservice)*

A high-performance, multi-tenant document intelligence API built with **FastAPI**, **PostgreSQL + pgvector**, **Hybrid Search (Dense + BM25)**, **Cross-Encoder Reranking**, **Strict Citation Verification**, and **Server-Sent Events (SSE)** streaming.

Designed for companies, law firms, clinics, and SaaS platforms that need tamper-proof AI answers that **cite exact page numbers, filenames, and verbatim quotes** with zero hallucinations.

---

## ⚡ The 4 Core Architectural Capabilities

1. **Strict Citation Engine**:
   - Returns the exact `page_number`, `source_filename`, and `exact_quote`.
   - Built-in post-validation string containment check verifies quotes exist verbatim in the source text.
   - If information cannot be found, it reliably returns `"I cannot find this information in the uploaded documents."` with `confidence: "low"` rather than fabricating facts.
2. **Hybrid Search (pgvector + BM25)**:
   - Combines dense semantic vector search with PostgreSQL `tsvector` full-text search using **Reciprocal Rank Fusion (RRF)**. Handles product IDs, invoice numbers, dates, and semantic intent simultaneously.
3. **Pydantic v2 Structured Output**:
   - Guarantees strict, machine-readable JSON responses ready for frontend dashboards, n8n, Zapier, or customer CRM workflows.
4. **FastAPI Streaming Endpoint (SSE)**:
   - Token-by-token streaming over HTTP Server-Sent Events (`/query/stream`) with low latency, followed by a final citation verification payload.

---

## 🚀 One-Click Deployment on Render

This repository includes a production-ready `render.yaml` Blueprint file that deploys all 4 components in a single click:
1. **Web Service**: FastAPI ASGI application (`rag-api`)
2. **Background Worker**: Celery task runner (`rag-worker`)
3. **Managed Database**: PostgreSQL 16 with `pgvector`
4. **Managed Cache**: Redis 7

### Deploy Steps:
1. Fork or push this repository to GitHub/GitLab.
2. Go to your [Render Dashboard](https://dashboard.render.com/) and click **New > Blueprint**.
3. Connect your repository. Render automatically reads `render.yaml` and provisions the Web Service, Celery Worker, PostgreSQL, and Redis.
4. (Optional) In the Render dashboard, add your `OPENAI_API_KEY` (or `GEMINI_API_KEY`) under Environment Variables.
5. Hit **Apply** — your API is live!

---

## 💻 Local Development Setup

### Option 1: Docker Compose (Recommended)
Spin up the entire stack (PostgreSQL + pgvector, Redis, FastAPI, and Celery Worker) with one command:

```bash
docker compose up --build
```
The API will be live at `http://localhost:8000`. Interactive Swagger documentation: `http://localhost:8000/docs`.

### Option 2: Local Python Environment
```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Start the FastAPI server
uvicorn app.main:app --reload --port 8000
```

> **Smart Fallback**: If Celery/Redis is not running locally, the API seamlessly processes document parsing and vectorization in the background using FastAPI's built-in asynchronous worker.

---

## 📡 API Reference & cURL Examples

### 1. Create a Tenant & Issue API Key (Admin)
```bash
curl -X POST "http://localhost:8000/api/v1/tenants" \
     -H "Content-Type: application/json" \
     -H "X-Master-Key: enterprise-super-secret-admin-key" \
     -d '{"name": "Acme Legal Partners"}'
```
**Response:**
```json
{
  "id": "e81d4fae-832d-4c3d-b2a1-fa4431d10291",
  "name": "Acme Legal Partners",
  "api_key": "acme_leg_xK81nBq29Pz_...",
  "created_at": "2026-11-16T10:30:00Z"
}
```

---

### 2. Upload a Document (PDF, DOCX, TXT)
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "X-API-Key: YOUR_TENANT_API_KEY" \
     -F "file=@sample_contract.pdf"
```
**Response:**
```json
{
  "id": "7b8e5c3a-23d4-4a6f-998f-0a12b3c4d5e6",
  "filename": "sample_contract.pdf",
  "file_size_bytes": 1048576,
  "status": "processing",
  "total_pages": 0,
  "created_at": "2026-11-16T10:31:00Z"
}
```

---

### 3. Query Documents (Strict JSON Response)
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_TENANT_API_KEY" \
     -d '{
       "query": "What is the penalty if the contract is terminated within 30 days?",
       "top_k": 3
     }'
```
**Response:**
```json
{
  "answer": "If terminated within the first 30 days of execution, a cancellation fee equal to 15% of the total contract value shall apply.",
  "confidence": "high",
  "citations": [
    {
      "document_id": "7b8e5c3a-23d4-4a6f-998f-0a12b3c4d5e6",
      "source_filename": "sample_contract.pdf",
      "page_number": 4,
      "exact_quote": "A cancellation fee equal to 15% of the total contract value shall apply if terminated within thirty (30) days."
    }
  ],
  "information_found": true
}
```

---

### 4. Query with Real-time SSE Token Streaming
```bash
curl -N -X POST "http://localhost:8000/api/v1/query/stream" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_TENANT_API_KEY" \
     -d '{
       "query": "What are the payment terms?",
       "top_k": 3
     }'
```
**SSE Stream Output:**
```text
event: message
data: {"type": "token", "content": "Payment "}

event: message
data: {"type": "token", "content": "is due "}

event: message
data: {"type": "token", "content": "within 30 days of invoice receipt."}

event: message
data: {"type": "final", "confidence": "high", "citations": [{"document_id": "...", "source_filename": "sample_contract.pdf", "page_number": 2, "exact_quote": "Invoices are payable net 30 days from date of receipt."}], "information_found": true}
```

---

## 🛡️ Zero-Cost Embedding & Reranking Defaults

| Task | Default Provider (0 Cost / Self-Hosted) | Cloud Alternative (Configurable in `.env`) |
| :--- | :--- | :--- |
| **Embeddings** | `FastEmbed` (`BAAI/bge-small-en-v1.5`, 384-dim, ONNX CPU) | OpenAI `text-embedding-3-small` (1536-dim) |
| **Reranker** | `FlashRank` (`ms-marco-TinyBERT-L-2-v2`, CPU) | Cohere Rerank API (`rerank-english-v3.0`) |
| **LLM Output** | Configurable via `OPENAI_API_KEY` or `GEMINI_API_KEY` | Zero-shot passage synthesis fallback |

---

## 📂 Project Architecture

```
rag-api/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── documents.py      # Upload, list, delete, status endpoints
│   │       ├── query.py          # Synchronous JSON & SSE streaming endpoints
│   │       └── tenants.py        # Multi-tenant API key onboarding
│   ├── core/
│   │   ├── config.py             # Pydantic Settings & environment variables
│   │   ├── logging.py            # Loguru structured logger
│   │   └── security.py           # API key hashing & tenant resolution
│   ├── db/
│   │   ├── base.py               # SQLAlchemy DeclarativeBase
│   │   ├── models.py             # Tenant, Document, DocumentChunk schemas
│   │   └── session.py            # Async (FastAPI) and Sync (Celery) DB engines
│   ├── services/
│   │   ├── chunker.py            # Page-preserving recursive character splitter
│   │   ├── embeddings.py         # FastEmbed (0-cost default) + OpenAI
│   │   ├── hybrid_search.py      # pgvector dense + BM25 tsvector + RRF
│   │   ├── parser.py             # PyMuPDF page-by-page text extractor
│   │   ├── rag_engine.py         # Strict citation prompt + quote verification
│   │   └── reranker.py           # FlashRank CPU reranker + Cohere
│   ├── workers/
│   │   ├── celery_app.py         # Celery instance & broker configuration
│   │   └── tasks.py              # Ingestion background task
│   └── main.py                   # FastAPI entrypoint, CORS, Lifespan, /health
├── Dockerfile                    # Multi-stage production container
├── docker-compose.yml            # Local development cluster
├── render.yaml                   # 1-Click Render Blueprint infrastructure
├── requirements.txt              # Production Python dependencies
└── .env.example                  # Environment configuration template
```

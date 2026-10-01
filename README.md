# Multi-Tenant RAG

Enterprise-grade **multi-tenant Retrieval-Augmented Generation (RAG)** platform. Ingest documents into isolated per-tenant vector stores, retrieve with a hybrid pipeline (dense + BM25 + reranking), and answer questions with a guarded LLM — all while keeping tenant data strictly separated.

## Features

### 1. Multi-Tenant Isolation
- Every tenant gets its own **ChromaDB collection** (`tenant_<uuid>`), so vectors from one tenant can never leak into another's results.
- PostgreSQL-backed metadata tracks tenants, users, documents, versions, and chunks.
- Schema migrations managed with **Alembic**.

### 2. Hybrid Retrieval
- **Dense retrieval** via OpenAI `text-embedding-3-small` embeddings in ChromaDB.
- **Lexical retrieval** via a live in-memory **BM25** index, rebuilt when the chunk count changes (so re-ingestion never serves a stale index).
- **Reciprocal Rank Fusion (RRF)** merges the two ranked lists with configurable weights. Dense and BM25 share the same per-chunk ID namespace, so hits on the same chunk from either retriever combine instead of collapsing.
- **Reranking** re-sorts the fused candidates before the LLM sees them.

### 3. Structured Chunking
- Chunks carry **page numbers**, **section paths**, and **character offsets** sourced from the parser, so answers can be cited back to a document and page.
- The PDF loader preserves page boundaries with internal markers that the chunker consumes and strips before any text reaches a user or an LLM.
- A **relevance threshold** (`RAG_RELEVANCE_THRESHOLD`) refuses to answer from weak context instead of fabricating.

### 4. Query Translation
Seven strategies to defeat single-vector blindness:
- `compress`, `expand`, `rephrase`, `multi_query`, `decompose`, `step_back`, `hyde`

Each runs through a pluggable `TranslationModel` interface (Ollama HTTP provider included; a deterministic rule-based model for tests), validates the structured JSON output, and returns typed `QueryVariant` objects. The live query path calls the service when an Ollama host is reachable and otherwise falls back to deterministic compound-query splitting.

### 5. Security Guardrails
- **Prompt-injection detection** rejects malicious queries before retrieval.
- **PII redaction** strips emails, phones, and other sensitive tokens from both the query and the generated answer.
- Input validation with clear rejection messages.

### 6. Inference Gateway
A single `LLMGateway` abstracts the model behind one interface, supporting:
- **OpenAI** (default, e.g. `gpt-4o-mini`)
- **vLLM** (OpenAI-compatible endpoint, with a real `/health` probe before use so a dead vLLM actually falls back)

Provider selection is runtime-configured; the rest of the pipeline is provider-agnostic.

### 7. Evaluation & Harness
- **Retrieval metrics**: Hit@K, Recall@K, MRR — evaluated against labeled JSONL datasets (`evals/rag_eval.jsonl`).
- **Harness evaluation** scores the final answer (faithfulness, relevance) via an LLM judge. The judge fails loudly on malformed output instead of returning inflated defaults.
- `scripts/run_retrieval_eval.py` runs the full retrieval benchmark across `dense`, `hybrid`, and `rerank` modes.

### 8. Streamlit UI
- `app.py` is the entry point with explicit navigation.
- `pages/login.py` — Google OAuth authentication.
- `pages/rag.py` — document upload, tenant-scoped chat, source previews (with page numbers), and latency breakdown.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  ingestion/       │────▶│  ChromaDB    │
│  UI (login,  │     │  parsing, chunk-  │     │  per-tenant  │
│   rag pages) │     │  ing, embed, up-  │     │  collections │
└─────────────┘     │ sert              │     └──────────────┘
                     └────────┬─────────┘
                              │
┌─────────────┐     ┌──────────▼─────────┐     ┌──────────────┐
│  User Query  │────▶│  query/             │────▶│  LLM Gateway │
│              │     │  translate → RRF    │     │  OpenAI/vLLM │
└─────────────┘     │  → rerank → answer  │     └──────────────┘
                     └─────────────────────┘
```

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Copy and edit environment variables
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, DATABASE_URL, etc.

# 3. Set up the database
alembic upgrade head

# 4. Run the Streamlit app
streamlit run app.py

# 5. (Optional) Run retrieval evaluation
python scripts/run_retrieval_eval.py evals/rag_eval.jsonl --k 1 2 3 4 5 6 --mode hybrid
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | yes | OpenAI API key for embeddings and LLM inference |
| `DATABASE_URL` | yes | PostgreSQL connection URL (`postgresql+psycopg://user:pass@host:port/db`) |
| `CHROMA_PATH` | no | ChromaDB persistence directory (default `./chroma_db`) |
| `LANGCHAIN_API_KEY` | no | LangSmith tracing key |
| `LANGCHAIN_PROJECT` | no | LangSmith project name |
| `VLLM_ENABLED` | no | Set `true` to use a vLLM endpoint |
| `VLLM_BASE_URL` | no | vLLM base URL (default `http://localhost:8000/v1`) |
| `VLLM_MODEL` | no | vLLM model name |
| `OLLAMA_TRANSLATION_MODEL` | no | Ollama model for query translation |
| `OLLAMA_TRANSLATION_BASE_URL` | no | Ollama base URL (default `http://localhost:11434`) |

## Project layout

```
src/
  ingestion/     document parsing (PDF/HTML/TXT/Office), chunking,
                 embedding, and Chroma upsert per tenant
  query/         hybrid retrieval, RRF fusion, reranking,
                 query translation, and the main RAG orchestrator
  security/      prompt-injection guardrails and PII redaction
  inference/     LLM gateway (OpenAI / vLLM)
  db/            SQLAlchemy models and session factory
  harness/       end-to-end answer evaluation harness
  evaluation/    retrieval metrics (Hit@K, Recall@K, MRR)
  auth/          user authentication and provisioning
  prompts/       prompt template loading (LangSmith-backed with fallback)
tests/           focused unit tests for translation, router, security,
                 and retrieval evaluation
scripts/         eval runner and chunk-catalog exporter
alembic/         database migrations
pages/           Streamlit pages (login, rag)
docs/            sample documents
evals/           labeled retrieval evaluation dataset
```

## Performance

Measured on the bundled `small_letter.pdf` evaluation set (10 labeled queries, hybrid mode):

| Metric | Value |
|---|---|
| Hit@6 | 1.000 |
| Recall@6 | 1.000 |
| MRR | 1.000 |
| Mean retrieval latency | 3.0 ms |

## Testing

```bash
pytest
```

## Roadmap / known gaps

This is a reference implementation, not a production-hardened service. Honest gaps versus the leaders:

- **Document parsing**: pypdf text only. No OCR, no table extraction, no layout awareness. Add Docling or DeepDoc for tables and scanned PDFs.
- **Reranking**: lexical heuristic, not a learned cross-encoder. Swap in `bge-reranker`, Cohere, or Jina.
- **Retrieval store**: Chroma + in-memory BM25. A native hybrid store (OpenSearch, Qdrant, or pgvector + Postgres full-text) would remove the BM25 rebuild and the Chroma/Postgres dual-write.
- **Tenancy and access**: one collection per tenant; every user currently gets their own tenant and only a `MEMBER`/`OWNER` role exists. Add org-level tenants, roles, and per-document ACLs enforced at query time.
- **Integration**: Streamlit only. A FastAPI layer, MCP server, and connectors are the highest-value additions.
- **Answers**: citations are now requested in the prompt and page numbers are surfaced in sources, but there is no chat history, no follow-up rewriting, and no streaming.
- **Advanced retrieval**: graph retrieval and agentic multi-step research are not on the live path.
- **Evaluation**: 10 questions on a one-page letter. Build larger test sets including unanswerable and multi-document questions, and add regression checks in CI.
- **Operations**: no Docker, no CI, synchronous ingestion. Add containers, background workers (the `OutboxEvent` table is wired for this), and tracing.
- **Security**: the injection filter is regex-based and covers the query only. Retrieved text should be delimited in the prompt and the model instructed to treat it as data. PII redaction should become a per-tenant policy. Add rate limits and real audit logs.

## License

Add your license choice here.
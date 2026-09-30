# Multi-Tenant RAG

Enterprise-grade **multi-tenant Retrieval-Augmented Generation (RAG)** platform. Ingest documents into isolated per-tenant vector stores, retrieve with a hybrid pipeline (dense + BM25 + cross-encoder reranking), and answer questions with a guarded LLM — all while keeping tenant data strictly separated.

## Features

### 1. Multi-Tenant Isolation
- Every tenant gets its own **ChromaDB collection** (`tenant_<uuid>`), so vectors from one tenant can never leak into another's results.
- PostgreSQL-backed metadata tracks tenants, users, documents, versions, and chunks.
- Schema migrations managed with **Alembic**.

### 2. Hybrid Retrieval
- **Dense retrieval** via OpenAI `text-embedding-3-small` embeddings in ChromaDB.
- **Lexical retrieval** via a live in-memory **BM25** index, rebuilt when the chunk count changes (so re-ingestion never serves a stale index).
- **Reciprocal Rank Fusion (RRF)** merges the two ranked lists with configurable weights.
- **Cross-encoder reranking** (exact-phrase, term-coverage, density, and positional scores) re-sorts the fused candidates before the LLM sees them.
- A **dense-hit preservation guard** guarantees the top semantic match is never dropped from the final context — fixing cross-encoder blindness on short queries whose answer lives in a chunk with different vocabulary.

### 3. Query Translation
Seven strategies to defeat single-vector blindness:
- `compress`, `expand`, `rephrase`, `multi_query`, `decompose`, `step_back`, `hyde`

Each runs through a pluggable `TranslationModel` interface (Ollama HTTP provider included; a deterministic rule-based model for tests), validates the structured JSON output, and returns typed `QueryVariant` objects.

### 4. Security Guardrails
- **Prompt-injection detection** rejects malicious queries before retrieval.
- **PII redaction** strips emails, phones, and other sensitive tokens from both the query and the generated answer.
- Input validation with clear rejection messages.

### 5. Inference Gateway
A single `LLMGateway` abstracts the model behind one interface, supporting:
- **OpenAI** (default, e.g. `gpt-4o-mini`)
- **vLLM** (OpenAI-compatible endpoint)

Provider selection is runtime-configured; the rest of the pipeline is provider-agnostic.

### 6. Evaluation & Harness
- **Retrieval metrics**: Hit@K, Recall@K, MRR — evaluated against labeled JSONL datasets (`evals/rag_eval.jsonl`).
- **Harness evaluation** scores the final answer (groundedness, correctness) end-to-end.
- `scripts/run_retrieval_eval.py` runs the full retrieval benchmark across `dense`, `hybrid`, and `rerank` modes.

### 7. Streamlit UI
- `app.py` is the entry point with explicit navigation.
- `pages/login.py` — Google OAuth authentication.
- `pages/rag.py` — document upload, tenant-scoped chat, source previews, and latency breakdown.

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

## Project layout

```
src/
  ingestion/     document parsing (PDF/HTML/TXT/Office), chunking,
                 embedding, and Chroma upsert per tenant
  query/         hybrid retrieval, RRF fusion, cross-encoder reranking,
                 query translation, and the main RAG orchestrator
  security/      prompt-injection guardrails and PII redaction
  inference/     LLM gateway (OpenAI / vLLM)
  db/            SQLAlchemy models and session factory
  harness/       end-to-end answer evaluation harness
  evaluation/    retrieval metrics (Hit@K, Recall@K, MRR)
  auth/          user authentication and provisioning
  prompts/       prompt template loading (LangSmith-backed with fallback)
tests/             focused unit tests for translation, router, security,
                   and retrieval evaluation
scripts/           eval runner and chunk-catalog exporter
alembic/           database migrations
pages/             Streamlit pages (login, rag)
docs/              sample documents
evals/             labeled retrieval evaluation dataset
```

## Performance

Measured on the bundled `small_letter.pdf` evaluation set (10 labeled queries, hybrid mode):

| Metric | Value |
|---|---|
| Hit@6 | 1.000 |
| Recall@6 | 1.000 |
| MRR | 1.000 |
| Mean retrieval latency | 5.4 ms |

## Testing

```bash
pytest
```

## License

Add your license choice here.
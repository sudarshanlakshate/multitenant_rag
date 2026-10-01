from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.inference.llm_gateway import llm_gateway
from src.ingestion.processor import _get_chroma_collection
from src.prompts.load_prompts import load_rag_prompt
from src.query.query_router_rrf import (
    BM25Retriever,
    Document as RouterDocument,
    SearchResult,
    reciprocal_rank_fusion,
)
from src.query.query_translation import (
    QueryTranslationService,
    QueryTranslationStrategy,
)
from src.query.reranker import reranker
from src.security.sanitizer import redact_pii, validate_input_query

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional assistant for enterprise question-answering tasks. "
            "Use the provided context to answer the question accurately and concisely. "
            "If the provided context does not contain enough information to answer, state clearly "
            "that the documents do not provide this information. Do not make up facts.\n\n"
            "Each context block is labeled [n] and carries a source, page, and section. "
            "When a claim is supported by a block, cite it inline as [n]. "
            "If the answer combines several blocks, cite each one. "
            "If no block supports a claim, do not make the claim.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

_TENANT_BM25_CACHE: dict[tuple[str, int], tuple[float, BM25Retriever]] = {}


def _purge_stale_bm25_cache() -> None:
    """Drop cache entries keyed by the old tenant-id-only format.

    A one-time migration so the new (tenant_id, chunk_count) keys are the
    only shape in the dict. Safe to call at import time.
    """
    stale = [key for key in list(_TENANT_BM25_CACHE) if not isinstance(key, tuple)]
    for key in stale:
        _TENANT_BM25_CACHE.pop(key, None)


_purge_stale_bm25_cache()


def _get_prompt():
    """Load prompt from LangSmith with robust fallback."""
    try:
        return load_rag_prompt()
    except Exception as exc:
        logger.debug("Using default prompt template (%s)", exc)
        return DEFAULT_PROMPT


def _get_tenant_bm25(tenant_id: str, chroma: Chroma) -> BM25Retriever | None:
    """Lazy build and cache in-memory BM25 index for tenant documents.

    The cache key includes the live Chroma chunk count so a corpus that
    has changed since the last build (re-ingestion, wipe, cleanup) forces
    a rebuild instead of returning a stale index.
    """
    now = time.time()
    try:
        live_count = chroma._collection.count()
    except Exception:
        live_count = -1

    cache_key = (tenant_id, live_count)
    if cache_key in _TENANT_BM25_CACHE:
        cached_time, cached_bm25 = _TENANT_BM25_CACHE[cache_key]
        if now - cached_time < 60.0:  # cache for 60s
            return cached_bm25

    try:
        data = chroma.get(include=["documents", "metadatas"])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        ids = data.get("ids", [])

        if not documents:
            return None

        router_docs = [
            RouterDocument(
                id=ids[i] if i < len(ids) else f"chunk-{i}",
                text=text,
                metadata=metadatas[i] if i < len(metadatas) else {},
            )
            for i, text in enumerate(documents)
            if text and text.strip()
        ]

        if not router_docs:
            return None

        retriever = BM25Retriever(router_docs)
        _TENANT_BM25_CACHE[cache_key] = (now, retriever)
        return retriever
    except Exception as exc:
        logger.debug("BM25 tenant index build skipped (%s)", exc)
        return None


def decompose_query(query: str) -> list[str]:
    """
    Decompose compound queries and expand terms to prevent single-vector
    blindness.

    This is a lightweight deterministic fallback that handles compound
    questions (split on '?', ' and ', ' as well as '). The full
    QueryTranslationService (seven strategies: compress, expand, rephrase,
    multi_query, decompose, step_back, hyde) is the production path and is
    wired in via `run_query_translation()` below.
    """
    variants = [query.strip()]

    # 1. Split on question marks for compound questions
    parts = [p.strip() for p in re.split(r"\?+", query) if p.strip()]
    if len(parts) > 1:
        for p in parts:
            clean = re.sub(r"^(and|or|also)\s+", "", p, flags=re.IGNORECASE).strip()
            if len(clean) > 2 and clean not in variants:
                variants.append(clean)

    # 2. Split on ' and ' / ' as well as '
    elif re.search(r"\s+(?:and|as well as)\s+", query, flags=re.IGNORECASE):
        subparts = re.split(r"\s+(?:and|as well as)\s+", query, flags=re.IGNORECASE)
        for sp in subparts:
            clean = sp.strip()
            if len(clean) > 2 and clean not in variants:
                variants.append(clean)

    return list(dict.fromkeys(variants))


def run_query_translation(query: str) -> list[str]:
    """
    Run the production QueryTranslationService (seven strategies) against the
    user query and return the flattened list of variant texts for retrieval.

    Falls back to the deterministic `decompose_query` when no translation
    model is configured (e.g. no Ollama host reachable), so the pipeline never
    hard-fails on a missing model.
    """
    try:
        from src.inference.llm_gateway import llm_gateway

        ollama_model = os.getenv("OLLAMA_TRANSLATION_MODEL", "qwen2.5:7b-instruct")
        ollama_base = os.getenv("OLLAMA_TRANSLATION_BASE_URL", "http://localhost:11434")

        from src.query.query_translation import (
            OllamaTranslationModel,
            QueryTranslationService,
            QueryTranslationStrategy,
        )

        model = OllamaTranslationModel(
            model=ollama_model,
            base_url=ollama_base,
            timeout_seconds=float(os.getenv("OLLAMA_TRANSLATION_TIMEOUT", "30.0")),
        )
        service = QueryTranslationService(model=model)

        strategies = (
            QueryTranslationStrategy.MULTI_QUERY,
            QueryTranslationStrategy.DECOMPOSE,
            QueryTranslationStrategy.EXPAND,
        )
        results = service.translate_many(query, strategies)

        variants: list[str] = []
        for result in results:
            for variant in result.variants:
                if variant.text and variant.text not in variants:
                    variants.append(variant.text)

        if variants:
            return variants
    except Exception as exc:
        logger.debug("Query translation skipped (%s); using deterministic fallback", exc)

    return decompose_query(query)


def format_docs(docs: list[Document]) -> str:
    """Concatenate retrieved document chunks into clean delimited context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def format_docs_with_citations(docs: list[Document]) -> str:
    """Concatenate retrieved chunks with inline citation labels.

    Each block is prefixed with a numeric label and annotated with its
    source, page, and section so the model can cite it and the UI can
    render clickable source links.
    """
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page")
        section = doc.metadata.get("section_path")

        header_parts = [f"[{idx}] {source}"]
        if page is not None:
            header_parts.append(f"page {page}")
        if section:
            header_parts.append(f"section: {section}")

        blocks.append(" ".join(header_parts) + "\n" + doc.page_content.strip())

    return "\n\n---\n\n".join(blocks)


def query_tenant_rag(
    query: str,
    tenant_id: str,
    k: int = 3,
    include_eval: bool = False,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute enterprise multi-tenant RAG retrieval and answer generation:
    1. Input Validation & Prompt Injection Guardrails
    2. Hybrid Retrieval: Dense (Chroma) + Lexical (BM25)
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-Encoder Reranking
    5. Inference with vLLM / OpenAI Gateway
    6. PII Redaction & Audit Logging
    """
    # 0. Security Guardrails Check
    sec_result = validate_input_query(query)
    if not sec_result.is_safe:
        logger.warning("Query rejected by security guardrail: tenant=%s flags=%s", tenant_id, sec_result.flagged_reasons)
        return {
            "answer": "Security Policy Notice: The submitted query violates security policies or contains forbidden prompt injection patterns.",
            "sources": [],
            "query": query,
            "chunks_found": 0,
            "retrieval_time_ms": 0.0,
            "rerank_time_ms": 0.0,
            "inference_time_ms": 0.0,
            "total_time_ms": 0.0,
            "llm_provider": "security_guardrail",
            "security_flags": sec_result.flagged_reasons,
        }

    tenant_str = str(tenant_id)
    chroma: Chroma = _get_chroma_collection(tenant_str)

    # 1. Hybrid Retrieval Phase (Dense + BM25 + RRF)
    t_retrieval_start = time.perf_counter()
    retrieved_candidates: list[Document] = []

    try:
        variants = run_query_translation(query)
        dense_results: list[SearchResult] = []
        dense_seen: set[str] = set()

        candidate_depth = max(k * 3, 10)

        # Dense Retrieval across variants
        for var in variants:
            results = chroma.similarity_search_with_score(var, k=candidate_depth)
            for rank_pos, (doc, dist) in enumerate(results, start=1):
                doc_key = doc.page_content.strip()
                if doc_key not in dense_seen:
                    dense_seen.add(doc_key)
                    # Convert distance to similarity score
                    score = 1.0 / (1.0 + max(0.0, dist))
                    # Use the Chroma point ID (per-chunk), NOT the document_id.
                    # RRF merges by document.id, so dense and BM25 must share
                    # the same ID namespace. Using document_id collapses every
                    # dense chunk from one document into a single fused entry
                    # and prevents dense/BM25 hits on the same chunk from
                    # combining.
                    point_id = doc.metadata.get("chunk_id") or doc.metadata.get(
                        "document_id", f"dense_{rank_pos}"
                    )
                    dense_results.append(
                        SearchResult(
                            document=RouterDocument(
                                id=point_id,
                                text=doc.page_content,
                                metadata=doc.metadata,
                            ),
                            score=score,
                            retriever="dense",
                            rank=rank_pos,
                        )
                    )

        # Lexical (BM25) Retrieval
        bm25_retriever = _get_tenant_bm25(tenant_str, chroma)
        ranked_lists: list[list[SearchResult]] = [dense_results]

        if bm25_retriever:
            bm25_hits = bm25_retriever.search(query, top_k=candidate_depth)
            if bm25_hits:
                ranked_lists.append(bm25_hits)

        # Reciprocal Rank Fusion
        fused_hits = reciprocal_rank_fusion(
            ranked_lists,
            k=60,
            top_k=candidate_depth,
            weights={"dense": 1.2, "bm25": 1.0},
        )

        for hit in fused_hits:
            retrieved_candidates.append(
                Document(
                    page_content=hit.document.text,
                    metadata=hit.document.metadata,
                )
            )

    except Exception as exc:
        logger.error("Error retrieving documents from ChromaDB/BM25 for tenant %s: %s", tenant_str, exc)
        return {
            "answer": f"Unable to retrieve documents: {exc}",
            "sources": [],
            "query": query,
            "chunks_found": 0,
            "retrieval_time_ms": 0.0,
            "rerank_time_ms": 0.0,
            "inference_time_ms": 0.0,
            "llm_provider": "none",
        }

    t_retrieval_end = time.perf_counter()
    retrieval_time_ms = round((t_retrieval_end - t_retrieval_start) * 1000.0, 2)

    if not retrieved_candidates:
        return {
            "answer": "No relevant documents found in your workspace. Please upload and index documents first.",
            "sources": [],
            "query": query,
            "chunks_found": 0,
            "retrieval_time_ms": retrieval_time_ms,
            "rerank_time_ms": 0.0,
            "inference_time_ms": 0.0,
            "total_time_ms": retrieval_time_ms,
            "llm_provider": "none",
        }

# 2. Cross-Encoder Reranking Phase
    t_rerank_start = time.perf_counter()
    reranked_pairs = reranker.rerank(query, retrieved_candidates, top_k=max(k, 3))
    t_rerank_end = time.perf_counter()
    rerank_time_ms = round((t_rerank_end - t_rerank_start) * 1000.0, 2)

    final_docs = [doc for doc, _ in reranked_pairs]
    final_scores = [score for _, score in reranked_pairs]

    # Guard against reranker blindness: the top dense/RRF hit is the strongest
    # semantic match and must never be dropped from the context, even when the
    # cross-encoder scores it low on term overlap (e.g. a short query whose
    # answer lives in a chunk with different vocabulary). Insert it first if
    # it is not already present in the final top-k.
    if retrieved_candidates:
        top_dense = retrieved_candidates[0]
        top_dense_key = top_dense.page_content.strip()
        if not any(doc.page_content.strip() == top_dense_key for doc in final_docs):
            final_docs.insert(0, top_dense)
            final_scores.insert(0, 0.0)

    # Relevance threshold: refuse to answer from weak context instead of
    # fabricating. The threshold is configurable via the environment and
    # defaults to a conservative 0.0 (never refuse) so existing behaviour is
    # preserved unless explicitly opted in.
    try:
        relevance_threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.0"))
    except ValueError:
        relevance_threshold = 0.0

    best_score = max(final_scores) if final_scores else 0.0
    if relevance_threshold > 0.0 and best_score < relevance_threshold:
        logger.info(
            "Query rejected by relevance threshold: best_score=%.3f threshold=%.3f",
            best_score, relevance_threshold,
        )
        return {
            "answer": (
                "The retrieved documents do not contain enough relevant "
                "information to answer this question. Please rephrase or "
                "upload additional documents."
            ),
            "sources": [],
            "query": query,
            "chunks_found": 0,
            "retrieval_mode": "hybrid_rrf_rerank",
            "retrieval_time_ms": retrieval_time_ms,
            "rerank_time_ms": rerank_time_ms,
            "inference_time_ms": 0.0,
            "total_time_ms": round(retrieval_time_ms + rerank_time_ms, 2),
            "llm_provider": "none",
            "relevance_score": best_score,
        }

    # 3. Inference Phase via LLM Gateway (vLLM / OpenAI)
    # Build context with inline citations so the model can reference sources,
    # and so the returned sources carry page numbers for the UI.
    context = format_docs_with_citations(final_docs)
    prompt = _get_prompt()
    llm, provider_name = llm_gateway.get_llm(temperature=0.0)

    chain = prompt | llm | StrOutputParser()

    t_inference_start = time.perf_counter()
    try:
        raw_answer = chain.invoke({"context": context, "question": query})
        answer = redact_pii(raw_answer)
    except Exception as exc:
        logger.error("Error generating answer with LLM (%s): %s", provider_name, exc)
        answer = f"Error generating answer: {exc}"
    t_inference_end = time.perf_counter()
    inference_time_ms = round((t_inference_end - t_inference_start) * 1000.0, 2)

    # 4. Format Sources
    sources = []
    for idx, (doc, r_score) in enumerate(zip(final_docs, final_scores)):
        content = doc.page_content.strip()
        preview = content[:200] + ("..." if len(content) > 200 else "")
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown Document"),
                "chunk_index": doc.metadata.get("chunk_index", idx),
                "page": doc.metadata.get("page"),
                "section_path": doc.metadata.get("section_path"),
                "distance": r_score,
                "character_count": len(content),
                "preview": preview,
                "content": content,
            }
        )

    total_time_ms = round(retrieval_time_ms + rerank_time_ms + inference_time_ms, 2)

    response: dict[str, Any] = {
        "answer": answer,
        "sources": sources,
        "query": query,
        "chunks_found": len(final_docs),
        "retrieval_mode": "hybrid_rrf_rerank",
        "retrieval_time_ms": retrieval_time_ms,
        "rerank_time_ms": rerank_time_ms,
        "inference_time_ms": inference_time_ms,
        "total_time_ms": total_time_ms,
        "llm_provider": provider_name,
    }

    # 5. Optional Harness Evaluation
    if include_eval:
        try:
            from src.harness.evaluator import RAGEvaluator
            evaluator = RAGEvaluator()
            eval_report = evaluator.evaluate(query=query, answer=answer, context_chunks=sources)
            response["evaluation"] = eval_report.to_dict()
        except Exception as exc:
            logger.warning("Harness evaluation step failed: %s", exc)

    return response


if __name__ == "__main__":
    result = query_tenant_rag("What is company name ?", "10aa8269-4204-4c3e-8d0e-18c9b6735cad", include_eval=True)
    print("Provider:", result["llm_provider"])
    print("Retrieval Time:", result["retrieval_time_ms"], "ms")
    print("Rerank Time:", result.get("rerank_time_ms", 0), "ms")
    print("Inference Time:", result["inference_time_ms"], "ms")
    print("Total Latency:", result["total_time_ms"], "ms")
    print("Answer:", result["answer"])
"""
Run retrieval evaluation against the REAL tenant-scoped Chroma collection.

Usage from the project root:
    python scripts/run_retrieval_eval.py evals/rag_eval.jsonl --k 1 2 3 4 5 6 --mode hybrid
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from src.ingestion.processor import CachedOpenAIEmbeddings

from src.evaluation.retrieval_eval import (
    EvalCase,
    RetrievedItem,
    aggregate_metrics,
    evaluate_retrieval,
    load_eval_cases,
)
from src.query.query_router_rrf import (
    BM25Retriever,
    Document as RouterDocument,
    SearchResult,
    reciprocal_rank_fusion,
)
from src.query.reranker import reranker

_CACHED_STORES: dict[str, Chroma] = {}
_CACHED_BM25: dict[str, BM25Retriever] = {}


def get_cached_vectorstore(tenant_id: str) -> Chroma:
    """Use cached tenant Chroma collection to avoid repeatedly reopening SQLite."""
    if tenant_id in _CACHED_STORES:
        return _CACHED_STORES[tenant_id]

    import re
    clean_tenant = re.sub(r"[^a-z0-9_-]", "_", str(tenant_id).lower())
    chroma_path = Path(
        os.getenv(
            "CHROMA_PATH",
            str(PROJECT_ROOT / "chroma_db"),
        )
    )

    embeddings = CachedOpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY", "mock-key"),
    )

    store = Chroma(
        collection_name=f"tenant_{clean_tenant}",
        embedding_function=embeddings,
        persist_directory=str(chroma_path),
    )
    _CACHED_STORES[tenant_id] = store
    return store


def get_cached_bm25(tenant_id: str, vectorstore: Chroma) -> BM25Retriever | None:
    """Build BM25 index once per tenant for eval.

    The cache key includes the live chunk count so a corpus that has
    changed since the last build (re-ingestion, wipe, cleanup) forces a
    rebuild instead of returning a stale index.
    """
    try:
        live_count = vectorstore._collection.count()
    except Exception:
        live_count = -1

    cache_key = (tenant_id, live_count)
    if cache_key in _CACHED_BM25:
        return _CACHED_BM25[cache_key]

    all_data = vectorstore.get(include=["documents", "metadatas"])
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    ids = all_data.get("ids", [])

    router_docs = [
        RouterDocument(
            id=ids[i] if i < len(ids) else f"doc_{i}",
            text=text,
            metadata=metadatas[i] if i < len(metadatas) else {},
        )
        for i, text in enumerate(documents)
        if text and text.strip()
    ]

    if not router_docs:
        return None

    bm25 = BM25Retriever(router_docs)
    _CACHED_BM25[cache_key] = bm25
    return bm25


def chunk_key(metadata: dict) -> str | None:
    """Build the stable evaluation key used by Chroma metadata."""
    document_id = metadata.get("document_id")
    version = metadata.get("version", 1)
    chunk_index = metadata.get("chunk_index")

    if document_id is None or chunk_index is None:
        return None

    return f"{document_id}:v{version}:c{chunk_index}"


def retrieve(
    vectorstore: Chroma,
    tenant_id: str,
    query: str,
    k: int,
    mode: str = "hybrid",
) -> tuple[list[RetrievedItem], float]:
    started = perf_counter()

    if mode == "dense":
        documents = vectorstore.similarity_search(query, k=k)
    else:
        candidate_k = max(k * 3, 10)
        dense_docs = vectorstore.similarity_search(query, k=candidate_k)
        
        bm25 = get_cached_bm25(tenant_id, vectorstore)
        if bm25:
            bm25_results = bm25.search(query, top_k=candidate_k)
            dense_results = [
                SearchResult(
                    document=RouterDocument(
                        id=d.metadata.get("document_id", f"d_{idx}"),
                        text=d.page_content,
                        metadata=d.metadata,
                    ),
                    score=1.0 / (idx + 1),
                    retriever="dense",
                    rank=idx + 1,
                )
                for idx, d in enumerate(dense_docs)
            ]
            fused = reciprocal_rank_fusion([dense_results, bm25_results], k=60, top_k=candidate_k)
            fused_docs = [
                type("LCDoc", (), {"page_content": h.document.text, "metadata": h.document.metadata})()
                for h in fused
            ]
        else:
            fused_docs = dense_docs

        if mode == "rerank":
            reranked = reranker.rerank(query, fused_docs, top_k=k)
            documents = [doc for doc, _ in reranked]
        else:
            documents = fused_docs[:k]

    latency_ms = (perf_counter() - started) * 1000

    results = [
        RetrievedItem(
            document_id=doc.metadata.get("document_id"),
            chunk_id=chunk_key(doc.metadata),
            rank=rank,
        )
        for rank, doc in enumerate(documents, start=1)
    ]

    return results, latency_ms


def evaluate_case(
    case: EvalCase,
    k: int,
    mode: str = "hybrid",
) -> tuple[object, float, list[RetrievedItem]]:
    vectorstore = get_cached_vectorstore(case.tenant_id)
    retrieved, latency_ms = retrieve(
        vectorstore,
        case.tenant_id,
        case.question,
        k,
        mode=mode,
    )
    metrics = evaluate_retrieval(case, retrieved)

    return metrics, latency_ms, retrieved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "rerank"],
        default="hybrid",
        help="Retrieval strategy: dense baseline, hybrid (BM25+Dense+RRF), or rerank",
    )
    args = parser.parse_args()

    cases = load_eval_cases(args.dataset)

    if not cases:
        raise SystemExit("Evaluation dataset is empty.")

    print("=" * 80, flush=True)
    print(f"RETRIEVAL EVALUATION ({args.mode.upper()} MODE)", flush=True)
    print("=" * 80, flush=True)
    print(f"Cases: {len(cases)}", flush=True)
    print(f"K values: {args.k}", flush=True)
    print(flush=True)

    for k in args.k:
        metrics = []
        latencies = []

        for case in cases:
            result, latency_ms, retrieved = evaluate_case(case, k, mode=args.mode)

            metrics.append(result)
            latencies.append(latency_ms)

            print(
                f"[{case.id}] "
                f"hit={result.hit} "
                f"recall={result.recall:.3f} "
                f"rr={result.reciprocal_rank:.3f} "
                f"latency={latency_ms:.1f}ms",
                flush=True,
            )

            if not result.hit:
                print(
                    f"   MISS -> Expected: {case.expected_chunk_ids} | Retrieved chunks: {[r.chunk_id for r in retrieved]} | docs: {[r.document_id for r in retrieved]}",
                    flush=True,
                )

        aggregate = aggregate_metrics(metrics)
        mean_latency = sum(latencies) / len(latencies)

        print(flush=True)
        print(f"--- K={k} ({args.mode.upper()}) ---", flush=True)
        print(f"Hit@{k}:       {aggregate['hit_rate']:.3f}", flush=True)
        print(f"Recall@{k}:    {aggregate['mean_recall']:.3f}", flush=True)
        print(f"MRR:           {aggregate['mrr']:.3f}", flush=True)
        print(f"Mean latency:  {mean_latency:.1f} ms", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()

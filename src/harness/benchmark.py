from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Latency and throughput benchmark results for inference or retrieval."""

    operation_name: str
    total_duration_ms: float
    retrieval_duration_ms: float
    inference_duration_ms: float
    tokens_generated_approx: int
    throughput_tokens_per_sec: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_rag_query(
    query_fn: Callable[..., dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], BenchmarkResult]:
    """
    Wrap a RAG query execution to profile latency and throughput.
    """
    t_start = time.perf_counter()
    result = query_fn(*args, **kwargs)
    t_end = time.perf_counter()

    total_ms = (t_end - t_start) * 1000.0

    retrieval_ms = result.get("retrieval_time_ms", 0.0)
    inference_ms = result.get("inference_time_ms", total_ms - retrieval_ms)

    answer_text = result.get("answer", "")
    approx_tokens = max(1, len(answer_text.split()) * 4 // 3)
    throughput = (approx_tokens / (inference_ms / 1000.0)) if inference_ms > 0 else 0.0

    bench = BenchmarkResult(
        operation_name="rag_query",
        total_duration_ms=round(total_ms, 2),
        retrieval_duration_ms=round(retrieval_ms, 2),
        inference_duration_ms=round(inference_ms, 2),
        tokens_generated_approx=approx_tokens,
        throughput_tokens_per_sec=round(throughput, 2),
        metadata={
            "chunks_retrieved": result.get("chunks_found", 0),
            "provider": result.get("llm_provider", "unknown"),
        },
    )

    return result, bench

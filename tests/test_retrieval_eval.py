from src.evaluation.retrieval_eval import (
    EvalCase,
    RetrievedItem,
    aggregate_metrics,
    evaluate_retrieval,
)


def test_hit_and_mrr_when_relevant_chunk_is_rank_one() -> None:
    case = EvalCase(
        id="q1",
        question="What is X?",
        tenant_id="tenant-1",
        expected_chunk_ids=frozenset({"chunk-2"}),
    )

    retrieved = [
        RetrievedItem(document_id="doc-1", chunk_id="chunk-2", rank=1),
        RetrievedItem(document_id="doc-1", chunk_id="chunk-9", rank=2),
    ]

    metrics = evaluate_retrieval(case, retrieved)

    assert metrics.hit is True
    assert metrics.recall == 1.0
    assert metrics.reciprocal_rank == 1.0


def test_recall_counts_multiple_expected_chunks() -> None:
    case = EvalCase(
        id="q2",
        question="Compare X and Y",
        tenant_id="tenant-1",
        expected_chunk_ids=frozenset({"chunk-2", "chunk-3"}),
    )

    retrieved = [
        RetrievedItem(document_id="doc-1", chunk_id="chunk-2", rank=1),
        RetrievedItem(document_id="doc-1", chunk_id="chunk-8", rank=2),
        RetrievedItem(document_id="doc-1", chunk_id="chunk-3", rank=3),
    ]

    metrics = evaluate_retrieval(case, retrieved)

    assert metrics.hit is True
    assert metrics.recall == 1.0
    assert metrics.reciprocal_rank == 1.0


def test_mrr_penalizes_late_first_relevant_result() -> None:
    case = EvalCase(
        id="q3",
        question="What is X?",
        tenant_id="tenant-1",
        expected_document_ids=frozenset({"doc-2"}),
    )

    retrieved = [
        RetrievedItem(document_id="doc-1", chunk_id="chunk-1", rank=1),
        RetrievedItem(document_id="doc-2", chunk_id="chunk-2", rank=2),
    ]

    metrics = evaluate_retrieval(case, retrieved)

    assert metrics.hit is True
    assert metrics.recall == 1.0
    assert metrics.reciprocal_rank == 0.5


def test_aggregate_metrics() -> None:
    metrics = [
        type("M", (), {
            "hit": True,
            "recall": 1.0,
            "reciprocal_rank": 1.0,
        })(),
        type("M", (), {
            "hit": False,
            "recall": 0.0,
            "reciprocal_rank": 0.0,
        })(),
    ]

    result = aggregate_metrics(metrics)

    assert result == {
        "hit_rate": 0.5,
        "mean_recall": 0.5,
        "mrr": 0.5,
    }

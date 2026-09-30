"""
Retrieval evaluation utilities for the production RAG project.

This module intentionally does NOT change the live RAG pipeline.
It evaluates retrieval against a labeled JSONL dataset.

Dataset contract:

{
  "id": "q001",
  "question": "...",
  "tenant_id": "...",
  "expected_document_ids": ["..."],
  "expected_chunk_ids": ["..."],
  "expected_answer": "..."
}

At least one of expected_document_ids / expected_chunk_ids should be supplied
for retrieval metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Iterable


@dataclass(frozen=True)
class EvalCase:
    """One labeled RAG evaluation question."""

    id: str
    question: str
    tenant_id: str
    expected_document_ids: frozenset[str] = field(default_factory=frozenset)
    expected_chunk_ids: frozenset[str] = field(default_factory=frozenset)
    expected_answer: str | None = None


@dataclass(frozen=True)
class RetrievedItem:
    """Minimal retrieval result needed by the evaluator."""

    document_id: str | None
    chunk_id: str | None
    rank: int


@dataclass(frozen=True)
class RetrievalMetrics:
    """Metrics for one query at one retrieval depth."""

    hit: bool
    recall: float
    reciprocal_rank: float


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load newline-delimited JSON evaluation cases."""

    cases: list[EvalCase] = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # Empty lines are allowed for easier editing.
            if not line:
                continue

            payload = json.loads(line)

            question = str(payload["question"]).strip()
            tenant_id = str(payload["tenant_id"]).strip()

            if not question:
                raise ValueError(f"{path}:{line_number}: question is empty")

            if not tenant_id:
                raise ValueError(f"{path}:{line_number}: tenant_id is empty")

            cases.append(
                EvalCase(
                    id=str(payload["id"]),
                    question=question,
                    tenant_id=tenant_id,
                    expected_document_ids=frozenset(
                        str(value)
                        for value in payload.get("expected_document_ids", [])
                    ),
                    expected_chunk_ids=frozenset(
                        str(value)
                        for value in payload.get("expected_chunk_ids", [])
                    ),
                    expected_answer=payload.get("expected_answer"),
                )
            )

    return cases


def _relevant(
    item: RetrievedItem,
    case: EvalCase,
) -> bool:
    """Return True when a retrieved item matches labeled evidence."""

    if case.expected_chunk_ids and item.chunk_id in case.expected_chunk_ids:
        return True

    if case.expected_document_ids and item.document_id in case.expected_document_ids:
        return True

    return False


def evaluate_retrieval(
    case: EvalCase,
    retrieved: Iterable[RetrievedItem],
) -> RetrievalMetrics:
    """
    Calculate Hit@K, Recall@K and reciprocal rank for one query.

    Important:
    - If chunk IDs are labeled, recall is measured over expected chunks.
    - Otherwise, if document IDs are labeled, recall is measured over expected
      documents.
    - MRR uses the first relevant retrieved item.
    """

    results = list(retrieved)

    expected = (
        set(case.expected_chunk_ids)
        if case.expected_chunk_ids
        else set(case.expected_document_ids)
    )

    if not expected:
        raise ValueError(
            f"Evaluation case {case.id!r} has no expected document/chunk IDs."
        )

    relevant_retrieved: set[str | None] = set()
    for item in results:
        if _relevant(item, case):
            relevant_retrieved.add(item.chunk_id or item.document_id)

    relevant_retrieved.discard(None)

    first_relevant_rank: int | None = None

    for item in results:
        if _relevant(item, case):
            first_relevant_rank = item.rank
            break

    return RetrievalMetrics(
        hit=bool(relevant_retrieved),
        recall=len(relevant_retrieved) / len(expected),
        reciprocal_rank=(
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        ),
    )


def aggregate_metrics(
    metrics: Iterable[RetrievalMetrics],
) -> dict[str, float]:
    """Aggregate per-query metrics using macro averages."""

    rows = list(metrics)

    if not rows:
        return {
            "hit_rate": 0.0,
            "mean_recall": 0.0,
            "mrr": 0.0,
        }

    return {
        "hit_rate": sum(row.hit for row in rows) / len(rows),
        "mean_recall": sum(row.recall for row in rows) / len(rows),
        "mrr": sum(row.reciprocal_rank for row in rows) / len(rows),
    }

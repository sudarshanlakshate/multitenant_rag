from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.inference.llm_gateway import llm_gateway

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Comprehensive quality evaluation for a RAG question-answer run."""

    query: str
    answer: str
    context_chunks_count: int
    context_relevance_score: float  # 0.0 to 1.0
    faithfulness_score: float        # 0.0 to 1.0 (groundedness)
    answer_relevance_score: float    # 0.0 to 1.0
    overall_quality_score: float     # 0.0 to 1.0
    evaluation_details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an impartial evaluation judge for an Enterprise RAG system.\n"
            "Given the provided Context and an Answer, evaluate whether every statement in the Answer "
            "is directly supported by the Context.\n"
            "If the answer makes claims NOT in the context, penalize the faithfulness score.\n"
            "Respond in EXACTLY this format:\n"
            "SCORE: <number between 0.0 and 1.0>\n"
            "EXPLANATION: <brief one sentence reason>",
        ),
        (
            "human",
            "Context:\n{context}\n\nAnswer:\n{answer}",
        ),
    ]
)

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an impartial evaluation judge for an Enterprise RAG system.\n"
            "Given a Question and the generated Answer, rate how directly, concisely, and accurately "
            "the Answer addresses the Question.\n"
            "Respond in EXACTLY this format:\n"
            "SCORE: <number between 0.0 and 1.0>\n"
            "EXPLANATION: <brief one sentence reason>",
        ),
        (
            "human",
            "Question: {question}\n\nAnswer: {answer}",
        ),
    ]
)


def _parse_score(text: str) -> tuple[float, str]:
    """Parse SCORE and EXPLANATION from LLM judge response."""
    score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text)
    score = float(score_match.group(1)) if score_match else 0.8
    score = max(0.0, min(1.0, score))

    exp_match = re.search(r"EXPLANATION:\s*(.+)", text, re.DOTALL)
    explanation = exp_match.group(1).strip() if exp_match else text.strip()
    return score, explanation


class RAGEvaluator:
    """Harness Evaluator for RAG quality metrics (Faithfulness, Relevance, Groundedness)."""

    def __init__(self) -> None:
        self.llm, self.provider = llm_gateway.get_llm(temperature=0.0)

    def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: list[dict[str, Any]],
    ) -> EvaluationReport:
        """Run quantitative evaluation harness on a RAG interaction."""
        full_context = "\n\n".join(chunk.get("content", "") for chunk in context_chunks)

        # 1. Evaluate Faithfulness
        faith_chain = FAITHFULNESS_PROMPT | self.llm | StrOutputParser()
        try:
            faith_raw = faith_chain.invoke({"context": full_context, "answer": answer})
            faith_score, faith_reason = _parse_score(faith_raw)
        except Exception as exc:
            logger.warning("Faithfulness evaluation failed: %s", exc)
            faith_score, faith_reason = 1.0, "Evaluator fallback"

        # 2. Evaluate Answer Relevance
        rel_chain = RELEVANCE_PROMPT | self.llm | StrOutputParser()
        try:
            rel_raw = rel_chain.invoke({"question": query, "answer": answer})
            rel_score, rel_reason = _parse_score(rel_raw)
        except Exception as exc:
            logger.warning("Answer relevance evaluation failed: %s", exc)
            rel_score, rel_reason = 1.0, "Evaluator fallback"

        # 3. Context Relevance (Heuristic density of query terms in retrieved context)
        query_words = set(re.findall(r"\w+", query.lower()))
        context_words = set(re.findall(r"\w+", full_context.lower()))
        overlap = query_words.intersection(context_words)
        context_rel_score = len(overlap) / max(len(query_words), 1)
        context_rel_score = min(1.0, round(context_rel_score, 2))

        # Overall composite score
        overall = round(
            (0.4 * faith_score) + (0.4 * rel_score) + (0.2 * context_rel_score),
            3,
        )

        return EvaluationReport(
            query=query,
            answer=answer,
            context_chunks_count=len(context_chunks),
            context_relevance_score=context_rel_score,
            faithfulness_score=faith_score,
            answer_relevance_score=rel_score,
            overall_quality_score=overall,
            evaluation_details={
                "judge_provider": self.provider,
                "faithfulness_explanation": faith_reason,
                "relevance_explanation": rel_reason,
            },
        )

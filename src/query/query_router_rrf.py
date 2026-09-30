from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
import re
from typing import Any, Protocol, TypedDict


# ============================================================================
# Type aliases
# ============================================================================
#
# These aliases describe the shape of reusable concepts in this module.
#
# Vector  -> one embedding vector
# EmbedFn -> function that accepts text and returns an embedding vector
#
# Example:
#     embed_fn: EmbedFn
#
# means:
#     embed_fn is a callable: (str) -> list[float]
#

Vector = list[float]
EmbedFn = Callable[[str], Vector]


# ============================================================================
# Shared result types
# ============================================================================

@dataclass(frozen=True)
class Document:
    """A searchable document or document chunk."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """One result returned by one retrieval strategy."""

    document: Document
    score: float
    retriever: str
    rank: int = 0
    rrf_score: float = 0.0


class RouterSearchResult(TypedDict):
    """
    Explicit return shape of QueryRouter.search().

    TypedDict is useful here because the method intentionally returns
    dictionary-style data rather than a dataclass.
    """

    query: str
    decision: QueryDecision
    results: list[SearchResult]
    retriever_results: dict[str, list[SearchResult]]


# ============================================================================
# Retriever contract
# ============================================================================

class Retriever(Protocol):
    """
    Structural interface for all retrievers.

    Any class with:
        name: str

    and:
        search(query: str, top_k: int) -> list[SearchResult]

    satisfies this protocol.
    """

    name: str

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        ...


# ============================================================================
# Query classification / routing
# ============================================================================

class QueryType:
    EXACT = "exact"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class QueryDecision:
    query_type: str
    retrievers: tuple[str, ...]
    reason: str


class QueryClassifier:
    """
    Lightweight deterministic router.

    This is intentionally not an LLM call. Routing decisions should be
    observable and predictable before introducing model-based routing.
    """

    IDENTIFIER_RE = re.compile(
        r"""
        (
            \b[A-Z]{2,}[-_]\d{2,}\b |
            \b[A-Z]{2,}\d{2,}\b |
            \b\d{3,}\b |
            \b[0-9a-f]{8}-[0-9a-f-]{27,}\b
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    QUOTED_RE = re.compile(r"""["']([^"']{2,})["']""")

    def classify(self, query: str) -> QueryDecision:
        q: str = query.strip()

        if not q:
            raise ValueError("query must not be empty")

        if self.QUOTED_RE.search(q) or self.IDENTIFIER_RE.search(q):
            return QueryDecision(
                query_type=QueryType.EXACT,
                retrievers=("exact", "bm25"),
                reason="Quoted phrase or identifier detected.",
            )

        tokens: list[str] = self._tokens(q)

        if len(tokens) <= 3:
            return QueryDecision(
                query_type=QueryType.KEYWORD,
                retrievers=("exact", "bm25", "vector"),
                reason=(
                    "Short query; lexical precision matters, while vector "
                    "search protects against vocabulary mismatch."
                ),
            )

        if self._looks_natural_language(q):
            return QueryDecision(
                query_type=QueryType.SEMANTIC,
                retrievers=("vector", "bm25", "exact"),
                reason=(
                    "Natural-language question detected; semantic retrieval "
                    "is primary, lexical retrieval remains a recall path."
                ),
            )

        return QueryDecision(
            query_type=QueryType.HYBRID,
            retrievers=("bm25", "vector", "exact"),
            reason="No strong routing signal; use all retrieval paths.",
        )

    @staticmethod
    def _tokens(query: str) -> list[str]:
        return re.findall(r"\b[\w.-]+\b", query.lower())

    @staticmethod
    def _looks_natural_language(query: str) -> bool:
        q: str = query.lower().strip()

        question_words: tuple[str, ...] = (
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "which",
            "can",
            "could",
            "should",
            "explain",
            "difference",
            "compare",
        )

        return q.endswith("?") or q.startswith(question_words)


# ============================================================================
# Exact matcher
# ============================================================================

class ExactRetriever:
    name: str = "exact"

    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents: list[Document] = list(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        q: str = self._normalize(query)

        if not q:
            return []

        results: list[SearchResult] = []

        for doc in self.documents:
            text: str = self._normalize(doc.text)

            if q in text:
                score: float = 1.0 + len(q) / max(len(text), 1)

                results.append(
                    SearchResult(
                        document=doc,
                        score=score,
                        retriever=self.name,
                    )
                )

        results.sort(key=lambda result: result.score, reverse=True)

        for rank, result in enumerate(results[:top_k], start=1):
            result.rank = rank

        return results[:top_k]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())


# ============================================================================
# Simple BM25 implementation
# ============================================================================

class BM25Retriever:
    """
    Small dependency-free BM25 implementation.

    For a large production corpus, move BM25 to a real search engine or
    a sparse/BM25 index. This implementation exists so routing + fusion
    can be tested locally without infrastructure.
    """

    name: str = "bm25"
    TOKEN_RE = re.compile(r"\b\w+\b")

    def __init__(
        self,
        documents: Sequence[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents: list[Document] = list(documents)
        self.k1: float = k1
        self.b: float = b

        self.tokenized: list[list[str]] = [
            self._tokenize(doc.text)
            for doc in self.documents
        ]

        self.doc_lengths: list[int] = [
            len(tokens)
            for tokens in self.tokenized
        ]

        self.avgdl: float = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

        self.doc_freq: Counter[str] = Counter()

        for tokens in self.tokenized:
            for token in set(tokens):
                self.doc_freq[token] += 1

        self.n_docs: int = len(self.documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        query_tokens: list[str] = self._tokenize(query)

        if not query_tokens or not self.documents:
            return []

        scores: list[float] = []

        for tokens in self.tokenized:
            term_frequency: Counter[str] = Counter(tokens)
            doc_len: int = len(tokens)
            score: float = 0.0

            for term in query_tokens:
                df: int = self.doc_freq.get(term, 0)

                if df == 0:
                    continue

                idf: float = math.log(
                    1 + (self.n_docs - df + 0.5) / (df + 0.5)
                )

                tf: int = term_frequency.get(term, 0)

                if tf == 0:
                    continue

                denominator: float = (
                    tf
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b * doc_len / max(self.avgdl, 1e-12)
                    )
                )

                score += idf * (
                    tf * (self.k1 + 1) / denominator
                )

            scores.append(score)

        ranked_indices: list[int] = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[SearchResult] = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1,
        ):
            if scores[index] <= 0:
                continue

            results.append(
                SearchResult(
                    document=self.documents[index],
                    score=scores[index],
                    retriever=self.name,
                    rank=rank,
                )
            )

        return results

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return cls.TOKEN_RE.findall(text.lower())


# ============================================================================
# In-memory vector retriever for local testing
# ============================================================================

class VectorRetriever:
    """
    Simple vector retriever.

    embed_fn must accept text and return a numeric embedding vector.

    In production, replace this class with a Chroma adapter and a real
    embedding provider.
    """

    name: str = "vector"

    def __init__(
        self,
        documents: Sequence[Document],
        embed_fn: EmbedFn,
    ) -> None:
        self.documents: list[Document] = list(documents)
        self.embed_fn: EmbedFn = embed_fn

        self.document_vectors: list[Vector] = [
            self.embed_fn(doc.text)
            for doc in self.documents
        ]

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        query_vector: Vector = self.embed_fn(query)

        scored: list[tuple[int, float]] = []

        for index, vector in enumerate(self.document_vectors):
            score: float = self._cosine_similarity(
                query_vector,
                vector,
            )
            scored.append((index, score))

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[SearchResult] = []

        for rank, (index, score) in enumerate(
            scored[:top_k],
            start=1,
        ):
            results.append(
                SearchResult(
                    document=self.documents[index],
                    score=score,
                    retriever=self.name,
                    rank=rank,
                )
            )

        return results

    @staticmethod
    def _cosine_similarity(
        a: Sequence[float],
        b: Sequence[float],
    ) -> float:
        if len(a) != len(b):
            raise ValueError("Vector dimensions must match.")

        dot: float = sum(
            x * y
            for x, y in zip(a, b)
        )

        norm_a: float = math.sqrt(
            sum(x * x for x in a)
        )

        norm_b: float = math.sqrt(
            sum(y * y for y in b)
        )

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================

def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[SearchResult]],
    *,
    k: int = 60,
    top_k: int = 10,
    weights: dict[str, float] | None = None,
) -> list[SearchResult]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score:
        sum(weight / (k + rank))

    Ranks are 1-based:
        rank 1 -> 1 / (k + 1)
        rank 2 -> 1 / (k + 2)

    RRF intentionally ignores the original BM25/vector/exact score scales.
    """

    if k <= 0:
        raise ValueError("RRF k must be > 0")

    fused: dict[str, SearchResult] = {}
    scores: defaultdict[str, float] = defaultdict(float)

    for results in ranked_lists:
        for position, result in enumerate(results, start=1):
            doc_id: str = result.document.id

            weight: float = 1.0

            if weights is not None:
                weight = weights.get(result.retriever, 1.0)

            scores[doc_id] += weight / (k + position)

            if doc_id not in fused:
                fused[doc_id] = SearchResult(
                    document=result.document,
                    score=result.score,
                    retriever="rrf",
                )

    ordered_ids: list[str] = sorted(
        scores,
        key=lambda doc_id: scores[doc_id],
        reverse=True,
    )

    output: list[SearchResult] = []

    for rank, doc_id in enumerate(
        ordered_ids[:top_k],
        start=1,
    ):
        result: SearchResult = fused[doc_id]

        result.rrf_score = scores[doc_id]
        result.rank = rank

        output.append(result)

    return output


# ============================================================================
# End-to-end query router
# ============================================================================

class QueryRouter:
    """
    Classify a query, execute selected retrievers, then fuse their rankings.
    """

    def __init__(
        self,
        classifier: QueryClassifier,
        retrievers: dict[str, Retriever],
        *,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self.classifier: QueryClassifier = classifier
        self.retrievers: dict[str, Retriever] = retrievers
        self.rrf_k: int = rrf_k
        self.candidate_k: int = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> RouterSearchResult:
        decision: QueryDecision = self.classifier.classify(query)

        ranked_lists: list[list[SearchResult]] = []

        for retriever_name in decision.retrievers:
            retriever: Retriever | None = self.retrievers.get(
                retriever_name
            )

            if retriever is None:
                raise RuntimeError(
                    f"Router selected '{retriever_name}', "
                    "but no retriever was registered."
                )

            results: list[SearchResult] = retriever.search(
                query,
                top_k=self.candidate_k,
            )

            ranked_lists.append(results)

        fused: list[SearchResult] = reciprocal_rank_fusion(
            ranked_lists,
            k=self.rrf_k,
            top_k=top_k,
        )

        retriever_results: dict[str, list[SearchResult]] = {
            name: results
            for name, results in zip(
                decision.retrievers,
                ranked_lists,
            )
        }

        return {
            "query": query,
            "decision": decision,
            "results": fused,
            "retriever_results": retriever_results,
        }


# ============================================================================
# Deterministic tiny embedding used ONLY for local tests
# ============================================================================

def keyword_embedding(text: str) -> Vector:
    """
    Tiny deterministic embedding for tests.

    DO NOT use this as a production embedding model.
    It only gives us a local vector retriever without an external model.
    """

    vocabulary: tuple[str, ...] = (
        "postgresql",
        "database",
        "password",
        "encoding",
        "rag",
        "retrieval",
        "semantic",
        "search",
        "invoice",
        "document",
        "migration",
        "alembic",
        "qdrant",
        "vector",
        "bm25",
        "exact",
    )

    tokens: set[str] = set(
        re.findall(r"\b\w+\b", text.lower())
    )

    return [
        1.0 if word in tokens else 0.0
        for word in vocabulary
    ]

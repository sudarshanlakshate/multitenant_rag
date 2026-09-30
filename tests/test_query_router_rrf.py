
from src.query.query_router_rrf import (
    BM25Retriever,
    Document,
    ExactRetriever,
    QueryClassifier,
    QueryRouter,
    VectorRetriever,
    keyword_embedding,
    reciprocal_rank_fusion,
)


DOCUMENTS = [
    Document(
        id="doc-1",
        text="PostgreSQL password URL encoding uses percent encoding. Encode @ as %40.",
    ),
    Document(
        id="doc-2",
        text="Alembic manages PostgreSQL database schema migrations.",
    ),
    Document(
        id="doc-3",
        text="Vector search retrieves semantically similar RAG chunks.",
    ),
    Document(
        id="doc-4",
        text="BM25 is a lexical retrieval algorithm for keyword search.",
    ),
    Document(
        id="doc-5",
        text="Qdrant supports hybrid dense and sparse retrieval with RRF.",
    ),
]


def build_router():
    retrievers = {
        "exact": ExactRetriever(DOCUMENTS),
        "bm25": BM25Retriever(DOCUMENTS),
        "vector": VectorRetriever(DOCUMENTS, keyword_embedding),
    }

    return QueryRouter(
        classifier=QueryClassifier(),
        retrievers=retrievers,
        rrf_k=60,
        candidate_k=5,
    )


def test_classifier_identifier():
    decision = QueryClassifier().classify("ERR404")
    assert decision.query_type == "exact"
    assert "exact" in decision.retrievers
    assert "bm25" in decision.retrievers


def test_classifier_question():
    decision = QueryClassifier().classify(
        "How does semantic vector search work?"
    )
    assert decision.query_type == "semantic"
    assert "vector" in decision.retrievers


def test_exact_retriever():
    results = ExactRetriever(DOCUMENTS).search("percent encoding", top_k=5)
    assert results
    assert results[0].document.id == "doc-1"


def test_bm25_retriever():
    results = BM25Retriever(DOCUMENTS).search("Alembic migrations", top_k=5)
    assert results
    assert results[0].document.id == "doc-2"


def test_vector_retriever():
    results = VectorRetriever(
        DOCUMENTS,
        keyword_embedding,
    ).search("Qdrant RRF vector", top_k=5)

    assert results
    assert results[0].document.id == "doc-5"


def test_rrf_promotes_agreement():
    first = [
        ExactRetriever(DOCUMENTS).search("encoding", top_k=5)[0]
    ]

    second = [
        BM25Retriever(DOCUMENTS).search("PostgreSQL password encoding", top_k=5)[0]
    ]

    fused = reciprocal_rank_fusion(
        [first, second],
        k=60,
        top_k=5,
    )

    assert fused[0].document.id == "doc-1"
    assert fused[0].rrf_score > 0


def test_end_to_end_router():
    router = build_router()

    output = router.search(
        "How does semantic vector search work?",
        top_k=3,
    )

    assert output["decision"].query_type == "semantic"
    assert len(output["results"]) == 3
    assert all(result.rrf_score > 0 for result in output["results"])


if __name__ == "__main__":
    tests = [
        test_classifier_identifier,
        test_classifier_question,
        test_exact_retriever,
        test_bm25_retriever,
        test_vector_retriever,
        test_rrf_promotes_agreement,
        test_end_to_end_router,
    ]

    for test_fn in tests:
        test_fn()
        print(f"PASS: {test_fn.__name__}")

    print("\nAll tests passed.")

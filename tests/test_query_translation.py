from __future__ import annotations

from src.query.query_translation import (
    QueryTranslationService,
    QueryTranslationStrategy,
    QueryVariantKind,
    RuleBasedTranslationModel,
    TranslationError,
    translate_many,
)


def test_all_strategies() -> None:
    service: QueryTranslationService = QueryTranslationService(
        model=RuleBasedTranslationModel()
    )

    query: str = (
        "How does hybrid RAG combine BM25 and vector search "
        "for enterprise retrieval?"
    )

    expected_kinds: dict[
        QueryTranslationStrategy,
        QueryVariantKind,
    ] = {
        QueryTranslationStrategy.COMPRESS: QueryVariantKind.QUERY,
        QueryTranslationStrategy.EXPAND: QueryVariantKind.QUERY,
        QueryTranslationStrategy.REPHRASE: QueryVariantKind.QUERY,
        QueryTranslationStrategy.MULTI_QUERY: QueryVariantKind.QUERY,
        QueryTranslationStrategy.DECOMPOSE: QueryVariantKind.SUBQUERY,
        QueryTranslationStrategy.STEP_BACK: QueryVariantKind.QUERY,
        QueryTranslationStrategy.HYDE: (
            QueryVariantKind.HYPOTHETICAL_DOCUMENT
        ),
    }

    for strategy, expected_kind in expected_kinds.items():
        result = service.translate(query, strategy)

        assert result.original_query == query
        assert result.strategy == strategy
        assert result.variants

        for variant in result.variants:
            assert isinstance(variant.text, str)
            assert variant.text
            assert variant.kind == expected_kind
            assert variant.strategy == strategy


def test_variant_counts() -> None:
    service: QueryTranslationService = QueryTranslationService(
        model=RuleBasedTranslationModel()
    )

    query: str = "Explain hybrid retrieval."

    multi: object = service.translate(
        query,
        QueryTranslationStrategy.MULTI_QUERY,
    )
    assert len(multi.variants) == 3

    decomposed: object = service.translate(
        query,
        QueryTranslationStrategy.DECOMPOSE,
    )
    assert len(decomposed.variants) == 2


def test_translate_many() -> None:
    service: QueryTranslationService = QueryTranslationService(
        model=RuleBasedTranslationModel()
    )

    strategies: tuple[
        QueryTranslationStrategy,
        ...,
    ] = (
        QueryTranslationStrategy.COMPRESS,
        QueryTranslationStrategy.REPHRASE,
        QueryTranslationStrategy.MULTI_QUERY,
    )

    results = translate_many(
        service=service,
        query="Explain hybrid retrieval.",
        strategies=strategies,
    )

    assert len(results) == 3


def test_empty_query() -> None:
    service: QueryTranslationService = QueryTranslationService(
        model=RuleBasedTranslationModel()
    )

    try:
        service.translate(
            "",
            QueryTranslationStrategy.REPHRASE,
        )
    except ValueError:
        return

    raise AssertionError("Empty query must raise ValueError.")


def test_invalid_json() -> None:
    class BadModel:
        def generate(self, prompt: str) -> str:
            return "not-json"

    service: QueryTranslationService = QueryTranslationService(
        model=BadModel()
    )

    try:
        service.translate(
            "Explain hybrid retrieval.",
            QueryTranslationStrategy.REPHRASE,
        )
    except TranslationError:
        return

    raise AssertionError("Invalid JSON must raise TranslationError.")


def test_invalid_variants_type() -> None:
    class BadModel:
        def generate(self, prompt: str) -> str:
            return '{"variants": [123]}'

    service: QueryTranslationService = QueryTranslationService(
        model=BadModel()
    )

    try:
        service.translate(
            "Explain hybrid retrieval.",
            QueryTranslationStrategy.REPHRASE,
        )
    except TranslationError:
        return

    raise AssertionError(
        "Non-string variants must raise TranslationError."
    )


if __name__ == "__main__":
    test_all_strategies()
    test_variant_counts()
    test_translate_many()
    test_empty_query()
    test_invalid_json()
    test_invalid_variants_type()
    print("All query_translation type-safe tests passed.")

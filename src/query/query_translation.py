from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, TypeAlias, cast


# ============================================================================
# Shared type aliases
# ============================================================================

MetadataValue: TypeAlias = str | int | float | bool | None
Metadata: TypeAlias = dict[str, MetadataValue]


# ============================================================================
# Query translation domain types
# ============================================================================

class QueryTranslationStrategy(StrEnum):
    COMPRESS = "compress"
    EXPAND = "expand"
    REPHRASE = "rephrase"
    MULTI_QUERY = "multi_query"
    DECOMPOSE = "decompose"
    STEP_BACK = "step_back"
    HYDE = "hyde"


class QueryVariantKind(StrEnum):
    QUERY = "query"
    SUBQUERY = "subquery"
    HYPOTHETICAL_DOCUMENT = "hypothetical_document"


@dataclass(frozen=True)
class QueryVariant:
    """One translated representation of the original user query."""

    text: str
    kind: QueryVariantKind
    strategy: QueryTranslationStrategy
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class QueryTranslationResult:
    """Complete result returned by the translation service."""

    original_query: str
    strategy: QueryTranslationStrategy
    variants: list[QueryVariant]
    metadata: Metadata = field(default_factory=dict)


class TranslationError(RuntimeError):
    """Raised when translation fails or the model returns invalid data."""


class TranslationModel(Protocol):
    """Contract implemented by any translation model/provider."""

    def generate(self, prompt: str) -> str:
        """Generate structured text from a prompt."""
        ...


# ============================================================================
# Small runtime type helpers
# ============================================================================

def _as_object_dict(value: object) -> dict[str, object]:
    """
    Convert an unknown JSON object into a typed dictionary.

    We deliberately do not use `dict[str, Any]`.
    JSON is untrusted external data, so its structure is checked at runtime.
    """
    if not isinstance(value, dict):
        raise TranslationError("Expected a JSON object.")

    result: dict[str, object] = {}

    for key, item in value.items():
        if not isinstance(key, str):
            raise TranslationError("JSON object keys must be strings.")
        result[key] = item

    return result


def _as_string_list(value: object) -> list[str]:
    """Validate that a JSON value is a list containing only strings."""
    if not isinstance(value, list):
        raise TranslationError("'variants' must be a JSON list.")

    result: list[str] = []

    for item in value:
        if not isinstance(item, str):
            raise TranslationError("Every query variant must be a string.")

        normalized: str = " ".join(item.split())

        if normalized:
            result.append(normalized)

    if not result:
        raise TranslationError("Translation produced no usable variants.")

    return result


# ============================================================================
# Offline deterministic model
# ============================================================================

class RuleBasedTranslationModel:
    """
    Deterministic local model used for unit tests and development.

    This is intentionally NOT a production query-rewriting model.
    """

    def generate(self, prompt: str) -> str:
        strategy_match = re.search(
            r"STRATEGY:\s*([a-z_]+)",
            prompt,
        )
        query_match = re.search(
            r"ORIGINAL_QUERY:\s*(.+)",
            prompt,
        )

        if strategy_match is None:
            raise TranslationError("Prompt is missing STRATEGY.")

        if query_match is None:
            raise TranslationError("Prompt is missing ORIGINAL_QUERY.")

        strategy_text: str = strategy_match.group(1)
        query: str = query_match.group(1).strip()

        if strategy_text == QueryTranslationStrategy.COMPRESS.value:
            words: list[str] = query.split()
            translated: str = " ".join(words[: min(8, len(words))])
            return json.dumps({"variants": [translated]})

        if strategy_text == QueryTranslationStrategy.EXPAND.value:
            translated = (
                f"{query} related concepts definitions implementation examples"
            )
            return json.dumps({"variants": [translated]})

        if strategy_text == QueryTranslationStrategy.REPHRASE.value:
            translated = f"Explain {query} using different wording"
            return json.dumps({"variants": [translated]})

        if strategy_text == QueryTranslationStrategy.MULTI_QUERY.value:
            variants: list[str] = [
                query,
                f"What is {query}?",
                f"How does {query} work?",
            ]
            return json.dumps({"variants": variants})

        if strategy_text == QueryTranslationStrategy.DECOMPOSE.value:
            variants = [
                f"What is the main concept behind {query}?",
                f"What are the important components of {query}?",
            ]
            return json.dumps({"variants": variants})

        if strategy_text == QueryTranslationStrategy.STEP_BACK.value:
            translated = (
                "What are the general principles and concepts "
                f"related to {query}?"
            )
            return json.dumps({"variants": [translated]})

        if strategy_text == QueryTranslationStrategy.HYDE.value:
            translated = (
                f"A technical document would explain {query}, "
                "including its architecture, behavior, and practical usage."
            )
            return json.dumps({"variants": [translated]})

        raise TranslationError(
            f"Unsupported strategy returned by test model: {strategy_text}"
        )


# ============================================================================
# Ollama provider
# ============================================================================

class OllamaTranslationModel:
    """
    Ollama provider using the local /api/chat HTTP endpoint.

    The model name is injected so the code does not assume a particular
    Qwen tag. Use `ollama list` and pass the exact installed tag.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
    ) -> None:
        normalized_model: str = model.strip()

        if not normalized_model:
            raise ValueError("model must not be empty.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0.")

        self.model: str = normalized_model
        self.base_url: str = base_url.rstrip("/")
        self.timeout_seconds: float = timeout_seconds

    def generate(self, prompt: str) -> str:
        """Send one non-streaming chat request to Ollama."""
        request_payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "format": "json",
        }

        encoded_payload: bytes = json.dumps(
            request_payload
        ).encode("utf-8")

        request: urllib.request.Request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=encoded_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw_response: bytes = response.read()

        except urllib.error.URLError as exc:
            raise TranslationError(
                f"Could not connect to Ollama at {self.base_url}: {exc}"
            ) from exc

        response_text: str = raw_response.decode("utf-8")

        # json.loads() is typed as returning Any by typeshed.
        # Cast it to object immediately so unknown/Any does not leak.
        parsed: object = cast(object, json.loads(response_text))

        response_object: dict[str, object] = _as_object_dict(parsed)

        raw_message: object = response_object.get("message")

        message_object: dict[str, object] = _as_object_dict(raw_message)

        raw_content: object = message_object.get("content")

        if not isinstance(raw_content, str):
            raise TranslationError(
                "Ollama response field 'message.content' must be a string."
            )

        content: str = raw_content.strip()

        if not content:
            raise TranslationError("Ollama returned empty generated content.")

        return content


# ============================================================================
# Query translation service
# ============================================================================

class QueryTranslationService:
    """
    Transforms a user query before retrieval.

    Responsibilities:
      1. validate input
      2. build the strategy-specific prompt
      3. call the injected model
      4. validate structured model output
      5. create typed QueryVariant objects
    """

    def __init__(self, model: TranslationModel) -> None:
        self.model: TranslationModel = model

    def translate(
        self,
        query: str,
        strategy: QueryTranslationStrategy,
    ) -> QueryTranslationResult:
        normalized_query: str = self._validate_query(query)

        prompt: str = self._build_prompt(
            query=normalized_query,
            strategy=strategy,
        )

        raw_output: str = self.model.generate(prompt)

        translated_texts: list[str] = self._parse_variants(
            raw_output
        )

        variant_kind: QueryVariantKind = self._variant_kind(strategy)

        variants: list[QueryVariant] = []

        for text in translated_texts:
            variants.append(
                QueryVariant(
                    text=text,
                    kind=variant_kind,
                    strategy=strategy,
                )
            )

        metadata: Metadata = {
            "variant_count": len(variants),
            "model_type": type(self.model).__name__,
        }

        return QueryTranslationResult(
            original_query=normalized_query,
            strategy=strategy,
            variants=variants,
            metadata=metadata,
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        normalized: str = " ".join(query.split())

        if not normalized:
            raise ValueError("query must not be empty.")

        if len(normalized) > 4000:
            raise ValueError(
                "query is too long; maximum length is 4000 characters."
            )

        return normalized

    @staticmethod
    def _build_prompt(
        query: str,
        strategy: QueryTranslationStrategy,
    ) -> str:
        instructions: dict[QueryTranslationStrategy, str] = {
            QueryTranslationStrategy.COMPRESS:
                "Compress the query while preserving retrieval intent.",
            QueryTranslationStrategy.EXPAND:
                "Expand the query with useful terminology, synonyms, "
                "and related technical concepts.",
            QueryTranslationStrategy.REPHRASE:
                "Rewrite the query with different wording while "
                "preserving its intent.",
            QueryTranslationStrategy.MULTI_QUERY:
                "Generate 3 meaningfully different search queries "
                "for the same information need.",
            QueryTranslationStrategy.DECOMPOSE:
                "Break the information need into independent, "
                "answerable retrieval sub-queries.",
            QueryTranslationStrategy.STEP_BACK:
                "Generalize the query into a broader conceptual "
                "question that provides useful background.",
            QueryTranslationStrategy.HYDE:
                "Generate one concise hypothetical technical document "
                "passage that could answer the query. It is not evidence.",
        }

        instruction: str = instructions[strategy]

        return (
            f"STRATEGY: {strategy.value}\n"
            f"ORIGINAL_QUERY: {query}\n\n"
            f"TASK: {instruction}\n"
            "Return JSON only in this exact shape: "
            '{"variants": ["text 1", "text 2"]}. '
            "Do not include markdown."
        )

    @staticmethod
    def _parse_variants(raw_output: str) -> list[str]:
        try:
            parsed: object = cast(
                object,
                json.loads(raw_output),
            )
        except json.JSONDecodeError as exc:
            raise TranslationError(
                "Translation model returned invalid JSON."
            ) from exc

        payload: dict[str, object] = _as_object_dict(parsed)

        raw_variants: object = payload.get("variants")

        return _as_string_list(raw_variants)

    @staticmethod
    def _variant_kind(
        strategy: QueryTranslationStrategy,
    ) -> QueryVariantKind:
        if strategy == QueryTranslationStrategy.DECOMPOSE:
            return QueryVariantKind.SUBQUERY

        if strategy == QueryTranslationStrategy.HYDE:
            return QueryVariantKind.HYPOTHETICAL_DOCUMENT

        return QueryVariantKind.QUERY


def translate_many(
    service: QueryTranslationService,
    query: str,
    strategies: Sequence[QueryTranslationStrategy],
) -> list[QueryTranslationResult]:
    """Run multiple strategies against one original user query."""
    results: list[QueryTranslationResult] = []

    for strategy in strategies:
        results.append(
            service.translate(
                query=query,
                strategy=strategy,
            )
        )

    return results

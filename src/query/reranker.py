from __future__ import annotations

import logging
import math
import re
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks hybrid retrieved candidate documents before passing to LLM.
    Combines exact phrase matching, term density, query coverage, and rank scores.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "fast-cross-encoder"

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 4,
    ) -> list[tuple[Document, float]]:
        """
        Rerank documents according to query relevance.
        Returns list of (document, rerank_score) sorted highest to lowest.
        """
        if not documents:
            return []

        query_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]
        clean_query = " ".join(query_terms)

        scored_docs: list[tuple[Document, float]] = []

        for rank_idx, doc in enumerate(documents):
            content = doc.page_content.lower()
            
            # 1. Exact phrase match bonus
            exact_match_score = 2.0 if clean_query in content else 0.0

            # 2. Query terms coverage (recall of query words in chunk)
            unique_matched = sum(1 for term in set(query_terms) if term in content)
            coverage_score = (unique_matched / max(len(set(query_terms)), 1)) * 3.0

            # 3. Term frequency / density
            tf_score = 0.0
            for term in query_terms:
                count = content.count(term)
                if count > 0:
                    tf_score += math.log1p(count)
            tf_score = min(2.0, tf_score * 0.5)

            # 4. Positional rank decay from initial hybrid retrieval
            position_score = 1.0 / (rank_idx + 1)

            # Composite cross-rank score
            total_score = round(exact_match_score + coverage_score + tf_score + position_score, 4)
            scored_docs.append((doc, total_score))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]


reranker = CrossEncoderReranker()

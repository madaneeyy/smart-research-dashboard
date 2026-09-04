from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Lexical retriever based on BM25.

    BM25 is particularly useful for:
        - exact function names
        - class names
        - variable names
        - technical terminology
        - file-specific terminology

    Example:
        train_test_split
        RandomForestClassifier
        LogisticRegression
        StandardScaler
    """

    def __init__(
        self,
        chunks: Sequence[Dict[str, Any]],
    ) -> None:
        self.chunks = list(chunks)

        if not self.chunks:
            raise ValueError(
                "BM25Retriever requires at least one chunk."
            )

        # --------------------------------------------------
        # Prepare documents for BM25
        # --------------------------------------------------

        self.tokenized_documents = [
            self._tokenize(
                self._build_search_text(chunk)
            )
            for chunk in self.chunks
        ]

        # Remove completely empty documents
        valid_documents = [
            tokens
            for tokens in self.tokenized_documents
            if tokens
        ]

        if not valid_documents:
            raise ValueError(
                "No searchable content found in chunks."
            )

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    # ======================================================
    # Public API
    # ======================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k chunks using BM25.

        Returns copies of the original chunks with:

            bm25_score

        added to each result.
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        # --------------------------------------------------
        # Tokenize query
        # --------------------------------------------------

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # --------------------------------------------------
        # Calculate BM25 scores
        # --------------------------------------------------

        scores = self.bm25.get_scores(
            query_tokens
        )

        # --------------------------------------------------
        # Rank documents
        # --------------------------------------------------

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: List[Dict[str, Any]] = []

        for index in ranked_indices[:top_k]:

            chunk = dict(
                self.chunks[index]
            )

            chunk["bm25_score"] = float(
                scores[index]
            )

            results.append(chunk)

        return results

    # ======================================================
    # Search text construction
    # ======================================================

    @staticmethod
    def _build_search_text(
        chunk: Dict[str, Any],
    ) -> str:
        """
        Build the text that BM25 searches.

        We deliberately include metadata because repository
        retrieval benefits heavily from exact matches in:

            - path
            - section
            - parent section
            - section path
            - symbol
            - language
            - chunk type

        Content remains the most important component.
        """

        fields = [
            chunk.get("path", ""),
            chunk.get("category", ""),
            chunk.get("section", ""),
            chunk.get("parent_section", ""),
            chunk.get("section_path", ""),
            chunk.get("chunk_type", ""),
            chunk.get("language", ""),
            chunk.get("symbol", ""),
            chunk.get("content", ""),
        ]

        return " ".join(
            str(field)
            for field in fields
            if field
        )

    # ======================================================
    # Tokenization
    # ======================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> List[str]:
        """
        Tokenize repository/code/documentation text.

        Unlike a basic whitespace tokenizer, this preserves
        useful programming identifiers.

        Examples:

            train_test_split
            RandomForestClassifier
            LogisticRegression
            StandardScaler
            _private_method
            fit_transform

        We also create additional subtokens for identifiers
        so that both exact and component-level matching work.
        """

        if not text:
            return []

        text = str(text)

        # --------------------------------------------------
        # Extract identifier-like tokens
        # --------------------------------------------------

        raw_tokens = re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*|"
            r"\d+(?:\.\d+)?",
            text,
        )

        tokens: List[str] = []

        for token in raw_tokens:

            token_lower = token.lower()

            if not token_lower:
                continue

            # --------------------------------------------------
            # Keep complete identifier
            # --------------------------------------------------

            tokens.append(
                token_lower
            )

            # --------------------------------------------------
            # Handle snake_case identifiers
            #
            # train_test_split
            #       ↓
            # train
            # test
            # split
            # --------------------------------------------------

            if "_" in token_lower:

                parts = [
                    part
                    for part in token_lower.split("_")
                    if part
                ]

                tokens.extend(parts)

            # --------------------------------------------------
            # Handle CamelCase identifiers
            #
            # RandomForestClassifier
            #       ↓
            # random
            # forest
            # classifier
            # --------------------------------------------------

            camel_parts = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\d|$)|"
                r"[A-Z]?[a-z]+|"
                r"\d+",
                token,
            )

            if len(camel_parts) > 1:

                tokens.extend(
                    part.lower()
                    for part in camel_parts
                    if part
                )

        return tokens
from __future__ import annotations

from typing import Any, Dict, List, Sequence
import re

import numpy as np

from .bm25_retriever import BM25Retriever
from .retriever import SimpleRetriever


class HybridRetriever:
    """
    Hybrid retrieval pipeline:

        Semantic Retrieval
                +
            BM25 Retrieval
                |
                v
        Reciprocal Rank Fusion
                |
                v
          Candidate Pool
                |
                v
              MMR
                |
                v
          Final Results


    RRF combines semantic and BM25 rankings.

    MMR then balances:

        relevance to query
        +
        diversity between selected chunks


    MMR formula:

        MMR =
            lambda * relevance
            -
            (1 - lambda) * redundancy


    lambda = 1.0
        Pure relevance.

    lambda = 0.9
        Mostly relevance.

    lambda = 0.7
        Balanced relevance/diversity.

    lambda = 0.5
        Stronger diversity.

    lambda = 0.0
        Maximum diversity.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        mmr_lambda: float = 0.7,
        duplicate_threshold: float = 0.97,
        near_duplicate_threshold: float = 0.92,
        metadata_bonus_weight: float = 0.05,
    ) -> None:

        # --------------------------------------------------------
        # Validate weights
        # --------------------------------------------------------

        if semantic_weight < 0:
            raise ValueError(
                "semantic_weight cannot be negative."
            )

        if bm25_weight < 0:
            raise ValueError(
                "bm25_weight cannot be negative."
            )

        if semantic_weight == 0 and bm25_weight == 0:
            raise ValueError(
                "At least one retrieval weight "
                "must be greater than 0."
            )

        # --------------------------------------------------------
        # Validate RRF
        # --------------------------------------------------------

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than 0."
            )

        # --------------------------------------------------------
        # Validate candidate multiplier
        # --------------------------------------------------------

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be "
                "greater than 0."
            )

        # --------------------------------------------------------
        # Validate MMR lambda
        # --------------------------------------------------------

        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(
                "mmr_lambda must be between 0 and 1."
            )

        # --------------------------------------------------------
        # Store configuration
        # --------------------------------------------------------

        self.semantic_weight = float(
            semantic_weight
        )

        self.bm25_weight = float(
            bm25_weight
        )

        self.rrf_k = int(
            rrf_k
        )

        self.candidate_multiplier = int(
            candidate_multiplier
        )

        self.mmr_lambda = float(
            mmr_lambda
        )

        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError("duplicate_threshold must be between 0 and 1.")
        if not 0.0 <= near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be between 0 and 1.")
        if near_duplicate_threshold > duplicate_threshold:
            raise ValueError("near_duplicate_threshold cannot exceed duplicate_threshold.")
        if metadata_bonus_weight < 0.0:
            raise ValueError("metadata_bonus_weight cannot be negative.")

        self.duplicate_threshold = float(duplicate_threshold)
        self.near_duplicate_threshold = float(near_duplicate_threshold)
        self.metadata_bonus_weight = float(metadata_bonus_weight)

    # ============================================================
    # PUBLIC RETRIEVE
    # ============================================================

    def retrieve(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform:

            Semantic Retrieval
                +
            BM25
                |
                v
            RRF
                |
                v
            Candidate Pool
                |
                v
            MMR
                |
                v
            Final Results
        """

        # ========================================================
        # VALIDATION
        # ========================================================

        if not question or not question.strip():
            return []

        if not chunks:
            return []

        chunks = list(chunks)

        if top_k <= 0:
            return []

        top_k = min(
            top_k,
            len(chunks),
        )

        # ========================================================
        # CANDIDATE COUNT
        # ========================================================

        candidate_k = min(
            max(
                top_k * self.candidate_multiplier,
                10,
            ),
            len(chunks),
        )

        # ========================================================
        # SEMANTIC RETRIEVAL
        # ========================================================

        semantic_results = (
            SimpleRetriever.retrieve(
                question=question,
                chunks=chunks,
                top_k=candidate_k,
            )
        )

        # ========================================================
        # BM25 RETRIEVAL
        # ========================================================

        bm25_retriever = BM25Retriever(
            chunks
        )

        bm25_results = (
            bm25_retriever.retrieve(
                query=question,
                top_k=candidate_k,
            )
        )

        # ========================================================
        # RECIPROCAL RANK FUSION
        # ========================================================

        fused: Dict[
            str,
            Dict[str, Any],
        ] = {}

        # --------------------------------------------------------
        # Semantic ranking
        # --------------------------------------------------------

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):

            document_id = (
                self._document_id(
                    result
                )
            )

            if document_id not in fused:

                fused[document_id] = {
                    "result": dict(
                        result
                    ),
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "hybrid_score": 0.0,
                }

            fused[document_id][
                "semantic_rank"
            ] = rank

            fused[document_id][
                "hybrid_score"
            ] += (
                self.semantic_weight
                / (
                    self.rrf_k
                    + rank
                )
            )

        # --------------------------------------------------------
        # BM25 ranking
        # --------------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            document_id = (
                self._document_id(
                    result
                )
            )

            if document_id not in fused:

                fused[document_id] = {
                    "result": dict(
                        result
                    ),
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "hybrid_score": 0.0,
                }

            # Merge any BM25-specific metadata.
            fused_result = fused[
                document_id
            ][
                "result"
            ]

            for key, value in result.items():

                if key not in fused_result:

                    fused_result[key] = value

            fused[document_id][
                "bm25_rank"
            ] = rank

            fused[document_id][
                "hybrid_score"
            ] += (
                self.bm25_weight
                / (
                    self.rrf_k
                    + rank
                )
            )

        # ========================================================
        # SORT RRF RESULTS
        # ========================================================

        ranked = sorted(
            fused.values(),
            key=lambda item: item[
                "hybrid_score"
            ],
            reverse=True,
        )

        # ========================================================
        # BUILD CANDIDATE LIST
        # ========================================================

        candidates: List[
            Dict[str, Any]
        ] = []

        for item in ranked:

            result = dict(
                item["result"]
            )

            result[
                "hybrid_score"
            ] = float(
                item[
                    "hybrid_score"
                ]
            )

            result[
                "semantic_rank"
            ] = item[
                "semantic_rank"
            ]

            result[
                "bm25_rank"
            ] = item[
                "bm25_rank"
            ]

            candidates.append(
                result
            )

        # ========================================================
        # MMR RERANK
        # ========================================================

        return self._mmr_rerank(
            question=question,
            candidates=candidates,
            top_k=top_k,
        )

    # ============================================================
    # MMR RERANKING
    # ============================================================

    def _mmr_rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Metadata-aware MMR reranking.

        The original MMR only considered semantic similarity.  This version
        additionally considers:

        * exact duplicate protection
        * near-duplicate protection
        * source/section metadata
        * technical-query lexical matches
        * complementary chunks from different sections

        A chunk from the same document is NOT automatically treated as a
        duplicate.  Different sections can contain useful complementary
        information and are therefore allowed to survive MMR.
        """
        if not candidates:
            return []

        top_k = min(top_k, len(candidates))
        model = SimpleRetriever._get_model()

        query_embedding = np.asarray(
            model.encode(
                question,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

        candidate_texts = [
            str(candidate.get("content", "")) for candidate in candidates
        ]
        candidate_embeddings = np.asarray(
            model.encode(
                candidate_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        if len(candidate_embeddings) != len(candidates):
            raise ValueError(
                "Number of candidate embeddings does not match number of candidates."
            )

        relevance_scores = np.dot(candidate_embeddings, query_embedding)
        query_type = self._classify_query(question)
        query_terms = self._query_terms(question)

        # Exact duplicate content is removed before MMR.  This is deliberately
        # stricter than source-level filtering: two chunks from the same file
        # can still be useful if their sections differ.
        seen_content = set()
        duplicate_indices = set()
        for i, candidate in enumerate(candidates):
            normalized = self._normalize_text(candidate.get("content", ""))
            if normalized and normalized in seen_content:
                duplicate_indices.add(i)
            elif normalized:
                seen_content.add(normalized)

        selected_indices: List[int] = []
        remaining_indices = [
            i for i in range(len(candidates)) if i not in duplicate_indices
        ]

        while remaining_indices and len(selected_indices) < top_k:
            best_index = None
            best_score = -float("inf")
            best_relevance = 0.0
            best_redundancy = 0.0
            best_metadata_bonus = 0.0
            best_relationship = "independent"

            for index in remaining_indices:
                relevance = float(relevance_scores[index])

                if not selected_indices:
                    redundancy = 0.0
                    relationship = "first"
                else:
                    similarities = np.dot(
                        candidate_embeddings[selected_indices],
                        candidate_embeddings[index],
                    )
                    max_pos = int(np.argmax(similarities))
                    raw_redundancy = float(similarities[max_pos])
                    selected_index = selected_indices[max_pos]

                    relationship = self._metadata_relationship(
                        candidates[index], candidates[selected_index]
                    )
                    redundancy = self._adjust_redundancy(
                        raw_redundancy,
                        relationship,
                    )

                    # Very high semantic similarity means the chunks are
                    # effectively duplicates even when their metadata differs.
                    if raw_redundancy >= self.near_duplicate_threshold:
                        redundancy = max(redundancy, raw_redundancy)

                metadata_bonus = self._metadata_bonus(
                    question=question,
                    query_type=query_type,
                    query_terms=query_terms,
                    candidate=candidates[index],
                    selected=[candidates[i] for i in selected_indices],
                )

                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * redundancy
                    + self.metadata_bonus_weight * metadata_bonus
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index
                    best_relevance = relevance
                    best_redundancy = redundancy
                    best_metadata_bonus = metadata_bonus
                    best_relationship = relationship

            if best_index is None:
                break

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        # Build final results and expose diagnostics for your experiments.
        results: List[Dict[str, Any]] = []
        selected_so_far: List[int] = []

        for index in selected_indices:
            relevance = float(relevance_scores[index])

            if not selected_so_far:
                redundancy = 0.0
                relationship = "first"
            else:
                similarities = np.dot(
                    candidate_embeddings[selected_so_far],
                    candidate_embeddings[index],
                )
                max_pos = int(np.argmax(similarities))
                raw_redundancy = float(similarities[max_pos])
                previous_index = selected_so_far[max_pos]
                relationship = self._metadata_relationship(
                    candidates[index], candidates[previous_index]
                )
                redundancy = self._adjust_redundancy(
                    raw_redundancy,
                    relationship,
                )
                if raw_redundancy >= self.near_duplicate_threshold:
                    redundancy = max(redundancy, raw_redundancy)

            metadata_bonus = self._metadata_bonus(
                question=question,
                query_type=query_type,
                query_terms=query_terms,
                candidate=candidates[index],
                selected=[candidates[i] for i in selected_so_far],
            )

            mmr_score = (
                self.mmr_lambda * relevance
                - (1.0 - self.mmr_lambda) * redundancy
                + self.metadata_bonus_weight * metadata_bonus
            )

            result = dict(candidates[index])
            result["mmr_score"] = float(mmr_score)
            result["mmr_relevance"] = float(relevance)
            result["mmr_redundancy"] = float(redundancy)
            result["mmr_lambda"] = float(self.mmr_lambda)
            result["query_type"] = query_type
            result["metadata_bonus"] = float(metadata_bonus)
            result["metadata_relationship"] = relationship
            result["duplicate_protected"] = False

            results.append(result)
            selected_so_far.append(index)

        return results

    # ============================================================
    # METADATA-AWARE RERANKING HELPERS
    # ============================================================

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """Normalize text for exact duplicate detection."""
        text = str(value or "").lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _query_terms(question: str) -> set[str]:
        """Extract useful lexical terms from the query."""
        return {
            token
            for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower())
            if len(token) >= 3
        }

    @classmethod
    def _classify_query(cls, question: str) -> str:
        """Classify a query as technical/exact or conceptual."""
        q = question.lower().strip()
        conceptual_markers = (
            "how does",
            "how do",
            "why does",
            "why do",
            "explain",
            "what is",
            "what are",
            "difference between",
            "compare",
            "overview",
            "concept",
        )
        if any(marker in q for marker in conceptual_markers):
            return "conceptual"
        return "technical"

    @staticmethod
    def _source(chunk: Dict[str, Any]) -> str:
        return str(
            chunk.get("path")
            or chunk.get("source")
            or chunk.get("file")
            or ""
        ).strip().lower()

    @staticmethod
    def _section(chunk: Dict[str, Any]) -> str:
        return str(
            chunk.get("section")
            or chunk.get("heading")
            or chunk.get("title")
            or ""
        ).strip().lower()

    @classmethod
    def _metadata_relationship(
        cls,
        candidate: Dict[str, Any],
        selected: Dict[str, Any],
    ) -> str:
        """Describe how two chunks relate using source/section metadata."""
        same_source = bool(cls._source(candidate)) and (
            cls._source(candidate) == cls._source(selected)
        )
        same_section = bool(cls._section(candidate)) and (
            cls._section(candidate) == cls._section(selected)
        )

        if same_source and same_section:
            return "same_source_same_section"
        if same_source:
            return "same_source_different_section"
        if same_section:
            return "different_source_same_section"
        return "independent"

    @staticmethod
    def _adjust_redundancy(
        similarity: float,
        relationship: str,
    ) -> float:
        """Adjust semantic redundancy using metadata.

        Same section is penalized most because chunks are likely to overlap.
        Different sections in the same document are treated as complementary
        unless their semantic similarity is extremely high.
        """
        if relationship == "same_source_same_section":
            factor = 1.00
        elif relationship == "same_source_different_section":
            factor = 0.55
        elif relationship == "different_source_same_section":
            factor = 0.75
        else:
            factor = 0.85
        return float(similarity * factor)

    @classmethod
    def _metadata_bonus(
        cls,
        question: str,
        query_type: str,
        query_terms: set[str],
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]],
    ) -> float:
        """Return a small 0..1 metadata/lexical bonus.

        The bonus is intentionally small: metadata should guide MMR, not
        overpower semantic relevance.
        """
        content = cls._normalize_text(candidate.get("content", ""))
        section = cls._section(candidate)
        source = cls._source(candidate)

        if not content:
            return 0.0

        candidate_terms = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", content))
        overlap = len(query_terms & candidate_terms) / max(len(query_terms), 1)

        bonus = min(overlap, 1.0) * 0.65

        # Exact technical terms are especially valuable for implementation /
        # API-style queries.
        if query_type == "technical":
            technical_markers = (
                "implementation",
                "api",
                "parameter",
                "parameters",
                "class",
                "function",
                "method",
                "syntax",
                "code",
                "example",
            )
            if any(marker in question.lower() for marker in technical_markers):
                if overlap >= 0.25:
                    bonus += 0.20

        # A different section after we already selected another section from
        # the same source is useful complementary context.
        if selected and source:
            same_source_selected = [
                item for item in selected if cls._source(item) == source
            ]
            if same_source_selected and section:
                selected_sections = {
                    cls._section(item) for item in same_source_selected
                }
                if section not in selected_sections:
                    bonus += 0.15

        return float(min(bonus, 1.0))

    # ============================================================
    # DOCUMENT ID
    # ============================================================

    @staticmethod
    def _document_id(
        chunk: Dict[str, Any],
    ) -> str:
        """
        Generate a stable identifier for a chunk.
        """

        path = str(
            chunk.get(
                "path",
                "",
            )
        )

        chunk_index = chunk.get(
            "chunk_index",
            None,
        )

        if path or chunk_index is not None:

            return (
                f"{path}|"
                f"{chunk_index}"
            )

        # Fallback to content.
        return str(
            chunk.get(
                "content",
                "",
            )
        )

    # ============================================================
    # DEBUG FORMATTER
    # ============================================================

    @staticmethod
    def format_result(
        result: Dict[str, Any],
    ) -> str:
        """
        Format one retrieval result for debugging.
        """

        path = result.get(
            "path",
            "",
        )

        section = result.get(
            "section",
            "",
        )

        semantic_rank = result.get(
            "semantic_rank",
            None,
        )

        bm25_rank = result.get(
            "bm25_rank",
            None,
        )

        similarity = result.get(
            "similarity",
            None,
        )

        bm25_score = result.get(
            "bm25_score",
            None,
        )

        hybrid_score = result.get(
            "hybrid_score",
            None,
        )

        mmr_score = result.get(
            "mmr_score",
            None,
        )

        mmr_relevance = result.get(
            "mmr_relevance",
            None,
        )

        mmr_redundancy = result.get(
            "mmr_redundancy",
            None,
        )

        mmr_lambda = result.get(
            "mmr_lambda",
            None,
        )

        return (
            f"path={path!r}, "
            f"section={section!r}, "
            f"semantic_rank={semantic_rank}, "
            f"bm25_rank={bm25_rank}, "
            f"similarity={similarity}, "
            f"bm25_score={bm25_score}, "
            f"hybrid_score={hybrid_score}, "
            f"mmr_score={mmr_score}, "
            f"mmr_relevance={mmr_relevance}, "
            f"mmr_redundancy={mmr_redundancy}, "
            f"mmr_lambda={mmr_lambda}"
        )
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Tuple
import re

import numpy as np

from .bm25_retriever import BM25Retriever
from .retriever import SimpleRetriever


class HybridRetriever:
    """
    Production-oriented hybrid RAG retriever.

    Pipeline
    --------
    1. Semantic retrieval
    2. BM25 retrieval
    3. Weighted Reciprocal Rank Fusion (RRF)
    4. Relevance filtering
    5. Exact duplicate removal
    6. Query-aware relevance scoring
    7. Protected high-relevance selection
    8. Complementarity-aware MMR selection
    9. Final relevance safety check

    Design goals
    ------------
    The retriever is optimized for a research assistant where the primary
    objective is NOT simply diversity.

    Priority order:

        relevance
            >
        exact technical matching
            >
        supporting/complementary evidence
            >
        diversity

    Important principles
    --------------------
    * MMR must not rescue weak candidates.
    * Exact technical/API identifiers should receive lexical protection.
    * Different sections of the same document may be complementary.
    * Exact duplicate content should never be returned twice.
    * Strong primary evidence should be protected from aggressive MMR.
    * Weak candidates should be allowed to disappear instead of forcing top-k.
    * Metadata should guide ranking, not overpower semantic relevance.
    """

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(
        self,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        mmr_lambda: float = 0.75,

        # ------------------------------------------------------------
        # Relevance filtering
        # ------------------------------------------------------------
        relevance_filter_enabled: bool = True,
        relevance_threshold: float = 0.30,
        relevance_relative_threshold: float = 0.70,

        # ------------------------------------------------------------
        # Duplicate protection
        # ------------------------------------------------------------
        near_duplicate_threshold: float = 0.92,

        # ------------------------------------------------------------
        # Metadata / query-aware scoring
        # ------------------------------------------------------------
        metadata_bonus_weight: float = 0.08,
        lexical_bonus_weight: float = 0.12,
        bm25_presence_bonus: float = 0.04,

        # ------------------------------------------------------------
        # Complementarity
        # ------------------------------------------------------------
        complementarity_bonus_weight: float = 0.10,

        # ------------------------------------------------------------
        # Primary-result protection
        # ------------------------------------------------------------
        protected_primary_count: int = 2,
        protected_primary_margin: float = 0.08,

        # ------------------------------------------------------------
        # Final result safety
        # ------------------------------------------------------------
        minimum_results: int = 1,

        max_chunks_per_source: int = 2,
        same_source_penalty: float = 0.10,
    ) -> None:

        # ============================================================
        # VALIDATION
        # ============================================================

        if semantic_weight < 0:
            raise ValueError("semantic_weight cannot be negative.")

        if bm25_weight < 0:
            raise ValueError("bm25_weight cannot be negative.")

        if semantic_weight == 0 and bm25_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than 0."
            )

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0.")

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater than 0."
            )

        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(
                "mmr_lambda must be between 0 and 1."
            )

        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError(
                "relevance_threshold must be between 0 and 1."
            )

        if not 0.0 <= relevance_relative_threshold <= 1.0:
            raise ValueError(
                "relevance_relative_threshold must be between 0 and 1."
            )

        if not 0.0 <= near_duplicate_threshold <= 1.0:
            raise ValueError(
                "near_duplicate_threshold must be between 0 and 1."
            )

        if metadata_bonus_weight < 0:
            raise ValueError(
                "metadata_bonus_weight cannot be negative."
            )

        if lexical_bonus_weight < 0:
            raise ValueError(
                "lexical_bonus_weight cannot be negative."
            )

        if bm25_presence_bonus < 0:
            raise ValueError(
                "bm25_presence_bonus cannot be negative."
            )

        if complementarity_bonus_weight < 0:
            raise ValueError(
                "complementarity_bonus_weight cannot be negative."
            )

        if protected_primary_count < 0:
            raise ValueError(
                "protected_primary_count cannot be negative."
            )

        if not 0.0 <= protected_primary_margin <= 1.0:
            raise ValueError(
                "protected_primary_margin must be between 0 and 1."
            )

        if minimum_results < 0:
            raise ValueError(
                "minimum_results cannot be negative."
            )

        if max_chunks_per_source <= 0:
            raise ValueError("max_chunks_per_source must be greater than 0.")

        if same_source_penalty < 0:
            raise ValueError("same_source_penalty cannot be negative.")

        # ============================================================
        # STORE CONFIGURATION
        # ============================================================

        self.semantic_weight = float(semantic_weight)
        self.bm25_weight = float(bm25_weight)
        self.rrf_k = int(rrf_k)

        self.candidate_multiplier = int(candidate_multiplier)

        self.mmr_lambda = float(mmr_lambda)

        self.relevance_filter_enabled = bool(
            relevance_filter_enabled
        )

        self.relevance_threshold = float(
            relevance_threshold
        )

        self.relevance_relative_threshold = float(
            relevance_relative_threshold
        )

        self.near_duplicate_threshold = float(
            near_duplicate_threshold
        )

        self.metadata_bonus_weight = float(
            metadata_bonus_weight
        )

        self.lexical_bonus_weight = float(
            lexical_bonus_weight
        )

        self.bm25_presence_bonus = float(
            bm25_presence_bonus
        )

        self.complementarity_bonus_weight = float(
            complementarity_bonus_weight
        )

        self.protected_primary_count = int(
            protected_primary_count
        )

        self.protected_primary_margin = float(
            protected_primary_margin
        )

        self.minimum_results = int(
            minimum_results
        )

        self.max_chunks_per_source = int(
            max_chunks_per_source
        )

        self.same_source_penalty = float(
            same_source_penalty
        )

    # ================================================================
    # PUBLIC RETRIEVE
    # ================================================================

    def retrieve(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most useful chunks for a query.

        The returned list can contain fewer than top_k results if weak
        candidates do not pass the final relevance requirements.
        """

        if not question or not question.strip():
            return []

        if not chunks:
            return []

        chunks = list(chunks)

        if top_k <= 0:
            return []

        top_k = min(top_k, len(chunks))

        # ============================================================
        # CANDIDATE POOL
        # ============================================================

        candidate_k = min(
            max(
                top_k * self.candidate_multiplier,
                10,
            ),
            len(chunks),
        )

        # ============================================================
        # SEMANTIC RETRIEVAL
        # ============================================================

        semantic_results = SimpleRetriever.retrieve(
            question=question,
            chunks=chunks,
            top_k=candidate_k,
        )

        # ============================================================
        # BM25 RETRIEVAL
        # ============================================================

        bm25_retriever = BM25Retriever(chunks)

        bm25_results = bm25_retriever.retrieve(
            query=question,
            top_k=candidate_k,
        )

        # ============================================================
        # RRF
        # ============================================================

        fused: Dict[str, Dict[str, Any]] = {}

        # ------------------------------------------------------------
        # Semantic ranking
        # ------------------------------------------------------------

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):
            document_id = self._document_id(result)

            if document_id not in fused:
                fused[document_id] = {
                    "result": dict(result),
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "hybrid_score": 0.0,
                }

            fused[document_id]["semantic_rank"] = rank

            fused[document_id]["hybrid_score"] += (
                self.semantic_weight
                / (self.rrf_k + rank)
            )

        # ------------------------------------------------------------
        # BM25 ranking
        # ------------------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            document_id = self._document_id(result)

            if document_id not in fused:
                fused[document_id] = {
                    "result": dict(result),
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "hybrid_score": 0.0,
                }

            fused_result = fused[document_id]["result"]

            # Preserve metadata from both retrievers.
            for key, value in result.items():
                if key not in fused_result:
                    fused_result[key] = value

            fused[document_id]["bm25_rank"] = rank

            fused[document_id]["hybrid_score"] += (
                self.bm25_weight
                / (self.rrf_k + rank)
            )

        # ============================================================
        # RRF SORT
        # ============================================================

        ranked = sorted(
            fused.values(),
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        # ============================================================
        # BUILD CANDIDATES
        # ============================================================

        candidates: List[Dict[str, Any]] = []

        for item in ranked:
            result = dict(item["result"])

            result["hybrid_score"] = float(
                item["hybrid_score"]
            )

            result["semantic_rank"] = item[
                "semantic_rank"
            ]

            result["bm25_rank"] = item[
                "bm25_rank"
            ]

            candidates.append(result)

        # ============================================================
        # RERANK
        # ============================================================

        return self._rerank(
            question=question,
            candidates=candidates,
            top_k=top_k,
        )

    # ================================================================
    # MAIN RERANKER
    # ================================================================

    def _rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:

        if not candidates:
            return []

        top_k = min(top_k, len(candidates))

        model = SimpleRetriever._get_model()

        # ============================================================
        # EMBEDDINGS
        # ============================================================

        query_embedding = np.asarray(
            model.encode(
                question,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

        candidate_texts = [
            str(
                candidate.get(
                    "content",
                    "",
                )
            )
            for candidate in candidates
        ]

        # Reuse the same persistent per-chunk embedding cache used by
        # SimpleRetriever.retrieve(). Do NOT call model.encode() directly
        # here, otherwise the RRF candidate set gets embedded a second time.
        candidate_embeddings = SimpleRetriever.get_embeddings(
            candidates
        )

        if len(candidate_embeddings) != len(candidates):
            raise ValueError(
                "Number of candidate embeddings does not "
                "match number of candidates."
            )

        # ============================================================
        # BASE SEMANTIC RELEVANCE
        # ============================================================

        semantic_scores = np.dot(
            candidate_embeddings,
            query_embedding,
        )

        query_type = self._classify_query(question)
        query_terms = self._query_terms(question)

        # ============================================================
        # RELEVANCE FILTER
        # ============================================================

        filtered_indices, filter_diagnostics = (
            self._relevance_filter(
                question=question,
                candidates=candidates,
                semantic_scores=semantic_scores,
                query_type=query_type,
                query_terms=query_terms,
            )
        )

        # ============================================================
        # EXACT DUPLICATE REMOVAL
        # ============================================================

        filtered_indices = self._remove_exact_duplicates(
            candidates=candidates,
            indices=filtered_indices,
        )

        filtered_indices = self._remove_near_duplicates(
            candidates=candidates,
            indices=filtered_indices,
            candidate_embeddings=candidate_embeddings,
        )

        if not filtered_indices:
            strongest = int(
                np.argmax(semantic_scores)
            )
            filtered_indices = [strongest]

        # ============================================================
        # QUERY-AWARE BASE SCORES
        # ============================================================

        query_scores: Dict[int, float] = {}

        for index in filtered_indices:

            score = self._query_aware_score(
                question=question,
                query_type=query_type,
                query_terms=query_terms,
                candidate=candidates[index],
                semantic_score=float(
                    semantic_scores[index]
                ),
            )

            query_scores[index] = score

        # ============================================================
        # SORT BY QUERY-AWARE RELEVANCE
        # ============================================================

        relevance_order = sorted(
            filtered_indices,
            key=lambda idx: query_scores[idx],
            reverse=True,
        )

        # ============================================================
        # PRIMARY PROTECTION
        # ============================================================
        #
        # We do NOT let MMR immediately throw away the strongest
        # evidence.
        #
        # This directly addresses one of our biggest findings:
        #
        #   MMR can improve diversity while hurting useful coverage.
        #
        # The strongest primary evidence is therefore protected.
        # ============================================================

        protected_indices = self._protected_primary_indices(
            relevance_order=relevance_order,
            query_scores=query_scores,
            top_k=top_k,
        )

        # ============================================================
        # COMPLEMENTARITY-AWARE SELECTION
        # ============================================================

        selected_indices: List[int] = []

        # ------------------------------------------------------------
        # First select protected primary chunks.
        # ------------------------------------------------------------

        for index in protected_indices:
            if len(selected_indices) >= top_k:
                break

            selected_indices.append(index)

        # ------------------------------------------------------------
        # Then select remaining chunks using controlled MMR.
        # ------------------------------------------------------------

        remaining_indices = [
            index
            for index in relevance_order
            if index not in selected_indices
        ]

        while (
            remaining_indices
            and len(selected_indices) < top_k
        ):

            best_index = None
            best_score = -float("inf")

            best_relevance = 0.0
            best_redundancy = 0.0
            best_metadata_bonus = 0.0
            best_complementarity = 0.0
            best_relationship = "independent"

            for index in remaining_indices:

                relevance = query_scores[index]

                candidate_source = self._source(candidates[index])
                same_source_count = sum(
                    1
                    for selected_index in selected_indices
                    if candidate_source
                    and candidate_source == self._source(candidates[selected_index])
                )

                # ------------------------------------------------
                # Redundancy against already-selected chunks.
                # ------------------------------------------------

                if not selected_indices:

                    redundancy = 0.0
                    relationship = "first"

                else:

                    similarities = np.dot(
                        candidate_embeddings[
                            selected_indices
                        ],
                        candidate_embeddings[index],
                    )

                    max_pos = int(
                        np.argmax(similarities)
                    )

                    raw_redundancy = float(
                        similarities[max_pos]
                    )

                    selected_index = (
                        selected_indices[max_pos]
                    )

                    relationship = (
                        self._metadata_relationship(
                            candidates[index],
                            candidates[
                                selected_index
                            ],
                        )
                    )

                    redundancy = (
                        self._adjust_redundancy(
                            raw_redundancy,
                            relationship,
                        )
                    )

                    # Very high similarity overrides metadata.
                    if (
                        raw_redundancy
                        >= self.near_duplicate_threshold
                    ):
                        redundancy = max(
                            redundancy,
                            raw_redundancy,
                        )

                # ------------------------------------------------
                # Metadata relevance
                # ------------------------------------------------

                metadata_bonus = (
                    self._metadata_bonus(
                        question=question,
                        query_type=query_type,
                        query_terms=query_terms,
                        candidate=candidates[index],
                        selected=[
                            candidates[i]
                            for i in selected_indices
                        ],
                    )
                )

                # ------------------------------------------------
                # Complementarity
                # ------------------------------------------------

                complementarity = (
                    self._complementarity_score(
                        candidate=candidates[index],
                        selected=[
                            candidates[i]
                            for i in selected_indices
                        ],
                        query_type=query_type,
                    )
                )

                # ------------------------------------------------
                # Controlled MMR
                # ------------------------------------------------

                mmr_score = (
                    self.mmr_lambda
                    * relevance
                    - (
                        1.0
                        - self.mmr_lambda
                    )
                    * redundancy
                    + self.metadata_bonus_weight
                    * metadata_bonus
                    + self.complementarity_bonus_weight
                    * complementarity
                )

                if same_source_count >= self.max_chunks_per_source:
                    mmr_score -= self.same_source_penalty * (
                        1.0 + same_source_count - self.max_chunks_per_source
                    )

                # ------------------------------------------------
                # Do not allow a much weaker chunk to win solely
                # because it is diverse.
                # ------------------------------------------------

                if selected_indices:

                    strongest_selected_score = max(
                        query_scores[i]
                        for i in selected_indices
                    )

                    minimum_allowed = (
                        strongest_selected_score
                        - self.protected_primary_margin
                    )

                    if relevance < minimum_allowed:
                        # Diversity cannot rescue a chunk that is
                        # substantially weaker than the selected
                        # evidence.
                        mmr_score -= 0.10

                if mmr_score > best_score:

                    best_score = mmr_score
                    best_index = index

                    best_relevance = relevance
                    best_redundancy = redundancy
                    best_metadata_bonus = metadata_bonus
                    best_complementarity = complementarity
                    best_relationship = relationship

            if best_index is None:
                break

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        # ============================================================
        # FINAL RESULT CONSTRUCTION
        # ============================================================

        results: List[Dict[str, Any]] = []

        for rank, index in enumerate(
            selected_indices,
            start=1,
        ):

            relevance = query_scores[index]

            # --------------------------------------------------------
            # Recalculate redundancy diagnostics.
            # --------------------------------------------------------

            if rank == 1:

                redundancy = 0.0
                relationship = "first"

            else:

                previous_indices = selected_indices[
                    : rank - 1
                ]

                similarities = np.dot(
                    candidate_embeddings[
                        previous_indices
                    ],
                    candidate_embeddings[index],
                )

                max_pos = int(
                    np.argmax(similarities)
                )

                raw_redundancy = float(
                    similarities[max_pos]
                )

                previous_index = previous_indices[
                    max_pos
                ]

                relationship = (
                    self._metadata_relationship(
                        candidates[index],
                        candidates[
                            previous_index
                        ],
                    )
                )

                redundancy = (
                    self._adjust_redundancy(
                        raw_redundancy,
                        relationship,
                    )
                )

                if (
                    raw_redundancy
                    >= self.near_duplicate_threshold
                ):
                    redundancy = max(
                        redundancy,
                        raw_redundancy,
                    )

            metadata_bonus = (
                self._metadata_bonus(
                    question=question,
                    query_type=query_type,
                    query_terms=query_terms,
                    candidate=candidates[index],
                    selected=[
                        candidates[i]
                        for i in selected_indices[: rank - 1]
                    ],
                )
            )

            complementarity = (
                self._complementarity_score(
                    candidate=candidates[index],
                    selected=[
                        candidates[i]
                        for i in selected_indices[: rank - 1]
                    ],
                    query_type=query_type,
                )
            )

            final_score = (
                self.mmr_lambda
                * relevance
                - (
                    1.0
                    - self.mmr_lambda
                )
                * redundancy
                + self.metadata_bonus_weight
                * metadata_bonus
                + self.complementarity_bonus_weight
                * complementarity
            )

            result = dict(
                candidates[index]
            )

            # ========================================================
            # DIAGNOSTICS
            # ========================================================

            result["retrieval_rank"] = rank

            result["query_type"] = query_type

            result["query_relevance_score"] = float(
                relevance
            )

            result["mmr_score"] = float(
                final_score
            )

            result["mmr_relevance"] = float(
                relevance
            )

            result["mmr_redundancy"] = float(
                redundancy
            )

            result["mmr_lambda"] = float(
                self.mmr_lambda
            )

            result["metadata_bonus"] = float(
                metadata_bonus
            )

            result["complementarity_score"] = float(
                complementarity
            )

            result["metadata_relationship"] = (
                relationship
            )

            result["primary_protected"] = (
                index in protected_indices
            )

            result["relevance_filter_enabled"] = (
                self.relevance_filter_enabled
            )

            result["relevance_threshold_used"] = (
                filter_diagnostics[
                    "threshold_used"
                ]
            )

            result["candidate_pool_size"] = (
                len(candidates)
            )

            result["post_filter_pool_size"] = (
                len(filtered_indices)
            )

            result["exact_duplicates_removed"] = (
                filter_diagnostics[
                    "exact_duplicates_removed"
                ]
            )

            result["semantic_score"] = float(
                semantic_scores[index]
            )

            # --------------------------------------------------------
            # Technical-query diagnostics
            # --------------------------------------------------------

            result["lexical_overlap"] = float(
                self._lexical_overlap(
                    query_terms,
                    candidates[index],
                )
            )

            result["technical_match_protected"] = (
                self._is_exact_technical_match(
                    query_type=query_type,
                    query_terms=query_terms,
                    candidate=candidates[index],
                )
            )

            result["symbol_path_match"] = float(
                self._symbol_path_match(
                    query_terms=query_terms,
                    candidate=candidates[index],
                )
            )

            results.append(result)

        # ============================================================
        # FINAL SAFETY FILTER
        # ============================================================

        results = self._final_relevance_safety_filter(
            results=results,
            semantic_scores=semantic_scores,
            candidates=candidates,
            minimum_results=self.minimum_results,
        )

        return results

    # ================================================================
    # RELEVANCE FILTER
    # ================================================================

    def _relevance_filter(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        semantic_scores: np.ndarray,
        query_type: str,
        query_terms: Set[str],
    ) -> Tuple[List[int], Dict[str, Any]]:

        if not candidates:
            return [], {
                "threshold_used": None,
                "exact_duplicates_removed": 0,
            }

        if not self.relevance_filter_enabled:
            return list(range(len(candidates))), {
                "threshold_used": None,
                "exact_duplicates_removed": 0,
            }

        top_relevance = float(
            np.max(semantic_scores)
        )

        absolute_cutoff = float(
            self.relevance_threshold
        )

        relative_cutoff = float(
            top_relevance
            * self.relevance_relative_threshold
        )

        threshold_used = max(
            absolute_cutoff,
            relative_cutoff,
        )

        keep_indices: List[int] = []

        for index, candidate in enumerate(
            candidates
        ):

            semantic_ok = (
                float(
                    semantic_scores[index]
                )
                >= threshold_used
            )

            # --------------------------------------------------------
            # Exact technical protection.
            # --------------------------------------------------------

            technical_match = (
                self._is_exact_technical_match(
                    query_type=query_type,
                    query_terms=query_terms,
                    candidate=candidate,
                )
            )

            # --------------------------------------------------------
            # Strong BM25 result with reasonable semantic support.
            #
            # This protects exact terminology without allowing
            # completely unrelated BM25 matches through.
            # --------------------------------------------------------

            bm25_protected = (
                candidate.get("bm25_rank") is not None
                and float(
                    semantic_scores[index]
                )
                >= max(
                    0.20,
                    threshold_used * 0.75,
                )
                and self._lexical_overlap(
                    query_terms,
                    candidate,
                )
                >= 0.30
            )

            if (
                semantic_ok
                or technical_match
                or bm25_protected
            ):
                keep_indices.append(index)

        # Never erase the whole pool.
        if not keep_indices:

            strongest = int(
                np.argmax(semantic_scores)
            )

            keep_indices = [strongest]

        return keep_indices, {
            "threshold_used": float(
                threshold_used
            ),
            "exact_duplicates_removed": 0,
        }

    # ================================================================
    # EXACT DUPLICATES
    # ================================================================

    def _remove_exact_duplicates(
        self,
        candidates: List[Dict[str, Any]],
        indices: List[int],
    ) -> List[int]:

        seen: Set[str] = set()
        result: List[int] = []

        for index in indices:

            normalized = self._normalize_text(
                candidates[index].get(
                    "content",
                    "",
                )
            )

            if not normalized:
                result.append(index)
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            result.append(index)

        return result

    # ================================================================
    # NEAR-DUPLICATE REMOVAL
    # ================================================================

    @classmethod
    def _remove_near_duplicates(
        cls,
        candidates: List[Dict[str, Any]],
        indices: List[int],
        candidate_embeddings: np.ndarray,
    ) -> List[int]:
        if len(indices) <= 1:
            return list(indices)

        kept: List[int] = []
        for index in indices:
            duplicate = False
            for kept_index in kept:
                similarity = float(np.dot(
                    candidate_embeddings[index],
                    candidate_embeddings[kept_index],
                ))
                if similarity >= 0.97:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(index)

        return kept

    # ================================================================
    # QUERY-AWARE RELEVANCE SCORE
    # ================================================================

    def _query_aware_score(
        self,
        question: str,
        query_type: str,
        query_terms: Set[str],
        candidate: Dict[str, Any],
        semantic_score: float,
    ) -> float:

        score = float(semantic_score)

        lexical_overlap = self._lexical_overlap(
            query_terms,
            candidate,
        )

        # ------------------------------------------------------------
        # Lexical relevance
        # ------------------------------------------------------------

        score += (
            self.lexical_bonus_weight
            * lexical_overlap
        )

        # ------------------------------------------------------------
        # Exact technical match
        # ------------------------------------------------------------

        if self._is_exact_technical_match(
            query_type=query_type,
            query_terms=query_terms,
            candidate=candidate,
        ):

            score += 0.08

        # ------------------------------------------------------------
        # Symbol/path matching
        # ------------------------------------------------------------

        symbol_match = self._symbol_path_match(
            query_terms=query_terms,
            candidate=candidate,
        )
        score += 0.12 * symbol_match

        # ------------------------------------------------------------
        # BM25 presence
        # ------------------------------------------------------------

        if candidate.get("bm25_rank") is not None:

            score += self.bm25_presence_bonus

        # ------------------------------------------------------------
        # Metadata relevance
        # ------------------------------------------------------------

        metadata_bonus = self._metadata_bonus(
            question=question,
            query_type=query_type,
            query_terms=query_terms,
            candidate=candidate,
            selected=[],
        )

        score += (
            self.metadata_bonus_weight
            * metadata_bonus
        )

        return float(score)

    # ================================================================
    # PRIMARY PROTECTION
    # ================================================================

    def _protected_primary_indices(
        self,
        relevance_order: List[int],
        query_scores: Dict[int, float],
        top_k: int,
    ) -> List[int]:

        if not relevance_order:
            return []

        count = min(
            self.protected_primary_count,
            top_k,
            len(relevance_order),
        )

        strongest_score = query_scores[
            relevance_order[0]
        ]

        protected: List[int] = []

        for index in relevance_order:

            if len(protected) >= count:
                break

            score = query_scores[index]

            # Only protect chunks reasonably close to the strongest.
            if (
                score
                >= strongest_score
                - self.protected_primary_margin
            ):
                protected.append(index)

        return protected

    # ================================================================
    # COMPLEMENTARITY
    # ================================================================

    @classmethod
    def _complementarity_score(
        cls,
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]],
        query_type: str,
    ) -> float:

        if not selected:
            return 0.0

        candidate_source = cls._source(
            candidate
        )

        candidate_section = cls._section(
            candidate
        )

        if not candidate_source and not candidate_section:
            return 0.0

        score = 0.0

        selected_sources = {
            cls._source(item)
            for item in selected
            if cls._source(item)
        }

        selected_sections = {
            cls._section(item)
            for item in selected
            if cls._section(item)
        }

        # ------------------------------------------------------------
        # Different section within same document.
        #
        # This is particularly useful for conceptual questions.
        # ------------------------------------------------------------

        if (
            candidate_source
            and candidate_source in selected_sources
            and candidate_section
            and candidate_section not in selected_sections
        ):
            score += 0.70

        # ------------------------------------------------------------
        # Different source can provide complementary evidence.
        # ------------------------------------------------------------

        elif (
            candidate_source
            and candidate_source not in selected_sources
        ):
            score += 0.35

        # ------------------------------------------------------------
        # Conceptual questions benefit slightly more from
        # complementary sections.
        # ------------------------------------------------------------

        if query_type == "conceptual":
            score += 0.15

        return float(
            min(score, 1.0)
        )

    # ================================================================
    # METADATA BONUS
    # ================================================================

    @classmethod
    def _metadata_bonus(
        cls,
        question: str,
        query_type: str,
        query_terms: Set[str],
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]],
    ) -> float:

        content = cls._normalize_text(
            candidate.get(
                "content",
                "",
            )
        )

        section = cls._section(
            candidate
        )

        source = cls._source(
            candidate
        )

        if not content:
            return 0.0

        candidate_terms = set(
            re.findall(
                r"[a-zA-Z_][a-zA-Z0-9_]*",
                content,
            )
        )

        overlap = len(
            query_terms & candidate_terms
        ) / max(
            len(query_terms),
            1,
        )

        bonus = (
            min(overlap, 1.0)
            * 0.65
        )

        # ------------------------------------------------------------
        # Technical-query bonus
        # ------------------------------------------------------------

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
                "source",
                "internals",
            )

            if any(
                marker in question.lower()
                for marker in technical_markers
            ):

                if overlap >= 0.25:
                    bonus += 0.20

        # ------------------------------------------------------------
        # Section-title lexical match
        # ------------------------------------------------------------

        if section:

            section_terms = set(
                re.findall(
                    r"[a-zA-Z_][a-zA-Z0-9_]*",
                    section,
                )
            )

            section_overlap = len(
                query_terms & section_terms
            ) / max(
                len(query_terms),
                1,
            )

            bonus += (
                min(section_overlap, 1.0)
                * 0.20
            )

        # ------------------------------------------------------------
        # Complementary same-source section.
        # ------------------------------------------------------------

        if selected and source:

            same_source_selected = [
                item
                for item in selected
                if cls._source(item) == source
            ]

            if (
                same_source_selected
                and section
            ):

                selected_sections = {
                    cls._section(item)
                    for item in same_source_selected
                }

                if section not in selected_sections:
                    bonus += 0.15

        return float(
            min(bonus, 1.0)
        )

    # ================================================================
    # SYMBOL / PATH MATCHING
    # ================================================================

    @classmethod
    def _symbol_path_match(
        cls,
        query_terms: Set[str],
        candidate: Dict[str, Any],
    ) -> float:
        if not query_terms:
            return 0.0

        path = str(candidate.get("path") or candidate.get("file") or "").lower()
        path_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", path))

        symbols = candidate.get("symbols") or []
        symbol_tokens = set()
        exact_symbols = set()
        for symbol in symbols:
            if isinstance(symbol, dict):
                value = str(
                    symbol.get("symbol")
                    or symbol.get("qualified_name")
                    or symbol.get("name")
                    or ""
                ).lower()
            else:
                value = str(symbol).lower()
            if value:
                exact_symbols.add(value)
                symbol_tokens.update(
                    re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", value)
                )

        exact_hits = 0
        token_hits = 0
        for term in query_terms:
            if term in exact_symbols:
                exact_hits += 1
            elif term in symbol_tokens or term in path_tokens:
                token_hits += 1

        return float(min(
            1.0,
            exact_hits / max(len(query_terms), 1)
            + 0.5 * token_hits / max(len(query_terms), 1),
        ))

    # ================================================================
    # TECHNICAL MATCH
    # ================================================================

    @classmethod
    def _is_exact_technical_match(
        cls,
        query_type: str,
        query_terms: Set[str],
        candidate: Dict[str, Any],
    ) -> bool:

        if query_type != "technical":
            return False

        overlap = cls._lexical_overlap(
            query_terms,
            candidate,
        )

        symbol_path_match = cls._symbol_path_match(
            query_terms=query_terms,
            candidate=candidate,
        )

        return overlap >= 0.50 or symbol_path_match >= 0.75

    # ================================================================
    # LEXICAL OVERLAP
    # ================================================================

    @classmethod
    def _lexical_overlap(
        cls,
        query_terms: Set[str],
        candidate: Dict[str, Any],
    ) -> float:

        if not query_terms:
            return 0.0

        content = cls._normalize_text(
            candidate.get(
                "content",
                "",
            )
        )

        candidate_terms = set(
            re.findall(
                r"[a-zA-Z_][a-zA-Z0-9_]*",
                content,
            )
        )

        return float(
            len(
                query_terms
                & candidate_terms
            )
            / max(
                len(query_terms),
                1,
            )
        )

    # ================================================================
    # QUERY CLASSIFICATION
    # ================================================================

    @classmethod
    def _classify_query(
        cls,
        question: str,
    ) -> str:

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
            "differences between",
            "compare",
            "comparison",
            "overview",
            "concept",
            "why",
        )

        if any(
            marker in q
            for marker in conceptual_markers
        ):
            return "conceptual"

        return "technical"

    # ================================================================
    # QUERY TERMS
    # ================================================================

    @staticmethod
    def _query_terms(
        question: str,
    ) -> Set[str]:

        tokens = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*",
            question.lower(),
        )

        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "does",
            "what",
            "how",
            "why",
            "are",
            "is",
            "was",
            "were",
            "before",
            "after",
            "from",
            "into",
            "using",
            "used",
            "use",
            "can",
            "could",
            "would",
            "should",
            "about",
            "does",
            "work",
            "works",
        }

        return {
            token
            for token in tokens
            if len(token) >= 3
            and token not in stopwords
        }

    # ================================================================
    # METADATA HELPERS
    # ================================================================

    @staticmethod
    def _source(
        chunk: Dict[str, Any],
    ) -> str:

        return str(
            chunk.get("path")
            or chunk.get("source")
            or chunk.get("file")
            or ""
        ).strip().lower()

    @staticmethod
    def _section(
        chunk: Dict[str, Any],
    ) -> str:

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

        same_source = bool(
            cls._source(candidate)
        ) and (
            cls._source(candidate)
            == cls._source(selected)
        )

        same_section = bool(
            cls._section(candidate)
        ) and (
            cls._section(candidate)
            == cls._section(selected)
        )

        if (
            same_source
            and same_section
        ):
            return "same_source_same_section"

        if same_source:
            return "same_source_different_section"

        if same_section:
            return "different_source_same_section"

        return "independent"

    # ================================================================
    # REDUNDANCY
    # ================================================================

    @staticmethod
    def _adjust_redundancy(
        similarity: float,
        relationship: str,
    ) -> float:

        # ------------------------------------------------------------
        # Same section = strongest redundancy assumption.
        # ------------------------------------------------------------

        if relationship == (
            "same_source_same_section"
        ):
            factor = 1.00

        # ------------------------------------------------------------
        # Different sections in same source are likely complementary.
        # ------------------------------------------------------------

        elif relationship == (
            "same_source_different_section"
        ):
            factor = 0.55

        # ------------------------------------------------------------
        # Same heading across different sources is moderately
        # redundant.
        # ------------------------------------------------------------

        elif relationship == (
            "different_source_same_section"
        ):
            factor = 0.75

        else:
            factor = 0.85

        return float(
            similarity * factor
        )

    # ================================================================
    # FINAL SAFETY FILTER
    # ================================================================

    def _final_relevance_safety_filter(
        self,
        results: List[Dict[str, Any]],
        semantic_scores: np.ndarray,
        candidates: List[Dict[str, Any]],
        minimum_results: int,
    ) -> List[Dict[str, Any]]:

        if not results:
            return []

        # ------------------------------------------------------------
        # Do not normally return chunks that are substantially below
        # the strongest retrieved evidence.
        #
        # We intentionally use the semantic score here rather than
        # MMR score.  MMR score contains diversity bonuses and should
        # not determine whether a chunk is fundamentally relevant.
        # ------------------------------------------------------------

        strongest_semantic = max(
            float(result.get(
                "semantic_score",
                0.0,
            ))
            for result in results
        )

        safety_cutoff = max(
            self.relevance_threshold * 0.75,
            strongest_semantic
            * self.relevance_relative_threshold
            * 0.75,
        )

        filtered: List[Dict[str, Any]] = []

        for result in results:

            semantic_score = float(
                result.get(
                    "semantic_score",
                    0.0,
                )
            )

            technical_protected = bool(
                result.get(
                    "technical_match_protected",
                    False,
                )
            )

            if (
                semantic_score >= safety_cutoff
                or technical_protected
            ):
                filtered.append(result)

        # ------------------------------------------------------------
        # Never return zero results when a valid candidate exists.
        # ------------------------------------------------------------

        if (
            len(filtered) < minimum_results
            and results
        ):

            for result in results:

                if result not in filtered:
                    filtered.append(result)

                if len(filtered) >= minimum_results:
                    break

        return filtered

    # ================================================================
    # NORMALIZATION
    # ================================================================

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:

        text = str(
            value or ""
        ).lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    # ================================================================
    # DOCUMENT ID
    # ================================================================

    @staticmethod
    def _document_id(
        chunk: Dict[str, Any],
    ) -> str:

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

        if (
            path
            or chunk_index is not None
        ):

            return (
                f"{path}|"
                f"{chunk_index}"
            )

        return str(
            chunk.get(
                "content",
                "",
            )
        )

    # ================================================================
    # DEBUG FORMATTER
    # ================================================================

    @staticmethod
    def format_result(
        result: Dict[str, Any],
    ) -> str:

        path = result.get(
            "path",
            "",
        )

        section = result.get(
            "section",
            "",
        )

        return (
            f"path={path!r}, "
            f"section={section!r}, "
            f"semantic_rank="
            f"{result.get('semantic_rank')}, "
            f"bm25_rank="
            f"{result.get('bm25_rank')}, "
            f"hybrid_score="
            f"{result.get('hybrid_score')}, "
            f"semantic_score="
            f"{result.get('semantic_score')}, "
            f"query_relevance="
            f"{result.get('query_relevance_score')}, "
            f"mmr_score="
            f"{result.get('mmr_score')}, "
            f"mmr_redundancy="
            f"{result.get('mmr_redundancy')}, "
            f"metadata_bonus="
            f"{result.get('metadata_bonus')}, "
            f"complementarity="
            f"{result.get('complementarity_score')}, "
            f"query_type="
            f"{result.get('query_type')}, "
            f"primary_protected="
            f"{result.get('primary_protected')}"
        )
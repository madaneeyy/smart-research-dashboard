from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .bm25_retriever import BM25Retriever
from .semantic_retriever import SimpleRetriever


class HybridRetriever:
    """
    Hybrid retriever combining:

        1. Semantic retrieval
        2. BM25 lexical retrieval
        3. Weighted Reciprocal Rank Fusion (RRF)
        4. Maximal Marginal Relevance (MMR)

    Retrieval pipeline:

        Query
          |
          +--> Semantic Retriever
          |
          +--> BM25 Retriever
          |
          v
        RRF Fusion
          |
          v
        Candidate Pool
          |
          v
        MMR Reranking
          |
          v
        Final Top-K

    RRF combines the rankings of semantic and lexical
    retrieval without requiring their raw scores to be
    on the same scale.

    MMR then reduces redundancy among the final results
    while preserving relevance to the query.
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
    ) -> None:
        """
        Initialize the hybrid retriever.

        Parameters
        ----------
        semantic_weight:
            Weight assigned to semantic RRF ranking.

        bm25_weight:
            Weight assigned to BM25 RRF ranking.

        rrf_k:
            RRF constant.

            Larger values make rank differences less aggressive.

        candidate_multiplier:
            Number of candidates retrieved from each retriever
            relative to final top_k.

        mmr_lambda:
            Controls the relevance/diversity trade-off.

                1.0 -> relevance only
                0.7 -> relevance-focused + some diversity
                0.5 -> balanced
                0.0 -> diversity-focused
        """

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
                "At least one retrieval weight must be greater than 0."
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than 0."
            )

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater than 0."
            )

        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(
                "mmr_lambda must be between 0 and 1."
            )

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

    # ============================================================
    # PUBLIC API
    # ============================================================

    def retrieve(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks using:

            Semantic Retrieval
                +
            BM25 Retrieval
                +
            Weighted RRF
                +
            MMR reranking

        Parameters
        ----------
        question:
            User query.

        chunks:
            Chunk collection produced by the document chunker.

        top_k:
            Number of final results.

        Returns
        -------
        list[dict]

            Each result contains the original chunk metadata
            plus retrieval information such as:

                similarity
                bm25_score
                semantic_rank
                bm25_rank
                hybrid_score
                mmr_score
        """

        # ========================================================
        # VALIDATE QUESTION
        # ========================================================

        if not question or not question.strip():
            return []

        # ========================================================
        # VALIDATE CHUNKS
        # ========================================================

        if not chunks:
            return []

        chunks = list(
            chunks
        )

        # ========================================================
        # VALIDATE TOP-K
        # ========================================================

        if top_k <= 0:
            return []

        top_k = min(
            top_k,
            len(chunks),
        )

        # ========================================================
        # CANDIDATE COUNT
        # ========================================================
        #
        # Example:
        #
        # top_k = 5
        # candidate_multiplier = 4
        #
        # candidate_k = 20
        #
        # We retrieve 20 candidates from each retriever
        # before fusion.
        #
        # This gives RRF and MMR enough candidates to work with.
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

        semantic_results = SimpleRetriever.retrieve(
            question=question,
            chunks=chunks,
            top_k=candidate_k,
        )

        # ========================================================
        # BM25 RETRIEVAL
        # ========================================================

        bm25_retriever = BM25Retriever(
            chunks
        )

        bm25_results = bm25_retriever.retrieve(
            query=question,
            top_k=candidate_k,
        )

        # ========================================================
        # RRF FUSION
        # ========================================================

        fused: Dict[
            str,
            Dict[str, Any],
        ] = {}

        # ========================================================
        # ADD SEMANTIC RESULTS
        # ========================================================

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):
            document_id = self._document_id(
                result
            )

            if document_id not in fused:
                fused[document_id] = {
                    "result": dict(result),
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
                    self.rrf_k + rank
                )
            )

        # ========================================================
        # ADD BM25 RESULTS
        # ========================================================

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            document_id = self._document_id(
                result
            )

            if document_id not in fused:
                fused[document_id] = {
                    "result": dict(result),
                    "semantic_rank": None,
                    "bm25_rank": None,
                    "hybrid_score": 0.0,
                }

            # ----------------------------------------------------
            # Merge metadata from BM25 result.
            #
            # Do not overwrite fields already supplied by
            # semantic retrieval unless necessary.
            # ----------------------------------------------------

            fused_result = fused[document_id][
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
                    self.rrf_k + rank
                )
            )

        # ========================================================
        # SORT BY RRF SCORE
        # ========================================================

        ranked = sorted(
            fused.values(),
            key=lambda item: item[
                "hybrid_score"
            ],
            reverse=True,
        )

        # ========================================================
        # BUILD RRF CANDIDATE RESULTS
        # ========================================================

        rrf_candidates: List[
            Dict[str, Any]
        ] = []

        for item in ranked:

            result = dict(
                item["result"]
            )

            result[
                "hybrid_score"
            ] = float(
                item["hybrid_score"]
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

            rrf_candidates.append(
                result
            )

        # ========================================================
        # MMR RERANKING
        # ========================================================
        #
        # RRF determines candidate relevance.
        #
        # MMR now chooses the final top_k while balancing:
        #
        #     relevance
        #
        #     vs.
        #
        #     redundancy
        #
        # ========================================================

        results = self._mmr_rerank(
            question=question,
            candidates=rrf_candidates,
            chunks=chunks,
            top_k=top_k,
        )

        return results

    # ============================================================
    # MMR RERANKING
    # ============================================================

    def _mmr_rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        chunks: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Rerank RRF candidates using Maximal Marginal Relevance.

        MMR formula:

            MMR(D) =
                lambda * relevance(D, query)
                -
                (1 - lambda)
                * max_similarity(D, selected_documents)

        The same SentenceTransformer model and chunk embedding
        cache used by SimpleRetriever are reused.

        This avoids:

            - loading another model
            - creating another embedding cache
            - generating duplicate embeddings
        """

        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        # ========================================================
        # LOAD NUMPY
        # ========================================================

        # numpy is already imported globally.
        # It is intentionally used directly here.

        # ========================================================
        # LOAD SAME EMBEDDING MODEL
        # ========================================================

        model = SimpleRetriever._get_model()

        # ========================================================
        # EMBED QUERY
        # ========================================================

        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        # ========================================================
        # FIND ORIGINAL CHUNKS
        # ========================================================
        #
        # Candidates are copies of the original chunks.
        #
        # We identify them using the same stable ID used by RRF.
        # ========================================================

        candidate_ids = {
            self._document_id(candidate)
            for candidate in candidates
        }

        chunk_lookup: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for chunk in chunks:

            document_id = self._document_id(
                chunk
            )

            if document_id in candidate_ids:
                chunk_lookup[
                    document_id
                ] = chunk

        # ========================================================
        # PRESERVE RRF CANDIDATE ORDER
        # ========================================================

        ordered_chunks: List[
            Dict[str, Any]
        ] = []

        valid_candidates: List[
            Dict[str, Any]
        ] = []

        for candidate in candidates:

            document_id = self._document_id(
                candidate
            )

            chunk = chunk_lookup.get(
                document_id
            )

            if chunk is not None:

                ordered_chunks.append(
                    chunk
                )

                valid_candidates.append(
                    candidate
                )

        candidates = valid_candidates

        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------

        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        # ========================================================
        # LOAD EXISTING CHUNK EMBEDDINGS
        # ========================================================
        #
        # IMPORTANT:
        #
        # This uses the exact same embedding cache as
        # SimpleRetriever.
        #
        # No duplicate embedding generation occurs if the cache
        # is available.
        # ========================================================

        candidate_embeddings = (
            SimpleRetriever._get_chunk_embeddings(
                ordered_chunks
            )
        )

        candidate_embeddings = np.asarray(
            candidate_embeddings,
            dtype=np.float32,
        )

        # ========================================================
        # VALIDATE EMBEDDING COUNT
        # ========================================================

        if len(candidate_embeddings) != len(
            candidates
        ):
            raise ValueError(
                "MMR candidate embedding count does not "
                "match candidate count."
            )

        # ========================================================
        # NORMALIZE QUERY
        # ========================================================

        query_norm = np.linalg.norm(
            query_embedding
        )

        if query_norm > 0:

            query_embedding = (
                query_embedding
                / query_norm
            )

        # ========================================================
        # NORMALIZE CANDIDATE EMBEDDINGS
        # ========================================================

        candidate_norms = np.linalg.norm(
            candidate_embeddings,
            axis=1,
            keepdims=True,
        )

        candidate_norms[
            candidate_norms == 0
        ] = 1.0

        candidate_embeddings = (
            candidate_embeddings
            / candidate_norms
        )

        # ========================================================
        # QUERY RELEVANCE
        # ========================================================
        #
        # Because embeddings are normalized:
        #
        # cosine similarity =
        # dot product
        # ========================================================

        relevance_scores = (
            candidate_embeddings
            @ query_embedding
        )

        # ========================================================
        # MMR SELECTION
        # ========================================================

        selected_indices: List[
            int
        ] = []

        remaining_indices = list(
            range(
                len(candidates)
            )
        )

        while (
            remaining_indices
            and len(selected_indices) < top_k
        ):

            best_index = None

            best_mmr_score = (
                -float("inf")
            )

            for index in remaining_indices:

                # ------------------------------------------------
                # Relevance to query
                # ------------------------------------------------

                relevance = float(
                    relevance_scores[index]
                )

                # ------------------------------------------------
                # Redundancy with already selected results
                # ------------------------------------------------

                if not selected_indices:

                    redundancy = 0.0

                else:

                    selected_embeddings = (
                        candidate_embeddings[
                            selected_indices
                        ]
                    )

                    similarities = (
                        selected_embeddings
                        @ candidate_embeddings[
                            index
                        ]
                    )

                    redundancy = float(
                        np.max(
                            similarities
                        )
                    )

                # ------------------------------------------------
                # MMR formula
                # ------------------------------------------------

                mmr_score = (
                    self.mmr_lambda
                    * relevance
                    -
                    (
                        1.0
                        - self.mmr_lambda
                    )
                    * redundancy
                )

                # ------------------------------------------------
                # Keep best candidate
                # ------------------------------------------------

                if (
                    mmr_score
                    > best_mmr_score
                ):

                    best_mmr_score = (
                        mmr_score
                    )

                    best_index = index

            # ----------------------------------------------------
            # Safety check
            # ----------------------------------------------------

            if best_index is None:
                break

            # ----------------------------------------------------
            # Select result
            # ----------------------------------------------------

            selected_indices.append(
                best_index
            )

            remaining_indices.remove(
                best_index
            )

        # ========================================================
        # BUILD FINAL RESULTS
        # ========================================================

        results: List[
            Dict[str, Any]
        ] = []

        for index in selected_indices:

            result = dict(
                candidates[index]
            )

            # ----------------------------------------------------
            # Store the actual MMR score used during selection.
            # ----------------------------------------------------

            result[
                "mmr_score"
            ] = float(
                self._calculate_mmr_score(
                    index=index,
                    selected_indices=selected_indices,
                    relevance_scores=relevance_scores,
                    candidate_embeddings=candidate_embeddings,
                )
            )

            results.append(
                result
            )

        return results

    # ============================================================
    # MMR SCORE HELPER
    # ============================================================

    def _calculate_mmr_score(
        self,
        index: int,
        selected_indices: List[int],
        relevance_scores: np.ndarray,
        candidate_embeddings: np.ndarray,
    ) -> float:
        """
        Calculate the MMR score for debugging/output.

        This does not affect ranking.
        """

        relevance = float(
            relevance_scores[index]
        )

        # --------------------------------------------------------
        # If no other document exists, there is no redundancy.
        # --------------------------------------------------------

        previous_indices = [
            i
            for i in selected_indices
            if i != index
        ]

        if not previous_indices:

            redundancy = 0.0

        else:

            similarities = (
                candidate_embeddings[
                    previous_indices
                ]
                @ candidate_embeddings[
                    index
                ]
            )

            redundancy = float(
                np.max(
                    similarities
                )
            )

        return (
            self.mmr_lambda
            * relevance
            -
            (
                1.0
                - self.mmr_lambda
            )
            * redundancy
        )

    # ============================================================
    # DOCUMENT ID
    # ============================================================

    @staticmethod
    def _document_id(
        chunk: Dict[str, Any],
    ) -> str:
        """
        Create a stable identifier for a chunk.

        Preferred identifier:

            path + chunk_index

        Example:

            doc/modules/linear_model.rst|32

        A content-based fallback is used when neither path
        nor chunk_index is available.
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

        # --------------------------------------------------------
        # Preferred identifier
        # --------------------------------------------------------

        if (
            path
            or chunk_index is not None
        ):

            return (
                f"{path}|"
                f"{chunk_index}"
            )

        # --------------------------------------------------------
        # Fallback
        # --------------------------------------------------------

        content = str(
            chunk.get(
                "content",
                "",
            )
        )

        return content

    # ============================================================
    # DEBUG / INSPECTION
    # ============================================================

    @staticmethod
    def format_result(
        result: Dict[str, Any],
    ) -> str:
        """
        Format a hybrid result for debugging/testing.

        This does not affect retrieval.
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

        return (
            f"path={path!r}, "
            f"section={section!r}, "
            f"semantic_rank={semantic_rank}, "
            f"bm25_rank={bm25_rank}, "
            f"similarity={similarity}, "
            f"bm25_score={bm25_score}, "
            f"hybrid_score={hybrid_score}, "
            f"mmr_score={mmr_score}"
        )
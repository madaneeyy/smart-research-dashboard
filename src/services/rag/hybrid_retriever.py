from __future__ import annotations
from typing import Any, Dict, List, Sequence
from .bm25_retriever import BM25Retriever
from .semantic_retriever import SimpleRetriever

class HybridRetriever:
    """
    Hybrid retriever combining:

        1. Semantic retrieval
        2. BM25 lexical retrieval

    The two rankings are combined using Weighted Reciprocal
    Rank Fusion (RRF).

    Why RRF?
    --------
    Semantic similarity and BM25 scores are on different scales.

    For example:

        semantic similarity -> roughly [-1, 1]

        BM25 score          -> depends on the corpus and query

    Therefore, directly adding the raw scores would not be
    mathematically meaningful.

    Instead, this retriever combines the rankings produced by
    each retriever.

    Final RRF contribution:

        weight / (rrf_k + rank)

    where rank starts at 1.

    This allows both lexical and semantic retrieval to
    contribute to the final ranking.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_multiplier: int=4,
        mmr_lamda: float = 0.7,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Parameters
        ----------
        semantic_weight:
            Weight assigned to the semantic ranking.

        bm25_weight:
            Weight assigned to the BM25 ranking.

        rrf_k:
            RRF constant used to reduce the impact of very
            high-ranked results.

        Examples
        --------
        Default balanced retrieval:

            HybridRetriever()

        More semantic:

            HybridRetriever(
                semantic_weight=0.7,
                bm25_weight=0.3,
            )

        More lexical:

            HybridRetriever(
                semantic_weight=0.3,
                bm25_weight=0.7,
            )
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

        self.semantic_weight = float(
            semantic_weight
        )

        self.bm25_weight = float(
            bm25_weight
        )

        self.rrf_k = int(
            rrf_k
        )
        if candidate_multiplier <= 0:
          raise ValueError(
        "candidate_multiplier must be greater than 0."
        )

        self.candidate_multiplier = int(
           candidate_multiplier
        )
        if not 0.0 <= mmr_lamda <= 1.0:
          raise ValueError(
          "mmr_lambda must be between 0 and 1."
        )

          self.mmr_lambda = float(mmr_lambda)

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
        Retrieve the most relevant chunks using both:

            - semantic retrieval
            - BM25 retrieval

        Parameters
        ----------
        question:
            User's search/query question.

        chunks:
            Chunk collection produced by the DocumentChunker.

        top_k:
            Number of final hybrid results.

        Returns
        -------
        list[dict]
            Ranked chunks containing:

                - original chunk metadata
                - similarity
                - bm25_score
                - semantic_rank
                - bm25_rank
                - hybrid_score

        Notes
        -----
        More candidates than top_k are retrieved from each
        individual retriever before fusion.

        This is important because a chunk may rank:

            #1 in semantic retrieval
            #15 in BM25

        and should still have a chance to appear in the
        final hybrid ranking.
        """

        # --------------------------------------------------------
        # Validate query
        # --------------------------------------------------------

        if not question or not question.strip():
            return []

        # --------------------------------------------------------
        # Validate chunks
        # --------------------------------------------------------

        if not chunks:
            return []

        chunks = list(chunks)

        # --------------------------------------------------------
        # Normalize top_k
        # --------------------------------------------------------

        if top_k <= 0:
            return []

        top_k = min(
            top_k,
            len(chunks),
        )

        # --------------------------------------------------------
        # Candidate count
        # --------------------------------------------------------
        #
        # Do not retrieve only top_k from each retriever.
        #
        # Example:
        #
        # top_k = 5
        #
        # We retrieve up to 10 candidates from each side
        # before fusion.
        #
        # This gives RRF more information to work with.
        # --------------------------------------------------------

        candidate_k = min(
            max(top_k * self.candidate_multiplier, 10),
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
        # FUSION
        # ========================================================

        fused: Dict[
            str,
            Dict[str, Any],
        ] = {}

        # --------------------------------------------------------
        # Add semantic rankings
        # --------------------------------------------------------

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
                / (self.rrf_k + rank)
            )

        # --------------------------------------------------------
        # Add BM25 rankings
        # --------------------------------------------------------

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
            # If the document was already found by semantic
            # retrieval, merge the BM25 metadata into the
            # existing result.
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
                / (self.rrf_k + rank)
            )

        # ========================================================
        # SORT BY HYBRID SCORE
        # ========================================================

        ranked = sorted(
            fused.values(),
            key=lambda item: item[
                "hybrid_score"
            ],
            reverse=True,
        )

        # ========================================================
        # BUILD FINAL RESULTS
        # ========================================================

        results: List[
            Dict[str, Any]
        ] = []

        for item in ranked[:top_k]:

            result = dict(
                item["result"]
            )

            # ----------------------------------------------------
            # Add hybrid ranking information
            # ----------------------------------------------------

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

            results.append(
                result
            )

        return results

    def _mmr_rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Maximal Marginal Relevance (MMR).

        MMR balances:

            1. Relevance to the query
            2. Diversity among selected chunks

        A high mmr_lambda favors relevance.
        A lower mmr_lambda favors diversity.

        Formula:

            MMR =
                lambda * relevance
                -
                (1 - lambda) * redundancy
        """

        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        # --------------------------------------------------------
        # Obtain embeddings for the query and candidate chunks
        # --------------------------------------------------------

        query_embedding = SimpleRetriever.embed_query(
            question
        )

        candidate_embeddings = []

        for candidate in candidates:
            embedding = SimpleRetriever.embed_text(
                candidate["content"]
            )

            candidate_embeddings.append(
                embedding
            )

        # --------------------------------------------------------
        # Convert embeddings to numpy arrays
        # --------------------------------------------------------

        import numpy as np

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        candidate_embeddings = np.asarray(
            candidate_embeddings,
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # Normalize embeddings
        #
        # After normalization:
        #
        # dot product == cosine similarity
        # --------------------------------------------------------

        query_norm = np.linalg.norm(
            query_embedding
        )

        if query_norm > 0:
            query_embedding = (
                query_embedding / query_norm
            )

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

        # --------------------------------------------------------
        # Query-to-document similarity
        # --------------------------------------------------------

        relevance_scores = (
            candidate_embeddings
            @ query_embedding
        )

        # --------------------------------------------------------
        # MMR selection
        # --------------------------------------------------------

        selected_indices = []

        remaining_indices = list(
            range(len(candidates))
        )

        while (
            remaining_indices
            and len(selected_indices) < top_k
        ):

            best_index = None
            best_score = -float("inf")

            for index in remaining_indices:

                relevance = float(
                    relevance_scores[index]
                )

                # ------------------------------------------------
                # First document:
                #
                # No redundancy penalty.
                # ------------------------------------------------

                if not selected_indices:

                    redundancy = 0.0

                else:

                    similarities = (
                        candidate_embeddings[index]
                        @ candidate_embeddings[
                            selected_indices
                        ].T
                    )

                    redundancy = float(
                        np.max(similarities)
                    )

                # ------------------------------------------------
                # MMR formula
                # ------------------------------------------------

                mmr_score = (
                    self.mmr_lambda * relevance
                    -
                    (1.0 - self.mmr_lambda)
                    * redundancy
                )

                if mmr_score > best_score:

                    best_score = mmr_score
                    best_index = index

            # ----------------------------------------------------
            # Add best candidate
            # ----------------------------------------------------

            selected_indices.append(
                best_index
            )

            remaining_indices.remove(
                best_index
            )

        # --------------------------------------------------------
        # Build final results
        # --------------------------------------------------------

        results = []

        for index in selected_indices:

            result = dict(
                candidates[index]
            )

            result["mmr_score"] = float(
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
    # DOCUMENT ID
    # ============================================================

    @staticmethod
    def _document_id(
        chunk: Dict[str, Any],
    ) -> str:
        """
        Create a stable identifier for a chunk.

        The chunker assigns a chunk_index to every chunk and
        preserves the source path.

        Therefore:

            path + chunk_index

        is the preferred identifier.

        A content-based fallback is used when chunk_index/path
        are unavailable.
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

        if path or chunk_index is not None:
            return (
                f"{path}|"
                f"{chunk_index}"
            )

        # --------------------------------------------------------
        # Fallback identifier
        # --------------------------------------------------------
        #
        # This should rarely be necessary because the chunker
        # normally supplies path and chunk_index.
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

        return (
            f"path={path!r}, "
            f"section={section!r}, "
            f"semantic_rank={semantic_rank}, "
            f"bm25_rank={bm25_rank}, "
            f"similarity={similarity}, "
            f"bm25_score={bm25_score}, "
            f"hybrid_score={hybrid_score}"
        )
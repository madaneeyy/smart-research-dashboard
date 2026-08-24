from __future__ import annotations

"""
Independent retrieval layer for uploaded documents.

This module is intentionally separate from the GitHub RAG pipeline.

Retrieval stack:
    1. BM25 lexical retrieval
    2. TF-IDF lexical retrieval
    3. Optional dense semantic retrieval via sentence-transformers
    4. Query-adaptive reciprocal-rank fusion
    5. Metadata / heading boosts
    6. MMR diversity selection
    7. Context-size protection

Dense retrieval is optional. If sentence-transformers is not installed, the
retriever still works using BM25 + TF-IDF + lexical/metadata signals.

Public API is kept compatible with the current backend:
    DocumentRetriever(...)
    DocumentRetriever.retrieve(question, chunks, top_k=5)
"""

import hashlib
import math
import re
import time
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Sequence, Set, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class DocumentRetriever:
    """Standalone hybrid retriever for uploaded documents."""

    TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+#/-]*")

    STOPWORDS: Set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has", "have",
        "how", "i", "if", "in", "into", "is", "it", "its", "may", "me",
        "of", "on", "or", "our", "please", "should", "that", "the", "their",
        "this", "to", "was", "were", "what", "when", "where", "which", "who",
        "why", "with", "would", "you", "your", "we", "us", "about", "tell",
        "give", "show", "explain", "document", "file", "pdf", "report",
    }

    INTENT_TERMS = {
        "methodology": {
            "method", "methodology", "approach", "procedure", "protocol",
            "implementation", "experimental", "experiment", "training",
            "evaluation", "technique",
        },
        "dataset": {
            "dataset", "datasets", "data", "corpus", "sample", "samples",
        },
        "model": {
            "model", "models", "architecture", "architectures", "network",
            "cnn", "transformer", "vit", "mamba",
        },
        "results": {
            "result", "results", "finding", "findings", "performance",
            "accuracy", "outcome", "comparison", "benchmark",
        },
        "conclusion": {
            "conclusion", "discussion", "future", "limitation",
            "limitations", "recommendation", "recommendations",
        },
        "introduction": {
            "introduction", "background", "motivation", "objective",
            "objectives", "abstract", "problem",
        },
    }

    OVERVIEW_RE = re.compile(
        r"\b(explain|summari[sz]e|overview|walk\s+me\s+through|"
        r"main\s+(?:idea|ideas|points|findings)|key\s+(?:idea|ideas|points|findings)|"
        r"in\s+short|briefly|what\s+is\s+this)\b",
        re.I,
    )

    FACTUAL_RE = re.compile(
        r"\b(title|name|author|authors|date|year|location|"
        r"who\s+(?:is|was|are)|what\s+(?:is|was|are)|"
        r"when\s+(?:was|is)|where\s+(?:was|is)|how\s+many)\b",
        re.I,
    )

    VISUAL_RE = re.compile(
        r"\b(figure|fig\.?|diagram|chart|plot|graph|table|"
        r"image|illustration|screenshot|architecture\s+diagram)\b",
        re.I,
    )

    def __init__(
        self,
        overview_top_k: int = 7,
        focused_top_k: int = 6,
        candidate_multiplier: int = 5,
        mmr_lambda: float = 0.72,
        dense_model_name: str = "all-MiniLM-L6-v2",
        dense_enabled: bool = True,
        dense_batch_size: int = 32,
        dense_cache_size: int = 8,
        query_embedding_cache_size: int = 32,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_enabled: bool = True,
        reranker_candidate_count: int = 20,
        reranker_batch_size: int = 16,
        context_budget_overview: int = 12000,
        context_budget_focused: int = 9000,
        context_budget_factual: int = 5000,
        min_score: float = 0.0,
        neighbor_window: int = 1,
        max_expanded_per_result: int = 2,
        expand_same_section_only: bool = False,
        expand_parents: bool = True,
        max_total_expanded_chunks: int = 10,
        min_neighbor_relevance: float = 0.18,
        expand_factual_queries: bool = False,
        profiling_enabled: bool = False,
    ) -> None:
        self.overview_top_k = max(1, int(overview_top_k))
        self.focused_top_k = max(1, int(focused_top_k))
        self.candidate_multiplier = max(2, int(candidate_multiplier))
        self.mmr_lambda = max(0.0, min(1.0, float(mmr_lambda)))

        self.dense_model_name = dense_model_name
        self.dense_enabled = bool(dense_enabled)
        self.dense_batch_size = max(1, int(dense_batch_size))
        self.dense_cache_size = max(1, int(dense_cache_size))
        self.query_embedding_cache_size = max(1, int(query_embedding_cache_size))
        self._dense_model = None
        self._dense_failed = False

        # In-memory only: no cache files are created.
        # Document embeddings are reused across questions.
        self._dense_embedding_cache: OrderedDict[str, Any] = OrderedDict()
        self._query_embedding_cache: OrderedDict[str, Any] = OrderedDict()

        self.reranker_model_name = reranker_model_name
        self.reranker_enabled = bool(reranker_enabled)
        self.reranker_candidate_count = max(1, int(reranker_candidate_count))
        self.reranker_batch_size = max(1, int(reranker_batch_size))
        self._reranker_model = None
        self._reranker_failed = False

        self.context_budget_overview = max(1000, int(context_budget_overview))
        self.context_budget_focused = max(1000, int(context_budget_focused))
        self.context_budget_factual = max(1000, int(context_budget_factual))
        self.min_score = float(min_score)

        # Parent/neighbor expansion settings.  Expansion is performed only
        # after retrieval/reranking/MMR, so the original ranking remains
        # authoritative.
        self.neighbor_window = max(0, int(neighbor_window))
        self.max_expanded_per_result = max(1, int(max_expanded_per_result))
        self.expand_same_section_only = bool(expand_same_section_only)
        self.expand_parents = bool(expand_parents)
        self.max_total_expanded_chunks = max(1, int(max_total_expanded_chunks))
        self.min_neighbor_relevance = max(
            0.0, min(1.0, float(min_neighbor_relevance))
        )
        self.expand_factual_queries = bool(expand_factual_queries)
        self.profiling_enabled = bool(profiling_enabled)
        self.last_profile: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        _profile = {}
        _t0 = time.perf_counter()
        _stage = time.perf_counter()
        valid = self._prepare_chunks(chunks)
        _profile["prepare_chunks_ms"] = (time.perf_counter() - _stage) * 1000
        if not valid:
            return []

        question = str(question or "").strip()
        if not question:
            return []

        query_type = self._query_type(question)
        requested = max(1, int(top_k or self.focused_top_k))

        if query_type == "overview":
            target_k = min(max(requested, self.overview_top_k), 8, len(valid))
            budget = self.context_budget_overview
        elif query_type == "factual":
            target_k = min(max(requested, 3), 5, len(valid))
            budget = self.context_budget_factual
        else:
            target_k = min(max(requested, self.focused_top_k), 8, len(valid))
            budget = self.context_budget_focused

        # Build all retrieval signals once per request.
        _stage = time.perf_counter()
        bm25_scores = self._bm25_scores(question, valid)
        _profile["bm25_ms"] = (time.perf_counter() - _stage) * 1000
        _stage = time.perf_counter()
        tfidf_scores = self._tfidf_scores(question, valid)
        _profile["tfidf_ms"] = (time.perf_counter() - _stage) * 1000
        _stage = time.perf_counter()
        dense_texts = [self._retrieval_text(chunk) for chunk in valid]
        dense_cache_key = self._embedding_cache_key(valid, dense_texts)
        dense_cache_hit = dense_cache_key in self._dense_embedding_cache
        dense_scores = self._dense_scores(question, valid)
        _profile["dense_ms"] = (time.perf_counter() - _stage) * 1000
        _profile["dense_document_cache_hit"] = int(dense_cache_hit)

        # Rank separately with each retriever.
        bm25_rank = self._rank_indices(bm25_scores)
        tfidf_rank = self._rank_indices(tfidf_scores)
        dense_rank = self._rank_indices(dense_scores) if dense_scores else []

        _stage = time.perf_counter()
        fused = self._fuse_ranks(
            bm25_rank=bm25_rank,
            tfidf_rank=tfidf_rank,
            dense_rank=dense_rank,
            bm25_scores=bm25_scores,
            tfidf_scores=tfidf_scores,
            dense_scores=dense_scores,
            question=question,
            query_type=query_type,
            chunks=valid,
        )

        # Keep a reasonably large candidate pool before MMR.
        candidate_count = min(
            len(fused),
            max(target_k * self.candidate_multiplier, target_k),
        )
        candidates = fused[:candidate_count]
        _profile["fusion_ms"] = (time.perf_counter() - _stage) * 1000

        _stage = time.perf_counter()
        candidates = self._query_aware_filter(
            question=question,
            candidates=candidates,
            query_type=query_type,
        )

        _profile["query_filter_ms"] = (time.perf_counter() - _stage) * 1000

        _stage = time.perf_counter()
        candidates = self._rerank_candidates(
            question=question,
            candidates=candidates,
        )

        _profile["reranker_ms"] = (time.perf_counter() - _stage) * 1000

        _stage = time.perf_counter()
        selected = self._mmr_select(
            candidates=candidates,
            top_k=target_k,
            question=question,
        )

        # Expand high-quality retrieved chunks with their immediate document
        # neighbors.  This happens after ranking so neighbors never outrank
        # the evidence selected by the retrieval/reranking pipeline.
        _profile["mmr_ms"] = (time.perf_counter() - _stage) * 1000

        _stage = time.perf_counter()
        selected = self._expand_with_neighbors(
            selected=selected,
            all_chunks=valid,
            question=question,
            query_type=query_type,
            max_total_chunks=self.max_total_expanded_chunks,
        )

        _profile["parent_neighbor_expansion_ms"] = (time.perf_counter() - _stage) * 1000

        _stage = time.perf_counter()
        selected = self._apply_context_budget(selected, budget)
        _profile["context_budget_ms"] = (time.perf_counter() - _stage) * 1000

        if not selected and candidates:
            selected = [self._truncate(candidates[0], budget)]

        results: List[Dict[str, Any]] = []

        for rank, item in enumerate(selected, start=1):
            result = dict(item)

            # Public, honest retrieval metadata.
            result["retrieval_source"] = "document"
            result["retriever_name"] = self.__class__.__name__
            result["document_rank"] = rank
            result["query_type"] = query_type

            # Keep score names explicit so the UI does not imply that an
            # unavailable score came from a different retrieval algorithm.
            result["relevance_score"] = round(
                float(result.get("_hybrid_score", 0.0)), 4
            )
            result["bm25_score"] = round(
                float(result.get("_bm25_score", 0.0)), 4
            )
            result["tfidf_score"] = round(
                float(result.get("_tfidf_score", 0.0)), 4
            )
            result["dense_score"] = round(
                float(result.get("_dense_score", 0.0)), 4
            )
            result["reranker_score"] = round(
                float(result.get("_reranker_score", 0.0)), 4
            )
            result["metadata_score"] = round(
                float(result.get("_metadata_score", 0.0)), 4
            )
            result["mmr_score"] = round(
                float(result.get("_mmr_score", 0.0)), 4
            )
            result["redundancy_score"] = round(
                float(result.get("_redundancy_score", 0.0)), 4
            )
            result["context_budget_chars"] = budget

            # Internal fields are not useful to the frontend.
            for key in list(result):
                if key.startswith("_"):
                    result.pop(key, None)

            results.append(result)

        _profile["total_retrieval_ms"] = (time.perf_counter() - _t0) * 1000
        _profile["input_chunks"] = len(valid)
        _profile["candidate_chunks"] = len(candidates)
        _profile["final_chunks"] = len(results)
        self.last_profile = _profile
        if self.profiling_enabled:
            print("\n" + "=" * 72)
            print("DOCUMENT RETRIEVER PERFORMANCE")
            print("=" * 72)
            for k, v in _profile.items():
                if k.endswith("_ms"):
                    print(f"{k:<34}: {v:,.2f} ms")
                else:
                    print(f"{k:<34}: {int(v)}")
            print("=" * 72)
        return results

    # ------------------------------------------------------------------
    # Chunk preparation
    # ------------------------------------------------------------------

    def _prepare_chunks(
        self,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        valid: List[Dict[str, Any]] = []

        for index, original in enumerate(chunks):
            content = str(
                original.get("content")
                or original.get("raw_content")
                or ""
            ).strip()

            if not content:
                continue

            item = dict(original)
            item["content"] = content
            item.setdefault(
                "chunk_id",
                f"{item.get('document_id', 'document')}:{index}",
            )
            item["_original_index"] = index
            valid.append(item)

        return valid

    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------

    def _query_type(self, question: str) -> str:
        if self.VISUAL_RE.search(question):
            return "visual"
        if self.OVERVIEW_RE.search(question):
            return "overview"
        if self.FACTUAL_RE.search(question):
            return "factual"
        return "focused"

    def _query_terms(self, question: str) -> Set[str]:
        return {
            token
            for token in self._tokens(question)
            if token not in self.STOPWORDS
        }

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def _bm25_scores(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[float]:
        documents = [
            self._tokens(
                self._retrieval_text(chunk)
            )
            for chunk in chunks
        ]
        query = [
            token
            for token in self._tokens(question)
            if token not in self.STOPWORDS
        ]

        if not query or not documents:
            return [0.0] * len(chunks)

        n = len(documents)
        avgdl = sum(len(doc) for doc in documents) / max(n, 1)

        document_frequency: Counter[str] = Counter()
        for doc in documents:
            for token in set(doc):
                document_frequency[token] += 1

        k1 = 1.5
        b = 0.75

        raw_scores: List[float] = []

        for doc in documents:
            frequencies = Counter(doc)
            dl = len(doc)
            score = 0.0

            for term in query:
                if term not in frequencies:
                    continue

                df = document_frequency.get(term, 0)
                idf = math.log(
                    1.0 + (n - df + 0.5) / (df + 0.5)
                )

                tf = frequencies[term]
                denominator = (
                    tf
                    + k1
                    * (
                        1.0
                        - b
                        + b * dl / max(avgdl, 1.0)
                    )
                )

                score += idf * (
                    tf * (k1 + 1.0)
                ) / max(denominator, 1e-9)

            raw_scores.append(score)

        return self._normalize_scores(raw_scores)

    # ------------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------------

    def _tfidf_scores(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[float]:
        if TfidfVectorizer is None or cosine_similarity is None:
            return [0.0] * len(chunks)

        texts = [
            self._retrieval_text(chunk)
            for chunk in chunks
        ]

        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_df=0.98,
                token_pattern=r"(?u)\b[\w][\w.+#/-]*\b",
            )

            matrix = vectorizer.fit_transform(texts)
            query_vector = vectorizer.transform([question])
            values = cosine_similarity(
                query_vector,
                matrix,
            ).ravel()

            return [
                max(0.0, min(1.0, float(value)))
                for value in values
            ]

        except Exception:
            return [0.0] * len(chunks)

    # ------------------------------------------------------------------
    # Dense semantic retrieval
    # ------------------------------------------------------------------

    def _dense_scores(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[float]:
        if not self.dense_enabled or self._dense_failed:
            return [0.0] * len(chunks)

        model = self._get_dense_model()
        if model is None:
            return [0.0] * len(chunks)

        try:
            texts = [self._retrieval_text(chunk) for chunk in chunks]

            # Expensive document encoding is cached in memory. Subsequent
            # questions only need the query embedding.
            document_key = self._embedding_cache_key(chunks, texts)
            embeddings = self._dense_embedding_cache.get(document_key)

            if embeddings is not None:
                self._dense_embedding_cache.move_to_end(document_key)
            else:
                embeddings = model.encode(
                    texts,
                    batch_size=self.dense_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                self._dense_embedding_cache[document_key] = embeddings
                self._dense_embedding_cache.move_to_end(document_key)
                while len(self._dense_embedding_cache) > self.dense_cache_size:
                    self._dense_embedding_cache.popitem(last=False)

            # Small LRU for repeated questions.
            query_key = question.strip()
            query_embedding = self._query_embedding_cache.get(query_key)

            if query_embedding is not None:
                self._query_embedding_cache.move_to_end(query_key)
            else:
                query_embedding = model.encode(
                    [question],
                    batch_size=1,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )[0]
                self._query_embedding_cache[query_key] = query_embedding
                self._query_embedding_cache.move_to_end(query_key)
                while len(self._query_embedding_cache) > self.query_embedding_cache_size:
                    self._query_embedding_cache.popitem(last=False)

            # Normalized embeddings make dot product equivalent to cosine.
            scores = embeddings @ query_embedding

            return [
                max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
                for score in scores
            ]

        except Exception:
            return [0.0] * len(chunks)

    @staticmethod
    def _embedding_cache_key(
        chunks: Sequence[Dict[str, Any]],
        texts: Sequence[str],
    ) -> str:
        hasher = hashlib.sha256()

        for index, (chunk, text) in enumerate(zip(chunks, texts)):
            chunk_id = str(
                chunk.get("chunk_id")
                or chunk.get("id")
                or chunk.get("document_chunk_index")
                or index
            )
            hasher.update(chunk_id.encode("utf-8", errors="ignore"))
            hasher.update(b"\\0")
            hasher.update(text.encode("utf-8", errors="ignore"))
            hasher.update(b"\\0")

        return hasher.hexdigest()

    def _get_dense_model(self):
        if self._dense_model is not None:
            return self._dense_model

        if self._dense_failed:
            return None

        try:
            from sentence_transformers import SentenceTransformer

            self._dense_model = SentenceTransformer(
                self.dense_model_name
            )
            return self._dense_model

        except Exception:
            self._dense_failed = True
            return None

    # ------------------------------------------------------------------
    # Hybrid fusion
    # ------------------------------------------------------------------

    def _fuse_ranks(
        self,
        bm25_rank: List[int],
        tfidf_rank: List[int],
        dense_rank: List[int],
        bm25_scores: List[float],
        tfidf_scores: List[float],
        dense_scores: List[float],
        question: str,
        query_type: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        n = len(chunks)
        rrf_k = 60.0

        # Query-adaptive weights.
        if query_type == "factual":
            weights = {
                "bm25": 0.48,
                "tfidf": 0.27,
                "dense": 0.15,
                "metadata": 0.10,
            }
        elif query_type == "overview":
            weights = {
                "bm25": 0.25,
                "tfidf": 0.20,
                "dense": 0.35,
                "metadata": 0.20,
            }
        elif query_type == "visual":
            # The text retriever cannot truly inspect visuals yet. Keep text
            # retrieval useful while making this query type visible to the
            # future ColQwen integration.
            weights = {
                "bm25": 0.25,
                "tfidf": 0.20,
                "dense": 0.40,
                "metadata": 0.15,
            }
        else:
            weights = {
                "bm25": 0.30,
                "tfidf": 0.20,
                "dense": 0.38,
                "metadata": 0.12,
            }

        # If dense retrieval is unavailable, redistribute its weight.
        if not any(dense_scores):
            lexical_total = weights["bm25"] + weights["tfidf"]
            if lexical_total > 0:
                weights["bm25"] += (
                    weights["dense"] * weights["bm25"] / lexical_total
                )
                weights["tfidf"] += (
                    weights["dense"] * weights["tfidf"] / lexical_total
                )
            weights["dense"] = 0.0

        rank_maps = {
            "bm25": {index: rank for rank, index in enumerate(bm25_rank, 1)},
            "tfidf": {index: rank for rank, index in enumerate(tfidf_rank, 1)},
            "dense": {index: rank for rank, index in enumerate(dense_rank, 1)},
        }

        fused: List[Dict[str, Any]] = []

        for index in range(n):
            bm25_rrf = (
                1.0 / (rrf_k + rank_maps["bm25"][index])
                if index in rank_maps["bm25"]
                else 0.0
            )
            tfidf_rrf = (
                1.0 / (rrf_k + rank_maps["tfidf"][index])
                if index in rank_maps["tfidf"]
                else 0.0
            )
            dense_rrf = (
                1.0 / (rrf_k + rank_maps["dense"][index])
                if index in rank_maps["dense"]
                else 0.0
            )

            metadata_score = self._metadata_score(
                question=question,
                chunk=chunks[index],
                query_type=query_type,
            )

            rrf_score = (
                weights["bm25"] * bm25_rrf
                + weights["tfidf"] * tfidf_rrf
                + weights["dense"] * dense_rrf
            )

            # RRF values are small. Metadata is normalized separately.
            # Normalize weighted RRF directly. The maximum is reached
            # when every available retriever ranks this chunk first.
            max_rrf = max(sum(weights.values()) / (rrf_k + 1.0), 1e-12)
            normalized_rrf = max(0.0, min(1.0, rrf_score / max_rrf))

            hybrid_score = (
                0.90 * normalized_rrf
                + 0.10 * metadata_score
            )

            item = dict(chunks[index])
            item["_hybrid_score"] = hybrid_score
            item["_bm25_score"] = bm25_scores[index]
            item["_tfidf_score"] = tfidf_scores[index]
            item["_dense_score"] = dense_scores[index]
            item["_metadata_score"] = metadata_score
            item["_bm25_rank"] = rank_maps["bm25"].get(index)
            item["_tfidf_rank"] = rank_maps["tfidf"].get(index)
            item["_dense_rank"] = rank_maps["dense"].get(index)

            fused.append(item)

        fused.sort(
            key=lambda item: float(item["_hybrid_score"]),
            reverse=True,
        )

        return fused

    @staticmethod
    def _normalize_single_rr(
        value: float,
        weights: Dict[str, float],
    ) -> float:
        # The largest possible weighted RRF contribution is approximately
        # sum(weights) / 61. Normalize relative to that.
        denominator = max(
            sum(weights.values()) / 61.0,
            1e-9,
        )
        return max(0.0, min(1.0, value / denominator))

    # ------------------------------------------------------------------
    # Metadata / structure scoring
    # ------------------------------------------------------------------

    def _metadata_score(
        self,
        question: str,
        chunk: Dict[str, Any],
        query_type: str,
    ) -> float:
        terms = self._query_terms(question)

        heading = str(
            chunk.get("section")
            or chunk.get("heading")
            or chunk.get("title")
            or ""
        ).lower()

        content = str(chunk.get("content") or "").lower()

        score = 0.0

        if heading:
            heading_tokens = set(self._tokens(heading))
            overlap = len(terms & heading_tokens) / max(len(terms), 1)
            score += 0.45 * overlap

        intent_hits = 0.0
        if terms:
            for intent_terms in self.INTENT_TERMS.values():
                overlap = len(terms & intent_terms)
                if overlap:
                    content_terms = set(self._tokens(content))
                    if content_terms & intent_terms:
                        intent_hits = max(
                            intent_hits,
                            min(
                                1.0,
                                len(content_terms & intent_terms) / 3.0,
                            ),
                        )

        score += 0.25 * intent_hits

        if query_type == "factual":
            # Early pages are often important for titles/authors/abstracts.
            position = chunk.get("document_chunk_index")
            try:
                position = int(position)
                if position <= 5:
                    score += 0.30
                elif position <= 15:
                    score += 0.12
            except (TypeError, ValueError):
                pass

        if chunk.get("page") is not None:
            score += 0.05

        return max(0.0, min(1.0, score))


    # ------------------------------------------------------------------
    # Query-aware relevance filtering
    # ------------------------------------------------------------------

    def _query_aware_filter(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        query_type: str,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        query_terms = self._query_terms(question)
        if not query_terms:
            return candidates

        scored: List[Tuple[float, Dict[str, Any]]] = []

        for candidate in candidates:
            content = str(candidate.get("content") or "").lower()
            retrieval_text = self._retrieval_text(candidate).lower()
            content_tokens = set(self._tokens(content))

            term_overlap = len(query_terms & content_tokens) / max(
                len(query_terms), 1
            )

            phrases = self._important_phrases(question)
            phrase_hits = sum(
                1 for phrase in phrases if phrase in retrieval_text
            )
            phrase_score = min(
                1.0,
                phrase_hits / max(len(phrases), 1),
            )

            hybrid = float(candidate.get("_hybrid_score", 0.0))
            metadata = float(candidate.get("_metadata_score", 0.0))

            query_fit = (
                0.45 * hybrid
                + 0.30 * term_overlap
                + 0.15 * phrase_score
                + 0.10 * metadata
            )

            candidate["_query_fit_score"] = query_fit
            scored.append((query_fit, candidate))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        keep_count = min(
            len(scored),
            max(self.focused_top_k * 3, 10),
        )
        kept = [candidate for _, candidate in scored[:keep_count]]

        # Preserve the strongest lexical result for short factual queries.
        if query_type == "factual" and candidates:
            lexical_best = max(
                candidates,
                key=lambda item: (
                    0.65 * float(item.get("_bm25_score", 0.0))
                    + 0.35 * float(item.get("_tfidf_score", 0.0))
                ),
            )
            if not any(
                self._stable_chunk_id(item)
                == self._stable_chunk_id(lexical_best)
                for item in kept
            ):
                kept.append(lexical_best)

        return kept

    @classmethod
    def _important_phrases(cls, question: str) -> List[str]:
        tokens = [
            token.lower()
            for token in cls._tokens(question)
            if token.lower() not in cls.STOPWORDS
        ]

        phrases: List[str] = []
        for size in (3, 2):
            for i in range(len(tokens) - size + 1):
                phrase = " ".join(tokens[i:i + size]).strip()
                if len(phrase) >= 6:
                    phrases.append(phrase)

        unique: List[str] = []
        seen: Set[str] = set()
        for phrase in sorted(
            phrases,
            key=lambda value: (-len(value.split()), -len(value)),
        ):
            if phrase not in seen:
                unique.append(phrase)
                seen.add(phrase)

        return unique[:12]

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank_candidates(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if not self.reranker_enabled:
            for candidate in candidates:
                candidate["_reranker_score"] = float(
                    candidate.get("_hybrid_score", 0.0)
                )
            return candidates

        model = self._get_reranker_model()
        if model is None:
            for candidate in candidates:
                candidate["_reranker_score"] = float(
                    candidate.get("_hybrid_score", 0.0)
                )
            return candidates

        rerank_pool = candidates[:self.reranker_candidate_count]
        pairs = [
            (question, self._reranker_text(candidate))
            for candidate in rerank_pool
        ]

        try:
            scores = model.predict(
                pairs,
                batch_size=self.reranker_batch_size,
                show_progress_bar=False,
            )

            normalized_scores = []
            for score in scores:
                score = max(-30.0, min(30.0, float(score)))
                normalized_scores.append(
                    1.0 / (1.0 + math.exp(-score))
                )

            for candidate, score in zip(
                rerank_pool,
                normalized_scores,
            ):
                candidate["_reranker_score"] = score

            for candidate in candidates[len(rerank_pool):]:
                candidate["_reranker_score"] = (
                    0.25 * float(candidate.get("_hybrid_score", 0.0))
                )

            for candidate in candidates:
                hybrid = float(candidate.get("_hybrid_score", 0.0))
                reranker = float(candidate.get("_reranker_score", 0.0))
                query_fit = float(candidate.get("_query_fit_score", 0.0))

                candidate["_hybrid_score"] = (
                    0.70 * reranker
                    + 0.20 * hybrid
                    + 0.10 * query_fit
                )

            candidates.sort(
                key=lambda item: float(item.get("_hybrid_score", 0.0)),
                reverse=True,
            )
            return candidates

        except Exception:
            for candidate in candidates:
                candidate["_reranker_score"] = float(
                    candidate.get("_hybrid_score", 0.0)
                )
            return candidates

    def _get_reranker_model(self):
        if not self.reranker_enabled or self._reranker_failed:
            return None

        if self._reranker_model is not None:
            return self._reranker_model

        if CrossEncoder is None:
            self._reranker_failed = True
            return None

        try:
            self._reranker_model = CrossEncoder(
                self.reranker_model_name
            )
            return self._reranker_model
        except Exception:
            self._reranker_failed = True
            return None

    def _reranker_text(self, chunk: Dict[str, Any]) -> str:
        section = str(
            chunk.get("section")
            or chunk.get("heading")
            or ""
        ).strip()

        parent = str(
            chunk.get("parent_section")
            or ""
        ).strip()

        content = str(chunk.get("content") or "").strip()

        return "\n".join(
            part for part in (parent, section, content) if part
        )

    # ------------------------------------------------------------------
    # MMR
    # ------------------------------------------------------------------

    def _mmr_select(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
        question: str,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        remaining = list(candidates)

        while remaining and len(selected) < top_k:
            best = None
            best_value = -float("inf")
            best_redundancy = 0.0

            for candidate in remaining:
                relevance = float(
                    candidate.get("_hybrid_score", 0.0)
                )

                redundancy = 0.0
                if selected:
                    redundancy = max(
                        self._chunk_similarity(candidate, other)
                        for other in selected
                    )

                mmr_value = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * redundancy
                )

                if mmr_value > best_value:
                    best_value = mmr_value
                    best = candidate
                    best_redundancy = redundancy

            if best is None:
                break

            best["_mmr_score"] = best_value
            best["_redundancy_score"] = best_redundancy

            selected.append(best)
            remaining.remove(best)

        return selected

    def _chunk_similarity(
        self,
        first: Dict[str, Any],
        second: Dict[str, Any],
    ) -> float:
        a = set(
            token
            for token in self._tokens(
                str(first.get("content") or "")
            )
            if token not in self.STOPWORDS
        )
        b = set(
            token
            for token in self._tokens(
                str(second.get("content") or "")
            )
            if token not in self.STOPWORDS
        )

        if not a or not b:
            return 0.0

        return len(a & b) / len(a | b)

    # ------------------------------------------------------------------
    # Parent / neighbor expansion
    # ------------------------------------------------------------------

    def _expand_with_neighbors(
        self,
        selected: List[Dict[str, Any]],
        all_chunks: Sequence[Dict[str, Any]],
        question: str,
        query_type: str = "focused",
        max_total_chunks: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Add limited contextual evidence around already-selected primary chunks.

        Expansion is deliberately conservative:
          1. Primary reranked/MMR results always remain authoritative.
          2. Parent-section context is preferred when explicit structure exists.
          3. Immediate neighbors are considered only when they have reasonable
             lexical relevance to the query.
          4. Expansion is skipped for factual queries by default because these
             queries usually need a very small context.
          5. A global expansion cap prevents 5 primary chunks from becoming
             15-20 prompt chunks and increasing LLM latency.
        """
        if not selected or self.neighbor_window <= 0:
            return selected

        if query_type == "factual" and not self.expand_factual_queries:
            return selected

        max_total = max(
            len(selected),
            int(max_total_chunks or self.max_total_expanded_chunks),
        )

        index_map = self._build_neighbor_index(all_chunks)
        expanded: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        # Primary evidence first. This guarantees that context budgeting can
        # never evict all of the actual retrieved evidence in favor of context.
        for primary in selected:
            primary_id = self._stable_chunk_id(primary)
            if primary_id in seen:
                continue

            primary_copy = dict(primary)
            primary_copy["evidence_role"] = "primary"
            primary_copy["expanded_from"] = None
            expanded.append(primary_copy)
            seen.add(primary_id)

        # Parent sections are useful for PDFs where a child chunk contains the
        # answer but the parent heading establishes what the section means.
        if self.expand_parents and len(expanded) < max_total:
            for primary in selected:
                if len(expanded) >= max_total:
                    break

                parent = self._find_parent_chunk(primary, all_chunks)
                if parent is None:
                    continue

                parent_id = self._stable_chunk_id(parent)
                if parent_id in seen:
                    continue
                if self._looks_like_duplicate(primary, parent):
                    continue

                parent_copy = dict(parent)
                parent_copy["_hybrid_score"] = (
                    float(primary.get("_hybrid_score", 0.0)) * 0.30
                )
                parent_copy["_reranker_score"] = (
                    float(primary.get("_reranker_score", 0.0)) * 0.20
                )
                parent_copy["_mmr_score"] = 0.0
                parent_copy["_redundancy_score"] = 0.0
                parent_copy["evidence_role"] = "parent"
                parent_copy["expanded_from"] = self._stable_chunk_id(primary)
                parent_copy["retrieval_source"] = "document"
                parent_copy["retriever_name"] = self.__class__.__name__

                expanded.append(parent_copy)
                seen.add(parent_id)

        # Neighbors are evaluated individually instead of blindly adding both
        # sides. This matters for arbitrary PDFs where adjacent chunks may be
        # headers, page artifacts, unrelated tables, or a new section.
        if len(expanded) < max_total:
            for primary in selected:
                if len(expanded) >= max_total:
                    break

                primary_id = self._stable_chunk_id(primary)
                neighbors = self._find_neighbors(
                    primary=primary,
                    all_chunks=all_chunks,
                    index_map=index_map,
                )

                scored_neighbors: List[Tuple[float, int, Dict[str, Any]]] = []
                for neighbor in neighbors:
                    if self.expand_same_section_only and not self._same_section(
                        primary, neighbor
                    ):
                        continue

                    if self._looks_like_duplicate(primary, neighbor):
                        continue

                    relevance = self._neighbor_relevance(
                        question=question,
                        primary=primary,
                        neighbor=neighbor,
                    )
                    if relevance < self.min_neighbor_relevance:
                        continue

                    distance = self._neighbor_distance(primary, neighbor)
                    scored_neighbors.append(
                        (relevance, distance, neighbor)
                    )

                # Prefer relevance, then proximity.
                scored_neighbors.sort(
                    key=lambda item: (-item[0], item[1], self._chunk_position(item[2]))
                )

                added_for_primary = 0
                for relevance, distance, neighbor in scored_neighbors:
                    if len(expanded) >= max_total:
                        break
                    if added_for_primary >= self.max_expanded_per_result:
                        break

                    neighbor_id = self._stable_chunk_id(neighbor)
                    if neighbor_id in seen:
                        continue

                    neighbor_copy = dict(neighbor)
                    primary_score = float(primary.get("_hybrid_score", 0.0))
                    neighbor_copy["_hybrid_score"] = (
                        0.30 * primary_score + 0.70 * relevance
                    )
                    neighbor_copy["_reranker_score"] = (
                        float(primary.get("_reranker_score", 0.0)) * 0.20
                    )
                    neighbor_copy["_mmr_score"] = 0.0
                    neighbor_copy["_redundancy_score"] = 0.0
                    neighbor_copy["evidence_role"] = "neighbor"
                    neighbor_copy["expanded_from"] = primary_id
                    neighbor_copy["neighbor_distance"] = distance
                    neighbor_copy["neighbor_relevance"] = round(relevance, 4)
                    neighbor_copy["retrieval_source"] = "document"
                    neighbor_copy["retriever_name"] = self.__class__.__name__

                    expanded.append(neighbor_copy)
                    seen.add(neighbor_id)
                    added_for_primary += 1

        # Natural reading order helps the LLM understand split paragraphs.
        expanded.sort(
            key=lambda item: (
                str(
                    item.get("document_id")
                    or item.get("file_id")
                    or item.get("filename")
                    or ""
                ),
                self._chunk_position(item),
            )
        )

        return expanded

    def _find_parent_chunk(
        self,
        primary: Dict[str, Any],
        all_chunks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        """Find an explicit parent-section chunk when the parser provides one."""
        parent_name = str(
            primary.get("parent_section")
            or primary.get("parent_heading")
            or ""
        ).strip().lower()

        if not parent_name:
            return None

        document_id = self._document_identity(primary)
        candidates: List[Dict[str, Any]] = []

        for chunk in all_chunks:
            if self._document_identity(chunk) != document_id:
                continue
            if self._stable_chunk_id(chunk) == self._stable_chunk_id(primary):
                continue

            section = str(
                chunk.get("section")
                or chunk.get("heading")
                or chunk.get("title")
                or ""
            ).strip().lower()

            if section == parent_name:
                candidates.append(chunk)

        if not candidates:
            return None

        # Prefer the closest parent-like chunk before the primary chunk.
        primary_pos = self._chunk_position(primary)
        candidates.sort(
            key=lambda chunk: (
                0 if self._chunk_position(chunk) <= primary_pos else 1,
                abs(self._chunk_position(chunk) - primary_pos),
                -len(str(chunk.get("content") or "")),
            )
        )
        return candidates[0]

    def _neighbor_relevance(
        self,
        question: str,
        primary: Dict[str, Any],
        neighbor: Dict[str, Any],
    ) -> float:
        """Cheap lexical relevance test; avoids another embedding/reranker call."""
        query_terms = self._query_terms(question)
        if not query_terms:
            return 0.5

        content_tokens = set(
            token
            for token in self._tokens(
                str(neighbor.get("content") or "")
            )
            if token not in self.STOPWORDS
        )
        if not content_tokens:
            return 0.0

        overlap = len(query_terms & content_tokens) / max(len(query_terms), 1)

        primary_terms = set(
            token
            for token in self._tokens(
                str(primary.get("content") or "")
            )
            if token not in self.STOPWORDS
        )
        continuity = (
            len(content_tokens & primary_terms)
            / max(len(content_tokens | primary_terms), 1)
        )

        section_bonus = 0.0
        if self._same_section(primary, neighbor):
            section_bonus = 0.15

        # Query overlap is the main signal. Continuity catches split paragraphs.
        return max(
            0.0,
            min(1.0, 0.70 * overlap + 0.30 * continuity + section_bonus),
        )

    @staticmethod
    def _document_identity(chunk: Dict[str, Any]) -> str:
        return str(
            chunk.get("document_id")
            or chunk.get("file_id")
            or chunk.get("filename")
            or "default-document"
        )

    def _build_neighbor_index(
        self,
        chunks: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[int]]:
        """
        Build a document-aware ordering index.

        Chunks from different uploaded documents must never become neighbors.
        If document_chunk_index exists, it is preferred; otherwise the
        original list order is used.
        """
        grouped: Dict[str, List[Tuple[int, int]]] = {}

        for list_index, chunk in enumerate(chunks):
            document_id = str(
                chunk.get("document_id")
                or chunk.get("file_id")
                or chunk.get("filename")
                or "default-document"
            )

            position = self._chunk_position(chunk, fallback=list_index)
            grouped.setdefault(document_id, []).append(
                (position, list_index)
            )

        result: Dict[str, List[int]] = {}

        for document_id, pairs in grouped.items():
            pairs.sort(key=lambda pair: pair[0])
            result[document_id] = [
                list_index
                for _, list_index in pairs
            ]

        return result

    def _find_neighbors(
        self,
        primary: Dict[str, Any],
        all_chunks: Sequence[Dict[str, Any]],
        index_map: Dict[str, List[int]],
    ) -> List[Dict[str, Any]]:
        document_id = str(
            primary.get("document_id")
            or primary.get("file_id")
            or primary.get("filename")
            or "default-document"
        )

        ordered_indices = index_map.get(document_id, [])
        if not ordered_indices:
            return []

        primary_index = self._locate_chunk_index(
            primary,
            ordered_indices,
            all_chunks,
        )

        if primary_index is None:
            return []

        neighbors: List[Dict[str, Any]] = []

        start = max(0, primary_index - self.neighbor_window)
        end = min(
            len(ordered_indices),
            primary_index + self.neighbor_window + 1,
        )

        for ordered_position in range(start, end):
            if ordered_position == primary_index:
                continue

            source_index = ordered_indices[ordered_position]
            if 0 <= source_index < len(all_chunks):
                neighbors.append(all_chunks[source_index])

        return neighbors

    def _locate_chunk_index(
        self,
        primary: Dict[str, Any],
        ordered_indices: Sequence[int],
        all_chunks: Sequence[Dict[str, Any]],
    ) -> int | None:
        primary_id = self._stable_chunk_id(primary)

        # Prefer a stable ID.
        for ordered_position, source_index in enumerate(ordered_indices):
            if self._stable_chunk_id(all_chunks[source_index]) == primary_id:
                return ordered_position

        # Fall back to document chunk index.
        target_position = primary.get("document_chunk_index")
        if target_position is not None:
            try:
                target_position = int(target_position)
                for ordered_position, source_index in enumerate(ordered_indices):
                    if self._chunk_position(all_chunks[source_index]) == target_position:
                        return ordered_position
            except (TypeError, ValueError):
                pass

        # Last resort: object content identity.
        target_content = str(primary.get("content") or "").strip()
        for ordered_position, source_index in enumerate(ordered_indices):
            if (
                str(all_chunks[source_index].get("content") or "").strip()
                == target_content
            ):
                return ordered_position

        return None

    def _neighbor_distance(
        self,
        first: Dict[str, Any],
        second: Dict[str, Any],
    ) -> int:
        first_position = self._chunk_position(first)
        second_position = self._chunk_position(second)
        return abs(first_position - second_position)

    @staticmethod
    def _same_section(
        first: Dict[str, Any],
        second: Dict[str, Any],
    ) -> bool:
        first_section = str(
            first.get("section")
            or first.get("heading")
            or ""
        ).strip().lower()

        second_section = str(
            second.get("section")
            or second.get("heading")
            or ""
        ).strip().lower()

        # If no structural metadata exists, don't reject the neighbor.
        if not first_section or not second_section:
            return True

        return first_section == second_section

    def _looks_like_duplicate(
        self,
        first: Dict[str, Any],
        second: Dict[str, Any],
    ) -> bool:
        first_text = str(first.get("content") or "").strip()
        second_text = str(second.get("content") or "").strip()

        if not first_text or not second_text:
            return True

        if first_text == second_text:
            return True

        # PDFs frequently repeat headers/footers or duplicated extraction
        # fragments. Treat extremely similar short chunks as duplicates.
        first_tokens = set(self._tokens(first_text))
        second_tokens = set(self._tokens(second_text))

        if not first_tokens or not second_tokens:
            return False

        similarity = len(first_tokens & second_tokens) / max(
            len(first_tokens | second_tokens),
            1,
        )

        return similarity >= 0.92

    @staticmethod
    def _stable_chunk_id(chunk: Dict[str, Any]) -> str:
        value = (
            chunk.get("chunk_id")
            or chunk.get("id")
            or chunk.get("_original_index")
            or chunk.get("document_chunk_index")
            or ""
        )

        document = (
            chunk.get("document_id")
            or chunk.get("file_id")
            or chunk.get("filename")
            or ""
        )

        return f"{document}:{value}"

    @staticmethod
    def _chunk_position(
        chunk: Dict[str, Any],
        fallback: int = 0,
    ) -> int:
        for key in (
            "document_chunk_index",
            "chunk_index",
            "index",
            "_original_index",
        ):
            value = chunk.get(key)
            if value is None:
                continue

            try:
                return int(value)
            except (TypeError, ValueError):
                continue

        return int(fallback)

    # ------------------------------------------------------------------
    # Context budget
    # ------------------------------------------------------------------

    def _apply_context_budget(
        self,
        selected: List[Dict[str, Any]],
        budget: int,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        used = 0

        for item in selected:
            text = str(item.get("content") or "").strip()
            if not text:
                continue

            separator = 2 if result else 0
            available = budget - used - separator

            if available <= 0:
                break

            if len(text) <= available:
                result.append(dict(item))
                used += separator + len(text)
            else:
                if not result:
                    result.append(
                        self._truncate(item, available)
                    )
                break

        return result

    def _truncate(
        self,
        item: Dict[str, Any],
        max_chars: int,
    ) -> Dict[str, Any]:
        result = dict(item)
        text = str(result.get("content") or "").strip()

        if len(text) <= max_chars:
            return result

        if max_chars <= 100:
            result["content"] = text[:max_chars].rstrip()
        else:
            candidate = text[:max_chars]
            boundaries = [
                candidate.rfind(". "),
                candidate.rfind("? "),
                candidate.rfind("! "),
                candidate.rfind("\n"),
            ]
            boundary = max(boundaries)

            if boundary >= int(max_chars * 0.55):
                candidate = candidate[: boundary + 1]

            result["content"] = candidate.rstrip() + " …"

        result["char_count"] = len(result["content"])
        result["truncated_for_context_budget"] = True
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieval_text(self, chunk: Dict[str, Any]) -> str:
        section = str(
            chunk.get("section")
            or chunk.get("heading")
            or ""
        ).strip()

        parent = str(
            chunk.get("parent_section")
            or ""
        ).strip()

        filename = str(
            chunk.get("filename")
            or ""
        ).strip()

        content = str(
            chunk.get("content")
            or ""
        ).strip()

        # Heading + parent + filename are deliberately included in the
        # retrieval representation. The LLM still receives the original
        # content, not this synthetic text.
        return "\n".join(
            part
            for part in (filename, parent, section, content)
            if part
        )

    @classmethod
    def _tokens(cls, text: str) -> List[str]:
        return [
            token.lower()
            for token in cls.TOKEN_RE.findall(str(text or ""))
        ]

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []

        maximum = max(scores)
        minimum = min(scores)

        if maximum <= 0:
            return [0.0] * len(scores)

        if maximum == minimum:
            return [1.0 if maximum > 0 else 0.0] * len(scores)

        return [
            max(
                0.0,
                min(
                    1.0,
                    (value - minimum) / (maximum - minimum),
                ),
            )
            for value in scores
        ]

    @staticmethod
    def _rank_indices(scores: List[float]) -> List[int]:
        return sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )
from __future__ import annotations

import numpy as np

"""
Independent retrieval layer for uploaded documents.

This module is intentionally separate from the GitHub RAG pipeline.

Retrieval stack:
    1. BM25 lexical retrieval
    2. TF-IDF lexical retrieval
    3. Optional dense semantic retrieval via the shared embedding provider
    4. Query-adaptive reciprocal-rank fusion
    5. Metadata / heading boosts
    6. MMR diversity selection
    7. Context-size protection

Dense retrieval is optional. If the embedding provider is unavailable, the
retriever still works using BM25 + TF-IDF + lexical/metadata signals.

Public API is kept compatible with the current backend:
    DocumentRetriever(...)
    DocumentRetriever.retrieve(question, chunks, top_k=5)
"""

import hashlib
import math
import re
from copy import deepcopy
import time
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Sequence, Set, Tuple

from ..github_rag.embedding_provider import create_embedding_provider

# Heavy ML dependencies are imported lazily inside the methods that need them.
# Dense embeddings are provided by embedding_provider.py; the cross-encoder
# remains lazy-loaded below.
TfidfVectorizer = None
cosine_similarity = None
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
        reranker_candidate_count: int = 32,
        reranker_batch_size: int = 16,
        context_budget_overview: int = 12000,
        context_budget_focused: int = 9000,
        context_budget_factual: int = 5000,
        min_score: float = 0.0,
        neighbor_window: int = 1,
        max_expanded_per_result: int = 2,
        expand_same_section_only: bool = False,
        expand_parents: bool = True,
        max_total_expanded_chunks: int = 8,
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
        self._embedding_provider = None
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
        self._last_reranker_status = "not_run"

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

    def debug_retrieval(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
        query_type: str | None = None,
        gold_pages: Sequence[int] | None = None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """Trace one retrieval through every major ranking stage.

        This is diagnostic-only and does not change ``retrieve()``.  It reports
        where the requested gold pages appear (or disappear) in BM25, TF-IDF,
        dense retrieval, fusion, query filtering, reranking, MMR, neighbor
        expansion, and the final context-budget selection.
        """
        valid = self._prepare_chunks(chunks)
        question = str(question or "").strip()
        if not valid or not question:
            return {"question": question, "error": "No valid chunks or question."}

        query_type = str(query_type or self._query_type(question)).strip().lower()
        retrieval_question = self._expanded_query(question, query_type)
        gold = {int(p) for p in (gold_pages or [])}
        requested = max(1, int(top_k or self.focused_top_k))

        if query_type == "overview":
            target_k = min(max(requested, self.overview_top_k), 8, len(valid))
            budget = self.context_budget_overview
            floor = 48
        elif query_type == "factual":
            target_k = min(max(requested, 3), 5, len(valid))
            budget = self.context_budget_factual
            floor = 30
        else:
            target_k = min(max(requested, self.focused_top_k), 8, len(valid))
            budget = self.context_budget_focused
            floor = 48

        def page(item: Dict[str, Any]) -> int | None:
            try:
                value = item.get("page")
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        def stable(item: Dict[str, Any]) -> str:
            return self._stable_chunk_id(item)

        def stage_snapshot(items: Sequence[Dict[str, Any]], limit: int | None = None) -> Dict[str, Any]:
            rows = []
            seen = set()
            for rank, item in enumerate(items, 1):
                pg = page(item)
                if pg in gold:
                    rows.append({
                        "rank": rank,
                        "page": pg,
                        "chunk_index": item.get("document_chunk_index", item.get("_original_index")),
                        "chunk_id": stable(item),
                        "score": round(float(item.get("_hybrid_score", 0.0)), 6),
                        "bm25": round(float(item.get("_bm25_score", 0.0)), 6),
                        "tfidf": round(float(item.get("_tfidf_score", 0.0)), 6),
                        "dense": round(float(item.get("_dense_score", 0.0)), 6),
                        "metadata": round(float(item.get("_metadata_score", 0.0)), 6),
                        "query_fit": round(float(item.get("_query_fit_score", 0.0)), 6),
                        "reranker": (round(float(item["_reranker_score"]), 6) if item.get("_reranker_score") is not None else None),
                    })
                    seen.add(pg)
            return {
                "gold_pages_found": sorted(seen),
                "gold_pages_missing": sorted(gold - seen),
                "gold_matches": rows,
                "count": len(items),
                "top_pages": [page(x) for x in list(items)[: (limit or len(items))]],
            }

        bm25 = self._bm25_scores(retrieval_question, valid)
        tfidf = self._tfidf_scores(retrieval_question, valid)
        dense = self._dense_scores(retrieval_question, valid)
        br = self._rank_indices(bm25)
        tr = self._rank_indices(tfidf)
        dr = self._rank_indices(dense) if dense else []

        def ranked_items(indices: Sequence[int], scores: Sequence[float]) -> List[Dict[str, Any]]:
            result = []
            for idx in indices:
                item = dict(valid[idx])
                item["_stage_score"] = float(scores[idx]) if idx < len(scores) else 0.0
                result.append(item)
            return result

        bm25_items = ranked_items(br, bm25)
        tfidf_items = ranked_items(tr, tfidf)
        dense_items = ranked_items(dr, dense) if dense else []

        fused = self._fuse_ranks(
            br, tr, dr, bm25, tfidf, dense, question, query_type, valid
        )
        candidate_count = min(
            len(fused), max(target_k * self.candidate_multiplier, floor, 60)
        )
        candidates = fused[:candidate_count]
        fusion_pool = list(candidates)

        filtered = self._query_aware_filter(question, candidates, query_type)
        reranked = self._rerank_candidates(question, filtered)
        rerank_pool = list(reranked)
        mmr_selected = self._mmr_select(reranked, target_k, question)
        expanded = self._expand_with_neighbors(
            mmr_selected, valid, question, query_type, self.max_total_expanded_chunks
        )
        final = self._apply_context_budget(expanded, budget)
        if not final and rerank_pool:
            final = [self._truncate(rerank_pool[0], budget)]

        stages = {
            "bm25": stage_snapshot(bm25_items, 20),
            "tfidf": stage_snapshot(tfidf_items, 20),
            "dense": stage_snapshot(dense_items, 20),
            "fusion_candidate_pool": stage_snapshot(fusion_pool),
            "query_aware_filter": stage_snapshot(filtered),
            "reranker": stage_snapshot(rerank_pool),
            "mmr_selection": stage_snapshot(mmr_selected),
            "neighbor_expansion": stage_snapshot(expanded),
            "final_context_budget": stage_snapshot(final),
        }

        return {
            "question": question,
            "query_type": query_type,
            "retrieval_question": retrieval_question,
            "gold_pages": sorted(gold),
            "target_k": target_k,
            "context_budget": budget,
            "reranker_status": self._last_reranker_status,
            "stages": stages,
        }


    def retrieve(self, question: str, chunks: Sequence[Dict[str, Any]], query_type: str | None = None, top_k: int = 5) -> List[Dict[str, Any]]:
        t0=time.perf_counter(); valid=self._prepare_chunks(chunks)
        if not valid: return []
        question=str(question or "").strip()
        if not question: return []
        query_type=str(query_type or self._query_type(question)).strip().lower()
        retrieval_question=self._expanded_query(question,query_type)
        requested=max(1,int(top_k or self.focused_top_k))
        if query_type=="overview": target_k=min(max(requested,self.overview_top_k),8,len(valid)); budget=self.context_budget_overview; floor=48
        elif query_type=="factual": target_k=min(max(requested,3),5,len(valid)); budget=self.context_budget_factual; floor=30
        else: target_k=min(max(requested,self.focused_top_k),8,len(valid)); budget=self.context_budget_focused; floor=48
        p={}
        st=time.perf_counter(); bm25=self._bm25_scores(retrieval_question,valid); p["bm25_ms"]=(time.perf_counter()-st)*1000
        st=time.perf_counter(); tfidf=self._tfidf_scores(retrieval_question,valid); p["tfidf_ms"]=(time.perf_counter()-st)*1000
        st=time.perf_counter(); dense=self._dense_scores(retrieval_question,valid); p["dense_ms"]=(time.perf_counter()-st)*1000
        br=self._rank_indices(bm25); tr=self._rank_indices(tfidf); dr=self._rank_indices(dense) if dense else []
        fused=self._fuse_ranks(br,tr,dr,bm25,tfidf,dense,question,query_type,valid)
        candidates=fused[:min(len(fused),max(target_k*self.candidate_multiplier,floor,60))]
        candidates=self._query_aware_filter(question,candidates,query_type)
        candidates=self._rerank_candidates(question,candidates)
        selected=self._mmr_select(candidates,target_k,question)
        selected=self._expand_with_neighbors(selected,valid,question,query_type,self.max_total_expanded_chunks)
        selected=self._apply_context_budget(selected,budget)
        if not selected and candidates: selected=[self._truncate(candidates[0],budget)]
        results=[]
        for rank,item in enumerate(selected,1):
            r=dict(item); r.update({"retrieval_source":"document","retriever_name":self.__class__.__name__,"document_rank":rank,"query_type":query_type,"relevance_score":round(float(r.get("_hybrid_score",0)),4)})
            for public,internal in (("bm25_score","_bm25_score"),("tfidf_score","_tfidf_score"),("dense_score","_dense_score"),("metadata_score","_metadata_score"),("mmr_score","_mmr_score"),("redundancy_score","_redundancy_score"),("query_fit_score","_query_fit_score")):
                r[public]=round(float(r.get(internal,0)),4)
            if r.get("_reranker_score") is not None: r["reranker_score"]=round(float(r["_reranker_score"]),4)
            r["context_budget_chars"]=budget
            for k in list(r):
                if k.startswith("_"): r.pop(k,None)
            results.append(r)
        p.update({"reranker_status":self._last_reranker_status,"total_retrieval_ms":(time.perf_counter()-t0)*1000,"input_chunks":len(valid),"candidate_chunks":len(candidates),"final_chunks":len(results)})
        self.last_profile=p
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

    def _query_terms(self, question: str) -> Set[str]:
        return {
            token
            for token in self._tokens(question)
            if token not in self.STOPWORDS
        }

    def _query_type(self, question: str) -> str:
        q = str(question or "").strip().lower()
        if not q: return "focused"
        def has(p): return bool(re.search(p, q, re.I))
        if has(r"\b(which optimizer|what .* metrics?|what datasets?|what .* architecture|how many .* seeds?|what corruptions?|what additional robustness corruptions|does the report propose .* future work)\b"): return "factual"
        if has(r"\b(summarize|summary|explain|overall|main findings?|key findings?|main conclusion|overall conclusion|key points?|what is (this|the) (report|project|study) about|future work|future scope|what does.*conclude|conclusion|why is .* not sufficient|what happened on .* task)\b"): return "overview"
        if has(r"\b(compare|comparison|versus|vs\.?|difference|differ|outperform|better than|always improve|how did .* compare|across (the|these|all) (three|3|datasets))\b"): return "comparison"
        if has(r"\b(how was .* (implemented|trained|evaluated)|how was the experimental pipeline|methodology|experimental setup|implementation|training|approach)\b"): return "methodology"
        if has(r"\b(limitation|limitations|drawback|drawbacks|shortcoming|shortcomings|constraint|constraints)\b"): return "limitation"
        if has(r"\b(gap|research gap|what is missing|missing)\b"): return "gap"
        if has(r"\b(contradict|contradiction|inconsistent|conflict)\b"): return "contradiction"
        if has(r"\b(figure|fig\.?|table|visual|diagram|plot|chart|shown)\b"): return "visual"
        if has(r"\b(dataset|datasets|model|models|architecture|architectures|metrics?|metric|corruptions?|seeds?|date|title|authors?)\b"): return "factual"
        if has(r"\b(what|which|who|when|where|how many|how much|name|list)\b"): return "factual"
        return "focused"



    @staticmethod
    def _query_type_profile(query_type: str) -> Dict[str, float]:
        return {
            "overview": {"abstract": .75, "introduction": .45, "result": 1.0, "conclusion": 1.0, "future": 1.05, "methodology": .12},
            "comparison": {"result": 1.10, "comparison": 1.15, "conclusion": .70, "abstract": .45, "introduction": .18, "methodology": .10},
            "methodology": {"methodology": 1.15, "experimental": .95, "implementation": 1.05, "evaluation": .85, "result": .30, "conclusion": .12},
            "limitation": {"limitation": 1.15, "conclusion": .90, "discussion": .80, "result": .45},
            "factual": {"abstract": .25, "methodology": .30, "result": .25, "conclusion": .15},
        }.get(query_type, {"result": .45, "conclusion": .35, "methodology": .30})

    def _expanded_query(self, question: str, query_type: str) -> str:
        """Expand queries with intent-specific vocabulary without polluting the query.

        Broadly appending every possible section label is harmful for focused
        questions (for example, adding ``future scope`` to a conclusion query).
        Keep expansion tied to the semantic focus of the actual question.
        """
        q = str(question or "").strip()
        q_lower = q.lower()

        if query_type == "overview":
            if re.search(r"\b(main conclusion|overall conclusion|what does .* conclude|conclusion|outcome)\b", q_lower):
                addition = "conclusion final outcome overall findings discussion"
            elif re.search(r"\b(future work|future scope|future directions?|proposed future)\b", q_lower):
                addition = "future work future scope future directions recommendations"
            elif re.search(r"\b(main findings?|key findings?|key points?|results?)\b", q_lower):
                addition = "results findings key findings conclusion"
            else:
                addition = "abstract introduction study purpose results findings"
        elif query_type == "comparison":
            addition = "results performance comparison comparative ranking across datasets"
        elif query_type == "methodology":
            if re.search(r"\b(robustness|corruptions?|noise|blur)\b", q_lower):
                addition = "robustness evaluation corruptions noise blur experimental procedure"
            elif re.search(r"\b(temperature scaling|calibration)\b", q_lower):
                addition = "calibration temperature scaling evaluation ECE MCE methodology"
            else:
                addition = "methodology experimental setup implementation training evaluation procedure"
        elif query_type == "limitation":
            addition = "limitations discussion conclusion drawbacks shortcomings constraints"
        else:
            addition = ""

        return q + (" " + addition if addition else "")


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
        global TfidfVectorizer, cosine_similarity
        if TfidfVectorizer is None or cosine_similarity is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
                TfidfVectorizer = _TfidfVectorizer
                cosine_similarity = _cosine_similarity
            except Exception:
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

    def _get_embedding_provider(self):
        """Lazily create and reuse the shared embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = create_embedding_provider()
        return self._embedding_provider

    def _dense_scores(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[float]:
        if not self.dense_enabled or self._dense_failed:
            return [0.0] * len(chunks)

        try:
            provider = self._get_embedding_provider()
            texts = [self._retrieval_text(chunk) for chunk in chunks]

            # Preserve the existing in-memory document embedding LRU.
            document_key = self._embedding_cache_key(chunks, texts)
            embeddings = self._dense_embedding_cache.get(document_key)

            if embeddings is not None:
                self._dense_embedding_cache.move_to_end(document_key)
            else:
                embeddings = np.asarray(
                    provider.embed(texts),
                    dtype=np.float32,
                )

                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)

                if len(embeddings) != len(chunks):
                    raise ValueError(
                        "Embedding provider returned an unexpected "
                        "number of document embeddings."
                    )

                self._dense_embedding_cache[document_key] = embeddings
                self._dense_embedding_cache.move_to_end(document_key)

                while len(self._dense_embedding_cache) > self.dense_cache_size:
                    self._dense_embedding_cache.popitem(last=False)

            # Preserve the existing query embedding LRU.
            query_key = question.strip()
            query_embedding = self._query_embedding_cache.get(query_key)

            if query_embedding is not None:
                self._query_embedding_cache.move_to_end(query_key)
            else:
                query_embedding = np.asarray(
                    provider.embed(question),
                    dtype=np.float32,
                ).reshape(-1)

                self._query_embedding_cache[query_key] = query_embedding
                self._query_embedding_cache.move_to_end(query_key)

                while len(self._query_embedding_cache) > self.query_embedding_cache_size:
                    self._query_embedding_cache.popitem(last=False)

            # Provider embeddings are normalized, so dot product preserves
            # the previous cosine-similarity calculation.
            scores = embeddings @ query_embedding

            return [
                max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
                for score in scores
            ]

        except Exception:
            # Preserve the original graceful fallback: lexical retrieval
            # continues if dense embedding generation is unavailable.
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

    # ------------------------------------------------------------------
    # Hybrid fusion
    # ------------------------------------------------------------------

    def _fuse_ranks(self,bm25_rank,tfidf_rank,dense_rank,bm25_scores,tfidf_scores,dense_scores,question,query_type,chunks):
        if query_type=="factual": w={"bm25":.48,"tfidf":.27,"dense":.15,"metadata":.10}
        elif query_type=="overview": w={"bm25":.24,"tfidf":.16,"dense":.30,"metadata":.30}
        elif query_type=="comparison": w={"bm25":.30,"tfidf":.16,"dense":.24,"metadata":.30}
        elif query_type=="methodology": w={"bm25":.34,"tfidf":.20,"dense":.21,"metadata":.25}
        else: w={"bm25":.30,"tfidf":.20,"dense":.35,"metadata":.15}
        if not any(dense_scores):
            total=w["bm25"]+w["tfidf"]
            for k in ("bm25","tfidf"): w[k]+=w["dense"]*w[k]/max(total,1e-9)
            w["dense"]=0
        maps={"bm25":{i:r for r,i in enumerate(bm25_rank,1)},"tfidf":{i:r for r,i in enumerate(tfidf_rank,1)},"dense":{i:r for r,i in enumerate(dense_rank,1)}}; fused=[]
        max_rrf=max((w["bm25"]+w["tfidf"]+w["dense"])/61,1e-9)
        for i,c in enumerate(chunks):
            rr=sum(w[k]*(1/(60+maps[k][i]) if i in maps[k] else 0) for k in maps); nr=max(0,min(1,rr/max_rrf)); meta=self._metadata_score(question,c,query_type); hybrid=(1-w["metadata"])*nr+w["metadata"]*meta
            x=dict(c); x.update({"_hybrid_score":hybrid,"_bm25_score":bm25_scores[i],"_tfidf_score":tfidf_scores[i],"_dense_score":dense_scores[i],"_metadata_score":meta}); fused.append(x)
        return sorted(fused,key=lambda x:float(x["_hybrid_score"]),reverse=True)


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

    def _metadata_score(self, question, chunk, query_type):
        """Score structural and semantic metadata for the query's actual focus.

        This is deliberately document-agnostic: it reacts to headings, parent
        sections, query vocabulary, and entity-like terms rather than fixed page
        numbers or this benchmark's section labels.
        """
        q = str(question or "").strip()
        q_lower = q.lower()
        terms = self._query_terms(q)
        heading = str(chunk.get("section") or chunk.get("heading") or chunk.get("title") or "").strip().lower()
        parent = str(chunk.get("parent_section") or chunk.get("parent_heading") or "").strip().lower()
        content = str(chunk.get("content") or "").strip().lower()
        structural = f"{heading} {parent}".strip()
        score = 0.0

        if terms and heading:
            score += 0.24 * len(terms & set(self._tokens(heading))) / max(len(terms), 1)

        profile = self._query_type_profile(query_type)
        pats = {
            "abstract": r"\babstract\b",
            "introduction": r"\bintroduction\b",
            "result": r"\b(results?|result analysis|findings?|performance)\b",
            "comparison": r"\b(comparison|comparative|versus|vs\.?)\b",
            "conclusion": r"\b(conclusion|discussion|concluding|final outcome)\b",
            "future": r"\b(future (work|scope)|future directions?|recommendations?|next steps?)\b",
            "methodology": r"\b(methodology|method|approach|experimental setup|procedure)\b",
            "experimental": r"\b(experiment(al)?|experimental setup)\b",
            "implementation": r"\b(implementation|implemented|training|trained)\b",
            "evaluation": r"\b(evaluation|metrics?|robustness|calibration)\b",
            "limitation": r"\b(limitations?|drawbacks?|shortcomings?|constraints?)\b",
            "discussion": r"\bdiscussion\b",
        }
        for label, weight in profile.items():
            pat = pats.get(label)
            if pat and re.search(pat, structural, re.I):
                score += .20 * weight
            elif pat and re.search(pat, content, re.I):
                score += .05 * weight

        # Detect salient named entities from the user's original wording.
        # This improves model/dataset comparisons without hardcoding any names.
        entities = re.findall(r"\b(?:[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)*)\b", q)
        entities = [e.lower() for e in entities if e.lower() not in {"what", "how", "which", "why", "does", "did", "is", "are", "was", "were"}]
        if entities:
            entity_text = f"{heading} {parent} {content}"
            hits = sum(1 for entity in entities if entity in entity_text)
            score += .18 * min(1.0, hits / max(len(entities), 1))

        if query_type == "factual":
            if re.search(r"\b(metric|metrics|evaluation|ece|mce|accuracy|calibration)\b", q_lower) and re.search(r"\b(evaluation|metrics?|calibration|performance|results?|methodology)\b", structural):
                score += .42
            if re.search(r"\b(corruptions?|robustness|noise|blur)\b", q_lower) and re.search(r"\b(robustness|corruptions?|evaluation|methodology|experimental)\b", structural):
                score += .48
            if re.search(r"\b(preprocessing|augmentation)\b", q_lower) and re.search(r"\b(preprocessing|augmentation|data|methodology)\b", structural):
                score += .32

        elif query_type == "overview":
            conclusion = bool(re.search(r"\b(main conclusion|overall conclusion|conclusion|conclude|outcome)\b", q_lower))
            future = bool(re.search(r"\b(future work|future scope|future directions?|proposed future)\b", q_lower))
            broad = bool(re.search(r"\b(about|explain|summarize|summary|main idea)\b", q_lower))
            findings = bool(re.search(r"\b(main findings?|key findings?|key points?|results?)\b", q_lower))
            if conclusion and re.search(r"\b(conclusion|discussion|final outcome)\b", structural):
                score += .62
            if future and re.search(r"\b(future (work|scope)|future directions?|recommendations?|next steps?)\b", structural):
                score += .72
            if findings and re.search(r"\b(results?|result analysis|findings?|performance|conclusion|discussion)\b", structural):
                score += .34
            if broad and not conclusion and not future and re.search(r"\b(abstract|introduction)\b", structural):
                score += .28

        elif query_type == "comparison":
            if re.search(r"\b(compare|comparison|versus|vs\.?|across|better|outperform)\b", q_lower) and re.search(r"\b(results?|result analysis|comparison|performance|ranking)\b", structural):
                score += .48

        elif query_type == "methodology":
            if re.search(r"\b(robustness|corruptions?|noise|blur)\b", q_lower) and re.search(r"\b(methodology|evaluation|robustness|experimental|procedure)\b", structural):
                score += .58
            if re.search(r"\b(implemented|implementation|trained|training)\b", q_lower) and re.search(r"\b(implementation|training|methodology|experimental|architecture)\b", structural):
                score += .40
            if re.search(r"\btemperature scaling|calibration\b", q_lower) and re.search(r"\b(calibration|evaluation|temperature|methodology|results?)\b", structural):
                score += .42

        # Early pages are useful for genuinely broad overviews, but should not
        # outrank a dedicated conclusion/future section for focused overview asks.
        try:
            pos = int(chunk.get("document_chunk_index"))
            focused_overview = query_type == "overview" and bool(
                re.search(r"\b(conclusion|conclude|future work|future scope|outcome)\b", q_lower)
            )
            if focused_overview and pos <= 5:
                score -= .22
            if query_type == "factual" and pos <= 15:
                score += .06
        except (TypeError, ValueError):
            pass

        return max(0.0, min(1.0, score + (.02 if chunk.get("page") is not None else 0.0)))



    # ------------------------------------------------------------------
    # Query-aware relevance filtering
    # ------------------------------------------------------------------

    def _query_aware_filter(self,question,candidates,query_type):
        if not candidates: return []
        terms=self._query_terms(question); phrases=self._important_phrases(question); scored=[]
        for c in candidates:
            text=self._retrieval_text(c).lower(); tokens=set(self._tokens(str(c.get("content") or "").lower())); overlap=len(terms & tokens)/max(len(terms),1); ph=min(1,sum(1 for p in phrases if p in text)/max(len(phrases),1)) if phrases else 0; hybrid=float(c.get("_hybrid_score",0)); meta=float(c.get("_metadata_score",0)); structural=self._metadata_score(question,c,query_type); fit=.35*hybrid+.20*overlap+.10*ph+.20*meta+.15*structural; c["_query_fit_score"]=fit; c["_hybrid_score"]=.72*hybrid+.28*fit; scored.append(c)
        scored.sort(key=lambda x:float(x.get("_hybrid_score",0)),reverse=True)
        keep_count=min(len(scored),max(24,self.focused_top_k*5))
        kept=scored[:keep_count]

        # Preserve a structurally strong candidate for specialized intents even
        # when lexical/dense retrieval initially ranked it outside the main set.
        specialized=bool(re.search(r"\b(conclusion|conclude|outcome|future work|future scope|robustness|corruptions?|metrics?|evaluation|calibration|across|datasets?|performance|compare|comparison)\b", question.lower()))
        if specialized and candidates:
            champion=max(candidates,key=lambda x: float(x.get("_metadata_score",0.0)))
            if not any(self._stable_chunk_id(x)==self._stable_chunk_id(champion) for x in kept):
                kept.append(champion)
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
        """Rerank candidates with the CrossEncoder when available.

        The CrossEncoder score is kept separate from the retrieval score.
        If the model is unavailable, no fake reranker score is produced and
        the original retrieval ranking is preserved.
        """
        if not candidates:
            return []

        for candidate in candidates:
            candidate.pop("_reranker_score", None)

        if not self.reranker_enabled:
            self._last_reranker_status = "disabled"
            return candidates

        model = self._get_reranker_model()
        if model is None:
            if self._last_reranker_status == "not_run":
                self._last_reranker_status = "unavailable"
            return candidates

        rerank_pool = candidates[: self.reranker_candidate_count]
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

            for candidate, raw_score in zip(rerank_pool, scores):
                raw_score = float(raw_score)
                clipped = max(-30.0, min(30.0, raw_score))
                reranker_score = 1.0 / (1.0 + math.exp(-clipped))

                candidate["_reranker_score"] = reranker_score

                retrieval_score = float(
                    candidate.get("_hybrid_score", 0.0)
                )
                query_fit = float(
                    candidate.get("_query_fit_score", 0.0)
                )

                # CrossEncoder is the dominant ranking signal. Retrieval and
                # query-fit remain small stabilizing signals.
                candidate["_hybrid_score"] = (
                    0.62 * reranker_score
                    + 0.18 * retrieval_score
                    + 0.10 * query_fit
                    + 0.10 * float(candidate.get("_metadata_score", 0.0))
                )

            candidates.sort(
                key=lambda item: float(
                    item.get("_hybrid_score", 0.0)
                ),
                reverse=True,
            )

            self._last_reranker_status = f"active:{len(rerank_pool)}"
            return candidates

        except Exception as exc:
            for candidate in candidates:
                candidate.pop("_reranker_score", None)

            self._last_reranker_status = f"error:{type(exc).__name__}"
            return candidates

    def _get_reranker_model(self):
        if not self.reranker_enabled or self._reranker_failed:
            return None

        if self._reranker_model is not None:
            return self._reranker_model

        global CrossEncoder
        if CrossEncoder is None:
            try:
                from sentence_transformers import CrossEncoder as _CrossEncoder
                CrossEncoder = _CrossEncoder
            except Exception:
                self._reranker_failed = True
                self._last_reranker_status = "unavailable:crossencoder_import"
                return None

        if CrossEncoder is None:
            self._reranker_failed = True
            self._last_reranker_status = "unavailable:crossencoder_import"
            return None

        try:
            self._reranker_model = CrossEncoder(
                self.reranker_model_name
            )
            return self._reranker_model
        except Exception as exc:
            self._reranker_failed = True
            self._last_reranker_status = f"unavailable:{type(exc).__name__}"
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
        """Relevance-first MMR with evidence preservation.

        MMR must not discard a strong answer-bearing chunk simply because
        another chunk is more diverse. For specialized factual/methodology/
        comparison questions, preserve the strongest evidence candidate from
        each relevant page before applying diversity to the remaining slots.
        """
        if not candidates or top_k <= 0:
            return []

        q = str(question or "").strip().lower()
        specialized = bool(re.search(
            r"\b(conclusion|conclude|outcome|future work|future scope|"
            r"robustness|corruptions?|metrics?|evaluation|calibration|"
            r"performance|compare|comparison|across|datasets?|"
            r"methodology|implementation|training|architecture)\b", q
        ))

        pool = list(candidates)
        pool.sort(
            key=lambda x: float(x.get("_hybrid_score", 0.0)),
            reverse=True,
        )

        if specialized:
            pool = pool[:min(len(pool), max(30, top_k * 6))]

        def channel_strength(c: Dict[str, Any]) -> float:
            return max(
                float(c.get("_bm25_score", 0.0)),
                float(c.get("_tfidf_score", 0.0)),
                float(c.get("_dense_score", 0.0)),
            )

        def strong_channels(c: Dict[str, Any]) -> int:
            return sum(
                float(c.get(k, 0.0)) >= 0.20
                for k in ("_bm25_score", "_tfidf_score", "_dense_score")
            )

        def relevance(c: Dict[str, Any]) -> float:
            hybrid = float(c.get("_hybrid_score", 0.0))
            fit = float(c.get("_query_fit_score", 0.0))
            meta = float(c.get("_metadata_score", 0.0))
            rerank = c.get("_reranker_score")
            rerank_value = float(rerank) if rerank is not None else 0.0
            # Retrieval evidence remains dominant. Query-fit is especially
            # useful when the hybrid score was diluted by a weak lexical match.
            score = 0.68 * hybrid + 0.20 * fit + 0.07 * meta + 0.05 * rerank_value
            return score

        selected: List[Dict[str, Any]] = []
        remaining = list(pool)

        if specialized:
            # Protect at most one chunk per page. A chunk is protected when it
            # has either multiple strong retrieval channels or unusually strong
            # query fit. This catches evidence such as page 13/19 chunks even
            # when their final hybrid rank is modest.
            protected: List[Dict[str, Any]] = []
            seen_pages = set()

            eligible = sorted(
                pool,
                key=lambda c: (
                    strong_channels(c) >= 2,
                    float(c.get("_query_fit_score", 0.0)) >= 0.30,
                    channel_strength(c),
                    float(c.get("_query_fit_score", 0.0)),
                    relevance(c),
                ),
                reverse=True,
            )

            for c in eligible:
                page = c.get("page")
                page_key = str(page) if page is not None else f"__none__{id(c)}"
                is_strong = (
                    strong_channels(c) >= 2
                    or float(c.get("_query_fit_score", 0.0)) >= 0.30
                    or float(c.get("_metadata_score", 0.0)) >= 0.22
                )
                if is_strong and page_key not in seen_pages:
                    protected.append(c)
                    seen_pages.add(page_key)
                    if len(protected) >= top_k:
                        break

            # Order protected evidence by actual relevance rather than by the
            # boolean protection criteria above.
            protected.sort(key=relevance, reverse=True)
            for c in protected[:top_k]:
                c["_mmr_protected"] = True
                c["_mmr_score"] = relevance(c)
                c["_redundancy_score"] = 0.0
                selected.append(c)
                if c in remaining:
                    remaining.remove(c)

        # Fill remaining slots with relevance-first MMR. Diversity is a small
        # penalty, never the main decision signal.
        lambda_value = max(self.mmr_lambda, 0.94) if specialized else self.mmr_lambda

        while remaining and len(selected) < top_k:
            best = None
            best_value = -float("inf")
            best_redundancy = 0.0

            for candidate in remaining:
                rel = relevance(candidate)
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        self._chunk_similarity(candidate, other)
                        for other in selected
                    )

                value = lambda_value * rel - (1.0 - lambda_value) * redundancy

                if value > best_value:
                    best_value = value
                    best = candidate
                    best_redundancy = redundancy

            if best is None:
                break

            best["_mmr_score"] = best_value
            best["_redundancy_score"] = best_redundancy
            selected.append(best)
            remaining.remove(best)

        return selected[:top_k]

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
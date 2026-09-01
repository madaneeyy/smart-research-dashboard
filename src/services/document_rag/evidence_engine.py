from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List, Sequence

from src.services.document_rag.document_retriever import DocumentRetriever
from src.services.document_rag.query_classifier import QueryClassifier


class EvidenceEngine:
    """Source-aware evidence orchestration above DocumentRetriever.

    DocumentRetriever remains responsible for lexical/semantic ranking,
    reranking, MMR and local context expansion.  This layer decides whether
    retrieval should happen globally or independently per attached document.
    """

    MULTI_SOURCE_TYPES = {
        "overview",
        "comparison",
        "limitation",
        "gap",
        "contradiction",
    }

    def __init__(
        self,
        retriever: DocumentRetriever,
        overview_per_source: int = 2,
        analysis_per_source: int = 2,
        max_final_evidence: int = 12,
    ) -> None:
        self.retriever = retriever
        self.overview_per_source = max(1, int(overview_per_source))
        self.analysis_per_source = max(1, int(analysis_per_source))
        self.max_final_evidence = max(2, int(max_final_evidence))

    def retrieve(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
        document_ids: Sequence[str],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        ids = [str(value) for value in document_ids if value]
        unique_ids = list(OrderedDict.fromkeys(ids))
        source_chunks = self._group_by_document(chunks, unique_ids)

        classification = QueryClassifier.classify(
            question,
            document_count=len(unique_ids),
        )
        query_type = str(classification["query_type"])

        if not unique_ids:
            return self._empty_result(classification)

        multi_source = len(unique_ids) > 1 and query_type in self.MULTI_SOURCE_TYPES

        if multi_source:
            evidence = self._retrieve_with_source_coverage(
                question=question,
                source_chunks=source_chunks,
                query_type=query_type,
                requested_top_k=top_k,
            )
            strategy = "per_document_coverage"
        else:
            evidence = self.retriever.retrieve(
                question=question,
                chunks=list(chunks),
                top_k=min(max(1, int(top_k)), len(chunks)) if chunks else 1,
            )
            strategy = "global_document_retrieval"

        represented = OrderedDict()
        for item in evidence:
            document_id = self._document_id(item)
            if document_id:
                represented[document_id] = represented.get(document_id, 0) + 1

        coverage = []
        for document_id in unique_ids:
            items = [item for item in evidence if self._document_id(item) == document_id]
            coverage.append(
                {
                    "document_id": document_id,
                    "chunk_count_available": len(source_chunks.get(document_id, [])),
                    "evidence_count": len(items),
                    "covered": bool(items),
                    "filename": self._filename(items[0]) if items else self._filename(source_chunks.get(document_id, [{}])[0]),
                }
            )

        return {
            "query_type": query_type,
            "query_reason": classification["reason"],
            "multi_source": multi_source,
            "strategy": strategy,
            "selected_document_ids": unique_ids,
            "selected_document_count": len(unique_ids),
            "documents_with_evidence": len(represented),
            "coverage_complete": len(represented) == len(unique_ids),
            "source_coverage": coverage,
            "evidence": evidence,
        }

    def _retrieve_with_source_coverage(
        self,
        question: str,
        source_chunks: Dict[str, List[Dict[str, Any]]],
        query_type: str,
        requested_top_k: int,
    ) -> List[Dict[str, Any]]:
        per_source = (
            self.overview_per_source
            if query_type == "overview"
            else self.analysis_per_source
        )

        candidates_by_source: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        for document_id, items in source_chunks.items():
            if not items:
                candidates_by_source[document_id] = []
                continue

            local_k = min(per_source, len(items))
            try:
                candidates = self.retriever.retrieve(
                    question=question,
                    chunks=items,
                    top_k=local_k,
                )
            except Exception:
                candidates = []

            # For a collection overview, the best available representative is
            # useful even when its lexical relevance is weak. The query is about
            # the source itself, so source coverage is intentional.
            if not candidates and query_type == "overview":
                candidates = [dict(items[0])]
                candidates[0]["retrieval_source"] = "document"
                candidates[0]["retriever_name"] = "DocumentRetriever"
                candidates[0]["query_type"] = query_type

            candidates_by_source[document_id] = candidates

        final: List[Dict[str, Any]] = []

        # Round-robin first candidates guarantees source coverage before using
        # extra evidence slots.
        for round_index in range(per_source):
            for document_id, candidates in candidates_by_source.items():
                if round_index >= len(candidates):
                    continue
                item = dict(candidates[round_index])
                item["evidence_role"] = "source_representative" if round_index == 0 else "supporting"
                item["source_coverage_rank"] = round_index + 1
                final.append(item)

        # For analysis queries, additional evidence is useful, but cap the
        # overall prompt size.  Keep representative coverage first.
        total_budget = max(
            len(source_chunks),
            min(self.max_final_evidence, max(int(requested_top_k), len(source_chunks) * per_source)),
        )
        if len(final) > total_budget:
            final = final[:total_budget]

        return final

    @staticmethod
    def _group_by_document(
        chunks: Sequence[Dict[str, Any]],
        document_ids: Sequence[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = OrderedDict(
            (str(document_id), []) for document_id in document_ids
        )

        for chunk in chunks:
            document_id = EvidenceEngine._document_id(chunk)
            if document_id in grouped:
                grouped[document_id].append(dict(chunk))

        return grouped

    @staticmethod
    def _document_id(chunk: Dict[str, Any]) -> str:
        return str(
            chunk.get("document_id")
            or chunk.get("file_id")
            or chunk.get("filename")
            or ""
        )

    @staticmethod
    def _filename(chunk: Dict[str, Any]) -> str:
        return str(
            chunk.get("filename")
            or chunk.get("document_path")
            or chunk.get("path")
            or ""
        )

    @staticmethod
    def _empty_result(classification: Dict[str, object]) -> Dict[str, Any]:
        return {
            "query_type": classification["query_type"],
            "query_reason": classification["reason"],
            "multi_source": False,
            "strategy": "none",
            "selected_document_ids": [],
            "selected_document_count": 0,
            "documents_with_evidence": 0,
            "coverage_complete": False,
            "source_coverage": [],
            "evidence": [],
        }

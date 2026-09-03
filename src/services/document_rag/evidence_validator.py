from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set


class EvidenceValidator:
    """Deterministic evidence-quality checks performed before Qwen generation.

    This is deliberately model-independent. It does not decide whether a
    generated sentence is true. Instead, it determines whether the retrieved
    evidence is strong enough to support source-specific claims and exposes
    uncertainty signals to the answer layer.
    """

    STOPWORDS: Set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has", "have",
        "how", "i", "if", "in", "into", "is", "it", "its", "me", "may",
        "of", "on", "or", "our", "should", "that", "the", "their", "this",
        "to", "was", "were", "what", "when", "where", "which", "who", "why",
        "with", "would", "you", "your", "we", "us", "these", "those", "do",
        "document", "documents", "file", "files", "source", "sources",
    }

    def validate(
        self,
        question: str,
        evidence_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        selected_ids = [
            str(value)
            for value in evidence_result.get("selected_document_ids", [])
            if str(value).strip()
        ]
        packages = list(evidence_result.get("source_coverage") or [])
        evidence = list(evidence_result.get("evidence") or [])
        query_type = str(evidence_result.get("query_type") or "focused")

        # EvidenceEngine stores evidence once at the top level. Build a small
        # per-document index here instead of expecting each coverage package
        # to contain its own evidence list.
        evidence_by_document: Dict[str, List[Dict[str, Any]]] = {}
        for item in evidence:
            document_id = str(
                item.get("document_id")
                or item.get("file_id")
                or item.get("filename")
                or ""
            ).strip()
            if document_id:
                evidence_by_document.setdefault(document_id, []).append(item)

        source_statuses: List[Dict[str, Any]] = []
        for package in packages:
            document_id = str(package.get("document_id") or "").strip()
            source_evidence = evidence_by_document.get(document_id, [])

            # Accept the current EvidenceEngine field name and the older name
            # so this validator remains compatible during the transition.
            normalized_package = dict(package)
            normalized_package["available_chunk_count"] = int(
                package.get("available_chunk_count")
                or package.get("chunk_count_available")
                or 0
            )

            source_statuses.append(
                self._validate_source(
                    question=question,
                    query_type=query_type,
                    package=normalized_package,
                    source_evidence=source_evidence,
                )
            )

        supported_sources = sum(
            1 for item in source_statuses if item["status"] in {"supported", "partially_supported"}
        )
        conflicting = any(
            item["status"] == "conflicting" for item in source_statuses
        )
        insufficient_sources = sum(
            1 for item in source_statuses if item["status"] == "insufficient_evidence"
        )

        coverage_complete = bool(
            selected_ids and supported_sources == len(selected_ids)
        )

        overall_status = self._overall_status(
            selected_count=len(selected_ids),
            supported_sources=supported_sources,
            insufficient_sources=insufficient_sources,
            conflicting=conflicting,
            evidence_count=len(evidence),
        )

        return {
            "status": overall_status,
            "query_type": query_type,
            "selected_source_count": len(selected_ids),
            "represented_source_count": supported_sources,
            "coverage_complete": coverage_complete,
            "insufficient_source_count": insufficient_sources,
            "conflicting_sources": conflicting,
            "source_statuses": source_statuses,
            "grounding_rules": self._grounding_rules(query_type),
        }

    def _validate_source(
        self,
        question: str,
        query_type: str,
        package: Dict[str, Any],
        source_evidence: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        filename = str(package.get("filename") or "Untitled source")
        available = int(package.get("available_chunk_count") or 0)
        evidence_count = len(source_evidence)

        if available <= 0 or evidence_count <= 0:
            return {
                "document_id": str(package.get("document_id") or ""),
                "filename": filename,
                "status": "insufficient_evidence",
                "available_chunk_count": available,
                "evidence_count": evidence_count,
                "query_overlap": 0.0,
                "reason": "No usable evidence was retrieved from this source.",
            }

        query_terms = self._terms(question)
        combined = " ".join(
            str(item.get("content") or "") for item in source_evidence
        )
        content_terms = self._terms(combined)
        overlap = (
            len(query_terms & content_terms) / len(query_terms)
            if query_terms
            else 1.0
        )

        # Overview/comparison/analysis questions need at least a real source
        # representation; exact lexical overlap is not mandatory because dense
        # retrieval may correctly match semantically related passages.
        if query_type in {"overview", "comparison", "limitation", "gap", "contradiction"}:
            if evidence_count >= 2 or overlap >= 0.12:
                status = "supported"
            else:
                status = "partially_supported"
        else:
            if overlap >= 0.20:
                status = "supported"
            elif overlap >= 0.08:
                status = "partially_supported"
            else:
                status = "insufficient_evidence"

        if query_type in {"limitation", "gap", "contradiction"} and evidence_count > 0:
            reason = (
                "Retrieved evidence is available, but the answer must distinguish "
                "explicit source statements from inference."
            )
        elif query_type == "overview":
            reason = "Source has representative evidence for independent characterization."
        else:
            reason = "Retrieved evidence passed the source-level grounding checks."

        return {
            "document_id": str(package.get("document_id") or ""),
            "filename": filename,
            "status": status,
            "available_chunk_count": available,
            "evidence_count": evidence_count,
            "query_overlap": round(overlap, 4),
            "reason": reason,
        }

    @staticmethod
    def _overall_status(
        selected_count: int,
        supported_sources: int,
        insufficient_sources: int,
        conflicting: bool,
        evidence_count: int,
    ) -> str:
        if not selected_count or not evidence_count:
            return "insufficient_evidence"
        if conflicting:
            return "conflicting"
        if supported_sources == selected_count:
            return "supported"
        if supported_sources > 0:
            return "partially_supported"
        if insufficient_sources == selected_count:
            return "insufficient_evidence"
        return "partially_supported"

    @staticmethod
    def _grounding_rules(query_type: str) -> List[str]:
        base = [
            "Use only supplied evidence for source-specific facts.",
            "Keep evidence associated with its own source.",
            "Do not convert missing evidence into a factual negative claim.",
            "When support is weak, explicitly state that evidence is insufficient.",
        ]

        if query_type == "overview":
            base.extend([
                "Describe each selected source independently before synthesis.",
                "Do not infer a relationship between sources merely because they were attached together.",
            ])
        elif query_type == "comparison":
            base.extend([
                "Make source-to-source comparisons only on attributes represented in evidence.",
                "Preserve differing conditions, datasets, methods, and contexts.",
            ])
        elif query_type == "limitation":
            base.extend([
                "Label author-stated limitations as explicit.",
                "Label model-derived weaknesses as inference.",
            ])
        elif query_type == "gap":
            base.extend([
                "Label explicitly stated future work separately from inferred gaps.",
                "Do not claim that a topic is globally unexplored from absence in these sources.",
            ])
        elif query_type == "contradiction":
            base.extend([
                "Report conflicting claims with their original source and conditions.",
                "Do not reconcile conflicting evidence without support.",
            ])

        return base

    def _terms(self, text: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if token not in self.STOPWORDS and len(token) > 1
        }
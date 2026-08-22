from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional


class ContextScope(str, Enum):
    GENERAL = "general"
    GITHUB = "github"
    DOCUMENT = "document"
    HYBRID = "hybrid"


class ContextRouter:
    """Cheap deterministic router that selects the authoritative RAG source."""

    HYBRID_PATTERNS = (
        r"\bcompare\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\.?\b",
        r"\bdifference between\b", r"\bcompared with\b", r"\bcompare .* with\b",
        r"\bcompare .* to\b",
    )

    DOCUMENT_PATTERNS = (
        r"\bmy (?:project|report|paper|thesis|dissertation)\b",
        r"\bthis (?:document|pdf|paper|report|thesis)\b",
        r"\bthe (?:document|pdf|paper|report|thesis)\b",
        r"\bthis file\b", r"\bthe uploaded\b", r"\buploaded (?:document|file|pdf)\b",
        r"\bfinal year project\b", r"\bproject report\b", r"\bresearch paper\b",
        r"\bwhat dataset did\b", r"\bwhat methodology\b",
        r"\bwhat were the (?:limitations|findings|results|contributions)\b",
        r"\bchapter\s+\d+\b", r"\bsection\s+\d+\b", r"\bpage\s+\d+\b",
    )

    GITHUB_PATTERNS = (
        r"\bgithub\b", r"\brepository\b", r"\brepo\b", r"\bcodebase\b",
        r"\bsource code\b", r"\bimplementation\b", r"\bimplemented\b",
        r"\bdefined\b", r"\bwhere is .* defined\b", r"\bwhere is .* implemented\b",
        r"\bclass\b", r"\bfunction\b", r"\bmethod\b", r"\bmodule\b",
        r"\btests?\b", r"\btest suite\b", r"\binstallation\b", r"\bpip\b",
    )

    @classmethod
    def _matches(cls, text: str, patterns: tuple[str, ...]) -> List[str]:
        return [p for p in patterns if re.search(p, text, re.I)]

    @classmethod
    def route(
        cls, question: str, history: Optional[List[Dict[str, str]]] = None,
        github_reference: Optional[Dict[str, Any]] = None, has_documents: bool = False,
        resolved_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        q = str(question or "").strip().lower()
        resolved = str(resolved_query or q).lower()
        gh = cls._matches(q, cls.GITHUB_PATTERNS)
        doc = cls._matches(q, cls.DOCUMENT_PATTERNS)
        hybrid = cls._matches(q, cls.HYBRID_PATTERNS)
        has_github = bool(github_reference and github_reference.get("repository_url"))

        repo_terms=[]
        if github_reference:
            for key in ("owner", "repo"):
                value=str(github_reference.get(key) or "").strip().lower().removesuffix(".git")
                if value: repo_terms.append(value)
        explicit_repo = has_github and any(
            term and re.search(rf"\b{re.escape(term)}\b", q) for term in repo_terms
        )

        if hybrid and has_github and has_documents:
            scope, reason = ContextScope.HYBRID, "explicit comparison across available sources"
        elif explicit_repo:
            scope, reason = ContextScope.GITHUB, "explicit repository reference"
        elif doc and has_documents and not hybrid:
            scope, reason = ContextScope.DOCUMENT, "document-oriented question"
        elif gh and has_github and not doc:
            scope, reason = ContextScope.GITHUB, "repository/code-oriented question"
        elif hybrid and has_github and has_documents:
            scope, reason = ContextScope.HYBRID, "comparison requires both sources"
        elif has_github:
            scope, reason = ContextScope.GITHUB, "active GitHub repository"
        elif has_documents:
            scope, reason = ContextScope.DOCUMENT, "active uploaded document"
        else:
            scope, reason = ContextScope.GENERAL, "no active research source"

        # Short follow-ups such as "what about its tests?" should remain on
        # the current source; explicit document language above still wins.
        if scope == ContextScope.GENERAL and has_documents and any(
            token in resolved for token in ("document", "report", "paper", "project", "pdf")
        ):
            scope, reason = ContextScope.DOCUMENT, "conversation-resolved document reference"

        return {
            "scope": scope.value, "reason": reason,
            "scores": {"github": len(gh), "document": len(doc), "hybrid": len(hybrid)},
            "signals": {"github": gh, "document": doc, "hybrid": hybrid, "explicit_repository": explicit_repo},
        }

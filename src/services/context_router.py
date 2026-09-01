

import re
from enum import Enum
from typing import Any, Dict, List, Optional


class ContextScope(str, Enum):
    GENERAL = "general"
    GITHUB = "github"
    DOCUMENT = "document"
    HYBRID = "hybrid"


class ContextRouter:
    """
    Deterministic router for selected research sources.

    The router decides which source families must participate in retrieval.
    It does not decide whether a source contains the answer; retrieval does.
    """

    DOCUMENT_PATTERNS = (
        r"\bmy\s+(?:project|report|paper|thesis|dissertation)\b",
        r"\bthis\s+(?:document|pdf|paper|report|thesis)\b",
        r"\bthe\s+(?:document|pdf|paper|report|thesis)\b",
        r"\bthis\s+file\b",
        r"\bthe\s+uploaded\b",
        r"\buploaded\s+(?:document|file|pdf)\b",
        r"\bfinal\s+year\s+project\b",
        r"\bproject\s+report\b",
        r"\bresearch\s+paper\b",
        r"\bwhat\s+dataset\b",
        r"\bwhat\s+methodology\b",
        r"\bwhat\s+were\s+the\s+(?:limitations|findings|results|contributions)\b",
        r"\bchapter\s+\d+\b",
        r"\bsection\s+\d+\b",
        r"\bpage\s+\d+\b",
    )

    GITHUB_PATTERNS = (
        r"\bgithub\b",
        r"\brepository\b",
        r"\brepo\b",
        r"\bcodebase\b",
        r"\bsource\s+code\b",
        r"\bimplementation\b",
        r"\bimplemented\b",
        r"\bwhere\s+is\b.*\bdefined\b",
        r"\bwhere\s+is\b.*\bimplemented\b",
        r"\bclass\b",
        r"\bfunction\b",
        r"\bmethod\b",
        r"\bmodule\b",
        r"\btest(?:s)?\b",
        r"\binstallation\b",
        r"\bpip\b",
    )

    CROSS_SOURCE_PATTERNS = (
        # Explicit comparisons / relationships.
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference\s+between\b",
        r"\bcompared\s+with\b",
        r"\brelate\b",
        r"\brelationship\s+between\b",
        r"\bcorrespond\b.*\b(?:paper|report|pdf|document)\b",

        # Explicit source-to-source questions.
        r"\b(?:paper|report|pdf|document)\b.*\b(?:repository|repo|github|implementation|code|codebase)\b",
        r"\b(?:repository|repo|github|implementation|code|codebase)\b.*\b(?:paper|report|pdf|document)\b",

        # "Where can I see these ideas in the repository?" style.
        r"\b(?:where|how|which|what)\b.*\b(?:in|from)\b.*\b(?:repository|repo|github|implementation|codebase)\b",
        r"\b(?:described|discussed|proposed|presented|mentioned)\b.*\b(?:paper|report|pdf|document)\b.*\b(?:repository|repo|github|implementation|codebase)\b",

        # Strong first/second source references.
        r"\baccording\s+to\s+(?:the\s+)?(?:paper|report|pdf|document)\b.*\b(?:repository|repo|code)\b",
        r"\bbased\s+on\s+(?:the\s+)?(?:paper|report|pdf|document)\b.*\b(?:repository|repo|code)\b",
        r"\bwhat\s+does\s+(?:the\s+)?(?:repository|repo|code)\b.*\b(?:paper|report|pdf|document)\b",
    )

    CROSS_SOURCE_TERMS = (
        "both sources",
        "each source",
        "these sources",
        "those sources",
        "the paper",
        "the report",
        "the pdf",
        "the document",
        "the repository",
        "the repo",
        "the code",
        "the implementation",
        "in the repository",
        "in the repo",
        "in the code",
        "according to the paper",
        "according to the report",
        "described in the paper",
        "described in the report",
        "mentioned in the paper",
        "mentioned in the report",
    )

    @classmethod
    def _matches(cls, text: str, patterns: tuple[str, ...]) -> List[str]:
        return [
            pattern
            for pattern in patterns
            if re.search(pattern, text, re.I)
        ]

    @classmethod
    def _cross_source_signal(cls, text: str) -> bool:
        q = text.lower()
        if any(term in q for term in cls.CROSS_SOURCE_TERMS):
            return True
        return bool(cls._matches(q, cls.CROSS_SOURCE_PATTERNS))

    @classmethod
    def route(
        cls,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        github_reference: Optional[Dict[str, Any]] = None,
        has_documents: bool = False,
        has_github: Optional[bool] = None,
        resolved_query: Optional[str] = None,
        previous_scope: Optional[str] = None,
        has_unavailable_documents: bool = False,
    ) -> Dict[str, Any]:
        q = str(question or "").strip().lower()
        resolved = str(resolved_query or q).strip().lower()

        github_matches = cls._matches(q, cls.GITHUB_PATTERNS)
        document_matches = cls._matches(q, cls.DOCUMENT_PATTERNS)

        reference_has_github = bool(
            github_reference
            and github_reference.get("repository_url")
        )
        selected_github = (
            reference_has_github
            if has_github is None
            else bool(has_github)
        )

        both_sources = bool(selected_github and has_documents)
        cross_source = cls._cross_source_signal(q)

        # Explicit cross-source relationship always wins when both source
        # types are actually selected. This is the critical invariant.
        if both_sources and cross_source:
            scope = ContextScope.HYBRID
            reason = "cross-source question with both selected source types"

        elif both_sources:
            # A question explicitly naming only one source can use that source.
            # Truly neutral questions use both because both were deliberately
            # selected for the chat; retrieval decides relevance.
            if document_matches and not github_matches:
                scope = ContextScope.DOCUMENT
                reason = "document-specific question with both source types selected"
            elif github_matches and not document_matches:
                scope = ContextScope.GITHUB
                reason = "GitHub-specific question with both source types selected"
            else:
                scope = ContextScope.HYBRID
                reason = "both source types selected for a neutral research question"

        elif selected_github:
            scope = ContextScope.GITHUB
            reason = "selected GitHub source available"

        elif has_documents:
            scope = ContextScope.DOCUMENT
            reason = "selected document source available"

        elif previous_scope in {
            ContextScope.GITHUB.value,
            ContextScope.DOCUMENT.value,
            ContextScope.HYBRID.value,
        }:
            scope = ContextScope(previous_scope)
            reason = "continuing the active research source"

        else:
            scope = ContextScope.GENERAL
            reason = "no active research source"

        return {
            "scope": scope.value,
            "reason": reason,
            "scores": {
                "github": len(github_matches),
                "document": len(document_matches),
                "hybrid": int(cross_source),
            },
            "signals": {
                "github": github_matches,
                "document": document_matches,
                "cross_source": cross_source,
                "selected_github": selected_github,
                "selected_documents": bool(has_documents),
                "has_unavailable_documents": bool(has_unavailable_documents),
                "resolved_query": resolved,
            },
        }

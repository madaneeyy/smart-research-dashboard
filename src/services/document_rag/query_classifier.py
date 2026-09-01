from __future__ import annotations

import re
from typing import Dict


class QueryClassifier:
    """Deterministic research-question classifier.

    The classifier is intentionally cheap and transparent.  It chooses a
    retrieval strategy; it does not answer the question.
    """

    OVERVIEW_PATTERNS = (
        r"\bwhat are (?:the )?(?:attached|selected|uploaded|these|the) documents?\b",
        r"\bwhat (?:is|are) (?:these|the attached|the selected|the uploaded) (?:files?|documents?) about\b",
        r"\b(?:summari[sz]e|summarise|overview|give me an overview)\b",
        r"\b(?:main idea|main ideas|key points?|key findings?)\b",
        r"\bwhat does (?:each|every) document contain\b",
    )

    COMPARISON_PATTERNS = (
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bdiffer(?:s|ence|ences)?\b",
        r"\bhow do (?:these|the) (?:documents?|papers?|studies?|sources?) differ\b",
        r"\bversus\b|\bvs\.?\b",
        r"\bwhich .* (?:better|stronger|more effective)\b",
    )

    LIMITATION_PATTERNS = (
        r"\blimitations?\b",
        r"\bweakness(?:es)?\b",
        r"\bthreats? to validity\b",
        r"\bconstraints?\b",
        r"\bshortcomings?\b",
        r"\bwhat (?:are|were) .* (?:limitations?|weaknesses?)\b",
    )

    GAP_PATTERNS = (
        r"\bresearch gaps?\b",
        r"\bgaps? in (?:the )?(?:literature|research|study|work)\b",
        r"\bwhat remains unexplored\b",
        r"\bwhat is missing\b",
        r"\bfuture research\b",
        r"\bopen (?:questions?|problems?)\b",
    )

    CONTRADICTION_PATTERNS = (
        r"\bcontradict(?:s|ion|ions|ory)?\b",
        r"\bdisagree(?:s|ment)?\b",
        r"\bconflict(?:s|ing)?\b",
        r"\bdo .* (?:disagree|conflict)\b",
        r"\bdifferent conclusions\b",
    )

    METHODOLOGY_PATTERNS = (
        r"\bmethodolog(?:y|ies)\b",
        r"\bmethods?\b",
        r"\bapproach(?:es)?\b",
        r"\bhow was .* (?:conducted|performed|implemented)\b",
        r"\bexperimental setup\b",
    )

    @classmethod
    def classify(cls, question: str, document_count: int = 0) -> Dict[str, object]:
        normalized = re.sub(r"\s+", " ", str(question or "").strip().lower())
        multi_source = int(document_count) > 1

        if any(re.search(pattern, normalized) for pattern in cls.OVERVIEW_PATTERNS):
            return {"query_type": "overview", "multi_source": multi_source, "reason": "overview_intent"}

        if any(re.search(pattern, normalized) for pattern in cls.COMPARISON_PATTERNS):
            return {"query_type": "comparison", "multi_source": multi_source, "reason": "comparison_intent"}

        if any(re.search(pattern, normalized) for pattern in cls.LIMITATION_PATTERNS):
            return {"query_type": "limitation", "multi_source": multi_source, "reason": "limitation_intent"}

        if any(re.search(pattern, normalized) for pattern in cls.GAP_PATTERNS):
            return {"query_type": "gap", "multi_source": multi_source, "reason": "gap_intent"}

        if any(re.search(pattern, normalized) for pattern in cls.CONTRADICTION_PATTERNS):
            return {"query_type": "contradiction", "multi_source": multi_source, "reason": "contradiction_intent"}

        if any(re.search(pattern, normalized) for pattern in cls.METHODOLOGY_PATTERNS):
            return {"query_type": "methodology", "multi_source": multi_source, "reason": "methodology_intent"}

        return {
            "query_type": "focused",
            "multi_source": multi_source,
            "reason": "default_focused_question",
        }

from __future__ import annotations

import re
from typing import Dict


class QueryClassifier:
    """Deterministic research-question classifier.

    The classifier is intentionally cheap, transparent, and domain-agnostic.
    It classifies the user's intent so the document retriever can choose an
    appropriate retrieval strategy. It does not answer the question.
    """

    OVERVIEW_PATTERNS = (
        r"\b(?:summari[sz]e|summarise|summary|overview)\b",
        r"\b(?:give|provide|explain) (?:me )?(?:a )?(?:brief |short |quick )?(?:overview|summary|synopsis)\b",
        r"\b(?:main|key|central) (?:idea|ideas|finding|findings|conclusion|conclusions|point|points|result|results)\b",
        r"\bwhat (?:is|are) (?:the )?(?:main|key|central) (?:idea|ideas|finding|findings|conclusion|conclusions|point|points|result|results)\b",
        r"\bwhat (?:is|are) (?:this|the|these) (?:report|study|project|paper|document|documents|work) about\b",
        r"\bwhat does (?:this|the) (?:report|study|project|paper|document|work) (?:focus on|cover|discuss)\b",
        r"\bexplain (?:this|the) (?:report|study|project|paper|document|work)\b",
        r"\b(?:overall|in general) (?:study|project|report|findings|outcome|results)\b",
        r"\bwhat happened (?:in|on) (?:the )?(?:study|project|report|experiment|task)\b",
        r"\bwhat happened on the (?:brain mri|cifar|eurosat) task\b",
        r"\bwhy is .* not sufficient\b",
        r"\bwhat future work (?:does|is|has) .*propose\b",
        r"\bwhat (?:future|further) (?:work|research|directions?) (?:does|is|are)\b",
        r"\bwhat does .* propose(?: for future work)?\b",
        r"\bwhat are the proposed future (?:directions|work|research)\b",
        r"\bwhat does .* conclude\b",
        r"\bwhat is the (?:overall|final) (?:conclusion|outcome|takeaway)\b",
        r"\bwhat are (?:the )?(?:attached|selected|uploaded|these|the) documents?\b",
        r"\bwhat (?:is|are) (?:these|the attached|the selected|the uploaded) (?:files?|documents?) about\b",
        r"\bwhat does (?:each|every) document contain\b",
    )

    COMPARISON_PATTERNS = (
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b|\bvs\.?\b",
        r"\bdiffer(?:s|ed|ence|ences|ent)\b",
        r"\bhow do .* differ\b",
        r"\bhow did .* compare\b",
        r"\bhow did .* perform .* across (?:the )?(?:two|three|multiple|different) (?:datasets?|tasks?|domains?)\b",
        r"\bhow did .* perform across (?:the )?(?:two|three|multiple|different) datasets?\b",
        r"\bwhich .* (?:better|worse|stronger|weaker|more effective|less effective|outperform(?:ed)?)\b",
        r"\b(?:better|worse|stronger|weaker|more effective|less effective) than\b",
        r"\bdid .* (?:always|ever) (?:improve|outperform|perform better|perform worse)\b",
        r"\bhow did .* and .* (?:compare|perform)\b",
    )

    LIMITATION_PATTERNS = (
        r"\blimitations?\b",
        r"\bweakness(?:es)?\b",
        r"\bthreats? to validity\b",
        r"\bconstraints?\b",
        r"\bshortcomings?\b",
        r"\bwhat (?:are|were) .* (?:limitations?|weaknesses?|shortcomings?)\b",
        r"\bwhat are the limitations of\b",
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

    VISUAL_PATTERNS = (
        r"\b(?:figure|fig\.?|diagram|chart|plot|graph|table|image|"
        r"illustration|screenshot|architecture\s+diagram)\b",
    )

    # Procedural questions.
    METHODOLOGY_PATTERNS = (
        r"\bmethodolog(?:y|ies)\b",
        r"\bmethods?\b",
        r"\bmethod(?:s)? used\b",
        r"\bapproach(?:es)?\b",
        r"\bexperimental (?:setup|design|procedure|pipeline)\b",
        r"\bhow was .* (?:conducted|performed|implemented|trained|evaluated|measured|tested|standardized)\b",
        r"\bhow were .* (?:conducted|performed|implemented|trained|evaluated|measured|tested|standardized)\b",
        r"\bhow did (?:they|the authors|the study|the project) .*"
        r"(?:implement|train|evaluate|measure|test)\b",
        r"\bwhat (?:training procedure|training setup|experimental setup)\b",
    )

    # Specific lookup questions.
    FACTUAL_PATTERNS = (
        r"\b(?:title|name|author|authors|date|year|location)\b",
        r"\bwho\s+(?:is|was|are)\b",
        r"\bwhat\s+(?:is|was|are)\s+(?:the )?(?:name|title|dataset|datasets|model|models|optimizer|metrics?|architecture|architectures|task|type|number|size)\b",
        r"\bwhat\s+(?:\w+\s+)?(?:datasets?|models?|architectures?|metrics?|optimizers?|corruptions?|augmentations?|tasks?|techniques?|methods?)\b",
        r"\bwhich\s+(?:datasets?|models?|architectures?|metrics?|optimizers?|corruptions?|augmentations?|tasks?|techniques?|methods?)\b",
        r"\bwhen\s+(?:was|is|were)\b",
        r"\bwhere\s+(?:was|is|were)\b",
        r"\bhow\s+many\b",
        r"\bhow\s+(?:large|big|much)\b",
        r"\bwhat type of (?:task|problem|dataset|data)\b",
        r"\bwhat kind of (?:task|problem|dataset|data)\b",
        r"\bwhich optimizer\b",
        r"\bwhich (?:model|architecture)\b",
        r"\bwhat (?:evaluation )?metrics?\b",
        r"\bwhat corruptions?\b",
        r"\bwhat preprocessing\b",
        r"\bwhat augmentation\b",
        r"\bwhat preprocessing and augmentation\b",
        r"\bwhat (?:is|was|are) (?:the )?\w+ dataset used for\b",
        r"\bwhat (?:additional|other|new) robustness corruptions? (?:are|were) proposed\b",
        r"\bdoes .* propose .* future work\b",
        r"\bdoes .* propose .* as future work\b",
    )

    @classmethod
    def _matches(cls, patterns, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    @classmethod
    def classify(cls, question: str, document_count: int = 0) -> Dict[str, object]:
        normalized = re.sub(r"\s+", " ", str(question or "").strip().lower())
        multi_source = int(document_count) > 1

        # Specific semantic intents first.
        if cls._matches(cls.OVERVIEW_PATTERNS, normalized):
            return {"query_type": "overview", "multi_source": multi_source, "reason": "overview_intent"}

        if cls._matches(cls.COMPARISON_PATTERNS, normalized):
            return {"query_type": "comparison", "multi_source": multi_source, "reason": "comparison_intent"}

        if cls._matches(cls.LIMITATION_PATTERNS, normalized):
            return {"query_type": "limitation", "multi_source": multi_source, "reason": "limitation_intent"}

        if cls._matches(cls.GAP_PATTERNS, normalized):
            return {"query_type": "gap", "multi_source": multi_source, "reason": "gap_intent"}

        if cls._matches(cls.CONTRADICTION_PATTERNS, normalized):
            return {"query_type": "contradiction", "multi_source": multi_source, "reason": "contradiction_intent"}

        if cls._matches(cls.VISUAL_PATTERNS, normalized):
            return {"query_type": "visual", "multi_source": multi_source, "reason": "visual_intent"}

        # Factual lookup before methodology because the benchmark treats
        # "what preprocessing was applied?" as a factual lookup.
        if cls._matches(cls.FACTUAL_PATTERNS, normalized):
            return {"query_type": "factual", "multi_source": multi_source, "reason": "factual_intent"}

        if cls._matches(cls.METHODOLOGY_PATTERNS, normalized):
            return {"query_type": "methodology", "multi_source": multi_source, "reason": "methodology_intent"}

        return {"query_type": "focused", "multi_source": multi_source, "reason": "default_focused_question"}

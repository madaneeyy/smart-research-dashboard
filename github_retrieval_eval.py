"""
GitHub Retrieval Evaluation Harness
===================================

Phase 1.2 evaluation harness.

This evaluates three independent layers:

1. Query Analysis
   - Identifier extraction
   - Identifier decomposition
   - Intent detection

2. Retrieval
   - Whether an expected file/path is retrieved
   - Rank of the expected result
   - Top-k retrieved paths

3. Repository Discovery / Scalability
   - Whether repository traversal succeeds
   - Whether large repositories hit safety limits

The goal is to distinguish:
    "the query analyzer is wrong"
from:
    "the query analyzer is correct but retrieval is wrong"
from:
    "retrieval is correct but repository discovery failed."

This file is intentionally repository-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable


from src.services.github.github_content import GitHubContentService


# ============================================================================
# Configuration
# ============================================================================

TOP_K = 10
MAX_FILES = 15


# ============================================================================
# Evaluation cases
# ============================================================================

@dataclass
class EvalCase:
    name: str
    repository: str
    category: str
    query: str

    # At least one expected path must match.
    expected_paths: list[str] = field(default_factory=list)

    # Expected query-analysis identifiers.
    expected_identifiers: list[str] = field(default_factory=list)

    # Expected identifier parts.
    expected_identifier_parts: list[str] = field(default_factory=list)

    # Expected intents.
    expected_intents: list[str] = field(default_factory=list)

    # Some cases intentionally don't require identifiers.
    require_identifier: bool = False


EVAL_CASES: list[EvalCase] = [

    # ------------------------------------------------------------------------
    # CTranslate2
    # ------------------------------------------------------------------------

    EvalCase(
        name="ctranslate2_overview",
        repository="https://github.com/OpenNMT/CTranslate2",
        category="overview",
        query="What is CTranslate2 and what problem does it solve?",
        expected_paths=[
            "README.md",
        ],
        expected_identifiers=[
            "ctranslate2",
        ],
        expected_identifier_parts=[
            "c",
            "translate",
            "2",
        ],
        expected_intents=[
            "overview",
        ],
        require_identifier=True,
    ),

    EvalCase(
        name="ctranslate2_generation_options",
        repository="https://github.com/OpenNMT/CTranslate2",
        category="implementation",
        query="How is GenerationOptions implemented?",
        expected_paths=[
            "include/ctranslate2/generation.h",
        ],
        expected_identifiers=[
            "generationoptions",
        ],
        expected_identifier_parts=[
            "generation",
            "options",
        ],
        expected_intents=[
            "implementation",
        ],
        require_identifier=True,
    ),

    EvalCase(
        name="ctranslate2_installation",
        repository="https://github.com/OpenNMT/CTranslate2",
        category="installation",
        query="How do I install CTranslate2 using pip?",
        expected_paths=[
            "README.md",
            "python/README.md",
            "docs/README.md",
        ],
        expected_identifiers=[
            "ctranslate2",
        ],
        expected_identifier_parts=[
            "c",
            "translate",
            "2",
        ],
        expected_intents=[
            "installation",
        ],
        require_identifier=True,
    ),

    # ------------------------------------------------------------------------
    # Python
    # ------------------------------------------------------------------------

    EvalCase(
        name="requests_session",
        repository="https://github.com/psf/requests",
        category="python_symbol",
        query="How is the Session class implemented?",
        expected_paths=[
            "src/requests/sessions.py",
        ],
        expected_identifiers=[
            "session",
        ],
        expected_identifier_parts=[
            "session",
        ],
        expected_intents=[
            "implementation",
        ],
        require_identifier=True,
    ),

    # ------------------------------------------------------------------------
    # Go
    # ------------------------------------------------------------------------

    EvalCase(
        name="cobra_command",
        repository="https://github.com/spf13/cobra",
        category="go_symbol",
        query="How is the Command type implemented?",
        expected_paths=[
            "command.go",
            "command_test.go",
        ],
        expected_identifiers=[
            "command",
        ],
        expected_identifier_parts=[
            "command",
        ],
        expected_intents=[
            "implementation",
        ],
        require_identifier=True,
    ),

    # ------------------------------------------------------------------------
    # Rust
    # ------------------------------------------------------------------------

    EvalCase(
        name="ripgrep_regex",
        repository="https://github.com/BurntSushi/ripgrep",
        category="rust_code",
        query="Where is regex matching implemented?",
        expected_paths=[
            "crates/regex/src",
            "crates/regex",
        ],
        expected_identifiers=[
            "regex",
            "matching",
        ],
        expected_identifier_parts=[
            "regex",
            "matching",
        ],
        expected_intents=[
            "implementation",
        ],
        require_identifier=True,
    ),

    # ------------------------------------------------------------------------
    # TypeScript / JavaScript
    # ------------------------------------------------------------------------

    EvalCase(
        name="nextjs_router",
        repository="https://github.com/vercel/next.js",
        category="typescript_architecture",
        query="How is routing implemented in Next.js?",
        expected_paths=[
            "packages/next/src",
            "packages/next",
        ],
        expected_identifiers=[
            "next.js",
        ],
        expected_identifier_parts=[
            "next",
            "js",
        ],
        expected_intents=[
            "implementation",
            "architecture",
        ],
        require_identifier=True,
    ),

    # ------------------------------------------------------------------------
    # Large monorepo
    # ------------------------------------------------------------------------

    EvalCase(
        name="pytorch_dataloader",
        repository="https://github.com/pytorch/pytorch",
        category="large_monorepo",
        query="How is DataLoader implemented?",
        expected_paths=[
            "torch/utils/data/dataloader.py",
            "torch/utils/data",
        ],
        expected_identifiers=[
            "dataloader",
        ],
        expected_identifier_parts=[
            "data",
            "loader",
        ],
        expected_intents=[
            "implementation",
        ],
        require_identifier=True,
    ),

    # ------------------------------------------------------------------------
    # Basic question
    # ------------------------------------------------------------------------

    EvalCase(
        name="black_overview",
        repository="https://github.com/psf/black",
        category="basic_question",
        query="What is Black and what is it used for?",
        expected_paths=[
            "README.md",
        ],
        expected_identifiers=[
            "black",
        ],
        expected_identifier_parts=[
            "black",
        ],
        expected_intents=[
            "overview",
        ],
        require_identifier=True,
    ),

    # ------------------------------------------------------------------------
    # Documentation-heavy repository
    # ------------------------------------------------------------------------

    EvalCase(
        name="fastapi_dependency",
        repository="https://github.com/fastapi/fastapi",
        category="docs_heavy",
        query="How does dependency injection work in FastAPI?",
        expected_paths=[
            "docs/en/docs/tutorial/dependencies",
            "docs/en/docs/tutorial/dependencies/dependencies.md",
            "fastapi/dependencies",
        ],
        expected_identifiers=[
            "fastapi",
        ],
        expected_identifier_parts=[
            "fast",
            "api",
        ],
        expected_intents=[
            "documentation",
        ],
        require_identifier=True,
    ),
]


# ============================================================================
# Result structures
# ============================================================================

@dataclass
class QueryAnalysisResult:
    passed: bool
    identifier_passed: bool
    decomposition_passed: bool
    intent_passed: bool

    identifiers: list[str]
    identifier_parts: list[str]
    intents: list[str]

    missing_identifiers: list[str]
    missing_parts: list[str]
    missing_intents: list[str]


@dataclass
class RetrievalResult:
    passed: bool
    discovery_failed: bool

    retrieved_paths: list[str]
    matched_expected_paths: list[str]
    expected_ranks: dict[str, int]

    error: str | None = None


@dataclass
class CaseResult:
    case: EvalCase
    elapsed: float

    analysis: QueryAnalysisResult
    retrieval: RetrievalResult

    overall_passed: bool


# ============================================================================
# Utility functions
# ============================================================================

def normalize(value: Any) -> str:
    """
    Normalize strings for comparison.

    We intentionally keep this conservative because repository paths and
    identifiers can contain meaningful punctuation.
    """
    return str(value).strip().lower().replace("\\", "/")


def path_matches(actual: str, expected: str) -> bool:
    """
    Match a retrieved repository path against an expected path.

    Supports:

        exact file:
            src/foo/bar.py

        directory:
            crates/regex/src

        repository subtree:
            packages/next/src
    """

    actual_n = normalize(actual).strip("/")
    expected_n = normalize(expected).strip("/")

    if actual_n == expected_n:
        return True

    # Expected path can represent a directory/subtree.
    if actual_n.startswith(expected_n + "/"):
        return True

    # Allow expected directory to match a returned path that contains it.
    if expected_n.endswith("/src"):
        if actual_n.startswith(expected_n + "/"):
            return True

    return False


def any_expected_path_matches(
    actual_path: str,
    expected_paths: Iterable[str],
) -> list[str]:
    matches = []

    for expected in expected_paths:
        if path_matches(actual_path, expected):
            matches.append(expected)

    return matches


def contains_expected(
    values: Iterable[str],
    expected: Iterable[str],
) -> tuple[list[str], list[str]]:
    """
    Case-insensitive containment check.
    """

    normalized_values = {
        normalize(v)
        for v in values
    }

    found = []
    missing = []

    for item in expected:
        normalized = normalize(item)

        if normalized in normalized_values:
            found.append(item)
        else:
            missing.append(item)

    return found, missing


def get_identifier_parts(analysis: dict[str, Any]) -> list[str]:
    """
    Extract identifier decomposition from the analyzer.

    The current service may expose the decomposition under different keys
    depending on the implementation.

    Supported:

        identifier_parts
        identifiers_parts
        parts
    """

    for key in (
        "identifier_parts",
        "identifiers_parts",
        "parts",
    ):
        value = analysis.get(key)

        if isinstance(value, (list, tuple)):
            return [str(x) for x in value]

    return []


# ============================================================================
# Query analysis evaluation
# ============================================================================

def evaluate_query_analysis(
    case: EvalCase,
    analysis: dict[str, Any],
) -> QueryAnalysisResult:

    identifiers = [
        str(x)
        for x in analysis.get("identifiers", [])
    ]

    identifier_parts = get_identifier_parts(analysis)

    intents = [
        str(x)
        for x in analysis.get("intents", [])
    ]

    # ------------------------------------------------------------------------
    # Identifiers
    # ------------------------------------------------------------------------

    if case.expected_identifiers:

        found_identifiers, missing_identifiers = contains_expected(
            identifiers,
            case.expected_identifiers,
        )

        identifier_passed = len(missing_identifiers) == 0

    else:
        found_identifiers = identifiers
        missing_identifiers = []
        identifier_passed = True

    # ------------------------------------------------------------------------
    # Identifier decomposition
    # ------------------------------------------------------------------------

    if case.expected_identifier_parts:

        found_parts, missing_parts = contains_expected(
            identifier_parts,
            case.expected_identifier_parts,
        )

        decomposition_passed = len(missing_parts) == 0

    else:
        found_parts = identifier_parts
        missing_parts = []
        decomposition_passed = True

    # ------------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------------

    if case.expected_intents:

        found_intents, missing_intents = contains_expected(
            intents,
            case.expected_intents,
        )

        intent_passed = len(missing_intents) == 0

    else:
        found_intents = intents
        missing_intents = []
        intent_passed = True

    # ------------------------------------------------------------------------
    # Required identifier
    # ------------------------------------------------------------------------

    if case.require_identifier and not identifiers:
        identifier_passed = False

        if not missing_identifiers:
            missing_identifiers = list(case.expected_identifiers)

    return QueryAnalysisResult(
        passed=(
            identifier_passed
            and decomposition_passed
            and intent_passed
        ),
        identifier_passed=identifier_passed,
        decomposition_passed=decomposition_passed,
        intent_passed=intent_passed,
        identifiers=identifiers,
        identifier_parts=identifier_parts,
        intents=intents,
        missing_identifiers=missing_identifiers,
        missing_parts=missing_parts,
        missing_intents=missing_intents,
    )


# ============================================================================
# Retrieval evaluation
# ============================================================================

def evaluate_retrieval(
    case: EvalCase,
    retrieved: list[dict[str, Any]],
) -> RetrievalResult:

    retrieved_paths = [
        str(item.get("path", ""))
        for item in retrieved
        if item.get("path")
    ]

    matched_expected_paths: list[str] = []
    expected_ranks: dict[str, int] = {}

    for rank, actual_path in enumerate(retrieved_paths, start=1):

        matches = any_expected_path_matches(
            actual_path,
            case.expected_paths,
        )

        for expected in matches:
            if expected not in matched_expected_paths:
                matched_expected_paths.append(expected)

            if expected not in expected_ranks:
                expected_ranks[expected] = rank

    passed = bool(matched_expected_paths)

    return RetrievalResult(
        passed=passed,
        discovery_failed=False,
        retrieved_paths=retrieved_paths,
        matched_expected_paths=matched_expected_paths,
        expected_ranks=expected_ranks,
    )


# ============================================================================
# Single-case evaluation
# ============================================================================

def run_case(case: EvalCase) -> CaseResult:

    started = time.perf_counter()

    # ------------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------------

    try:
        analysis_dict = (
            GitHubContentService.analyze_query(case.query)
        )

        analysis_result = evaluate_query_analysis(
            case,
            analysis_dict,
        )

    except Exception as exc:

        elapsed = time.perf_counter() - started

        analysis_result = QueryAnalysisResult(
            passed=False,
            identifier_passed=False,
            decomposition_passed=False,
            intent_passed=False,
            identifiers=[],
            identifier_parts=[],
            intents=[],
            missing_identifiers=list(case.expected_identifiers),
            missing_parts=list(case.expected_identifier_parts),
            missing_intents=list(case.expected_intents),
        )

        retrieval_result = RetrievalResult(
            passed=False,
            discovery_failed=False,
            retrieved_paths=[],
            matched_expected_paths=[],
            expected_ranks={},
            error=f"Query analysis error: {type(exc).__name__}: {exc}",
        )

        return CaseResult(
            case=case,
            elapsed=elapsed,
            analysis=analysis_result,
            retrieval=retrieval_result,
            overall_passed=False,
        )

    # ------------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------------

    try:

        retrieved = (
            GitHubContentService.select_relevant_source_files(
                case.repository,
                case.query,
                max_files=MAX_FILES,
            )
        )

        retrieval_result = evaluate_retrieval(
            case,
            retrieved,
        )

    except Exception as exc:

        retrieval_result = RetrievalResult(
            passed=False,
            discovery_failed=True,
            retrieved_paths=[],
            matched_expected_paths=[],
            expected_ranks={},
            error=f"{type(exc).__name__}: {exc}",
        )

    elapsed = time.perf_counter() - started

    # ------------------------------------------------------------------------
    # Overall
    #
    # For this phase, a case passes only when:
    #
    #   1. Query analysis is correct
    #   2. Expected retrieval target is found
    # ------------------------------------------------------------------------

    overall_passed = (
        analysis_result.passed
        and retrieval_result.passed
        and not retrieval_result.discovery_failed
    )

    return CaseResult(
        case=case,
        elapsed=elapsed,
        analysis=analysis_result,
        retrieval=retrieval_result,
        overall_passed=overall_passed,
    )


# ============================================================================
# Printing
# ============================================================================

def print_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def print_case_result(result: CaseResult) -> None:

    case = result.case
    analysis = result.analysis
    retrieval = result.retrieval

    print()
    print("=" * 96)
    print(
        f"{'PASS' if result.overall_passed else 'FAIL'}: "
        f"{case.name}"
    )
    print("-" * 96)

    print(f"Repository : {case.repository.rsplit('/', 1)[-1]}")
    print(f"Category   : {case.category}")
    print(f"Query      : {case.query}")
    print(f"Time       : {result.elapsed:.3f}s")

    # ------------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------------

    print()
    print("QUERY ANALYSIS")

    print(
        f"  Identifier extraction   : "
        f"{print_bool(analysis.identifier_passed)}"
    )

    print(
        f"  Identifier decomposition: "
        f"{print_bool(analysis.decomposition_passed)}"
    )

    print(
        f"  Intent detection        : "
        f"{print_bool(analysis.intent_passed)}"
    )

    print(f"  Identifiers             : {analysis.identifiers}")
    print(f"  Parts                   : {analysis.identifier_parts}")
    print(f"  Intents                 : {analysis.intents}")

    if analysis.missing_identifiers:
        print(
            f"  Missing identifiers     : "
            f"{analysis.missing_identifiers}"
        )

    if analysis.missing_parts:
        print(
            f"  Missing identifier parts: "
            f"{analysis.missing_parts}"
        )

    if analysis.missing_intents:
        print(
            f"  Missing intents         : "
            f"{analysis.missing_intents}"
        )

    # ------------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------------

    print()
    print("RETRIEVAL")

    if retrieval.discovery_failed:

        print("  Repository discovery   : FAIL")
        print(
            f"  Error                   : "
            f"{retrieval.error}"
        )

    else:

        print("  Repository discovery   : PASS")

        print(
            f"  Expected path found    : "
            f"{print_bool(retrieval.passed)}"
        )

        if retrieval.expected_ranks:

            print("  Expected ranks:")

            for path, rank in retrieval.expected_ranks.items():
                print(f"      #{rank:<3} {path}")

        print()
        print("  Top results:")

        for rank, path in enumerate(
            retrieval.retrieved_paths[:TOP_K],
            start=1,
        ):

            marker = ""

            if any_expected_path_matches(
                path,
                case.expected_paths,
            ):
                marker = "  <-- EXPECTED"

            print(
                f"      {rank:>2}. "
                f"{path}"
                f"{marker}"
            )

    # ------------------------------------------------------------------------
    # Overall diagnosis
    # ------------------------------------------------------------------------

    print()
    print("DIAGNOSIS")

    if retrieval.discovery_failed:

        print(
            "  Repository discovery failed. "
            "This is a scalability/network/discovery issue, "
            "not primarily a ranking issue."
        )

    elif not retrieval.passed:

        print(
            "  Retrieval failed: none of the expected paths "
            "were found in the retrieved candidates."
        )

    elif not analysis_result_passed(analysis):

        print(
            "  Retrieval succeeded, but query analysis "
            "is incomplete or incorrect."
        )

    else:

        print(
            "  Query analysis and retrieval both passed."
        )


def analysis_result_passed(
    analysis: QueryAnalysisResult,
) -> bool:
    return (
        analysis.identifier_passed
        and analysis.decomposition_passed
        and analysis.intent_passed
    )


# ============================================================================
# Summary
# ============================================================================

def print_summary(results: list[CaseResult]) -> None:

    total = len(results)

    overall_passed = sum(
        1
        for result in results
        if result.overall_passed
    )

    retrieval_passed = sum(
        1
        for result in results
        if result.retrieval.passed
        and not result.retrieval.discovery_failed
    )

    analysis_passed = sum(
        1
        for result in results
        if analysis_result_passed(result.analysis)
    )

    identifier_passed = sum(
        1
        for result in results
        if result.analysis.identifier_passed
    )

    decomposition_passed = sum(
        1
        for result in results
        if result.analysis.decomposition_passed
    )

    intent_passed = sum(
        1
        for result in results
        if result.analysis.intent_passed
    )

    discovery_passed = sum(
        1
        for result in results
        if not result.retrieval.discovery_failed
    )

    print()
    print()
    print("#" * 96)
    print("PHASE 1.2 RETRIEVAL EVALUATION SUMMARY")
    print("#" * 96)

    print(f"Total cases              : {total}")
    print(
        f"Overall passed           : "
        f"{overall_passed}/{total} "
        f"({overall_passed / total * 100:.1f}%)"
    )

    print(
        f"Query analysis passed    : "
        f"{analysis_passed}/{total} "
        f"({analysis_passed / total * 100:.1f}%)"
    )

    print(
        f"Retrieval hit rate       : "
        f"{retrieval_passed}/{total} "
        f"({retrieval_passed / total * 100:.1f}%)"
    )

    print(
        f"Repository discovery     : "
        f"{discovery_passed}/{total} "
        f"({discovery_passed / total * 100:.1f}%)"
    )

    print()
    print("QUERY ANALYSIS BREAKDOWN")

    print(
        f"  Identifier extraction  : "
        f"{identifier_passed}/{total} "
        f"({identifier_passed / total * 100:.1f}%)"
    )

    print(
        f"  Identifier decomposition: "
        f"{decomposition_passed}/{total} "
        f"({decomposition_passed / total * 100:.1f}%)"
    )

    print(
        f"  Intent detection       : "
        f"{intent_passed}/{total} "
        f"({intent_passed / total * 100:.1f}%)"
    )

    print()
    print("CASE RESULTS")

    for result in results:

        status = "PASS" if result.overall_passed else "FAIL"

        retrieval_status = (
            "DISCOVERY_ERROR"
            if result.retrieval.discovery_failed
            else (
                "PASS"
                if result.retrieval.passed
                else "FAIL"
            )
        )

        analysis_status = (
            "PASS"
            if analysis_result_passed(result.analysis)
            else "FAIL"
        )

        rank_text = "-"

        if result.retrieval.expected_ranks:
            ranks = list(
                result.retrieval.expected_ranks.values()
            )
            rank_text = ",".join(
                str(rank)
                for rank in sorted(ranks)
            )

        print(
            f"  {status:<4} "
            f"{result.case.name:<32} "
            f"analysis={analysis_status:<4} "
            f"retrieval={retrieval_status:<15} "
            f"rank={rank_text}"
        )

    # ------------------------------------------------------------------------
    # Failed cases
    # ------------------------------------------------------------------------

    failed = [
        result
        for result in results
        if not result.overall_passed
    ]

    if failed:

        print()
        print("FAILED CASES")

        for result in failed:

            print(
                f"\n  - {result.case.name}"
            )

            if not analysis_result_passed(result.analysis):

                print("      Query analysis:")

                if result.analysis.missing_identifiers:
                    print(
                        "        Missing identifiers: "
                        f"{result.analysis.missing_identifiers}"
                    )

                if result.analysis.missing_parts:
                    print(
                        "        Missing parts: "
                        f"{result.analysis.missing_parts}"
                    )

                if result.analysis.missing_intents:
                    print(
                        "        Missing intents: "
                        f"{result.analysis.missing_intents}"
                    )

            if result.retrieval.discovery_failed:

                print(
                    "      Discovery error: "
                    f"{result.retrieval.error}"
                )

            elif not result.retrieval.passed:

                print(
                    "      Expected paths not retrieved:"
                )

                for path in result.case.expected_paths:
                    print(f"        - {path}")

                print(
                    "      Top results:"
                )

                for path in result.retrieval.retrieved_paths[:5]:
                    print(f"        - {path}")

            elif result.retrieval.expected_ranks:

                print(
                    "      Retrieval succeeded at rank(s): "
                    f"{list(result.retrieval.expected_ranks.values())}"
                )


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    print("GitHub Retrieval Evaluation Harness")
    print(f"Cases: {len(EVAL_CASES)}")

    # ------------------------------------------------------------------------
    # Cache before
    # ------------------------------------------------------------------------

    try:
        before_cache = (
            GitHubContentService.cache_stats()
        )
    except Exception as exc:
        before_cache = {
            "error": f"{type(exc).__name__}: {exc}"
        }

    print(f"Cache stats before run: {before_cache}")

    # ------------------------------------------------------------------------
    # Run cases
    # ------------------------------------------------------------------------

    results: list[CaseResult] = []

    for case in EVAL_CASES:

        result = run_case(case)

        results.append(result)

        print_case_result(result)

    # ------------------------------------------------------------------------
    # Cache after
    # ------------------------------------------------------------------------

    try:
        after_cache = (
            GitHubContentService.cache_stats()
        )
    except Exception as exc:
        after_cache = {
            "error": f"{type(exc).__name__}: {exc}"
        }

    print(
        f"\nCache stats after run: "
        f"{after_cache}"
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print_summary(results)


if __name__ == "__main__":
    main()
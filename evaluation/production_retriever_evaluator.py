from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from src.services.github.repository_indexer import GitHubRepositoryIndexer
from src.services.github.content_acquirer import GitHubContentAcquirer
from src.services.rag.chunker import DocumentChunker
from src.services.rag.hybrid_retriever import HybridRetriever
from src.services.rag.retriever import SimpleRetriever


# =============================================================================
# CONFIGURATION
# =============================================================================

REPOSITORY_URL = (
    "https://github.com/scikit-learn/scikit-learn.git"
)

GOLD_DATASET = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "scikit_learn_retrieval_gold_v1.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# RETRIEVER CONFIGURATION
# =============================================================================

TOP_K = 5

SEMANTIC_WEIGHT = 0.5
BM25_WEIGHT = 0.5

RRF_K = 60

CANDIDATE_MULTIPLIER = 8

MMR_LAMBDA = 0.70

METADATA_BONUS_WEIGHT = 0.05


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_path(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\\",
        "/",
    )

    text = re.sub(
        r"/+",
        "/",
        text,
    )

    text = text.strip()

    while text.startswith("./"):
        text = text[2:]

    text = text.lstrip("/")

    return text


def normalize_anchor(value: Any) -> str:
    """
    Normalize section / anchor names.

    Examples:

        "Ordinary Least Squares"
        " ordinary   least   squares "

    become the same representation.
    """

    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\n",
        " ",
    )

    text = text.replace(
        "\r",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.strip().lower()

    # RST roles.
    text = re.sub(
        r":class:`([^`]+)`",
        r"\1",
        text,
    )

    text = re.sub(
        r":func:`([^`]+)`",
        r"\1",
        text,
    )

    text = re.sub(
        r":ref:`([^`]+)`",
        r"\1",
        text,
    )

    # Normalize colon spacing.
    text = re.sub(
        r"\s*:\s*",
        ": ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_section(
    value: Any,
) -> str:
    """
    Normalize a potentially hierarchical section.

    Example:

        Ridge regression and classification >
        Setting the regularization parameter:
        leave-one-out Cross-Validation

    """

    text = normalize_anchor(
        value
    )

    text = text.replace(
        "→",
        ">",
    )

    text = re.sub(
        r"\s*>\s*",
        ">",
        text,
    )

    return text


# =============================================================================
# CHUNK IDENTIFICATION
# =============================================================================

def chunk_id(
    chunk: Dict[str, Any],
) -> str:

    path = normalize_path(
        chunk.get("path")
        or chunk.get("file")
        or chunk.get("source")
        or ""
    )

    index = chunk.get(
        "chunk_index"
    )

    if index is not None:
        return (
            f"{path}|{index}"
        )

    content = str(
        chunk.get(
            "content",
            "",
        )
    )

    digest = hashlib.sha1(
        content.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()

    return (
        f"{path}|content:{digest}"
    )


def content_fingerprint(
    chunk: Dict[str, Any],
) -> str:

    content = str(
        chunk.get(
            "content",
            "",
        )
    )

    return hashlib.sha1(
        content.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()


# =============================================================================
# GOLD DATASET
# =============================================================================

def load_gold_dataset() -> List[Dict[str, Any]]:

    if not GOLD_DATASET.exists():
        raise FileNotFoundError(
            f"Gold dataset not found:\n"
            f"{GOLD_DATASET}"
        )

    with open(
        GOLD_DATASET,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if isinstance(
        data,
        dict,
    ):

        queries = data.get(
            "queries"
        )

        if not isinstance(
            queries,
            list,
        ):
            raise ValueError(
                "Gold dataset does not "
                "contain a valid 'queries' list."
            )

    elif isinstance(
        data,
        list,
    ):

        queries = data

    else:

        raise ValueError(
            "Gold dataset must be either "
            "a list or an object containing "
            "'queries'."
        )

    output = []

    for index, item in enumerate(
        queries,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"Gold query #{index} "
                f"is not an object."
            )

        query = (
            item.get("query")
            or item.get("question")
            or item.get("text")
        )

        if not query:
            raise ValueError(
                f"Gold query #{index} "
                f"does not contain a query."
            )

        gold = item.get(
            "gold"
        )

        if not isinstance(
            gold,
            list,
        ):
            raise ValueError(
                f"Gold query #{index} "
                f"does not contain a gold list."
            )

        output.append(
            {
                **item,
                "_query": str(query),
                "_query_index": index,
            }
        )

    return output


# =============================================================================
# GOLD ITEM NORMALIZATION
# =============================================================================

def normalize_gold_item(
    item: Dict[str, Any],
) -> Dict[str, Any]:

    path = normalize_path(
        item.get("path")
        or item.get("file")
        or item.get("source")
        or ""
    )

    anchor = str(
        item.get("anchor")
        or item.get("section")
        or item.get("heading")
        or ""
    ).strip()

    relevance = item.get(
        "relevance",
        1,
    )

    try:
        relevance = int(
            relevance
        )
    except (
        TypeError,
        ValueError,
    ):
        relevance = 1

    if relevance not in (
        1,
        2,
        3,
    ):
        relevance = 1

    chunk_index = item.get(
        "chunk_index"
    )

    if chunk_index is not None:

        try:
            chunk_index = int(
                chunk_index
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    return {
        "path": path,
        "anchor": anchor,
        "normalized_anchor": normalize_anchor(
            anchor
        ),
        "relevance": relevance,
        "chunk_index": chunk_index,
        "raw": item,
    }


def build_gold_items(
    gold_query: Dict[str, Any],
) -> List[Dict[str, Any]]:

    return [
        normalize_gold_item(item)
        for item in gold_query.get(
            "gold",
            [],
        )
    ]


# =============================================================================
# SECTION EXTRACTION
# =============================================================================

def section_candidates(
    result: Dict[str, Any],
) -> List[str]:

    candidates = []

    fields = [
        "section",
        "heading",
        "title",
        "parent_section",
        "parent_heading",
    ]

    for field in fields:

        value = result.get(
            field
        )

        if value:
            candidates.append(
                str(value)
            )

    metadata = result.get(
        "metadata"
    )

    if isinstance(
        metadata,
        dict,
    ):

        for field in fields:

            value = metadata.get(
                field
            )

            if value:
                candidates.append(
                    str(value)
                )

    # Remove duplicates while preserving order.
    output = []

    seen = set()

    for value in candidates:

        normalized = normalize_section(
            value
        )

        if normalized not in seen:

            seen.add(
                normalized
            )

            output.append(
                value
            )

    return output


# =============================================================================
# HIERARCHICAL SECTION PARSING
# =============================================================================

def section_parts(
    section: str,
) -> List[str]:

    normalized = normalize_section(
        section
    )

    if not normalized:
        return []

    parts = [
        part.strip()
        for part in normalized.split(">")
        if part.strip()
    ]

    return parts


def anchor_match_strength(
    gold_anchor: str,
    retrieved_section: str,
) -> int:
    """
    Return the quality of an anchor match.

    0 = no match

    1 = weak containment

    2 = hierarchical component match

    3 = exact anchor match

    The important behavior:

        Gold:
            Ridge regression and classification

        Retrieved:
            Ridge regression and classification >
            Setting the regularization parameter...

        => 2

        Gold:
            Setting the regularization parameter...

        Retrieved:
            Ridge regression and classification >
            Setting the regularization parameter...

        => 2

    Exact standalone section:

        Gold:
            Ordinary Least Squares

        Retrieved:
            Ordinary Least Squares

        => 3
    """

    gold = normalize_anchor(
        gold_anchor
    )

    section = normalize_section(
        retrieved_section
    )

    if not gold or not section:
        return 0

    # -------------------------------------------------------------------------
    # Exact entire-section match
    # -------------------------------------------------------------------------

    if section == gold:
        return 3

    parts = section_parts(
        section
    )

    normalized_parts = [
        normalize_anchor(part)
        for part in parts
    ]

    # -------------------------------------------------------------------------
    # Exact hierarchical component
    # -------------------------------------------------------------------------

    if gold in normalized_parts:
        return 2

    # -------------------------------------------------------------------------
    # Exact final section component
    # -------------------------------------------------------------------------

    if (
        normalized_parts
        and normalized_parts[-1] == gold
    ):
        return 2

    # -------------------------------------------------------------------------
    # Conservative containment
    # -------------------------------------------------------------------------

    if len(
        gold.split()
    ) >= 2:

        if gold in section:
            return 1

    return 0


# =============================================================================
# BEST GOLD MATCH
# =============================================================================

def best_gold_match(
    result: Dict[str, Any],
    gold_items: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    CRITICAL ALIGNMENT FUNCTION.

    The previous evaluator returned the FIRST matching gold anchor.

    That was wrong.

    Example:

        Gold:
            Ridge regression and classification
            Setting the regularization parameter: ...

        Retrieved:
            Ridge regression and classification >
            Setting the regularization parameter: ...

    Both anchors technically occur in the hierarchical path.

    The evaluator must choose the MORE SPECIFIC child anchor.

    We therefore rank candidates using:

        1. exact chunk index
        2. exact anchor
        3. anchor specificity / word count
        4. match strength
        5. relevance

    This prevents child chunks from being incorrectly assigned
    to their parent anchor.
    """

    result_path = normalize_path(
        result.get("path")
        or result.get("file")
        or result.get("source")
        or ""
    )

    result_index = result.get(
        "chunk_index"
    )

    sections = section_candidates(
        result
    )

    candidates = []

    for gold in gold_items:

        if gold["path"] != result_path:
            continue

        # ---------------------------------------------------------------------
        # Exact chunk match if gold explicitly specifies one.
        # ---------------------------------------------------------------------

        exact_chunk = False

        if (
            gold["chunk_index"] is not None
            and result_index is not None
            and str(
                gold["chunk_index"]
            )
            == str(
                result_index
            )
        ):
            exact_chunk = True

        # ---------------------------------------------------------------------
        # Evaluate every section representation.
        # ---------------------------------------------------------------------

        best_strength = 0
        best_section = ""

        for section in sections:

            strength = anchor_match_strength(
                gold["anchor"],
                section,
            )

            if strength > best_strength:

                best_strength = strength
                best_section = section

        if not exact_chunk and best_strength == 0:
            continue

        anchor_length = len(
            gold["normalized_anchor"].split()
        )

        # Exact chunk is overwhelmingly strong.
        exact_chunk_score = (
            1_000_000
            if exact_chunk
            else 0
        )

        # Exact whole-section match beats hierarchical match.
        exact_section_score = (
            100_000
            if best_strength == 3
            else 0
        )

        candidates.append(
            {
                "gold": gold,
                "strength": best_strength,
                "section": best_section,
                "exact_chunk_score":
                    exact_chunk_score,
                "exact_section_score":
                    exact_section_score,
                "anchor_length":
                    anchor_length,
            }
        )

    if not candidates:
        return None

    # -------------------------------------------------------------------------
    # MOST SPECIFIC MATCH WINS
    # -------------------------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            item["exact_chunk_score"],
            item["exact_section_score"],
            item["strength"],
            item["anchor_length"],
            item["gold"]["relevance"],
        ),
        reverse=True,
    )

    return candidates[0]["gold"]


# =============================================================================
# ANCHOR KEY
# =============================================================================

def gold_anchor_key(
    gold: Dict[str, Any],
) -> Tuple[str, str]:

    return (
        gold["path"],
        gold["normalized_anchor"],
    )


def result_anchor_key(
    result: Dict[str, Any],
    gold: Dict[str, Any],
) -> Tuple[str, str]:

    return gold_anchor_key(
        gold
    )


# =============================================================================
# VALIDATE GOLD DATASET
# =============================================================================

def validate_gold_dataset(
    gold_queries: Sequence[Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:

    total = 0
    matched = 0

    missing = []

    for gold_query in gold_queries:

        gold_items = build_gold_items(
            gold_query
        )

        for gold in gold_items:

            total += 1

            found = False

            for chunk in chunks:

                chunk_path = normalize_path(
                    chunk.get("path")
                    or chunk.get("file")
                    or chunk.get("source")
                    or ""
                )

                if (
                    chunk_path
                    != gold["path"]
                ):
                    continue

                sections = section_candidates(
                    chunk
                )

                for section in sections:

                    strength = anchor_match_strength(
                        gold["anchor"],
                        section,
                    )

                    if strength > 0:

                        found = True
                        break

                if found:
                    break

            if found:

                matched += 1

            else:

                missing.append(
                    {
                        "query":
                            gold_query["_query"],
                        "path":
                            gold["path"],
                        "anchor":
                            gold["anchor"],
                        "relevance":
                            gold["relevance"],
                    }
                )

    return {
        "total_gold_items":
            total,

        "matched_gold_items":
            matched,

        "missing_gold_items":
            len(missing),

        "match_rate":
            (
                matched / total
                if total
                else 0.0
            ),

        "missing":
            missing,
    }


# =============================================================================
# RETRIEVED GOLD MATCHES
# =============================================================================

def get_retrieved_matches(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:

    matches = []

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):

        match = best_gold_match(
            result,
            gold_items,
        )

        matches.append(
            {
                "rank": rank,
                "result": result,
                "gold": match,
            }
        )

    return matches


# =============================================================================
# UNIQUE GOLD ANCHORS RETRIEVED
# =============================================================================

def unique_gold_matches(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:

    output = []

    seen = set()

    matches = get_retrieved_matches(
        results,
        gold_items,
        k,
    )

    for item in matches:

        gold = item["gold"]

        if gold is None:
            continue

        key = gold_anchor_key(
            gold
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        output.append(
            {
                **item,
                "anchor_key": key,
            }
        )

    return output


# =============================================================================
# PRECISION
# =============================================================================

def precision_at_k(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int = 5,
) -> float:

    top = results[:k]

    if not top:
        return 0.0

    relevant = 0

    for result in top:

        if (
            best_gold_match(
                result,
                gold_items,
            )
            is not None
        ):
            relevant += 1

    return (
        relevant / len(top)
    )


# =============================================================================
# RECALL
# =============================================================================

def recall_at_k(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int = 5,
) -> float:

    if not gold_items:
        return 0.0

    gold_anchor_set = {
        gold_anchor_key(item)
        for item in gold_items
    }

    retrieved = {
        item["anchor_key"]
        for item in unique_gold_matches(
            results,
            gold_items,
            k,
        )
    }

    return (
        len(
            retrieved
            & gold_anchor_set
        )
        / len(
            gold_anchor_set
        )
    )


# =============================================================================
# PRIMARY RECALL
# =============================================================================

def primary_recall_at_k(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int = 5,
) -> float:

    primary = [
        item
        for item in gold_items
        if item["relevance"] == 3
    ]

    if not primary:
        return 0.0

    primary_keys = {
        gold_anchor_key(item)
        for item in primary
    }

    retrieved = {
        item["anchor_key"]
        for item in unique_gold_matches(
            results,
            gold_items,
            k,
        )
    }

    return (
        len(
            retrieved
            & primary_keys
        )
        / len(
            primary_keys
        )
    )


# =============================================================================
# SUPPORTING RECALL
# =============================================================================

def supporting_recall_at_k(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int = 5,
) -> float:

    supporting = [
        item
        for item in gold_items
        if item["relevance"] == 2
    ]

    if not supporting:
        return 0.0

    supporting_keys = {
        gold_anchor_key(item)
        for item in supporting
    }

    retrieved = {
        item["anchor_key"]
        for item in unique_gold_matches(
            results,
            gold_items,
            k,
        )
    }

    return (
        len(
            retrieved
            & supporting_keys
        )
        / len(
            supporting_keys
        )
    )


# =============================================================================
# ANCHOR COVERAGE
# =============================================================================

def anchor_coverage_at_k(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int = 5,
) -> float:

    gold_keys = {
        gold_anchor_key(item)
        for item in gold_items
    }

    if not gold_keys:
        return 0.0

    retrieved_keys = {
        item["anchor_key"]
        for item in unique_gold_matches(
            results,
            gold_items,
            k,
        )
    }

    return (
        len(
            retrieved_keys
            & gold_keys
        )
        / len(
            gold_keys
        )
    )


# =============================================================================
# HIT RATE
# =============================================================================

def hit_rate_at_k(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int = 5,
) -> float:

    return (
        1.0
        if any(
            best_gold_match(
                result,
                gold_items,
            )
            is not None
            for result in results[:k]
        )
        else 0.0
    )


# =============================================================================
# PRIMARY HIT
# =============================================================================

def primary_hit_at_k(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int = 5,
) -> float:

    primary = [
        item
        for item in gold_items
        if item["relevance"] == 3
    ]

    if not primary:
        return 0.0

    return (
        1.0
        if any(
            best_gold_match(
                result,
                primary,
            )
            is not None
            for result in results[:k]
        )
        else 0.0
    )


# =============================================================================
# MRR
# =============================================================================

def reciprocal_rank(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
) -> float:

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if (
            best_gold_match(
                result,
                gold_items,
            )
            is not None
        ):
            return 1.0 / rank

    return 0.0


# =============================================================================
# PRIMARY MRR
# =============================================================================

def primary_reciprocal_rank(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
) -> float:

    primary = [
        item
        for item in gold_items
        if item["relevance"] == 3
    ]

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if (
            best_gold_match(
                result,
                primary,
            )
            is not None
        ):
            return 1.0 / rank

    return 0.0


# =============================================================================
# NDCG
# =============================================================================

def dcg(
    gains: Sequence[float],
) -> float:

    total = 0.0

    for rank, gain in enumerate(
        gains,
        start=1,
    ):

        total += (
            gain
            / math.log2(
                rank + 1
            )
        )

    return total


def ndcg_at_k(
    results: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    k: int = 5,
) -> float:
    """
    Anchor-level NDCG.

    IMPORTANT:

    If two retrieved chunks belong to the same gold anchor,
    only the FIRST occurrence contributes gain.

    This prevents:

        Log loss
        Log loss

    from being counted as:

        3 + 3

    and producing NDCG > 1.

    """

    actual_gains = []

    seen_anchors = set()

    for result in results[:k]:

        match = best_gold_match(
            result,
            gold_items,
        )

        if match is None:

            actual_gains.append(
                0.0
            )

            continue

        key = gold_anchor_key(
            match
        )

        if key in seen_anchors:

            # Duplicate chunk from the same
            # anchor does not create another
            # independent evidence unit.
            actual_gains.append(
                0.0
            )

        else:

            seen_anchors.add(
                key
            )

            actual_gains.append(
                float(
                    match["relevance"]
                )
            )

    while len(
        actual_gains
    ) < k:

        actual_gains.append(
            0.0
        )

    # -------------------------------------------------------------------------
    # Ideal ranking.
    #
    # Each gold anchor can contribute only once.
    # -------------------------------------------------------------------------

    unique_gold = {}

    for gold in gold_items:

        key = gold_anchor_key(
            gold
        )

        if (
            key not in unique_gold
            or
            gold["relevance"]
            > unique_gold[key]["relevance"]
        ):

            unique_gold[key] = gold

    ideal_gains = sorted(
        [
            float(
                item["relevance"]
            )
            for item in unique_gold.values()
        ],
        reverse=True,
    )[:k]

    while len(
        ideal_gains
    ) < k:

        ideal_gains.append(
            0.0
        )

    actual_dcg = dcg(
        actual_gains
    )

    ideal_dcg = dcg(
        ideal_gains
    )

    if ideal_dcg <= 0:
        return 0.0

    value = (
        actual_dcg
        / ideal_dcg
    )

    # Numerical safety.
    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


# =============================================================================
# DUPLICATES
# =============================================================================

def duplicate_count(
    results: Sequence[Dict[str, Any]],
) -> int:

    seen = set()

    duplicates = 0

    for result in results:

        fingerprint = (
            content_fingerprint(
                result
            )
        )

        if fingerprint in seen:

            duplicates += 1

        else:

            seen.add(
                fingerprint
            )

    return duplicates


def unique_ratio(
    results: Sequence[Dict[str, Any]],
) -> float:

    if not results:
        return 1.0

    unique = len(
        {
            content_fingerprint(
                result
            )
            for result in results
        }
    )

    return (
        unique
        / len(results)
    )


# =============================================================================
# SAME DOCUMENT RATIO
# =============================================================================

def same_document_ratio(
    results: Sequence[Dict[str, Any]],
) -> float:

    if len(results) <= 1:
        return 0.0

    paths = [
        normalize_path(
            result.get("path")
            or result.get("file")
            or result.get("source")
            or ""
        )
        for result in results
    ]

    same_pairs = 0
    total_pairs = 0

    for i in range(
        len(paths)
    ):

        for j in range(
            i + 1,
            len(paths),
        ):

            total_pairs += 1

            if (
                paths[i]
                and paths[i]
                == paths[j]
            ):

                same_pairs += 1

    if total_pairs == 0:
        return 0.0

    return (
        same_pairs
        / total_pairs
    )


# =============================================================================
# SEMANTIC REDUNDANCY
# =============================================================================

def semantic_redundancy(
    results: Sequence[Dict[str, Any]],
) -> Optional[float]:

    if len(results) <= 1:
        return 0.0

    try:

        import numpy as np

        model = (
            SimpleRetriever._get_model()
        )

        texts = [
            str(
                result.get(
                    "content",
                    "",
                )
            )
            for result in results
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        values = []

        for i in range(
            len(matrix)
        ):

            for j in range(
                i + 1,
                len(matrix),
            ):

                values.append(
                    float(
                        np.dot(
                            matrix[i],
                            matrix[j],
                        )
                    )
                )

        if not values:
            return 0.0

        return (
            sum(values)
            / len(values)
        )

    except Exception:

        return None


# =============================================================================
# RESULT SUMMARY
# =============================================================================

def summarize_result(
    result: Dict[str, Any],
    rank: int,
    gold_items: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:

    match = best_gold_match(
        result,
        gold_items,
    )

    sections = section_candidates(
        result
    )

    preview = re.sub(
        r"\s+",
        " ",
        str(
            result.get(
                "content",
                "",
            )
        ),
    ).strip()

    if len(preview) > 500:

        preview = (
            preview[:500]
            + "..."
        )

    return {
        "rank":
            rank,

        "id":
            chunk_id(result),

        "path":
            result.get("path"),

        "chunk_index":
            result.get("chunk_index"),

        "section":
            (
                result.get("section")
                or result.get("heading")
                or result.get("title")
            ),

        "section_candidates":
            sections,

        "semantic_rank":
            result.get("semantic_rank"),

        "bm25_rank":
            result.get("bm25_rank"),

        "similarity":
            result.get("similarity"),

        "bm25_score":
            result.get("bm25_score"),

        "hybrid_score":
            result.get("hybrid_score"),

        "mmr_score":
            result.get("mmr_score"),

        "mmr_relevance":
            result.get("mmr_relevance"),

        "mmr_redundancy":
            result.get("mmr_redundancy"),

        "mmr_lambda":
            result.get("mmr_lambda"),

        "metadata_bonus":
            result.get("metadata_bonus"),

        "metadata_relationship":
            result.get(
                "metadata_relationship"
            ),

        "is_gold":
            match is not None,

        "gold_anchor":
            (
                match["anchor"]
                if match
                else None
            ),

        "gold_relevance":
            (
                match["relevance"]
                if match
                else None
            ),

        "gold_path":
            (
                match["path"]
                if match
                else None
            ),

        "match_type":
            (
                "anchor"
                if match
                else None
            ),

        "preview":
            preview,
    }


# =============================================================================
# QUERY EVALUATION
# =============================================================================

def evaluate_query(
    gold_query: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    retriever: HybridRetriever,
) -> Dict[str, Any]:

    query = gold_query[
        "_query"
    ]

    gold_items = build_gold_items(
        gold_query
    )

    results = retriever.retrieve(
        question=query,
        chunks=chunks,
        top_k=TOP_K,
    )

    summarized = [
        summarize_result(
            result,
            rank,
            gold_items,
        )
        for rank, result in enumerate(
            results,
            start=1,
        )
    ]

    metrics = {
        "precision@5":
            precision_at_k(
                results,
                gold_items,
                5,
            ),

        "recall@5":
            recall_at_k(
                results,
                gold_items,
                5,
            ),

        "primary_recall@5":
            primary_recall_at_k(
                results,
                gold_items,
                5,
            ),

        "supporting_recall@5":
            supporting_recall_at_k(
                results,
                gold_items,
                5,
            ),

        "anchor_coverage@5":
            anchor_coverage_at_k(
                results,
                gold_items,
                5,
            ),

        "primary_hit@5":
            primary_hit_at_k(
                results,
                gold_items,
                5,
            ),

        "hit_rate@5":
            hit_rate_at_k(
                results,
                gold_items,
                5,
            ),

        "ndcg@5":
            ndcg_at_k(
                results,
                gold_items,
                5,
            ),

        "mrr":
            reciprocal_rank(
                results,
                gold_items,
            ),

        "primary_mrr":
            primary_reciprocal_rank(
                results,
                gold_items,
            ),

        "duplicate_count":
            duplicate_count(
                results
            ),

        "unique_ratio":
            unique_ratio(
                results
            ),

        "same_document_ratio":
            same_document_ratio(
                results
            ),

        "semantic_redundancy":
            semantic_redundancy(
                results
            ),

        "returned_count":
            len(results),
    }

    return {
        "query_id":
            gold_query.get(
                "id",
                gold_query.get(
                    "query_id",
                    gold_query[
                        "_query_index"
                    ],
                ),
            ),

        "query_index":
            gold_query[
                "_query_index"
            ],

        "query":
            query,

        "category":
            (
                gold_query.get(
                    "category"
                )
                or gold_query.get(
                    "query_type"
                )
                or "unknown"
            ),

        "gold": [
            {
                "path":
                    item["path"],

                "anchor":
                    item["anchor"],

                "relevance":
                    item["relevance"],

                "chunk_index":
                    item["chunk_index"],
            }
            for item in gold_items
        ],

        "results":
            summarized,

        "metrics":
            metrics,
    }


# =============================================================================
# AGGREGATION
# =============================================================================

METRICS = [
    "precision@5",
    "recall@5",
    "primary_recall@5",
    "supporting_recall@5",
    "anchor_coverage@5",
    "primary_hit@5",
    "hit_rate@5",
    "ndcg@5",
    "mrr",
    "primary_mrr",
    "duplicate_count",
    "unique_ratio",
    "same_document_ratio",
    "semantic_redundancy",
    "returned_count",
]


def aggregate(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:

    output = {
        "query_count":
            len(results)
    }

    for metric in METRICS:

        values = [
            result["metrics"].get(
                metric
            )
            for result in results
            if result["metrics"].get(
                metric
            )
            is not None
        ]

        if values:

            output[metric] = (
                sum(values)
                / len(values)
            )

        else:

            output[metric] = None

    return output


def aggregate_by_category(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:

    groups = defaultdict(list)

    for result in results:

        groups[
            result.get(
                "category",
                "unknown",
            )
        ].append(
            result
        )

    return {
        category:
            aggregate(items)
        for category, items
        in groups.items()
    }


# =============================================================================
# FORMATTING
# =============================================================================

def pct(
    value: Optional[float],
) -> str:

    if value is None:
        return "N/A"

    return (
        f"{value * 100:.2f}%"
    )


def num(
    value: Optional[float],
) -> str:

    if value is None:
        return "N/A"

    return (
        f"{value:.4f}"
    )


# =============================================================================
# TEXT REPORT
# =============================================================================

def generate_report(
    evaluation: Dict[str, Any],
) -> str:

    lines = []

    add = lines.append

    add("=" * 110)
    add(
        "PRODUCTION RETRIEVER — "
        "ANCHOR-AWARE 24 QUERY EVALUATION"
    )
    add("=" * 110)

    add(
        f"Generated: "
        f"{evaluation['timestamp']}"
    )

    add(
        f"Repository: "
        f"{evaluation['repository']}"
    )

    add(
        f"Gold dataset: "
        f"{evaluation['gold_dataset']}"
    )

    add("")
    add(
        "ALIGNMENT METHOD"
    )
    add("-" * 110)

    add(
        "Gold relevance is evaluated at "
        "PATH + ANCHOR level."
    )

    add(
        "When a retrieved section contains "
        "both a parent and child gold anchor, "
        "the most specific matching anchor wins."
    )

    add(
        "Repeated chunks from the same anchor "
        "count only once for Recall, Coverage "
        "and NDCG."
    )

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------

    config = evaluation[
        "configuration"
    ]

    add("")
    add(
        "CONFIGURATION"
    )
    add("-" * 110)

    add(
        f"Top-K                 : "
        f"{config['top_k']}"
    )

    add(
        f"Semantic weight       : "
        f"{config['semantic_weight']}"
    )

    add(
        f"BM25 weight           : "
        f"{config['bm25_weight']}"
    )

    add(
        f"RRF k                 : "
        f"{config['rrf_k']}"
    )

    add(
        f"Candidate multiplier  : "
        f"{config['candidate_multiplier']}"
    )

    add(
        f"MMR lambda            : "
        f"{config['mmr_lambda']}"
    )

    add(
        f"Metadata bonus weight : "
        f"{config['metadata_bonus_weight']}"
    )

    add(
        f"Repository chunks     : "
        f"{config['chunk_count']}"
    )

    add(
        f"Gold queries          : "
        f"{config['gold_query_count']}"
    )

    # -------------------------------------------------------------------------
    # GOLD VALIDATION
    # -------------------------------------------------------------------------

    validation = evaluation[
        "gold_validation"
    ]

    add("")
    add("=" * 110)
    add(
        "GOLD DATASET VALIDATION"
    )
    add("=" * 110)

    add(
        f"Gold items            : "
        f"{validation['total_gold_items']}"
    )

    add(
        f"Matched               : "
        f"{validation['matched_gold_items']}"
    )

    add(
        f"Missing               : "
        f"{validation['missing_gold_items']}"
    )

    add(
        f"Match rate            : "
        f"{pct(validation['match_rate'])}"
    )

    if validation["missing"]:

        add("")
        add(
            "MISSING GOLD ANCHORS"
        )

        for item in validation[
            "missing"
        ]:

            add(
                f"  Query: "
                f"{item['query']}"
            )

            add(
                f"  Path: "
                f"{item['path']}"
            )

            add(
                f"  Anchor: "
                f"{item['anchor']}"
            )

    # -------------------------------------------------------------------------
    # OVERALL
    # -------------------------------------------------------------------------

    overall = evaluation[
        "aggregate"
    ]

    add("")
    add("=" * 110)
    add(
        "OVERALL RETRIEVAL PERFORMANCE"
    )
    add("=" * 110)

    add(
        f"Precision@5           : "
        f"{pct(overall['precision@5'])}"
    )

    add(
        f"Recall@5              : "
        f"{pct(overall['recall@5'])}"
    )

    add(
        f"Primary Recall@5      : "
        f"{pct(overall['primary_recall@5'])}"
    )

    add(
        f"Supporting Recall@5   : "
        f"{pct(overall['supporting_recall@5'])}"
    )

    add(
        f"Anchor Coverage@5     : "
        f"{pct(overall['anchor_coverage@5'])}"
    )

    add(
        f"Primary Hit@5         : "
        f"{pct(overall['primary_hit@5'])}"
    )

    add(
        f"Hit Rate@5            : "
        f"{pct(overall['hit_rate@5'])}"
    )

    add(
        f"NDCG@5                : "
        f"{num(overall['ndcg@5'])}"
    )

    add(
        f"MRR                   : "
        f"{num(overall['mrr'])}"
    )

    add(
        f"Primary MRR           : "
        f"{num(overall['primary_mrr'])}"
    )

    add(
        f"Unique Ratio          : "
        f"{pct(overall['unique_ratio'])}"
    )

    add(
        f"Duplicate Count       : "
        f"{num(overall['duplicate_count'])}"
    )

    add(
        f"Same Document Ratio   : "
        f"{pct(overall['same_document_ratio'])}"
    )

    add(
        f"Semantic Redundancy   : "
        f"{num(overall['semantic_redundancy'])}"
    )

    # -------------------------------------------------------------------------
    # CATEGORY
    # -------------------------------------------------------------------------

    add("")
    add("=" * 110)
    add(
        "CATEGORY BREAKDOWN"
    )
    add("=" * 110)

    for category, data in evaluation[
        "by_category"
    ].items():

        add("")
        add(
            f"[{category}]"
        )

        add(
            f"Queries              : "
            f"{data['query_count']}"
        )

        add(
            f"Precision@5          : "
            f"{pct(data['precision@5'])}"
        )

        add(
            f"Recall@5             : "
            f"{pct(data['recall@5'])}"
        )

        add(
            f"Primary Recall@5     : "
            f"{pct(data['primary_recall@5'])}"
        )

        add(
            f"Supporting Recall@5  : "
            f"{pct(data['supporting_recall@5'])}"
        )

        add(
            f"Anchor Coverage@5    : "
            f"{pct(data['anchor_coverage@5'])}"
        )

        add(
            f"NDCG@5               : "
            f"{num(data['ndcg@5'])}"
        )

        add(
            f"MRR                  : "
            f"{num(data['mrr'])}"
        )

    # -------------------------------------------------------------------------
    # QUERY RESULTS
    # -------------------------------------------------------------------------

    add("")
    add("=" * 110)
    add(
        "QUERY-BY-QUERY RESULTS"
    )
    add("=" * 110)

    for query_result in evaluation[
        "query_results"
    ]:

        metrics = query_result[
            "metrics"
        ]

        add("")
        add(
            f"Q{query_result['query_index']:02d}"
        )

        add(
            f"Query: "
            f"{query_result['query']}"
        )

        add(
            f"Category: "
            f"{query_result['category']}"
        )

        add(
            f"P@5="
            f"{pct(metrics['precision@5'])} "
            f"R@5="
            f"{pct(metrics['recall@5'])} "
            f"PrimaryR@5="
            f"{pct(metrics['primary_recall@5'])} "
            f"SupportR@5="
            f"{pct(metrics['supporting_recall@5'])} "
            f"Anchor="
            f"{pct(metrics['anchor_coverage@5'])} "
            f"MRR="
            f"{num(metrics['mrr'])} "
            f"NDCG="
            f"{num(metrics['ndcg@5'])}"
        )

        add(
            "Gold anchors:"
        )

        for gold in query_result[
            "gold"
        ]:

            add(
                f"  - "
                f"{gold['anchor']} "
                f"(relevance="
                f"{gold['relevance']})"
            )

        add(
            "Retrieved:"
        )

        for item in query_result[
            "results"
        ]:

            marker = (
                "GOLD"
                if item["is_gold"]
                else "----"
            )

            add(
                f"  #{item['rank']} "
                f"[{marker}] "
                f"{item['path']} "
                f"| chunk="
                f"{item['chunk_index']}"
            )

            add(
                f"      section="
                f"{item['section']!r}"
            )

            add(
                f"      semantic_rank="
                f"{item['semantic_rank']} "
                f"bm25_rank="
                f"{item['bm25_rank']}"
            )

            add(
                f"      similarity="
                f"{num(item['similarity'])} "
                f"bm25="
                f"{num(item['bm25_score'])} "
                f"hybrid="
                f"{num(item['hybrid_score'])}"
            )

            add(
                f"      mmr="
                f"{num(item['mmr_score'])} "
                f"relevance="
                f"{num(item['mmr_relevance'])} "
                f"redundancy="
                f"{num(item['mmr_redundancy'])}"
            )

            add(
                f"      metadata_bonus="
                f"{num(item['metadata_bonus'])} "
                f"relationship="
                f"{item['metadata_relationship']}"
            )

            if item["is_gold"]:

                add(
                    f"      MATCHED ANCHOR="
                    f"{item['gold_anchor']!r} "
                    f"relevance="
                    f"{item['gold_relevance']}"
                )

            add(
                f"      preview="
                f"{item['preview']}"
            )

    # -------------------------------------------------------------------------
    # PROBLEM QUERIES
    # -------------------------------------------------------------------------

    add("")
    add("=" * 110)
    add(
        "QUERIES NEEDING ATTENTION"
    )
    add("=" * 110)

    problem_count = 0

    for query_result in evaluation[
        "query_results"
    ]:

        metrics = query_result[
            "metrics"
        ]

        if (
            metrics["primary_hit@5"]
            < 1.0
            or
            metrics["anchor_coverage@5"]
            < 1.0
        ):

            problem_count += 1

            add("")
            add(
                f"Q{query_result['query_index']:02d}: "
                f"{query_result['query']}"
            )

            add(
                f"  Primary Hit@5     : "
                f"{pct(metrics['primary_hit@5'])}"
            )

            add(
                f"  Primary Recall@5  : "
                f"{pct(metrics['primary_recall@5'])}"
            )

            add(
                f"  Supporting Recall : "
                f"{pct(metrics['supporting_recall@5'])}"
            )

            add(
                f"  Anchor Coverage   : "
                f"{pct(metrics['anchor_coverage@5'])}"
            )

            add(
                f"  MRR               : "
                f"{num(metrics['mrr'])}"
            )

            add(
                f"  NDCG@5            : "
                f"{num(metrics['ndcg@5'])}"
            )

    if problem_count == 0:

        add(
            "All queries achieved "
            "full primary and anchor coverage."
        )

    add("")
    add("=" * 110)
    add(
        "END OF REPORT"
    )
    add("=" * 110)

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 110)
    print(
        "PRODUCTION RETRIEVER — "
        "FINAL ANCHOR-AWARE EVALUATION"
    )
    print("=" * 110)

    # =========================================================================
    # 1. GOLD
    # =========================================================================

    print(
        "\n[1/5] Loading gold dataset..."
    )

    gold_queries = (
        load_gold_dataset()
    )

    print(
        f"Loaded "
        f"{len(gold_queries)} "
        f"gold queries."
    )

    if len(gold_queries) != 24:

        print(
            "\nWARNING:"
        )

        print(
            f"Expected 24 queries, "
            f"but found "
            f"{len(gold_queries)}."
        )

    # =========================================================================
    # 2. REPOSITORY
    # =========================================================================

    print(
        "\n[2/5] Loading repository..."
    )

    print(
        "Discovering repository..."
    )

    files = (
        GitHubRepositoryIndexer.discover(
            REPOSITORY_URL
        )
    )

    print(
        f"Files discovered: "
        f"{len(files)}"
    )

    print(
        "Acquiring repository content..."
    )

    documents = (
        GitHubContentAcquirer.acquire(
            files
        )
    )

    print(
        f"Documents acquired: "
        f"{len(documents)}"
    )

    print(
        "Creating chunks..."
    )

    chunker = DocumentChunker()

    chunks = list(
        chunker.chunk_documents(
            documents
        )
    )

    print(
        f"Chunks created: "
        f"{len(chunks)}"
    )

    if not chunks:

        raise RuntimeError(
            "No chunks were created."
        )

    # =========================================================================
    # 3. GOLD VALIDATION
    # =========================================================================

    print(
        "\n[3/5] Validating gold anchors..."
    )

    validation = (
        validate_gold_dataset(
            gold_queries,
            chunks,
        )
    )

    print(
        f"Gold items: "
        f"{validation['total_gold_items']}"
    )

    print(
        f"Matched: "
        f"{validation['matched_gold_items']}"
    )

    print(
        f"Missing: "
        f"{validation['missing_gold_items']}"
    )

    print(
        f"Match rate: "
        f"{pct(validation['match_rate'])}"
    )

    if validation[
        "missing_gold_items"
    ]:

        print(
            "\nWARNING: Missing gold anchors:"
        )

        for item in validation[
            "missing"
        ]:

            print(
                f"  {item['path']} "
                f"-> "
                f"{item['anchor']}"
            )

    # =========================================================================
    # 4. RETRIEVER
    # =========================================================================

    print(
        "\n[4/5] Constructing "
        "HybridRetriever..."
    )

    retriever = HybridRetriever(
        semantic_weight=SEMANTIC_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        rrf_k=RRF_K,
        candidate_multiplier=CANDIDATE_MULTIPLIER,
        mmr_lambda=MMR_LAMBDA,
        metadata_bonus_weight=METADATA_BONUS_WEIGHT,
    )

    print(
        "Configuration:"
    )

    print(
        f"  semantic_weight      = "
        f"{SEMANTIC_WEIGHT}"
    )

    print(
        f"  bm25_weight          = "
        f"{BM25_WEIGHT}"
    )

    print(
        f"  rrf_k                = "
        f"{RRF_K}"
    )

    print(
        f"  candidate_multiplier = "
        f"{CANDIDATE_MULTIPLIER}"
    )

    print(
        f"  mmr_lambda           = "
        f"{MMR_LAMBDA}"
    )

    print(
        f"  metadata_bonus       = "
        f"{METADATA_BONUS_WEIGHT}"
    )

    # =========================================================================
    # 5. EVALUATE
    # =========================================================================

    print(
        "\n[5/5] Running "
        "24-query evaluation..."
    )

    query_results = []

    for index, gold_query in enumerate(
        gold_queries,
        start=1,
    ):

        print(
            f"\rEvaluating "
            f"{index}/{len(gold_queries)}",
            end="",
            flush=True,
        )

        try:

            result = evaluate_query(
                gold_query,
                chunks,
                retriever,
            )

            query_results.append(
                result
            )

        except Exception:

            print(
                "\n\nERROR:"
            )

            print(
                gold_query["_query"]
            )

            traceback.print_exc()

            raise

    print()

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    overall = aggregate(
        query_results
    )

    by_category = (
        aggregate_by_category(
            query_results
        )
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    evaluation = {
        "timestamp":
            datetime.now().isoformat(),

        "repository":
            REPOSITORY_URL,

        "gold_dataset":
            str(GOLD_DATASET),

        "evaluation_type":
            "production_retriever_final_anchor_aware",

        "alignment_method":
            (
                "path + anchor; "
                "most-specific anchor wins; "
                "explicit chunk_index only "
                "when supplied by gold"
            ),

        "ndcg_method":
            (
                "anchor-level; "
                "duplicate chunks from the "
                "same anchor receive zero "
                "additional gain"
            ),

        "configuration": {
            "top_k":
                TOP_K,

            "semantic_weight":
                SEMANTIC_WEIGHT,

            "bm25_weight":
                BM25_WEIGHT,

            "rrf_k":
                RRF_K,

            "candidate_multiplier":
                CANDIDATE_MULTIPLIER,

            "mmr_lambda":
                MMR_LAMBDA,

            "metadata_bonus_weight":
                METADATA_BONUS_WEIGHT,

            "chunk_count":
                len(chunks),

            "gold_query_count":
                len(gold_queries),
        },

        "gold_validation":
            validation,

        "aggregate":
            overall,

        "by_category":
            by_category,

        "query_results":
            query_results,
    }

    # =========================================================================
    # SAVE
    # =========================================================================

    json_path = (
        OUTPUT_DIR
        /
        f"production_retriever_final_"
        f"{timestamp}.json"
    )

    txt_path = (
        OUTPUT_DIR
        /
        f"production_retriever_final_"
        f"{timestamp}.txt"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=2,
            ensure_ascii=False,
        )

    report = generate_report(
        evaluation
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report
        )

    # =========================================================================
    # TERMINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 110)

    print()
    print(
        f"Gold match rate       : "
        f"{pct(validation['match_rate'])}"
    )

    print(
        f"Precision@5           : "
        f"{pct(overall['precision@5'])}"
    )

    print(
        f"Recall@5              : "
        f"{pct(overall['recall@5'])}"
    )

    print(
        f"Primary Recall@5      : "
        f"{pct(overall['primary_recall@5'])}"
    )

    print(
        f"Supporting Recall@5   : "
        f"{pct(overall['supporting_recall@5'])}"
    )

    print(
        f"Anchor Coverage@5     : "
        f"{pct(overall['anchor_coverage@5'])}"
    )

    print(
        f"Primary Hit@5         : "
        f"{pct(overall['primary_hit@5'])}"
    )

    print(
        f"Hit Rate@5            : "
        f"{pct(overall['hit_rate@5'])}"
    )

    print(
        f"NDCG@5                : "
        f"{num(overall['ndcg@5'])}"
    )

    print(
        f"MRR                   : "
        f"{num(overall['mrr'])}"
    )

    print(
        f"Primary MRR           : "
        f"{num(overall['primary_mrr'])}"
    )

    print(
        f"Unique Ratio          : "
        f"{pct(overall['unique_ratio'])}"
    )

    print(
        f"Same Document Ratio   : "
        f"{pct(overall['same_document_ratio'])}"
    )

    print(
        f"Semantic Redundancy   : "
        f"{num(overall['semantic_redundancy'])}"
    )

    # =========================================================================
    # SANITY CHECKS
    # =========================================================================

    print()
    print(
        "SANITY CHECKS"
    )

    ndcg = overall[
        "ndcg@5"
    ]

    if ndcg is not None:

        if (
            ndcg < 0.0
            or ndcg > 1.0
        ):

            print(
                "WARNING: NDCG is outside "
                "[0, 1]."
            )

        else:

            print(
                "NDCG range check       : PASS"
            )

    else:

        print(
            "NDCG range check       : N/A"
        )

    if validation[
        "missing_gold_items"
    ] == 0:

        print(
            "Gold alignment check   : PASS"
        )

    else:

        print(
            "Gold alignment check   : "
            "WARNING"
        )

    print()
    print(
        "JSON:"
    )

    print(
        json_path
    )

    print()
    print(
        "TEXT:"
    )

    print(
        txt_path
    )

    print()
    print(
        "Run complete."
    )


if __name__ == "__main__":
    main()
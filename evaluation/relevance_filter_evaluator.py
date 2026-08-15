from __future__ import annotations

import sys
import json
import math
import hashlib
import re
import traceback
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple, Optional, Sequence

import numpy as np


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

try:
    from src.services.github.repository_indexer import (
        GitHubRepositoryIndexer,
    )

    from src.services.github.content_acquirer import (
        GitHubContentAcquirer,
    )

    from src.services.rag.chunker import (
        DocumentChunker,
    )

    from src.services.rag.hybrid_retriever import (
        HybridRetriever,
    )

    from src.services.rag.retriever import (
        SimpleRetriever,
    )

except Exception:
    print("\nERROR: Could not import project RAG components.")
    print("\nProject root:")
    print(PROJECT_ROOT)
    print("\nOriginal error:")
    traceback.print_exc()
    raise SystemExit(1)


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
# RETRIEVAL CONFIGURATION
# =============================================================================

TOP_K_VALUES = [1, 3, 5, 10]

SEMANTIC_WEIGHT = 0.5
BM25_WEIGHT = 0.5

RRF_K = 60

CANDIDATE_MULTIPLIER = 8

MMR_LAMBDAS = [
    1.0,
    0.9,
    0.7,
    0.5,
]


# =============================================================================
# NEW EXPERIMENT PARAMETERS
# =============================================================================

# Retrieve substantially more candidates before filtering.
CANDIDATE_POOL_SIZE = 40


# Semantic relevance threshold.
#
# IMPORTANT:
# We are not making this extremely aggressive initially.
#
# The purpose of this experiment is to determine whether filtering weak
# candidates before MMR improves precision without destroying recall.
#
SEMANTIC_THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
]


# Metadata score is deliberately small.
#
# Metadata should guide retrieval.
# It should NOT overpower semantic relevance.
METADATA_WEIGHT = 0.10


# Exact lexical match contribution.
LEXICAL_WEIGHT = 0.15


# MMR is only applied after relevance filtering.
MMR_ENABLED = True


# =============================================================================
# TEXT HELPERS
# =============================================================================

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "training",
    "using",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


def tokenize(value: Any) -> List[str]:
    text = normalize_text(value)

    return re.findall(
        r"[a-zA-Z_][a-zA-Z0-9_]*",
        text,
    )


def query_terms(query: str) -> Set[str]:
    return {
        token
        for token in tokenize(query)
        if token not in STOPWORDS
        and len(token) > 1
    }


def canonical_path(value: Any) -> str:
    if value is None:
        return ""

    path = str(value).strip()

    path = path.replace(
        "\\",
        "/",
    )

    while "//" in path:
        path = path.replace(
            "//",
            "/",
        )

    return path.lstrip("./")


# =============================================================================
# CHUNK IDENTIFICATION
# =============================================================================

def chunk_id(
    chunk: Dict[str, Any],
) -> str:

    explicit_id = (
        chunk.get("id")
        or chunk.get("chunk_id")
        or chunk.get("document_id")
    )

    if explicit_id:
        return str(explicit_id)

    path = canonical_path(
        chunk.get("path", "")
    )

    index = chunk.get(
        "chunk_index",
        None,
    )

    if path or index is not None:
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

    return f"content|{digest}"


# =============================================================================
# GOLD DATASET
# =============================================================================

def load_gold_dataset() -> List[Dict[str, Any]]:

    if not GOLD_DATASET.exists():
        raise FileNotFoundError(
            f"\nGold dataset not found:\n"
            f"{GOLD_DATASET}"
        )

    with open(
        GOLD_DATASET,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        queries = data

    elif isinstance(data, dict):

        queries = None

        for key in (
            "queries",
            "gold_queries",
            "examples",
            "dataset",
            "items",
        ):
            value = data.get(key)

            if isinstance(value, list):
                queries = value
                break

        if queries is None:
            raise ValueError(
                "Could not find query list in gold dataset."
            )

    else:
        raise ValueError(
            "Unsupported gold dataset format."
        )

    normalized = []

    for index, item in enumerate(
        queries
    ):

        if not isinstance(item, dict):
            continue

        gold = dict(item)

        gold["_query_index"] = index

        query = (
            gold.get("query")
            or gold.get("question")
            or gold.get("text")
            or gold.get("q")
        )

        if not query:
            raise ValueError(
                f"Gold query {index} has no query/question field."
            )

        gold["_query"] = str(query)

        normalized.append(
            gold
        )

    return normalized


# =============================================================================
# GOLD DATASET + ANCHOR-BASED ALIGNMENT
# =============================================================================
#
# IMPORTANT:
# The gold dataset intentionally does NOT identify fixed chunk indices.
# It stores:
#     path + human-readable anchor/section + graded relevance (1/2/3)
#
# The current DocumentChunker may change chunk boundaries, so the evaluator
# MUST resolve those anchors against the CURRENT chunks before scoring.
# This prevents the old broken comparison:
#
#     gold:      doc/modules/linear_model.rst|None
#     retrieved: doc/modules/linear_model.rst|2
#
# from producing false 0% metrics.
# =============================================================================

ANCHOR_MATCH_MIN_SCORE = 50


def safe_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def normalize_anchor(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("`", "")
    text = re.sub(
        r"[^\w\s²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉.-]",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def load_gold_dataset() -> List[Dict[str, Any]]:
    if not GOLD_DATASET.exists():
        raise FileNotFoundError(
            f"\nGold dataset not found:\n{GOLD_DATASET}"
        )

    with open(GOLD_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        queries = data
    elif isinstance(data, dict):
        queries = data.get("queries")
        if not isinstance(queries, list):
            raise ValueError(
                "Gold dataset does not contain a 'queries' list."
            )
    else:
        raise ValueError("Gold dataset must be a JSON object or list.")

    normalized = []

    for index, item in enumerate(queries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Gold query #{index} is not an object.")

        query = (
            item.get("query")
            or item.get("question")
            or item.get("text")
            or item.get("q")
        )

        if not query:
            raise ValueError(
                f"Gold query #{index} has no query field."
            )

        gold_items = item.get("gold")

        # This is the actual schema used by
        # scikit_learn_retrieval_gold_v1.json.
        if not isinstance(gold_items, list) or not gold_items:
            raise ValueError(
                f"Gold query {item.get('id', index)} must contain "
                "a non-empty 'gold' list."
            )

        normalized.append(
            {
                **item,
                "_query_index": index,
                "_query": str(query),
            }
        )

    return normalized


def normalize_gold_item(item: Dict[str, Any]) -> Dict[str, Any]:
    path = canonical_path(
        item.get(
            "path",
            item.get("file", item.get("source", "")),
        )
    )

    anchor = str(
        item.get(
            "anchor",
            item.get("section", ""),
        )
    )

    relevance = safe_int(item.get("relevance"))

    if relevance not in (1, 2, 3):
        raise ValueError(
            f"Invalid gold relevance {relevance} "
            f"for anchor {anchor!r}."
        )

    return {
        "path": path,
        "anchor": anchor,
        "anchor_norm": normalize_anchor(anchor),
        "relevance": relevance,
    }


def _section_values(chunk: Dict[str, Any]) -> List[str]:
    values: List[str] = []

    for key in (
        "section",
        "parent_section",
        "section_path",
        "title",
        "heading",
    ):
        value = chunk.get(key)

        if value:
            if isinstance(value, (list, tuple)):
                values.extend(str(v) for v in value if v)
            else:
                values.append(str(value))

    return values


def _section_segments(chunk: Dict[str, Any]) -> List[str]:
    values: List[str] = []

    for value in _section_values(chunk):
        for part in re.split(
            r"\s*(?:>|/|::|\\)\s*",
            str(value),
        ):
            if part.strip():
                values.append(part.strip())

    return values


def anchor_match_score(
    chunk: Dict[str, Any],
    gold: Dict[str, Any],
) -> int:
    """
    Resolve a gold path+anchor to a CURRENT chunk.

    Matching is metadata-first. Semantic similarity is deliberately NOT used
    to define gold relevance because doing so would leak retrieval behavior
    into the ground truth.
    """

    if (
        canonical_path(chunk.get("path", ""))
        != gold["path"]
    ):
        return 0

    anchor = gold["anchor_norm"]

    if not anchor:
        return 0

    section_values = [
        normalize_anchor(v)
        for v in _section_values(chunk)
        if v
    ]

    segments = [
        normalize_anchor(v)
        for v in _section_segments(chunk)
        if v
    ]

    content_norm = normalize_anchor(
        chunk.get("content", "")
    )

    # Exact section metadata.
    if anchor in section_values:
        return 100

    # Exact section-path segment.
    if anchor in segments:
        return 95

    # Section path ending with anchor.
    for value in section_values:
        if value.endswith(anchor):
            return 90

    # Anchor contained in metadata.
    if any(anchor in value for value in section_values):
        # Do not let a broad anchor such as
        # "ordinary least squares" accidentally absorb
        # "ordinary least squares complexity" as an exact match.
        for value in section_values:
            if value != anchor and value.startswith(anchor):
                remainder = value[len(anchor):].strip(
                    " -:>|"
                )

                if remainder and len(remainder.split()) <= 6:
                    return 70

        return 80

    # Last resort: anchor appears in content.
    if anchor in content_norm:
        return 55

    return 0


def resolve_gold_anchors(
    gold_query: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Resolve every gold path+anchor to current chunk IDs.

    If an anchored section spans multiple current chunks, all chunks belonging
    to the strongest metadata match are assigned the anchor's relevance.
    """

    resolved: Dict[str, Dict[str, Any]] = {}
    missing: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []

    for raw_item in gold_query["gold"]:
        item = normalize_gold_item(raw_item)

        candidates: List[Tuple[int, Dict[str, Any]]] = []

        for chunk in chunks:
            score = anchor_match_score(chunk, item)

            if score >= ANCHOR_MATCH_MIN_SCORE:
                candidates.append((score, chunk))

        if not candidates:
            missing.append(
                {
                    "path": item["path"],
                    "anchor": item["anchor"],
                    "relevance": item["relevance"],
                }
            )
            continue

        best_score = max(
            score for score, _ in candidates
        )

        # Only keep candidates with the strongest interpretation.
        selected = [
            (score, chunk)
            for score, chunk in candidates
            if score >= best_score
        ]

        # Content-only matches are not strong enough to label many chunks.
        if best_score < 90:
            selected = selected[:1]

        if len(selected) > 1:
            ambiguous.append(
                {
                    "path": item["path"],
                    "anchor": item["anchor"],
                    "relevance": item["relevance"],
                    "candidate_count": len(selected),
                }
            )

        for score, chunk in selected:
            cid = chunk_id(chunk)

            existing = resolved.get(cid)

            # If multiple anchors map to the same chunk, keep the stronger
            # relevance label.
            if (
                existing is None
                or item["relevance"] > existing["relevance"]
            ):
                resolved[cid] = {
                    "chunk_id": cid,
                    "path": item["path"],
                    "anchor": item["anchor"],
                    "relevance": item["relevance"],
                    "match_score": score,
                    "chunk_index": chunk.get("chunk_index"),
                }

    return {
        "resolved": resolved,
        "missing": missing,
        "ambiguous": ambiguous,
        "gold_anchor_count": len(gold_query["gold"]),
        "resolved_chunk_count": len(resolved),
    }


def validate_gold_against_chunks(
    gold_queries: Sequence[Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    total_anchors = 0
    resolved_anchors = 0
    missing: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    resolved_chunk_counts: List[int] = []

    for gold in gold_queries:
        resolution = resolve_gold_anchors(
            gold,
            chunks,
        )

        total_anchors += resolution[
            "gold_anchor_count"
        ]

        resolved_anchors += (
            resolution["gold_anchor_count"]
            - len(resolution["missing"])
        )

        missing.extend(
            [
                {
                    **item,
                    "query_id": gold.get("id"),
                }
                for item in resolution["missing"]
            ]
        )

        ambiguous.extend(
            [
                {
                    **item,
                    "query_id": gold.get("id"),
                }
                for item in resolution["ambiguous"]
            ]
        )

        resolved_chunk_counts.append(
            resolution["resolved_chunk_count"]
        )

    return {
        "gold_anchor_count": total_anchors,
        "resolved_anchor_count": resolved_anchors,
        "missing_anchor_count": len(missing),
        "gold_anchor_match_rate": (
            resolved_anchors / total_anchors
            if total_anchors
            else 0.0
        ),
        "missing_anchors": missing,
        "ambiguous_anchors": ambiguous,
        "resolved_chunks_total": int(
            sum(resolved_chunk_counts)
        ),
        "chunk_count": len(chunks),
    }


def relevance_for_result(
    result: Dict[str, Any],
    gold_map: Dict[str, Any],
) -> int:
    info = gold_map.get(chunk_id(result))
    return (
        int(info["relevance"])
        if info
        else 0
    )


def anchor_for_result(
    result: Dict[str, Any],
    gold_map: Dict[str, Any],
) -> Optional[str]:
    info = gold_map.get(chunk_id(result))
    return (
        info["anchor"]
        if info
        else None
    )


# =============================================================================
# END GOLD ALIGNMENT
# =============================================================================

# =============================================================================
# REPOSITORY LOADING
# =============================================================================

def load_repository_chunks():

    print("\n")
    print("=" * 80)
    print("LOADING REPOSITORY")
    print("=" * 80)

    print(
        f"\nRepository:\n{REPOSITORY_URL}"
    )

    print(
        "\n[1/3] Discovering repository..."
    )

    files = (
        GitHubRepositoryIndexer.discover(
            REPOSITORY_URL
        )
    )

    print(
        f"Files discovered: {len(files)}"
    )

    print(
        "\n[2/3] Acquiring repository content..."
    )

    documents = (
        GitHubContentAcquirer.acquire(
            files
        )
    )

    print(
        f"Documents acquired: {len(documents)}"
    )

    print(
        "\n[3/3] Creating semantic chunks..."
    )

    chunker = DocumentChunker()

    chunks = (
        chunker.chunk_documents(
            documents
        )
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    return chunks


# =============================================================================
# RETRIEVER
# =============================================================================

def create_retriever(
    mmr_lambda: float,
) -> HybridRetriever:

    return HybridRetriever(
        semantic_weight=SEMANTIC_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        rrf_k=RRF_K,
        candidate_multiplier=CANDIDATE_MULTIPLIER,
        mmr_lambda=mmr_lambda,
    )


# =============================================================================
# EMBEDDING MODEL
# =============================================================================

def get_embedding_model():

    return SimpleRetriever._get_model()


def embed_texts(
    texts: List[str],
) -> np.ndarray:

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


# =============================================================================
# COSINE SIMILARITY
# =============================================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


# =============================================================================
# QUERY TYPE
# =============================================================================

def classify_query(
    query: str,
) -> str:

    q = normalize_text(
        query
    )

    conceptual_markers = (
        "how does",
        "how do",
        "what is",
        "what are",
        "why",
        "explain",
        "difference between",
        "compare",
        "overview",
        "concept",
    )

    technical_markers = (
        "implementation",
        "implemented",
        "api",
        "parameter",
        "parameters",
        "class",
        "function",
        "method",
        "syntax",
        "code",
        "example",
        "argument",
    )

    if any(
        marker in q
        for marker in technical_markers
    ):
        return "technical"

    if any(
        marker in q
        for marker in conceptual_markers
    ):
        return "conceptual"

    return "general"


# =============================================================================
# METADATA
# =============================================================================

def get_source(
    chunk: Dict[str, Any],
) -> str:

    return canonical_path(
        chunk.get("path")
        or chunk.get("source")
        or chunk.get("file")
        or ""
    )


def get_section(
    chunk: Dict[str, Any],
) -> str:

    return normalize_text(
        chunk.get("section")
        or chunk.get("heading")
        or chunk.get("title")
        or ""
    )


def get_category(
    chunk: Dict[str, Any],
) -> str:

    return normalize_text(
        chunk.get("category")
        or chunk.get("chunk_type")
        or ""
    )


# =============================================================================
# LEXICAL RELEVANCE
# =============================================================================

def lexical_score(
    query: str,
    chunk: Dict[str, Any],
) -> float:

    q_terms = query_terms(
        query
    )

    if not q_terms:
        return 0.0

    content_terms = set(
        tokenize(
            chunk.get(
                "content",
                "",
            )
        )
    )

    if not content_terms:
        return 0.0

    overlap = (
        len(
            q_terms
            & content_terms
        )
        / len(q_terms)
    )

    return float(
        min(
            overlap,
            1.0,
        )
    )


# =============================================================================
# METADATA RELEVANCE
# =============================================================================

def metadata_score(
    query: str,
    query_type: str,
    chunk: Dict[str, Any],
) -> float:

    score = 0.0

    q = normalize_text(
        query
    )

    terms = query_terms(
        query
    )

    section = get_section(
        chunk
    )

    source = get_source(
        chunk
    )

    content = normalize_text(
        chunk.get(
            "content",
            "",
        )
    )

    # -------------------------------------------------------------------------
    # Exact query term in section
    # -------------------------------------------------------------------------

    if section:

        section_terms = set(
            tokenize(
                section
            )
        )

        if terms:

            section_overlap = (
                len(
                    terms
                    & section_terms
                )
                / len(terms)
            )

            score += (
                0.50
                * min(
                    section_overlap,
                    1.0,
                )
            )

    # -------------------------------------------------------------------------
    # Query term in source path
    # -------------------------------------------------------------------------

    if source:

        source_terms = set(
            tokenize(
                source
            )
        )

        if terms:

            source_overlap = (
                len(
                    terms
                    & source_terms
                )
                / len(terms)
            )

            score += (
                0.20
                * min(
                    source_overlap,
                    1.0,
                )
            )

    # -------------------------------------------------------------------------
    # Technical query bonus
    # -------------------------------------------------------------------------

    if query_type == "technical":

        technical_terms = (
            "implementation",
            "implemented",
            "parameter",
            "parameters",
            "class",
            "function",
            "method",
            "api",
            "code",
            "example",
            "syntax",
        )

        if any(
            term in q
            for term in technical_terms
        ):

            if any(
                term in content
                for term in technical_terms
            ):
                score += 0.20

    # -------------------------------------------------------------------------
    # Conceptual query bonus
    # -------------------------------------------------------------------------

    if query_type == "conceptual":

        conceptual_terms = (
            "overview",
            "definition",
            "explains",
            "classification",
            "regression",
            "algorithm",
            "concept",
            "example",
        )

        if any(
            term in content
            for term in conceptual_terms
        ):
            score += 0.10

    return float(
        min(
            score,
            1.0,
        )
    )


# =============================================================================
# COMBINED RELEVANCE
# =============================================================================

def combined_relevance(
    query: str,
    query_type: str,
    chunk: Dict[str, Any],
    semantic_score: float,
) -> Dict[str, float]:

    lexical = lexical_score(
        query,
        chunk,
    )

    metadata = metadata_score(
        query,
        query_type,
        chunk,
    )

    final = (
        (1.0 - LEXICAL_WEIGHT - METADATA_WEIGHT)
        * semantic_score
        + LEXICAL_WEIGHT
        * lexical
        + METADATA_WEIGHT
        * metadata
    )

    return {
        "semantic": float(
            semantic_score
        ),
        "lexical": float(
            lexical
        ),
        "metadata": float(
            metadata
        ),
        "combined": float(
            final
        ),
    }


# =============================================================================
# REDUNDANCY
# =============================================================================

def metadata_relationship(
    candidate: Dict[str, Any],
    selected: Dict[str, Any],
) -> str:

    candidate_source = get_source(
        candidate
    )

    selected_source = get_source(
        selected
    )

    candidate_section = get_section(
        candidate
    )

    selected_section = get_section(
        selected
    )

    same_source = (
        bool(candidate_source)
        and candidate_source
        == selected_source
    )

    same_section = (
        bool(candidate_section)
        and candidate_section
        == selected_section
    )

    if (
        same_source
        and same_section
    ):
        return "same_source_same_section"

    if same_source:
        return "same_source_different_section"

    if same_section:
        return "different_source_same_section"

    return "independent"


def adjusted_redundancy(
    similarity: float,
    relationship: str,
) -> float:

    if relationship == (
        "same_source_same_section"
    ):
        factor = 1.0

    elif relationship == (
        "same_source_different_section"
    ):
        # Same document but different section
        # can be complementary.
        factor = 0.55

    elif relationship == (
        "different_source_same_section"
    ):
        factor = 0.75

    else:
        factor = 0.85

    return float(
        similarity * factor
    )


# =============================================================================
# MMR
# =============================================================================

def mmr_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    mmr_lambda: float,
) -> List[Dict[str, Any]]:

    if not candidates:
        return []

    top_k = min(
        top_k,
        len(candidates),
    )

    model = get_embedding_model()

    query_embedding = np.asarray(
        model.encode(
            query,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )

    texts = [
        str(
            candidate.get(
                "content",
                "",
            )
        )
        for candidate in candidates
    ]

    embeddings = np.asarray(
        model.encode(
            texts,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )

    # Precompute relevance.
    relevance = []

    for index, candidate in enumerate(
        candidates
    ):

        semantic = cosine_similarity(
            query_embedding,
            embeddings[index],
        )

        candidate["_mmr_semantic"] = (
            semantic
        )

        relevance.append(
            semantic
        )

    selected_indices = []

    remaining = set(
        range(
            len(candidates)
        )
    )

    while (
        remaining
        and len(selected_indices)
        < top_k
    ):

        best_index = None
        best_score = -float(
            "inf"
        )

        for index in remaining:

            relevance_score = (
                relevance[index]
            )

            if not selected_indices:

                redundancy = 0.0

            else:

                redundancy_values = []

                for selected_index in selected_indices:

                    similarity = cosine_similarity(
                        embeddings[index],
                        embeddings[
                            selected_index
                        ],
                    )

                    relationship = (
                        metadata_relationship(
                            candidates[index],
                            candidates[
                                selected_index
                            ],
                        )
                    )

                    adjusted = (
                        adjusted_redundancy(
                            similarity,
                            relationship,
                        )
                    )

                    redundancy_values.append(
                        adjusted
                    )

                redundancy = max(
                    redundancy_values
                )

            score = (
                mmr_lambda
                * relevance_score
                -
                (
                    1.0
                    - mmr_lambda
                )
                * redundancy
            )

            if score > best_score:

                best_score = score
                best_index = index

        if best_index is None:
            break

        selected_indices.append(
            best_index
        )

        remaining.remove(
            best_index
        )

    output = []

    for rank, index in enumerate(
        selected_indices,
        start=1,
    ):

        result = dict(
            candidates[index]
        )

        result["mmr_rank"] = rank

        # Recompute the ACTUAL MMR score for this selected result.
        # The old implementation incorrectly stored only semantic relevance,
        # which made mmr_score == semantic_relevance for every result.
        relevance_score = float(
            candidates[index].get(
                "_mmr_semantic",
                0.0,
            )
        )

        if not selected_indices:
            redundancy_score = 0.0
        else:
            redundancy_values = []

            for selected_index in selected_indices:
                similarity = cosine_similarity(
                    embeddings[index],
                    embeddings[selected_index],
                )

                relationship = metadata_relationship(
                    candidates[index],
                    candidates[selected_index],
                )

                redundancy_values.append(
                    adjusted_redundancy(
                        similarity,
                        relationship,
                    )
                )

            redundancy_score = (
                max(redundancy_values)
                if redundancy_values
                else 0.0
            )

        result["mmr_relevance"] = relevance_score
        result["mmr_redundancy"] = float(
            redundancy_score
        )
        result["mmr_score"] = float(
            mmr_lambda * relevance_score
            - (1.0 - mmr_lambda)
            * redundancy_score
        )

        output.append(
            result
        )

    return output


# =============================================================================
# CANDIDATE RETRIEVAL
# =============================================================================

def retrieve_candidate_pool(
    query: str,
    chunks: List[Dict[str, Any]],
    mmr_lambda: float,
) -> List[Dict[str, Any]]:

    # We deliberately request a large pool.
    retriever = create_retriever(
        mmr_lambda
    )

    results = retriever.retrieve(
        question=query,
        chunks=chunks,
        top_k=CANDIDATE_POOL_SIZE,
    )

    return list(
        results or []
    )


# =============================================================================
# SEMANTIC SCORING
# =============================================================================

def score_candidates_semantically(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not candidates:
        return []

    model = get_embedding_model()

    query_embedding = np.asarray(
        model.encode(
            query,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )

    texts = [
        str(
            candidate.get(
                "content",
                "",
            )
        )
        for candidate in candidates
    ]

    embeddings = np.asarray(
        model.encode(
            texts,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )

    scored = []

    for candidate, embedding in zip(
        candidates,
        embeddings,
    ):

        result = dict(
            candidate
        )

        semantic = cosine_similarity(
            query_embedding,
            embedding,
        )

        result[
            "_semantic_relevance"
        ] = semantic

        scored.append(
            result
        )

    return scored


# =============================================================================
# RELEVANCE FILTER
# =============================================================================

def relevance_filter(
    query: str,
    candidates: List[Dict[str, Any]],
    threshold: float,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    query_type = classify_query(
        query
    )

    scored = score_candidates_semantically(
        query,
        candidates,
    )

    accepted = []
    rejected = []

    for candidate in scored:

        semantic = float(
            candidate.get(
                "_semantic_relevance",
                0.0,
            )
        )

        scores = combined_relevance(
            query=query,
            query_type=query_type,
            chunk=candidate,
            semantic_score=semantic,
        )

        candidate[
            "_lexical_relevance"
        ] = scores["lexical"]

        candidate[
            "_metadata_relevance"
        ] = scores["metadata"]

        candidate[
            "_combined_relevance"
        ] = scores["combined"]

        # IMPORTANT:
        #
        # Threshold is based primarily on semantic relevance.
        #
        # We do NOT allow metadata to rescue an obviously irrelevant
        # semantic candidate.
        #
        if semantic >= threshold:
            accepted.append(
                candidate
            )
        else:
            rejected.append(
                candidate
            )

    accepted.sort(
        key=lambda item:
        item.get(
            "_combined_relevance",
            0.0,
        ),
        reverse=True,
    )

    return (
        accepted,
        rejected,
    )


# =============================================================================
# FINAL RERANK
# =============================================================================

def final_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    mmr_lambda: float,
) -> List[Dict[str, Any]]:

    if not candidates:
        return []

    # First rank by combined relevance.
    candidates = sorted(
        candidates,
        key=lambda item:
        item.get(
            "_combined_relevance",
            item.get(
                "_semantic_relevance",
                0.0,
            ),
        ),
        reverse=True,
    )

    # MMR happens ONLY after filtering.
    if MMR_ENABLED:

        results = mmr_rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            mmr_lambda=mmr_lambda,
        )

    else:

        results = candidates[
            :top_k
        ]

    return results


# =============================================================================
# RESULT IDENTIFICATION + GRADED METRICS
# =============================================================================

def result_is_relevant(
    result: Dict[str, Any],
    gold_map: Dict[str, Any],
) -> bool:
    return relevance_for_result(
        result,
        gold_map,
    ) > 0


def precision_at_k(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    top = results[:k]

    if not top:
        return 0.0

    return sum(
        result_is_relevant(result, gold_map)
        for result in top
    ) / len(top)


def weighted_precision_at_k(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    top = results[:k]

    if not top:
        return 0.0

    return sum(
        relevance_for_result(result, gold_map)
        for result in top
    ) / (3.0 * len(top))


def recall_at_k(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    if not gold_map:
        return 0.0

    retrieved = {
        chunk_id(result)
        for result in results[:k]
    }

    gold_ids = set(gold_map)

    return len(
        retrieved & gold_ids
    ) / len(gold_ids)


def graded_recall_at_k(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    if not gold_map:
        return 0.0

    total_relevance = sum(
        info["relevance"]
        for info in gold_map.values()
    )

    if total_relevance == 0:
        return 0.0

    retrieved_ids = {
        chunk_id(result)
        for result in results[:k]
    }

    covered = sum(
        info["relevance"]
        for cid, info in gold_map.items()
        if cid in retrieved_ids
    )

    return covered / total_relevance


def hit_rate_at_k(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    return float(
        any(
            result_is_relevant(
                result,
                gold_map,
            )
            for result in results[:k]
        )
    )


def primary_hit_rate_at_k(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    return float(
        any(
            relevance_for_result(
                result,
                gold_map,
            ) == 3
            for result in results[:k]
        )
    )


def reciprocal_rank(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
) -> float:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result_is_relevant(
            result,
            gold_map,
        ):
            return 1.0 / rank

    return 0.0


def primary_reciprocal_rank(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
) -> float:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if (
            relevance_for_result(
                result,
                gold_map,
            )
            == 3
        ):
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    if not gold_map:
        return 0.0

    dcg = 0.0

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):
        relevance = relevance_for_result(
            result,
            gold_map,
        )

        if relevance:
            dcg += (
                (2**relevance - 1)
                / math.log2(rank + 1)
            )

    ideal_relevances = sorted(
        (
            info["relevance"]
            for info in gold_map.values()
        ),
        reverse=True,
    )[:k]

    idcg = sum(
        (2**relevance - 1)
        / math.log2(rank + 1)
        for rank, relevance in enumerate(
            ideal_relevances,
            start=1,
        )
    )

    return (
        dcg / idcg
        if idcg
        else 0.0
    )


# =============================================================================
# END RESULT METRICS
# =============================================================================

# =============================================================================
# DUPLICATES
# =============================================================================

def duplicate_count(
    results: List[Dict[str, Any]],
) -> int:

    ids = [
        chunk_id(result)
        for result in results
    ]

    return (
        len(ids)
        - len(set(ids))
    )


def unique_ratio(
    results: List[Dict[str, Any]],
) -> float:

    if not results:
        return 1.0

    ids = [
        chunk_id(result)
        for result in results
    ]

    return (
        len(set(ids))
        / len(ids)
    )


def content_duplicate_count(
    results: List[Dict[str, Any]],
) -> int:

    normalized = [
        normalize_text(
            result.get(
                "content",
                "",
            )
        )
        for result in results
    ]

    return (
        len(normalized)
        - len(set(normalized))
    )


# =============================================================================
# SEMANTIC REDUNDANCY
# =============================================================================

def semantic_redundancy(
    results: List[Dict[str, Any]],
) -> float:

    if len(results) < 2:
        return 0.0

    embeddings = embed_texts(
        [
            str(
                result.get(
                    "content",
                    "",
                )
            )
            for result in results
        ]
    )

    similarities = []

    for i in range(
        len(embeddings)
    ):

        for j in range(
            i + 1,
            len(embeddings),
        ):

            similarities.append(
                cosine_similarity(
                    embeddings[i],
                    embeddings[j],
                )
            )

    if not similarities:
        return 0.0

    return float(
        np.mean(
            similarities
        )
    )


# =============================================================================
# GOLD FILTER SAFETY
# =============================================================================

def evaluate_filter_safety(
    rejected: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
) -> Dict[str, Any]:
    rejected_gold = []

    for result in rejected:
        relevance = relevance_for_result(
            result,
            gold_map,
        )

        if relevance > 0:
            rejected_gold.append(
                {
                    "id": chunk_id(result),
                    "relevance": relevance,
                    "anchor": anchor_for_result(
                        result,
                        gold_map,
                    ),
                }
            )

    gold_ids = set(gold_map)

    rejected_ids = {
        item["id"]
        for item in rejected_gold
    }

    return {
        "rejected_count": len(rejected),
        "rejected_gold_count": len(rejected_gold),
        "gold_filter_recall": (
            1.0
            if not gold_ids
            else 1.0
            - (
                len(rejected_ids)
                / len(gold_ids)
            )
        ),
        "rejected_gold_ids": sorted(
            rejected_ids
        ),
        "rejected_gold": rejected_gold,
    }


# =============================================================================
# SINGLE QUERY EXPERIMENT
# =============================================================================

# =============================================================================
# SINGLE QUERY EXPERIMENT
# =============================================================================

def evaluate_query(
    gold: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    threshold: float,
    mmr_lambda: float,
) -> Dict[str, Any]:

    query = gold["_query"]
    query_type = classify_query(query)

    # -------------------------------------------------------------------------
    # CRITICAL: resolve gold anchors against CURRENT chunks.
    # -------------------------------------------------------------------------

    resolution = resolve_gold_anchors(
        gold,
        chunks,
    )

    gold_map = resolution["resolved"]

    # -------------------------------------------------------------------------
    # Candidate retrieval
    # -------------------------------------------------------------------------

    candidates = retrieve_candidate_pool(
        query=query,
        chunks=chunks,
        mmr_lambda=1.0,
    )

    # -------------------------------------------------------------------------
    # Score + filter
    # -------------------------------------------------------------------------

    accepted, rejected = relevance_filter(
        query=query,
        candidates=candidates,
        threshold=threshold,
    )

    # -------------------------------------------------------------------------
    # Final reranking
    # -------------------------------------------------------------------------

    final_results = final_rerank(
        query=query,
        candidates=accepted,
        top_k=max(TOP_K_VALUES),
        mmr_lambda=mmr_lambda,
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = {}

    for k in TOP_K_VALUES:
        metrics[f"precision@{k}"] = precision_at_k(
            final_results,
            gold_map,
            k,
        )

        metrics[f"weighted_precision@{k}"] = (
            weighted_precision_at_k(
                final_results,
                gold_map,
                k,
            )
        )

        metrics[f"recall@{k}"] = recall_at_k(
            final_results,
            gold_map,
            k,
        )

        metrics[f"graded_recall@{k}"] = (
            graded_recall_at_k(
                final_results,
                gold_map,
                k,
            )
        )

        metrics[f"hit_rate@{k}"] = hit_rate_at_k(
            final_results,
            gold_map,
            k,
        )

        metrics[f"primary_hit_rate@{k}"] = (
            primary_hit_rate_at_k(
                final_results,
                gold_map,
                k,
            )
        )

        metrics[f"ndcg@{k}"] = ndcg_at_k(
            final_results,
            gold_map,
            k,
        )

    metrics["mrr"] = reciprocal_rank(
        final_results,
        gold_map,
    )

    metrics["primary_mrr"] = (
        primary_reciprocal_rank(
            final_results,
            gold_map,
        )
    )

    metrics["duplicate_count"] = duplicate_count(
        final_results
    )

    metrics["content_duplicate_count"] = (
        content_duplicate_count(
            final_results
        )
    )

    metrics["unique_ratio"] = unique_ratio(
        final_results
    )

    metrics["semantic_redundancy"] = (
        semantic_redundancy(
            final_results
        )
    )

    safety = evaluate_filter_safety(
        rejected,
        gold_map,
    )

    # -------------------------------------------------------------------------
    # Result diagnostics
    # -------------------------------------------------------------------------

    summarized_results = []

    for rank, result in enumerate(
        final_results,
        start=1,
    ):
        relevance = relevance_for_result(
            result,
            gold_map,
        )

        summarized_results.append(
            {
                "rank": rank,
                "id": chunk_id(result),
                "path": canonical_path(
                    result.get("path", "")
                ),
                "chunk_index": result.get(
                    "chunk_index"
                ),
                "section": result.get(
                    "section",
                    result.get("heading", ""),
                ),
                "semantic_relevance": result.get(
                    "_semantic_relevance"
                ),
                "lexical_relevance": result.get(
                    "_lexical_relevance"
                ),
                "metadata_relevance": result.get(
                    "_metadata_relevance"
                ),
                "combined_relevance": result.get(
                    "_combined_relevance"
                ),
                "mmr_score": result.get(
                    "mmr_score"
                ),
                "gold_relevance": relevance,
                "gold_anchor": anchor_for_result(
                    result,
                    gold_map,
                ),
                "gold_match_score": (
                    gold_map[
                        chunk_id(result)
                    ]["match_score"]
                    if chunk_id(result) in gold_map
                    else None
                ),
                "relevant": relevance > 0,
                "preview": normalize_text(
                    result.get("content", "")
                )[:500],
            }
        )

    return {
        "query_id": gold.get(
            "id",
            gold.get(
                "query_id",
                gold["_query_index"],
            ),
        ),
        "query_index": gold["_query_index"],
        "query": query,
        "query_type": query_type,
        "category": gold.get(
            "category",
            gold.get(
                "type",
                "unknown",
            ),
        ),
        "threshold": threshold,
        "mmr_lambda": mmr_lambda,

        # Keep the actual anchor-based gold information.
        "gold_anchors": [
            normalize_gold_item(item)
            for item in gold["gold"]
        ],
        "gold_resolution": {
            "resolved_chunk_count": (
                resolution[
                    "resolved_chunk_count"
                ]
            ),
            "missing": resolution[
                "missing"
            ],
            "ambiguous": resolution[
                "ambiguous"
            ],
            "resolved": list(
                gold_map.values()
            ),
        },

        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "filter_rate": (
            len(rejected) / len(candidates)
            if candidates
            else 0.0
        ),
        "filter_safety": safety,
        "metrics": metrics,
        "results": summarized_results,
    }


# =============================================================================
# AGGREGATION
# =============================================================================

def average_metric(
    results: List[Dict[str, Any]],
    metric: str,
) -> float:

    values = []

    for result in results:

        value = result[
            "metrics"
        ].get(metric)

        if value is not None:
            values.append(
                float(value)
            )

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def aggregate_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    output = {}

    for k in TOP_K_VALUES:

        for metric_name in (
            "precision",
            "weighted_precision",
            "recall",
            "graded_recall",
            "hit_rate",
            "primary_hit_rate",
        ):
            output[
                f"{metric_name}@{k}"
            ] = average_metric(
                results,
                f"{metric_name}@{k}",
            )

        output[
            f"ndcg@{k}"
        ] = average_metric(
            results,
            f"ndcg@{k}",
        )

    output["mrr"] = average_metric(
        results,
        "mrr",
    )

    output["primary_mrr"] = average_metric(
        results,
        "primary_mrr",
    )

    output[
        "duplicate_count_avg"
    ] = average_metric(
        results,
        "duplicate_count",
    )

    output[
        "content_duplicate_count_avg"
    ] = average_metric(
        results,
        "content_duplicate_count",
    )

    output[
        "unique_ratio_avg"
    ] = average_metric(
        results,
        "unique_ratio",
    )

    output[
        "semantic_redundancy_avg"
    ] = average_metric(
        results,
        "semantic_redundancy",
    )

    output[
        "candidate_count_avg"
    ] = float(
        np.mean(
            [
                r["candidate_count"]
                for r in results
            ]
        )
    ) if results else 0.0

    output[
        "accepted_count_avg"
    ] = float(
        np.mean(
            [
                r["accepted_count"]
                for r in results
            ]
        )
    ) if results else 0.0

    output[
        "filter_rate_avg"
    ] = float(
        np.mean(
            [
                r["filter_rate"]
                for r in results
            ]
        )
    ) if results else 0.0

    output[
        "gold_filter_recall_avg"
    ] = float(
        np.mean(
            [
                r[
                    "filter_safety"
                ][
                    "gold_filter_recall"
                ]
                for r in results
            ]
        )
    ) if results else 0.0

    output[
        "queries_with_filtered_gold"
    ] = sum(
        1
        for r in results
        if r[
            "filter_safety"
        ][
            "rejected_gold_count"
        ] > 0
    )

    return output


def aggregate_by_type(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    groups = {}

    for result in results:

        query_type = result[
            "query_type"
        ]

        groups.setdefault(
            query_type,
            [],
        ).append(
            result
        )

    output = {}

    for query_type, items in groups.items():

        output[
            query_type
        ] = {
            "query_count": len(
                items
            ),
            "metrics": aggregate_results(
                items
            ),
        }

    return output


# =============================================================================
# FORMATTERS
# =============================================================================

def pct(
    value: float,
) -> str:

    return (
        f"{value * 100:.2f}%"
    )


def num(
    value: Any,
    digits: int = 4,
) -> str:

    if value is None:
        return "N/A"

    return (
        f"{float(value):.{digits}f}"
    )


# =============================================================================
# TEXT REPORT
# =============================================================================

def generate_report(
    evaluation: Dict[str, Any],
) -> str:

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "RELEVANCE FILTER + METADATA-AWARE MMR EVALUATION"
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        f"\nGenerated: {evaluation['timestamp']}"
    )

    lines.append(
        f"\nRepository:\n{evaluation['repository']}"
    )

    lines.append(
        f"\nGold dataset:\n{evaluation['gold_dataset']}"
    )

    lines.append(
        "\n"
        + "=" * 100
    )

    lines.append(
        "EXPERIMENT CONFIGURATION"
    )

    lines.append(
        "=" * 100
    )

    config = evaluation[
        "configuration"
    ]

    for key, value in config.items():

        lines.append(
            f"{key}: {value}"
        )

    # -------------------------------------------------------------------------
    # Threshold comparison
    # -------------------------------------------------------------------------

    lines.append(
        "\n"
        + "=" * 100
    )

    lines.append(
        "THRESHOLD COMPARISON"
    )

    lines.append(
        "=" * 100
    )

    for threshold_key, experiment in (
        evaluation[
            "experiments"
        ].items()
    ):

        aggregate = experiment[
            "aggregate"
        ]

        lines.append(
            f"\nThreshold={experiment['threshold']} "
            f"| MMR λ={experiment['mmr_lambda']}"
        )

        lines.append(
            f"  Precision@1 : "
            f"{pct(aggregate['precision@1'])}"
        )

        lines.append(
            f"  Precision@3 : "
            f"{pct(aggregate['precision@3'])}"
        )

        lines.append(
            f"  Precision@5 : "
            f"{pct(aggregate['precision@5'])}"
        )

        lines.append(
            f"  Recall@5    : "
            f"{pct(aggregate['recall@5'])}"
        )

        lines.append(
            f"  GradedRecall@5: "
            f"{pct(aggregate['graded_recall@5'])}"
        )

        lines.append(
            f"  PrimaryHit@5: "
            f"{pct(aggregate['primary_hit_rate@5'])}"
        )

        lines.append(
            f"  HitRate@5   : "
            f"{pct(aggregate['hit_rate@5'])}"
        )

        lines.append(
            f"  NDCG@5      : "
            f"{num(aggregate['ndcg@5'])}"
        )

        lines.append(
            f"  MRR         : "
            f"{num(aggregate['mrr'])}"
        )

        lines.append(
            f"  Redundancy  : "
            f"{num(aggregate['semantic_redundancy_avg'])}"
        )

        lines.append(
            f"  Unique ratio: "
            f"{pct(aggregate['unique_ratio_avg'])}"
        )

        lines.append(
            f"  Avg accepted: "
            f"{num(aggregate['accepted_count_avg'], 2)}"
        )

        lines.append(
            f"  Filter rate : "
            f"{pct(aggregate['filter_rate_avg'])}"
        )

        lines.append(
            f"  Gold safety : "
            f"{pct(aggregate['gold_filter_recall_avg'])}"
        )

        lines.append(
            f"  Queries that filtered gold: "
            f"{aggregate['queries_with_filtered_gold']}"
        )

    # -------------------------------------------------------------------------
    # Query type
    # -------------------------------------------------------------------------

    lines.append(
        "\n"
        + "=" * 100
    )

    lines.append(
        "QUERY TYPE BREAKDOWN"
    )

    lines.append(
        "=" * 100
    )

    for threshold_key, experiment in (
        evaluation[
            "experiments"
        ].items()
    ):

        lines.append(
            f"\nThreshold={experiment['threshold']}"
        )

        by_type = experiment[
            "by_query_type"
        ]

        for query_type, data in by_type.items():

            metrics = data[
                "metrics"
            ]

            lines.append(
                f"\n  {query_type.upper()} "
                f"({data['query_count']} queries)"
            )

            lines.append(
                f"    Precision@5: "
                f"{pct(metrics['precision@5'])}"
            )

            lines.append(
                f"    Recall@5:    "
                f"{pct(metrics['recall@5'])}"
            )

            lines.append(
                f"    GradedRecall@5: "
                f"{pct(metrics['graded_recall@5'])}"
            )

            lines.append(
                f"    HitRate@5:   "
                f"{pct(metrics['hit_rate@5'])}"
            )

            lines.append(
                f"    MRR:         "
                f"{num(metrics['mrr'])}"
            )

    # -------------------------------------------------------------------------
    # Query-level failures
    # -------------------------------------------------------------------------

    lines.append(
        "\n"
        + "=" * 100
    )

    lines.append(
        "QUERY-LEVEL FAILURES"
    )

    lines.append(
        "=" * 100
    )

    for threshold_key, experiment in (
        evaluation[
            "experiments"
        ].items()
    ):

        lines.append(
            f"\nThreshold={experiment['threshold']}"
        )

        for result in experiment[
            "queries"
        ]:

            p5 = result[
                "metrics"
            ][
                "precision@5"
            ]

            r5 = result[
                "metrics"
            ][
                "recall@5"
            ]

            gold_filtered = result[
                "filter_safety"
            ][
                "rejected_gold_count"
            ]

            if (
                p5 < 0.5
                or r5 < 0.5
                or gold_filtered > 0
            ):

                lines.append(
                    "\n"
                    f"  Query #{result['query_index']}: "
                    f"{result['query']}"
                )

                lines.append(
                    f"    Type: {result['query_type']}"
                )

                lines.append(
                    f"    P@5: {pct(p5)}"
                )

                lines.append(
                    f"    R@5: {pct(r5)}"
                )

                lines.append(
                    f"    Candidates: "
                    f"{result['candidate_count']}"
                )

                lines.append(
                    f"    Accepted: "
                    f"{result['accepted_count']}"
                )

                lines.append(
                    f"    Gold filtered: "
                    f"{gold_filtered}"
                )

                if gold_filtered:

                    lines.append(
                        "    WARNING: "
                        "relevance threshold removed a gold chunk."
                    )

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "RELEVANCE FILTER + METADATA-AWARE MMR EVALUATION"
    )

    print(
        "=" * 100
    )

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nGold dataset:\n{GOLD_DATASET}"
    )

    # -------------------------------------------------------------------------
    # Gold
    # -------------------------------------------------------------------------

    print(
        "\n[1/3] Loading gold dataset..."
    )

    gold_queries = (
        load_gold_dataset()
    )

    print(
        f"Loaded {len(gold_queries)} gold queries."
    )

    # -------------------------------------------------------------------------
    # Repository
    # -------------------------------------------------------------------------

    print(
        "\n[2/3] Loading repository..."
    )

    chunks = (
        load_repository_chunks()
    )

    print(
        f"Loaded {len(chunks)} chunks."
    )

    # -------------------------------------------------------------------------
    # GOLD ALIGNMENT VALIDATION
    # -------------------------------------------------------------------------

    print(
        "\n[3/4] Resolving gold anchors against current chunks..."
    )

    gold_validation = validate_gold_against_chunks(
        gold_queries,
        chunks,
    )

    print(
        f"Gold anchors:     "
        f"{gold_validation['gold_anchor_count']}"
    )

    print(
        f"Resolved anchors: "
        f"{gold_validation['resolved_anchor_count']}"
    )

    print(
        f"Missing anchors:  "
        f"{gold_validation['missing_anchor_count']}"
    )

    print(
        f"Anchor match rate: "
        f"{pct(gold_validation['gold_anchor_match_rate'])}"
    )

    if gold_validation["missing_anchor_count"]:
        print(
            "\nWARNING: Some gold anchors could not be resolved."
        )

        for missing in gold_validation[
            "missing_anchors"
        ][:20]:
            print(
                f"  - {missing['path']} :: "
                f"{missing['anchor']}"
            )

        raise RuntimeError(
            "Gold alignment is incomplete. "
            "Fix the missing anchors before trusting metrics."
        )

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    print(
        "\n[3/3] Running experiments..."
    )

    experiments = {}

    # Start with λ=0.7 because this has been the most interesting configuration
    # in your previous experiments.
    #
    # We will still test all thresholds.
    selected_lambda = 0.7

    for threshold in SEMANTIC_THRESHOLDS:

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"THRESHOLD = {threshold}"
        )

        print(
            f"MMR λ = {selected_lambda}"
        )

        print(
            "-" * 100
        )

        query_results = []

        for index, gold in enumerate(
            gold_queries,
            start=1,
        ):

            print(
                f"\n[{index}/{len(gold_queries)}] "
                f"{gold['_query']}"
            )

            result = evaluate_query(
                gold=gold,
                chunks=chunks,
                threshold=threshold,
                mmr_lambda=selected_lambda,
            )

            query_results.append(
                result
            )

            metrics = result[
                "metrics"
            ]

            print(
                f"  P@5={pct(metrics['precision@5'])} "
                f"R@5={pct(metrics['recall@5'])} "
                f"MRR={num(metrics['mrr'])}"
            )

            print(
                f"  Candidates={result['candidate_count']} "
                f"Accepted={result['accepted_count']} "
                f"Filtered={result['rejected_count']}"
            )

            if (
                result[
                    "filter_safety"
                ][
                    "rejected_gold_count"
                ]
                > 0
            ):

                print(
                    "  !!! WARNING: "
                    "gold chunk was filtered !!!"
                )

        aggregate = (
            aggregate_results(
                query_results
            )
        )

        experiments[
            f"threshold_{threshold}"
        ] = {
            "threshold": threshold,
            "mmr_lambda": selected_lambda,
            "aggregate": aggregate,
            "by_query_type": aggregate_by_type(
                query_results
            ),
            "queries": query_results,
        }

        print(
            "\nSUMMARY"
        )

        print(
            f"Precision@5: "
            f"{pct(aggregate['precision@5'])}"
        )

        print(
            f"Recall@5: "
            f"{pct(aggregate['recall@5'])}"
        )

        print(
            f"HitRate@5: "
            f"{pct(aggregate['hit_rate@5'])}"
        )

        print(
            f"MRR: "
            f"{num(aggregate['mrr'])}"
        )

        print(
            f"Semantic redundancy: "
            f"{num(aggregate['semantic_redundancy_avg'])}"
        )

        print(
            f"Average candidates accepted: "
            f"{num(aggregate['accepted_count_avg'], 2)}"
        )

        print(
            f"Gold filter safety: "
            f"{pct(aggregate['gold_filter_recall_avg'])}"
        )

    # -------------------------------------------------------------------------
    # Final evaluation object
    # -------------------------------------------------------------------------

    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "repository": REPOSITORY_URL,
        "gold_dataset": str(
            GOLD_DATASET
        ),
        "configuration": {
            "top_k_values": TOP_K_VALUES,
            "semantic_weight": SEMANTIC_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "rrf_k": RRF_K,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "semantic_thresholds": SEMANTIC_THRESHOLDS,
            "metadata_weight": METADATA_WEIGHT,
            "lexical_weight": LEXICAL_WEIGHT,
            "mmr_enabled": MMR_ENABLED,
            "mmr_lambda": selected_lambda,
            "anchor_match_min_score": ANCHOR_MATCH_MIN_SCORE,
            "gold_alignment": "path + human-readable anchor resolved against current chunks",
        },
        "gold_query_count": len(
            gold_queries
        ),
        "chunk_count": len(
            chunks
        ),
        "gold_validation": gold_validation,
        "experiments": experiments,
    }

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = (
        OUTPUT_DIR
        / f"relevance_filter_evaluation_{timestamp}.json"
    )

    txt_path = (
        OUTPUT_DIR
        / f"relevance_filter_evaluation_{timestamp}.txt"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            evaluation,
            f,
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
    ) as f:

        f.write(
            report
        )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"\nJSON result:"
        f"\n{json_path}"
    )

    print(
        f"\nText report:"
        f"\n{txt_path}"
    )

    print(
        "\nDo NOT commit these result files to Git."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
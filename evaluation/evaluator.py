from __future__ import annotations

"""
Gold-grounded retrieval evaluator for the Smart Research Dashboard RAG.

Key design decision:
    The gold dataset uses repository path + human-readable section anchors,
    NOT fixed chunk indices. This evaluator resolves those anchors against
    the chunks produced by the CURRENT DocumentChunker before scoring.

Therefore chunk boundaries can change without invalidating the gold set.

Run from project root:
    python evaluation/evaluator.py

Outputs:
    evaluation/results/retrieval_evaluation_<timestamp>.json
    evaluation/results/retrieval_evaluation_<timestamp>.txt
"""


import hashlib
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Actual project imports
# ---------------------------------------------------------------------------

try:
    from src.services.github.repository_indexer import GitHubRepositoryIndexer
    from src.services.github.content_acquirer import GitHubContentAcquirer
    from src.services.rag.chunker import DocumentChunker
    from src.services.rag.hybrid_retriever import HybridRetriever
except Exception:
    print("\nERROR: Could not import the project's RAG components.")
    print(f"Project root: {PROJECT_ROOT}")
    traceback.print_exc()
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPOSITORY_URL = "https://github.com/scikit-learn/scikit-learn.git"
GOLD_DATASET = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "scikit_learn_retrieval_gold_v1.json"
)
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_VALUES = [1, 3, 5, 10]
SEMANTIC_WEIGHT = 0.5
BM25_WEIGHT = 0.5
RRF_K = 60
CANDIDATE_MULTIPLIER = 4
MMR_LAMBDAS = [1.0, 0.9, 0.7, 0.5]

# A gold anchor can resolve to more than one chunk when a section is split.
# That is intentional: every chunk belonging to the anchored section is
# potentially useful context for the answer.
ANCHOR_MATCH_MIN_SCORE = 50

# ---------------------------------------------------------------------------
# Text / numeric helpers
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\\", "/").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_anchor(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("`", "")
    text = re.sub(r"[^\w\s²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_path(value: Any) -> str:
    if value is None:
        return ""
    path = str(value).strip().replace("\\", "/")
    while "//" in path:
        path = path.replace("//", "/")
    return path.lstrip("./")


def safe_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def format_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def format_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


# ---------------------------------------------------------------------------
# Chunk identity / content fingerprints
# ---------------------------------------------------------------------------


def chunk_id(chunk: Dict[str, Any]) -> str:
    path = canonical_path(chunk.get("path", ""))
    index = chunk.get("chunk_index")
    if path or index is not None:
        return f"{path}|{index}"
    content = str(chunk.get("content", ""))
    digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()
    return f"content|{digest}"


def content_fingerprint(chunk: Dict[str, Any]) -> str:
    """Fingerprint normalized content to detect duplicate payloads even if IDs differ."""
    content = normalize_text(chunk.get("content", ""))
    digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()
    return digest


def result_id(result: Dict[str, Any]) -> str:
    return chunk_id(result)


# ---------------------------------------------------------------------------
# Gold dataset
# ---------------------------------------------------------------------------


def load_gold_dataset() -> Dict[str, Any]:
    if not GOLD_DATASET.exists():
        raise FileNotFoundError(f"Gold dataset not found: {GOLD_DATASET}")

    with open(GOLD_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        queries = data
        metadata: Dict[str, Any] = {}
    elif isinstance(data, dict):
        queries = data.get("queries")
        metadata = data
    else:
        raise ValueError("Gold dataset must be a JSON object or list.")

    if not isinstance(queries, list):
        raise ValueError("Gold dataset does not contain a 'queries' list.")

    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(queries, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Gold query #{i} is not an object.")
        query = item.get("query") or item.get("question") or item.get("text")
        if not query:
            raise ValueError(f"Gold query #{i} has no query field.")
        gold_items = item.get("gold")
        if not isinstance(gold_items, list) or not gold_items:
            raise ValueError(
                f"Gold query {item.get('id', i)} must contain a non-empty 'gold' list."
            )
        normalized.append({**item, "_query_index": i, "_query": str(query)})

    return {"metadata": metadata, "queries": normalized}


def normalize_gold_item(item: Dict[str, Any]) -> Dict[str, Any]:
    path = canonical_path(item.get("path", item.get("file", "")))
    anchor = str(item.get("anchor", item.get("section", "")))
    relevance = safe_int(item.get("relevance")) or 0
    if relevance not in (1, 2, 3):
        raise ValueError(f"Invalid gold relevance {relevance} for anchor {anchor!r}.")
    return {
        "path": path,
        "anchor": anchor,
        "anchor_norm": normalize_anchor(anchor),
        "relevance": relevance,
    }


# ---------------------------------------------------------------------------
# Anchor resolution
# ---------------------------------------------------------------------------


def _section_values(chunk: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("section", "parent_section", "section_path", "title", "heading"):
        value = chunk.get(key)
        if value:
            values.append(str(value))
    return values


def _section_segments(chunk: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for value in _section_values(chunk):
        for part in re.split(r"\s*(?:>|/|::|\\)\s*", str(value)):
            if part.strip():
                values.append(part.strip())
    return values


def anchor_match_score(chunk: Dict[str, Any], gold: Dict[str, Any]) -> int:
    """Return confidence that a chunk belongs to a gold path+anchor.

    Scores are deliberately metadata-first. We do not use semantic similarity
    to define gold relevance because that would leak the retriever's behavior
    into the evaluation labels.
    """
    if canonical_path(chunk.get("path", "")) != gold["path"]:
        return 0

    anchor = gold["anchor_norm"]
    if not anchor:
        return 0

    section_values = [normalize_anchor(v) for v in _section_values(chunk) if v]
    segments = [normalize_anchor(v) for v in _section_segments(chunk) if v]
    content = normalize_text(chunk.get("content", ""))
    content_norm = normalize_anchor(content)

    # Exact section metadata is strongest.
    if anchor in section_values:
        return 100
    if anchor in segments:
        return 95

    # Exact section path ending / containing the anchor as a segment.
    for value in section_values:
        if value.endswith(anchor):
            return 90

    # Heading/section metadata may contain a little formatting around anchor.
    if any(anchor in value for value in section_values):
        # Avoid treating a broad anchor such as "ordinary least squares" as
        # the more specific "ordinary least squares complexity" section.
        for value in section_values:
            if value != anchor and value.startswith(anchor):
                remainder = value[len(anchor):].strip(" -:>|")
                if remainder and len(remainder.split()) <= 6:
                    return 70
        return 80

    # Last resort: anchor appears in chunk content, preferably as a heading.
    if anchor in content_norm:
        return 55

    return 0


def resolve_gold_anchors(
    gold_query: Dict[str, Any], chunks: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Resolve every path+anchor gold target to current chunk IDs.

    If a section is split across several chunks, every chunk in that section is
    mapped to the anchor. Relevance is inherited from the anchor (3/2/1).
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

        best_score = max(score for score, _ in candidates)
        # Prefer the strongest metadata interpretation. If exact section
        # metadata exists, do not pull in weaker content-only matches.
        selected = [(score, chunk) for score, chunk in candidates if score >= best_score]

        # For exact section/path matches, there can legitimately be multiple
        # chunks in the same section. For content-only matches, keep only the
        # best match to avoid falsely labeling unrelated chunks.
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
            if existing is None or item["relevance"] > existing["relevance"]:
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


def build_gold_relevance_map(
    gold_query: Dict[str, Any], chunks: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    return resolve_gold_anchors(gold_query, chunks)


# ---------------------------------------------------------------------------
# Retrieval construction / repository loading
# ---------------------------------------------------------------------------


def create_retriever(mmr_lambda: float) -> HybridRetriever:
    return HybridRetriever(
        semantic_weight=SEMANTIC_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        rrf_k=RRF_K,
        candidate_multiplier=CANDIDATE_MULTIPLIER,
        mmr_lambda=mmr_lambda,
    )


def load_repository_chunks() -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("LOADING REPOSITORY")
    print("=" * 80)
    print(f"\nRepository: {REPOSITORY_URL}")

    print("\n[1/3] Discovering repository...")
    files = GitHubRepositoryIndexer.discover(REPOSITORY_URL)
    print(f"Files discovered: {len(files)}")

    print("\n[2/3] Acquiring repository content...")
    documents = GitHubContentAcquirer.acquire(files)
    print(f"Documents acquired: {len(documents)}")

    print("\n[3/3] Creating chunks with current DocumentChunker...")
    chunks = DocumentChunker().chunk_documents(documents)
    print(f"Chunks created: {len(chunks)}")
    return list(chunks)


# ---------------------------------------------------------------------------
# Gold validation
# ---------------------------------------------------------------------------


def validate_gold_against_chunks(
    gold_queries: Sequence[Dict[str, Any]], chunks: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    total_anchors = 0
    resolved_anchors = 0
    missing: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    resolved_chunk_counts: List[int] = []

    for gold in gold_queries:
        resolution = resolve_gold_anchors(gold, chunks)
        total_anchors += resolution["gold_anchor_count"]
        resolved_anchors += resolution["gold_anchor_count"] - len(resolution["missing"])
        missing.extend([{**m, "query_id": gold.get("id")} for m in resolution["missing"]])
        ambiguous.extend([{**a, "query_id": gold.get("id")} for a in resolution["ambiguous"]])
        resolved_chunk_counts.append(resolution["resolved_chunk_count"])

    return {
        "gold_anchor_count": total_anchors,
        "resolved_anchor_count": resolved_anchors,
        "missing_anchor_count": len(missing),
        "gold_anchor_match_rate": resolved_anchors / total_anchors if total_anchors else 0.0,
        "missing_anchors": missing,
        "ambiguous_anchors": ambiguous,
        "resolved_chunks_total": int(sum(resolved_chunk_counts)),
        "chunk_count": len(chunks),
    }


# ---------------------------------------------------------------------------
# Relevance lookup
# ---------------------------------------------------------------------------


def relevance_for_result(result: Dict[str, Any], gold_map: Dict[str, Any]) -> int:
    info = gold_map.get(result_id(result))
    return int(info["relevance"]) if info else 0


def anchor_for_result(result: Dict[str, Any], gold_map: Dict[str, Any]) -> Optional[str]:
    info = gold_map.get(result_id(result))
    return info["anchor"] if info else None


# ---------------------------------------------------------------------------
# Graded retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any], k: int) -> float:
    selected = results[:k]
    if not selected:
        return 0.0
    return sum(relevance_for_result(r, gold_map) > 0 for r in selected) / len(selected)


def weighted_precision_at_k(
    results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any], k: int
) -> float:
    selected = results[:k]
    if not selected:
        return 0.0
    return sum(relevance_for_result(r, gold_map) for r in selected) / (3.0 * len(selected))


def recall_at_k(results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any], k: int) -> float:
    gold_ids = set(gold_map)
    if not gold_ids:
        return 0.0
    retrieved = {result_id(r) for r in results[:k]} & gold_ids
    return len(retrieved) / len(gold_ids)


def graded_recall_at_k(results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any], k: int) -> float:
    if not gold_map:
        return 0.0
    total = sum(info["relevance"] for info in gold_map.values())
    retrieved = {result_id(r) for r in results[:k]}
    covered = sum(info["relevance"] for cid, info in gold_map.items() if cid in retrieved)
    return covered / total if total else 0.0


def hit_rate_at_k(results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any], k: int) -> float:
    return 1.0 if any(relevance_for_result(r, gold_map) > 0 for r in results[:k]) else 0.0


def primary_hit_rate_at_k(results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any], k: int) -> float:
    return 1.0 if any(relevance_for_result(r, gold_map) == 3 for r in results[:k]) else 0.0


def reciprocal_rank(results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any]) -> float:
    for rank, result in enumerate(results, 1):
        if relevance_for_result(result, gold_map) > 0:
            return 1.0 / rank
    return 0.0


def primary_reciprocal_rank(results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any]) -> float:
    for rank, result in enumerate(results, 1):
        if relevance_for_result(result, gold_map) == 3:
            return 1.0 / rank
    return 0.0


def dcg_at_k(results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any], k: int) -> float:
    total = 0.0
    for rank, result in enumerate(results[:k], 1):
        rel = relevance_for_result(result, gold_map)
        if rel:
            total += (2**rel - 1) / math.log2(rank + 1)
    return total


def ndcg_at_k(results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any], k: int) -> float:
    if not gold_map:
        return 0.0
    actual = dcg_at_k(results, gold_map, k)
    ideal_relevances = sorted((info["relevance"] for info in gold_map.values()), reverse=True)[:k]
    ideal = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, 1))
    return actual / ideal if ideal else 0.0


# ---------------------------------------------------------------------------
# Duplicate / diversity diagnostics
# ---------------------------------------------------------------------------


def duplicate_count(results: Sequence[Dict[str, Any]]) -> int:
    ids = [result_id(r) for r in results]
    return len(ids) - len(set(ids))


def content_duplicate_count(results: Sequence[Dict[str, Any]]) -> int:
    fps = [content_fingerprint(r) for r in results if str(r.get("content", "")).strip()]
    return len(fps) - len(set(fps))


def unique_ratio(results: Sequence[Dict[str, Any]]) -> float:
    if not results:
        return 1.0
    ids = [result_id(r) for r in results]
    return len(set(ids)) / len(ids)


def content_unique_ratio(results: Sequence[Dict[str, Any]]) -> float:
    if not results:
        return 1.0
    fps = [content_fingerprint(r) for r in results]
    return len(set(fps)) / len(fps)


def same_document_ratio(results: Sequence[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    paths = [canonical_path(r.get("path", "")) for r in results]
    counts = Counter(paths)
    return max(counts.values()) / len(paths) if counts else 0.0


def complementary_preservation(
    results: Sequence[Dict[str, Any]], gold_map: Dict[str, Any]
) -> Dict[str, Any]:
    """Measure whether supporting gold chunks survive alongside primary chunks."""
    retrieved_ids = {result_id(r) for r in results}
    primary = {cid for cid, info in gold_map.items() if info["relevance"] == 3}
    supporting = {cid for cid, info in gold_map.items() if info["relevance"] == 2}
    contextual = {cid for cid, info in gold_map.items() if info["relevance"] == 1}

    return {
        "primary_total": len(primary),
        "primary_retrieved": len(primary & retrieved_ids),
        "primary_recall": len(primary & retrieved_ids) / len(primary) if primary else 0.0,
        "supporting_total": len(supporting),
        "supporting_retrieved": len(supporting & retrieved_ids),
        "supporting_recall": len(supporting & retrieved_ids) / len(supporting) if supporting else 0.0,
        "contextual_total": len(contextual),
        "contextual_retrieved": len(contextual & retrieved_ids),
        "contextual_recall": len(contextual & retrieved_ids) / len(contextual) if contextual else 0.0,
        "both_primary_and_supporting": bool(primary & retrieved_ids) and bool(supporting & retrieved_ids),
    }


# ---------------------------------------------------------------------------
# Semantic redundancy
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return normalized @ normalized.T


def compute_semantic_redundancy(results: Sequence[Dict[str, Any]]) -> Optional[float]:
    contents = [str(r.get("content", "")) for r in results]
    if len(contents) < 2:
        return None
    try:
        from src.services.rag.retriever import SimpleRetriever

        encoder = SimpleRetriever._get_model()
        embeddings = encoder.encode(
            contents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix = cosine_similarity_matrix(np.asarray(embeddings, dtype=np.float32))
        values = [float(matrix[i, j]) for i in range(len(contents)) for j in range(i + 1, len(contents))]
        return float(np.mean(values)) if values else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def summarize_result(
    result: Dict[str, Any],
    rank: int,
    gold_map: Dict[str, Any],
) -> Dict[str, Any]:
    cid = result_id(result)
    gold_info = gold_map.get(cid)
    return {
        "rank": rank,
        "id": cid,
        "path": canonical_path(result.get("path", "")),
        "chunk_index": result.get("chunk_index"),
        "chunk_type": result.get("chunk_type"),
        "category": result.get("category"),
        "section": result.get("section"),
        "parent_section": result.get("parent_section"),
        "section_path": result.get("section_path"),
        "semantic_rank": result.get("semantic_rank"),
        "bm25_rank": result.get("bm25_rank"),
        "similarity": safe_float(result.get("similarity")),
        "bm25_score": safe_float(result.get("bm25_score")),
        "hybrid_score": safe_float(result.get("hybrid_score")),
        "mmr_score": safe_float(result.get("mmr_score")),
        "mmr_relevance": safe_float(result.get("mmr_relevance")),
        "mmr_redundancy": safe_float(result.get("mmr_redundancy")),
        "gold_relevance": int(gold_info["relevance"]) if gold_info else 0,
        "gold_anchor": gold_info["anchor"] if gold_info else None,
        "gold_match_score": gold_info["match_score"] if gold_info else None,
        "relevant": bool(gold_info),
        "preview": str(result.get("content", "")).replace("\n", " ")[:500],
    }


# ---------------------------------------------------------------------------
# Query evaluation
# ---------------------------------------------------------------------------


def evaluate_query(
    gold: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    mmr_lambda: float,
) -> Dict[str, Any]:
    query = gold["_query"]
    resolution = build_gold_relevance_map(gold, chunks)
    gold_map = resolution["resolved"]

    retriever = create_retriever(mmr_lambda)
    max_k = max(TOP_K_VALUES)
    raw_results = retriever.retrieve(question=query, chunks=list(chunks), top_k=max_k)
    raw_results = list(raw_results or [])

    summarized = [
        summarize_result(result, rank, gold_map)
        for rank, result in enumerate(raw_results, 1)
    ]

    metrics: Dict[str, Any] = {}
    for k in TOP_K_VALUES:
        metrics[f"precision@{k}"] = precision_at_k(raw_results, gold_map, k)
        metrics[f"weighted_precision@{k}"] = weighted_precision_at_k(raw_results, gold_map, k)
        metrics[f"recall@{k}"] = recall_at_k(raw_results, gold_map, k)
        metrics[f"graded_recall@{k}"] = graded_recall_at_k(raw_results, gold_map, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(raw_results, gold_map, k)
        metrics[f"primary_hit_rate@{k}"] = primary_hit_rate_at_k(raw_results, gold_map, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(raw_results, gold_map, k)

    metrics["mrr"] = reciprocal_rank(raw_results, gold_map)
    metrics["primary_mrr"] = primary_reciprocal_rank(raw_results, gold_map)
    metrics["duplicate_count"] = duplicate_count(raw_results)
    metrics["content_duplicate_count"] = content_duplicate_count(raw_results)
    metrics["unique_ratio"] = unique_ratio(raw_results)
    metrics["content_unique_ratio"] = content_unique_ratio(raw_results)
    metrics["same_document_ratio"] = same_document_ratio(raw_results)
    metrics["semantic_redundancy"] = compute_semantic_redundancy(raw_results)

    complementary = complementary_preservation(raw_results, gold_map)

    return {
        "query_id": gold.get("id", gold.get("query_id", gold["_query_index"])),
        "query_index": gold["_query_index"],
        "query": query,
        "type": gold.get("type", gold.get("query_type", "unknown")),
        "category": gold.get("category", gold.get("type", "unknown")),
        "gold_anchors": [normalize_gold_item(x) for x in gold["gold"]],
        "gold_resolution": {
            "resolved_chunk_count": resolution["resolved_chunk_count"],
            "missing": resolution["missing"],
            "ambiguous": resolution["ambiguous"],
            "resolved": list(resolution["resolved"].values()),
        },
        "retrieved_count": len(summarized),
        "metrics": metrics,
        "complementary_preservation": complementary,
        "results": summarized,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def average_metric(results: Sequence[Dict[str, Any]], metric: str) -> float:
    values = []
    for result in results:
        value = result["metrics"].get(metric)
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def aggregate_results(query_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    metric_names = []
    for k in TOP_K_VALUES:
        metric_names.extend(
            [
                f"precision@{k}",
                f"weighted_precision@{k}",
                f"recall@{k}",
                f"graded_recall@{k}",
                f"hit_rate@{k}",
                f"primary_hit_rate@{k}",
                f"ndcg@{k}",
            ]
        )
    metric_names.extend(
        [
            "mrr",
            "primary_mrr",
            "duplicate_count",
            "content_duplicate_count",
            "unique_ratio",
            "content_unique_ratio",
            "same_document_ratio",
            "semantic_redundancy",
        ]
    )
    for name in metric_names:
        aggregate[name] = average_metric(query_results, name)

    # Aggregate complementary preservation separately.
    for key in (
        "primary_recall",
        "supporting_recall",
        "contextual_recall",
    ):
        vals = [float(r["complementary_preservation"][key]) for r in query_results]
        aggregate[f"complementary_{key}"] = float(np.mean(vals)) if vals else 0.0

    return aggregate


def aggregate_by_type(query_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in query_results:
        groups[str(result.get("type", "unknown"))].append(result)
    return {
        key: {"query_count": len(values), "metrics": aggregate_results(values)}
        for key, values in groups.items()
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_text_report(evaluation: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    add("=" * 110)
    add("GOLD-GROUNDED RAG RETRIEVAL EVALUATION REPORT")
    add("=" * 110)
    add(f"Generated: {evaluation['timestamp']}")
    add(f"Repository: {evaluation['repository']}")
    add(f"Gold dataset: {evaluation['gold_dataset']}")

    gv = evaluation["gold_validation"]
    add("\n" + "-" * 110)
    add("GOLD DATASET VALIDATION")
    add("-" * 110)
    add(f"Gold anchors:          {gv['gold_anchor_count']}")
    add(f"Resolved anchors:      {gv['resolved_anchor_count']}")
    add(f"Missing anchors:       {gv['missing_anchor_count']}")
    add(f"Anchor match rate:     {format_pct(gv['gold_anchor_match_rate'])}")
    add(f"Resolved chunks total:  {gv['resolved_chunks_total']}")

    if gv["missing_anchors"]:
        add("\nMISSING GOLD ANCHORS:")
        for item in gv["missing_anchors"]:
            add(f"  - {item['query_id']}: {item['path']} :: {item['anchor']} (rel={item['relevance']})")

    if gv["ambiguous_anchors"]:
        add("\nANCHORS RESOLVED TO MULTIPLE CURRENT CHUNKS:")
        for item in gv["ambiguous_anchors"][:50]:
            add(f"  - {item['query_id']}: {item['anchor']} -> {item['candidate_count']} chunks")

    add("\n" + "=" * 110)
    add("RESULTS BY MMR LAMBDA")
    add("=" * 110)

    for lambda_key, data in evaluation["evaluations"].items():
        a = data["aggregate"]
        add(f"\nMMR λ={lambda_key}")
        add("-" * 110)
        for k in (1, 3, 5, 10):
            add(
                f"Recall@{k:<2}: {format_pct(a[f'recall@{k}']):>8} | "
                f"GradedRecall@{k:<2}: {format_pct(a[f'graded_recall@{k}']):>8} | "
                f"nDCG@{k:<2}: {format_pct(a[f'ndcg@{k}']):>8}"
            )
        add(f"Precision@5:            {format_pct(a['precision@5'])}")
        add(f"Weighted Precision@5:   {format_pct(a['weighted_precision@5'])}")
        add(f"Hit@5:                  {format_pct(a['hit_rate@5'])}")
        add(f"Primary Hit@5:          {format_pct(a['primary_hit_rate@5'])}")
        add(f"MRR:                    {format_num(a['mrr'])}")
        add(f"Primary MRR:            {format_num(a['primary_mrr'])}")
        add(f"Unique ID ratio:        {format_pct(a['unique_ratio'])}")
        add(f"Unique content ratio:   {format_pct(a['content_unique_ratio'])}")
        add(f"Duplicate IDs/query:    {format_num(a['duplicate_count'])}")
        add(f"Duplicate content/query:{format_num(a['content_duplicate_count'])}")
        add(f"Same-document ratio:    {format_pct(a['same_document_ratio'])}")
        add(f"Semantic redundancy:    {format_num(a['semantic_redundancy'])}")
        add(f"Primary recall:         {format_pct(a['complementary_primary_recall'])}")
        add(f"Supporting recall:      {format_pct(a['complementary_supporting_recall'])}")

        add("\nBY QUERY TYPE:")
        for qtype, breakdown in data["by_type"].items():
            ba = breakdown["metrics"]
            add(
                f"  {qtype:<20} n={breakdown['query_count']:<2} "
                f"Recall@5={format_pct(ba['recall@5']):>8} "
                f"nDCG@5={format_pct(ba['ndcg@5']):>8} "
                f"MRR={format_num(ba['mrr']):>7}"
            )

    # Main detailed diagnostics use λ=0.7 because that is the configuration
    # previously inspected during development, while all lambdas are scored.
    main_eval = evaluation["evaluations"].get("0.7")
    if main_eval:
        add("\n" + "=" * 110)
        add("QUERY-LEVEL DIAGNOSTICS — MMR λ=0.7")
        add("=" * 110)
        for result in main_eval["query_results"]:
            m = result["metrics"]
            add("\n" + "-" * 110)
            add(f"{result['query_id']} | {result['type']} | {result['query']}")
            add(
                f"Recall@5={format_pct(m['recall@5'])} | "
                f"GradedRecall@5={format_pct(m['graded_recall@5'])} | "
                f"nDCG@5={format_pct(m['ndcg@5'])} | "
                f"MRR={format_num(m['mrr'])} | "
                f"Unique={format_pct(m['unique_ratio'])}"
            )

            cp = result["complementary_preservation"]
            add(
                f"Primary recall={format_pct(cp['primary_recall'])} | "
                f"Supporting recall={format_pct(cp['supporting_recall'])}"
            )

            if result["gold_resolution"]["missing"]:
                add("GOLD RESOLUTION WARNING:")
                for item in result["gold_resolution"]["missing"]:
                    add(f"  ! missing {item['path']} :: {item['anchor']}")

            add("Retrieved:")
            for item in result["results"]:
                marker = "✓" if item["relevant"] else "✗"
                rel = item["gold_relevance"]
                anchor = item["gold_anchor"] or "-"
                add(
                    f"  {marker} #{item['rank']:<2} rel={rel} "
                    f"{item['path']} | chunk={item['chunk_index']} | "
                    f"section={item['section']} | gold={anchor}"
                )
                add(
                    f"       semantic_rank={item['semantic_rank']} "
                    f"bm25_rank={item['bm25_rank']} "
                    f"hybrid={format_num(item['hybrid_score'])} "
                    f"mmr={format_num(item['mmr_score'])}"
                )
                add(f"       {item['preview']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_evaluation(evaluation: Dict[str, Any]) -> Tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"retrieval_evaluation_{timestamp}.json"
    txt_path = OUTPUT_DIR / f"retrieval_evaluation_{timestamp}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(generate_text_report(evaluation))

    return json_path, txt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 110)
    print("GOLD-GROUNDED RAG RETRIEVAL EVALUATION FRAMEWORK")
    print("=" * 110)
    print(f"\nProject root:\n{PROJECT_ROOT}")
    print(f"\nGold dataset:\n{GOLD_DATASET}")

    print("\n[1/5] Loading gold dataset...")
    gold_data = load_gold_dataset()
    gold_queries = gold_data["queries"]
    print(f"Loaded {len(gold_queries)} gold queries.")

    print("\n[2/5] Loading repository and current chunks...")
    chunks = load_repository_chunks()
    if not chunks:
        raise RuntimeError("No chunks were created.")

    print("\n[3/5] Resolving gold anchors against current chunks...")
    gold_validation = validate_gold_against_chunks(gold_queries, chunks)
    print(f"Gold anchors:       {gold_validation['gold_anchor_count']}")
    print(f"Resolved anchors:   {gold_validation['resolved_anchor_count']}")
    print(f"Missing anchors:    {gold_validation['missing_anchor_count']}")
    print(f"Match rate:         {format_pct(gold_validation['gold_anchor_match_rate'])}")

    if gold_validation["missing_anchor_count"]:
        print("\nWARNING: some gold anchors could not be resolved.")
        print("The affected queries will be evaluated with the anchors that did resolve.")

    print("\n[4/5] Running retrieval evaluation...")
    evaluations: Dict[str, Any] = {}

    for mmr_lambda in MMR_LAMBDAS:
        print("\n" + "-" * 110)
        print(f"Evaluating MMR λ={mmr_lambda}")
        print("-" * 110)
        query_results: List[Dict[str, Any]] = []

        for position, gold in enumerate(gold_queries, 1):
            print(f"\r  Query {position}/{len(gold_queries)}", end="", flush=True)
            result = evaluate_query(gold, chunks, mmr_lambda)
            query_results.append(result)
        print()

        evaluations[f"{mmr_lambda:.1f}"] = {
            "mmr_lambda": mmr_lambda,
            "aggregate": aggregate_results(query_results),
            "by_type": aggregate_by_type(query_results),
            "query_results": query_results,
        }

    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "repository": REPOSITORY_URL,
        "gold_dataset": str(GOLD_DATASET),
        "gold_dataset_name": gold_data["metadata"].get("dataset_name"),
        "gold_relevance_scale": gold_data["metadata"].get("relevance_scale"),
        "configuration": {
            "top_k_values": TOP_K_VALUES,
            "semantic_weight": SEMANTIC_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "rrf_k": RRF_K,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "mmr_lambdas": MMR_LAMBDAS,
            "anchor_match_min_score": ANCHOR_MATCH_MIN_SCORE,
            "chunk_count": len(chunks),
            "gold_query_count": len(gold_queries),
        },
        "gold_validation": gold_validation,
        "evaluations": evaluations,
    }

    print("\n[5/5] Saving evaluation results...")
    json_path, txt_path = save_evaluation(evaluation)

    print("\n" + "=" * 110)
    print("EVALUATION COMPLETE")
    print("=" * 110)
    print("\nMAIN SUMMARY")
    print("-" * 110)

    for lambda_key, data in evaluations.items():
        a = data["aggregate"]
        print(f"\nMMR λ={lambda_key}")
        print(f"  Recall@1    : {format_pct(a['recall@1'])}")
        print(f"  Recall@3    : {format_pct(a['recall@3'])}")
        print(f"  Recall@5    : {format_pct(a['recall@5'])}")
        print(f"  Recall@10   : {format_pct(a['recall@10'])}")
        print(f"  GradedR@5   : {format_pct(a['graded_recall@5'])}")
        print(f"  nDCG@5      : {format_pct(a['ndcg@5'])}")
        print(f"  Precision@5 : {format_pct(a['precision@5'])}")
        print(f"  Hit@5       : {format_pct(a['hit_rate@5'])}")
        print(f"  Primary@5   : {format_pct(a['primary_hit_rate@5'])}")
        print(f"  MRR         : {format_num(a['mrr'])}")
        print(f"  Primary MRR : {format_num(a['primary_mrr'])}")
        print(f"  Unique IDs  : {format_pct(a['unique_ratio'])}")
        print(f"  Unique text : {format_pct(a['content_unique_ratio'])}")
        print(f"  Redundancy  : {format_num(a['semantic_redundancy'])}")
        print(f"  Primary rec : {format_pct(a['complementary_primary_recall'])}")
        print(f"  Support rec : {format_pct(a['complementary_supporting_recall'])}")

    print("\n" + "=" * 110)
    print("OUTPUT FILES")
    print("=" * 110)
    print(f"\nJSON:\n{json_path}")
    print(f"\nREPORT:\n{txt_path}")
    print("\nSend me the generated .txt report and/or .json file for analysis.")


if __name__ == "__main__":
    main()
from __future__ import annotations

"""
Controlled ranking ablation for the Smart Research Dashboard RAG.

A: Hybrid RRF -> relevance filter -> metadata-aware MMR (lambda=0.7)
B: Hybrid RRF -> relevance filter -> query-aware combined-score ranking (NO MMR)
C: Hybrid RRF -> relevance filter -> complementarity-aware greedy selection

The three experiments share the exact same raw Hybrid/RRF candidate pool and
threshold. This isolates the ranking/selection stage.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the existing evaluation framework for gold alignment, scoring,
# repository loading, relevance filtering, and the current MMR implementation.
import relevance_filter_evaluator as base  # type: ignore

from src.services.rag.bm25_retriever import BM25Retriever
from src.services.rag.retriever import SimpleRetriever


# ---------------------------------------------------------------------------
# Controlled experiment configuration
# ---------------------------------------------------------------------------
TOP_K = 5
THRESHOLD = 0.20
MMR_LAMBDA = 0.70

# Keep the same hybrid weights/RRF/candidate multiplier as the existing setup.
SEMANTIC_WEIGHT = base.SEMANTIC_WEIGHT
BM25_WEIGHT = base.BM25_WEIGHT
RRF_K = base.RRF_K
CANDIDATE_MULTIPLIER = base.CANDIDATE_MULTIPLIER
CANDIDATE_POOL_SIZE = base.CANDIDATE_POOL_SIZE

OUTPUT_DIR = base.OUTPUT_DIR
REPOSITORY_URL = base.REPOSITORY_URL
GOLD_DATASET = base.GOLD_DATASET
ANCHOR_MATCH_MIN_SCORE = base.ANCHOR_MATCH_MIN_SCORE


# ---------------------------------------------------------------------------
# Raw Hybrid/RRF candidate pool
# ---------------------------------------------------------------------------
def _document_id(chunk: Dict[str, Any]) -> str:
    path = str(chunk.get("path", ""))
    index = chunk.get("chunk_index")
    if path or index is not None:
        return f"{path}|{index}"
    return str(chunk.get("content", ""))


def raw_hybrid_candidates(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    pool_size: int = CANDIDATE_POOL_SIZE,
) -> List[Dict[str, Any]]:
    """Build the raw Hybrid/RRF pool WITHOUT MMR or relevance filtering."""
    chunks = list(chunks)
    if not query.strip() or not chunks:
        return []

    # Mirror HybridRetriever's candidate sizing when its requested top_k is
    # CANDIDATE_POOL_SIZE: top_k * candidate_multiplier, minimum 10.
    candidate_k = min(
        max(pool_size * CANDIDATE_MULTIPLIER, 10),
        len(chunks),
    )

    semantic_results = SimpleRetriever.retrieve(
        question=query,
        chunks=chunks,
        top_k=candidate_k,
    )

    bm25_results = BM25Retriever(chunks).retrieve(
        query=query,
        top_k=candidate_k,
    )

    fused: Dict[str, Dict[str, Any]] = {}

    for rank, result in enumerate(semantic_results, start=1):
        did = _document_id(result)
        if did not in fused:
            fused[did] = {
                "result": dict(result),
                "semantic_rank": None,
                "bm25_rank": None,
                "hybrid_score": 0.0,
            }
        fused[did]["semantic_rank"] = rank
        fused[did]["hybrid_score"] += SEMANTIC_WEIGHT / (RRF_K + rank)

    for rank, result in enumerate(bm25_results, start=1):
        did = _document_id(result)
        if did not in fused:
            fused[did] = {
                "result": dict(result),
                "semantic_rank": None,
                "bm25_rank": None,
                "hybrid_score": 0.0,
            }

        merged = fused[did]["result"]
        for key, value in result.items():
            if key not in merged:
                merged[key] = value

        fused[did]["bm25_rank"] = rank
        fused[did]["hybrid_score"] += BM25_WEIGHT / (RRF_K + rank)

    ranked = sorted(
        fused.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    candidates: List[Dict[str, Any]] = []
    for item in ranked[:pool_size]:
        result = dict(item["result"])
        result["hybrid_score"] = float(item["hybrid_score"])
        result["semantic_rank"] = item["semantic_rank"]
        result["bm25_rank"] = item["bm25_rank"]
        candidates.append(result)

    return candidates


# ---------------------------------------------------------------------------
# Query-aware ranking
# ---------------------------------------------------------------------------
def _tokens(text: Any) -> set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", str(text or "").lower()))


def _query_terms(query: str) -> set[str]:
    return _tokens(query) - base.STOPWORDS


def _section(candidate: Dict[str, Any]) -> str:
    return base.get_section(candidate)


def _source(candidate: Dict[str, Any]) -> str:
    return base.get_source(candidate)


def query_aware_score(query: str, candidate: Dict[str, Any]) -> float:
    """Use the framework's semantic + lexical + metadata relevance score."""
    semantic = float(candidate.get("_semantic_relevance", 0.0))
    return float(
        base.combined_relevance(
            query=query,
            query_type=base.classify_query(query),
            chunk=candidate,
            semantic_score=semantic,
        )["combined"]
    )


def rerank_no_mmr(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    scored = []
    for candidate in candidates:
        result = dict(candidate)
        result["query_aware_score"] = query_aware_score(query, result)
        scored.append(result)

    scored.sort(key=lambda x: x["query_aware_score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Complementarity-aware selection
# ---------------------------------------------------------------------------
def complementarity_score(
    query: str,
    query_terms: set[str],
    candidate: Dict[str, Any],
    selected: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """Score a candidate for useful new evidence, not just raw relevance.

    The selection objective rewards:
      1. query-aware relevance,
      2. new query-term coverage,
      3. section novelty,
      4. source novelty.

    It only applies a light redundancy penalty, because related sections can be
    complementary evidence rather than duplicates.
    """
    relevance = query_aware_score(query, candidate)

    content_terms = _tokens(candidate.get("content", ""))
    section_terms = _tokens(candidate.get("section", ""))
    candidate_terms = content_terms | section_terms

    selected_terms = set()
    selected_sections = set()
    selected_sources = set()
    for item in selected:
        selected_terms |= _tokens(item.get("content", ""))
        selected_terms |= _tokens(item.get("section", ""))
        selected_sections.add(_section(item))
        selected_sources.add(_source(item))

    # Coverage is query-specific: reward terms the current top results have not
    # already covered.
    if query_terms:
        new_terms = (candidate_terms & query_terms) - selected_terms
        coverage_gain = len(new_terms) / len(query_terms)
    else:
        coverage_gain = 0.0

    section = _section(candidate)
    source = _source(candidate)
    section_novelty = 1.0 if section and section not in selected_sections else 0.0
    source_novelty = 1.0 if source and source not in selected_sources else 0.0

    # Light redundancy penalty only for very similar semantic content.
    redundancy = 0.0
    if selected:
        model = SimpleRetriever._get_model()
        candidate_embedding = np.asarray(
            model.encode(
                [str(candidate.get("content", ""))],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0],
            dtype=np.float32,
        )
        selected_embeddings = np.asarray(
            model.encode(
                [str(item.get("content", "")) for item in selected],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        similarities = selected_embeddings @ candidate_embedding
        redundancy = float(np.max(similarities)) if len(similarities) else 0.0

    score = (
        0.65 * relevance
        + 0.20 * coverage_gain
        + 0.10 * section_novelty
        + 0.05 * source_novelty
        - 0.10 * max(0.0, redundancy - 0.80)
    )

    return float(score), {
        "query_aware_relevance": float(relevance),
        "coverage_gain": float(coverage_gain),
        "section_novelty": float(section_novelty),
        "source_novelty": float(source_novelty),
        "redundancy": float(redundancy),
    }


def rerank_complementarity(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    query_terms = _query_terms(query)
    remaining = [dict(c) for c in candidates]
    selected: List[Dict[str, Any]] = []

    # Precompute relevance once so the selection is deterministic and efficient.
    for candidate in remaining:
        candidate["query_aware_score"] = query_aware_score(query, candidate)

    while remaining and len(selected) < top_k:
        best_index = None
        best_score = -float("inf")
        best_diag: Dict[str, float] = {}

        for i, candidate in enumerate(remaining):
            score, diag = complementarity_score(
                query=query,
                query_terms=query_terms,
                candidate=candidate,
                selected=selected,
            )
            if score > best_score:
                best_score = score
                best_index = i
                best_diag = diag

        if best_index is None:
            break

        result = remaining.pop(best_index)
        result["complementarity_score"] = float(best_score)
        result["complementarity_diagnostics"] = best_diag
        result["selection_rank"] = len(selected) + 1
        selected.append(result)

    return selected


# ---------------------------------------------------------------------------
# D: Coverage-aware constrained selection
# ---------------------------------------------------------------------------
def _normalize_section(value: Any) -> str:
    return base.normalize_text(value or "")


def _section_parts(section: str) -> List[str]:
    """Return normalized hierarchical section parts when available."""
    section = _normalize_section(section)
    if not section:
        return []
    # Our chunks commonly use ' > ' for parent/child headings.
    parts = [p.strip() for p in re.split(r"\s*>\s*", section) if p.strip()]
    return parts


def _parent_section(section: str) -> str:
    parts = _section_parts(section)
    return " > ".join(parts[:-1]) if len(parts) > 1 else ""


def _content_terms(candidate: Dict[str, Any]) -> set[str]:
    return _tokens(candidate.get("content", ""))


def _section_novelty(candidate: Dict[str, Any], selected: Sequence[Dict[str, Any]]) -> float:
    section = _section(candidate)
    if not section:
        return 0.0
    selected_sections = {_section(item) for item in selected if _section(item)}
    return 0.0 if section in selected_sections else 1.0


def _parent_novelty(candidate: Dict[str, Any], selected: Sequence[Dict[str, Any]]) -> float:
    parent = _parent_section(_section(candidate))
    if not parent:
        return 0.0
    selected_parents = {
        _parent_section(_section(item))
        for item in selected
        if _parent_section(_section(item))
    }
    return 0.0 if parent in selected_parents else 1.0


def _query_term_gain(
    query_terms: set[str],
    candidate: Dict[str, Any],
    selected: Sequence[Dict[str, Any]],
) -> float:
    if not query_terms:
        return 0.0
    covered = set()
    for item in selected:
        covered |= _content_terms(item)
        covered |= _tokens(_section(item))
    candidate_terms = _content_terms(candidate) | _tokens(_section(candidate))
    new_terms = (candidate_terms & query_terms) - covered
    return len(new_terms) / len(query_terms)


def _content_novelty(
    candidate: Dict[str, Any],
    selected: Sequence[Dict[str, Any]],
) -> float:
    """How much lexical content the candidate adds beyond selected chunks."""
    if not selected:
        return 1.0
    candidate_terms = _content_terms(candidate)
    if not candidate_terms:
        return 0.0
    covered = set()
    for item in selected:
        covered |= _content_terms(item)
    return float(len(candidate_terms - covered) / max(len(candidate_terms), 1))


def _pairwise_redundancy(
    candidate: Dict[str, Any],
    selected: Sequence[Dict[str, Any]],
    model: Any,
) -> float:
    if not selected:
        return 0.0
    texts = [str(candidate.get("content", ""))] + [str(x.get("content", "")) for x in selected]
    embeddings = np.asarray(
        model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if len(embeddings) <= 1:
        return 0.0
    sims = embeddings[1:] @ embeddings[0]
    return float(np.max(sims))


def constrained_coverage_selection(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    relevance_floor: float = 0.72,
    relevance_weight: float = 0.72,
    query_gain_weight: float = 0.08,
    section_gain_weight: float = 0.08,
    parent_gain_weight: float = 0.04,
    content_gain_weight: float = 0.08,
    redundancy_weight: float = 0.16,
) -> List[Dict[str, Any]]:
    """Select a small evidence set without using gold labels.

    Design:
      1. Always select the strongest query-aware result first.
      2. For later slots, require candidates to remain reasonably close to the
         best relevance score so diversity cannot promote weak material.
      3. Reward *new evidence*: new query terms, sections, parent sections and
         lexical content.
      4. Penalize only strong semantic redundancy.

    This is deliberately NOT gold-aware. Gold labels are used only by the
    evaluator after selection.
    """
    if not candidates:
        return []

    top_k = min(top_k, len(candidates))
    remaining = [dict(c) for c in candidates]

    for candidate in remaining:
        candidate["query_aware_score"] = query_aware_score(query, candidate)

    remaining.sort(key=lambda x: x["query_aware_score"], reverse=True)
    best_relevance = float(remaining[0]["query_aware_score"])
    relevance_floor_value = best_relevance * relevance_floor
    query_terms = _query_terms(query)
    model = SimpleRetriever._get_model()

    selected: List[Dict[str, Any]] = []

    # Slot 1: pure relevance anchor. This protects primary retrieval quality.
    first = remaining.pop(0)
    first["coverage_selection_score"] = float(first["query_aware_score"])
    first["coverage_diagnostics"] = {
        "query_aware_relevance": float(first["query_aware_score"]),
        "query_term_gain": 1.0,
        "section_gain": 1.0 if _section(first) else 0.0,
        "parent_section_gain": 1.0 if _parent_section(_section(first)) else 0.0,
        "content_gain": 1.0,
        "redundancy": 0.0,
        "relevance_floor": relevance_floor_value,
        "accepted_by_relevance_floor": True,
    }
    first["selection_rank"] = 1
    selected.append(first)

    while remaining and len(selected) < top_k:
        best_index = None
        best_score = -float("inf")
        best_diag: Dict[str, Any] = {}

        for i, candidate in enumerate(remaining):
            relevance = float(candidate["query_aware_score"])
            passes_floor = relevance >= relevance_floor_value

            # If no candidate passes the relevance floor, relax it for the
            # remaining slot rather than returning fewer than top_k results.
            if not passes_floor and any(
                float(x["query_aware_score"]) >= relevance_floor_value for x in remaining
            ):
                continue

            query_gain = _query_term_gain(query_terms, candidate, selected)
            section_gain = _section_novelty(candidate, selected)
            parent_gain = _parent_novelty(candidate, selected)
            content_gain = _content_novelty(candidate, selected)
            redundancy = _pairwise_redundancy(candidate, selected, model)

            # Redundancy only becomes meaningful when semantic similarity is
            # already high. This avoids punishing genuinely related evidence.
            redundancy_penalty = max(0.0, redundancy - 0.70)

            score = (
                relevance_weight * relevance
                + query_gain_weight * query_gain
                + section_gain_weight * section_gain
                + parent_gain_weight * parent_gain
                + content_gain_weight * content_gain
                - redundancy_weight * redundancy_penalty
            )

            if score > best_score:
                best_score = score
                best_index = i
                best_diag = {
                    "query_aware_relevance": relevance,
                    "query_term_gain": query_gain,
                    "section_gain": section_gain,
                    "parent_section_gain": parent_gain,
                    "content_gain": content_gain,
                    "redundancy": redundancy,
                    "redundancy_penalty": redundancy_penalty,
                    "relevance_floor": relevance_floor_value,
                    "accepted_by_relevance_floor": passes_floor,
                }

        if best_index is None:
            break

        result = remaining.pop(best_index)
        result["coverage_selection_score"] = float(best_score)
        result["coverage_diagnostics"] = best_diag
        result["selection_rank"] = len(selected) + 1
        selected.append(result)

    return selected


def rerank_coverage_aware(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    return constrained_coverage_selection(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# New metrics
# ---------------------------------------------------------------------------
def primary_recall_at_k(
    results: Sequence[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    primary = {cid for cid, info in gold_map.items() if info["relevance"] == 3}
    if not primary:
        return 0.0
    retrieved = {base.chunk_id(r) for r in results[:k]}
    return len(primary & retrieved) / len(primary)


def supporting_recall_at_k(
    results: Sequence[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    supporting = {cid for cid, info in gold_map.items() if info["relevance"] == 2}
    if not supporting:
        return 0.0
    retrieved = {base.chunk_id(r) for r in results[:k]}
    return len(supporting & retrieved) / len(supporting)


def anchor_coverage_at_k(
    results: Sequence[Dict[str, Any]],
    gold_map: Dict[str, Any],
    k: int,
) -> float:
    """Unique gold-anchor coverage, independent of how many chunks an anchor spans."""
    gold_anchors = {
        info["anchor"]
        for info in gold_map.values()
        if info.get("anchor")
    }
    if not gold_anchors:
        return 0.0

    retrieved_anchors = {
        gold_map[base.chunk_id(r)]["anchor"]
        for r in results[:k]
        if base.chunk_id(r) in gold_map and gold_map[base.chunk_id(r)].get("anchor")
    }
    return len(retrieved_anchors) / len(gold_anchors)


def evaluate_new_metrics(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
) -> Dict[str, float]:
    return {
        "primary_recall@5": primary_recall_at_k(results, gold_map, 5),
        "supporting_recall@5": supporting_recall_at_k(results, gold_map, 5),
        "anchor_coverage@5": anchor_coverage_at_k(results, gold_map, 5),
    }


# ---------------------------------------------------------------------------
# One query / experiment
# ---------------------------------------------------------------------------
def prepare_candidates(
    query: str,
    chunks: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw = raw_hybrid_candidates(query, chunks, CANDIDATE_POOL_SIZE)
    accepted, rejected = base.relevance_filter(query, raw, THRESHOLD)
    return raw, accepted, rejected


def run_method(
    method: str,
    query: str,
    accepted: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if method == "A_current_mmr":
        return base.mmr_rerank(
            query=query,
            candidates=[dict(c) for c in accepted],
            top_k=TOP_K,
            mmr_lambda=MMR_LAMBDA,
        )

    if method == "B_query_aware_no_mmr":
        return rerank_no_mmr(
            query=query,
            candidates=[dict(c) for c in accepted],
            top_k=TOP_K,
        )

    if method == "C_complementarity":
        return rerank_complementarity(
            query=query,
            candidates=[dict(c) for c in accepted],
            top_k=TOP_K,
        )

    if method == "D_coverage_aware":
        return rerank_coverage_aware(
            query=query,
            candidates=[dict(c) for c in accepted],
            top_k=TOP_K,
        )

    raise ValueError(f"Unknown method: {method}")


def summarize_results(
    results: List[Dict[str, Any]],
    gold_map: Dict[str, Any],
) -> List[Dict[str, Any]]:
    output = []
    for rank, result in enumerate(results, 1):
        relevance = base.relevance_for_result(result, gold_map)
        output.append({
            "rank": rank,
            "id": base.chunk_id(result),
            "path": base.canonical_path(result.get("path", "")),
            "section": result.get("section", result.get("heading", "")),
            "semantic_relevance": result.get("_semantic_relevance"),
            "lexical_relevance": result.get("_lexical_relevance"),
            "metadata_relevance": result.get("_metadata_relevance"),
            "combined_relevance": result.get("_combined_relevance"),
            "mmr_score": result.get("mmr_score"),
            "query_aware_score": result.get("query_aware_score"),
            "complementarity_score": result.get("complementarity_score"),
            "gold_relevance": relevance,
            "gold_anchor": base.anchor_for_result(result, gold_map),
            "gold_match_score": (
                gold_map[base.chunk_id(result)]["match_score"]
                if base.chunk_id(result) in gold_map
                else None
            ),
            "relevant": relevance > 0,
            "preview": base.normalize_text(result.get("content", ""))[:500],
        })
    return output


def evaluate_query(
    gold: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    method: str,
) -> Dict[str, Any]:
    query = gold["_query"]
    resolution = base.resolve_gold_anchors(gold, chunks)
    gold_map = resolution["resolved"]

    raw, accepted, rejected = prepare_candidates(query, chunks)
    results = run_method(method, query, accepted)

    # Reuse the existing core metrics and add the three objective-specific ones.
    metrics = {
        "precision@5": base.precision_at_k(results, gold_map, 5),
        "weighted_precision@5": base.weighted_precision_at_k(results, gold_map, 5),
        "recall@5": base.recall_at_k(results, gold_map, 5),
        "graded_recall@5": base.graded_recall_at_k(results, gold_map, 5),
        "hit_rate@5": base.hit_rate_at_k(results, gold_map, 5),
        "primary_hit_rate@5": base.primary_hit_rate_at_k(results, gold_map, 5),
        "ndcg@5": base.ndcg_at_k(results, gold_map, 5),
        "mrr": base.reciprocal_rank(results, gold_map),
        "primary_mrr": base.primary_reciprocal_rank(results, gold_map),
        "primary_recall@5": primary_recall_at_k(results, gold_map, 5),
        "supporting_recall@5": supporting_recall_at_k(results, gold_map, 5),
        "anchor_coverage@5": anchor_coverage_at_k(results, gold_map, 5),
        "duplicate_count": base.duplicate_count(results),
        "unique_ratio": base.unique_ratio(results),
        "semantic_redundancy": base.semantic_redundancy(results),
    }

    rejected_gold = [
        {
            "id": base.chunk_id(r),
            "relevance": base.relevance_for_result(r, gold_map),
            "anchor": base.anchor_for_result(r, gold_map),
        }
        for r in rejected
        if base.relevance_for_result(r, gold_map) > 0
    ]

    return {
        "query_id": gold.get("id"),
        "query_index": gold.get("_query_index"),
        "query": query,
        "query_type": base.classify_query(query),
        "gold_category": gold.get("type", gold.get("category", "unknown")),
        "method": method,
        "threshold": THRESHOLD,
        "mmr_lambda": MMR_LAMBDA if method == "A_current_mmr" else None,
        "candidate_count": len(raw),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "filter_rate": len(rejected) / len(raw) if raw else 0.0,
        "gold_filter_recall": 1.0 - (len(rejected_gold) / len(gold_map)) if gold_map else 1.0,
        "rejected_gold_count": len(rejected_gold),
        "metrics": metrics,
        "gold_anchors": [base.normalize_gold_item(item) for item in gold["gold"]],
        "results": summarize_results(results, gold_map),
    }


def aggregate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = [
        "precision@5",
        "weighted_precision@5",
        "recall@5",
        "graded_recall@5",
        "hit_rate@5",
        "primary_hit_rate@5",
        "ndcg@5",
        "mrr",
        "primary_mrr",
        "primary_recall@5",
        "supporting_recall@5",
        "anchor_coverage@5",
        "duplicate_count",
        "unique_ratio",
        "semantic_redundancy",
    ]
    out: Dict[str, Any] = {}
    for name in metric_names:
        values = [float(x["metrics"][name]) for x in items if x["metrics"].get(name) is not None]
        out[name] = float(np.mean(values)) if values else 0.0

    out["candidate_count_avg"] = float(np.mean([x["candidate_count"] for x in items])) if items else 0.0
    out["accepted_count_avg"] = float(np.mean([x["accepted_count"] for x in items])) if items else 0.0
    out["filter_rate_avg"] = float(np.mean([x["filter_rate"] for x in items])) if items else 0.0
    out["gold_filter_recall_avg"] = float(np.mean([x["gold_filter_recall"] for x in items])) if items else 0.0
    out["queries_with_filtered_gold"] = sum(x["rejected_gold_count"] > 0 for x in items)
    return out


def aggregate_by_query_type(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["gold_category"], []).append(item)
    return {
        key: {"query_count": len(value), "metrics": aggregate(value)}
        for key, value in groups.items()
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def report(evaluation: Dict[str, Any]) -> str:
    lines = [
        "=" * 110,
        "RANKING ABLATION: MMR vs QUERY-AWARE vs COMPLEMENTARITY vs COVERAGE-AWARE",
        "=" * 110,
        f"Generated: {evaluation['timestamp']}",
        f"Repository: {evaluation['repository']}",
        f"Gold dataset: {evaluation['gold_dataset']}",
        "",
        "CONTROLLED CONFIGURATION",
        f"  threshold: {THRESHOLD}",
        f"  MMR lambda: {MMR_LAMBDA}",
        "  D relevance floor: 72% of best query-aware score",
        "  D selection: relevance-first + evidence gain + constrained redundancy",
        f"  candidate pool: {CANDIDATE_POOL_SIZE}",
        f"  semantic weight: {SEMANTIC_WEIGHT}",
        f"  BM25 weight: {BM25_WEIGHT}",
        f"  RRF k: {RRF_K}",
        "  All methods share the same raw RRF candidate pool.",
        "",
        "METHOD SUMMARY",
        "-" * 110,
    ]

    for method, data in evaluation["experiments"].items():
        a = data["aggregate"]
        lines += [
            f"\n{method}",
            f"  Precision@5              : {pct(a['precision@5'])}",
            f"  Recall@5                 : {pct(a['recall@5'])}",
            f"  GradedRecall@5           : {pct(a['graded_recall@5'])}",
            f"  Primary Recall@5         : {pct(a['primary_recall@5'])}",
            f"  Supporting Recall@5      : {pct(a['supporting_recall@5'])}",
            f"  Anchor Coverage@5        : {pct(a['anchor_coverage@5'])}",
            f"  Primary Hit@5            : {pct(a['primary_hit_rate@5'])}",
            f"  NDCG@5                   : {a['ndcg@5']:.4f}",
            f"  MRR                      : {a['mrr']:.4f}",
            f"  Primary MRR              : {a['primary_mrr']:.4f}",
            f"  Semantic redundancy      : {a['semantic_redundancy']:.4f}",
            f"  Avg accepted             : {a['accepted_count_avg']:.2f}",
            f"  Filter rate              : {pct(a['filter_rate_avg'])}",
            f"  Gold filter safety       : {pct(a['gold_filter_recall_avg'])}",
        ]

    lines += ["", "QUERY-TYPE BREAKDOWN", "-" * 110]
    for method, data in evaluation["experiments"].items():
        lines.append(f"\n{method}")
        for qtype, group in data["by_query_type"].items():
            a = group["metrics"]
            lines += [
                f"  {qtype} ({group['query_count']} queries)",
                f"    P@5={pct(a['precision@5'])}  R@5={pct(a['recall@5'])}  "
                f"PrimaryR@5={pct(a['primary_recall@5'])}  "
                f"SupportR@5={pct(a['supporting_recall@5'])}  "
                f"AnchorCov@5={pct(a['anchor_coverage@5'])}",
            ]

    lines += ["", "QUERY-LEVEL COMPARISON", "-" * 110]
    query_ids = sorted({q["query_id"] for d in evaluation["experiments"].values() for q in d["queries"]})
    methods = list(evaluation["experiments"])
    for qid in query_ids:
        rows = []
        query_text = None
        for method in methods:
            item = next(q for q in evaluation["experiments"][method]["queries"] if q["query_id"] == qid)
            query_text = item["query"]
            m = item["metrics"]
            rows.append(
                f"{method}: P5={pct(m['precision@5'])}, R5={pct(m['recall@5'])}, "
                f"PR5={pct(m['primary_recall@5'])}, SR5={pct(m['supporting_recall@5'])}, "
                f"AC5={pct(m['anchor_coverage@5'])}"
            )
        lines.append(f"\n{qid}: {query_text}")
        lines.extend("  " + row for row in rows)

    return "\n".join(lines)


def main() -> None:
    print("=" * 110)
    print("RANKING ABLATION: MMR vs QUERY-AWARE vs COMPLEMENTARITY vs COVERAGE-AWARE")
    print("=" * 110)
    print(f"\nThreshold={THRESHOLD} | MMR λ={MMR_LAMBDA} | Candidate pool={CANDIDATE_POOL_SIZE}")

    gold_queries = base.load_gold_dataset()
    chunks = base.load_repository_chunks()

    validation = base.validate_gold_against_chunks(gold_queries, chunks)
    print(
        f"\nGold anchors: {validation['gold_anchor_count']} | "
        f"Resolved: {validation['resolved_anchor_count']} | "
        f"Missing: {validation['missing_anchor_count']}"
    )
    if validation["missing_anchor_count"]:
        raise RuntimeError("Gold alignment is incomplete; aborting controlled experiment.")

    methods = [
        "A_current_mmr",
        "B_query_aware_no_mmr",
        "C_complementarity",
        "D_coverage_aware",
    ]
    experiments: Dict[str, Any] = {}

    for method in methods:
        print(f"\n{'-' * 110}\n{method}\n{'-' * 110}")
        queries = []
        for index, gold in enumerate(gold_queries, 1):
            result = evaluate_query(gold, chunks, method)
            queries.append(result)
            m = result["metrics"]
            print(
                f"[{index:02d}] {gold['_query'][:70]:70s} "
                f"P5={pct(m['precision@5']):>8} "
                f"R5={pct(m['recall@5']):>8} "
                f"PR5={pct(m['primary_recall@5']):>8} "
                f"SR5={pct(m['supporting_recall@5']):>8} "
                f"AC5={pct(m['anchor_coverage@5']):>8}"
            )

        experiments[method] = {
            "aggregate": aggregate(queries),
            "by_query_type": aggregate_by_query_type(queries),
            "queries": queries,
        }

    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "repository": REPOSITORY_URL,
        "gold_dataset": str(GOLD_DATASET),
        "configuration": {
            "methods": methods,
            "top_k": TOP_K,
            "threshold": THRESHOLD,
            "mmr_lambda": MMR_LAMBDA,
            "coverage_selection": {
                "relevance_floor": 0.72,
                "relevance_weight": 0.72,
                "query_gain_weight": 0.08,
                "section_gain_weight": 0.08,
                "parent_gain_weight": 0.04,
                "content_gain_weight": 0.08,
                "redundancy_weight": 0.16,
            },
            "semantic_weight": SEMANTIC_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "rrf_k": RRF_K,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "metadata_weight": base.METADATA_WEIGHT,
            "lexical_weight": base.LEXICAL_WEIGHT,
            "anchor_match_min_score": ANCHOR_MATCH_MIN_SCORE,
            "gold_alignment": "path + human-readable anchor resolved against current chunks",
            "candidate_pool_control": "raw Hybrid/RRF candidates; no MMR or relevance filter before the shared ranking stage",
        },
        "gold_query_count": len(gold_queries),
        "chunk_count": len(chunks),
        "gold_validation": validation,
        "metric_definitions": {
            "primary_recall@5": "relevance=3 gold chunks retrieved / all relevance=3 gold chunks",
            "supporting_recall@5": "relevance=2 gold chunks retrieved / all relevance=2 gold chunks",
            "anchor_coverage@5": "distinct gold anchors represented in top 5 / distinct gold anchors for query",
        },
        "experiments": experiments,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"ranking_ablation_evaluation_{timestamp}.json"
    txt_path = OUTPUT_DIR / f"ranking_ablation_evaluation_{timestamp}.txt"

    json_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(report(evaluation), encoding="utf-8")

    print("\n" + "=" * 110)
    print("EXPERIMENT COMPLETE")
    print("=" * 110)
    print(f"\nJSON:   {json_path}")
    print(f"REPORT: {txt_path}")

    print("\nFINAL SUMMARY")
    print("-" * 110)
    for method, data in experiments.items():
        a = data["aggregate"]
        print(
            f"{method:28s} "
            f"P@5={pct(a['precision@5']):>8} "
            f"R@5={pct(a['recall@5']):>8} "
            f"PrimaryR={pct(a['primary_recall@5']):>8} "
            f"SupportR={pct(a['supporting_recall@5']):>8} "
            f"AnchorCov={pct(a['anchor_coverage@5']):>8} "
            f"NDCG={a['ndcg@5']:.4f} "
            f"MRR={a['mrr']:.4f}"
        )


if __name__ == "__main__":
    main()
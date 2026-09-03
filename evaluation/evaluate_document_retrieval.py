from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_chunks(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("chunks", [])
    if not isinstance(data, list):
        raise ValueError("chunks.json must be a list or {'chunks': [...]}")
    out = []
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item.setdefault("chunk_index", i)
        item["content"] = str(item.get("content") or item.get("text") or "")
        out.append(item)
    return out


def page(item: Dict[str, Any]):
    try:
        return int(item.get("page"))
    except (TypeError, ValueError):
        return None


def rr(retrieved, gold):
    for rank, item in enumerate(retrieved, 1):
        if page(item) in gold:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved, gold, k):
    if not gold:
        return 0.0
    found = {page(x) for x in retrieved[:k] if page(x) in gold}
    return len(found) / len(gold)


def precision_at_k(retrieved, gold, k):
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(page(x) in gold for x in top) / len(top)


def ndcg_at_k(retrieved, gold, k):
    top = retrieved[:k]
    dcg = sum(
        (1 if page(x) in gold else 0) / math.log2(rank + 1)
        for rank, x in enumerate(top, 1)
    )
    ideal_n = min(k, len(gold))
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def instantiate_retriever():
    module = importlib.import_module(
        "src.services.document_rag.document_retriever"
    )
    cls = getattr(module, "DocumentRetriever")
    try:
        return cls()
    except TypeError as exc:
        raise RuntimeError(
            "DocumentRetriever() needs constructor arguments in your current "
            "version. Edit instantiate_retriever() with the same constructor "
            "used by your backend."
        ) from exc


def classify(question):
    module = importlib.import_module(
        "src.services.document_rag.query_classifier"
    )
    classifier = getattr(module, "QueryClassifier")
    result = classifier.classify(question, document_count=1)
    return str(result.get("query_type", "unknown"))


def run_retrieval(chunks, questions, top_k):
    retriever = instantiate_retriever()
    signature = inspect.signature(retriever.retrieve)
    accepts_query_type = "query_type" in signature.parameters

    results = []
    for q in questions:
        expected = str(q["query_type"])
        actual = classify(q["question"])

        kwargs = {
            "question": q["question"],
            "chunks": chunks,
            "top_k": top_k,
        }

        # Current improved retriever supports query_type. This also keeps the
        # evaluator compatible with an older local version.
        if accepts_query_type:
            kwargs["query_type"] = expected

        retrieved = retriever.retrieve(**kwargs)
        if retrieved is None:
            retrieved = []

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "expected_query_type": expected,
            "actual_query_type": actual,
            "retrieved": list(retrieved),
        })
    return results


def evaluate(dataset, results):
    qmap = {q["id"]: q for q in dataset["questions"]}
    rows = []

    for result in results:
        q = qmap[result["question_id"]]
        retrieved = result.get("retrieved") or []
        gold = {int(x) for x in q["gold_pages"]}

        first_rank = next(
            (r for r, item in enumerate(retrieved, 1) if page(item) in gold),
            None,
        )

        row = {
            "id": q["id"],
            "question": q["question"],
            "expected_query_type": q["query_type"],
            "actual_query_type": result.get("actual_query_type"),
            "classification_correct": result.get("actual_query_type") == q["query_type"],
            "gold_pages": sorted(gold),
            "retrieved_pages": [page(x) for x in retrieved[:10]],
            "retrieved_chunk_indices": [x.get("chunk_index") for x in retrieved[:10]],
            "first_relevant_rank": first_rank,
            "mrr": rr(retrieved, gold),
        }

        for k in (1, 3, 5, 10):
            row[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
            row[f"precision@{k}"] = precision_at_k(retrieved, gold, k)
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)

        rows.append(row)

    overall = {
        "questions": len(rows),
        "mrr": statistics.mean(r["mrr"] for r in rows) if rows else 0,
        "classification_accuracy": statistics.mean(
            r["classification_correct"] for r in rows
        ) if rows else 0,
    }
    for k in (1, 3, 5, 10):
        overall[f"recall@{k}"] = statistics.mean(
            r[f"recall@{k}"] for r in rows
        ) if rows else 0
        overall[f"precision@{k}"] = statistics.mean(
            r[f"precision@{k}"] for r in rows
        ) if rows else 0
        overall[f"ndcg@{k}"] = statistics.mean(
            r[f"ndcg@{k}"] for r in rows
        ) if rows else 0

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["expected_query_type"]].append(r)

    by_type = {}
    for qtype, group in sorted(grouped.items()):
        by_type[qtype] = {
            "questions": len(group),
            "recall@5": statistics.mean(r["recall@5"] for r in group),
            "recall@10": statistics.mean(r["recall@10"] for r in group),
            "precision@5": statistics.mean(r["precision@5"] for r in group),
            "mrr": statistics.mean(r["mrr"] for r in group),
        }

    return {
        "document": dataset.get("document", {}),
        "overall": overall,
        "by_query_type": by_type,
        "zero_recall@10": [r["id"] for r in rows if r["recall@10"] == 0],
        "worst_recall@5": [
            r["id"] for r in sorted(rows, key=lambda x: (x["recall@5"], x["mrr"]))[:10]
        ],
        "rows": rows,
    }


def print_report(report):
    print("\n" + "=" * 70)
    print("SMART RESEARCH DASHBOARD — DOCUMENT RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\nOVERALL")
    for k, v in report["overall"].items():
        print(f"{k:28s}: {v:.4f}" if isinstance(v, float) else f"{k:28s}: {v}")

    print("\nBY QUERY TYPE")
    print("-" * 70)
    for qtype, m in report["by_query_type"].items():
        print(
            f"{qtype:14s} n={m['questions']:2d}  "
            f"R@5={m['recall@5']:.3f}  "
            f"R@10={m['recall@10']:.3f}  "
            f"P@5={m['precision@5']:.3f}  "
            f"MRR={m['mrr']:.3f}"
        )

    print("\nZERO RECALL@10")
    print(", ".join(report["zero_recall@10"]) or "None")

    print("\nWORST RECALL@5")
    print(", ".join(report["worst_recall@5"]) or "None")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        default=Path(__file__).with_name("document_retrieval_eval_dataset.json"),
        type=Path,
    )
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument(
        "--output",
        default=Path("retrieval_eval_results.json"),
        type=Path,
    )
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    chunks = normalize_chunks(load_json(args.chunks))

    results = run_retrieval(
        chunks=chunks,
        questions=dataset["questions"],
        top_k=args.top_k,
    )

    report = evaluate(dataset, results)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_report(report)
    print(f"\nDetailed results: {args.output.resolve()}")


if __name__ == "__main__":
    main()

import sys
from typing import Dict, List, Any

import numpy as np

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


# ============================================================
# CONFIGURATION
# ============================================================

REPOSITORY_URL = (
    "https://github.com/scikit-learn/scikit-learn.git"
)

TOP_K = 5

CANDIDATE_MULTIPLIER = 4

# MMR configurations.
#
# 1.0 -> relevance only
# 0.9 -> mostly relevance
# 0.7 -> balanced
# 0.5 -> stronger diversity

MMR_LAMBDAS = [
    1.0,
    0.9,
    0.7,
    0.5,
]

# Use:
#
#     python -m tests.test_hybrid_retrieval
#
# or:
#
#     python -m tests.test_hybrid_retrieval --verbose
#
VERBOSE = "--verbose" in sys.argv


# ============================================================
# TEST QUERIES
# ============================================================

QUERIES = [
    "linear regression",

    "How does random forest classification work?",

    "standardize features before training",

    "train_test_split",

    "logistic regression implementation",
]


# ============================================================
# REPOSITORY PIPELINE
# ============================================================

def load_test_chunks():
    """
    Discover the repository, acquire its content,
    and create structure-aware chunks.
    """

    print("\nDiscovering repository...")

    files = GitHubRepositoryIndexer.discover(
        REPOSITORY_URL
    )

    print(
        f"Files discovered: {len(files)}"
    )

    print(
        "\nAcquiring repository content..."
    )

    documents = GitHubContentAcquirer.acquire(
        files
    )

    print(
        f"Documents acquired: {len(documents)}"
    )

    print(
        "\nCreating semantic chunks..."
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    return chunks


# ============================================================
# CHUNK STATISTICS
# ============================================================

def print_chunk_statistics(
    chunks: List[Dict[str, Any]],
):
    """
    Display basic information about generated chunks.
    """

    if not chunks:
        print(
            "\nNo chunks were generated."
        )
        return

    print("\n")
    print("=" * 100)
    print("CHUNK INFORMATION")
    print("=" * 100)

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    sample = chunks[0]

    print("\nAvailable fields:")

    for key in sample.keys():
        print(
            f"  - {key}"
        )

    print("\nSample chunk metadata:")

    metadata_fields = [
        "path",
        "category",
        "section",
        "parent_section",
        "section_path",
        "chunk_index",
        "chunk_type",
        "language",
        "char_count",
    ]

    for field in metadata_fields:

        if field in sample:

            print(
                f"  {field}: "
                f"{sample.get(field)}"
            )


# ============================================================
# RESULT HELPERS
# ============================================================

def get_result_id(
    result: Dict[str, Any],
) -> str:
    """
    Generate a stable identifier for a retrieved chunk.
    """

    path = str(
        result.get(
            "path",
            "",
        )
    )

    chunk_index = result.get(
        "chunk_index",
        None,
    )

    if path or chunk_index is not None:

        return (
            f"{path}|"
            f"{chunk_index}"
        )

    content = str(
        result.get(
            "content",
            "",
        )
    )

    return content


def shorten_path(
    path: str,
    max_length: int = 65,
) -> str:
    """
    Make long repository paths easier to read
    in terminal output.
    """

    if len(path) <= max_length:
        return path

    return (
        "..."
        + path[
            -(max_length - 3):
        ]
    )


def get_content_preview(
    result: Dict[str, Any],
    max_length: int = 120,
) -> str:
    """
    Return a short one-line preview of a chunk.
    """

    content = str(
        result.get(
            "content",
            "",
        )
    )

    content = (
        content
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )

    if len(content) > max_length:

        content = (
            content[:max_length]
            + "..."
        )

    return content


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def embed_result(result):
    """
    Generate an embedding for a retrieved result using
    the exact same SentenceTransformer model used by
    SimpleRetriever.
    """

    model = SimpleRetriever._get_model()

    content = str(
        result.get(
            "content",
            "",
        )
    )

    embedding = model.encode(
        content,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embedding,
        dtype=np.float32,
    )

def cosine_similarity(
    embedding_a,
    embedding_b,
) -> float:
    """
    Calculate cosine similarity between two embeddings.
    """

    a = np.asarray(
        embedding_a,
        dtype=np.float32,
    )

    b = np.asarray(
        embedding_b,
        dtype=np.float32,
    )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:

        return 0.0

    return float(
        np.dot(a, b)
        / (
            norm_a
            * norm_b
        )
    )


# ============================================================
# REDUNDANCY MEASUREMENT
# ============================================================

def calculate_redundancy(
    results: List[Dict[str, Any]],
) -> float:
    """
    Calculate average pairwise semantic similarity
    between retrieved chunks.

    Higher value:
        More redundant results.

    Lower value:
        More diverse results.

    This is useful for visually demonstrating the
    effect of MMR.
    """

    if len(results) < 2:

        return 0.0

    embeddings = []

    for result in results:

        embedding = embed_result(
            result
        )

        embeddings.append(
            embedding
        )

    similarities = []

    for i in range(
        len(embeddings)
    ):

        for j in range(
            i + 1,
            len(embeddings),
        ):

            similarity = (
                cosine_similarity(
                    embeddings[i],
                    embeddings[j],
                )
            )

            similarities.append(
                similarity
            )

    if not similarities:

        return 0.0

    return float(
        np.mean(
            similarities
        )
    )


# ============================================================
# RANKING DISPLAY
# ============================================================

def print_configuration_results(
    results_by_lambda: Dict[
        float,
        List[Dict[str, Any]],
    ],
):
    """
    Display each result position across
    different MMR lambda values.
    """

    print("\n")
    print("-" * 100)
    print("RANKING COMPARISON")
    print("-" * 100)

    for rank in range(
        TOP_K
    ):

        print(
            f"\nPOSITION {rank + 1}"
        )

        print(
            "-" * 100
        )

        for mmr_lambda in MMR_LAMBDAS:

            results = results_by_lambda[
                mmr_lambda
            ]

            if rank >= len(results):

                continue

            result = results[
                rank
            ]

            path = shorten_path(
                str(
                    result.get(
                        "path",
                        "",
                    )
                )
            )

            hybrid_score = float(
                result.get(
                    "hybrid_score",
                    0.0,
                )
            )

            mmr_score = result.get(
                "mmr_score",
                None,
            )

            similarity = result.get(
                "similarity",
                None,
            )

            semantic_rank = result.get(
                "semantic_rank",
                None,
            )

            bm25_rank = result.get(
                "bm25_rank",
                None,
            )

            mmr_text = "-"

            if mmr_score is not None:

                mmr_text = (
                    f"{float(mmr_score):.6f}"
                )

            similarity_text = "-"

            if similarity is not None:

                similarity_text = (
                    f"{float(similarity):.4f}"
                )

            print(
                f"λ={mmr_lambda:.1f} | "
                f"Hybrid={hybrid_score:.6f} | "
                f"MMR={mmr_text:>10} | "
                f"Semantic={similarity_text:>7} | "
                f"SemRank={str(semantic_rank):>3} | "
                f"BM25Rank={str(bm25_rank):>3} | "
                f"{path}"
            )


# ============================================================
# DETAILED RESULTS
# ============================================================

def print_detailed_results(
    results: List[Dict[str, Any]],
    mmr_lambda: float,
):
    """
    Print detailed information for one MMR configuration.
    """

    print("\n")
    print("-" * 100)

    print(
        f"DETAILED RESULTS — MMR λ={mmr_lambda:.1f}"
    )

    print("-" * 100)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n[{rank}] "
            f"{result.get('path', '')}"
        )

        print(
            f"    Chunk index  : "
            f"{result.get('chunk_index', '-')}"
        )

        print(
            f"    Chunk type   : "
            f"{result.get('chunk_type', '-')}"
        )

        print(
            f"    Category     : "
            f"{result.get('category', '-')}"
        )

        print(
            f"    Section      : "
            f"{result.get('section', '-')}"
        )

        semantic_rank = result.get(
            "semantic_rank",
            None,
        )

        print(
            f"    Semantic rank: "
            f"{semantic_rank}"
        )

        bm25_rank = result.get(
            "bm25_rank",
            None,
        )

        print(
            f"    BM25 rank    : "
            f"{bm25_rank}"
        )

        hybrid_score = result.get(
            "hybrid_score",
            None,
        )

        if hybrid_score is not None:

            print(
                f"    Hybrid score : "
                f"{float(hybrid_score):.6f}"
            )

        similarity = result.get(
            "similarity",
            None,
        )

        if similarity is not None:

            print(
                f"    Semantic sim : "
                f"{float(similarity):.4f}"
            )

        mmr_score = result.get(
            "mmr_score",
            None,
        )

        if mmr_score is not None:

            print(
                f"    MMR score    : "
                f"{float(mmr_score):.6f}"
            )

        preview = get_content_preview(
            result,
            180,
        )

        print(
            f"    Preview      : "
            f"{preview}"
        )

        if VERBOSE and rank == 1:

            content = str(
                result.get(
                    "content",
                    "",
                )
            )

            print(
                "\n    FULL CONTENT:"
            )

            print(
                "    "
                + "-" * 85
            )

            for line in content.splitlines():

                print(
                    "    "
                    + line
                )

            print(
                "    "
                + "-" * 85
            )


# ============================================================
# RANK MOVEMENT
# ============================================================

def print_rank_movement(
    baseline: List[Dict[str, Any]],
    mmr_results: List[Dict[str, Any]],
):
    """
    Compare relevance-only results against MMR results.
    """

    print("\n")
    print("-" * 100)
    print("RANK MOVEMENT — λ=1.0 → λ=0.7")
    print("-" * 100)

    baseline_positions = {}

    for position, result in enumerate(
        baseline,
        start=1,
    ):

        baseline_positions[
            get_result_id(result)
        ] = position

    changed = 0

    for new_position, result in enumerate(
        mmr_results,
        start=1,
    ):

        result_id = get_result_id(
            result
        )

        old_position = (
            baseline_positions.get(
                result_id
            )
        )

        path = shorten_path(
            str(
                result.get(
                    "path",
                    "",
                )
            )
        )

        if old_position is None:

            print(
                f"  NEW     → "
                f"#{new_position} | "
                f"{path}"
            )

            changed += 1

        elif old_position != new_position:

            movement = (
                old_position
                - new_position
            )

            direction = (
                "↑"
                if movement > 0
                else "↓"
            )

            print(
                f"  {direction} "
                f"#{old_position} → "
                f"#{new_position} | "
                f"{path}"
            )

            changed += 1

        else:

            print(
                f"  =       "
                f"#{new_position} | "
                f"{path}"
            )

    print(
        "\nPositions changed: "
        f"{changed}/{TOP_K}"
    )


# ============================================================
# RESULT OVERLAP
# ============================================================

def print_result_overlap(
    baseline: List[Dict[str, Any]],
    mmr_results: List[Dict[str, Any]],
):
    """
    Show which chunks MMR introduced and removed.
    """

    baseline_ids = {
        get_result_id(result)
        for result in baseline
    }

    mmr_ids = {
        get_result_id(result)
        for result in mmr_results
    }

    added = (
        mmr_ids
        - baseline_ids
    )

    removed = (
        baseline_ids
        - mmr_ids
    )

    print("\n")
    print("-" * 100)
    print("RESULT SET CHANGES")
    print("-" * 100)

    print(
        f"\nNew chunks introduced by MMR: "
        f"{len(added)}"
    )

    for result in mmr_results:

        if get_result_id(result) in added:

            print(
                f"  + "
                f"{shorten_path(str(result.get('path', '')))}"
            )

    print(
        f"\nChunks removed by MMR: "
        f"{len(removed)}"
    )

    for result in baseline:

        if get_result_id(result) in removed:

            print(
                f"  - "
                f"{shorten_path(str(result.get('path', '')))}"
            )


# ============================================================
# REDUNDANCY COMPARISON
# ============================================================

def print_redundancy_comparison(
    results_by_lambda: Dict[
        float,
        List[Dict[str, Any]],
    ],
):
    """
    Compare semantic redundancy across
    MMR lambda configurations.
    """

    print("\n")
    print("-" * 100)
    print("SEMANTIC REDUNDANCY")
    print("-" * 100)

    redundancy_by_lambda = {}

    for mmr_lambda in MMR_LAMBDAS:

        results = results_by_lambda[
            mmr_lambda
        ]

        redundancy = calculate_redundancy(
            results
        )

        redundancy_by_lambda[
            mmr_lambda
        ] = redundancy

        print(
            f"\nλ={mmr_lambda:.1f}"
        )

        print(
            f"    Average pairwise similarity: "
            f"{redundancy:.4f}"
        )

    # --------------------------------------------------------
    # Compare baseline vs balanced MMR
    # --------------------------------------------------------

    baseline_redundancy = (
        redundancy_by_lambda[
            1.0
        ]
    )

    balanced_redundancy = (
        redundancy_by_lambda[
            0.7
        ]
    )

    difference = (
        baseline_redundancy
        - balanced_redundancy
    )

    print("\n")
    print(
        "λ=1.0 vs λ=0.7"
    )

    print(
        f"    Relevance-only : "
        f"{baseline_redundancy:.4f}"
    )

    print(
        f"    MMR λ=0.7     : "
        f"{balanced_redundancy:.4f}"
    )

    if difference > 0:

        percentage = (
            (
                difference
                / baseline_redundancy
            )
            * 100
            if baseline_redundancy != 0
            else 0
        )

        print(
            f"\n    ✓ MMR reduced "
            f"semantic redundancy."
        )

        print(
            f"    ✓ Reduction: "
            f"{difference:.4f}"
            f" "
            f"({percentage:.2f}%)"
        )

    elif difference < 0:

        print(
            "\n    ! MMR produced "
            "higher average similarity "
            "for this query."
        )

    else:

        print(
            "\n    = No change in "
            "average semantic redundancy."
        )


# ============================================================
# LAMBDA SUMMARY
# ============================================================

def print_lambda_summary(
    results_by_lambda: Dict[
        float,
        List[Dict[str, Any]],
    ],
):
    """
    Print a concise summary of the results
    for every lambda.
    """

    print("\n")
    print("-" * 100)
    print("MMR LAMBDA SUMMARY")
    print("-" * 100)

    for mmr_lambda in MMR_LAMBDAS:

        results = results_by_lambda[
            mmr_lambda
        ]

        print(
            f"\nλ={mmr_lambda:.1f}"
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            path = shorten_path(
                str(
                    result.get(
                        "path",
                        "",
                    )
                ),
                70,
            )

            mmr_score = result.get(
                "mmr_score",
                None,
            )

            if mmr_score is None:

                mmr_text = "-"

            else:

                mmr_text = (
                    f"{float(mmr_score):.6f}"
                )

            print(
                f"    {rank}. "
                f"{path}"
                f" | MMR={mmr_text}"
            )


# ============================================================
# RUN ONE QUERY
# ============================================================

def run_mmr_experiment(
    chunks: List[Dict[str, Any]],
    query: str,
):
    """
    Run the complete MMR experiment for one query.
    """

    print("\n")
    print("=" * 100)

    print(
        f"QUERY: {query}"
    )

    print("=" * 100)

    results_by_lambda = {}

    # ========================================================
    # Execute retrieval for every lambda
    # ========================================================

    for mmr_lambda in MMR_LAMBDAS:

        print(
            f"\nRunning "
            f"λ={mmr_lambda:.1f}..."
        )

        retriever = HybridRetriever(
            semantic_weight=0.5,
            bm25_weight=0.5,
            rrf_k=60,
            candidate_multiplier=CANDIDATE_MULTIPLIER,
            mmr_lambda=mmr_lambda,
        )

        results = retriever.retrieve(
            question=query,
            chunks=chunks,
            top_k=TOP_K,
        )

        results_by_lambda[
            mmr_lambda
        ] = results

    # ========================================================
    # 1. Ranking comparison
    # ========================================================

    print_configuration_results(
        results_by_lambda
    )

    # ========================================================
    # 2. Detailed balanced results
    # ========================================================

    print_detailed_results(
        results_by_lambda[0.7],
        0.7,
    )

    # ========================================================
    # 3. Rank movement
    # ========================================================

    print_rank_movement(
        baseline=results_by_lambda[1.0],
        mmr_results=results_by_lambda[0.7],
    )

    # ========================================================
    # 4. Result set changes
    # ========================================================

    print_result_overlap(
        baseline=results_by_lambda[1.0],
        mmr_results=results_by_lambda[0.7],
    )

    # ========================================================
    # 5. Redundancy
    # ========================================================

    print_redundancy_comparison(
        results_by_lambda
    )

    # ========================================================
    # 6. Lambda summary
    # ========================================================

    print_lambda_summary(
        results_by_lambda
    )

    return results_by_lambda


# ============================================================
# TEST HYBRID RETRIEVAL
# ============================================================

def test_hybrid_retrieval(
    chunks: List[Dict[str, Any]],
):
    """
    Run the RRF + MMR experiment across all queries.
    """

    print("\n")
    print("=" * 100)
    print("RRF + MMR RETRIEVAL EXPERIMENT")
    print("=" * 100)

    print("\nConfiguration:")

    print(
        f"  Semantic weight      : 0.5"
    )

    print(
        f"  BM25 weight          : 0.5"
    )

    print(
        f"  RRF k                : 60"
    )

    print(
        f"  Candidate multiplier : "
        f"{CANDIDATE_MULTIPLIER}"
    )

    print(
        f"  Candidate pool size  : "
        f"up to {TOP_K * CANDIDATE_MULTIPLIER}"
    )

    print(
        f"  Final top_k          : "
        f"{TOP_K}"
    )

    print(
        "\nMMR configurations:"
    )

    print(
        "  λ=1.0 → relevance only"
    )

    print(
        "  λ=0.9 → mostly relevance"
    )

    print(
        "  λ=0.7 → balanced"
    )

    print(
        "  λ=0.5 → stronger diversity"
    )

    # ========================================================
    # Run every query
    # ========================================================

    for query_number, query in enumerate(
        QUERIES,
        start=1,
    ):

        print("\n")
        print(
            f"QUERY {query_number}/"
            f"{len(QUERIES)}"
        )

        run_mmr_experiment(
            chunks=chunks,
            query=query,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("RRF + MMR HYBRID RETRIEVAL TEST")
    print("=" * 100)

    if VERBOSE:

        print(
            "\nOutput mode: VERBOSE"
        )

        print(
            "Full content will be shown "
            "for the first balanced result."
        )

    else:

        print(
            "\nOutput mode: COMPACT"
        )

        print(
            "Use --verbose to show "
            "full content for the first "
            "balanced result."
        )

    # ========================================================
    # Step 1: Repository → documents → chunks
    # ========================================================

    print(
        "\n[1/2] Loading repository..."
    )

    chunks = load_test_chunks()

    if not chunks:

        print(
            "\nERROR: No chunks were generated."
        )

        return

    # ========================================================
    # Chunk statistics
    # ========================================================

    print_chunk_statistics(
        chunks
    )

    # ========================================================
    # Step 2: RRF + MMR experiment
    # ========================================================

    print(
        "\n[2/2] Running retrieval experiments..."
    )

    test_hybrid_retrieval(
        chunks
    )

    # ========================================================
    # Complete
    # ========================================================

    print("\n")
    print("=" * 100)

    print(
        "RRF + MMR RETRIEVAL TEST COMPLETE"
    )

    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
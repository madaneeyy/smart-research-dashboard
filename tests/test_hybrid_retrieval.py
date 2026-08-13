import sys
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


REPOSITORY_URL = (
    "https://github.com/scikit-learn/scikit-learn.git"
)

VERBOSE = "--verbose" in sys.argv


# ============================================================
# Configuration
# ============================================================

QUERIES = [
    "linear regression",
    "How does random forest classification work?",
    "standardize features before training",
    "train_test_split",
    "logistic regression implementation",
]

TOP_K = 5


# ============================================================
# Repository pipeline
# ============================================================

def load_test_chunks():
    """
    Run the repository -> documents -> chunks pipeline.

    Returns
    -------
    list[dict]
        Structure-aware chunks produced by DocumentChunker.
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

    # DocumentChunker.chunk_documents()
    # is an instance method.
    chunker = DocumentChunker()

    chunks = chunker.chunk_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    return chunks


# ============================================================
# Chunk inspection
# ============================================================

def print_chunk_statistics(chunks):
    """
    Print useful information about the generated chunks.
    """

    if not chunks:
        print(
            "\nNo chunks were created."
        )
        return

    print("\n" + "-" * 70)
    print("CHUNK STRUCTURE")
    print("-" * 70)

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    sample = chunks[0]

    print("\nAvailable fields:")

    for key in sample.keys():
        print(
            f"  - {key}"
        )

    print(
        "\nSample chunk metadata:"
    )

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
                f"{field}: "
                f"{sample.get(field)}"
            )


# ============================================================
# Hybrid retrieval testing
# ============================================================

def test_hybrid_retrieval(chunks):
    """
    Run hybrid retrieval tests.

    Normal mode:
        Prints compact previews for all results.

    --verbose mode:
        Prints the full content only for the
        top-ranked result of each query.
    """

    print("\n")
    print("=" * 70)
    print("INITIALIZING HYBRID RETRIEVER")
    print("=" * 70)

    retriever = HybridRetriever(
        semantic_weight=0.5,
        bm25_weight=0.5,
        rrf_k=60,
    )

    print(
        "\nHybrid configuration:"
    )

    print(
        "  Semantic weight:",
        retriever.semantic_weight,
    )

    print(
        "  BM25 weight:",
        retriever.bm25_weight,
    )

    print(
        "  RRF k:",
        retriever.rrf_k,
    )

    # ========================================================
    # Run queries
    # ========================================================

    for query in QUERIES:

        print("\n")
        print("=" * 70)
        print(
            f"QUERY: {query}"
        )
        print("=" * 70)

        results = retriever.retrieve(
            question=query,
            chunks=chunks,
            top_k=TOP_K,
        )

        if not results:

            print(
                "\nNo results found."
            )

            continue

        # ====================================================
        # Compact ranking table
        # ====================================================

        print(
            "\n"
            f"{'Rank':<6}"
            f"{'Hybrid':<12}"
            f"{'SemRank':<9}"
            f"{'BM25Rank':<10}"
            f"{'Semantic':<11}"
            f"{'BM25':<11}"
            f"Path"
        )

        print(
            "-" * 110
        )

        for i, result in enumerate(
            results,
            start=1,
        ):

            hybrid_score = float(
                result.get(
                    "hybrid_score",
                    0.0,
                )
            )

            similarity = result.get(
                "similarity"
            )

            if similarity is not None:

                similarity = float(
                    similarity
                )

            else:

                similarity = 0.0

            bm25_score = result.get(
                "bm25_score"
            )

            if bm25_score is not None:

                bm25_score = float(
                    bm25_score
                )

            else:

                bm25_score = 0.0

            semantic_rank = result.get(
                "semantic_rank",
                "-",
            )

            bm25_rank = result.get(
                "bm25_rank",
                "-",
            )

            path = result.get(
                "path",
                "",
            )

            # Keep paths short enough
            # for terminal output.

            if len(path) > 55:

                path = (
                    "..."
                    + path[-52:]
                )

            print(
                f"{i:<6}"
                f"{hybrid_score:<12.6f}"
                f"{str(semantic_rank):<9}"
                f"{str(bm25_rank):<10}"
                f"{similarity:<11.4f}"
                f"{bm25_score:<11.4f}"
                f"{path}"
            )

        # ====================================================
        # Detailed result information
        # ====================================================

        print(
            "\nRESULT DETAILS"
        )

        print(
            "-" * 70
        )

        for i, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{i}] "
                f"{result.get('path', '')}"
            )

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            print(
                f"    Hybrid Score : "
                f"{float(result.get('hybrid_score', 0.0)):.6f}"
            )

            similarity = result.get(
                "similarity"
            )

            if similarity is not None:

                print(
                    f"    Semantic     : "
                    f"{float(similarity):.4f}"
                    f"  "
                    f"(rank "
                    f"{result.get('semantic_rank', '-')})"
                )

            bm25_score = result.get(
                "bm25_score"
            )

            if bm25_score is not None:

                print(
                    f"    BM25         : "
                    f"{float(bm25_score):.4f}"
                    f"  "
                    f"(rank "
                    f"{result.get('bm25_rank', '-')})"
                )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            print(
                f"    Category     : "
                f"{result.get('category', '')}"
            )

            print(
                f"    Section      : "
                f"{result.get('section', '')}"
            )

            print(
                f"    Chunk        : "
                f"{result.get('chunk_index', '')}"
                f" "
                f"({result.get('chunk_type', '')})"
            )

            print(
                f"    Language     : "
                f"{result.get('language', '')}"
            )

            if result.get("symbol"):

                print(
                    f"    Symbol       : "
                    f"{result.get('symbol')}"
                )

            # ------------------------------------------------
            # Content
            # ------------------------------------------------

            content = result.get(
                "content",
                "",
            )

            if (
                VERBOSE
                and i == 1
            ):

                # In verbose mode we only
                # print the COMPLETE content
                # of the top-ranked result.

                print(
                    "\n    FULL CONTENT:"
                )

                print(
                    "    "
                    + "-" * 60
                )

                formatted_content = (
                    content
                    .replace(
                        "\r\n",
                        "\n",
                    )
                    .replace(
                        "\r",
                        "\n",
                    )
                )

                for line in formatted_content.split(
                    "\n"
                ):

                    print(
                        "    "
                        + line
                    )

                print(
                    "    "
                    + "-" * 60
                )

            else:

                # Results 2-5 always receive
                # a compact preview.

                preview = (
                    content
                    .replace(
                        "\n",
                        " ",
                    )
                    .replace(
                        "\r",
                        " ",
                    )
                    .strip()
                )

                if len(preview) > 280:

                    preview = (
                        preview[:280]
                        + "..."
                    )

                print(
                    f"    Preview      : "
                    f"{preview}"
                )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    if VERBOSE:

        print(
            "\nOutput mode: VERBOSE"
        )

        print(
            "Full content will be shown "
            "only for the top result."
        )

    else:

        print(
            "\nOutput mode: COMPACT"
        )

        print(
            "Use --verbose to show "
            "the full top result."
        )

    # --------------------------------------------------------
    # 1. Load repository and create chunks
    # --------------------------------------------------------

    print(
        "\n[1/2] Loading repository and chunks..."
    )

    chunks = load_test_chunks()

    if not chunks:

        print(
            "\nERROR: No chunks were generated."
        )

        return

    # --------------------------------------------------------
    # Chunk statistics
    # --------------------------------------------------------

    print_chunk_statistics(
        chunks
    )

    # --------------------------------------------------------
    # 2. Test hybrid retrieval
    # --------------------------------------------------------

    print(
        "\n[2/2] Testing hybrid retrieval..."
    )

    test_hybrid_retrieval(
        chunks
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "HYBRID RETRIEVAL TEST COMPLETE"
    )
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
from src.services.github.repository_indexer import (
    GitHubRepositoryIndexer,
)

from src.services.github.content_acquirer import (
    GitHubContentAcquirer,
)

from src.services.rag.chunker import (
    DocumentChunker,
)

from src.services.rag.bm25_retriever import (
    BM25Retriever,
)


REPOSITORY_URL = (
    "https://github.com/scikit-learn/scikit-learn.git"
)


print("=" * 70)
print("BM25 RETRIEVAL TEST")
print("=" * 70)


# ============================================================
# 1. Discover repository
# ============================================================

print("\n[1/4] Discovering repository...")

files = GitHubRepositoryIndexer.discover(
    REPOSITORY_URL
)

print(
    f"Files discovered: {len(files)}"
)


# ============================================================
# 2. Acquire repository content
# ============================================================

print("\n[2/4] Acquiring repository content...")

documents = GitHubContentAcquirer.acquire(
    files
)

print(
    f"Documents acquired: {len(documents)}"
)


# ============================================================
# 3. Create semantic chunks
# ============================================================

print("\n[3/4] Creating chunks...")

chunker = DocumentChunker()

chunks = chunker.chunk_documents(
    documents
)

print(
    f"Chunks created: {len(chunks)}"
)


# ============================================================
# 4. Initialize BM25
# ============================================================

print("\n[4/4] Initializing BM25 retriever...")

retriever = BM25Retriever(
    chunks
)

print("BM25 retriever ready.")


# ============================================================
# Queries
# ============================================================

queries = [
    "linear regression",
    "How does random forest classification work?",
    "standardize features before training",
    "train_test_split",
    "logistic regression implementation",
]


# ============================================================
# Run BM25 retrieval
# ============================================================

for query in queries:

    print("\n")
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    if not results:

        print("\nNo results found.")

        continue


    for i, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- RESULT {i} ---"
        )

        # ----------------------------------------------------
        # BM25 score
        # ----------------------------------------------------

        score = result.get(
            "bm25_score",
            result.get(
                "score",
                0.0,
            ),
        )

        print(
            "BM25 Score:",
            round(
                score,
                4,
            ),
        )

        # ----------------------------------------------------
        # File information
        # ----------------------------------------------------

        print(
            "Path:",
            result.get(
                "path",
                "",
            ),
        )

        print(
            "Category:",
            result.get(
                "category",
                "",
            ),
        )

        print(
            "Section:",
            result.get(
                "section",
                "",
            ),
        )

        # ----------------------------------------------------
        # Chunk metadata
        # ----------------------------------------------------

        print(
            "Chunk index:",
            result.get(
                "chunk_index",
                "",
            ),
        )

        print(
            "Chunk type:",
            result.get(
                "chunk_type",
                "",
            ),
        )

        print(
            "Symbol:",
            result.get(
                "symbol",
                "",
            ),
        )

        print(
            "Language:",
            result.get(
                "language",
                "",
            ),
        )

        # ----------------------------------------------------
        # Content
        # ----------------------------------------------------

        content = result.get(
            "content",
            "",
        )

        print("\nContent:")

        print(
            content[:1200]
        )

        print(
            "\n" + "-" * 70
        )


print("\n")
print("=" * 70)
print("BM25 RETRIEVAL TEST COMPLETE")
print("=" * 70)
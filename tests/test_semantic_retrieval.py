from src.services.github.repository_indexer import (
    GitHubRepositoryIndexer,
)

from src.services.github.content_acquirer import (
    GitHubContentAcquirer,
)

from src.services.rag.retriever import (
    SimpleRetriever,
)


REPOSITORY_URL = (
    "https://github.com/scikit-learn/scikit-learn.git"
)


print("=" * 60)
print("SEMANTIC RETRIEVAL TEST")
print("=" * 60)


# --------------------------------------------------
# 1. Discover repository
# --------------------------------------------------

print("\nDiscovering repository...")

files = GitHubRepositoryIndexer.discover(
    REPOSITORY_URL
)

print(f"Files discovered: {len(files)}")


# --------------------------------------------------
# 2. Acquire content
# --------------------------------------------------

print("\nAcquiring repository content...")

documents = GitHubContentAcquirer.acquire(
    files
)

print(
    f"Documents acquired: {len(documents)}"
)


# --------------------------------------------------
# 3. Create chunks
# --------------------------------------------------

print("\nChunking documents...")

# IMPORTANT:
# Replace this import with your actual chunker.
from src.services.rag.chunker import (
    DocumentChunker,
)

chunks = DocumentChunker.chunk_documents(
    documents
)

print(f"Chunks created: {len(chunks)}")


# --------------------------------------------------
# 4. Test semantic retrieval
# --------------------------------------------------

queries = [
    "linear regression",
    "How does random forest classification work?",
    "standardize features before training",
    "train_test_split",
    "logistic regression implementation"
]


for query in queries:

    print("\n")
    print("=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    results = SimpleRetriever.retrieve(
        query,
        chunks,
        top_k=5,
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- RESULT {i} ---"
        )

        print(
            "Similarity:",
            round(
                result["similarity"],
                4,
            ),
        )

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

        print(
            "\n",
            result["content"][:800],
        )
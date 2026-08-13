import sys
from pathlib import Path

# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from src.services.github.repository_indexer import GitHubRepositoryIndexer
from src.services.github.content_acquirer import GitHubContentAcquirer
from src.services.rag.chunker import TextChunker


# --------------------------------------------------
# Retrieval test
# --------------------------------------------------

def test_retrieval(query, chunks, top_k=5):
    query_terms = set(query.lower().split())

    scored = []

    for chunk in chunks:
        content = chunk["content"].lower()

        score = sum(
            content.count(term)
            for term in query_terms
            if len(term) > 2
        )

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\nQuery: {query}")
    print(f"Matching chunks: {len(scored)}")

    for i, (score, chunk) in enumerate(scored[:top_k]):
        print(f"\n--- RESULT {i + 1} ---")
        print("Score:", score)
        print("Path:", chunk["path"])
        print("Category:", chunk["category"])
        print("Section:", chunk.get("section", ""))
        print("Words:", len(chunk["content"].split()))
        print("\n", chunk["content"][:1000])


# --------------------------------------------------
# Main test
# --------------------------------------------------

def main():

    print("Discovering repository...")

    files = GitHubRepositoryIndexer.discover(
        "https://github.com/NVIDIA/Megatron-LM"
    )

    print("Files discovered:", len(files))

    print("\nAcquiring repository content...")

    documents = GitHubContentAcquirer.acquire(files)

    print("Documents acquired:", len(documents))

    print("\nChunking documents...")

    chunks = TextChunker.chunk_documents(documents)

    print("Chunks created:", len(chunks))

    total_words = sum(
        len(chunk["content"].split())
        for chunk in chunks
    )

    print("Total chunk words:", total_words)

    # --------------------------------------------------
    # Retrieval experiments
    # --------------------------------------------------

    test_retrieval(
        "tensor parallelism",
        chunks,
        top_k=5,
    )

    test_retrieval(
        "pipeline parallelism",
        chunks,
        top_k=5,
    )

    test_retrieval(
        "distributed training",
        chunks,
        top_k=5,
    )


if __name__ == "__main__":
    main()
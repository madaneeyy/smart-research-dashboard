from src.services.research import ResearchService
from src.services.relevance import RelevanceScorer


def main():
    query = "transformer models"

    print("=" * 80)
    print("RESEARCH SEARCH SMOKE TEST")
    print("=" * 80)
    print(f"Query: {query!r}")
    print()

    service = ResearchService()

    results = service.search(
        query,
        arxiv_limit=5,
        github_limit=5,
        paperswithcode_limit=5,
        huggingface_limit=5,
    )

    if not results:
        print("No results returned.")
        return

    print(f"Results returned: {len(results)}")
    print()
    print("-" * 80)

    for index, item in enumerate(results, start=1):
        score = RelevanceScorer.score(query, item)

        print(
            f"{index:02d}. "
            f"[score={score:5.1f}] "
            f"[{item.source}] "
            f"{item.title}"
        )

        print(f"    ID:  {item.id}")
        print(f"    URL: {item.url}")
        print()

    print("-" * 80)

    scores = [
        RelevanceScorer.score(query, item)
        for item in results
    ]

    if scores == sorted(scores, reverse=True):
        print("PASS: Results are ordered by relevance score.")
    else:
        print("FAIL: Results are NOT ordered by relevance score.")

    print()

    print("Score order:")
    print(scores)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
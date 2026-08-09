from src.services.research import ResearchService


service = ResearchService()

results = service.search(
    "learning",
    sources=["paperswithcode"],
    paperswithcode_limit=5,
)

print(f"FOUND: {len(results)}")

for result in results:
    print(
        f"[{result.source}] "
        f"{result.title} | "
        f"{result.id} | "
        f"{result.url}"
    )
from src.collectors.arxiv import (
    research_paper_to_item,
    search_arxiv,
)
from src.collectors.github import (
    github_repository_to_item,
    search_github_repositories,
)
from src.collectors.huggingface import (
    huggingface_model_to_item,
    search_huggingface_models,
)
from src.collectors.paperswithcode import (
    paperswithcode_paper_to_item,
    search_paperswithcode_papers,
)
from src.models.research import ResearchItem


class ResearchService:
    def search(self, query: str) -> list[ResearchItem]:
        if not query.strip():
            raise ValueError("query must not be empty")

        results = []

        arxiv_papers = search_arxiv(
            search_query=query,
        )

        # ... rest stays the same
        results.extend(
            research_paper_to_item(paper)
            for paper in arxiv_papers
        )

        github_repositories = search_github_repositories(
            query=query,
        )

        results.extend(
            github_repository_to_item(repository)
            for repository in github_repositories
        )

        paperswithcode_papers = search_paperswithcode_papers(
    query=query,
)

        results.extend(
            paperswithcode_paper_to_item(paper)
            for paper in paperswithcode_papers
        )

        huggingface_models = search_huggingface_models(
            search=query,
        )

        results.extend(
            huggingface_model_to_item(model)
            for model in huggingface_models
        )

        return results
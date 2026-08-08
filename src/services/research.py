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
    def search(
        self,
        query: str,
        *,
        arxiv_limit: int = 20,
        github_limit: int = 20,
        paperswithcode_limit: int = 20,
        huggingface_limit: int = 20,
    ) -> list[ResearchItem]:

        if not query.strip():
            raise ValueError("query must not be empty")

        results: list[ResearchItem] = []

        # ArXiv
        try:
            arxiv_papers = search_arxiv(
                search_query=query,
                max_results=arxiv_limit,
            )

            results.extend(
                research_paper_to_item(paper)
                for paper in arxiv_papers
            )
        except Exception:
            pass

        # GitHub
        try:
            github_repositories = search_github_repositories(
                query=query,
                per_page=github_limit,
            )

            results.extend(
                github_repository_to_item(repository)
                for repository in github_repositories
            )
        except Exception:
            pass

        # PapersWithCode
        try:
            paperswithcode_papers = search_paperswithcode_papers(
                query=query,
                length=paperswithcode_limit,
            )

            results.extend(
                paperswithcode_paper_to_item(paper)
                for paper in paperswithcode_papers
            )
        except Exception:
            pass

        # Hugging Face
        try:
            huggingface_models = search_huggingface_models(
                search=query,
                limit=huggingface_limit,
            )

            results.extend(
                huggingface_model_to_item(model)
                for model in huggingface_models
            )
        except Exception:
            pass

        # Remove duplicates while preserving source priority.
        unique_results: list[ResearchItem] = []
        seen_ids: set[str] = set()

        for result in results:
            if result.id in seen_ids:
                continue

            seen_ids.add(result.id)
            unique_results.append(result)

        return unique_results
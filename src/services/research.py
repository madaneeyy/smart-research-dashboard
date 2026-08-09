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
from src.services.relevance import RelevanceScorer


class ResearchService:
    VALID_SOURCES = {
        "arxiv",
        "github",
        "paperswithcode",
        "huggingface",
    }

    def search(
        self,
        query: str,
        sources: list[str] | None = None,
        arxiv_limit: int = 20,
        github_limit: int = 20,
        paperswithcode_limit: int = 20,
        huggingface_limit: int = 20,
    ) -> list[ResearchItem]:
        """
        Search across the selected research sources.

        Results are:
        1. Collected from each selected source
        2. Converted into ResearchItem objects
        3. Deduplicated
        4. Ranked by relevance to the query
        """

        if not query.strip():
            raise ValueError("query must not be empty")

        # ---------------------------------------------------------
        # Determine selected sources
        # ---------------------------------------------------------
        if sources is None:
            selected_sources = self.VALID_SOURCES
        else:
            unknown_sources = set(sources) - self.VALID_SOURCES

            if unknown_sources:
                raise ValueError(
                    f"unknown source: {sorted(unknown_sources)[0]}"
                )

            selected_sources = set(sources)

        results: list[ResearchItem] = []

        # ---------------------------------------------------------
        # arXiv
        # ---------------------------------------------------------
        if "arxiv" in selected_sources:
            try:
                papers = search_arxiv(
                    search_query=query,
                    max_results=arxiv_limit,
                )

                results.extend(
                    research_paper_to_item(paper)
                    for paper in papers
                )

            except Exception:
                # One failing source should not break the
                # entire research search.
                pass

        # ---------------------------------------------------------
        # GitHub
        # ---------------------------------------------------------
        if "github" in selected_sources:
            try:
                repositories = search_github_repositories(
                    query=query,
                    per_page=github_limit,
                )

                results.extend(
                    github_repository_to_item(repository)
                    for repository in repositories
                )

            except Exception:
                pass

        # ---------------------------------------------------------
        # PapersWithCode
        # ---------------------------------------------------------
        if "paperswithcode" in selected_sources:
            try:
                papers = search_paperswithcode_papers(
                    query=query,
                    length=paperswithcode_limit,
                )

                results.extend(
                    paperswithcode_paper_to_item(paper)
                    for paper in papers
                )

            except Exception:
                pass

        # ---------------------------------------------------------
        # Hugging Face
        # ---------------------------------------------------------
        if "huggingface" in selected_sources:
            try:
                models = search_huggingface_models(
                    search=query,
                    limit=huggingface_limit,
                )

                results.extend(
                    huggingface_model_to_item(model)
                    for model in models
                )

            except Exception:
                pass

        # ---------------------------------------------------------
        # Deduplicate
        # ---------------------------------------------------------
        deduplicated_results = self._deduplicate_results(results)

        # ---------------------------------------------------------
        # Rank by relevance
        # ---------------------------------------------------------
        return RelevanceScorer.rank(
            query,
            deduplicated_results,
        )

    # =============================================================
    # DEDUPLICATION
    # =============================================================

    @staticmethod
    def _deduplicate_results(
        results: list[ResearchItem],
    ) -> list[ResearchItem]:
        """
        Remove duplicate research items while preserving the
        first occurrence.

        Deduplication uses:
        1. Cross-source research ID
        2. Normalized URL
        3. Source-specific ID
        """

        unique_results: list[ResearchItem] = []
        seen_keys: set[str] = set()

        for result in results:
            keys = ResearchService._deduplication_keys(result)

            # If any identity key has already been seen,
            # this result is considered a duplicate.
            if any(key in seen_keys for key in keys):
                continue

            unique_results.append(result)
            seen_keys.update(keys)

        return unique_results

    @staticmethod
    def _deduplication_keys(
        result: ResearchItem,
    ) -> set[str]:
        """
        Generate stable identity keys for a ResearchItem.

        Cross-source example:

            arXiv:
                id = "1234.5678"

            PapersWithCode:
                id = "1234.5678"

        Both become:

            research-id:1234.5678

        Therefore, if the arXiv result was already encountered,
        the PapersWithCode result will be treated as a duplicate.
        """

        keys: set[str] = set()

        # ---------------------------------------------------------
        # 1. Cross-source research ID
        # ---------------------------------------------------------
        raw_id = str(result.id).strip().lower()

        if raw_id:
            normalized_id = raw_id

            if normalized_id.startswith("arxiv-"):
                normalized_id = normalized_id.removeprefix("arxiv-")

            if normalized_id.startswith("arxiv:"):
                normalized_id = normalized_id.removeprefix("arxiv:")

            keys.add(
                f"research-id:{normalized_id}"
            )

        # ---------------------------------------------------------
        # 2. Normalized URL
        # ---------------------------------------------------------
        raw_url = str(result.url).strip().lower()

        if raw_url:
            normalized_url = raw_url.rstrip("/")

            keys.add(
                f"url:{normalized_url}"
            )

        # ---------------------------------------------------------
        # 3. Source-specific ID
        # ---------------------------------------------------------
        if raw_id and result.source:
            keys.add(
                f"source-id:"
                f"{result.source.strip().lower()}:"
                f"{raw_id}"
            )

        return keys
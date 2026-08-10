from datetime import datetime, timezone
import logging

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


logger = logging.getLogger(__name__)


class ResearchService:
    VALID_SOURCES = {
        "arxiv",
        "github",
        "paperswithcode",
        "huggingface",
    }

    VALID_SORT_OPTIONS = {
        "relevance",
        "published",
        "updated",
    }

    def search(
        self,
        query: str,
        sources: list[str] | None = None,
        arxiv_limit: int = 20,
        github_limit: int = 20,
        paperswithcode_limit: int = 20,
        huggingface_limit: int = 20,
        sort_by: str = "relevance",
    ) -> list[ResearchItem]:
        """
        Search across the selected research sources.

        Pipeline:

        1. Collect results from each selected source
        2. Normalize results into ResearchItem objects
        3. Deduplicate results
        4. Rank/sort results
        5. Return final results

        sort_by options:
            - "relevance": relevance score, highest first
            - "published": newest publication date first
            - "updated": most recently updated first
        """

        if not query.strip():
            raise ValueError("query must not be empty")

        # ---------------------------------------------------------
        # Validate sorting option
        # ---------------------------------------------------------
        if sort_by not in self.VALID_SORT_OPTIONS:
            raise ValueError(
                f"unknown sort option: {sort_by}"
            )

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
                logger.exception(
                    "ArXiv search failed for query=%r",
                    query,
                )

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
                logger.exception(
                    "GitHub search failed for query=%r",
                    query,
                )

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
                logger.exception(
                    "PapersWithCode search failed for query=%r",
                    query,
                )

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
                logger.exception(
                    "Hugging Face search failed for query=%r",
                    query,
                )

        # ---------------------------------------------------------
        # Deduplicate
        # ---------------------------------------------------------
        deduplicated_results = self._deduplicate_results(
            results
        )

        # ---------------------------------------------------------
        # Sort / rank
        # ---------------------------------------------------------
        if sort_by == "relevance":
            return RelevanceScorer.rank(
                query,
                deduplicated_results,
            )

        if sort_by == "published":
            return self._sort_by_date(
                deduplicated_results,
                field="published",
            )

        if sort_by == "updated":
            return self._sort_by_date(
                deduplicated_results,
                field="updated",
            )

        raise ValueError(
            f"unsupported sort option: {sort_by}"
        )

    # =============================================================
    # SORTING
    # =============================================================

    @staticmethod
    def _sort_by_date(
        results: list[ResearchItem],
        field: str,
    ) -> list[ResearchItem]:
        """
        Sort results by a date field in descending order.

        Handles:

        - timezone-aware datetime objects
        - timezone-naive datetime objects
        - ISO datetime strings
        - ISO strings ending with Z
        - missing dates
        - invalid dates

        All valid dates are normalized to UTC.

        Missing/invalid dates are placed at the end.
        """

        def normalize_date(
            value,
        ) -> datetime | None:
            # -----------------------------------------------------
            # Missing value
            # -----------------------------------------------------
            if value is None:
                return None

            # -----------------------------------------------------
            # datetime object
            # -----------------------------------------------------
            if isinstance(value, datetime):

                # Convert naive datetime to UTC.
                if value.tzinfo is None:
                    return value.replace(
                        tzinfo=timezone.utc
                    )

                # Convert aware datetime to UTC.
                return value.astimezone(
                    timezone.utc
                )

            # -----------------------------------------------------
            # String datetime
            # -----------------------------------------------------
            if isinstance(value, str):
                value = value.strip()

                if not value:
                    return None

                try:
                    # Convert trailing Z into UTC offset.
                    parsed = datetime.fromisoformat(
                        value.replace(
                            "Z",
                            "+00:00",
                        )
                    )

                except ValueError:
                    return None

                # Naive datetime string.
                if parsed.tzinfo is None:
                    return parsed.replace(
                        tzinfo=timezone.utc
                    )

                # Aware datetime string.
                return parsed.astimezone(
                    timezone.utc
                )

            # -----------------------------------------------------
            # Unsupported value type
            # -----------------------------------------------------
            return None

        dated_results: list[
            tuple[ResearchItem, datetime]
        ] = []

        undated_results: list[ResearchItem] = []

        # ---------------------------------------------------------
        # Separate dated and undated results
        # ---------------------------------------------------------
        for result in results:
            value = getattr(
                result,
                field,
                None,
            )

            normalized = normalize_date(value)

            if normalized is None:
                undated_results.append(result)
            else:
                dated_results.append(
                    (result, normalized)
                )

        # ---------------------------------------------------------
        # Sort valid dates newest first
        # ---------------------------------------------------------
        dated_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # ---------------------------------------------------------
        # Return dated results first and undated results last
        # ---------------------------------------------------------
        return [
            result
            for result, _ in dated_results
        ] + undated_results

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
            keys = ResearchService._deduplication_keys(
                result
            )

            if any(
                key in seen_keys
                for key in keys
            ):
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
        raw_id = str(
            result.id
        ).strip().lower()

        if raw_id:
            normalized_id = raw_id

            if normalized_id.startswith("arxiv-"):
                normalized_id = normalized_id.removeprefix(
                    "arxiv-"
                )

            if normalized_id.startswith("arxiv:"):
                normalized_id = normalized_id.removeprefix(
                    "arxiv:"
                )

            keys.add(
                f"research-id:{normalized_id}"
            )

        # ---------------------------------------------------------
        # 2. Normalized URL
        # ---------------------------------------------------------
        raw_url = str(
            result.url
        ).strip().lower()

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
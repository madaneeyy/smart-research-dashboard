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
from src.services.semantic_search import SemanticSearch


logger = logging.getLogger(__name__)


class ResearchService:

    # =============================================================
    # VALID OPTIONS
    # =============================================================

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

    VALID_SEARCH_MODES = {
        "keyword",
        "semantic",
        "hybrid",
    }

    # =============================================================
    # SEARCH
    # =============================================================

    def search(
        self,
        query: str,
        sources: list[str] | None = None,
        arxiv_limit: int = 20,
        github_limit: int = 20,
        paperswithcode_limit: int = 20,
        huggingface_limit: int = 20,
        sort_by: str = "relevance",
        search_mode: str = "keyword",
    ) -> list[ResearchItem]:
        """
        Search across the selected research sources.

        Pipeline:

        1. Validate input
        2. Collect results
        3. Normalize results
        4. Deduplicate results
        5. Sort/rank results
        6. Return final results

        Search modes:

        - keyword
        - semantic
        - hybrid
        """

        # ---------------------------------------------------------
        # Validate query
        # ---------------------------------------------------------

        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        # ---------------------------------------------------------
        # Validate sorting option
        # ---------------------------------------------------------

        if sort_by not in self.VALID_SORT_OPTIONS:
            raise ValueError(
                f"unknown sort option: {sort_by}"
            )

        # ---------------------------------------------------------
        # Validate search mode
        # ---------------------------------------------------------

        if search_mode not in self.VALID_SEARCH_MODES:
            raise ValueError(
                f"unknown search mode: {search_mode}"
            )

        # ---------------------------------------------------------
        # Determine selected sources
        # ---------------------------------------------------------

        if sources is None:

            selected_sources = self.VALID_SOURCES

        else:

            unknown_sources = (
                set(sources)
                - self.VALID_SOURCES
            )

            if unknown_sources:
                raise ValueError(
                    f"unknown source: "
                    f"{sorted(unknown_sources)[0]}"
                )

            selected_sources = set(sources)

        results: list[ResearchItem] = []

        # =========================================================
        # arXiv
        # =========================================================

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

        # =========================================================
        # GitHub
        # =========================================================

        if "github" in selected_sources:

            try:

                repositories = (
                    search_github_repositories(
                        query=query,
                        per_page=github_limit,
                    )
                )

                results.extend(
                    github_repository_to_item(
                        repository
                    )
                    for repository in repositories
                )

            except Exception:

                logger.exception(
                    "GitHub search failed for query=%r",
                    query,
                )

        # =========================================================
        # PapersWithCode
        # =========================================================

        if "paperswithcode" in selected_sources:

            try:

                papers = (
                    search_paperswithcode_papers(
                        query=query,
                        length=paperswithcode_limit,
                    )
                )

                results.extend(
                    paperswithcode_paper_to_item(
                        paper
                    )
                    for paper in papers
                )

            except Exception:

                logger.exception(
                    "PapersWithCode search failed "
                    "for query=%r",
                    query,
                )

        # =========================================================
        # Hugging Face
        # =========================================================

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
                    "Hugging Face search failed "
                    "for query=%r",
                    query,
                )

        # =========================================================
        # DEDUPLICATION
        # =========================================================

        deduplicated_results = (
            self._deduplicate_results(
                results
            )
        )

        # =========================================================
        # DATE SORTING
        # =========================================================

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

        # =========================================================
        # RELEVANCE
        # =========================================================

        return self._rank_by_search_mode(
            query=query,
            results=deduplicated_results,
            search_mode=search_mode,
        )

    # =============================================================
    # SEARCH MODE RANKING
    # =============================================================

    @classmethod
    def _rank_by_search_mode(
        cls,
        query: str,
        results: list[ResearchItem],
        search_mode: str,
    ) -> list[ResearchItem]:

        if not results:
            return []

        # ---------------------------------------------------------
        # Keyword search
        # ---------------------------------------------------------

        if search_mode == "keyword":

            return RelevanceScorer.rank(
                query,
                results,
            )

        # ---------------------------------------------------------
        # Semantic search
        # ---------------------------------------------------------

        if search_mode == "semantic":

            semantic_scores = (
                SemanticSearch.score(
                    query,
                    results,
                )
            )

            ranked = list(
                zip(
                    results,
                    semantic_scores,
                )
            )

            ranked.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            return [
                result
                for result, _ in ranked
            ]

        # ---------------------------------------------------------
        # Hybrid search
        # ---------------------------------------------------------

        if search_mode == "hybrid":

            return cls._hybrid_rank(
                query=query,
                results=results,
            )

        raise ValueError(
            f"unsupported search mode: "
            f"{search_mode}"
        )

    # =============================================================
    # HYBRID RANKING
    # =============================================================

    @classmethod
    def _hybrid_rank(
        cls,
        query: str,
        results: list[ResearchItem],
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> list[ResearchItem]:

        if not results:
            return []

        # ---------------------------------------------------------
        # Keyword ranking
        # ---------------------------------------------------------

        keyword_ranked = (
            RelevanceScorer.rank(
                query,
                results,
            )
        )

        # ---------------------------------------------------------
        # Convert keyword ranking into
        # normalized positional scores.
        #
        # First result = 1.0
        # Last result = 0.0
        # ---------------------------------------------------------

        keyword_scores: dict[str, float] = {}

        total = len(keyword_ranked)

        for index, result in enumerate(
            keyword_ranked
        ):

            if total == 1:

                score = 1.0

            else:

                score = (
                    1.0
                    - (
                        index
                        / (total - 1)
                    )
                )

            keyword_scores[
                cls._result_identity(result)
            ] = score

        # ---------------------------------------------------------
        # Semantic scores
        # ---------------------------------------------------------

        semantic_values = (
            SemanticSearch.score(
                query,
                results,
            )
        )

        semantic_scores = {
            cls._result_identity(result): score
            for result, score in zip(
                results,
                semantic_values,
            )
        }

        # ---------------------------------------------------------
        # Combine keyword + semantic scores
        #
        # Hybrid score:
        #
        # 40% keyword
        # 60% semantic
        # ---------------------------------------------------------

        hybrid_results = []

        for result in results:

            identity = (
                cls._result_identity(result)
            )

            keyword_score = (
                keyword_scores.get(
                    identity,
                    0.0,
                )
            )

            semantic_score = (
                semantic_scores.get(
                    identity,
                    0.0,
                )
            )

            hybrid_score = (
                keyword_weight
                * keyword_score
                +
                semantic_weight
                * semantic_score
            )

            hybrid_results.append(
                (
                    result,
                    hybrid_score,
                )
            )

        # ---------------------------------------------------------
        # Highest hybrid score first
        # ---------------------------------------------------------

        hybrid_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            result
            for result, _ in hybrid_results
        ]

    # =============================================================
    # RESULT IDENTITY
    # =============================================================

    @staticmethod
    def _result_identity(
        result: ResearchItem,
    ) -> str:

        return (
            f"{result.source}:"
            f"{result.id}:"
            f"{result.url}"
        )

    # =============================================================
    # DATE SORTING
    # =============================================================

    @staticmethod
    def _sort_by_date(
        results: list[ResearchItem],
        field: str,
    ) -> list[ResearchItem]:

        def normalize_date(
            value,
        ) -> datetime | None:

            if value is None:
                return None

            # -----------------------------------------------------
            # Already a datetime
            # -----------------------------------------------------

            if isinstance(
                value,
                datetime,
            ):

                if value.tzinfo is None:

                    return value.replace(
                        tzinfo=timezone.utc
                    )

                return value.astimezone(
                    timezone.utc
                )

            # -----------------------------------------------------
            # String date
            # -----------------------------------------------------

            if isinstance(
                value,
                str,
            ):

                value = value.strip()

                if not value:
                    return None

                try:

                    parsed = (
                        datetime.fromisoformat(
                            value.replace(
                                "Z",
                                "+00:00",
                            )
                        )
                    )

                except ValueError:

                    return None

                if parsed.tzinfo is None:

                    return parsed.replace(
                        tzinfo=timezone.utc
                    )

                return parsed.astimezone(
                    timezone.utc
                )

            return None

        dated_results: list[
            tuple[ResearchItem, datetime]
        ] = []

        undated_results: list[
            ResearchItem
        ] = []

        for result in results:

            value = getattr(
                result,
                field,
                None,
            )

            normalized = normalize_date(
                value
            )

            if normalized is None:

                undated_results.append(
                    result
                )

            else:

                dated_results.append(
                    (
                        result,
                        normalized,
                    )
                )

        # ---------------------------------------------------------
        # Newest first
        # ---------------------------------------------------------

        dated_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

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

        unique_results: list[
            ResearchItem
        ] = []

        seen_keys: set[str] = set()

        for result in results:

            keys = (
                ResearchService
                ._deduplication_keys(
                    result
                )
            )

            # -----------------------------------------------------
            # Skip duplicate
            # -----------------------------------------------------

            if any(
                key in seen_keys
                for key in keys
            ):
                continue

            # -----------------------------------------------------
            # Keep unique result
            # -----------------------------------------------------

            unique_results.append(
                result
            )

            seen_keys.update(
                keys
            )

        return unique_results

    # =============================================================
    # DEDUPLICATION KEYS
    # =============================================================

    @staticmethod
    def _deduplication_keys(
        result: ResearchItem,
    ) -> set[str]:

        keys: set[str] = set()

        # ---------------------------------------------------------
        # Research ID
        # ---------------------------------------------------------

        raw_id = str(
            result.id
        ).strip().lower()

        if raw_id:

            normalized_id = raw_id

            # -----------------------------------------------------
            # Normalize arXiv IDs
            # -----------------------------------------------------

            if normalized_id.startswith(
                "arxiv-"
            ):

                normalized_id = (
                    normalized_id.removeprefix(
                        "arxiv-"
                    )
                )

            if normalized_id.startswith(
                "arxiv:"
            ):

                normalized_id = (
                    normalized_id.removeprefix(
                        "arxiv:"
                    )
                )

            keys.add(
                f"research-id:"
                f"{normalized_id}"
            )

        # ---------------------------------------------------------
        # URL
        # ---------------------------------------------------------

        raw_url = str(
            result.url
        ).strip().lower()

        if raw_url:

            normalized_url = (
                raw_url.rstrip("/")
            )

            keys.add(
                f"url:{normalized_url}"
            )

        # ---------------------------------------------------------
        # Source + ID
        # ---------------------------------------------------------

        if raw_id and result.source:

            keys.add(
                f"source-id:"
                f"{result.source.strip().lower()}:"
                f"{raw_id}"
            )

        return keys
import re

from src.models.research import ResearchItem
from src.services.semantic_search import SemanticSearch


class RelevanceScorer:
    """
    Calculate relevance scores for ResearchItem objects.

    Keyword scoring is explainable and based on:
    1. Exact title match
    2. Full query phrase in title
    3. Query tokens in title
    4. Full query phrase in tags
    5. Query tokens in tags
    6. Full query phrase in description
    7. Query tokens in description

    Hybrid scoring combines:
        40% keyword relevance
        60% semantic relevance
    """

    # =========================================================
    # KEYWORD SCORE
    # =========================================================

    @staticmethod
    def score(
        query: str,
        item: ResearchItem,
    ) -> float:
        """
        Calculate the keyword relevance score for one item.

        Higher score means more keyword relevance.
        """

        query = query.strip().lower()

        if not query:
            raise ValueError("query must not be empty")

        title = (item.title or "").lower()
        description = (item.description or "").lower()

        tags = [
            str(tag).lower()
            for tag in (item.tags or [])
        ]

        score = 0.0

        # -----------------------------------------------------
        # Query tokens
        # -----------------------------------------------------

        query_words = RelevanceScorer._tokenize(query)

        if not query_words:
            return score

        title_words = RelevanceScorer._tokenize(title)
        description_words = RelevanceScorer._tokenize(
            description
        )

        # -----------------------------------------------------
        # 1. Exact title match
        # -----------------------------------------------------

        exact_title_match = title == query

        if exact_title_match:
            score += 10.0

            # Exact title is already the strongest match.
            return score

        # -----------------------------------------------------
        # 2. Full query phrase in title
        # -----------------------------------------------------

        if RelevanceScorer._contains_phrase(
            title,
            query,
        ):
            score += 6.0

        # -----------------------------------------------------
        # 3. Query words in title
        # -----------------------------------------------------

        matched_title_words = sum(
            1
            for word in query_words
            if word in title_words
        )

        for word in query_words:
            if word in title_words:
                score += 2.0

        # -----------------------------------------------------
        # 4. Complete query coverage in title
        # -----------------------------------------------------

        if matched_title_words == len(query_words):
            score += 3.0

        # -----------------------------------------------------
        # 5. Full query phrase in tags
        # -----------------------------------------------------

        if any(
            RelevanceScorer._contains_phrase(
                tag,
                query,
            )
            for tag in tags
        ):
            score += 4.0

        # -----------------------------------------------------
        # 6. Individual query words in tags
        # -----------------------------------------------------

        for word in query_words:
            if any(
                word in tag
                for tag in tags
            ):
                score += 1.0

        # -----------------------------------------------------
        # 7. Full query phrase in description
        # -----------------------------------------------------

        if RelevanceScorer._contains_phrase(
            description,
            query,
        ):
            score += 3.0

        # -----------------------------------------------------
        # 8. Individual query words in description
        # -----------------------------------------------------

        for word in query_words:
            if word in description_words:
                score += 1.0

        return score

    # =========================================================
    # KEYWORD RANKING
    # =========================================================

    @staticmethod
    def rank(
        query: str,
        items: list[ResearchItem],
    ) -> list[ResearchItem]:
        """
        Return items ordered from most keyword-relevant
        to least keyword-relevant.
        """

        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        return sorted(
            items,
            key=lambda item: RelevanceScorer.score(
                query,
                item,
            ),
            reverse=True,
        )

    # =========================================================
    # HYBRID RANKING
    # =========================================================

    @staticmethod
    def hybrid_rank(
        query: str,
        items: list[ResearchItem],
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> list[ResearchItem]:
        """
        Rank results using both keyword and semantic relevance.

        Default weighting:

            40% keyword relevance
            60% semantic relevance

        Final score:

            hybrid_score =
                0.4 * keyword_score
                +
                0.6 * semantic_score
        """

        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        if not items:
            return []

        if keyword_weight < 0:
            raise ValueError(
                "keyword_weight must not be negative"
            )

        if semantic_weight < 0:
            raise ValueError(
                "semantic_weight must not be negative"
            )

        total_weight = (
            keyword_weight
            + semantic_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "at least one ranking weight must be greater than zero"
            )

        # -----------------------------------------------------
        # Normalize weights
        # -----------------------------------------------------

        keyword_weight = (
            keyword_weight / total_weight
        )

        semantic_weight = (
            semantic_weight / total_weight
        )

        # -----------------------------------------------------
        # Keyword scores
        # -----------------------------------------------------

        raw_keyword_scores = {
            RelevanceScorer._result_identity(item):
                RelevanceScorer.score(
                    query,
                    item,
                )
            for item in items
        }

        keyword_scores = (
            RelevanceScorer._normalize_scores(
                raw_keyword_scores
            )
        )

        # -----------------------------------------------------
        # Semantic scores
        # -----------------------------------------------------

        semantic_values = SemanticSearch.score(
            query,
            items,
        )

        raw_semantic_scores = {
            RelevanceScorer._result_identity(item):
                float(score)
            for item, score in zip(
                items,
                semantic_values,
            )
        }

        semantic_scores = (
            RelevanceScorer._normalize_scores(
                raw_semantic_scores
            )
        )

        # -----------------------------------------------------
        # Combine keyword + semantic scores
        # -----------------------------------------------------

        hybrid_results = []

        for item in items:

            identity = (
                RelevanceScorer._result_identity(
                    item
                )
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

            # ---------------------------------------------
            # HYBRID SCORE
            # ---------------------------------------------

            hybrid_score = (
                keyword_weight
                * keyword_score
                +
                semantic_weight
                * semantic_score
            )

            hybrid_results.append(
                (
                    item,
                    hybrid_score,
                )
            )

        # -----------------------------------------------------
        # Highest hybrid score first
        # -----------------------------------------------------

        hybrid_results.sort(
            key=lambda pair: pair[1],
            reverse=True,
        )

        return [
            item
            for item, _ in hybrid_results
        ]

    # =========================================================
    # RESULT IDENTITY
    # =========================================================

    @staticmethod
    def _result_identity(
        item: ResearchItem,
    ) -> str:
        """
        Return a stable identity for a ResearchItem.

        Source is included so two different sources with
        the same ID do not accidentally collide.
        """

        return (
            f"{item.source}:{item.id}"
        )

    # =========================================================
    # NORMALIZE SCORES
    # =========================================================

    @staticmethod
    def _normalize_scores(
        scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Normalize scores to the range 0.0 - 1.0.

        Highest score becomes 1.0.
        Lowest score becomes 0.0.

        If every score is identical, every item receives 1.0.
        """

        if not scores:
            return {}

        values = list(scores.values())

        minimum = min(values)
        maximum = max(values)

        # All scores are identical.
        if maximum == minimum:
            return {
                identity: 1.0
                for identity in scores
            }

        return {
            identity: (
                (score - minimum)
                / (maximum - minimum)
            )
            for identity, score in scores.items()
        }

    # =========================================================
    # TOKENIZATION
    # =========================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        """
        Convert text into normalized word tokens.

        Examples:

            transformer_models
                -> ["transformer", "models"]

            transformer-models
                -> ["transformer", "models"]

            NVIDIA/Megatron-LM
                -> ["nvidia", "megatron", "lm"]
        """

        return re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

    # =========================================================
    # PHRASE MATCHING
    # =========================================================

    @staticmethod
    def _contains_phrase(
        text: str,
        query: str,
    ) -> bool:
        """
        Check whether all query words appear consecutively.

        Separators such as spaces, underscores, hyphens,
        and slashes are treated equivalently.
        """

        text_words = (
            RelevanceScorer._tokenize(text)
        )

        query_words = (
            RelevanceScorer._tokenize(query)
        )

        if not query_words:
            return False

        query_length = len(query_words)

        for index in range(
            len(text_words)
            - query_length
            + 1
        ):
            if (
                text_words[
                    index:index + query_length
                ]
                == query_words
            ):
                return True

        return False
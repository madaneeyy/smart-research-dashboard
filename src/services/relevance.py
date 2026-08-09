import re

from src.models.research import ResearchItem


class RelevanceScorer:
    """
    Calculate a simple, explainable relevance score for ResearchItem objects.

    Scoring priorities:
    1. Exact title match
    2. Full query phrase in title
    3. Query tokens in title
    4. Full query phrase in tags
    5. Query tokens in tags
    6. Full query phrase in description
    7. Query tokens in description
    """

    @staticmethod
    def score(
        query: str,
        item: ResearchItem,
    ) -> float:
        """
        Calculate the relevance score for one ResearchItem.

        Higher score means more relevant.
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

        # ---------------------------------------------------------
        # Query tokens
        # ---------------------------------------------------------
        query_words = RelevanceScorer._tokenize(query)

        if not query_words:
            return score

        title_words = RelevanceScorer._tokenize(title)
        description_words = RelevanceScorer._tokenize(description)

        # ---------------------------------------------------------
        # 1. Exact title match
        # ---------------------------------------------------------
        exact_title_match = title == query

        if exact_title_match:
            score += 10.0

            # Exact title already represents the strongest
            # possible title match. Avoid adding token bonuses.
            return score

        # ---------------------------------------------------------
        # 2. Full query phrase appears in title
        # ---------------------------------------------------------
        if RelevanceScorer._contains_phrase(title, query):
            score += 6.0

        # ---------------------------------------------------------
        # 3. Query words in title
        # ---------------------------------------------------------
        matched_title_words = sum(
            1
            for word in query_words
            if word in title_words
        )

        for word in query_words:
            if word in title_words:
                score += 2.0

        # ---------------------------------------------------------
        # 4. Query coverage bonus in title
        #
        # Example:
        #
        # query:
        #     transformer models
        #
        # title:
        #     transformer models
        #
        # Both query words are present, so the result gets an
        # additional bonus for covering the complete query.
        # ---------------------------------------------------------
        if matched_title_words == len(query_words):
            score += 3.0

        # ---------------------------------------------------------
        # 5. Full query phrase appears in tags
        # ---------------------------------------------------------
        if any(
            RelevanceScorer._contains_phrase(tag, query)
            for tag in tags
        ):
            score += 4.0

        # ---------------------------------------------------------
        # 6. Individual query words in tags
        # ---------------------------------------------------------
        for word in query_words:
            if any(word in tag for tag in tags):
                score += 1.0

        # ---------------------------------------------------------
        # 7. Full query phrase appears in description
        # ---------------------------------------------------------
        if RelevanceScorer._contains_phrase(
            description,
            query,
        ):
            score += 3.0

        # ---------------------------------------------------------
        # 8. Individual query words in description
        # ---------------------------------------------------------
        for word in query_words:
            if word in description_words:
                score += 1.0

        return score

    @staticmethod
    def rank(
        query: str,
        items: list[ResearchItem],
    ) -> list[ResearchItem]:
        """
        Return items ordered from most relevant to least relevant.

        Python's sort is stable, so items with identical scores
        preserve their original order.
        """

        if not query.strip():
            raise ValueError("query must not be empty")

        return sorted(
            items,
            key=lambda item: RelevanceScorer.score(
                query,
                item,
            ),
            reverse=True,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into normalized word tokens.

        Underscores, hyphens, slashes, and other separators are
        treated as word boundaries.

        Examples:

            "transformer_models"
                -> ["transformer", "models"]

            "transformer-models"
                -> ["transformer", "models"]

            "NVIDIA/Megatron-LM"
                -> ["nvidia", "megatron", "lm"]
        """

        return re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

    @staticmethod
    def _contains_phrase(
        text: str,
        query: str,
    ) -> bool:
        """
        Check whether all query words appear consecutively.

        Separators such as spaces, underscores, hyphens, and
        slashes are treated equivalently.

        Examples:

            query = "transformer models"

            "Transformer Models"
                -> True

            "Transformer_Models"
                -> True

            "Transformer-Models"
                -> True

            "Models for Transformer"
                -> False
        """

        text_words = RelevanceScorer._tokenize(text)
        query_words = RelevanceScorer._tokenize(query)

        if not query_words:
            return False

        query_length = len(query_words)

        for index in range(
            len(text_words) - query_length + 1
        ):
            if (
                text_words[index:index + query_length]
                == query_words
            ):
                return True

        return False
import re

from src.models.research import ResearchItem


class RelevanceScorer:
    """
    Calculate a simple, explainable relevance score for ResearchItem objects.
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

        # 1. Exact title match
        exact_title_match = title == query

        if exact_title_match:
            score += 10.0

        # 2. Full query appears in title
        if query in title and not exact_title_match:
            score += 6.0

        # 3. Full query appears in description
        if query in description:
            score += 3.0

        # 4. Full query appears in tags
        if any(query in tag for tag in tags):
            score += 4.0

        # Token-level matching
        query_words = RelevanceScorer._tokenize(query)

        if not query_words:
            return score

        title_words = RelevanceScorer._tokenize(title)
        description_words = RelevanceScorer._tokenize(description)

        # 5. Individual query words in title
        #
        # Don't add the +2 bonus to an exact title match.
        # Otherwise "transformer" would score 12 instead of 10.
        if not exact_title_match:
            for word in query_words:
                if word in title_words:
                    score += 2.0

        # 6. Individual query words in description
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

        Python's sort is stable, so items with equal scores
        preserve their original order.
        """

        if not query.strip():
            raise ValueError("query must not be empty")

        return sorted(
            items,
            key=lambda item: RelevanceScorer.score(query, item),
            reverse=True,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into normalized word tokens.
        """

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )
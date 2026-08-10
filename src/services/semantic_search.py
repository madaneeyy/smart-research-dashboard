from functools import lru_cache

from sentence_transformers import SentenceTransformer


class SemanticSearch:
    """
    Generate semantic embeddings and calculate similarity
    between a search query and research items.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    @classmethod
    @lru_cache(maxsize=1)
    def _get_model(cls) -> SentenceTransformer:
        """
        Load the embedding model once and reuse it.
        """
        return SentenceTransformer(cls.MODEL_NAME)

    @staticmethod
    def _build_text(result) -> str:
        """
        Build the text representation used for semantic search.
        """

        parts: list[str] = []

        if result.title:
            parts.append(result.title)

        if result.description:
            parts.append(result.description)

        if result.authors:
            parts.append(
                " ".join(result.authors)
            )

        if result.tags:
            parts.append(
                " ".join(result.tags)
            )

        tasks = getattr(
            result,
            "tasks",
            [],
        )

        if tasks:
            parts.append(
                " ".join(tasks)
            )

        conference = getattr(
            result,
            "conference",
            None,
        )

        if conference:
            parts.append(conference)

        return " ".join(parts)

    @classmethod
    def score(
        cls,
        query: str,
        results: list,
    ) -> list[float]:
        """
        Calculate semantic similarity scores.

        Returns one score per result.
        Scores are normalized approximately between 0 and 1.
        """

        if not results:
            return []

        model = cls._get_model()

        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
        )

        result_texts = [
            cls._build_text(result)
            for result in results
        ]

        result_embeddings = model.encode(
            result_texts,
            normalize_embeddings=True,
        )

        scores = (
            result_embeddings
            @ query_embedding
        )

        # Convert cosine similarity from
        # approximately [-1, 1] to [0, 1].
        normalized_scores = [
            float((score + 1) / 2)
            for score in scores
        ]

        return normalized_scores
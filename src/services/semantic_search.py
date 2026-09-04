from functools import lru_cache

from src.services.github_rag.embedding_provider import create_embedding_provider


class SemanticSearch:
    """
    Generate semantic embeddings and calculate similarity
    between a search query and research items.

    Embeddings are generated through the shared embedding provider,
    which can use the configured remote Hugging Face provider.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    @classmethod
    @lru_cache(maxsize=1)
    def _get_embedding_provider(cls):
        """
        Create and reuse the shared embedding provider.
        """
        return create_embedding_provider()

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

        provider = cls._get_embedding_provider()

        # Generate normalized query embedding.
        query_embedding = provider.embed([query])[0]

        result_texts = [
            cls._build_text(result)
            for result in results
        ]

        # Generate normalized result embeddings.
        result_embeddings = provider.embed(result_texts)

        # Because embeddings are normalized, dot product is
        # equivalent to cosine similarity.
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


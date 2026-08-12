from sentence_transformers import SentenceTransformer
import numpy as np


class SimpleRetriever:
    """
    Embedding-based semantic retriever.

    Converts the question and document chunks into embeddings
    and ranks chunks according to cosine similarity.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    _model = None

    @classmethod
    def _get_model(cls):
        """
        Load the embedding model only once.
        """

        if cls._model is None:
            cls._model = SentenceTransformer(
                cls.MODEL_NAME
            )

        return cls._model

    @classmethod
    def retrieve(
        cls,
        question: str,
        chunks: list[str],
        top_k: int = 3,
    ) -> list[str]:
        """
        Return the most semantically relevant chunks.
        """

        if not question or not question.strip():
            return []

        if not chunks:
            return []

        model = cls._get_model()

        # Create embedding for the question
        question_embedding = model.encode(
            question,
            normalize_embeddings=True,
        )

        # Create embeddings for all chunks
        chunk_embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
        )

        # Cosine similarity becomes a dot product
        # because embeddings are normalized.
        scores = np.dot(
            chunk_embeddings,
            question_embedding,
        )

        # Sort highest similarity first
        ranked_indices = np.argsort(
            scores
        )[::-1]

        # Return top K chunks
        results = []

        for index in ranked_indices[:top_k]:
            results.append(
                chunks[index]
            )

        return results
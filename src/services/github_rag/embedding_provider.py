from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np


class EmbeddingProvider(ABC):
    """
    Common interface for generating text embeddings.

    The rest of the retrieval system should depend on this interface,
    not directly on SentenceTransformer or any external API.
    """

    @abstractmethod
    def embed(
        self,
        texts: str | Sequence[str],
    ) -> np.ndarray:
        """
        Generate embeddings for one or more texts.

        Returns:
            numpy array of shape:

                (embedding_dimension,)
            
            for a single text, or

                (number_of_texts, embedding_dimension)

            for multiple texts.
        """
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local development provider.

    Uses the existing SentenceTransformer model.

    This keeps the current behavior intact while we migrate
    the retrieval system to the new provider interface.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        print(
            f"Loading local embedding model: {self.MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            self.MODEL_NAME
        )

    def embed(
        self,
        texts: str | Sequence[str],
    ) -> np.ndarray:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    Remote embedding provider using Hugging Face Inference Providers.

    The actual embedding model runs remotely.

    The Render backend therefore does NOT need to load
    SentenceTransformer/PyTorch just to create embeddings.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        token: str | None = None,
    ) -> None:

        from huggingface_hub import InferenceClient

        self.token = token or os.getenv("HF_TOKEN")

        if not self.token:
            raise RuntimeError(
                "HF_TOKEN is not set. "
                "Add your Hugging Face access token "
                "to the environment variables."
            )

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.token,
        )

    def embed(
        self,
        texts: str | Sequence[str],
    ) -> np.ndarray:

        # Hugging Face accepts either one text or a list of texts.
        result = self.client.feature_extraction(
            text=texts,
            model=self.MODEL_NAME,
        )

        embeddings = np.asarray(
            result,
            dtype=np.float32,
        )

        # Make sure a single text still produces a
        # consistent 2-dimensional array internally.
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize locally so that our existing
        # cosine-similarity/dot-product behavior remains unchanged.
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        embeddings = embeddings / np.maximum(
            norms,
            1e-12,
        )

        return embeddings


def create_embedding_provider() -> EmbeddingProvider:
    """
    Create the embedding provider based on EMBEDDING_MODE.

    Supported modes:

        local
            Uses SentenceTransformer locally.

        remote
            Uses Hugging Face remotely.
    """

    mode = os.getenv(
        "EMBEDDING_MODE",
        "local",
    ).strip().lower()

    if mode == "remote":
        print(
            "Using remote Hugging Face embedding provider."
        )

        return HuggingFaceEmbeddingProvider()

    if mode == "local":
        print(
            "Using local SentenceTransformer embedding provider."
        )

        return LocalEmbeddingProvider()

    raise ValueError(
        f"Unsupported EMBEDDING_MODE: {mode}. "
        "Use 'local' or 'remote'."
    )
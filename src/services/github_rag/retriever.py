import numpy as np
import hashlib
import os

from .embedding_provider import create_embedding_provider


class SimpleRetriever:
    """
    Embedding-based semantic retriever with:

    1. Shared embedding-provider abstraction
    2. Persistent chunk-embedding cache
    3. Automatic embedding-cache invalidation when chunks change
    4. Cosine-similarity ranking
    5. Preservation of all chunk metadata

    Embedding behavior
    ------------------
    The retriever does not load the embedding model directly.

    The embedding provider is selected through EMBEDDING_MODE:

        local
            Uses the existing local embedding provider.

        remote
            Uses the Hugging Face embedding provider.

    This allows the Render backend to use remote embeddings without
    loading the embedding model/PyTorch into the main process.
    """

    # ============================================================
    # EMBEDDING PROVIDER
    # ============================================================

    _embedding_provider = None

    # ============================================================
    # CACHE CONFIGURATION
    # ============================================================

    # Cache directory:
    #
    # src/services/rag/
    #     cache/
    #         chunk_<hash>.npy
    #
    CACHE_DIR = os.path.join(
        os.path.dirname(__file__),
        "cache",
    )

    # ============================================================
    # EMBEDDING CACHE IDENTIFIER
    # ============================================================

    @classmethod
    def _chunk_cache_key(cls, chunk):
        """Create a stable cache key for one chunk."""
        cache_data = {}

        transient = {
            "similarity", "score", "hybrid_score",
            "semantic_score", "semantic_rank", "bm25_rank",
            "retrieval_rank", "query_relevance_score",
            "mmr_score", "mmr_relevance", "mmr_redundancy",
            "metadata_bonus", "complementarity_score",
        }

        for key, value in chunk.items():
            if key not in transient:
                cache_data[key] = value

        serialized = repr(sorted(
            cache_data.items(),
            key=lambda item: str(item[0]),
        ))

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()[:32]

    @classmethod
    def _get_chunk_cache_path(cls, chunk):
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        return os.path.join(
            cls.CACHE_DIR,
            f"chunk_{cls._chunk_cache_key(chunk)}.npy",
        )

    @classmethod
    def _get_cache_path(cls, chunks):
        """Backward-compatible helper for older callers."""
        if not chunks:
            return os.path.join(
                cls.CACHE_DIR,
                "chunk_empty.npy",
            )
        return cls._get_chunk_cache_path(chunks[0])

    # Maximum number/age of persisted chunk embeddings.
    # Override with environment variables in deployment.
    MAX_EMBEDDING_CACHE_FILES = int(
        os.getenv("RAG_MAX_EMBEDDING_CACHE_FILES", "5000")
    )
    EMBEDDING_CACHE_TTL_SECONDS = int(
        os.getenv("RAG_EMBEDDING_CACHE_TTL", str(30 * 24 * 60 * 60))
    )

    @classmethod
    def _get_embedding_provider(cls):
        """
        Return the shared embedding provider.

        The provider is created once per Python process and reused
        for both query and chunk embeddings.
        """
        if cls._embedding_provider is None:
            cls._embedding_provider = create_embedding_provider()

        return cls._embedding_provider

    # ============================================================
    # CHUNK EMBEDDINGS
    # ============================================================

    @classmethod
    def _get_chunk_embeddings(cls, chunks):
        """
        Return embeddings using a persistent PER-CHUNK cache.

        This means different queries can reuse embeddings for overlapping
        chunks instead of creating a new batch cache for every query.
        """
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = [None] * len(chunks)
        missing_chunks = []
        missing_positions = []

        for position, chunk in enumerate(chunks):
            cache_path = cls._get_chunk_cache_path(chunk)

            try:
                if os.path.exists(cache_path):
                    embedding = np.load(
                        cache_path,
                        allow_pickle=False,
                    )
                    embedding = np.asarray(
                        embedding,
                        dtype=np.float32,
                    ).reshape(-1)

                    if embedding.size:
                        embeddings[position] = embedding
                        continue

            except (OSError, ValueError):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

            missing_chunks.append(chunk)
            missing_positions.append(position)

        if missing_chunks:
            provider = cls._get_embedding_provider()

            print(
                f"Embedding cache miss: "
                f"{len(missing_chunks)}/{len(chunks)} chunks."
            )

            texts = [
                str(chunk.get("content", ""))
                for chunk in missing_chunks
            ]

            generated = np.asarray(
                provider.embed(texts),
                dtype=np.float32,
            )

            if generated.ndim == 1:
                generated = generated.reshape(1, -1)

            if len(generated) != len(missing_chunks):
                raise ValueError(
                    "Embedding model returned an unexpected "
                    "number of embeddings."
                )

            for position, embedding, chunk in zip(
                missing_positions,
                generated,
                missing_chunks,
            ):
                embedding = np.asarray(
                    embedding,
                    dtype=np.float32,
                ).reshape(-1)

                cache_path = cls._get_chunk_cache_path(chunk)
                temporary_path = (
                    f"{cache_path}.{os.getpid()}.tmp"
                )

                try:
                    with open(temporary_path, "wb") as file:
                        np.save(file, embedding)

                    os.replace(
                        temporary_path,
                        cache_path,
                    )
                except OSError:
                    try:
                        os.remove(temporary_path)
                    except OSError:
                        pass

                embeddings[position] = embedding

            cls._prune_embedding_cache()

        if any(embedding is None for embedding in embeddings):
            raise RuntimeError(
                "Failed to create embeddings for all chunks."
            )

        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise ValueError(
                "Cached chunk embeddings have inconsistent dimensions."
            )

        return np.vstack(embeddings).astype(
            np.float32,
            copy=False,
        )

    @classmethod
    def get_embeddings(cls, chunks):
        """
        Public cache-aware embedding accessor.

        HybridRetriever should use this instead of generating
        chunk embeddings through a model directly.
        """
        return cls._get_chunk_embeddings(chunks)

    # ============================================================
    # SEMANTIC RETRIEVAL
    # ============================================================

    @classmethod
    def retrieve(
        cls,
        question: str,
        chunks: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Return the most semantically relevant chunks.

        Each returned result contains:

            - Every original chunk field
            - similarity

        This means the retriever automatically preserves
        metadata added by the improved chunker.

        Example:

            {
                "content": "...",
                "path": "...",
                "category": "...",
                "section": "...",
                "chunk_index": 4,
                "chunk_type": "function",
                "symbol": "train_test_split",
                "language": "python",
                "similarity": 0.72
            }
        """

        # --------------------------------------------------------
        # Validate question
        # --------------------------------------------------------

        if not question or not question.strip():
            return []

        # --------------------------------------------------------
        # Validate chunks
        # --------------------------------------------------------

        if not chunks:
            return []

        # --------------------------------------------------------
        # Normalize top_k
        # --------------------------------------------------------

        top_k = max(
            1,
            min(
                top_k,
                len(chunks),
            ),
        )

        # --------------------------------------------------------
        # Generate query embedding
        # --------------------------------------------------------

        provider = cls._get_embedding_provider()

        question_embedding = np.asarray(
            provider.embed(question),
            dtype=np.float32,
        ).reshape(-1)

        # --------------------------------------------------------
        # Load/create chunk embeddings
        # --------------------------------------------------------

        chunk_embeddings = (
            cls._get_chunk_embeddings(
                chunks
            )
        )

        # --------------------------------------------------------
        # Validate embedding count
        # --------------------------------------------------------

        if len(chunk_embeddings) != len(
            chunks
        ):
            raise ValueError(
                "Cached embedding count does not match "
                "chunk count. Delete the retriever cache "
                "and run again."
            )

        # --------------------------------------------------------
        # COSINE SIMILARITY
        # --------------------------------------------------------
        #
        # Both question and chunk embeddings are normalized.
        #
        # Therefore:
        #
        # cosine_similarity(A, B)
        #       =
        # dot(A, B)
        #
        # --------------------------------------------------------

        scores = np.dot(
            chunk_embeddings,
            question_embedding,
        )

        # --------------------------------------------------------
        # Rank highest similarity first
        # --------------------------------------------------------

        ranked_indices = np.argsort(
            scores
        )[::-1]

        # --------------------------------------------------------
        # Build results
        # --------------------------------------------------------

        results = []

        for index in ranked_indices[
            :top_k
        ]:

            # Make a copy rather than modifying the
            # original chunk.
            result = dict(
                chunks[index]
            )

            # Add retrieval score.
            result["similarity"] = float(
                scores[index]
            )

            results.append(
                result
            )

        return results

    @classmethod
    def _prune_embedding_cache(cls):
        """
        Bound the on-disk embedding cache.

        Old entries are removed by TTL first. If the cache still exceeds
        the configured maximum, the oldest files are removed.
        """
        if not os.path.isdir(cls.CACHE_DIR):
            return

        now = __import__("time").time()
        files = []

        for filename in os.listdir(cls.CACHE_DIR):
            if not (
                filename.startswith("chunk_")
                and filename.endswith(".npy")
            ):
                continue

            path = os.path.join(cls.CACHE_DIR, filename)

            try:
                stat = os.stat(path)
            except OSError:
                continue

            age = now - stat.st_mtime

            if (
                cls.EMBEDDING_CACHE_TTL_SECONDS >= 0
                and age > cls.EMBEDDING_CACHE_TTL_SECONDS
            ):
                try:
                    os.remove(path)
                except OSError:
                    pass
                continue

            files.append((stat.st_mtime, path))

        limit = cls.MAX_EMBEDDING_CACHE_FILES

        if limit < 0:
            return

        if len(files) > limit:
            files.sort(key=lambda item: item[0])

            for _, path in files[: len(files) - limit]:
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ============================================================
    # CACHE MANAGEMENT
    # ============================================================

    @classmethod
    def clear_embedding_cache(cls) -> None:
        """
        Delete generated chunk embedding caches.

        The embedding model/provider is managed separately by
        embedding_provider.py.
        """

        if not os.path.isdir(
            cls.CACHE_DIR
        ):
            return

        deleted = 0

        for filename in os.listdir(
            cls.CACHE_DIR
        ):

            if (
                (
                    filename.startswith("embeddings_")
                    or filename.startswith("chunk_")
                )
                and filename.endswith(
                    ".npy"
                )
            ):

                try:

                    os.remove(
                        os.path.join(
                            cls.CACHE_DIR,
                            filename,
                        )
                    )

                    deleted += 1

                except OSError:
                    pass

        print(
            f"Deleted {deleted} embedding cache file(s)."
        )
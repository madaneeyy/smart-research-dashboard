from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import os


class SimpleRetriever:
    """
    Embedding-based semantic retriever with:

    1. Persistent local embedding-model cache
    2. Persistent chunk-embedding cache
    3. Automatic embedding-cache invalidation when chunks change
    4. Cosine-similarity ranking
    5. Preservation of all chunk metadata

    Model behavior
    --------------
    First run:
        Hugging Face -> local model directory

    Later runs:
        local model directory -> SentenceTransformer

    Chunk embedding behavior
    ------------------------
    First run:
        Generate embeddings -> save to disk

    Later runs:
        Load embeddings from disk

    The embedding cache is automatically invalidated when
    chunk content or metadata changes.
    """

    # ============================================================
    # MODEL CONFIGURATION
    # ============================================================

    MODEL_NAME = "all-MiniLM-L6-v2"

    _model = None

    # Cache directory:
    #
    # src/services/rag/
    #     cache/
    #         models/
    #             all-MiniLM-L6-v2/
    #         embeddings_xxxxx.npy
    #
    CACHE_DIR = os.path.join(
        os.path.dirname(__file__),
        "cache",
    )

    MODEL_CACHE_DIR = os.path.join(
        CACHE_DIR,
        "models",
        "all-MiniLM-L6-v2",
    )

    # ============================================================
    # MODEL LOADING
    # ============================================================

    @classmethod
    def _get_model(cls):
        """
        Load the embedding model.

        If the model already exists locally:
            Load ONLY from local files.

        If the model does not exist:
            Download it from Hugging Face and store it locally.

        The model is loaded only once per Python process.
        """

        # --------------------------------------------------------
        # Already loaded in this Python process
        # --------------------------------------------------------

        if cls._model is not None:
            return cls._model

        os.makedirs(
            cls.MODEL_CACHE_DIR,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Check whether model exists locally
        # --------------------------------------------------------

        model_exists = os.path.exists(
            os.path.join(
                cls.MODEL_CACHE_DIR,
                "config.json",
            )
        )

        # --------------------------------------------------------
        # LOCAL MODEL
        # --------------------------------------------------------

        if model_exists:

            print(
                "Loading embedding model from local cache:"
            )

            print(
                cls.MODEL_CACHE_DIR
            )

            cls._model = SentenceTransformer(
                cls.MODEL_CACHE_DIR,
                local_files_only=True,
            )

            print(
                "Embedding model loaded from local cache."
            )

        # --------------------------------------------------------
        # FIRST RUN
        # --------------------------------------------------------

        else:

            print(
                "Embedding model not found locally."
            )

            print(
                f"Downloading {cls.MODEL_NAME}..."
            )

            cls._model = SentenceTransformer(
                cls.MODEL_NAME,
                cache_folder=cls.MODEL_CACHE_DIR,
            )

            print(
                "Embedding model downloaded and cached locally:"
            )

            print(
                cls.MODEL_CACHE_DIR
            )

        return cls._model

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
            model = cls._get_model()

            print(
                f"Embedding cache miss: "
                f"{len(missing_chunks)}/{len(chunks)} chunks."
            )

            texts = [
                str(chunk.get("content", ""))
                for chunk in missing_chunks
            ]

            generated = np.asarray(
                model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
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

        HybridRetriever should use this instead of calling
        SentenceTransformer.encode() directly for chunk text.
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
        # Load embedding model
        # --------------------------------------------------------

        model = cls._get_model()

        # --------------------------------------------------------
        # Encode question
        # --------------------------------------------------------

        question_embedding = model.encode(
            question,
            normalize_embeddings=True,
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype=np.float32,
        )

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

        This does NOT delete the locally cached
        SentenceTransformer model.
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
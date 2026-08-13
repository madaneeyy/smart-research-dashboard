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
    def _get_cache_path(cls, chunks):
        """
        Generate a deterministic embedding-cache path.

        The cache hash includes:

            - chunk content
            - path
            - section
            - category
            - chunk index
            - chunk type
            - symbol
            - language
            - other metadata

        Therefore, if the chunker changes the chunks,
        the old embedding cache will automatically stop
        being used.
        """

        os.makedirs(
            cls.CACHE_DIR,
            exist_ok=True,
        )

        chunk_data = []

        for chunk in chunks:

            # ----------------------------------------------------
            # Preserve all chunk metadata.
            #
            # Sorting makes the hash independent of dictionary
            # insertion order.
            # ----------------------------------------------------

            chunk_metadata = {}

            for key, value in chunk.items():

                # Do not include fields that should not affect
                # the actual embedding content.
                #
                # Similarity/ranking fields should never be part
                # of the source chunk itself anyway, but excluding
                # them makes the cache safer.
                if key in {
                    "similarity",
                    "score",
                }:
                    continue

                chunk_metadata[key] = value

            chunk_data.append(
                chunk_metadata
            )

        cache_string = repr(
            chunk_data
        )

        cache_hash = hashlib.sha256(
            cache_string.encode("utf-8")
        ).hexdigest()[:32]

        return os.path.join(
            cls.CACHE_DIR,
            f"embeddings_{cache_hash}.npy",
        )

    # ============================================================
    # CHUNK EMBEDDINGS
    # ============================================================

    @classmethod
    def _get_chunk_embeddings(cls, chunks):
        """
        Return embeddings for all chunks.

        If an embedding cache exists:
            Load it.

        Otherwise:
            Generate embeddings using the local model
            and persist them to disk.
        """

        cache_path = cls._get_cache_path(
            chunks
        )

        # --------------------------------------------------------
        # CACHE HIT
        # --------------------------------------------------------

        if os.path.exists(cache_path):

            print(
                "Chunk embeddings cache hit."
            )

            return np.load(
                cache_path
            )

        # --------------------------------------------------------
        # CACHE MISS
        # --------------------------------------------------------

        model = cls._get_model()

        print(
            f"Creating embeddings for {len(chunks)} chunks..."
        )

        texts = [
            chunk.get(
                "content",
                "",
            )
            for chunk in chunks
        ]

        # --------------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------------

        chunk_embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        chunk_embeddings = np.asarray(
            chunk_embeddings,
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # Atomic-ish cache write
        # --------------------------------------------------------

        temporary_path = (
            f"{cache_path}.tmp"
        )

        np.save(
            temporary_path,
            chunk_embeddings,
        )

        # np.save adds .npy if necessary.
        generated_temp = temporary_path

        if not generated_temp.endswith(
            ".npy"
        ):
            generated_temp += ".npy"

        os.replace(
            generated_temp,
            cache_path,
        )

        print(
            "Chunk embeddings cached."
        )

        return chunk_embeddings

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
                filename.startswith(
                    "embeddings_"
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
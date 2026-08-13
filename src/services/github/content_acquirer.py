import base64
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests


class GitHubContentAcquirer:
    """
    Fetches GitHub repository file contents.

    Acquisition strategy:
        1. Load from local cache when available
        2. GitHub API
        3. Raw GitHub URL fallback

    The output is normalized so the rest of the RAG pipeline
    does not need to care where the content came from.

    Persistent caching:
        cache/
            documents/
                <file-cache-key>.json

    Cache invalidation:
        The GitHub file SHA is used as part of the cache key.
        If the SHA changes, the file is fetched again.
    """

    API_BASE = "https://api.github.com"
    RAW_BASE = "https://raw.githubusercontent.com"

    API_TIMEOUT = 20
    RAW_TIMEOUT = 20

    MAX_RETRIES = 2

    # Files larger than this are skipped by default.
    MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

    # Extensions that should not be sent into the text RAG pipeline.
    IGNORED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".bmp",
        ".tiff",
        ".mp3",
        ".wav",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".rar",
        ".pdf",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".onnx",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
    }

    # ---------------------------------------------------------
    # CACHE
    # ---------------------------------------------------------

    CACHE_DIR = os.path.join(
        os.path.dirname(__file__),
        "cache",
        "documents",
    )

    session = requests.Session()

    # =========================================================
    # PUBLIC ACQUISITION
    # =========================================================

    @classmethod
    def acquire(
        cls,
        files: List[Dict[str, Any]],
        github_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Acquire content for discovered GitHub files.

        Uses a persistent disk cache before making network
        requests.

        Parameters
        ----------
        files:
            Output from GitHubRepositoryIndexer.discover()

        github_token:
            Optional GitHub personal access token.

        Returns
        -------
        List[Dict[str, Any]]
            Documents containing file metadata + actual content.
        """

        documents: List[Dict[str, Any]] = []

        headers = cls._build_headers(github_token)

        total = len(files)

        cls._ensure_cache_dir()

        print(f"Acquiring content for {total} files...")

        cached_count = 0
        fetched_count = 0

        for index, file_info in enumerate(files, start=1):

            path = file_info.get("path", "")

            if not path:
                print(
                    f"[{index}/{total}] "
                    "Skipped: missing path"
                )
                continue

            if cls._should_skip_file(path, file_info):
                print(
                    f"[{index}/{total}] "
                    f"Skipped: {path}"
                )
                continue

            # -------------------------------------------------
            # 1. Try local cache
            # -------------------------------------------------

            cached_document = cls._load_cached_document(
                file_info
            )

            if cached_document is not None:

                cached_count += 1

                print(
                    f"[{index}/{total}] "
                    f"Cache hit: {path}"
                )

                documents.append(cached_document)

                continue

            # -------------------------------------------------
            # 2. Cache miss → fetch from GitHub
            # -------------------------------------------------

            fetched_count += 1

            print(
                f"[{index}/{total}] "
                f"Fetching: {path}"
            )

            content = None
            source = None

            # -------------------------------------------------
            # GitHub API
            # -------------------------------------------------

            try:

                content = cls._fetch_from_api(
                    file_info=file_info,
                    headers=headers,
                )

                if content:

                    source = "github_api"

                    print(
                        "  -> API: success"
                    )

            except Exception as exc:

                print(
                    f"  -> API failed: {exc}"
                )

            # -------------------------------------------------
            # Raw GitHub fallback
            # -------------------------------------------------

            if not content:

                try:

                    content = cls._fetch_from_raw(
                        file_info=file_info,
                        headers=headers,
                    )

                    if content:

                        source = "github_raw"

                        print(
                            "  -> Raw: success"
                        )

                except Exception as exc:

                    print(
                        f"  -> Raw failed: {exc}"
                    )

            # -------------------------------------------------
            # If both failed
            # -------------------------------------------------

            if not content:

                print(
                    "  -> Skipped: "
                    "could not acquire content"
                )

                continue

            # -------------------------------------------------
            # Normalize document
            # -------------------------------------------------

            document = {
                "content": content,
                "path": file_info.get(
                    "path",
                    "",
                ),
                "category": file_info.get(
                    "category",
                    "unknown",
                ),
                "score": file_info.get(
                    "score",
                    0,
                ),
                "sha": file_info.get(
                    "sha",
                    "",
                ),
                "size": file_info.get(
                    "size",
                    0,
                ),
                "source": source,
            }

            documents.append(document)

            # -------------------------------------------------
            # Save to cache
            # -------------------------------------------------

            try:

                cls._save_cached_document(
                    file_info=file_info,
                    document=document,
                )

                print(
                    "  -> Saved to cache"
                )

            except Exception as exc:

                # Cache failure should NEVER break
                # the acquisition pipeline.
                print(
                    f"  -> Cache save failed: {exc}"
                )

        print()

        print(
            f"Successfully acquired "
            f"{len(documents)}/{total} files."
        )

        print(
            f"Cache hits: {cached_count}"
        )

        print(
            f"Network fetches: {fetched_count}"
        )

        return documents

    # =========================================================
    # CACHE HELPERS
    # =========================================================

    @classmethod
    def _ensure_cache_dir(cls) -> None:
        """
        Create the document cache directory.
        """

        os.makedirs(
            cls.CACHE_DIR,
            exist_ok=True,
        )

    @classmethod
    def _get_cache_key(
        cls,
        file_info: Dict[str, Any],
    ) -> str:
        """
        Generate a stable cache key for a GitHub file.

        SHA is preferred because it identifies the exact
        GitHub blob version.

        If SHA is unavailable, fall back to path + URL +
        branch.
        """

        path = str(
            file_info.get(
                "path",
                "",
            )
        )

        sha = str(
            file_info.get(
                "sha",
                "",
            )
        )

        url = str(
            file_info.get(
                "url",
                "",
            )
        )

        branch = str(
            file_info.get(
                "branch",
                "main",
            )
        )

        identity = (
            f"path={path}|"
            f"sha={sha}|"
            f"url={url}|"
            f"branch={branch}"
        )

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    @classmethod
    def _get_cache_path(
        cls,
        file_info: Dict[str, Any],
    ) -> str:
        """
        Return the JSON cache path for a file.
        """

        cache_key = cls._get_cache_key(
            file_info
        )

        return os.path.join(
            cls.CACHE_DIR,
            f"{cache_key}.json",
        )

    @classmethod
    def _load_cached_document(
        cls,
        file_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Load a cached document if one exists.

        Returns None when:
            - no cache exists
            - cache is corrupted
            - cached metadata does not match
        """

        cache_path = cls._get_cache_path(
            file_info
        )

        if not os.path.exists(cache_path):
            return None

        try:

            with open(
                cache_path,
                "r",
                encoding="utf-8",
            ) as file:

                document = json.load(file)

            if not isinstance(
                document,
                dict,
            ):
                return None

            content = document.get(
                "content"
            )

            if not isinstance(
                content,
                str,
            ):
                return None

            # -------------------------------------------------
            # Verify important metadata
            # -------------------------------------------------

            expected_path = file_info.get(
                "path",
                "",
            )

            cached_path = document.get(
                "path",
                "",
            )

            if expected_path != cached_path:
                return None

            expected_sha = file_info.get(
                "sha",
                "",
            )

            cached_sha = document.get(
                "sha",
                "",
            )

            # If both SHAs are available they MUST match.
            if expected_sha and cached_sha:

                if expected_sha != cached_sha:
                    return None

            # -------------------------------------------------
            # Update metadata that may change between runs
            # -------------------------------------------------

            document["category"] = file_info.get(
                "category",
                document.get(
                    "category",
                    "unknown",
                ),
            )

            document["score"] = file_info.get(
                "score",
                document.get(
                    "score",
                    0,
                ),
            )

            document["size"] = file_info.get(
                "size",
                document.get(
                    "size",
                    0,
                ),
            )

            # Clearly mark that this came from cache.
            document["source"] = "cache"

            return document

        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):

            # Corrupt cache → ignore it and fetch again.
            return None

    @classmethod
    def _save_cached_document(
        cls,
        file_info: Dict[str, Any],
        document: Dict[str, Any],
    ) -> None:
        """
        Save a document to disk as JSON.

        Uses atomic replacement so an interrupted write
        does not normally leave a half-written cache file.
        """

        cls._ensure_cache_dir()

        cache_path = cls._get_cache_path(
            file_info
        )

        temp_path = (
            cache_path
            + ".tmp"
        )

        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                document,
                file,
                ensure_ascii=False,
            )

        os.replace(
            temp_path,
            cache_path,
        )

    # =========================================================
    # GitHub API
    # =========================================================

    @classmethod
    def _fetch_from_api(
        cls,
        file_info: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Optional[str]:

        url = file_info.get(
            "url"
        )

        if not url:

            raise ValueError(
                "No GitHub API URL available"
            )

        response = cls._request_with_retry(
            url=url,
            headers=headers,
            timeout=cls.API_TIMEOUT,
        )

        if response.status_code != 200:

            raise RuntimeError(
                f"HTTP {response.status_code}"
            )

        data = response.json()

        # GitHub blob API normally returns:
        #
        # {
        #     "content": "...base64...",
        #     "encoding": "base64"
        # }

        encoded_content = data.get(
            "content"
        )

        if not encoded_content:

            raise ValueError(
                "GitHub API response "
                "contained no content"
            )

        encoding = data.get(
            "encoding",
            "",
        )

        if encoding != "base64":

            raise ValueError(
                f"Unsupported encoding: {encoding}"
            )

        try:

            decoded = base64.b64decode(
                encoded_content
            )

            return decoded.decode(
                "utf-8",
                errors="replace",
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to decode API content: "
                f"{exc}"
            )

    # =========================================================
    # Raw GitHub fallback
    # =========================================================

    @classmethod
    def _fetch_from_raw(
        cls,
        file_info: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Optional[str]:

        raw_url = cls._build_raw_url(
            file_info
        )

        if not raw_url:

            raise ValueError(
                "Could not construct "
                "raw GitHub URL"
            )

        response = cls._request_with_retry(
            url=raw_url,
            headers=headers,
            timeout=cls.RAW_TIMEOUT,
        )

        if response.status_code != 200:

            raise RuntimeError(
                f"HTTP {response.status_code}"
            )

        return response.text

    # =========================================================
    # Raw URL construction
    # =========================================================

    @classmethod
    def _build_raw_url(
        cls,
        file_info: Dict[str, Any],
    ) -> Optional[str]:

        path = file_info.get(
            "path"
        )

        if not path:
            return None

        api_url = file_info.get(
            "url",
            "",
        )

        # Example:
        #
        # https://api.github.com/repos/
        # NVIDIA/Megatron-LM/git/blobs/SHA
        #
        # We need repository owner/name
        # + branch.

        if "/repos/" not in api_url:
            return None

        try:

            repository_part = (
                api_url.split(
                    "/repos/",
                    1,
                )[1]
            )

            parts = repository_part.split(
                "/"
            )

            if len(parts) < 2:
                return None

            owner = parts[0]
            repo = parts[1]

        except Exception:

            return None

        branch = file_info.get(
            "branch",
            "main",
        )

        return (
            f"{cls.RAW_BASE}/"
            f"{owner}/"
            f"{repo}/"
            f"{branch}/"
            f"{path}"
        )

    # =========================================================
    # HTTP retry logic
    # =========================================================

    @classmethod
    def _request_with_retry(
        cls,
        url: str,
        headers: Dict[str, str],
        timeout: int,
    ) -> requests.Response:

        last_exception = None

        for attempt in range(
            cls.MAX_RETRIES + 1
        ):

            try:

                response = cls.session.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                )

                # Retry temporary server/rate-limit errors.
                if response.status_code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:

                    if attempt < cls.MAX_RETRIES:

                        delay = 2 ** attempt

                        print(
                            f"    Retrying in "
                            f"{delay}s..."
                        )

                        time.sleep(
                            delay
                        )

                        continue

                return response

            except requests.RequestException as exc:

                last_exception = exc

                if attempt < cls.MAX_RETRIES:

                    delay = 2 ** attempt

                    print(
                        f"    Request error. "
                        f"Retrying in {delay}s..."
                    )

                    time.sleep(
                        delay
                    )

                else:

                    raise

        if last_exception:
            raise last_exception

        raise RuntimeError(
            "Request failed unexpectedly"
        )

    # =========================================================
    # Headers
    # =========================================================

    @classmethod
    def _build_headers(
        cls,
        github_token: Optional[str],
    ) -> Dict[str, str]:

        headers = {
            "Accept": (
                "application/vnd.github+json"
            ),
            "User-Agent": (
                "Smart-Research-Dashboard"
            ),
        }

        if github_token:

            headers["Authorization"] = (
                f"Bearer {github_token}"
            )

        return headers

    # =========================================================
    # File filtering
    # =========================================================

    @classmethod
    def _should_skip_file(
        cls,
        path: str,
        file_info: Dict[str, Any],
    ) -> bool:

        # -------------------------------------------------
        # Size protection
        # -------------------------------------------------

        size = file_info.get(
            "size",
            0,
        )

        try:

            size = int(size)

        except (
            TypeError,
            ValueError,
        ):

            size = 0

        if size > cls.MAX_FILE_SIZE:
            return True

        # -------------------------------------------------
        # Extension filtering
        # -------------------------------------------------

        path_lower = path.lower()

        for extension in cls.IGNORED_EXTENSIONS:

            if path_lower.endswith(
                extension
            ):

                return True

        return False
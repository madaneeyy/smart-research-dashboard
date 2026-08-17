"""
GitHub Repository Cache
=======================

Two-level cache for GitHub repository data:

    L1: In-memory cache
        ↓
    L2: Persistent disk cache
        ↓
    GitHub API

The cache stores repository-level data, NOT query-specific answers.

Cached:
    - Repository metadata
    - Repository tree
    - Individual file contents

Not cached:
    - User questions
    - Final LLM answers
    - Query-specific retrieved context

This allows multiple different questions against the same repository
without repeatedly downloading the repository from GitHub.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

from pathlib import Path
from typing import Any, Optional


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_ROOT = (
    PROJECT_ROOT
    / "cache"
    / "github"
)

# Repository data remains valid for 24 hours by default.
DEFAULT_TTL_SECONDS = 24 * 60 * 60

# Prevent accidentally caching enormous individual files.
DEFAULT_MAX_FILE_SIZE = 5 * 1024 * 1024


# ============================================================
# CACHE
# ============================================================


class GitHubCache:
    """
    Two-level GitHub repository cache.

    Level 1:
        In-memory dictionaries.

    Level 2:
        Persistent JSON/text files on disk.

    Example:

        cache = GitHubCache()

        tree = cache.get_tree(
            "https://github.com/scikit-learn/scikit-learn",
            "main",
        )

        if tree is None:
            # Fetch from GitHub
            tree = github_tree

            cache.set_tree(
                repo_url,
                branch,
                tree,
            )
    """

    def __init__(
        self,
        cache_root: Optional[Path | str] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ) -> None:

        self.cache_root = Path(
            cache_root
            if cache_root is not None
            else CACHE_ROOT
        )

        self.ttl_seconds = int(ttl_seconds)
        self.max_file_size = int(max_file_size)

        # ----------------------------------------------------
        # L1 MEMORY CACHE
        # ----------------------------------------------------

        self._tree_memory: dict[str, Any] = {}

        self._file_memory: dict[str, str] = {}

        self._metadata_memory: dict[str, dict[str, Any]] = {}

        # ----------------------------------------------------
        # THREAD SAFETY
        # ----------------------------------------------------

        self._lock = threading.RLock()

        # ----------------------------------------------------
        # CREATE CACHE DIRECTORY
        # ----------------------------------------------------

        self.cache_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # REPOSITORY IDENTIFICATION
    # ========================================================

    @staticmethod
    def normalize_repo_url(
        repo_url: str,
    ) -> str:
        """
        Normalize common GitHub repository URL formats.

        Examples:

            https://github.com/user/repo

            https://github.com/user/repo/

            https://github.com/user/repo.git

        all become:

            https://github.com/user/repo
        """

        if not repo_url:
            raise ValueError(
                "GitHub repository URL cannot be empty."
            )

        url = repo_url.strip()

        url = url.rstrip("/")

        if url.endswith(".git"):
            url = url[:-4]

        # Remove trailing whitespace again after .git.
        url = url.rstrip("/")

        return url

    @classmethod
    def repository_key(
        cls,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> str:
        """
        Generate a deterministic filesystem-safe repository key.

        Example:

            scikit-learn__scikit-learn__main__a1b2c3...
        """

        normalized = cls.normalize_repo_url(
            repo_url
        )

        branch_value = (
            branch.strip()
            if branch
            else "default"
        )

        # ----------------------------------------------------
        # Extract owner/repository for readability.
        # ----------------------------------------------------

        match = re.search(
            r"github\.com/([^/]+)/([^/]+)$",
            normalized,
            re.IGNORECASE,
        )

        if match:
            owner = match.group(1)
            repository = match.group(2)

            readable = (
                f"{owner}__{repository}"
            )
        else:
            readable = "github_repo"

        # ----------------------------------------------------
        # Hash prevents collisions.
        # ----------------------------------------------------

        digest_source = (
            f"{normalized}|{branch_value}"
        )

        digest = hashlib.sha256(
            digest_source.encode("utf-8")
        ).hexdigest()[:16]

        safe_branch = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "_",
            branch_value,
        )

        return (
            f"{readable}"
            f"__{safe_branch}"
            f"__{digest}"
        )

    # ========================================================
    # REPOSITORY DIRECTORY
    # ========================================================

    def repository_dir(
        self,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> Path:

        key = self.repository_key(
            repo_url,
            branch,
        )

        return self.cache_root / key

    # ========================================================
    # FILE PATHS
    # ========================================================

    def metadata_path(
        self,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> Path:

        return (
            self.repository_dir(
                repo_url,
                branch,
            )
            / "metadata.json"
        )

    def tree_path(
        self,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> Path:

        return (
            self.repository_dir(
                repo_url,
                branch,
            )
            / "tree.json"
        )

    def files_dir(
        self,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> Path:

        return (
            self.repository_dir(
                repo_url,
                branch,
            )
            / "files"
        )

    def file_cache_path(
        self,
        repo_url: str,
        file_path: str,
        branch: Optional[str] = None,
    ) -> Path:

        normalized_path = (
            file_path
            .replace("\\", "/")
            .strip("/")
        )

        # Hash the actual path to avoid filesystem problems
        # with characters such as ":" or very long paths.
        path_hash = hashlib.sha256(
            normalized_path.encode("utf-8")
        ).hexdigest()[:16]

        readable = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "_",
            normalized_path,
        )

        # Keep filenames manageable.
        readable = readable[-120:]

        filename = (
            f"{readable}"
            f"__{path_hash}.txt"
        )

        return (
            self.files_dir(
                repo_url,
                branch,
            )
            / filename
        )

    # ========================================================
    # TTL
    # ========================================================

    def _is_fresh(
        self,
        path: Path,
    ) -> bool:

        if not path.exists():
            return False

        try:
            age = (
                time.time()
                - path.stat().st_mtime
            )

            return age <= self.ttl_seconds

        except OSError:
            return False

    # ========================================================
    # JSON UTILITIES
    # ========================================================

    @staticmethod
    def _read_json(
        path: Path,
    ) -> Optional[Any]:

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as f:

                return json.load(f)

        except (
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return None

    @staticmethod
    def _write_json_atomic(
        path: Path,
        data: Any,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Atomic replacement.
        os.replace(
            temporary,
            path,
        )

    # ========================================================
    # TEXT UTILITIES
    # ========================================================

    @staticmethod
    def _read_text(
        path: Path,
    ) -> Optional[str]:

        try:

            return path.read_text(
                encoding="utf-8"
            )

        except (
            OSError,
            UnicodeDecodeError,
        ):
            return None

    @staticmethod
    def _write_text_atomic(
        path: Path,
        content: str,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary.write_text(
            content,
            encoding="utf-8",
        )

        os.replace(
            temporary,
            path,
        )

    # ========================================================
    # METADATA
    # ========================================================

    def get_metadata(
        self,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:

        key = self.repository_key(
            repo_url,
            branch,
        )

        with self._lock:

            # L1
            memory_value = (
                self._metadata_memory.get(key)
            )

            if memory_value is not None:
                return memory_value

            # L2
            path = self.metadata_path(
                repo_url,
                branch,
            )

            if not self._is_fresh(path):
                return None

            value = self._read_json(path)

            if not isinstance(
                value,
                dict,
            ):
                return None

            self._metadata_memory[key] = value

            return value

    def set_metadata(
        self,
        repo_url: str,
        metadata: dict[str, Any],
        branch: Optional[str] = None,
    ) -> None:

        key = self.repository_key(
            repo_url,
            branch,
        )

        metadata = dict(metadata)

        metadata.setdefault(
            "repo_url",
            self.normalize_repo_url(repo_url),
        )

        metadata.setdefault(
            "branch",
            branch or "default",
        )

        metadata["cached_at"] = (
            time.time()
        )

        with self._lock:

            self._metadata_memory[key] = metadata

            self._write_json_atomic(
                self.metadata_path(
                    repo_url,
                    branch,
                ),
                metadata,
            )

    # ========================================================
    # TREE CACHE
    # ========================================================

    def get_tree(
        self,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Return cached repository tree.

        Returns None on cache miss.
        """

        key = self.repository_key(
            repo_url,
            branch,
        )

        with self._lock:

            # ------------------------------------------------
            # L1
            # ------------------------------------------------

            if key in self._tree_memory:

                return self._tree_memory[key]

            # ------------------------------------------------
            # L2
            # ------------------------------------------------

            path = self.tree_path(
                repo_url,
                branch,
            )

            if not self._is_fresh(path):
                return None

            tree = self._read_json(path)

            if tree is None:
                return None

            self._tree_memory[key] = tree

            return tree

    def set_tree(
        self,
        repo_url: str,
        tree: Any,
        branch: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Store repository tree in memory and on disk.
        """

        key = self.repository_key(
            repo_url,
            branch,
        )

        with self._lock:

            # L1
            self._tree_memory[key] = tree

            # L2
            self._write_json_atomic(
                self.tree_path(
                    repo_url,
                    branch,
                ),
                tree,
            )

            # Optional metadata.
            if metadata is not None:

                self.set_metadata(
                    repo_url,
                    metadata,
                    branch,
                )

    # ========================================================
    # FILE CACHE
    # ========================================================

    def get_file(
        self,
        repo_url: str,
        file_path: str,
        branch: Optional[str] = None,
    ) -> Optional[str]:
        """
        Return cached file contents.

        Returns None on cache miss.
        """

        key = (
            self.repository_key(
                repo_url,
                branch,
            )
            + "::"
            + file_path
        )

        with self._lock:

            # ------------------------------------------------
            # L1
            # ------------------------------------------------

            if key in self._file_memory:

                return self._file_memory[key]

            # ------------------------------------------------
            # L2
            # ------------------------------------------------

            path = self.file_cache_path(
                repo_url,
                file_path,
                branch,
            )

            if not self._is_fresh(path):
                return None

            content = self._read_text(path)

            if content is None:
                return None

            self._file_memory[key] = content

            return content

    def set_file(
        self,
        repo_url: str,
        file_path: str,
        content: str,
        branch: Optional[str] = None,
    ) -> bool:
        """
        Store file contents.

        Returns:
            True  -> cached successfully
            False -> skipped because file is too large
        """

        if content is None:
            return False

        if not isinstance(
            content,
            str,
        ):
            content = str(content)

        size = len(
            content.encode("utf-8")
        )

        if size > self.max_file_size:
            return False

        key = (
            self.repository_key(
                repo_url,
                branch,
            )
            + "::"
            + file_path
        )

        with self._lock:

            # L1
            self._file_memory[key] = content

            # L2
            self._write_text_atomic(
                self.file_cache_path(
                    repo_url,
                    file_path,
                    branch,
                ),
                content,
            )

        return True

    # ========================================================
    # CACHE STATUS
    # ========================================================

    def has_repository(
        self,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> bool:
        """
        True when a fresh repository tree exists.
        """

        return (
            self.get_tree(
                repo_url,
                branch,
            )
            is not None
        )

    def cache_age_seconds(
        self,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> Optional[float]:
        """
        Return repository tree cache age in seconds.
        """

        path = self.tree_path(
            repo_url,
            branch,
        )

        if not path.exists():
            return None

        try:

            return max(
                0.0,
                time.time()
                - path.stat().st_mtime,
            )

        except OSError:
            return None

    # ========================================================
    # INVALIDATION
    # ========================================================

    def invalidate_repository(
        self,
        repo_url: str,
        branch: Optional[str] = None,
    ) -> None:
        """
        Completely remove one repository from cache.
        """

        key = self.repository_key(
            repo_url,
            branch,
        )

        with self._lock:

            # Remove L1 tree.
            self._tree_memory.pop(
                key,
                None,
            )

            # Remove L1 metadata.
            self._metadata_memory.pop(
                key,
                None,
            )

            # Remove L1 files.
            prefix = key + "::"

            file_keys = [
                k
                for k in self._file_memory
                if k.startswith(prefix)
            ]

            for file_key in file_keys:
                self._file_memory.pop(
                    file_key,
                    None,
                )

            # Remove L2.
            directory = self.repository_dir(
                repo_url,
                branch,
            )

            if directory.exists():

                import shutil

                shutil.rmtree(
                    directory,
                    ignore_errors=True,
                )

    def invalidate_all(self) -> None:
        """
        Clear the entire GitHub cache.
        """

        with self._lock:

            self._tree_memory.clear()

            self._file_memory.clear()

            self._metadata_memory.clear()

            if self.cache_root.exists():

                import shutil

                shutil.rmtree(
                    self.cache_root,
                    ignore_errors=True,
                )

            self.cache_root.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    def stats(self) -> dict[str, Any]:
        """
        Return cache statistics useful during development.
        """

        repositories = 0

        try:

            if self.cache_root.exists():

                repositories = sum(
                    1
                    for path
                    in self.cache_root.iterdir()
                    if path.is_dir()
                )

        except OSError:
            repositories = 0

        return {
            "cache_root": str(
                self.cache_root
            ),
            "repositories_on_disk": repositories,
            "memory_trees": len(
                self._tree_memory
            ),
            "memory_files": len(
                self._file_memory
            ),
            "memory_metadata": len(
                self._metadata_memory
            ),
            "ttl_seconds": self.ttl_seconds,
            "ttl_hours": round(
                self.ttl_seconds / 3600,
                2,
            ),
        }


# ============================================================
# SINGLETON
# ============================================================

# Use this from github_content_v2.py:
#
#     from src.services.github_cache import github_cache
#
# This means the entire application shares the same L1 cache.
github_cache = GitHubCache()
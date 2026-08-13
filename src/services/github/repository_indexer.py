import base64
import json
import os
import re
import time
from urllib.parse import quote, urlparse

import requests
from dotenv import load_dotenv


load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


class GitHubRepositoryIndexer:
    """
    Repository-agnostic GitHub repository discovery and indexing.

    Pipeline:

        GitHub repository
            ↓
        repository tree
            ↓
        file filtering
            ↓
        file classification
            ↓
        relevance scoring
            ↓
        balanced file selection
            ↓
        file content retrieval

    The resulting documents can then be passed to the
    RAG chunking and embedding pipeline.
    """

    API_BASE_URL = "https://api.github.com"

    # =========================================================
    # PERSISTENT CACHE
    # =========================================================

    CACHE_DIR = os.path.join(
        os.path.dirname(__file__),
        "cache",
        "github",
    )

    TREE_CACHE_TTL_SECONDS = int(
        os.getenv("GITHUB_TREE_CACHE_TTL", "300")
    )

    CACHE_VERSION = 1

    _session = None

    # =========================================================
    # LIMITS
    # =========================================================

    MAX_FILES = 120

    MAX_FILE_SIZE = 300_000

    # =========================================================
    # DOCUMENTATION FILES
    # =========================================================

    CORE_DOCUMENTATION_FILES = {
        "readme.md",
        "readme.rst",
        "readme.txt",
        "architecture.md",
        "documentation.md",
    }

    SECONDARY_DOCUMENTATION_FILES = {
        "guide.md",
        "guides.md",
        "tutorial.md",
        "tutorials.md",
        "api.md",
        "design.md",
        "overview.md",
    }

    LOW_PRIORITY_DOCUMENTATION_FILES = {
        "contributing.md",
        "contribute.md",
        "security.md",
        "agents.md",
        "claude.md",
        "code_of_conduct.md",
        "support.md",
    }

    # =========================================================
    # CONFIGURATION FILES
    # =========================================================

    CONFIGURATION_FILES = {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "requirements-dev.txt",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "cargo.toml",
        "go.mod",
        "go.sum",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "pubspec.yaml",
        "composer.json",
        "gemfile",
        "dockerfile",
        "makefile",
        "cmakelists.txt",
        "environment.yml",
        "environment.yaml",
    }

    # =========================================================
    # LOW-VALUE CONFIGURATION
    # =========================================================

    LOW_VALUE_CONFIG_FILES = {
        ".pre-commit-config.yaml",
        ".gitlab-ci.yml",
        ".travis.yml",
        "codecov.yml",
        "greptile.json",
        "renovate.json",
        "dependabot.yml",
    }

    LOW_VALUE_CONFIG_DIRECTORIES = {
        ".claude",
        ".github",
        ".gitlab",
    }

    # =========================================================
    # DIRECTORY TYPES
    # =========================================================

    DOCUMENTATION_DIRECTORIES = {
        "docs",
        "doc",
        "documentation",
        "wiki",
        "guides",
    }

    EXAMPLE_DIRECTORIES = {
        "examples",
        "example",
        "samples",
        "sample",
        "tutorials",
        "tutorial",
    }

    SOURCE_DIRECTORIES = {
        "src",
        "lib",
        "app",
        "apps",
        "packages",
        "core",
        "server",
        "client",
        "backend",
        "frontend",
        "cmd",
        "internal",
        "modules",
        "components",
        "services",
    }

    TEST_DIRECTORIES = {
        "test",
        "tests",
        "__tests__",
        "testing",
        "functional_tests",
        "integration_tests",
    }

    # =========================================================
    # SOURCE EXTENSIONS
    # =========================================================

    SOURCE_EXTENSIONS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".kt",
        ".kts",
        ".go",
        ".rs",
        ".cpp",
        ".cc",
        ".cxx",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".swift",
        ".dart",
        ".rb",
        ".php",
        ".scala",
        ".sh",
        ".bash",
        ".sql",
    }

    # =========================================================
    # DOCUMENTATION EXTENSIONS
    # =========================================================

    DOCUMENTATION_EXTENSIONS = {
        ".md",
        ".mdx",
        ".rst",
        ".txt",
    }

    # =========================================================
    # CONFIGURATION EXTENSIONS
    # =========================================================

    CONFIGURATION_EXTENSIONS = {
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".xml",
        ".ini",
        ".cfg",
        ".conf",
    }

    # =========================================================
    # IGNORED DIRECTORIES
    # =========================================================

    IGNORED_DIRECTORIES = {
        ".git",
        ".idea",
        ".vscode",
        "node_modules",
        "vendor",
        "dist",
        "build",
        "target",
        "coverage",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".venv",
        "venv",
        "env",
        ".tox",
        "bin",
        "obj",
        ".next",
        ".dart_tool",
        ".gradle",
    }

    # =========================================================
    # IGNORED FILE EXTENSIONS
    # =========================================================

    IGNORED_FILE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".svg",
        ".mp4",
        ".mov",
        ".avi",
        ".mp3",
        ".wav",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".pdf",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
    }

    # =========================================================
    # CATEGORY LIMITS
    # =========================================================

    CATEGORY_LIMITS = {
        "documentation": 30,
        "configuration": 15,
        "source": 55,
        "examples": 20,
    }

    # =========================================================
    # HEADERS
    # =========================================================

    @staticmethod
    def _headers() -> dict[str, str]:
        """
        Build GitHub API request headers.
        """

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Smart-Research-Dashboard",
        }

        if GITHUB_TOKEN:
            headers["Authorization"] = (
                f"Bearer {GITHUB_TOKEN}"
            )

        return headers

    @classmethod
    def _get_session(cls) -> requests.Session:
        """Return a reusable HTTP session."""
        if cls._session is None:
            cls._session = requests.Session()
            cls._session.headers.update(cls._headers())
        return cls._session

    @classmethod
    def _request(cls, method: str, url: str, **kwargs):
        """Make an authenticated GitHub API request."""
        kwargs.setdefault("timeout", 30)
        response = cls._get_session().request(method, url, **kwargs)

        if response.status_code == 401:
            raise RuntimeError(
                "GitHub authentication failed. Check GITHUB_TOKEN in your .env file."
            )

        if response.status_code == 403 and "rate limit" in response.text.lower():
            raise RuntimeError(
                "GitHub API rate limit exceeded. Set a valid GITHUB_TOKEN in your .env file."
            )

        response.raise_for_status()
        return response

    @classmethod
    def _tree_cache_path(cls, owner: str, repository: str) -> str:
        """Return the persistent cache path for one repository tree."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        safe_name = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            f"{owner}__{repository}",
        )
        return os.path.join(
            cls.CACHE_DIR,
            f"tree_{safe_name}.json",
        )

    @classmethod
    def _load_tree_cache(cls, owner: str, repository: str):
        """Load a recent tree cache entry, if available."""
        path = cls._tree_cache_path(owner, repository)

        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as file:
                cached = json.load(file)

            if cached.get("version") != cls.CACHE_VERSION:
                return None

            created_at = float(cached.get("created_at", 0))
            if time.time() - created_at > cls.TREE_CACHE_TTL_SECONDS:
                return None

            tree = cached.get("tree")
            if not isinstance(tree, list):
                return None

            return tree

        except (OSError, ValueError, TypeError):
            return None

    @classmethod
    def _save_tree_cache(cls, owner: str, repository: str, tree: list[dict], commit_sha: str | None = None):
        """Persist the repository tree to disk."""
        path = cls._tree_cache_path(owner, repository)
        temporary_path = f"{path}.tmp"

        payload = {
            "version": cls.CACHE_VERSION,
            "created_at": time.time(),
            "owner": owner,
            "repository": repository,
            "commit_sha": commit_sha,
            "tree": tree,
        }

        try:
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)
            os.replace(temporary_path, path)
        except OSError:
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass

    @classmethod
    def clear_tree_cache(cls, github_url: str | None = None) -> None:
        """Clear all tree caches or the cache for one repository."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

        if github_url is None:
            for filename in os.listdir(cls.CACHE_DIR):
                if filename.startswith("tree_") and filename.endswith(".json"):
                    try:
                        os.remove(os.path.join(cls.CACHE_DIR, filename))
                    except OSError:
                        pass
            return

        owner, repository = cls._parse_github_url(github_url)
        path = cls._tree_cache_path(owner, repository)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    # =========================================================
    # PARSE GITHUB URL
    # =========================================================

    @staticmethod
    def _parse_github_url(
        github_url: str,
    ) -> tuple[str, str]:
        """
        Extract owner and repository from a GitHub URL.

        Supports:

            https://github.com/owner/repo
            https://github.com/owner/repo.git
        """

        parsed = urlparse(
            str(github_url).strip()
        )

        if parsed.netloc.lower() not in {
            "github.com",
            "www.github.com",
        }:
            raise ValueError(
                "Expected a GitHub repository URL."
            )

        parts = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        if len(parts) < 2:
            raise ValueError(
                "Invalid GitHub repository URL."
            )

        owner = parts[0]

        repository = re.sub(
            r"\.git$",
            "",
            parts[1],
            flags=re.IGNORECASE,
        )

        return owner, repository

    # =========================================================
    # FETCH REPOSITORY TREE
    # =========================================================

    @classmethod
    def fetch_tree(
        cls,
        github_url: str,
    ) -> list[dict]:
        """
        Fetch the complete recursive repository tree.

        The recursive tree is persisted on disk for
        ``TREE_CACHE_TTL_SECONDS`` (5 minutes by default), so repeated
        runs do not repeatedly hit the expensive Git tree endpoint.
        """

        owner, repository = cls._parse_github_url(github_url)

        cached_tree = cls._load_tree_cache(owner, repository)
        if cached_tree is not None:
            print(f"Tree cache hit: {owner}/{repository}")
            return cached_tree

        print(f"Fetching repository tree: {owner}/{repository}")

        repo_url = f"{cls.API_BASE_URL}/repos/{owner}/{repository}"
        response = cls._request("GET", repo_url, timeout=30)
        repo_data = response.json()

        default_branch = repo_data.get("default_branch")
        if not default_branch:
            raise ValueError("Could not determine repository default branch.")

        encoded_branch = quote(default_branch, safe="")
        ref_url = (
            f"{cls.API_BASE_URL}/repos/{owner}/{repository}"
            f"/git/ref/heads/{encoded_branch}"
        )

        response = cls._request("GET", ref_url, timeout=30)
        ref_data = response.json()
        commit_sha = ref_data.get("object", {}).get("sha")

        if not commit_sha:
            raise ValueError("Could not determine repository commit.")

        tree_url = (
            f"{cls.API_BASE_URL}/repos/{owner}/{repository}"
            f"/git/trees/{commit_sha}"
        )

        response = cls._request(
            "GET",
            tree_url,
            params={"recursive": "1"},
            timeout=60,
        )

        data = response.json()
        tree = data.get("tree", [])

        cls._save_tree_cache(
            owner,
            repository,
            tree,
            commit_sha=commit_sha,
        )

        print(f"Tree cached: {len(tree)} entries")
        return tree

    # =========================================================
    # FILE EXTENSION
    # =========================================================

    @staticmethod
    def _file_extension(
        path: str,
    ) -> str:
        """
        Return lowercase file extension.
        """

        filename = (
            path
            .rsplit("/", 1)[-1]
            .lower()
        )

        if "." not in filename:
            return ""

        return (
            "."
            + filename.rsplit(
                ".",
                1,
            )[-1]
        )

    # =========================================================
    # PATH PARTS
    # =========================================================

    @staticmethod
    def _path_parts(
        path: str,
    ) -> list[str]:
        """
        Split a repository path into lowercase components.
        """

        return [
            part.lower()
            for part in path.replace(
                "\\",
                "/",
            ).split("/")
            if part
        ]

    # =========================================================
    # CHECK IGNORED PATH
    # =========================================================

    @classmethod
    def _is_ignored_path(
        cls,
        path: str,
    ) -> bool:

        parts = cls._path_parts(
            path
        )

        return any(
            part in cls.IGNORED_DIRECTORIES
            for part in parts
        )

    # =========================================================
    # CHECK TEST PATH
    # =========================================================

    @classmethod
    def _is_test_path(
        cls,
        path: str,
    ) -> bool:

        parts = cls._path_parts(
            path
        )

        return any(
            part in cls.TEST_DIRECTORIES
            for part in parts
        )

    # =========================================================
    # CHECK EXAMPLE PATH
    # =========================================================

    @classmethod
    def _is_example_path(
        cls,
        path: str,
    ) -> bool:

        parts = cls._path_parts(
            path
        )

        return any(
            directory in cls.EXAMPLE_DIRECTORIES
            for directory in parts
        )

    # =========================================================
    # LOW-VALUE CONFIGURATION
    # =========================================================

    @classmethod
    def _is_low_value_configuration(
        cls,
        path: str,
    ) -> bool:
        """
        Identify configuration files that usually provide
        little value for repository question answering.
        """

        normalized = path.replace(
            "\\",
            "/",
        )

        filename = (
            normalized
            .rsplit("/", 1)[-1]
            .lower()
        )

        parts = cls._path_parts(
            normalized
        )

        # -----------------------------------------------------
        # Known CI / automation files
        # -----------------------------------------------------

        if filename in cls.LOW_VALUE_CONFIG_FILES:
            return True

        # -----------------------------------------------------
        # Editor / automation directories
        # -----------------------------------------------------

        if any(
            directory
            in cls.LOW_VALUE_CONFIG_DIRECTORIES
            for directory in parts
        ):
            return True

        # -----------------------------------------------------
        # Documentation metadata JSON
        # -----------------------------------------------------

        extension = cls._file_extension(
            normalized
        )

        if (
            extension == ".json"
            and "docs" in parts
            and filename not in {
                "package.json",
            }
        ):
            return True

        return False

    # =========================================================
    # CLASSIFY FILE
    # =========================================================

    @classmethod
    def classify_file(
        cls,
        path: str,
    ) -> str:
        """
        Classify a repository file.

        Important ordering:

            1. Known documentation
            2. Known configuration
            3. Actual source extension
            4. Documentation extension
            5. Configuration extension
            6. Directory fallback

        This prevents:

            docs/example.py

        from incorrectly becoming documentation.
        """

        normalized = path.replace(
            "\\",
            "/",
        )

        filename = (
            normalized
            .rsplit("/", 1)[-1]
            .lower()
        )

        parts = cls._path_parts(
            normalized
        )

        extension = cls._file_extension(
            normalized
        )

        # -----------------------------------------------------
        # Core documentation
        # -----------------------------------------------------

        if filename in cls.CORE_DOCUMENTATION_FILES:

            if cls._is_example_path(
                normalized
            ):
                return "examples"

            return "documentation"

        # -----------------------------------------------------
        # Secondary documentation
        # -----------------------------------------------------

        if filename in cls.SECONDARY_DOCUMENTATION_FILES:

            if cls._is_example_path(
                normalized
            ):
                return "examples"

            return "documentation"

        # -----------------------------------------------------
        # Low-priority admin documentation
        # -----------------------------------------------------

        if filename in cls.LOW_PRIORITY_DOCUMENTATION_FILES:
            return "other"

        # -----------------------------------------------------
        # Known configuration
        # -----------------------------------------------------

        if filename in cls.CONFIGURATION_FILES:
            return "configuration"

        # -----------------------------------------------------
        # SOURCE CODE FIRST
        # -----------------------------------------------------

        if extension in cls.SOURCE_EXTENSIONS:

            if cls._is_example_path(
                normalized
            ):
                return "examples"

            return "source"

        # -----------------------------------------------------
        # Documentation files
        # -----------------------------------------------------

        if extension in cls.DOCUMENTATION_EXTENSIONS:

            if cls._is_example_path(
                normalized
            ):
                return "examples"

            return "documentation"

        # -----------------------------------------------------
        # Configuration extensions
        # -----------------------------------------------------

        if extension in cls.CONFIGURATION_EXTENSIONS:
            return "configuration"

        # -----------------------------------------------------
        # Example directory fallback
        # -----------------------------------------------------

        if cls._is_example_path(
            normalized
        ):
            return "examples"

        # -----------------------------------------------------
        # Documentation directory fallback
        # -----------------------------------------------------

        if any(
            directory
            in cls.DOCUMENTATION_DIRECTORIES
            for directory in parts
        ):
            return "documentation"

        # -----------------------------------------------------
        # Source directory fallback
        # -----------------------------------------------------

        if any(
            directory
            in cls.SOURCE_DIRECTORIES
            for directory in parts
        ):
            return "source"

        return "other"

    # =========================================================
    # SCORE FILE
    # =========================================================

    @classmethod
    def score_file(
        cls,
        path: str,
        size: int | None = None,
    ) -> int:
        """
        Calculate a generic repository relevance score.

        Higher score means the file is more likely to help
        answer questions about the repository.
        """

        normalized = path.replace(
            "\\",
            "/",
        )

        filename = (
            normalized
            .rsplit("/", 1)[-1]
            .lower()
        )

        parts = cls._path_parts(
            normalized
        )

        extension = cls._file_extension(
            normalized
        )

        category = cls.classify_file(
            normalized
        )

        # -----------------------------------------------------
        # Base category scores
        # -----------------------------------------------------

        category_scores = {
            "documentation": 75,
            "configuration": 55,
            "source": 65,
            "examples": 50,
            "other": 5,
        }

        score = category_scores.get(
            category,
            5,
        )

        # =====================================================
        # README SCORING
        # =====================================================

        if filename in {
            "readme.md",
            "readme.rst",
            "readme.txt",
        }:

            # Root README is extremely important.
            if len(parts) == 1:

                score += 60

            # README inside core source/package directories.
            elif any(
                directory
                in SOURCE_DIRECTORIES
                for directory in []
            ):
                score += 25

            # README inside examples.
            elif cls._is_example_path(
                normalized
            ):

                score += 30

            # Deep documentation README.
            else:

                score += 25

        # -----------------------------------------------------
        # Architecture / overview / design docs
        # -----------------------------------------------------

        if filename in {
            "architecture.md",
            "documentation.md",
            "overview.md",
            "design.md",
        }:

            score += 30

        # -----------------------------------------------------
        # API / guide / tutorial docs
        # -----------------------------------------------------

        if filename in {
            "api.md",
            "guide.md",
            "guides.md",
            "tutorial.md",
            "tutorials.md",
        }:

            score += 20

        # -----------------------------------------------------
        # Documentation directory
        # -----------------------------------------------------

        if any(
            directory
            in cls.DOCUMENTATION_DIRECTORIES
            for directory in parts
        ):

            score += 5

        # =====================================================
        # CONFIGURATION
        # =====================================================

        if filename in {
            "pyproject.toml",
            "package.json",
            "cargo.toml",
            "go.mod",
            "pom.xml",
            "pubspec.yaml",
            "composer.json",
        }:

            score += 30

        # Runtime / training / application config is useful.
        if extension in {
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
        }:

            if not cls._is_low_value_configuration(
                normalized
            ):

                score += 5

        # =====================================================
        # SOURCE FILE HINTS
        # =====================================================

        source_keywords = {
            "model": 12,
            "train": 12,
            "training": 12,
            "inference": 12,
            "auth": 15,
            "database": 12,
            "schema": 10,
            "service": 8,
            "controller": 8,
            "router": 8,
            "config": 8,
            "settings": 8,
            "api": 8,
        }

        for keyword, bonus in source_keywords.items():

            if keyword in filename:
                score += bonus

        # =====================================================
        # DOCUMENTATION KEYWORDS
        # =====================================================

        documentation_keywords = {
            "architecture": 20,
            "guide": 10,
            "tutorial": 10,
            "api": 8,
            "design": 12,
            "overview": 12,
        }

        for keyword, bonus in documentation_keywords.items():

            if keyword in filename:
                score += bonus

        # =====================================================
        # EXAMPLE FILES
        # =====================================================

        if cls._is_example_path(
            normalized
        ):

            score += 5

            # Example READMEs are especially useful.
            if filename.startswith("readme"):
                score += 15

        # =====================================================
        # TEST PENALTIES
        # =====================================================

        if cls._is_test_path(
            normalized
        ):

            score -= 25

        if (
            "test_" in filename
            or filename.startswith("test.")
            or filename.endswith("_test.py")
            or filename.endswith(".test.js")
            or filename.endswith(".test.ts")
            or filename.endswith(".spec.js")
            or filename.endswith(".spec.ts")
        ):

            score -= 15

        # =====================================================
        # LOW-VALUE CONFIGURATION
        # =====================================================

        if cls._is_low_value_configuration(
            normalized
        ):

            score -= 35

        # =====================================================
        # CI / AUTOMATION PENALTY
        # =====================================================

        if any(
            keyword in filename
            for keyword in {
                "pre-commit",
                "codecov",
                "gitlab-ci",
                "jenkins",
                "travis",
            }
        ):

            score -= 20

        # =====================================================
        # ADMIN DOCUMENTATION PENALTY
        # =====================================================

        if filename in cls.LOW_PRIORITY_DOCUMENTATION_FILES:

            score -= 40

        # =====================================================
        # DEEP NESTING PENALTY
        # =====================================================

        depth = len(parts) - 1

        if depth >= 5:
            score -= 10

        if depth >= 8:
            score -= 15

        # =====================================================
        # HUGE FILE PENALTY
        # =====================================================

        if (
            size is not None
            and size > cls.MAX_FILE_SIZE
        ):

            score -= 100

        return max(
            score,
            0,
        )

    # =========================================================
    # SELECT FILES
    # =========================================================

    @classmethod
    def select_files(
        cls,
        tree: list[dict],
    ) -> list[dict]:
        """
        Select a balanced set of useful files.

        Category limits prevent a repository with thousands
        of source files from completely dominating the
        retrieval corpus.
        """

        candidates = []

        # -----------------------------------------------------
        # Build candidate list
        # -----------------------------------------------------

        for item in tree:

            if item.get("type") != "blob":
                continue

            path = item.get("path")

            if not path:
                continue

            if cls._is_ignored_path(
                path
            ):
                continue

            extension = cls._file_extension(
                path
            )

            if (
                extension
                in cls.IGNORED_FILE_EXTENSIONS
            ):
                continue

            size = item.get("size")

            if (
                size is not None
                and size > cls.MAX_FILE_SIZE
            ):
                continue

            category = cls.classify_file(
                path
            )

            score = cls.score_file(
                path,
                size,
            )

            if score <= 0:
                continue

            candidates.append(
                {
                    "path": path,
                    "sha": item.get("sha"),
                    "size": size,
                    "url": item.get("url"),
                    "score": score,
                    "category": category,
                }
            )

        # -----------------------------------------------------
        # Group candidates
        # -----------------------------------------------------

        grouped = {
            "documentation": [],
            "configuration": [],
            "source": [],
            "examples": [],
            "other": [],
        }

        for candidate in candidates:

            category = candidate[
                "category"
            ]

            grouped[
                category
            ].append(
                candidate
            )

        # -----------------------------------------------------
        # Sort categories
        # -----------------------------------------------------

        for category in grouped:

            grouped[category].sort(
                key=lambda item: (
                    -item["score"],
                    item["path"],
                )
            )

        # -----------------------------------------------------
        # Balanced category selection
        # -----------------------------------------------------

        selected = []

        for category, limit in (
            cls.CATEGORY_LIMITS.items()
        ):

            selected.extend(
                grouped[
                    category
                ][:limit]
            )

        # -----------------------------------------------------
        # Fill remaining slots with globally relevant files
        # -----------------------------------------------------

        selected_paths = {
            item["path"]
            for item in selected
        }

        remaining = [
            item
            for item in candidates
            if item["path"]
            not in selected_paths
        ]

        remaining.sort(
            key=lambda item: (
                -item["score"],
                item["path"],
            )
        )

        slots = (
            cls.MAX_FILES
            - len(selected)
        )

        if slots > 0:

            selected.extend(
                remaining[:slots]
            )

        # -----------------------------------------------------
        # Final deterministic ordering
        # -----------------------------------------------------

        category_order = {
            "documentation": 0,
            "configuration": 1,
            "source": 2,
            "examples": 3,
            "other": 4,
        }

        selected.sort(
            key=lambda item: (
                category_order.get(
                    item["category"],
                    5,
                ),
                -item["score"],
                item["path"],
            )
        )

        return selected[
            :cls.MAX_FILES
        ]

    # =========================================================
    # FETCH FILE CONTENT
    # =========================================================

    @classmethod
    def fetch_file(
        cls,
        github_url: str,
        path: str,
    ) -> str:
        """
        Fetch one file's content from GitHub.
        """

        owner, repository = (
            cls._parse_github_url(
                github_url
            )
        )

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}"
            f"/contents/{path}"
        )

        response = cls._request(
            "GET",
            url,
            timeout=30,
        )

        data = response.json()

        if data.get("type") != "file":

            raise ValueError(
                f"Not a file: {path}"
            )

        content = data.get(
            "content"
        )

        if not content:
            return ""

        return base64.b64decode(
            content
        ).decode(
            "utf-8",
            errors="replace",
        )

    # =========================================================
    # DISCOVER
    # =========================================================

    @classmethod
    def discover(
        cls,
        github_url: str,
    ) -> list[dict]:
        """
        Discover useful files without downloading contents.
        """

        tree = cls.fetch_tree(
            github_url
        )

        return cls.select_files(
            tree
        )

    # =========================================================
    # INDEX REPOSITORY
    # =========================================================

    @classmethod
    def index_repository(
        cls,
        github_url: str,
    ) -> list[dict]:
        """
        Discover selected files and download their contents.

        Returns:

            [
                {
                    "path": "...",
                    "content": "...",
                    "score": 100,
                    "category": "source",
                }
            ]

        These documents are ready for the RAG pipeline.
        """

        files = cls.discover(
            github_url
        )

        documents = []

        for file_info in files:

            path = file_info[
                "path"
            ]

            try:

                content = cls.fetch_file(
                    github_url,
                    path,
                )

            except Exception:
                continue

            if not content.strip():
                continue

            documents.append(
                {
                    "path": path,
                    "content": content,
                    "score": file_info[
                        "score"
                    ],
                    "category": file_info[
                        "category"
                    ],
                }
            )

        return documents
    
    @classmethod
    def fetch_file_content(cls, file):
        """
        Fetch the actual content of a GitHub file using
        the GitHub Blob API.

        The `file` dictionary is expected to contain:
        - path
        - sha
        - url
        """

        url = file.get("url")

        if not url:
            return ""

        try:
            response = cls._request(
                "GET",
                url,
                timeout=30,
            )

            data = response.json()

            encoded_content = data.get("content")

            if not encoded_content:
                return ""

            # GitHub sometimes inserts newlines into base64
            encoded_content = encoded_content.replace("\n", "")

            decoded_content = base64.b64decode(
                encoded_content
            )

            return decoded_content.decode(
                "utf-8",
                errors="replace",
            )

        except requests.RequestException as exc:
            print(
                f"Failed to fetch {file.get('path', 'unknown')}: "
                f"{exc}"
            )
            return ""

        except (ValueError, UnicodeDecodeError) as exc:
            print(
                f"Failed to decode {file.get('path', 'unknown')}: "
                f"{exc}"
            )
            return ""
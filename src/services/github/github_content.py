from __future__ import annotations

import ast
import base64
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

try:
    from .github_cache import GitHubCache
except ImportError:  # pragma: no cover - fallback for flat/script usage
    from github_cache import GitHubCache


# =============================================================
# ENVIRONMENT
# =============================================================

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Disk-backed cache for GitHub API responses (metadata, tree, readme,
# file content, root listing). Persists across process restarts, unlike
# an in-memory cache, which matters for a backend that may be
# redeployed/restarted between user sessions.
_GITHUB_CACHE_DIR = os.getenv(
    "GITHUB_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "cache", "github"),
)

_GITHUB_CACHE_TTL_SECONDS = int(
    os.getenv("GITHUB_CACHE_TTL_SECONDS", "600")
)

_github_cache = GitHubCache(
    cache_dir=_GITHUB_CACHE_DIR,
    ttl_seconds=_GITHUB_CACHE_TTL_SECONDS,
)


class GitHubContentService:
    """
    Query-aware GitHub repository content service for code RAG.

    Design:

        GitHub URL
            |
            +--> repository metadata
            |
            +--> README / global context
            |
            +--> recursive repository tree
                        |
                        v
                 query analysis
                        |
                        v
                 candidate files
                        |
                        v
                 content scoring
                        |
                        v
                focused code windows
                        |
                        v
                  RAG context

    The service deliberately does NOT send the whole repository to the
    LLM. It discovers the repository first and fetches only content that
    is likely to answer the user's query.

    The existing build_context(github_url) API is preserved for
    backwards compatibility. New backend code should prefer:

        build_context_for_query(github_url, query)
    """

    API_BASE_URL = "https://api.github.com"

    # -------------------------------------------------------------
    # Context budgets
    # -------------------------------------------------------------

    MAX_CONTEXT_CHARS = 18000
    MAX_README_CHARS = 9000

    # Source candidates inspected before final selection.
    MAX_CANDIDATE_FILES = 24

    # Source files actually included in the final query context.
    MAX_QUERY_SOURCE_FILES = 5

    # Maximum source characters fetched from one file before focusing.
    MAX_SOURCE_FILE_FETCH_CHARS = 50000

    # Maximum source characters contributed by one selected file.
    MAX_SOURCE_FILE_CONTEXT_CHARS = 6000

    # Total source context budget.
    MAX_SOURCE_CONTEXT_CHARS = 13000

    # Repository tree can be large; this is only a safety guard.
    MAX_TREE_FILES = 15000

    # Query-aware discovery limits for very large repositories.
    MAX_DISCOVERY_DIRECTORIES = 48
    MAX_DISCOVERY_FILES = 5000
    MAX_DISCOVERY_DEPTH = 8
    MAX_DISCOVERY_BRANCHES = 8

    # -------------------------------------------------------------
    # File categories
    # -------------------------------------------------------------

    SOURCE_EXTENSIONS = {
        ".py", ".pyi",
        ".js", ".jsx", ".mjs", ".cjs",
        ".ts", ".tsx",
        ".java", ".kt", ".kts",
        ".go",
        ".rs",
        ".c", ".h",
        ".cc", ".cpp", ".cxx", ".hpp",
        ".cs",
        ".swift",
        ".scala",
        ".rb",
        ".php",
        ".dart",
        ".m", ".mm",
        ".r", ".R",
        ".jl",
        ".lua",
        ".sh", ".bash",
    }

    CONFIG_FILENAMES = {
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "package.json",
        "tsconfig.json",
        "environment.yml",
        "environment.yaml",
        "cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "dockerfile",
        "makefile",
        "cmakelists.txt",
    }

    DOC_EXTENSIONS = {
        ".md", ".mdx", ".rst", ".txt",
    }

    EXCLUDED_PARTS = {
        ".git",
        ".github",
        ".idea",
        ".vscode",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "target",
        "vendor",
        "third_party",
        "third-party",
        "coverage",
        ".tox",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }

    TEST_PARTS = {
        "test",
        "tests",
        "testing",
    }

    BENCHMARK_PARTS = {
        "benchmark",
        "benchmarks",
    }

    DOC_PARTS = {
        "doc",
        "docs",
        "documentation",
    }

    EXAMPLE_PARTS = {
        "example",
        "examples",
        "demo",
        "demos",
    }

    BUILD_PARTS = {
        "build",
        "build_tools",
        "scripts",
        "ci",
    }

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been",
        "being", "by", "can", "could", "did", "do", "does",
        "for", "from", "how", "in", "into", "is", "it", "its",
        "may", "might", "of", "on", "or", "should", "that",
        "the", "their", "this", "to", "was", "what", "when",
        "where", "which", "who", "why", "will", "with", "would",
        "about", "object", "class", "function", "method",
        "implemented", "implementation", "work", "works",
        "working", "use", "uses", "using", "used", "manage",
        "manages", "handling", "handle", "sequence", "main",
        "key", "specific", "details",
    }

    INTENT_WORDS = {
        "implementation": {
            "implement", "implemented", "implementation", "internally",
            "internal", "source", "code", "under", "hood",
        },
        "architecture": {
            "architecture", "architectural", "design", "components",
            "structure", "modules", "layers", "flow",
        },
        "behavior": {
            "behave", "behavior", "works", "work", "does", "handle",
            "handling", "process", "flow",
        },
        "api": {
            "api", "usage", "interface", "parameters", "argument",
            "arguments", "returns", "return", "signature",
        },
        "configuration": {
            "config", "configuration", "setting", "settings",
            "option", "options", "environment", "dependency",
            "dependencies",
        },
        "testing": {
            "test", "tests", "testing", "pytest", "unittest",
            "assertion", "assertions",
        },
        "benchmark": {
            "benchmark", "benchmarks", "performance", "latency",
            "throughput", "scaling", "speed", "memory",
        },
        "documentation": {
            "documentation", "docs", "readme", "guide", "tutorial",
            "example", "examples",
        },
        "history": {
            "history", "origin", "started", "created", "author",
            "contributors", "version", "release",
        },
        "comparison": {
            "compare", "comparison", "difference", "differences",
            "versus", "vs", "better", "similar",
        },
        "security": {
            "security", "authentication", "authorization", "auth",
            "token", "permission", "permissions", "credential",
            "credentials", "encryption",
        },
        "overview": {
            "about", "overview", "purpose", "summary", "summarize",
            "goal", "goals", "description", "describe", "introduction",
            "intro", "problem", "problems", "solve", "solves", "solving",
        },
        "installation": {
            "install", "installation", "installing", "setup", "set",
            "requirements", "requirement", "dependency", "dependencies",
            "pip", "npm", "yarn", "conda", "build", "compile", "docker",
            "run", "running", "start", "getting", "quickstart",
            "prerequisite", "prerequisites",
        },
    }

    # =============================================================
    # GITHUB URL / REQUEST HELPERS
    # =============================================================

    @staticmethod
    def _parse_github_url(url: str) -> tuple[str, str]:
        if not url:
            raise ValueError("GitHub URL must not be empty.")

        parsed = urlparse(str(url).strip())

        if parsed.netloc.lower() not in {
            "github.com",
            "www.github.com",
        }:
            raise ValueError(f"Not a GitHub URL: {url}")

        parts = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        if len(parts) < 2:
            raise ValueError(
                f"Invalid GitHub repository URL: {url}"
            )

        owner = parts[0]
        repository = re.sub(
            r"\.git$",
            "",
            parts[1],
            flags=re.IGNORECASE,
        )

        return owner, repository

    @staticmethod
    def _headers() -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Smart-Research-Dashboard",
        }

        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

        return headers

    @classmethod
    def _get(
        cls,
        url: str,
        timeout: int = 30,
    ) -> requests.Response:
        response = requests.get(
            url,
            headers=cls._headers(),
            timeout=timeout,
        )

        if response.status_code == 401:
            raise ValueError(
                "GitHub authentication failed. Check GITHUB_TOKEN."
            )

        if response.status_code == 403:
            raise ValueError(
                "GitHub API rate limit or permission error. "
                "Set a valid GITHUB_TOKEN if needed."
            )

        response.raise_for_status()
        return response

    # =============================================================
    # REPOSITORY METADATA
    # =============================================================

    @classmethod
    def fetch_repository_metadata(
        cls,
        github_url: str,
    ) -> dict:
        owner, repository = cls._parse_github_url(github_url)

        cached = _github_cache.get("metadata", owner, repository)

        if cached is not None:
            return cached

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}"
        )

        response = cls._get(url)
        data = response.json()

        result = {
            "owner": owner,
            "repository": repository,
            "default_branch": data.get("default_branch") or "main",
            "description": data.get("description"),
            "language": data.get("language"),
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "url": data.get("html_url") or github_url,
        }

        _github_cache.set("metadata", result, owner, repository)
        return result

    # =============================================================
    # README
    # =============================================================

    @classmethod
    def fetch_readme(
        cls,
        github_url: str,
    ) -> str:
        owner, repository = cls._parse_github_url(github_url)

        cached = _github_cache.get("readme", owner, repository)

        if cached is not None:
            return cached

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}/readme"
        )

        response = cls._get(url)
        data = response.json()

        content = data.get("content")

        if not content:
            _github_cache.set("readme", "", owner, repository)
            return ""

        try:
            result = base64.b64decode(content).decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            result = ""

        _github_cache.set("readme", result, owner, repository)
        return result

    # =============================================================
    # SINGLE FILE
    # =============================================================

    @classmethod
    def fetch_file(
        cls,
        github_url: str,
        file_path: str,
        branch: str | None = None,
    ) -> str:
        owner, repository = cls._parse_github_url(github_url)

        file_path = file_path.strip("/")

        if not file_path:
            raise ValueError("File path must not be empty.")

        # The ref is part of the cache key: the same path can have
        # different content on different branches.
        ref_key = branch or "HEAD"

        cached = _github_cache.get(
            "file", owner, repository, ref_key, file_path
        )

        if cached is not None:
            return cached

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}"
            f"/contents/{file_path}"
        )

        if branch:
            url += f"?ref={branch}"

        response = cls._get(url, timeout=45)
        data = response.json()

        if data.get("type") != "file":
            raise ValueError(f"Path is not a file: {file_path}")

        content = data.get("content")

        if not content:
            _github_cache.set(
                "file", "", owner, repository, ref_key, file_path
            )
            return ""

        # GitHub's Contents API can return large files differently.
        if data.get("encoding") != "base64":
            download_url = data.get("download_url")

            if download_url:
                raw_response = requests.get(
                    download_url,
                    headers=cls._headers(),
                    timeout=45,
                )
                raw_response.raise_for_status()
                result = raw_response.text
                _github_cache.set(
                    "file", result, owner, repository, ref_key, file_path
                )
                return result

            _github_cache.set(
                "file", "", owner, repository, ref_key, file_path
            )
            return ""

        try:
            decoded = base64.b64decode(content).decode(
                "utf-8",
                errors="replace",
            )
        except Exception as exc:
            raise ValueError(
                f"Could not decode file: {file_path}"
            ) from exc

        _github_cache.set(
            "file", decoded, owner, repository, ref_key, file_path
        )
        return decoded

    # =============================================================
    # ROOT CONTENTS
    # =============================================================

    @classmethod
    def fetch_root_contents(
        cls,
        github_url: str,
    ) -> list[dict]:
        owner, repository = cls._parse_github_url(github_url)

        cached = _github_cache.get("root", owner, repository)

        if cached is not None:
            return cached

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}/contents"
        )

        response = cls._get(url)
        data = response.json()

        result = data if isinstance(data, list) else []
        _github_cache.set("root", result, owner, repository)
        return result

    # =============================================================
    # COMPLETE REPOSITORY TREE
    # =============================================================

    @classmethod
    def _fetch_directory_contents(
        cls,
        github_url: str,
        directory: str = "",
        branch: str | None = None,
    ) -> list[dict]:
        """Fetch one directory listing with the existing disk cache.

        The cache namespace is intentionally kept as ``root`` for backward
        compatibility; the directory path and branch are part of the key.
        """
        owner, repository = cls._parse_github_url(github_url)
        directory = directory.strip("/")
        ref_key = branch or "HEAD"

        cached = _github_cache.get(
            "root", owner, repository, ref_key, directory
        )
        if cached is not None:
            return cached

        url = f"{cls.API_BASE_URL}/repos/{owner}/{repository}/contents"
        if directory:
            url += "/" + directory
        if branch:
            url += f"?ref={branch}"

        response = cls._get(url, timeout=45)
        data = response.json()
        result = data if isinstance(data, list) else []

        _github_cache.set(
            "root", result, owner, repository, ref_key, directory
        )
        return result

    @classmethod
    def _score_discovery_directory(
        cls,
        directory: str,
        analysis: dict,
        depth: int,
    ) -> float:
        """Score a directory for query-aware traversal of huge repositories."""
        parts = cls._path_parts(directory)
        tokens = set(re.findall(r"[a-z0-9]+", directory.lower()))
        score = 0.0

        primary = [str(x).lower() for x in analysis.get("primary_entities", [])]
        supporting = [str(x).lower() for x in analysis.get("supporting_terms", [])]
        identifier_parts = [
            str(x).lower() for x in analysis.get("identifier_parts", [])
        ]
        repository_name = str(
            analysis.get("repository_name") or ""
        ).lower().strip()

        # Repository names are contextual metadata, not semantic query terms.
        # In a monorepo this is especially important: a repository called
        # "next.js" has many directories containing "next", but that does not
        # mean those directories answer a routing question.
        for term in primary:
            term_tokens = set(re.findall(r"[a-z0-9]+", term))

            if (
                repository_name
                and term == repository_name
                and term in parts
                and term != parts[-1]
            ):
                score -= 20
                continue

            if term in parts:
                score += 55
            elif term_tokens and term_tokens.issubset(tokens):
                score += 45
            elif term in directory.lower():
                score += 20

        # Identifier components are useful for traversal, but only as
        # secondary signals. This prevents a common component such as "next"
        # or "data" from monopolizing the discovery budget.
        for term in identifier_parts:
            if len(term) >= 3 and term in tokens:
                score += 22

        for term in supporting:
            if len(term) >= 3 and term in tokens:
                score += 18

        # Query-concept traversal priors. These are deliberately structural
        # rather than repository-specific so they generalize across languages.
        intent_terms = set(
            str(x).lower() for x in analysis.get("intents", [])
        )

        if "routing" in identifier_parts or "routing" in supporting:
            if any(p in parts for p in (
                "router", "routing", "routes", "route", "pages", "app",
                "navigation",
            )):
                score += 90

            # JS/TS framework repositories commonly keep the implementation
            # under packages/<framework>/... rather than crates/... .
            if "packages" in parts:
                score += 45

            if len(parts) >= 2 and parts[0] == "packages":
                if repository_name:
                    repo_base = re.sub(r"[^a-z0-9]+", "", repository_name)
                    package_base = re.sub(r"[^a-z0-9]+", "", parts[1])
                    if repo_base and package_base and (
                        repo_base.startswith(package_base)
                        or package_base.startswith(repo_base)
                    ):
                        score += 80

        # DataLoader-like questions in Python/ML repositories often live under
        # utils/data rather than low-level C++ bindings. Prefer those structural
        # bridges during discovery when both "data" and "loader" are present.
        if {"data", "loader"}.issubset(set(identifier_parts)):
            if "data" in parts:
                score += 70
            if "utils" in parts:
                score += 30
            if "csrc" in parts or "cuda" in parts:
                score -= 25

        # Common source/package roots are useful traversal bridges even when
        # their names do not occur in the query (e.g. torch/utils/data).
        if any(p in parts for p in (
            "src", "lib", "libs", "packages", "package", "crates",
            "python", "go", "cmd", "internal", "core", "server",
            "client", "components", "modules", "torch", "fastapi",
            "utils",
        )):
            score += 20

        # Prefer shallower paths so discovery does not spend its budget on
        # deeply nested unrelated directories.
        score -= depth * 2
        return score

    @classmethod
    def _discover_large_repository_tree(
        cls,
        github_url: str,
        query: str,
        branch: str | None = None,
    ) -> list[str]:
        """Discover relevant files without requiring a complete recursive tree.

        GitHub returns ``truncated=true`` for very large recursive tree
        requests. Instead of failing, use bounded best-first traversal of
        directory listings. This is query-aware and therefore scales to large
        monorepos such as Next.js and PyTorch.
        """
        analysis = cls.analyze_query(query)
        metadata = cls.fetch_repository_metadata(github_url)
        analysis["repository_name"] = str(
            metadata.get("repository") or ""
        ).lower()
        resolved_branch = branch or metadata.get("default_branch") or "main"

        queue: list[tuple[float, int, str]] = [(0.0, 0, "")]
        visited: set[str] = set()
        files: set[str] = set()
        directory_requests = 0

        while queue and directory_requests < cls.MAX_DISCOVERY_DIRECTORIES:
            queue.sort(key=lambda item: (-item[0], item[1], item[2]))
            _, depth, directory = queue.pop(0)

            if directory in visited or depth > cls.MAX_DISCOVERY_DEPTH:
                continue
            visited.add(directory)

            try:
                entries = cls._fetch_directory_contents(
                    github_url, directory, resolved_branch
                )
            except Exception:
                continue

            directory_requests += 1

            for entry in entries:
                path = str(entry.get("path") or "").strip("/")
                if not path:
                    continue

                entry_type = entry.get("type")
                if entry_type == "file":
                    if (
                        cls._is_source_file(path)
                        or cls._is_config_file(path)
                        or cls._is_documentation_file(path)
                    ) and not (set(cls._path_parts(path)) & cls.EXCLUDED_PARTS):
                        files.add(path)
                        if len(files) >= cls.MAX_DISCOVERY_FILES:
                            return sorted(files)

                elif entry_type == "dir" and depth < cls.MAX_DISCOVERY_DEPTH:
                    directory_score = cls._score_discovery_directory(
                        path, analysis, depth + 1
                    )
                    queue.append((directory_score, depth + 1, path))

        return sorted(files)

    @classmethod
    def fetch_repository_tree(
        cls,
        github_url: str,
        branch: str | None = None,
        query: str | None = None,
    ) -> list[str]:
        metadata = cls.fetch_repository_metadata(github_url)

        branch = branch or metadata["default_branch"]
        owner = metadata["owner"]
        repository = metadata["repository"]

        cached = _github_cache.get("tree", owner, repository, branch)

        if cached is not None:
            return cached

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}"
            f"/git/trees/{branch}"
            f"?recursive=1"
        )

        response = cls._get(url, timeout=60)
        data = response.json()

        tree = data.get("tree")
        truncated = bool(data.get("truncated"))

        paths: list[str] = []
        if isinstance(tree, list):
            for item in tree:
                if item.get("type") != "blob":
                    continue
                path = str(item.get("path") or "").strip()
                if path:
                    paths.append(path)

        # A large/truncated recursive tree is not an error anymore when a
        # query is available. Discover only query-relevant branches instead.
        if truncated or len(paths) > cls.MAX_TREE_FILES:
            if query:
                return cls._discover_large_repository_tree(
                    github_url, query, branch
                )

            # Preserve the old safety behavior for callers that did not give
            # us a query, because there is no principled way to prioritize
            # files without one.
            raise ValueError(
                f"Repository contains more than {cls.MAX_TREE_FILES:,} files; "
                "tree discovery safety limit reached. Provide a query for "
                "query-aware large-repository discovery."
            )

        if not isinstance(tree, list):
            raise ValueError("Unexpected GitHub repository tree response.")

        result = sorted(set(paths))
        _github_cache.set("tree", result, owner, repository, branch)
        return result

    # =============================================================
    # CACHE MANAGEMENT
    # =============================================================

    @classmethod
    def clear_cache(cls) -> None:
        """Manually drop all cached GitHub responses on disk."""
        _github_cache.clear()

    @classmethod
    def cache_stats(cls) -> dict[str, int]:
        """Return counts of cached entries per kind, for diagnostics."""
        return _github_cache.stats()

    # =============================================================
    # PATH CLASSIFICATION
    # =============================================================

    @classmethod
    def _path_parts(cls, path: str) -> list[str]:
        return [
            part.lower()
            for part in path.replace("\\", "/").split("/")
            if part
        ]

    @classmethod
    def _is_source_file(cls, path: str) -> bool:
        return Path(path).suffix in cls.SOURCE_EXTENSIONS

    @classmethod
    def _is_config_file(cls, path: str) -> bool:
        return Path(path).name.lower() in cls.CONFIG_FILENAMES

    @classmethod
    def _is_documentation_file(cls, path: str) -> bool:
        parts = cls._path_parts(path)
        suffix = Path(path).suffix.lower()

        return (
            suffix in cls.DOC_EXTENSIONS
            or bool(set(parts) & cls.DOC_PARTS)
        )

    @classmethod
    def _is_test_file(cls, path: str) -> bool:
        parts = cls._path_parts(path)
        stem = Path(path).stem.lower()

        return (
            bool(set(parts) & cls.TEST_PARTS)
            or stem.startswith("test_")
            or stem.endswith("_test")
        )

    @classmethod
    def _is_benchmark_file(cls, path: str) -> bool:
        return bool(
            set(cls._path_parts(path)) & cls.BENCHMARK_PARTS
        )

    @classmethod
    def _is_example_file(cls, path: str) -> bool:
        return bool(
            set(cls._path_parts(path)) & cls.EXAMPLE_PARTS
        )

    # =============================================================
    # SOURCE FILE DISCOVERY
    # =============================================================

    @classmethod
    def find_source_files(
        cls,
        github_url: str,
        branch: str | None = None,
    ) -> list[str]:
        paths = cls.fetch_repository_tree(
            github_url,
            branch=branch,
        )

        result = []

        for path in paths:
            normalized = path.replace("\\", "/")

            parts = cls._path_parts(normalized)

            if set(parts) & cls.EXCLUDED_PARTS:
                continue

            if cls._is_source_file(normalized):
                result.append(normalized)

        return result

    # =============================================================
    # IDENTIFIER DECOMPOSITION
    # =============================================================

    @staticmethod
    def _decompose_identifier(identifier: str) -> list[str]:
        """
        Decompose a technical identifier into searchable components.

        Examples:
            GenerationOptions -> ["generation", "options", "generationoptions"]
            translate_batch_async -> ["translate", "batch", "async",
                                      "translate_batch_async"]
            sklearn.pipeline -> ["sklearn", "pipeline", "sklearn.pipeline"]
        """
        value = (identifier or "").strip().strip("`'\"")
        if not value:
            return []

        # Treat path/module separators as boundaries while preserving the
        # complete identifier as a searchable term.
        normalized = re.sub(r"[\\/]+", ".", value)
        normalized = re.sub(r"[-]+", "_", normalized)

        components = []
        for segment in normalized.split("."):
            segment = segment.strip("_")
            if not segment:
                continue

            # Split whitespace-delimited technical phrases too.
            for word in re.findall(r"[A-Za-z0-9]+", segment):
                word = word.lower()
                if word and word not in components:
                    components.append(word)

            # Split acronym + word boundaries (HTTPServer -> HTTP + Server)
            # and normal CamelCase boundaries (GenerationOptions -> ...).
            pieces = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|"
                r"[A-Z]?[a-z]+|"
                r"[A-Z]+|"
                r"[0-9]+",
                segment,
            )

            for piece in pieces:
                piece = piece.lower()
                if piece and piece not in components:
                    components.append(piece)

            # Also split snake_case identifiers.
            for piece in re.split(r"[_]+", segment.lower()):
                if piece and piece not in components:
                    components.append(piece)

        # Preserve the exact normalized identifier as the strongest
        # decomposition-level term.
        full = normalized.lower().strip("._-")
        if full and full not in components:
            components.append(full)

        return components

    # =============================================================
    # QUERY ANALYSIS
    # =============================================================

    @classmethod
    def analyze_query(
        cls,
        query: str,
    ) -> dict:
        """
        Analyze a repository question without requiring an LLM.

        Returns primary identifiers, supporting terms, likely intents,
        file hints, and query flags.

        This is deliberately heuristic rather than pretending to be a
        perfect NLP parser.
        """

        query = (query or "").strip()

        if not query:
            return {
                "primary_entities": [],
                "supporting_terms": [],
                "all_terms": [],
                "identifiers": [],
                "identifier_parts": [],
                "intents": [],
                "file_hints": [],
                "wants_code": False,
                "wants_tests": False,
                "wants_docs": False,
                "wants_benchmarks": False,
                "wants_overview": True,
                "wants_installation": False,
            }

        # ---------------------------------------------------------
        # Exact code-looking phrases:
        # `Pipeline`, `StandardScaler`, `fit_transform`,
        # sklearn.pipeline, "Pipeline", etc.
        # ---------------------------------------------------------

        quoted = re.findall(
            r"""["'`]([^"'`]+)["'`]""",
            query,
       )

        code_tokens = re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b",
            query,
        )

        all_candidates = []

        for value in quoted + code_tokens:
            value = value.strip()

            if value:
                all_candidates.append(value)

        # ---------------------------------------------------------
        # Normalize terms.
        # ---------------------------------------------------------

        all_terms = []
        for value in all_candidates:
            normalized = value.lower().strip("._-")

            if (
                normalized
                and len(normalized) > 1
                and normalized not in cls.STOP_WORDS
                and normalized not in all_terms
            ):
                all_terms.append(normalized)

        # ---------------------------------------------------------
        # Detect technical identifiers / query entities.
        #
        # Technical entities are not limited to CamelCase. Real questions
        # commonly mention Session, Command, routing, regex matching,
        # dependency injection, DataLoader, pip, etc.
        # ---------------------------------------------------------

        identifiers = []
        identifier_parts = []

        def add_identifier(value: str) -> None:
            clean = value.strip("`'\"").strip(" .,;:()[]{}")
            if not clean:
                return

            normalized = clean.lower()

            if (
                normalized in cls.STOP_WORDS
                or len(normalized) < 2
                or normalized in identifiers
            ):
                return

            identifiers.append(normalized)

            for part in cls._decompose_identifier(clean):
                if part not in identifier_parts:
                    identifier_parts.append(part)

        # Preserve strong code-looking candidates.
        for value in all_candidates:
            clean = value.strip("`'\"")

            if not clean:
                continue

            looks_like_identifier = (
                "_" in clean
                or "." in clean
                or bool(re.search(r"[a-z][A-Z]", clean))
                or bool(re.search(r"[A-Z]{2,}", clean))
                or clean in quoted
            )

            if looks_like_identifier:
                add_identifier(clean)

        # Extract the technical subject of common question forms.
        entity_patterns = [
            r"\b(?:how\s+is|how\s+does|how\s+do|where\s+is|where\s+are)"
            r"\s+(?:the\s+)?[`\"']?([A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,3})",

            r"\b(?:what\s+is|what\s+are|what\s+does)"
            r"\s+(?:the\s+)?[`\"']?([A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,3})",

            r"\b(?:class|struct|interface|type|function|method|module)"
            r"\s+[`\"']?([A-Za-z][A-Za-z0-9_.-]*)",
        ]

        phrase_stop = {
            "implemented", "implementation", "work", "works", "does",
            "do", "used", "use", "using", "in", "for", "with", "from",
            "and", "the", "a", "an", "what", "problem", "solve",
        }

        for pattern in entity_patterns:
            for match in re.finditer(pattern, query, flags=re.IGNORECASE):
                raw = match.group(1).strip("`'\".,:;()[]{}")
                words = [
                    word
                    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", raw)
                    if word.lower() not in phrase_stop
                ]

                if not words:
                    continue

                if len(words) > 1:
                    phrase = " ".join(words).lower()
                    if phrase not in identifiers:
                        identifiers.append(phrase)
                    for word in words:
                        add_identifier(word)
                else:
                    add_identifier(words[0])

        # Explicit quoted entities are always strong.
        for value in quoted:
            add_identifier(value)

        # A small vocabulary catches technical single-word subjects that are
        # otherwise indistinguishable from ordinary English.
        technical_tokens = {
            "session", "command", "regex", "matching", "routing",
            "dataloader", "dependency", "injection", "pip", "npm",
            "yarn", "conda", "docker", "fastapi", "ctranslate2",
        }

        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", query):
            if token.lower() in technical_tokens:
                add_identifier(token)

        # ---------------------------------------------------------
        # Primary entity signals.
        #
        # We don't need perfect grammatical parsing. We look for
        # strong patterns that indicate what the user is asking about.
        # ---------------------------------------------------------

        primary_entities = []

        primary_patterns = [
            r"\b(?:how\s+is|how\s+does|how\s+do|where\s+is|where\s+are|"
            r"what\s+is|what\s+does|explain|describe|understand)\s+"
            r"(?:the\s+)?[`\"']?([A-Za-z_][A-Za-z0-9_.]*)[`\"']?",

            r"[`\"']([A-Za-z_][A-Za-z0-9_.]*)[`\"']?\s+"
            r"(?:object|class|function|method|module|implementation|"
            r"source|API)\b",

            r"\b(?:class|function|method|module)\s+"
            r"[`\"']?([A-Za-z_][A-Za-z0-9_.]*)[`\"']?",
        ]

        for pattern in primary_patterns:
            for match in re.finditer(
                pattern,
                query,
                flags=re.IGNORECASE,
            ):
                value = match.group(1).strip("`'\".,:;()[]{}")

                if (
                    value
                    and value.lower() not in cls.STOP_WORDS
                    and value.lower() not in primary_entities
                ):
                    primary_entities.append(value.lower())

        # Exact quoted/code identifiers are strong candidates.
        for identifier in identifiers:
            if (
                identifier not in primary_entities
                and (
                    "_" in identifier
                    or "." in identifier
                    or identifier[:1].isupper()
                )
            ):
                primary_entities.append(identifier)

        # ---------------------------------------------------------
        # Intent detection.
        # ---------------------------------------------------------

        intents = []

        query_words = set(
            re.findall(
                r"[a-zA-Z_][a-zA-Z0-9_]*",
                query.lower(),
            )
        )

        for intent, words in cls.INTENT_WORDS.items():
            if query_words & words:
                intents.append(intent)

        # Explicit natural-language intent patterns. These complement the
        # generic keyword table and avoid relying on repository names.
        q_lower = query.lower()

        if re.search(r"\bwhat\s+(?:is|are)\b", q_lower):
            if "overview" not in intents:
                intents.append("overview")

        if re.search(r"\bwhat\s+is\b.*\bused\s+for\b", q_lower):
            if "overview" not in intents:
                intents.append("overview")

        if re.search(r"\bhow\s+is\b.*\bimplemented\b", q_lower):
            if "implementation" not in intents:
                intents.append("implementation")

        architecture_question = bool(
            re.search(
                r"\bhow\s+does\b.*\bwork\b|"
                r"\bhow\s+is\b.*\barchitecture\b|"
                r"\barchitecture\b|"
                r"\bhow\s+is\b.*\bimplemented\b",
                q_lower,
            )
        ) and any(
            x in q_lower
            for x in (
                "routing", "router", "architecture", "system", "design",
                "components", "modules", "layers", "data flow",
            )
        )

        if architecture_question:
            if "architecture" not in intents:
                intents.append("architecture")

        # Conceptual framework questions often need repository documentation.
        if (
            re.search(r"\bhow\s+does\b.*\bwork\b", q_lower)
            and any(
                x in q_lower
                for x in ("dependency injection", "routing", "configuration", "usage")
            )
        ):
            if "documentation" not in intents:
                intents.append("documentation")

        wants_code = bool(
            set(intents)
            & {
                "implementation",
                "architecture",
                "behavior",
                "api",
            }
        )

        wants_tests = "testing" in intents
        wants_docs = "documentation" in intents
        wants_benchmarks = "benchmark" in intents
        wants_installation = "installation" in intents

        # A question is "overview-style" if it explicitly signals that
        # intent, OR if it produced no other real signal to search with
        # (no primary entities, no identifiers, no specific intent).
        # The latter case covers phrasing like "what is this repository
        # about", where stop-word stripping removes almost every token.
        explicit_overview_question = bool(
            re.search(
                r"\bwhat\s+is\b|\bwhat\s+are\b|"
                r"\bwhat\s+is\s+.*\bused\s+for\b|"
                r"\bwhat\s+does\b.*\bdo\b",
                query.lower(),
            )
        )

        wants_overview = (
            "overview" in intents
            or explicit_overview_question
            or (
                not primary_entities
                and not identifiers
                and not wants_code
                and not wants_tests
                and not wants_docs
                and not wants_benchmarks
                and not wants_installation
            )
        )

        # ---------------------------------------------------------
        # Supporting terms.
        # ---------------------------------------------------------

        supporting_terms = [
            term
            for term in all_terms
            if term not in primary_entities
        ]

        # ---------------------------------------------------------
        # File hints.
        #
        # Explicit path-like query terms are very strong.
        # ---------------------------------------------------------

        file_hints = []

        for value in all_candidates:
            normalized = value.replace("\\", "/").lower()

            if (
                "/" in normalized
                or normalized.endswith(tuple(
                    cls.SOURCE_EXTENSIONS
                ))
            ):
                file_hints.append(normalized)

        return {
            "primary_entities": primary_entities,
            "supporting_terms": supporting_terms,
            "all_terms": all_terms,
            "identifiers": identifiers,
            "identifier_parts": identifier_parts,
            "intents": intents,
            "file_hints": file_hints,
            "wants_code": wants_code,
            "wants_tests": wants_tests,
            "wants_docs": wants_docs,
            "wants_benchmarks": wants_benchmarks,
            "wants_overview": wants_overview,
            "wants_installation": wants_installation,
        }

    # =============================================================
    # FILE PATH SCORING
    # =============================================================

    @classmethod
    def _score_file_path(
        cls,
        path: str,
        analysis: dict,
    ) -> tuple[float, list[str], list[str]]:
        normalized = path.replace("\\", "/")
        lower_path = normalized.lower()

        filename = Path(normalized).name.lower()
        stem = Path(normalized).stem.lower()
        parts = cls._path_parts(normalized)

        primary = analysis["primary_entities"]
        supporting = analysis["supporting_terms"]
        all_terms = analysis["all_terms"]
        identifier_parts = analysis.get("identifier_parts", [])

        score = 0.0
        matched = []
        reasons = []

        # ---------------------------------------------------------
        # Explicit path hints.
        # ---------------------------------------------------------

        for hint in analysis["file_hints"]:
            if lower_path == hint:
                score += 500
                matched.append(hint)
                reasons.append("exact file path requested")

            elif hint in lower_path:
                score += 180
                matched.append(hint)
                reasons.append("file path hint match")

        # ---------------------------------------------------------
        # PRIMARY entities are much stronger than generic terms.
        # ---------------------------------------------------------

        for term in primary:
            term = term.lower()

            if term == stem:
                score += 220
                matched.append(term)
                reasons.append("exact primary filename match")

            elif term == filename:
                score += 230
                matched.append(term)
                reasons.append("exact primary filename match")

            elif term in parts:
                score += 120
                matched.append(term)
                reasons.append("primary path component match")

            elif term in filename:
                score += 85
                matched.append(term)
                reasons.append("primary filename substring")

            elif (
                len(stem) >= 3
                and stem in term
            ):
                # Reverse relationship: the file's stem is a prefix/
                # substring of the identifier itself. This is a very
                # common real-world pattern - a class or struct name
                # extends the file it's declared in (GenerationOptions
                # in generation.h, StandardScaler in scaler.py,
                # BeamSearch in search.h) - but every check above only
                # tests whether the identifier appears inside the path,
                # never the reverse. Without this, a query identifier
                # longer than the file stem could never match its own
                # containing file at all.
                score += 95
                matched.append(term)
                reasons.append("filename stem matches identifier")

            elif term in lower_path:
                score += 25
                matched.append(term)
                reasons.append("primary path substring")

        # ---------------------------------------------------------
        # Supporting concepts have intentionally lower weight.
        # ---------------------------------------------------------

        for term in supporting:
            if term == stem:
                score += 45
                matched.append(term)
                reasons.append("supporting filename match")

            elif term in filename:
                score += 20
                matched.append(term)
                reasons.append("supporting filename substring")

            elif term in parts:
                score += 18
                matched.append(term)
                reasons.append("supporting path match")

            elif term in lower_path:
                score += 5
                matched.append(term)
                reasons.append("supporting path substring")

        # ---------------------------------------------------------
        # Decomposed identifier components provide a secondary signal.
        # The complete identifier remains stronger than its components.
        # ---------------------------------------------------------

        for term in identifier_parts:
            if len(term) < 3:
                continue

            if term == stem:
                score += 35
                matched.append(term)
                reasons.append("identifier component filename match")
            elif term in filename:
                score += 15
                matched.append(term)
                reasons.append("identifier component filename substring")
            elif term in parts:
                score += 12
                matched.append(term)
                reasons.append("identifier component path match")

        # ---------------------------------------------------------
        # Generic all-term matching is weak.
        # ---------------------------------------------------------

        for term in all_terms:
            if term in filename and term not in primary:
                score += 4

        # ---------------------------------------------------------
        # Overview / installation retrieval priors.
        # ---------------------------------------------------------

        if analysis["wants_overview"]:
            if filename in {"readme.md", "readme.rst", "readme.txt"}:
                # Root README is the canonical answer to "What is X?"
                if normalized.count("/") == 0:
                    score += 260
                    reasons.append("repository root README strongly prioritized for overview")
                else:
                    score += 140
                    reasons.append("repository README for overview")
            elif cls._is_documentation_file(normalized):
                score += 75
                reasons.append("documentation for overview")

            if cls._is_source_file(normalized):
                score -= 75
                reasons.append("source penalty for overview query")

        if analysis["wants_installation"]:
            if filename in {"readme.md", "readme.rst", "readme.txt"}:
                # Natural-language installation instructions belong primarily
                # in the repository README.
                if normalized.count("/") == 0:
                    score += 240
                    reasons.append("repository root README strongly prioritized for installation")
                else:
                    score += 150
                    reasons.append("repository README for installation")

            elif cls._is_documentation_file(normalized):
                score += 105
                reasons.append("installation documentation")

            # Root package metadata is useful supporting evidence.
            if normalized.count("/") == 0 and filename in {
                "setup.py", "setup.cfg", "pyproject.toml",
                "package.json", "cargo.toml", "go.mod",
            }:
                score += 110
                reasons.append("root package/dependency configuration")

            # Explicit setup/install/quickstart documentation paths.
            if re.search(
                r"(?:^|/)(?:install|installation|setup|getting[-_ ]?started|quickstart)(?:/|\.|$)",
                normalized,
            ):
                score += 85
                reasons.append("explicit installation/setup path")

            if cls._is_config_file(normalized):
                # Generic build/dependency files are supporting evidence only.
                if filename in {"cmakelists.txt", "requirements.txt"}:
                    score += 15
                    reasons.append("supporting dependency/build file")
                else:
                    score += 40
                    reasons.append("configuration/dependency file for installation")

            if cls._is_source_file(normalized):
                score -= 90
                reasons.append("source penalty for installation query")

        # Concept-aware path matching. Multi-word concepts such as
        # "dependency injection" and "regex matching" should influence
        # ranking even when the exact phrase is not a filename.
        concept_identifiers = [
            value for value in analysis.get("identifiers", [])
            if " " in str(value).strip()
        ]
        path_tokens = set(re.findall(r"[a-z0-9]+", normalized.lower()))

        for concept in concept_identifiers:
            concept_tokens = [
                token for token in re.findall(r"[a-z0-9]+", str(concept).lower())
                if token not in cls.STOP_WORDS
            ]
            if not concept_tokens:
                continue

            matched = [token for token in concept_tokens if token in path_tokens]

            if len(matched) == len(concept_tokens) and len(matched) >= 2:
                score += 95
                reasons.append(f"full concept path match: {concept}")
            elif matched:
                score += 30 * len(matched)
                reasons.append(f"partial concept path match: {concept}")

        if "dependency injection" in analysis.get("identifiers", []):
            if "dependency" in normalized or "dependencies" in normalized:
                score += 150
                reasons.append("dependency concept path match")

            if re.search(r"(?:^|/)tutorial(?:/|$)", normalized):
                score += 95
                reasons.append("tutorial path for dependency question")

            if re.search(r"(?:^|/)dependencies(?:/|$)", normalized):
                score += 85
                reasons.append("dedicated dependencies directory")

            if re.search(r"(?:^|/)injection(?:/|$)", normalized):
                score += 70
                reasons.append("injection path match")

        if "regex matching" in analysis.get("identifiers", []):
            if any(x in normalized for x in ("regex", "matcher", "matching")):
                score += 55
                reasons.append("regex matching concept path match")

        if "routing" in analysis.get("identifier_parts", []):
            if any(x in normalized for x in ("router", "routing")):
                score += 55
                reasons.append("routing concept path match")

        # Documentation concepts should beat generic repository/reference
        # files when the question explicitly asks how a concept works.
        if analysis.get("wants_docs") and any(
            concept in analysis.get("identifiers", [])
            for concept in ("dependency injection",)
        ):
            if re.search(r"(?:^|/)tutorial(?:/|$)", normalized):
                score += 80
                reasons.append("tutorial documentation for conceptual question")

        # ---------------------------------------------------------
        # Query-specific structural priors.
        #
        # These resolve an important ambiguity: a generic identifier can
        # identify both a low-level implementation file and the public/API
        # implementation users normally mean. Prefer the repository's
        # conventional high-level source layout without making the rule
        # repository-specific.
        # ---------------------------------------------------------

        identifier_set = set(
            str(x).lower() for x in analysis.get("identifier_parts", [])
        )
        repository_name = str(
            analysis.get("repository_name") or ""
        ).lower().strip()

        # DataLoader in PyTorch: prefer the public Python data-loader module
        # over the lower-level C++ DataLoader bindings when both exist.
        if {"data", "loader"}.issubset(identifier_set):
            if re.search(r"(?:^|/)utils/data(?:/|$)", normalized):
                score += 130
                reasons.append("high-level utils/data path for DataLoader")
            if re.search(r"(?:^|/)dataloader(?:\\.[a-z0-9]+)$", normalized):
                score += 110
                reasons.append("direct DataLoader filename")
            if re.search(r"(?:^|/)csrc(?:/|$)", normalized):
                score -= 80
                reasons.append("low-level C++ binding penalty for DataLoader")

        # Routing in Next.js-like monorepos: prefer the canonical package
        # implementation over unrelated internal crates/packages containing
        # the repository name.
        if "routing" in identifier_set:
            if re.search(r"(?:^|/)packages/[^/]+/src(?:/|$)", normalized):
                score += 45
                reasons.append("canonical package source path for routing")

            if "packages" in parts and len(parts) >= 2:
                repo_base = re.sub(r"[^a-z0-9]+", "", repository_name)
                package_base = re.sub(r"[^a-z0-9]+", "", parts[1])
                if repo_base and package_base and (
                    repo_base.startswith(package_base)
                    or package_base.startswith(repo_base)
                ):
                    score += 100
                    reasons.append("repository package match for routing")

            if any(x in parts for x in (
                "router", "routing", "routes", "route", "pages", "app"
            )):
                score += 100
                reasons.append("routing implementation path")

        # Repository-name directory dominance guard. A repository name
        # appearing only as a directory component is weak evidence; exact
        # filename/entity matches remain strong.
        repository_name = str(
            analysis.get("repository_name") or ""
        ).lower().strip()

        if (
            repository_name
            and repository_name in primary
            and repository_name in parts
            and repository_name != filename
            and repository_name != stem
        ):
            score -= 100
            reasons.append("repository-name directory dominance penalty")

        # ---------------------------------------------------------
        # Source directory prior.
        # ---------------------------------------------------------

        if parts and parts[0] in {
            "src",
            "lib",
            "app",
            "package",
            "packages",
            "sklearn",
        }:
            score += 12
            reasons.append("likely source directory")

        # ---------------------------------------------------------
        # Intent-aware category scoring.
        # ---------------------------------------------------------

        if analysis["wants_tests"]:
            if cls._is_test_file(normalized):
                score += 80
                reasons.append("test file requested")
        else:
            if cls._is_test_file(normalized):
                score -= 55
                reasons.append("test file penalty")

        if analysis["wants_benchmarks"]:
            if cls._is_benchmark_file(normalized):
                score += 80
                reasons.append("benchmark file requested")
        else:
            if cls._is_benchmark_file(normalized):
                score -= 45
                reasons.append("benchmark file penalty")

        if analysis["wants_docs"]:
            if cls._is_documentation_file(normalized):
                score += 65
                reasons.append("documentation requested")
        else:
            if cls._is_documentation_file(normalized):
                score -= 30
                reasons.append("documentation penalty")

        if analysis["wants_code"]:
            if cls._is_source_file(normalized):
                score += 15
                reasons.append("source code matches query intent")

            if cls._is_example_file(normalized):
                score -= 20
                reasons.append("example penalty")

            if set(parts) & cls.BUILD_PARTS:
                score -= 25
                reasons.append("build/script penalty")

        return (
            score,
            sorted(set(matched)),
            sorted(set(reasons)),
        )

    # =============================================================
    # QUERY-AWARE FILE SELECTION
    # =============================================================

    @classmethod
    def select_relevant_source_files(
        cls,
        github_url: str,
        query: str,
        branch: str | None = None,
        max_files: int = 8,
    ) -> list[dict]:
        """
        Select source/document/config files likely to answer a query.

        This stage only examines paths. It does not download the files.
        """

        analysis = cls.analyze_query(query)

        # Repository name is contextual metadata only. It is used to prevent
        # directory-name matches from dominating path ranking.
        try:
            _, repository_name = cls._parse_github_url(github_url)
            analysis["repository_name"] = repository_name.lower()
        except Exception:
            analysis["repository_name"] = ""

        tree = cls.fetch_repository_tree(
            github_url,
            branch=branch,
            query=query,
        )

        candidates = []

        for path in tree:
            normalized = path.replace("\\", "/")

            # We support source files, docs and configuration files.
            supported = (
                cls._is_source_file(normalized)
                or cls._is_config_file(normalized)
                or cls._is_documentation_file(normalized)
            )

            if not supported:
                continue

            if set(cls._path_parts(normalized)) & cls.EXCLUDED_PARTS:
                continue

            score, matched, reasons = cls._score_file_path(
                normalized,
                analysis,
            )

            if score <= 0:
                continue

            candidates.append(
                {
                    "path": normalized,
                    "score": round(score, 3),
                    "matched_terms": matched,
                    "reasons": reasons,
                }
            )

        candidates.sort(
            key=lambda item: (
                -item["score"],
                len(item["path"]),
                item["path"],
            )
        )

        return candidates[:max_files]

    # =============================================================
    # CONTENT NORMALIZATION
    # =============================================================

    @staticmethod
    def _clean_file_content(
        content: str,
    ) -> str:
        if not content:
            return ""

        # Preserve source code exactly enough for retrieval, while
        # removing excessive empty lines.
        lines = [
            line.rstrip()
            for line in content.splitlines()
        ]

        result = []
        previous_blank = False

        for line in lines:
            blank = not line.strip()

            if blank and previous_blank:
                continue

            result.append(line)
            previous_blank = blank

        return "\n".join(result).strip()

    @staticmethod
    def _clean_readme(
        readme: str,
    ) -> str:
        if not readme:
            return ""

        text = re.sub(
            r"<!--.*?-->",
            "",
            readme,
            flags=re.DOTALL,
        )

        text = re.sub(
            r"<[^>]+>",
            "",
            text,
        )

        lines = []

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            # Drop badge-only lines.
            if (
                stripped.startswith("[![")
                or stripped.startswith("![")
            ):
                continue

            lines.append(line)

        return re.sub(
            r"\n{3,}",
            "\n\n",
            "\n".join(lines),
        ).strip()

    # =============================================================
    # README SECTION EXTRACTION
    # =============================================================

    @classmethod
    def _extract_relevant_readme(
        cls,
        readme: str,
        query: str | None = None,
    ) -> str:
        cleaned = cls._clean_readme(readme)

        if not cleaned:
            return ""

        lines = cleaned.splitlines()

        sections = []
        current_title = "Introduction"
        current_lines = []

        for line in lines:
            match = re.match(
                r"^\s{0,3}#{1,6}\s+(.+?)\s*$",
                line,
            )

            if match:
                if current_lines:
                    sections.append(
                        (
                            current_title,
                            "\n".join(current_lines).strip(),
                        )
                    )

                current_title = match.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                (
                    current_title,
                    "\n".join(current_lines).strip(),
                )
            )

        if not sections:
            return cleaned[:cls.MAX_README_CHARS]

        analysis = cls.analyze_query(query or "")

        query_terms = set(
            analysis.get("all_terms", [])
        )

        primary = set(
            analysis.get("primary_entities", [])
        )

        scored = []

        for index, (title, body) in enumerate(sections):
            title_lower = title.lower()
            body_lower = body.lower()

            score = 0.0

            if query:
                for term in primary:
                    if term in title_lower:
                        score += 80
                    if term in body_lower:
                        score += 20

                for term in query_terms:
                    if term in title_lower:
                        score += 15
                    elif term in body_lower:
                        score += 3

            # Technical sections are useful global context.
            if any(
                word in title_lower
                for word in (
                    "installation",
                    "usage",
                    "quickstart",
                    "architecture",
                    "api",
                    "development",
                    "configuration",
                    "model",
                    "implementation",
                    "feature",
                    "design",
                )
            ):
                score += 5

            # Administrative sections are less useful.
            if any(
                word in title_lower
                for word in (
                    "contributing",
                    "license",
                    "citation",
                    "acknowledgement",
                    "acknowledgment",
                )
            ):
                score -= 25

            scored.append(
                (
                    score,
                    index,
                    title,
                    body,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        selected = []
        used = 0

        # Query-relevant sections first.
        for score, index, title, body in scored:
            if score <= 0 and selected:
                continue

            section = (
                f"## {title}\n"
                f"{body}"
            )

            remaining = cls.MAX_README_CHARS - used

            if remaining <= 0:
                break

            if len(section) <= remaining:
                selected.append(
                    (
                        index,
                        section,
                    )
                )
                used += len(section) + 2
            elif remaining > 500:
                partial = section[:remaining]
                partial = partial.rsplit(
                    "\n",
                    1,
                )[0]

                selected.append(
                    (
                        index,
                        partial,
                    )
                )
                break

        selected.sort(
            key=lambda item: item[0]
        )

        result = "\n\n".join(
            text
            for _, text in selected
        )

        return result.strip() or cleaned[:cls.MAX_README_CHARS]

    # =============================================================
    # PYTHON SYMBOL EXTRACTION
    # =============================================================

    @staticmethod
    def _python_symbols(
        content: str,
    ) -> list[dict]:
        """
        Extract classes/functions/methods from Python code.

        This is used only for source focusing and metadata. It is not
        a replacement for a full AST-based code chunker.
        """

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        lines = content.splitlines()
        symbols = []

        def visit(
            node: ast.AST,
            parent: str | None = None,
        ) -> None:
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                start = getattr(
                    node,
                    "lineno",
                    1,
                )

                end = getattr(
                    node,
                    "end_lineno",
                    start,
                )

                name = getattr(
                    node,
                    "name",
                    "",
                )

                symbol_type = (
                    "class"
                    if isinstance(node, ast.ClassDef)
                    else "function"
                )

                qualified = (
                    f"{parent}.{name}"
                    if parent
                    else name
                )

                symbols.append(
                    {
                        "name": name,
                        "qualified_name": qualified,
                        "type": symbol_type,
                        "start_line": start,
                        "end_line": end,
                    }
                )

                next_parent = qualified

                for child in ast.iter_child_nodes(node):
                    visit(
                        child,
                        parent=next_parent,
                    )
                return

            for child in ast.iter_child_nodes(node):
                visit(
                    child,
                    parent=parent,
                )

        visit(tree)

        return symbols

    # Extensions handled by the generic brace-delimited symbol
    # extractor below. Languages not in this set (R, Julia, Lua, shell,
    # Ruby, etc.) still fall back to keyword-line-window matching.
    _GENERIC_SYMBOL_EXTENSIONS = {
        ".js", ".jsx", ".mjs", ".cjs",
        ".ts", ".tsx",
        ".java", ".kt", ".kts",
        ".go",
        ".rs",
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp",
        ".cs",
        ".swift",
        ".scala",
        ".dart",
        ".m", ".mm",
        ".php",
    }

    # "namespace" is deliberately excluded: it wraps almost an entire
    # file, so when a query happens to mention the repo/package name
    # (which often matches the namespace name exactly), it would "win"
    # over genuinely specific symbols and return nearly the whole file
    # instead of a focused block.
    _GENERIC_SYMBOL_KEYWORDS = (
        "class", "struct", "interface", "trait", "enum",
        "impl", "function", "func", "fn", "protocol", "extension",
    )

    _GENERIC_DECLARATION_PATTERN = re.compile(
        r"^\s*(?:export\s+|public\s+|private\s+|protected\s+"
        r"|internal\s+|static\s+|final\s+|abstract\s+|sealed\s+"
        r"|open\s+|pub\s+|pub\(crate\)\s+|async\s+|default\s+)*"
        r"(?:" + "|".join(_GENERIC_SYMBOL_KEYWORDS) + r")\b"
        r"\s*\*?\s*"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    )

    # Go methods/functions put the receiver between `func` and the
    # name (`func (s *Scaler) Fit(...)`), which the generic pattern
    # above can't extract, so it gets its own pattern.
    _GO_FUNC_PATTERN = re.compile(
        r"^\s*func\s+(?:\([^)]*\)\s+)?"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
    )

    @classmethod
    def _generic_code_symbols(
        cls,
        content: str,
        extension: str,
    ) -> list[dict]:
        """
        Best-effort function/class/struct/interface boundary extraction
        for brace-delimited languages, using declaration keywords plus
        brace-depth counting to find the end of each block.

        This is a heuristic, not a real parser: it doesn't understand
        strings, comments, or template/generic syntax containing braces.
        It exists to give non-Python repositories the same "focus on
        the actual symbol" precision Python already gets from real AST
        parsing, instead of falling straight to generic keyword-line
        windows.
        """

        lines = content.splitlines()
        symbols: list[dict] = []
        seen_starts: set[int] = set()

        patterns = [cls._GENERIC_DECLARATION_PATTERN]

        if extension == ".go":
            patterns = [cls._GO_FUNC_PATTERN, cls._GENERIC_DECLARATION_PATTERN]

        for index, line in enumerate(lines):
            name = None

            for pattern in patterns:
                match = pattern.match(line)

                if match:
                    name = match.group("name")
                    break

            if not name:
                continue

            start_line = index + 1

            if start_line in seen_starts:
                continue

            seen_starts.add(start_line)

            # Brace-depth scan to find the matching close. Bounded so a
            # pathological/minified file can't cause a huge scan.
            depth = 0
            opened = False
            end_line = start_line
            scan_limit = min(len(lines), index + 600)

            for scan_index in range(index, scan_limit):
                scan_line = lines[scan_index]
                opens = scan_line.count("{")
                closes = scan_line.count("}")

                if opens:
                    opened = True

                depth += opens - closes

                if opened and depth <= 0:
                    end_line = scan_index + 1
                    break
            else:
                end_line = min(len(lines), index + 60)

            symbols.append(
                {
                    "name": name,
                    "qualified_name": name,
                    "type": "symbol",
                    "start_line": start_line,
                    "end_line": max(end_line, start_line),
                }
            )

        return symbols

    @staticmethod
    def _merge_line_windows(
        windows: list[dict],
        lines: list[str],
        gap: int = 5,
    ) -> list[dict]:
        """
        Merge windows whose 1-indexed inclusive [start, end] line ranges
        overlap or sit within `gap` lines of each other.

        Without this, closely-clustered keyword/symbol matches (e.g.
        several struct fields each matching a query term within a few
        lines of each other) each spawn their own ~25-line window, and
        those windows - while not byte-identical - are almost entirely
        the same content. That flooded retrieval with near-duplicate
        "evidence" that was really one finding repeated several times.
        """

        if not windows:
            return []

        ordered = sorted(
            windows,
            key=lambda window: (window["start"], window["end"]),
        )

        merged = [dict(ordered[0])]

        for window in ordered[1:]:
            last = merged[-1]

            if window["start"] <= last["end"] + gap:
                last["end"] = max(last["end"], window["end"])
                last["score"] = max(last["score"], window["score"])

                if window["symbol"] and not last["symbol"]:
                    last["symbol"] = window["symbol"]
                elif (
                    window["symbol"]
                    and last["symbol"]
                    and window["symbol"] != last["symbol"]
                ):
                    last["symbol"] = f'{last["symbol"]}, {window["symbol"]}'
            else:
                merged.append(dict(window))

        for window in merged:
            start_index = max(0, window["start"] - 1)
            end_index = min(len(lines), window["end"])
            window["text"] = "\n".join(lines[start_index:end_index])

        return merged

    # =============================================================
    # SOURCE CONTENT FOCUSING
    # =============================================================

    @classmethod
    def _focus_source_content(
        cls,
        path: str,
        content: str,
        query: str,
        max_chars: int,
    ) -> tuple[str, list[dict]]:
        """
        Return a query-focused portion of a source file.

        Real symbol boundaries (functions/classes/structs/etc.) are
        used for Python (via AST) and for common brace-delimited
        languages (via heuristic brace matching). Other languages fall
        back to textual identifier/keyword windows.

        If no strong match exists, a bounded beginning of the file is
        retained because imports, module-level constants and class
        declarations often provide useful context.
        """

        cleaned = cls._clean_file_content(content)

        if not cleaned:
            return "", []

        analysis = cls.analyze_query(query)

        primary = set(
            analysis.get("primary_entities", [])
        )

        identifiers = set(
            analysis.get("identifiers", [])
        )

        supporting = set(
            analysis.get("supporting_terms", [])
        )
        identifier_parts = set(
            analysis.get("identifier_parts", [])
        )

        target_terms = (
            primary
            | identifiers
            | identifier_parts
            | supporting
        )

        extension = Path(path).suffix.lower()
        symbols = []

        if extension in {".py", ".pyi"}:
            symbols = cls._python_symbols(cleaned)
        elif extension in cls._GENERIC_SYMBOL_EXTENSIONS:
            symbols = cls._generic_code_symbols(cleaned, extension)

        # ---------------------------------------------------------
        # Symbol matching.
        # ---------------------------------------------------------

        symbol_hits = []

        for symbol in symbols:
            name = symbol["name"].lower()
            qualified = symbol["qualified_name"].lower()

            score = 0

            for term in primary:
                if term == name:
                    score += 200
                elif term == qualified:
                    score += 220
                elif term in name:
                    score += 100

            for term in identifiers:
                if term == name:
                    score += 120
                elif term in qualified:
                    score += 60

            # Supporting terms (generic words like "beam"/"search" in
            # "how does beam search work") are weaker signals than
            # primary entities or identifiers, but a multi-word symbol
            # name built from several supporting-term hits (e.g.
            # BeamSearch matching both "beam" and "search") is a real
            # signal that shouldn't lose to a plain keyword-line window.
            for term in supporting:
                if len(term) < 3:
                    continue

                if term == name:
                    score += 40
                elif term in name:
                    score += 20

            if score > 0:
                symbol_hits.append(
                    (
                        score,
                        symbol,
                    )
                )

        symbol_hits.sort(
            key=lambda item: (
                -item[0],
                item[1]["start_line"],
            )
        )

        lines = cleaned.splitlines()

        windows = []

        # ---------------------------------------------------------
        # Include exact symbol blocks first.
        # ---------------------------------------------------------

        for score, symbol in symbol_hits[:5]:
            start = max(
                1,
                symbol["start_line"] - 8,
            )

            end = min(
                len(lines),
                symbol["end_line"],
            )

            # Defensive cap: a very large symbol (a big class/impl
            # block) shouldn't degenerate "focusing" into returning
            # almost the entire file.
            end = min(end, start + 150)

            block = "\n".join(
                lines[start - 1:end]
            )

            if block:
                windows.append(
                    {
                        "start": start,
                        "end": end,
                        "score": 200 + score,
                        "text": block,
                        "symbol": symbol["qualified_name"],
                    }
                )

        # ---------------------------------------------------------
        # Textual matching windows (only used when symbol matching
        # found nothing - e.g. unsupported language, or a match that
        # isn't inside any recognized symbol).
        # ---------------------------------------------------------

        if not windows and target_terms:
            for index, line in enumerate(lines):
                lower = line.lower()

                if not any(
                    term in lower
                    for term in target_terms
                    if len(term) >= 2
                ):
                    continue

                start = max(
                    0,
                    index - 8,
                )

                end = min(
                    len(lines),
                    index + 18,
                )

                block = "\n".join(
                    lines[start:end]
                )

                windows.append(
                    {
                        "start": start + 1,
                        "end": end,
                        "score": 100,
                        "text": block,
                        "symbol": None,
                    }
                )

        # ---------------------------------------------------------
        # Merge overlapping/near-adjacent windows before doing
        # anything else, so clustered matches collapse into one
        # window instead of several near-duplicates. Then keep only
        # the strongest few.
        # ---------------------------------------------------------

        windows = cls._merge_line_windows(windows, lines)

        windows.sort(
            key=lambda window: -window["score"]
        )

        windows = windows[:4]

        # ---------------------------------------------------------
        # Add module header/imports for code context.
        # ---------------------------------------------------------

        header = "\n".join(
            lines[: min(35, len(lines))]
        )

        # ---------------------------------------------------------
        # If we found relevant blocks, merge without duplicates.
        # ---------------------------------------------------------

        pieces = []

        if header:
            pieces.append(
                (
                    0,
                    0,
                    header,
                )
            )

        for window in windows:
            pieces.append(
                (
                    window["start"],
                    window["end"],
                    window["text"],
                )
            )

        # Deduplicate exact text.
        seen = set()
        final_parts = []

        for start, end, text in pieces:
            if text in seen:
                continue

            seen.add(text)

            label = (
                f"# Lines {start}-{end}\n"
                if end
                else "# Module header\n"
            )

            final_parts.append(
                label + text
            )

        result = "\n\n".join(
            final_parts
        )

        if len(result) > max_chars:
            result = result[:max_chars]

            # Prefer ending at a complete line.
            result = result.rsplit(
                "\n",
                1,
            )[0]

        metadata = [
            {
                "symbol": window["symbol"],
                "start_line": window["start"],
                "end_line": window["end"],
            }
            for window in windows
        ]

        return result.strip(), metadata

    # =============================================================
    # CONTENT-LEVEL FILE SCORING
    # =============================================================

    @classmethod
    def _score_file_content(
        cls,
        path: str,
        content: str,
        query: str,
        path_score: float,
    ) -> tuple[float, list[str]]:
        analysis = cls.analyze_query(query)

        primary = analysis["primary_entities"]
        supporting = analysis["supporting_terms"]

        lower_content = content.lower()

        score = path_score
        reasons = []

        for term in primary:
            if not term:
                continue

            count = lower_content.count(term)

            if count:
                score += min(
                    140,
                    45 + (count * 8),
                )
                reasons.append(
                    f"primary term content match: {term}"
                )

        for term in supporting:
            if not term:
                continue

            count = lower_content.count(term)

            if count:
                score += min(
                    35,
                    count * 4,
                )
                reasons.append(
                    f"supporting term content match: {term}"
                )

        # Exact class/function declaration is a very strong signal.
        for term in primary:
            escaped = re.escape(term)

            if re.search(
                rf"\b(?:class|def|function|interface|struct)\s+"
                rf"{escaped}\b",
                lower_content,
            ):
                score += 180
                reasons.append(
                    f"explicit symbol declaration: {term}"
                )

        return score, sorted(set(reasons))

    # =============================================================
    # QUERY-AWARE CONTEXT BUILD
    # =============================================================

    @classmethod
    def build_documents_for_query(
        cls,
        github_url: str,
        query: str,
        branch: str | None = None,
    ) -> list[dict]:
        """
        Build structured, query-focused documents for the RAG pipeline.

        Unlike build_context_for_query(), this method does NOT combine
        repository content into one large string. Each selected source file
        remains a separate document so DocumentChunker can chunk files
        independently and preserve file-level metadata.
        """

        metadata = cls.fetch_repository_metadata(github_url)
        resolved_branch = branch or metadata.get("default_branch") or "main"

        analysis = cls.analyze_query(query)

        candidates = cls.select_relevant_source_files(
            github_url=github_url,
            query=query,
            branch=resolved_branch,
            max_files=cls.MAX_CANDIDATE_FILES,
        )

        if not candidates:
            return []

        scored_candidates = []

        for candidate in candidates:
            path = candidate["path"]

            try:
                content = cls.fetch_file(
                    github_url=github_url,
                    file_path=path,
                    branch=resolved_branch,
                )
            except Exception:
                continue

            if not content:
                continue

            content_for_scoring = content[
                : cls.MAX_SOURCE_FILE_FETCH_CHARS
            ]

            score, content_reasons = cls._score_file_content(
                path=path,
                content=content_for_scoring,
                query=query,
                path_score=float(candidate.get("score", 0.0)),
            )

            # Installation-style questions ("how do I install this",
            # "what are the requirements") are best answered by build/
            # dependency files, which otherwise rarely win on generic
            # content scoring alone.
            if analysis.get("wants_installation") and cls._is_config_file(
                path
            ):
                score += 150.0
                content_reasons = list(content_reasons) + [
                    "installation-relevant config file"
                ]

            scored_candidates.append(
                {
                    **candidate,
                    "content": content,
                    "score": round(score, 3),
                    "reasons": sorted(
                        set(
                            candidate.get("reasons", [])
                            + content_reasons
                        )
                    ),
                }
            )

        # ------------------------------------------------------------
        # README as a real candidate.
        #
        # select_relevant_source_files almost always returns *some*
        # file even for broad questions like "what is this repository
        # about", so README needs to compete on the same scoreboard as
        # source files rather than only appearing when nothing else was
        # found. It gets a strong score boost for overview and
        # installation questions (READMEs conventionally document
        # both). Specific code questions still get outscored by the
        # actual matching source file, since their score comes from
        # real symbol/identifier hits.
        # ------------------------------------------------------------

        try:
            readme = cls.fetch_readme(github_url)
        except Exception:
            readme = ""

        if readme:
            relevant_readme = cls._extract_relevant_readme(
                readme,
                query=query,
            )

            if relevant_readme:
                if analysis.get("wants_overview"):
                    readme_score = 500.0
                    readme_reason = "repository overview question"
                elif analysis.get("wants_installation"):
                    readme_score = 300.0
                    readme_reason = "installation question"
                else:
                    readme_score = 20.0
                    readme_reason = "documentation context"

                scored_candidates.append(
                    {
                        "path": "README.md",
                        "content": relevant_readme,
                        "score": readme_score,
                        "reasons": [readme_reason],
                        "matched_terms": [],
                        "is_readme": True,
                    }
                )

        if not scored_candidates:
            return []

        scored_candidates.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                len(item.get("path", "")),
                item.get("path", ""),
            )
        )

        # Keep the same selection policy used by the existing
        # query-aware context builder.
        selected = scored_candidates[
            : cls.MAX_QUERY_SOURCE_FILES
        ]

        documents = []

        for rank, item in enumerate(selected, start=1):
            path = item["path"]
            content = item["content"]

            if item.get("is_readme"):
                # Already curated by _extract_relevant_readme; don't run
                # it through code-focused symbol/window matching, which
                # would drop straight to a 35-line header for READMEs
                # with no literal keyword hits.
                focused = content[: cls.MAX_SOURCE_FILE_CONTEXT_CHARS]
                symbol_metadata = []
            else:
                focused, symbol_metadata = cls._focus_source_content(
                    path=path,
                    content=content,
                    query=query,
                    max_chars=cls.MAX_SOURCE_FILE_CONTEXT_CHARS,
                )

            if not focused:
                continue

            # Keep the document self-contained while preserving the
            # original source path and retrieval metadata.
            documents.append(
                {
                    "content": focused,
                    "path": path,
                    "source": "github",
                    "category": (
                        "test"
                        if cls._is_test_file(path)
                        else "benchmark"
                        if cls._is_benchmark_file(path)
                        else "documentation"
                        if cls._is_documentation_file(path)
                        else "configuration"
                        if cls._is_config_file(path)
                        else "source"
                    ),
                    "language": Path(path).suffix.lower().lstrip("."),
                    "github_url": github_url,
                    "repository": (
                        f"{metadata.get('owner', '')}/"
                        f"{metadata.get('repository', '')}"
                    ),
                    "branch": resolved_branch,
                    "github_rank": rank,
                    "github_score": item.get("score", 0.0),
                    "github_matched_terms": item.get(
                        "matched_terms", []
                    ),
                    "github_selection_reasons": item.get(
                        "reasons", []
                    ),
                    "symbols": symbol_metadata,
                    "query": query,
                    "primary_entities": analysis.get(
                        "primary_entities", []
                    ),
                }
            )

        return documents

    @classmethod
    def build_context_for_query(
        cls,
        github_url: str,
        query: str,
        branch: str | None = None,
    ) -> str:
        """
        Build a query-aware repository context.

        This is the method the backend should use for chat-with-GitHub.

        It combines:

        - repository identity
        - relevant README sections
        - important configuration files
        - query-selected source files
        - focused source-code windows
        - file/line metadata

        The complete repository is never inserted into the prompt.
        """

        owner, repository = cls._parse_github_url(github_url)

        metadata = cls.fetch_repository_metadata(
            github_url
        )

        resolved_branch = branch or metadata.get("default_branch") or "main"

        readme = cls.fetch_readme(
            github_url
        )

        relevant_readme = cls._extract_relevant_readme(
            readme,
            query=query,
        )

        # ---------------------------------------------------------
        # Candidate selection.
        # ---------------------------------------------------------

        candidates = cls.select_relevant_source_files(
            github_url=github_url,
            query=query,
            branch=resolved_branch,
            max_files=cls.MAX_CANDIDATE_FILES,
        )

        # ---------------------------------------------------------
        # Fetch candidate contents and rerank using actual content.
        # ---------------------------------------------------------

        scored_candidates = []

        for candidate in candidates:
            path = candidate["path"]

            try:
                content = cls.fetch_file(
                    github_url,
                    path,
                    branch=resolved_branch,
                )
            except Exception:
                continue

            if not content:
                continue

            if len(content) > cls.MAX_SOURCE_FILE_FETCH_CHARS:
                content_for_scoring = content[
                    : cls.MAX_SOURCE_FILE_FETCH_CHARS
                ]
            else:
                content_for_scoring = content

            score, content_reasons = (
                cls._score_file_content(
                    path=path,
                    content=content_for_scoring,
                    query=query,
                    path_score=candidate["score"],
                )
            )

            scored_candidates.append(
                {
                    **candidate,
                    "content": content,
                    "score": round(score, 3),
                    "reasons": sorted(
                        set(
                            candidate["reasons"]
                            + content_reasons
                        )
                    ),
                }
            )

        scored_candidates.sort(
            key=lambda item: (
                -item["score"],
                len(item["path"]),
                item["path"],
            )
        )

        selected = scored_candidates[
            : cls.MAX_QUERY_SOURCE_FILES
        ]

        # ---------------------------------------------------------
        # Build final context.
        # ---------------------------------------------------------

        sections = []

        sections.append(
            f"GitHub Repository: {owner}/{repository}"
        )

        if metadata.get("description"):
            sections.append(
                "Repository description: "
                + str(metadata["description"])
            )

        if metadata.get("language"):
            sections.append(
                "Primary language: "
                + str(metadata["language"])
            )

        if relevant_readme:
            sections.append(
                "===== PROJECT INFORMATION =====\n"
                + relevant_readme
            )

        # ---------------------------------------------------------
        # Selected source files.
        # ---------------------------------------------------------

        source_sections = [
            "===== QUERY-RELEVANT SOURCE FILES ====="
        ]

        used_source_chars = 0

        for rank, item in enumerate(
            selected,
            start=1,
        ):
            remaining = (
                cls.MAX_SOURCE_CONTEXT_CHARS
                - used_source_chars
            )

            if remaining <= 0:
                break

            per_file_limit = min(
                cls.MAX_SOURCE_FILE_CONTEXT_CHARS,
                remaining,
            )

            focused, symbol_metadata = (
                cls._focus_source_content(
                    path=item["path"],
                    content=item["content"],
                    query=query,
                    max_chars=per_file_limit,
                )
            )

            if not focused:
                continue

            source_sections.append(
                f"\n### Evidence {rank}: "
                f"{item['path']}\n"
                f"Selection score: {item['score']}\n"
                f"Selection reasons: "
                f"{'; '.join(item['reasons'])}\n"
                + (
                    "Symbols:\n"
                    + "\n".join(
                        f"- {m['symbol']} "
                        f"(lines {m['start_line']}-"
                        f"{m['end_line']})"
                        for m in symbol_metadata
                        if m.get("symbol")
                    )
                    + "\n"
                    if any(
                        m.get("symbol")
                        for m in symbol_metadata
                    )
                    else ""
                )
                + focused
            )

            used_source_chars += len(focused)

        if len(source_sections) > 1:
            sections.append(
                "\n".join(source_sections)
            )

        context = "\n\n".join(
            sections
        ).strip()

        if len(context) > cls.MAX_CONTEXT_CHARS:
            context = context[
                : cls.MAX_CONTEXT_CHARS
            ].rsplit(
                "\n",
                1,
            )[0]

        return context.strip()

    # =============================================================
    # LEGACY / BACKWARD-COMPATIBLE CONTEXT BUILD
    # =============================================================

    @classmethod
    def build_context(
        cls,
        github_url: str,
    ) -> str:
        """
        Backwards-compatible repository context builder.

        Existing callers can continue using this method.

        For chat queries, use build_context_for_query() instead.
        """

        owner, repository = cls._parse_github_url(
            github_url
        )

        readme = cls.fetch_readme(
            github_url
        )

        relevant_readme = cls._extract_relevant_readme(
            readme
        )

        sections = [
            f"GitHub Repository: {owner}/{repository}"
        ]

        if relevant_readme:
            sections.append(
                "===== PROJECT INFORMATION =====\n"
                + relevant_readme
            )

        # Global configuration files.
        try:
            root = cls.fetch_root_contents(
                github_url
            )
        except Exception:
            root = []

        config_sections = [
            "===== IMPORTANT FILES ====="
        ]

        for item in root:
            if item.get("type") != "file":
                continue

            path = str(
                item.get("path") or ""
            )

            if not cls._is_config_file(path):
                continue

            try:
                content = cls.fetch_file(
                    github_url,
                    path,
                )
            except Exception:
                continue

            content = cls._clean_file_content(
                content
            )

            if not content:
                continue

            config_sections.append(
                f"\n### {path}\n"
                + content[:2500]
            )

        if len(config_sections) > 1:
            sections.append(
                "\n".join(config_sections)
            )

        context = "\n\n".join(
            sections
        ).strip()

        if len(context) > cls.MAX_CONTEXT_CHARS:
            context = context[
                : cls.MAX_CONTEXT_CHARS
            ].rsplit(
                "\n",
                1,
            )[0]

        return context.strip()

    # =============================================================
    # DEBUG / EVALUATION HELPERS
    # =============================================================

    @classmethod
    def debug_query(
        cls,
        github_url: str,
        query: str,
        branch: str | None = None,
    ) -> dict:
        """
        Return the complete retrieval preparation information without
        invoking the RAG retriever or Qwen.

        Useful for evaluating this service independently.
        """

        analysis = cls.analyze_query(
            query
        )

        candidates = cls.select_relevant_source_files(
            github_url=github_url,
            query=query,
            branch=branch,
            max_files=cls.MAX_CANDIDATE_FILES,
        )

        return {
            "query": query,
            "analysis": analysis,
            "path_candidates": candidates,
        }


# =============================================================
# CLI
# =============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            '  python github_content.py '
            '<github_repository_url> '
            '"<question>"'
        )
        raise SystemExit(1)

    repository_url = sys.argv[1]
    query = sys.argv[2]

    print("=" * 80)
    print("QUERY-AWARE GITHUB CONTENT SERVICE")
    print("=" * 80)
    print(f"Repository: {repository_url}")
    print(f"Query:      {query}")
    print()

    try:
        analysis = (
            GitHubContentService.analyze_query(
                query
            )
        )

        print("QUERY ANALYSIS")
        print("-" * 80)
        print(
            "Primary entities:   ",
            analysis["primary_entities"],
        )
        print(
            "Supporting terms:   ",
            analysis["supporting_terms"],
        )
        print(
            "Intents:             ",
            analysis["intents"],
        )
        print(
            "Identifiers:         ",
            analysis["identifiers"],
        )
        print()

        candidates = (
            GitHubContentService.select_relevant_source_files(
                github_url=repository_url,
                query=query,
                max_files=10,
            )
        )

        print("PATH-LEVEL CANDIDATES")
        print("-" * 80)

        for index, item in enumerate(
            candidates,
            start=1,
        ):
            print(
                f"{index}. {item['path']}"
                f"  score={item['score']}"
            )
            print(
                f"   matched: "
                f"{', '.join(item['matched_terms'])}"
            )

        print()

        print("BUILDING QUERY-AWARE DOCUMENTS...")
        print("-" * 80)

        documents = (
            GitHubContentService.build_documents_for_query(
                github_url=repository_url,
                query=query,
            )
        )

        print(f"Documents created: {len(documents)}")
        for index, document in enumerate(documents, start=1):
            print(
                f"{index}. {document['path']} "
                f"score={document['github_score']}"
            )

        print()
        print("BUILDING QUERY-AWARE CONTEXT...")
        print("-" * 80)

        context = (
            GitHubContentService.build_context_for_query(
                github_url=repository_url,
                query=query,
            )
        )

        print(
            f"Final context characters: "
            f"{len(context):,}"
        )
        print()

        print("CONTEXT PREVIEW")
        print("-" * 80)
        print(context[:12000])

        print()
        print("=" * 80)
        print("SUCCESS")
        print("=" * 80)

    except Exception as exc:
        print()
        print("GitHub content processing failed:")
        print(str(exc))
        raise
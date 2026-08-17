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


# =============================================================
# ENVIRONMENT
# =============================================================

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


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

    # Jupyter notebooks are common in research/tutorial repositories and
    # must be searchable for repository questions.
    NOTEBOOK_EXTENSIONS = {".ipynb"}

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
            "intro",
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

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}"
        )

        response = cls._get(url)
        data = response.json()

        return {
            "owner": owner,
            "repository": repository,
            "default_branch": data.get("default_branch") or "main",
            "description": data.get("description"),
            "language": data.get("language"),
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "url": data.get("html_url") or github_url,
        }

    # =============================================================
    # README
    # =============================================================

    @classmethod
    def fetch_readme(
        cls,
        github_url: str,
    ) -> str:
        owner, repository = cls._parse_github_url(github_url)

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}/readme"
        )

        response = cls._get(url)
        data = response.json()

        content = data.get("content")

        if not content:
            return ""

        try:
            return base64.b64decode(content).decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            return ""

    # =============================================================
    # SINGLE FILE
    # =============================================================

    @classmethod
    def fetch_file(
        cls,
        github_url: str,
        file_path: str,
    ) -> str:
        owner, repository = cls._parse_github_url(github_url)

        file_path = file_path.strip("/")

        if not file_path:
            raise ValueError("File path must not be empty.")

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}"
            f"/contents/{file_path}"
        )

        response = cls._get(url, timeout=45)
        data = response.json()

        if data.get("type") != "file":
            raise ValueError(f"Path is not a file: {file_path}")

        content = data.get("content")

        if not content:
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
                return raw_response.text

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

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}/contents"
        )

        response = cls._get(url)
        data = response.json()

        return data if isinstance(data, list) else []

    # =============================================================
    # COMPLETE REPOSITORY TREE
    # =============================================================

    @classmethod
    def fetch_repository_tree(
        cls,
        github_url: str,
        branch: str | None = None,
    ) -> list[str]:
        metadata = cls.fetch_repository_metadata(github_url)

        branch = branch or metadata["default_branch"]
        owner = metadata["owner"]
        repository = metadata["repository"]

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}"
            f"/git/trees/{branch}"
            f"?recursive=1"
        )

        response = cls._get(
            url,
            timeout=60,
        )

        data = response.json()

        if data.get("truncated"):
            raise ValueError(
                "GitHub returned a truncated repository tree. "
                "The repository is too large for a complete recursive "
                "tree request."
            )

        tree = data.get("tree")

        if not isinstance(tree, list):
            raise ValueError(
                "Unexpected GitHub repository tree response."
            )

        paths = []

        for item in tree:
            if item.get("type") != "blob":
                continue

            path = str(item.get("path") or "").strip()

            if path:
                paths.append(path)

        if len(paths) > cls.MAX_TREE_FILES:
            raise ValueError(
                f"Repository contains more than "
                f"{cls.MAX_TREE_FILES:,} files; "
                "tree discovery safety limit reached."
            )

        return sorted(set(paths))

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
                "intents": [],
                "file_hints": [],
                "wants_code": False,
                "wants_tests": False,
                "wants_docs": False,
                "wants_benchmarks": False,
                "wants_overview": True,
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
        # Detect technical identifiers.
        # ---------------------------------------------------------

        identifiers = []

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
                normalized = clean.lower()

                if (
                    normalized not in cls.STOP_WORDS
                    and normalized not in identifiers
                ):
                    identifiers.append(normalized)

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

        # A question is "overview-style" if it explicitly signals that
        # intent, OR if it produced no other real signal to search with
        # (no primary entities, no identifiers, no specific intent).
        # The latter case covers phrasing like "what is this repository
        # about" or "what does this project do", where stop-word
        # stripping removes almost every token.
        wants_overview = (
            "overview" in intents
            or (
                not primary_entities
                and not identifiers
                and not wants_code
                and not wants_tests
                and not wants_docs
                and not wants_benchmarks
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
            "intents": intents,
            "file_hints": file_hints,
            "wants_code": wants_code,
            "wants_tests": wants_tests,
            "wants_docs": wants_docs,
            "wants_benchmarks": wants_benchmarks,
            "wants_overview": wants_overview,
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
        # Generic all-term matching is weak.
        # ---------------------------------------------------------

        for term in all_terms:
            if term in filename and term not in primary:
                score += 4

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

        tree = cls.fetch_repository_tree(
            github_url,
            branch=branch,
        )

        candidates = []

        for path in tree:
            normalized = path.replace("\\", "/")

            # We support source files, docs and configuration files.
            supported = (
                cls._is_source_file(normalized)
                or Path(normalized).suffix.lower()
                in cls.NOTEBOOK_EXTENSIONS
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
    def _clean_notebook_content(content: str) -> str:
        """Convert a raw .ipynb JSON document into searchable notebook text."""
        if not content:
            return ""

        try:
            notebook = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content

        cells = notebook.get("cells", [])
        if not isinstance(cells, list):
            return content

        parts = []

        for index, cell in enumerate(cells, start=1):
            if not isinstance(cell, dict):
                continue

            cell_type = cell.get("cell_type", "unknown")
            source = cell.get("source", "")

            if isinstance(source, list):
                source = "".join(source)

            if not isinstance(source, str) or not source.strip():
                continue

            parts.append(
                f"# Notebook cell {index} [{cell_type}]\n"
                f"{source.strip()}"
            )

        return "\n\n".join(parts).strip()

    @staticmethod
    def _clean_file_content(
        content: str,
    ) -> str:
        if not content:
            return ""

        # Convert Jupyter notebooks into readable code/markdown before
        # normal whitespace cleanup.
        if isinstance(content, str):
            # The caller does not need to know whether the source is a
            # notebook; notebook JSON is detected by the file-specific
            # path in _focus_source_content.
            pass

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

        For Python files, exact symbols/classes/functions are preferred.
        For other languages, textual identifier/keyword windows are used.

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

        target_terms = (
            primary
            | identifiers
            | supporting
        )

        symbols = []

        if Path(path).suffix.lower() in {
            ".py",
            ".pyi",
        }:
            symbols = cls._python_symbols(cleaned)

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

        for _, symbol in symbol_hits[:3]:
            start = max(
                1,
                symbol["start_line"] - 8,
            )

            end = min(
                len(lines),
                symbol["end_line"],
            )

            block = "\n".join(
                lines[start - 1:end]
            )

            if block:
                windows.append(
                    {
                        "start": start,
                        "end": end,
                        "score": 200,
                        "text": block,
                        "symbol": symbol["qualified_name"],
                    }
                )

        # ---------------------------------------------------------
        # Textual matching windows.
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

                if len(windows) >= 4:
                    break

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
        if Path(path).suffix.lower() in cls.NOTEBOOK_EXTENSIONS:
            content = cls._clean_notebook_content(content)

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
        # README previously only appeared when NO source file scored at
        # all. In practice, select_relevant_source_files almost always
        # returns *some* file even for broad questions like "what is
        # this repository about", so that fallback rarely triggered and
        # an arbitrary/generic source file won by default instead.
        #
        # README now competes on the same scoreboard as source files,
        # with a strong score boost when the question looks like an
        # overview question. Specific code questions still get outscored
        # by the actual matching source file, since their score comes
        # from real symbol/identifier hits.
        # ------------------------------------------------------------

        try:
            readme = cls.fetch_readme(github_url)
        except Exception:
            readme = ""

        readme_added = False

        if readme:
            relevant_readme = cls._extract_relevant_readme(
                readme,
                query=query,
            )

            if relevant_readme:
                readme_score = (
                    500.0
                    if analysis.get("wants_overview")
                    else 20.0
                )

                scored_candidates.append(
                    {
                        "path": "README.md",
                        "content": relevant_readme,
                        "score": readme_score,
                        "reasons": [
                            "repository overview question"
                            if analysis.get("wants_overview")
                            else "documentation context"
                        ],
                        "matched_terms": [],
                        "is_readme": True,
                    }
                )
                readme_added = True

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
                        "notebook"
                        if Path(path).suffix.lower()
                        in cls.NOTEBOOK_EXTENSIONS
                        else "test"
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
            branch=branch,
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
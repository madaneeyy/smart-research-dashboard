import base64
import os
import re
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
    Fetch research-related content from public GitHub repositories.

    Pipeline:

        GitHub Repository
              ↓
        README extraction
              ↓
        Important files
              ↓
        Curated research context
              ↓
        Chunking
              ↓
        Embedding-based RAG
    """

    API_BASE_URL = "https://api.github.com"

    # =========================================================
    # CONTEXT LIMITS
    # =========================================================

    # We now allow a larger context because the embedding
    # retriever will decide which parts are actually relevant.
    #
    # Target:
    # approximately 8,000 - 15,000 characters
    #
    # Hard maximum:
    # 18,000 characters
    MAX_CONTEXT_CHARS = 18000

    # Maximum README content we are willing to keep.
    MAX_README_CHARS = 12000

    # Maximum content from each important configuration file.
    MAX_FILE_CHARS = 2500

    # =========================================================
    # GITHUB URL PARSING
    # =========================================================

    @staticmethod
    def _parse_github_url(
        url: str,
    ) -> tuple[str, str]:
        """
        Extract owner and repository from a GitHub URL.
        """

        if not url:
            raise ValueError(
                "GitHub URL must not be empty."
            )

        parsed = urlparse(
            str(url).strip()
        )

        if parsed.netloc.lower() not in {
            "github.com",
            "www.github.com",
        }:
            raise ValueError(
                f"Not a GitHub URL: {url}"
            )

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
        repository = parts[1]

        repository = re.sub(
            r"\.git$",
            "",
            repository,
            flags=re.IGNORECASE,
        )

        return owner, repository

    # =========================================================
    # REQUEST HEADERS
    # =========================================================

    @staticmethod
    def _headers() -> dict[str, str]:
        """
        Build headers for GitHub API requests.
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

    # =========================================================
    # FETCH README
    # =========================================================

    @classmethod
    def fetch_readme(
        cls,
        github_url: str,
    ) -> str:
        """
        Fetch the repository README.
        """

        owner, repository = (
            cls._parse_github_url(
                github_url
            )
        )

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}/readme"
        )

        response = requests.get(
            url,
            headers=cls._headers(),
            timeout=30,
        )

        if response.status_code == 404:
            raise ValueError(
                "GitHub repository or README "
                "not found."
            )

        response.raise_for_status()

        data = response.json()

        content = data.get(
            "content"
        )

        if not content:
            raise ValueError(
                "GitHub returned no README content."
            )

        try:
            return base64.b64decode(
                content
            ).decode(
                "utf-8",
                errors="replace",
            )

        except Exception as exc:
            raise ValueError(
                "Could not decode GitHub README."
            ) from exc

    # =========================================================
    # FETCH A SINGLE FILE
    # =========================================================

    @classmethod
    def fetch_file(
        cls,
        github_url: str,
        file_path: str,
    ) -> str:
        """
        Fetch a single file from a GitHub repository.
        """

        owner, repository = (
            cls._parse_github_url(
                github_url
            )
        )

        file_path = file_path.strip(
            "/"
        )

        if not file_path:
            raise ValueError(
                "File path must not be empty."
            )

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}"
            f"/contents/{file_path}"
        )

        response = requests.get(
            url,
            headers=cls._headers(),
            timeout=30,
        )

        if response.status_code == 404:
            raise ValueError(
                f"File not found: {file_path}"
            )

        response.raise_for_status()

        data = response.json()

        if data.get("type") != "file":
            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        content = data.get(
            "content"
        )

        if not content:
            raise ValueError(
                f"No content returned for: {file_path}"
            )

        if data.get("encoding") != "base64":
            raise ValueError(
                f"Unsupported encoding for: {file_path}"
            )

        try:
            return base64.b64decode(
                content
            ).decode(
                "utf-8",
                errors="replace",
            )

        except Exception as exc:
            raise ValueError(
                f"Could not decode file: {file_path}"
            ) from exc

    # =========================================================
    # GET ROOT DIRECTORY
    # =========================================================

    @classmethod
    def fetch_root_contents(
        cls,
        github_url: str,
    ) -> list[dict]:
        """
        Get files and directories from repository root.
        """

        owner, repository = (
            cls._parse_github_url(
                github_url
            )
        )

        url = (
            f"{cls.API_BASE_URL}"
            f"/repos/{owner}/{repository}"
            f"/contents"
        )

        response = requests.get(
            url,
            headers=cls._headers(),
            timeout=30,
        )

        if response.status_code == 404:
            raise ValueError(
                "GitHub repository not found."
            )

        response.raise_for_status()

        data = response.json()

        if not isinstance(
            data,
            list,
        ):
            raise ValueError(
                "Unexpected GitHub directory response."
            )

        return data

    # =========================================================
    # FIND IMPORTANT FILES
    # =========================================================

    @classmethod
    def find_important_files(
        cls,
        github_url: str,
    ) -> list[str]:
        """
        Identify useful project/configuration files.
        """

        contents = (
            cls.fetch_root_contents(
                github_url
            )
        )

        important_names = {
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "package.json",
            "environment.yml",
            "environment.yaml",
            "dockerfile",
            "makefile",
        }

        important_files = []

        for item in contents:

            if item.get("type") != "file":
                continue

            name = (
                item.get("name", "")
                .strip()
                .lower()
            )

            if name in important_names:

                path = item.get(
                    "path"
                )

                if path:
                    important_files.append(
                        path
                    )

        return important_files

    # =========================================================
    # FETCH IMPORTANT FILES
    # =========================================================

    @classmethod
    def fetch_important_files(
        cls,
        github_url: str,
    ) -> dict[str, str]:
        """
        Fetch important configuration files.
        """

        files = (
            cls.find_important_files(
                github_url
            )
        )

        results = {}

        for file_path in files:

            try:

                content = cls.fetch_file(
                    github_url,
                    file_path,
                )

                results[file_path] = content

            except Exception:
                # One failed file should not break
                # the entire repository analysis.
                continue

        return results

    # =========================================================
    # CLEAN README
    # =========================================================

    @staticmethod
    def _clean_readme(
        readme: str,
    ) -> str:
        """
        Remove obvious README noise while preserving
        useful technical information.
        """

        if not readme:
            return ""

        text = readme

        # Remove HTML comments.
        text = re.sub(
            r"<!--.*?-->",
            "",
            text,
            flags=re.DOTALL,
        )

        # Remove HTML tags.
        text = re.sub(
            r"<[^>]+>",
            "",
            text,
        )

        cleaned_lines = []

        for line in text.splitlines():

            stripped = line.strip()

            if not stripped:
                continue

            # Remove badge/image-only lines.
            if (
                stripped.startswith("[![")
                or stripped.startswith("![")
            ):
                continue

            cleaned_lines.append(
                line
            )

        text = "\n".join(
            cleaned_lines
        )

        # Normalize excessive blank lines.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # =========================================================
    # EXTRACT RELEVANT README SECTIONS
    # =========================================================
    @classmethod
    def _extract_relevant_readme(
        cls,
        readme: str,
    ) -> str:
        """
        Extract a large, useful portion of the README
        for embedding-based RAG.

        We keep important technical sections while
        avoiding administrative sections such as
        contributing, licensing, and citations.
        """

        cleaned = cls._clean_readme(readme)

        if not cleaned:
            return ""

        lines = cleaned.splitlines()

        # -----------------------------------------------------
        # Split README into sections
        # -----------------------------------------------------

        sections = []

        current_title = "Introduction"
        current_lines = []

        for line in lines:

            heading_match = re.match(
                r"^\s{0,3}#{1,6}\s+(.+?)\s*$",
                line,
            )

            if heading_match:

                # Save previous section.
                if current_lines:

                    sections.append(
                        (
                            current_title,
                            "\n".join(
                                current_lines
                            ).strip(),
                        )
                    )

                current_title = (
                    heading_match.group(1).strip()
                )

                current_lines = []

            else:
                current_lines.append(line)

        # Save final section.
        if current_lines:

            sections.append(
                (
                    current_title,
                    "\n".join(
                        current_lines
                    ).strip(),
                )
            )

        # -----------------------------------------------------
        # Keywords we consider valuable for RAG
        # -----------------------------------------------------

        priority_keywords = [
            "about",
            "overview",
            "introduction",
            "architecture",
            "component",
            "feature",
            "capabilit",
            "model",
            "training",
            "transformer",
            "research",
            "parallel",
            "performance",
            "scal",
            "framework",
            "library",
            "design",
            "method",
            "benchmark",
            "memory",
            "gpu",
            "distributed",
            "inference",
            "usage",
            "example",
            "requirement",
            "install",
            "quickstart",
            "configuration",
            "data",
        ]

        excluded_keywords = [
            "contribut",
            "license",
            "citation",
            "acknowledg",
            "security",
        ]

        # -----------------------------------------------------
        # Score sections
        # -----------------------------------------------------

        scored_sections = []

        for index, (title, body) in enumerate(sections):

            if not body:
                continue

            title_lower = title.lower()

            # Skip administrative sections.
            if any(
                keyword in title_lower
                for keyword in excluded_keywords
            ):
                continue

            score = 0

            for keyword in priority_keywords:

                if keyword in title_lower:
                    score += 3

            # Give early README sections a small bonus.
            if index <= 2:
                score += 2

            scored_sections.append(
                (
                    score,
                    index,
                    title,
                    body,
                )
            )

        # -----------------------------------------------------
        # Sort by relevance
        # -----------------------------------------------------

        scored_sections.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        # -----------------------------------------------------
        # Select sections until README budget is reached
        # -----------------------------------------------------

        selected = []

        current_length = 0

        for (
            score,
            index,
            title,
            body,
        ) in scored_sections:

            section_text = (
                f"## {title}\n"
                f"{body}"
            )

            section_length = (
                len(section_text) + 2
            )

            remaining = (
                cls.MAX_README_CHARS
                - current_length
            )

            if remaining <= 0:
                break

            if section_length <= remaining:

                selected.append(
                    (
                        index,
                        section_text,
                    )
                )

                current_length += (
                    section_length
                )

            else:

                # If there is still a reasonable amount
                # of space, keep a partial section.
                if remaining > 500:

                    partial = section_text[
                        :remaining
                    ]

                    # Avoid cutting in the middle of
                    # a line where possible.
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

        # -----------------------------------------------------
        # Restore original README order
        # -----------------------------------------------------

        selected.sort(
            key=lambda item: item[0]
        )

        result = "\n\n".join(
            section_text
            for _, section_text in selected
        )

        # -----------------------------------------------------
        # Fallback
        # -----------------------------------------------------

        if not result.strip():

            result = cleaned[
                :cls.MAX_README_CHARS
            ]

        return result.strip()

    # =========================================================
    # CLEAN IMPORTANT FILE
    # =========================================================

    @staticmethod
    def _clean_file_content(
        content: str,
    ) -> str:
        """
        Remove excessive blank lines and obvious comments
        while preserving configuration/code values.
        """

        if not content:
            return ""

        lines = []

        for line in content.splitlines():

            stripped = line.strip()

            if not stripped:
                continue

            # Don't remove code comments aggressively.
            # Configuration comments can occasionally be
            # useful, so we preserve them.

            lines.append(
                line
            )

        return "\n".join(
            lines
        ).strip()

    # =========================================================
    # BUILD RAG CONTEXT
    # =========================================================

    @classmethod
    def build_context(
        cls,
        github_url: str,
    ) -> str:
        """
        Build a larger, curated research context.

        Target:
            ~8,000 - 15,000 characters

        Hard maximum:
            18,000 characters

        The important idea is that this method should preserve
        enough information for embedding-based retrieval.

        We do NOT want to aggressively reduce the repository
        to a few hundred words.
        """

        owner, repository = (
            cls._parse_github_url(
                github_url
            )
        )

        # -----------------------------------------------------
        # Fetch README
        # -----------------------------------------------------

        readme = cls.fetch_readme(
            github_url
        )

        relevant_readme = (
            cls._extract_relevant_readme(
                readme
            )
        )

        # -----------------------------------------------------
        # Fetch important files
        # -----------------------------------------------------

        important_files = (
            cls.fetch_important_files(
                github_url
            )
        )

        # -----------------------------------------------------
        # Build context sections
        # -----------------------------------------------------

        sections = []

        # Repository identity.
        sections.append(
            f"GitHub Repository: "
            f"{owner}/{repository}"
        )

        # README.
        if relevant_readme:

            sections.append(
                "===== PROJECT INFORMATION =====\n"
                + relevant_readme
            )

        # Important files.
        if important_files:

            file_sections = [
                "===== IMPORTANT FILES ====="
            ]

            for (
                file_path,
                content,
            ) in important_files.items():

                cleaned_content = (
                    cls._clean_file_content(
                        content
                    )
                )

                if not cleaned_content:
                    continue

                if (
                    len(cleaned_content)
                    > cls.MAX_FILE_CHARS
                ):

                    cleaned_content = (
                        cleaned_content[
                            : cls.MAX_FILE_CHARS
                        ]
                        .rsplit(
                            "\n",
                            1,
                        )[0]
                    )

                file_sections.append(
                    f"\n### {file_path}\n"
                    f"{cleaned_content}"
                )

            if len(file_sections) > 1:

                sections.append(
                    "\n".join(
                        file_sections
                    )
                )

        # -----------------------------------------------------
        # Combine
        # -----------------------------------------------------

        context = "\n\n".join(
            sections
        ).strip()

        # -----------------------------------------------------
        # FINAL SAFETY LIMIT
        # -----------------------------------------------------

        if len(context) > cls.MAX_CONTEXT_CHARS:

            context = (
                context[
                    : cls.MAX_CONTEXT_CHARS
                ]
                .rsplit(
                    "\n",
                    1,
                )[0]
            )

        return context.strip()
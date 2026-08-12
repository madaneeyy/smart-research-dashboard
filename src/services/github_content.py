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

    Phase 2:
        Step 1 -> README
        Step 2 -> Important repository files
        Step 3 -> Compact AI-ready research context
    """

    API_BASE_URL = "https://api.github.com"

    # =========================================================
    # CONTEXT LIMITS
    # =========================================================

    # Target range:
    # ~4,000 - 6,000 characters
    #
    # We use 6,000 as a hard maximum so the AI receives
    # useful information without unnecessary repository noise.
    MAX_CONTEXT_CHARS = 6000

    # Maximum amount of README information we want to keep.
    MAX_README_CHARS = 4500

    # Maximum amount from each important configuration file.
    MAX_FILE_CHARS = 700

    # =========================================================
    # GITHUB URL PARSING
    # =========================================================

    @staticmethod
    def _parse_github_url(
        url: str,
    ) -> tuple[str, str]:

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

        If GITHUB_TOKEN is available, authenticate the
        request to avoid the unauthenticated API rate limit.
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

        encoding = data.get(
            "encoding"
        )

        if encoding != "base64":
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
        Get files and directories from the
        repository root.
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
        Identify useful files from the repository root.

        We intentionally keep this limited.
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
        Fetch the contents of important repository files.
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
                # Ignore an individual file failure.
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
        Remove README noise that is generally not useful
        for answering research questions.
        """

        if not readme:
            return ""

        text = readme

        # -----------------------------------------------------
        # Remove HTML comments
        # -----------------------------------------------------

        text = re.sub(
            r"<!--.*?-->",
            "",
            text,
            flags=re.DOTALL,
        )

        # -----------------------------------------------------
        # Remove HTML tags
        # -----------------------------------------------------

        text = re.sub(
            r"<[^>]+>",
            "",
            text,
        )

        # -----------------------------------------------------
        # Remove image/badge lines
        # -----------------------------------------------------

        lines = []

        for line in text.splitlines():

            stripped = line.strip()

            if not stripped:
                continue

            # Skip badge/image-heavy lines.
            if (
                stripped.startswith("[![")
                or stripped.startswith("![")
            ):
                continue

            lines.append(
                line
            )

        text = "\n".join(
            lines
        )

        # -----------------------------------------------------
        # Remove excessive blank lines
        # -----------------------------------------------------

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # =========================================================
    # EXTRACT IMPORTANT README SECTIONS
    # =========================================================

    @classmethod
    def _extract_relevant_readme(
        cls,
        readme: str,
    ) -> str:
        """
        Extract README sections that are most useful for
        answering research-related questions.

        We prioritize sections describing:

        - project overview
        - architecture
        - components
        - features
        - capabilities
        - research
        - models
        - training
        - usage/concepts

        We intentionally avoid long installation,
        contribution, and changelog sections.
        """

        cleaned = cls._clean_readme(
            readme
        )

        if not cleaned:
            return ""

        lines = cleaned.splitlines()

        # -----------------------------------------------------
        # Parse markdown sections
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
                    heading_match.group(1)
                    .strip()
                )

                current_lines = []

            else:

                current_lines.append(
                    line
                )

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
        # Keywords we care about
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
        ]

        excluded_keywords = [
            "install",
            "getting started",
            "contribut",
            "license",
            "changelog",
            "news",
            "citation",
            "acknowledg",
            "security",
            "release",
        ]

        selected = []

        # -----------------------------------------------------
        # Select relevant sections
        # -----------------------------------------------------

        for title, body in sections:

            title_lower = title.lower()

            if any(
                keyword in title_lower
                for keyword in excluded_keywords
            ):
                continue

            score = sum(
                1
                for keyword in priority_keywords
                if keyword in title_lower
            )

            if score > 0:

                selected.append(
                    (
                        score,
                        title,
                        body,
                    )
                )

        # Highest-value sections first.
        selected.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        result_parts = []

        # -----------------------------------------------------
        # Always include the introduction if useful
        # -----------------------------------------------------

        for title, body in sections:

            if title == "Introduction" and body:

                result_parts.append(
                    f"## {title}\n{body}"
                )

                break

        # -----------------------------------------------------
        # Add prioritized sections
        # -----------------------------------------------------

        for (
            _score,
            title,
            body,
        ) in selected:

            section = (
                f"## {title}\n{body}"
            )

            # Avoid duplicates.
            if section in result_parts:
                continue

            result_parts.append(
                section
            )

        result = "\n\n".join(
            result_parts
        )

        # -----------------------------------------------------
        # If section parsing didn't find enough useful
        # information, use the beginning of the README.
        # -----------------------------------------------------

        if len(result.strip()) < 1000:

            result = cleaned[:3000]

        # -----------------------------------------------------
        # Keep README portion within budget.
        # -----------------------------------------------------

        if len(result) > cls.MAX_README_CHARS:

            result = (
                result[
                    :cls.MAX_README_CHARS
                ]
                .rsplit(
                    "\n",
                    1,
                )[0]
            )

        return result.strip()

    # =========================================================
    # CLEAN IMPORTANT FILE
    # =========================================================

    @staticmethod
    def _clean_file_content(
        content: str,
    ) -> str:
        """
        Keep useful configuration information while removing
        excessive comments and blank lines.
        """

        if not content:
            return ""

        lines = []

        for line in content.splitlines():

            stripped = line.strip()

            if not stripped:
                continue

            # Skip obvious long comment-only lines.
            if stripped.startswith(
                "#"
            ):
                continue

            lines.append(
                line
            )

        return "\n".join(
            lines
        ).strip()

    # =========================================================
    # BUILD RESEARCH CONTEXT
    # =========================================================

    @classmethod
    def build_context(
        cls,
        github_url: str,
    ) -> str:
        """
        Build compact AI-ready research context.

        The context intentionally contains only information
        useful for answering questions about the repository.

        Target:
            ~4,000 - 6,000 characters

        Hard maximum:
            6,000 characters
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

        # -----------------------------------------------------
        # Fetch important configuration files
        # -----------------------------------------------------

        important_files = (
            cls.fetch_important_files(
                github_url
            )
        )

        # -----------------------------------------------------
        # Build compact context
        # -----------------------------------------------------

        context_parts = []

        # -----------------------------------------------------
        # Repository identity
        # -----------------------------------------------------

        context_parts.append(
            f"GitHub Repository: "
            f"{owner}/{repository}"
        )

        # -----------------------------------------------------
        # Relevant README
        # -----------------------------------------------------

        relevant_readme = (
            cls._extract_relevant_readme(
                readme
            )
        )

        if relevant_readme:

            context_parts.append(
                "\n===== PROJECT INFORMATION =====\n"
                + relevant_readme
            )

        # -----------------------------------------------------
        # Important files
        # -----------------------------------------------------

        if important_files:

            file_parts = []

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

                # Limit each configuration file.
                if len(cleaned_content) > cls.MAX_FILE_CHARS:

                    cleaned_content = (
                        cleaned_content[
                            :cls.MAX_FILE_CHARS
                        ]
                        .rsplit(
                            "\n",
                            1,
                        )[0]
                    )

                file_parts.append(
                    f"### {file_path}\n"
                    f"{cleaned_content}"
                )

            if file_parts:

                context_parts.append(
                    "\n===== IMPORTANT FILES =====\n"
                    + "\n\n".join(
                        file_parts
                    )
                )

        # -----------------------------------------------------
        # Combine
        # -----------------------------------------------------

        context = "\n\n".join(
            context_parts
        ).strip()

        # -----------------------------------------------------
        # FINAL SAFETY LIMIT
        #
        # This should rarely be reached because the README
        # and file limits already control the size.
        #
        # If it is reached, cut at the last complete line.
        # -----------------------------------------------------

        if len(context) > cls.MAX_CONTEXT_CHARS:

            context = (
                context[
                    :cls.MAX_CONTEXT_CHARS
                ]
                .rsplit(
                    "\n",
                    1,
                )[0]
            )

        return context.strip()
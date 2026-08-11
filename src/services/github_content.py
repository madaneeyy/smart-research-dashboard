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
    """

    API_BASE_URL = "https://api.github.com"

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
            url.strip()
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

        Example:

            fetch_file(
                url,
                "requirements.txt"
            )
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

        # -----------------------------------------------------
        # Make sure this is a file
        # -----------------------------------------------------

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

        We intentionally keep this limited for now.
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

        Returns:

            {
                "requirements.txt": "...",
                "pyproject.toml": "..."
            }
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
                # The rest of the repository can
                # still be analyzed.
                continue

        return results

    # =========================================================
    # BUILD RESEARCH CONTEXT
    # =========================================================

    @classmethod
    def build_context(
        cls,
        github_url: str,
    ) -> str:
        """
        Build AI-ready research context.

        Includes:

            1. Repository identity
            2. README
            3. Important configuration files
        """

        owner, repository = (
            cls._parse_github_url(
                github_url
            )
        )

        readme = cls.fetch_readme(
            github_url
        )

        important_files = (
            cls.fetch_important_files(
                github_url
            )
        )

        context_parts = []

        # -----------------------------------------------------
        # Repository
        # -----------------------------------------------------

        context_parts.append(
            f"GitHub Repository: "
            f"{owner}/{repository}"
        )

        # -----------------------------------------------------
        # README
        # -----------------------------------------------------

        context_parts.append(
            "\n===== README =====\n"
        )

        context_parts.append(
            readme
        )

        # -----------------------------------------------------
        # Important files
        # -----------------------------------------------------

        for (
            file_path,
            content,
        ) in important_files.items():

            context_parts.append(
                f"\n===== {file_path} =====\n"
            )

            context_parts.append(
                content
            )

        return "\n".join(
            context_parts
        )
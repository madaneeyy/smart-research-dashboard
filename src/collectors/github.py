from datetime import datetime

from pydantic import BaseModel, HttpUrl

from src.clients.github import get_github
from src.models.research import ResearchItem


# =============================================================
# GITHUB REPOSITORY MODEL
# =============================================================

class GitHubRepository(BaseModel):
    id: int
    name: str
    full_name: str
    description: str | None = None

    html_url: HttpUrl

    stars: int
    forks: int

    language: str | None = None
    topics: list[str] = []

    created_at: datetime
    updated_at: datetime


# =============================================================
# CONVERT GITHUB REPOSITORY → RESEARCH ITEM
# =============================================================

def github_repository_to_item(
    repository: GitHubRepository,
) -> ResearchItem:
    """
    Convert a GitHubRepository into the common ResearchItem model.

    Common GitHub information is mapped to the standard fields,
    while GitHub-specific information such as stars, forks, and
    language is preserved in their dedicated ResearchItem fields.
    """

    # ---------------------------------------------------------
    # Tags
    # ---------------------------------------------------------

    tags = list(repository.topics)

    if repository.language:
        tags.append(repository.language)

    # ---------------------------------------------------------
    # ResearchItem
    # ---------------------------------------------------------

    return ResearchItem(
        # -----------------------------------------------------
        # Common fields
        # -----------------------------------------------------

        id=repository.full_name,

        title=repository.name,

        description=repository.description,

        authors=[],

        source="github",

        url=repository.html_url,

        # GitHub repository creation date
        published=repository.created_at,

        # GitHub's updated_at field
        updated=repository.updated_at,

        tags=tags,

        # -----------------------------------------------------
        # GitHub-specific fields
        # -----------------------------------------------------

        stars=repository.stars,

        forks=repository.forks,

        language=repository.language,
    )


# =============================================================
# PARSE GITHUB API RESPONSE
# =============================================================

def parse_github_repository(
    data: dict,
) -> GitHubRepository:
    """
    Convert a raw GitHub API repository dictionary into a
    GitHubRepository model.
    """

    return GitHubRepository(
        id=data["id"],
        name=data["name"],
        full_name=data["full_name"],
        description=data.get("description"),

        html_url=data["html_url"],

        # GitHub API field:
        # stargazers_count → stars
        stars=data["stargazers_count"],

        # GitHub API field:
        # forks_count → forks
        forks=data["forks_count"],

        language=data.get("language"),

        topics=data.get("topics", []),

        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


# =============================================================
# SEARCH GITHUB REPOSITORIES
# =============================================================

def search_github_repositories(
    query: str,
    page: int = 1,
    per_page: int = 20,
) -> list[GitHubRepository]:
    """
    Search GitHub repositories.

    Parameters:
        query:
            GitHub search query.

        page:
            Page number. Must be >= 1.

        per_page:
            Number of repositories per page. Must be >= 1.

    Returns:
        A list of GitHubRepository objects.
    """

    # ---------------------------------------------------------
    # Validate pagination
    # ---------------------------------------------------------

    if page < 1:
        raise ValueError(
            "page must be 1 or greater"
        )

    if per_page < 1:
        raise ValueError(
            "per_page must be at least 1"
        )

    # ---------------------------------------------------------
    # GitHub API request
    # ---------------------------------------------------------

    response = get_github(
        "/search/repositories",
        params={
            "q": query,
            "page": page,
            "per_page": per_page,
        },
    )

    # ---------------------------------------------------------
    # Parse response
    # ---------------------------------------------------------

    data = response.json()

    return [
        parse_github_repository(item)
        for item in data.get("items", [])
    ]
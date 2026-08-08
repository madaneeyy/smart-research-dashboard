from datetime import datetime

from pydantic import BaseModel, HttpUrl

from src.clients.github import get_github
from src.models.research import ResearchItem


class GitHubRepository(BaseModel):
    id: int
    name: str
    full_name: str
    description: str | None
    html_url: HttpUrl
    stars: int
    forks: int
    language: str | None
    topics: list[str]
    created_at: datetime
    updated_at: datetime

def github_repository_to_item(
    repository: GitHubRepository,
) -> ResearchItem:
    tags = list(repository.topics)

    if repository.language:
        tags.append(repository.language)

    return ResearchItem(
        id=repository.full_name,
        title=repository.name,
        description=repository.description or "",
        authors=[],
        source="github",
        url=repository.html_url,
        published=repository.created_at,
        updated=repository.updated_at,
        tags=tags,
    )

def parse_github_repository(data: dict) -> GitHubRepository:
    return GitHubRepository(
        id=data["id"],
        name=data["name"],
        full_name=data["full_name"],
        description=data.get("description"),
        html_url=data["html_url"],
        stars=data["stargazers_count"],
        forks=data["forks_count"],
        language=data.get("language"),
        topics=data.get("topics", []),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


def search_github_repositories(
    query: str,
    page: int = 1,
    per_page: int = 20,
) -> list[GitHubRepository]:
    if page < 1:
        raise ValueError("page must be 1 or greater")

    if per_page < 1:
        raise ValueError("per_page must be at least 1")

    response = get_github(
        "/search/repositories",
        params={
            "q": query,
            "page": page,
            "per_page": per_page,
        },
    )

    data = response.json()

    return [
        parse_github_repository(item)
        for item in data.get("items", [])
    ]
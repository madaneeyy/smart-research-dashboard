import httpx
import pytest

from src.clients.github import get_github

from src.collectors.github import (
    GitHubRepository,
    github_repository_to_item,
    parse_github_repository,
    search_github_repositories,
)


# =========================================================
# GitHub Client
# =========================================================


def test_get_github_returns_response(monkeypatch):

    def mock_get(*args, **kwargs):
        request = httpx.Request(
            "GET",
            "https://api.github.com/repos/test/repo",
        )

        return httpx.Response(
            status_code=200,
            json={
                "name": "test-repo",
            },
            request=request,
        )

    monkeypatch.setattr(
        httpx,
        "get",
        mock_get,
    )

    response = get_github(
        "/repos/test/repo"
    )

    assert response.status_code == 200
    assert response.json()["name"] == "test-repo"


def test_get_github_raises_for_http_error(monkeypatch):

    def mock_get(*args, **kwargs):
        request = httpx.Request(
            "GET",
            "https://api.github.com/repos/test/repo",
        )

        return httpx.Response(
            status_code=404,
            request=request,
        )

    monkeypatch.setattr(
        httpx,
        "get",
        mock_get,
    )

    with pytest.raises(httpx.HTTPStatusError):
        get_github(
            "/repos/test/repo"
        )


# =========================================================
# GitHub Repository Parsing
# =========================================================


def test_parse_github_repository():

    data = {
        "id": 123456,
        "name": "test-repo",
        "full_name": "test-user/test-repo",
        "description": "A test repository.",
        "html_url": "https://github.com/test-user/test-repo",
        "stargazers_count": 100,
        "forks_count": 20,
        "language": "Python",
        "topics": [
            "ai",
            "research",
        ],
        "created_at": "2026-01-01T12:00:00Z",
        "updated_at": "2026-08-01T12:00:00Z",
    }

    repository = parse_github_repository(data)

    assert repository.id == 123456
    assert repository.name == "test-repo"
    assert repository.full_name == (
        "test-user/test-repo"
    )

    assert repository.description == (
        "A test repository."
    )

    assert str(repository.html_url) == (
        "https://github.com/test-user/test-repo"
    )

    assert repository.stars == 100
    assert repository.forks == 20
    assert repository.language == "Python"

    assert repository.topics == [
        "ai",
        "research",
    ]

    # Verify created_at.
    assert repository.created_at.year == 2026

    # Verify exact updated_at timestamp.
    assert repository.updated_at is not None
    assert repository.updated_at.isoformat().startswith(
        "2026-08-01T12:00:00"
    )


# =========================================================
# GitHub Search
# =========================================================


def test_search_github_repositories(
    monkeypatch,
):

    response_data = {
        "items": [
            {
                "id": 123456,
                "name": "test-repo",
                "full_name": "test-user/test-repo",
                "description": "A test repository.",
                "html_url": (
                    "https://github.com/test-user/test-repo"
                ),
                "stargazers_count": 100,
                "forks_count": 20,
                "language": "Python",
                "topics": [
                    "ai",
                    "research",
                ],
                "created_at": (
                    "2026-01-01T12:00:00Z"
                ),
                "updated_at": (
                    "2026-08-01T12:00:00Z"
                ),
            }
        ]
    }

    class MockResponse:

        def json(self):
            return response_data

    def mock_get_github(
        *args,
        **kwargs,
    ):
        return MockResponse()

    monkeypatch.setattr(
        "src.collectors.github.get_github",
        mock_get_github,
    )

    repositories = search_github_repositories(
        query="machine learning",
        per_page=1,
    )

    assert len(repositories) == 1

    repository = repositories[0]

    assert repository.id == 123456
    assert repository.name == "test-repo"

    assert repository.full_name == (
        "test-user/test-repo"
    )

    assert repository.stars == 100
    assert repository.forks == 20
    assert repository.language == "Python"

    assert repository.topics == [
        "ai",
        "research",
    ]

    # Verify updated_at survives the search flow.
    assert repository.updated_at is not None
    assert repository.updated_at.isoformat().startswith(
        "2026-08-01T12:00:00"
    )


# =========================================================
# Validation
# =========================================================


def test_search_github_repositories_rejects_invalid_page():

    with pytest.raises(
        ValueError,
        match="page must be 1 or greater",
    ):
        search_github_repositories(
            query="machine learning",
            page=0,
        )


def test_search_github_repositories_rejects_invalid_per_page():

    with pytest.raises(
        ValueError,
        match="per_page must be at least 1",
    ):
        search_github_repositories(
            query="machine learning",
            per_page=0,
        )


# =========================================================
# ResearchItem Conversion
# =========================================================


def test_github_repository_to_item():

    repository = GitHubRepository(
        id=123456,
        name="test-repo",
        full_name="test-user/test-repo",
        description="A test repository.",
        html_url=(
            "https://github.com/test-user/test-repo"
        ),
        stars=100,
        forks=20,
        language="Python",
        topics=[
            "ai",
            "research",
        ],
        created_at="2026-01-01T12:00:00Z",
        updated_at="2026-08-01T12:00:00Z",
    )

    item = github_repository_to_item(
        repository
    )

    # -----------------------------------------------------
    # Common ResearchItem fields
    # -----------------------------------------------------

    assert item.id == "test-user/test-repo"

    assert item.title == "test-repo"

    assert item.description == (
        "A test repository."
    )

    assert item.authors == []

    assert item.source == "github"

    assert str(item.url) == (
        "https://github.com/test-user/test-repo"
    )

    # -----------------------------------------------------
    # Dates
    # -----------------------------------------------------

    assert item.published is not None
    assert item.published.year == 2026

    assert item.published.isoformat().startswith(
        "2026-01-01T12:00:00"
    )

    # GitHub updated_at → ResearchItem.updated
    assert item.updated is not None
    assert item.updated.isoformat().startswith(
        "2026-08-01T12:00:00"
    )

    # -----------------------------------------------------
    # Tags
    # -----------------------------------------------------

    assert item.tags == [
        "ai",
        "research",
        "Python",
    ]

    # -----------------------------------------------------
    # GitHub-specific fields
    # -----------------------------------------------------

    assert item.stars == 100

    assert item.forks == 20

    assert item.language == "Python"
import httpx
import pytest

from src.clients.github import get_github


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

    monkeypatch.setattr(httpx, "get", mock_get)

    response = get_github("/repos/test/repo")

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

    monkeypatch.setattr(httpx, "get", mock_get)

    with pytest.raises(httpx.HTTPStatusError):
        get_github("/repos/test/repo")
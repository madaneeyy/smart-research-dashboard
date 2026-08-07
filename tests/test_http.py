import httpx
import pytest

from src.clients.http import get


def test_get_returns_response_text(monkeypatch):
    def mock_get(*args, **kwargs):
        request = httpx.Request(
            "GET",
            "https://example.com",
        )

        return httpx.Response(
            status_code=200,
            text="hello",
            request=request,
        )

    monkeypatch.setattr(httpx, "get", mock_get)

    result = get("https://example.com")

    assert result == "hello"

def test_get_raises_for_http_error(monkeypatch):
    def mock_get(*args, **kwargs):
        request = httpx.Request(
            "GET",
            "https://example.com",
        )

        return httpx.Response(
            status_code=500,
            request=request,
        )

    monkeypatch.setattr(httpx, "get", mock_get)

    with pytest.raises(httpx.HTTPStatusError):
        get("https://example.com")

import httpx
import pytest

from src.clients.paperswithcode import get_paperswithcode_rows


def test_get_paperswithcode_rows(monkeypatch):
    request = httpx.Request(
        "GET",
        "https://datasets-server.huggingface.co/rows",
    )

    response = httpx.Response(
        status_code=200,
        json={
            "rows": [
                {
                    "row_idx": 0,
                    "row": {
                        "title": "Test Paper",
                    },
                }
            ]
        },
        request=request,
    )

    def mock_get(*args, **kwargs):
        return response

    monkeypatch.setattr(httpx, "get", mock_get)

    result = get_paperswithcode_rows(
        "pwc-archive/papers-with-abstracts",
        offset=0,
        length=1,
    )

    assert result.status_code == 200
    assert result.json()["rows"][0]["row"]["title"] == "Test Paper"


def test_get_paperswithcode_rows_rejects_negative_offset():
    with pytest.raises(ValueError):
        get_paperswithcode_rows(
            "pwc-archive/papers-with-abstracts",
            offset=-1,
        )


def test_get_paperswithcode_rows_rejects_invalid_length():
    with pytest.raises(ValueError):
        get_paperswithcode_rows(
            "pwc-archive/papers-with-abstracts",
            length=0,
        )


def test_get_paperswithcode_rows_raises_for_http_error(monkeypatch):
    request = httpx.Request(
        "GET",
        "https://datasets-server.huggingface.co/rows",
    )

    response = httpx.Response(
        status_code=404,
        request=request,
    )

    def mock_get(*args, **kwargs):
        return response

    monkeypatch.setattr(httpx, "get", mock_get)

    with pytest.raises(httpx.HTTPStatusError):
        get_paperswithcode_rows(
            "pwc-archive/papers-with-abstracts",
        )
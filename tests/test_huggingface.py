
import httpx
import pytest

from src.clients.huggingface import get_huggingface
from src.collectors.huggingface import (
    HuggingFaceModel,
    parse_huggingface_model,
    search_huggingface_models,
    huggingface_model_to_item,
)
def test_get_huggingface_returns_response(monkeypatch):
    def mock_get(*args, **kwargs):
        request = httpx.Request(
            "GET",
            "https://huggingface.co/api/models",
        )

        return httpx.Response(
            status_code=200,
            json=[
                {
                    "modelId": "test/model",
                }
            ],
            request=request,
        )

    monkeypatch.setattr(httpx, "get", mock_get)

    response = get_huggingface("/models")

    assert response.status_code == 200
    assert response.json()[0]["modelId"] == "test/model"


def test_get_huggingface_raises_for_http_error(monkeypatch):
    def mock_get(*args, **kwargs):
        request = httpx.Request(
            "GET",
            "https://huggingface.co/api/models",
        )

        return httpx.Response(
            status_code=404,
            request=request,
        )

    monkeypatch.setattr(httpx, "get", mock_get)

    with pytest.raises(httpx.HTTPStatusError):
        get_huggingface("/models")

def test_parse_huggingface_model():
    data = {
        "modelId": "test-user/test-model",
        "author": "test-user",
        "pipeline_tag": "text-classification",
        "downloads": 1500,
        "likes": 42,
        "library_name": "transformers",
        "tags": ["pytorch", "text-classification"],
        "createdAt": "2026-01-01T12:00:00Z",
        "lastModified": "2026-08-01T12:00:00Z",
    }

    model = parse_huggingface_model(data)

    assert model.model_id == "test-user/test-model"
    assert model.author == "test-user"
    assert model.pipeline_tag == "text-classification"
    assert model.downloads == 1500
    assert model.likes == 42
    assert model.library_name == "transformers"
    assert model.tags == ["pytorch", "text-classification"]

    assert model.created_at.year == 2026
    assert model.last_modified.year == 2026

    assert str(model.url) == (
        "https://huggingface.co/test-user/test-model"
    )
def test_search_huggingface_models_passes_filters(monkeypatch):
    response_data = []

    captured = {}

    class MockResponse:
        def json(self):
            return response_data

    def mock_get_huggingface(path, params=None):
        captured["path"] = path
        captured["params"] = params
        return MockResponse()

    monkeypatch.setattr(
        "src.collectors.huggingface.get_huggingface",
        mock_get_huggingface,
    )

    search_huggingface_models(
        search="BERT",
        limit=10,
        author="google",
        library="transformers",
        pipeline_tag="text-classification",
    )

    assert captured["path"] == "/models"
    assert captured["params"] == {
        "limit": 10,
        "search": "BERT",
        "author": "google",
        "library": "transformers",
        "pipeline_tag": "text-classification",
    }

def test_huggingface_model_to_item():
    model = HuggingFaceModel(
        model_id="test-user/test-model",
        author="test-user",
        pipeline_tag="text-classification",
        downloads=1000,
        likes=50,
        library_name="transformers",
        tags=["nlp", "text-classification"],
        created_at="2026-01-01T12:00:00Z",
        last_modified="2026-08-01T12:00:00Z",
        url="https://huggingface.co/test-user/test-model",
    )

    item = huggingface_model_to_item(model)

    assert item.id == "test-user/test-model"
    assert item.title == "test-user/test-model"
    assert item.description is None
    assert item.authors == ["test-user"]
    assert item.source == "huggingface"
    assert str(item.url) == (
        "https://huggingface.co/test-user/test-model"
    )
    assert item.published.year == 2026
    assert item.updated.year == 2026
    assert item.tags == ["nlp", "text-classification"]
import httpx
import pytest

from src.clients.huggingface import get_huggingface

from src.collectors.huggingface import (
    HuggingFaceModel,
    huggingface_model_to_item,
    parse_huggingface_model,
    search_huggingface_models,
)


# =============================================================
# CLIENT TESTS
# =============================================================

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

    monkeypatch.setattr(
        httpx,
        "get",
        mock_get,
    )

    response = get_huggingface("/models")

    assert response.status_code == 200

    assert response.json()[0]["modelId"] == (
        "test/model"
    )


def test_get_huggingface_raises_for_http_error(
    monkeypatch,
):

    def mock_get(*args, **kwargs):
        request = httpx.Request(
            "GET",
            "https://huggingface.co/api/models",
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
        get_huggingface("/models")


# =============================================================
# PARSING TEST
# =============================================================

def test_parse_huggingface_model():

    data = {
        "modelId": "test-user/test-model",
        "author": "test-user",
        "pipeline_tag": "text-classification",
        "downloads": 1500,
        "likes": 42,
        "library_name": "transformers",
        "tags": [
            "pytorch",
            "text-classification",
        ],
        "createdAt": "2026-01-01T12:00:00Z",
        "lastModified": "2026-08-01T12:00:00Z",
    }

    model = parse_huggingface_model(data)

    # ---------------------------------------------------------
    # Basic fields
    # ---------------------------------------------------------

    assert model.model_id == (
        "test-user/test-model"
    )

    assert model.author == "test-user"

    assert model.pipeline_tag == (
        "text-classification"
    )

    # ---------------------------------------------------------
    # Hugging Face metadata
    # ---------------------------------------------------------

    assert model.downloads == 1500

    assert model.likes == 42

    assert model.library_name == (
        "transformers"
    )

    # ---------------------------------------------------------
    # Tags
    # ---------------------------------------------------------

    assert model.tags == [
        "pytorch",
        "text-classification",
    ]

    # ---------------------------------------------------------
    # createdAt
    # ---------------------------------------------------------

    assert model.created_at is not None

    assert model.created_at.year == 2026
    assert model.created_at.month == 1
    assert model.created_at.day == 1

    # ---------------------------------------------------------
    # lastModified
    # ---------------------------------------------------------

    assert model.last_modified is not None

    assert model.last_modified.year == 2026
    assert model.last_modified.month == 8
    assert model.last_modified.day == 1

    # ---------------------------------------------------------
    # URL
    # ---------------------------------------------------------

    assert str(model.url) == (
        "https://huggingface.co/"
        "test-user/test-model"
    )


# =============================================================
# AUTHOR FALLBACK TEST
# =============================================================

def test_parse_huggingface_model_derives_author():

    data = {
        "modelId": "google/bert-base",
    }

    model = parse_huggingface_model(data)

    assert model.model_id == (
        "google/bert-base"
    )

    assert model.author == "google"


def test_parse_huggingface_model_allows_missing_author():

    data = {
        "modelId": "bert-base",
    }

    model = parse_huggingface_model(data)

    assert model.author is None


# =============================================================
# MISSING OPTIONAL FIELDS TEST
# =============================================================

def test_parse_huggingface_model_handles_missing_fields():

    data = {
        "modelId": "test-user/minimal-model",
    }

    model = parse_huggingface_model(data)

    assert model.model_id == (
        "test-user/minimal-model"
    )

    assert model.downloads == 0

    assert model.likes == 0

    assert model.pipeline_tag is None

    assert model.library_name is None

    assert model.tags == []

    assert model.created_at is None

    assert model.last_modified is None

    assert model.author == "test-user"

    assert str(model.url) == (
        "https://huggingface.co/"
        "test-user/minimal-model"
    )


# =============================================================
# SEARCH TEST
# =============================================================

def test_search_huggingface_models_passes_filters(
    monkeypatch,
):

    response_data = []

    captured = {}

    class MockResponse:

        def json(self):
            return response_data

    def mock_get_huggingface(
        path,
        params=None,
    ):
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


def test_search_huggingface_models_rejects_invalid_limit():

    with pytest.raises(
        ValueError,
        match="limit must be at least 1",
    ):
        search_huggingface_models(
            search="BERT",
            limit=0,
        )


# =============================================================
# SEARCH RESULT PARSING TEST
# =============================================================

def test_search_huggingface_models_returns_models(
    monkeypatch,
):

    response_data = [
        {
            "modelId": "google/bert-base",
            "author": "google",
            "pipeline_tag": "fill-mask",
            "downloads": 10000,
            "likes": 500,
            "library_name": "transformers",
            "tags": ["nlp"],
            "createdAt": "2026-01-01T12:00:00Z",
            "lastModified": "2026-08-01T12:00:00Z",
        },
        {
            "modelId": "openai/test-model",
            "author": "openai",
            "pipeline_tag": "text-generation",
            "downloads": 5000,
            "likes": 200,
            "library_name": "transformers",
            "tags": ["language-model"],
            "createdAt": "2026-02-01T12:00:00Z",
            "lastModified": "2026-08-02T12:00:00Z",
        },
    ]

    class MockResponse:

        def json(self):
            return response_data

    def mock_get_huggingface(
        path,
        params=None,
    ):
        return MockResponse()

    monkeypatch.setattr(
        "src.collectors.huggingface.get_huggingface",
        mock_get_huggingface,
    )

    models = search_huggingface_models(
        search="machine learning",
        limit=2,
    )

    assert len(models) == 2

    assert models[0].model_id == (
        "google/bert-base"
    )

    assert models[0].downloads == 10000

    assert models[0].likes == 500

    assert models[1].model_id == (
        "openai/test-model"
    )


# =============================================================
# NORMALIZATION TEST
# =============================================================

def test_huggingface_model_to_item():

    model = HuggingFaceModel(
        model_id="test-user/test-model",
        author="test-user",
        pipeline_tag="text-classification",
        downloads=1000,
        likes=50,
        library_name="transformers",
        tags=[
            "nlp",
            "text-classification",
        ],
        created_at="2026-01-01T12:00:00Z",
        last_modified="2026-08-01T12:00:00Z",
        url=(
            "https://huggingface.co/"
            "test-user/test-model"
        ),
    )

    item = huggingface_model_to_item(model)

    # ---------------------------------------------------------
    # Basic fields
    # ---------------------------------------------------------

    assert item.id == (
        "test-user/test-model"
    )

    assert item.title == (
        "test-user/test-model"
    )

    assert item.description is None

    assert item.authors == [
        "test-user"
    ]

    assert item.source == "huggingface"

    assert str(item.url) == (
        "https://huggingface.co/"
        "test-user/test-model"
    )

    # ---------------------------------------------------------
    # Published date
    #
    # createdAt
    #      ↓
    # published
    # ---------------------------------------------------------

    assert item.published is not None

    assert item.published.year == 2026
    assert item.published.month == 1
    assert item.published.day == 1

    # ---------------------------------------------------------
    # Updated date
    #
    # lastModified
    #      ↓
    # updated
    # ---------------------------------------------------------

    assert item.updated is not None

    assert item.updated.year == 2026
    assert item.updated.month == 8
    assert item.updated.day == 1

    # ---------------------------------------------------------
    # Tags
    # ---------------------------------------------------------

    assert item.tags == [
        "nlp",
        "text-classification",
    ]

    # ---------------------------------------------------------
    # Hugging Face-specific fields
    # ---------------------------------------------------------

    assert item.downloads == 1000

    assert item.likes == 50

    assert item.library == (
        "transformers"
    )

    assert item.pipeline_tag == (
        "text-classification"
    )


# =============================================================
# DATE MAPPING TEST
# =============================================================

def test_huggingface_dates_map_correctly_to_research_item():
    """
    Explicitly verify:

        Hugging Face createdAt
            ↓
        ResearchItem.published

        Hugging Face lastModified
            ↓
        ResearchItem.updated
    """

    model = HuggingFaceModel(
        model_id="test-user/date-test-model",
        author="test-user",
        pipeline_tag="text-classification",
        downloads=100,
        likes=10,
        library_name="transformers",
        tags=["nlp"],
        created_at="2025-03-15T10:30:00Z",
        last_modified="2026-08-10T14:45:00Z",
        url=(
            "https://huggingface.co/"
            "test-user/date-test-model"
        ),
    )

    item = huggingface_model_to_item(model)

    # ---------------------------------------------------------
    # createdAt → published
    # ---------------------------------------------------------

    assert item.published is not None

    assert item.published.year == 2025
    assert item.published.month == 3
    assert item.published.day == 15

    # ---------------------------------------------------------
    # lastModified → updated
    # ---------------------------------------------------------

    assert item.updated is not None

    assert item.updated.year == 2026
    assert item.updated.month == 8
    assert item.updated.day == 10

    # ---------------------------------------------------------
    # Updated should be newer than published
    # ---------------------------------------------------------

    assert item.updated > item.published


# =============================================================
# SOURCE-SPECIFIC METADATA TEST
# =============================================================

def test_huggingface_metadata_is_preserved():

    model = HuggingFaceModel(
        model_id="test-user/metadata-model",
        author="test-user",
        pipeline_tag="image-classification",
        downloads=12400,
        likes=234,
        library_name="transformers",
        tags=[
            "pytorch",
            "vision",
        ],
        created_at="2026-01-01T12:00:00Z",
        last_modified="2026-08-10T12:00:00Z",
        url=(
            "https://huggingface.co/"
            "test-user/metadata-model"
        ),
    )

    item = huggingface_model_to_item(model)

    # These are the values that the Streamlit
    # result card will eventually display.

    assert item.downloads == 12400

    assert item.likes == 234

    assert item.library == "transformers"

    assert item.pipeline_tag == (
        "image-classification"
    )


# =============================================================
# MISSING DATES TEST
# =============================================================

def test_huggingface_missing_dates_are_preserved():

    model = HuggingFaceModel(
        model_id="test-user/no-date-model",
        author="test-user",
        pipeline_tag=None,
        downloads=0,
        likes=0,
        library_name=None,
        tags=[],
        created_at=None,
        last_modified=None,
        url=(
            "https://huggingface.co/"
            "test-user/no-date-model"
        ),
    )

    item = huggingface_model_to_item(model)

    assert item.published is None

    assert item.updated is None

    assert item.downloads == 0

    assert item.likes == 0

    assert item.library is None

    assert item.pipeline_tag is None
from datetime import datetime

from pydantic import BaseModel, HttpUrl

from src.clients.huggingface import get_huggingface
from src.models.research import ResearchItem


class HuggingFaceModel(BaseModel):
    model_id: str
    author: str | None
    pipeline_tag: str | None
    downloads: int
    likes: int
    library_name: str | None
    tags: list[str]
    created_at: datetime | None
    last_modified: datetime | None
    url: HttpUrl


def parse_huggingface_model(data: dict) -> HuggingFaceModel:
    model_id = data["modelId"]

    author = data.get("author")

    if author is None and "/" in model_id:
        author = model_id.split("/", 1)[0]

    return HuggingFaceModel(
        model_id=model_id,
        author=author,
        pipeline_tag=data.get("pipeline_tag"),
        downloads=data.get("downloads", 0),
        likes=data.get("likes", 0),
        library_name=data.get("library_name"),
        tags=data.get("tags", []),
        created_at=data.get("createdAt"),
        last_modified=data.get("lastModified"),
        url=f"https://huggingface.co/{model_id}",
    )


def search_huggingface_models(
    search: str | None = None,
    limit: int = 20,
    author: str | None = None,
    library: str | None = None,
    pipeline_tag: str | None = None,
) -> list[HuggingFaceModel]:
    if limit < 1:
        raise ValueError("limit must be at least 1")

    params = {
        "limit": limit,
    }

    if search:
        params["search"] = search

    if author:
        params["author"] = author

    if library:
        params["library"] = library

    if pipeline_tag:
        params["pipeline_tag"] = pipeline_tag

    response = get_huggingface(
        "/models",
        params=params,
    )

    data = response.json()

    return [
        parse_huggingface_model(item)
        for item in data
    ]

def huggingface_model_to_item(
    model: HuggingFaceModel,
) -> ResearchItem:
    return ResearchItem(
        id=model.model_id,
        title=model.model_id,
        description=None,
        authors=[model.author] if model.author else [],
        source="huggingface",
        url=model.url,
        published=model.created_at,
        updated=model.last_modified,
        tags=model.tags,
    )
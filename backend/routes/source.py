from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.source_service import (
    create_source,
    delete_source,
    get_source,
    list_sources,
)


class CreateSourceRequest(BaseModel):
    source_type: str
    title: str
    url: str | None = None
    metadata: dict[str, Any] | None = None


class SourceResponse(BaseModel):
    id: UUID
    workspace_id: UUID
    source_type: str
    title: str
    url: str | None
    metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class DeleteSourceResponse(BaseModel):
    message: str


workspace_source_router = APIRouter(
    prefix="/workspaces/{workspace_id}/sources",
    tags=["Sources"],
)


source_router = APIRouter(
    prefix="/sources",
    tags=["Sources"],
)


@workspace_source_router.post(
    "",
    response_model=SourceResponse,
)
def create_source_endpoint(
    workspace_id: str,
    source: CreateSourceRequest,
):
    return create_source(
        workspace_id=workspace_id,
        source_type=source.source_type,
        title=source.title,
        url=source.url,
        metadata=source.metadata,
    )


@workspace_source_router.get(
    "",
    response_model=list[SourceResponse],
)
def get_sources(workspace_id: str):
    return list_sources(workspace_id)


@source_router.get(
    "/{source_id}",
    response_model=SourceResponse,
)
def get_source_endpoint(source_id: str):
    source = get_source(source_id)

    if source is None:
        raise HTTPException(
            status_code=404,
            detail="Source not found.",
        )

    return source


@source_router.delete(
    "/{source_id}",
    response_model=DeleteSourceResponse,
)
def delete_source_endpoint(source_id: str):
    deleted = delete_source(source_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Source not found.",
        )

    return {
        "message": "Source deleted successfully."
    }
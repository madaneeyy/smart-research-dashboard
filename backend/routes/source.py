from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.source_service import (
    create_source,
    delete_source,
    delete_source_from_workspace,
    get_source,
    list_sources,
)
from backend.services.arxiv_ingestion_service import ingest_arxiv_source
from backend.services.workspace_document_service import delete_workspace_document


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
    source_type = source.source_type.strip().lower()

    try:
        if source_type in {"arxiv", "arxiv_paper"}:
            return ingest_arxiv_source(
                workspace_id=workspace_id,
                title=source.title,
                url=source.url,
                metadata=source.metadata,
            )

        return create_source(
            workspace_id=workspace_id,
            source_type=source.source_type,
            title=source.title,
            url=source.url,
            metadata=source.metadata,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Could not ingest/add source to workspace.",
                "error": str(exc),
            },
        ) from exc


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

@workspace_source_router.delete(
    "/{source_id}",
    response_model=DeleteSourceResponse,
)
def delete_workspace_source_endpoint(
    workspace_id: str,
    source_id: str,
):
    source = get_source(source_id)

    if source is None or str(source.get("workspace_id")) != str(workspace_id):
        raise HTTPException(
            status_code=404,
            detail="Source not found in this workspace.",
        )

    source_type = str(source.get("source_type") or "").strip().lower()
    metadata = source.get("metadata") or {}
    document_id = (
        str(metadata.get("document_id") or "").strip()
        if isinstance(metadata, dict)
        else ""
    )

    deleted = delete_source_from_workspace(
        workspace_id=workspace_id,
        source_id=source_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Source not found in this workspace.",
        )

    # ArXiv sources are backed by a workspace_documents association. Remove
    # that association too; delete_workspace_document preserves the underlying
    # document/chunks if another workspace still uses them.
    if source_type in {"arxiv", "arxiv_paper"} and document_id:
        try:
            delete_workspace_document(
                document_id=document_id,
                workspace_id=workspace_id,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        "The arXiv source was removed, but its document "
                        "association could not be cleaned up."
                    ),
                    "error": str(exc),
                    "document_id": document_id,
                },
            ) from exc

    return {
        "message": "Source removed from workspace."
    }


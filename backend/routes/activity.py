from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.activity_service import (
    create_activity,
    list_recent_activity,
)


router = APIRouter(
    prefix="/workspaces/{workspace_id}/activity",
    tags=["Activity"],
)


ActivityType = Literal[
    "document_added",
    "paper_added",
    "model_added",
    "repository_added",
    "chat_started",
    "research_performed",
]


class CreateActivityRequest(BaseModel):
    activity_type: ActivityType
    title: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    reference_id: str | None = Field(default=None, max_length=200)
    reference_type: str | None = Field(default=None, max_length=80)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActivityResponse(BaseModel):
    id: UUID
    workspace_id: UUID
    activity_type: ActivityType
    title: str
    description: str | None
    reference_id: str | None
    reference_type: str | None
    metadata: dict[str, Any] | None
    created_at: datetime


@router.post("", response_model=ActivityResponse)
def create_workspace_activity(
    workspace_id: str,
    request: CreateActivityRequest,
) -> dict[str, Any]:
    try:
        return create_activity(
            workspace_id=workspace_id,
            activity_type=request.activity_type,
            title=request.title,
            description=request.description,
            reference_id=request.reference_id,
            reference_type=request.reference_type,
            metadata=request.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Unable to record workspace activity.",
                "error": str(exc),
            },
        ) from exc


@router.get("", response_model=list[ActivityResponse])
def get_workspace_activity(
    workspace_id: str,
    limit: int = Query(default=8, ge=1, le=50),
) -> list[dict[str, Any]]:
    try:
        return list_recent_activity(workspace_id, limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Unable to load workspace activity.",
                "error": str(exc),
            },
        ) from exc

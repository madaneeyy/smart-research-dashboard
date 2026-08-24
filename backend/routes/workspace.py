from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.workspace_service import (
    create_workspace,
    delete_workspace,
    get_workspace,
    list_workspaces,
    update_workspace,
)


class CreateWorkspaceRequest(BaseModel):
    name: str
    description: str | None = None


class UpdateWorkspaceRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class WorkspaceResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    owner_id: UUID | None
    created_at: datetime
    updated_at: datetime


class DeleteWorkspaceResponse(BaseModel):
    message: str


router = APIRouter(
    prefix="/workspaces",
    tags=["Workspaces"],
)


@router.post("", response_model=WorkspaceResponse)
def create_workspace_endpoint(
    workspace: CreateWorkspaceRequest,
):
    return create_workspace(
        name=workspace.name,
        description=workspace.description,
    )


@router.get("", response_model=list[WorkspaceResponse])
def get_workspaces():
    return list_workspaces()


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
)
def get_workspace_endpoint(workspace_id: str):
    workspace = get_workspace(workspace_id)

    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail="Workspace not found.",
        )

    return workspace


@router.patch(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
)
def update_workspace_endpoint(
    workspace_id: str,
    workspace: UpdateWorkspaceRequest,
):
    updated_workspace = update_workspace(
        workspace_id=workspace_id,
        name=workspace.name,
        description=workspace.description,
    )

    if updated_workspace is None:
        raise HTTPException(
            status_code=404,
            detail="Workspace not found.",
        )

    return updated_workspace


@router.delete(
    "/{workspace_id}",
    response_model=DeleteWorkspaceResponse,
)
def delete_workspace_endpoint(workspace_id: str):
    deleted = delete_workspace(workspace_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Workspace not found.",
        )

    return {
        "message": "Workspace deleted successfully."
    }
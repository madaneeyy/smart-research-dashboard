from typing import Any

from backend.db.database import supabase


def create_workspace(
    name: str,
    description: str | None = None,
) -> dict[str, Any]:
    response = (
        supabase
        .table("workspaces")
        .insert({
            "name": name,
            "description": description,
        })
        .execute()
    )

    if not response.data:
        raise RuntimeError("Failed to create workspace.")

    return response.data[0]

def list_workspaces() -> list[dict[str, Any]]:
    response = (
        supabase
        .table("workspaces")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return response.data

def get_workspace(workspace_id: str) -> dict[str, Any] | None:
    response = (
        supabase
        .table("workspaces")
        .select("*")
        .eq("id", workspace_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]

def update_workspace(
    workspace_id: str,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any] | None:

    updates: dict[str, Any] = {}

    if name is not None:
        updates["name"] = name

    if description is not None:
        updates["description"] = description

    if not updates:
        return get_workspace(workspace_id)

    response = (
        supabase
        .table("workspaces")
        .update(updates)
        .eq("id", workspace_id)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]

def delete_workspace(workspace_id: str) -> bool:
    existing_workspace = get_workspace(workspace_id)

    if existing_workspace is None:
        return False

    (
        supabase
        .table("workspaces")
        .delete()
        .eq("id", workspace_id)
        .execute()
    )

    return True
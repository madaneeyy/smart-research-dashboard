from typing import Any

from backend.db.database import supabase


def create_source(
    workspace_id: str,
    source_type: str,
    title: str,
    url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:

    response = (
        supabase
        .table("workspace_sources")
        .insert({
            "workspace_id": workspace_id,
            "source_type": source_type,
            "title": title,
            "url": url,
            "metadata": metadata or {},
        })
        .execute()
    )

    if not response.data:
        raise RuntimeError("Failed to create source.")
    

    return response.data[0]

def list_sources(
    workspace_id: str,
) -> list[dict[str, Any]]:
    response = (
        supabase
        .table("workspace_sources")
        .select("*")
        .eq("workspace_id", workspace_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data

def get_source(source_id: str) -> dict[str, Any] | None:
    response = (
        supabase
        .table("workspace_sources")
        .select("*")
        .eq("id", source_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]

def delete_source(source_id: str) -> bool:
    existing_source = get_source(source_id)

    if existing_source is None:
        return False

    (
        supabase
        .table("workspace_sources")
        .delete()
        .eq("id", source_id)
        .execute()
    )

    return True



def delete_source_from_workspace(
    workspace_id: str,
    source_id: str,
) -> bool:
    response = (
        supabase
        .table("workspace_sources")
        .delete()
        .eq("workspace_id", workspace_id)
        .eq("id", source_id)
        .execute()
    )

    return bool(response.data)
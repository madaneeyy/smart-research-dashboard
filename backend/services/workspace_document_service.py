from typing import Any

from backend.db.database import supabase


def create_workspace_document(
    workspace_id: str,
    document_id: str,
    filename: str,
    content_type: str | None = None,
    pages: int | None = None,
    characters: int | None = None,
    size_bytes: int | None = None,
    status: str = "ready",
) -> dict[str, Any]:
    response = (
        supabase
        .table("workspace_documents")
        .insert({
            "workspace_id": workspace_id,
            "document_id": document_id,
            "filename": filename,
            "content_type": content_type,
            "pages": pages,
            "characters": characters,
            "size_bytes": size_bytes,
            "status": status,
        })
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Failed to create workspace document."
        )

    print("CREATE WORKSPACE DOCUMENT RESPONSE:")
    print(response.data[0])

    return response.data[0]


def list_workspace_documents(
    workspace_id: str,
) -> list[dict[str, Any]]:
    response = (
        supabase
        .table("workspace_documents")
        .select("*")
        .eq("workspace_id", workspace_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def get_workspace_document(
    document_id: str,
    workspace_id: str,
) -> dict[str, Any] | None:
    response = (
        supabase
        .table("workspace_documents")
        .select("*")
        .eq("workspace_id", workspace_id)
        .eq("document_id", document_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]


def delete_workspace_document(
    document_id: str,
    workspace_id: str,
) -> bool:
    existing_document = get_workspace_document(
        document_id=document_id,
        workspace_id=workspace_id,
    )

    if existing_document is None:
        return False

    (
        supabase
        .table("workspace_documents")
        .delete()
        .eq("workspace_id", workspace_id)
        .eq("document_id", document_id)
        .execute()
    )

    return True
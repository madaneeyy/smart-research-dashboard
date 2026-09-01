from typing import Any

from backend.db.database import supabase

from backend.services.chat_service import (
    delete_chat,
    get_workspace_chats,
)

from backend.services.workspace_document_service import (
    delete_workspace_document,
    list_workspace_documents,
)


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
        raise RuntimeError(
            "Failed to create workspace."
        )

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


def get_workspace(
    workspace_id: str,
) -> dict[str, Any] | None:
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


def delete_workspace(
    workspace_id: str,
) -> bool:
    """
    Delete a workspace and its workspace-owned data.

    Deletion order:

        chats
          ├── chat_messages
          └── chat_sources

        workspace_sources

        workspace_documents
          └── underlying document data
              only when no other workspace uses it

        workspace
    """

    existing_workspace = get_workspace(
        workspace_id
    )

    if existing_workspace is None:
        return False

    # ----------------------------------------------------------
    # 1. Delete all chats belonging to the workspace.
    #
    # delete_chat() already removes:
    #   - chat_messages
    #   - chat_sources
    #   - chat
    # ----------------------------------------------------------

    chats = get_workspace_chats(
        workspace_id
    )

    for chat in chats:
        chat_id = chat.get("id")

        if not chat_id:
            continue

        deleted = delete_chat(
            str(chat_id)
        )

        if not deleted:
            raise RuntimeError(
                f"Failed to delete chat {chat_id}."
            )

    # ----------------------------------------------------------
    # 2. Delete workspace source associations.
    # ----------------------------------------------------------

    (
        supabase
        .table("workspace_sources")
        .delete()
        .eq("workspace_id", workspace_id)
        .execute()
    )

    # ----------------------------------------------------------
    # 3. Delete workspace documents.
    #
    # delete_workspace_document() handles:
    #
    #   - workspace_documents association
    #   - checking whether another workspace still uses
    #     the underlying document
    #   - document_chunks
    #   - Storage object
    #   - documents record
    # ----------------------------------------------------------

    workspace_documents = (
        list_workspace_documents(
            workspace_id
        )
    )

    for workspace_document in workspace_documents:
        document_id = (
            workspace_document.get(
                "document_id"
            )
        )

        if not document_id:
            continue

        deleted = delete_workspace_document(
            document_id=str(document_id),
            workspace_id=workspace_id,
        )

        if not deleted:
            raise RuntimeError(
                f"Failed to delete workspace document "
                f"{document_id}."
            )

    # ----------------------------------------------------------
    # 4. Finally delete the workspace itself.
    # ----------------------------------------------------------

    response = (
        supabase
        .table("workspaces")
        .delete()
        .eq("id", workspace_id)
        .execute()
    )

    if not response.data:
        return False

    return True
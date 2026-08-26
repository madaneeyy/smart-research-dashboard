from typing import Any

from backend.db.database import supabase


def create_chat(
    workspace_id: str,
    title: str = "New Chat",
) -> dict[str, Any]:
    response = (
        supabase
        .table("chats")
        .insert(
            {
                "workspace_id": workspace_id,
                "title": title,
            }
        )
        .execute()
    )

    if not response.data:
        raise RuntimeError("Failed to create chat.")

    return response.data[0]


def add_chat_source(
    chat_id: str,
    source_type: str,
    source_id: str,
) -> dict[str, Any]:
    response = (
        supabase
        .table("chat_sources")
        .insert(
            {
                "chat_id": chat_id,
                "source_type": source_type,
                "source_id": source_id,
            }
        )
        .execute()
    )

    if not response.data:
        raise RuntimeError("Failed to add chat source.")

    return response.data[0]

def get_chat(
    chat_id: str,
) -> dict[str, Any] | None:
    response = (
        supabase
        .table("chats")
        .select("*")
        .eq("id", chat_id)
        .maybe_single()
        .execute()
    )

    return response.data


def get_chat_messages(
    chat_id: str,
) -> list[dict[str, Any]]:
    response = (
        supabase
        .table("chat_messages")
        .select("*")
        .eq("chat_id", chat_id)
        .order("created_at", desc=False)
        .execute()
    )

    return response.data or []


def get_chat_sources(
    chat_id: str,
) -> list[dict[str, Any]]:
    response = (
        supabase
        .table("chat_sources")
        .select("*")
        .eq("chat_id", chat_id)
        .order("created_at", desc=False)
        .execute()
    )

    return response.data or []

def add_chat_message(
    chat_id: str,
    role: str,
    content: str,
) -> dict[str, Any]:
    response = (
        supabase
        .table("chat_messages")
        .insert(
            {
                "chat_id": chat_id,
                "role": role,
                "content": content,
            }
        )
        .execute()
    )

    if not response.data:
        raise RuntimeError("Failed to save chat message.")

    return response.data[0]

def get_workspace_chats(
    workspace_id: str,
) -> list[dict[str, Any]]:
    response = (
        supabase
        .table("chats")
        .select("*")
        .eq("workspace_id", workspace_id)
        .order("updated_at", desc=True)
        .execute()
    )

    return response.data or []

def delete_chat(chat_id: str) -> bool:
    existing_chat = get_chat(chat_id)

    if existing_chat is None:
        return False

    # Delete dependent records first.
    (
        supabase
        .table("chat_messages")
        .delete()
        .eq("chat_id", chat_id)
        .execute()
    )

    (
        supabase
        .table("chat_sources")
        .delete()
        .eq("chat_id", chat_id)
        .execute()
    )

    # Delete the chat itself.
    (
        supabase
        .table("chats")
        .delete()
        .eq("id", chat_id)
        .execute()
    )

    return True
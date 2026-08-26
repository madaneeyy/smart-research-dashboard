from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.chat_service import (
    create_chat,
    add_chat_source,
    get_chat,
    get_chat_messages,
    get_chat_sources,
    add_chat_message,
    get_workspace_chats,
    delete_chat,
)


router = APIRouter(
    prefix="/chats",
    tags=["chats"],
)


class CreateChatRequest(BaseModel):
    workspace_id: str
    title: Optional[str] = "New Chat"
    source_type: Optional[str] = None
    source_id: Optional[str] = None


@router.post("")
def create_chat_endpoint(
    request: CreateChatRequest,
) -> Dict[str, Any]:

    chat = create_chat(
        workspace_id=request.workspace_id,
        title=request.title or "New Chat",
    )

    source = None

    if request.source_type and request.source_id:
        source = add_chat_source(
            chat_id=str(chat["id"]),
            source_type=request.source_type,
            source_id=request.source_id,
        )

    return {
        "chat": chat,
        "source": source,
    }

@router.get("/{chat_id}")
def get_chat_endpoint(
    chat_id: str,
) -> Dict[str, Any]:
    chat = get_chat(chat_id)

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat not found.",
        )

    return chat


@router.get("/{chat_id}/messages")
def get_chat_messages_endpoint(
    chat_id: str,
) -> list[dict[str, Any]]:
    chat = get_chat(chat_id)

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat not found.",
        )

    return get_chat_messages(chat_id)

class AddChatMessageRequest(BaseModel):
    role: str
    content: str


@router.post("/{chat_id}/messages")
def add_chat_message_endpoint(
    chat_id: str,
    request: AddChatMessageRequest,
) -> Dict[str, Any]:

    chat = get_chat(chat_id)

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat not found.",
        )

    if request.role not in {"user", "assistant"}:
        raise HTTPException(
            status_code=400,
            detail="Role must be 'user' or 'assistant'.",
        )

    if not request.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Message content must not be empty.",
        )

    return add_chat_message(
        chat_id=chat_id,
        role=request.role,
        content=request.content,
    )


@router.get("/{chat_id}/sources")
def get_chat_sources_endpoint(
    chat_id: str,
) -> list[dict[str, Any]]:
    chat = get_chat(chat_id)

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Chat not found.",
        )

    return get_chat_sources(chat_id)

@router.get("/workspace/{workspace_id}")
def get_workspace_chats_endpoint(
    workspace_id: str,
) -> list[dict[str, Any]]:
    return get_workspace_chats(workspace_id)

@router.delete("/{chat_id}")
def delete_chat_endpoint(
    chat_id: str,
) -> Dict[str, Any]:

    deleted = delete_chat(chat_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Chat not found.",
        )

    return {
        "message": "Chat deleted successfully."
    }
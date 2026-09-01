from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.chat_service import (
    create_chat, add_chat_source, add_chat_sources, get_chat,
    get_chat_messages, get_chat_sources, add_chat_message,
    get_workspace_chats, delete_chat, remove_chat_source,
)
from backend.services.source_service import get_arxiv_source_for_document

router=APIRouter(prefix="/chats", tags=["chats"])

class CreateChatRequest(BaseModel):
    workspace_id: str
    title: Optional[str] = "New Chat"
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    document_ids: list[str] = Field(default_factory=list)
    sources: list[dict[str, str]] = Field(default_factory=list)

@router.post("")
def create_chat_endpoint(request: CreateChatRequest) -> Dict[str, Any]:
    try:
        chat=create_chat(request.workspace_id,(request.title or "New Chat").strip() or "New Chat")
        chat_id=str(chat["id"]); sources=[]
        if request.source_type and request.source_id:
            sources.append(add_chat_source(chat_id,request.source_type.strip(),request.source_id.strip()))
        ids = list(
            dict.fromkeys(
                str(x).strip()
                for x in request.document_ids
                if str(x).strip()
            )
        )

        generic_sources: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for item in request.sources:
            source_type = str(item.get("source_type") or "").strip()
            source_id = str(item.get("source_id") or "").strip()

            if not source_type or not source_id:
                continue

            key = (source_type, source_id)
            if key in seen:
                continue

            seen.add(key)
            generic_sources.append({
                "source_type": source_type,
                "source_id": source_id,
            })

        combined: list[dict[str, str]] = []

        for document_id in ids:
            arxiv_source = get_arxiv_source_for_document(
                request.workspace_id,
                document_id,
            )
            if arxiv_source:
                combined.append({
                    "source_type": "arxiv",
                    "source_id": document_id,
                })
            else:
                combined.append({
                    "source_type": "document",
                    "source_id": document_id,
                })

        # Preserve generic sources exactly as supplied (GitHub and future
        # source types use this path).
        combined.extend(generic_sources)

        if combined:
            sources.extend(
                add_chat_sources(chat_id, combined)
            )

        return {
            "chat": chat,
            "source": sources[0] if sources else None,
            "sources": sources,
        }
    except Exception as exc:
        raise HTTPException(status_code=503,detail={"message":"Unable to create chat or attach its sources.","error":str(exc)}) from exc

@router.get("/workspace/{workspace_id}")
def get_workspace_chats_endpoint(workspace_id: str)->list[dict[str,Any]]:
    return get_workspace_chats(workspace_id)

@router.get("/{chat_id}")
def get_chat_endpoint(chat_id:str)->Dict[str,Any]:
    chat=get_chat(chat_id)
    if not chat: raise HTTPException(status_code=404,detail="Chat not found.")
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
    role:str
    content:str

@router.post("/{chat_id}/messages")
def add_chat_message_endpoint(chat_id:str,request:AddChatMessageRequest)->Dict[str,Any]:
    if not get_chat(chat_id): raise HTTPException(status_code=404,detail="Chat not found.")
    if request.role not in {"user","assistant"}: raise HTTPException(status_code=400,detail="Role must be 'user' or 'assistant'.")
    if not request.content.strip(): raise HTTPException(status_code=400,detail="Message content must not be empty.")
    try: return add_chat_message(chat_id,request.role,request.content)
    except Exception as exc: raise HTTPException(status_code=503,detail={"message":"Unable to save chat message.","error":str(exc)}) from exc

class AddChatSourceRequest(BaseModel):
    source_type:str
    source_id:str

@router.post("/{chat_id}/sources")
def add_chat_source_endpoint(chat_id:str,request:AddChatSourceRequest)->Dict[str,Any]:
    if not get_chat(chat_id): raise HTTPException(status_code=404,detail="Chat not found.")
    if not request.source_type.strip() or not request.source_id.strip(): raise HTTPException(status_code=400,detail="Source type and source ID must not be empty.")
    try: return add_chat_source(chat_id,request.source_type,request.source_id)
    except Exception as exc: raise HTTPException(status_code=503,detail={"message":"Unable to attach source to conversation.","error":str(exc)}) from exc

@router.delete("/{chat_id}/sources/{source_type}/{source_id}")
def delete_chat_source_endpoint(chat_id:str,source_type:str,source_id:str)->Dict[str,Any]:
    if not get_chat(chat_id): raise HTTPException(status_code=404,detail="Chat not found.")
    try: removed=remove_chat_source(chat_id,source_type,source_id)
    except Exception as exc: raise HTTPException(status_code=503,detail={"message":"Unable to remove chat source.","error":str(exc)}) from exc
    if not removed: raise HTTPException(status_code=404,detail="Chat source not found.")
    return {"message":"Chat source removed successfully."}

@router.get("/{chat_id}/sources")
def get_chat_sources_endpoint(chat_id:str)->list[dict[str,Any]]:
    if not get_chat(chat_id): raise HTTPException(status_code=404,detail="Chat not found.")
    return get_chat_sources(chat_id)

@router.patch("/{chat_id}")
def update_chat_endpoint(chat_id:str,request:dict[str,Any])->Dict[str,Any]:
    from backend.services.chat_service import update_chat_title
    if not get_chat(chat_id): raise HTTPException(status_code=404,detail="Chat not found.")
    title=str(request.get("title") or "").strip()
    if not title: raise HTTPException(status_code=400,detail="Title must not be empty.")
    try: return update_chat_title(chat_id,title) or get_chat(chat_id)
    except Exception as exc: raise HTTPException(status_code=503,detail={"message":"Unable to update chat.","error":str(exc)}) from exc

@router.delete("/{chat_id}")
def delete_chat_endpoint(chat_id:str)->Dict[str,Any]:
    try: deleted=delete_chat(chat_id)
    except Exception as exc: raise HTTPException(status_code=503,detail={"message":"Unable to delete chat.","error":str(exc)}) from exc
    if not deleted: raise HTTPException(status_code=404,detail="Chat not found.")
    return {"message":"Chat deleted successfully."}
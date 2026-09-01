from __future__ import annotations

from threading import RLock
from typing import Any, Iterable

from backend.db.database import supabase

# The project uses one shared synchronous Supabase client. FastAPI runs sync
# handlers in a thread pool, so serialize these small chat DB operations to
# prevent concurrent reuse of the same HTTP/2 connection pool on Windows.
_chat_db_lock = RLock()

def _execute(builder: Any) -> Any:
    with _chat_db_lock:
        return builder.execute()

def create_chat(workspace_id: str, title: str = "New Chat") -> dict[str, Any]:
    response = _execute(
        supabase.table("chats").insert({"workspace_id": workspace_id, "title": title})
    )
    if not response.data:
        raise RuntimeError("Failed to create chat.")
    return response.data[0]

def update_chat_title(chat_id: str, title: str) -> dict[str, Any] | None:
    title = title.strip()
    if not title:
        return get_chat(chat_id)
    response = _execute(supabase.table("chats").update({"title": title}).eq("id", chat_id))
    return response.data[0] if response.data else None

def get_chat(chat_id: str) -> dict[str, Any] | None:
    response = _execute(supabase.table("chats").select("*").eq("id", chat_id).limit(1))
    return response.data[0] if response.data else None

def add_chat_source(chat_id: str, source_type: str, source_id: str) -> dict[str, Any]:
    source_type = source_type.strip(); source_id = source_id.strip()
    if not source_type or not source_id:
        raise ValueError("Source type and source ID must not be empty.")
    existing = _execute(
        supabase.table("chat_sources").select("*")
        .eq("chat_id", chat_id).eq("source_type", source_type).eq("source_id", source_id).limit(1)
    )
    if existing.data:
        return existing.data[0]
    response = _execute(
        supabase.table("chat_sources").insert({"chat_id": chat_id, "source_type": source_type, "source_id": source_id})
    )
    if not response.data:
        raise RuntimeError("Failed to add chat source.")
    return response.data[0]

def add_chat_sources(chat_id: str, sources: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    normalized=[]; seen=set()
    for item in sources:
        st=str(item.get("source_type") or "").strip(); sid=str(item.get("source_id") or "").strip()
        if not st or not sid:
            raise ValueError("Every chat source needs source_type and source_id.")
        key=(st,sid)
        if key not in seen:
            seen.add(key); normalized.append({"source_type":st,"source_id":sid})
    if not normalized: return []
    existing_response=_execute(supabase.table("chat_sources").select("*").eq("chat_id",chat_id))
    existing_rows=existing_response.data or []
    existing_keys={(str(r.get("source_type") or ""),str(r.get("source_id") or "")) for r in existing_rows}
    missing=[{"chat_id":chat_id,**item} for item in normalized if (item["source_type"],item["source_id"]) not in existing_keys]
    inserted=[]
    if missing:
        inserted_response=_execute(supabase.table("chat_sources").insert(missing))
        inserted=inserted_response.data or []
    all_rows=existing_rows+inserted
    by_key={(str(r.get("source_type") or ""),str(r.get("source_id") or "")):r for r in all_rows}
    return [by_key[(item["source_type"],item["source_id"])] for item in normalized if (item["source_type"],item["source_id"]) in by_key]

def remove_chat_source(chat_id: str, source_type: str, source_id: str) -> bool:
    response=_execute(
        supabase.table("chat_sources").delete().eq("chat_id",chat_id).eq("source_type",source_type).eq("source_id",source_id)
    )
    return bool(response.data)

def get_chat_messages(chat_id: str) -> list[dict[str, Any]]:
    response=_execute(supabase.table("chat_messages").select("*").eq("chat_id",chat_id).order("created_at",desc=False))
    return response.data or []

def get_chat_sources(chat_id: str) -> list[dict[str, Any]]:
    response=_execute(supabase.table("chat_sources").select("*").eq("chat_id",chat_id).order("created_at",desc=False))
    return response.data or []

def add_chat_message(chat_id: str, role: str, content: str) -> dict[str, Any]:
    response=_execute(supabase.table("chat_messages").insert({"chat_id":chat_id,"role":role,"content":content}))
    if not response.data: raise RuntimeError("Failed to save chat message.")
    return response.data[0]

def get_workspace_chats(workspace_id: str) -> list[dict[str, Any]]:
    response=_execute(supabase.table("chats").select("*").eq("workspace_id",workspace_id).order("updated_at",desc=True))
    return response.data or []

def delete_chat(chat_id: str) -> bool:
    if get_chat(chat_id) is None: return False
    _execute(supabase.table("chat_messages").delete().eq("chat_id",chat_id))
    _execute(supabase.table("chat_sources").delete().eq("chat_id",chat_id))
    _execute(supabase.table("chats").delete().eq("id",chat_id))
    return True
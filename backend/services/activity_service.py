from __future__ import annotations

from typing import Any

from backend.db.database import supabase


ACTIVITY_TYPES = {
    "document_added",
    "paper_added",
    "model_added",
    "repository_added",
    "chat_started",
    "research_performed",
}


def create_activity(
    workspace_id: str,
    activity_type: str,
    title: str,
    description: str | None = None,
    reference_id: str | None = None,
    reference_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a lightweight workspace activity event."""
    normalized_type = str(activity_type or "").strip().lower()
    if normalized_type not in ACTIVITY_TYPES:
        raise ValueError(f"Unsupported activity type: {activity_type}")

    normalized_title = str(title or "").strip()
    if not normalized_title:
        raise ValueError("Activity title must not be empty.")

    row = {
        "workspace_id": workspace_id,
        "activity_type": normalized_type,
        "title": normalized_title,
        "description": str(description).strip() if description else None,
        "reference_id": str(reference_id).strip() if reference_id else None,
        "reference_type": str(reference_type).strip() if reference_type else None,
        "metadata": metadata or {},
    }

    response = supabase.table("recent_activity").insert(row).execute()
    if not response.data:
        raise RuntimeError("Failed to create activity.")

    return response.data[0]


def list_recent_activity(
    workspace_id: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return the newest activity events for a workspace."""
    safe_limit = max(1, min(int(limit), 50))

    response = (
        supabase
        .table("recent_activity")
        .select("*")
        .eq("workspace_id", workspace_id)
        .order("created_at", desc=True)
        .limit(safe_limit)
        .execute()
    )

    return response.data or []

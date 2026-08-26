from typing import Any

from backend.db.database import supabase


def create_document_chunks(
    document_id: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    rows = []

    for index, chunk in enumerate(chunks):
        content = str(
            chunk.get("content")
            or chunk.get("raw_content")
            or ""
        ).strip()

        if not content:
            continue

        rows.append(
            {
                "document_id": document_id,
                "chunk_index": index,
                "content": content,
                "page": chunk.get("page"),
            }
        )

    if not rows:
        return []

    response = (
        supabase
        .table("document_chunks")
        .insert(rows)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Failed to save document chunks."
        )

    return response.data


def get_document_chunks(
    document_id: str,
) -> list[dict[str, Any]]:
    response = (
        supabase
        .table("document_chunks")
        .select("*")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )

    return response.data or []
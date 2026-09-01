from typing import Any

from backend.db.database import supabase


BUCKET_NAME = "documents"


def build_document_storage_path(
    document_id: str,
    filename: str,
) -> str:
    """Return the canonical Storage path for an underlying document."""
    return f"{document_id}/{filename}"


def get_document(
    document_id: str,
) -> dict[str, Any] | None:
    """Return the shared underlying document record, if it exists."""
    response = (
        supabase
        .table("documents")
        .select("*")
        .eq("id", document_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    return response.data[0]


def upload_document_file(
    storage_path: str,
    file_bytes: bytes,
    content_type: str | None = None,
) -> str:
    file_options: dict[str, Any] = {
        "upsert": False,
    }

    if content_type:
        file_options["content-type"] = content_type

    supabase.storage.from_(BUCKET_NAME).upload(
        storage_path,
        file_bytes,
        file_options=file_options,
    )

    return storage_path


def create_document(
    document_id: str,
    filename: str,
    storage_path: str,
    content_type: str | None = None,
    pages: int | None = None,
    characters: int | None = None,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    response = (
        supabase
        .table("documents")
        .insert({
            "id": document_id,
            "filename": filename,
            "content_type": content_type,
            "storage_path": storage_path,
            "pages": pages,
            "characters": characters,
            "size_bytes": size_bytes,
        })
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Failed to create document record."
        )

    return response.data[0]
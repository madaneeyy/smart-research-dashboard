from __future__ import annotations

"""
ArXiv -> workspace ingestion.

ArXiv papers are intentionally ingested through the existing uploaded-document
pipeline. This module owns only acquisition, normalization, persistence and
workspace association; downstream retrieval remains the document retriever.
"""

import re
from typing import Any
from urllib.request import Request, urlopen

from backend.db.database import supabase
from backend.services.document_chunk_service import create_document_chunks, get_document_chunks
from backend.services.document_storage_service import (
    build_document_storage_path,
    create_document,
    get_document,
    upload_document_file,
)
from backend.services.source_service import create_source, list_sources
from backend.services.workspace_document_service import (
    create_workspace_document,
    get_workspace_document,
)
from src.services.document_rag.document_cache import DocumentCache
from src.services.document_rag.document_chunker import UploadedDocumentChunker
from src.services.document_rag.document_service import DocumentService


ARXIV_ABS_RE = re.compile(
    r"^/(?:abs|pdf)/(?P<identifier>[^/?#]+)",
    re.IGNORECASE,
)
ARXIV_ID_RE = re.compile(
    r"^(?:arxiv:)?(?P<identifier>"
    r"(?:\d{4}\.\d{4,5}(?:v\d+)?|"
    r"[a-zA-Z-]+(?:\.[a-zA-Z-]+)?/\d{7}(?:v\d+)?)"
    r")$",
)


def normalize_arxiv_id(value: str) -> str:
    """Normalize an arXiv ID or arXiv URL to an unversioned canonical ID."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("An arXiv ID or URL is required.")

    identifier = raw

    if raw.lower().startswith(("http://", "https://")):
        from urllib.parse import urlparse

        parsed = urlparse(raw)
        host = parsed.netloc.lower()
        if host not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            raise ValueError("The URL must point to arxiv.org.")

        match = ARXIV_ABS_RE.match(parsed.path)
        if not match:
            raise ValueError("Could not extract an arXiv ID from the URL.")

        identifier = match.group("identifier")
        if identifier.lower().endswith(".pdf"):
            identifier = identifier[:-4]

    identifier = identifier.strip().rstrip("/")
    if identifier.lower().startswith("arxiv:"):
        identifier = identifier[6:]

    match = ARXIV_ID_RE.match(identifier)
    if not match:
        raise ValueError(f"Unsupported arXiv identifier: {value}")

    identifier = match.group("identifier")
    identifier = re.sub(r"v\d+$", "", identifier, flags=re.IGNORECASE)
    return identifier


def canonical_arxiv_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def canonical_arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _download_pdf(pdf_url: str, max_bytes: int = 50 * 1024 * 1024) -> bytes:
    request = Request(
        pdf_url,
        headers={
            "User-Agent": (
                "SmartResearchDashboard/0.1 "
                "(research project; arXiv ingestion)"
            )
        },
    )

    with urlopen(request, timeout=45) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("The arXiv PDF is larger than the 50 MB ingestion limit.")

        data = response.read(max_bytes + 1)

    if len(data) > max_bytes:
        raise ValueError("The arXiv PDF is larger than the 50 MB ingestion limit.")

    if not data:
        raise ValueError("The arXiv PDF download was empty.")

    # A PDF must start with the PDF signature. arXiv can otherwise return an
    # HTML error page when a request is rejected.
    if not data.startswith(b"%PDF-"):
        preview = data[:200].decode("utf-8", errors="ignore").strip()
        raise ValueError(
            "arXiv did not return a PDF. "
            + (f"Response started with: {preview[:120]}" if preview else "")
        )

    return data


def _metadata_document_id(metadata: dict[str, Any]) -> str:
    return str(metadata.get("document_id") or "").strip()


def _find_workspace_arxiv_source(
    workspace_id: str,
    arxiv_id: str,
) -> dict[str, Any] | None:
    for source in list_sources(workspace_id) or []:
        source_type = str(source.get("source_type") or "").strip().lower()
        if source_type not in {"arxiv", "arxiv_paper"}:
            continue

        metadata = source.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue

        try:
            existing_id = normalize_arxiv_id(
                str(metadata.get("arxiv_id") or source.get("url") or "")
            )
        except ValueError:
            continue

        if existing_id == arxiv_id:
            return source

    return None


def _find_shared_arxiv_document(arxiv_id: str) -> str | None:
    """
    Reuse a previously ingested paper from another workspace when its
    canonical document/chunks are still present.
    """
    response = (
        supabase
        .table("workspace_sources")
        .select("metadata")
        .in_("source_type", ["arxiv", "arxiv_paper"])
        .execute()
    )

    for row in response.data or []:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue

        try:
            existing_id = normalize_arxiv_id(
                str(metadata.get("arxiv_id") or "")
            )
        except ValueError:
            continue

        if existing_id != arxiv_id:
            continue

        document_id = _metadata_document_id(metadata)
        if not document_id:
            continue

        document = get_document(document_id)
        if document and get_document_chunks(document_id):
            return document_id

    return None


def _persist_document(
    document: dict[str, Any],
    raw_bytes: bytes,
) -> tuple[str, dict[str, Any]]:
    """
    Persist the extracted document exactly like the current upload pipeline.

    Returns (document_id, canonical document record).
    """
    document_id = str(document["document_id"])
    existing_document = get_document(document_id)

    if existing_document:
        if get_document_chunks(document_id):
            return document_id, existing_document

        chunker = UploadedDocumentChunker(
            target_chars=1000,
            max_chars=1600,
            overlap_chars=180,
            min_chunk_chars=20,
        )
        chunks = chunker.chunk_documents([document])
        if not chunks:
            raise ValueError("No usable chunks were created from the arXiv PDF.")

        create_document_chunks(document_id=document_id, chunks=chunks)
        repaired = get_document(document_id)
        if not repaired:
            raise RuntimeError("The repaired arXiv document could not be reloaded.")
        return document_id, repaired

    chunker = UploadedDocumentChunker(
        target_chars=1000,
        max_chars=1600,
        overlap_chars=180,
        min_chunk_chars=20,
    )
    chunks = chunker.chunk_documents([document])
    if not chunks:
        raise ValueError("No usable chunks were created from the arXiv PDF.")

    filename = str(document.get("filename") or f"{document_id}.pdf")
    storage_path = build_document_storage_path(document_id, filename)

    upload_document_file(
        storage_path=storage_path,
        file_bytes=raw_bytes,
        content_type="application/pdf",
    )

    create_document(
        document_id=document_id,
        filename=filename,
        storage_path=storage_path,
        content_type="application/pdf",
        pages=document.get("pages"),
        characters=document.get("characters"),
        size_bytes=len(raw_bytes),
    )

    create_document_chunks(
        document_id=document_id,
        chunks=chunks,
    )

    persisted = get_document(document_id)
    if not persisted:
        raise RuntimeError("The persisted arXiv document could not be reloaded.")

    return document_id, persisted


def ingest_arxiv_source(
    workspace_id: str,
    title: str,
    url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Ingest one arXiv paper into the existing document family and workspace.

    The operation is idempotent at the workspace level and reuses a shared
    underlying document/chunk set whenever one already exists.
    """
    incoming_metadata = dict(metadata or {})

    identifier_candidates = [
        str(incoming_metadata.get("arxiv_id") or "").strip(),
        str(incoming_metadata.get("research_id") or "").strip(),
        str(url or "").strip(),
    ]

    arxiv_id = None
    for candidate in identifier_candidates:
        if not candidate:
            continue
        try:
            arxiv_id = normalize_arxiv_id(candidate)
            break
        except ValueError:
            continue

    if not arxiv_id:
        raise ValueError(
            "Could not determine the arXiv ID. Provide an arXiv ID or arXiv URL."
        )

    canonical_url = canonical_arxiv_url(arxiv_id)
    pdf_url = canonical_arxiv_pdf_url(arxiv_id)

    existing_source = _find_workspace_arxiv_source(workspace_id, arxiv_id)

    # Fast idempotent path: source + document association + chunks already exist.
    if existing_source:
        existing_metadata = existing_source.get("metadata") or {}
        document_id = _metadata_document_id(existing_metadata)

        if document_id:
            document = get_document(document_id)
            workspace_document = get_workspace_document(
                document_id=document_id,
                workspace_id=workspace_id,
            )
            if document and get_document_chunks(document_id):
                if not workspace_document:
                    create_workspace_document(
                        workspace_id=workspace_id,
                        document_id=document_id,
                        filename=str(document.get("filename") or f"{arxiv_id}.pdf"),
                        content_type=document.get("content_type") or "application/pdf",
                        pages=document.get("pages"),
                        characters=document.get("characters"),
                        size_bytes=document.get("size_bytes"),
                        status="ready",
                    )
                return existing_source

    # Prefer a document already ingested into another workspace.
    document_id = _find_shared_arxiv_document(arxiv_id)
    document = get_document(document_id) if document_id else None

    if not document or not get_document_chunks(document_id):
        raw_bytes = _download_pdf(pdf_url)
        filename = f"{arxiv_id.replace('/', '_')}.pdf"

        document = DocumentCache.get_or_extract(
            filename=filename,
            raw_bytes=raw_bytes,
            content_type="application/pdf",
            extractor=DocumentService.extract,
        )

        document_id, document = _persist_document(
            document=document,
            raw_bytes=raw_bytes,
        )

    workspace_document = get_workspace_document(
        document_id=document_id,
        workspace_id=workspace_id,
    )

    if not workspace_document:
        create_workspace_document(
            workspace_id=workspace_id,
            document_id=document_id,
            filename=str(document.get("filename") or f"{arxiv_id}.pdf"),
            content_type=document.get("content_type") or "application/pdf",
            pages=document.get("pages"),
            characters=document.get("characters"),
            size_bytes=document.get("size_bytes"),
            status="ready",
        )

    final_metadata: dict[str, Any] = {
        **incoming_metadata,
        "arxiv_id": arxiv_id,
        "document_id": document_id,
        "canonical_url": canonical_url,
        "pdf_url": pdf_url,
        "title": title,
        "ingestion_status": "ready",
        "source_family": "document",
    }

    # Re-check after ingestion in case another request created the source while
    # this request was downloading/extracting.
    existing_source = _find_workspace_arxiv_source(workspace_id, arxiv_id)
    if existing_source:
        existing_metadata = existing_source.get("metadata") or {}
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}

        merged_metadata = {
            **existing_metadata,
            **final_metadata,
            "document_id": document_id,
            "arxiv_id": arxiv_id,
            "canonical_url": canonical_url,
            "pdf_url": pdf_url,
            "ingestion_status": "ready",
            "source_family": "document",
        }

        response = (
            supabase
            .table("workspace_sources")
            .update({
                "title": title.strip() or existing_source.get("title") or f"arXiv:{arxiv_id}",
                "url": canonical_url,
                "metadata": merged_metadata,
            })
            .eq("id", existing_source["id"])
            .execute()
        )
        return response.data[0] if response.data else existing_source

    return create_source(
        workspace_id=workspace_id,
        source_type="arxiv",
        title=title.strip() or f"arXiv:{arxiv_id}",
        url=canonical_url,
        metadata=final_metadata,
    )

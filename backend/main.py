from __future__ import annotations
import re
import json

import os
import time
from typing import Any, Dict, List, Optional

import ollama
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.routes.workspace import router as workspace_router
from backend.routes.source import (
    source_router,
    workspace_source_router,
)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# PROJECT RAG COMPONENTS
# ============================================================

from src.services.rag.chunker import DocumentChunker
from src.services.rag.hybrid_retriever import HybridRetriever
from src.services.github.github_content import GitHubContentService
from src.services.document_rag.document_service import DocumentService
from src.services.document_rag.document_cache import DocumentCache
from src.services.context_router import ContextRouter
from src.services.document_rag.document_chunker import UploadedDocumentChunker
from src.services.document_rag.document_retriever import DocumentRetriever
import inspect

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("RAG_MAX_TOP_K", "10"))

OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1000"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")

MAX_CONTEXT_CHARACTERS = int(
    os.getenv("MAX_CONTEXT_CHARACTERS", "2000000")
)

# ============================================================
# HYBRID RETRIEVER CONFIGURATION
# ============================================================
#
# These values are deliberately kept in one place.
# They are the production integration of the retriever that we
# evaluated with the existing retrieval framework.
#
# IMPORTANT:
# Do not recreate/reinitialize this inside /ask.
# The embedding model/cache should be reused between requests.
# ============================================================

hybrid_retriever = HybridRetriever(
    semantic_weight=0.5,
    bm25_weight=0.5,
    rrf_k=60,
    candidate_multiplier=4,
    mmr_lambda=0.75,
    relevance_filter_enabled=True,
    relevance_threshold=0.30,
    relevance_relative_threshold=0.70,
    near_duplicate_threshold=0.92,
    metadata_bonus_weight=0.08,
    lexical_bonus_weight=0.12,
    bm25_presence_bonus=0.04,
    complementarity_bonus_weight=0.10,
    protected_primary_count=2,
    protected_primary_margin=0.08,
    minimum_results=1,
)

# The chunker is lightweight and can safely be reused.
document_chunker = DocumentChunker(
    max_chars=1800,
    min_chars=250,
    overlap_chars=250,
)

uploaded_document_chunker = UploadedDocumentChunker(
    target_chars=1000,
    max_chars=1600,
    overlap_chars=180,
    min_chunk_chars=20,
)

document_retriever = DocumentRetriever(
    overview_top_k=7,
    focused_top_k=6,
    candidate_multiplier=5,
    mmr_lambda=0.72,
     profiling_enabled=True,
)

ollama_client = ollama.Client(host=OLLAMA_HOST)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Smart Research AI API",
    description="Structure-aware RAG with HybridRetriever and Qwen.",
    version="2.4.0",
)
app.include_router(workspace_router)
app.include_router(source_router)
app.include_router(workspace_source_router)
# ============================================================
# CORS
# ============================================================

cors_value = os.getenv("CORS_ORIGINS", "*").strip()

if cors_value == "*":
    cors_origins = ["*"]
    allow_credentials = False
else:
    cors_origins = [
        item.strip()
        for item in cors_value.split(",")
        if item.strip()
    ]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question asked by the user.",
    )

    # New preferred input for GitHub chat.
    # When provided, the backend fetches and prepares repository context
    # through GitHubContentService instead of requiring the frontend to
    # construct the context itself.
    github_url: Optional[str] = Field(
        default=None,
        description="GitHub repository URL to research.",
    )

    branch: Optional[str] = Field(
        default=None,
        description="Optional Git branch. Uses the repository default branch when omitted.",
    )

    # Backward-compatible path: an already-built research/repository context
    # can still be supplied by existing callers.
    context: Optional[str] = Field(
        default=None,
        description="Optional pre-built research/repository context.",
    )

    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Optional previous user/assistant messages.",
    )
    chat_id: Optional[str] = Field(default=None, description="Stable chat/session identifier.")

    # Uploaded document IDs attached to this chat/request.
    # Optional so existing GitHub/general-chat clients remain compatible.
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Uploaded document IDs to use as RAG context.",
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
    )

    # Optional metadata for callers that already have a single source.
    source_path: Optional[str] = None
    source_category: Optional[str] = None



# ============================================================
# HELPERS
# ============================================================

def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _ollama_stat(response: Any, name: str) -> Optional[float]:
    value = getattr(response, name, None)
    if value is None and isinstance(response, dict):
        value = response.get(name)
    if value is None:
        return None
    try:
        return round(float(value) / 1_000_000, 2)
    except (TypeError, ValueError):
        return None



def _profile_start() -> float:
    return time.perf_counter()


def _profile_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _print_stage_timings(
    timings: Dict[str, float],
    ollama_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Print the slowest request stages first."""
    print("\n" + "=" * 80)
    print("LATENCY PROFILING")
    print("=" * 80)

    total = timings.get("total_ms", 0.0)

    for name, value in sorted(
        timings.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        pct = (value / total * 100.0) if total > 0 else 0.0
        print(f"{name:32s}: {value:10,.2f} ms  ({pct:6.1f}%)")

    if ollama_metrics:
        print("-" * 80)
        for name, value in ollama_metrics.items():
            if value is not None:
                print(f"ollama.{name:27s}: {value}")

    print("=" * 80 + "\n")

def build_github_documents(
    github_url: str,
    question: str,
    branch: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build structured GitHub documents for the RAG pipeline.

    IMPORTANT:
    GitHubContentService must expose build_documents_for_query().
    This keeps repository discovery/file ranking in the GitHub service
    while allowing the backend to chunk each selected file separately.
    """
    try:
        builder = getattr(
            GitHubContentService,
            "build_documents_for_query",
            None,
        )

        if not callable(builder):
            raise RuntimeError(
                "GitHubContentService.build_documents_for_query() "
                "is missing. Add the query-aware document builder to "
                "src/services/github_content.py before starting the backend."
            )

        documents = builder(
            github_url=github_url,
            query=question,
            branch=branch,
        )

        if not isinstance(documents, list):
            raise TypeError(
                "build_documents_for_query() must return a list of documents."
            )

        cleaned: List[Dict[str, Any]] = []

        for document in documents:
            if not isinstance(document, dict):
                continue

            content = str(
                document.get("content") or ""
            ).strip()

            path = str(
                document.get("path") or ""
            ).strip()

            if not content or not path:
                continue

            cleaned.append(
                {
                    **document,
                    "content": content,
                    "path": path,
                    "category": str(
                        document.get("category")
                        or "source"
                    ),
                }
            )

        return cleaned

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid GitHub repository request.",
                "error": str(exc),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Failed to build GitHub repository documents.",
                "error": str(exc),
                "github_url": github_url,
            },
        )

def clean_history(
    history: List[Dict[str, str]],
    max_messages: int = 8,
) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []

    for message in history[-max_messages:]:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content,
            }
        )

    return cleaned


def get_chunk_content(chunk: Dict[str, Any]) -> str:
    return str(
        chunk.get("content")
        or chunk.get("raw_content")
        or ""
    ).strip()


def get_chunk_source(chunk: Dict[str, Any]) -> str:
    return str(
        chunk.get("path")
        or chunk.get("source")
        or chunk.get("file")
        or ""
    ).strip()


def get_chunk_section(chunk: Dict[str, Any]) -> str:
    return str(
        chunk.get("section")
        or chunk.get("heading")
        or chunk.get("title")
        or ""
    ).strip()


def format_retrieved_context(
    chunks: List[Dict[str, Any]],
) -> str:
    evidence: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        content = get_chunk_content(chunk)

        if not content:
            continue

        source = get_chunk_source(chunk) or "Unknown source"
        section = get_chunk_section(chunk) or "Unknown section"

        parent_section = str(
            chunk.get("parent_section") or ""
        ).strip()

        section_path = chunk.get("section_path") or []

        section_path_text = ""
        if isinstance(section_path, list) and section_path:
            section_path_text = " > ".join(
                str(item) for item in section_path
            )

        metadata_lines = [
            f"Evidence {index}",
            f"Source: {source}",
            f"Section: {section}",
        ]

        page = chunk.get("page")
        if page is not None:
            metadata_lines.append(f"Page: {page}")

        filename = str(chunk.get("filename") or "").strip()
        if filename and filename != source:
            metadata_lines.append(f"Document: {filename}")

        if parent_section:
            metadata_lines.append(
                f"Parent section: {parent_section}"
            )

        if section_path_text:
            metadata_lines.append(
                f"Section path: {section_path_text}"
            )

        metadata_lines.extend(
            [
                "Content:",
                content,
            ]
        )

        evidence.append(
            "\n".join(metadata_lines)
        )

    return "\n\n------------------------------\n\n".join(evidence)


def build_system_prompt(
    retrieved_context: str,
) -> str:
    """Build a compact, source-grounded prompt for RAG answers.

    Keep instructions short because this prompt is sent to Qwen on every
    request; retrieved evidence remains unchanged.
    """
    return f"""
You are a technical research assistant. Answer the user's question directly,
accurately, and concisely using the retrieved evidence below.

GROUNDING
- Treat retrieved evidence as authoritative for source-specific claims.
- Do not invent facts, files, functions, parameters, results, citations,
  implementation details, or author intent.
- Combine evidence from multiple chunks when they complement each other.
- Preserve exact technical terminology, identifiers, filenames, APIs,
  algorithms, architecture names, and configuration values.
- Prefer claims explicitly supported by the evidence.

INFERENCE
- You may use general technical knowledge to explain what the evidence means,
  but never turn that explanation into a source claim.
- When interpreting, use wording such as "This means", "This can be
  interpreted as", or "In practice".
- For "why" questions, give the stated reason when available; otherwise
  explain only what is reasonably implied. Do not invent author intent.
- If an important part of the question cannot be answered from the evidence,
  say so briefly. Do not add generic limitation/disclaimer sections.

CODE / IMPLEMENTATION
- Describe only what the retrieved code supports.
- Preserve exact identifiers and mention relevant files/modules when present.
- If code is incomplete, state only the limitation that affects the answer.

ANSWER STYLE
- Start with the answer; do not begin with "The retrieved evidence...",
  "The provided context...", or similar unless necessary.
- Use bullets/numbered sections for multi-part questions.
- Be concise for simple questions and detailed only when needed.
- Avoid unnecessarily strong claims such as "guarantees", "proves",
  "ensures", "always", or "optimal" unless explicitly supported.
- Do not fabricate source sections, filenames, URLs, or citations.

FINAL CHECK
Before answering: answer the actual question; keep source-specific claims
grounded; distinguish inference from source statements; preserve terminology;
avoid unnecessary disclaimers; keep the response no more complicated than
necessary.

RETRIEVED EVIDENCE
{retrieved_context}
END RETRIEVED EVIDENCE
""".strip()

def build_messages(
    question: str,
    history: List[Dict[str, str]],
    system_prompt: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(clean_history(history))

    messages.append(
        {
            "role": "user",
            "content": question.strip(),
        }
    )

    return messages


def serialize_chunk(
    chunk: Dict[str, Any],
    rank: int,
) -> Dict[str, Any]:
    """
    Expose useful evidence and diagnostics to Streamlit.

    We intentionally preserve the fields generated by the current
    HybridRetriever rather than inventing a second ranking system in
    the backend.
    """

    output: Dict[str, Any] = {
        "rank": rank,
        "content": get_chunk_content(chunk),
        "raw_content": chunk.get("raw_content"),
        "source": get_chunk_source(chunk),
        "section": get_chunk_section(chunk),
        "parent_section": chunk.get("parent_section"),
        "section_path": chunk.get("section_path"),
        "chunk_index": chunk.get("chunk_index"),
        "chunk_type": chunk.get("chunk_type"),
        "language": chunk.get("language"),
        "char_count": chunk.get("char_count"),
        "category": chunk.get("category"),
        "document_id": chunk.get("document_id"),
        "filename": chunk.get("filename"),
        "page": chunk.get("page"),
        "pages": chunk.get("pages"),
        "retrieval_source": chunk.get("retrieval_source"),
        "retriever_name": chunk.get("retriever_name"),
        "github_score": chunk.get("github_score"),
        "github_rank": chunk.get("github_rank"),
        "github_reasons": chunk.get("github_reasons"),
        "github_matched_terms": chunk.get(
            "github_matched_terms"
        ),
    }

    diagnostic_fields = [
        "retrieval_rank",
        "semantic_rank",
        "bm25_rank",
        "hybrid_score",
        "semantic_score",
        "bm25_score",
        "rrf_score",
        "query_relevance_score",
        "mmr_score",
        "mmr_redundancy",
        "metadata_bonus",
        "lexical_bonus",
        "complementarity_score",
        "query_type",
        "primary_protected",
        "candidate_pool_size",
        "post_filter_pool_size",

        # Uploaded-document retrieval diagnostics.
        "relevance_score",
        "similarity_score",
        "lexical_score",
        "phrase_score",
        "section_score",
        "intent_score",
        "redundancy_score",
        "retrieval_source",
        "retriever_name",

        # GitHub file-selection diagnostics.
        "github_score",
        "github_rank",
        "github_reasons",
        "github_matched_terms",
    ]

    for field_name in diagnostic_fields:
        if field_name in chunk:
            output[field_name] = chunk[field_name]

    return output


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "Smart Research AI API",
        "model": OLLAMA_MODEL,
        "ollama_host": OLLAMA_HOST,
        "retriever": "HybridRetriever",
        "chunker": "DocumentChunker",
        "document_chunker": "UploadedDocumentChunker",
        "document_retriever": "DocumentRetriever",
        "top_k": DEFAULT_TOP_K,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        response = ollama_client.list()

        models = []

        for model in getattr(response, "models", []):
            name = getattr(model, "model", None)

            if name:
                models.append(str(name))

        return {
            "status": "ok",
            "ollama": "connected",
            "model": OLLAMA_MODEL,
            "model_available": OLLAMA_MODEL in models,
            "models": models,
            "github_retriever": "HybridRetriever",
            "github_chunker": "DocumentChunker",
            "document_retriever": "DocumentRetriever",
            "document_chunker": "UploadedDocumentChunker",
            "github_content": "GitHubContentService",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "ollama": "unavailable",
                "model": OLLAMA_MODEL,
                "ollama_host": OLLAMA_HOST,
                "error": str(exc),
            },
        )


# ============================================================
# GITHUB URL DETECTION
# ============================================================

# Matches GitHub repository URLs embedded anywhere in a message. It accepts:
#   https://github.com/owner/repo
#   http://github.com/owner/repo
#   https://www.github.com/owner/repo.git
#   github.com/owner/repo/tree/main
#   github.com/owner/repo/blob/main/file.py
#   github.com/owner/repo/issues/123
#   github.com/owner/repo/pull/123
#   github.com/owner/repo/discussions/123
#   git@github.com:owner/repo.git
#   ssh://git@github.com/owner/repo.git
#
# The current GitHubContentService operates at repository level, so all
# repository resource URLs are normalized to https://github.com/owner/repo.
import re

GITHUB_HTTPS_RE = re.compile(
    r"(?P<url>https?://(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?P<path>/[^\s<>'\"`\]]*)?)",
    re.IGNORECASE,
)

GITHUB_BARE_RE = re.compile(
    r"(?P<url>(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?P<path>/[^\s<>'\"`\]]*)?)",
    re.IGNORECASE,
)

GITHUB_SSH_RE = re.compile(
    r"(?P<url>(?:ssh://git@|git@)github\.com[:/](?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?P<path>/[^\s<>'\"`]*)?)",
    re.IGNORECASE,
)

GITHUB_RESOURCE_TYPES = {
    "tree", "blob", "raw", "commit", "commits", "issues", "issue",
    "pull", "pulls", "discussions", "releases", "actions", "wiki",
    "projects", "security", "settings", "compare", "tags", "branches",
}


def extract_github_reference(text: str) -> Dict[str, Optional[str]]:
    """Extract and normalize a GitHub repository reference from free text."""
    empty: Dict[str, Optional[str]] = {
        "original_url": None,
        "repository_url": None,
        "owner": None,
        "repo": None,
        "resource_type": None,
    }
    if not text:
        return empty

    match = (
        GITHUB_HTTPS_RE.search(text)
        or GITHUB_SSH_RE.search(text)
        or GITHUB_BARE_RE.search(text)
    )
    if not match:
        return empty

    owner = match.group("owner")
    repo = match.group("repo").rstrip(".,;:!?)]}")
    path = (match.groupdict().get("path") or "").rstrip(".,;:!?)]}")

    # .git is a transport suffix, not part of the repository name.
    repo = repo[:-4] if repo.lower().endswith(".git") else repo

    resource_type = None
    path_parts = [part for part in path.split("/") if part]
    if path_parts and path_parts[0].lower() in GITHUB_RESOURCE_TYPES:
        resource_type = path_parts[0].lower()

    original = match.group("url").rstrip(".,;:!?)]}")
    repository_url = f"https://github.com/{owner}/{repo}"

    resource_path = None
    if path_parts:
        if resource_type:
            resource_path = "/".join(path_parts[1:]) or None
        else:
            resource_path = "/".join(path_parts)

    return {
        "original_url": original,
        "repository_url": repository_url,
        "owner": owner,
        "repo": repo,
        "resource_type": resource_type,
        "resource_path": resource_path,
    }


def extract_github_url(text: str) -> Dict[str, Optional[str]]:
    """Compatibility wrapper for callers expecting extract_github_url()."""
    return extract_github_reference(text)


def remove_github_reference(text: str, reference: Dict[str, Optional[str]]) -> str:
    """Remove the detected URL so the RAG query contains only the question."""
    original = reference.get("original_url")
    if not original:
        return text.strip()

    cleaned = text.replace(original, " ")
    # Also remove the normalized repository URL if the input was a bare URL
    # or if punctuation/transport syntax made the exact replacement differ.
    repository_url = reference.get("repository_url")
    if repository_url:
        cleaned = re.sub(re.escape(repository_url), " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "What is this repository about?"

# ============================================================
# RAG RESPONSE FINALIZATION
# ============================================================

def _build_rag_response_payload(
    *,
    question: str,
    answer: str,
    response: Any,
    timings: Dict[str, float],
    request_start: float,
    model_history: List[Dict[str, str]],
    context_origin: str,
    context_scope: str,
    route: Dict[str, Any],
    documents: List[Dict[str, Any]],
    uploaded_documents: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
    retrieved_context: str,
    active_retriever_name: str,
    active_chunker_name: str,
    github_reference: Dict[str, Any],
    github_url: str,
    repository_switched: bool,
    source_switched: bool,
    active_document_ids: List[str],
    resolution: Dict[str, Any],
    requested_top_k: int,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    serialized_sources = [
        serialize_chunk(chunk, index)
        for index, chunk in enumerate(retrieved_chunks, start=1)
    ]
    first_chunk = retrieved_chunks[0] if retrieved_chunks else {}

    ollama_metrics = {
        "total_duration_ms": _ollama_stat(response, "total_duration"),
        "load_duration_ms": _ollama_stat(response, "load_duration"),
        "prompt_eval_duration_ms": _ollama_stat(response, "prompt_eval_duration"),
        "eval_duration_ms": _ollama_stat(response, "eval_duration"),
        "prompt_eval_count": getattr(response, "prompt_eval_count", None),
        "eval_count": getattr(response, "eval_count", None),
    }
    eval_count = ollama_metrics.get("eval_count")
    eval_duration_ms = ollama_metrics.get("eval_duration_ms")
    if (
        isinstance(eval_count, (int, float))
        and isinstance(eval_duration_ms, (int, float))
        and eval_duration_ms > 0
    ):
        ollama_metrics["generation_tokens_per_second"] = round(
            float(eval_count) / (float(eval_duration_ms) / 1000), 2
        )

    timings["total_ms"] = _profile_ms(request_start)
    timings["pre_llm_ms"] = round(
        max(0.0, timings["total_ms"] - timings.get("qwen_wall_ms", 0.0)), 2
    )

    _print_stage_timings(timings=timings, ollama_metrics=ollama_metrics)
    print("\\n" + "=" * 72)
    print("RAG REQUEST PERFORMANCE")
    print("=" * 72)
    print(f"Question: {question}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Repository switched: {repository_switched}")
    print(f"LLM history turns: {len(model_history)}")
    print(f"Context origin: {context_origin}")
    print(f"Context scope: {context_scope}")
    print(f"Context route reason: {route.get('reason')}")
    print(f"Documents acquired: {len(documents)}")
    print(f"Uploaded documents: {len(uploaded_documents)}")
    document_chars = sum(len(str(document.get("content") or "")) for document in documents)
    print(f"Document chars: {document_chars:,}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunks retrieved: {len(retrieved_chunks)}")
    print(f"Retrieved context chars: {len(retrieved_context):,}")
    print(f"System prompt chars: {len(messages[0].get('content', '')) if messages else 0}")
    print(f"Total message chars: {sum(len(str(message.get('content', ''))) for message in messages):,}")
    for name, value in timings.items():
        print(f"{name:28s}: {value:,.2f} ms")
    for name, value in ollama_metrics.items():
        if value is not None:
            print(f"ollama.{name:19s}: {value}")
    print("=" * 72 + "\\n")

    return {
        "question": question,
        "answer": answer,
        "model": OLLAMA_MODEL,
        "retriever": active_retriever_name or ("HybridRetriever" if context_origin != "general_chat" else "none"),
        "chunker": active_chunker_name or ("DocumentChunker" if context_origin != "general_chat" else "none"),
        "context_origin": context_origin,
        "github_resource_type": github_reference.get("resource_type"),
        "github_detected": bool(github_reference.get("repository_url")),
        "chat_mode": context_scope,
        "context_scope": context_scope,
        "context_route": route,
        "source_switched": source_switched,
        "github_url": github_url or None,
        "active_document_ids": active_document_ids,
        "uploaded_documents": [
            {
                "document_id": document.get("document_id"),
                "filename": document.get("filename"),
                "pages": document.get("pages"),
                "characters": document.get("characters"),
                "cache_hit": document.get("cache_hit", False),
            }
            for document in uploaded_documents
        ],
        "repository_switched": repository_switched,
        "llm_history_turns": len(model_history),
        "query_resolution": resolution,
        "chunks_created": len(chunks),
        "chunks_retrieved": len(retrieved_chunks),
        "performance": {
            "timings_ms": timings,
            "ollama": ollama_metrics,
            "documents": len(documents),
            "document_characters": sum(len(str(document.get("content") or "")) for document in documents),
            "system_prompt_characters": len(messages[0].get("content", "")) if messages else 0,
            "retrieved_context_characters": len(retrieved_context),
            "message_characters": sum(len(str(message.get("content", ""))) for message in messages),
            "latency_breakdown": dict(sorted(timings.items(), key=lambda item: item[1], reverse=True)),
        },
        "sources": serialized_sources,
        "retrieval": {
            "top_k": requested_top_k,
            "query_type": first_chunk.get("query_type"),
            "candidate_pool_size": first_chunk.get("candidate_pool_size"),
            "post_filter_pool_size": first_chunk.get("post_filter_pool_size"),
        },
    }


# ============================================================
# ASK
# ============================================================


# ============================================================
# UPLOADED DOCUMENT CONTEXT
# ============================================================

# Runtime document store. The document bytes are not kept after extraction;
# only normalized text is retained. This is intentionally in-memory for the
# first document-RAG phase and can later be replaced by persistent storage.
UPLOADED_DOCUMENTS: Dict[str, Dict[str, Any]] = {}

ACTIVE_DOCUMENT_CONTEXT: Dict[str, List[str]] = {}


def set_active_documents(
    chat_id: Optional[str],
    document_ids: List[str],
) -> None:
    if not chat_id:
        return

    valid_ids = [
        document_id
        for document_id in document_ids
        if document_id in UPLOADED_DOCUMENTS
    ]

    ACTIVE_DOCUMENT_CONTEXT[chat_id] = valid_ids


def get_active_document_ids(
    chat_id: Optional[str],
) -> List[str]:
    if not chat_id:
        return []

    return list(
        ACTIVE_DOCUMENT_CONTEXT.get(chat_id, [])
    )


def get_documents_for_chat(
    chat_id: Optional[str],
    document_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    ids = (
        document_ids
        if document_ids is not None
        else get_active_document_ids(chat_id)
    )

    documents = []

    for document_id in ids:
        document = UPLOADED_DOCUMENTS.get(document_id)

        if not document:
            continue

        documents.append(document)

    return documents


# ============================================================
# CONVERSATION-AWARE QUERY RESOLUTION
# ============================================================
_CONVERSATIONAL_PATTERNS = (
    r"\bwhat about\b",
    r"\bhow about\b",
    r"\bwhat does (?:it|this|that)\b",
    r"\bhow does (?:it|this|that)\b",
    r"\bhow is (?:it|this|that)\b",
    r"\bwhere is (?:it|this|that)\b",
    r"\bwhere are (?:it|this|that)\b",
    r"\bwhy does (?:it|this|that)\b",
    r"\b(?:its|their)\s+(?:tests?|implementation|usage|code|files?|architecture)\b",
    r"\bthe implementation\b",
    r"\bthe tests?\b",
)

def _extract_conversation_terms(history: List[Dict[str, str]]) -> List[str]:
    terms=[]; seen=set()
    for msg in reversed(clean_history(history, max_messages=8)):
        if msg.get('role')!='user': continue
        content=str(msg.get('content') or '')
        for groups in re.findall(r'`([^`]+)`|(?<![\w])([A-Z][A-Za-z0-9_]{2,})|(?<![\w])([A-Za-z_][A-Za-z0-9_]*\(\))', content):
            candidate=next((g for g in groups if g), '').strip('`.,:;!?()[]{}')
            if len(candidate)<3 or candidate.lower() in seen: continue
            seen.add(candidate.lower()); terms.append(candidate)
            if len(terms)>=6: return terms
    return terms

def resolve_conversational_query(question: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    original=question.strip(); recent=clean_history(history, max_messages=8)
    base={"original_query":original,"resolved_query":original,"was_resolved":False,"resolution_type":"none","referenced_terms":[]}
    if not recent: return base
    low=original.lower(); followup=any(re.search(p,low) for p in _CONVERSATIONAL_PATTERNS)
    short=len(re.findall(r'\w+',low))<=5
    if not followup and not short: return base
    previous=next((str(m.get('content') or '').strip() for m in reversed(recent) if m.get('role')=='user' and str(m.get('content') or '').strip()!=original),'')
    terms=_extract_conversation_terms(recent)
    parts=[original]
    if terms: parts.append('Conversation entities: '+', '.join(terms))
    if previous: parts.append('Previous user question: '+previous)
    return {"original_query":original,"resolved_query":"\n".join(parts),"was_resolved":True,"resolution_type":"reference_followup" if followup else "short_followup","referenced_terms":terms}

ACTIVE_GITHUB_CONTEXT: Dict[str, Dict[str, Optional[str]]] = {}
ACTIVE_SOURCE_CONTEXT: Dict[str, str] = {}

def get_active_source_scope(chat_id: Optional[str]) -> Optional[str]:
    return ACTIVE_SOURCE_CONTEXT.get(chat_id) if chat_id else None

def set_active_source_scope(chat_id: Optional[str], scope: str) -> None:
    if chat_id:
        ACTIVE_SOURCE_CONTEXT[chat_id] = scope
def get_active_github_context(chat_id: Optional[str]):
    return ACTIVE_GITHUB_CONTEXT.get(chat_id) if chat_id else None
def set_active_github_context(chat_id: Optional[str], reference: Dict[str, Optional[str]]):
    if chat_id and reference.get('repository_url'):
        ACTIVE_GITHUB_CONTEXT[chat_id]=dict(reference)


@app.get("/document-cache-stats")
def document_cache_stats() -> Dict[str, Any]:
    return DocumentCache.stats()


@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    chat_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract an uploaded document and attach it to the current chat.

    The endpoint intentionally stores normalized text rather than raw bytes.
    Uploaded documents are later processed by the independent
    UploadedDocumentChunker + DocumentRetriever pipeline.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    try:
        raw_bytes = await file.read()

        document = DocumentCache.get_or_extract(
            filename=file.filename,
            raw_bytes=raw_bytes,
            content_type=file.content_type,
            extractor=DocumentService.extract,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    document_id = document["document_id"]

    UPLOADED_DOCUMENTS[document_id] = document

    if chat_id:
        current_ids = get_active_document_ids(chat_id)

        if document_id not in current_ids:
            current_ids.append(document_id)

        set_active_documents(
            chat_id,
            current_ids,
        )

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "pages": document.get("pages"),
        "characters": document["characters"],
        "size_bytes": document["size_bytes"],
        "context_origin": "upload",
        "active_document_ids": (
            get_active_document_ids(chat_id)
            if chat_id
            else [document_id]
        ),
    }


@app.post("/ask")
def ask_ai(
    request: AskRequest,
    stream: bool = Query(False, description="Stream Qwen output when true."),
) -> Any:

    request_start = time.perf_counter()
    timings: Dict[str, float] = {}

    question = request.question.strip()

    # ------------------------------------------------------------
    # GITHUB CONTEXT PRIORITY
    # ------------------------------------------------------------
    # A URL written in the current message always wins. This allows
    # repository switching inside the same chat.
    #
    # Priority:
    #   1. current-message GitHub URL
    #   2. explicit request.github_url
    #   3. active repository for this chat
    #   4. no GitHub context
    # ------------------------------------------------------------

    active_github = get_active_github_context(request.chat_id)
    current_reference = extract_github_reference(question)

    # A new repository in the current message starts a new repository
    # context for the LLM. The persisted chat history is NOT deleted;
    # it is simply not used as model history for this request.
    previous_repo = str(
        (active_github or {}).get("repository_url") or ""
    ).rstrip("/").lower()
    current_repo = str(
        current_reference.get("repository_url") or ""
    ).rstrip("/").lower()

    repository_switched = bool(
        current_repo
        and current_repo != previous_repo
    )

    if current_reference.get("repository_url"):
        github_reference = current_reference
        github_url = current_reference["repository_url"]

        question = remove_github_reference(
            question,
            current_reference,
        )

        set_active_github_context(
            request.chat_id,
            current_reference,
        )

    elif request.github_url:
        explicit_reference = extract_github_reference(
            request.github_url.strip()
        )

        if explicit_reference.get("repository_url"):
            explicit_repo = str(
                explicit_reference.get("repository_url") or ""
            ).rstrip("/").lower()

            repository_switched = bool(
                explicit_repo
                and explicit_repo != previous_repo
            )

            github_reference = explicit_reference
            github_url = explicit_reference["repository_url"]

            set_active_github_context(
                request.chat_id,
                explicit_reference,
            )
        else:
            github_reference = {}
            github_url = ""

    elif active_github and active_github.get("repository_url"):
        github_url = active_github["repository_url"]

        github_reference = {
            "original_url": active_github.get("repository_url"),
            "repository_url": active_github.get("repository_url"),
            "owner": active_github.get("owner"),
            "repo": active_github.get("repo"),
            "resource_type": active_github.get("resource_type"),
            "resource_path": active_github.get("resource_path"),
        }

    else:
        github_reference = {}
        github_url = ""

    # ------------------------------------------------------------
    # CONVERSATION-AWARE QUERY RESOLUTION
    # ------------------------------------------------------------
    # Only the retrieval query is enriched. The original question
    # remains unchanged for the final LLM response.
    # ------------------------------------------------------------

    resolution = resolve_conversational_query(
        question=question,
        history=request.history,
    )
    retrieval_query = resolution["resolved_query"]

    # Important distinction:
    # - request.history is still used by the resolver.
    # - model_history controls what Qwen sees.
    # When switching repositories, old repository turns must not contaminate
    # the answer. The active repository's retrieved evidence becomes the
    # authoritative context.
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    # ========================================================
    # 0. BUILD / ACCEPT RESEARCH CONTEXT
    # ========================================================
    #
    # Preferred flow:
    #
    #   frontend -> github_url + question
    #             -> GitHubContentService
    #             -> query-aware repository context
    #
    # Backward-compatible flow:
    #
    #   frontend -> pre-built context + question
    #
    # This keeps GitHub acquisition out of Streamlit and keeps the
    # existing RAG pipeline unchanged after context creation.
    # ========================================================

    # github_url has already been normalized above. Do not overwrite the
    # automatically detected value here.
    context_start = time.perf_counter()

    documents: List[Dict[str, Any]] = []

    if request.document_ids is not None:
        set_active_documents(request.chat_id, request.document_ids)

    active_document_ids = get_active_document_ids(request.chat_id)
    uploaded_documents = get_documents_for_chat(request.chat_id, request.document_ids)

    route = ContextRouter.route(
        question=question,
        history=request.history,
        github_reference=github_reference,
        has_documents=bool(uploaded_documents),
        resolved_query=retrieval_query,
    )
    context_scope = route["scope"]
    previous_scope = get_active_source_scope(request.chat_id)
    source_switched = bool(previous_scope and previous_scope != context_scope and context_scope != "general")
    set_active_source_scope(request.chat_id, context_scope)

    # Never carry retrieval terms from the previous source into a newly
    # selected repository/document. Conversation-aware resolution is only
    # useful once the source is stable.
    if repository_switched or source_switched:
        retrieval_query = question

    model_history = [] if repository_switched or source_switched else request.history

    if context_scope == "github":
        if not github_url:
            raise HTTPException(status_code=400, detail="No active GitHub repository is available for this question.")
        documents = [
            {
                **document,
                "source": "github",
                "source_type": "github_repository",
            }
            for document in build_github_documents(
                github_url=github_url,
                question=retrieval_query,
                branch=request.branch,
            )
        ]
        context_origin = "github"

    elif context_scope == "document":
        if not uploaded_documents:
            raise HTTPException(status_code=400, detail="No active uploaded document is available for this question.")
        documents = [
            {
                **document,
                "source": "upload",
                "source_type": "uploaded_document",
            }
            for document in uploaded_documents
        ]
        context_origin = "upload"

    elif context_scope == "hybrid":
        if not github_url or not uploaded_documents:
            raise HTTPException(status_code=400, detail="A hybrid question requires both an active GitHub repository and an uploaded document.")
        github_documents = build_github_documents(
            github_url=github_url,
            question=retrieval_query,
            branch=request.branch,
        )

        upload_documents = [
            {
                **document,
                "source": "upload",
                "source_type": "uploaded_document",
            }
            for document in uploaded_documents
        ]

        github_documents = [
            {
                **document,
                "source": "github",
                "source_type": "github_repository",
            }
            for document in github_documents
        ]

        documents = upload_documents + github_documents
        context_origin = "github+upload"

    elif request.context and request.context.strip():
        context = request.context.strip()
        if len(context) > MAX_CONTEXT_CHARACTERS:
            raise HTTPException(status_code=413, detail=f"Research context is too large. Maximum allowed size: {MAX_CONTEXT_CHARACTERS:,} characters.")
        documents = [{
            "content": context,
            "path": request.source_path or "research/repository_context.md",
            "category": request.source_category or "documentation",
            "source": "provided_context",
            "source_type": "provided_context",
        }]
        context_origin = "provided_context"
        context_scope = "provided_context"

    else:
        documents = []
        context_origin = "general_chat"
        context_scope = "general"

    timings["github_or_context_ms"] = _elapsed_ms(context_start)

    if context_origin != "general_chat" and not documents:
        raise HTTPException(
            status_code=502 if github_url else 400,
            detail=(
                "GitHubContentService returned no usable documents."
                if github_url
                else "Research context must not be empty."
            ),
        )

    requested_top_k = (
        request.top_k
        if request.top_k is not None
        else DEFAULT_TOP_K
    )

    requested_top_k = max(
        1,
        min(requested_top_k, MAX_TOP_K),
    )

    # ========================================================
    # 1-4. CONTEXT PREPARATION / RETRIEVAL / PROMPT
    # ========================================================

    chunks: List[Dict[str, Any]] = []
    retrieved_chunks: List[Dict[str, Any]] = []
    retrieved_context = ""
    active_retriever_name = ""
    active_chunker_name = ""

    if context_origin == "general_chat":
        prompt_start = _profile_start()
        system_prompt = """
You are a helpful AI assistant.

Answer the user's question clearly and accurately.
Use the conversation history when relevant. Do not invent facts,
sources, files, repository details, or citations.
For technical questions, explain concepts accurately and preserve
technical terminology. Answer directly and appropriately concisely.
""".strip()
        messages = build_messages(
            question=question,
            history=model_history,
            system_prompt=system_prompt,
        )
        timings["prompt_build_ms"] = _profile_ms(prompt_start)
    else:
        # IMPORTANT: GitHub and uploaded documents are retrieved independently.
        # This prevents a document chunk from competing directly with repository
        # chunks and makes the evidence provenance explicit.
        chunk_start = _profile_start()
        try:
            if context_scope == "github":
                chunks = document_chunker.chunk_documents(documents)
                active_chunker_name = "DocumentChunker"
            elif context_scope == "document":
                chunks = uploaded_document_chunker.chunk_documents(documents)
                active_chunker_name = "UploadedDocumentChunker"
            elif context_scope == "hybrid":
                github_documents = [d for d in documents if d.get("source") == "github"]
                upload_documents = [d for d in documents if d.get("source") == "upload"]
                github_chunks = document_chunker.chunk_documents(github_documents)
                upload_chunks = uploaded_document_chunker.chunk_documents(upload_documents)
                chunks = github_chunks + upload_chunks
                active_chunker_name = "DocumentChunker + UploadedDocumentChunker"
            else:
                chunks = document_chunker.chunk_documents(documents)
                active_chunker_name = "DocumentChunker"

            timings["chunking_ms"] = _profile_ms(chunk_start)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to chunk research documents.",
                    "error": str(exc),
                    "context_origin": context_origin,
                    "document_count": len(documents),
                    "document_names": [
                        document.get("filename")
                        or document.get("path")
                        or document.get("source")
                        for document in documents
                    ],
                },
            )

        if not chunks:
            raise HTTPException(status_code=400, detail="No usable research chunks were created.")

        retrieval_start = _profile_start()
        try:
            if context_scope == "github":
                retrieved_chunks = hybrid_retriever.retrieve(
                    question=retrieval_query,
                    chunks=chunks,
                    top_k=min(requested_top_k, len(chunks)),
                )
                active_retriever_name = "HybridRetriever"

            elif context_scope == "document":
                retrieved_chunks = document_retriever.retrieve(
                    question=retrieval_query,
                    chunks=chunks,
                    top_k=min(requested_top_k, len(chunks)),
                )
                active_retriever_name = "DocumentRetriever"

            elif context_scope == "hybrid":
                github_chunks = [c for c in chunks if c.get("source") == "github"]
                upload_chunks = [c for c in chunks if c.get("source") == "upload"]

                # Retrieve independently, then merge. Each source gets its own
                # relevance/MMR decision before the final evidence is assembled.
                github_results = hybrid_retriever.retrieve(
                    question=retrieval_query,
                    chunks=github_chunks,
                    top_k=min(requested_top_k, len(github_chunks)) if github_chunks else 0,
                ) if github_chunks else []
                upload_results = document_retriever.retrieve(
                    question=retrieval_query,
                    chunks=upload_chunks,
                    top_k=min(requested_top_k, len(upload_chunks)) if upload_chunks else 0,
                ) if upload_chunks else []

                retrieved_chunks = (github_results + upload_results)[:requested_top_k]
                active_retriever_name = "HybridRetriever + DocumentRetriever"

            else:
                retrieved_chunks = hybrid_retriever.retrieve(
                    question=retrieval_query,
                    chunks=chunks,
                    top_k=min(requested_top_k, len(chunks)),
                )
                active_retriever_name = "HybridRetriever"

            timings["retrieval_ms"] = _profile_ms(retrieval_start)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Document/GitHub retrieval failed.",
                    "error": str(exc),
                    "context_origin": context_origin,
                    "chunk_count": len(chunks),
                    "retriever": active_retriever_name,
                },
            )

        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No sufficiently relevant evidence was found in the active source.",
            )

        context_format_start = _profile_start()
        retrieved_context = format_retrieved_context(retrieved_chunks)
        timings["context_formatting_ms"] = _profile_ms(context_format_start)
        if not retrieved_context.strip():
            raise HTTPException(status_code=500, detail="Retriever returned chunks but their content was empty.")

        prompt_start = time.perf_counter()
        system_prompt = build_system_prompt(retrieved_context)

        if context_scope == "document":
            system_prompt = (
                "IMPORTANT SOURCE RULE: This request is document-only. Use "
                "only the retrieved uploaded-document evidence. Ignore any "
                "active GitHub repository unless the user explicitly asks "
                "for a comparison.\n\n" + system_prompt
            )
        elif context_scope == "github":
            system_prompt = (
                "IMPORTANT SOURCE RULE: This request is GitHub-only. Use "
                "only the active repository and retrieved GitHub evidence. "
                "Ignore uploaded documents.\n\n" + system_prompt
            )
        elif context_scope == "hybrid":
            system_prompt = (
                "IMPORTANT SOURCE RULE: This request is hybrid. Use both "
                "retrieved GitHub and uploaded-document evidence, keeping "
                "claims tied to the correct source.\n\n" + system_prompt
            )

        if repository_switched:
            system_prompt = (
                "IMPORTANT CURRENT REPOSITORY CONTEXT:\n"
                "A new GitHub repository was selected in the current "
                "message. Answer this request using ONLY the current "
                "repository and the retrieved evidence below. Do not use "
                "facts, files, claims, or conclusions from a previously "
                "discussed repository. Do not say that the previous "
                "repository lacks information about this question. "
                "If the current repository evidence is insufficient, "
                "state that directly.\n\n"
                + system_prompt
            )
        elif source_switched:
            system_prompt = (
                "IMPORTANT SOURCE CONTEXT:\n"
                "The active source changed for this request. Use ONLY the "
                "currently retrieved source evidence. Do not carry facts, "
                "claims, files, or conclusions from the previous source into "
                "this answer. If the current source evidence is insufficient, "
                "state that directly.\n\n"
                + system_prompt
            )

        messages = build_messages(
            question=question,
            history=model_history,
            system_prompt=system_prompt,
        )
        timings["prompt_build_ms"] = _elapsed_ms(prompt_start)

    # ========================================================
    # 5. QWEN / OLLAMA
    # ========================================================

    llm_start = _profile_start()

    try:
        if stream:
            ollama_stream = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={
                    "num_ctx": OLLAMA_NUM_CTX,
                    "temperature": OLLAMA_TEMPERATURE,
                    "num_predict": OLLAMA_NUM_PREDICT,
                },
                keep_alive=OLLAMA_KEEP_ALIVE,
                stream=True,
            )

            def event_stream():
                """Stream tokens, then send the unchanged RAG metadata as one final event."""
                answer_parts: List[str] = []
                last_response: Any = None

                try:
                    for chunk in ollama_stream:
                        last_response = chunk
                        content = getattr(getattr(chunk, "message", None), "content", None)
                        if content:
                            text = str(content)
                            answer_parts.append(text)
                            yield (
                                "data: "
                                + json.dumps({"type": "token", "content": text}, ensure_ascii=False)
                                + "\n\n"
                            )

                    answer = "".join(answer_parts).strip()
                    timings["qwen_wall_ms"] = _profile_ms(llm_start)

                    if not answer:
                        raise RuntimeError("Qwen returned an empty answer.")

                    payload = _build_rag_response_payload(
                        question=question,
                        answer=answer,
                        response=last_response,
                        timings=timings,
                        request_start=request_start,
                        model_history=model_history,
                        context_origin=context_origin,
                        context_scope=context_scope,
                        route=route,
                        documents=documents,
                        uploaded_documents=uploaded_documents,
                        chunks=chunks,
                        retrieved_chunks=retrieved_chunks,
                        retrieved_context=retrieved_context,
                        active_retriever_name=active_retriever_name,
                        active_chunker_name=active_chunker_name,
                        github_reference=github_reference,
                        github_url=github_url,
                        repository_switched=repository_switched,
                        source_switched=source_switched,
                        active_document_ids=active_document_ids,
                        resolution=resolution,
                        requested_top_k=requested_top_k,
                        messages=messages,
                    )

                    yield (
                        "data: "
                        + json.dumps({"type": "done", "data": payload}, ensure_ascii=False)
                        + "\n\n"
                    )

                except Exception as exc:
                    yield (
                        "data: "
                        + json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False)
                        + "\n\n"
                    )

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "num_ctx": OLLAMA_NUM_CTX,
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
            keep_alive=OLLAMA_KEEP_ALIVE,
            stream=False,
        )

        timings["qwen_wall_ms"] = _profile_ms(llm_start)

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Qwen/Ollama request failed.",
                "model": OLLAMA_MODEL,
                "ollama_host": OLLAMA_HOST,
                "error": str(exc),
            },
        )

    # ========================================================
    # 6. EXTRACT ANSWER
    # ========================================================

    try:
        answer = str(
            response.message.content
        ).strip()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Unexpected Ollama response format.",
                "error": str(exc),
            },
        )

    if not answer:
        raise HTTPException(
            status_code=500,
            detail="Qwen returned an empty answer.",
        )

    # ========================================================
    # 7. RETURN ANSWER + EVIDENCE
    # ========================================================

    return _build_rag_response_payload(
        question=question,
        answer=answer,
        response=response,
        timings=timings,
        request_start=request_start,
        model_history=model_history,
        context_origin=context_origin,
        context_scope=context_scope,
        route=route,
        documents=documents,
        uploaded_documents=uploaded_documents,
        chunks=chunks,
        retrieved_chunks=retrieved_chunks,
        retrieved_context=retrieved_context,
        active_retriever_name=active_retriever_name,
        active_chunker_name=active_chunker_name,
        github_reference=github_reference,
        github_url=github_url,
        repository_switched=repository_switched,
        source_switched=source_switched,
        active_document_ids=active_document_ids,
        resolution=resolution,
        requested_top_k=requested_top_k,
        messages=messages,
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("BACKEND_PORT", "8000")),
        reload=True,
    )
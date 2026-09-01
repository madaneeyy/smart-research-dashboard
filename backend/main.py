import math
import re
import json

import os
import time
from typing import Any, Dict, List, Optional, OrderedDict

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
from backend.routes.research import router as research_router
from backend.services.workspace_document_service import (
    create_workspace_document,
    get_workspace_document,
    list_workspace_documents,
    delete_workspace_document
)
from backend.services.document_storage_service import (
    build_document_storage_path,
    upload_document_file,
    create_document,
    get_document,
)
from backend.services.document_chunk_service import (
    create_document_chunks,
    get_document_chunks,
)
from backend.routes.chat import router as chat_router
from backend.services.chat_service import get_chat, get_chat_sources
from backend.services.source_service import list_sources
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
from src.services.document_rag.evidence_engine import EvidenceEngine
from src.services.document_rag.evidence_validator import EvidenceValidator
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

evidence_engine = EvidenceEngine(
    retriever=document_retriever,
    overview_per_source=2,
    analysis_per_source=2,
    max_final_evidence=12,
)
evidence_validator = EvidenceValidator()

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
app.include_router(chat_router)
app.include_router(research_router)
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

    github_urls: Optional[List[str]] = Field(
        default=None,
        description="GitHub repository URLs selected for this chat.",
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
    Build query-focused GitHub documents.

    The selected GitHub source is a retrieval target, not just a UI hint.
    If the query-specific selector returns nothing, retry with a broad
    repository-overview query so valid repositories do not become
    "no usable documents" merely because the filename/path vocabulary
    doesn't match the question lexically.
    """
    normalized_url = str(github_url or "").strip()
    if not normalized_url:
        return []

    builder = getattr(
        GitHubContentService,
        "build_documents_for_query",
        None,
    )

    if not callable(builder):
        raise HTTPException(
            status_code=500,
            detail=(
                "GitHubContentService.build_documents_for_query() "
                "is not available."
            ),
        )

    attempts = [
        str(question or "").strip(),
        "repository overview project structure main components",
    ]

    last_error: Optional[Exception] = None

    for attempt_index, query in enumerate(attempts):
        if not query:
            continue

        try:
            documents = builder(
                github_url=normalized_url,
                query=query,
                branch=branch,
            )

            if isinstance(documents, list):
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
                            "source": "github",
                            "github_url": normalized_url,
                            "query": query,
                            "category": str(
                                document.get("category")
                                or "source"
                            ),
                        }
                    )

                if cleaned:
                    print(
                        f"GitHub documents found using "
                        f"{'original' if attempt_index == 0 else 'fallback'} query: "
                        f"{len(cleaned)}"
                    )
                    return cleaned

        except ValueError as exc:
            last_error = exc
            break
        except Exception as exc:
            last_error = exc
            print(
                "GitHub query-aware retrieval warning:",
                str(exc),
            )

    # Last compatibility fallback. This is intentionally only reached after
    # both query-aware attempts failed.
    legacy_builder = getattr(
        GitHubContentService,
        "build_context_for_query",
        None,
    )

    if callable(legacy_builder):
        try:
            context = str(
                legacy_builder(
                    github_url=normalized_url,
                    query=str(question or "").strip()
                    or "repository overview",
                    branch=branch,
                )
                or ""
            ).strip()

            if context:
                return [
                    {
                        "content": context,
                        "path": "repository-context.md",
                        "source": "github",
                        "github_url": normalized_url,
                        "query": question,
                        "category": "documentation",
                    }
                ]
        except Exception as exc:
            last_error = exc
            print(
                "GitHub legacy context fallback warning:",
                str(exc),
            )

    if isinstance(last_error, ValueError):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid GitHub repository request.",
                "error": str(last_error),
            },
        )

    return []

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
    evidence_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Format evidence while preserving explicit source provenance."""
    if not chunks:
        return ""

    groups: "OrderedDict[tuple[str, str], List[Dict[str, Any]]]" = OrderedDict()
    for chunk in chunks:
        st = str(chunk.get("source_type") or chunk.get("source") or "unknown").strip().lower()
        if st in {"github", "github_repository"}:
            key = ("github_repository", str(chunk.get("repository") or chunk.get("github_url") or "GitHub repository").strip())
        elif st in {"arxiv", "arxiv_paper"}:
            key = ("arxiv", str(chunk.get("arxiv_id") or chunk.get("document_id") or chunk.get("filename") or "arXiv paper").strip())
        elif st in {"upload", "uploaded_document", "document"}:
            key = ("uploaded_document", str(chunk.get("document_id") or chunk.get("filename") or chunk.get("path") or "Uploaded document").strip())
        else:
            key = (st or "unknown", str(chunk.get("path") or chunk.get("filename") or "Unknown source").strip())
        groups.setdefault(key, []).append(chunk)

    blocks: List[str] = []
    for idx, ((kind, name), source_chunks) in enumerate(groups.items(), start=1):
        first = source_chunks[0]
        if kind == "github_repository":
            source_kind = "GitHub repository"
            source_name = str(first.get("repository") or first.get("github_url") or name).strip()
        elif kind == "arxiv":
            source_kind = "arXiv paper"
            source_name = str(first.get("arxiv_title") or first.get("filename") or first.get("path") or name).strip()
        elif kind == "uploaded_document":
            source_kind = "Uploaded document"
            source_name = str(first.get("filename") or first.get("path") or name).strip()
        else:
            source_kind = str(first.get("source_type") or first.get("source") or kind)
            source_name = str(first.get("filename") or first.get("path") or name).strip()

        lines = [
            f"===== SOURCE {idx} =====",
            f"Source type: {source_kind}",
            f"Source name: {source_name}",
            f"Retrieved evidence items: {len(source_chunks)}",
            "",
            "EVIDENCE:",
        ]
        n = 0
        for chunk in source_chunks:
            content = get_chunk_content(chunk)
            if not content:
                continue
            n += 1
            lines += ["", f"Evidence {n}"]
            path = str(chunk.get("path") or chunk.get("filename") or "Unknown").strip()
            lines.append(f"File/path: {path}")
            section = get_chunk_section(chunk)
            if section:
                lines.append(f"Section: {section}")
            if chunk.get("page") is not None:
                lines.append(f"Page: {chunk.get('page')}")
            if chunk.get("chunk_index") is not None:
                lines.append(f"Chunk index: {chunk.get('chunk_index')}")
            lines += ["Content:", content]
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_system_prompt(
    retrieved_context: str,
    evidence_result: Optional[Dict[str, Any]] = None,
    question: str = "",
) -> str:
    """
    Single general-purpose prompt for every grounded question.

    Query classification is used by retrieval, not by answer generation.
    """
    source_count = retrieved_context.count("===== SOURCE ")
    coverage_complete = bool(evidence_result and evidence_result.get("coverage_complete", False))

    return f"""
You are a research assistant answering the user's question.

USER QUESTION:
{question.strip()}

You have been given retrieved evidence from the user's selected sources.
Read the evidence carefully and answer the question directly.

CORE INSTRUCTIONS:
- Use the retrieved evidence as the main factual basis of your answer.
- Read all supplied evidence before deciding what to say.
- A passage does not need to contain the exact wording of the question to be relevant.
- You may summarize, synthesize, explain, compare, and connect related information when the connection is supported by the evidence.
- When several passages contribute to the same answer, combine them into a coherent explanation.
- Keep source-specific facts associated with the correct source.
- Treat every SOURCE block as an independent source. "GitHub repository" means current implementation evidence; "Uploaded document" means uploaded document evidence; "arXiv paper" means literature/paper evidence from arXiv.
- For cross-source questions, explicitly connect evidence from both relevant sources and distinguish the paper/document description from the current implementation.
- Do not invent source-specific facts, numbers, methods, results, quotations, citations, authors, or relationships.
- Do not refuse simply because the evidence uses different terminology from the question.
- If the evidence answers only part of the question, answer that part and clearly state what is not established.
- If the retrieved evidence genuinely does not establish the requested information, say that the information is not established by the retrieved evidence.
- Never say that there is "no evidence" when evidence passages are supplied.
- If sources disagree, explain the disagreement rather than silently choosing one.
- General knowledge may be used for brief explanations of terms or concepts, but do not use it to invent missing source-specific facts.
- Prefer a useful, natural answer over a discussion of the retrieval process.

SOURCE ORGANIZATION:
There are {source_count} selected source(s).
Coverage complete: {coverage_complete}

Each SOURCE block below is independent. Preserve those boundaries.

RETRIEVED EVIDENCE:
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
    unavailable_document_ids: List[str],
    evidence_result: Dict[str, Any],
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
    if evidence_result:
        print(f"Evidence query reason: {evidence_result.get('query_reason')}")
    print(f"Documents acquired: {len(documents)}")
    print(f"Uploaded documents: {len(uploaded_documents)}")
    document_chars = sum(len(str(document.get("content") or "")) for document in documents)
    print(f"Document chars: {document_chars:,}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunks retrieved: {len(retrieved_chunks)}")
    if evidence_result:
        print(f"Evidence query type: {evidence_result.get('query_type')}")
        print(f"Evidence strategy: {evidence_result.get('strategy')}")
        print(f"Selected documents: {evidence_result.get('selected_document_count', 0)}")
        print(f"Documents with evidence: {evidence_result.get('documents_with_evidence', 0)}")
        print(f"Coverage complete: {evidence_result.get('coverage_complete', False)}")
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
        "unavailable_document_ids": unavailable_document_ids,
        "source_unavailable": bool(unavailable_document_ids),
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
        "evidence": evidence_result,
        "evidence_validation": evidence_result.get("validation") if evidence_result else None,
        "retrieval": {
            "top_k": requested_top_k,
            "query_type": (evidence_result.get("query_type") if evidence_result else first_chunk.get("query_type")),
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
        str(document_id)
        for document_id in document_ids
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
    """Return documents from the legacy in-memory chat-upload store."""
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


def get_persistent_document_chunks(
    document_ids: List[str],
    workspace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load persistent document chunks and preserve arXiv provenance."""
    chunks: List[Dict[str, Any]] = []

    arxiv_by_document_id: Dict[str, Dict[str, Any]] = {}

    if workspace_id:
        try:
            for source in list_sources(str(workspace_id)) or []:
                if _normalize_source_type(source.get("source_type")) not in {"arxiv", "arxiv_paper"}:
                    continue
                metadata = source.get("metadata") or {}
                if not isinstance(metadata, dict):
                    continue
                source_document_id = str(metadata.get("document_id") or "").strip()
                if source_document_id:
                    arxiv_by_document_id[source_document_id] = source
        except Exception as exc:
            print(f"arXiv provenance lookup warning: {exc}")

    for document_id in document_ids:
        document_id = str(document_id)
        persistent_chunks = get_document_chunks(document_id)
        arxiv_source = arxiv_by_document_id.get(document_id)

        if arxiv_source:
            metadata = arxiv_source.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}

            for chunk in persistent_chunks:
                chunks.append({
                    **chunk,
                    "source": "arxiv",
                    "source_type": "arxiv",
                    "filename": arxiv_source.get("title") or chunk.get("filename") or f"{document_id}.pdf",
                    "path": metadata.get("canonical_url") or arxiv_source.get("url") or f"arxiv:{metadata.get('arxiv_id') or document_id}",
                    "arxiv_id": metadata.get("arxiv_id"),
                    "arxiv_title": arxiv_source.get("title"),
                    "arxiv_url": metadata.get("canonical_url") or arxiv_source.get("url"),
                })
        else:
            for chunk in persistent_chunks:
                chunks.append({
                    **chunk,
                    "source": "upload",
                    "source_type": "uploaded_document",
                })

    return chunks


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
    low = original.lower()
    followup = any(
        re.search(pattern, low)
        for pattern in _CONVERSATIONAL_PATTERNS
    )

    short = len(re.findall(r"\w+", low)) <= 5

    # A short question is not automatically a follow-up.
    # It must contain explicit reference/continuity language.
    short_reference = bool(
        short
        and re.search(
            r"\b(?:it|this|that|these|those|the above|the previous|"
            r"same|another|also|what about|how about)\b",
            low,
        )
    )

    if not followup and not short_reference:
        return base
    previous=next((str(m.get('content') or '').strip() for m in reversed(recent) if m.get('role')=='user' and str(m.get('content') or '').strip()!=original),'')
    terms=_extract_conversation_terms(recent)
    parts=[original]
    if terms: parts.append('Conversation entities: '+', '.join(terms))
    if previous: parts.append('Previous user question: '+previous)
    return {"original_query":original,"resolved_query":"\n".join(parts),"was_resolved":True,"resolution_type":"reference_followup" if followup else "short_followup","referenced_terms":terms}

def question_explicitly_references_missing_document(
    question: str,
    history: List[Dict[str, str]],
) -> bool:
    """Return True only when the user explicitly refers to a missing document."""
    text = question.strip().lower()
    explicit_patterns = (
        r"\bthe document\b",
        r"\bthat document\b",
        r"\bthis document\b",
        r"\bthe file\b",
        r"\bthat file\b",
        r"\bthis file\b",
        r"\bthe report\b",
        r"\bthat report\b",
        r"\bthis report\b",
        r"\baccording to (?:the|this|that) (?:document|file|report)\b",
        r"\bwhat did (?:the|this|that) (?:document|file|report)\b",
        r"\bwhat (?:does|did) it say\b",
    )
    if any(re.search(pattern, text) for pattern in explicit_patterns):
        return True

    # Conversational follow-ups can refer to a missing source implicitly.
    # Require a short/reference-like question rather than treating all short
    # questions as source-specific.
    followup_patterns = (
        r"\bwhat about it\b",
        r"\bhow about it\b",
        r"\bwhat does it say\b",
        r"\bwhat did it say\b",
        r"\bwhat were (?:the|its) (?:findings|results|methods|methodology)\b",
    )
    return any(re.search(pattern, text) for pattern in followup_patterns)


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

def get_chat_github_urls(chat_id: Optional[str]) -> List[str]:
    """Resolve persisted GitHub repository URLs from this chat's sources."""
    if not chat_id:
        return []

    try:
        rows = get_chat_sources(chat_id)
    except Exception:
        return []

    urls: List[str] = []

    for row in rows or []:
        source_type = str(row.get("source_type") or "").strip().lower()

        if source_type not in {"github", "github_repository"}:
            continue

        source_id = str(row.get("source_id") or "").strip()

        if not source_id:
            continue

        reference = extract_github_reference(source_id)
        repository_url = reference.get("repository_url")

        if repository_url and repository_url not in urls:
            urls.append(repository_url)

    return urls



def get_unavailable_chat_document_ids(
    chat_id: Optional[str],
    active_workspace_document_ids: List[str],
) -> List[str]:
    """
    Return document IDs that are unavailable in the chat's workspace.

    This includes:
      1. document IDs explicitly requested by the current request that no
         longer belong to the workspace; and
      2. historical document chat_sources that used to belong to the workspace
         but were later removed.

    Keeping the current request in this calculation is important because the
    frontend may still have a stale selection after a workspace document is
    deleted.
    """
    if not chat_id:
        return []

    chat = get_chat(chat_id)
    if not chat:
        return []

    workspace_id = chat.get("workspace_id")
    if not workspace_id:
        return []

    try:
        current_workspace_documents = list_workspace_documents(
            str(workspace_id)
        )
    except Exception:
        # If availability cannot be verified, do not falsely claim deletion.
        return []

    current_ids = {
        str(item.get("document_id"))
        for item in current_workspace_documents
        if item.get("document_id")
    }

    active_ids = {
        str(item)
        for item in active_workspace_document_ids
        if item
    }

    unavailable = {
        document_id
        for document_id in active_ids
        if document_id not in current_ids
    }

    try:
        chat_sources = get_chat_sources(chat_id)
    except Exception:
        chat_sources = []

    for source in chat_sources:
        if str(source.get("source_type") or "").lower() not in {"document", "arxiv", "arxiv_paper"}:
            continue

        source_id = source.get("source_id")
        if not source_id:
            continue

        source_id = str(source_id)
        if source_id not in current_ids:
            unavailable.add(source_id)

    return list(unavailable)


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
@app.post("/workspaces/{workspace_id}/documents")
async def upload_workspace_document(
    workspace_id: str,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Persist a document once and create only a workspace association.

    Canonical model:
      documents       -> one row per content-addressed document
      document_chunks -> one canonical chunk set per document
      Storage         -> one canonical object per document
      workspace_documents -> one association per workspace

    A previously-created document with zero chunks is repaired before the
    workspace association is returned/created. This fixes documents that were
    created by an earlier interrupted upload pipeline.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    try:
        raw_bytes = await file.read()

        if not raw_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

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

    document_id = str(document["document_id"])

    # ----------------------------------------------------------
    # 1. Workspace-level idempotency.
    # ----------------------------------------------------------
    existing_workspace_document = get_workspace_document(
        document_id=document_id,
        workspace_id=workspace_id,
    )

    # ----------------------------------------------------------
    # 2. Load/reuse the canonical document.
    # ----------------------------------------------------------
    existing_document = get_document(document_id)
    existing_chunks = (
        get_document_chunks(document_id)
        if existing_document is not None
        else []
    )

    # If the document exists but its chunk set is missing, repair it.
    if not existing_chunks:
        try:
            chunks = uploaded_document_chunker.chunk_documents(
                [document]
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to chunk the document.",
                    "error": str(exc),
                    "document_id": document_id,
                    "filename": file.filename,
                },
            )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "No usable chunks were created from the document.",
                    "document_id": document_id,
                    "filename": file.filename,
                },
            )

        if existing_document is None:
            # Chunking is validated before we create the canonical document so
            # a parser failure does not leave another incomplete documents row.
            storage_path = build_document_storage_path(
                document_id=document_id,
                filename=document["filename"],
            )

            try:
                upload_document_file(
                    storage_path=storage_path,
                    file_bytes=raw_bytes,
                    content_type=file.content_type,
                )

                create_document(
                    document_id=document_id,
                    filename=document["filename"],
                    storage_path=storage_path,
                    content_type=file.content_type,
                    pages=document.get("pages"),
                    characters=document.get("characters"),
                    size_bytes=len(raw_bytes),
                )

                create_document_chunks(
                    document_id=document_id,
                    chunks=chunks,
                )

                existing_document = get_document(document_id)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Failed to persist the document.",
                        "error": str(exc),
                        "document_id": document_id,
                    },
                )
        else:
            # Existing canonical row, but an earlier upload never created the
            # chunk set. Repair it exactly once.
            try:
                create_document_chunks(
                    document_id=document_id,
                    chunks=chunks,
                )
                existing_chunks = get_document_chunks(document_id)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Failed to repair the document chunk set.",
                        "error": str(exc),
                        "document_id": document_id,
                    },
                )

    if existing_document is None:
        existing_document = get_document(document_id)

    if existing_workspace_document is not None:
        # Keep the legacy in-memory cache warm for any older code paths, but
        # the persistent document/chunks are authoritative.
        UPLOADED_DOCUMENTS[document_id] = document

        return {
            "workspace_document_id": existing_workspace_document["id"],
            "workspace_id": workspace_id,
            "document_id": document_id,
            "filename": existing_workspace_document.get(
                "filename",
                document["filename"],
            ),
            "content_type": existing_workspace_document.get(
                "content_type",
                file.content_type,
            ),
            "pages": existing_workspace_document.get(
                "pages",
                document.get("pages"),
            ),
            "characters": existing_workspace_document.get(
                "characters",
                document.get("characters"),
            ),
            "size_bytes": existing_workspace_document.get(
                "size_bytes",
                document.get("size_bytes"),
            ),
            "status": existing_workspace_document.get(
                "status",
                "ready",
            ),
            "context_origin": "workspace_upload",
            "already_exists": True,
            "reused_underlying_document": True,
        }

    UPLOADED_DOCUMENTS[document_id] = document

    # ----------------------------------------------------------
    # 3. Create only the workspace association.
    # ----------------------------------------------------------
    try:
        workspace_document = create_workspace_document(
            workspace_id=workspace_id,
            document_id=document_id,
            filename=(
                existing_document.get("filename", document["filename"])
                if existing_document
                else document["filename"]
            ),
            content_type=(
                existing_document.get("content_type", file.content_type)
                if existing_document
                else file.content_type
            ),
            pages=(
                existing_document.get("pages", document.get("pages"))
                if existing_document
                else document.get("pages")
            ),
            characters=(
                existing_document.get("characters", document.get("characters"))
                if existing_document
                else document.get("characters")
            ),
            size_bytes=(
                existing_document.get("size_bytes", document.get("size_bytes"))
                if existing_document
                else document.get("size_bytes")
            ),
            status="ready",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "The document was available but could not be "
                    "associated with the workspace."
                ),
                "error": str(exc),
                "document_id": document_id,
            },
        )

    return {
        "workspace_document_id": workspace_document["id"],
        "workspace_id": workspace_id,
        "document_id": document_id,
        "filename": workspace_document.get(
            "filename",
            document["filename"],
        ),
        "content_type": workspace_document.get(
            "content_type",
            file.content_type,
        ),
        "pages": workspace_document.get(
            "pages",
            document.get("pages"),
        ),
        "characters": workspace_document.get(
            "characters",
            document.get("characters"),
        ),
        "size_bytes": workspace_document.get(
            "size_bytes",
            document.get("size_bytes"),
        ),
        "status": workspace_document.get(
            "status",
            "ready",
        ),
        "context_origin": "workspace_upload",
        "already_exists": False,
        "reused_underlying_document": existing_document is not None,
    }


@app.get("/debug/document/{document_id}")
def debug_document(document_id: str) -> Dict[str, Any]:
    document = UPLOADED_DOCUMENTS.get(document_id)

    return {
        "document_id": document_id,
        "exists_in_uploaded_documents": document is not None,
        "filename": (
            document.get("filename")
            if document
            else None
        ),
        "characters": (
            document.get("characters")
            if document
            else None
        ),
    }

@app.get("/workspaces/{workspace_id}/documents")
def get_workspace_documents(
    workspace_id: str,
) -> List[Dict[str, Any]]:
    """
    Return all documents associated with a workspace.
    """

    try:
        return list_workspace_documents(workspace_id)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Could not load workspace documents.",
                "error": str(exc),
            },
        )



@app.get(
    "/workspaces/{workspace_id}/documents/{document_id}/preview"
)
def preview_workspace_document(
    workspace_id: str,
    document_id: str,
) -> Dict[str, Any]:
    """
    Return readable extracted content for a document that
    belongs to the requested workspace.

    The workspace association is checked first so a document
    cannot be previewed merely by knowing its document_id.
    """

    workspace_document = (
        get_workspace_document(
            document_id=document_id,
            workspace_id=workspace_id,
        )
    )

    if workspace_document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found in this workspace.",
        )

    chunks = get_document_chunks(
        document_id
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No readable content is available for this document.",
        )

    content_parts: list[str] = []

    for chunk in chunks:
        content = str(
            chunk.get("content")
            or ""
        ).strip()

        if not content:
            continue

        page = chunk.get("page")

        if page is not None:
            content_parts.append(
                f"[Page {page}]\n{content}"
            )
        else:
            content_parts.append(content)

    content = "\n\n".join(
        content_parts
    ).strip()

    if not content:
        raise HTTPException(
            status_code=404,
            detail="No readable content is available for this document.",
        )

    return {
        "document_id": document_id,
        "filename": workspace_document.get(
            "filename",
            "Untitled document",
        ),
        "content_type": workspace_document.get(
            "content_type"
        ),
        "pages": workspace_document.get(
            "pages"
        ),
        "characters": workspace_document.get(
            "characters"
        ),
        "content": content,
    }


@app.delete("/workspaces/{workspace_id}/documents/{document_id}")
def delete_workspace_document_endpoint(
    workspace_id: str,
    document_id: str,
) -> Dict[str, Any]:

    deleted = delete_workspace_document(
        workspace_id=workspace_id,
        document_id=document_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found in this workspace.",
        )

    # Also clear the legacy in-memory document store.
    UPLOADED_DOCUMENTS.pop(
        str(document_id),
        None,
    )

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "workspace_id": workspace_id,
    }

def _normalize_source_type(value: Any) -> str:
    return str(value or "").strip().lower()


def _resolve_chat_selected_sources(
    chat_id: Optional[str],
) -> tuple[List[str], List[str]]:
    """
    Resolve persistent chat_sources into canonical document IDs and GitHub URLs.

    This is the single source of truth for ChatPage selections.
    """
    document_ids: List[str] = []
    github_urls: List[str] = []

    if not chat_id:
        return document_ids, github_urls

    try:
        rows = get_chat_sources(chat_id) or []
    except Exception as exc:
        print(
            f"Chat source lookup warning for {chat_id}: {exc}"
        )
        return document_ids, github_urls

    for row in rows:
        source_type = _normalize_source_type(
            row.get("source_type")
        )
        source_id = str(
            row.get("source_id") or ""
        ).strip()

        if not source_id:
            continue

        if source_type in {"document", "arxiv", "arxiv_paper"}:
            if source_id not in document_ids:
                document_ids.append(source_id)
            continue

        if source_type in {
            "github",
            "github_repository",
        }:
            if source_id not in github_urls:
                github_urls.append(source_id)

    return document_ids, github_urls


@app.post("/ask")
def ask_ai(
    request: AskRequest,
    stream: bool = Query(False, description="Stream Qwen output when true."),
) -> Any:

    request_start = time.perf_counter()
    timings: Dict[str, float] = {}

    question = request.question.strip()

    # ------------------------------------------------------------
    # CHAT SOURCE OF TRUTH
    # ------------------------------------------------------------
    # Persisted chat_sources are authoritative. Explicit frontend values are
    # merged for compatibility with clients that send them on /ask.
    persisted_document_ids, persisted_github_urls = (
        _resolve_chat_selected_sources(request.chat_id)
    )

    effective_document_ids = list(
        dict.fromkeys(
            [
                *(
                    str(value).strip()
                    for value in (request.document_ids or [])
                    if str(value).strip()
                ),
                *persisted_document_ids,
            ]
        )
    )

    effective_github_urls = list(
        dict.fromkeys(
            [
                *(
                    str(value).strip()
                    for value in (request.github_urls or [])
                    if str(value).strip()
                ),
                *persisted_github_urls,
            ]
        )
    )

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
    persisted_chat_github_urls = get_chat_github_urls(request.chat_id)

    requested_github_urls: List[str] = []

    for raw_url in effective_github_urls:
        reference = extract_github_reference(str(raw_url or "").strip())
        repository_url = reference.get("repository_url")

        if repository_url and repository_url not in requested_github_urls:
            requested_github_urls.append(repository_url)

    if request.github_url:
        reference = extract_github_reference(request.github_url.strip())
        repository_url = reference.get("repository_url")

        if repository_url and repository_url not in requested_github_urls:
            requested_github_urls.append(repository_url)

    current_reference = extract_github_reference(question)
    current_repo_url = current_reference.get("repository_url")

    if current_repo_url:
        if current_repo_url not in requested_github_urls:
            requested_github_urls.insert(0, current_repo_url)

        question = remove_github_reference(
            question,
            current_reference,
        )

        set_active_github_context(
            request.chat_id,
            current_reference,
        )

    elif requested_github_urls:
        first_reference = extract_github_reference(
            requested_github_urls[0]
        )

        if first_reference.get("repository_url"):
            set_active_github_context(
                request.chat_id,
                first_reference,
            )

    elif active_github and active_github.get("repository_url"):
        repository_url = str(
            active_github["repository_url"]
        ).rstrip("/")

        if repository_url:
            requested_github_urls = [repository_url]

    elif persisted_chat_github_urls:
        requested_github_urls = list(
            persisted_chat_github_urls
        )

    # Include any GitHub sources persisted in chat_sources that were not
    # already present in the active-github compatibility state.
    for repository_url in effective_github_urls:
        if repository_url and repository_url not in requested_github_urls:
            requested_github_urls.append(repository_url)

        first_reference = extract_github_reference(
            requested_github_urls[0]
        )

        if first_reference.get("repository_url"):
            set_active_github_context(
                request.chat_id,
                first_reference,
            )

    previous_repo = str(
        (active_github or {}).get("repository_url") or ""
    ).rstrip("/").lower()

    repository_switched = bool(
        current_repo_url
        and current_repo_url.rstrip("/").lower() != previous_repo
    )

    github_url = (
        requested_github_urls[0]
        if requested_github_urls
        else ""
    )

    github_reference = (
        extract_github_reference(github_url)
        if github_url
        else {}
    )

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

    # Only enrich retrieval with conversation context for a genuine follow-up.
    # Standalone questions use ONLY the current question.
    is_conversational_followup = bool(
        resolution.get("was_resolved")
        and resolution.get("resolution_type")
        in {"reference_followup", "short_followup"}
    )
    retrieval_query = (
        resolution["resolved_query"]
        if is_conversational_followup
        else question
    )

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
    chunks: List[Dict[str, Any]] = []
    retrieved_chunks: List[Dict[str, Any]] = []
    retrieved_context = ""
    active_retriever_name = ""
    active_chunker_name = ""
    evidence_result: Dict[str, Any] = {}
    evidence_validation: Dict[str, Any] = {}
    if request.document_ids is not None:
        set_active_documents(request.chat_id, effective_document_ids)

    active_document_ids = get_active_document_ids(request.chat_id)

    # Workspace documents are persistent. document_chunks is the source of
    # truth for their availability; the in-memory store remains only for the
    # legacy chat-upload endpoint.
    chat_workspace_id = None
    if request.chat_id:
        try:
            chat_record = get_chat(request.chat_id)
            chat_workspace_id = str(chat_record.get("workspace_id")) if chat_record and chat_record.get("workspace_id") else None
        except Exception:
            chat_workspace_id = None

    persistent_document_chunks = get_persistent_document_chunks(
        active_document_ids,
        workspace_id=chat_workspace_id,
    )
    uploaded_documents = get_documents_for_chat(
        request.chat_id,
        effective_document_ids,
    )

    has_documents = bool(persistent_document_chunks) or bool(uploaded_documents)

    previous_scope = get_active_source_scope(request.chat_id)
    unavailable_document_ids = get_unavailable_chat_document_ids(
        request.chat_id,
        active_document_ids,
    )
    has_unavailable_documents = bool(unavailable_document_ids)

    # If the current request explicitly references a document that no longer
    # exists in this workspace, fail closed for source-specific questions even
    # if other documents remain attached to the chat.
    if (
        unavailable_document_ids
        and not github_url
        and question_explicitly_references_missing_document(
            question,
            request.history,
        )
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "The document is no longer available. "
                    "It was removed from this workspace, so I cannot answer "
                    "that source-specific question."
                ),
                "context_scope": "document",
                "source_unavailable": True,
                "unavailable_document_ids": unavailable_document_ids,
            },
        )

    route = ContextRouter.route(
        question=question,
        history=request.history,
        github_reference=github_reference,
        has_documents=has_documents,
        has_github=bool(requested_github_urls),
        resolved_query=retrieval_query,
        previous_scope=previous_scope,
        has_unavailable_documents=has_unavailable_documents,
    )
    context_scope = route["scope"]

    # Selected source truth: when both a persisted/uploaded document and a
    # GitHub repository are selected, never silently drop either source family.
    if persistent_document_chunks and requested_github_urls:
        context_scope = "hybrid"
        route["scope"] = "hybrid"
        route["reason"] = "both selected source families are active"

    print("\n" + "=" * 72)
    print("SOURCE ROUTING")
    print("=" * 72)
    print(f"Question:              {question!r}")
    print(f"Selected documents:    {len(active_document_ids)}")
    print(f"Selected GitHub repos: {len(requested_github_urls)}")
    print(f"Context scope:         {context_scope}")
    print(f"Route reason:          {route.get('reason')}")
    print(f"Route signals:         {route.get('signals')}")
    print("=" * 72 + "\n")

    # Explicitly attached workspace documents are authoritative for this
    # request when no GitHub repository is active.  The router may otherwise
    # classify a generic-looking question (for example, “what are the main
    # findings?”) as general chat even though the user deliberately attached
    # documents.  We want retrieval to make that relevance decision instead:
    #   attached docs + relevant question   -> document RAG
    #   attached docs + unrelated question  -> general Qwen fallback
    # Do not override a hybrid route merely because the legacy singular
    # github_url is empty; selected repositories are tracked in
    # requested_github_urls/chat_sources.

    source_switched = bool(
        previous_scope
        and previous_scope != context_scope
        and context_scope != "general"
    )
    set_active_source_scope(request.chat_id, context_scope)

    # Never carry retrieval terms from the previous source into a newly
    # selected repository/document. Conversation-aware resolution is only
    # useful once the source is stable.
    if repository_switched or source_switched:
        retrieval_query = question

    if repository_switched or source_switched:
        model_history = []
    elif context_scope in {"document", "hybrid"}:
        # Standalone research questions must be isolated from previous turns.
        # Only genuine conversational follow-ups receive prior conversation.
        is_conversational_followup = bool(
            resolution.get("was_resolved")
            and resolution.get("resolution_type")
            in {"reference_followup", "short_followup"}
        )
        model_history = (
            [
                message
                for message in request.history[-6:]
                if str(message.get("role", "")).strip().lower()
                in {"user", "assistant"}
            ]
            if is_conversational_followup
            else []
        )
    else:
        model_history = request.history

    if context_scope == "github":
        if not github_url:
            raise HTTPException(status_code=400, detail="No active GitHub repository is available for this question.")
        documents = []

        for selected_github_url in requested_github_urls:
            github_docs = build_github_documents(
                github_url=selected_github_url,
                question=retrieval_query,
                branch=request.branch,
            )

            documents.extend(
                {
                    **document,
                    "source": "github",
                    "source_type": "github_repository",
                    "github_url": selected_github_url,
                }
                for document in github_docs
            )
        context_origin = "github"

    elif context_scope == "document":
        if not persistent_document_chunks and not uploaded_documents:
            if unavailable_document_ids and question_explicitly_references_missing_document(
                question,
                request.history,
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            "The document is no longer available. "
                            "It was removed from this workspace, so I cannot answer "
                            "that source-specific question."
                        ),
                        "context_scope": "document",
                        "source_unavailable": True,
                        "unavailable_document_ids": unavailable_document_ids,
                    },
                )

            # The selected document is gone, but the current question does not
            # explicitly depend on that missing source. Continue with general chat.
            context_origin = "general_chat"
            context_scope = "general"
            active_chunker_name = "none"
            active_retriever_name = "none"

        if context_scope == "document":
            chunks = list(persistent_document_chunks)
            context_origin = "upload"
            active_chunker_name = "PersistentDocumentChunks"

    elif context_scope == "hybrid":
        if not requested_github_urls or not persistent_document_chunks:
            missing_parts: List[str] = []
            if not requested_github_urls:
                missing_parts.append("GitHub repository")
            if not persistent_document_chunks:
                if unavailable_document_ids:
                    missing_parts.append("uploaded document (removed)")
                else:
                    missing_parts.append("uploaded document")

            raise HTTPException(
                status_code=409 if unavailable_document_ids else 400,
                detail={
                    "message": (
                        "A cross-source question cannot be answered because "
                        + ", ".join(missing_parts)
                        + " is unavailable."
                    ),
                    "context_scope": "hybrid",
                    "source_unavailable": bool(unavailable_document_ids),
                    "unavailable_document_ids": unavailable_document_ids,
                },
            )

        github_documents: List[Dict[str, Any]] = []

        for selected_github_url in requested_github_urls:
            github_docs = build_github_documents(
                github_url=selected_github_url,
                question=retrieval_query,
                branch=request.branch,
            )

            github_documents.extend(
                {
                    **document,
                    "source": "github",
                    "source_type": "github_repository",
                    "github_url": selected_github_url,
                }
                for document in github_docs
            )

        github_chunks_created = document_chunker.chunk_documents(github_documents)

        # GitHub overview documents are already normalized source records. If
        # the generic document chunker rejects a small/structured file set,
        # preserve the GitHub evidence as direct chunks rather than dropping
        # the entire repository from the hybrid request.
        if github_documents and not github_chunks_created:
            github_chunks_created = []
            for document in github_documents:
                content = str(document.get("content") or "").strip()
                if not content:
                    continue
                github_chunks_created.append({
                    **document,
                    "content": content,
                    "raw_content": content,
                    "path": str(document.get("path") or "github/source.md"),
                    "source": "github",
                    "source_type": "github_repository",
                    "chunk_type": "text",
                    "chunk_index": 0,
                    "char_count": len(content),
                })

        chunks = list(persistent_document_chunks)
        chunks.extend(github_chunks_created)

        print(
            "HYBRID CONTEXT PREP: "
            f"upload_chunks={len(persistent_document_chunks)}, "
            f"github_documents={len(github_documents)}, "
            f"github_chunks={len(github_chunks_created)}"
        )

        documents = github_documents
        context_origin = "github+upload"
        active_chunker_name = "PersistentDocumentChunks + DocumentChunker"

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

    if context_scope != "general" and not documents and not chunks:
        raise HTTPException(
            status_code=502 if requested_github_urls else 400,
            detail=(
                "GitHubContentService returned no usable documents."
                if requested_github_urls
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
                # Chunks were already loaded from persistent document_chunks.
                active_chunker_name = "PersistentDocumentChunks"
            elif context_scope == "hybrid":
                # IMPORTANT:
                # Do NOT reconstruct source families from the flattened `chunks`
                # list here. The preparation stage already created authoritative
                # GitHub and upload lists. Re-splitting after chunking was the
                # reason the GitHub family could disappear from retrieval.

                github_chunks = [
                    chunk
                    for chunk in github_chunks_created
                    if get_chunk_content(chunk)
                ]

                upload_chunks = [
                    chunk
                    for chunk in persistent_document_chunks
                    if get_chunk_content(chunk)
                ]

                print(
                    "HYBRID RETRIEVAL INPUTS: "
                    f"upload={len(upload_chunks)}, "
                    f"github={len(github_chunks)}"
                )

                # Rank each source family using the SAME production ranker
                # already used when that family is retrieved on its own
                # (HybridRetriever for GitHub, DocumentRetriever for uploads),
                # instead of maintaining a bespoke local reranker here.
                #
                # This file previously had a third, independent scoring
                # implementation (pure embedding cosine similarity with an
                # ad-hoc lexical patch) that could silently disagree with
                # HybridRetriever's own lexical/path/symbol/BM25 scoring.
                # That drift was the root cause of GitHub evidence dropping
                # out of hybrid comparison answers. HybridRetriever's
                # relevance filter already guarantees at least one result
                # whenever `chunks` is non-empty (see _relevance_filter's
                # "never erase the whole pool" fallback), so the
                # source-family coverage guarantee below still holds even
                # though HybridRetriever's own gate is active here.
                source_count = int(bool(upload_chunks)) + int(bool(github_chunks))
                per_source_k = (
                    max(
                        1,
                        math.ceil(requested_top_k / source_count),
                    )
                    if source_count
                    else 0
                )

                github_results = (
                    hybrid_retriever.retrieve(
                        question=retrieval_query,
                        chunks=github_chunks,
                        top_k=min(per_source_k, len(github_chunks)),
                    )
                    if github_chunks
                    else []
                )

                upload_results = (
                    document_retriever.retrieve(
                        question=retrieval_query,
                        chunks=upload_chunks,
                        top_k=min(per_source_k, len(upload_chunks)),
                    )
                    if upload_chunks
                    else []
                )

                retrieved_chunks = []

                # Guarantee source-family coverage before filling leftovers.
                for index in range(per_source_k):
                    if index < len(upload_results):
                        retrieved_chunks.append(upload_results[index])
                    if index < len(github_results):
                        retrieved_chunks.append(github_results[index])

                if len(retrieved_chunks) < requested_top_k:
                    seen_ids = {
                        (
                            str(item.get("source") or ""),
                            str(item.get("document_id") or ""),
                            str(item.get("path") or ""),
                            str(item.get("chunk_index") or ""),
                            str(item.get("content") or "")[:120],
                        )
                        for item in retrieved_chunks
                    }

                    leftovers = (
                        upload_results[per_source_k:]
                        + github_results[per_source_k:]
                    )

                    for item in leftovers:
                        key = (
                            str(item.get("source") or ""),
                            str(item.get("document_id") or ""),
                            str(item.get("path") or ""),
                            str(item.get("chunk_index") or ""),
                            str(item.get("content") or "")[:120],
                        )

                        if key in seen_ids:
                            continue

                        retrieved_chunks.append(item)
                        seen_ids.add(key)

                        if len(retrieved_chunks) >= requested_top_k:
                            break

                print(
                    "HYBRID SOURCE RESULTS: "
                    f"upload={len(upload_results)}, "
                    f"github={len(github_results)}, "
                    f"final={len(retrieved_chunks)}"
                )

                print(
                    "HYBRID GITHUB PATHS: "
                    + ", ".join(
                        str(item.get("path") or "")
                        for item in github_results
                    )
                )

                active_retriever_name = "SourceAwareHybridRetriever"

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
                evidence_result = evidence_engine.retrieve(
                    question=retrieval_query,
                    chunks=chunks,
                    document_ids=active_document_ids,
                    top_k=requested_top_k,
                )
                retrieved_chunks = list(evidence_result.get("evidence") or [])
                evidence_validation = evidence_validator.validate(
                    question=retrieval_query,
                    evidence_result=evidence_result,
                )
                evidence_result["validation"] = evidence_validation
                active_retriever_name = "EvidenceEngine + DocumentRetriever"

            elif context_scope == "hybrid":
                # retrieved_chunks was already computed above (chunk_start
                # phase) using the authoritative source-tagged
                # github_chunks_created / persistent_document_chunks lists.
                # Do NOT re-derive source families from the flattened
                # `chunks` list here: document_chunker.chunk_documents()
                # does not guarantee it preserves the "source" key on its
                # output chunk dicts, so re-splitting after chunking can
                # silently drop the GitHub family from retrieval. This was
                # the root cause of cross-source questions returning only
                # document evidence.
                pass

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
            # A selected document does not mean every question is about it.
            # If retrieval finds no relevant evidence, intentionally fall back
            # to general Qwen rather than failing the request.
            if context_scope == "document":
                context_origin = "general_chat"
                context_scope = "general"
                active_retriever_name = "none"
                active_chunker_name = "none"
                retrieved_context = ""

                prompt_start = _profile_start()
                system_prompt = """
You are a helpful AI assistant.

The user has selected documents, but the retrieval step returned no usable
passages for this particular question.

Answer helpfully using general knowledge and the conversation history.
Do not pretend the answer came from the selected documents.
If the question clearly asks for source-specific facts, explain that the
selected evidence did not establish those facts.
""".strip()
                messages = build_messages(
                    question=question,
                    history=model_history,
                    system_prompt=system_prompt,
                )
                timings["prompt_build_ms"] = _profile_ms(prompt_start)
            else:
                raise HTTPException(
                    status_code=404,
                    detail="No sufficiently relevant evidence was found in the active source.",
                )
        else:
            github_count = sum(1 for item in retrieved_chunks if str(item.get("source_type") or item.get("source") or "").lower() in {"github","github_repository"})
            upload_count = sum(1 for item in retrieved_chunks if str(item.get("source_type") or item.get("source") or "").lower() in {"upload","uploaded_document","document"})
            print(f"SOURCE EVIDENCE COUNTS: github={github_count}, upload={upload_count}, total={len(retrieved_chunks)}")
            context_format_start = _profile_start()
            retrieved_context = format_retrieved_context(retrieved_chunks, evidence_result)
            timings["context_formatting_ms"] = _profile_ms(context_format_start)
            if not retrieved_context.strip():
                raise HTTPException(status_code=500, detail="Retriever returned chunks but their content was empty.")

            prompt_start = time.perf_counter()
            system_prompt = build_system_prompt(
                retrieved_context,
                evidence_result,
                question=question,
            )

            messages = build_messages(
                question=question,
                history=model_history,
                system_prompt=system_prompt,
            )
            timings["prompt_build_ms"] = _elapsed_ms(prompt_start)

            nonempty_retrieved = [
                chunk for chunk in retrieved_chunks
                if get_chunk_content(chunk)
            ]
            print("\n" + "=" * 72)
            print("LLM EVIDENCE HANDOFF")
            print("=" * 72)
            print(f"Retrieved evidence chunks : {len(nonempty_retrieved)}")
            print(f"Prompt contains evidence  : {bool(retrieved_context.strip())}")
            print(f"Retrieved context chars   : {len(retrieved_context):,}")
            print(f"System prompt chars       : {len(system_prompt):,}")
            print("=" * 72 + "\n")

    # ========================================================
    # 5. QWEN / OLLAMA
    # ========================================================

    llm_start = _profile_start()

    print("\n" + "=" * 72)
    print("LLM CONTEXT CONTROL")
    print("=" * 72)
    print(f"Follow-up detected:   {is_conversational_followup}")
    print(f"History sent to Qwen:  {len(model_history)}")
    print(f"Retrieval query:       {retrieval_query!r}")
    print(f"Current question:      {question!r}")
    print("=" * 72 + "\n")

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
                        unavailable_document_ids=unavailable_document_ids,
                        evidence_result=evidence_result,
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
        evidence_result=evidence_result,
        unavailable_document_ids=unavailable_document_ids,
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
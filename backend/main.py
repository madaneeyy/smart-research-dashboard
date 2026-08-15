from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

ollama_client = ollama.Client(host=OLLAMA_HOST)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Smart Research AI API",
    description="Structure-aware RAG with HybridRetriever and Qwen.",
    version="2.1.0",
)


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

    context: str = Field(
        ...,
        min_length=1,
        description="Research/repository content to retrieve from.",
    )

    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Optional previous user/assistant messages.",
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
    )

    # Optional metadata from the frontend. These allow the backend
    # to preserve meaningful source information in chunk metadata.
    source_path: Optional[str] = None
    source_category: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

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
    return f"""
You are a technical research assistant.

Answer the user's question using the retrieved evidence provided below.

Your goal is to give a clear, useful, technically accurate answer while
staying faithful to the retrieved source.

============================================================
GROUNDING RULES
============================================================

1. Use the retrieved evidence as the primary source for
   repository-specific or research-specific claims.

2. Answer the user's question directly. Do not begin by describing
   what type of document or repository the evidence comes from unless
   that information is directly relevant to the question.

3. Do not invent repository-specific facts, implementation details,
   filenames, functions, parameters, results, or conclusions.

4. You may combine information from multiple retrieved chunks when
   they provide complementary parts of the answer.

5. Preserve the terminology used by the source, especially:
   - class names
   - function names
   - filenames
   - modules
   - APIs
   - parameters
   - algorithms
   - architecture names
   - configuration values

============================================================
FACTS VS INFERENCE
============================================================

Prefer statements directly supported by the retrieved evidence.

You may provide a reasonable technical interpretation when it helps
explain the evidence, but do not present that interpretation as an
explicit statement from the repository.

For example:

Source:
"A window is applied on the attention map to limit backward attention
and focus on short term patterns."

Good:
"The repository applies a window to the attention map to limit
backward attention and focus on short-term patterns."

Also acceptable:
"This means the model is designed to place more emphasis on nearby
historical time steps."

Do NOT say:
"The authors chose this approach because long-range dependencies are
irrelevant."

unless the retrieved evidence explicitly says that.

When an explanation is an inference, use natural wording such as:

"This means..."
"This can be interpreted as..."
"In practice, this allows..."
"Conceptually..."

Do not repeatedly announce that something is an inference unless the
distinction is important.

============================================================
RATIONALE AND "WHY" QUESTIONS
============================================================

If the user asks why something was implemented:

- Give the reason if the retrieved evidence states it.
- If the reason is strongly implied by the evidence, you may explain
  the implication naturally without attributing an unstated intention
  to the authors.
- If the evidence genuinely does not provide enough information,
  briefly say that the repository does not specify the exact reason.

Do not invent author intent.

Avoid unnecessary disclaimers such as:

"The retrieved evidence does not describe a research study..."

unless the user specifically asked whether the source is a research
study.

============================================================
CLAIM STRENGTH
============================================================

Do not unnecessarily strengthen claims.

Prefer:

"The repository applies..."
"The implementation uses..."
"The section describes..."
"The model focuses on..."

Avoid strong claims such as:

"guarantees"
"ensures"
"proves"
"eliminates"
"always"
"optimal"

unless the evidence explicitly supports them.

============================================================
INCOMPLETE EVIDENCE
============================================================

If the evidence is sufficient to answer the question, simply answer it.

If a small detail is missing, answer the rest of the question normally
and briefly mention the missing detail at the end.

Only say that the evidence is insufficient when the missing information
actually prevents you from answering an important part of the question.

Do not add a generic limitation section to every answer.

============================================================
CODE AND IMPLEMENTATION QUESTIONS
============================================================

For code questions:

- describe what the retrieved code actually does
- preserve exact identifiers
- mention relevant files or modules when available
- combine related implementation chunks when necessary
- do not invent code that is not present in the evidence

If the retrieved code is incomplete, explain only the limitation that
actually matters to the user's question.

============================================================
GENERAL TECHNICAL KNOWLEDGE
============================================================

You may use general technical knowledge to make the retrieved evidence
easier to understand.

However, do not use general knowledge to invent repository-specific
facts.

When explaining something beyond what the repository explicitly states,
phrase it as an explanation rather than attributing it to the authors.

For example:

"The repository uses a linear layer here. In a time-series setting,
this acts as a projection from the input features into the model's
representation space."

This is an explanation of the implementation, not a claim about the
authors' stated rationale.

============================================================
ANSWER STYLE
============================================================

Start with the answer.

Do not start with:

"The retrieved evidence..."
"The provided context..."
"This is not a research paper..."
"The source does not..."
"Based on the retrieved evidence..."

unless that information is necessary to answer the question.

For straightforward questions, give a straightforward answer.

For multi-part questions, use numbered sections or bullets.

For implementation questions, explain the relevant flow clearly.

For conceptual questions, explain the concept first and then connect it
to the repository.

Be concise when the question is simple and detailed when the question
requires explanation.

============================================================
SOURCE REFERENCES
============================================================

When useful, naturally mention the relevant source section or file.

Examples:

"The repository's `Adaptations for time series` section states..."

"The implementation in `model.py`..."

"The README explains..."

Do not fabricate filenames, sections, URLs, or citations.

============================================================
FINAL INTERNAL CHECK
============================================================

Before answering, check:

- Did I answer the actual question?
- Are repository-specific facts supported?
- Did I accidentally invent author intent?
- Did I turn an inference into a source claim?
- Did I preserve technical terminology?
- Am I adding an unnecessary disclaimer?
- Am I making the answer more complicated than the question requires?

If the answer is supported, answer confidently and directly.

============================================================
RETRIEVED RESEARCH EVIDENCE
============================================================

{retrieved_context}

============================================================
END RETRIEVED RESEARCH EVIDENCE
============================================================
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
            "retriever": "HybridRetriever",
            "chunker": "DocumentChunker",
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
# ASK
# ============================================================

@app.post("/ask")
def ask_ai(request: AskRequest) -> Dict[str, Any]:

    question = request.question.strip()
    context = request.context.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    if not context:
        raise HTTPException(
            status_code=400,
            detail="Research context must not be empty.",
        )

    if len(context) > MAX_CONTEXT_CHARACTERS:
        raise HTTPException(
            status_code=413,
            detail=(
                "Research context is too large. "
                f"Maximum allowed size: "
                f"{MAX_CONTEXT_CHARACTERS:,} characters."
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
    # 1. STRUCTURE-AWARE CHUNKING
    # ========================================================

    source_path = (
        request.source_path
        or "research/repository_context.md"
    )

    source_category = (
        request.source_category
        or "documentation"
    )

    try:
        chunks = document_chunker.chunk_document(
            content=context,
            path=source_path,
            category=source_category,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to chunk research context.",
                "error": str(exc),
            },
        )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No usable research chunks were created.",
        )

    # ========================================================
    # 2. HYBRID RETRIEVAL
    # ========================================================

    try:
        retrieved_chunks = hybrid_retriever.retrieve(
            question=question,
            chunks=chunks,
            top_k=min(
                requested_top_k,
                len(chunks),
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Hybrid retrieval failed.",
                "error": str(exc),
            },
        )

    if not retrieved_chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                "HybridRetriever did not find any relevant "
                "evidence for this question."
            ),
        )

    # ========================================================
    # 3. FORMAT RETRIEVED EVIDENCE
    # ========================================================

    retrieved_context = format_retrieved_context(
        retrieved_chunks
    )

    if not retrieved_context.strip():
        raise HTTPException(
            status_code=500,
            detail=(
                "Retriever returned chunks but their content "
                "was empty."
            ),
        )

    # ========================================================
    # 4. BUILD QWEN PROMPT
    # ========================================================

    system_prompt = build_system_prompt(
        retrieved_context
    )

    messages = build_messages(
        question=question,
        history=request.history,
        system_prompt=system_prompt,
    )

    # ========================================================
    # 5. QWEN / OLLAMA
    # ========================================================

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "num_ctx": OLLAMA_NUM_CTX,
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
            keep_alive=OLLAMA_KEEP_ALIVE,
        )

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

    serialized_sources = [
        serialize_chunk(chunk, index)
        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        )
    ]

    first_chunk = (
        retrieved_chunks[0]
        if retrieved_chunks
        else {}
    )

    return {
        "question": question,
        "answer": answer,
        "model": OLLAMA_MODEL,

        "retriever": "HybridRetriever",
        "chunker": "DocumentChunker",

        "chunks_created": len(chunks),
        "chunks_retrieved": len(retrieved_chunks),

        "sources": serialized_sources,

        "retrieval": {
            "top_k": requested_top_k,
            "query_type": first_chunk.get("query_type"),
            "candidate_pool_size": first_chunk.get(
                "candidate_pool_size"
            ),
            "post_filter_pool_size": first_chunk.get(
                "post_filter_pool_size"
            ),
        },
    }


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
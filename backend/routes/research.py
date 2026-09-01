from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.research import ResearchService


router = APIRouter(
    prefix="/research",
    tags=["Research"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ResearchSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Research topic or question.",
    )

    sources: list[str] | None = Field(
        default=None,
        description=(
            "Sources to search. "
            "Valid values: arxiv, github, "
            "paperswithcode, huggingface."
        ),
    )

    sort_by: str = Field(
        default="relevance",
        description=(
            "Result ordering: relevance, "
            "published, or updated."
        ),
    )

    search_mode: str = Field(
        default="keyword",
        description=(
            "Search mode: keyword, semantic, or hybrid."
        ),
    )

    arxiv_limit: int = Field(
        default=20,
        ge=1,
        le=50,
    )

    github_limit: int = Field(
        default=20,
        ge=1,
        le=50,
    )

    paperswithcode_limit: int = Field(
        default=20,
        ge=1,
        le=50,
    )

    huggingface_limit: int = Field(
        default=20,
        ge=1,
        le=50,
    )


# ============================================================
# RESPONSE MODEL
# ============================================================

class ResearchItemResponse(BaseModel):
    # ---------------------------------------------------------
    # Common fields
    # ---------------------------------------------------------

    id: str
    title: str
    description: str | None = None
    authors: list[str]

    source: str
    url: str

    published: datetime | None = None
    updated: datetime | None = None

    tags: list[str] = Field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # GitHub-specific
    # ---------------------------------------------------------

    stars: int | None = None
    forks: int | None = None
    language: str | None = None

    # ---------------------------------------------------------
    # Hugging Face-specific
    # ---------------------------------------------------------

    downloads: int | None = None
    likes: int | None = None
    library: str | None = None
    pipeline_tag: str | None = None

    # ---------------------------------------------------------
    # PapersWithCode-specific
    # ---------------------------------------------------------

    tasks: list[str] = Field(
        default_factory=list
    )

    conference: str | None = None

    # ---------------------------------------------------------
    # Flexible provider metadata
    # ---------------------------------------------------------

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


# ============================================================
# SEARCH ENDPOINT
# ============================================================

@router.post(
    "/search",
    response_model=list[ResearchItemResponse],
)
def search_research(
    request: ResearchSearchRequest,
):
    """
    Search across the configured research providers.

    This endpoint intentionally delegates all research logic
    to the existing ResearchService rather than duplicating
    collection/ranking code inside the API layer.
    """

    service = ResearchService()

    try:
        results = service.search(
            query=request.query,
            sources=request.sources,
            arxiv_limit=request.arxiv_limit,
            github_limit=request.github_limit,
            paperswithcode_limit=request.paperswithcode_limit,
            huggingface_limit=request.huggingface_limit,
            sort_by=request.sort_by,
            search_mode=request.search_mode,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Research search failed.",
                "error": str(exc),
            },
        ) from exc

    return [
        ResearchItemResponse(
            # -------------------------------------------------
            # Common
            # -------------------------------------------------

            id=result.id,
            title=result.title,
            description=result.description,
            authors=result.authors,
            source=result.source,
            url=str(result.url),

            published=result.published,
            updated=result.updated,

            tags=result.tags,

            # -------------------------------------------------
            # GitHub
            # -------------------------------------------------

            stars=result.stars,
            forks=result.forks,
            language=result.language,

            # -------------------------------------------------
            # Hugging Face
            # -------------------------------------------------

            downloads=result.downloads,
            likes=result.likes,
            library=result.library,
            pipeline_tag=result.pipeline_tag,

            # -------------------------------------------------
            # PapersWithCode
            # -------------------------------------------------

            tasks=result.tasks,
            conference=result.conference,

            # -------------------------------------------------
            # Flexible metadata
            # -------------------------------------------------

            metadata=result.metadata,
        )
        for result in results
    ]
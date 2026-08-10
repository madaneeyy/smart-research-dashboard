from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class ResearchItem(BaseModel):
    # ---------------------------------------------------------
    # Common fields
    # ---------------------------------------------------------

    id: str
    title: str
    description: str | None = None
    authors: list[str]

    source: str
    url: HttpUrl

    published: datetime | None = None
    updated: datetime | None = None

    tags: list[str] = Field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # GitHub-specific metadata
    # ---------------------------------------------------------

    stars: int | None = None
    forks: int | None = None
    language: str | None = None

    # ---------------------------------------------------------
    # Hugging Face-specific metadata
    # ---------------------------------------------------------

    downloads: int | None = None
    likes: int | None = None
    library: str | None = None
    pipeline_tag: str | None = None

    # ---------------------------------------------------------
    # PapersWithCode-specific metadata
    # ---------------------------------------------------------

    tasks: list[str] = Field(
        default_factory=list
    )

    conference: str | None = None

    # ---------------------------------------------------------
    # Flexible source-specific metadata
    # ---------------------------------------------------------

    metadata: dict[str, object] = Field(
        default_factory=dict
    )
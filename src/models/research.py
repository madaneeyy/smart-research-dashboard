from datetime import datetime

from pydantic import BaseModel, HttpUrl


class ResearchItem(BaseModel):
    id: str
    title: str
    description: str
    authors: list[str]

    source: str

    url: HttpUrl

    published: datetime | None = None
    updated: datetime | None = None

    tags: list[str] = []
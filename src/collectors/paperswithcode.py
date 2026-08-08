from datetime import datetime

from pydantic import BaseModel, HttpUrl
from src.clients.paperswithcode import get_paperswithcode_rows
from src.models.research import ResearchItem

PAPERS_DATASET = "pwc-archive/papers-with-abstracts"
class PapersWithCodeMethod(BaseModel):
    name: str | None = None
    full_name: str | None = None
    description: str | None = None
    introduced_year: int | None = None
    source_title: str | None = None
    source_url: HttpUrl | None = None


class PapersWithCodePaper(BaseModel):
    paper_url: HttpUrl
    arxiv_id: str | None = None
    nips_id: float | None = None
    openreview_id: str | None = None

    title: str
    abstract: str | None = None
    short_abstract: str | None = None

    url_abs: HttpUrl | None = None
    url_pdf: HttpUrl | None = None

    proceeding: str | None = None

    authors: list[str]
    tasks: list[str]

    date: datetime | None = None

    conference_url_abs: HttpUrl | None = None
    conference_url_pdf: HttpUrl | None = None
    conference: str | None = None

    reproduces_paper: str | None = None

    methods: list[PapersWithCodeMethod]


class PaperCodeLink(BaseModel):
    paper_url: HttpUrl
    paper_title: str

    paper_arxiv_id: str | None = None
    paper_url_abs: HttpUrl | None = None
    paper_url_pdf: HttpUrl | None = None

    repo_url: HttpUrl | None = None

    is_official: bool
    mentioned_in_paper: bool
    mentioned_in_github: bool

    framework: str | None = None


def parse_paperswithcode_method(data: dict) -> PapersWithCodeMethod:
    return PapersWithCodeMethod(
        name=data.get("name"),
        full_name=data.get("full_name"),
        description=data.get("description"),
        introduced_year=data.get("introduced_year"),
        source_title=data.get("source_title"),
        source_url=data.get("source_url"),
    )


def parse_paperswithcode_paper(data: dict) -> PapersWithCodePaper:
    return PapersWithCodePaper(
        paper_url=data["paper_url"],
        arxiv_id=data.get("arxiv_id"),
        nips_id=data.get("nips_id"),
        openreview_id=data.get("openreview_id"),
        title=data["title"],
        abstract=data.get("abstract"),
        short_abstract=data.get("short_abstract"),
        url_abs=data.get("url_abs"),
        url_pdf=data.get("url_pdf"),
        proceeding=data.get("proceeding"),
        authors=data.get("authors", []),
        tasks=data.get("tasks", []),
        date=data.get("date"),
        conference_url_abs=data.get("conference_url_abs"),
        conference_url_pdf=data.get("conference_url_pdf"),
        conference=data.get("conference"),
        reproduces_paper=data.get("reproduces_paper"),
        methods=[
            parse_paperswithcode_method(method)
            for method in data.get("methods", [])
        ],
    )
def paperswithcode_paper_to_item(
    paper: PapersWithCodePaper,
) -> ResearchItem:
    return ResearchItem(
        id=paper.arxiv_id or str(paper.paper_url),
        title=paper.title,
        description=paper.abstract or "",
        authors=paper.authors,
        source="paperswithcode",
        url=paper.paper_url,
        published=paper.date,
        updated=None,
        tags=paper.tasks,
    )



def parse_paper_code_link(data: dict) -> PaperCodeLink:
    return PaperCodeLink(
        paper_url=data["paper_url"],
        paper_title=data["paper_title"],
        paper_arxiv_id=data.get("paper_arxiv_id"),
        paper_url_abs=data.get("paper_url_abs"),
        paper_url_pdf=data.get("paper_url_pdf"),
        repo_url=data.get("repo_url"),
        is_official=data.get("is_official", False),
        mentioned_in_paper=data.get("mentioned_in_paper", False),
        mentioned_in_github=data.get("mentioned_in_github", False),
        framework=data.get("framework"),
    )

def load_paperswithcode_papers(
    *,
    offset: int = 0,
    length: int = 100,
) -> list[PapersWithCodePaper]:
    response = get_paperswithcode_rows(
        PAPERS_DATASET,
        offset=offset,
        length=length,
    )

    data = response.json()

    return [
        parse_paperswithcode_paper(item["row"])
        for item in data.get("rows", [])
    ]

def search_paperswithcode_papers(
    offset: int = 0,
    length: int = 20,
) -> list[PapersWithCodePaper]:
    if offset < 0:
        raise ValueError("offset must be 0 or greater")

    if length < 1:
        raise ValueError("length must be at least 1")

    response = get_paperswithcode_rows(
        "pwc-archive/papers-with-abstracts",
        offset=offset,
        length=length,
    )

    data = response.json()

    return [
        parse_paperswithcode_paper(row["row"])
        for row in data.get("rows", [])
    ]




def search_paper_code_links(
    offset: int = 0,
    length: int = 20,
) -> list[PaperCodeLink]:
    if offset < 0:
        raise ValueError("offset must be 0 or greater")

    if length < 1:
        raise ValueError("length must be at least 1")

    response = get_paperswithcode_rows(
        "pwc-archive/links-between-paper-and-code",
        offset=offset,
        length=length,
    )

    data = response.json()

    return [
        parse_paper_code_link(row["row"])
        for row in data.get("rows", [])
    ]
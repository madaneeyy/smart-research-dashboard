from datetime import datetime

from pydantic import BaseModel, HttpUrl

from src.clients.paperswithcode import (
    get_paperswithcode_rows,
)
from src.models.research import ResearchItem


# =============================================================
# DATASET CONSTANTS
# =============================================================

PAPERS_DATASET = (
    "pwc-archive/papers-with-abstracts"
)

PAPER_CODE_LINKS_DATASET = (
    "pwc-archive/links-between-paper-and-code"
)


# =============================================================
# PAPERSWITHCODE METHOD MODEL
# =============================================================

class PapersWithCodeMethod(BaseModel):
    name: str | None = None
    full_name: str | None = None
    description: str | None = None
    introduced_year: int | None = None
    source_title: str | None = None
    source_url: HttpUrl | None = None


# =============================================================
# PAPERSWITHCODE PAPER MODEL
# =============================================================

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


# =============================================================
# PAPER ↔ CODE LINK MODEL
# =============================================================

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


# =============================================================
# PARSE PAPERSWITHCODE METHOD
# =============================================================

def parse_paperswithcode_method(
    data: dict,
) -> PapersWithCodeMethod:
    """
    Convert a raw PapersWithCode method dictionary
    into a PapersWithCodeMethod model.
    """

    return PapersWithCodeMethod(
        name=data.get("name"),
        full_name=data.get("full_name"),
        description=data.get("description"),
        introduced_year=data.get(
            "introduced_year"
        ),
        source_title=data.get(
            "source_title"
        ),
        source_url=data.get(
            "source_url"
        ),
    )


# =============================================================
# PARSE PAPERSWITHCODE PAPER
# =============================================================

def parse_paperswithcode_paper(
    data: dict,
) -> PapersWithCodePaper:
    """
    Convert a raw PapersWithCode dataset row
    into a PapersWithCodePaper model.
    """

    return PapersWithCodePaper(
        paper_url=data["paper_url"],

        arxiv_id=data.get("arxiv_id"),
        nips_id=data.get("nips_id"),
        openreview_id=data.get(
            "openreview_id"
        ),

        title=data["title"],

        abstract=data.get("abstract"),
        short_abstract=data.get(
            "short_abstract"
        ),

        url_abs=data.get("url_abs"),
        url_pdf=data.get("url_pdf"),

        proceeding=data.get(
            "proceeding"
        ),

        authors=data.get(
            "authors",
            [],
        ),

        tasks=data.get(
            "tasks",
            [],
        ),

        date=data.get("date"),

        conference_url_abs=data.get(
            "conference_url_abs"
        ),

        conference_url_pdf=data.get(
            "conference_url_pdf"
        ),

        conference=data.get(
            "conference"
        ),

        reproduces_paper=data.get(
            "reproduces_paper"
        ),

        methods=[
            parse_paperswithcode_method(
                method
            )
            for method in data.get(
                "methods",
                [],
            )
        ],
    )


# =============================================================
# PAPERSWITHCODE PAPER → RESEARCH ITEM
# =============================================================

def paperswithcode_paper_to_item(
    paper: PapersWithCodePaper,
) -> ResearchItem:
    """
    Convert a PapersWithCode paper into
    the common ResearchItem model.
    """

    # ---------------------------------------------------------
    # Source-specific metadata
    # ---------------------------------------------------------

    metadata = {
        "arxiv_id": paper.arxiv_id,
        "nips_id": paper.nips_id,
        "openreview_id": paper.openreview_id,
        "short_abstract": paper.short_abstract,
        "url_abs": (
            str(paper.url_abs)
            if paper.url_abs
            else None
        ),
        "url_pdf": (
            str(paper.url_pdf)
            if paper.url_pdf
            else None
        ),
        "proceeding": paper.proceeding,
        "conference": paper.conference,
        "conference_url_abs": (
            str(paper.conference_url_abs)
            if paper.conference_url_abs
            else None
        ),
        "conference_url_pdf": (
            str(paper.conference_url_pdf)
            if paper.conference_url_pdf
            else None
        ),
        "reproduces_paper": (
            paper.reproduces_paper
        ),
        "methods": [
            {
                "name": method.name,
                "full_name": method.full_name,
                "description": method.description,
                "introduced_year": (
                    method.introduced_year
                ),
                "source_title": (
                    method.source_title
                ),
                "source_url": (
                    str(method.source_url)
                    if method.source_url
                    else None
                ),
            }
            for method in paper.methods
        ],
    }

    # ---------------------------------------------------------
    # ResearchItem
    # ---------------------------------------------------------

    return ResearchItem(
        id=(
            paper.arxiv_id
            or str(paper.paper_url)
        ),
        title=paper.title,
        description=paper.abstract or "",
        authors=paper.authors,
        source="paperswithcode",
        url=paper.paper_url,
        published=paper.date,
        updated=None,
        tags=paper.tasks,
        metadata=metadata,
    )


# =============================================================
# PARSE PAPER ↔ CODE LINK
# =============================================================

def parse_paper_code_link(
    data: dict,
) -> PaperCodeLink:
    """
    Convert a raw PapersWithCode
    paper-code relationship into a model.
    """

    return PaperCodeLink(
        paper_url=data["paper_url"],

        paper_title=data[
            "paper_title"
        ],

        paper_arxiv_id=data.get(
            "paper_arxiv_id"
        ),

        paper_url_abs=data.get(
            "paper_url_abs"
        ),

        paper_url_pdf=data.get(
            "paper_url_pdf"
        ),

        repo_url=data.get(
            "repo_url"
        ),

        is_official=data.get(
            "is_official",
            False,
        ),

        mentioned_in_paper=data.get(
            "mentioned_in_paper",
            False,
        ),

        mentioned_in_github=data.get(
            "mentioned_in_github",
            False,
        ),

        framework=data.get(
            "framework"
        ),
    )


# =============================================================
# LOAD PAPERSWITHCODE PAPERS
# =============================================================

def load_paperswithcode_papers(
    *,
    offset: int = 0,
    length: int = 100,
) -> list[PapersWithCodePaper]:
    """
    Load papers directly from the
    PapersWithCode archived dataset.

    Parameters
    ----------
    offset:
        Number of rows to skip.

    length:
        Number of rows to fetch.
    """

    # ---------------------------------------------------------
    # Validate pagination
    # ---------------------------------------------------------

    if offset < 0:
        raise ValueError(
            "offset must be 0 or greater"
        )

    if length < 1:
        raise ValueError(
            "length must be at least 1"
        )

    # ---------------------------------------------------------
    # Request dataset rows
    # ---------------------------------------------------------

    response = get_paperswithcode_rows(
        PAPERS_DATASET,
        offset=offset,
        length=length,
    )

    # ---------------------------------------------------------
    # Parse response
    # ---------------------------------------------------------

    data = response.json()

    return [
        parse_paperswithcode_paper(
            item["row"]
        )
        for item in data.get(
            "rows",
            [],
        )
    ]


# =============================================================
# PAPER QUERY MATCHING
# =============================================================

def _paper_matches_query(
    paper: PapersWithCodePaper,
    query: str,
) -> bool:
    """
    Check whether a paper matches a search query.

    The query is matched against:

    - title
    - abstract
    - short abstract
    - tasks
    - authors
    - method names
    """

    query_lower = query.strip().lower()

    if not query_lower:
        return False

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    if query_lower in paper.title.lower():
        return True

    # ---------------------------------------------------------
    # Abstract
    # ---------------------------------------------------------

    if query_lower in (
        paper.abstract or ""
    ).lower():
        return True

    # ---------------------------------------------------------
    # Short abstract
    # ---------------------------------------------------------

    if query_lower in (
        paper.short_abstract or ""
    ).lower():
        return True

    # ---------------------------------------------------------
    # Tasks
    # ---------------------------------------------------------

    if any(
        query_lower in task.lower()
        for task in paper.tasks
    ):
        return True

    # ---------------------------------------------------------
    # Authors
    # ---------------------------------------------------------

    if any(
        query_lower in author.lower()
        for author in paper.authors
    ):
        return True

    # ---------------------------------------------------------
    # Methods
    # ---------------------------------------------------------

    if any(
        method.name
        and query_lower
        in method.name.lower()
        for method in paper.methods
    ):
        return True

    # ---------------------------------------------------------
    # No match
    # ---------------------------------------------------------

    return False


# =============================================================
# SEARCH PAPERSWITHCODE PAPERS
# =============================================================

def search_paperswithcode_papers(
    query: str | None = None,
    offset: int = 0,
    length: int = 20,
    *,
    page_size: int = 100,
    max_scan: int = 500,
) -> list[PapersWithCodePaper]:
    """
    Search PapersWithCode papers.

    The PapersWithCode archived dataset does not provide
    a native full-text search endpoint, so searching is
    performed locally while scanning dataset rows.

    Parameters
    ----------
    query:
        Optional search text.

    offset:
        Starting dataset offset.

    length:
        Maximum number of matching papers to return.

    page_size:
        Number of rows fetched per dataset request.

    max_scan:
        Maximum number of dataset rows examined.
    """

    # ---------------------------------------------------------
    # Validate query
    # ---------------------------------------------------------

    if query is not None and not query.strip():
        raise ValueError(
            "query must not be empty"
        )

    # ---------------------------------------------------------
    # Validate pagination
    # ---------------------------------------------------------

    if offset < 0:
        raise ValueError(
            "offset must be 0 or greater"
        )

    if length < 1:
        raise ValueError(
            "length must be at least 1"
        )

    if page_size < 1:
        raise ValueError(
            "page_size must be at least 1"
        )

    if max_scan < 1:
        raise ValueError(
            "max_scan must be at least 1"
        )

    # ---------------------------------------------------------
    # No query
    #
    # Normal dataset browsing.
    # ---------------------------------------------------------

    if query is None:
        return load_paperswithcode_papers(
            offset=offset,
            length=length,
        )

    # ---------------------------------------------------------
    # Search locally
    # ---------------------------------------------------------

    results: list[
        PapersWithCodePaper
    ] = []

    current_offset = offset
    scanned = 0

    while (
        len(results) < length
        and scanned < max_scan
    ):
        remaining = max_scan - scanned

        request_length = min(
            page_size,
            remaining,
        )

        papers = (
            load_paperswithcode_papers(
                offset=current_offset,
                length=request_length,
            )
        )

        # -----------------------------------------------------
        # Dataset exhausted
        # -----------------------------------------------------

        if not papers:
            break

        scanned += len(papers)

        # -----------------------------------------------------
        # Find matches
        # -----------------------------------------------------

        for paper in papers:
            if _paper_matches_query(
                paper,
                query,
            ):
                results.append(paper)

                if len(results) >= length:
                    break

        # -----------------------------------------------------
        # Move to next page
        # -----------------------------------------------------

        current_offset += len(papers)

        # -----------------------------------------------------
        # Fewer rows than requested means
        # we reached the end of the dataset.
        # -----------------------------------------------------

        if len(papers) < request_length:
            break

    return results


# =============================================================
# SEARCH PAPER ↔ CODE LINKS
# =============================================================

def search_paper_code_links(
    offset: int = 0,
    length: int = 20,
) -> list[PaperCodeLink]:
    """
    Load paper-to-code relationships from
    the PapersWithCode dataset.
    """

    # ---------------------------------------------------------
    # Validate pagination
    # ---------------------------------------------------------

    if offset < 0:
        raise ValueError(
            "offset must be 0 or greater"
        )

    if length < 1:
        raise ValueError(
            "length must be at least 1"
        )

    # ---------------------------------------------------------
    # Request dataset rows
    # ---------------------------------------------------------

    response = get_paperswithcode_rows(
        PAPER_CODE_LINKS_DATASET,
        offset=offset,
        length=length,
    )

    # ---------------------------------------------------------
    # Parse response
    # ---------------------------------------------------------

    data = response.json()

    return [
        parse_paper_code_link(
            row["row"]
        )
        for row in data.get(
            "rows",
            [],
        )
    ]
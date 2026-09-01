from datetime import datetime
import xml.etree.ElementTree as ET

from pydantic import BaseModel, HttpUrl

from src.clients.http import get
from src.models.research import ResearchItem


# =============================================================
# ARXIV CONSTANTS
# =============================================================

ARXIV_API_URL = (
    "https://export.arxiv.org/api/query"
)

ATOM_NS = (
    "http://www.w3.org/2005/Atom"
)

ARXIV_NS = (
    "http://arxiv.org/schemas/atom"
)


# =============================================================
# ARXIV RESEARCH PAPER MODEL
# =============================================================

class ResearchPaper(BaseModel):
    id: str
    title: str
    authors: list[str]

    abstract: str

    published: datetime
    updated: datetime

    categories: list[str]
    primary_category: str

    pdf_url: HttpUrl
    arxiv_url: HttpUrl

    source: str = "arxiv"


# =============================================================
# CONVERT ARXIV PAPER → RESEARCH ITEM
# =============================================================

def research_paper_to_item(
    paper: ResearchPaper,
) -> ResearchItem:
    """
    Convert an arXiv ResearchPaper into the common
    ResearchItem model.

    Mappings:

        arXiv title
            → ResearchItem.title

        arXiv abstract
            → ResearchItem.description

        arXiv authors
            → ResearchItem.authors

        arXiv published
            → ResearchItem.published

        arXiv updated
            → ResearchItem.updated

        arXiv categories
            → ResearchItem.tags

        arXiv URL
            → ResearchItem.url
    """

    return ResearchItem(
        # -----------------------------------------------------
        # Common fields
        # -----------------------------------------------------

        id=paper.id,

        title=paper.title,

        description=paper.abstract,

        authors=list(paper.authors),

        source=paper.source,

        url=paper.arxiv_url,

        # -----------------------------------------------------
        # Dates
        # -----------------------------------------------------

        published=paper.published,

        updated=paper.updated,

        # -----------------------------------------------------
        # Categories → Tags
        # -----------------------------------------------------

        tags=list(paper.categories),
    )


# =============================================================
# PARSE ARXIV API RESPONSE
# =============================================================

def parse_arxiv_response(
    xml_text: str,
) -> list[ResearchPaper]:
    """
    Parse an arXiv Atom XML response into a list of
    ResearchPaper objects.

    The parser extracts:

        ID
        title
        abstract
        authors
        published date
        updated date
        categories
        primary category
        arXiv URL
        PDF URL
    """

    # ---------------------------------------------------------
    # Remove leading whitespace
    # ---------------------------------------------------------

    # XML declarations such as:
    #
    # <?xml version="1.0" encoding="UTF-8"?>
    #
    # must appear at the beginning of the XML document.
    #
    # arXiv responses/tests may contain leading whitespace,
    # so strip it before parsing.

    xml_text = xml_text.lstrip("\ufeff \t\r\n")

    root = ET.fromstring(xml_text)

    papers: list[ResearchPaper] = []

    # ---------------------------------------------------------
    # Iterate through entries
    # ---------------------------------------------------------

    for entry in root.findall(
        f"{{{ATOM_NS}}}entry"
    ):

        # -----------------------------------------------------
        # ID
        # -----------------------------------------------------

        entry_id = entry.findtext(
            f"{{{ATOM_NS}}}id",
            "",
        ).strip()

        paper_id = entry_id.rsplit(
            "/",
            1,
        )[-1]

        # -----------------------------------------------------
        # Title
        # -----------------------------------------------------

        title = entry.findtext(
            f"{{{ATOM_NS}}}title",
            "",
        ).strip()

        # -----------------------------------------------------
        # Abstract
        # -----------------------------------------------------

        abstract = entry.findtext(
            f"{{{ATOM_NS}}}summary",
            "",
        ).strip()

        # -----------------------------------------------------
        # Published date
        # -----------------------------------------------------

        published = entry.findtext(
            f"{{{ATOM_NS}}}published",
            "",
        ).strip()

        # -----------------------------------------------------
        # Updated date
        # -----------------------------------------------------

        updated = entry.findtext(
            f"{{{ATOM_NS}}}updated",
            "",
        ).strip()

        # -----------------------------------------------------
        # Authors
        # -----------------------------------------------------

        authors = []

        for author in entry.findall(
            f"{{{ATOM_NS}}}author"
        ):
            name = author.findtext(
                f"{{{ATOM_NS}}}name",
                "",
            ).strip()

            if name:
                authors.append(name)

        # -----------------------------------------------------
        # Categories
        # -----------------------------------------------------

        categories = [
            category.attrib["term"]
            for category in entry.findall(
                f"{{{ATOM_NS}}}category"
            )
            if "term" in category.attrib
        ]

        # -----------------------------------------------------
        # Primary category
        # -----------------------------------------------------

        primary_category_element = entry.find(
            f"{{{ARXIV_NS}}}primary_category"
        )

        primary_category = ""

        if (
            primary_category_element is not None
            and "term" in primary_category_element.attrib
        ):
            primary_category = (
                primary_category_element.attrib[
                    "term"
                ]
            )

        # -----------------------------------------------------
        # Links
        # -----------------------------------------------------

        arxiv_url = ""
        pdf_url = ""

        for link in entry.findall(
            f"{{{ATOM_NS}}}link"
        ):

            link_type = link.attrib.get(
                "type"
            )

            link_rel = link.attrib.get(
                "rel"
            )

            href = link.attrib.get(
                "href",
                "",
            )

            # arXiv abstract/web page
            if link_rel == "alternate":
                arxiv_url = href

            # PDF
            elif link_type == "application/pdf":
                pdf_url = href

        # -----------------------------------------------------
        # Build ResearchPaper
        # -----------------------------------------------------

        paper = ResearchPaper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            published=published,
            updated=updated,
            categories=categories,
            primary_category=primary_category,
            pdf_url=pdf_url,
            arxiv_url=arxiv_url,
        )

        papers.append(paper)

    return papers


# =============================================================
# FETCH ARXIV XML
# =============================================================

def fetch_arxiv_xml(
    search_query: str,
    start: int = 0,
    max_results: int = 20,
) -> str:
    """
    Fetch raw XML data from the arXiv API.
    """

    # ---------------------------------------------------------
    # Validate start
    # ---------------------------------------------------------

    if start < 0:
        raise ValueError(
            "start must be 0 or greater"
        )

    # ---------------------------------------------------------
    # Validate max_results
    # ---------------------------------------------------------

    if max_results < 1:
        raise ValueError(
            "max_results must be at least 1"
        )

    # ---------------------------------------------------------
    # Request parameters
    # ---------------------------------------------------------

    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
    }

    # ---------------------------------------------------------
    # Request headers
    # ---------------------------------------------------------

    headers = {
        "User-Agent": (
            "SmartResearchDashboard/0.1 "
            "(research project)"
        )
    }

    # ---------------------------------------------------------
    # API request
    # ---------------------------------------------------------

    return get(
        ARXIV_API_URL,
        params=params,
        headers=headers,
        timeout=30.0,
    )


# =============================================================
# SEARCH ARXIV
# =============================================================

def search_arxiv(
    search_query: str,
    start: int = 0,
    max_results: int = 20,
) -> list[ResearchPaper]:
    """
    Search arXiv and return parsed research papers.
    """

    xml_text = fetch_arxiv_xml(
        search_query=search_query,
        start=start,
        max_results=max_results,
    )

    return parse_arxiv_response(
        xml_text
    )


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    papers = search_arxiv(
        search_query="cat:cs.AI",
        max_results=5,
    )

    for paper in papers:
        print(
            paper.id,
            "-",
            paper.title,
        )
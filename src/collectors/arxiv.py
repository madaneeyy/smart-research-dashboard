from src.models.research import ResearchItem
from datetime import datetime
import xml.etree.ElementTree as ET
from src.clients.http import get
from pydantic import BaseModel, HttpUrl



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
    
def research_paper_to_item(paper: ResearchPaper) -> ResearchItem:
    return ResearchItem(
        id=paper.id,
        title=paper.title,
        description=paper.abstract,
        authors=paper.authors,
        source=paper.source,
        url=paper.arxiv_url,
        published=paper.published,
        updated=paper.updated,
        tags=paper.categories,
    )


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"


def parse_arxiv_response(xml_text: str) -> list[ResearchPaper]:
    root = ET.fromstring(xml_text)

    papers = []

    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        entry_id = entry.findtext(
            f"{{{ATOM_NS}}}id",
            "",
        ).strip()

        title = entry.findtext(
            f"{{{ATOM_NS}}}title",
            "",
        ).strip()

        abstract = entry.findtext(
            f"{{{ATOM_NS}}}summary",
            "",
        ).strip()

        published = entry.findtext(
            f"{{{ATOM_NS}}}published",
            "",
        ).strip()

        updated = entry.findtext(
            f"{{{ATOM_NS}}}updated",
            "",
        ).strip()

        authors = [
            author.findtext(
                f"{{{ATOM_NS}}}name",
                "",
            ).strip()
            for author in entry.findall(
                f"{{{ATOM_NS}}}author"
            )
        ]

        categories = [
            category.attrib["term"]
            for category in entry.findall(
                f"{{{ATOM_NS}}}category"
            )
            if "term" in category.attrib
        ]

        primary_category_element = entry.find(
            "{http://arxiv.org/schemas/atom}primary_category"
        )

        primary_category = (
            primary_category_element.attrib["term"]
            if primary_category_element is not None
            and "term" in primary_category_element.attrib
            else ""
        )

        arxiv_url = ""
        pdf_url = ""

        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            link_type = link.attrib.get("type")
            link_rel = link.attrib.get("rel")
            href = link.attrib.get("href", "")

            if link_rel == "alternate":
                arxiv_url = href

            elif link_type == "application/pdf":
                pdf_url = href

        paper_id = entry_id.rsplit("/", 1)[-1]

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

def fetch_arxiv_xml(
    search_query: str,
    start: int = 0,
    max_results: int = 20,
) -> str:
    if start < 0:
        raise ValueError("start must be 0 or greater")

    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
    }

    headers = {
        "User-Agent": "SmartResearchDashboard/0.1 (research project)"
    }

    return get(
        ARXIV_API_URL,
        params=params,
        headers=headers,
        timeout=30.0,
    )

def search_arxiv(
    search_query: str,
    start: int = 0,
    max_results: int = 20,
) -> list[ResearchPaper]:
    xml_text = fetch_arxiv_xml(
        search_query=search_query,
        start=start,
        max_results=max_results,
    )

    return parse_arxiv_response(xml_text)

if __name__ == "__main__":
    papers = search_arxiv(
        search_query="cat:cs.AI",
        max_results=5,
    )

    for paper in papers:
        print(paper.id, "-", paper.title)
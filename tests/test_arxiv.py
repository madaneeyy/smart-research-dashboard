import httpx
import pytest

from src.collectors.arxiv import (
    ResearchPaper,
    fetch_arxiv_xml,
    parse_arxiv_response,
    research_paper_to_item,
    search_arxiv,
)


# =============================================================
# PARSING
# =============================================================


def test_parse_arxiv_response():
    xml = """
    <feed
        xmlns="http://www.w3.org/2005/Atom"
        xmlns:arxiv="http://arxiv.org/schemas/atom"
    >
        <entry>
            <id>https://arxiv.org/abs/1234.5678v1</id>

            <title>Test Research Paper</title>

            <updated>2026-08-01T12:00:00Z</updated>
            <published>2026-08-01T12:00:00Z</published>

            <link
                href="https://arxiv.org/abs/1234.5678v1"
                rel="alternate"
                type="text/html"
            />

            <link
                href="https://arxiv.org/pdf/1234.5678v1"
                rel="related"
                type="application/pdf"
                title="pdf"
            />

            <summary>
                This is a test abstract.
            </summary>

            <category term="cs.AI" />
            <category term="cs.CV" />

            <author>
                <name>Test Author</name>
            </author>

            <arxiv:primary_category term="cs.AI" />
        </entry>
    </feed>
    """

    papers = parse_arxiv_response(xml)

    assert len(papers) == 1

    paper = papers[0]

    # ---------------------------------------------------------
    # Basic fields
    # ---------------------------------------------------------

    assert paper.id == "1234.5678v1"

    assert paper.title == "Test Research Paper"

    assert paper.authors == [
        "Test Author",
    ]

    assert paper.abstract == (
        "This is a test abstract."
    )

    # ---------------------------------------------------------
    # Categories
    # ---------------------------------------------------------

    assert paper.categories == [
        "cs.AI",
        "cs.CV",
    ]

    assert paper.primary_category == "cs.AI"

    # ---------------------------------------------------------
    # URLs
    # ---------------------------------------------------------

    assert str(paper.pdf_url) == (
        "https://arxiv.org/pdf/1234.5678v1"
    )

    assert str(paper.arxiv_url) == (
        "https://arxiv.org/abs/1234.5678v1"
    )

    # ---------------------------------------------------------
    # Source
    # ---------------------------------------------------------

    assert paper.source == "arxiv"

    # ---------------------------------------------------------
    # Published date
    # ---------------------------------------------------------

    assert paper.published.year == 2026
    assert paper.published.month == 8
    assert paper.published.day == 1

    # ---------------------------------------------------------
    # Updated date
    # ---------------------------------------------------------

    assert paper.updated.year == 2026
    assert paper.updated.month == 8
    assert paper.updated.day == 1


# =============================================================
# MULTIPLE PAPERS
# =============================================================


def test_parse_multiple_arxiv_papers():
    xml = """
    <feed
        xmlns="http://www.w3.org/2005/Atom"
        xmlns:arxiv="http://arxiv.org/schemas/atom"
    >

        <entry>
            <id>https://arxiv.org/abs/1111.1111v1</id>

            <title>First Test Paper</title>

            <updated>2026-08-01T12:00:00Z</updated>
            <published>2026-08-01T12:00:00Z</published>

            <link
                href="https://arxiv.org/abs/1111.1111v1"
                rel="alternate"
                type="text/html"
            />

            <link
                href="https://arxiv.org/pdf/1111.1111v1"
                rel="related"
                type="application/pdf"
            />

            <summary>
                First test abstract.
            </summary>

            <category term="cs.AI" />

            <author>
                <name>Author One</name>
            </author>

            <arxiv:primary_category
                term="cs.AI"
            />
        </entry>

        <entry>
            <id>https://arxiv.org/abs/2222.2222v1</id>

            <title>Second Test Paper</title>

            <updated>2026-08-02T12:00:00Z</updated>
            <published>2026-08-02T12:00:00Z</published>

            <link
                href="https://arxiv.org/abs/2222.2222v1"
                rel="alternate"
                type="text/html"
            />

            <link
                href="https://arxiv.org/pdf/2222.2222v1"
                rel="related"
                type="application/pdf"
            />

            <summary>
                Second test abstract.
            </summary>

            <category term="cs.CV" />

            <author>
                <name>Author Two</name>
            </author>

            <arxiv:primary_category
                term="cs.CV"
            />
        </entry>

    </feed>
    """

    papers = parse_arxiv_response(xml)

    assert len(papers) == 2

    # ---------------------------------------------------------
    # First paper
    # ---------------------------------------------------------

    first = papers[0]

    assert first.id == "1111.1111v1"

    assert first.title == "First Test Paper"

    assert first.authors == [
        "Author One",
    ]

    assert first.abstract == (
        "First test abstract."
    )

    assert first.categories == [
        "cs.AI",
    ]

    assert first.primary_category == "cs.AI"

    assert first.published.year == 2026
    assert first.published.month == 8
    assert first.published.day == 1

    assert first.updated.year == 2026
    assert first.updated.month == 8
    assert first.updated.day == 1

    # ---------------------------------------------------------
    # Second paper
    # ---------------------------------------------------------

    second = papers[1]

    assert second.id == "2222.2222v1"

    assert second.title == "Second Test Paper"

    assert second.authors == [
        "Author Two",
    ]

    assert second.abstract == (
        "Second test abstract."
    )

    assert second.categories == [
        "cs.CV",
    ]

    assert second.primary_category == "cs.CV"

    assert second.published.year == 2026
    assert second.published.month == 8
    assert second.published.day == 2

    assert second.updated.year == 2026
    assert second.updated.month == 8
    assert second.updated.day == 2


# =============================================================
# VALIDATION
# =============================================================


def test_search_arxiv_rejects_negative_start():
    with pytest.raises(
        ValueError,
        match="start must be 0 or greater",
    ):
        search_arxiv(
            search_query="cat:cs.AI",
            start=-1,
        )


def test_search_arxiv_rejects_invalid_max_results():
    with pytest.raises(
        ValueError,
        match="max_results must be at least 1",
    ):
        search_arxiv(
            search_query="cat:cs.AI",
            max_results=0,
        )


# =============================================================
# HTTP ERROR
# =============================================================


def test_search_arxiv_raises_for_http_error(
    monkeypatch,
):
    def mock_get(*args, **kwargs):
        request = httpx.Request(
            "GET",
            "https://export.arxiv.org/api/query",
        )

        response = httpx.Response(
            status_code=500,
            request=request,
        )

        raise httpx.HTTPStatusError(
            "Server error",
            request=request,
            response=response,
        )

    monkeypatch.setattr(
        "src.collectors.arxiv.get",
        mock_get,
    )

    with pytest.raises(
        httpx.HTTPStatusError,
    ):
        search_arxiv(
            search_query="cat:cs.AI",
            max_results=1,
        )


# =============================================================
# SEARCH
# =============================================================


def test_search_arxiv_returns_papers(
    monkeypatch,
):
    xml = """
    <feed
        xmlns="http://www.w3.org/2005/Atom"
        xmlns:arxiv="http://arxiv.org/schemas/atom"
    >
        <entry>
            <id>https://arxiv.org/abs/9999.9999v1</id>

            <title>Mocked arXiv Paper</title>

            <updated>2026-08-01T12:00:00Z</updated>
            <published>2026-08-01T12:00:00Z</published>

            <link
                href="https://arxiv.org/abs/9999.9999v1"
                rel="alternate"
                type="text/html"
            />

            <link
                href="https://arxiv.org/pdf/9999.9999v1"
                rel="related"
                type="application/pdf"
            />

            <summary>
                Mocked paper abstract.
            </summary>

            <category term="cs.AI" />

            <author>
                <name>Mock Author</name>
            </author>

            <arxiv:primary_category
                term="cs.AI"
            />
        </entry>
    </feed>
    """

    def mock_get(*args, **kwargs):
        return xml

    monkeypatch.setattr(
        "src.collectors.arxiv.get",
        mock_get,
    )

    papers = search_arxiv(
        search_query="cat:cs.AI",
        max_results=1,
    )

    assert len(papers) == 1

    paper = papers[0]

    assert paper.id == "9999.9999v1"

    assert paper.title == (
        "Mocked arXiv Paper"
    )

    assert paper.authors == [
        "Mock Author",
    ]

    assert paper.abstract == (
        "Mocked paper abstract."
    )

    assert paper.categories == [
        "cs.AI",
    ]

    assert paper.primary_category == (
        "cs.AI"
    )

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------

    assert paper.published.year == 2026
    assert paper.published.month == 8
    assert paper.published.day == 1

    assert paper.updated.year == 2026
    assert paper.updated.month == 8
    assert paper.updated.day == 1

    # ---------------------------------------------------------
    # URLs
    # ---------------------------------------------------------

    assert str(paper.arxiv_url) == (
        "https://arxiv.org/abs/9999.9999v1"
    )

    assert str(paper.pdf_url) == (
        "https://arxiv.org/pdf/9999.9999v1"
    )


# =============================================================
# FETCH XML
# =============================================================


def test_fetch_arxiv_xml_returns_response_text(
    monkeypatch,
):
    xml = "test xml"

    def mock_get(*args, **kwargs):
        return xml

    monkeypatch.setattr(
        "src.collectors.arxiv.get",
        mock_get,
    )

    result = fetch_arxiv_xml(
        search_query="cat:cs.AI",
        max_results=1,
    )

    assert result == xml


# =============================================================
# CONVERSION TO RESEARCH ITEM
# =============================================================


def test_research_paper_to_item():
    paper = ResearchPaper(
        id="test-001",
        title="Test Paper",
        authors=[
            "Test Author",
        ],
        abstract="Test abstract.",
        published="2026-08-01T12:00:00Z",
        updated="2026-08-01T12:00:00Z",
        categories=[
            "cs.AI",
            "cs.CV",
        ],
        primary_category="cs.AI",
        pdf_url=(
            "https://arxiv.org/pdf/test-001"
        ),
        arxiv_url=(
            "https://arxiv.org/abs/test-001"
        ),
    )

    item = research_paper_to_item(
        paper
    )

    # ---------------------------------------------------------
    # Common fields
    # ---------------------------------------------------------

    assert item.id == "test-001"

    assert item.title == "Test Paper"

    assert item.description == (
        "Test abstract."
    )

    assert item.authors == [
        "Test Author",
    ]

    assert item.source == "arxiv"

    assert str(item.url) == (
        "https://arxiv.org/abs/test-001"
    )

    # ---------------------------------------------------------
    # Published date
    # ---------------------------------------------------------

    assert item.published is not None

    assert item.published.year == 2026
    assert item.published.month == 8
    assert item.published.day == 1

    # ---------------------------------------------------------
    # Updated date
    # ---------------------------------------------------------

    assert item.updated is not None

    assert item.updated.year == 2026
    assert item.updated.month == 8
    assert item.updated.day == 1

    # ---------------------------------------------------------
    # Tags
    # ---------------------------------------------------------

    assert item.tags == [
        "cs.AI",
        "cs.CV",
    ]


# =============================================================
# DATE MAPPING
# =============================================================


def test_research_paper_to_item_with_different_dates():
    paper = ResearchPaper(
        id="1234.5678",
        title="Test Research Paper",
        authors=[
            "Test Author",
        ],
        abstract=(
            "This is a test abstract."
        ),
        published=(
            "2026-08-01T00:00:00Z"
        ),
        updated=(
            "2026-08-02T00:00:00Z"
        ),
        categories=[
            "cs.AI",
            "cs.LG",
        ],
        primary_category="cs.AI",
        pdf_url=(
            "https://arxiv.org/pdf/1234.5678"
        ),
        arxiv_url=(
            "https://arxiv.org/abs/1234.5678"
        ),
    )

    item = research_paper_to_item(
        paper
    )

    # ---------------------------------------------------------
    # Basic fields
    # ---------------------------------------------------------

    assert item.id == "1234.5678"

    assert item.title == (
        "Test Research Paper"
    )

    assert item.description == (
        "This is a test abstract."
    )

    assert item.authors == [
        "Test Author",
    ]

    assert item.source == "arxiv"

    assert str(item.url) == (
        "https://arxiv.org/abs/1234.5678"
    )

    # ---------------------------------------------------------
    # published → published
    # ---------------------------------------------------------

    assert item.published is not None

    assert item.published.year == 2026
    assert item.published.month == 8
    assert item.published.day == 1

    # ---------------------------------------------------------
    # updated → updated
    # ---------------------------------------------------------

    assert item.updated is not None

    assert item.updated.year == 2026
    assert item.updated.month == 8
    assert item.updated.day == 2

    # ---------------------------------------------------------
    # Ensure updated is newer
    # ---------------------------------------------------------

    assert item.updated > item.published

    # ---------------------------------------------------------
    # Tags
    # ---------------------------------------------------------

    assert item.tags == [
        "cs.AI",
        "cs.LG",
    ]


# =============================================================
# MISSING OPTIONAL PRIMARY CATEGORY
# =============================================================


def test_parse_arxiv_response_without_primary_category():
    xml = """
    <feed
        xmlns="http://www.w3.org/2005/Atom"
        xmlns:arxiv="http://arxiv.org/schemas/atom"
    >
        <entry>
            <id>https://arxiv.org/abs/1234.0001v1</id>

            <title>Paper Without Primary Category</title>

            <updated>2026-08-01T12:00:00Z</updated>
            <published>2026-08-01T12:00:00Z</published>

            <link
                href="https://arxiv.org/abs/1234.0001v1"
                rel="alternate"
                type="text/html"
            />

            <link
                href="https://arxiv.org/pdf/1234.0001v1"
                rel="related"
                type="application/pdf"
            />

            <summary>
                Test abstract.
            </summary>

            <category term="cs.AI" />

            <author>
                <name>Test Author</name>
            </author>
        </entry>
    </feed>
    """

    papers = parse_arxiv_response(xml)

    assert len(papers) == 1

    paper = papers[0]

    assert paper.id == "1234.0001v1"

    assert paper.categories == [
        "cs.AI",
    ]

    assert paper.primary_category == ""


# =============================================================
# EMPTY FEED
# =============================================================


def test_parse_arxiv_response_empty_feed():
    xml = """
    <feed
        xmlns="http://www.w3.org/2005/Atom"
        xmlns:arxiv="http://arxiv.org/schemas/atom"
    >
    </feed>
    """

    papers = parse_arxiv_response(xml)

    assert papers == []
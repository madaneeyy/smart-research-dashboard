import pytest

from src.collectors.paperswithcode import (
    PapersWithCodePaper,
    parse_paperswithcode_paper,
    parse_paper_code_link,
    paperswithcode_paper_to_item,
    search_paper_code_links,
    search_paperswithcode_papers,
)


# =============================================================
# PARSE PAPERSWITHCODE PAPER
# =============================================================


def test_parse_paperswithcode_paper():
    data = {
        "paper_url": (
            "https://paperswithcode.com/paper/test-paper"
        ),
        "arxiv_id": "1234.5678",
        "nips_id": None,
        "openreview_id": None,
        "title": "Test Research Paper",
        "abstract": "This is a test abstract.",
        "short_abstract": "A short test abstract.",
        "url_abs": (
            "https://arxiv.org/abs/1234.5678"
        ),
        "url_pdf": (
            "https://arxiv.org/pdf/1234.5678"
        ),
        "proceeding": None,
        "authors": ["Test Author"],
        "tasks": ["image classification"],
        "date": "2026-08-01T00:00:00Z",
        "conference_url_abs": None,
        "conference_url_pdf": None,
        "conference": "Test Conference",
        "reproduces_paper": None,
        "methods": [
            {
                "name": "Test Method",
                "full_name": "Test Method Full Name",
                "description": "A test method.",
                "introduced_year": 2026,
                "source_title": "Test Research Paper",
                "source_url": (
                    "https://arxiv.org/abs/1234.5678"
                ),
            }
        ],
    }

    paper = parse_paperswithcode_paper(data)

    assert str(paper.paper_url) == (
        "https://paperswithcode.com/paper/test-paper"
    )

    assert paper.arxiv_id == "1234.5678"
    assert paper.title == "Test Research Paper"
    assert paper.abstract == "This is a test abstract."
    assert paper.short_abstract == (
        "A short test abstract."
    )

    assert str(paper.url_abs) == (
        "https://arxiv.org/abs/1234.5678"
    )

    assert str(paper.url_pdf) == (
        "https://arxiv.org/pdf/1234.5678"
    )

    assert paper.authors == ["Test Author"]
    assert paper.tasks == ["image classification"]

    # ---------------------------------------------------------
    # Date
    # ---------------------------------------------------------

    assert paper.date is not None
    assert paper.date.year == 2026
    assert paper.date.month == 8
    assert paper.date.day == 1

    # ---------------------------------------------------------
    # Conference
    # ---------------------------------------------------------

    assert paper.conference == "Test Conference"

    # ---------------------------------------------------------
    # Methods
    # ---------------------------------------------------------

    assert len(paper.methods) == 1

    method = paper.methods[0]

    assert method.name == "Test Method"

    assert method.full_name == (
        "Test Method Full Name"
    )

    assert method.description == (
        "A test method."
    )

    assert method.introduced_year == 2026

    assert method.source_title == (
        "Test Research Paper"
    )

    assert str(method.source_url) == (
        "https://arxiv.org/abs/1234.5678"
    )


# =============================================================
# PARSE PAPER CODE LINK
# =============================================================


def test_parse_paper_code_link():
    data = {
        "paper_url": (
            "https://paperswithcode.com/paper/test-paper"
        ),
        "paper_title": "Test Research Paper",
        "paper_arxiv_id": "1234.5678",
        "paper_url_abs": (
            "https://arxiv.org/abs/1234.5678"
        ),
        "paper_url_pdf": (
            "https://arxiv.org/pdf/1234.5678"
        ),
        "repo_url": (
            "https://github.com/test-user/test-repo"
        ),
        "is_official": True,
        "mentioned_in_paper": True,
        "mentioned_in_github": False,
        "framework": "PyTorch",
    }

    link = parse_paper_code_link(data)

    assert str(link.paper_url) == (
        "https://paperswithcode.com/paper/test-paper"
    )

    assert link.paper_title == (
        "Test Research Paper"
    )

    assert link.paper_arxiv_id == "1234.5678"

    assert str(link.paper_url_abs) == (
        "https://arxiv.org/abs/1234.5678"
    )

    assert str(link.paper_url_pdf) == (
        "https://arxiv.org/pdf/1234.5678"
    )

    assert str(link.repo_url) == (
        "https://github.com/test-user/test-repo"
    )

    assert link.is_official is True
    assert link.mentioned_in_paper is True
    assert link.mentioned_in_github is False
    assert link.framework == "PyTorch"


# =============================================================
# SEARCH PAPERSWITHCODE PAPERS
# =============================================================


def test_search_paperswithcode_papers(monkeypatch):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": (
                        "https://paperswithcode.com/"
                        "paper/test-paper"
                    ),
                    "arxiv_id": "1234.5678",
                    "title": "Test Research Paper",
                    "abstract": "Test abstract.",
                    "short_abstract": None,
                    "url_abs": (
                        "https://arxiv.org/abs/"
                        "1234.5678"
                    ),
                    "url_pdf": (
                        "https://arxiv.org/pdf/"
                        "1234.5678"
                    ),
                    "authors": ["Test Author"],
                    "tasks": ["machine learning"],
                    "date": "2026-08-01T00:00:00Z",
                    "conference": None,
                    "methods": [],
                }
            }
        ]
    }

    class MockResponse:
        def json(self):
            return response_data

    def mock_get_rows(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(
        "src.collectors.paperswithcode.get_paperswithcode_rows",
        mock_get_rows,
    )

    papers = search_paperswithcode_papers(
        offset=0,
        length=1,
    )

    assert len(papers) == 1

    paper = papers[0]

    assert paper.title == "Test Research Paper"
    assert paper.arxiv_id == "1234.5678"

    # ---------------------------------------------------------
    # Date
    # ---------------------------------------------------------

    assert paper.date is not None
    assert paper.date.year == 2026
    assert paper.date.month == 8
    assert paper.date.day == 1


# =============================================================
# SEARCH PAPER CODE LINKS
# =============================================================


def test_search_paper_code_links(monkeypatch):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": (
                        "https://paperswithcode.com/"
                        "paper/test-paper"
                    ),
                    "paper_title": "Test Research Paper",
                    "paper_arxiv_id": "1234.5678",
                    "paper_url_abs": (
                        "https://arxiv.org/abs/"
                        "1234.5678"
                    ),
                    "paper_url_pdf": (
                        "https://arxiv.org/pdf/"
                        "1234.5678"
                    ),
                    "repo_url": (
                        "https://github.com/"
                        "test-user/test-repo"
                    ),
                    "is_official": True,
                    "mentioned_in_paper": True,
                    "mentioned_in_github": False,
                    "framework": "PyTorch",
                }
            }
        ]
    }

    class MockResponse:
        def json(self):
            return response_data

    def mock_get_rows(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(
        "src.collectors.paperswithcode.get_paperswithcode_rows",
        mock_get_rows,
    )

    links = search_paper_code_links(
        offset=0,
        length=1,
    )

    assert len(links) == 1

    link = links[0]

    assert link.paper_title == (
        "Test Research Paper"
    )

    assert str(link.repo_url) == (
        "https://github.com/"
        "test-user/test-repo"
    )

    assert link.is_official is True


# =============================================================
# PAPERSWITHCODE -> RESEARCH ITEM
# =============================================================


def test_paperswithcode_paper_to_item():
    paper = PapersWithCodePaper(
        paper_url=(
            "https://paperswithcode.com/"
            "paper/test-paper"
        ),
        arxiv_id="1234.5678",
        title="Test Research Paper",
        abstract="This is a test abstract.",
        short_abstract=(
            "A short test abstract."
        ),
        url_abs=(
            "https://arxiv.org/abs/1234.5678"
        ),
        url_pdf=(
            "https://arxiv.org/pdf/1234.5678"
        ),
        authors=["Test Author"],
        tasks=["machine learning"],
        date="2026-08-01T00:00:00Z",
        conference=None,
        methods=[],
    )

    item = paperswithcode_paper_to_item(
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
        "Test Author"
    ]

    assert item.source == (
        "paperswithcode"
    )

    assert str(item.url) == (
        "https://paperswithcode.com/"
        "paper/test-paper"
    )

    # ---------------------------------------------------------
    # Date mapping
    # ---------------------------------------------------------

    assert item.published is not None

    assert item.published.year == 2026
    assert item.published.month == 8
    assert item.published.day == 1

    # PapersWithCode currently has
    # no separate updated date.

    assert item.updated is None

    # ---------------------------------------------------------
    # Tags
    # ---------------------------------------------------------

    assert item.tags == [
        "machine learning"
    ]

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    assert item.metadata["arxiv_id"] == (
        "1234.5678"
    )

    assert item.metadata["short_abstract"] == (
        "A short test abstract."
    )

    assert item.metadata["url_abs"] == (
        "https://arxiv.org/abs/1234.5678"
    )

    assert item.metadata["url_pdf"] == (
        "https://arxiv.org/pdf/1234.5678"
    )

    assert item.metadata["conference"] is None

    assert item.metadata["proceeding"] is None

    assert item.metadata["reproduces_paper"] is None

    assert item.metadata["methods"] == []


# =============================================================
# DATE MAPPING TEST
# =============================================================


def test_paperswithcode_date_maps_to_published():
    """
    Explicitly verifies:

        PapersWithCode date
                    ↓
        PapersWithCodePaper.date
                    ↓
        ResearchItem.published
    """

    paper = PapersWithCodePaper(
        paper_url=(
            "https://paperswithcode.com/"
            "paper/date-test"
        ),
        arxiv_id="9999.9999",
        title="Date Mapping Test",
        abstract="Testing date mapping.",
        short_abstract=None,
        url_abs=(
            "https://arxiv.org/abs/9999.9999"
        ),
        url_pdf=(
            "https://arxiv.org/pdf/9999.9999"
        ),
        authors=["Test Author"],
        tasks=["machine learning"],
        date="2026-08-10T15:30:00Z",
        conference=None,
        methods=[],
    )

    item = paperswithcode_paper_to_item(
        paper
    )

    assert item.published is not None

    assert item.published.year == 2026
    assert item.published.month == 8
    assert item.published.day == 10

    assert item.published.hour == 15
    assert item.published.minute == 30

    assert item.updated is None


# =============================================================
# VALIDATION TESTS
# =============================================================


def test_search_paperswithcode_rejects_negative_offset():
    with pytest.raises(
        ValueError,
        match="offset must be 0 or greater",
    ):
        search_paperswithcode_papers(
            offset=-1
        )


def test_search_paperswithcode_rejects_invalid_length():
    with pytest.raises(
        ValueError,
        match="length must be at least 1",
    ):
        search_paperswithcode_papers(
            length=0
        )


def test_search_paperswithcode_rejects_invalid_page_size():
    with pytest.raises(
        ValueError,
        match="page_size must be at least 1",
    ):
        search_paperswithcode_papers(
            query="machine learning",
            page_size=0,
        )


def test_search_paperswithcode_rejects_invalid_max_scan():
    with pytest.raises(
        ValueError,
        match="max_scan must be at least 1",
    ):
        search_paperswithcode_papers(
            query="machine learning",
            max_scan=0,
        )


def test_search_paperswithcode_rejects_empty_query():
    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        search_paperswithcode_papers(
            query="   "
        )


# =============================================================
# QUERY MATCHING TESTS
# =============================================================


def test_search_paperswithcode_matches_title(
    monkeypatch,
):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": (
                        "https://paperswithcode.com/"
                        "paper/title-match"
                    ),
                    "arxiv_id": "1111.1111",
                    "title": (
                        "Deep Learning for Vision"
                    ),
                    "abstract": "An abstract.",
                    "short_abstract": None,
                    "authors": [],
                    "tasks": [],
                    "date": (
                        "2026-08-01T00:00:00Z"
                    ),
                    "methods": [],
                }
            }
        ]
    }

    class MockResponse:
        def json(self):
            return response_data

    monkeypatch.setattr(
        "src.collectors.paperswithcode.get_paperswithcode_rows",
        lambda *args, **kwargs: MockResponse(),
    )

    results = search_paperswithcode_papers(
        query="deep learning",
        length=1,
    )

    assert len(results) == 1

    assert results[0].title == (
        "Deep Learning for Vision"
    )


def test_search_paperswithcode_matches_task(
    monkeypatch,
):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": (
                        "https://paperswithcode.com/"
                        "paper/task-match"
                    ),
                    "arxiv_id": "2222.2222",
                    "title": "Test Paper",
                    "abstract": "An abstract.",
                    "short_abstract": None,
                    "authors": [],
                    "tasks": [
                        "image classification"
                    ],
                    "date": (
                        "2026-08-01T00:00:00Z"
                    ),
                    "methods": [],
                }
            }
        ]
    }

    class MockResponse:
        def json(self):
            return response_data

    monkeypatch.setattr(
        "src.collectors.paperswithcode.get_paperswithcode_rows",
        lambda *args, **kwargs: MockResponse(),
    )

    results = search_paperswithcode_papers(
        query="classification",
        length=1,
    )

    assert len(results) == 1

    assert results[0].tasks == [
        "image classification"
    ]


def test_search_paperswithcode_matches_author(
    monkeypatch,
):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": (
                        "https://paperswithcode.com/"
                        "paper/author-match"
                    ),
                    "arxiv_id": "3333.3333",
                    "title": "Test Paper",
                    "abstract": "An abstract.",
                    "short_abstract": None,
                    "authors": [
                        "Geoffrey Hinton"
                    ],
                    "tasks": [],
                    "date": (
                        "2026-08-01T00:00:00Z"
                    ),
                    "methods": [],
                }
            }
        ]
    }

    class MockResponse:
        def json(self):
            return response_data

    monkeypatch.setattr(
        "src.collectors.paperswithcode.get_paperswithcode_rows",
        lambda *args, **kwargs: MockResponse(),
    )

    results = search_paperswithcode_papers(
        query="hinton",
        length=1,
    )

    assert len(results) == 1

    assert results[0].authors == [
        "Geoffrey Hinton"
    ]


def test_search_paperswithcode_matches_method(
    monkeypatch,
):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": (
                        "https://paperswithcode.com/"
                        "paper/method-match"
                    ),
                    "arxiv_id": "4444.4444",
                    "title": "Test Paper",
                    "abstract": "An abstract.",
                    "short_abstract": None,
                    "authors": [],
                    "tasks": [],
                    "date": (
                        "2026-08-01T00:00:00Z"
                    ),
                    "methods": [
                        {
                            "name": "Transformer",
                            "full_name": (
                                "Transformer Architecture"
                            ),
                            "description": (
                                "A neural architecture."
                            ),
                        }
                    ],
                }
            }
        ]
    }

    class MockResponse:
        def json(self):
            return response_data

    monkeypatch.setattr(
        "src.collectors.paperswithcode.get_paperswithcode_rows",
        lambda *args, **kwargs: MockResponse(),
    )

    results = search_paperswithcode_papers(
        query="transformer",
        length=1,
    )

    assert len(results) == 1

    assert results[0].methods[0].name == (
        "Transformer"
    )


# =============================================================
# SEARCH WITHOUT QUERY
# =============================================================


def test_search_paperswithcode_without_query(
    monkeypatch,
):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": (
                        "https://paperswithcode.com/"
                        "paper/no-query"
                    ),
                    "arxiv_id": "5555.5555",
                    "title": "Browsing Paper",
                    "abstract": "An abstract.",
                    "short_abstract": None,
                    "authors": [],
                    "tasks": [],
                    "date": (
                        "2026-08-01T00:00:00Z"
                    ),
                    "methods": [],
                }
            }
        ]
    }

    class MockResponse:
        def json(self):
            return response_data

    monkeypatch.setattr(
        "src.collectors.paperswithcode.get_paperswithcode_rows",
        lambda *args, **kwargs: MockResponse(),
    )

    results = search_paperswithcode_papers(
        query=None,
        offset=0,
        length=1,
    )

    assert len(results) == 1

    assert results[0].title == (
        "Browsing Paper"
    )


# =============================================================
# NO MATCH
# =============================================================


def test_search_paperswithcode_returns_empty_when_no_match(
    monkeypatch,
):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": (
                        "https://paperswithcode.com/"
                        "paper/no-match"
                    ),
                    "arxiv_id": "6666.6666",
                    "title": "Unrelated Paper",
                    "abstract": (
                        "Something completely different."
                    ),
                    "short_abstract": None,
                    "authors": [
                        "Another Author"
                    ],
                    "tasks": [
                        "computer vision"
                    ],
                    "date": (
                        "2026-08-01T00:00:00Z"
                    ),
                    "methods": [],
                }
            }
        ]
    }

    class MockResponse:
        def json(self):
            return response_data

    monkeypatch.setattr(
        "src.collectors.paperswithcode.get_paperswithcode_rows",
        lambda *args, **kwargs: MockResponse(),
    )

    results = search_paperswithcode_papers(
        query="quantum computing",
        length=1,
    )

    assert results == []
from src.collectors.paperswithcode import (
    PapersWithCodePaper,
    parse_paperswithcode_paper,
    parse_paper_code_link,
    paperswithcode_paper_to_item,
    search_paper_code_links,
    search_paperswithcode_papers,
)


def test_parse_paperswithcode_paper():
    data = {
        "paper_url": "https://paperswithcode.com/paper/test-paper",
        "arxiv_id": "1234.5678",
        "nips_id": None,
        "openreview_id": None,
        "title": "Test Research Paper",
        "abstract": "This is a test abstract.",
        "short_abstract": "A short test abstract.",
        "url_abs": "https://arxiv.org/abs/1234.5678",
        "url_pdf": "https://arxiv.org/pdf/1234.5678",
        "proceeding": None,
        "authors": ["Test Author"],
        "tasks": ["image classification"],
        "date": "2026-08-01T00:00:00",
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
                "source_url": "https://arxiv.org/abs/1234.5678",
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
    assert paper.short_abstract == "A short test abstract."

    assert str(paper.url_abs) == (
        "https://arxiv.org/abs/1234.5678"
    )

    assert str(paper.url_pdf) == (
        "https://arxiv.org/pdf/1234.5678"
    )

    assert paper.authors == ["Test Author"]
    assert paper.tasks == ["image classification"]

    assert paper.date.year == 2026
    assert paper.conference == "Test Conference"

    assert len(paper.methods) == 1

    method = paper.methods[0]

    assert method.name == "Test Method"
    assert method.full_name == "Test Method Full Name"
    assert method.description == "A test method."
    assert method.introduced_year == 2026
    assert method.source_title == "Test Research Paper"
    assert str(method.source_url) == (
        "https://arxiv.org/abs/1234.5678"
    )


def test_parse_paper_code_link():
    data = {
        "paper_url": "https://paperswithcode.com/paper/test-paper",
        "paper_title": "Test Research Paper",
        "paper_arxiv_id": "1234.5678",
        "paper_url_abs": "https://arxiv.org/abs/1234.5678",
        "paper_url_pdf": "https://arxiv.org/pdf/1234.5678",
        "repo_url": "https://github.com/test-user/test-repo",
        "is_official": True,
        "mentioned_in_paper": True,
        "mentioned_in_github": False,
        "framework": "PyTorch",
    }

    link = parse_paper_code_link(data)

    assert str(link.paper_url) == (
        "https://paperswithcode.com/paper/test-paper"
    )

    assert link.paper_title == "Test Research Paper"
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


def test_search_paperswithcode_papers(monkeypatch):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": "https://paperswithcode.com/paper/test-paper",
                    "arxiv_id": "1234.5678",
                    "title": "Test Research Paper",
                    "abstract": "Test abstract.",
                    "short_abstract": None,
                    "url_abs": "https://arxiv.org/abs/1234.5678",
                    "url_pdf": "https://arxiv.org/pdf/1234.5678",
                    "authors": ["Test Author"],
                    "tasks": ["machine learning"],
                    "date": "2026-08-01T00:00:00",
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
    assert papers[0].title == "Test Research Paper"
    assert papers[0].arxiv_id == "1234.5678"


def test_search_paper_code_links(monkeypatch):
    response_data = {
        "rows": [
            {
                "row": {
                    "paper_url": "https://paperswithcode.com/paper/test-paper",
                    "paper_title": "Test Research Paper",
                    "paper_arxiv_id": "1234.5678",
                    "paper_url_abs": "https://arxiv.org/abs/1234.5678",
                    "paper_url_pdf": "https://arxiv.org/pdf/1234.5678",
                    "repo_url": "https://github.com/test-user/test-repo",
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
    assert links[0].paper_title == "Test Research Paper"
    assert str(links[0].repo_url) == (
        "https://github.com/test-user/test-repo"
    )
    assert links[0].is_official is True

def test_paperswithcode_paper_to_item():
    paper = PapersWithCodePaper(
        paper_url="https://paperswithcode.com/paper/test-paper",
        arxiv_id="1234.5678",
        title="Test Research Paper",
        abstract="This is a test abstract.",
        short_abstract="A short test abstract.",
        url_abs="https://arxiv.org/abs/1234.5678",
        url_pdf="https://arxiv.org/pdf/1234.5678",
        authors=["Test Author"],
        tasks=["machine learning"],
        date="2026-08-01T00:00:00Z",
        conference=None,
        methods=[],
    )

    item = paperswithcode_paper_to_item(paper)

    assert item.id == "1234.5678"
    assert item.title == "Test Research Paper"
    assert item.description == "This is a test abstract."
    assert item.authors == ["Test Author"]
    assert item.source == "paperswithcode"
    assert str(item.url) == (
        "https://paperswithcode.com/paper/test-paper"
    )
    assert item.published.year == 2026
    assert item.updated is None
    assert item.tags == ["machine learning"]
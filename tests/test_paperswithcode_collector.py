from src.collectors.paperswithcode import (
    load_paperswithcode_papers,
)


def test_load_paperswithcode_papers(monkeypatch):
    response_data = {
        "rows": [
            {
                "row_idx": 0,
                "row": {
                    "paper_url": "https://paperswithcode.com/paper/test-paper",
                    "arxiv_id": "1234.5678",
                    "title": "Test Research Paper",
                    "abstract": "A test abstract.",
                    "short_abstract": None,
                    "url_abs": "https://arxiv.org/abs/1234.5678",
                    "url_pdf": "https://arxiv.org/pdf/1234.5678",
                    "authors": ["Test Author"],
                    "tasks": ["image classification"],
                    "date": "2026-08-01T00:00:00",
                    "conference": "Test Conference",
                    "methods": [],
                },
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

    papers = load_paperswithcode_papers(
        offset=0,
        length=1,
    )

    assert len(papers) == 1

    paper = papers[0]

    assert paper.title == "Test Research Paper"
    assert paper.arxiv_id == "1234.5678"
    assert paper.authors == ["Test Author"]
    assert paper.tasks == ["image classification"]
    assert paper.conference == "Test Conference"
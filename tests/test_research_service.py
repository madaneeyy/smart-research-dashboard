import pytest

from src.collectors.arxiv import ResearchPaper
from src.collectors.github import GitHubRepository
from src.collectors.huggingface import HuggingFaceModel
from src.collectors.paperswithcode import PapersWithCodePaper
from src.models.research import ResearchItem
from src.services.research import ResearchService


def test_search_returns_research_items_from_all_sources(monkeypatch):
    arxiv_paper = ResearchPaper(
        id="arxiv-123",
        title="ArXiv Paper",
        authors=["Author A"],
        abstract="An ArXiv paper.",
        published="2026-08-01T00:00:00Z",
        updated="2026-08-01T00:00:00Z",
        categories=["machine learning"],
        primary_category="cs.LG",
        pdf_url="https://arxiv.org/pdf/arxiv-123",
        arxiv_url="https://arxiv.org/abs/arxiv-123",
    )

    github_repository = GitHubRepository(
        id=123456,
        name="test-repo",
        full_name="user/test-repo",
        description="A GitHub repository.",
        html_url="https://github.com/user/test-repo",
        stars=100,
        forks=20,
        language="Python",
        topics=["machine-learning"],
        created_at="2026-08-01T00:00:00Z",
        updated_at="2026-08-01T00:00:00Z",
    )

    paperswithcode_paper = PapersWithCodePaper(
        paper_url="https://paperswithcode.com/paper/test-paper",
        arxiv_id="1234.5678",
        title="PapersWithCode Paper",
        abstract="A PapersWithCode paper.",
        authors=["Author B"],
        tasks=["machine learning"],
        date="2026-08-01T00:00:00Z",
        methods=[],
    )

    huggingface_model = HuggingFaceModel(
        model_id="test-user/test-model",
        author="test-user",
        pipeline_tag="text-classification",
        downloads=1000,
        likes=50,
        library_name="transformers",
        tags=["nlp"],
        created_at="2026-08-01T00:00:00Z",
        last_modified="2026-08-01T00:00:00Z",
        url="https://huggingface.co/test-user/test-model",
    )

    def mock_arxiv(*args, **kwargs):
        return [arxiv_paper]

    def mock_github(*args, **kwargs):
        return [github_repository]

    def mock_paperswithcode(*args, **kwargs):
        return [paperswithcode_paper]

    def mock_huggingface(*args, **kwargs):
        return [huggingface_model]

    monkeypatch.setattr(
        "src.services.research.search_arxiv",
        mock_arxiv,
    )

    monkeypatch.setattr(
        "src.services.research.search_github_repositories",
        mock_github,
    )

    monkeypatch.setattr(
        "src.services.research.search_paperswithcode_papers",
        mock_paperswithcode,
    )

    monkeypatch.setattr(
        "src.services.research.search_huggingface_models",
        mock_huggingface,
    )

    service = ResearchService()

    results = service.search("machine learning")

    assert len(results) == 4

    assert all(
        isinstance(result, ResearchItem)
        for result in results
    )

    assert results[0].source == "arxiv"
    assert results[0].title == "ArXiv Paper"

    assert results[1].source == "github"
    assert results[1].title == "test-repo"

    assert results[2].source == "paperswithcode"
    assert results[2].title == "PapersWithCode Paper"

    assert results[3].source == "huggingface"
    assert results[3].title == "test-user/test-model"

def test_search_rejects_empty_query():
    service = ResearchService()

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        service.search("")
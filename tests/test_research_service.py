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

    # Search results are now ranked by relevance,
    # so do not assume source order.
    assert {result.source for result in results} == {
        "arxiv",
        "github",
        "paperswithcode",
        "huggingface",
    }

    assert {result.title for result in results} == {
        "ArXiv Paper",
        "test-repo",
        "PapersWithCode Paper",
        "test-user/test-model",
    }


def test_search_rejects_empty_query():
    service = ResearchService()

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        service.search("")


def test_search_rejects_whitespace_query():
    service = ResearchService()

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        service.search("   ")


def test_search_continues_when_one_source_fails(monkeypatch):
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
        raise RuntimeError("GitHub is unavailable")

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

    assert len(results) == 3

    assert all(
        isinstance(result, ResearchItem)
        for result in results
    )

    # GitHub failed, but the other three sources succeeded.
    assert {result.source for result in results} == {
        "arxiv",
        "paperswithcode",
        "huggingface",
    }

    assert {result.title for result in results} == {
        "ArXiv Paper",
        "PapersWithCode Paper",
        "test-user/test-model",
    }


def test_search_removes_duplicate_results(monkeypatch):
    arxiv_paper = ResearchPaper(
        id="1234.5678",
        title="Same Research Paper",
        authors=["Author A"],
        abstract="A research paper.",
        published="2026-08-01T00:00:00Z",
        updated="2026-08-01T00:00:00Z",
        categories=["machine learning"],
        primary_category="cs.LG",
        pdf_url="https://arxiv.org/pdf/1234.5678",
        arxiv_url="https://arxiv.org/abs/1234.5678",
    )

    paperswithcode_paper = PapersWithCodePaper(
        paper_url="https://paperswithcode.com/paper/same-research-paper",
        arxiv_id="1234.5678",
        title="Same Research Paper",
        abstract="A research paper.",
        authors=["Author A"],
        tasks=["machine learning"],
        date="2026-08-01T00:00:00Z",
        methods=[],
    )

    def mock_arxiv(*args, **kwargs):
        return [arxiv_paper]

    def mock_github(*args, **kwargs):
        return []

    def mock_paperswithcode(*args, **kwargs):
        return [paperswithcode_paper]

    def mock_huggingface(*args, **kwargs):
        return []

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

    assert len(results) == 1

    assert results[0].id == "1234.5678"
    assert results[0].title == "Same Research Paper"
    assert results[0].source == "arxiv"


def test_search_uses_per_source_limits(monkeypatch):
    calls = {}

    def mock_arxiv(*args, **kwargs):
        calls["arxiv"] = kwargs
        return []

    def mock_github(*args, **kwargs):
        calls["github"] = kwargs
        return []

    def mock_paperswithcode(*args, **kwargs):
        calls["paperswithcode"] = kwargs
        return []

    def mock_huggingface(*args, **kwargs):
        calls["huggingface"] = kwargs
        return []

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

    results = service.search(
        "transformers",
        arxiv_limit=5,
        github_limit=10,
        paperswithcode_limit=3,
        huggingface_limit=7,
    )

    assert results == []

    assert calls["arxiv"]["max_results"] == 5
    assert calls["github"]["per_page"] == 10
    assert calls["paperswithcode"]["length"] == 3
    assert calls["huggingface"]["limit"] == 7


def test_search_can_filter_to_one_source(monkeypatch):
    calls = []

    def mock_arxiv(*args, **kwargs):
        calls.append("arxiv")
        return []

    def mock_github(*args, **kwargs):
        calls.append("github")
        return []

    def mock_paperswithcode(*args, **kwargs):
        calls.append("paperswithcode")
        return []

    def mock_huggingface(*args, **kwargs):
        calls.append("huggingface")
        return []

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

    results = service.search(
        "transformers",
        sources=["arxiv"],
    )

    assert results == []
    assert calls == ["arxiv"]


def test_search_can_filter_to_multiple_sources(monkeypatch):
    calls = []

    def mock_arxiv(*args, **kwargs):
        calls.append("arxiv")
        return []

    def mock_github(*args, **kwargs):
        calls.append("github")
        return []

    def mock_paperswithcode(*args, **kwargs):
        calls.append("paperswithcode")
        return []

    def mock_huggingface(*args, **kwargs):
        calls.append("huggingface")
        return []

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

    results = service.search(
        "transformers",
        sources=["arxiv", "github"],
    )

    assert results == []
    assert calls == ["arxiv", "github"]


def test_search_rejects_unknown_source():
    service = ResearchService()

    with pytest.raises(
        ValueError,
        match="unknown source",
    ):
        service.search(
            "transformers",
            sources=["not-a-source"],
        )
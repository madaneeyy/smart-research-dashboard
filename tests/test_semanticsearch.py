from src.models.research import ResearchItem
from src.services.research import ResearchService


def make_result(
    result_id: str,
    title: str,
    description: str,
) -> ResearchItem:

    return ResearchItem(
        id=result_id,
        title=title,
        description=description,
        authors=["Test Author"],
        source="arxiv",
        url=f"https://arxiv.org/abs/{result_id}",
    )


def test_invalid_search_mode():

    service = ResearchService()

    try:

        service.search(
            "machine learning",
            sources=[],
            search_mode="invalid",
        )

    except ValueError as exc:

        assert "unknown search mode" in str(exc)

    else:

        raise AssertionError(
            "Expected ValueError"
        )


def test_keyword_search_mode():

    service = ResearchService()

    results = [
        make_result(
            "1",
            "Machine Learning",
            "Machine learning research",
        ),
    ]

    ranked = service._rank_by_search_mode(
        query="machine learning",
        results=results,
        search_mode="keyword",
    )

    assert len(ranked) == 1


def test_semantic_search_mode():

    service = ResearchService()

    results = [
        make_result(
            "1",
            "Deep Learning",
            "Neural networks and representation learning",
        ),
        make_result(
            "2",
            "Cooking Recipes",
            "Different recipes for cooking food",
        ),
    ]

    ranked = service._rank_by_search_mode(
        query="neural network research",
        results=results,
        search_mode="semantic",
    )

    assert len(ranked) == 2

    assert ranked[0].id == "1"


def test_hybrid_search_mode():

    service = ResearchService()

    results = [
        make_result(
            "1",
            "Deep Learning",
            "Neural networks and representation learning",
        ),
        make_result(
            "2",
            "Cooking Recipes",
            "Different recipes for cooking food",
        ),
    ]

    ranked = service._rank_by_search_mode(
        query="neural network research",
        results=results,
        search_mode="hybrid",
    )

    assert len(ranked) == 2

    assert ranked[0].id == "1"
from src.models.research import ResearchItem
from src.services.relevance import RelevanceScorer


def make_item(
    title: str,
    description: str = "",
    tags: list[str] | None = None,
) -> ResearchItem:
    return ResearchItem(
        id=title,
        title=title,
        description=description,
        authors=[],
        source="test",
        url=f"https://example.com/{title.replace(' ', '-')}",
        published=None,
        updated=None,
        tags=tags or [],
    )


def test_exact_title_match_scores_highest():
    item = make_item(
        title="transformer",
    )

    score = RelevanceScorer.score(
        "transformer",
        item,
    )

    assert score == 10.0


def test_title_match_scores_higher_than_description_match():
    title_item = make_item(
        title="Transformer Models",
        description="A research paper.",
    )

    description_item = make_item(
        title="Machine Learning",
        description="This paper studies transformer models.",
    )

    title_score = RelevanceScorer.score(
        "transformer",
        title_item,
    )

    description_score = RelevanceScorer.score(
        "transformer",
        description_item,
    )

    assert title_score > description_score


def test_tag_match_adds_relevance():
    item = make_item(
        title="Machine Learning",
        tags=["transformer"],
    )

    score = RelevanceScorer.score(
        "transformer",
        item,
    )

    assert score >= 4.0


def test_rank_orders_most_relevant_first():
    weak = make_item(
        title="Computer Vision",
        description="A study of images.",
    )

    medium = make_item(
        title="Machine Learning",
        description="This paper discusses transformer models.",
    )

    strong = make_item(
        title="Transformer Models",
        description="Transformer architectures for machine learning.",
    )

    results = RelevanceScorer.rank(
        "transformer",
        [weak, medium, strong],
    )

    assert results[0].title == "Transformer Models"
    assert results[1].title == "Machine Learning"
    assert results[2].title == "Computer Vision"


def test_empty_query_is_rejected():
    item = make_item(
        title="Transformer",
    )

    try:
        RelevanceScorer.score(
            "",
            item,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == "query must not be empty"
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

def test_underscore_is_treated_as_word_separator():
    item = make_item(
        title="Transformer_Models",
    )

    score = RelevanceScorer.score(
        "transformer models",
        item,
    )

    assert score > 0.0


def test_hyphen_is_treated_as_word_separator():
    item = make_item(
        title="Transformer-Models",
    )

    score = RelevanceScorer.score(
        "transformer models",
        item,
    )

    assert score > 0.0


def test_full_query_match_scores_higher_than_single_word_match():
    full_match = make_item(
        title="Transformer Models",
    )

    single_match = make_item(
        title="Vision Transformer",
    )

    full_score = RelevanceScorer.score(
        "transformer models",
        full_match,
    )

    single_score = RelevanceScorer.score(
        "transformer models",
        single_match,
    )

    assert full_score > single_score


def test_query_words_in_title_score_higher_than_description_only():
    title_match = make_item(
        title="Transformer Models",
    )

    description_match = make_item(
        title="Machine Learning",
        description="This paper discusses transformer models.",
    )

    title_score = RelevanceScorer.score(
        "transformer models",
        title_match,
    )

    description_score = RelevanceScorer.score(
        "transformer models",
        description_match,
    )

    assert title_score > description_score


def test_rank_prefers_full_query_title_match():
    weak = make_item(
        title="Computer Vision",
    )

    partial = make_item(
        title="Vision Transformer",
    )

    strong = make_item(
        title="Transformer Models",
    )

    results = RelevanceScorer.rank(
        "transformer models",
        [weak, partial, strong],
    )

    assert results[0].title == "Transformer Models"
    assert results[1].title == "Vision Transformer"
    assert results[2].title == "Computer Vision"

def test_rank_returns_empty_list_for_empty_results():
    results = RelevanceScorer.rank(
        "transformer",
        [],
    )

    assert results == []
def test_rank_preserves_all_results():
    items = [
        make_item("Computer Vision"),
        make_item("Transformer"),
        make_item("Machine Learning"),
    ]

    results = RelevanceScorer.rank(
        "transformer",
        items,
    )

    assert len(results) == len(items)
    assert {item.id for item in results} == {
        item.id for item in items
    }

def test_rank_returns_results_in_descending_score_order():
    items = [
        make_item("Computer Vision"),
        make_item("Transformer"),
        make_item("Transformer Models"),
    ]

    results = RelevanceScorer.rank(
        "transformer models",
        items,
    )

    scores = [
        RelevanceScorer.score("transformer models", item)
        for item in results
    ]

    assert scores == sorted(scores, reverse=True)

def test_rank_returns_results_in_descending_score_order():
    items = [
        make_item("Computer Vision"),
        make_item("Transformer"),
        make_item("Transformer Models"),
    ]

    results = RelevanceScorer.rank(
        "transformer models",
        items,
    )

    scores = [
        RelevanceScorer.score(
            "transformer models",
            item,
        )
        for item in results
    ]

    assert scores == sorted(scores, reverse=True)

def test_rank_preserves_all_results():
    items = [
        make_item("Computer Vision"),
        make_item("Transformer"),
        make_item("Machine Learning"),
    ]

    results = RelevanceScorer.rank(
        "transformer",
        items,
    )

    assert len(results) == len(items)

    assert {
        item.id for item in results
    } == {
        item.id for item in items
    }
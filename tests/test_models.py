from src.models.research import ResearchItem


def test_research_item():
    item = ResearchItem(
        id="test-001",
        title="Test Research Item",
        description="A test research item.",
        authors=["Test Author"],
        source="test",
        url="https://example.com/test",
        tags=["ai", "research"],
    )

    assert item.id == "test-001"
    assert item.title == "Test Research Item"
    assert item.description == "A test research item."
    assert item.authors == ["Test Author"]
    assert item.source == "test"
    assert str(item.url) == "https://example.com/test"
    assert item.tags == ["ai", "research"]
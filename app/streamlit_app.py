import html

import streamlit as st

from src.services.research import ResearchService


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Research Dashboard",
    layout="wide",
)


# ============================================================
# DESIGN TOKENS
# ============================================================

PAPER = "#F7F5F0"
INK = "#20232B"
MUTED = "#6B7280"
RULE = "#E3E0D8"
ACCENT = "#2D3561"

SOURCE_COLORS = {
    "arxiv": "#B31B1B",
    "github": "#24292F",
    "paperswithcode": "#0F9E9E",
    "huggingface": "#C98A00",
}

SOURCE_NAMES = {
    "arxiv": "arXiv",
    "github": "GitHub",
    "paperswithcode": "PapersWithCode",
    "huggingface": "Hugging Face",
}


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');
    .block-container {{
        padding-top: 3rem;
        max-width: 880px;
    }}
    body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Source Serif 4', serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }}
    .eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {MUTED};
        margin-bottom: 0.35rem;
    }}
    .app-title {{
        font-size: 2.1rem;
        margin-bottom: 0.15rem;
    }}
    .app-subtitle {{
        color: {MUTED};
        font-size: 0.95rem;
        margin-bottom: 1.75rem;
    }}
    hr.divider {{
        border: none;
        border-top: 1px solid {RULE};
        margin: 0.5rem 0 1.75rem 0;
    }}
    /* Search input: underline style, no box */
    div[data-testid="stTextInput"] input {{
        border: none !important;
        border-bottom: 2px solid {INK} !important;
        border-radius: 0 !important;
        background: transparent !important;
        font-family: 'Source Serif 4', serif;
        font-size: 1.3rem !important;
        padding: 0.3rem 0.1rem !important;
    }}
    div[data-testid="stTextInput"] input:focus {{
        border-bottom: 2px solid {ACCENT} !important;
        box-shadow: none !important;
    }}
    /* Result card */
    .result-card {{
        background: #FFFFFF;
        border: 1px solid {RULE};
        border-left: 4px solid var(--spine);
        border-radius: 3px;
        padding: 1.15rem 1.4rem 1.05rem 1.25rem;
        margin-bottom: 1rem;
    }}
    .result-title-row {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
    }}
    .result-title {{
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        font-size: 1.2rem;
        color: {INK};
        margin: 0;
    }}
    .result-title a {{
        color: {INK};
        text-decoration: none;
    }}
    .result-title a:hover {{
        color: {ACCENT};
    }}
    .result-source {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--spine);
        white-space: nowrap;
        padding-top: 0.2rem;
    }}
    .result-meta {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: {MUTED};
        margin: 0.35rem 0 0.7rem 0;
    }}
    .result-stats {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: {INK};
        background: {PAPER};
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 3px;
        margin-bottom: 0.7rem;
    }}
    .result-description {{
        font-size: 0.92rem;
        line-height: 1.55;
        color: {INK};
        margin-bottom: 0.7rem;
    }}
    .result-tags {{
        margin-bottom: 0.5rem;
    }}
    .result-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: {MUTED};
        border: 1px solid {RULE};
        border-radius: 3px;
        padding: 0.12rem 0.45rem;
        margin-right: 0.35rem;
        display: inline-block;
    }}
    .result-footer {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid {RULE};
        margin-top: 0.6rem;
        padding-top: 0.6rem;
    }}
    .result-footer-source {{
        font-size: 0.78rem;
        color: {MUTED};
    }}
    .result-open {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: {ACCENT};
        text-decoration: none;
    }}
    .result-open:hover {{
        text-decoration: underline;
    }}
    .results-count {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: {MUTED};
        margin-bottom: 1rem;
    }}
    .empty-state {{
        border: 1px dashed {RULE};
        border-radius: 4px;
        padding: 2rem;
        text-align: center;
        color: {MUTED};
        font-size: 0.92rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="eyebrow">arxiv · github · paperswithcode · huggingface</div>', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">Research Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Search papers, repositories, and models across four sources at once.</div>', unsafe_allow_html=True)


# ============================================================
# SEARCH INPUT
# ============================================================

query = st.text_input(
    "Search",
    placeholder="Search research — e.g. transformer models",
    label_visibility="collapsed",
)

st.write("")


# ============================================================
# SOURCE SELECTION
# ============================================================

available_sources = [
    "arxiv",
    "github",
    "paperswithcode",
    "huggingface",
]

filter_col, sort_col = st.columns([0.68, 0.32])

with filter_col:

    selected_sources = st.multiselect(
        "Sources",
        options=available_sources,
        default=available_sources,
        format_func=lambda source: SOURCE_NAMES.get(source, source.title()),
    )

with sort_col:

    sort_option = st.selectbox(
        "Sort by",
        options=[
            "Most relevant",
            "Newest",
            "Recently updated",
        ],
    )

sort_mapping = {
    "Most relevant": "relevance",
    "Newest": "published",
    "Recently updated": "updated",
}

sort_by = sort_mapping[sort_option]

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def esc(value) -> str:
    """HTML-escape a value, treating None as empty."""

    if value is None:
        return ""

    return html.escape(str(value))


def format_number(value) -> str:
    """Format numbers such as 1200 -> 1.2K."""

    if value is None:
        return "—"

    try:
        value = int(value)
    except (TypeError, ValueError):
        return str(value)

    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,}"


def build_stats_line(result, source: str) -> str:
    """Build the monospace stats line for a given source."""

    if source == "github":

        stars = format_number(getattr(result, "stars", None))
        forks = format_number(getattr(result, "forks", None))
        language = getattr(result, "language", None) or "—"

        return f"★ {stars} stars &nbsp;·&nbsp; ⑂ {forks} forks &nbsp;·&nbsp; {esc(language)}"

    if source == "huggingface":

        downloads = format_number(getattr(result, "downloads", None))
        likes = format_number(getattr(result, "likes", None))
        library = getattr(result, "library", None)
        pipeline_tag = getattr(result, "pipeline_tag", None)

        parts = [f"↓ {downloads} downloads", f"♥ {likes} likes"]

        if library:
            parts.append(esc(library))

        if pipeline_tag:
            parts.append(esc(pipeline_tag))

        return " &nbsp;·&nbsp; ".join(parts)

    if source == "paperswithcode":

        tasks = getattr(result, "tasks", [])
        conference = getattr(result, "conference", None)

        parts = []

        if tasks:
            parts.append(", ".join(esc(t) for t in tasks[:4]))

        if conference:
            parts.append(f"presented at {esc(conference)}")

        return " &nbsp;·&nbsp; ".join(parts) if parts else ""

    return ""


def build_meta_line(result, source: str) -> str:
    """Build the ID / date meta line shown under the title."""

    segments = [SOURCE_NAMES.get(source, source.title()).upper(), esc(result.id)]

    published = getattr(result, "published", None)
    updated = getattr(result, "updated", None)

    if published:
        segments.append(published.strftime("%b %d, %Y"))
    elif updated:
        segments.append(f"updated {updated.strftime('%b %d, %Y')}")

    return " &nbsp;·&nbsp; ".join(segments)


def build_authors_line(result) -> str:
    """Build an authors line where relevant for the source."""

    authors = getattr(result, "authors", [])

    if not authors:
        return ""

    shown = authors[:5]
    text = ", ".join(esc(a) for a in shown)

    if len(authors) > 5:
        text += f" +{len(authors) - 5} more"

    return text


def build_tags_html(result) -> str:
    """Build tag pill markup."""

    tags = getattr(result, "tags", [])

    if not tags:
        return ""

    pills = "".join(f'<span class="result-tag">{esc(tag)}</span>' for tag in tags)

    return f'<div class="result-tags">{pills}</div>'


# ============================================================
# RESULT CARD
# ============================================================

def render_result_card(result):
    """Render one complete research result as a single HTML block."""

    source = getattr(result, "source", "unknown").lower().strip()
    source_display = SOURCE_NAMES.get(source, source.title())
    spine_color = SOURCE_COLORS.get(source, MUTED)

    stats_line = build_stats_line(result, source)
    meta_line = build_meta_line(result, source)
    authors_line = build_authors_line(result)
    tags_html = build_tags_html(result)

    description = (result.description or "").strip()

    if len(description) > 600:
        description = description[:600] + "…"

    description_html = (
        f'<div class="result-description">{esc(description)}</div>' if description else ""
    )

    authors_html = f'<div class="result-meta">By {authors_line}</div>' if authors_line else ""

    stats_html = f'<div class="result-stats">{stats_line}</div>' if stats_line else ""

    # Build the card as ONE unbroken string with no blank lines.
    # Streamlit's markdown renderer treats a blank line as the end of a raw
    # HTML block, so any empty conditional section here would cause the
    # remaining tags to be parsed as plain text instead of HTML.
    parts = [
        f'<div class="result-card" style="--spine: {spine_color};">',
        '<div class="result-title-row">',
        f'<p class="result-title"><a href="{esc(str(result.url))}" target="_blank">{esc(result.title)}</a></p>',
        f'<div class="result-source">{esc(source_display)}</div>',
        '</div>',
        f'<div class="result-meta">{meta_line}</div>',
        stats_html,
        authors_html,
        description_html,
        tags_html,
        '<div class="result-footer">',
        f'<div class="result-footer-source">{esc(source_display)}</div>',
        f'<a class="result-open" href="{esc(str(result.url))}" target="_blank">Open →</a>',
        '</div>',
        '</div>',
    ]

    card_html = "".join(part for part in parts if part)

    st.markdown(card_html, unsafe_allow_html=True)


# ============================================================
# SEARCH
# ============================================================

if query:

    if not selected_sources:

        st.warning("Select at least one source to search.")

    else:

        try:

            service = ResearchService()

            results = service.search(
                query=query,
                sources=selected_sources,
                sort_by=sort_by,
            )

        except Exception as error:

            st.error("Something went wrong while searching the research sources.")
            st.exception(error)

            results = []

        if results:

            st.markdown(f'<div class="results-count">{len(results)} results</div>', unsafe_allow_html=True)

            for result in results:
                render_result_card(result)

        else:

            st.markdown(
                '<div class="empty-state">No results found. Try a different search term or add more sources.</div>',
                unsafe_allow_html=True,
            )
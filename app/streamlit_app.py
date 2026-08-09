import streamlit as st

from src.services.research import ResearchService


st.set_page_config(
    page_title="Smart Research Dashboard",
    layout="wide",
)

st.title("Smart Research Dashboard")
st.write("Search research across multiple sources.")

query = st.text_input(
    "Search research",
    placeholder="e.g. transformer models",
)

available_sources = [
    "arxiv",
    "github",
    "paperswithcode",
    "huggingface",
]

selected_sources = st.multiselect(
    "Sources",
    options=available_sources,
    default=available_sources,
)

if query:
    service = ResearchService()

    results = service.search(
        query,
        sources=selected_sources,
    )

    st.subheader(f"Results ({len(results)})")

    for result in results:
        with st.container(border=True):
            st.markdown(f"### {result.title}")

            st.caption(result.source)

            if result.description:
                st.write(result.description)

            if result.tags:
                st.write(
                    "**Tags:** "
                    + ", ".join(result.tags)
                )

            if result.published:
                st.write(
                    f"**Published:** "
                    f"{result.published:%Y-%m-%d}"
                )

            st.link_button(
                "Open source",
                str(result.url),
            )
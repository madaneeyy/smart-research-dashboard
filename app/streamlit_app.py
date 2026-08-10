import streamlit as st

from src.services.research import ResearchService
from src.services.ai_research import AIResearchService


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Research Dashboard",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "ai_answers" not in st.session_state:
    st.session_state.ai_answers = {}

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ============================================================
# CONSTANTS
# ============================================================

RESULTS_PER_PAGE = 10

AVAILABLE_SOURCES = [
    "arxiv",
    "github",
    "paperswithcode",
    "huggingface",
]


# ============================================================
# HEADER
# ============================================================

st.title("🔎 Smart Research Dashboard")

st.write(
    "Search research papers, repositories, and models "
    "across multiple sources."
)


# ============================================================
# SEARCH INPUT
# ============================================================

query = st.text_input(
    "Search research",
    placeholder=(
        "e.g. models that understand images and classify them"
    ),
)


# ============================================================
# RESET PAGE WHEN QUERY CHANGES
# ============================================================

if query != st.session_state.last_query:

    st.session_state.page = 1

    st.session_state.ai_answers = {}

    st.session_state.last_query = query


# ============================================================
# SOURCE SELECTION
# ============================================================

st.subheader("Sources")

selected_sources = st.multiselect(
    "Select research sources",
    options=AVAILABLE_SOURCES,
    default=AVAILABLE_SOURCES,
    format_func=lambda source: {
        "arxiv": "arXiv",
        "github": "GitHub",
        "paperswithcode": "PapersWithCode",
        "huggingface": "Hugging Face",
    }.get(source, source.title()),
)


# ============================================================
# SEARCH OPTIONS
# ============================================================

option_columns = st.columns(2)


# ------------------------------------------------------------
# SORTING
# ------------------------------------------------------------

with option_columns[0]:

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


# ------------------------------------------------------------
# SEARCH MODE
# ------------------------------------------------------------

with option_columns[1]:

    search_mode = st.selectbox(
        "Search mode",
        options=[
            "Keyword",
            "Semantic",
            "Hybrid",
        ],
    )


search_mode_mapping = {
    "Keyword": "keyword",
    "Semantic": "semantic",
    "Hybrid": "hybrid",
}

search_mode_value = search_mode_mapping[
    search_mode
]


# ============================================================
# SOURCE DISPLAY HELPERS
# ============================================================

def source_name(source: str) -> str:
    """
    Return a human-readable source name.
    """

    names = {
        "github": "GitHub",
        "arxiv": "arXiv",
        "paperswithcode": "PapersWithCode",
        "huggingface": "Hugging Face",
    }

    return names.get(
        source.lower(),
        source.title(),
    )


# ============================================================
# GITHUB METADATA
# ============================================================

def display_github_metadata(result):
    """
    Display GitHub-specific metadata.
    """

    stars = getattr(
        result,
        "stars",
        None,
    )

    forks = getattr(
        result,
        "forks",
        None,
    )

    language = getattr(
        result,
        "language",
        None,
    )

    metadata = []

    if stars is not None:
        metadata.append(
            f"⭐ {stars:,} stars"
        )

    if forks is not None:
        metadata.append(
            f"🍴 {forks:,} forks"
        )

    if language:
        metadata.append(
            f"💻 {language}"
        )

    if metadata:
        st.write(
            " • ".join(metadata)
        )


# ============================================================
# HUGGING FACE METADATA
# ============================================================

def display_huggingface_metadata(result):
    """
    Display Hugging Face-specific metadata.
    """

    downloads = getattr(
        result,
        "downloads",
        None,
    )

    likes = getattr(
        result,
        "likes",
        None,
    )

    library = getattr(
        result,
        "library",
        None,
    )

    pipeline_tag = getattr(
        result,
        "pipeline_tag",
        None,
    )

    metadata = []

    if downloads is not None:

        if downloads >= 1_000_000:

            downloads_text = (
                f"{downloads / 1_000_000:.1f}M"
            )

        elif downloads >= 1_000:

            downloads_text = (
                f"{downloads / 1_000:.1f}K"
            )

        else:

            downloads_text = (
                f"{downloads:,}"
            )

        metadata.append(
            f"⬇ {downloads_text} downloads"
        )

    if likes is not None:
        metadata.append(
            f"♥ {likes:,} likes"
        )

    if library:
        metadata.append(
            f"🤗 {library}"
        )

    if pipeline_tag:
        metadata.append(
            f"🏷 {pipeline_tag}"
        )

    if metadata:
        st.write(
            " • ".join(metadata)
        )


# ============================================================
# ARXIV METADATA
# ============================================================

def display_arxiv_metadata(result):
    """
    Display arXiv-specific metadata.
    """

    authors = getattr(
        result,
        "authors",
        None,
    )

    if authors:

        st.write(
            "**👤 Authors:** "
            + ", ".join(authors)
        )


# ============================================================
# PAPERSWITHCODE METADATA
# ============================================================

def display_paperswithcode_metadata(result):
    """
    Display PapersWithCode-specific metadata.
    """

    authors = getattr(
        result,
        "authors",
        None,
    )

    tasks = getattr(
        result,
        "tasks",
        None,
    )

    conference = getattr(
        result,
        "conference",
        None,
    )

    if authors:

        st.write(
            "**👤 Authors:** "
            + ", ".join(authors)
        )

    if tasks:

        st.write(
            "**🎯 Tasks:** "
            + ", ".join(tasks)
        )

    if conference:

        st.write(
            f"**🎓 Conference:** {conference}"
        )


# ============================================================
# ASK AI
# ============================================================

def display_ask_ai(result):
    """
    Display the Ask AI interface for a research result.
    """

    result_id = str(result.id)

    with st.expander(
        "🤖 Ask AI about this result"
    ):

        st.caption(
            "Ask questions about this research "
            "resource using the available metadata."
        )

        ai_question = st.text_input(
            "Your question",
            placeholder=(
                "e.g. What is this research about?"
            ),
            key=f"ai_question_{result_id}",
        )

        ask_button = st.button(
            "Ask AI",
            key=f"ask_ai_{result_id}",
            use_container_width=True,
        )

        if ask_button:

            if not ai_question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                try:

                    with st.spinner(
                        "AI is analyzing the research..."
                    ):

                        ai_service = (
                            AIResearchService()
                        )

                        # IMPORTANT:
                        #
                        # AIResearchService.ask()
                        # requires:
                        #
                        #   1. ResearchItem
                        #   2. Question
                        #
                        answer = ai_service.ask(
                            result,
                            ai_question.strip(),
                        )

                    st.session_state.ai_answers[
                        result_id
                    ] = {
                        "question":
                            ai_question.strip(),

                        "answer":
                            answer,
                    }

                except Exception as exc:

                    st.error(
                        "The AI request failed."
                    )

                    st.exception(exc)

        # ----------------------------------------------------
        # Display previous answer
        # ----------------------------------------------------

        if result_id in st.session_state.ai_answers:

            previous_answer = (
                st.session_state.ai_answers[
                    result_id
                ]
            )

            st.markdown(
                "### 🤖 AI Answer"
            )

            st.markdown(
                previous_answer["answer"]
            )


# ============================================================
# RESULT CARD
# ============================================================

def display_result_card(result):
    """
    Render a single research result card.
    """

    source = (
        result.source.lower()
    )

    with st.container(
        border=True
    ):

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        st.markdown(
            f"### {result.title}"
        )

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        st.caption(
            f"Source: {source_name(source)}"
        )

        # ----------------------------------------------------
        # Source-specific metadata
        # ----------------------------------------------------

        if source == "github":

            display_github_metadata(
                result
            )

        elif source == "huggingface":

            display_huggingface_metadata(
                result
            )

        elif source == "arxiv":

            display_arxiv_metadata(
                result
            )

        elif source == "paperswithcode":

            display_paperswithcode_metadata(
                result
            )

        # ----------------------------------------------------
        # Description
        # ----------------------------------------------------

        if result.description:

            st.write(
                result.description
            )

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        if result.tags:

            st.write(
                "**🏷 Tags:** "
                + " • ".join(
                    result.tags
                )
            )

        # ----------------------------------------------------
        # Dates
        # ----------------------------------------------------

        date_columns = st.columns(2)

        with date_columns[0]:

            if result.published:

                st.caption(
                    "📅 Published"
                )

                st.write(
                    f"{result.published:%Y-%m-%d}"
                )

        with date_columns[1]:

            if result.updated:

                st.caption(
                    "🔄 Updated"
                )

                st.write(
                    f"{result.updated:%Y-%m-%d %H:%M:%S %Z}"
                )

        # ----------------------------------------------------
        # Open source
        # ----------------------------------------------------

        st.link_button(
            "🔗 Open source",
            str(result.url),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Ask AI
        # ----------------------------------------------------

        display_ask_ai(result)


# ============================================================
# SEARCH
# ============================================================

if query:

    if not selected_sources:

        st.warning(
            "Please select at least one source."
        )

    else:

        service = ResearchService()

        try:

            # ------------------------------------------------
            # Search
            # ------------------------------------------------

            results = service.search(
                query,
                sources=selected_sources,
                sort_by=sort_by,
                search_mode=search_mode_value,
            )

        except TypeError:

            # ------------------------------------------------
            # Compatibility fallback
            #
            # If the current ResearchService.search()
            # does not yet accept search_mode, fall back
            # to the existing search interface.
            # ------------------------------------------------

            results = service.search(
                query,
                sources=selected_sources,
                sort_by=sort_by,
            )

        except Exception as exc:

            st.error(
                "Search failed."
            )

            st.exception(exc)

            results = []

        # ====================================================
        # RESULTS HEADER
        # ====================================================

        st.subheader(
            f"{len(results)} results"
        )

        # ====================================================
        # SEARCH MODE INDICATOR
        # ====================================================

        if search_mode == "Keyword":

            st.caption(
                "🔤 Keyword search — matches "
                "explicit terms in research metadata."
            )

        elif search_mode == "Semantic":

            st.caption(
                "🧠 Semantic search — finds results "
                "based on conceptual similarity."
            )

        elif search_mode == "Hybrid":

            st.caption(
                "⚡ Hybrid search — combines keyword "
                "and semantic relevance."
            )

        # ====================================================
        # NO RESULTS
        # ====================================================

        if not results:

            st.info(
                "No results found. Try a different "
                "search query or select more sources."
            )

        else:

            # =================================================
            # PAGINATION
            # =================================================

            total_results = len(
                results
            )

            total_pages = max(
                1,
                (
                    total_results
                    + RESULTS_PER_PAGE
                    - 1
                )
                // RESULTS_PER_PAGE,
            )

            # -------------------------------------------------
            # Keep page valid
            # -------------------------------------------------

            if (
                st.session_state.page
                > total_pages
            ):

                st.session_state.page = (
                    total_pages
                )

            current_page = (
                st.session_state.page
            )

            start_index = (
                (current_page - 1)
                * RESULTS_PER_PAGE
            )

            end_index = min(
                start_index
                + RESULTS_PER_PAGE,
                total_results,
            )

            page_results = results[
                start_index:end_index
            ]

            # =================================================
            # PAGE INFORMATION
            # =================================================

            st.caption(
                f"Showing "
                f"{start_index + 1}–{end_index} "
                f"of {total_results}"
            )

            # =================================================
            # DISPLAY RESULTS
            # =================================================

            for result in page_results:

                display_result_card(
                    result
                )

            # =================================================
            # PAGINATION CONTROLS
            # =================================================

            if total_pages > 1:

                pagination_columns = (
                    st.columns(
                        [
                            1,
                            2,
                            1,
                        ]
                    )
                )

                # ------------------------------------------------
                # Previous
                # ------------------------------------------------

                with pagination_columns[0]:

                    if st.button(
                        "← Previous",
                        disabled=(
                            current_page <= 1
                        ),
                        use_container_width=True,
                    ):

                        st.session_state.page -= 1

                        st.rerun()

                # ------------------------------------------------
                # Page indicator
                # ------------------------------------------------

                with pagination_columns[1]:

                    st.markdown(
                        f"""
                        <div style="
                            text-align: center;
                            padding-top: 7px;
                            font-weight: 600;
                        ">
                            Page {current_page}
                            of {total_pages}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ------------------------------------------------
                # Next
                # ------------------------------------------------

                with pagination_columns[2]:

                    if st.button(
                        "Next →",
                        disabled=(
                            current_page
                            >= total_pages
                        ),
                        use_container_width=True,
                    ):

                        st.session_state.page += 1

                        st.rerun()

else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.info(
        "Enter a research topic above to start searching."
    )

    st.markdown(
        """
        ### Try searching for

        - `transformer models`
        - `large language models`
        - `image classification`
        - `computer vision`
        - `retrieval augmented generation`
        - `multimodal AI`
        """
    )
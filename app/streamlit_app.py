import sys
import uuid
from pathlib import Path
from typing import Any

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.research import ResearchService

st.set_page_config(
    page_title="Smart Research AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://127.0.0.1:8000/ask"
RESULTS_PER_PAGE = 10
AVAILABLE_SOURCES = ["arxiv", "github", "paperswithcode", "huggingface"]
DOCUMENT_FILE_TYPES = [
    "pdf", "docx", "txt", "md", "markdown",
    "csv", "json",
    "py", "js", "jsx", "ts", "tsx",
    "java", "c", "h", "cpp", "cc", "hpp",
    "go", "rs", "rb", "php", "swift",
    "kt", "kts", "dart", "cs",
    "yaml", "yml", "toml", "xml",
    "html", "css", "sql", "sh", "ps1",
]

SOURCE_NAMES = {
    "github": "GitHub",
    "arxiv": "arXiv",
    "paperswithcode": "PapersWithCode",
    "huggingface": "Hugging Face",
}

# ============================================================================
# SESSION STATE
# ============================================================================

defaults = {
    "page": 1,
    "last_query": "",
    "ai_answers": {},
    "chats": {},
    "active_chat_id": None,
    "chat_counter": 0,
    "app_mode": "Chat",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================================
# CHAT HELPERS
# ============================================================================

def create_chat(title: str = "New Chat") -> str:
    st.session_state.chat_counter += 1

    # Stable ID for the backend's per-chat GitHub context.
    # The same ID is sent with every message in this conversation.
    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "title": title,
        "messages": [],
        "github_url": None,
    }

    st.session_state.active_chat_id = chat_id
    return chat_id


def active_chat() -> dict[str, Any]:
    chat_id = st.session_state.active_chat_id
    if chat_id not in st.session_state.chats:
        create_chat()
    return st.session_state.chats[st.session_state.active_chat_id]


def chat_title(question: str) -> str:
    text = " ".join(question.strip().split())
    return text if len(text) <= 42 else text[:39].rstrip() + "..."


def call_backend(
    question: str,
    history: list[dict[str, str]],
    chat_id: str,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "question": question.strip(),
        "history": history,
        "top_k": 5,
        # Backend uses this to remember the active GitHub repository
        # when later messages omit the URL.
        "chat_id": chat_id,
        "document_ids": document_ids,
    }

    response = requests.post(
        BACKEND_URL,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()

    data = response.json()
    answer = (
        data.get("answer")
        or data.get("response")
        or data.get("output")
    )

    if not answer:
        raise ValueError(
            "FastAPI returned no AI answer. "
            f"Response: {data}"
        )

    return {
        "answer": answer,
        "sources": data.get("sources", []),
        "retrieval": data.get("retrieval", {}),
        "chunks_created": data.get("chunks_created", 0),
        "chunks_retrieved": data.get("chunks_retrieved", 0),
        "model": data.get("model", ""),
        "context_origin": data.get("context_origin", ""),
        "context_scope": data.get("context_scope", ""),
        "context_route": data.get("context_route", {}),
        "source_switched": data.get("source_switched", False),
        "github_url": data.get("github_url"),
        "active_document_ids": data.get("active_document_ids", []),
        "uploaded_documents": data.get("uploaded_documents", []),
    }


def upload_document(
    uploaded_file: Any,
    chat_id: str,
) -> dict[str, Any]:
    """Upload one document to the FastAPI document-RAG endpoint."""
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    response = requests.post(
        f"{BACKEND_URL.rsplit('/ask', 1)[0]}/upload-document",
        params={"chat_id": chat_id},
        files=files,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()



def show_backend_error(exc: Exception) -> None:
    if isinstance(exc, requests.exceptions.ConnectionError):
        st.error("Could not connect to the AI backend.")
        st.info(
            "Make sure FastAPI is running on "
            "http://127.0.0.1:8000"
        )
    elif isinstance(exc, requests.exceptions.Timeout):
        st.error("The AI request timed out.")
        st.info(
            "Qwen or retrieval may still be processing the request."
        )
    elif isinstance(exc, requests.exceptions.HTTPError):
        st.error("The AI backend returned an error.")
        try:
            if exc.response is not None:
                try:
                    st.json(exc.response.json())
                except Exception:
                    st.code(exc.response.text)
        except Exception:
            st.exception(exc)
    else:
        st.error("The AI request failed.")
        st.exception(exc)


# ============================================================================
# CHAT UI
# ============================================================================

def render_chat_sidebar() -> None:
    """
    Render navigation/settings only.

    Document uploads intentionally do NOT live in the sidebar anymore.
    They are attached directly from the chat composer using st.chat_input's
    native file attachment button.
    """
    with st.sidebar:
        st.markdown("## 🔎 Smart Research AI")

        if st.button(
            "＋ New Chat",
            use_container_width=True,
            type="primary",
        ):
            create_chat()
            st.rerun()

        st.divider()
        st.caption("CHATS")

        chats = list(st.session_state.chats.values())

        if not chats:
            st.caption("No conversations yet.")

        for chat_item in reversed(chats):
            chat_id = chat_item["id"]
            label = chat_item.get("title") or "New Chat"

            if chat_id == st.session_state.active_chat_id:
                label = "● " + label

            if st.button(
                label,
                key=f"select_chat_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.active_chat_id = chat_id
                st.session_state.app_mode = "Chat"
                st.rerun()

        st.divider()

        if st.button(
            "🔍 Research Search",
            use_container_width=True,
        ):
            st.session_state.app_mode = "Research"
            st.rerun()

        st.divider()

        current_chat = st.session_state.chats.get(
            st.session_state.active_chat_id
        )

        active_repo = (
            current_chat.get("github_url")
            if current_chat
            else None
        )

        if active_repo:
            st.caption("📦 Active GitHub repository")
            st.code(active_repo, language="text")

        attached_documents = (
            current_chat.get("document_ids", [])
            if current_chat
            else []
        )

        if attached_documents:
            st.caption(
                f"📎 {len(attached_documents)} document(s) attached "
                "to this chat"
            )

        st.caption(
            "Attach documents directly from the paperclip button "
            "inside the message box."
        )


def render_chat() -> None:
    render_chat_sidebar()
    chat = active_chat()

    st.markdown(
        f"### {chat.get('title', 'New Chat')}"
    )
    st.caption("Smart Research AI")

    messages = chat.get("messages", [])

    if not messages:
        st.markdown(
            """
            <div style="text-align:center;padding:12vh 1rem 15vh 1rem">
                <div style="font-size:2.2rem">🔎</div>
                <div style="font-size:2rem;font-weight:650">
                    What can I help you with?
                </div>
                <div style="color:#777;max-width:650px;margin:auto">
                    Ask questions about research, programming,
                    machine learning, repositories, papers,
                    or anything you want to investigate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            metadata = message.get("metadata")
            if metadata and message["role"] == "assistant":
                render_chat_diagnostics(metadata)

    # ------------------------------------------------------------------
    # NATIVE CHAT ATTACHMENT COMPOSER
    # ------------------------------------------------------------------
    #
    # Modern Streamlit versions expose a native attachment/paperclip
    # control when accept_file is enabled. The submission is a dict-like
    # object with:
    #
    #   submission.text
    #   submission.files
    #
    # This keeps the upload UX exactly where users expect it: beside the
    # message composer instead of in the sidebar.
    # ------------------------------------------------------------------
    submission = st.chat_input(
        "Message Smart Research AI...",
        key=f"chat_composer_{chat['id']}",
        accept_file="multiple",
        file_type=DOCUMENT_FILE_TYPES,
        max_upload_size=200,
        submit_mode="disable",
    )

    if submission is None:
        return

    prompt = str(
        submission.get("text", "")
        if hasattr(submission, "get")
        else getattr(submission, "text", "")
    ).strip()

    attached_files = (
        submission.get("files", [])
        if hasattr(submission, "get")
        else getattr(submission, "files", [])
    ) or []

    # ------------------------------------------------------------------
    # 1. Upload every newly attached document.
    # ------------------------------------------------------------------
    uploaded_this_turn: list[str] = []
    upload_errors: list[str] = []

    for uploaded_file in attached_files:
        upload_key = (
            f"{chat['id']}:"
            f"{uploaded_file.name}:"
            f"{len(uploaded_file.getvalue())}"
        )

        processed_keys = chat.setdefault(
            "uploaded_file_keys",
            set(),
        )

        if upload_key in processed_keys:
            continue

        try:
            with st.spinner(
                f"Indexing {uploaded_file.name}..."
            ):
                upload_result = upload_document(
                    uploaded_file,
                    chat["id"],
                )

            document_id = upload_result["document_id"]

            chat.setdefault("document_ids", [])

            if document_id not in chat["document_ids"]:
                chat["document_ids"].append(document_id)

            processed_keys.add(upload_key)
            uploaded_this_turn.append(
                upload_result.get(
                    "filename",
                    uploaded_file.name,
                )
            )

        except Exception as upload_error:
            upload_errors.append(
                f"{uploaded_file.name}: {upload_error}"
            )

    if upload_errors:
        for error in upload_errors:
            st.error(f"Could not attach document — {error}")

    # ------------------------------------------------------------------
    # 2. A file can be submitted without text.
    # ------------------------------------------------------------------
    if not prompt and attached_files and not uploaded_this_turn:
        return

    # If the user only attached a file, automatically give the assistant
    # a useful initial instruction rather than sending an empty question.
    if not prompt and uploaded_this_turn:
        prompt = (
            "I attached "
            + ", ".join(
                f"`{name}`"
                for name in uploaded_this_turn
            )
            + ". Please summarize what this document contains and "
              "tell me what I can ask you about it."
        )

    if not prompt:
        return

    # ------------------------------------------------------------------
    # 3. Build conversation history BEFORE adding this turn.
    # ------------------------------------------------------------------
    history = [
        {
            "role": m["role"],
            "content": m["content"],
        }
        for m in messages
        if m.get("role") in {"user", "assistant"}
    ]

    # Keep the visible user message compact when a file is attached.
    visible_prompt = prompt

    if uploaded_this_turn:
        attachment_text = "\n\n".join(
            f"📎 **{name}**"
            for name in uploaded_this_turn
        )
        visible_prompt = (
            f"{prompt}\n\n{attachment_text}"
        )

    chat["messages"].append({
        "role": "user",
        "content": visible_prompt,
    })

    if len(chat["messages"]) == 1:
        chat["title"] = chat_title(prompt)

    with st.chat_message("user"):
        st.markdown(visible_prompt)

    with st.chat_message("assistant"):
        with st.spinner(
            "Thinking and retrieving relevant information..."
        ):
            try:
                data = call_backend(
                    prompt,
                    history,
                    chat["id"],
                    chat.get("document_ids", []),
                )

                answer = data["answer"]

                # Keep frontend state synchronized with backend's active
                # GitHub repository.
                active_repo = data.get(
                    "active_github_repository"
                )

                if active_repo:
                    chat["github_url"] = active_repo

                st.markdown(answer)

                chat["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": data,
                })

                render_chat_diagnostics(data)

            except Exception as exc:
                show_backend_error(exc)

                if (
                    chat["messages"]
                    and chat["messages"][-1]["role"] == "user"
                ):
                    chat["messages"].pop()


def render_chat_diagnostics(data: dict[str, Any]) -> None:
    summary = []

    if data.get("model"):
        summary.append(f"Model: `{data['model']}`")
    if data.get("context_origin"):
        summary.append(
            f"Context: `{data['context_origin']}`"
        )

    if data.get("context_scope"):
        summary.append(
            f"Scope: `{data['context_scope']}`"
        )
    if data.get("chunks_created"):
        summary.append(
            f"Chunks indexed: {data['chunks_created']}"
        )
    if data.get("chunks_retrieved"):
        summary.append(
            f"Evidence retrieved: {data['chunks_retrieved']}"
        )

    uploaded = data.get("uploaded_documents") or []
    if uploaded:
        names = ", ".join(
            str(item.get("filename") or "document")
            for item in uploaded
        )
        summary.append(f"📄 {names}")

    if summary:
        st.caption(" • ".join(summary))

    sources = data.get("sources") or []
    if not sources:
        return

    with st.expander(
        f"📚 Retrieved evidence ({len(sources)} chunks)"
    ):
        for index, source in enumerate(sources, 1):
            path = source.get("source") or "Unknown source"
            section = source.get("section") or "Unknown section"

            st.markdown(f"**Evidence {index}**")
            st.caption(f"Source: {path}  •  Section: {section}")

            if source.get("content"):
                st.code(source["content"], language="text")

            cols = st.columns(4)

            score = source.get("query_relevance_score")
            if score is not None:
                cols[0].caption(f"Relevance: {score:.3f}")

            score = source.get("semantic_score")
            if score is not None:
                cols[1].caption(f"Semantic: {score:.3f}")

            score = source.get("mmr_score")
            if score is not None:
                cols[2].caption(f"MMR: {score:.3f}")

            score = source.get("complementarity_score")
            if score is not None:
                cols[3].caption(
                    f"Complementarity: {score:.3f}"
                )

            if index < len(sources):
                st.divider()


# ============================================================================
# ORIGINAL RESEARCH SEARCH UI
# ============================================================================

def source_name(source: str) -> str:
    return SOURCE_NAMES.get(source.lower(), source.title())


def display_github_metadata(result: Any) -> None:
    values = []
    stars = getattr(result, "stars", None)
    forks = getattr(result, "forks", None)
    language = getattr(result, "language", None)

    if stars is not None:
        values.append(f"⭐ {stars:,} stars")
    if forks is not None:
        values.append(f"🍴 {forks:,} forks")
    if language:
        values.append(f"💻 {language}")

    if values:
        st.write(" • ".join(values))


def display_huggingface_metadata(result: Any) -> None:
    values = []
    downloads = getattr(result, "downloads", None)
    likes = getattr(result, "likes", None)
    library = getattr(result, "library", None)
    pipeline_tag = getattr(result, "pipeline_tag", None)

    if downloads is not None:
        if downloads >= 1_000_000:
            text = f"{downloads / 1_000_000:.1f}M"
        elif downloads >= 1_000:
            text = f"{downloads / 1_000:.1f}K"
        else:
            text = f"{downloads:,}"
        values.append(f"⬇ {text} downloads")

    if likes is not None:
        values.append(f"♥ {likes:,} likes")
    if library:
        values.append(f"🤗 {library}")
    if pipeline_tag:
        values.append(f"🏷 {pipeline_tag}")

    if values:
        st.write(" • ".join(values))


def display_arxiv_metadata(result: Any) -> None:
    authors = getattr(result, "authors", None)
    if authors:
        st.write(
            "**👤 Authors:** "
            + ", ".join(str(x) for x in authors)
        )


def display_paperswithcode_metadata(result: Any) -> None:
    authors = getattr(result, "authors", None)
    tasks = getattr(result, "tasks", None)
    conference = getattr(result, "conference", None)

    if authors:
        st.write(
            "**👤 Authors:** "
            + ", ".join(str(x) for x in authors)
        )
    if tasks:
        st.write(
            "**🎯 Tasks:** "
            + ", ".join(str(x) for x in tasks)
        )
    if conference:
        st.write(f"**🎓 Conference:** {conference}")


def research_ask_ai(result: Any) -> None:
    result_id = str(result.id)

    with st.expander("🤖 Ask AI about this result"):
        st.caption(
            "For GitHub repositories, the backend analyzes the repository "
            "for your specific question before running Hybrid RAG."
        )

        question = st.text_input(
            "Your question",
            placeholder="e.g. How does this approach work?",
            key=f"research_question_{result_id}",
        )

        if st.button(
            "Ask AI",
            key=f"research_ask_{result_id}",
            use_container_width=True,
        ):
            if not question.strip():
                st.warning("Please enter a question.")
                return

            try:
                source = str(
                    getattr(result, "source", "") or ""
                ).lower().strip()
                url = getattr(result, "url", None)
                url = str(url).strip() if url else None

                context = [
                    f"Title: {getattr(result, 'title', '')}",
                    f"Source: {getattr(result, 'source', '')}",
                    f"URL: {getattr(result, 'url', '')}",
                ]

                for attr, label in [
                    ("description", "Description"),
                    ("published", "Published"),
                    ("updated", "Updated"),
                    ("stars", "GitHub stars"),
                    ("forks", "GitHub forks"),
                    ("language", "Programming language"),
                    ("downloads", "Downloads"),
                    ("likes", "Likes"),
                    ("library", "Library"),
                    ("pipeline_tag", "Pipeline tag"),
                    ("conference", "Conference"),
                ]:
                    value = getattr(result, attr, None)
                    if value is not None and value != "":
                        context.append(f"{label}: {value}")

                for attr, label in [
                    ("authors", "Authors"),
                    ("tags", "Tags"),
                    ("tasks", "Tasks"),
                ]:
                    value = getattr(result, attr, None)
                    if value:
                        context.append(
                            f"{label}: "
                            + ", ".join(str(x) for x in value)
                        )

                previous = st.session_state.ai_answers.get(
                    result_id,
                    {},
                )

                history = []
                if previous.get("question"):
                    history.append({
                        "role": "user",
                        "content": previous["question"],
                    })
                if previous.get("answer"):
                    history.append({
                        "role": "assistant",
                        "content": previous["answer"],
                    })

                # Use the current chat's stable ID so a GitHub repository
                # selected here can remain active for later chat messages.
                research_chat_id = st.session_state.active_chat_id
                if not research_chat_id:
                    research_chat_id = create_chat("Research Chat")

                payload = {
                    "question": question.strip(),
                    "history": history,
                    "top_k": 5,
                    "chat_id": research_chat_id,
                }

                if source == "github" and url:
                    payload["github_url"] = url
                else:
                    payload["context"] = "\n".join(context)

                with st.spinner(
                    "AI is retrieving relevant evidence..."
                ):
                    response = requests.post(
                        BACKEND_URL,
                        json=payload,
                        timeout=300,
                    )
                    response.raise_for_status()

                data = response.json()
                answer = (
                    data.get("answer")
                    or data.get("response")
                    or data.get("output")
                )

                if not answer:
                    raise ValueError(
                        "FastAPI returned no AI answer. "
                        f"Response: {data}"
                    )

                st.session_state.ai_answers[result_id] = {
                    "question": question.strip(),
                    "answer": answer,
                    "sources": data.get("sources", []),
                    "retrieval": data.get("retrieval", {}),
                    "chunks_created": data.get(
                        "chunks_created", 0
                    ),
                    "chunks_retrieved": data.get(
                        "chunks_retrieved", 0
                    ),
                    "model": data.get("model", ""),
                    "context_origin": data.get(
                        "context_origin", ""
                    ),
                    "github_url": data.get("github_url"),
                }

            except Exception as exc:
                show_backend_error(exc)

        previous = st.session_state.ai_answers.get(result_id)
        if not previous:
            return

        st.markdown("### 🤖 AI Answer")
        st.markdown(previous["answer"])

        render_chat_diagnostics(previous)


def display_result_card(result: Any) -> None:
    source = str(
        getattr(result, "source", "") or ""
    ).lower()

    with st.container(border=True):
        st.markdown(
            f"### {getattr(result, 'title', 'Untitled')}"
        )
        st.caption(f"Source: {source_name(source)}")

        if source == "github":
            display_github_metadata(result)
        elif source == "huggingface":
            display_huggingface_metadata(result)
        elif source == "arxiv":
            display_arxiv_metadata(result)
        elif source == "paperswithcode":
            display_paperswithcode_metadata(result)

        description = getattr(result, "description", None)
        if description:
            st.write(description)

        tags = getattr(result, "tags", None)
        if tags:
            st.write(
                "**🏷 Tags:** "
                + " • ".join(str(x) for x in tags)
            )

        cols = st.columns(2)

        published = getattr(result, "published", None)
        if published:
            with cols[0]:
                st.caption("📅 Published")
                try:
                    st.write(published.strftime("%Y-%m-%d"))
                except Exception:
                    st.write(str(published))

        updated = getattr(result, "updated", None)
        if updated:
            with cols[1]:
                st.caption("🔄 Updated")
                try:
                    st.write(
                        updated.strftime(
                            "%Y-%m-%d %H:%M:%S %Z"
                        )
                    )
                except Exception:
                    st.write(str(updated))

        url = getattr(result, "url", None)
        if url:
            st.link_button(
                "🔗 Open source",
                str(url),
                use_container_width=True,
            )

        research_ask_ai(result)


def render_research() -> None:
    with st.sidebar:
        st.markdown("## 🔎 Smart Research AI")
        if st.button(
            "← Back to Chat",
            use_container_width=True,
        ):
            st.session_state.app_mode = "Chat"
            st.rerun()

    st.title("🔍 Research Search")
    st.write(
        "Search research papers, repositories, and models "
        "across multiple sources."
    )

    query = st.text_input(
        "Search research",
        placeholder=(
            "e.g. models that understand images and classify them"
        ),
        key="research_query",
    )

    if query != st.session_state.last_query:
        st.session_state.page = 1
        st.session_state.ai_answers = {}
        st.session_state.last_query = query

    st.subheader("Sources")

    selected_sources = st.multiselect(
        "Select research sources",
        options=AVAILABLE_SOURCES,
        default=AVAILABLE_SOURCES,
        format_func=source_name,
    )

    cols = st.columns(2)

    with cols[0]:
        sort_option = st.selectbox(
            "Sort by",
            ["Most relevant", "Newest", "Recently updated"],
        )

    with cols[1]:
        search_mode = st.selectbox(
            "Search mode",
            ["Keyword", "Semantic", "Hybrid"],
        )

    sort_by = {
        "Most relevant": "relevance",
        "Newest": "published",
        "Recently updated": "updated",
    }[sort_option]

    search_mode_value = {
        "Keyword": "keyword",
        "Semantic": "semantic",
        "Hybrid": "hybrid",
    }[search_mode]

    if not query:
        st.info(
            "Enter a research topic above to start searching."
        )
        return

    if not selected_sources:
        st.warning("Please select at least one source.")
        return

    try:
        service = ResearchService()
        try:
            results = service.search(
                query,
                sources=selected_sources,
                sort_by=sort_by,
                search_mode=search_mode_value,
            )
        except TypeError:
            results = service.search(
                query,
                sources=selected_sources,
                sort_by=sort_by,
            )
    except Exception as exc:
        st.error("Search failed.")
        st.exception(exc)
        return

    st.subheader(f"{len(results)} results")

    if search_mode == "Keyword":
        st.caption(
            "🔤 Keyword search — matches explicit terms "
            "in research metadata."
        )
    elif search_mode == "Semantic":
        st.caption(
            "🧠 Semantic search — finds results based on "
            "conceptual similarity."
        )
    else:
        st.caption(
            "⚡ Hybrid search — combines keyword and "
            "semantic relevance."
        )

    if not results:
        st.info(
            "No results found. Try a different query "
            "or select more sources."
        )
        return

    total = len(results)
    total_pages = max(
        1,
        (total + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE,
    )

    st.session_state.page = min(
        st.session_state.page,
        total_pages,
    )

    current = st.session_state.page
    start = (current - 1) * RESULTS_PER_PAGE
    end = min(start + RESULTS_PER_PAGE, total)

    st.caption(f"Showing {start + 1}–{end} of {total}")

    for result in results[start:end]:
        display_result_card(result)

    if total_pages > 1:
        cols = st.columns([1, 2, 1])

        with cols[0]:
            if st.button(
                "← Previous",
                disabled=current <= 1,
                use_container_width=True,
            ):
                st.session_state.page -= 1
                st.rerun()

        with cols[1]:
            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    padding-top:7px;
                    font-weight:600;
                ">
                    Page {current} of {total_pages}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with cols[2]:
            if st.button(
                "Next →",
                disabled=current >= total_pages,
                use_container_width=True,
            ):
                st.session_state.page += 1
                st.rerun()


# ============================================================================
# ENTRY POINT
# ============================================================================

if not st.session_state.chats:
    create_chat()

if st.session_state.app_mode == "Research":
    render_research()
else:
    render_chat()
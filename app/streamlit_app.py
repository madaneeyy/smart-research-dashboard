
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


# ============================================================================
# FUTURISTIC PRODUCT UI
# ============================================================================

def inject_app_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        :root {
            --sr-bg:#0B0F14; --sr-bg2:#111720; --sr-surface:#151C26;
            --sr-elevated:#1B2430; --sr-border:#263241; --sr-text:#F3F6FA;
            --sr-secondary:#9AA7B5; --sr-muted:#657383; --sr-blue:#6C8CFF;
            --sr-teal:#7CE7D8; --sr-green:#59D68A; --sr-yellow:#F5C76A;
            --sr-red:#FF6B7A; --sr-github:#E6EDF3;
        }
        html, body, [class*="css"] { font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
        .stApp { background:var(--sr-bg); color:var(--sr-text); }
        .main .block-container { max-width:1180px; padding-top:1.35rem; padding-bottom:7rem; }
        section[data-testid="stSidebar"] { background:#0D1218; border-right:1px solid var(--sr-border); }
        section[data-testid="stSidebar"] > div { padding:.95rem .8rem; }
        section[data-testid="stSidebar"] .stButton > button {
            background:transparent; border:1px solid transparent; color:var(--sr-secondary);
            border-radius:8px; min-height:2.2rem; transition:all 180ms ease;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background:var(--sr-surface); border-color:var(--sr-border); color:var(--sr-text); transform:translateX(1px);
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background:var(--sr-blue); border-color:var(--sr-blue); color:#fff; font-weight:600; text-align:center;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { filter:brightness(1.08); transform:translateY(-1px); }
        div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div, div[data-baseweb="select"] > div {
            background:var(--sr-surface); border-color:var(--sr-border); border-radius:9px;
        }
        div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea { color:var(--sr-text); }
        div[data-testid="stChatInput"] { background:rgba(11,15,20,.92); border-top:0; }
        div[data-testid="stChatInput"] > div {
            background:var(--sr-surface); border:1px solid var(--sr-border); border-radius:14px;
            box-shadow:0 14px 45px rgba(0,0,0,.28); transition:border-color 180ms ease,box-shadow 180ms ease;
        }
        div[data-testid="stChatInput"] > div:focus-within {
            border-color:rgba(108,140,255,.65); box-shadow:0 14px 45px rgba(0,0,0,.32),0 0 0 1px rgba(108,140,255,.12);
        }
        div[data-testid="stChatMessage"] { border:0; background:transparent; padding:.7rem 0; }
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { line-height:1.72; }
        pre { background:#0A0E13 !important; border:1px solid var(--sr-border) !important; border-radius:9px !important; }
        code { font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace !important; }
        div[data-testid="stExpander"] { background:rgba(21,28,38,.55); border:1px solid var(--sr-border); border-radius:10px; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background:rgba(21,28,38,.45); border-color:var(--sr-border); border-radius:12px;
            transition:border-color 180ms ease,transform 180ms ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:#334154; transform:translateY(-1px); }
        .sr-brand { display:flex; align-items:center; gap:.65rem; padding:.35rem .2rem 1rem; }
        .sr-mark {
            width:31px; height:31px; border-radius:9px; display:grid; place-items:center;
            background:linear-gradient(135deg,#6C8CFF,#7CE7D8); color:#071016; font-weight:800; font-size:11px;
            box-shadow:0 5px 18px rgba(108,140,255,.18);
        }
        .sr-brand-title { font-size:14px; font-weight:700; color:var(--sr-text); }
        .sr-brand-sub { font-size:10px; color:var(--sr-muted); margin-top:1px; }
        .sr-section-label { font-size:10px; font-weight:700; letter-spacing:.12em; color:var(--sr-muted); margin:1.1rem .15rem .45rem; }
        .sr-repo { border:1px solid var(--sr-border); background:rgba(21,28,38,.62); border-radius:9px; padding:.65rem .7rem; margin-top:.7rem; }
        .sr-repo-label { font-size:10px; color:var(--sr-teal); text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
        .sr-repo-name { font-size:12px; color:var(--sr-text); margin-top:.25rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .sr-context {
            display:inline-flex; align-items:center; gap:.45rem; border:1px solid var(--sr-border);
            background:rgba(21,28,38,.65); border-radius:999px; padding:.34rem .65rem; color:var(--sr-secondary); font-size:11px;
        }
        .sr-dot { width:6px; height:6px; border-radius:50%; background:var(--sr-teal); box-shadow:0 0 9px rgba(124,231,216,.45); }
        .sr-hero { padding:10vh 1rem 10vh; text-align:center; }
        .sr-eyebrow { color:var(--sr-teal); font-size:11px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.7rem; }
        .sr-hero h1 { color:var(--sr-text); font-size:clamp(28px,4vw,42px); letter-spacing:-.045em; margin:0; }
        .sr-hero p { max-width:620px; margin:.8rem auto 0; color:var(--sr-secondary); font-size:14px; line-height:1.7; }
        .sr-research-head { border-bottom:1px solid var(--sr-border); padding-bottom:1rem; margin-bottom:1.2rem; }
        .sr-research-head h1 { font-size:28px; letter-spacing:-.035em; margin-bottom:.3rem; }
        .sr-source-badge {
            display:inline-block; font-size:10px; font-weight:700; color:var(--sr-secondary);
            border:1px solid var(--sr-border); border-radius:999px; padding:.22rem .48rem; background:rgba(21,28,38,.55);
        }
        .sr-diagnostics { color:var(--sr-muted); font-size:11px; margin-top:.45rem; }
        @media (max-width:800px) {
            .main .block-container { padding-left:.8rem; padding-right:.8rem; }
            .sr-hero { padding-top:7vh; }
        }
        </style>
        """, unsafe_allow_html=True
    )

inject_app_css()

BACKEND_URL = "http://127.0.0.1:8000/ask"
RESULTS_PER_PAGE = 10
AVAILABLE_SOURCES = ["arxiv", "github", "paperswithcode", "huggingface"]
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
    "pending_prompt": None,
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
) -> dict[str, Any]:
    payload = {
        "question": question.strip(),
        "history": history,
        "top_k": 5,
        # Backend uses this to remember the active GitHub repository
        # when later messages omit the URL.
        "chat_id": chat_id,
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
        "github_url": data.get("github_url"),
    }


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
def render_chat_sidebar() -> None:
    with st.sidebar:
        # ---------------------------------------------------------
        # BRAND
        # ---------------------------------------------------------
        brand_col1, brand_col2 = st.columns([0.22, 0.78], vertical_alignment="center")

        with brand_col1:
            st.markdown(
                """
                <div style="
                    width:31px;
                    height:31px;
                    border-radius:9px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background:linear-gradient(135deg,#6C8CFF,#7CE7D8);
                    color:#071016;
                    font-weight:800;
                    font-size:10px;
                    box-shadow:0 5px 18px rgba(108,140,255,.18);
                ">SR</div>
                """,
                unsafe_allow_html=True,
            )

        with brand_col2:
            st.markdown(
                "**Smart Research AI**",
                help="AI-powered research workstation",
            )
            st.caption("Research workstation")

        st.markdown("---")

        # ---------------------------------------------------------
        # NEW CHAT
        # ---------------------------------------------------------
        if st.button(
            "＋  New Chat",
            key="new_chat_button",
            use_container_width=True,
            type="primary",
        ):
            create_chat()
            st.session_state.app_mode = "Chat"
            st.rerun()

        # ---------------------------------------------------------
        # RECENT CHATS
        # ---------------------------------------------------------
        st.markdown(
            '<div class="sr-section-label">RECENT CHATS</div>',
            unsafe_allow_html=True,
        )

        chats = list(st.session_state.chats.values())

        if not chats:
            st.caption("Your conversations will appear here.")

        for chat in reversed(chats):
            chat_id = chat["id"]
            title = chat.get("title") or "New Chat"
            is_active = chat_id == st.session_state.active_chat_id

            label = f"●  {title}" if is_active else title

            if st.button(
                label,
                key=f"select_chat_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.active_chat_id = chat_id
                st.session_state.app_mode = "Chat"
                st.rerun()

        # ---------------------------------------------------------
        # RESEARCH
        # ---------------------------------------------------------
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        if st.button(
            "⌕  Research Search",
            key="research_search_button",
            use_container_width=True,
        ):
            st.session_state.app_mode = "Research"
            st.rerun()

        # ---------------------------------------------------------
        # FLEXIBLE SPACE
        # ---------------------------------------------------------
        st.markdown(
            """
            <div style="
                height:18vh;
                min-height:70px;
                max-height:180px;
            "></div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------
        # ACTIVE GITHUB CONTEXT
        # ---------------------------------------------------------
        current_chat = st.session_state.chats.get(
            st.session_state.active_chat_id
        )

        active_repo = (
            current_chat.get("github_url")
            if current_chat
            else None
        )

        if active_repo:
            repo_path = (
                active_repo
                .rstrip("/")
                .split("github.com/")[-1]
            )

            parts = repo_path.split("/")
            display_repo = (
                f"{parts[0]}/{parts[1]}"
                if len(parts) >= 2
                else repo_path
            )

            st.markdown(
                f"""
                <div class="sr-repo">
                    <div class="sr-repo-label">ACTIVE CONTEXT</div>
                    <div style="
                        display:flex;
                        align-items:center;
                        gap:7px;
                        margin-top:5px;
                        color:#9AA7B5;
                        font-size:11px;
                    ">
                        <span class="sr-dot"></span>
                        GitHub repository
                    </div>
                    <div class="sr-repo-name">{display_repo}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------
        # FOOTER
        # ---------------------------------------------------------
        st.caption("GitHub context persists within each conversation.")


# ============================================================================
# CHAT UI
# ============================================================================

def render_chat() -> None:
    render_chat_sidebar()
    chat=active_chat()
    active_repo=chat.get("github_url")
    context_html=""
    if active_repo:
        repo=active_repo.rstrip("/").split("/")[-1]
        context_html=f'<div class="sr-context"><span class="sr-dot"></span>GitHub · {repo}</div>'

    st.markdown(
        f"""<div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:.8rem;">
        <div><div style="font-size:18px;font-weight:700;letter-spacing:-.025em;">Smart Research AI</div>
        <div style="font-size:11px;color:#657383;margin-top:2px;">{chat.get("title","New Chat")}</div></div>
        <div>{context_html}</div></div>""", unsafe_allow_html=True
    )

    messages=chat.get("messages",[])
    if not messages:
        st.markdown(
            """<div class="sr-hero"><div class="sr-eyebrow">AI research workstation</div>
            <h1>What can I help you investigate?</h1>
            <p>Explore repositories, understand implementations, search research, retrieve evidence,
            and turn technical questions into clear answers.</p></div>""", unsafe_allow_html=True
        )
        cols=st.columns(2)
        for i,suggestion in enumerate([
            "Explain this GitHub repository","How is authentication implemented?",
            "Find the main architecture","Compare these two approaches"
        ]):
            with cols[i%2]:
                if st.button(suggestion,key=f"empty_prompt_{i}",use_container_width=True):
                    st.session_state.pending_prompt=suggestion
                    st.rerun()

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata") and message["role"]=="assistant":
                render_chat_diagnostics(message["metadata"])

    prompt=st.chat_input("Message Smart Research AI…")
    pending=st.session_state.pop("pending_prompt",None)
    if pending and not prompt:
        prompt=pending
    if not prompt:
        return
    prompt=prompt.strip()
    if not prompt:
        return

    history=[{"role":m["role"],"content":m["content"]} for m in messages if m.get("role") in {"user","assistant"}]
    chat["messages"].append({"role":"user","content":prompt})
    if len(chat["messages"])==1:
        chat["title"]=chat_title(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Researching and synthesizing…"):
            try:
                data=call_backend(prompt,history,chat["id"])
                answer=data["answer"]
                # Preserve the stable chat_id architecture and accept either backend context key.
                active_repo=data.get("active_github_repository") or data.get("github_url")
                if active_repo:
                    chat["github_url"]=active_repo
                st.markdown(answer)
                chat["messages"].append({"role":"assistant","content":answer,"metadata":data})
                render_chat_diagnostics(data)
            except Exception as exc:
                show_backend_error(exc)
                if chat["messages"] and chat["messages"][-1]["role"]=="user":
                    chat["messages"].pop()




def render_chat_diagnostics(data: dict[str, Any]) -> None:
    summary=[]
    if data.get("model"): summary.append(str(data["model"]))
    if data.get("context_origin"): summary.append(str(data["context_origin"]))
    if data.get("chunks_retrieved"): summary.append(f"{data['chunks_retrieved']} evidence chunks")
    if summary:
        st.markdown('<div class="sr-diagnostics">◦ '+' &nbsp;·&nbsp; '.join(summary)+'</div>',unsafe_allow_html=True)

    sources=data.get("sources") or []
    if not sources:
        return
    with st.expander(f"Evidence · {len(sources)} chunks"):
        for index,source in enumerate(sources,1):
            path=source.get("source") or "Unknown source"
            section=source.get("section") or "Unknown section"
            st.markdown(f"**{index:02d}  {path}**  \n<span style='color:#657383;font-size:11px'>{section}</span>",unsafe_allow_html=True)
            if source.get("content"):
                st.code(source["content"],language="text")
            cols=st.columns(4)
            for col,key,label in [
                (cols[0],"query_relevance_score","Relevance"),
                (cols[1],"semantic_score","Semantic"),
                (cols[2],"mmr_score","MMR"),
                (cols[3],"complementarity_score","Complementarity"),
            ]:
                score=source.get(key)
                if score is not None: col.caption(f"{label}  {score:.3f}")
            if index<len(sources): st.divider()




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
    source=str(getattr(result,"source","") or "").lower()
    with st.container(border=True):
        st.markdown(f'<div class="sr-source-badge">{source_name(source)}</div>',unsafe_allow_html=True)
        st.markdown(f"### {getattr(result,'title','Untitled')}")
        if source=="github": display_github_metadata(result)
        elif source=="huggingface": display_huggingface_metadata(result)
        elif source=="arxiv": display_arxiv_metadata(result)
        elif source=="paperswithcode": display_paperswithcode_metadata(result)
        description=getattr(result,"description",None)
        if description: st.write(description)
        tags=getattr(result,"tags",None)
        if tags: st.caption(" · ".join(str(x) for x in tags))
        cols=st.columns(2)
        published=getattr(result,"published",None)
        if published:
            with cols[0]:
                st.caption("Published")
                try: st.write(published.strftime("%Y-%m-%d"))
                except Exception: st.write(str(published))
        updated=getattr(result,"updated",None)
        if updated:
            with cols[1]:
                st.caption("Updated")
                try: st.write(updated.strftime("%Y-%m-%d %H:%M:%S %Z"))
                except Exception: st.write(str(updated))
        url=getattr(result,"url",None)
        if url: st.link_button("Open source ↗",str(url),use_container_width=True)
        research_ask_ai(result)




def render_research() -> None:
    with st.sidebar:
        st.markdown(
            """<div class="sr-brand"><div class="sr-mark">SR</div><div>
            <div class="sr-brand-title">Smart Research AI</div>
            <div class="sr-brand-sub">Research workspace</div></div></div>""",unsafe_allow_html=True
        )
        if st.button("←  Back to Chat",use_container_width=True):
            st.session_state.app_mode="Chat"
            st.rerun()
        st.markdown('<div class="sr-section-label">RESEARCH SOURCES</div>',unsafe_allow_html=True)
        st.caption("GitHub · arXiv · PapersWithCode · Hugging Face")

    st.markdown(
        """<div class="sr-research-head"><div class="sr-eyebrow">Research workspace</div>
        <h1>Search technical knowledge</h1><div style="color:#9AA7B5;font-size:13px;">
        Search papers, repositories, implementations, and models from one place.</div></div>""",
        unsafe_allow_html=True
    )
    query=st.text_input("Search research",placeholder="e.g. models that understand images and classify them",key="research_query",label_visibility="collapsed")
    if query!=st.session_state.last_query:
        st.session_state.page=1
        st.session_state.ai_answers={}
        st.session_state.last_query=query

    controls=st.columns([1.6,1.1,1.1])
    with controls[0]:
        selected_sources=st.multiselect("Sources",options=AVAILABLE_SOURCES,default=AVAILABLE_SOURCES,format_func=source_name)
    with controls[1]:
        sort_option=st.selectbox("Sort",["Most relevant","Newest","Recently updated"])
    with controls[2]:
        search_mode=st.selectbox("Mode",["Keyword","Semantic","Hybrid"])

    sort_by={"Most relevant":"relevance","Newest":"published","Recently updated":"updated"}[sort_option]
    search_mode_value={"Keyword":"keyword","Semantic":"semantic","Hybrid":"hybrid"}[search_mode]
    if not query:
        st.markdown("""<div style="padding:4rem 1rem;text-align:center;color:#657383;">
        <div style="font-size:13px;font-weight:600;color:#9AA7B5;">Start with a technical question</div>
        <div style="font-size:12px;margin-top:.4rem;">Try an architecture, model, paper, repository, or implementation topic.</div>
        </div>""",unsafe_allow_html=True)
        return
    if not selected_sources:
        st.warning("Select at least one source.")
        return
    try:
        service=ResearchService()
        try:
            results=service.search(query,sources=selected_sources,sort_by=sort_by,search_mode=search_mode_value)
        except TypeError:
            results=service.search(query,sources=selected_sources,sort_by=sort_by)
    except Exception as exc:
        st.error("Search failed.")
        st.exception(exc)
        return

    st.markdown(
        f"""<div style="display:flex;justify-content:space-between;align-items:center;margin:1.4rem 0 .8rem;">
        <div style="font-size:14px;font-weight:600;color:#F3F6FA;">{len(results)} results</div>
        <div style="font-size:11px;color:#657383;">{search_mode} · {sort_option}</div></div>""",
        unsafe_allow_html=True
    )
    if search_mode=="Keyword": st.caption("Keyword search · matches explicit terms in research metadata.")
    elif search_mode=="Semantic": st.caption("Semantic search · finds results by conceptual similarity.")
    else: st.caption("Hybrid search · combines keyword and semantic relevance.")
    if not results:
        st.info("No results found. Try a different query or select more sources.")
        return

    total=len(results)
    total_pages=max(1,(total+RESULTS_PER_PAGE-1)//RESULTS_PER_PAGE)
    st.session_state.page=min(st.session_state.page,total_pages)
    current=st.session_state.page
    start=(current-1)*RESULTS_PER_PAGE
    end=min(start+RESULTS_PER_PAGE,total)
    st.caption(f"Showing {start+1}–{end} of {total}")
    for result in results[start:end]:
        display_result_card(result)
    if total_pages>1:
        cols=st.columns([1,2,1])
        with cols[0]:
            if st.button("← Previous",disabled=current<=1,use_container_width=True):
                st.session_state.page-=1; st.rerun()
        with cols[1]:
            st.markdown(f"<div style='text-align:center;padding-top:8px;color:#9AA7B5;font-size:12px;'>Page {current} of {total_pages}</div>",unsafe_allow_html=True)
        with cols[2]:
            if st.button("Next →",disabled=current>=total_pages,use_container_width=True):
                st.session_state.page+=1; st.rerun()




# ============================================================================
# ENTRY POINT
# ============================================================================

if not st.session_state.chats:
    create_chat()

if st.session_state.app_mode == "Research":
    render_research()
else:
    render_chat()

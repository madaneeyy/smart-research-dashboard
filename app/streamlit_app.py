from importlib import metadata
import sys
import uuid
import json
from pathlib import Path
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo
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


# ---------------------------------------------------------------------------
# PROFESSIONAL DOCUMENT SOURCE UI
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Sidebar spacing */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
        padding-bottom: 1.5rem;
    }

    /* Source section */
    .source-panel {
        margin: 0.25rem 0 0.75rem 0;
    }

    .source-heading {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .source-subtitle {
        font-size: 0.72rem;
        line-height: 1.35;
        margin-bottom: 0.55rem;
    }

    .upload-hint {
        border: 1px dashed rgba(128, 128, 128, 0.55);
        border-radius: 12px;
        padding: 0.7rem 0.75rem;
        text-align: center;
        margin-top: 0.45rem;
        font-size: 0.74rem;
        line-height: 1.35;
    }

    .document-count {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.76rem;
        font-weight: 600;
        margin: 0.15rem 0 0.45rem 0;
    }

    .document-empty {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 0.6rem 0.7rem;
        font-size: 0.74rem;
        margin-bottom: 0.5rem;
    }

    /* Keep sidebar controls visually compact */
    section[data-testid="stSidebar"] .stSelectbox label {
        font-size: 0.76rem;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        margin-top: 0.35rem;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
        min-height: 96px;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
        font-size: 0.74rem;
    }

    /* Selected source indicator */
    .active-source {
        border-radius: 9px;
        padding: 0.45rem 0.6rem;
        margin-top: 0.45rem;
        font-size: 0.72rem;
        line-height: 1.35;
    }

    /* =========================================================
       SOURCES PAGE
       ========================================================= */

    .sources-page-header {
        margin-bottom: 1.1rem;
    }

    .sources-workspace-name {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.65;
        margin-bottom: 0.35rem;
    }

    .sources-page-title {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 0.3rem;
    }

    .sources-page-description {
        font-size: 0.92rem;
        line-height: 1.5;
        opacity: 0.68;
    }

    .sources-summary {
        display: flex;
        align-items: baseline;
        gap: 0.4rem;
        margin: 0.9rem 0 0.8rem 0;
        font-size: 0.86rem;
    }

    .sources-summary span {
        opacity: 0.55;
    }

    .source-card-header {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        min-height: 3.2rem;
        margin-bottom: 0.55rem;
    }

    .source-icon {
        width: 2.15rem;
        height: 2.15rem;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 auto;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.08);
        font-size: 1.1rem;
    }

    .source-card-title-area {
        min-width: 0;
    }

    .source-card-title {
        font-size: 1.02rem;
        font-weight: 650;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .source-card-provider {
        font-size: 0.77rem;
        opacity: 0.58;
        margin-top: 0.18rem;
    }

    .source-card-description {
        font-size: 0.86rem;
        line-height: 1.5;
        opacity: 0.72;
        margin: 0.65rem 0 0.85rem 0;
    }

    .source-card-url {
        font-size: 0.74rem;
        opacity: 0.52;
        margin-top: 0.1rem;
        overflow-wrap: anywhere;
    }

    .source-card-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.55rem;
    }

    .source-meta-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        background: rgba(128, 128, 128, 0.08);
        font-size: 0.7rem;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    "app_mode": "Workspace",  # or "Chat", "Research", or "Sources"
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value
    
if "workspaces" not in st.session_state:
        st.session_state.workspaces = []

if "active_workspace_id" not in st.session_state:
        st.session_state.active_workspace_id = None
if "workspace_toast" not in st.session_state:
    st.session_state.workspace_toast = None
if "sources_add_mode" not in st.session_state:
    st.session_state.sources_add_mode = None
if "selected_workspace_document_id" not in st.session_state:
    st.session_state.selected_workspace_document_id = None

if "selected_workspace_source_id" not in st.session_state:
    st.session_state.selected_workspace_source_id = None
    
if "selected_workspace_document_name" not in st.session_state:
    st.session_state.selected_workspace_document_name = None




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

        # Documents attached to this chat.
        "document_ids": [],
        "document_names": {},

        # None means "All documents".
        "selected_document_id": None,

        # Prevent duplicate uploads after Streamlit reruns.
        "uploaded_file_keys": set(),
    }

    st.session_state.active_chat_id = chat_id
    return chat_id

def load_persistent_chat(chat_id: str) -> dict[str, Any]:
    """
    Load a persistent chat from FastAPI and convert it into
    the local chat structure expected by the existing Chat UI.
    """

    chat_response = requests.get(
        f"{BACKEND_URL}/chats/{chat_id}",
        timeout=30,
    )
    chat_response.raise_for_status()
    chat_data = chat_response.json()

    messages_response = requests.get(
        f"{BACKEND_URL}/chats/{chat_id}/messages",
        timeout=30,
    )
    messages_response.raise_for_status()
    messages_data = messages_response.json()

    sources_response = requests.get(
        f"{BACKEND_URL}/chats/{chat_id}/sources",
        timeout=30,
    )
    sources_response.raise_for_status()
    sources_data = sources_response.json()

    document_ids: list[str] = []
    document_names: dict[str, str] = {}

    for source in sources_data:
        if source.get("source_type") == "document":
            source_id = source.get("source_id")

            if not source_id:
                continue

            source_id = str(source_id)

            if source_id not in document_ids:
                document_ids.append(source_id)

            document_names.setdefault(
                source_id,
                "Workspace document",
            )

    messages: list[dict[str, Any]] = []

    for message in messages_data:
        messages.append(
            {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
            }
        )

    return {
        "id": str(chat_data["id"]),
        "title": chat_data.get("title") or "New Chat",
        "messages": messages,
        "github_url": None,
        "document_ids": document_ids,
        "document_names": document_names,
        "selected_document_id": (
            document_ids[0]
            if document_ids
            else None
        ),
        "uploaded_file_keys": set(),
    }
def active_chat() -> dict[str, Any]:
    chat_id = st.session_state.active_chat_id

    if not chat_id:
        create_chat()
        return st.session_state.chats[
            st.session_state.active_chat_id
        ]

    if chat_id not in st.session_state.chats:
        try:
            chat = load_persistent_chat(
                str(chat_id)
            )

            st.session_state.chats[str(chat_id)] = chat

        except requests.RequestException as exc:
            st.error(
                f"Could not load chat: {exc}"
            )

            create_chat()

    return st.session_state.chats[
        st.session_state.active_chat_id
    ]

def chat_title(question: str) -> str:
    text = " ".join(question.strip().split())
    return text if len(text) <= 42 else text[:39].rstrip() + "..."

def create_persistent_chat(
    workspace_id: str,
    title: str,
    source_type: str,
    source_id: str,
) -> dict[str, Any]:
    response = requests.post(
        f"{BACKEND_URL}/chats",
        json={
            "workspace_id": workspace_id,
            "title": title,
            "source_type": source_type,
            "source_id": source_id,
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()

def save_chat_message(
    chat_id: str,
    role: str,
    content: str,
) -> dict[str, Any]:
    response = requests.post(
        f"{BACKEND_URL}/chats/{chat_id}/messages",
        json={
            "role": role,
            "content": content,
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()

def call_backend(
    question: str,
    history: list[dict[str, str]],
    chat_id: str,
    document_ids: list[str] | None = None,
    answer_placeholder: Any | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call FastAPI and stream Qwen tokens without changing the final payload."""
    payload = {
        "question": question.strip(),
        "history": history,
        "top_k": 5,
        "chat_id": chat_id,
        "document_ids": document_ids,
    }
    if extra_payload:
        payload.update(extra_payload)

    response = requests.post(
        f"{BACKEND_URL}/ask",
        params={"stream": "true"},
        json=payload,
        timeout=300,
        stream=True,
        headers={"Accept": "text/event-stream"},
    )
    response.raise_for_status()

    answer_parts: list[str] = []
    final_data: dict[str, Any] | None = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue

        line = raw_line.strip()
        if not line.startswith("data: "):
            continue

        try:
            event = json.loads(line[6:])
        except Exception as exc:
            raise ValueError(f"Invalid streaming response from FastAPI: {line}") from exc

        event_type = event.get("type")

        if event_type == "token":
            text = str(event.get("content") or "")
            if text:
                answer_parts.append(text)
                if answer_placeholder is not None:
                    answer_placeholder.markdown("".join(answer_parts))

        elif event_type == "done":
            data = event.get("data")
            if isinstance(data, dict):
                final_data = data

        elif event_type == "error":
            raise RuntimeError(str(event.get("error") or "Streaming generation failed."))

    if final_data is None:
        raise ValueError("FastAPI ended the stream without a final response payload.")

    answer = (
        final_data.get("answer")
        or "".join(answer_parts)
        or final_data.get("response")
        or final_data.get("output")
    )

    if not answer:
        raise ValueError(
            "FastAPI returned no AI answer. "
            f"Response: {final_data}"
        )

    final_data["answer"] = answer
    return final_data

BACKEND_URL = "http://127.0.0.1:8000"


def fetch_workspaces() -> list[dict[str, Any]]:
    response = requests.get(
        f"{BACKEND_URL}/workspaces",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

def fetch_recent_chats(
    workspace_id: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    response = requests.get(
        f"{BACKEND_URL}/chats/workspace/{workspace_id}",
        timeout=10,
    )

    response.raise_for_status()

    chats = response.json()

    if not isinstance(chats, list):
        return []

    return chats[:limit]

def fetch_workspace_counts(
    workspace_id: str,
) -> dict[str, int]:
    """
    Return source/document counts for the active workspace.
    """

    sources_response = requests.get(
        f"{BACKEND_URL}/workspaces/{workspace_id}/sources",
        timeout=10,
    )
    sources_response.raise_for_status()

    documents_response = requests.get(
        f"{BACKEND_URL}/workspaces/{workspace_id}/documents",
        timeout=10,
    )
    documents_response.raise_for_status()

    sources = sources_response.json()
    documents = documents_response.json()

    if not isinstance(sources, list):
        sources = []

    if not isinstance(documents, list):
        documents = []

    papers = 0
    repositories = 0

    for source in sources:
        source_type = str(
            source.get("source_type") or ""
        ).lower()

        if source_type in {
            "arxiv",
            "paperswithcode",
        }:
            papers += 1

        elif source_type == "github":
            repositories += 1

    return {
        "papers": papers,
        "repositories": repositories,
        "documents": len(documents),
    }


def get_current_workspace_source_urls() -> set[str]:
    workspace_id = (
        st.session_state.active_workspace_id
    )

    if not workspace_id:
        return set()

    try:
        response = requests.get(
            f"{BACKEND_URL}/workspaces/{workspace_id}/sources",
            timeout=10,
        )

        response.raise_for_status()

        sources = response.json()

        return {
            (
                str(source.get("source_type", "")).lower()
                + "|"
                + str(source.get("url", "")).strip()
            )
            for source in sources
            if source.get("url")
        }

    except Exception as exc:
        st.warning(
            f"Could not load workspace sources: {exc}"
        )
        return set()

def create_workspace(
    name: str,
    description: str | None = None,
) -> dict[str, Any]:
    response = requests.post(
        f"{BACKEND_URL}/workspaces",
        json={
            "name": name,
            "description": description,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

def make_json_safe(value: Any) -> Any:
    """
    Recursively convert arbitrary Python values into
    JSON-compatible values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    # Pydantic models
    if hasattr(value, "model_dump"):
        return make_json_safe(
            value.model_dump()
        )

    # Date / datetime / URL-like values
    if hasattr(value, "isoformat"):
        return value.isoformat()

    # Everything else gets a readable representation.
    return str(value)


def serialize_research_result(
    result: Any,
) -> dict[str, Any]:
    """
    Extract the useful ResearchItem fields into a
    JSON-safe metadata dictionary.
    """

    def safe(value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {
                str(key): safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                safe(item)
                for item in value
            ]

        if hasattr(value, "isoformat"):
            return value.isoformat()

        return str(value)

    metadata = {
        "research_id": getattr(
            result,
            "id",
            None,
        ),

        "title": getattr(
            result,
            "title",
            None,
        ),

        "description": getattr(
            result,
            "description",
            None,
        ),

        "authors": getattr(
            result,
            "authors",
            [],
        ),

        "source": getattr(
            result,
            "source",
            None,
        ),

        "url": (
            str(getattr(result, "url", ""))
            if getattr(result, "url", None)
            else None
        ),

        "published": getattr(
            result,
            "published",
            None,
        ),

        "updated": getattr(
            result,
            "updated",
            None,
        ),

        "tags": getattr(
            result,
            "tags",
            [],
        ),

        # GitHub
        "stars": getattr(
            result,
            "stars",
            None,
        ),

        "forks": getattr(
            result,
            "forks",
            None,
        ),

        "language": getattr(
            result,
            "language",
            None,
        ),

        # Hugging Face
        "downloads": getattr(
            result,
            "downloads",
            None,
        ),

        "likes": getattr(
            result,
            "likes",
            None,
        ),

        "library": getattr(
            result,
            "library",
            None,
        ),

        "pipeline_tag": getattr(
            result,
            "pipeline_tag",
            None,
        ),

        # PapersWithCode
        "tasks": getattr(
            result,
            "tasks",
            [],
        ),

        "conference": getattr(
            result,
            "conference",
            None,
        ),

        # Flexible provider-specific metadata
        "provider_metadata": getattr(
            result,
            "metadata",
            {},
        ),
    }

    return safe(metadata)

def add_source_to_workspace(
    workspace_id: str,
    source_type: str,
    title: str,
    url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    
    print(
       "ADDING SOURCE:",
        source_type,
        title,
        url,
    )

    payload = {
        "source_type": source_type,
        "title": title,
        "url": url,
        "metadata": metadata or {},
    }

    # ------------------------------------------------------------
    # Convert EVERYTHING in the payload into JSON-safe values.
    #
    # default=str means that if a provider has accidentally put
    # something unusual (module, datetime, URL object, etc.) into
    # metadata, it becomes a string instead of crashing.
    # ------------------------------------------------------------

    payload = json.loads(
        json.dumps(
            payload,
            default=str,
            allow_nan=False,
        )
    )

    response = requests.post(
        f"{BACKEND_URL}/workspaces/{workspace_id}/sources",
        json=payload,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()

def add_research_result_to_workspace(
    source_type: str,
    title: str,
    url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:

    workspace_id = (
        st.session_state.active_workspace_id
    )

    if not workspace_id:
        st.warning(
            "Please select a workspace first."
        )
        return False

    try:
        add_source_to_workspace(
            workspace_id=str(workspace_id),
            source_type=source_type,
            title=title,
            url=url,
            metadata=metadata,
        )

        return True

    except requests.RequestException as exc:
        st.error(
            f"Could not add source to workspace: {exc}"
        )
        return False

def render_workspace_selector() -> None:
    st.markdown("### 📁 Workspace")

    try:
        workspaces = fetch_workspaces()

    except requests.RequestException as exc:
        st.error(f"Could not load workspaces: {exc}")
        return

    # Keep the latest workspace list in Streamlit state.
    st.session_state.workspaces = workspaces

    # ------------------------------------------------------------
    # No workspaces exist
    # ------------------------------------------------------------

    if not workspaces:
        st.info("No workspaces yet.")
        return

    # ------------------------------------------------------------
    # Make sure an active workspace always exists.
    # ------------------------------------------------------------

    active_id = st.session_state.active_workspace_id

    workspace_ids = {
        str(workspace["id"])
        for workspace in workspaces
    }

    if active_id is None or str(active_id) not in workspace_ids:
        active_id = str(workspaces[0]["id"])
        st.session_state.active_workspace_id = active_id

    # ------------------------------------------------------------
    # Workspace selector
    # ------------------------------------------------------------

    selected_index = 0

    for index, workspace in enumerate(workspaces):
        if str(workspace["id"]) == str(active_id):
            selected_index = index
            break

    selected_workspace = st.selectbox(
        "Current workspace",
        options=workspaces,
        index=selected_index,
        format_func=lambda workspace: (
            f"🔬 {workspace.get('name', 'Untitled Workspace')}"
        ),
        key="workspace_selector",
        label_visibility="collapsed",
    )

    selected_id = str(selected_workspace["id"])

    # ------------------------------------------------------------
    # Workspace changed
    # ------------------------------------------------------------

    if selected_id != str(st.session_state.active_workspace_id):

        st.session_state.active_workspace_id = selected_id

        # Switching workspace should always return
        # to that workspace's overview.
        st.session_state.app_mode = "Workspace"

        st.rerun()

def render_create_workspace() -> None:
    with st.popover("＋ New Workspace", use_container_width=True):
        st.markdown("#### Create workspace")

        name = st.text_input(
            "Name",
            placeholder="e.g. Vision Research",
        )

        description = st.text_area(
            "Description",
            placeholder="What are you researching?",
            height=90,
        )

        if st.button(
            "Create Workspace",
            type="primary",
            use_container_width=True,
        ):
            if not name.strip():
                st.warning("Please enter a workspace name.")
                return

            try:
                workspace = create_workspace(
                    name=name.strip(),
                    description=description.strip() or None,
                )

                st.session_state.active_workspace_id = workspace["id"]
                st.session_state.app_mode = "Workspace"

                st.success(
                    f"Created '{workspace['name']}'."
                )

                st.rerun()

            except requests.RequestException as exc:
                st.error(
                    f"Could not create workspace: {exc}"
                )

def render_workspace_sidebar() -> None:
    """
    Render workspace-level navigation.

    This is intentionally separate from the chat sidebar because
    workspace selection should remain available across the app.
    """

    with st.sidebar:

        st.markdown("## 🔎 Smart Research AI")

        render_workspace_selector()

        render_create_workspace()

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

def render_document_selector(
    chat: dict[str, Any],
) -> list[str]:
    """
    Render the document source selector.

    The selector is intentionally always visible:
    - no documents -> disabled-looking empty state
    - one/multiple documents -> All documents + individual files

    Returns the document IDs to send to the backend.
    """

    document_ids = chat.get("document_ids", [])
    document_names = chat.get("document_names", {})

    st.markdown(
        '<div class="source-panel">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="source-heading">📎 Documents</div>',
        unsafe_allow_html=True,
    )

    if not document_ids:
        st.markdown(
            '<div class="document-empty">'
            'No documents attached yet.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Sidebar drag/drop uploader is always available.
        st.file_uploader(
            "Drop files here or browse",
            type=DOCUMENT_FILE_TYPES,
            accept_multiple_files=True,
            key=f"sidebar_document_uploader_{chat['id']}",
            label_visibility="collapsed",
            help=(
                "Upload one or more documents. "
                "They will be indexed and become available "
                "as sources for your questions."
            ),
        )

        st.markdown(
            '<div class="upload-hint">'
            '<strong>Drop files here</strong><br>'
            'or click to browse<br>'
            '<span>PDF, DOCX, TXT, MD, Python, JS and more</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        return []

    count = len(document_ids)

    st.markdown(
        f'<div class="document-count">'
        f'📎 {count} document'
        f'{"s" if count != 1 else ""} attached'
        f'</div>',
        unsafe_allow_html=True,
    )

    options: list[tuple[str, str | None]] = [
        ("All documents", None)
    ]

    for document_id in document_ids:
        filename = document_names.get(
            document_id,
            f"Document {document_id[:8]}",
        )
        options.append((filename, document_id))

    labels = [label for label, _ in options]

    selected_document_id = chat.get(
        "selected_document_id"
    )

    selected_index = 0

    if selected_document_id in document_ids:
        for index, (_, document_id) in enumerate(options):
            if document_id == selected_document_id:
                selected_index = index
                break

    selected_label = st.selectbox(
        "Document to use",
        options=labels,
        index=selected_index,
        key=f"document_selector_{chat['id']}",
        help=(
            "Choose which document(s) the RAG system "
            "should use for your next question."
        ),
    )

    selected_id = None

    for label, document_id in options:
        if label == selected_label:
            selected_id = document_id
            break

    chat["selected_document_id"] = selected_id

    if selected_id is None:
        st.markdown(
            '<div class="active-source">'
            '📚 <strong>Searching all attached documents</strong>'
            '</div>',
            unsafe_allow_html=True,
        )

        selected_ids = document_ids.copy()

    else:
        selected_name = document_names.get(
            selected_id,
            "Selected document",
        )

        st.markdown(
            f'<div class="active-source">'
            f'📄 <strong>Using:</strong> '
            f'{selected_name}'
            f'</div>',
            unsafe_allow_html=True,
        )

        selected_ids = [selected_id]

    # Sidebar drag/drop uploader remains available after documents exist.
    st.file_uploader(
        "Add more documents",
        type=DOCUMENT_FILE_TYPES,
        accept_multiple_files=True,
        key=f"sidebar_document_uploader_{chat['id']}",
        label_visibility="collapsed",
        help="Drop files here or click to add more documents.",
    )

    st.markdown(
        '<div class="upload-hint">'
        '＋ <strong>Add more documents</strong><br>'
        'Drop files here or click to browse'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    return selected_ids


def get_sidebar_uploaded_files(
    chat: dict[str, Any],
) -> list[Any]:
    """
    Retrieve files selected in the sidebar uploader for this chat.

    Streamlit stores uploader state in session_state using the same key
    used by st.file_uploader.
    """
    key = f"sidebar_document_uploader_{chat['id']}"

    files = st.session_state.get(key)

    if not files:
        return []

    if not isinstance(files, list):
        return [files]

    return files


def render_chat_sidebar() -> list[str]:
    """
    Render chat navigation/settings and the document selector.

    Returns the document IDs selected for the next chat question.
    """

    selected_document_ids: list[str] = []

    # Workspace navigation is shared across the application.
    render_workspace_sidebar()

    with st.sidebar:

        st.divider()

        if st.button(
            "＋ New Chat",
            use_container_width=True,
            type="primary",
        ):
            create_chat()
            st.rerun()

        st.divider()

        st.caption("CHATS")

        chats = list(
            st.session_state.chats.values()
        )

        if not chats:
            st.caption("No conversations yet.")

        for chat_item in reversed(chats):

            chat_id = chat_item["id"]

            label = (
                chat_item.get("title")
                or "New Chat"
            )

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
            st.caption(
                "📦 Active GitHub repository"
            )

            st.code(
                active_repo,
                language="text",
            )

        attached_documents = (
            current_chat.get("document_ids", [])
            if current_chat
            else []
        )

        # Always render the document source panel.
        if current_chat:
            selected_document_ids = (
                render_document_selector(
                    current_chat
                )
            )

        st.caption(
            "Attach documents directly from the "
            "paperclip button inside the message box."
        )

    return selected_document_ids


def render_workspace_overview() -> None:
    render_workspace_sidebar()

    workspace_id = st.session_state.active_workspace_id

    if not workspace_id:
        st.info("Select or create a workspace to get started.")
        return

    workspace = next(
        (
            item
            for item in st.session_state.workspaces
            if str(item["id"]) == str(workspace_id)
        ),
        None,
    )

    if workspace is None:
        st.warning("The selected workspace could not be found.")
        return

    name = workspace.get("name", "Untitled Workspace")
    description = (
        workspace.get("description")
        or "Your research workspace."
    )

    # Header
    st.markdown("**RESEARCH WORKSPACE**")
    st.markdown(f"# 🔬 {name}")
    st.caption(description)


    st.markdown("### Get started")

    col1, col2,col3= st.columns(3)

    with col1:
        with st.container(border=True,height=220):
            st.markdown("#### 🔎 Discover research")
            st.caption(
                "Find papers, repositories, models and datasets."
            )

            if st.button(
                "Discover research →",
                key="workspace_discover",
                use_container_width=True,
            ):
                st.session_state.app_mode = "Research"
                st.rerun()

    with col2:
        with st.container(border=True,height=220):
            st.markdown("#### 📚  Sources")
            st.caption(
                "Browse, organize, and investigate your collected sources."
            )

            if st.button(
                "View sources →",
                key="workspace_sources",
                use_container_width=True,
            ):
                st.session_state.app_mode = "Sources"

                st.rerun()

    with col3:
        with st.container(border=True,height=220):
            st.markdown("#### 💬 Ask AI")
            st.caption(
                "Investigate the research you've collected."
            )

            if st.button(
                "Open research chat →",
                key="workspace_chat",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.app_mode = "Chat"

                if not st.session_state.active_chat_id:
                    create_chat()

                st.rerun()

    st.markdown("")
        # ------------------------------------------------------------------
    # Recent chats
    # ------------------------------------------------------------------

    st.markdown("")
    st.markdown("### Recent chats")

    try:
        recent_chats = fetch_recent_chats(
            workspace_id=str(workspace_id),
            limit=5,
        )
    except requests.RequestException as exc:
        st.error(
            f"Could not load recent chats: {exc}"
        )
        recent_chats = []

    if not recent_chats:
        st.caption(
            "No conversations yet. Open a source and click Ask AI to start one."
        )
    else:
        for chat_item in recent_chats:
            chat_id = str(chat_item.get("id"))
            title = (
                chat_item.get("title")
                or "New Chat"
            )

            updated_at = chat_item.get(
                "updated_at"
            )

            updated_text = ""

            if updated_at:
                try:
                    timestamp=datetime.fromisoformat(
                        str(updated_at).replace("Z","+00:00")
                    )
                    local_timestamp = timestamp.astimezone(
                        ZoneInfo("Asia/Kathmandu")
                    )
                    updated_text = local_timestamp.strftime(
                        "%d %b %Y, %I:%M %p"
                    )
                except Exception:
                    updated_text = str(updated_at)

            chat_col, button_col = st.columns(
                [5, 1],
                gap="small",
            )

            with chat_col:
                st.markdown(
                    f"**💬 {title}**"
                )

                if updated_text:
                    st.caption(
                        f"Last updated: {updated_text}"
                    )
                else:
                    st.caption(
                        "Research conversation"
                    )

            with button_col:
                if st.button(
                    "Open →",
                    key=f"recent_chat_{chat_id}",
                    use_container_width=True,
                ):
                    st.session_state.active_chat_id = chat_id
                    st.session_state.app_mode = "Chat"

                    # Clear any stale Sources → Ask AI navigation state.
                    st.session_state.selected_workspace_document_id = None
                    st.session_state.selected_workspace_document_name = None
                    st.session_state.selected_workspace_source_id = None

                    st.rerun()

    st.markdown("")

    # ------------------------------------------------------------------
    # Your research
    # Keep this section outside the recent-chats conditional so every
    # workspace always shows its source/document counts, even when it
    # has no chats yet.
    # ------------------------------------------------------------------
    st.markdown("### Your research")

    try:
        workspace_counts = fetch_workspace_counts(
            workspace_id=str(workspace_id),
        )
    except requests.RequestException as exc:
        st.warning(
            f"Could not load workspace statistics: {exc}"
        )
        workspace_counts = {
            "papers": 0,
            "repositories": 0,
            "documents": 0,
        }

    stats = st.columns(3)

    with stats[0]:
        st.metric(
            "Papers",
            workspace_counts.get("papers", 0),
        )

    with stats[1]:
        st.metric(
            "Repositories",
            workspace_counts.get("repositories", 0),
        )

    with stats[2]:
        st.metric(
            "Documents",
            workspace_counts.get("documents", 0),
        )

    st.divider()

    st.caption(
        "Start by discovering research or adding material "
        "to this workspace."
    )
   
def render_sources() -> None:
    """Display research sources and uploaded documents for the active workspace."""

    render_workspace_sidebar()

    workspace_id = st.session_state.active_workspace_id

    if not workspace_id:
        st.info("Select or create a workspace to view its sources.")
        return

    workspace = next(
        (
            item
            for item in st.session_state.workspaces
            if str(item["id"]) == str(workspace_id)
        ),
        None,
    )

    if workspace is None:
        st.warning("The selected workspace could not be found.")
        return

    workspace_name = workspace.get("name", "Untitled Workspace")

    # ------------------------------------------------------------
    # Back to workspace overview
    # ------------------------------------------------------------
    if st.button(
        "← Workspace",
        key="sources_back_to_workspace",
    ):
        st.session_state.app_mode = "Workspace"
        st.rerun()

    st.markdown("")

    # ------------------------------------------------------------
    # Load both source collections
    # ------------------------------------------------------------
    try:
        sources_response = requests.get(
            f"{BACKEND_URL}/workspaces/{workspace_id}/sources",
            timeout=10,
        )
        sources_response.raise_for_status()
        research_sources = sources_response.json()

        documents_response = requests.get(
            f"{BACKEND_URL}/workspaces/{workspace_id}/documents",
            timeout=10,
        )
        documents_response.raise_for_status()
        workspace_documents = documents_response.json()

    except requests.RequestException as exc:
        st.error(f"Could not load workspace sources: {exc}")
        return

    if not isinstance(research_sources, list):
        research_sources = []

    if not isinstance(workspace_documents, list):
        workspace_documents = []

    # ------------------------------------------------------------
    # Normalize both backend collections into one UI model
    # ------------------------------------------------------------
    combined_sources: list[dict[str, Any]] = []

    for source in research_sources:
        metadata = source.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        combined_sources.append(
            {
                "kind": "research",
                "id": source.get("id"),
                "title": source.get("title") or "Untitled source",
                "source_type": str(
                    source.get("source_type") or ""
                ).lower(),
                "url": source.get("url"),
                "description": str(
                    metadata.get("description") or ""
                ).strip(),
                "metadata": metadata,
            }
        )

    for document in workspace_documents:
        pages = document.get("pages")
        status = document.get("status") or "ready"

        description = (
            f"{pages} pages · {status.title()}"
            if pages is not None
            else f"Uploaded document · {status.title()}"
        )

        combined_sources.append(
            {
                "kind": "document",
                "id": document.get("id"),
                "document_id": document.get("document_id"),
                "title": document.get("filename") or "Untitled document",
                "source_type": "document",
                "url": None,
                "description": description,
                "metadata": {
                    "content_type": document.get("content_type"),
                    "pages": pages,
                    "characters": document.get("characters"),
                    "size_bytes": document.get("size_bytes"),
                    "status": status,
                },
            }
        )

    source_type_labels = {
        "arxiv": "Papers",
        "github": "Repositories",
        "paperswithcode": "PapersWithCode",
        "huggingface": "Hugging Face",
        "document": "Documents",
        "documents": "Documents",
    }

    icon_map = {
        "arxiv": "📄",
        "github": "🐙",
        "paperswithcode": "📊",
        "huggingface": "🤗",
        "document": "📎",
        "documents": "📎",
    }

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------
    st.markdown(
        f"""
        <div class="sources-page-header">
            <div class="sources-workspace-name">🔬 {workspace_name}</div>
            <div class="sources-page-title">Sources</div>
            <div class="sources-page-description">
                View and manage everything you've collected for this workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Add sources popover
    # ------------------------------------------------------------
    _, top_action = st.columns([5, 1])

    with top_action:
        with st.popover(
            "＋ Add sources",
            use_container_width=True,
        ):
            st.markdown("### Add sources")
            st.caption(f"Add material to **{workspace_name}**")
            st.divider()

            st.markdown("#### 📄 Add documents")
            st.caption(
                "Upload papers, notes, code, datasets, or other research material."
            )

            if st.button(
                "Upload files →",
                key="sources_upload_documents",
                use_container_width=True,
            ):
                st.session_state.sources_add_mode = "documents"
                st.rerun()

            st.markdown("")
            st.markdown("#### 🔎 Discover research")
            st.caption(
                "Find papers, repositories, models, and datasets to add to this workspace."
            )

            if st.button(
                "Discover research →",
                key="sources_discover_research",
                use_container_width=True,
            ):
                st.session_state.sources_add_mode = "research"
                st.session_state.app_mode = "Research"
                st.rerun()

    # ------------------------------------------------------------
    # Workspace document upload
    # ------------------------------------------------------------
    if st.session_state.sources_add_mode == "documents":
        st.markdown("---")
        st.markdown("### 📄 Add documents")
        st.caption(
            f"Upload documents to **{workspace_name}**. "
            "They will become available as sources in this workspace."
        )

        uploaded_files = st.file_uploader(
            "Choose files",
            type=DOCUMENT_FILE_TYPES,
            accept_multiple_files=True,
            key="workspace_document_uploader",
            label_visibility="collapsed",
            help=(
                "Upload PDFs, DOCX, Markdown, text files, "
                "or supported source/code files."
            ),
        )

        upload_col1, upload_col2 = st.columns(2)

        with upload_col1:
            if st.button(
                "Add to workspace",
                type="primary",
                use_container_width=True,
                key="confirm_workspace_document_upload",
                disabled=not uploaded_files,
            ):
                successful_uploads: list[str] = []
                failed_uploads: list[str] = []

                for uploaded_file in uploaded_files:
                    try:
                        with st.spinner(f"Adding {uploaded_file.name}..."):
                            response = requests.post(
                                f"{BACKEND_URL}/workspaces/{workspace_id}/documents",
                                files={
                                    "file": (
                                        uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        uploaded_file.type
                                        or "application/octet-stream",
                                    )
                                },
                                timeout=120,
                            )
                            response.raise_for_status()
                            result = response.json()

                        successful_uploads.append(
                            result.get("filename", uploaded_file.name)
                        )

                    except requests.RequestException as exc:
                        failed_uploads.append(
                            f"{uploaded_file.name}: {exc}"
                        )
                    except Exception as exc:
                        failed_uploads.append(
                            f"{uploaded_file.name}: {exc}"
                        )

                if successful_uploads:
                    count = len(successful_uploads)
                    st.success(
                        f"Added {count} document{'s' if count != 1 else ''} "
                        "to the workspace."
                    )

                for error in failed_uploads:
                    st.error(f"Could not add document — {error}")

                if not failed_uploads:
                    st.session_state.sources_add_mode = None
                    if "workspace_document_uploader" in st.session_state:
                        del st.session_state["workspace_document_uploader"]
                    st.rerun()

        with upload_col2:
            if st.button(
                "Cancel",
                use_container_width=True,
                key="cancel_workspace_document_upload",
            ):
                st.session_state.sources_add_mode = None
                if "workspace_document_uploader" in st.session_state:
                    del st.session_state["workspace_document_uploader"]
                st.rerun()

    # ------------------------------------------------------------
    # Search + filter
    # ------------------------------------------------------------
    search_col, filter_col = st.columns([4, 1])

    with search_col:
        search_query = st.text_input(
            "Search sources",
            placeholder="🔎 Search your sources...",
            label_visibility="collapsed",
            key="sources_search",
        )

    source_types = sorted(
        {
            str(source.get("source_type", "")).lower()
            for source in combined_sources
            if source.get("source_type")
        }
    )

    filter_options = ["All"] + source_types

    with filter_col:
        selected_filter = st.selectbox(
            "Filter",
            filter_options,
            format_func=lambda value: (
                "All"
                if value == "All"
                else source_type_labels.get(
                    value,
                    value.replace("_", " ").title(),
                )
            ),
            label_visibility="collapsed",
            key="sources_filter",
        )

    # ------------------------------------------------------------
    # Filter sources in memory
    # ------------------------------------------------------------
    filtered_sources: list[dict[str, Any]] = []
    query = search_query.strip().lower()

    for source in combined_sources:
        source_type = str(
            source.get("source_type", "")
        ).lower()

        title = str(
            source.get("title", "")
        )

        url = str(
            source.get("url", "")
        )

        description = str(
            source.get("description", "")
        )

        metadata = source.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        if (
            selected_filter != "All"
            and source_type != selected_filter
        ):
            continue

        if query:
            searchable_text = " ".join(
                [
                    title,
                    description,
                    url,
                    source_type,
                    str(metadata.get("authors", "")),
                    str(metadata.get("tags", "")),
                    str(metadata.get("language", "")),
                    str(metadata.get("content_type", "")),
                ]
            ).lower()

            if query not in searchable_text:
                continue

        filtered_sources.append(source)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    st.markdown(
        f"""
        <div class="sources-summary">
            <strong>{len(filtered_sources)} source{'' if len(filtered_sources) == 1 else 's'}</strong>
            <span>in this workspace</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Empty state
    # ------------------------------------------------------------
    if not filtered_sources:
        if combined_sources:
            st.info("No sources match your search or filter.")
        else:
            st.info(
                "This workspace has no sources yet. "
                "Start by discovering research or adding documents."
            )
        return

    # ------------------------------------------------------------
    # Source cards
    # ------------------------------------------------------------
    for index, source in enumerate(filtered_sources):
        source_type = str(
            source.get("source_type", "")
        ).lower()
        source_kind = source.get("kind", "research")
        title = str(source.get("title") or "Untitled source")
        url = source.get("url")

        metadata = source.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        description = str(
            source.get("description") or ""
        ).strip()

        display_source_name = source_type_labels.get(
            source_type,
            source_type.replace("_", " ").title() or "Source",
        )
        icon = icon_map.get(source_type, "📚")

        authors = metadata.get("authors")
        tags = metadata.get("tags")
        language = metadata.get("language")

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="source-card-header">
                    <div class="source-icon">{icon}</div>
                    <div class="source-card-title-area">
                        <div class="source-card-title">{title}</div>
                        <div class="source-card-provider">{display_source_name}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if description:
                shortened_description = description
                if len(shortened_description) > 280:
                    shortened_description = (
                        shortened_description[:277].rstrip() + "..."
                    )
                st.markdown(
                    f'<div class="source-card-description">{shortened_description}</div>',
                    unsafe_allow_html=True,
                )

            meta_items: list[str] = []

            if source_kind == "document":
                pages = metadata.get("pages")
                status = metadata.get("status")
                content_type = metadata.get("content_type")

                if pages is not None:
                    meta_items.append(
                        f"{pages} page{'s' if pages != 1 else ''}"
                    )

                if status:
                    meta_items.append(str(status).title())

                if content_type:
                    meta_items.append(
                        str(content_type).replace(
                            "application/", ""
                        ).upper()
                    )
            else:
                if isinstance(authors, list) and authors:
                    author_count = len(authors)
                    meta_items.append(
                        f"{author_count} author{'s' if author_count != 1 else ''}"
                    )

                if isinstance(tags, list) and tags:
                    meta_items.append(f"{len(tags)} tags")

                if language:
                    meta_items.append(str(language))

            if meta_items:
                pills = "".join(
                    f'<span class="source-meta-pill">{item}</span>'
                    for item in meta_items[:3]
                )
                st.markdown(
                    f'<div class="source-card-meta">{pills}</div>',
                    unsafe_allow_html=True,
                )

            if url:
                st.markdown(
                    f'<div class="source-card-url">{str(url)}</div>',
                    unsafe_allow_html=True,
                )

            action_left, action_right = st.columns(2)

        with action_left:
            if url:
                st.link_button(
                    "Open source",
                    str(url),
                    use_container_width=True,
                    key=f"sources_open_{source.get('id', index)}",
                )
            elif source_kind == "document":
                st.button(
                    "Document",
                    disabled=True,
                    use_container_width=True,
                    key=f"sources_document_label_{source.get('id', index)}",
                )
            else:
                st.button(
                    "Open source",
                    disabled=True,
                    use_container_width=True,
                    key=f"sources_open_disabled_{source.get('id', index)}",
                )

        with action_right:
            if st.button(
                "Ask AI",
                use_container_width=True,
                key=f"sources_ask_{source.get('kind')}_{source.get('id', index)}",
            ):
                workspace_id = st.session_state.get(
                    "active_workspace_id"
                )

                if not workspace_id:
                    st.error(
                        "No active workspace is selected."
                    )
                    return

                if source_kind == "document":
                    document_id = source.get("document_id")

                    if not document_id:
                        st.error(
                            "This document does not have a valid document ID."
                        )
                        return

                    chat_title = (
                        source.get("title")
                        or "Document Chat"
                    )

                    try:
                        with st.spinner("Opening chat..."):
                            chat_result = create_persistent_chat(
                                workspace_id=str(workspace_id),
                                title=chat_title,
                                source_type="document",
                                source_id=str(document_id),
                            )

                        chat_data = chat_result.get("chat")

                        if not chat_data or not chat_data.get("id"):
                            st.error(
                                "The backend created no valid chat."
                            )
                            return

                        # Store the persistent chat ID.
                        st.session_state.active_chat_id = str(
                            chat_data["id"]
                        )

                        # Pass the selected document to Chat.
                        st.session_state.selected_workspace_document_id = (
                            str(document_id)
                        )

                        st.session_state.selected_workspace_document_name = (
                            chat_title
                        )

                        st.session_state.selected_workspace_source_id = None

                        # Open Chat.
                        st.session_state.app_mode = "Chat"

                        st.rerun()

                    except requests.RequestException as exc:
                        st.error(
                            f"Could not create chat: {exc}"
                        )

                else:
                    st.warning(
                        "Chat support for this source type will be added next."
                    )


def render_chat() -> None:
    back_col, _=st.columns([1,8])
    with back_col:
        if st.button(
            "← Workspace",
            use_container_width=True,
            key="chat_back_to_workspace",
        ):
            st.session_state.app_mode="Workspace"
            st.rerun()
    chat = active_chat()

    # ------------------------------------------------------------
    # Activate a document selected from the workspace Sources page.
    # ------------------------------------------------------------

    workspace_document_id = (
        st.session_state.selected_workspace_document_id
    )

    if workspace_document_id:

        if workspace_document_id not in chat["document_ids"]:
            chat["document_ids"].append(
                workspace_document_id
            )

        # We need a readable name for the existing
        # Chat document selector.
        workspace_document_name = (
            st.session_state.get(
                "selected_workspace_document_name"
            )
            or "Workspace document"
        )

        chat["document_names"][
            workspace_document_id
        ] = workspace_document_name

        chat["selected_document_id"] = (
            workspace_document_id
        )

        # Consume the navigation state.
        st.session_state.selected_workspace_document_id = None
        st.session_state.selected_workspace_document_name = None
    
    selected_document_ids = render_chat_sidebar()


    # ------------------------------------------------------------
    # Process documents uploaded through the sidebar dropzone.
    # ------------------------------------------------------------
    sidebar_files = get_sidebar_uploaded_files(chat)

    if sidebar_files:
        processed_keys = chat.setdefault(
            "uploaded_file_keys",
            set(),
        )

        for uploaded_file in sidebar_files:
            upload_key = (
                f"{chat['id']}:"
                f"{uploaded_file.name}:"
                f"{len(uploaded_file.getvalue())}"
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
                filename = upload_result.get(
                    "filename",
                    uploaded_file.name,
                )

                chat.setdefault("document_ids", [])

                if document_id not in chat["document_ids"]:
                    chat["document_ids"].append(document_id)

                chat.setdefault("document_names", {})
                chat["document_names"][document_id] = filename

                # New sidebar upload becomes the selected source.
                chat["selected_document_id"] = document_id

                processed_keys.add(upload_key)

                # Clear the uploader state after processing so the same
                # file isn't uploaded again on every Streamlit rerun.
                uploader_key = (
                    f"sidebar_document_uploader_{chat['id']}"
                )

                if uploader_key in st.session_state:
                    del st.session_state[uploader_key]

                st.toast(
                    f"Added {filename}",
                    icon="📎",
                )

                st.rerun()

            except Exception as upload_error:
                st.error(
                    f"Could not attach "
                    f"{uploaded_file.name}: "
                    f"{upload_error}"
                )

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
            filename = upload_result.get(
                "filename",
                uploaded_file.name,
            )

            chat.setdefault("document_ids", [])

            if document_id not in chat["document_ids"]:
                chat["document_ids"].append(document_id)

            chat.setdefault("document_names", {})
            chat["document_names"][document_id] = filename

            # A newly attached file is immediately the active document
            # for the question submitted with that attachment.
            chat["selected_document_id"] = document_id

            processed_keys.add(upload_key)
            uploaded_this_turn.append(filename)

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
    # If documents were attached with this exact question, use those
    # documents for this turn rather than an older sidebar selection.
    if uploaded_this_turn:
        selected_document_ids = [
            document_id
            for document_id in chat.get("document_ids", [])
            if chat.get("document_names", {}).get(document_id)
            in set(uploaded_this_turn)
        ]

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
    
    try:
        save_chat_message(
            chat_id=str(chat["id"]),
            role="user",
            content=visible_prompt,
        )
    except requests.RequestException as exc:
        st.error(
            f"Could not save your message: {exc}"
        )
        return

    if len(chat["messages"]) == 1:
        chat["title"] = chat_title(prompt)

    with st.chat_message("user"):
        st.markdown(visible_prompt)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        status_placeholder = st.empty()
        status_placeholder.caption("Thinking and retrieving relevant information...")
        try:
            
            data = call_backend(
                prompt,
                history,
                chat["id"],
                selected_document_ids,
                answer_placeholder=answer_placeholder,
            )

            answer = data["answer"]
            status_placeholder.empty()

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
            try:
                save_chat_message(
                    chat_id=str(chat["id"]),
                    role="assistant",
                    content=answer,
                )
            except requests.RequestException as exc:
                st.warning(
                    f"Answer was generated, but could not be saved: {exc}"
                )

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

        streamed_this_run = False

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

                answer_placeholder = st.empty()
                status_placeholder = st.empty()
                status_placeholder.caption("Retrieving evidence and generating answer...")

                data = call_backend(
                    question.strip(),
                    history,
                    research_chat_id,
                    answer_placeholder=answer_placeholder,
                    extra_payload={
                        key: value
                        for key, value in payload.items()
                        if key not in {"question", "history", "top_k", "chat_id", "document_ids"}
                    },
                )
                answer = data["answer"]
                status_placeholder.empty()
                streamed_this_run = True

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
                render_chat_diagnostics(data)

            except Exception as exc:
                show_backend_error(exc)

        previous = st.session_state.ai_answers.get(result_id)
        if not previous:
            return

        if not streamed_this_run:
            st.markdown("### 🤖 AI Answer")
            st.markdown(previous["answer"])
            render_chat_diagnostics(previous)
def display_result_card(result: Any,workspace_sources_urls:set[str]) -> None:
    source = str(
        getattr(result, "source", "") or ""
    ).lower()

    title = str(
        getattr(result, "title", "Untitled")
        or "Untitled"
    )

    url = getattr(result, "url", None)
    source_key = (
        f"{source}|"
        f"{str(url).strip() if url else ''}"
    )

    already_in_workspace = (
       source_key in workspace_sources_urls
    )

    with st.container(border=True):

        # --------------------------------------------------------
        # Title
        # --------------------------------------------------------

        st.markdown(
            f"### {title}"
        )

        st.caption(
            f"Source: {source_name(source)}"
        )

        # --------------------------------------------------------
        # Source-specific metadata
        # --------------------------------------------------------

        if source == "github":
            display_github_metadata(result)

        elif source == "huggingface":
            display_huggingface_metadata(result)

        elif source == "arxiv":
            display_arxiv_metadata(result)

        elif source == "paperswithcode":
            display_paperswithcode_metadata(result)

        # --------------------------------------------------------
        # Description
        # --------------------------------------------------------

        description = getattr(
            result,
            "description",
            None,
        )

        if description:
            st.write(description)

        # --------------------------------------------------------
        # Tags
        # --------------------------------------------------------

        tags = getattr(
            result,
            "tags",
            None,
        )

        if tags:
            st.write(
                "**🏷 Tags:** "
                + " • ".join(
                    str(x)
                    for x in tags
                )
            )

        # --------------------------------------------------------
        # Published / Updated
        # --------------------------------------------------------

        cols = st.columns(2)

        published = getattr(
            result,
            "published",
            None,
        )

        if published:
            with cols[0]:
                st.caption("📅 Published")

                try:
                    st.write(
                        published.strftime(
                            "%Y-%m-%d"
                        )
                    )
                except Exception:
                    st.write(
                        str(published)
                    )

        updated = getattr(
            result,
            "updated",
            None,
        )

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
                    st.write(
                        str(updated)
                    )

        # --------------------------------------------------------
        # Actions
        # --------------------------------------------------------

        action_cols = st.columns(2)

        # --------------------------------------------------------
        # Open source
        # --------------------------------------------------------

        with action_cols[0]:

            if url:
                st.link_button(
                    "🔗 Open source",
                    str(url),
                    use_container_width=True,
                )

        # --------------------------------------------------------
        # Add to workspace
        # --------------------------------------------------------

        with action_cols[1]:

            workspace_id = (
                st.session_state.active_workspace_id
            )

            if workspace_id:

                source_key = (
                    f"{source}_"
                    f"{title}_"
                    f"{url}"
                )

                added_key = (
                    f"workspace_source_added_"
                    f"{source_key}"
                )

                # Initialize state for this source.
                if added_key not in st.session_state:
                    st.session_state[added_key] = False

                # ------------------------------------------------
                # Already added
                # ------------------------------------------------

                if already_in_workspace or st.session_state[added_key]:

                    st.button(
                        "✓ In current workspace",
                        disabled=True,
                        use_container_width=True,
                        key=f"in_workspace_{source_key}",
                   )

                else:

                    if st.button(
                        "＋ Add to workspace",
                        use_container_width=True,
                        key=f"add_{source_key}",
             ):

                        metadata = serialize_research_result(
                            result
                   )

                        success = (
                            add_research_result_to_workspace(
                                source_type=source,
                                title=title,
                                url=(
                                    str(url)
                                    if url
                                    else None
                                ),
                                metadata=metadata,
                            )
                        )

                        if success:

                            st.session_state[added_key] = True

                            st.session_state[
                                "workspace_toast"
                            ] = (
                                 "Successfully added "
                                "to workspace."
                        )

                        st.rerun()

            else:

                st.button(
                    "＋ Add to workspace",
                    disabled=True,
                    use_container_width=True,
                    key=(
                        f"add_disabled_"
                        f"{source}_"
                        f"{title}_"
                        f"{url}"
                    ),
                )

        # --------------------------------------------------------
        # Existing AI action
        # --------------------------------------------------------

        research_ask_ai(result)
def render_research() -> None:
    render_workspace_sidebar()
    if st.button(
        "← Workspace",
        use_container_width=True,
        key="research_back_to_workspace",
    ):
        st.session_state.app_mode = "Workspace"
        st.rerun()

    st.divider()

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
    
    workspace_source_urls = (
    get_current_workspace_source_urls()
)

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
        display_result_card(result,workspace_source_urls)

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
                f"**Page {current} of {total_pages}**"
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
if st.session_state.workspace_toast:
    st.toast(
        st.session_state.workspace_toast,
        icon="✅",
    )

    st.session_state.workspace_toast = None
if st.session_state.app_mode == "Research":
    render_research()

elif st.session_state.app_mode == "Workspace":
    render_workspace_overview()
    
elif st.session_state.app_mode == "Sources":
    render_sources()

else:
    render_chat()
from datetime import datetime

from pydantic import BaseModel, HttpUrl

from src.clients.huggingface import get_huggingface
from src.models.research import ResearchItem


# =============================================================
# HUGGING FACE MODEL
# =============================================================

class HuggingFaceModel(BaseModel):
    model_id: str
    author: str | None = None

    pipeline_tag: str | None = None

    downloads: int = 0
    likes: int = 0

    library_name: str | None = None

    tags: list[str] = []

    created_at: datetime | None = None
    last_modified: datetime | None = None

    url: HttpUrl


# =============================================================
# PARSE HUGGING FACE API RESPONSE
# =============================================================

def parse_huggingface_model(
    data: dict,
) -> HuggingFaceModel:
    """
    Convert a raw Hugging Face API model dictionary into
    a HuggingFaceModel.

    Important source mappings:

        modelId      → model_id
        createdAt    → created_at
        lastModified → last_modified
        pipeline_tag → pipeline_tag
        library_name → library_name
    """

    model_id = data["modelId"]

    # ---------------------------------------------------------
    # Author
    # ---------------------------------------------------------

    author = data.get("author")

    # Some Hugging Face responses may not provide an explicit
    # author. In that case, derive it from:
    #
    # username/model-name
    #
    # Example:
    #
    # google/bert-base-uncased
    # ↓
    # google

    if author is None and "/" in model_id:
        author = model_id.split("/", 1)[0]

    # ---------------------------------------------------------
    # Parse model
    # ---------------------------------------------------------

    return HuggingFaceModel(
        model_id=model_id,

        author=author,

        pipeline_tag=data.get("pipeline_tag"),

        downloads=data.get("downloads", 0),

        likes=data.get("likes", 0),

        library_name=data.get("library_name"),

        tags=data.get("tags", []),

        created_at=data.get("createdAt"),

        last_modified=data.get("lastModified"),

        url=f"https://huggingface.co/{model_id}",
    )


# =============================================================
# SEARCH HUGGING FACE MODELS
# =============================================================

def search_huggingface_models(
    search: str | None = None,
    limit: int = 20,
    author: str | None = None,
    library: str | None = None,
    pipeline_tag: str | None = None,
) -> list[HuggingFaceModel]:
    """
    Search Hugging Face models.

    Parameters:
        search:
            Optional search text.

        limit:
            Maximum number of models to return.

        author:
            Optional Hugging Face author filter.

        library:
            Optional library filter.

        pipeline_tag:
            Optional pipeline/task filter.

    Returns:
        A list of HuggingFaceModel objects.
    """

    # ---------------------------------------------------------
    # Validate limit
    # ---------------------------------------------------------

    if limit < 1:
        raise ValueError(
            "limit must be at least 1"
        )

    # ---------------------------------------------------------
    # Build query parameters
    # ---------------------------------------------------------

    params = {
        "limit": limit,
    }

    if search:
        params["search"] = search

    if author:
        params["author"] = author

    if library:
        params["library"] = library

    if pipeline_tag:
        params["pipeline_tag"] = pipeline_tag

    # ---------------------------------------------------------
    # API request
    # ---------------------------------------------------------

    response = get_huggingface(
        "/models",
        params=params,
    )

    # ---------------------------------------------------------
    # Parse response
    # ---------------------------------------------------------

    data = response.json()

    return [
        parse_huggingface_model(item)
        for item in data
    ]


# =============================================================
# CONVERT HUGGING FACE MODEL → RESEARCH ITEM
# =============================================================

def huggingface_model_to_item(
    model: HuggingFaceModel,
) -> ResearchItem:
    """
    Convert a HuggingFaceModel into the common ResearchItem model.

    Common information is mapped to the standard ResearchItem
    fields.

    Hugging Face-specific information is mapped to the dedicated
    ResearchItem fields so that the Streamlit UI can display it.
    """

    return ResearchItem(
        # -----------------------------------------------------
        # Common fields
        # -----------------------------------------------------

        id=model.model_id,

        title=model.model_id,

        description=None,

        authors=(
            [model.author]
            if model.author
            else []
        ),

        source="huggingface",

        url=model.url,

        # Hugging Face createdAt
        # → ResearchItem.published

        published=model.created_at,

        # Hugging Face lastModified
        # → ResearchItem.updated

        updated=model.last_modified,

        tags=model.tags,

        # -----------------------------------------------------
        # Hugging Face-specific fields
        # -----------------------------------------------------

        downloads=model.downloads,

        likes=model.likes,

        library=model.library_name,

        pipeline_tag=model.pipeline_tag,
    )
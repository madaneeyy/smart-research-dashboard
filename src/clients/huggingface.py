import httpx


HUGGING_FACE_API_URL = "https://huggingface.co/api"


def get_huggingface(
    path: str,
    *,
    params: dict | None = None,
) -> httpx.Response:
    response = httpx.get(
        f"{HUGGING_FACE_API_URL}{path}",
        params=params,
        headers={
            "User-Agent": "SmartResearchDashboard/0.1 (research project)",
        },
        timeout=30.0,
    )

    response.raise_for_status()

    return response
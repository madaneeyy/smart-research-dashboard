import httpx

BASE_URL = "https://datasets-server.huggingface.co"


def get_paperswithcode_rows(
    dataset: str,
    *,
    offset: int = 0,
    length: int = 100,
) -> httpx.Response:
    if offset < 0:
        raise ValueError("offset must be 0 or greater")

    if length < 1:
        raise ValueError("length must be at least 1")

    response = httpx.get(
        f"{BASE_URL}/rows",
        params={
            "dataset": dataset,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        },
        timeout=30.0,
    )

    response.raise_for_status()

    return response


def search_paperswithcode_rows(
    dataset: str,
    query: str,
    *,
    offset: int = 0,
    length: int = 100,
) -> httpx.Response:
    if not query.strip():
        raise ValueError("query must not be empty")

    if offset < 0:
        raise ValueError("offset must be 0 or greater")

    if length < 1:
        raise ValueError("length must be at least 1")

    if length > 100:
        raise ValueError("length must be at most 100")

    response = httpx.get(
        f"{BASE_URL}/search",
        params={
            "dataset": dataset,
            "config": "default",
            "split": "train",
            "query": query,
            "offset": offset,
            "length": length,
        },
        timeout=30.0,
    )

    response.raise_for_status()

    return response
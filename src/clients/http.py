import httpx

def get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 30.0,
) -> str:
    response = httpx.get(
        url,
        params=params,
        headers=headers,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.text
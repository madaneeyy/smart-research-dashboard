import httpx

GITHUB_API_URL = "https://api.github.com"

GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2026-03-10",
}


def get_github(path: str, params: dict | None = None) -> httpx.Response:
    response = httpx.get(
        f"{GITHUB_API_URL}{path}",
        params=params,
        headers=GITHUB_HEADERS,
        timeout=30.0,
    )

    response.raise_for_status()

    return response
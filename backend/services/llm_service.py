import os
from typing import Any, Dict, List, Optional

import ollama
import requests


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1000"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"


ollama_client = ollama.Client(host=OLLAMA_HOST)


def get_provider() -> str:
    return LLM_PROVIDER


def get_model() -> str:
    if LLM_PROVIDER == "groq":
        return GROQ_MODEL

    return OLLAMA_MODEL


def _groq_headers() -> Dict[str, str]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }


def chat(
    *,
    messages: List[Dict[str, str]],
    stream: bool = False,
) -> Any:
    """
    Generate a response using the configured LLM provider.

    Supported providers:
    - ollama
    - groq
    """

    if LLM_PROVIDER == "ollama":
        return ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "num_ctx": OLLAMA_NUM_CTX,
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
            keep_alive=OLLAMA_KEEP_ALIVE,
            stream=stream,
        )

    if LLM_PROVIDER == "groq":
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": OLLAMA_TEMPERATURE,
            "max_tokens": OLLAMA_NUM_PREDICT,
            "stream": stream,
        }

        response = requests.post(
            GROQ_BASE_URL,
            headers=_groq_headers(),
            json=payload,
            timeout=(15, 120),
            stream=stream,
        )

        response.raise_for_status()

        if stream:
            # Return complete SSE lines rather than arbitrary network byte
            # chunks. This makes extract_stream_content() reliable when
            # FastAPI forwards Groq's streamed response to the frontend.
            return response.iter_lines(decode_unicode=False)

        return response.json()

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. "
        "Use 'ollama' or 'groq'."
    )


def check_connection() -> Dict[str, Any]:
    """
    Check whether the configured provider is reachable.
    """

    if LLM_PROVIDER == "ollama":
        try:
            result = ollama_client.list()

            models = []

            try:
                model_items = getattr(result, "models", []) or []

                for model in model_items:
                    model_name = getattr(model, "model", None)

                    if model_name:
                        models.append(str(model_name))

            except Exception:
                pass

            return {
                "provider": "ollama",
                "connected": True,
                "model": OLLAMA_MODEL,
                "model_available": OLLAMA_MODEL in models,
                "models": models,
                "host": OLLAMA_HOST,
            }

        except Exception as exc:
            return {
                "provider": "ollama",
                "connected": False,
                "model": OLLAMA_MODEL,
                "model_available": False,
                "host": OLLAMA_HOST,
                "error": str(exc),
            }

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            return {
                "provider": "groq",
                "connected": False,
                "model": GROQ_MODEL,
                "model_available": False,
                "error": "GROQ_API_KEY is not configured.",
            }

        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers=_groq_headers(),
                timeout=(10, 15),
            )

            response.raise_for_status()

            data = response.json()
            models = [
                str(model.get("id"))
                for model in data.get("data", [])
                if model.get("id")
            ]

            return {
                "provider": "groq",
                "connected": True,
                "model": GROQ_MODEL,
                "model_available": GROQ_MODEL in models,
                "models": models,
            }

        except Exception as exc:
            return {
                "provider": "groq",
                "connected": False,
                "model": GROQ_MODEL,
                "model_available": False,
                "error": str(exc),
            }

    return {
        "provider": LLM_PROVIDER,
        "connected": False,
        "model": get_model(),
        "model_available": False,
        "error": f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}",
    }


def extract_content(response: Any) -> str:
    """
    Extract assistant text from either Ollama or Groq responses.
    """

    if LLM_PROVIDER == "ollama":
        content = getattr(
            getattr(response, "message", None),
            "content",
            None,
        )

        return str(content or "").strip()

    if LLM_PROVIDER == "groq":
        try:
            return str(
                response["choices"][0]["message"]["content"] or ""
            ).strip()
        except (KeyError, IndexError, TypeError):
            return ""

    return ""


def extract_stream_content(chunk: Any) -> Optional[str]:
    """
    Extract a streamed text chunk from either provider.

    Ollama:
        chunk.message.content

    Groq:
        OpenAI-compatible SSE:
        data: {"choices":[{"delta":{"content":"..."}}]}
    """

    if LLM_PROVIDER == "ollama":
        content = getattr(
            getattr(chunk, "message", None),
            "content",
            None,
        )

        if content:
            return str(content)

        return None

    if LLM_PROVIDER == "groq":
        try:
            import json

            line = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            line = line.strip()

            if not line or not line.startswith("data:"):
                return None

            data = line[5:].strip()

            if data == "[DONE]":
                return None

            parsed = json.loads(data)
            choices = parsed.get("choices") or []

            if not choices:
                return None

            delta = choices[0].get("delta") or {}
            content = delta.get("content")

            if content:
                return str(content)

        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, IndexError):
            return None

        return None

    return None
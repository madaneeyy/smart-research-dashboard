"""
Document cache for Smart Research AI.

Caches extracted document content so the same uploaded file does not
need to be parsed repeatedly.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Callable, Dict, Optional


class DocumentCache:
    """Thread-safe in-memory cache for extracted documents."""

    _cache: Dict[str, Dict[str, Any]] = {}
    _lock = threading.RLock()

    _hits = 0
    _misses = 0

    @classmethod
    def make_key(
        cls,
        filename: str,
        raw_bytes: bytes,
    ) -> str:
        """Create a stable SHA-256 cache key."""

        digest = hashlib.sha256()

        digest.update(
            filename.strip().lower().encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(raw_bytes)

        return digest.hexdigest()

    @classmethod
    def get(
        cls,
        filename: str,
        raw_bytes: bytes,
    ) -> Optional[Dict[str, Any]]:
        """Return a cached document if available."""

        key = cls.make_key(
            filename,
            raw_bytes,
        )

        with cls._lock:
            cached = cls._cache.get(key)

            if cached is None:
                cls._misses += 1
                return None

            cls._hits += 1

            return dict(cached)

    @classmethod
    def set(
        cls,
        filename: str,
        raw_bytes: bytes,
        document: Dict[str, Any],
    ) -> str:
        """Store an extracted document."""

        key = cls.make_key(
            filename,
            raw_bytes,
        )

        with cls._lock:
            cls._cache[key] = dict(document)

        return key

    @classmethod
    def get_or_extract(
        cls,
        filename: str,
        raw_bytes: bytes,
        extractor: Callable[..., Dict[str, Any]],
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return the cached document or extract it and cache the result.
        """

        cached = cls.get(
            filename=filename,
            raw_bytes=raw_bytes,
        )

        if cached is not None:
            cached["cache_hit"] = True
            return cached

        document = extractor(
            filename=filename,
            raw_bytes=raw_bytes,
            content_type=content_type,
        )

        key = cls.set(
            filename=filename,
            raw_bytes=raw_bytes,
            document=document,
        )

        result = dict(document)

        result["cache_key"] = key
        result["cache_hit"] = False

        return result

    @classmethod
    def invalidate(
        cls,
        filename: str,
        raw_bytes: bytes,
    ) -> bool:
        """Remove one document from the cache."""

        key = cls.make_key(
            filename,
            raw_bytes,
        )

        with cls._lock:
            return cls._cache.pop(key, None) is not None

    @classmethod
    def clear(cls) -> None:
        """Clear the entire document cache."""

        with cls._lock:
            cls._cache.clear()

            cls._hits = 0
            cls._misses = 0

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """Return cache statistics."""

        with cls._lock:
            entries = len(cls._cache)
            hits = cls._hits
            misses = cls._misses

        requests = hits + misses

        return {
            "entries": entries,
            "hits": hits,
            "misses": misses,
            "requests": requests,
            "hit_rate": (
                round(
                    (hits / requests) * 100,
                    2,
                )
                if requests
                else 0.0
            ),
        }
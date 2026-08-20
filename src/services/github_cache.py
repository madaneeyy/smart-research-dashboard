from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class GitHubCache:
    """Small atomic JSON disk cache for GitHub API responses."""

    def __init__(self, cache_dir: str | Path = ".cache/github", ttl_seconds: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = max(0, int(ttl_seconds))

    @staticmethod
    def _digest(kind: str, *parts: str) -> str:
        value = "\0".join([kind, *[str(x or "") for x in parts]])
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _path(self, kind: str, *parts: str) -> Path:
        directory = self.cache_dir / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self._digest(kind, *parts)}.json"

    def get(self, kind: str, *parts: str) -> Any | None:
        path = self._path(kind, *parts)
        try:
            if not path.exists():
                return None
            if time.time() - path.stat().st_mtime > self.ttl_seconds:
                return None
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError, json.JSONDecodeError):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def set(self, kind: str, value: Any, *parts: str) -> None:
        path = self._path(kind, *parts)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def stats(self) -> dict[str, int]:
        kinds = ("metadata", "readme", "file", "tree", "root")
        result = {k: 0 for k in kinds}
        for kind in kinds:
            directory = self.cache_dir / kind
            if directory.exists():
                result[kind] = sum(1 for p in directory.glob("*.json") if p.is_file())
        result["total"] = sum(result.values())
        return result

    def clear(self) -> None:
        if not self.cache_dir.exists():
            return
        for path in self.cache_dir.rglob("*.json"):
            try:
                path.unlink()
            except OSError:
                pass
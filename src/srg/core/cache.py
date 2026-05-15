"""Disk cache for evaluation runs (reproducibility, fewer API calls)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(query: str, top_k: int) -> str:
    raw = f"{query.strip().lower()}|top_k={int(top_k)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def load_cached(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_cached(cache_dir: Path, key: str, data: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    slim = {k: v for k, v in data.items() if k != "payload"}
    path.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")

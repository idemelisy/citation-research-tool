"""Named project snapshots (JSON) for SRG session restore."""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import FeedbackEvent

SCHEMA_VERSION = 1
DEFAULT_MAX_BLOB_BYTES = 20 * 1024 * 1024  # 20 MB total embedded PDFs
RESTORED_PDF_SIGNATURE = "__restored_from_project__"


def sanitize_project_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("_") or "untitled"
    return cleaned[:120]


def feedback_events_to_jsonable(events: list[FeedbackEvent]) -> list[dict[str, Any]]:
    return [
        {
            "event_type": e.event_type,
            "left_id": e.left_id,
            "right_id": e.right_id,
            "accepted": e.accepted,
            "created_at": e.created_at.isoformat(),
        }
        for e in events
    ]


def feedback_events_from_jsonable(rows: list[dict[str, Any]]) -> list[FeedbackEvent]:
    out: list[FeedbackEvent] = []
    for r in rows:
        ts = r.get("created_at")
        created = datetime.fromisoformat(ts) if isinstance(ts, str) else datetime.utcnow()
        out.append(
            FeedbackEvent(
                event_type=str(r.get("event_type", "")),
                left_id=str(r.get("left_id", "")),
                right_id=str(r.get("right_id", "")),
                accepted=bool(r.get("accepted", False)),
                created_at=created,
            )
        )
    return out


def collect_pdf_blobs(
    files: list[dict[str, Any]],
    max_total_bytes: int = DEFAULT_MAX_BLOB_BYTES,
) -> tuple[dict[str, str], list[str]]:
    """
    Encode PDF contents as base64. Returns (filename -> b64, warnings).
    Stops adding when max_total_bytes (decoded) would be exceeded.
    """
    blobs: dict[str, str] = {}
    warnings: list[str] = []
    total = 0
    for f in files:
        name = str(f.get("name") or "document.pdf")
        content = f.get("content")
        if not isinstance(content, (bytes, bytearray)):
            continue
        raw = bytes(content)
        if total + len(raw) > max_total_bytes:
            warnings.append(f"Skipped embedding '{name}' (would exceed project size cap).")
            continue
        blobs[name] = base64.standard_b64encode(raw).decode("ascii")
        total += len(raw)
    return blobs, warnings


class ProjectStore:
    """Save/load versioned JSON project files under a directory."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path.cwd() / "srg_projects"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{sanitize_project_filename(name)}.json"

    def list_project_stems(self) -> list[str]:
        """Filename stems (without .json), newest first — stable keys for Load UI."""
        return sorted(
            (p.stem for p in self.root.glob("*.json")),
            key=lambda s: (self.root / f"{s}.json").stat().st_mtime,
            reverse=True,
        )

    def project_display_name(self, stem: str) -> str:
        path = self.root / f"{stem}.json"
        if not path.is_file():
            return stem
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("name") or stem)
        except (json.JSONDecodeError, OSError):
            return stem

    def save(self, name: str, snapshot: dict[str, Any]) -> Path:
        path = self._path(name)
        doc = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": datetime.utcnow().isoformat(),
            "name": name.strip() or path.stem,
            "session": snapshot,
        }
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Project not found: {path}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported project schema: {doc.get('schema_version')}")
        session = doc.get("session")
        if not isinstance(session, dict):
            raise ValueError("Invalid project file: missing session")
        return session

    def load_stem(self, stem: str) -> dict[str, Any]:
        """Load by file stem (e.g. from list_project_stems)."""
        path = self.root / f"{stem}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Project not found: {path}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported project schema: {doc.get('schema_version')}")
        session = doc.get("session")
        if not isinstance(session, dict):
            raise ValueError("Invalid project file: missing session")
        return session

    def load_document(self, name: str) -> dict[str, Any]:
        """Full file including metadata."""
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Project not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def delete_stem(self, stem: str) -> None:
        path = self.root / f"{stem}.json"
        if path.is_file():
            path.unlink()

    def delete(self, name: str) -> None:
        path = self._path(name)
        if path.is_file():
            path.unlink()

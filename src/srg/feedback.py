from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from typing import Iterable

from .schema import FeedbackEvent


class FeedbackStore:
    def __init__(self, db_path: str = "srg_feedback.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                event_type TEXT NOT NULL,
                left_id TEXT NOT NULL,
                right_id TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(self, event: FeedbackEvent) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?)",
                (event.event_type, event.left_id, event.right_id, int(event.accepted), event.created_at.isoformat()),
            )
            self.conn.commit()

    def list_all(self) -> list[FeedbackEvent]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT event_type, left_id, right_id, accepted, created_at FROM feedback ORDER BY created_at DESC"
            ).fetchall()
        out: list[FeedbackEvent] = []
        for r in rows:
            ts = r[4]
            created = datetime.fromisoformat(ts) if ts else datetime.utcnow()
            out.append(
                FeedbackEvent(
                    event_type=r[0],
                    left_id=r[1],
                    right_id=r[2],
                    accepted=bool(r[3]),
                    created_at=created,
                )
            )
        return out

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM feedback")
            self.conn.commit()

    def bulk_add(self, events: Iterable[FeedbackEvent]) -> None:
        for event in events:
            self.add(event)

    def replace_all(self, events: Iterable[FeedbackEvent]) -> None:
        """Clear the store and insert events (e.g. project load)."""
        self.clear()
        self.bulk_add(events)


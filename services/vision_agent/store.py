"""SQLite-backed event log for one site's Vision Agent.

A single writer thread (the capture/detection loop) and the async MCP tool
handlers all go through this class; a plain lock keeps it correct without
needing a separate connection per thread.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime

from services.common.schemas import BoundingBox, DetectionEvent, ViolationType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp);
"""


class EventStore:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, event: DetectionEvent) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events "
                "(id, site_id, timestamp, violation_type, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, snapshot_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.site_id,
                    event.timestamp.isoformat(),
                    event.violation_type.value,
                    event.confidence,
                    event.bbox.x1,
                    event.bbox.y1,
                    event.bbox.x2,
                    event.bbox.y2,
                    event.snapshot_path,
                ),
            )
            self._conn.commit()

    def recent(self, since: datetime | None = None, limit: int = 100) -> list[DetectionEvent]:
        with self._lock:
            if since is not None:
                cur = self._conn.execute(
                    "SELECT * FROM events WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                    (since.isoformat(), limit),
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
                )
            rows = cur.fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _row_to_event(row: tuple) -> DetectionEvent:
        (id_, site_id, ts, vtype, conf, x1, y1, x2, y2, snapshot_path) = row
        return DetectionEvent(
            id=id_,
            site_id=site_id,
            timestamp=datetime.fromisoformat(ts),
            violation_type=ViolationType(vtype),
            confidence=conf,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            snapshot_path=snapshot_path,
        )

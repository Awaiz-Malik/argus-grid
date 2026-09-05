"""SQLite-backed incident log for the Orchestrator."""

from __future__ import annotations

import json
import sqlite3
import threading

from services.common.schemas import Incident

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents (created_at);
"""


class IncidentStore:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, incident: Incident) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO incidents (id, created_at, payload_json) VALUES (?, ?, ?)",
                (incident.id, incident.created_at.isoformat(), incident.model_dump_json()),
            )
            self._conn.commit()

    def recent(self, limit: int = 50) -> list[Incident]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT payload_json FROM incidents ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = cur.fetchall()
        return [Incident.model_validate(json.loads(row[0])) for row in rows]

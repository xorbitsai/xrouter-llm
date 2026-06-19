"""SQLite-backed log of routing decisions (call history)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class CallStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        parent = Path(self.path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    config TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    task TEXT,
                    selected TEXT NOT NULL,
                    candidates TEXT NOT NULL,
                    expected_quality REAL,
                    cost REAL,
                    latency REAL
                )
                """
            )

    def record(
        self,
        *,
        ts: float,
        config: str,
        prompt: str,
        task: str | None,
        selected: list[str],
        candidates: list[dict[str, Any]],
        expected_quality: float,
        cost: float,
        latency: float,
    ) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calls
                    (ts, config, prompt, task, selected, candidates, expected_quality, cost, latency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    config,
                    prompt,
                    task,
                    json.dumps(selected),
                    json.dumps(candidates),
                    expected_quality,
                    cost,
                    latency,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM calls ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def model_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._connect() as conn:
            rows = conn.execute("SELECT selected FROM calls").fetchall()
        for row in rows:
            for model_id in json.loads(row["selected"]):
                counts[model_id] = counts.get(model_id, 0) + 1
        return counts

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "ts": row["ts"],
            "config": row["config"],
            "prompt": row["prompt"],
            "task": row["task"],
            "selected": json.loads(row["selected"]),
            "candidates": json.loads(row["candidates"]),
            "expected_quality": row["expected_quality"],
            "cost": row["cost"],
            "latency": row["latency"],
        }

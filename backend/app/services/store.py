"""SQLite-backed conversation memory and session storage."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid
from app.config import settings


class SessionStore:
    def __init__(self, path: str | None = None):
        self.path = Path(path or settings.app_database_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    context TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    def connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES (?, ?, '{}')", (sid, datetime.now(timezone.utc).isoformat()))
        return sid

    def get(self, sid: str) -> dict[str, Any] | None:
        with self.connect() as db:
            session = db.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
            if not session:
                return None
            messages = [dict(r) for r in db.execute("SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id", (sid,))]
            return {
                "id": sid,
                "created_at": session["created_at"],
                "context": json.loads(session["context"]),
                "messages": messages,
            }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT id, created_at FROM sessions ORDER BY created_at DESC LIMIT 30")]

    def save(self, sid: str, question: str, response: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        intent = response.get("intent", {})
        existing_ctx = (self.get(sid) or {}).get("context", {})
        last_filters = existing_ctx.get("last_filters", {})

        new_ctx = {
            "last_intent": intent,
            "last_metric": intent.get("metric"),
            "last_period": intent.get("period") or existing_ctx.get("last_period"),
            "last_filters": {
                "country": intent.get("country") or last_filters.get("country"),
                "product_line": intent.get("product_line") or last_filters.get("product_line"),
            },
            "previous_question": question,
        }

        with self.connect() as db:
            db.execute("UPDATE sessions SET context = ? WHERE id = ?", (json.dumps(new_ctx, default=str), sid))
            summary = response.get("report", {}).get("summary", "Analysis completed.")
            db.executemany(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [(sid, "user", question, now), (sid, "assistant", summary, now)],
            )
            db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?)",
                (response["run_id"], sid, json.dumps({**response, "question": question}, default=str), now),
            )

    def run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT response_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        return json.loads(row["response_json"]) if row else None

    def list_runs(self, sid: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT response_json FROM runs WHERE session_id = ? ORDER BY created_at, rowid", (sid,)).fetchall()
        return [json.loads(r["response_json"]) for r in rows]

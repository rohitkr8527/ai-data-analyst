"""Shared Evidence and Artifact Store for multi-agent research runs."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
import uuid
from app.config import settings


class SharedEvidenceStore:
    def __init__(self, path: str | None = None):
        self.db_path = str(path or settings.dataset_store_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._lock, self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS investigation_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    code_hash TEXT DEFAULT '',
                    dataset_ids_json TEXT DEFAULT '[]',
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_art_run ON investigation_artifacts(run_id, task_id);
                CREATE TABLE IF NOT EXISTS evidence_notes (
                    note_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    methodology TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'observed',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_note_run ON evidence_notes(run_id, task_id);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def save_artifact(
        self,
        run_id: str,
        task_id: str,
        agent_name: str,
        artifact_type: str,
        title: str,
        content: Any,
        code_hash: str = "",
        dataset_ids: list[str] | None = None,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        art_id = artifact_id or f"art_{uuid.uuid4().hex[:12]}"
        content_json = json.dumps(content, sort_keys=True, default=str)
        sha256 = hashlib.sha256(content_json.encode("utf-8")).hexdigest()[:16]
        created_at = datetime.now(timezone.utc).isoformat()
        ds_list = list(dataset_ids or [])

        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO investigation_artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (art_id, run_id, task_id, agent_name, artifact_type, title, content_json, code_hash, json.dumps(ds_list), sha256, created_at),
            )

        return {
            "artifact_id": art_id, "run_id": run_id, "task_id": task_id,
            "agent_name": agent_name, "artifact_type": artifact_type, "title": title,
            "code_hash": code_hash, "dataset_ids": ds_list, "sha256": sha256, "created_at": created_at,
        }

    def list_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM investigation_artifacts WHERE run_id = ? ORDER BY created_at ASC", (run_id,)
            ).fetchall()
        return [
            {
                "artifact_id": r["artifact_id"], "run_id": r["run_id"], "task_id": r["task_id"],
                "agent_name": r["agent_name"], "artifact_type": r["artifact_type"], "title": r["title"],
                "content": json.loads(r["content_json"] or "{}"), "code_hash": r["code_hash"],
                "dataset_ids": json.loads(r["dataset_ids_json"] or "[]"), "sha256": r["sha256"], "created_at": r["created_at"],
            }
            for r in rows
        ]

    def save_note(
        self,
        run_id: str,
        task_id: str,
        agent_name: str,
        claim: str,
        methodology: str = "",
        confidence: float = 0.0,
        status: str = "observed",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        note_id = f"note_{uuid.uuid4().hex[:10]}"
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO evidence_notes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (note_id, run_id, task_id, agent_name, claim, methodology, float(confidence), status, json.dumps(metadata or {}, default=str), created_at),
            )
        return note_id

    def list_notes(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM evidence_notes WHERE run_id = ? ORDER BY created_at ASC", (run_id,)).fetchall()
        return [
            {
                "note_id": r["note_id"], "run_id": r["run_id"], "task_id": r["task_id"],
                "agent_name": r["agent_name"], "claim": r["claim"], "methodology": r["methodology"],
                "confidence": r["confidence"], "status": r["status"],
                "metadata": json.loads(r["metadata_json"] or "{}"), "created_at": r["created_at"],
            }
            for r in rows
        ]

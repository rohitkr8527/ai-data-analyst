"""Controlled storage for validated query datasets.

Stores query rows and computed tables in SQLite + thread-safe LRU cache, passing
lightweight DatasetRef handles inside LangGraph state and checkpoints.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import uuid
from typing import Any
import pandas as pd
from app.config import settings
from app.orchestration.contracts import DatasetRef


class DatasetStore:
    """Persists validated datasets outside LangGraph checkpoints."""

    def __init__(self, path: str | None = None):
        self.path = Path(path or settings.dataset_store_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem_cache: dict[str, dict[str, Any]] = {}
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    query_id TEXT,
                    sql_hash TEXT,
                    columns_json TEXT NOT NULL,
                    rows_json TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_datasets_run ON datasets(run_id);
                """
            )

    def put(
        self,
        run_id: str,
        task_id: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        query_id: str | None = None,
        sql_hash: str | None = None,
        dataset_id: str | None = None,
    ) -> DatasetRef:
        ds_id = dataset_id or f"ds_{uuid.uuid4().hex[:12]}"
        if not sql_hash and rows is not None:
            raw_sig = json.dumps(
                {"cols": columns, "rows": rows},
                sort_keys=True,
                default=str,
            )
            sql_hash = hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16]

        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "dataset_id": ds_id,
            "run_id": run_id,
            "task_id": task_id,
            "query_id": query_id,
            "sql_hash": sql_hash,
            "columns": list(columns),
            "rows": list(rows),
            "row_count": len(rows),
            "created_at": now,
        }

        with self._lock:
            self._mem_cache[ds_id] = payload
            with self._connect() as db:
                db.execute(
                    """
                    INSERT OR REPLACE INTO datasets
                    (dataset_id, run_id, task_id, query_id, sql_hash, columns_json, rows_json, row_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ds_id,
                        run_id,
                        task_id,
                        query_id,
                        sql_hash,
                        json.dumps(columns, default=str),
                        json.dumps(rows, default=str),
                        len(rows),
                        now,
                    ),
                )

        return DatasetRef(
            dataset_id=ds_id,
            run_id=run_id,
            task_id=task_id,
            columns=list(columns),
            row_count=len(rows),
            query_id=query_id,
            sql_hash=sql_hash,
            preview=list(rows[:5]),
        )

    def get(self, dataset_id: str) -> dict[str, Any] | None:
        with self._lock:
            if dataset_id in self._mem_cache:
                return self._mem_cache[dataset_id]
            with self._connect() as db:
                row = db.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
            if not row:
                return None
            payload = {
                "dataset_id": row["dataset_id"],
                "run_id": row["run_id"],
                "task_id": row["task_id"],
                "query_id": row["query_id"],
                "sql_hash": row["sql_hash"],
                "columns": json.loads(row["columns_json"]),
                "rows": json.loads(row["rows_json"]),
                "row_count": row["row_count"],
                "created_at": row["created_at"],
            }
            self._mem_cache[dataset_id] = payload
            return payload

    def get_rows(self, dataset_id: str) -> list[dict[str, Any]]:
        item = self.get(dataset_id)
        return list(item["rows"]) if item else []

    def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        item = self.get(dataset_id)
        if not item:
            return pd.DataFrame()
        rows = item["rows"]
        cols = item["columns"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows)

    def list_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT dataset_id, run_id, task_id, query_id, sql_hash, columns_json, row_count, created_at FROM datasets WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        return [
            {
                "dataset_id": r["dataset_id"],
                "run_id": r["run_id"],
                "task_id": r["task_id"],
                "query_id": r["query_id"],
                "sql_hash": r["sql_hash"],
                "columns": json.loads(r["columns_json"]),
                "row_count": r["row_count"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

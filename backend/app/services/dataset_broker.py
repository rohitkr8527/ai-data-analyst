"""Authorized Dataset Broker mediating safe dataset acquisition for autonomous agents."""
from __future__ import annotations
from typing import Any, Callable
import pandas as pd
from app.orchestration.contracts import DatasetRef
from app.services.dataset_store import DatasetStore
from app.services.db import ReadOnlyGateway
from app.services.evidence_store import SharedEvidenceStore
from app.services.schema_catalog import SchemaCatalog


def summarize_dataframe(df: pd.DataFrame, dataset_name: str = "dataset", max_sample_rows: int = 3) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "name": dataset_name,
            "row_count": 0,
            "columns": list(df.columns) if df is not None else [],
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns} if df is not None else {},
            "null_counts": {},
            "numeric_ranges": {},
            "sample_rows": [],
        }

    numeric_ranges: dict[str, dict[str, float]] = {}
    for col in list(df.columns)[:16]:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                numeric_ranges[str(col)] = {
                    "min": round(float(s.min()), 4),
                    "mean": round(float(s.mean()), 4),
                    "median": round(float(s.median()), 4),
                    "max": round(float(s.max()), 4),
                }

    sample_rows = []
    for row in df[list(df.columns)[:12]].head(max_sample_rows).to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if pd.isna(v):
                clean[str(k)] = None
            elif hasattr(v, "isoformat"):
                clean[str(k)] = v.isoformat()
            elif isinstance(v, (int, float, bool)):
                clean[str(k)] = v
            else:
                sv = str(v)
                clean[str(k)] = f"{sv[:120]}...[truncated]" if len(sv) > 120 else sv
        sample_rows.append(clean)

    return {
        "name": dataset_name,
        "row_count": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "null_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "numeric_ranges": numeric_ranges,
        "sample_rows": sample_rows,
    }


class AuthorizedDatasetBroker:
    def __init__(
        self,
        db: ReadOnlyGateway,
        dataset_store: DatasetStore,
        catalog: SchemaCatalog | None = None,
        evidence_store: SharedEvidenceStore | None = None,
    ):
        self.db = db
        self.dataset_store = dataset_store
        self.catalog = catalog or getattr(db, "catalog", None) or SchemaCatalog(
            engine=getattr(db, "engine", None),
            database_url=getattr(db, "url", None),
            approved_tables=getattr(db, "approved_tables", None),
        )
        self.evidence_store = evidence_store

    def inspect_schema(self, table_name: str | None = None) -> str:
        return self.catalog.inspect_table_details([table_name]) if table_name else self.catalog.format_llm_schema_context()

    def fetch_via_sql(
        self,
        run_id: str,
        task_id: str,
        sql: str,
        params: dict[str, Any] | None = None,
        reason: str = "",
        allowed_tables: set[str] | None = None,
    ) -> tuple[DatasetRef, pd.DataFrame, str, str]:
        clean_sql = sql.strip().rstrip(";")
        self.db.validate_sql(clean_sql, allowed_tables=allowed_tables)
        qres = self.db.execute(clean_sql, params or {}, trusted_template=False, allowed_tables=allowed_tables, run_id=run_id, task_id=task_id)
        cols, rows, query_id = list(qres["columns"]), list(qres["rows"]), str(qres["id"])
        ds_ref = self.dataset_store.put(
            run_id=run_id, task_id=task_id, columns=cols, rows=rows,
            query_id=query_id, sql_hash=str(qres.get("query_hash", "")),
        )
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
        if self.evidence_store is not None:
            self.evidence_store.save_note(
                run_id=run_id, task_id=task_id, agent_name="dataset_broker",
                claim=reason or f"Authorized dataset acquisition ({ds_ref.row_count} rows)",
                methodology="ReadOnlyGateway validated SELECT snapshot", confidence=1.0, status="acquired",
                metadata={"dataset_id": ds_ref.dataset_id, "query_id": query_id, "sql": clean_sql},
            )
        return ds_ref, df, query_id, clean_sql

    def request_dataset(
        self,
        run_id: str,
        task_id: str,
        request_text_or_sql: str,
        sql_provider: Callable[[str], tuple[str, dict[str, Any]]] | None = None,
        reason: str = "",
    ) -> tuple[DatasetRef, pd.DataFrame, dict[str, Any]]:
        stripped = (request_text_or_sql or "").strip()
        if stripped.upper().startswith(("SELECT ", "WITH ")):
            ds_ref, df, qid, safe_sql = self.fetch_via_sql(run_id, task_id, stripped, {}, reason or "Direct SQL data request")
            return ds_ref, df, {"query_id": qid, "sql": safe_sql, "source": "direct_sql"}
        if sql_provider is None:
            raise ValueError("Dataset request requires either a validated SELECT SQL query or an authorized sql_provider.")
        sql, params = sql_provider(stripped)
        ds_ref, df, qid, safe_sql = self.fetch_via_sql(run_id, task_id, sql, params, reason or stripped)
        return ds_ref, df, {"query_id": qid, "sql": safe_sql, "source": "sql_provider"}

"""Secure Read-Only SQL execution gateway with MySQL dialect enforcement.

Uses SQLGlot AST inspection to block mutations, injection vectors, and unauthorized tables.
Executes queries against local MySQL with connection pooling, bounded concurrency, timeouts, and row caps.
"""
from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
import hashlib
import logging
import threading
import time
import uuid
from typing import Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError, OperationalError
import sqlglot
from sqlglot import exp
from app.config import settings
from app.exceptions import DatabaseConnectionError, QuerySecurityError
from app.services.schema_catalog import APPROVED_TABLES, SchemaCatalog
from app.services.telemetry import trace_span

logger = logging.getLogger(__name__)

ALLOWED_FUNCTIONS = {
    "sum", "avg", "count", "min", "max", "round", "coalesce", "nullif", "abs",
    "date_format", "year", "month", "day", "quarter", "concat", "concat_ws",
    "upper", "lower", "trim", "substring", "ifnull", "case", "datediff", "cast",
    "row_number", "rank", "dense_rank", "ntile", "lag", "lead", "first_value", "last_value",
    "stddev", "stddev_pop", "stddev_samp", "variance", "var_pop", "var_samp",
    "floor", "ceil", "ceiling", "power", "pow", "sqrt", "log", "log10", "ln",
    "exp", "mod", "sign", "greatest", "least", "timestampdiff", "str_to_date",
    "date_sub", "date_add", "extract", "length", "char_length", "replace",
    "left", "right", "convert", "if",
}

DISALLOWED_FUNCTIONS = {
    "sleep", "benchmark", "load_file", "into_outfile", "into_dumpfile",
    "user", "system_user", "session_user", "current_user", "database", "schema",
    "version", "connection_id", "get_lock", "release_lock",
    "json_arrayagg", "json_objectagg", "group_concat",
}


class ReadOnlyGateway:
    """Enforces strict defense-in-depth read-only SQL execution on MySQL."""

    def __init__(
        self,
        database_url: str | None = None,
        engine: Engine | None = None,
        approved_tables: set[str] | None = None,
    ):
        self.url = database_url or settings.database_url
        self.engine = engine
        self.approved_tables = (
            {t.lower() for t in approved_tables} if approved_tables is not None else set(APPROVED_TABLES)
        )
        self._semaphore = threading.BoundedSemaphore(max(1, settings.max_parallel_db_queries))
        if not self.engine:
            if not self.url:
                raise DatabaseConnectionError("DATABASE_CONNECTION_FAILED: database_url is empty.")
            try:
                self.engine = create_engine(
                    self.url,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    pool_size=max(3, settings.max_parallel_db_queries),
                    max_overflow=max(2, settings.global_max_db_connections),
                    connect_args={
                        "connect_timeout": 5,
                        "read_timeout": settings.max_query_seconds + 2,
                        "write_timeout": 5,
                    }
                )
            except Exception as exc:
                raise DatabaseConnectionError(
                    f"DATABASE_CONNECTION_FAILED: Could not initialize MySQL engine: {exc}"
                ) from exc
        self.catalog = SchemaCatalog(
            engine=self.engine,
            database_url=self.url,
            approved_tables=self.approved_tables,
        )

    def schema(self) -> dict[str, list[dict[str, str]]]:
        """Return table column dictionary."""
        snapshot = self.catalog.get_snapshot()
        return {
            tname: [{"name": c.name, "type": c.data_type} for c in meta.columns]
            for tname, meta in snapshot.tables.items()
        }

    def validate_sql(self, sql: str, allowed_tables: set[str] | None = None) -> None:
        """Parse and inspect the SQL AST using SQLGlot with MySQL dialect."""
        cleaned = sql.strip().rstrip(";")

        # Prohibit multi-statement separators and inline comments
        if ";" in cleaned:
            raise QuerySecurityError("Statement separators (semicolons) are prohibited.")
        if "--" in cleaned or "/*" in cleaned or "#" in cleaned:
            raise QuerySecurityError("SQL comments are prohibited.")

        # Parse AST
        try:
            expressions = sqlglot.parse(cleaned, read="mysql")
        except Exception as exc:
            raise QuerySecurityError(f"SQL parsing failed: {exc}") from exc

        if not expressions or len(expressions) != 1:
            raise QuerySecurityError("Exactly one SELECT statement is required.")

        tree = expressions[0]
        if not isinstance(tree, (exp.Select, exp.Union)):
            raise QuerySecurityError("Only SELECT statements are permitted.")

        # Prohibit mutations anywhere in AST (CTEs, subqueries)
        forbidden_nodes = (
            exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop,
            exp.Alter, exp.Command, exp.Transaction,
            exp.Merge, exp.Into
        )
        for node in tree.walk():
            if isinstance(node, forbidden_nodes):
                raise QuerySecurityError(f"Unsupported or dangerous SQL expression: {type(node).__name__}")

        # Prohibit SELECT * (while allowing COUNT(*))
        if any(isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count) for node in tree.walk()):
            raise QuerySecurityError("SELECT * is not permitted. Explicit columns or aggregations are required.")

        # Table whitelist verification
        permitted_tables = (allowed_tables or self.approved_tables) & self.approved_tables
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name}
        tables_referenced = {
            t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in cte_names
        }
        if not tables_referenced:
            raise QuerySecurityError("Query must select from at least one authorized table.")

        unapproved = tables_referenced - permitted_tables
        if unapproved:
            raise QuerySecurityError(f"Query references unapproved tables: {', '.join(sorted(unapproved))}")

        # Function check
        for f in tree.find_all(exp.Func):
            fname = (f.name if isinstance(f, exp.Anonymous) else f.sql_name() or "").lower()
            if fname in DISALLOWED_FUNCTIONS:
                raise QuerySecurityError(f"Prohibited function in query: {fname}")
            if fname and fname not in ALLOWED_FUNCTIONS:
                logger.debug("Inspecting function: %s", fname)

    def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        trusted_template: bool = False,
        allowed_tables: set[str] | None = None,
        run_id: str = "",
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute validated query against MySQL and return structured evidence."""
        if not trusted_template:
            self.validate_sql(sql, allowed_tables=allowed_tables)

        params = params or {}
        query_id = str(uuid.uuid4())
        hash_input = f"{sql.strip()}|{sorted(params.items())}"
        query_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

        with self._semaphore:
            start_time = time.monotonic()
            try:
                with trace_span(
                    "mysql.execute",
                    run_id=run_id,
                    task_id=task_id,
                    attributes={"query_id": query_id, "query_hash": query_hash},
                ):
                    with self.engine.connect() as conn:
                        stmt = text(sql)
                        result = conn.execute(stmt, params)
                        cols = list(result.keys())
                        raw_rows = result.fetchmany(settings.max_result_rows + 1)

                        if len(raw_rows) > settings.max_result_rows:
                            raise QuerySecurityError(
                                f"Query result exceeded maximum row limit ({settings.max_result_rows}). "
                                "Add a LIMIT clause or grouping aggregation."
                            )

                        duration_ms = round((time.monotonic() - start_time) * 1000)

                        def _serialize_val(val: Any) -> Any:
                            if isinstance(val, Decimal):
                                return float(val)
                            if isinstance(val, (date, datetime)):
                                return val.isoformat()
                            return val

                        dict_rows = [
                            {k: _serialize_val(v) for k, v in dict(r._mapping).items()}
                            for r in raw_rows
                        ]

                        logger.info(
                            "SQL query executed (id: %s, hash: %s, rows: %d, duration: %d ms)",
                            query_id, query_hash, len(dict_rows), duration_ms
                        )

                        return {
                            "id": query_id,
                            "query_hash": query_hash,
                            "sql": sql,
                            "columns": cols,
                            "rows": dict_rows,
                            "duration_ms": duration_ms
                        }

            except OperationalError as exc:
                msg = str(exc).lower()
                if "access denied" in msg or "can't connect" in msg or "connection refused" in msg:
                    raise DatabaseConnectionError(
                        "DATABASE_CONNECTION_FAILED: Could not connect to the configured MySQL database. Check host, port, credentials, and server status."
                    ) from exc
                raise QuerySecurityError(f"Database rejected query execution: {exc}") from exc
            except DatabaseError as exc:
                raise QuerySecurityError(f"Database query error: {exc}") from exc

"""Dynamic schema discovery, relationship mapping, and metric metadata catalog.

Inspects live MySQL schema with caching and invalidation support.
Provides compact, relevant schema representations for LLM prompts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from app.config import settings
from app.exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)

APPROVED_TABLES = {
    "customers",
    "orders",
    "orderdetails",
    "products",
    "productlines",
    "payments",
    "employees",
    "offices"
}

TABLE_DESCRIPTIONS = {
    "customers": "Customer profiles, credit limits, contact info, and geographical location (city, state, country).",
    "orders": "Order records with orderDate, requiredDate, shippedDate, customerNumber, and status (Shipped, Resolved, Cancelled, In Process, On Hold, Disputed).",
    "orderdetails": "Line items per order: orderNumber, productCode, quantityOrdered, priceEach, orderLineNumber. Key source for gross merchandise revenue.",
    "products": "Product catalog: productCode, productName, productLine, productScale, productVendor, productDescription, quantityInStock, buyPrice, MSRP.",
    "productlines": "Categories of scale models: productLine (e.g. Classic Cars, Motorcycles, Planes, Ships, Trains, Trucks and Buses, Vintage Cars) and textDescription.",
    "payments": "Cash payment transactions received: customerNumber, checkNumber, paymentDate, amount. Note: payments received != order-line revenue.",
    "employees": "Sales representatives and internal staff: employeeNumber, lastName, firstName, email, officeCode, reportsTo, jobTitle.",
    "offices": "Sales offices worldwide: officeCode, city, phone, state, country, territory."
}


@dataclass
class ColumnMeta:
    name: str
    data_type: str
    primary_key: bool = False
    nullable: bool = True
    description: str = ""


@dataclass
class TableMeta:
    name: str
    description: str
    columns: list[ColumnMeta]
    primary_keys: list[str]
    foreign_keys: list[dict[str, Any]]
    row_count: int = 0


@dataclass
class SchemaSnapshot:
    tables: dict[str, TableMeta]
    relationships: list[dict[str, str]]
    date_range: dict[str, str]
    created_at: float = field(default_factory=time.time)


class SchemaCatalog:
    """Discovers and caches schema metadata from SQL databases with thread-safe invalidation."""

    def __init__(
        self,
        engine: Engine | None = None,
        database_url: str | None = None,
        approved_tables: set[str] | None = None,
    ):
        self.url = database_url or settings.database_url
        self._engine = engine
        self.approved_tables = (
            {t.lower() for t in approved_tables} if approved_tables is not None else set(APPROVED_TABLES)
        )
        self._cache: SchemaSnapshot | None = None
        self._lock = threading.Lock()

    def get_engine(self) -> Engine:
        if self._engine is None:
            if not self.url:
                raise DatabaseConnectionError("DATABASE_CONNECTION_FAILED: database_url is empty.")
            try:
                self._engine = create_engine(self.url, pool_pre_ping=True, pool_recycle=300)
            except Exception as exc:
                raise DatabaseConnectionError(
                    f"DATABASE_CONNECTION_FAILED: Could not create engine for MySQL ({type(exc).__name__})."
                ) from exc
        return self._engine

    def invalidate(self) -> None:
        """Force refresh on subsequent access."""
        with self._lock:
            self._cache = None

    def get_snapshot(self, ttl_seconds: int = 3600) -> SchemaSnapshot:
        """Retrieve cached schema or discover dynamically."""
        with self._lock:
            now = time.time()
            if self._cache and (now - self._cache.created_at) < ttl_seconds:
                return self._cache

            engine = self.get_engine()
            try:
                inspector = inspect(engine)
                table_names = [t for t in inspector.get_table_names() if t.lower() in self.approved_tables]
            except Exception as exc:
                raise DatabaseConnectionError(
                    f"DATABASE_CONNECTION_FAILED: Could not inspect MySQL database ({type(exc).__name__}): {exc}"
                ) from exc

            tables_meta: dict[str, TableMeta] = {}
            relationships: list[dict[str, str]] = []

            with engine.connect() as conn:
                for tname in table_names:
                    cols_info = inspector.get_columns(tname)
                    pk_info = inspector.get_pk_constraint(tname).get("constrained_columns", [])
                    fk_info = inspector.get_foreign_keys(tname)

                    try:
                        cnt = conn.execute(text(f"SELECT COUNT(*) FROM `{tname}`")).scalar() or 0
                    except Exception:
                        cnt = 0

                    col_metas = [
                        ColumnMeta(
                            name=c["name"],
                            data_type=str(c["type"]),
                            primary_key=c["name"] in pk_info,
                            nullable=c.get("nullable", True)
                        )
                        for c in cols_info
                    ]

                    for fk in fk_info:
                        referred_table = fk.get("referred_table")
                        if referred_table and referred_table in APPROVED_TABLES:
                            relationships.append({
                                "from_table": tname,
                                "from_cols": fk.get("constrained_columns", []),
                                "to_table": referred_table,
                                "to_cols": fk.get("referred_columns", [])
                            })

                    tables_meta[tname] = TableMeta(
                        name=tname,
                        description=TABLE_DESCRIPTIONS.get(tname, ""),
                        columns=col_metas,
                        primary_keys=pk_info,
                        foreign_keys=fk_info,
                        row_count=cnt
                    )

                # Date range in database
                date_range = {"min_order_date": "2003-01-06", "max_order_date": "2005-05-31"}
                try:
                    res = conn.execute(text("SELECT MIN(orderDate), MAX(orderDate) FROM orders")).fetchone()
                    if res and res[0] and res[1]:
                        date_range["min_order_date"] = str(res[0])
                        date_range["max_order_date"] = str(res[1])
                except Exception as exc:
                    logger.warning("Could not query orders date range: %s", exc)

            snapshot = SchemaSnapshot(
                tables=tables_meta,
                relationships=relationships,
                date_range=date_range
            )
            self._cache = snapshot
            return snapshot

    def format_llm_schema_context(self, relevant_tables: list[str] | None = None) -> str:
        """Format a clear, token-efficient schema context for SQL Analyst and Planner prompts."""
        snapshot = self.get_snapshot()
        target_tables = relevant_tables or list(snapshot.tables.keys())

        lines = [
            "### ClassicModels MySQL Database Schema",
            f"Historical Date Range: {snapshot.date_range['min_order_date']} to {snapshot.date_range['max_order_date']}",
            "Dialect: MySQL 8.0 (Use backticks or standard identifiers, DATE_FORMAT for months, proper JOINs).",
            ""
        ]

        for tname in target_tables:
            meta = snapshot.tables.get(tname)
            if not meta:
                continue
            cols_str = ", ".join([f"`{c.name}` ({c.data_type}{' PK' if c.primary_key else ''})" for c in meta.columns])
            lines.append(f"- Table `{tname}` ({meta.row_count} rows): {meta.description}")
            lines.append(f"  Columns: {cols_str}")

        if "orders" in snapshot.tables and "orderdetails" in snapshot.tables:
            lines.append("\n### Standard Join Paths:")
            lines.append("- orders -> orderdetails: `orders.orderNumber = orderdetails.orderNumber`")
            lines.append("- orderdetails -> products: `orderdetails.productCode = products.productCode`")
            lines.append("- products -> productlines: `products.productLine = productlines.productLine`")
            lines.append("- orders -> customers: `orders.customerNumber = customers.customerNumber`")
            lines.append("- customers -> employees (sales rep): `customers.salesRepEmployeeNumber = employees.employeeNumber`")
            lines.append("- employees -> offices: `employees.officeCode = offices.officeCode`")
            lines.append("- customers -> payments: `customers.customerNumber = payments.customerNumber`")
        elif snapshot.relationships:
            lines.append("\n### Discovered Foreign Key Relationships:")
            for rel in snapshot.relationships:
                lines.append(f"- {rel['from_table']}.{rel['from_col']} -> {rel['to_table']}.{rel['to_col']}")

        return "\n".join(lines)

    def inspect_table_details(self, table_names: list[str] | None = None, sample_limit: int = 3) -> str:
        """Inspect detailed columns, types, FKs, row counts, and sample rows for authorized tables."""
        snapshot = self.get_snapshot()
        requested = (
            [t for t in snapshot.tables.keys() if t.lower() in {x.lower() for x in table_names}]
            if table_names
            else list(snapshot.tables.keys())
        )
        if not requested:
            return f"No matching authorized tables found. Authorized tables: {sorted(snapshot.tables.keys())}"

        engine = self.get_engine()
        lines: list[str] = ["### Detailed Table Inspection"]
        safe_limit = max(1, min(int(sample_limit), 5))

        with engine.connect() as conn:
            for tname in requested:
                meta = snapshot.tables.get(tname)
                if not meta:
                    continue
                lines.append(f"\nTable `{tname}` ({meta.row_count} rows): {meta.description}")
                for col in meta.columns:
                    pk_tag = " PK" if col.primary_key else ""
                    null_tag = "" if col.nullable else " NOT NULL"
                    lines.append(f"  - `{col.name}`: {col.data_type}{pk_tag}{null_tag}")
                if meta.foreign_keys:
                    for fk in meta.foreign_keys:
                        lines.append(
                            f"  - FK {fk.get('constrained_columns')} -> {fk.get('referred_table')}.{fk.get('referred_columns')}"
                        )
                try:
                    col_list = ", ".join(f"`{c.name}`" for c in meta.columns[:12])
                    if col_list:
                        rows = conn.execute(text(f"SELECT {col_list} FROM `{tname}` LIMIT {safe_limit}")).mappings().all()
                        sample_dicts = [{str(k): str(v) if v is not None else None for k, v in dict(r).items()} for r in rows]
                        lines.append(f"  - Sample rows: {sample_dicts}")
                except Exception as exc:
                    lines.append(f"  - Sample rows unavailable: {exc}")

        return "\n".join(lines)

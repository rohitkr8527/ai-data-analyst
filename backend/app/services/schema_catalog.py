"""Dynamic schema discovery and metadata catalog."""
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
    "customers", "orders", "orderdetails", "products",
    "productlines", "payments", "employees", "offices",
}

TABLE_DESCRIPTIONS = {
    "customers": "Customer profiles, credit limits, contact info, and geographical location (city, state, country).",
    "orders": "Order records with orderDate, requiredDate, shippedDate, customerNumber, and status.",
    "orderdetails": "Line items per order: orderNumber, productCode, quantityOrdered, priceEach, orderLineNumber.",
    "products": "Product catalog: productCode, productName, productLine, productScale, productVendor, quantityInStock, buyPrice, MSRP.",
    "productlines": "Categories of scale models: productLine and textDescription.",
    "payments": "Cash payment transactions received: customerNumber, checkNumber, paymentDate, amount.",
    "employees": "Sales representatives and internal staff: employeeNumber, lastName, firstName, officeCode, reportsTo, jobTitle.",
    "offices": "Sales offices worldwide: officeCode, city, phone, state, country, territory.",
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
    def __init__(
        self,
        engine: Engine | None = None,
        database_url: str | None = None,
        approved_tables: set[str] | None = None,
    ):
        self.url = database_url or settings.database_url
        self._engine = engine
        self.approved_tables = {t.lower() for t in approved_tables} if approved_tables is not None else set(APPROVED_TABLES)
        self._cache: SchemaSnapshot | None = None
        self._lock = threading.Lock()

    def get_engine(self) -> Engine:
        if self._engine is None:
            if not self.url:
                raise DatabaseConnectionError("DATABASE_CONNECTION_FAILED: database_url is empty.")
            try:
                self._engine = create_engine(self.url, pool_pre_ping=True, pool_recycle=300)
            except Exception as exc:
                raise DatabaseConnectionError(f"DATABASE_CONNECTION_FAILED: {exc}") from exc
        return self._engine

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def get_snapshot(self, ttl_seconds: int = 3600) -> SchemaSnapshot:
        with self._lock:
            if self._cache and (time.time() - self._cache.created_at) < ttl_seconds:
                return self._cache

            engine = self.get_engine()
            try:
                inspector = inspect(engine)
                table_names = [t for t in inspector.get_table_names() if t.lower() in self.approved_tables]
            except Exception as exc:
                raise DatabaseConnectionError(f"DATABASE_CONNECTION_FAILED: {exc}") from exc

            tables_meta: dict[str, TableMeta] = {}
            relationships: list[dict[str, str]] = []
            date_range = {"min_order_date": "2003-01-06", "max_order_date": "2005-05-31"}

            with engine.connect() as conn:
                for tname in table_names:
                    pk_info = inspector.get_pk_constraint(tname).get("constrained_columns", [])
                    fk_info = inspector.get_foreign_keys(tname)
                    try:
                        cnt = conn.execute(text(f"SELECT COUNT(*) FROM `{tname}`")).scalar() or 0
                    except Exception:
                        cnt = 0

                    col_metas = [
                        ColumnMeta(c["name"], str(c["type"]), c["name"] in pk_info, c.get("nullable", True))
                        for c in inspector.get_columns(tname)
                    ]
                    for fk in fk_info:
                        ref_tbl = fk.get("referred_table")
                        if ref_tbl and ref_tbl.lower() in self.approved_tables:
                            fc, tc = fk.get("constrained_columns", []), fk.get("referred_columns", [])
                            relationships.append({
                                "from_table": tname, "from_cols": fc, "from_col": ", ".join(fc),
                                "to_table": ref_tbl, "to_cols": tc, "to_col": ", ".join(tc),
                            })
                    tables_meta[tname] = TableMeta(tname, TABLE_DESCRIPTIONS.get(tname, ""), col_metas, pk_info, fk_info, cnt)

                try:
                    res = conn.execute(text("SELECT MIN(orderDate), MAX(orderDate) FROM orders")).fetchone()
                    if res and res[0] and res[1]:
                        date_range = {"min_order_date": str(res[0]), "max_order_date": str(res[1])}
                except Exception:
                    pass

            self._cache = SchemaSnapshot(tables=tables_meta, relationships=relationships, date_range=date_range)
            return self._cache

    def format_llm_schema_context(self, relevant_tables: list[str] | None = None) -> str:
        snapshot = self.get_snapshot()
        lines = [
            "### ClassicModels MySQL Database Schema",
            f"Historical Date Range: {snapshot.date_range['min_order_date']} to {snapshot.date_range['max_order_date']}",
            "Dialect: MySQL 8.0 (Use backticks or standard identifiers, DATE_FORMAT for months, proper JOINs).\n",
        ]
        for tname in (relevant_tables or list(snapshot.tables.keys())):
            meta = snapshot.tables.get(tname)
            if meta:
                cols = ", ".join(f"`{c.name}` ({c.data_type}{' PK' if c.primary_key else ''})" for c in meta.columns)
                lines.append(f"- Table `{tname}` ({meta.row_count} rows): {meta.description}\n  Columns: {cols}")

        if "orders" in snapshot.tables and "orderdetails" in snapshot.tables:
            lines.extend([
                "\n### Standard Join Paths:",
                "- orders -> orderdetails: `orders.orderNumber = orderdetails.orderNumber`",
                "- orderdetails -> products: `orderdetails.productCode = products.productCode`",
                "- products -> productlines: `products.productLine = productlines.productLine`",
                "- orders -> customers: `orders.customerNumber = customers.customerNumber`",
                "- customers -> employees: `customers.salesRepEmployeeNumber = employees.employeeNumber`",
                "- employees -> offices: `employees.officeCode = offices.officeCode`",
                "- customers -> payments: `customers.customerNumber = payments.customerNumber`",
            ])
        return "\n".join(lines)

    def inspect_table_details(self, table_names: list[str] | None = None, sample_limit: int = 3) -> str:
        snapshot = self.get_snapshot()
        wanted = {x.lower() for x in table_names} if table_names else None
        requested = [t for t in snapshot.tables if not wanted or t.lower() in wanted]
        if not requested:
            return f"No matching authorized tables found. Authorized: {sorted(snapshot.tables)}"

        lines = ["### Detailed Table Inspection"]
        safe_limit = max(1, min(int(sample_limit), 5))
        with self.get_engine().connect() as conn:
            for tname in requested:
                meta = snapshot.tables[tname]
                lines.append(f"\nTable `{tname}` ({meta.row_count} rows): {meta.description}")
                for col in meta.columns:
                    lines.append(f"  - `{col.name}`: {col.data_type}{' PK' if col.primary_key else ''}{'' if col.nullable else ' NOT NULL'}")
                for fk in meta.foreign_keys:
                    lines.append(f"  - FK {fk.get('constrained_columns')} -> {fk.get('referred_table')}.{fk.get('referred_columns')}")
                try:
                    col_list = ", ".join(f"`{c.name}`" for c in meta.columns[:12])
                    if col_list:
                        rows = conn.execute(text(f"SELECT {col_list} FROM `{tname}` LIMIT {safe_limit}")).mappings().all()
                        lines.append(f"  - Sample rows: {[{str(k): str(v) if v is not None else None for k, v in dict(r).items()} for r in rows]}")
                except Exception as exc:
                    lines.append(f"  - Sample rows unavailable: {exc}")
        return "\n".join(lines)

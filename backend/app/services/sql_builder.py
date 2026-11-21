"""Canonical SQL template builder and QuerySpec compiler for ClassicModels MySQL database.

Produces authorized, parameterized MySQL queries for both legacy `PlanTask` items
and dynamic `QuerySpec` specifications created by the Planning Agent, Hypothesis Agent,
or Refinement Planner.
"""
from __future__ import annotations
from typing import Any
from app.orchestration.contracts import FilterSpec, QuerySpec
from app.schemas import PlanTask, QuestionIntent
from app.services.semantics import (
    APPROVED_METRICS,
    BASE_ORDER_FROM,
    resolve_period,
)

EXTENDED_ORDER_FROM = """FROM orders o
JOIN orderdetails od ON o.orderNumber = od.orderNumber
JOIN products p ON od.productCode = p.productCode
JOIN customers c ON o.customerNumber = c.customerNumber
LEFT JOIN employees e ON c.salesRepEmployeeNumber = e.employeeNumber
LEFT JOIN offices ofc ON e.officeCode = ofc.officeCode"""

DIMENSION_SQL: dict[str, dict[str, str]] = {
    "product_line": {"select": "p.productLine AS label", "group": "p.productLine"},
    "product": {"select": "p.productName AS label, p.productLine AS category", "group": "p.productCode, p.productName, p.productLine"},
    "country": {"select": "c.country AS label", "group": "c.country"},
    "customer": {"select": "c.customerName AS label, c.country AS country", "group": "c.customerNumber, c.customerName, c.country"},
    "order_status": {"select": "o.status AS label", "group": "o.status"},
    "month": {"select": "DATE_FORMAT(o.orderDate, '%Y-%m') AS month", "group": "DATE_FORMAT(o.orderDate, '%Y-%m')"},
    "quarter": {"select": "CONCAT(YEAR(o.orderDate), '-Q', QUARTER(o.orderDate)) AS label", "group": "CONCAT(YEAR(o.orderDate), '-Q', QUARTER(o.orderDate))"},
    "year": {"select": "CAST(YEAR(o.orderDate) AS CHAR) AS label", "group": "CAST(YEAR(o.orderDate) AS CHAR)"},
    "sales_rep": {
        "select": "COALESCE(CONCAT(e.firstName, ' ', e.lastName), 'Unassigned') AS label",
        "group": "e.employeeNumber, e.firstName, e.lastName",
        "needs_extended": "true",
    },
    "office": {
        "select": "COALESCE(CONCAT(ofc.city, ' (', ofc.country, ')'), 'Unassigned') AS label",
        "group": "ofc.officeCode, ofc.city, ofc.country",
        "needs_extended": "true",
    },
}

FILTER_COLUMN_MAP: dict[str, str] = {
    "country": "c.country",
    "product_line": "p.productLine",
    "customer": "c.customerName",
    "product": "p.productName",
    "order_status": "o.status",
}


class SQLBuilder:
    """Builds parameterized MySQL statements for ClassicModels tasks and QuerySpecs."""

    def compile_query_spec(
        self,
        spec: QuerySpec,
        period_key: str | None = None,
        plan_filters: list[FilterSpec] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Deterministically compile a structured QuerySpec into parameterized MySQL."""
        period_info = resolve_period(period_key)
        metric_key = spec.metric if spec.metric in APPROVED_METRICS else "gross_revenue"
        metric_expr = APPROVED_METRICS[metric_key]["sql_expression"]

        params: dict[str, Any] = {}
        effective_filters: list[FilterSpec] = []
        if spec.inherit_plan_filters and plan_filters:
            effective_filters.extend(plan_filters)
        for f in spec.filters:
            if not any(ef.dimension == f.dimension and ef.value == f.value for ef in effective_filters):
                effective_filters.append(f)

        dims = list(spec.dimensions)
        has_order_status_dim = "order_status" in dims
        has_order_status_filter = any(f.dimension == "order_status" for f in effective_filters)

        # Payments table query path when metric is payments_received
        if metric_key == "payments_received":
            return self._compile_payments_query(spec, period_info, effective_filters)

        where_clauses: list[str] = []
        if not has_order_status_dim and not has_order_status_filter:
            where_clauses.append("o.status != 'Cancelled'")

        for idx, flt in enumerate(effective_filters):
            col = FILTER_COLUMN_MAP.get(flt.dimension)
            if not col:
                continue
            pname = f"flt_{flt.dimension}_{idx}"
            where_clauses.append(f"{col} = :{pname}")
            params[pname] = flt.value

        t_curr = "o.orderDate >= :target_start AND o.orderDate < :target_end"
        t_prev = "o.orderDate >= :previous_start AND o.orderDate < :previous_end"

        needs_extended = any(DIMENSION_SQL.get(d, {}).get("needs_extended") == "true" for d in dims)
        from_clause = EXTENDED_ORDER_FROM if needs_extended else BASE_ORDER_FROM

        # Case A: Period-over-period comparison (baseline or bridge breakdown)
        if spec.comparison == "previous_period":
            params["target_start"] = spec.custom_start or period_info["target_start"]
            params["target_end"] = spec.custom_end or period_info["target_end"]
            params["previous_start"] = period_info["previous_start"]
            params["previous_end"] = period_info["previous_end"]

            where_all = " AND ".join(where_clauses + [f"({t_prev} OR {t_curr})"])

            if not dims:
                sql = f"""SELECT
    ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous_revenue,
    ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current_revenue,
    COUNT(DISTINCT CASE WHEN {t_prev} THEN o.orderNumber END) AS previous_orders,
    COUNT(DISTINCT CASE WHEN {t_curr} THEN o.orderNumber END) AS current_orders,
    SUM(CASE WHEN {t_prev} THEN od.quantityOrdered ELSE 0 END) AS previous_units,
    SUM(CASE WHEN {t_curr} THEN od.quantityOrdered ELSE 0 END) AS current_units
{from_clause}
WHERE {where_all}"""
                return sql, params

            primary_dim = dims[0]
            dim_meta = DIMENSION_SQL.get(primary_dim, DIMENSION_SQL["product_line"])
            select_dim = dim_meta["select"]
            group_dim = dim_meta["group"]

            if metric_key == "total_orders":
                prev_expr = f"COUNT(DISTINCT CASE WHEN {t_prev} THEN o.orderNumber END)"
                curr_expr = f"COUNT(DISTINCT CASE WHEN {t_curr} THEN o.orderNumber END)"
            elif metric_key == "units_sold":
                prev_expr = f"COALESCE(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered ELSE 0 END), 0)"
                curr_expr = f"COALESCE(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered ELSE 0 END), 0)"
            else:
                prev_expr = f"ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2)"
                curr_expr = f"ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2)"

            order_clause = "ORDER BY (current - previous) ASC"
            if spec.order == "change_desc":
                order_clause = "ORDER BY (current - previous) DESC"
            elif spec.order == "value_desc":
                order_clause = "ORDER BY current DESC"

            limit_clause = ""
            if spec.limit:
                params["limit_n"] = spec.limit
                limit_clause = "\nLIMIT :limit_n"
            elif primary_dim in ("product", "customer"):
                params["limit_n"] = 15
                limit_clause = "\nLIMIT :limit_n"

            sql = f"""SELECT {select_dim},
       {prev_expr} AS previous,
       {curr_expr} AS current
{from_clause}
WHERE {where_all}
GROUP BY {group_dim}
{order_clause}{limit_clause}"""
            return sql, params

        # Case B: Single-period or full-history aggregation
        is_time_dim = any(d in ("month", "quarter", "year") for d in dims)
        if spec.period_scope == "custom" and spec.custom_start and spec.custom_end:
            params["target_start"] = spec.custom_start
            params["target_end"] = spec.custom_end
            where_clauses.append(t_curr)
        elif spec.period_scope == "plan_period" and (not is_time_dim or period_key is not None):
            # When querying monthly trend across a specific year (e.g. 2004), scope to that year;
            # when querying overall peak month or full history without an explicit period_key, don't restrict to default single month.
            if not is_time_dim or (period_key and len(period_key.strip()) == 4):
                params["target_start"] = period_info["target_start"]
                params["target_end"] = period_info["target_end"]
                where_clauses.append(t_curr)

        where_sql = " AND ".join(where_clauses) if where_clauses else "o.orderNumber IS NOT NULL"

        if not dims:
            sql = f"""SELECT ROUND({metric_expr}, 2) AS value,
       COUNT(DISTINCT o.orderNumber) AS order_count,
       COALESCE(SUM(od.quantityOrdered), 0) AS units_sold
{from_clause}
WHERE {where_sql}"""
            return sql, params

        primary_dim = dims[0]
        dim_meta = DIMENSION_SQL.get(primary_dim, DIMENSION_SQL["product_line"])
        select_dim = dim_meta["select"]
        if primary_dim == "product":
            select_dim = "p.productName AS product, p.productLine AS line"
        group_dim = dim_meta["group"]

        if primary_dim == "order_status":
            sql = f"""SELECT o.status AS label,
       COUNT(DISTINCT o.orderNumber) AS order_count,
       ROUND(SUM(od.quantityOrdered * od.priceEach), 2) AS total_value
{from_clause}
WHERE {where_sql}
GROUP BY o.status
ORDER BY total_value DESC"""
            return sql, params

        order_clause = "ORDER BY value DESC"
        if primary_dim == "month" and spec.order != "value_desc" and spec.limit != 1:
            order_clause = "ORDER BY month ASC"
        elif spec.order == "chronological":
            col_alias = "month" if primary_dim == "month" else "label"
            order_clause = f"ORDER BY {col_alias} ASC"
        elif spec.order == "value_asc":
            order_clause = "ORDER BY value ASC"

        limit_clause = ""
        if spec.limit:
            params["limit_n"] = spec.limit
            limit_clause = "\nLIMIT :limit_n"

        sql = f"""SELECT {select_dim},
       ROUND({metric_expr}, 2) AS value
{from_clause}
WHERE {where_sql}
GROUP BY {group_dim}
{order_clause}{limit_clause}"""
        return sql, params

    def _compile_payments_query(
        self,
        spec: QuerySpec,
        period_info: dict[str, str],
        filters: list[FilterSpec],
    ) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {}
        where_clauses: list[str] = []
        if spec.period_scope != "full_history":
            params["target_start"] = spec.custom_start or period_info["target_start"]
            params["target_end"] = spec.custom_end or period_info["target_end"]
            where_clauses.append("p.paymentDate >= :target_start AND p.paymentDate < :target_end")

        needs_customer_join = bool(spec.dimensions) or any(f.dimension in ("country", "customer") for f in filters)
        for idx, flt in enumerate(filters):
            if flt.dimension == "country":
                pname = f"flt_country_{idx}"
                where_clauses.append(f"c.country = :{pname}")
                params[pname] = flt.value
            elif flt.dimension == "customer":
                pname = f"flt_customer_{idx}"
                where_clauses.append(f"c.customerName = :{pname}")
                params[pname] = flt.value

        where_sql = " AND ".join(where_clauses) if where_clauses else "p.amount >= 0"
        from_sql = "FROM payments p JOIN customers c ON p.customerNumber = c.customerNumber" if needs_customer_join else "FROM payments p"

        if not spec.dimensions:
            return f"SELECT ROUND(SUM(p.amount), 2) AS value\n{from_sql}\nWHERE {where_sql}", params

        dim = spec.dimensions[0]
        if dim == "month":
            return (
                f"SELECT DATE_FORMAT(p.paymentDate, '%Y-%m') AS month, ROUND(SUM(p.amount), 2) AS value\n"
                f"{from_sql}\nWHERE {where_sql}\nGROUP BY DATE_FORMAT(p.paymentDate, '%Y-%m')\nORDER BY month ASC",
                params,
            )
        if dim == "country":
            return (
                f"SELECT c.country AS label, ROUND(SUM(p.amount), 2) AS value\n"
                f"FROM payments p JOIN customers c ON p.customerNumber = c.customerNumber\n"
                f"WHERE {where_sql}\nGROUP BY c.country\nORDER BY value DESC",
                params,
            )
        return (
            f"SELECT c.customerName AS label, ROUND(SUM(p.amount), 2) AS value\n"
            f"FROM payments p JOIN customers c ON p.customerNumber = c.customerNumber\n"
            f"WHERE {where_sql}\nGROUP BY c.customerNumber, c.customerName\nORDER BY value DESC\nLIMIT 15",
            params,
        )

    def build(self, task: PlanTask, intent: QuestionIntent) -> tuple[str, dict[str, Any]]:
        """Legacy compatibility wrapper building parameterized MySQL for PlanTask."""
        period_info = resolve_period(intent.period)
        metric_expr = APPROVED_METRICS.get(intent.metric, APPROVED_METRICS["gross_revenue"])["sql_expression"]

        params: dict[str, Any] = {
            "limit_n": intent.n
        }
        if intent.period is not None or task.kind in (
            "total", "top_products", "baseline", "country_bridge",
            "product_line_bridge", "product_bridge", "customer_bridge", "order_status",
            "by_country", "by_product_line", "by_customer",
        ):
            params["target_start"] = period_info["target_start"]
            params["target_end"] = period_info["target_end"]
            params["previous_start"] = period_info["previous_start"]
            params["previous_end"] = period_info["previous_end"]

        filters = ["o.status != 'Cancelled'"]
        if intent.country:
            filters.append("c.country = :country")
            params["country"] = intent.country
        if intent.product_line:
            filters.append("p.productLine = :product_line")
            params["product_line"] = intent.product_line

        where_base = " AND ".join(filters)
        t_curr = "o.orderDate >= :target_start AND o.orderDate < :target_end"
        t_prev = "o.orderDate >= :previous_start AND o.orderDate < :previous_end"

        if task.kind == "top_products":
            sql = f"""SELECT p.productName AS product, p.productLine AS line,
       ROUND({metric_expr}, 2) AS value
{BASE_ORDER_FROM}
WHERE {where_base} AND {t_curr}
GROUP BY p.productCode, p.productName, p.productLine
ORDER BY value DESC
LIMIT :limit_n"""
        elif task.kind == "total":
            if intent.metric == "payments_received":
                sql = """SELECT ROUND(SUM(p.amount), 2) AS value
FROM payments p
WHERE p.paymentDate >= :target_start AND p.paymentDate < :target_end"""
            else:
                sql = f"""SELECT ROUND({metric_expr}, 2) AS value
{BASE_ORDER_FROM}
WHERE {where_base} AND {t_curr}"""
        elif task.kind == "trend":
            if intent.n == 1:
                sql = f"""SELECT DATE_FORMAT(o.orderDate, '%Y-%m') AS month,
       ROUND({metric_expr}, 2) AS value
{BASE_ORDER_FROM}
WHERE {where_base}
GROUP BY DATE_FORMAT(o.orderDate, '%Y-%m')
ORDER BY value DESC
LIMIT :limit_n"""
            else:
                sql = f"""SELECT DATE_FORMAT(o.orderDate, '%Y-%m') AS month,
       ROUND({metric_expr}, 2) AS value
{BASE_ORDER_FROM}
WHERE {where_base}
GROUP BY DATE_FORMAT(o.orderDate, '%Y-%m')
ORDER BY month ASC"""
        elif task.kind in ("by_country", "country_bridge"):
            if task.kind == "country_bridge":
                sql = f"""SELECT c.country AS label,
       ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous,
       ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current
{BASE_ORDER_FROM}
WHERE o.status != 'Cancelled' AND ({t_prev} OR {t_curr})
GROUP BY c.country
ORDER BY (current - previous) ASC"""
            else:
                sql = f"""SELECT c.country AS label,
       ROUND({metric_expr}, 2) AS value
{BASE_ORDER_FROM}
WHERE {where_base} AND {t_curr}
GROUP BY c.country
ORDER BY value DESC
LIMIT :limit_n"""
        elif task.kind in ("by_product_line", "product_line_bridge"):
            if task.kind == "product_line_bridge":
                sql = f"""SELECT p.productLine AS label,
       ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous,
       ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current
{BASE_ORDER_FROM}
WHERE o.status != 'Cancelled' AND ({t_prev} OR {t_curr})
GROUP BY p.productLine
ORDER BY (current - previous) ASC"""
            else:
                sql = f"""SELECT p.productLine AS label,
       ROUND({metric_expr}, 2) AS value
{BASE_ORDER_FROM}
WHERE {where_base} AND {t_curr}
GROUP BY p.productLine
ORDER BY value DESC
LIMIT :limit_n"""
        elif task.kind == "by_customer":
            sql = f"""SELECT c.customerName AS label, c.country AS country,
       ROUND({metric_expr}, 2) AS value
{BASE_ORDER_FROM}
WHERE {where_base} AND {t_curr}
GROUP BY c.customerNumber, c.customerName, c.country
ORDER BY value DESC
LIMIT :limit_n"""
        elif task.kind == "baseline":
            sql = f"""SELECT
    ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous_revenue,
    ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current_revenue,
    COUNT(DISTINCT CASE WHEN {t_prev} THEN o.orderNumber END) AS previous_orders,
    COUNT(DISTINCT CASE WHEN {t_curr} THEN o.orderNumber END) AS current_orders
{BASE_ORDER_FROM}
WHERE o.status != 'Cancelled' AND ({t_prev} OR {t_curr})"""
        elif task.kind == "product_bridge":
            sql = f"""SELECT p.productName AS label, p.productLine AS category,
       ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous,
       ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current
{BASE_ORDER_FROM}
WHERE o.status != 'Cancelled' AND ({t_prev} OR {t_curr})
GROUP BY p.productCode, p.productName, p.productLine
ORDER BY (current - previous) ASC
LIMIT 15"""
        elif task.kind == "customer_bridge":
            sql = f"""SELECT c.customerName AS label, c.country AS country,
       ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous,
       ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current
{BASE_ORDER_FROM}
WHERE o.status != 'Cancelled' AND ({t_prev} OR {t_curr})
GROUP BY c.customerNumber, c.customerName, c.country
ORDER BY (current - previous) ASC
LIMIT 15"""
        elif task.kind == "order_status":
            sql = f"""SELECT o.status AS label,
       COUNT(DISTINCT o.orderNumber) AS order_count,
       ROUND(SUM(od.quantityOrdered * od.priceEach), 2) AS total_value
{BASE_ORDER_FROM}
WHERE ({t_curr})
GROUP BY o.status
ORDER BY total_value DESC"""
        else:
            sql = f"""SELECT ROUND({metric_expr}, 2) AS value
{BASE_ORDER_FROM}
WHERE {where_base} AND {t_curr}"""

        return sql, params

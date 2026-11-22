"""SQL template builder and QuerySpec compiler for ClassicModels."""
from __future__ import annotations
from typing import Any
from app.orchestration.contracts import FilterSpec, QuerySpec
from app.schemas import PlanTask, QuestionIntent
from app.services.semantics import APPROVED_METRICS, BASE_ORDER_FROM, resolve_period

EXTENDED_ORDER_FROM = f"""{BASE_ORDER_FROM}
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
    "sales_rep": {"select": "COALESCE(CONCAT(e.firstName, ' ', e.lastName), 'Unassigned') AS label", "group": "e.employeeNumber, e.firstName, e.lastName", "needs_extended": "true"},
    "office": {"select": "COALESCE(CONCAT(ofc.city, ' (', ofc.country, ')'), 'Unassigned') AS label", "group": "ofc.officeCode, ofc.city, ofc.country", "needs_extended": "true"},
}

FILTER_COLUMN_MAP: dict[str, str] = {
    "country": "c.country",
    "product_line": "p.productLine",
    "customer": "c.customerName",
    "product": "p.productName",
    "order_status": "o.status",
}


class SQLBuilder:
    def compile_query_spec(
        self,
        spec: QuerySpec,
        period_key: str | None = None,
        plan_filters: list[FilterSpec] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        period_info = resolve_period(period_key)
        metric_key = spec.metric if spec.metric in APPROVED_METRICS else "gross_revenue"
        metric_expr = APPROVED_METRICS[metric_key]["sql_expression"]

        effective_filters: list[FilterSpec] = list(plan_filters) if (spec.inherit_plan_filters and plan_filters) else []
        for f in spec.filters:
            if not any(ef.dimension == f.dimension and ef.value == f.value for ef in effective_filters):
                effective_filters.append(f)

        if metric_key == "payments_received":
            return self._compile_payments_query(spec, period_info, effective_filters)

        dims = list(spec.dimensions)
        params: dict[str, Any] = {}
        where_clauses: list[str] = []
        if "order_status" not in dims and not any(f.dimension == "order_status" for f in effective_filters):
            where_clauses.append("o.status != 'Cancelled'")

        for idx, flt in enumerate(effective_filters):
            if col := FILTER_COLUMN_MAP.get(flt.dimension):
                pname = f"flt_{flt.dimension}_{idx}"
                where_clauses.append(f"{col} = :{pname}")
                params[pname] = flt.value

        t_curr = "o.orderDate >= :target_start AND o.orderDate < :target_end"
        t_prev = "o.orderDate >= :previous_start AND o.orderDate < :previous_end"
        from_clause = EXTENDED_ORDER_FROM if any(DIMENSION_SQL.get(d, {}).get("needs_extended") == "true" for d in dims) else BASE_ORDER_FROM

        if spec.comparison == "previous_period":
            params.update({
                "target_start": spec.custom_start or period_info["target_start"],
                "target_end": spec.custom_end or period_info["target_end"],
                "previous_start": period_info["previous_start"],
                "previous_end": period_info["previous_end"],
            })
            where_all = " AND ".join(where_clauses + [f"({t_prev} OR {t_curr})"])

            if not dims:
                return f"""SELECT
    ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous_revenue,
    ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current_revenue,
    COUNT(DISTINCT CASE WHEN {t_prev} THEN o.orderNumber END) AS previous_orders,
    COUNT(DISTINCT CASE WHEN {t_curr} THEN o.orderNumber END) AS current_orders,
    SUM(CASE WHEN {t_prev} THEN od.quantityOrdered ELSE 0 END) AS previous_units,
    SUM(CASE WHEN {t_curr} THEN od.quantityOrdered ELSE 0 END) AS current_units
{from_clause}
WHERE {where_all}""", params

            primary_dim = dims[0]
            dim_meta = DIMENSION_SQL.get(primary_dim, DIMENSION_SQL["product_line"])
            if metric_key == "total_orders":
                prev_expr, curr_expr = f"COUNT(DISTINCT CASE WHEN {t_prev} THEN o.orderNumber END)", f"COUNT(DISTINCT CASE WHEN {t_curr} THEN o.orderNumber END)"
            elif metric_key == "units_sold":
                prev_expr, curr_expr = f"COALESCE(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered ELSE 0 END), 0)", f"COALESCE(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered ELSE 0 END), 0)"
            elif metric_key == "aov":
                prev_expr = f"ROUND(COALESCE(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END) / NULLIF(COUNT(DISTINCT CASE WHEN {t_prev} THEN o.orderNumber END), 0), 0), 2)"
                curr_expr = f"ROUND(COALESCE(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END) / NULLIF(COUNT(DISTINCT CASE WHEN {t_curr} THEN o.orderNumber END), 0), 0), 2)"
            else:
                prev_expr = f"ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2)"
                curr_expr = f"ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2)"

            order_clause = "ORDER BY (current - previous) DESC" if spec.order == "change_desc" else ("ORDER BY current DESC" if spec.order == "value_desc" else "ORDER BY (current - previous) ASC")
            limit_clause = ""
            if spec.limit or primary_dim in ("product", "customer"):
                params["limit_n"] = spec.limit or 15
                limit_clause = "\nLIMIT :limit_n"

            return f"""SELECT {dim_meta['select']},
       {prev_expr} AS previous,
       {curr_expr} AS current
{from_clause}
WHERE {where_all}
GROUP BY {dim_meta['group']}
{order_clause}{limit_clause}""", params

        is_time_dim = any(d in ("month", "quarter", "year") for d in dims)
        if spec.period_scope == "custom" and spec.custom_start and spec.custom_end:
            params["target_start"], params["target_end"] = spec.custom_start, spec.custom_end
            where_clauses.append(t_curr)
        elif spec.period_scope == "plan_period" and (not is_time_dim or (period_key and len(period_key.strip()) == 4)):
            params["target_start"], params["target_end"] = period_info["target_start"], period_info["target_end"]
            where_clauses.append(t_curr)

        where_sql = " AND ".join(where_clauses) if where_clauses else "o.orderNumber IS NOT NULL"
        if not dims:
            return f"""SELECT ROUND({metric_expr}, 2) AS value,
       COUNT(DISTINCT o.orderNumber) AS order_count,
       COALESCE(SUM(od.quantityOrdered), 0) AS units_sold
{from_clause}
WHERE {where_sql}""", params

        primary_dim = dims[0]
        if primary_dim == "order_status":
            return f"""SELECT o.status AS label,
       COUNT(DISTINCT o.orderNumber) AS order_count,
       ROUND(SUM(od.quantityOrdered * od.priceEach), 2) AS total_value
{from_clause}
WHERE {where_sql}
GROUP BY o.status
ORDER BY total_value DESC""", params

        dim_meta = DIMENSION_SQL.get(primary_dim, DIMENSION_SQL["product_line"])
        select_dim = "p.productName AS product, p.productLine AS line" if primary_dim == "product" else dim_meta["select"]
        if primary_dim == "month" and spec.order != "value_desc" and spec.limit != 1:
            order_clause = "ORDER BY month ASC"
        elif spec.order == "chronological":
            order_clause = f"ORDER BY {'month' if primary_dim == 'month' else 'label'} ASC"
        elif spec.order == "value_asc":
            order_clause = "ORDER BY value ASC"
        else:
            order_clause = "ORDER BY value DESC"

        limit_clause = ""
        if spec.limit:
            params["limit_n"] = spec.limit
            limit_clause = "\nLIMIT :limit_n"

        return f"""SELECT {select_dim},
       ROUND({metric_expr}, 2) AS value
{from_clause}
WHERE {where_sql}
GROUP BY {dim_meta['group']}
{order_clause}{limit_clause}""", params

    def _compile_payments_query(self, spec: QuerySpec, period_info: dict[str, str], filters: list[FilterSpec]) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {}
        where_clauses: list[str] = []
        needs_customer_join = bool(spec.dimensions) or any(f.dimension in ("country", "customer") for f in filters)
        for idx, flt in enumerate(filters):
            if flt.dimension in ("country", "customer"):
                pname = f"flt_{flt.dimension}_{idx}"
                col = "c.country" if flt.dimension == "country" else "c.customerName"
                where_clauses.append(f"{col} = :{pname}")
                params[pname] = flt.value

        from_sql = "FROM payments p JOIN customers c ON p.customerNumber = c.customerNumber" if needs_customer_join else "FROM payments p"

        if spec.comparison == "previous_period":
            params.update({
                "target_start": spec.custom_start or period_info["target_start"],
                "target_end": spec.custom_end or period_info["target_end"],
                "previous_start": period_info["previous_start"],
                "previous_end": period_info["previous_end"],
            })
            t_curr = "p.paymentDate >= :target_start AND p.paymentDate < :target_end"
            t_prev = "p.paymentDate >= :previous_start AND p.paymentDate < :previous_end"
            where_all = " AND ".join(where_clauses + [f"({t_prev} OR {t_curr})"])
            if not spec.dimensions:
                return (
                    f"SELECT ROUND(SUM(CASE WHEN {t_prev} THEN p.amount ELSE 0 END), 2) AS previous_revenue,\n"
                    f"       ROUND(SUM(CASE WHEN {t_curr} THEN p.amount ELSE 0 END), 2) AS current_revenue,\n"
                    f"       COUNT(CASE WHEN {t_prev} THEN 1 END) AS previous_orders,\n"
                    f"       COUNT(CASE WHEN {t_curr} THEN 1 END) AS current_orders\n"
                    f"{from_sql}\nWHERE {where_all}",
                    params,
                )
            sel_col, grp_col = ("c.country AS label", "c.country") if spec.dimensions[0] == "country" else ("c.customerName AS label", "c.customerNumber, c.customerName")
            return (
                f"SELECT {sel_col},\n"
                f"       ROUND(SUM(CASE WHEN {t_prev} THEN p.amount ELSE 0 END), 2) AS previous,\n"
                f"       ROUND(SUM(CASE WHEN {t_curr} THEN p.amount ELSE 0 END), 2) AS current\n"
                f"FROM payments p JOIN customers c ON p.customerNumber = c.customerNumber\n"
                f"WHERE {where_all}\nGROUP BY {grp_col}\nORDER BY (current - previous) ASC",
                params,
            )

        if spec.period_scope != "full_history":
            params["target_start"] = spec.custom_start or period_info["target_start"]
            params["target_end"] = spec.custom_end or period_info["target_end"]
            where_clauses.append("p.paymentDate >= :target_start AND p.paymentDate < :target_end")

        where_sql = " AND ".join(where_clauses) if where_clauses else "p.amount >= 0"
        if not spec.dimensions:
            return f"SELECT ROUND(SUM(p.amount), 2) AS value\n{from_sql}\nWHERE {where_sql}", params

        dim = spec.dimensions[0]
        if dim == "month":
            return f"SELECT DATE_FORMAT(p.paymentDate, '%Y-%m') AS month, ROUND(SUM(p.amount), 2) AS value\n{from_sql}\nWHERE {where_sql}\nGROUP BY DATE_FORMAT(p.paymentDate, '%Y-%m')\nORDER BY month ASC", params
        if dim == "country":
            return f"SELECT c.country AS label, ROUND(SUM(p.amount), 2) AS value\nFROM payments p JOIN customers c ON p.customerNumber = c.customerNumber\nWHERE {where_sql}\nGROUP BY c.country\nORDER BY value DESC", params
        return f"SELECT c.customerName AS label, ROUND(SUM(p.amount), 2) AS value\nFROM payments p JOIN customers c ON p.customerNumber = c.customerNumber\nWHERE {where_sql}\nGROUP BY c.customerNumber, c.customerName\nORDER BY value DESC\nLIMIT 15", params

    def build(self, task: PlanTask, intent: QuestionIntent) -> tuple[str, dict[str, Any]]:
        period_info = resolve_period(intent.period)
        metric_expr = APPROVED_METRICS.get(intent.metric, APPROVED_METRICS["gross_revenue"])["sql_expression"]
        params: dict[str, Any] = {"limit_n": intent.n}

        if intent.period is not None or task.kind != "trend":
            params.update({
                "target_start": period_info["target_start"],
                "target_end": period_info["target_end"],
                "previous_start": period_info["previous_start"],
                "previous_end": period_info["previous_end"],
            })

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
            return f"SELECT p.productName AS product, p.productLine AS line, ROUND({metric_expr}, 2) AS value\n{BASE_ORDER_FROM}\nWHERE {where_base} AND {t_curr}\nGROUP BY p.productCode, p.productName, p.productLine\nORDER BY value DESC\nLIMIT :limit_n", params
        if task.kind == "trend":
            order_lim = "ORDER BY value DESC\nLIMIT :limit_n" if intent.n == 1 else "ORDER BY month ASC"
            return f"SELECT DATE_FORMAT(o.orderDate, '%Y-%m') AS month, ROUND({metric_expr}, 2) AS value\n{BASE_ORDER_FROM}\nWHERE {where_base}\nGROUP BY DATE_FORMAT(o.orderDate, '%Y-%m')\n{order_lim}", params
        if task.kind in ("by_country", "by_product_line", "by_customer"):
            col_map = {
                "by_country": ("c.country AS label", "c.country"),
                "by_product_line": ("p.productLine AS label", "p.productLine"),
                "by_customer": ("c.customerName AS label, c.country AS country", "c.customerNumber, c.customerName, c.country"),
            }
            sel, grp = col_map[task.kind]
            return f"SELECT {sel}, ROUND({metric_expr}, 2) AS value\n{BASE_ORDER_FROM}\nWHERE {where_base} AND {t_curr}\nGROUP BY {grp}\nORDER BY value DESC\nLIMIT :limit_n", params
        if task.kind in ("country_bridge", "product_line_bridge", "product_bridge", "customer_bridge"):
            bridge_map = {
                "country_bridge": ("c.country AS label", "c.country", ""),
                "product_line_bridge": ("p.productLine AS label", "p.productLine", ""),
                "product_bridge": ("p.productName AS label, p.productLine AS category", "p.productCode, p.productName, p.productLine", "\nLIMIT 15"),
                "customer_bridge": ("c.customerName AS label, c.country AS country", "c.customerNumber, c.customerName, c.country", "\nLIMIT 15"),
            }
            sel, grp, lim = bridge_map[task.kind]
            return (
                f"SELECT {sel},\n"
                f"       ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous,\n"
                f"       ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current\n"
                f"{BASE_ORDER_FROM}\nWHERE o.status != 'Cancelled' AND ({t_prev} OR {t_curr})\nGROUP BY {grp}\nORDER BY (current - previous) ASC{lim}",
                params,
            )
        if task.kind == "baseline":
            return (
                f"SELECT ROUND(SUM(CASE WHEN {t_prev} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous_revenue,\n"
                f"       ROUND(SUM(CASE WHEN {t_curr} THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current_revenue,\n"
                f"       COUNT(DISTINCT CASE WHEN {t_prev} THEN o.orderNumber END) AS previous_orders,\n"
                f"       COUNT(DISTINCT CASE WHEN {t_curr} THEN o.orderNumber END) AS current_orders\n"
                f"{BASE_ORDER_FROM}\nWHERE o.status != 'Cancelled' AND ({t_prev} OR {t_curr})",
                params,
            )
        if task.kind == "order_status":
            return f"SELECT o.status AS label, COUNT(DISTINCT o.orderNumber) AS order_count, ROUND(SUM(od.quantityOrdered * od.priceEach), 2) AS total_value\n{BASE_ORDER_FROM}\nWHERE ({t_curr})\nGROUP BY o.status\nORDER BY total_value DESC", params
        if task.kind == "total" and intent.metric == "payments_received":
            return "SELECT ROUND(SUM(p.amount), 2) AS value\nFROM payments p\nWHERE p.paymentDate >= :target_start AND p.paymentDate < :target_end", params
        return f"SELECT ROUND({metric_expr}, 2) AS value\n{BASE_ORDER_FROM}\nWHERE {where_base} AND {t_curr}", params

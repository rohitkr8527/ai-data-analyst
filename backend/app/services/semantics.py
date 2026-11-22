"""ClassicModels business semantics, approved metrics, dimensions, and date resolution."""
from __future__ import annotations
from datetime import date
import re
from typing import Any
from app.schemas import QuestionIntent

DATA_MIN_DATE = "2003-01-06"
DATA_MAX_DATE = "2005-05-31"
DEFAULT_HISTORICAL_PERIOD = "2004-11"

APPROVED_METRICS: dict[str, dict[str, Any]] = {
    "gross_revenue": {
        "label": "Gross merchandise revenue",
        "unit": "USD",
        "description": "Total order line revenue before returns or shipping: SUM(od.quantityOrdered * od.priceEach). Excludes Cancelled orders.",
        "sql_expression": "SUM(od.quantityOrdered * od.priceEach)",
    },
    "total_orders": {
        "label": "Total orders",
        "unit": "orders",
        "description": "Count of distinct order records excluding Cancelled status: COUNT(DISTINCT o.orderNumber).",
        "sql_expression": "COUNT(DISTINCT o.orderNumber)",
    },
    "aov": {
        "label": "Average order value (AOV)",
        "unit": "USD",
        "description": "Gross merchandise revenue divided by completed order volume.",
        "sql_expression": "ROUND(SUM(od.quantityOrdered * od.priceEach) / NULLIF(COUNT(DISTINCT o.orderNumber), 0), 2)",
    },
    "units_sold": {
        "label": "Units sold",
        "unit": "units",
        "description": "Sum of all product units ordered on valid orders: SUM(od.quantityOrdered).",
        "sql_expression": "SUM(od.quantityOrdered)",
    },
    "payments_received": {
        "label": "Payments received",
        "unit": "USD",
        "description": "Total cash receipts recorded in payments table: SUM(p.amount).",
        "sql_expression": "SUM(p.amount)",
    },
}

APPROVED_DIMENSIONS: dict[str, dict[str, str]] = {
    "product_line": {"label": "Product line", "column": "p.productLine", "description": "Scale model categories"},
    "product": {"label": "Product", "column": "p.productName", "description": "Individual model names"},
    "country": {"label": "Country", "column": "c.country", "description": "Customer country of operation"},
    "customer": {"label": "Customer", "column": "c.customerName", "description": "Customer business entity name"},
    "order_status": {"label": "Order status", "column": "o.status", "description": "Fulfillment status"},
    "month": {"label": "Month", "column": "DATE_FORMAT(o.orderDate, '%Y-%m')", "description": "Order calendar month (YYYY-MM)"},
    "quarter": {"label": "Quarter", "column": "CONCAT(YEAR(o.orderDate), '-Q', QUARTER(o.orderDate))", "description": "Order calendar quarter"},
    "year": {"label": "Year", "column": "CAST(YEAR(o.orderDate) AS CHAR)", "description": "Order calendar year"},
    "sales_rep": {"label": "Sales Representative", "column": "CONCAT(e.firstName, ' ', e.lastName)", "description": "Assigned customer sales representative"},
    "office": {"label": "Sales Office", "column": "CONCAT(ofc.city, ' (', ofc.country, ')')", "description": "Sales office territory and city"},
}

BASE_ORDER_FROM = """FROM orders o
JOIN orderdetails od ON o.orderNumber = od.orderNumber
JOIN products p ON od.productCode = p.productCode
JOIN customers c ON o.customerNumber = c.customerNumber"""

_MONTH_NAMES = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_KNOWN_COUNTRIES = (
    "usa", "france", "australia", "uk", "germany", "japan", "spain", "italy",
    "canada", "finland", "norway", "singapore", "denmark", "sweden", "belgium",
    "switzerland", "new zealand",
)
_KNOWN_LINES = (
    "classic cars", "vintage cars", "motorcycles", "planes",
    "ships", "trains", "trucks and buses",
)


def shift_month(dt: date, delta: int) -> date:
    z = dt.year * 12 + dt.month - 1 + delta
    return date(z // 12, z % 12 + 1, 1)


def resolve_period(period_str: str | None = None) -> dict[str, str]:
    target_date = date.fromisoformat(f"{DEFAULT_HISTORICAL_PERIOD}-01")
    if period_str:
        s = period_str.strip()
        if m := re.match(r"^(\d{4})[-/](0[1-9]|1[0-2])$", s):
            target_date = date(int(m.group(1)), int(m.group(2)), 1)
        elif m_year := re.match(r"^(\d{4})$", s):
            yr = int(m_year.group(1))
            return {
                "is_annual": "true",
                "target_start": f"{yr}-01-01",
                "target_end": f"{yr + 1}-01-01",
                "previous_start": f"{yr - 1}-01-01",
                "previous_end": f"{yr}-01-01",
                "target_label": str(yr),
                "previous_label": str(yr - 1),
                "target_period": str(yr),
                "note": f"Annual period {yr} (data available {DATA_MIN_DATE} to {DATA_MAX_DATE})",
            }

    prev_date = shift_month(target_date, -1)
    next_date = shift_month(target_date, 1)
    return {
        "is_annual": "false",
        "target_start": target_date.isoformat(),
        "target_end": next_date.isoformat(),
        "previous_start": prev_date.isoformat(),
        "previous_end": target_date.isoformat(),
        "target_label": target_date.strftime("%B %Y"),
        "previous_label": prev_date.strftime("%B %Y"),
        "target_period": target_date.strftime("%Y-%m"),
        "note": f"Historical data range: {DATA_MIN_DATE} to {DATA_MAX_DATE}.",
    }


def extract_intent(question: str, previous: dict[str, Any] | None = None) -> QuestionIntent:
    q = question.lower().strip()
    prev = QuestionIntent.model_validate(previous) if previous else None

    metric = "gross_revenue"
    if re.search(r"\b(payments?|cash|receipts?)\b", q):
        metric = "payments_received"
    elif re.search(r"\b(orders?|order count)\b", q) and not re.search(r"\b(average|aov|value)\b", q):
        metric = "total_orders"
    elif re.search(r"\b(aov|average order)\b", q):
        metric = "aov"
    elif re.search(r"\b(units?|quantity)\b", q):
        metric = "units_sold"
    elif prev and not re.search(r"\b(sales|revenue|orders?|payments?)\b", q):
        metric = prev.metric

    period = None
    iso_months = [f"{m.group(1)}-{m.group(2)}" for m in re.finditer(r"\b(200[3-5])[-/](0[1-9]|1[0-2])\b", q)]
    named_months = [
        f"{m.group(2)}-{_MONTH_NAMES[m.group(1)]}"
        for m in re.finditer(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(200[3-5])\b", q)
    ]
    if all_months := (iso_months or named_months):
        period = max(all_months)
    elif years := re.findall(r"\b(200[3-5])\b", q):
        period = max(years)
    elif prev and prev.period and not re.search(r"\b(all time|full history|entire history|overall)\b", q):
        period = prev.period

    country = next(
        (c.upper() if c in ("usa", "uk") else c.title() for c in _KNOWN_COUNTRIES if re.search(rf"\b{c}\b", q)),
        None,
    )
    if not country and prev and prev.country and not re.search(r"\b(all countries|every country|global|worldwide)\b", q):
        country = prev.country

    product_line = next((pl.title() for pl in _KNOWN_LINES if pl in q), None)
    if not product_line and prev and prev.product_line and not re.search(r"\b(all product ?lines|all categories|every product ?line)\b", q):
        product_line = prev.product_line

    is_month_q = bool(re.search(r"\b(months?|monthly)\b", q))
    is_country_q = bool(re.search(r"\b(country|countries|nations?|territor(y|ies))\b", q))
    is_pl_q = bool(re.search(r"\b(product ?lines?|categor(y|ies))\b", q))
    is_cust_q = bool(re.search(r"\b(customers?|clients?|accounts?)\b", q))
    is_prod_q = bool(re.search(r"\b(products?|models?|items?)\b", q))
    is_ranking = bool(re.search(r"\b(top|best|leading|highest|peak|lowest|worst|most)\b", q))
    is_single_best = bool(
        re.search(r"\b(which|what|peak|best|lowest|worst)\s+(month|country|product ?line|category|customer|product)\b", q)
        or re.search(r"\b(which|what)\s+(month|country|product ?line|category|customer|product)\s+(has|had|generat|produc|saw|is)\b", q)
    )

    if m_top := re.search(r"\btop\s+(\d+)\b", q):
        n = min(50, max(1, int(m_top.group(1))))
    elif is_single_best:
        n = 1
    elif prev and prev.n and prev.n > 1:
        n = prev.n
    else:
        n = 10

    if re.search(r"\b(why|drop|decrease|decline|fell|down|root cause|investigat)\b", q):
        kind = "deep"
    elif re.search(r"\b(compare|compar|versus|vs|difference|bridge)\b", q):
        kind = "comparative"
    elif (is_month_q and (is_ranking or any(k in q for k in ("trend", "over time", "history")))) or re.search(r"\b(trend|over time|monthly|history)\b", q):
        kind = "trend"
    elif (is_country_q and (is_ranking or any(k in q for k in ("by", "across", "each")))) or re.search(r"\b(by country|per country|across countries|geographic)\b", q):
        kind = "by_country"
    elif (is_pl_q and (is_ranking or any(k in q for k in ("by", "across", "each")))) or re.search(r"\b(by product line|by category|per line)\b", q):
        kind = "by_product_line"
    elif is_cust_q and (is_ranking or "by" in q or "across" in q):
        kind = "by_customer"
    elif is_prod_q or (is_ranking and not (is_month_q or is_country_q or is_pl_q)):
        kind = "top_products"
    elif re.search(r"\b(total|revenue|sales|volume|payments)\b", q):
        kind = "total"
    elif prev:
        kind = prev.kind
    else:
        kind = "simple"

    return QuestionIntent(
        kind=kind, metric=metric, period=period, country=country,
        product_line=product_line, n=n, is_followup=bool(prev),
    )


def public_catalog() -> dict[str, Any]:
    return {
        "metrics": {k: {"label": v["label"], "unit": v["unit"], "description": v["description"]} for k, v in APPROVED_METRICS.items()},
        "dimensions": {k: {"label": v["label"], "description": v["description"]} for k, v in APPROVED_DIMENSIONS.items()},
        "data_range": {"min_date": DATA_MIN_DATE, "max_date": DATA_MAX_DATE, "default_comparison": "November 2004 vs October 2004"},
    }

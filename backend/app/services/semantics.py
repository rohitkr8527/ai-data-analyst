"""ClassicModels business semantics, approved metrics, dimensions, and date resolution.

Decouples semantic reporting definitions from LLM prompts.
Enforces strict accounting rules: payments received != order-line revenue;
cancelled orders are excluded from completed sales metrics.
"""
from __future__ import annotations
from datetime import date
import re
from typing import Any
from app.schemas import QuestionIntent

# ClassicModels Historical Date Range
DATA_MIN_DATE = "2003-01-06"
DATA_MAX_DATE = "2005-05-31"
DEFAULT_HISTORICAL_PERIOD = "2004-11"  # ClassicModels highest volume comparison month

APPROVED_METRICS: dict[str, dict[str, Any]] = {
    "gross_revenue": {
        "label": "Gross merchandise revenue",
        "unit": "USD",
        "description": "Total order line revenue before returns or shipping: SUM(od.quantityOrdered * od.priceEach). Excludes Cancelled orders.",
        "sql_expression": "SUM(od.quantityOrdered * od.priceEach)"
    },
    "total_orders": {
        "label": "Total orders",
        "unit": "orders",
        "description": "Count of distinct order records excluding Cancelled status: COUNT(DISTINCT o.orderNumber).",
        "sql_expression": "COUNT(DISTINCT o.orderNumber)"
    },
    "aov": {
        "label": "Average order value (AOV)",
        "unit": "USD",
        "description": "Gross merchandise revenue divided by completed order volume.",
        "sql_expression": "ROUND(SUM(od.quantityOrdered * od.priceEach) / NULLIF(COUNT(DISTINCT o.orderNumber), 0), 2)"
    },
    "units_sold": {
        "label": "Units sold",
        "unit": "units",
        "description": "Sum of all product units ordered on valid orders: SUM(od.quantityOrdered).",
        "sql_expression": "SUM(od.quantityOrdered)"
    },
    "payments_received": {
        "label": "Payments received",
        "unit": "USD",
        "description": "Total cash receipts recorded in payments table: SUM(p.amount). Note: Payments received is cash flow, distinct from order-line sales.",
        "sql_expression": "SUM(p.amount)"
    }
}

APPROVED_DIMENSIONS: dict[str, dict[str, str]] = {
    "product_line": {
        "label": "Product line",
        "column": "p.productLine",
        "description": "Scale model categories (e.g. Classic Cars, Vintage Cars, Motorcycles, Planes, Ships, Trains, Trucks and Buses)"
    },
    "product": {
        "label": "Product",
        "column": "p.productName",
        "description": "Individual model names (e.g. 1969 Harley Davidson Ultimate Chopper)"
    },
    "country": {
        "label": "Country",
        "column": "c.country",
        "description": "Customer country of operation (e.g. USA, France, Australia, UK, Germany)"
    },
    "customer": {
        "label": "Customer",
        "column": "c.customerName",
        "description": "Customer business entity name"
    },
    "order_status": {
        "label": "Order status",
        "column": "o.status",
        "description": "Fulfillment status: Shipped, Resolved, In Process, On Hold, Disputed, Cancelled"
    },
    "month": {
        "label": "Month",
        "column": "DATE_FORMAT(o.orderDate, '%Y-%m')",
        "description": "Order calendar month (YYYY-MM)"
    },
    "quarter": {
        "label": "Quarter",
        "column": "CONCAT(YEAR(o.orderDate), '-Q', QUARTER(o.orderDate))",
        "description": "Order calendar quarter (YYYY-QN)"
    },
    "year": {
        "label": "Year",
        "column": "CAST(YEAR(o.orderDate) AS CHAR)",
        "description": "Order calendar year (YYYY)"
    },
    "sales_rep": {
        "label": "Sales Representative",
        "column": "CONCAT(e.firstName, ' ', e.lastName)",
        "description": "Assigned customer sales representative from employees table"
    },
    "office": {
        "label": "Sales Office",
        "column": "CONCAT(ofc.city, ' (', ofc.country, ')')",
        "description": "Sales office territory and city from offices table"
    }
}

BASE_ORDER_FROM = """FROM orders o
JOIN orderdetails od ON o.orderNumber = od.orderNumber
JOIN products p ON od.productCode = p.productCode
JOIN customers c ON o.customerNumber = c.customerNumber"""


def shift_month(dt: date, delta: int) -> date:
    """Safely increment or decrement months on a 1st-of-month date."""
    z = dt.year * 12 + dt.month - 1 + delta
    return date(z // 12, z % 12 + 1, 1)


def resolve_period(period_str: str | None = None) -> dict[str, str]:
    """Resolve target and comparison historical period windows for ClassicModels.

    Since ClassicModels data spans 2003-01 to 2005-05, relative questions without explicit
    dates resolve to the well-documented comparison period (November 2004 vs October 2004)
    with explicit labeling.
    """
    if period_str:
        # Match YYYY-MM
        m = re.match(r"^(\d{4})[-/](0[1-9]|1[0-2])$", period_str.strip())
        if m:
            target_date = date(int(m.group(1)), int(m.group(2)), 1)
        else:
            # Match YYYY
            m_year = re.match(r"^(\d{4})$", period_str.strip())
            if m_year:
                target_date = date(int(m_year.group(1)), 1, 1)
                prev_date = date(int(m_year.group(1)) - 1, 1, 1)
                return {
                    "is_annual": "true",
                    "target_start": f"{target_date.year}-01-01",
                    "target_end": f"{target_date.year + 1}-01-01",
                    "previous_start": f"{prev_date.year}-01-01",
                    "previous_end": f"{target_date.year}-01-01",
                    "target_label": str(target_date.year),
                    "previous_label": str(prev_date.year),
                    "target_period": str(target_date.year),
                    "note": f"Annual period {target_date.year} (data available {DATA_MIN_DATE} to {DATA_MAX_DATE})"
                }
            else:
                target_date = date.fromisoformat(DEFAULT_HISTORICAL_PERIOD + "-01")
    else:
        target_date = date.fromisoformat(DEFAULT_HISTORICAL_PERIOD + "-01")

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
        "note": f"Historical data range: {DATA_MIN_DATE} to {DATA_MAX_DATE}."
    }


def extract_intent(question: str, previous: dict[str, Any] | None = None) -> QuestionIntent:
    """Extract structural parameters (metric, dimension, period, filter) from the question."""
    q = question.lower().split(";", 1)[0].strip()
    prev = QuestionIntent.model_validate(previous) if previous else None

    # Detect metric
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

    # Detect period
    period = None
    # Check for month YYYY-MM or Month YYYY
    m_month = re.search(r"\b(200[3-5])[-/](0[1-9]|1[0-2])\b", q)
    if m_month:
        period = f"{m_month.group(1)}-{m_month.group(2)}"
    else:
        m_named = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(200[3-5])\b", q)
        if m_named:
            month_names = {
                "january": "01", "february": "02", "march": "03", "april": "04",
                "may": "05", "june": "06", "july": "07", "august": "08",
                "september": "09", "october": "10", "november": "11", "december": "12"
            }
            period = f"{m_named.group(2)}-{month_names[m_named.group(1)]}"
        else:
            m_year = re.search(r"\b(200[3-5])\b", q)
            if m_year:
                period = m_year.group(1)
            elif prev and prev.period:
                period = prev.period

    # Detect country / region
    country = None
    known_countries = [
        "usa", "france", "australia", "uk", "germany", "japan",
        "spain", "italy", "canada", "finland", "norway", "singapore",
        "denmark", "sweden", "belgium", "switzerland", "new zealand"
    ]
    for c in known_countries:
        if re.search(rf"\b{c}\b", q):
            country = c.upper() if c in ("usa", "uk") else c.title()
            break
    if not country and prev and prev.country:
        country = prev.country

    # Detect product line
    product_line = None
    known_lines = [
        "classic cars", "vintage cars", "motorcycles",
        "planes", "ships", "trains", "trucks and buses"
    ]
    for pl in known_lines:
        if pl in q:
            product_line = pl.title()
            break
    if not product_line and prev and prev.product_line:
        product_line = prev.product_line

    # Detect target dimension / entity
    is_month_query = bool(re.search(r"\b(months?|monthly)\b", q))
    is_country_query = bool(re.search(r"\b(country|countries|nations?|territor(y|ies))\b", q))
    is_product_line_query = bool(re.search(r"\b(product ?lines?|categor(y|ies))\b", q))
    is_customer_query = bool(re.search(r"\b(customers?|clients?|accounts?)\b", q))
    is_product_query = bool(re.search(r"\b(products?|models?|items?)\b", q))
    is_ranking = bool(re.search(r"\b(top|best|leading|highest|peak|lowest|worst|most)\b", q))
    is_single_best = bool(re.search(r"\b(which|what|peak|best|lowest|worst)\s+(month|country|product ?line|category|customer|product)\b", q)) or bool(re.search(r"\b(which|what)\s+(month|country|product ?line|category|customer|product)\s+(has|had|generat|produc|saw|is)\b", q))

    # Top N
    m_top = re.search(r"\btop\s+(\d+)\b", q)
    if m_top:
        n = min(50, max(1, int(m_top.group(1))))
    elif is_single_best:
        n = 1
    elif prev and prev.n:
        n = prev.n
    else:
        n = 10

    # Intent kind
    if re.search(r"\b(why|drop|decrease|decline|fell|down|root cause|investigat)\b", q):
        kind = "deep"
    elif re.search(r"\b(compare|compar|versus|vs|difference|bridge)\b", q):
        kind = "comparative"
    elif is_month_query and (is_ranking or "trend" in q or "over time" in q or "history" in q):
        kind = "trend"
    elif is_country_query and (is_ranking or "by" in q or "across" in q or "each" in q):
        kind = "by_country"
    elif is_product_line_query and (is_ranking or "by" in q or "across" in q or "each" in q):
        kind = "by_product_line"
    elif is_customer_query and (is_ranking or "by" in q or "across" in q):
        kind = "by_customer"
    elif is_product_query or (is_ranking and not is_month_query and not is_country_query and not is_product_line_query):
        kind = "top_products"
    elif re.search(r"\b(trend|over time|monthly|history)\b", q):
        kind = "trend"
    elif re.search(r"\b(by country|per country|across countries|geographic)\b", q):
        kind = "by_country"
    elif re.search(r"\b(by product line|by category|per line)\b", q):
        kind = "by_product_line"
    elif re.search(r"\b(total|revenue|sales|volume|payments)\b", q):
        kind = "total"
    elif prev:
        kind = prev.kind
    else:
        kind = "simple"

    return QuestionIntent(
        kind=kind,
        metric=metric,
        period=period,
        country=country,
        product_line=product_line,
        n=n,
        is_followup=bool(prev)
    )


def public_catalog() -> dict[str, Any]:
    """Expose available metrics, dimensions, and database range to the frontend."""
    return {
        "metrics": {
            k: {
                "label": v["label"],
                "unit": v["unit"],
                "description": v["description"]
            }
            for k, v in APPROVED_METRICS.items()
        },
        "dimensions": {
            k: {
                "label": v["label"],
                "description": v["description"]
            }
            for k, v in APPROVED_DIMENSIONS.items()
        },
        "data_range": {
            "min_date": DATA_MIN_DATE,
            "max_date": DATA_MAX_DATE,
            "default_comparison": "November 2004 vs October 2004"
        }
    }

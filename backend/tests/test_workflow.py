"""Unit tests for workflow logic, semantics, and reporter accounting reconciliation."""
import pytest
from app.services.semantics import resolve_period, extract_intent
from app.agents.reporter import ReporterAgent, format_currency, calculate_pct
from app.schemas import QuestionIntent, ModelUsageItem


def test_resolve_period_monthly():
    p = resolve_period("2004-11")
    assert p["target_label"] == "November 2004"
    assert p["previous_label"] == "October 2004"
    assert p["target_start"] == "2004-11-01"
    assert p["target_end"] == "2004-12-01"
    assert p["previous_start"] == "2004-10-01"
    assert p["previous_end"] == "2004-11-01"


def test_resolve_period_annual():
    p = resolve_period("2004")
    assert p["target_label"] == "2004"
    assert p["previous_label"] == "2003"
    assert p["target_start"] == "2004-01-01"
    assert p["target_end"] == "2005-01-01"


def test_extract_intent_simple_top_n():
    intent = extract_intent("Show top 5 products by revenue in 2004")
    assert intent.kind == "top_products"
    assert intent.metric == "gross_revenue"
    assert intent.n == 5
    assert intent.period == "2004"


def test_extract_intent_deep_investigation():
    intent = extract_intent("Why did sales decrease in November 2004 compared with October 2004?")
    assert intent.kind == "deep"
    assert intent.metric == "gross_revenue"
    assert intent.period == "2004-11"


def test_extract_intent_followup_preserves_context():
    prev = extract_intent("Show sales by country in 2004")
    followup = extract_intent("Now only show customers from USA", prev.model_dump())
    assert followup.country == "USA"
    assert followup.period == "2004"
    assert followup.is_followup is True


def test_reporter_math_helpers():
    assert format_currency(1234.56) == "$1,234.56"
    assert format_currency(-500.0) == "-$500.00"
    assert calculate_pct(100.0, 150.0) == 50.0
    assert calculate_pct(200.0, 100.0) == -50.0
    assert calculate_pct(0.0, 100.0) is None


def test_reporter_reconciles_baseline_variance():
    reporter = ReporterAgent()
    intent = QuestionIntent(kind="deep", metric="gross_revenue", period="2004-11")

    # Mock evidence matching baseline and product_line_bridge
    baseline_evidence = {
        "task_id": "baseline",
        "query_id": "q-base-1",
        "purpose": "Baseline comparison",
        "columns": ["previous_revenue", "current_revenue", "previous_orders", "current_orders"],
        "rows": [{"previous_revenue": 500000.0, "current_revenue": 300000.0, "previous_orders": 50, "current_orders": 30}],
        "sql": "SELECT ...",
        "duration_ms": 25
    }

    bridge_evidence = {
        "task_id": "product_line_bridge",
        "query_id": "q-bridge-1",
        "purpose": "Product line bridge",
        "columns": ["label", "previous", "current"],
        "rows": [
            {"label": "Classic Cars", "previous": 300000.0, "current": 180000.0},  # delta: -120000
            {"label": "Vintage Cars", "previous": 200000.0, "current": 120000.0},  # delta: -80000
        ],  # sum of changes: -200,000, exactly matches baseline change!
        "sql": "SELECT ...",
        "duration_ms": 30
    }

    report = reporter.create_report(
        intent=intent,
        evidence_list=[baseline_evidence, bridge_evidence],
        hypotheses_text=["Sales dropped due to Classic Cars"],
        task_errors=[],
        trace_events=[],
        models_used=[]
    )

    assert report.status == "complete"
    assert len(report.kpis) >= 4
    net_variance_kpi = next(k for k in report.kpis if k["label"] == "Net Variance")
    assert net_variance_kpi["value"] == -200000.0
    assert len(report.charts) >= 1
    assert len(report.hypotheses) >= 1
    assert report.hypotheses[0].status == "verified"

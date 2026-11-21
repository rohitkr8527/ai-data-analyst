"""Comprehensive test suite for the Dynamic Supervisor-Worker Orchestration Architecture.

Covers all 12 required verification scenarios:
1. Simple SQL questions without unnecessary agents.
2. Comparative analysis using SQL and Python.
3. Root-cause investigations with hypothesis generation.
4. Parallel independent investigations.
5. Evidence review and bounded refinement loop.
6. SQL error recovery (bounded repair).
7. Budget exhaustion and partial analysis marking.
8. Provider failures (OpenRouter Jev and OpenAI explicit error propagation).
9. Conversation follow-ups and session context preservation.
10. Business report generation (structured BusinessReport + HTML & PDF rendering).
11. Concurrent workflow isolation and durable LangGraph checkpoints.
12. Numerical reconciliation and evidence references.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock
import pandas as pd
import pytest

from app.agents.router import JevRoutingDecision, OpenRouterJevRouter, RouteType
from app.exceptions import PlanValidationError, ProviderAuthError, ProviderRateLimitError, QuerySecurityError
from app.orchestration.contracts import (
    BudgetSpec,
    ChartRequest,
    FilterSpec,
    HypothesisDraft,
    HypothesisSet,
    HypothesisSpec,
    OutputRequirements,
    PlanDraft,
    PythonToolSpec,
    QuerySpec,
    RefinementRequest,
    ReportDraft,
    ReportSection,
    ReviewDraft,
    TaskDraft,
)
from app.orchestration.validator import PlanValidator, detect_cycle, topological_order
from app.orchestration.workflow import AnalyticalWorkflow
from app.schemas import ReportNarrative, SQLDraft
from app.services.analysis_tools import run_registered_analysis_tool
from app.services.dataset_store import DatasetStore
from app.services.db import ReadOnlyGateway
from app.services.models import ModelSelector, ModelUsage
from app.services.store import SessionStore


def _build_mock_gateway() -> ReadOnlyGateway:
    """Create a ReadOnlyGateway with deterministic MySQL responses for isolated orchestration tests."""
    gw = MagicMock(spec=ReadOnlyGateway)
    gw.catalog = MagicMock()
    gw.catalog.format_llm_schema_context.return_value = "### ClassicModels Schema"
    snap = MagicMock()
    snap.tables = {"orders": MagicMock(), "orderdetails": MagicMock(), "products": MagicMock(), "customers": MagicMock()}
    gw.catalog.get_snapshot.return_value = snap

    def fake_execute(sql: str, params: dict[str, Any] | None = None, trusted_template: bool = False, **kwargs):
        s_low = sql.lower()
        if "select *" in s_low or "unknown_col" in s_low:
            raise QuerySecurityError("Unknown column 'unknown_col' in field list")
        if "previous_revenue" in s_low and "current_revenue" in s_low:
            return {
                "id": "qid_baseline",
                "query_hash": "hash_base",
                "sql": sql,
                "columns": ["previous_revenue", "current_revenue", "previous_orders", "current_orders"],
                "rows": [
                    {
                        "previous_revenue": 500000.0,
                        "current_revenue": 300000.0,
                        "previous_orders": 50,
                        "current_orders": 30,
                    }
                ],
                "duration_ms": 12,
            }
        if "p.productline" in s_low and "previous" in s_low:
            return {
                "id": "qid_pl_bridge",
                "query_hash": "hash_pl",
                "sql": sql,
                "columns": ["label", "previous", "current"],
                "rows": [
                    {"label": "Classic Cars", "previous": 300000.0, "current": 180000.0},
                    {"label": "Vintage Cars", "previous": 200000.0, "current": 120000.0},
                ],
                "duration_ms": 15,
            }
        if "c.country" in s_low and "previous" in s_low:
            return {
                "id": "qid_country_bridge",
                "query_hash": "hash_ctry",
                "sql": sql,
                "columns": ["label", "previous", "current"],
                "rows": [
                    {"label": "USA", "previous": 280000.0, "current": 160000.0},
                    {"label": "France", "previous": 220000.0, "current": 140000.0},
                ],
                "duration_ms": 14,
            }
        if "p.productname" in s_low and "previous" in s_low:
            return {
                "id": "qid_prod_bridge",
                "query_hash": "hash_pb",
                "sql": sql,
                "columns": ["label", "category", "previous", "current"],
                "rows": [
                    {"label": "1968 Ford Mustang", "category": "Classic Cars", "previous": 80000.0, "current": 30000.0},
                    {"label": "1952 Alpine Renault", "category": "Classic Cars", "previous": 70000.0, "current": 40000.0},
                ],
                "duration_ms": 16,
            }
        if "c.customername" in s_low and "previous" in s_low:
            return {
                "id": "qid_cust_bridge",
                "query_hash": "hash_cb",
                "sql": sql,
                "columns": ["label", "country", "previous", "current"],
                "rows": [
                    {"label": "Mini Gifts Distributors", "country": "USA", "previous": 120000.0, "current": 50000.0},
                ],
                "duration_ms": 13,
            }
        if "o.status" in s_low:
            return {
                "id": "qid_status",
                "query_hash": "hash_st",
                "sql": sql,
                "columns": ["label", "order_count", "total_value"],
                "rows": [
                    {"label": "Shipped", "order_count": 28, "total_value": 280000.0},
                    {"label": "Cancelled", "order_count": 2, "total_value": 25000.0},
                ],
                "duration_ms": 9,
            }
        if "date_format" in s_low:
            return {
                "id": "qid_trend",
                "query_hash": "hash_tr",
                "sql": sql,
                "columns": ["month", "value"],
                "rows": [
                    {"month": "2004-09", "value": 320000.0},
                    {"month": "2004-10", "value": 500000.0},
                    {"month": "2004-11", "value": 300000.0},
                    {"month": "2004-12", "value": 340000.0},
                ],
                "duration_ms": 11,
            }
        if "e.firstname" in s_low or "salesrep" in s_low:
            return {
                "id": "qid_sales_rep",
                "query_hash": "hash_sr",
                "sql": sql,
                "columns": ["label", "previous", "current"],
                "rows": [
                    {"label": "Gerard Hernandez", "previous": 150000.0, "current": 90000.0},
                ],
                "duration_ms": 10,
            }
        return {
            "id": "qid_lookup",
            "query_hash": "hash_lkp",
            "sql": sql,
            "columns": ["product", "line", "value"],
            "rows": [
                {"product": "1992 Ferrari 360 Spider red", "line": "Classic Cars", "value": 276839.98},
                {"product": "2001 Ferrari Enzo", "line": "Classic Cars", "value": 190755.86},
            ],
            "duration_ms": 10,
        }

    gw.execute.side_effect = fake_execute
    return gw


def _build_workflow_fixture(
    tmp_path: Path,
    route_choice: RouteType = RouteType.SIMPLE_SQL,
    custom_plan_draft: PlanDraft | None = None,
    review_decisions: list[ReviewDraft] | None = None,
    first_sql_bad: bool = False,
) -> tuple[AnalyticalWorkflow, MagicMock, MagicMock]:
    gw = _build_mock_gateway()
    selector = MagicMock(spec=ModelSelector)
    ds_store = DatasetStore(path=str(tmp_path / "datasets.db"))
    cp_path = str(tmp_path / "checkpoints.db")

    sql_call_counter = {"n": 0}
    review_idx = {"n": 0}

    def fake_parse(task_type, response_model, system_instructions, input_payload, custom_effort=None):
        usage = ModelUsage(
            task_type=task_type.value,
            model="gpt-6.1-sol" if "plan" in task_type.value or "review" in task_type.value or "hypothesis" in task_type.value else "gpt-6-luna",
            reasoning_effort="medium",
            duration_ms=15,
            input_tokens=80,
            output_tokens=40,
            cost_usd=0.0002,
        )
        if response_model is PlanDraft:
            if custom_plan_draft is not None:
                return custom_plan_draft, usage
            return (
                PlanDraft(
                    objective=str(input_payload.get("question", "Analyze ClassicModels")),
                    complexity="lookup",
                    metrics=["gross_revenue"],
                    dimensions=["product"],
                    period_key="2004",
                    filters=[],
                    tasks=[
                        TaskDraft(
                            id="top_products",
                            title="Top Products by Revenue",
                            agent="sql",
                            purpose="Fetch top products by revenue",
                            depends_on=[],
                            priority=1,
                            query_spec=QuerySpec(metric="gross_revenue", dimensions=["product"], limit=10),
                        )
                    ],
                    hypotheses_required=False,
                    output=OutputRequirements(
                        format="answer_with_visuals",
                        charts=[ChartRequest(task_id="top_products", preferred_kind="bar", title="Top Products")],
                    ),
                    budgets=BudgetSpec(max_queries=4, max_refinements=1, max_seconds=60),
                ),
                usage,
            )
        if response_model is SQLDraft:
            sql_call_counter["n"] += 1
            if first_sql_bad and sql_call_counter["n"] == 1:
                return SQLDraft(sql="SELECT unknown_col FROM orders", reason="Intentional bad column"), usage
            ref_sql = input_payload.get("reference_sql") or (
                "SELECT p.productName AS product, p.productLine AS line, ROUND(SUM(od.quantityOrdered * od.priceEach), 2) AS value "
                "FROM orders o JOIN orderdetails od ON o.orderNumber = od.orderNumber JOIN products p ON od.productCode = p.productCode "
                "WHERE o.status != 'Cancelled' GROUP BY p.productCode, p.productName, p.productLine ORDER BY value DESC LIMIT :limit_n"
            )
            return SQLDraft(sql=ref_sql, reason="Validated SQL"), usage
        if response_model is HypothesisSet:
            return (
                HypothesisSet(
                    hypotheses=[
                        HypothesisDraft(
                            id="hyp_classic_cars",
                            statement="Classic Cars contraction accounted for the majority of the revenue decline.",
                            hypothesis_type="accounting_contribution",
                            dimension="product_line",
                            target_entity="Classic Cars",
                            expected_direction="decrease",
                            evidence_required="Compare Classic Cars revenue across periods.",
                            priority=1,
                            alternative_explanations=["Geographic overlap with USA market"],
                        ),
                        HypothesisDraft(
                            id="hyp_order_volume",
                            statement="Reduced order count rather than lower AOV drove the revenue drop.",
                            hypothesis_type="observed_pattern",
                            dimension="order_volume",
                            expected_direction="decrease",
                            evidence_required="Decompose revenue change into order volume vs AOV.",
                            priority=1,
                            alternative_explanations=["Order timing shifts"],
                        ),
                    ],
                    reasoning="Formulated from baseline and product line observations.",
                ),
                usage,
            )
        if response_model is ReviewDraft:
            if review_decisions and review_idx["n"] < len(review_decisions):
                dec = review_decisions[review_idx["n"]]
                review_idx["n"] += 1
                return dec, usage
            return ReviewDraft(sufficient=True, confidence=0.95, assessment="All evidence reconciled."), usage
        if response_model is ReportNarrative:
            return (
                ReportNarrative(
                    executive_summary="",
                    key_findings_summary=["Verified revenue variance across dimensions."],
                    recommended_actions=["Review Classic Cars inventory and US customer accounts."],
                    data_limitations=["Cross-dimensional contributions overlap."],
                    follow_up_questions=["Which sales representatives had the largest variance?"],
                ),
                usage,
            )
        if response_model is ReportDraft:
            return (
                ReportDraft(
                    title="Executive Business Report: ClassicModels Revenue Analysis",
                    executive_summary="Gross revenue declined by -$200,000.00 (-40.0%) driven by Classic Cars and order volume contraction.",
                    business_objective="Evaluate period-over-period revenue variance and identify primary drivers.",
                    kpi_overview="Previous: $500,000.00 | Current: $300,000.00 | Net Variance: -$200,000.00 (-40.0%)",
                    sections=[
                        ReportSection(
                            heading="1. Period-over-Period Revenue & Volume Summary",
                            body="Completed order volume fell from 50 to 30 orders while AOV remained steady at $10,000.00.",
                            evidence_ids=["qid_baseline"],
                        ),
                        ReportSection(
                            heading="2. Product Line & Geographic Attribution",
                            body="Classic Cars declined by -$120,000.00 (60.0% of net change) and Vintage Cars declined by -$80,000.00 (40.0%).",
                            evidence_ids=["qid_pl_bridge", "qid_country_bridge"],
                        ),
                    ],
                    recommendations=["Audit Classic Cars account orders in the USA and France."],
                    limitations=["Accounting contributions overlap across dimensions."],
                ),
                usage,
            )
        raise ValueError(f"Unhandled response_model in mock: {response_model}")

    selector.parse_structured.side_effect = fake_parse

    wf = AnalyticalWorkflow(
        gateway=gw,
        model_selector=selector,
        dataset_store=ds_store,
        checkpoint_path=cp_path,
    )
    wf.router = MagicMock(spec=OpenRouterJevRouter)
    wf.router.route.return_value = JevRoutingDecision(
        route=route_choice,
        confidence=0.96,
        probabilities={route_choice.value: 0.96},
        duration_ms=25,
        input_tokens=45,
        output_tokens=10,
    )
    return wf, gw, selector


# ----------------------------------------------------------------------
# Scenario 1: Simple SQL questions without unnecessary agents
# ----------------------------------------------------------------------
def test_scenario_1_simple_sql_no_unnecessary_agents(tmp_path):
    wf, gw, _ = _build_workflow_fixture(tmp_path, route_choice=RouteType.SIMPLE_SQL)
    res = wf.run("Show the top 10 products by revenue in 2004", session_id="s1")

    report = res["report"]
    assert report["status"] == "complete"
    assert len(report["evidence"]) == 1
    assert len(report["charts"]) == 1

    # Verify no Hypothesis Agent or Evidence Reviewer LLM call was made for simple lookup
    roles = [m["agent_role"] for m in report["models_used"]]
    assert "Hypothesis Agent" not in roles
    assert "Evidence Reviewer" not in roles
    assert "Report Generation Agent" not in roles
    assert gw.execute.call_count == 1


# ----------------------------------------------------------------------
# Scenario 2: Comparative analysis using SQL and Python
# ----------------------------------------------------------------------
def test_scenario_2_comparative_analysis_sql_and_python(tmp_path):
    comp_draft = PlanDraft(
        objective="Compare 2004 vs 2003 revenue by product line",
        complexity="comparison",
        metrics=["gross_revenue"],
        dimensions=["product_line"],
        period_key="2004",
        filters=[],
        tasks=[
            TaskDraft(
                id="baseline",
                title="Baseline Comparison",
                agent="sql",
                purpose="Establish baseline",
                depends_on=[],
                priority=1,
                query_spec=QuerySpec(metric="gross_revenue", dimensions=[], comparison="previous_period"),
            ),
            TaskDraft(
                id="product_line_bridge",
                title="Product Line Comparison",
                agent="sql",
                purpose="Compare product lines",
                depends_on=[],
                priority=1,
                query_spec=QuerySpec(metric="gross_revenue", dimensions=["product_line"], comparison="previous_period"),
            ),
            TaskDraft(
                id="compare_periods",
                title="Python Period Comparison",
                agent="python",
                purpose="Compute deltas and shares",
                depends_on=["product_line_bridge", "baseline"],
                priority=2,
                python_spec=PythonToolSpec(tool="contribution_analysis", input_task_ids=["product_line_bridge", "baseline"]),
            ),
        ],
        hypotheses_required=False,
        output=OutputRequirements(format="answer_with_visuals"),
        budgets=BudgetSpec(max_queries=5, max_refinements=1, max_seconds=60),
    )

    wf, _, _ = _build_workflow_fixture(tmp_path, route_choice=RouteType.COMPARATIVE_ANALYSIS, custom_plan_draft=comp_draft)
    res = wf.run("Compare 2004 vs 2003 revenue by product line", session_id="s2")
    report = res["report"]

    assert report["status"] == "complete"
    assert len(report["python_analyses"]) >= 1
    contrib = next(p for p in report["python_analyses"] if p["tool"] in ("contribution_analysis", "period_comparison"))
    assert contrib["calculation"]["summary"]["reconciled"] is True


# ----------------------------------------------------------------------
# Scenario 3 & 4: Root-cause investigation with hypotheses & parallel tasks
# ----------------------------------------------------------------------
def test_scenario_3_and_4_root_cause_hypotheses_and_parallel_execution(tmp_path):
    rca_draft = PlanDraft(
        objective="Why did revenue decrease in November 2004?",
        complexity="investigation",
        metrics=["gross_revenue"],
        dimensions=["product_line", "country", "product", "customer"],
        period_key="2004-11",
        filters=[],
        tasks=[
            TaskDraft(
                id="baseline",
                title="Baseline",
                agent="sql",
                purpose="Baseline revenue",
                depends_on=[],
                priority=1,
                query_spec=QuerySpec(metric="gross_revenue", dimensions=[], comparison="previous_period"),
            ),
            TaskDraft(
                id="product_line_bridge",
                title="Product Line Bridge",
                agent="sql",
                purpose="Product lines",
                depends_on=[],
                priority=1,
                query_spec=QuerySpec(metric="gross_revenue", dimensions=["product_line"], comparison="previous_period"),
            ),
            TaskDraft(
                id="country_bridge",
                title="Country Bridge",
                agent="sql",
                purpose="Countries",
                depends_on=[],
                priority=2,
                query_spec=QuerySpec(metric="gross_revenue", dimensions=["country"], comparison="previous_period"),
            ),
        ],
        hypotheses_required=True,
        output=OutputRequirements(format="answer_with_visuals"),
        budgets=BudgetSpec(max_queries=10, max_refinements=1, max_seconds=120),
    )

    wf, _, _ = _build_workflow_fixture(tmp_path, route_choice=RouteType.DEEP_INVESTIGATION, custom_plan_draft=rca_draft)
    res = wf.run("Why did revenue decrease in November 2004?", session_id="s3")
    report = res["report"]

    assert report["status"] == "complete"
    # Verify Hypothesis Agent ran and hypotheses were evaluated
    assert len(report["hypotheses"]) >= 2
    assert any(h["status"] in ("verified", "supported") for h in report["hypotheses"])

    # Verify Python revenue decomposition and contribution analysis ran after SQL dependencies
    py_tools = {p["tool"] for p in report["python_analyses"]}
    assert "revenue_decomposition" in py_tools
    assert "contribution_analysis" in py_tools

    # Verify parallel dispatch occurred in supervisor trace
    sup_traces = [t["message"] for t in report["trace"] if t["stage"] == "supervisor"]
    assert any("in parallel" in m for m in sup_traces)


# ----------------------------------------------------------------------
# Scenario 5: Evidence review and bounded refinement loop
# ----------------------------------------------------------------------
def test_scenario_5_evidence_review_triggers_targeted_refinement(tmp_path):
    rca_draft = PlanDraft(
        objective="Why did revenue decrease in November 2004?",
        complexity="investigation",
        metrics=["gross_revenue"],
        dimensions=["product_line"],
        period_key="2004-11",
        filters=[],
        tasks=[
            TaskDraft(
                id="baseline",
                title="Baseline",
                agent="sql",
                purpose="Baseline",
                depends_on=[],
                priority=1,
                query_spec=QuerySpec(metric="gross_revenue", dimensions=[], comparison="previous_period"),
            ),
        ],
        hypotheses_required=True,
        output=OutputRequirements(format="answer_with_visuals"),
        budgets=BudgetSpec(max_queries=12, max_refinements=2, max_seconds=120),
    )

    # Round 0 review requests a sales_rep breakdown; Round 1 review approves sufficiency
    reviews = [
        ReviewDraft(
            sufficient=False,
            confidence=0.6,
            assessment="Need sales representative breakdown to verify account rep impact.",
            refinement_requests=[
                RefinementRequest(
                    reason="Investigate sales representative variance in November 2004",
                    kind="additional_breakdown",
                    gap="sales_rep",
                    priority=1,
                )
            ],
        ),
        ReviewDraft(
            sufficient=True,
            confidence=0.95,
            assessment="Sales rep breakdown collected; evidence is now complete.",
            refinement_requests=[],
        ),
    ]

    wf, _, _ = _build_workflow_fixture(
        tmp_path,
        route_choice=RouteType.DEEP_INVESTIGATION,
        custom_plan_draft=rca_draft,
        review_decisions=reviews,
    )
    res = wf.run("Why did revenue decrease in November 2004?", session_id="s5")
    report = res["report"]

    assert report["status"] == "complete"
    assert report["evidence_review"]["round"] == 1
    assert report["evidence_review"]["decision"] == "sufficient"
    refinement_tasks = [t for t in report["task_graph"] if t["origin"] == "refinement"]
    assert len(refinement_tasks) == 1
    assert refinement_tasks[0]["status"] == "COMPLETED"


# ----------------------------------------------------------------------
# Scenario 6: SQL error recovery (bounded repair)
# ----------------------------------------------------------------------
def test_scenario_6_sql_error_recovery_via_bounded_repair(tmp_path):
    wf, _, _ = _build_workflow_fixture(tmp_path, route_choice=RouteType.SIMPLE_SQL, first_sql_bad=True)
    res = wf.run("Show the top 10 products by revenue in 2004", session_id="s6")
    report = res["report"]

    assert report["status"] == "complete"
    assert len(report["evidence"]) == 1
    roles = [m["agent_role"] for m in report["models_used"]]
    assert "SQL Agent (Repair)" in roles


# ----------------------------------------------------------------------
# Scenario 7: Budget exhaustion returns partial analysis with limitations
# ----------------------------------------------------------------------
def test_scenario_7_budget_exhaustion_marks_partial_analysis(tmp_path):
    tight_draft = PlanDraft(
        objective="Why did revenue decrease in November 2004?",
        complexity="investigation",
        metrics=["gross_revenue"],
        dimensions=["product_line", "country"],
        period_key="2004-11",
        filters=[],
        tasks=[
            TaskDraft(
                id="baseline",
                title="Baseline",
                agent="sql",
                purpose="Baseline",
                depends_on=[],
                priority=1,
                query_spec=QuerySpec(metric="gross_revenue", dimensions=[], comparison="previous_period"),
            ),
        ],
        hypotheses_required=True,
        output=OutputRequirements(format="answer_with_visuals"),
        budgets=BudgetSpec(max_queries=2, max_refinements=1, max_seconds=60),
    )

    reviews = [
        ReviewDraft(
            sufficient=False,
            confidence=0.5,
            assessment="Additional sales rep query required, but query budget is exhausted.",
            refinement_requests=[
                RefinementRequest(
                    reason="Need sales rep breakdown",
                    kind="additional_breakdown",
                    gap="sales_rep",
                    priority=1,
                )
            ],
        )
    ]

    wf, _, _ = _build_workflow_fixture(
        tmp_path,
        route_choice=RouteType.DEEP_INVESTIGATION,
        custom_plan_draft=tight_draft,
        review_decisions=reviews,
    )
    res = wf.run("Why did revenue decrease in November 2004?", session_id="s7")
    report = res["report"]

    assert report["status"] == "partial"
    assert report["evidence_review"]["decision"] == "stop_budget"
    assert any("budget" in lim.lower() for lim in report["limitations"])


# ----------------------------------------------------------------------
# Scenario 8: Provider failures propagate explicitly without silent fallback
# ----------------------------------------------------------------------
def test_scenario_8_provider_failures_propagate_explicitly(tmp_path):
    wf, _, selector = _build_workflow_fixture(tmp_path)
    selector.parse_structured.side_effect = ProviderRateLimitError("PROVIDER_RATE_LIMITED: Rate limit reached.")

    with pytest.raises(ProviderRateLimitError):
        wf.run("Show top 10 products by revenue")

    wf.router.route.side_effect = ProviderAuthError("PROVIDER_AUTH_FAILED: Invalid OpenRouter key.")
    with pytest.raises(ProviderAuthError):
        wf.run("Show top 10 products by revenue")


# ----------------------------------------------------------------------
# Scenario 9: Conversation follow-ups preserve session context
# ----------------------------------------------------------------------
def test_scenario_9_conversation_followups_preserve_context(tmp_path):
    store = SessionStore(path=str(tmp_path / "sessions.db"))
    wf, _, _ = _build_workflow_fixture(tmp_path, route_choice=RouteType.SIMPLE_SQL)

    sid = store.create_session()
    ans1 = wf.run("Show sales by country in 2004", session_id=sid)
    store.save(
        sid,
        "Show sales by country in 2004",
        {"run_id": ans1["run_id"], "session_id": sid, "route": ans1["route"], "report": ans1["report"], "intent": ans1["intent"]},
    )

    sess = store.get(sid)
    assert sess is not None
    assert sess["context"]["last_period"] == "2004"

    ans2 = wf.run("Now only show customers from USA", previous_context=sess["context"], session_id=sid)
    assert ans2["intent"]["country"] == "USA"
    assert ans2["intent"]["period"] == "2004"
    assert ans2["intent"]["is_followup"] is True


# ----------------------------------------------------------------------
# Scenario 10: Business report generation (HTML and PDF export)
# ----------------------------------------------------------------------
def test_scenario_10_business_report_generation_html_and_pdf(tmp_path):
    from app.agents.report_agent import render_report_html, render_report_pdf

    wf, _, _ = _build_workflow_fixture(tmp_path, route_choice=RouteType.DEEP_INVESTIGATION)
    res = wf.run("Generate a comprehensive executive business report on why revenue declined in November 2004", session_id="s10")
    report = res["report"]

    assert report["output_mode"] == "business_report"
    assert report["business_report"] is not None
    assert len(report["business_report"]["sections"]) >= 1

    html_doc = render_report_html(report, run_id=res["run_id"])
    assert html_doc.startswith("<!DOCTYPE html>")
    assert "Executive Business Report" in html_doc
    assert "Executive Analytical Synthesis Flow" not in html_doc
    assert "sandbox code provenance" not in html_doc

    pdf_bytes = render_report_pdf(report, run_id=res["run_id"])
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_report_curation_formatting_and_sanitization():
    from app.agents.report_agent import _format_cell_value, _prepare_report_view_data, render_report_html, render_report_pdf

    order_row = {"comparison": "MoM", "metric": "total_orders", "current": 33.0, "baseline": 13.0, "absolute_change": 20.0, "percentage_change": 153.85}
    rev_row = {"comparison": "MoM", "metric": "gross_revenue", "current": 979291.98, "baseline": 500233.86, "absolute_change": 479058.12, "percentage_change": 95.77}
    aov_row = {"comparison": "MoM", "metric": "aov", "current": 29675.51, "baseline": 38479.53, "absolute_change": -8804.01, "percentage_change": -22.88}

    assert _format_cell_value(order_row["metric"], "metric", order_row) == "Total Orders"
    assert _format_cell_value(order_row["current"], "current", order_row) == "33"
    assert _format_cell_value(order_row["baseline"], "baseline", order_row) == "13"
    assert _format_cell_value(order_row["absolute_change"], "absolute_change", order_row) == "+20"
    assert _format_cell_value(rev_row["baseline"], "baseline", rev_row) == "$500,233.86"
    assert _format_cell_value(aov_row["metric"], "metric", aov_row) == "AOV"

    noisy_report = {
        "title": "ClassicModels Executive Report: November 2004 Gross Revenue Performance",
        "summary": "November 2004 gross revenue was $979,291.98, up 95.8% from October.",
        "status": "partial",
        "period": "November 2004",
        "comparison": "October 2004",
        "kpis": [{"label": "Gross merchandise revenue (November 2004)", "value": 979291.98, "unit": "USD"}],
        "findings": [
            {"title": "Total Gross merchandise revenue", "detail": "Measured $979,291.98 in November 2004.", "status": "verified"},
            {"title": "Autonomous Python Analysis: Cross-source reconciliation", "detail": "november_revenue=979291.98, comparisons=[{'comparison': 'MoM', 'current': 979291.98}]", "status": "verified"},
        ],
        "tables": [
            {
                "title": "Query Output",
                "rows": [
                    {"period_label": "November 2004", "period_start": "2004-11-01", "period_end_exclusive": "2004-12-01", "gross_revenue": 979291.98, "total_orders": 33, "aov": 29675.51},
                    {"period_label": "October 2004", "period_start": "2004-10-01", "period_end_exclusive": "2004-11-01", "gross_revenue": 500233.86, "total_orders": 13, "aov": 38479.53},
                ],
            },
            {
                "title": "Authoritative period KPIs",
                "rows": [
                    {"period_label": "November 2004", "gross_revenue": 979291.98, "total_orders": 33, "aov": 29675.51, "purchasing_customers": 30},
                    {"period_label": "October 2004", "gross_revenue": 500233.86, "total_orders": 13, "aov": 38479.53, "purchasing_customers": 13},
                ],
            },
            {
                "title": "Cross-source reconciliation",
                "rows": [
                    {"period": "November 2004", "check": "monthly gross_revenue", "difference": 0.0},
                    {"period": "October 2004", "check": "monthly gross_revenue", "difference": 0.0},
                ],
            },
        ],
        "business_report": {
            "title": "ClassicModels Executive Report: November 2004 Gross Revenue Performance",
            "executive_summary": "November 2004 gross revenue was $979,291.98, up 95.8% from October.",
            "sections": [
                {"heading": "1. Trends and Period-over-Period Comparisons", "body": "October 2004 revenue was $500,233.86 across 13 orders."},
                {"heading": "2. Primary Findings & Dimensional Drivers", "body": "Order volume contributed $681,550.42 while AOV offset -$202,492.30."},
                {"heading": "3. Root-Cause & Hypothesis Evaluation", "body": "Lower AOV is an arithmetic offset to higher order volume."},
                {"heading": "4. Methodology, Data Scope & Accounting Rules", "body": "Gross revenue is ordered quantity multiplied by price."},
            ],
            "recommendations": ["Investigate lower average order value using order and product mix."],
            "limitations": [],
        },
        "limitations": [
            "comparative_commercial_breakdowns: Only SELECT statements are permitted.",
            "commercial_breakdowns_select_only: Query result exceeded maximum row limit (500). Add a LIMIT clause or grouping aggregation.",
            "Refinement stopped because max_refinements (2) was reached.",
            "Run exceeded wall-clock budget (240s)",
            "The revenue bridge is an arithmetic allocation and does not establish causation.",
        ],
    }

    view = _prepare_report_view_data(noisy_report)
    assert len(view["kpis"]) == 4
    assert len(view["sections"]) == 3
    assert view["findings"] == []
    assert [t["title"] for t in view["tables"]] == ["Authoritative period KPIs"]
    assert all("Only SELECT" not in lim and "max_refinements" not in lim and "wall-clock budget" not in lim for lim in view["limitations"])

    html_out = render_report_html(noisy_report)
    assert "comparisons=[{" not in html_out
    assert "Only SELECT statements" not in html_out
    assert "Cross-source reconciliation" not in html_out

    pdf_out = render_report_pdf(noisy_report)
    assert pdf_out.startswith(b"%PDF")


def test_token_efficiency_fast_paths_and_payload_truncation(tmp_path):
    from app.agents.evidence_reviewer import EvidenceReviewerAgent
    from app.agents.planner import PlannerAgent
    from app.agents.sql_analyst import SQLAnalyst
    from app.orchestration.contracts import AnalysisTask, TaskResult
    from app.schemas import QuestionIntent
    from app.services.dataset_broker import summarize_dataframe

    # 1. DataFrame summary truncates huge string cells and caps sample rows to 3
    huge_df = pd.DataFrame([{"id": i, "giant_json": "x" * 10000} for i in range(10)])
    prof = summarize_dataframe(huge_df, "huge")
    assert len(prof["sample_rows"]) == 3
    assert len(prof["sample_rows"][0]["giant_json"]) < 150
    assert prof["sample_rows"][0]["giant_json"].endswith("...[truncated]")

    # 2. Planner fast-path for simple lookups uses 0 LLM tokens
    live_selector = ModelSelector()
    planner = PlannerAgent(live_selector)
    intent = QuestionIntent(kind="top_products", metric="gross_revenue", period="2004", n=10)
    draft, plan_usage = planner.create_plan_draft(
        question="Show the top 10 products by revenue in 2004",
        intent=intent,
        route_decision={"route": "SIMPLE_SQL"},
        schema_context="### ClassicModels",
    )
    assert plan_usage.tokens == 0
    assert len(draft.tasks) == 1

    # 3. Structured SQL compiles deterministically in 0 LLM tokens
    sql_analyst = SQLAnalyst(live_selector)
    task = AnalysisTask.model_validate(draft.tasks[0].model_dump())
    sql, params, source, sql_usage = sql_analyst.draft_for_task(task, "2004", [], "### ClassicModels")
    assert source == "compiled"
    assert sql_usage is None
    assert "SELECT" in sql

    # 4. 3-Layer Evidence Reviewer skips redundant LLM call when deterministic checks pass
    ds_store = DatasetStore(str(tmp_path / "fast_ds.db"))
    ds_ref = ds_store.put("r1", "baseline", ["previous_revenue", "current_revenue", "previous_orders", "current_orders"], [{"previous_revenue": 100.0, "current_revenue": 150.0, "previous_orders": 2, "current_orders": 3}])
    reviewer = EvidenceReviewerAgent(ds_store, live_selector)
    plan, _ = PlanValidator().validate_and_normalize("p1", "Why did sales change?", draft.model_copy(update={"complexity": "investigation"}))
    t_res = {"baseline": TaskResult(task_id="baseline", agent="sql", status="COMPLETED", dataset=ds_ref, query_id="q1").model_dump()}
    ev_records, hyps, ds_map, delta = reviewer.aggregate(plan, t_res, [])
    rev, rev_usage = reviewer.review(plan, ev_records, hyps, ds_map, delta, [], 0, {"queries_executed": 1})
    assert rev.sufficient is True
    assert rev.ai_reviewed is False
    assert rev_usage is None


# ----------------------------------------------------------------------
# Scenario 11: Concurrent workflow isolation and checkpoint persistence
# ----------------------------------------------------------------------
def test_scenario_11_concurrent_workflow_isolation_and_checkpoints(tmp_path):
    wf, _, _ = _build_workflow_fixture(tmp_path, route_choice=RouteType.SIMPLE_SQL)

    def run_session(idx: int):
        sid = f"concurrent_session_{idx}"
        rid = f"run_{idx}"
        out = wf.run(f"Show top {idx + 3} products by revenue in 2004", session_id=sid, run_id=rid)
        return sid, rid, out

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run_session, range(3)))

    for idx, (sid, rid, out) in enumerate(results):
        assert out["run_id"] == rid
        assert out["thread_id"] == f"{sid}:{rid}"
        assert out["report"]["status"] == "complete"
        # Verify durable checkpoints were persisted for each isolated thread
        cp_hist = wf.get_checkpoint_history(out["thread_id"])
        assert len(cp_hist) >= 2


# ----------------------------------------------------------------------
# Scenario 12: Numerical reconciliation, DAG validation, and Python tools
# ----------------------------------------------------------------------
def test_scenario_12_plan_validator_rejects_cycles_and_unapproved_tools():
    validator = PlanValidator()
    cyclic_draft = PlanDraft(
        objective="Cyclic test",
        complexity="lookup",
        metrics=["gross_revenue"],
        dimensions=[],
        tasks=[
            TaskDraft(
                id="task_a",
                title="Task A",
                agent="sql",
                purpose="A",
                depends_on=["task_b"],
                query_spec=QuerySpec(metric="gross_revenue"),
            ),
            TaskDraft(
                id="task_b",
                title="Task B",
                agent="sql",
                purpose="B",
                depends_on=["task_a"],
                query_spec=QuerySpec(metric="gross_revenue"),
            ),
        ],
        output=OutputRequirements(format="direct_answer"),
        budgets=BudgetSpec(),
    )
    assert detect_cycle(cyclic_draft.tasks) is not None
    with pytest.raises(PlanValidationError):
        validator.validate_and_normalize("plan_cyc", "Cyclic question", cyclic_draft)


def test_scenario_12_all_approved_python_analysis_tools():
    baseline_df = pd.DataFrame(
        [{"previous_revenue": 500000.0, "current_revenue": 300000.0, "previous_orders": 50, "current_orders": 30}]
    )
    bridge_df = pd.DataFrame(
        [
            {"label": "Classic Cars", "previous": 300000.0, "current": 180000.0},
            {"label": "Vintage Cars", "previous": 200000.0, "current": 120000.0},
        ]
    )
    trend_df = pd.DataFrame(
        [
            {"month": "2004-06", "value": 310000.0, "orders": 31},
            {"month": "2004-07", "value": 325000.0, "orders": 33},
            {"month": "2004-08", "value": 419000.0, "orders": 42},
            {"month": "2004-09", "value": 383000.0, "orders": 38},
            {"month": "2004-10", "value": 552000.0, "orders": 55},
            {"month": "2004-11", "value": 979000.0, "orders": 98},
            {"month": "2004-12", "value": 428000.0, "orders": 43},
        ]
    )

    decomp = run_registered_analysis_tool("revenue_decomposition", [baseline_df])
    assert decomp["summary"]["reconciled"] is True
    assert decomp["summary"]["net_revenue_change"] == -200000.0
    assert decomp["summary"]["volume_effect_usd"] == -200000.0

    contrib = run_registered_analysis_tool("contribution_analysis", [bridge_df, baseline_df])
    assert contrib["summary"]["reconciled"] is True
    assert contrib["summary"]["primary_contributor_label"] == "Classic Cars"
    assert contrib["summary"]["primary_contributor_share_pct"] == 60.0

    pcomp = run_registered_analysis_tool("period_comparison", [bridge_df])
    assert pcomp["summary"]["net_change"] == -200000.0

    stats_res = run_registered_analysis_tool("statistical_summary", [trend_df])
    assert stats_res["summary"]["count"] == 7

    trend_res = run_registered_analysis_tool("trend_analysis", [trend_df])
    assert trend_res["summary"]["peak_period"] == "2004-11"

    anom_res = run_registered_analysis_tool("anomaly_detection", [trend_df], {"threshold": 1.8})
    assert anom_res["summary"]["anomaly_count"] >= 1

    corr_res = run_registered_analysis_tool("correlation_analysis", [trend_df])
    assert corr_res["summary"]["pearson_r"] > 0.99

    dist_res = run_registered_analysis_tool("distribution_analysis", [trend_df])
    assert dist_res["summary"]["entity_count"] == 7

    dq_res = run_registered_analysis_tool("data_quality_check", [trend_df])
    assert dq_res["summary"]["quality_ok"] is True

    fc_res = run_registered_analysis_tool("forecast_linear", [trend_df], {"horizon": 2})
    assert len(fc_res["forecasts"]) == 2


def test_live_mysql_dynamic_orchestration_november_2004_reconciliation(tmp_path):
    """Execute the full dynamic DAG against the live MySQL ClassicModels database."""
    live_gw = ReadOnlyGateway()
    wf, _, _ = _build_workflow_fixture(tmp_path, route_choice=RouteType.DEEP_INVESTIGATION)
    wf.gateway = live_gw

    rca_draft = PlanDraft(
        objective="Why did sales change in November 2004 compared with October 2004?",
        complexity="investigation",
        metrics=["gross_revenue"],
        dimensions=["product_line", "country", "product", "customer"],
        period_key="2004-11",
        filters=[],
        tasks=[
            TaskDraft(
                id="baseline",
                title="Baseline",
                agent="sql",
                purpose="Baseline",
                depends_on=[],
                priority=1,
                query_spec=QuerySpec(metric="gross_revenue", dimensions=[], comparison="previous_period"),
            ),
        ],
        hypotheses_required=True,
        output=OutputRequirements(format="business_report", report_formats=["html", "pdf"]),
        budgets=BudgetSpec(max_queries=10, max_refinements=1, max_seconds=120),
    )
    wf_live, _, _ = _build_workflow_fixture(
        tmp_path,
        route_choice=RouteType.DEEP_INVESTIGATION,
        custom_plan_draft=rca_draft,
    )
    wf_live.gateway = live_gw

    res = wf_live.run("Why did sales change in November 2004 compared with October 2004?", session_id="live_mysql_s1")
    report = res["report"]

    assert report["status"] == "complete"
    assert len(report["evidence"]) >= 6
    assert len(report["python_analyses"]) >= 3
    for py_item in report["python_analyses"]:
        if py_item["tool"] in ("revenue_decomposition", "contribution_analysis"):
            assert py_item["calculation"]["summary"]["reconciled"] is True


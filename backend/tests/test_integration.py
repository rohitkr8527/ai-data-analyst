"""Isolated unit and integration tests using test fixtures and dependency injection."""
import pytest
from unittest.mock import MagicMock
from app.schemas import QuestionIntent, PlanTask, AnalysisPlan, SQLDraft, ReportNarrative
from app.services.semantics import extract_intent, resolve_period
from app.agents.planner import PlannerAgent
from app.agents.sql_analyst import SQLAnalyst
from app.agents.reporter import ReporterAgent
from app.services.models import ModelSelector, TaskType, ModelUsage


@pytest.fixture
def mock_selector():
    selector = MagicMock(spec=ModelSelector)

    def mock_parse(task_type, response_model, system_instructions, input_payload, custom_effort=None):
        usage = ModelUsage(
            task_type=task_type.value,
            model="gpt-test-model",
            reasoning_effort="low",
            duration_ms=10,
            input_tokens=50,
            output_tokens=20
        )
        if response_model is AnalysisPlan:
            res = AnalysisPlan(
                tasks=[
                    PlanTask(id="baseline", kind="baseline", purpose="Establish baseline"),
                    PlanTask(id="product_line_bridge", kind="product_line_bridge", purpose="Product lines")
                ],
                hypotheses=["Classic Cars variance drove decline"]
            )
            return res, usage
        elif response_model is SQLDraft:
            res = SQLDraft(sql="SELECT productName, value FROM test_table", reason="Approved query")
            return res, usage
        elif response_model is ReportNarrative:
            res = ReportNarrative(
                executive_summary="Sales decreased due to supply chain variance in Classic Cars.",
                key_findings_summary=["Classic Cars had highest variance"],
                recommended_actions=["Audit supplier deliveries"],
                data_limitations=["Overlapping dimensions"]
            )
            return res, usage
        raise ValueError(f"Unexpected response model: {response_model}")

    selector.parse_structured.side_effect = mock_parse
    return selector


def test_planner_agent_plan_formulation(mock_selector):
    planner = PlannerAgent(model_selector=mock_selector)
    intent = QuestionIntent(kind="deep", metric="gross_revenue", period="2004-11")
    plan, usage = planner.plan(intent, "Why did sales decrease in November 2004?", "DEEP_INVESTIGATION", "schema context")

    assert plan is not None
    assert len(plan.tasks) >= 2
    assert plan.tasks[0].kind == "baseline"
    assert usage is not None
    assert usage.agent_role == "Investigation Planner"


def test_sql_analyst_draft_generation(mock_selector):
    analyst = SQLAnalyst(model_selector=mock_selector)
    task = PlanTask(id="top_products", kind="top_products", purpose="Measure top products")
    intent = QuestionIntent(kind="top_products", metric="gross_revenue", period="2004", n=10)

    sql, params, usage = analyst.draft(task, intent, "schema context")
    assert sql.startswith("SELECT")
    assert "target_start" in params
    assert usage.agent_role == "SQL Analyst"


def test_reporter_narrative_enrichment(mock_selector):
    reporter = ReporterAgent(model_selector=mock_selector)
    intent = QuestionIntent(kind="top_products", metric="gross_revenue", period="2004")

    mock_evidence = [{
        "task_id": "top_products",
        "query_id": "q-top-1",
        "purpose": "Top products",
        "columns": ["product", "value"],
        "rows": [
            {"product": "1969 Harley Davidson", "value": 150000.0},
            {"product": "1952 Alpine Renault", "value": 120000.0}
        ],
        "sql": "SELECT ...",
        "duration_ms": 15
    }]

    report = reporter.create_report(
        intent=intent,
        evidence_list=mock_evidence,
        hypotheses_text=[],
        task_errors=[],
        trace_events=[],
        models_used=[]
    )

    assert report.status == "complete"
    assert len(report.findings) >= 1
    assert len(report.charts) >= 1
    # Check that mock insight was incorporated
    assert "Classic Cars" in report.summary or "Harley" in report.summary
    assert len(report.models_used) >= 1
    assert report.models_used[-1].agent_role == "Business Insight Agent"


def test_intent_extraction_peak_month():
    intent = extract_intent("which month has the highest revenue")
    assert intent.kind == "trend"
    assert intent.n == 1
    assert intent.metric == "gross_revenue"


def test_sql_builder_single_peak_month():
    from app.services.sql_builder import SQLBuilder
    builder = SQLBuilder()
    intent = extract_intent("which month has the highest revenue")
    task = PlanTask(id="trend", kind="trend", purpose="Peak month")
    sql, params = builder.build(task, intent)
    assert "ORDER BY value DESC" in sql
    assert "LIMIT :limit_n" in sql
    assert params["limit_n"] == 1


def test_reporter_single_peak_month_suppresses_unsolicited_extras(mock_selector):
    reporter = ReporterAgent(model_selector=mock_selector)
    intent = extract_intent("which month has the highest revenue")

    mock_evidence = [{
        "task_id": "trend",
        "query_id": "q-peak-1",
        "purpose": "Peak month",
        "columns": ["month", "value"],
        "rows": [
            {"month": "2004-11", "value": 979291.98}
        ],
        "sql": "SELECT ...",
        "duration_ms": 12
    }]

    report = reporter.create_report(
        intent=intent,
        evidence_list=mock_evidence,
        hypotheses_text=[],
        task_errors=[],
        trace_events=[],
        models_used=[]
    )

    assert report.status == "complete"
    # Single-entity queries should not generate charts or unprompted strategic recommendations
    assert len(report.charts) == 0
    assert len(report.recommendations) == 0
    assert "November 2004" in report.findings[0].title or "November 2004" in report.findings[0].detail


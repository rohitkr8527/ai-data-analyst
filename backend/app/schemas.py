"""Pydantic API schemas and data contracts."""
from typing import Any, Literal
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    session_id: str | None = None
    run_id: str | None = None


class QuestionIntent(BaseModel):
    kind: str = "simple"
    metric: str = "gross_revenue"
    period: str | None = None
    country: str | None = None
    product_line: str | None = None
    n: int = Field(default=10, ge=1, le=50)
    is_followup: bool = False


class PlanTask(BaseModel):
    id: str
    kind: str
    purpose: str
    dimension: str | None = None


class AnalysisPlan(BaseModel):
    tasks: list[PlanTask] = Field(max_length=10)
    hypotheses: list[str] = Field(default_factory=list, max_length=8)
    comparison_baseline_required: bool = True


class SQLDraft(BaseModel):
    sql: str
    reason: str = ""


class ReportNarrative(BaseModel):
    executive_summary: str
    key_findings_summary: list[str] = Field(default_factory=list, max_length=4)
    recommended_actions: list[str] = Field(default_factory=list, max_length=4)
    data_limitations: list[str] = Field(default_factory=list, max_length=4)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=4)


class Evidence(BaseModel):
    task_id: str
    query_id: str
    purpose: str
    columns: list[str]
    rows: list[dict[str, Any]]
    sql: str
    duration_ms: int
    dataset_id: str | None = None


class Finding(BaseModel):
    title: str
    detail: str
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["verified", "inconclusive", "warning"] = "verified"


class HypothesisItem(BaseModel):
    id: str
    description: str
    dimension: str
    hypothesis_type: str = "accounting_contribution"
    status: Literal["verified", "inconclusive", "warning", "supported", "not_supported", "rejected", "revised"] = "verified"
    measured_effect: float = 0.0
    share_of_change: float | None = None
    confidence: float = 0.8
    evidence_query_ids: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    limitations: str = ""


class ChartSpec(BaseModel):
    title: str
    kind: Literal["bar", "line", "waterfall", "pie", "grouped_bar"] = "bar"
    labels: list[str]
    values: list[float]
    unit: str = "USD"
    x_label: str = ""
    y_label: str = ""
    source_task_id: str = ""


class ModelUsageItem(BaseModel):
    agent_role: str
    task_type: str
    model: str
    reasoning_effort: str
    duration_ms: int
    tokens: int = 0
    cost_usd: float = 0.0


class Report(BaseModel):
    title: str
    summary: str
    status: Literal["complete", "partial", "clarification_required", "unsupported"] = "complete"
    output_mode: str = "answer_with_visuals"
    period: str = ""
    comparison: str = ""
    kpis: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    hypotheses: list[HypothesisItem] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    python_analyses: list[dict[str, Any]] = Field(default_factory=list)
    evidence_review: dict[str, Any] = Field(default_factory=dict)
    task_graph: list[dict[str, Any]] = Field(default_factory=list)
    business_report: dict[str, Any] | None = None
    models_used: list[ModelUsageItem] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    run_id: str
    session_id: str
    route: str
    route_confidence: float = 1.0
    route_probabilities: dict[str, float] = Field(default_factory=dict)
    report: Report

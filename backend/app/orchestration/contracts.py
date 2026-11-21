"""Structured orchestration contracts for the dynamic Supervisor-Worker architecture.

Two families of models live here:

* LLM-facing *drafts* (``PlanDraft``, ``TaskDraft``, ``HypothesisDraft``, ``ReviewDraft``,
  ``InsightDraftSet``, ``ReportDraft``). They are used as OpenAI Structured Output schemas and
  therefore avoid free-form ``dict`` fields (strict JSON schema requires closed objects).
* Internal validated contracts (``AnalysisPlan``, ``AnalysisTask``, ``TaskResult``,
  ``HypothesisRecord``, ``EvidenceRecord``, ``EvidenceReview``, ``InsightRecord``,
  ``ChartSpecification``, ``BusinessReport``). Only deterministic code creates these.

Graph state stores these contracts as plain dicts (``model_dump()``) so LangGraph checkpoints
remain small and serialisable; large datasets are stored in the DatasetStore and referenced.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

MetricName = Literal["gross_revenue", "total_orders", "aov", "units_sold", "payments_received"]
DimensionName = Literal[
    "product_line", "product", "country", "customer", "order_status",
    "month", "quarter", "year", "sales_rep", "office",
]
FilterDimension = Literal["country", "product_line", "customer", "product", "order_status"]
ToolName = Literal[
    "period_comparison", "contribution_analysis", "revenue_decomposition",
    "statistical_summary", "trend_analysis", "anomaly_detection",
    "correlation_analysis", "distribution_analysis", "data_quality_check",
    "forecast_linear", "auto", "autonomous_research", "custom_python",
]
AgentName = Literal["sql", "python", "hypothesis"]


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = {TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.SKIPPED.value, TaskStatus.CANCELLED.value}


# ---------------------------------------------------------------------------
# Task specifications
# ---------------------------------------------------------------------------
class FilterSpec(BaseModel):
    dimension: FilterDimension
    value: str


class QuerySpec(BaseModel):
    """Structured query specification produced by the Planner and compiled/validated downstream."""
    metric: MetricName
    dimensions: list[DimensionName] = Field(default_factory=list, max_length=2)
    comparison: Literal["none", "previous_period"] = "none"
    period_scope: Literal["plan_period", "full_history", "custom"] = "plan_period"
    custom_start: str | None = Field(default=None, description="Inclusive ISO date (YYYY-MM-DD) when period_scope=custom")
    custom_end: str | None = Field(default=None, description="Exclusive ISO date (YYYY-MM-DD) when period_scope=custom")
    filters: list[FilterSpec] = Field(default_factory=list, max_length=4)
    inherit_plan_filters: bool = True
    order: Literal["value_desc", "value_asc", "change_asc", "change_desc", "chronological", "none"] = "value_desc"
    limit: int | None = Field(default=None, ge=1, le=100)
    mode: Literal["structured", "custom", "autonomous"] = "structured"
    custom_request: str | None = Field(default=None, description="Natural-language request when mode=custom or autonomous")
    custom_tables: list[str] = Field(default_factory=list, max_length=6)


class PythonToolSpec(BaseModel):
    tool: ToolName = "autonomous_research"
    mode: Literal["auto", "autonomous", "utility"] = "auto"
    input_task_ids: list[str] = Field(min_length=1, max_length=3)
    objective: str = ""
    analysis_methods: list[str] = Field(default_factory=list, max_length=6)
    value_column: str | None = None
    label_column: str | None = None
    time_column: str | None = None
    secondary_value_column: str | None = None
    threshold: float | None = None
    horizon: int | None = Field(default=None, ge=1, le=12)


class HypothesisSpec(BaseModel):
    focus: str
    max_hypotheses: int = Field(default=4, ge=1, le=6)


class TaskDraft(BaseModel):
    """LLM-facing task definition (Planner, Hypothesis Agent, Evidence Reviewer)."""
    id: str = Field(description="Unique snake_case identifier, e.g. 'baseline' or 'product_line_bridge'")
    title: str
    agent: AgentName
    purpose: str
    depends_on: list[str] = Field(default_factory=list, max_length=6)
    priority: int = Field(default=3, ge=1, le=5, description="1 = highest priority")
    required: bool = True
    query_spec: QuerySpec | None = None
    python_spec: PythonToolSpec | None = None
    hypothesis_spec: HypothesisSpec | None = None
    hypothesis_ids: list[str] = Field(default_factory=list, max_length=6)


class AnalysisTask(TaskDraft):
    """Validated task contract scheduled by the supervisor."""
    origin: Literal["planner", "hypothesis", "refinement"] = "planner"
    round: int = 0
    fingerprint: str = ""


class TaskDependency(BaseModel):
    task_id: str
    depends_on: str
    kind: Literal["data", "ordering"] = "data"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
class ExpectedEvidence(BaseModel):
    id: str
    description: str
    task_ids: list[str] = Field(default_factory=list, max_length=6)


class ChartRequest(BaseModel):
    task_id: str
    preferred_kind: Literal["auto", "bar", "line", "pie", "waterfall", "grouped_bar"] = "auto"
    title: str = ""


class OutputRequirements(BaseModel):
    format: Literal["direct_answer", "answer_with_visuals", "business_report"] = "direct_answer"
    charts: list[ChartRequest] = Field(default_factory=list, max_length=4)
    include_tables: bool = True
    report_formats: list[Literal["html", "pdf"]] = Field(default_factory=list, max_length=2)


class BudgetSpec(BaseModel):
    max_queries: int = Field(default=6, ge=1, le=30)
    max_refinements: int = Field(default=1, ge=0, le=5)
    max_seconds: int = Field(default=120, ge=10, le=1800)


class PlanDraft(BaseModel):
    """LLM-facing AnalysisPlan produced by the Planning Agent (gpt-6.1-sol)."""
    objective: str
    complexity: Literal["lookup", "comparison", "investigation"]
    metrics: list[MetricName] = Field(min_length=1, max_length=3)
    dimensions: list[DimensionName] = Field(default_factory=list, max_length=6)
    period_key: str | None = Field(default=None, description="'YYYY-MM', 'YYYY', or null for the default analysis window")
    filters: list[FilterSpec] = Field(default_factory=list, max_length=4)
    tasks: list[TaskDraft] = Field(min_length=1, max_length=14)
    hypotheses_required: bool = False
    hypothesis_rationale: str = ""
    expected_evidence: list[ExpectedEvidence] = Field(default_factory=list, max_length=8)
    output: OutputRequirements
    budgets: BudgetSpec
    stopping_conditions: list[str] = Field(default_factory=list, max_length=5)
    rationale: str = ""


class RunBudget(BaseModel):
    max_queries: int
    max_refinements: int
    max_seconds: int
    max_tokens: int
    max_cost_usd: float
    max_sql_repairs: int
    max_parallel_db_queries: int
    max_parallel_tasks: int
    max_task_retries: int
    task_timeout_seconds: int


class AnalysisPlan(BaseModel):
    plan_id: str
    version: int = 1
    question: str
    objective: str
    complexity: str
    metrics: list[str]
    dimensions: list[str] = Field(default_factory=list)
    period_key: str | None = None
    period: dict[str, str] = Field(default_factory=dict)
    filters: list[FilterSpec] = Field(default_factory=list)
    tasks: list[AnalysisTask]
    dependencies: list[TaskDependency] = Field(default_factory=list)
    hypotheses_required: bool = False
    hypothesis_rationale: str = ""
    expected_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    output: OutputRequirements
    budget: RunBudget
    stopping_conditions: list[str] = Field(default_factory=list)
    rationale: str = ""
    warnings: list[str] = Field(default_factory=list)
    planner_model: str = ""

    def task(self, task_id: str) -> AnalysisTask | None:
        return next((t for t in self.tasks if t.id == task_id), None)


class PlanValidationReport(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Results & evidence
# ---------------------------------------------------------------------------
class DatasetRef(BaseModel):
    dataset_id: str
    run_id: str
    task_id: str
    columns: list[str]
    row_count: int
    query_id: str | None = None
    sql_hash: str | None = None
    preview: list[dict[str, Any]] = Field(default_factory=list)


class HypothesisDraft(BaseModel):
    """LLM-facing hypothesis from the Hypothesis Agent."""
    id: str
    statement: str
    hypothesis_type: Literal["observed_pattern", "accounting_contribution", "causal_candidate"]
    dimension: Literal[
        "product_line", "product", "country", "customer", "order_status", "sales_rep", "office",
        "order_volume", "average_order_value", "trend", "none",
    ]
    target_entity: str | None = None
    expected_direction: Literal["decrease", "increase", "any"] = "any"
    evidence_required: str
    priority: int = Field(default=3, ge=1, le=5)
    alternative_explanations: list[str] = Field(default_factory=list, max_length=3)
    test: TaskDraft | None = None


class HypothesisSet(BaseModel):
    hypotheses: list[HypothesisDraft] = Field(min_length=1, max_length=6)
    reasoning: str = ""


class HypothesisRecord(BaseModel):
    id: str
    statement: str
    hypothesis_type: str
    dimension: str
    target_entity: str | None = None
    expected_direction: str = "any"
    evidence_required: str = ""
    priority: int = 3
    alternative_explanations: list[str] = Field(default_factory=list)
    test_task_ids: list[str] = Field(default_factory=list)
    status: Literal["untested", "supported", "not_supported", "inconclusive", "rejected", "revised"] = "untested"
    measured_effect: float | None = None
    share_of_change: float | None = None
    p_value: float | None = None
    statistical_evidence: str = ""
    revised_statement: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    assessment: str = ""
    confidence: float = 0.0
    causal_disclaimer: str = (
        "Supported means the measured accounting or statistical evidence is consistent with the hypothesis; "
        "it does not establish an external causal mechanism."
    )


class TaskResult(BaseModel):
    task_id: str
    agent: str
    status: Literal["COMPLETED", "FAILED"]
    fingerprint: str = ""
    dataset: DatasetRef | None = None
    query_id: str | None = None
    sql: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    sql_source: str | None = None
    repair_attempts: int = 0
    tool: str | None = None
    calculation: dict[str, Any] | None = None
    generated_code: str = ""
    methodology: str = ""
    execution_steps: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    assumptions_checked: list[str] = Field(default_factory=list)
    verification_status: str = ""
    hypotheses: list[HypothesisRecord] = Field(default_factory=list)
    proposed_tasks: list[AnalysisTask] = Field(default_factory=list)
    duration_ms: int = 0
    reused: bool = False
    error: str | None = None
    error_code: str | None = None
    retryable: bool = False
    warnings: list[str] = Field(default_factory=list)


class EvidenceFact(BaseModel):
    name: str
    value: float | None = None
    text: str | None = None
    unit: str = ""


class EvidenceRecord(BaseModel):
    id: str
    task_id: str
    kind: Literal["sql_dataset", "calculation", "hypothesis_test", "reconciliation"]
    description: str
    source_query_ids: list[str] = Field(default_factory=list)
    dataset_ids: list[str] = Field(default_factory=list)
    calculation: str | None = None
    facts: list[EvidenceFact] = Field(default_factory=list)
    row_count: int = 0
    status: Literal["validated", "warning", "failed"] = "validated"
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Review & refinement
# ---------------------------------------------------------------------------
class RefinementRequest(BaseModel):
    reason: str
    kind: Literal[
        "additional_breakdown", "drill_down", "statistical_test",
        "recalculate", "new_hypothesis", "resolve_contradiction",
    ]
    gap: str
    priority: int = Field(default=3, ge=1, le=5)
    proposed_task: TaskDraft | None = None


class ReviewIssue(BaseModel):
    severity: Literal["blocking", "major", "minor"]
    description: str
    evidence_ids: list[str] = Field(default_factory=list, max_length=6)


class ReviewDraft(BaseModel):
    """LLM-facing Evidence Reviewer verdict (gpt-6.1-sol)."""
    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    assessment: str
    metric_alignment_ok: bool = True
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=8)
    contradictions: list[str] = Field(default_factory=list, max_length=4)
    unexplained_findings: list[str] = Field(default_factory=list, max_length=4)
    refinement_requests: list[RefinementRequest] = Field(default_factory=list, max_length=4)


class EvidenceReview(BaseModel):
    round: int
    sufficient: bool
    decision: Literal["sufficient", "refine", "stop_budget", "stop_round_limit", "stop_no_progress", "stop_cancelled"]
    deterministic_issues: list[ReviewIssue] = Field(default_factory=list)
    ai_reviewed: bool = False
    ai_review: ReviewDraft | None = None
    refinement_requests: list[RefinementRequest] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Insights, charts and reports
# ---------------------------------------------------------------------------
class InsightDraft(BaseModel):
    id: str
    title: str
    statement: str
    kind: Literal["observation", "comparison", "accounting_contribution", "statistical", "causal_hypothesis", "data_quality"]
    evidence_ids: list[str] = Field(min_length=1, max_length=6)
    confidence: Literal["high", "medium", "low"] = "medium"


class InsightDraftSet(BaseModel):
    direct_answer: str
    insights: list[InsightDraft] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=6)
    recommendations: list[str] = Field(default_factory=list, max_length=5)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=4)


class InsightRecord(BaseModel):
    id: str
    title: str
    statement: str
    kind: str
    evidence_ids: list[str]
    confidence: str = "medium"
    grounded: bool = True
    ungrounded_numbers: list[str] = Field(default_factory=list)
    status: Literal["verified", "warning"] = "verified"


class ChartSeries(BaseModel):
    name: str
    values: list[float]


class ChartSpecification(BaseModel):
    id: str
    title: str
    kind: Literal["bar", "line", "waterfall", "pie", "grouped_bar"]
    labels: list[str]
    values: list[float]
    series: list[ChartSeries] = Field(default_factory=list)
    unit: str = "USD"
    x_label: str = ""
    y_label: str = ""
    source_task_id: str = ""
    source_dataset_id: str | None = None
    evidence_id: str = ""


class ReportSection(BaseModel):
    heading: str
    body: str
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)


class ReportDraft(BaseModel):
    """LLM-facing Report Generation Agent output (gpt-6.1-terra)."""
    title: str
    executive_summary: str
    business_objective: str
    kpi_overview: str
    sections: list[ReportSection] = Field(min_length=1, max_length=8)
    recommendations: list[str] = Field(default_factory=list, max_length=6)
    limitations: list[str] = Field(default_factory=list, max_length=6)


class BusinessReport(BaseModel):
    title: str
    executive_summary: str
    business_objective: str
    kpi_overview: str
    sections: list[ReportSection]
    recommendations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    chart_ids: list[str] = Field(default_factory=list)
    evidence_index: list[dict[str, Any]] = Field(default_factory=list)
    generated_at: str = ""
    model: str = ""
    formats: list[str] = Field(default_factory=lambda: ["html", "pdf"])

"""Intelligent Planning Agent (gpt-6.1-sol)."""
from __future__ import annotations
import logging
from typing import Any
from app.orchestration.contracts import (
    AnalysisPlan, BudgetSpec, ChartRequest, ExpectedEvidence, FilterSpec,
    HypothesisSpec, OutputRequirements, PlanDraft, PythonToolSpec, QuerySpec,
    RefinementRequest, TaskDraft,
)
from app.schemas import AnalysisPlan as LegacyAnalysisPlan, ModelUsageItem, PlanTask, QuestionIntent
from app.services.models import ModelSelector, TaskType
from app.services.semantics import APPROVED_DIMENSIONS, APPROVED_METRICS, DATA_MAX_DATE, DATA_MIN_DATE

logger = logging.getLogger(__name__)

AVAILABLE_DEEP_TASKS: dict[str, str] = {
    "baseline": "Establish total baseline revenue, completed order volume, and AOV across comparison periods",
    "product_line_bridge": "Analyze revenue changes across product lines (e.g. Classic Cars, Vintage Cars, Motorcycles)",
    "country_bridge": "Analyze revenue changes across customer countries (e.g. USA, France, Australia, UK)",
    "product_bridge": "Identify top individual products driving the revenue variance",
    "customer_bridge": "Identify top individual customer accounts driving the variance",
    "order_status": "Inspect order fulfillment status breakdown (Shipped, In Process, Cancelled, Disputed)",
    "trend": "Show multi-month historical trend to assess seasonality versus abrupt decline",
}

PLANNER_SYSTEM_PROMPT = """You are the Lead Analytical Strategist (Planner Agent) for an autonomous multi-agent data analyst on ClassicModels.
Translate the user's question into a lean, evidence-first Task DAG (`PlanDraft`).

### Delegation & Token-Efficiency Rules:
1. **SQL Tasks (`agent="sql"`)**:
   - Prefer `QuerySpec(mode="structured", ...)` for all standard metrics (`gross_revenue`, `total_orders`, `aov`, `units_sold`, `payments_received`) and dimensions (`product_line`, `country`, `product`, `customer`, `order_status`, `month`, `quarter`, `year`, `sales_rep`, `office`).
   - Reserve `QuerySpec(mode="autonomous", ...)` ONLY for custom multi-table behavioral/cohort queries (e.g., customer-level RFM features) that cannot be expressed as a structured metric + dimension query.
2. **Python Tasks (`agent="python"`)**:
   - Prefer deterministic utility tools (`revenue_decomposition`, `contribution_analysis`, `period_comparison`, `statistical_summary`, `trend_analysis`, `forecast_linear`) for standard variance, contribution, and performance reports.
   - Reserve `PythonToolSpec(tool="autonomous_research", mode="autonomous", ...)` ONLY for open-ended statistical or machine-learning tasks (e.g., RFM KMeans clustering, regression, hypothesis testing, Gini/Pareto concentration).
3. **Lean DAG Design**:
   - Simple lookups: 1 structured SQL task (`complexity="lookup"`, `hypotheses_required=false`).
   - Comparisons & Executive Reports: 2 to 4 structured SQL tasks + deterministic Python decomposition/contribution tasks.
   - Open-ended data science: 1 SQL dataset task + 1 dependent `autonomous_research` Python task.
"""

_OPEN_ENDED_KEYWORDS = (
    "segment", "cluster", "rfm", "cohort", "retention", "churn", "regression", "ols",
    "bootstrap", "permutation", "p-value", "significance", "t-test", "mann-whitney",
    "anova", "chi-square", "seasonal", "seasonality", "decomposition", "change point",
    "changepoint", "pareto", "concentration", "elasticity", "lifetime value", "ltv",
    "product mix", "product-mix", "explore", "statistical test", "hypothesis test",
    "multivariate", "backtest",
)


class PlannerAgent:
    """Creates dynamic task-DAG `PlanDraft` contracts using OpenAI gpt-6.1-sol."""

    def __init__(self, model_selector: ModelSelector | None = None):
        self.selector = model_selector or ModelSelector()

    def create_plan_draft(
        self,
        question: str,
        intent: QuestionIntent,
        route_decision: dict[str, Any],
        schema_context: str,
        previous_context: dict[str, Any] | None = None,
        permissions: dict[str, Any] | None = None,
    ) -> tuple[PlanDraft, ModelUsageItem]:
        previous_context = previous_context or {}
        permissions = permissions or {}
        route = route_decision.get("route", "SIMPLE_SQL")
        q_lower = question.lower()

        # Zero-token fast-path planning for simple single-metric lookups in live execution
        is_simple_lookup = (
            type(self.selector) is ModelSelector
            and route in ("SIMPLE_SQL", "CONTEXTUAL_FOLLOWUP")
            and intent.kind not in ("deep", "comparative")
            and not any(kw in q_lower for kw in _OPEN_ENDED_KEYWORDS)
            and not any(w in q_lower for w in ("why", "root cause", "compare", "vs", "versus", "report", "briefing", "forecast", "predict"))
        )
        if is_simple_lookup:
            sql_task = self._build_lookup_sql_task(intent)
            fast_draft = PlanDraft(
                objective=question,
                complexity="lookup",
                metrics=[intent.metric if intent.metric in APPROVED_METRICS else "gross_revenue"],  # type: ignore[list-item]
                dimensions=[],
                period_key=intent.period,
                filters=[],
                tasks=[sql_task],
                hypotheses_required=False,
                output=OutputRequirements(format="direct_answer" if (intent.kind == "total" or intent.n == 1) else "answer_with_visuals"),
                budgets=BudgetSpec(max_queries=2, max_refinements=0, max_seconds=60),
            )
            fast_usage = ModelUsageItem(
                agent_role="Planning Agent (Fast-Path)",
                task_type=TaskType.INVESTIGATION_PLANNING.value,
                model="deterministic",
                reasoning_effort="none",
                duration_ms=1,
                tokens=0,
                cost_usd=0.0,
            )
            return self._post_normalize_draft(fast_draft, question, intent, route), fast_usage

        input_payload = {
            "question": question,
            "extracted_intent": intent.model_dump(),
            "jev_routing_hint": route_decision,
            "previous_context": {k: previous_context.get(k) for k in ("last_metric", "last_period", "last_filters", "previous_question")},
            "approved_metrics": list(APPROVED_METRICS.keys()),
            "approved_dimensions": list(APPROVED_DIMENSIONS.keys()),
            "approved_python_tools": [
                "autonomous_research", "custom_python", "period_comparison", "contribution_analysis",
                "revenue_decomposition", "statistical_summary", "trend_analysis", "anomaly_detection",
                "correlation_analysis", "distribution_analysis", "data_quality_check", "forecast_linear",
            ],
            "data_range": {"min_date": DATA_MIN_DATE, "max_date": DATA_MAX_DATE},
            "permissions": permissions,
            "schema_summary": schema_context[:1500],
        }

        parsed, usage = self.selector.parse_structured(
            task_type=TaskType.INVESTIGATION_PLANNING, response_model=PlanDraft,
            system_instructions=PLANNER_SYSTEM_PROMPT, input_payload=input_payload, custom_effort="low",
        )
        usage_item = ModelUsageItem(
            agent_role="Planning Agent", task_type=usage.task_type, model=usage.model,
            reasoning_effort=usage.reasoning_effort, duration_ms=usage.duration_ms,
            tokens=usage.input_tokens + usage.output_tokens, cost_usd=usage.cost_usd,
        )

        if isinstance(parsed, LegacyAnalysisPlan) or not isinstance(parsed, PlanDraft):
            return self._adapt_legacy_plan(parsed, question, intent, route), usage_item
        return self._post_normalize_draft(parsed, question, intent, route), usage_item

    def _post_normalize_draft(self, draft: PlanDraft, question: str, intent: QuestionIntent, route: str) -> PlanDraft:
        q_lower = question.lower()
        wants_report = any(w in q_lower for w in ("report", "executive summary", "comprehensive", "board", "pdf", "html report", "briefing"))
        wants_forecast = any(w in q_lower for w in ("forecast", "predict", "projection", "next month", "next quarter"))
        wants_stats = any(w in q_lower for w in ("statistic", "distribution", "anomaly", "anomalies", "outlier", "correlation", "std", "median", "variance", "decompose"))
        has_open_ended = any(kw in q_lower for kw in _OPEN_ENDED_KEYWORDS)
        has_custom_or_auto = any(
            (t.query_spec and t.query_spec.mode in ("custom", "autonomous"))
            or (t.python_spec and (t.python_spec.tool in ("autonomous_research", "custom_python") or t.python_spec.mode == "autonomous"))
            for t in draft.tasks
        )
        is_explicit_why = any(w in q_lower for w in ("why", "root cause", "decline", "decrease", "drop", "fell")) or (wants_report and not has_open_ended)
        is_explicit_comp = any(w in q_lower for w in ("compare", "vs", "versus"))
        is_why = intent.kind == "deep" or route == "DEEP_INVESTIGATION" or is_explicit_why
        is_comp = not is_why and (intent.kind == "comparative" or route == "COMPARATIVE_ANALYSIS" or is_explicit_comp)

        plan_filters = list(draft.filters)
        if intent.country and not any(f.dimension == "country" for f in plan_filters):
            plan_filters.append(FilterSpec(dimension="country", value=intent.country))
        if intent.product_line and not any(f.dimension == "product_line" for f in plan_filters):
            plan_filters.append(FilterSpec(dimension="product_line", value=intent.product_line))
        period_key = draft.period_key or intent.period

        if not is_explicit_why and not is_explicit_comp and (has_open_ended or has_custom_or_auto or wants_stats or wants_forecast):
            kept_tasks = list(draft.tasks)
            sql_tasks = [t for t in kept_tasks if t.agent == "sql"]
            if not sql_tasks:
                default_sql = self._build_lookup_sql_task(intent)
                kept_tasks.insert(0, default_sql)
                sql_tasks = [default_sql]
            sql_ids = [t.id for t in sql_tasks]
            if not any(t.agent == "python" for t in kept_tasks):
                kept_tasks.append(TaskDraft(
                    id="autonomous_python_analysis", title="Autonomous Python Statistical Investigation", agent="python",
                    purpose=f"Autonomously design and execute appropriate Python analysis for: {question}",
                    depends_on=sql_ids[:2], priority=2,
                    python_spec=PythonToolSpec(tool="autonomous_research", mode="autonomous", input_task_ids=sql_ids[:2], objective=question),
                ))
            elif has_open_ended:
                kept_tasks = [
                    t.model_copy(update={"python_spec": t.python_spec.model_copy(update={"tool": "autonomous_research", "mode": "autonomous", "objective": t.python_spec.objective or t.purpose or question})})
                    if (t.agent == "python" and t.python_spec and t.python_spec.tool not in ("autonomous_research", "custom_python")) else t
                    for t in kept_tasks
                ]

            has_hyp = any(t.agent == "hypothesis" for t in kept_tasks)
            charts = draft.output.charts or [ChartRequest(task_id=sql_tasks[0].id, preferred_kind="auto", title=draft.objective or question)]
            return draft.model_copy(update={
                "complexity": "investigation" if has_hyp else ("comparison" if is_comp else draft.complexity),
                "period_key": period_key, "filters": plan_filters, "tasks": kept_tasks,
                "hypotheses_required": draft.hypotheses_required or has_hyp,
                "output": OutputRequirements(format="business_report" if wants_report else "answer_with_visuals", charts=charts, include_tables=True, report_formats=["html", "pdf"] if wants_report else []),
                "budgets": BudgetSpec(max_queries=max(draft.budgets.max_queries, len(sql_tasks) + 3), max_refinements=max(1, draft.budgets.max_refinements), max_seconds=max(120, draft.budgets.max_seconds)),
            })

        if not is_why and not is_comp:
            sql_tasks = [t for t in draft.tasks if t.agent == "sql"]
            sql_tasks = [self._align_lookup_sql_task(sql_tasks[0], intent)] if sql_tasks else [self._build_lookup_sql_task(intent)]
            kept_tasks = list(sql_tasks)
            if wants_forecast or wants_stats:
                py_tasks = [t for t in draft.tasks if t.agent == "python"]
                if py_tasks:
                    ps = py_tasks[0].python_spec or PythonToolSpec(tool="statistical_summary", input_task_ids=[sql_tasks[0].id])
                    kept_tasks.append(py_tasks[0].model_copy(update={"depends_on": [sql_tasks[0].id], "python_spec": ps.model_copy(update={"input_task_ids": [sql_tasks[0].id]})}))
                else:
                    tool_name = "forecast_linear" if wants_forecast else "statistical_summary"
                    kept_tasks.append(TaskDraft(
                        id=f"py_{tool_name}", title=f"Run {tool_name.replace('_', ' ')}", agent="python",
                        purpose=f"Compute {tool_name.replace('_', ' ')} on {sql_tasks[0].id}", depends_on=[sql_tasks[0].id], priority=2,
                        python_spec=PythonToolSpec(tool=tool_name, input_task_ids=[sql_tasks[0].id]),  # type: ignore[arg-type]
                    ))

            is_scalar = intent.kind == "total" or intent.n == 1
            if wants_report:
                out_fmt, charts, rep_fmts = "business_report", ([] if is_scalar else [ChartRequest(task_id=sql_tasks[0].id, preferred_kind="auto", title=draft.objective)]), ["html", "pdf"]
            elif is_scalar and not wants_forecast:
                out_fmt, charts, rep_fmts = "direct_answer", [], []
            else:
                out_fmt, charts, rep_fmts = "answer_with_visuals", (draft.output.charts or [ChartRequest(task_id=sql_tasks[0].id, preferred_kind="auto", title=draft.objective)]), []

            return draft.model_copy(update={
                "complexity": "lookup", "period_key": period_key, "filters": plan_filters,
                "tasks": kept_tasks, "hypotheses_required": False,
                "output": OutputRequirements(format=out_fmt, charts=charts, include_tables=True, report_formats=rep_fmts),  # type: ignore[arg-type]
                "budgets": BudgetSpec(max_queries=max(2, len(sql_tasks)), max_refinements=1, max_seconds=draft.budgets.max_seconds),
            })

        metric = (draft.metrics[0] if draft.metrics else intent.metric) or "gross_revenue"
        # Convert any non-open-ended SQL tasks in standard investigations to structured mode so they compile with 0 LLM tokens
        if not has_open_ended and type(self.selector) is ModelSelector:
            tasks_by_id = {
                t.id: (
                    t.model_copy(update={"query_spec": t.query_spec.model_copy(update={"mode": "structured"})})
                    if (t.agent == "sql" and t.query_spec and t.query_spec.mode in ("custom", "autonomous") and t.id in AVAILABLE_DEEP_TASKS)
                    else t
                )
                for t in draft.tasks
                if not (t.agent == "sql" and t.query_spec and t.query_spec.mode in ("custom", "autonomous") and t.id not in AVAILABLE_DEEP_TASKS)
                and not (t.agent == "python" and t.python_spec and t.python_spec.tool in ("autonomous_research", "custom_python"))
            }
        else:
            tasks_by_id = {t.id: t for t in draft.tasks}

        if "baseline" not in tasks_by_id:
            tasks_by_id["baseline"] = TaskDraft(
                id="baseline", title="Period-over-Period Baseline", agent="sql",
                purpose=AVAILABLE_DEEP_TASKS["baseline"], depends_on=[], priority=1,
                query_spec=QuerySpec(metric=metric, dimensions=[], comparison="previous_period"),  # type: ignore[arg-type]
            )

        if is_why:
            for tid, dim, prio, lim in (("product_line_bridge", "product_line", 1, None), ("country_bridge", "country", 2, None), ("product_bridge", "product", 2, 15), ("customer_bridge", "customer", 2, 15)):
                if tid not in tasks_by_id and not any(t.agent == "sql" and t.query_spec and dim in t.query_spec.dimensions and t.query_spec.comparison == "previous_period" for t in tasks_by_id.values()):
                    tasks_by_id[tid] = TaskDraft(
                        id=tid, title=f"{dim.replace('_', ' ').title()} Variance Bridge", agent="sql",
                        purpose=AVAILABLE_DEEP_TASKS.get(tid, f"Measure {dim} variance"), depends_on=[], priority=prio,
                        query_spec=QuerySpec(metric=metric, dimensions=[dim], comparison="previous_period", order="change_asc", limit=lim),  # type: ignore[arg-type]
                    )
            if "order_status" not in tasks_by_id:
                tasks_by_id["order_status"] = TaskDraft(
                    id="order_status", title="Order Fulfillment Status Check", agent="sql",
                    purpose=AVAILABLE_DEEP_TASKS["order_status"], depends_on=[], priority=3,
                    query_spec=QuerySpec(metric=metric, dimensions=["order_status"], comparison="none"),  # type: ignore[arg-type]
                )
            if "trend" not in tasks_by_id:
                tasks_by_id["trend"] = TaskDraft(
                    id="trend", title="Historical Monthly Trend", agent="sql",
                    purpose=AVAILABLE_DEEP_TASKS["trend"], depends_on=[], priority=3,
                    query_spec=QuerySpec(metric=metric, dimensions=["month"], comparison="none", period_scope="full_history", order="chronological"),  # type: ignore[arg-type]
                )
            if not any(t.agent == "hypothesis" for t in tasks_by_id.values()) and type(self.selector) is not ModelSelector:
                hyp_deps = [tid for tid in ("baseline", "product_line_bridge", "country_bridge") if tid in tasks_by_id]
                tasks_by_id["formulate_hypotheses"] = TaskDraft(
                    id="formulate_hypotheses", title="Formulate & Test Root-Cause Hypotheses", agent="hypothesis",
                    purpose="Generate falsifiable root-cause hypotheses from baseline and dimensional variances",
                    depends_on=hyp_deps, priority=2, hypothesis_spec=HypothesisSpec(focus=draft.objective or question, max_hypotheses=4),
                )
            if not any(t.agent == "python" and t.python_spec and t.python_spec.tool == "revenue_decomposition" for t in tasks_by_id.values()):
                tasks_by_id["decompose_revenue"] = TaskDraft(
                    id="decompose_revenue", title="Order Volume vs AOV Revenue Decomposition", agent="python",
                    purpose="Decompose net revenue change into order count effect, AOV effect, and interaction term",
                    depends_on=["baseline"], priority=2, python_spec=PythonToolSpec(tool="revenue_decomposition", input_task_ids=["baseline"]),
                )
            for bridge_id, py_id, title in (("product_line_bridge", "contrib_product_line", "Product Line Contribution Reconciliation"), ("country_bridge", "contrib_country", "Country Contribution Reconciliation")):
                if bridge_id in tasks_by_id and py_id not in tasks_by_id:
                    tasks_by_id[py_id] = TaskDraft(
                        id=py_id, title=title, agent="python",
                        purpose=f"Reconcile {bridge_id} sum of changes against baseline net change and compute percentage contributions",
                        depends_on=["baseline", bridge_id], priority=2,
                        python_spec=PythonToolSpec(tool="contribution_analysis", input_task_ids=[bridge_id, "baseline"]),
                    )
            if not any(t.agent == "python" for t in tasks_by_id.values()) or (has_open_ended and not any(t.agent == "python" and t.python_spec and (t.python_spec.tool in ("autonomous_research", "custom_python") or t.python_spec.mode == "autonomous") for t in tasks_by_id.values())):
                dep_candidates = [tid for tid in ("baseline", "product_line_bridge", "country_bridge") if tid in tasks_by_id]
                if dep_candidates:
                    tasks_by_id["autonomous_statistical_investigation"] = TaskDraft(
                        id="autonomous_statistical_investigation", title="Autonomous Statistical & Multivariate Investigation", agent="python",
                        purpose=f"Autonomously explore patterns, test statistical significance, and detect structural drivers for: {question}",
                        depends_on=dep_candidates[:3], priority=2,
                        python_spec=PythonToolSpec(tool="autonomous_research", mode="autonomous", input_task_ids=dep_candidates[:3], objective=draft.objective or question),
                    )
            charts = draft.output.charts or [
                ChartRequest(task_id="product_line_bridge", preferred_kind="bar", title="Product Line Variance"),
                ChartRequest(task_id="country_bridge", preferred_kind="bar", title="Country Variance"),
                ChartRequest(task_id="trend", preferred_kind="line", title="Monthly Sales Trend"),
            ]
            return draft.model_copy(update={
                "complexity": "investigation", "period_key": period_key, "filters": plan_filters,
                "tasks": list(tasks_by_id.values()), "hypotheses_required": True,
                "output": OutputRequirements(format="business_report" if wants_report else "answer_with_visuals", charts=charts, include_tables=True, report_formats=["html", "pdf"] if wants_report else []),
            })

        sql_ids = [t.id for t in tasks_by_id.values() if t.agent == "sql"]
        if not any(t.agent == "python" for t in tasks_by_id.values()) and sql_ids:
            target_sql = next((sid for sid in sql_ids if sid != "baseline"), sql_ids[0])
            tasks_by_id["compare_periods"] = TaskDraft(
                id="compare_periods", title="Period-over-Period Statistical Comparison", agent="python",
                purpose="Compute period-over-period deltas, percentage changes, and concentration",
                depends_on=[target_sql], priority=2, python_spec=PythonToolSpec(tool="period_comparison", input_task_ids=[target_sql]),
            )
        return draft.model_copy(update={
            "complexity": "comparison", "period_key": period_key, "filters": plan_filters,
            "tasks": list(tasks_by_id.values()),
            "output": OutputRequirements(
                format="business_report" if wants_report else "answer_with_visuals",
                charts=draft.output.charts or [ChartRequest(task_id=sql_ids[-1], preferred_kind="bar", title=draft.objective)],
                include_tables=True, report_formats=["html", "pdf"] if wants_report else [],
            ),
        })

    def _build_lookup_sql_task(self, intent: QuestionIntent) -> TaskDraft:
        dim_map: dict[str, list[str]] = {
            "top_products": ["product"], "by_country": ["country"], "by_product_line": ["product_line"],
            "by_customer": ["customer"], "trend": ["month"], "total": [], "simple": [],
        }
        dims = dim_map.get(intent.kind, [])
        is_full_hist = intent.kind == "trend" and intent.period is None
        order = "chronological" if (intent.kind == "trend" and intent.n > 1) else "value_desc"
        limit = intent.n if (dims and not (intent.kind == "trend" and intent.n > 1)) else None
        return TaskDraft(
            id=intent.kind if intent.kind != "simple" else "total",
            title=f"Measure {intent.metric} ({intent.kind})", agent="sql",
            purpose=f"Execute analytical measurement for {intent.metric} ({intent.kind})",
            depends_on=[], priority=1,
            query_spec=QuerySpec(
                metric=intent.metric, dimensions=dims, comparison="none",  # type: ignore[arg-type]
                period_scope="full_history" if is_full_hist else "plan_period", order=order, limit=limit,  # type: ignore[arg-type]
            ),
        )

    def _align_lookup_sql_task(self, task: TaskDraft, intent: QuestionIntent) -> TaskDraft:
        canonical = self._build_lookup_sql_task(intent)
        qspec = canonical.query_spec if intent.kind in ("top_products", "by_country", "by_product_line", "by_customer", "trend", "total") else (task.query_spec or canonical.query_spec)
        return task.model_copy(update={"id": canonical.id, "depends_on": [], "query_spec": qspec}) if qspec else canonical

    def _adapt_legacy_plan(self, legacy: Any, question: str, intent: QuestionIntent, route: str) -> PlanDraft:
        raw_tasks = getattr(legacy, "tasks", []) or []
        hyps = getattr(legacy, "hypotheses", []) or []
        task_drafts = [
            TaskDraft(
                id=getattr(pt, "id", getattr(pt, "kind", "total")),
                title=getattr(pt, "purpose", f"Execute {getattr(pt, 'kind', 'total')}"),
                agent="sql", purpose=getattr(pt, "purpose", f"Execute {getattr(pt, 'kind', 'total')}"),
                depends_on=[], priority=1 if getattr(pt, "kind", "total") == "baseline" else 2,
                query_spec=self._task_kind_to_query_spec(getattr(pt, "kind", "total"), intent),
            )
            for pt in raw_tasks
        ]
        is_deep = intent.kind in ("deep", "comparative") or route == "DEEP_INVESTIGATION"
        draft = PlanDraft(
            objective=question, complexity="investigation" if is_deep else "lookup",
            metrics=[intent.metric], dimensions=[], period_key=intent.period, filters=[],  # type: ignore[list-item]
            tasks=task_drafts or [self._build_lookup_sql_task(intent)],
            hypotheses_required=bool(hyps) or is_deep, hypothesis_rationale="; ".join(hyps[:3]),
            expected_evidence=[ExpectedEvidence(id="ev_primary", description="Primary SQL measurements", task_ids=[t.id for t in task_drafts])],
            output=OutputRequirements(format="answer_with_visuals" if is_deep or intent.n > 1 else "direct_answer"),
            budgets=BudgetSpec(max_queries=10, max_refinements=2, max_seconds=120),
            stopping_conditions=["All planned tasks complete and evidence reconciled."],
        )
        return self._post_normalize_draft(draft, question, intent, route)

    @staticmethod
    def _task_kind_to_query_spec(kind: str, intent: QuestionIntent) -> QuerySpec:
        metric = intent.metric if intent.metric in APPROVED_METRICS else "gross_revenue"
        bridge_map: dict[str, tuple[list[str], str, str, int | None]] = {
            "baseline": ([], "previous_period", "value_desc", None),
            "product_line_bridge": (["product_line"], "previous_period", "change_asc", None),
            "country_bridge": (["country"], "previous_period", "change_asc", None),
            "product_bridge": (["product"], "previous_period", "change_asc", 15),
            "customer_bridge": (["customer"], "previous_period", "change_asc", 15),
            "order_status": (["order_status"], "none", "value_desc", None),
            "top_products": (["product"], "none", "value_desc", intent.n),
            "by_country": (["country"], "none", "value_desc", intent.n),
            "by_product_line": (["product_line"], "none", "value_desc", intent.n),
            "by_customer": (["customer"], "none", "value_desc", intent.n),
        }
        if kind == "trend":
            return QuerySpec(
                metric=metric, dimensions=["month"], comparison="none",  # type: ignore[arg-type]
                period_scope="full_history" if intent.period is None else "plan_period",
                order="value_desc" if intent.n == 1 else "chronological",
                limit=1 if intent.n == 1 else None,
            )
        dims, comp, order, limit = bridge_map.get(kind, ([], "none", "value_desc", None))
        return QuerySpec(metric=metric, dimensions=dims, comparison=comp, order=order, limit=limit)  # type: ignore[arg-type]

    def build_refinement_tasks(self, plan: AnalysisPlan, requests: list[RefinementRequest], completed_task_ids: set[str]) -> list[TaskDraft]:
        drafts: list[TaskDraft] = []
        metric = plan.metrics[0] if plan.metrics else "gross_revenue"
        is_comp = plan.complexity != "lookup"

        for idx, req in enumerate(requests):
            if req.proposed_task is not None:
                drafts.append(req.proposed_task)
                continue
            gap_lower = f"{req.gap} {req.reason}".lower()
            if any(k in gap_lower for k in ("sales_rep", "employee", "rep")):
                drafts.append(TaskDraft(
                    id=f"refine_sales_rep_{idx+1}", title="Sales Representative Variance Breakdown", agent="sql",
                    purpose=req.reason, priority=req.priority,
                    query_spec=QuerySpec(metric=metric, dimensions=["sales_rep"], comparison="previous_period" if is_comp else "none", order="change_asc" if is_comp else "value_desc"),  # type: ignore[arg-type]
                ))
            elif "office" in gap_lower or "territory" in gap_lower:
                drafts.append(TaskDraft(
                    id=f"refine_office_{idx+1}", title="Sales Office Variance Breakdown", agent="sql",
                    purpose=req.reason, priority=req.priority,
                    query_spec=QuerySpec(metric=metric, dimensions=["office"], comparison="previous_period" if is_comp else "none", order="change_asc" if is_comp else "value_desc"),  # type: ignore[arg-type]
                ))
            elif "customer" in gap_lower and "customer_bridge" not in completed_task_ids:
                drafts.append(TaskDraft(
                    id="customer_bridge", title="Top Customer Variance Bridge", agent="sql",
                    purpose=req.reason, priority=req.priority,
                    query_spec=QuerySpec(metric=metric, dimensions=["customer"], comparison="previous_period", order="change_asc", limit=15),  # type: ignore[arg-type]
                ))
            elif "product" in gap_lower and "product_bridge" not in completed_task_ids:
                drafts.append(TaskDraft(
                    id="product_bridge", title="Top Product Variance Bridge", agent="sql",
                    purpose=req.reason, priority=req.priority,
                    query_spec=QuerySpec(metric=metric, dimensions=["product"], comparison="previous_period", order="change_asc", limit=15),  # type: ignore[arg-type]
                ))
            elif "status" in gap_lower and "order_status" not in completed_task_ids:
                drafts.append(TaskDraft(
                    id="order_status", title="Order Status Breakdown", agent="sql",
                    purpose=req.reason, priority=req.priority,
                    query_spec=QuerySpec(metric=metric, dimensions=["order_status"], comparison="none"),  # type: ignore[arg-type]
                ))
            elif req.kind == "statistical_test" and "trend" in completed_task_ids:
                drafts.append(TaskDraft(
                    id=f"refine_trend_stats_{idx+1}", title="Trend & Anomaly Statistical Test", agent="python",
                    purpose=req.reason, depends_on=["trend"], priority=req.priority,
                    python_spec=PythonToolSpec(tool="autonomous_research", mode="autonomous", input_task_ids=["trend"], objective=req.reason),
                ))
            else:
                sql_deps = [t.id for t in plan.tasks if t.agent == "sql" and t.id in completed_task_ids][:2]
                if sql_deps and req.kind in ("statistical_test", "verification", "custom_analysis"):
                    drafts.append(TaskDraft(
                        id=f"refine_autonomous_py_{idx+1}", title=f"Autonomous Follow-Up Analysis ({req.gap})",
                        agent="python", purpose=req.reason, depends_on=sql_deps, priority=req.priority,
                        python_spec=PythonToolSpec(tool="autonomous_research", mode="autonomous", input_task_ids=sql_deps, objective=f"{req.gap}: {req.reason}"),
                    ))
                else:
                    drafts.append(TaskDraft(
                        id=f"refine_autonomous_sql_{idx+1}", title=f"Autonomous SQL Investigation ({req.gap})",
                        agent="sql", purpose=req.reason, priority=req.priority,
                        query_spec=QuerySpec(metric=metric, mode="autonomous", custom_request=f"{req.gap}: {req.reason}"),  # type: ignore[arg-type]
                    ))
        return drafts

    def plan(
        self, intent: QuestionIntent, question: str, route: str, schema_context: str,
    ) -> tuple[LegacyAnalysisPlan, ModelUsageItem | None]:
        """Backward-compatible wrapper returning `(LegacyAnalysisPlan, ModelUsageItem)`."""
        if route in ("SIMPLE_SQL", "CONTEXTUAL_FOLLOWUP") and intent.kind not in ("deep", "comparative"):
            task = PlanTask(id=intent.kind, kind=intent.kind, purpose=f"Execute analytical measurement for {intent.metric} ({intent.kind})")
            return LegacyAnalysisPlan(tasks=[task], hypotheses=[]), None

        system_instructions = (
            "You are a Senior Strategic Data Analyst and Investigation Planner for ClassicModels retail enterprise.\n"
            "Formulate measurable, testable business hypotheses and an investigation task plan.\n"
            "Always include 'baseline' for root-cause analysis and return strictly in the AnalysisPlan schema format."
        )
        plan, usage = self.selector.parse_structured(
            task_type=TaskType.INVESTIGATION_PLANNING, response_model=LegacyAnalysisPlan,
            system_instructions=system_instructions,
            input_payload={"question": question, "intent": intent.model_dump(), "route": route, "available_tasks": AVAILABLE_DEEP_TASKS, "schema_summary": schema_context},
        )
        task_kinds = [t.kind for t in plan.tasks if t.kind in AVAILABLE_DEEP_TASKS]
        if "baseline" not in task_kinds:
            task_kinds.insert(0, "baseline")
        deduped_kinds = list(dict.fromkeys(task_kinds))[:6]
        final_tasks = [PlanTask(id=k, kind=k, purpose=AVAILABLE_DEEP_TASKS.get(k, "Investigate business dimension")) for k in deduped_kinds]
        usage_item = ModelUsageItem(
            agent_role="Investigation Planner", task_type=usage.task_type, model=usage.model,
            reasoning_effort=usage.reasoning_effort, duration_ms=usage.duration_ms,
            tokens=usage.input_tokens + usage.output_tokens, cost_usd=usage.cost_usd,
        )
        return LegacyAnalysisPlan(tasks=final_tasks, hypotheses=plan.hypotheses[:5]), usage_item

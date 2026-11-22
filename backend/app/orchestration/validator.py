"""Deterministic validator and normalizer for AnalysisPlan DAGs."""
from __future__ import annotations
import hashlib
import json
import re
from typing import Any
from app.config import settings
from app.exceptions import PlanValidationError
from app.orchestration.contracts import (
    AnalysisPlan, AnalysisTask, HypothesisSpec, OutputRequirements,
    PlanDraft, PlanValidationReport, QuerySpec, RunBudget,
    TaskDependency, TaskDraft,
)
from app.services.schema_catalog import APPROVED_TABLES
from app.services.semantics import APPROVED_DIMENSIONS, APPROVED_METRICS, resolve_period

APPROVED_PYTHON_TOOLS = {
    "period_comparison", "contribution_analysis", "revenue_decomposition",
    "statistical_summary", "trend_analysis", "anomaly_detection",
    "correlation_analysis", "distribution_analysis", "data_quality_check",
    "forecast_linear", "auto", "autonomous_research", "custom_python",
}


def compute_task_fingerprint(task: TaskDraft | AnalysisTask, period_key: str | None = None) -> str:
    payload: dict[str, Any] = {"agent": task.agent, "period_key": period_key or ""}
    if task.agent == "sql" and task.query_spec:
        payload["query_spec"] = task.query_spec.model_dump()
        if task.query_spec.mode in ("custom", "autonomous") and not task.query_spec.custom_request:
            payload["purpose"] = task.purpose
    elif task.agent == "python" and task.python_spec:
        payload["python_spec"] = task.python_spec.model_dump()
    elif task.agent == "hypothesis" and task.hypothesis_spec:
        payload["hypothesis_spec"] = task.hypothesis_spec.model_dump()
    else:
        payload["id"] = task.id
        payload["purpose"] = task.purpose
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def detect_cycle(tasks: list[TaskDraft | AnalysisTask]) -> list[str] | None:
    adj = {t.id: list(t.depends_on) for t in tasks}
    state = {t.id: 0 for t in tasks}
    stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for dep in adj.get(node, []):
            if dep not in state:
                continue
            if state[dep] == 1:
                return stack[stack.index(dep):] + [dep]
            if state[dep] == 0 and (cyc := dfs(dep)):
                return cyc
        stack.pop()
        state[node] = 2
        return None

    for tid in list(state):
        if state[tid] == 0 and (cyc := dfs(tid)):
            return cyc
    return None


def topological_order(tasks: list[AnalysisTask]) -> list[str]:
    by_id = {t.id: t for t in tasks}
    in_deg = {t.id: 0 for t in tasks}
    dependents: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            if dep in by_id:
                in_deg[t.id] += 1
                dependents[dep].append(t.id)

    ready = sorted([tid for tid, d in in_deg.items() if d == 0], key=lambda x: (by_id[x].priority, x))
    ordered: list[str] = []
    while ready:
        curr = ready.pop(0)
        ordered.append(curr)
        for nxt in dependents.get(curr, []):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                ready.append(nxt)
                ready.sort(key=lambda x: (by_id[x].priority, x))
    return ordered


class PlanValidator:
    def __init__(self, allowed_tables: set[str] | None = None):
        if allowed_tables is not None:
            self.allowed_tables = {t.lower() for t in allowed_tables}
        elif settings.allowed_tables.strip():
            self.allowed_tables = {t.strip().lower() for t in settings.allowed_tables.split(",") if t.strip()} & APPROVED_TABLES
        else:
            self.allowed_tables = set(APPROVED_TABLES)

    def build_budget(
        self,
        requested_queries: int | None = None,
        requested_refinements: int | None = None,
        requested_seconds: int | None = None,
    ) -> RunBudget:
        return RunBudget(
            max_queries=max(1, requested_queries or settings.max_investigation_queries),
            max_refinements=max(0, requested_refinements if requested_refinements is not None else settings.max_plan_refinements),
            max_seconds=max(10, requested_seconds or settings.max_run_seconds),
            max_tokens=settings.max_run_tokens,
            max_cost_usd=settings.max_run_cost_usd,
            max_sql_repairs=settings.max_sql_retries,
            max_parallel_db_queries=settings.max_parallel_db_queries,
            max_parallel_tasks=settings.max_parallel_tasks,
            max_task_retries=settings.max_task_retries,
            task_timeout_seconds=settings.task_timeout_seconds,
        )

    def validate_and_normalize(
        self,
        plan_id: str,
        question: str,
        draft: PlanDraft,
        planner_model: str = "",
        allow_forecasts: bool | None = None,
        allow_reports: bool | None = None,
    ) -> tuple[AnalysisPlan, PlanValidationReport]:
        errors: list[str] = []
        warnings: list[str] = []
        forecasts_ok = settings.allow_forecasting if allow_forecasts is None else allow_forecasts
        reports_ok = settings.allow_reports if allow_reports is None else allow_reports

        valid_metrics = [m for m in draft.metrics if m in APPROVED_METRICS]
        if not valid_metrics:
            errors.append(f"No approved metrics in plan: {draft.metrics}")
            valid_metrics = ["gross_revenue"]
        valid_dims = [d for d in draft.dimensions if d in APPROVED_DIMENSIONS]

        if early_cycle := detect_cycle(draft.tasks):
            errors.append(f"Dependency cycle detected in AnalysisPlan: {' -> '.join(early_cycle)}")

        seen_ids: set[str] = set()
        seen_fps: set[str] = set()
        normalized_tasks: list[AnalysisTask] = []

        for idx, td in enumerate(draft.tasks):
            clean_id = re.sub(r"[^a-zA-Z0-9_]", "_", td.id.strip().lower()).strip("_") or f"task_{idx+1}"
            if clean_id in seen_ids:
                warnings.append(f"Duplicate task ID '{clean_id}' skipped.")
                continue

            if td.agent == "sql":
                qspec = td.query_spec or QuerySpec(
                    metric=valid_metrics[0],  # type: ignore[arg-type]
                    dimensions=valid_dims[:1],  # type: ignore[arg-type]
                    comparison="previous_period" if draft.complexity in ("comparison", "investigation") else "none",
                )
                if qspec.metric not in APPROVED_METRICS:
                    errors.append(f"Task '{clean_id}' references unsupported metric '{qspec.metric}'.")
                if qspec.custom_tables and (unallowed := ({t.lower() for t in qspec.custom_tables} - self.allowed_tables)):
                    errors.append(f"Task '{clean_id}' requests unauthorized tables: {sorted(unallowed)}")
                td = td.model_copy(update={"id": clean_id, "query_spec": qspec})

            elif td.agent == "python":
                pspec = td.python_spec
                if pspec is None:
                    errors.append(f"Python task '{clean_id}' is missing python_spec.")
                    continue
                if pspec.tool not in APPROVED_PYTHON_TOOLS:
                    errors.append(f"Python task '{clean_id}' requested unapproved tool '{pspec.tool}'.")
                if pspec.tool == "forecast_linear" and not forecasts_ok:
                    warnings.append(f"Forecasting task '{clean_id}' skipped.")
                    continue
                merged_deps = list(dict.fromkeys(list(td.depends_on) + list(pspec.input_task_ids)))
                td = td.model_copy(update={"id": clean_id, "depends_on": merged_deps})

            elif td.agent == "hypothesis":
                if not draft.hypotheses_required and draft.complexity == "lookup":
                    warnings.append(f"Removed unnecessary hypothesis task '{clean_id}' for lookup question.")
                    continue
                hspec = td.hypothesis_spec or HypothesisSpec(focus=draft.objective or question, max_hypotheses=4)
                td = td.model_copy(update={"id": clean_id, "hypothesis_spec": hspec})

            fp = compute_task_fingerprint(td, draft.period_key)
            if fp in seen_fps:
                warnings.append(f"Duplicate task fingerprint for '{clean_id}' removed.")
                continue

            seen_ids.add(clean_id)
            seen_fps.add(fp)
            normalized_tasks.append(AnalysisTask(**td.model_dump(), origin="planner", round=0, fingerprint=fp))

        if not normalized_tasks:
            errors.append("Plan contains no executable tasks after validation.")

        budget = self.build_budget(draft.budgets.max_queries, draft.budgets.max_refinements, draft.budgets.max_seconds)
        sql_count = 0
        bounded_tasks: list[AnalysisTask] = []
        for t in normalized_tasks:
            if len(bounded_tasks) >= settings.max_plan_tasks:
                warnings.append(f"Task '{t.id}' dropped due to max_plan_tasks.")
                continue
            if t.agent == "sql":
                if sql_count >= budget.max_queries:
                    warnings.append(f"SQL task '{t.id}' dropped to respect max_queries.")
                    continue
                sql_count += 1
            bounded_tasks.append(t)

        valid_task_ids = {t.id for t in bounded_tasks}
        data_task_ids = {t.id for t in bounded_tasks if t.agent in ("sql", "python")}
        for i, t in enumerate(bounded_tasks):
            clean_deps = [d for d in t.depends_on if d in valid_task_ids and d != t.id]
            if t.agent == "python" and t.python_spec:
                clean_inputs = [d for d in t.python_spec.input_task_ids if d in data_task_ids and d != t.id]
                if not clean_inputs:
                    errors.append(f"Python task '{t.id}' has no valid upstream input tasks.")
                t = t.model_copy(update={
                    "depends_on": clean_deps,
                    "python_spec": t.python_spec.model_copy(update={"input_task_ids": clean_inputs or t.python_spec.input_task_ids}),
                })
            else:
                t = t.model_copy(update={"depends_on": clean_deps})
            bounded_tasks[i] = t

        if cycle := detect_cycle(bounded_tasks):
            errors.append(f"Dependency cycle detected in AnalysisPlan: {' -> '.join(cycle)}")

        output = draft.output
        if output.format == "business_report" and not reports_ok:
            warnings.append("Downgraded output format from business_report to answer_with_visuals.")
            output = OutputRequirements(format="answer_with_visuals", charts=output.charts, include_tables=output.include_tables)

        dep_edges = [
            TaskDependency(task_id=t.id, depends_on=dep, kind="data" if t.agent in ("python", "hypothesis") else "ordering")
            for t in bounded_tasks for dep in t.depends_on
        ]

        plan = AnalysisPlan(
            plan_id=plan_id, version=1, question=question, objective=draft.objective, complexity=draft.complexity,
            metrics=valid_metrics, dimensions=valid_dims, period_key=draft.period_key, period=resolve_period(draft.period_key),
            filters=list(draft.filters), tasks=bounded_tasks, dependencies=dep_edges,
            hypotheses_required=draft.hypotheses_required, hypothesis_rationale=draft.hypothesis_rationale,
            expected_evidence=list(draft.expected_evidence), output=output, budget=budget,
            stopping_conditions=list(draft.stopping_conditions) or ["All required tasks completed."],
            rationale=draft.rationale, warnings=warnings, planner_model=planner_model,
        )

        report = PlanValidationReport(valid=not errors, errors=errors, warnings=warnings)
        if not report.valid:
            raise PlanValidationError(f"PLAN_VALIDATION_FAILED: {'; '.join(errors)}", details={"errors": errors, "warnings": warnings})
        return plan, report

    def validate_dynamic_tasks(
        self,
        existing_plan: AnalysisPlan,
        existing_fingerprints: set[str],
        existing_task_ids: set[str],
        candidate_drafts: list[TaskDraft],
        origin: str,
        round_num: int,
        remaining_query_budget: int,
    ) -> tuple[list[AnalysisTask], list[str]]:
        accepted: list[AnalysisTask] = []
        warnings: list[str] = []
        known_ids = set(existing_task_ids)
        known_fps = set(existing_fingerprints)
        sql_added = 0

        for idx, td in enumerate(candidate_drafts):
            clean_id = re.sub(r"[^a-zA-Z0-9_]", "_", td.id.strip().lower()).strip("_") or f"{origin}_{round_num}_{idx+1}"
            if clean_id in known_ids:
                clean_id = f"{clean_id}_r{round_num}_{idx+1}"

            if td.agent == "sql":
                if sql_added >= remaining_query_budget or not td.query_spec or td.query_spec.metric not in APPROVED_METRICS:
                    warnings.append(f"Skipped dynamic SQL task '{clean_id}'.")
                    continue
                if td.query_spec.custom_tables and ({t.lower() for t in td.query_spec.custom_tables} - self.allowed_tables):
                    warnings.append(f"Skipped dynamic SQL task '{clean_id}': unallowed tables.")
                    continue
            elif td.agent == "python":
                if not td.python_spec or td.python_spec.tool not in APPROVED_PYTHON_TOOLS:
                    warnings.append(f"Skipped dynamic Python task '{clean_id}': invalid python_spec.")
                    continue
                valid_inputs = [d for d in td.python_spec.input_task_ids if d in known_ids]
                if not valid_inputs:
                    warnings.append(f"Skipped dynamic Python task '{clean_id}': unknown input_task_ids.")
                    continue
                td = td.model_copy(update={
                    "depends_on": list(dict.fromkeys(list(td.depends_on) + valid_inputs)),
                    "python_spec": td.python_spec.model_copy(update={"input_task_ids": valid_inputs}),
                })
            elif td.agent == "hypothesis":
                if td.hypothesis_spec is None:
                    td = td.model_copy(update={"hypothesis_spec": HypothesisSpec(focus=td.purpose or td.title, max_hypotheses=4)})
            else:
                continue

            clean_deps = [d for d in td.depends_on if d in known_ids and d != clean_id]
            td = td.model_copy(update={"id": clean_id, "depends_on": clean_deps})
            fp = compute_task_fingerprint(td, existing_plan.period_key)
            if fp in known_fps:
                warnings.append(f"Skipped duplicate dynamic task '{clean_id}'.")
                continue

            task_obj = AnalysisTask(**td.model_dump(), origin=origin, round=round_num, fingerprint=fp)  # type: ignore[arg-type]
            if detect_cycle(list(existing_plan.tasks) + accepted + [task_obj]):
                warnings.append(f"Skipped dynamic task '{clean_id}': dependency cycle.")
                continue

            if task_obj.agent == "sql":
                sql_added += 1
            known_ids.add(clean_id)
            known_fps.add(fp)
            accepted.append(task_obj)

        return accepted, warnings

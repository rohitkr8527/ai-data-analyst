"""Deterministic validator and normalizer for AnalysisPlan DAGs.

Validates:
- Unique task IDs and snake_case formatting
- Valid dependency references and acyclic topological ordering (Kahn / DFS cycle detection)
- Agent-to-spec compatibility (SQL tasks have QuerySpec, Python tasks have PythonToolSpec, etc.)
- Python analysis tasks only depend on upstream SQL or Python tasks
- Approved metrics, dimensions, and table permissions
- Budget clamping against runtime configuration ceilings
- Deterministic task fingerprinting to prevent duplicate execution
"""
from __future__ import annotations
import hashlib
import json
import re
from typing import Any
from app.config import settings
from app.exceptions import PlanValidationError
from app.orchestration.contracts import (
    AnalysisPlan,
    AnalysisTask,
    OutputRequirements,
    PlanDraft,
    PlanValidationReport,
    QuerySpec,
    RunBudget,
    TaskDependency,
    TaskDraft,
)
from app.services.schema_catalog import APPROVED_TABLES
from app.services.semantics import APPROVED_DIMENSIONS, APPROVED_METRICS, resolve_period

APPROVED_PYTHON_TOOLS = {
    "period_comparison",
    "contribution_analysis",
    "revenue_decomposition",
    "statistical_summary",
    "trend_analysis",
    "anomaly_detection",
    "correlation_analysis",
    "distribution_analysis",
    "data_quality_check",
    "forecast_linear",
    "auto",
    "autonomous_research",
    "custom_python",
}


def compute_task_fingerprint(task: TaskDraft | AnalysisTask, period_key: str | None = None) -> str:
    """Compute a deterministic content fingerprint for deduplication across plan and refinement waves."""
    payload: dict[str, Any] = {
        "agent": task.agent,
        "period_key": period_key or "",
    }
    if task.agent == "sql" and task.query_spec:
        payload["query_spec"] = task.query_spec.model_dump()
    elif task.agent == "python" and task.python_spec:
        payload["python_spec"] = task.python_spec.model_dump()
    elif task.agent == "hypothesis" and task.hypothesis_spec:
        payload["hypothesis_spec"] = task.hypothesis_spec.model_dump()
    else:
        payload["id"] = task.id
        payload["purpose"] = task.purpose
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def detect_cycle(tasks: list[TaskDraft | AnalysisTask]) -> list[str] | None:
    """Return a cycle path if the task dependency graph is not a DAG, else None."""
    adj: dict[str, list[str]] = {t.id: list(t.depends_on) for t in tasks}
    state: dict[str, int] = {t.id: 0 for t in tasks}  # 0=unvisited, 1=visiting, 2=done
    stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for dep in adj.get(node, []):
            if dep not in state:
                continue
            if state[dep] == 1:
                idx = stack.index(dep)
                return stack[idx:] + [dep]
            if state[dep] == 0:
                cyc = dfs(dep)
                if cyc:
                    return cyc
        stack.pop()
        state[node] = 2
        return None

    for tid in list(state.keys()):
        if state[tid] == 0:
            cyc = dfs(tid)
            if cyc:
                return cyc
    return None


def topological_order(tasks: list[AnalysisTask]) -> list[str]:
    """Return task IDs in deterministic topological order (priority-aware Kahn algorithm)."""
    by_id = {t.id: t for t in tasks}
    in_deg: dict[str, int] = {t.id: 0 for t in tasks}
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
    """Validates and normalizes a PlanDraft into an executable AnalysisPlan."""

    def __init__(self, allowed_tables: set[str] | None = None):
        if allowed_tables is not None:
            self.allowed_tables = {t.lower() for t in allowed_tables}
        elif settings.allowed_tables.strip():
            self.allowed_tables = {
                t.strip().lower() for t in settings.allowed_tables.split(",") if t.strip()
            } & APPROVED_TABLES
        else:
            self.allowed_tables = set(APPROVED_TABLES)

    def build_budget(self, requested_queries: int | None = None, requested_refinements: int | None = None, requested_seconds: int | None = None) -> RunBudget:
        max_q = min(
            settings.max_investigation_queries,
            max(1, requested_queries or settings.max_investigation_queries),
        )
        max_ref = min(
            settings.max_plan_refinements,
            max(0, requested_refinements if requested_refinements is not None else settings.max_plan_refinements),
        )
        max_sec = min(
            settings.max_run_seconds,
            max(10, requested_seconds or settings.max_run_seconds),
        )
        return RunBudget(
            max_queries=max_q,
            max_refinements=max_ref,
            max_seconds=max_sec,
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

        # 1. Validate metrics & dimensions
        valid_metrics = [m for m in draft.metrics if m in APPROVED_METRICS]
        if not valid_metrics:
            errors.append(f"No approved metrics in plan: {draft.metrics}")
            valid_metrics = ["gross_revenue"]

        valid_dims = [d for d in draft.dimensions if d in APPROVED_DIMENSIONS or d in ("month", "quarter", "year", "sales_rep", "office")]

        # Early cycle detection on raw draft tasks
        early_cycle = detect_cycle(draft.tasks)
        if early_cycle:
            errors.append(f"Dependency cycle detected in AnalysisPlan: {' -> '.join(early_cycle)}")

        # 2. Validate task IDs, specs, permissions, and deduplicate
        seen_ids: set[str] = set()
        seen_fingerprints: set[str] = set()
        normalized_tasks: list[AnalysisTask] = []

        for idx, td in enumerate(draft.tasks):
            clean_id = re.sub(r"[^a-zA-Z0-9_]", "_", td.id.strip().lower()).strip("_") or f"task_{idx+1}"
            if clean_id in seen_ids:
                warnings.append(f"Duplicate task ID '{clean_id}' skipped during plan validation.")
                continue

            # Agent-specific validation
            if td.agent == "sql":
                qspec = td.query_spec
                if qspec is None:
                    qspec = QuerySpec(
                        metric=valid_metrics[0],  # type: ignore[arg-type]
                        dimensions=valid_dims[:1],  # type: ignore[arg-type]
                        comparison="previous_period" if draft.complexity in ("comparison", "investigation") else "none",
                    )
                    warnings.append(f"Task '{clean_id}' missing query_spec; synthesized default QuerySpec.")
                if qspec.metric not in APPROVED_METRICS:
                    errors.append(f"Task '{clean_id}' references unsupported metric '{qspec.metric}'.")
                if qspec.custom_tables:
                    unallowed = {t.lower() for t in qspec.custom_tables} - self.allowed_tables
                    if unallowed:
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
                    warnings.append(f"Forecasting task '{clean_id}' skipped because forecasting is not enabled for this request.")
                    continue
                # Ensure python_spec.input_task_ids are included in depends_on
                merged_deps = list(dict.fromkeys(list(td.depends_on) + list(pspec.input_task_ids)))
                td = td.model_copy(update={"id": clean_id, "depends_on": merged_deps})

            elif td.agent == "hypothesis":
                if not draft.hypotheses_required and draft.complexity == "lookup":
                    warnings.append(f"Removed unnecessary hypothesis task '{clean_id}' for simple lookup question.")
                    continue
                if td.hypothesis_spec is None:
                    from app.orchestration.contracts import HypothesisSpec
                    td = td.model_copy(
                        update={
                            "id": clean_id,
                            "hypothesis_spec": HypothesisSpec(focus=draft.objective or question, max_hypotheses=4),
                        }
                    )
                else:
                    td = td.model_copy(update={"id": clean_id})

            fp = compute_task_fingerprint(td, draft.period_key)
            if fp in seen_fingerprints:
                warnings.append(f"Duplicate task fingerprint for '{clean_id}' removed.")
                continue

            seen_ids.add(clean_id)
            seen_fingerprints.add(fp)
            normalized_tasks.append(
                AnalysisTask(
                    **td.model_dump(),
                    origin="planner",
                    round=0,
                    fingerprint=fp,
                )
            )

        if not normalized_tasks:
            errors.append("Plan contains no executable tasks after validation.")

        # 3. Clean dangling dependencies and enforce SQL budget
        budget = self.build_budget(
            draft.budgets.max_queries,
            draft.budgets.max_refinements,
            draft.budgets.max_seconds,
        )

        sql_count = 0
        bounded_tasks: list[AnalysisTask] = []
        for t in normalized_tasks:
            if len(bounded_tasks) >= settings.max_plan_tasks:
                warnings.append(f"Task '{t.id}' dropped due to max_plan_tasks={settings.max_plan_tasks}.")
                continue
            if t.agent == "sql":
                if sql_count >= budget.max_queries:
                    warnings.append(f"SQL task '{t.id}' dropped to respect max_queries={budget.max_queries}.")
                    continue
                sql_count += 1
            bounded_tasks.append(t)

        valid_task_ids = {t.id for t in bounded_tasks}
        for i, t in enumerate(bounded_tasks):
            clean_deps = [d for d in t.depends_on if d in valid_task_ids and d != t.id]
            if len(clean_deps) != len(t.depends_on):
                dropped = set(t.depends_on) - set(clean_deps)
                warnings.append(f"Task '{t.id}' dropped unknown/self dependencies: {sorted(dropped)}")
            if t.agent == "python" and t.python_spec:
                clean_inputs = [d for d in t.python_spec.input_task_ids if d in valid_task_ids and d != t.id]
                if not clean_inputs:
                    errors.append(f"Python task '{t.id}' has no valid upstream input tasks.")
                t = t.model_copy(
                    update={
                        "depends_on": clean_deps,
                        "python_spec": t.python_spec.model_copy(update={"input_task_ids": clean_inputs or t.python_spec.input_task_ids}),
                    }
                )
            else:
                t = t.model_copy(update={"depends_on": clean_deps})
            bounded_tasks[i] = t

        # 4. Cycle detection
        cycle = detect_cycle(bounded_tasks)
        if cycle:
            errors.append(f"Dependency cycle detected in AnalysisPlan: {' -> '.join(cycle)}")

        # 5. Output requirements permission check
        output = draft.output
        if output.format == "business_report" and not reports_ok:
            warnings.append("Downgraded output format from business_report to answer_with_visuals per permissions.")
            output = OutputRequirements(
                format="answer_with_visuals",
                charts=output.charts,
                include_tables=output.include_tables,
                report_formats=[],
            )

        # Build explicit TaskDependency edges
        dep_edges: list[TaskDependency] = []
        for t in bounded_tasks:
            for dep in t.depends_on:
                dep_edges.append(
                    TaskDependency(
                        task_id=t.id,
                        depends_on=dep,
                        kind="data" if t.agent in ("python", "hypothesis") else "ordering",
                    )
                )

        period_info = resolve_period(draft.period_key)
        stopping = list(draft.stopping_conditions) or [
            "All required tasks completed and Evidence Reviewer confirmed sufficiency.",
            f"Query budget ({budget.max_queries}) or refinement limit ({budget.max_refinements}) reached.",
        ]

        plan = AnalysisPlan(
            plan_id=plan_id,
            version=1,
            question=question,
            objective=draft.objective,
            complexity=draft.complexity,
            metrics=valid_metrics,
            dimensions=valid_dims,
            period_key=draft.period_key,
            period=period_info,
            filters=list(draft.filters),
            tasks=bounded_tasks,
            dependencies=dep_edges,
            hypotheses_required=draft.hypotheses_required,
            hypothesis_rationale=draft.hypothesis_rationale,
            expected_evidence=list(draft.expected_evidence),
            output=output,
            budget=budget,
            stopping_conditions=stopping,
            rationale=draft.rationale,
            warnings=warnings,
            planner_model=planner_model,
        )

        report = PlanValidationReport(valid=(len(errors) == 0), errors=errors, warnings=warnings)
        if not report.valid:
            raise PlanValidationError(
                f"PLAN_VALIDATION_FAILED: {'; '.join(errors)}",
                details={"errors": errors, "warnings": warnings},
            )
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
        """Validate tasks dynamically proposed by the Hypothesis Agent or Refinement Planner."""
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
                if sql_added >= remaining_query_budget:
                    warnings.append(f"Skipped dynamic SQL task '{clean_id}': query budget exhausted.")
                    continue
                if td.query_spec is None:
                    warnings.append(f"Skipped dynamic SQL task '{clean_id}': missing query_spec.")
                    continue
                if td.query_spec.metric not in APPROVED_METRICS:
                    warnings.append(f"Skipped dynamic SQL task '{clean_id}': unapproved metric '{td.query_spec.metric}'.")
                    continue
                if td.query_spec.custom_tables:
                    unallowed = {t.lower() for t in td.query_spec.custom_tables} - self.allowed_tables
                    if unallowed:
                        warnings.append(f"Skipped dynamic SQL task '{clean_id}': unallowed tables {sorted(unallowed)}.")
                        continue
            elif td.agent == "python":
                if td.python_spec is None or td.python_spec.tool not in APPROVED_PYTHON_TOOLS:
                    warnings.append(f"Skipped dynamic Python task '{clean_id}': invalid python_spec.")
                    continue
                valid_inputs = [d for d in td.python_spec.input_task_ids if d in known_ids]
                if not valid_inputs:
                    warnings.append(f"Skipped dynamic Python task '{clean_id}': unknown input_task_ids.")
                    continue
                merged_deps = list(dict.fromkeys(list(td.depends_on) + valid_inputs))
                td = td.model_copy(
                    update={
                        "depends_on": merged_deps,
                        "python_spec": td.python_spec.model_copy(update={"input_task_ids": valid_inputs}),
                    }
                )
            elif td.agent == "hypothesis":
                if td.hypothesis_spec is None:
                    from app.orchestration.contracts import HypothesisSpec
                    td = td.model_copy(
                        update={"hypothesis_spec": HypothesisSpec(focus=td.purpose or td.title, max_hypotheses=4)}
                    )
            else:
                warnings.append(f"Skipped dynamic task '{clean_id}' with unsupported dynamic agent '{td.agent}'.")
                continue

            clean_deps = [d for d in td.depends_on if d in known_ids and d != clean_id]
            td = td.model_copy(update={"id": clean_id, "depends_on": clean_deps})
            fp = compute_task_fingerprint(td, existing_plan.period_key)
            if fp in known_fps:
                warnings.append(f"Skipped duplicate dynamic task '{clean_id}' (fingerprint {fp}).")
                continue

            task_obj = AnalysisTask(
                **td.model_dump(),
                origin=origin,  # type: ignore[arg-type]
                round=round_num,
                fingerprint=fp,
            )
            # Check that adding this task does not introduce a cycle
            if detect_cycle(list(existing_plan.tasks) + accepted + [task_obj]):
                warnings.append(f"Skipped dynamic task '{clean_id}' because it would create a dependency cycle.")
                continue

            if task_obj.agent == "sql":
                sql_added += 1
            known_ids.add(clean_id)
            known_fps.add(fp)
            accepted.append(task_obj)

        return accepted, warnings

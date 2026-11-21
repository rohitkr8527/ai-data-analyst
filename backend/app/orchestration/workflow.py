"""Dynamic Supervisor-Worker LangGraph Orchestration with Evidence-Driven Refinement."""
from __future__ import annotations
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable
from uuid import uuid4
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.evidence_reviewer import EvidenceReviewerAgent
from app.agents.hypothesis import HypothesisAgent
from app.agents.planner import PlannerAgent
from app.agents.python_analyst import PythonAnalysisAgent
from app.agents.report_agent import ReportGenerationAgent
from app.agents.reporter import ReporterAgent
from app.agents.router import JevRoutingDecision, OpenRouterJevRouter, RouteType
from app.agents.sql_analyst import SQLAnalyst
from app.config import settings
from app.exceptions import (
    DatabaseConnectionError, ProviderAuthError, ProviderError,
    ProviderRateLimitError, QuerySecurityError,
)
from app.orchestration.contracts import (
    TERMINAL_STATUSES, AnalysisPlan, AnalysisTask, EvidenceRecord,
    EvidenceReview, PlanDraft, QuerySpec, TaskResult, TaskStatus,
)
from app.orchestration.state import AnalysisState
from app.orchestration.validator import PlanValidator
from app.schemas import ModelUsageItem, PlanTask, QuestionIntent, Report
from app.services.dataset_broker import AuthorizedDatasetBroker
from app.services.dataset_store import DatasetStore
from app.services.db import ReadOnlyGateway
from app.services.evidence_store import SharedEvidenceStore
from app.services.models import ModelSelector
from app.services.sandbox import SandboxExecutionService
from app.services.schema_catalog import APPROVED_TABLES
from app.services.semantics import DATA_MAX_DATE, DATA_MIN_DATE, extract_intent, resolve_period
from app.services.telemetry import cancellation_registry, trace_span

logger = logging.getLogger(__name__)
WorkflowState = AnalysisState


class AnalyticalWorkflow:
    """Dynamic Supervisor-Worker multi-agent analytical workflow powered by LangGraph."""

    def __init__(
        self,
        gateway: ReadOnlyGateway | None = None,
        model_selector: ModelSelector | None = None,
        dataset_store: DatasetStore | None = None,
        checkpoint_path: str | None = None,
        sandbox: SandboxExecutionService | None = None,
        evidence_store: SharedEvidenceStore | None = None,
    ):
        self.gateway = gateway or ReadOnlyGateway()
        self.model_selector = model_selector or ModelSelector()
        self.dataset_store = dataset_store or DatasetStore()
        self.sandbox = sandbox or SandboxExecutionService()
        ds_path_str = str(getattr(self.dataset_store, "path", None) or settings.dataset_store_path)
        self.evidence_store = evidence_store or SharedEvidenceStore(ds_path_str)
        self.dataset_broker = AuthorizedDatasetBroker(
            self.gateway, self.dataset_store, getattr(self.gateway, "catalog", None), self.evidence_store,
        )
        self.validator = PlanValidator()
        self.router = OpenRouterJevRouter()
        self.planner = PlannerAgent(self.model_selector)
        self.sql_analyst = SQLAnalyst(self.model_selector)
        self.hypothesis_agent = HypothesisAgent(self.model_selector)
        self.python_agent = PythonAnalysisAgent(
            self.dataset_store, self.model_selector,
            sandbox=self.sandbox, dataset_broker=self.dataset_broker, evidence_store=self.evidence_store,
        )
        self.reviewer = EvidenceReviewerAgent(self.dataset_store, self.model_selector, sandbox=self.sandbox)
        self.reporter = ReporterAgent(self.model_selector)
        self.report_agent = ReportGenerationAgent(self.model_selector)

        cp_file = Path(checkpoint_path or settings.checkpoint_database_path).resolve()
        cp_file.parent.mkdir(parents=True, exist_ok=True)
        self._cp_conn = sqlite3.connect(str(cp_file), check_same_thread=False, timeout=20.0)
        self.checkpointer = SqliteSaver(self._cp_conn)
        self.checkpointer.setup()

        self._event_listeners: dict[str, Callable[[dict[str, Any]], None]] = {}
        self.graph = self._build_graph()

    def _emit_event(
        self, run_id: str, stage: str, message: str, status: str = "ok",
        task_id: str | None = None, extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evt: dict[str, Any] = {"stage": stage, "message": message, "status": status, "timestamp": time.time()}
        if task_id:
            evt["task_id"] = task_id
        if extra:
            evt.update(extra)
        cb = self._event_listeners.get(run_id)
        if cb:
            try:
                cb(evt)
            except Exception:
                pass
        return evt

    @staticmethod
    def _event(stage: str, message: str, status: str = "ok") -> dict[str, Any]:
        return {"stage": stage, "message": message, "status": status, "timestamp": time.time()}

    def _task_failure(
        self, run_id: str, task: AnalysisTask, agent: str, err_msg: str,
        stage: str | None = None, error_code: str | None = None,
        sql: str | None = None, repair_attempts: int = 0,
        models_used: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        fail_res = TaskResult(
            task_id=task.id, agent=agent, status="FAILED", fingerprint=task.fingerprint,
            sql=sql, repair_attempts=repair_attempts, error=err_msg, error_code=error_code,
        )
        evt = self._emit_event(run_id, stage or task.id, err_msg, "error", task_id=task.id)
        out: dict[str, Any] = {
            "tasks": {task.id: {"status": TaskStatus.FAILED.value, "error": err_msg}},
            "task_results": {task.id: fail_res.model_dump()},
            "errors": [{"task": task.id, "error": err_msg}],
            "trace": [evt],
        }
        if models_used is not None:
            out["models_used"] = models_used
        return out

    def prepare_context(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id") or str(uuid4())
        question = state["question"]
        prev_context = state.get("previous_context", {})
        with trace_span("node.prepare_context", run_id=run_id):
            intent = extract_intent(question, prev_context.get("last_intent"))
            allowed_tables = sorted(self.validator.allowed_tables & APPROVED_TABLES)
            schema_ctx = self.gateway.catalog.format_llm_schema_context(allowed_tables)
            permissions = {
                "allowed_tables": allowed_tables, "allow_reports": settings.allow_reports,
                "allow_forecasting": settings.allow_forecasting, "read_only": True,
            }
            evt = self._emit_event(
                run_id, "prepare",
                f"Resolved metric: '{intent.metric}', period: '{intent.period or 'default'}', tables: {len(allowed_tables)}",
            )
        return {
            "run_id": run_id, "started_at": state.get("started_at") or time.monotonic(),
            "intent": intent.model_dump(), "permissions": permissions, "schema_context": schema_ctx,
            "refinement_round": 0, "budget_state": {"queries_executed": 0, "tokens_used": 0, "cost_usd": 0.0},
            "trace": [evt],
        }

    def route_request(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        allowed_tables = state.get("permissions", {}).get("allowed_tables") or list(self.gateway.catalog.get_snapshot().tables.keys())
        with trace_span("node.jev_route", run_id=run_id):
            decision: JevRoutingDecision = self.router.route(
                question=state["question"], context=state.get("previous_context", {}), schema_summary=allowed_tables,
            )
        tokens = decision.input_tokens + decision.output_tokens
        model_item = {
            "agent_role": "Jev Decision Router", "task_type": "decision_routing",
            "model": settings.openrouter_jev_model, "reasoning_effort": "decision_choice",
            "duration_ms": decision.duration_ms, "tokens": tokens, "cost_usd": decision.cost,
        }
        prev_budget = state.get("budget_state", {})
        evt = self._emit_event(run_id, "jev_route", f"Jev route: {decision.route.value} (confidence: {decision.confidence:.2f})")
        return {
            "route_decision": {
                "route": decision.route.value, "confidence": decision.confidence,
                "probabilities": decision.probabilities, "complexity_hint": decision.complexity_hint,
                "requires_hypotheses_hint": decision.requires_hypotheses_hint,
                "preferred_output_hint": decision.preferred_output_hint, "duration_ms": decision.duration_ms,
            },
            "models_used": [model_item],
            "budget_state": {
                "tokens_used": int(prev_budget.get("tokens_used", 0)) + tokens,
                "cost_usd": round(float(prev_budget.get("cost_usd", 0.0)) + decision.cost, 6),
            },
            "trace": [evt],
        }

    def handle_clarification(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        period_info = resolve_period(state.get("intent", {}).get("period"))
        report = Report(
            title="Clarification Required",
            summary=(
                "Could you clarify your analytical request? For example, you can ask:\n"
                "- 'Show the top 10 products by revenue in 2004'\n"
                "- 'Compare 2004 sales by product line'\n"
                "- 'Why did sales decline in November 2004 compared with October 2004?'\n"
                f"Note: ClassicModels contains historical data spanning {DATA_MIN_DATE} to {DATA_MAX_DATE}."
            ),
            status="clarification_required", output_mode="direct_answer",
            period=period_info["target_label"], comparison=period_info["previous_label"],
            limitations=[f"Database scope is restricted to ClassicModels records from {DATA_MIN_DATE} to {DATA_MAX_DATE}."],
        )
        return {"report": report.model_dump(), "trace": [self._emit_event(run_id, "clarification", "Requested user clarification on analytical intent")]}

    def handle_unsupported(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        report = Report(
            title="Unsupported Request",
            summary=(
                "This question cannot be answered using the ClassicModels sales, orders, "
                "customers, and inventory database. Please ask a question related to sales performance, "
                "product line revenue, customer orders, or historical variance."
            ),
            status="unsupported", output_mode="direct_answer",
            limitations=["Out of domain analytical request."],
        )
        return {"report": report.model_dump(), "trace": [self._emit_event(run_id, "unsupported", "Request categorized as out-of-domain by Jev")]}

    def plan_investigation(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        intent = QuestionIntent.model_validate(state["intent"])
        self._emit_event(run_id, "planning", "Planning Agent designing dynamic task DAG...")
        with trace_span("node.planning_agent", run_id=run_id):
            draft, usage_item = self.planner.create_plan_draft(
                question=state["question"], intent=intent,
                route_decision=state.get("route_decision", {"route": "SIMPLE_SQL"}),
                schema_context=state["schema_context"],
                previous_context=state.get("previous_context", {}),
                permissions=state.get("permissions", {}),
            )
        prev_budget = state.get("budget_state", {})
        evt = self._emit_event(
            run_id, "plan",
            f"Planning Agent drafted {len(draft.tasks)} tasks (complexity={draft.complexity}, hypotheses={draft.hypotheses_required})",
        )
        return {
            "plan": draft.model_dump(),
            "models_used": [usage_item.model_dump()],
            "budget_state": {
                "tokens_used": int(prev_budget.get("tokens_used", 0)) + usage_item.tokens,
                "cost_usd": round(float(prev_budget.get("cost_usd", 0.0)) + usage_item.cost_usd, 6),
            },
            "trace": [evt],
        }

    def validate_plan_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        draft = PlanDraft.model_validate(state["plan"])
        permissions = state.get("permissions", {})
        with trace_span("node.validate_plan", run_id=run_id):
            plan, val_report = self.validator.validate_and_normalize(
                plan_id=f"plan_{run_id[:8]}", question=state["question"], draft=draft,
                planner_model=settings.openai_planner_model,
                allow_forecasts=permissions.get("allow_forecasting", True),
                allow_reports=permissions.get("allow_reports", True),
            )
        task_map = {
            t.id: {
                "task": t.model_dump(), "status": TaskStatus.PENDING.value, "attempts": 0,
                "depends_on": list(t.depends_on), "priority": t.priority, "agent": t.agent,
            }
            for t in plan.tasks
        }
        evt = self._emit_event(run_id, "validate_plan", f"Validated DAG with {len(plan.tasks)} tasks and {len(plan.dependencies)} dependency edges")
        return {"plan": plan.model_dump(), "plan_validation": val_report.model_dump(), "tasks": task_map, "trace": [evt]}

    def supervisor_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        plan = AnalysisPlan.model_validate(state["plan"])
        tasks_map = {k: dict(v) for k, v in state.get("tasks", {}).items()}
        task_results = state.get("task_results", {})
        budget_state = state.get("budget_state", {})
        elapsed_s = time.monotonic() - float(state.get("started_at") or time.monotonic())

        updates: dict[str, dict[str, Any]] = {}
        trace_events: list[dict[str, Any]] = []

        is_cancelled = cancellation_registry.is_cancelled(run_id)
        is_timed_out = elapsed_s >= plan.budget.max_seconds
        queries_used = int(budget_state.get("queries_executed", 0))
        tokens_used = int(budget_state.get("tokens_used", 0))
        cost_used = float(budget_state.get("cost_usd", 0.0))

        if is_cancelled or is_timed_out:
            reason = cancellation_registry.reason(run_id) if is_cancelled else f"Run exceeded wall-clock budget ({plan.budget.max_seconds}s)"
            for tid, entry in tasks_map.items():
                if entry["status"] not in TERMINAL_STATUSES:
                    updates[tid] = {"status": TaskStatus.CANCELLED.value, "error": reason}
            trace_events.append(self._emit_event(run_id, "supervisor", f"Stopping task scheduling: {reason}", "warning"))
            return {"tasks": updates, "stop_reason": reason, "trace": trace_events}

        existing_ids = set(tasks_map.keys())
        existing_fps = {
            entry["task"].get("fingerprint", "") for entry in tasks_map.values() if isinstance(entry.get("task"), dict)
        } | set(state.get("completed_fingerprints", {}).keys())

        candidate_drafts = [
            pt for raw_res in task_results.values()
            if (res := TaskResult.model_validate(raw_res)).agent == "hypothesis" and res.proposed_tasks
            for pt in res.proposed_tasks if pt.id not in existing_ids
        ]
        if candidate_drafts:
            accepted, _ = self.validator.validate_dynamic_tasks(
                existing_plan=plan, existing_fingerprints=existing_fps, existing_task_ids=existing_ids,
                candidate_drafts=candidate_drafts, origin="hypothesis",
                round_num=int(state.get("refinement_round", 0)),
                remaining_query_budget=max(0, plan.budget.max_queries - queries_used),
            )
            for new_task in accepted:
                tasks_map[new_task.id] = {
                    "task": new_task.model_dump(), "status": TaskStatus.PENDING.value, "attempts": 0,
                    "depends_on": list(new_task.depends_on), "priority": new_task.priority, "agent": new_task.agent,
                }
                updates[new_task.id] = tasks_map[new_task.id]
                plan.tasks.append(new_task)
                trace_events.append(self._emit_event(run_id, "supervisor", f"Scheduled hypothesis test task '{new_task.id}' ({new_task.agent})", task_id=new_task.id))

        for tid, entry in list(tasks_map.items()):
            if entry["status"] in TERMINAL_STATUSES:
                continue
            failed_deps = [
                d for d in entry.get("depends_on", [])
                if d in tasks_map and tasks_map[d]["status"] in (TaskStatus.FAILED.value, TaskStatus.SKIPPED.value, TaskStatus.CANCELLED.value)
            ]
            if failed_deps:
                entry["status"] = TaskStatus.SKIPPED.value
                updates[tid] = {"status": TaskStatus.SKIPPED.value, "error": f"Skipped due to upstream failure: {failed_deps}"}
                trace_events.append(self._emit_event(run_id, "supervisor", f"Skipped task '{tid}' (upstream {failed_deps} unavailable)", "warning", task_id=tid))

        ready_candidates = sorted(
            [
                (int(entry.get("priority", 3)), tid, entry)
                for tid, entry in tasks_map.items()
                if entry["status"] == TaskStatus.PENDING.value
                and all(d in tasks_map and tasks_map[d]["status"] == TaskStatus.COMPLETED.value for d in entry.get("depends_on", []))
            ],
            key=lambda item: (item[0], item[1]),
        )

        dispatched_ids: list[str] = []
        sql_in_wave = 0
        rem_query_budget = max(0, plan.budget.max_queries - queries_used)
        budget_exceeded = tokens_used >= plan.budget.max_tokens or (plan.budget.max_cost_usd > 0 and cost_used >= plan.budget.max_cost_usd)

        for _, tid, entry in ready_candidates:
            if budget_exceeded:
                updates[tid] = {"status": TaskStatus.SKIPPED.value, "error": "Skipped due to token/cost budget exhaustion."}
                continue
            if entry.get("agent", "sql") == "sql":
                if sql_in_wave >= rem_query_budget:
                    updates[tid] = {"status": TaskStatus.SKIPPED.value, "error": f"Skipped: max_queries budget ({plan.budget.max_queries}) reached."}
                    trace_events.append(self._emit_event(run_id, "budget", f"Task '{tid}' skipped: query budget ({plan.budget.max_queries}) exhausted", "warning", task_id=tid))
                    continue
                if sql_in_wave >= plan.budget.max_parallel_db_queries:
                    continue
                sql_in_wave += 1
            if len(dispatched_ids) >= plan.budget.max_parallel_tasks:
                continue
            updates[tid] = {"status": TaskStatus.RUNNING.value, "attempts": int(entry.get("attempts", 0)) + 1, "started_at": time.time()}
            dispatched_ids.append(tid)

        if dispatched_ids:
            trace_events.append(self._emit_event(run_id, "supervisor", f"Dispatching {len(dispatched_ids)} ready task(s) in parallel: {', '.join(dispatched_ids)}"))
        return {"plan": plan.model_dump(), "tasks": updates, "trace": trace_events}

    def supervisor_router(self, state: AnalysisState):
        tasks_map = state.get("tasks", {})
        plan_dict = state.get("plan", {})
        sends: list[Send] = []
        worker_map = {"sql": "sql_worker", "hypothesis": "hypothesis_worker", "python": "python_worker"}
        for entry in tasks_map.values():
            if entry.get("status") != TaskStatus.RUNNING.value:
                continue
            target = worker_map.get(entry.get("agent", "sql"))
            if target:
                sends.append(Send(target, {
                    "run_id": state.get("run_id", ""), "task": entry["task"], "attempt": entry.get("attempts", 1),
                    "plan": plan_dict, "intent": state.get("intent", {}), "permissions": state.get("permissions", {}),
                    "schema_context": state.get("schema_context", ""), "task_results": state.get("task_results", {}),
                    "completed_fingerprints": state.get("completed_fingerprints", {}),
                }))
        return sends if sends else "aggregate_evidence"

    def sql_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload.get("run_id", "")
        task = AnalysisTask.model_validate(payload["task"])
        plan = AnalysisPlan.model_validate(payload["plan"])
        intent = QuestionIntent.model_validate(payload["intent"])
        schema_ctx = payload.get("schema_context", "")
        allowed_tables = set(payload.get("permissions", {}).get("allowed_tables") or APPROVED_TABLES)
        completed_fps: dict[str, str] = payload.get("completed_fingerprints", {})
        existing_results: dict[str, dict[str, Any]] = payload.get("task_results", {})

        if task.fingerprint and task.fingerprint in completed_fps:
            prev_tid = completed_fps[task.fingerprint]
            if prev_tid in existing_results:
                prev_res = TaskResult.model_validate(existing_results[prev_tid])
                if prev_res.status == "COMPLETED" and prev_res.dataset:
                    reused_res = prev_res.model_copy(update={"task_id": task.id, "reused": True})
                    evt = self._emit_event(run_id, f"sql_{task.id}", f"Reused cached dataset from '{prev_tid}'", task_id=task.id)
                    return {
                        "tasks": {task.id: {"status": TaskStatus.COMPLETED.value}},
                        "task_results": {task.id: reused_res.model_dump()},
                        "trace": [evt],
                    }

        self._emit_event(run_id, f"sql_{task.id}", f"Generating & executing SQL for '{task.title}'...", task_id=task.id)
        models_used: list[dict[str, Any]] = []

        def _complete_sql(sql_str: str, params_dict: dict[str, Any], q_res: dict[str, Any], ds_ref, source: str, repairs: int, steps: list[dict[str, Any]] | None = None, msg_prefix: str = "Fetched") -> dict[str, Any]:
            res = TaskResult(
                task_id=task.id, agent="sql", status="COMPLETED", fingerprint=task.fingerprint,
                dataset=ds_ref, query_id=q_res["id"], sql=sql_str, params=params_dict,
                sql_source=source, repair_attempts=repairs, execution_steps=steps or [], duration_ms=q_res["duration_ms"],
            )
            raw_ev = {
                "id": task.id, "task_id": task.id, "query_id": q_res["id"], "purpose": task.purpose,
                "columns": q_res["columns"], "rows": q_res["rows"], "sql": sql_str,
                "duration_ms": q_res["duration_ms"], "dataset_id": ds_ref.dataset_id,
            }
            evt = self._emit_event(run_id, f"exec_{task.id}", f"{msg_prefix} {len(q_res['rows'])} rows in {q_res['duration_ms']} ms", task_id=task.id)
            return {
                "tasks": {task.id: {"status": TaskStatus.COMPLETED.value, "duration_ms": q_res["duration_ms"]}},
                "task_results": {task.id: res.model_dump()},
                "completed_fingerprints": {task.fingerprint: task.id} if task.fingerprint else {},
                "raw_evidence": [raw_ev], "models_used": models_used, "trace": [evt],
            }

        try:
            with trace_span("worker.sql", run_id=run_id, task_id=task.id):
                if task.query_spec and task.query_spec.mode == "autonomous":
                    final_sql, final_params, q_res, ds_ref, sql_steps, sql_usages, repair_count = self.sql_analyst.investigate_sql(
                        task=task, gateway=self.gateway, dataset_store=self.dataset_store,
                        period_key=plan.period_key, plan_filters=plan.filters, schema_context=schema_ctx,
                        allowed_tables=allowed_tables, run_id=run_id, on_event=self._event_listeners.get(run_id),
                    )
                    models_used.extend(u.model_dump() for u in sql_usages)
                    return _complete_sql(final_sql, final_params, q_res, ds_ref, "autonomous_research", repair_count, sql_steps, "Autonomous SQL explored and fetched")

                sql, params, sql_source, draft_usage = self.sql_analyst.draft_for_task(
                    task=task, period_key=plan.period_key, plan_filters=plan.filters, schema_context=schema_ctx,
                )
                if draft_usage:
                    models_used.append(draft_usage.model_dump())

                max_retries = plan.budget.max_sql_repairs
                repair_count = 0
                for attempt in range(max_retries + 1):
                    try:
                        q_res = self.gateway.execute(sql, params, trusted_template=False, allowed_tables=allowed_tables, run_id=run_id, task_id=task.id)
                        ds_ref = self.dataset_store.put(run_id=run_id, task_id=task.id, columns=q_res["columns"], rows=q_res["rows"], query_id=q_res["id"], sql_hash=q_res["query_hash"])
                        return _complete_sql(sql, params, q_res, ds_ref, sql_source, repair_count)
                    except QuerySecurityError as sec_err:
                        err_msg = str(sec_err)
                        repairable = any(tok in err_msg.lower() for tok in ("syntax", "unknown column", "select *", "parsing failed", "database query error", "rejected query", "only select", "exceeded maximum row limit", "statement separators"))
                        if attempt < max_retries and repairable:
                            repair_count += 1
                            logger.warning("Attempting SQL repair (%d/%d) for %s: %s", repair_count, max_retries, task.id, err_msg)
                            sql, rep_usage = self.sql_analyst.repair(
                                rejected_sql=sql, error_message=err_msg, task=task, intent=intent,
                                schema_context=schema_ctx, available_params=list(params.keys()),
                            )
                            models_used.append(rep_usage.model_dump())
                            continue
                        return self._task_failure(run_id, task, "sql", err_msg, error_code="QUERY_SECURITY_VIOLATION", sql=sql, repair_attempts=repair_count, models_used=models_used)

        except (ProviderAuthError, ProviderRateLimitError, DatabaseConnectionError):
            raise
        except Exception as exc:
            err_text = str(exc) if isinstance(exc, (QuerySecurityError, ProviderError)) else f"{type(exc).__name__}: {exc}"
            return self._task_failure(run_id, task, "sql", err_text, error_code="TASK_EXECUTION_ERROR", models_used=models_used)
        return {}

    def hypothesis_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload.get("run_id", "")
        task = AnalysisTask.model_validate(payload["task"])
        plan = AnalysisPlan.model_validate(payload["plan"])
        task_results: dict[str, dict[str, Any]] = payload.get("task_results", {})

        self._emit_event(run_id, f"hypothesis_{task.id}", "Generating measurable hypotheses...", task_id=task.id)
        upstream_summaries = [
            {"task_id": dep_id, "columns": dep_res.dataset.columns, "row_count": dep_res.dataset.row_count, "preview": dep_res.dataset.preview}
            for dep_id in task.depends_on if dep_id in task_results and (dep_res := TaskResult.model_validate(task_results[dep_id])).dataset
        ]
        try:
            with trace_span("worker.hypothesis", run_id=run_id, task_id=task.id):
                existing_ids = [t.id for t in plan.tasks]
                hyp_records, proposed_drafts, usage_item = self.hypothesis_agent.generate(
                    task=task, plan=plan, upstream_summaries=upstream_summaries, existing_task_ids=existing_ids,
                )
            accepted_proposed, _ = self.validator.validate_dynamic_tasks(
                existing_plan=plan, existing_fingerprints={t.fingerprint for t in plan.tasks},
                existing_task_ids=set(existing_ids), candidate_drafts=proposed_drafts,
                origin="hypothesis", round_num=0, remaining_query_budget=2,
            )
            res = TaskResult(
                task_id=task.id, agent="hypothesis", status="COMPLETED", fingerprint=task.fingerprint,
                hypotheses=hyp_records, proposed_tasks=accepted_proposed, duration_ms=usage_item.duration_ms,
            )
            evt = self._emit_event(run_id, f"hypothesis_{task.id}", f"Generated {len(hyp_records)} testable hypotheses ({len(accepted_proposed)} additional test tasks)", task_id=task.id)
            return {
                "tasks": {task.id: {"status": TaskStatus.COMPLETED.value, "duration_ms": usage_item.duration_ms}},
                "task_results": {task.id: res.model_dump()},
                "hypotheses": [h.model_dump() for h in hyp_records],
                "models_used": [usage_item.model_dump()], "trace": [evt],
            }
        except (ProviderAuthError, ProviderRateLimitError):
            raise
        except Exception as exc:
            return self._task_failure(run_id, task, "hypothesis", f"{type(exc).__name__}: {exc}", stage=f"hypothesis_{task.id}")

    def python_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload.get("run_id", "")
        task = AnalysisTask.model_validate(payload["task"])
        plan = AnalysisPlan.model_validate(payload["plan"])
        schema_ctx = payload.get("schema_context", "")
        task_results: dict[str, dict[str, Any]] = payload.get("task_results", {})

        input_tids = task.python_spec.input_task_ids if task.python_spec else task.depends_on
        dataset_ids = [
            dep_res.dataset.dataset_id for ut in input_tids
            if ut in task_results and (dep_res := TaskResult.model_validate(task_results[ut])).dataset
        ]
        is_auto = self.python_agent.should_use_autonomous_mode(task)
        self._emit_event(run_id, f"python_{task.id}", f"Running {'autonomous Python research' if is_auto else 'deterministic Python analysis'} ({task.title})...", task_id=task.id)
        try:
            if is_auto:
                def _sql_provider(req_text: str) -> tuple[str, dict[str, Any]]:
                    helper_task = AnalysisTask(
                        id=f"{task.id}_data_req", title=req_text[:80], agent="sql", purpose=req_text,
                        query_spec=QuerySpec(metric=(plan.metrics[0] if plan.metrics else "gross_revenue"), mode="custom", custom_request=req_text),
                    )
                    sql, params, _, _ = self.sql_analyst.draft_for_task(
                        task=helper_task, period_key=plan.period_key, plan_filters=plan.filters, schema_context=schema_ctx,
                    )
                    return sql, params

                with trace_span("worker.python.autonomous", run_id=run_id, task_id=task.id):
                    inv = self.python_agent.investigate(
                        task=task, upstream_dataset_ids=dataset_ids, run_id=run_id or "run_py",
                        sql_provider=_sql_provider, on_event=self._event_listeners.get(run_id),
                    )
                duration_ms = sum(int(s.get("duration_ms", 0)) for s in inv.execution_steps) or 1
                if inv.status == "FAILED":
                    raise RuntimeError(inv.error or "Autonomous Python investigation failed.")
                res = TaskResult(
                    task_id=task.id, agent="python", status="COMPLETED", fingerprint=task.fingerprint,
                    tool=inv.tool_name, calculation=inv.calculation, generated_code=inv.generated_code,
                    methodology=inv.methodology, execution_steps=inv.execution_steps, artifacts=inv.artifacts,
                    assumptions_checked=inv.assumptions_checked, repair_attempts=inv.repair_attempts, duration_ms=duration_ms,
                )
                evt = self._emit_event(run_id, f"python_{task.id}", f"Completed autonomous Python ({inv.methodology}) in {len(inv.execution_steps)} step(s)", task_id=task.id)
                return {
                    "tasks": {task.id: {"status": TaskStatus.COMPLETED.value, "duration_ms": duration_ms}},
                    "task_results": {task.id: res.model_dump()},
                    "models_used": [u.model_dump() for u in inv.usages], "trace": [evt],
                }

            with trace_span("worker.python", run_id=run_id, task_id=task.id):
                tool_name, calc_dict, usage_item, duration_ms = self.python_agent.execute(task=task, upstream_dataset_ids=dataset_ids)
            res = TaskResult(task_id=task.id, agent="python", status="COMPLETED", fingerprint=task.fingerprint, tool=tool_name, calculation=calc_dict, duration_ms=duration_ms)
            evt = self._emit_event(run_id, f"python_{task.id}", f"Completed Python tool '{tool_name}' in {duration_ms} ms", task_id=task.id)
            return {
                "tasks": {task.id: {"status": TaskStatus.COMPLETED.value, "duration_ms": duration_ms}},
                "task_results": {task.id: res.model_dump()},
                "models_used": [usage_item.model_dump()] if usage_item else [], "trace": [evt],
            }
        except Exception as exc:
            return self._task_failure(run_id, task, "python", f"{type(exc).__name__}: {exc}", stage=f"python_{task.id}")

    def aggregate_evidence_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        plan = AnalysisPlan.model_validate(state["plan"])
        task_results = state.get("task_results", {})
        with trace_span("node.aggregate_evidence", run_id=run_id):
            ev_records, eval_hyps, _, _ = self.reviewer.aggregate(plan=plan, task_results=task_results, existing_hypotheses=state.get("hypotheses", []))
        sql_completed = sum(1 for r in task_results.values() if r.get("agent") == "sql" and r.get("status") == "COMPLETED" and not r.get("reused"))
        total_tokens = sum(int(m.get("tokens", 0)) for m in state.get("models_used", []))
        total_cost = round(sum(float(m.get("cost_usd", 0.0)) for m in state.get("models_used", [])), 6)
        evt = self._emit_event(run_id, "aggregate_evidence", f"Aggregated {len(ev_records)} evidence records ({sql_completed} SQL queries, {len(eval_hyps)} evaluated hypotheses)")
        return {
            "evidence_records": [e.model_dump() for e in ev_records],
            "hypotheses": [h.model_dump() for h in eval_hyps],
            "budget_state": {"queries_executed": sql_completed, "tokens_used": total_tokens, "cost_usd": total_cost},
            "trace": [evt],
        }

    def review_evidence_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        plan = AnalysisPlan.model_validate(state["plan"])
        refinement_round = int(state.get("refinement_round", 0))
        self._emit_event(run_id, "review_evidence", f"Reviewing evidence sufficiency (round {refinement_round})...")
        with trace_span("node.review_evidence", run_id=run_id):
            ev_records, eval_hyps, datasets_by_task, baseline_delta = self.reviewer.aggregate(
                plan=plan, task_results=state.get("task_results", {}), existing_hypotheses=state.get("hypotheses", []),
            )
            review, usage_item = self.reviewer.review(
                plan=plan, evidence_records=ev_records, hypotheses=eval_hyps,
                datasets_by_task=datasets_by_task, baseline_delta=baseline_delta,
                task_errors=state.get("errors", []), refinement_round=refinement_round,
                budget_state=state.get("budget_state", {}),
            )
        evt = self._emit_event(run_id, "evidence_review", f"Evidence Reviewer decision: {review.decision} (sufficient={review.sufficient})")
        return {"evidence_review": review.model_dump(), "models_used": [usage_item.model_dump()] if usage_item else [], "trace": [evt]}

    def review_router(self, state: AnalysisState) -> str:
        review = state.get("evidence_review", {})
        return "refinement_planner" if (review.get("decision") == "refine" and review.get("refinement_requests")) else "insight_agent"

    def refinement_planner_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        plan = AnalysisPlan.model_validate(state["plan"])
        review = EvidenceReview.model_validate(state["evidence_review"])
        tasks_map = state.get("tasks", {})
        next_round = int(state.get("refinement_round", 0)) + 1

        completed_ids = {tid for tid, entry in tasks_map.items() if entry.get("status") == TaskStatus.COMPLETED.value}
        candidate_drafts = self.planner.build_refinement_tasks(plan=plan, requests=review.refinement_requests, completed_task_ids=completed_ids)
        existing_fps = {t.fingerprint for t in plan.tasks} | set(state.get("completed_fingerprints", {}).keys())
        rem_queries = max(0, plan.budget.max_queries - int(state.get("budget_state", {}).get("queries_executed", 0)))

        accepted, warnings = self.validator.validate_dynamic_tasks(
            existing_plan=plan, existing_fingerprints=existing_fps, existing_task_ids=set(tasks_map.keys()),
            candidate_drafts=candidate_drafts[: settings.max_refinement_tasks_per_round],
            origin="refinement", round_num=next_round, remaining_query_budget=rem_queries,
        )
        new_task_entries: dict[str, dict[str, Any]] = {}
        for t in accepted:
            plan.tasks.append(t)
            new_task_entries[t.id] = {
                "task": t.model_dump(), "status": TaskStatus.PENDING.value, "attempts": 0,
                "depends_on": list(t.depends_on), "priority": t.priority, "agent": t.agent,
            }
        plan.version += 1
        evt = self._emit_event(run_id, "refinement_planner", f"Refinement round {next_round}: added {len(accepted)} targeted task(s): {[t.id for t in accepted]}")
        return {
            "plan": plan.model_dump(), "tasks": new_task_entries, "refinement_round": next_round,
            "refinement_history": [{"round": next_round, "added_tasks": [t.id for t in accepted], "warnings": warnings, "requests": [r.model_dump() for r in review.refinement_requests]}],
            "trace": [evt],
        }

    def insight_agent_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        intent = QuestionIntent.model_validate(state["intent"])
        plan = AnalysisPlan.model_validate(state["plan"])
        hypotheses_records = state.get("hypotheses", [])
        task_results = state.get("task_results", {})

        ev_status_by_task = {ev.get("task_id"): ev.get("description", "") for ev in state.get("evidence_records", []) if ev.get("kind") == "calculation"}
        python_analyses = [
            {
                "task_id": tid, "tool": res.get("tool"),
                "methodology": res.get("methodology") or (res.get("calculation") or {}).get("methodology", ""),
                "generated_code": res.get("generated_code", ""), "execution_steps": res.get("execution_steps", []),
                "assumptions_checked": res.get("assumptions_checked", []), "artifacts": res.get("artifacts", []),
                "repair_attempts": res.get("repair_attempts", 0),
                "verification_status": (
                    "reproduced" if "[verification=reproduced]" in str(ev_status_by_task.get(tid, ""))
                    else ("warning" if "[verification=mismatch]" in str(ev_status_by_task.get(tid, "")) else res.get("verification_status", ""))
                ),
                "calculation": res.get("calculation"), "duration_ms": res.get("duration_ms", 0),
            }
            for tid, res in task_results.items()
            if res.get("agent") == "python" and res.get("status") == "COMPLETED" and res.get("calculation")
        ]

        models_used_items = [ModelUsageItem.model_validate(m) for m in state.get("models_used", [])]
        before_count = len(models_used_items)
        self._emit_event(run_id, "insight_agent", "Synthesizing verified findings and business insights...")
        with trace_span("node.insight_agent", run_id=run_id):
            report = self.reporter.create_report(
                intent=intent, evidence_list=state.get("raw_evidence", []),
                hypotheses_text=[h.get("statement", "") for h in hypotheses_records],
                task_errors=state.get("errors", []), trace_events=state.get("trace", []),
                models_used=models_used_items, python_analyses=python_analyses,
                evaluated_hypotheses=hypotheses_records, output_mode=plan.output.format,
            )
        new_models = [m.model_dump() for m in report.models_used[before_count:]]
        evt = self._emit_event(run_id, "synthesize", f"Insight Agent synthesized {len(report.findings)} verified findings")
        return {
            "report": report.model_dump(), "direct_answer": report.summary,
            "final_output_mode": plan.output.format, "models_used": new_models, "trace": [evt],
        }

    def output_requirements_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        mode = AnalysisPlan.model_validate(state["plan"]).output.format
        return {"final_output_mode": mode, "trace": [self._emit_event(run_id, "output_requirements", f"Routing to output pipelines for format='{mode}'")]}

    def output_router(self, state: AnalysisState) -> list[str]:
        plan = AnalysisPlan.model_validate(state["plan"])
        targets = ["direct_answer_node"]
        if plan.output.format in ("answer_with_visuals", "business_report") or state.get("report", {}).get("charts"):
            targets.append("chart_renderer")
        if plan.output.format == "business_report":
            targets.append("report_agent")
        return targets

    def chart_renderer_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        charts = state.get("report", {}).get("charts", [])
        validated_charts = [
            {**c, "id": f"chart_{idx+1}"}
            for idx, c in enumerate(charts)
            if (c.get("labels") or []) and (c.get("values") or []) and len(c["labels"]) == len(c["values"])
        ]
        return {"charts": validated_charts, "trace": [self._emit_event(run_id, "chart_renderer", f"Rendered {len(validated_charts)} verified chart(s)")]}

    def report_agent_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        plan = AnalysisPlan.model_validate(state["plan"])
        base_report = Report.model_validate(state["report"])
        ev_records = [EvidenceRecord.model_validate(e) for e in state.get("evidence_records", [])]
        self._emit_event(run_id, "report_agent", "Generating comprehensive BusinessReport...")
        with trace_span("node.report_agent", run_id=run_id):
            biz_report, usage_item = self.report_agent.generate_business_report(plan=plan, base_report=base_report, evidence_records=ev_records)
        evt = self._emit_event(run_id, "report_agent", f"Generated BusinessReport with {len(biz_report.sections)} sections")
        return {"business_report": biz_report.model_dump(), "models_used": [usage_item.model_dump()] if usage_item else [], "trace": [evt]}

    def direct_answer_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        return {"direct_answer": state.get("report", {}).get("summary", ""), "trace": [self._emit_event(run_id, "direct_answer", "Verified concise executive answer")]}

    def final_quality_gate_node(self, state: AnalysisState) -> dict[str, Any]:
        run_id = state.get("run_id", "")
        report_dict = dict(state.get("report", {}))
        review = state.get("evidence_review", {})
        tasks_map = state.get("tasks", {})
        biz_report = state.get("business_report")
        stop_reason = state.get("stop_reason", "")

        task_graph_summary = [
            {
                "id": tid, "title": entry.get("task", {}).get("title", tid),
                "agent": entry.get("agent", "sql"), "status": entry.get("status", "COMPLETED"),
                "depends_on": entry.get("depends_on", []), "priority": entry.get("priority", 3),
                "origin": entry.get("task", {}).get("origin", "planner"),
                "round": entry.get("task", {}).get("round", 0),
                "duration_ms": entry.get("duration_ms", 0), "error": entry.get("error"),
            }
            for tid, entry in tasks_map.items()
        ]
        report_dict["task_graph"] = task_graph_summary
        report_dict["evidence_review"] = review
        if biz_report:
            report_dict["business_report"] = biz_report
            if biz_report.get("executive_summary"):
                report_dict["summary"] = biz_report["executive_summary"]
            if biz_report.get("recommendations"):
                report_dict["recommendations"] = biz_report["recommendations"]
            if biz_report.get("limitations"):
                report_dict["limitations"] = biz_report["limitations"]
        report_dict["models_used"] = state.get("models_used", [])

        decision = review.get("decision", "sufficient")
        has_failed_tasks = any(entry.get("status") in (TaskStatus.FAILED.value, TaskStatus.CANCELLED.value) for entry in tasks_map.values())
        if (stop_reason or has_failed_tasks or decision in ("stop_budget", "stop_round_limit")) and (not review.get("sufficient", True) or has_failed_tasks or stop_reason):
            report_dict["status"] = "partial"
            lims = list(report_dict.get("limitations", []))
            if decision == "stop_budget" and not any("budget" in l.lower() for l in lims):
                lims.append("Analysis scope was bounded by the configured query execution budget.")
            report_dict["limitations"] = lims

        evt = self._emit_event(run_id, "quality_gate", f"Final Quality Gate passed (status={report_dict.get('status')}, tasks={len(task_graph_summary)})")
        report_dict["trace"] = list(state.get("trace", [])) + [evt]
        return {"report": report_dict, "trace": [evt]}

    def execute_measurements(self, state: AnalysisState) -> dict[str, Any]:
        """Legacy helper preserved for direct node unit tests."""
        tasks = [PlanTask.model_validate(t) for t in state.get("tasks", [])] if isinstance(state.get("tasks"), list) else []
        intent = QuestionIntent.model_validate(state["intent"])
        schema_ctx = state["schema_context"]
        results, errors, new_models, trace_events = [], [], [], []
        for task in tasks:
            sql, params, usage = self.sql_analyst.draft(task, intent, schema_ctx)
            new_models.append(usage.model_dump())
            try:
                qres = self.gateway.execute(sql, params, trusted_template=False)
                results.append({
                    "task_id": task.id, "query_id": qres["id"], "purpose": task.purpose,
                    "columns": qres["columns"], "rows": qres["rows"], "sql": sql, "duration_ms": qres["duration_ms"],
                })
                trace_events.append(self._event(f"exec_{task.kind}", f"Fetched {len(qres['rows'])} rows"))
            except Exception as exc:
                errors.append({"task": task.kind, "error": str(exc)})
        return {"raw_evidence": results, "errors": errors, "models_used": new_models, "trace": trace_events}

    def _build_graph(self):
        builder = StateGraph(AnalysisState)
        for name, fn in (
            ("prepare_context", self.prepare_context), ("route_request", self.route_request),
            ("handle_clarification", self.handle_clarification), ("handle_unsupported", self.handle_unsupported),
            ("plan_investigation", self.plan_investigation), ("validate_plan", self.validate_plan_node),
            ("supervisor", self.supervisor_node), ("sql_worker", self.sql_worker),
            ("hypothesis_worker", self.hypothesis_worker), ("python_worker", self.python_worker),
            ("aggregate_evidence", self.aggregate_evidence_node), ("review_evidence", self.review_evidence_node),
            ("refinement_planner", self.refinement_planner_node), ("insight_agent", self.insight_agent_node),
            ("output_requirements", self.output_requirements_node), ("chart_renderer", self.chart_renderer_node),
            ("report_agent", self.report_agent_node), ("direct_answer_node", self.direct_answer_node),
            ("final_quality_gate", self.final_quality_gate_node),
        ):
            builder.add_node(name, fn)

        builder.add_edge(START, "prepare_context")
        builder.add_edge("prepare_context", "route_request")

        def route_selector(state: AnalysisState) -> str:
            route = state.get("route_decision", {}).get("route", "SIMPLE_SQL")
            if route == RouteType.CLARIFICATION_REQUIRED.value:
                return "handle_clarification"
            if route == RouteType.UNSUPPORTED_REQUEST.value:
                return "handle_unsupported"
            return "plan_investigation"

        builder.add_conditional_edges(
            "route_request", route_selector,
            {"handle_clarification": "handle_clarification", "handle_unsupported": "handle_unsupported", "plan_investigation": "plan_investigation"},
        )
        builder.add_edge("handle_clarification", END)
        builder.add_edge("handle_unsupported", END)
        builder.add_edge("plan_investigation", "validate_plan")
        builder.add_edge("validate_plan", "supervisor")
        builder.add_conditional_edges("supervisor", self.supervisor_router, ["sql_worker", "hypothesis_worker", "python_worker", "aggregate_evidence"])
        builder.add_edge("sql_worker", "supervisor")
        builder.add_edge("hypothesis_worker", "supervisor")
        builder.add_edge("python_worker", "supervisor")
        builder.add_edge("aggregate_evidence", "review_evidence")
        builder.add_conditional_edges("review_evidence", self.review_router, {"refinement_planner": "refinement_planner", "insight_agent": "insight_agent"})
        builder.add_edge("refinement_planner", "supervisor")
        builder.add_edge("insight_agent", "output_requirements")
        builder.add_conditional_edges("output_requirements", self.output_router, ["chart_renderer", "report_agent", "direct_answer_node"])
        builder.add_edge("chart_renderer", "final_quality_gate")
        builder.add_edge("report_agent", "final_quality_gate")
        builder.add_edge("direct_answer_node", "final_quality_gate")
        builder.add_edge("final_quality_gate", END)
        return builder.compile(checkpointer=self.checkpointer)

    def get_checkpoint_history(self, thread_id: str) -> list[dict[str, Any]]:
        cfg = {"configurable": {"thread_id": thread_id}}
        history = []
        try:
            for snap in self.graph.get_state_history(cfg):
                history.append({
                    "checkpoint_id": snap.config.get("configurable", {}).get("checkpoint_id"),
                    "next": list(snap.next), "created_at": snap.created_at,
                    "step": snap.metadata.get("step") if snap.metadata else None,
                })
        except Exception as exc:
            logger.debug("Could not read checkpoint history for %s: %s", thread_id, exc)
        return history

    def run(
        self, question: str, previous_context: dict[str, Any] | None = None,
        session_id: str | None = None, run_id: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        rid = run_id or str(uuid4())
        sid = session_id or "default"
        thread_id = f"{sid}:{rid}"

        cancellation_registry.register(rid)
        if on_event:
            self._event_listeners[rid] = on_event

        initial_state: AnalysisState = {
            "run_id": rid, "session_id": sid, "question": question, "started_at": time.monotonic(),
            "previous_context": previous_context or {}, "tasks": {}, "task_results": {},
            "completed_fingerprints": {}, "hypotheses": [], "evidence_records": [], "raw_evidence": [],
            "refinement_round": 0, "refinement_history": [], "charts": [], "tables": [], "errors": [],
            "models_used": [], "trace": [], "budget_state": {"queries_executed": 0, "tokens_used": 0, "cost_usd": 0.0},
        }
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 60, "max_concurrency": settings.max_parallel_tasks},
            )
        finally:
            self._event_listeners.pop(rid, None)
            cancellation_registry.clear(rid)

        report_data = dict(final_state.get("report", {}))
        if not report_data.get("trace"):
            report_data["trace"] = final_state.get("trace", [])
        route_dec = final_state.get("route_decision", {})
        return {
            "run_id": rid, "thread_id": thread_id, "intent": final_state.get("intent", {}),
            "plan": final_state.get("plan", {}), "route": route_dec.get("route", "SIMPLE_SQL"),
            "route_confidence": route_dec.get("confidence", 1.0),
            "route_probabilities": route_dec.get("probabilities", {}), "report": report_data,
        }

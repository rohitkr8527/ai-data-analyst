"""SQL Analyst agent."""
from __future__ import annotations
import logging
from typing import Any, Callable, Literal
from pydantic import BaseModel
from app.agents.runtime import AgentLoopBudget, AutonomousAgentLoop, StepExecutionOutcome
from app.config import settings
from app.exceptions import DatabaseConnectionError, ProviderAuthError, ProviderRateLimitError, QuerySecurityError
from app.orchestration.contracts import AnalysisTask, DatasetRef, FilterSpec, QuerySpec
from app.schemas import ModelUsageItem, PlanTask, QuestionIntent, SQLDraft
from app.services.dataset_store import DatasetStore
from app.services.db import ReadOnlyGateway
from app.services.models import ModelSelector, TaskType
from app.services.semantics import APPROVED_METRICS, resolve_period
from app.services.sql_builder import SQLBuilder

logger = logging.getLogger(__name__)

_STD_SQL_PROMPT = (
    "You are a Senior MySQL Database Analyst. Generate ONE safe SELECT query in MySQL 8.0 dialect.\n"
    "Rules:\n"
    "- Never produce INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, or SELECT *.\n"
    "- Adhere to reference_sql column aliases ('month', 'label', 'value', 'product', 'previous', 'current'), GROUP BY, ORDER BY, and LIMIT.\n"
    "- Use ONLY the supplied named bind parameters without adding or renaming parameters.\n"
    "- Never use comments (-- or /*) or statement separators (;).\n"
    "- Exclude 'Cancelled' orders when measuring completed sales unless grouping by order_status."
)

_OPEN_SQL_PROMPT = (
    "You are an Autonomous Senior SQL Research Analyst. Design ONE safe SELECT query tailored to the analytical objective.\n"
    "Rules:\n"
    "- Never produce INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, or SELECT *.\n"
    "- Never use JSON_ARRAYAGG, JSON_OBJECTAGG, or GROUP_CONCAT.\n"
    "- Exclude 'Cancelled' orders when measuring completed sales unless investigating order_status.\n"
    "- Never use SQL comments (-- or /*) or statement separators (;)."
)


class SQLStepDecision(BaseModel):
    phase: Literal["discover_schema", "query", "repair", "conclude"]
    action: Literal["inspect_schema", "run_sql", "finish"]
    reasoning: str
    tables_to_inspect: list[str] = []
    sql_query: str = ""
    observation_notes: str = ""


def _to_usage_item(role: str, usage) -> ModelUsageItem:
    return ModelUsageItem(
        agent_role=role, task_type=usage.task_type, model=usage.model,
        reasoning_effort=usage.reasoning_effort, duration_ms=usage.duration_ms,
        tokens=usage.input_tokens + usage.output_tokens, cost_usd=usage.cost_usd,
    )


class SQLAnalyst:
    def __init__(self, model_selector: ModelSelector | None = None):
        self.selector = model_selector or ModelSelector()
        self.builder = SQLBuilder()

    def draft_for_task(
        self,
        task: AnalysisTask,
        period_key: str | None,
        plan_filters: list[FilterSpec] | None,
        schema_context: str,
    ) -> tuple[str, dict[str, Any], str, ModelUsageItem | None]:
        qspec = task.query_spec or QuerySpec(metric="gross_revenue")
        canonical_sql, params = self.builder.compile_query_spec(spec=qspec, period_key=period_key, plan_filters=plan_filters)

        if qspec.mode == "structured" and settings.sql_generation_mode == "compiled" and type(self.selector) is ModelSelector:
            return canonical_sql, params, "compiled", None

        period_info = resolve_period(period_key)
        if qspec.mode in ("custom", "autonomous"):
            draft, usage = self.selector.parse_structured(
                task_type=TaskType.SQL_RESEARCH,
                response_model=SQLDraft,
                system_instructions=_OPEN_SQL_PROMPT,
                input_payload={
                    "task_id": task.id, "task_title": task.title, "purpose": task.purpose,
                    "custom_request": qspec.custom_request or task.purpose, "custom_tables": qspec.custom_tables,
                    "period": period_info, "plan_filters": [f.model_dump() for f in (plan_filters or [])], "schema": schema_context,
                },
            )
            return draft.sql, {}, "llm_custom", _to_usage_item("SQL Research Agent", usage)

        draft, usage = self.selector.parse_structured(
            task_type=TaskType.SQL_GENERATION,
            response_model=SQLDraft,
            system_instructions=_STD_SQL_PROMPT,
            input_payload={
                "task_id": task.id, "task_title": task.title, "purpose": task.purpose,
                "query_spec": qspec.model_dump(),
                "metric": APPROVED_METRICS.get(qspec.metric, APPROVED_METRICS["gross_revenue"]),
                "period": period_info, "reference_sql": canonical_sql,
                "available_parameters": list(params.keys()), "schema": schema_context,
            },
        )
        return draft.sql, params, "llm", _to_usage_item("SQL Agent", usage)

    def draft(
        self,
        task: PlanTask,
        intent: QuestionIntent,
        schema_context: str,
    ) -> tuple[str, dict[str, Any], ModelUsageItem]:
        canonical_sql, params = self.builder.build(task, intent)
        draft, usage = self.selector.parse_structured(
            task_type=TaskType.SQL_GENERATION,
            response_model=SQLDraft,
            system_instructions=_STD_SQL_PROMPT,
            input_payload={
                "task": task.model_dump(), "intent": intent.model_dump(),
                "metric": APPROVED_METRICS.get(intent.metric, APPROVED_METRICS["gross_revenue"]),
                "period": resolve_period(intent.period), "reference_sql": canonical_sql,
                "available_parameters": list(params.keys()), "schema": schema_context,
            },
        )
        return draft.sql, params, _to_usage_item("SQL Analyst", usage)

    def repair(
        self,
        rejected_sql: str,
        error_message: str,
        task: PlanTask | AnalysisTask,
        intent: QuestionIntent | dict[str, Any],
        schema_context: str,
        available_params: list[str] | None = None,
    ) -> tuple[str, ModelUsageItem]:
        draft, usage = self.selector.parse_structured(
            task_type=TaskType.SQL_GENERATION,
            response_model=SQLDraft,
            system_instructions=(
                "Repair the rejected MySQL query. Return ONE safe SELECT query without SELECT *, comments, or semicolons, "
                "using only authorized schema tables/columns and existing named bind parameters."
            ),
            input_payload={
                "task": task.model_dump() if hasattr(task, "model_dump") else str(task),
                "intent": intent.model_dump() if hasattr(intent, "model_dump") else dict(intent or {}),
                "rejected_sql": rejected_sql[:3000], "error_message": error_message[:500],
                "available_parameters": available_params or [], "schema": schema_context,
            },
        )
        return draft.sql, _to_usage_item("SQL Agent (Repair)", usage)

    def investigate_sql(
        self,
        task: AnalysisTask,
        gateway: ReadOnlyGateway,
        dataset_store: DatasetStore,
        period_key: str | None = None,
        plan_filters: list[FilterSpec] | None = None,
        schema_context: str = "",
        allowed_tables: set[str] | None = None,
        run_id: str = "run_sql",
        budget: AgentLoopBudget | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any], DatasetRef, list[dict[str, Any]], list[ModelUsageItem], int]:
        qspec = task.query_spec or QuerySpec(metric="gross_revenue", mode="autonomous")
        objective = qspec.custom_request or task.purpose or task.title
        eff_tables = set(allowed_tables or gateway.approved_tables)
        base_schema = schema_context or gateway.catalog.format_llm_schema_context(sorted(eff_tables))
        dialect_name = gateway.engine.dialect.name if gateway.engine is not None else "mysql"

        loop = AutonomousAgentLoop(
            agent_name="sql_research_agent", run_id=run_id or "run_sql", task_id=task.id,
            objective=objective, budget=budget or AgentLoopBudget.from_settings(), on_event=on_event,
        )
        state_box: dict[str, Any] = {
            "inspected_details": "", "last_sql": "", "last_error": None,
            "best_result": None, "best_sql": "", "repair_count": 0,
        }
        sys_prompt = (
            f"You are an Autonomous SQL Research Agent working against a read-only {dialect_name.upper()} database.\n"
            "1. When table columns are visible in `schema_overview`, choose action='run_sql' on Step 1 with ONE safe SELECT query.\n"
            "2. Never use SELECT *, JSON_ARRAYAGG, JSON_OBJECTAGG, or GROUP_CONCAT.\n"
            "3. If SQL fails or returns 0 rows unexpectedly, issue a corrected query."
        )

        def _decide(step_idx: int, active_loop: AutonomousAgentLoop) -> tuple[SQLStepDecision, ModelUsageItem | None]:
            decision, usage = self.selector.parse_structured(
                task_type=TaskType.SQL_RESEARCH,
                response_model=SQLStepDecision,
                system_instructions=sys_prompt,
                input_payload={
                    "objective": objective, "task_title": task.title, "dialect": dialect_name,
                    "period": resolve_period(period_key) if period_key else {},
                    "plan_filters": [f.model_dump() for f in (plan_filters or [])],
                    "allowed_tables": sorted(eff_tables), "schema_overview": base_schema[:1800],
                    "detailed_schema_inspected": state_box["inspected_details"][:1500],
                    "previous_sql": state_box["last_sql"][:1500] if state_box["last_error"] else "",
                    "last_error": state_box["last_error"], "has_completed_query": state_box["best_result"] is not None,
                    "recent_steps": active_loop.build_compact_history(max_recent=2),
                    "step_index": step_idx, "remaining_steps": active_loop.remaining_steps(),
                },
                custom_effort="low",
            )
            return decision, _to_usage_item("SQL Research Agent", usage)

        def _execute(step_idx: int, decision: SQLStepDecision, active_loop: AutonomousAgentLoop) -> StepExecutionOutcome:
            if decision.action == "inspect_schema":
                targets = [t for t in decision.tables_to_inspect if t.lower() in eff_tables] or sorted(eff_tables)
                details = gateway.catalog.inspect_table_details(targets, sample_limit=3)
                state_box["inspected_details"] = details
                return StepExecutionOutcome(action="inspect_schema", status="ok", observation_summary=details[:1400])

            if decision.action == "finish":
                if state_box["best_result"] is not None:
                    return StepExecutionOutcome(
                        action="finish", status="completed",
                        observation_summary=decision.observation_notes or "Completed autonomous SQL research.",
                        should_finish=True,
                    )
                if not decision.sql_query.strip():
                    return StepExecutionOutcome(
                        action="finish_rejected", status="error",
                        observation_summary="No SQL query has succeeded yet. Choose action='run_sql' first.",
                    )

            sql = decision.sql_query.strip().rstrip(";")
            if not sql:
                return StepExecutionOutcome(action="run_sql", status="error", observation_summary="No sql_query provided.")

            is_repair = state_box["last_error"] is not None
            state_box["last_sql"] = sql
            try:
                qres = gateway.execute(sql, {}, trusted_template=False, allowed_tables=eff_tables, run_id=run_id, task_id=task.id)
                state_box["last_error"] = None
                state_box["best_result"] = qres
                state_box["best_sql"] = sql
                row_cnt = len(qres.get("rows", []))
                obs = f"Executed SQL in {qres.get('duration_ms', 0)}ms: rows={row_cnt}, columns={qres.get('columns', [])}, preview={qres.get('rows', [])[:3]}"
                auto_done = row_cnt > 0 and type(self.selector) is ModelSelector
                return StepExecutionOutcome(
                    action="run_sql", status="completed" if auto_done else "ok",
                    is_repair=is_repair, observation_summary=obs[:1000], should_finish=auto_done,
                )
            except (ProviderAuthError, ProviderRateLimitError, DatabaseConnectionError):
                raise
            except Exception as exc:
                state_box["repair_count"] += 1
                state_box["last_error"] = str(exc)[:800]
                return StepExecutionOutcome(
                    action="run_sql", status="error", is_repair=True,
                    observation_summary=f"SQL validation/execution failed: {exc}",
                )

        loop.run(_decide, _execute)
        best_qres = state_box["best_result"]
        if best_qres is None:
            raise QuerySecurityError(state_box["last_error"] or "Autonomous SQL Research failed to execute a valid query.")

        ds_ref = dataset_store.put(
            run_id=run_id, task_id=task.id, columns=best_qres["columns"],
            rows=best_qres["rows"], query_id=best_qres["id"], sql_hash=best_qres["query_hash"],
        )
        return state_box["best_sql"], {}, best_qres, ds_ref, [s.to_dict() for s in loop.steps], loop.usages, state_box["repair_count"]

"""SQL Analyst agent. Uses OpenAI gpt-6-luna with low reasoning effort to generate MySQL queries.

Untrusted generated SQL is parsed and strictly validated by ReadOnlyGateway.
Bounded repair policy fixes syntax/column errors without permitting security bypasses.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Callable, Literal
from pydantic import BaseModel, Field
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


class SQLStepDecision(BaseModel):
    """Closed structured decision schema for multi-step Autonomous SQL Research."""

    phase: Literal["discover_schema", "query", "repair", "conclude"]
    action: Literal["inspect_schema", "run_sql", "finish"]
    reasoning: str = Field(description="Reasoning behind schema discovery, SQL formulation, repair, or completion.")
    tables_to_inspect: list[str] = Field(
        default_factory=list,
        description="Authorized table names to inspect in detail when action='inspect_schema'.",
    )
    sql_query: str = Field(
        default="",
        description="Safe read-only SELECT query when action='run_sql'. Never use SELECT * or mutating statements.",
    )
    observation_notes: str = Field(
        default="",
        description="Key takeaways from observed rows/schema or conclusion summary when action='finish'.",
    )


class SQLAnalyst:
    """Generates and repairs MySQL queries using OpenAI gpt-6-luna."""

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
        """Generate a MySQL query for an AnalysisTask.

        Returns `(sql, params, sql_source, usage_item)`.
        """
        qspec = task.query_spec or QuerySpec(metric="gross_revenue")
        is_open_ended = qspec.mode in ("custom", "autonomous")
        canonical_sql, params = self.builder.compile_query_spec(
            spec=qspec,
            period_key=period_key,
            plan_filters=plan_filters,
        )

        # Use deterministic compiled SQL (0 LLM tokens) for standard structured QuerySpecs in live execution
        if qspec.mode == "structured" and type(self.selector) is ModelSelector:
            return canonical_sql, params, "compiled", None

        period_info = resolve_period(period_key)
        metric_meta = APPROVED_METRICS.get(qspec.metric, APPROVED_METRICS["gross_revenue"])

        if is_open_ended:
            custom_params: dict[str, Any] = {}
            system_instructions = (
                "You are an Autonomous Senior SQL Research Analyst.\n"
                "Design ONE safe SELECT query tailored to the analytical objective and custom_request.\n"
                "Rules:\n"
                "- Never produce INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, or commands.\n"
                "- Never produce SELECT *; always list explicit columns, aggregations, CTEs, or window functions.\n"
                "- Never use JSON_ARRAYAGG, JSON_OBJECTAGG, or GROUP_CONCAT to pack detail rows into JSON strings.\n"
                "- Design the exact granularity, JOINs, CTEs, cohorts, or window expressions needed rather than squeezing into a fixed template.\n"
                "- Exclude 'Cancelled' orders when measuring completed sales unless investigating order_status.\n"
                "- Never use SQL comments (-- or /*) or statement separators (;).\n"
                "- Return your answer strictly in the requested SQLDraft JSON format."
            )
            input_payload = {
                "task_id": task.id,
                "task_title": task.title,
                "purpose": task.purpose,
                "custom_request": qspec.custom_request or task.purpose,
                "custom_tables": qspec.custom_tables,
                "period": period_info,
                "plan_filters": [f.model_dump() for f in (plan_filters or [])],
                "schema": schema_context,
            }
            draft, usage = self.selector.parse_structured(
                task_type=TaskType.SQL_RESEARCH,
                response_model=SQLDraft,
                system_instructions=system_instructions,
                input_payload=input_payload,
            )
            usage_item = ModelUsageItem(
                agent_role="SQL Research Agent",
                task_type=usage.task_type,
                model=usage.model,
                reasoning_effort=usage.reasoning_effort,
                duration_ms=usage.duration_ms,
                tokens=usage.input_tokens + usage.output_tokens,
                cost_usd=usage.cost_usd,
            )
            return draft.sql, custom_params, "llm_custom", usage_item

        system_instructions = (
            "You are a Senior MySQL Database Analyst. Generate ONE safe SELECT query in MySQL 8.0 dialect.\n"
            "Rules:\n"
            "- Never produce INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, or commands.\n"
            "- Never produce SELECT *; always use explicit columns, aggregations, and aliases.\n"
            "- When a reference_sql is provided, adhere closely to its column aliases ('month', 'label', 'value', 'product', 'previous', 'current', 'previous_revenue', 'current_revenue'), GROUP BY, ORDER BY, and LIMIT.\n"
            "- If the task is to find the overall peak/highest or full historical trend without a specific period, do NOT add arbitrary date filters.\n"
            "- Use ONLY the supplied named bind parameters (e.g. :target_start, :target_end, :previous_start, :previous_end, :limit_n).\n"
            "- Do not add, remove, or rename bind parameters.\n"
            "- Never use comments (-- or /*) or statement separators (;).\n"
            "- Use MySQL DATE_FORMAT for date manipulations.\n"
            "- Exclude 'Cancelled' orders when measuring completed sales unless grouping by order_status.\n"
            "- Return your answer strictly in the requested SQLDraft JSON format."
        )

        input_payload = {
            "task_id": task.id,
            "task_title": task.title,
            "purpose": task.purpose,
            "query_spec": qspec.model_dump(),
            "metric": metric_meta,
            "period": period_info,
            "reference_sql": canonical_sql,
            "available_parameters": list(params.keys()),
            "schema": schema_context,
        }

        draft, usage = self.selector.parse_structured(
            task_type=TaskType.SQL_GENERATION,
            response_model=SQLDraft,
            system_instructions=system_instructions,
            input_payload=input_payload,
        )

        usage_item = ModelUsageItem(
            agent_role="SQL Agent",
            task_type=usage.task_type,
            model=usage.model,
            reasoning_effort=usage.reasoning_effort,
            duration_ms=usage.duration_ms,
            tokens=usage.input_tokens + usage.output_tokens,
            cost_usd=usage.cost_usd,
        )

        return draft.sql, params, "llm", usage_item

    def draft(
        self,
        task: PlanTask,
        intent: QuestionIntent,
        schema_context: str,
    ) -> tuple[str, dict[str, Any], ModelUsageItem]:
        """Generate a validated MySQL query tailored to a legacy PlanTask."""
        canonical_sql, params = self.builder.build(task, intent)
        period_info = resolve_period(intent.period)
        metric_meta = APPROVED_METRICS.get(intent.metric, APPROVED_METRICS["gross_revenue"])

        system_instructions = (
            "You are a Senior MySQL Database Analyst. Generate ONE safe SELECT query in MySQL 8.0 dialect.\n"
            "Rules:\n"
            "- Never produce INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, or commands.\n"
            "- Never produce SELECT *; always use explicit columns, aggregations, and aliases.\n"
            "- When a reference_sql is provided, adhere closely to its column aliases ('month', 'label', 'value', 'product'), GROUP BY, ORDER BY, and LIMIT.\n"
            "- If the task is to find the overall peak/highest or full historical trend without a specific period, do NOT add arbitrary date filters.\n"
            "- Use ONLY the supplied named bind parameters (e.g. :target_start, :target_end, :limit_n).\n"
            "- Do not add, remove, or rename bind parameters.\n"
            "- Never use comments (-- or /*) or statement separators (;).\n"
            "- Use MySQL DATE_FORMAT for date manipulations.\n"
            "- Exclude 'Cancelled' orders when measuring completed sales.\n"
            "- Return your answer strictly in the requested SQLDraft JSON format."
        )

        input_payload = {
            "task": task.model_dump(),
            "intent": intent.model_dump(),
            "metric": metric_meta,
            "period": period_info,
            "reference_sql": canonical_sql,
            "available_parameters": list(params.keys()),
            "schema": schema_context,
        }

        draft, usage = self.selector.parse_structured(
            task_type=TaskType.SQL_GENERATION,
            response_model=SQLDraft,
            system_instructions=system_instructions,
            input_payload=input_payload,
        )

        usage_item = ModelUsageItem(
            agent_role="SQL Analyst",
            task_type=usage.task_type,
            model=usage.model,
            reasoning_effort=usage.reasoning_effort,
            duration_ms=usage.duration_ms,
            tokens=usage.input_tokens + usage.output_tokens,
            cost_usd=usage.cost_usd,
        )

        return draft.sql, params, usage_item

    def repair(
        self,
        rejected_sql: str,
        error_message: str,
        task: PlanTask | AnalysisTask,
        intent: QuestionIntent | dict[str, Any],
        schema_context: str,
        available_params: list[str] | None = None,
    ) -> tuple[str, ModelUsageItem]:
        """Repair a rejected SQL query due to syntax or column error."""
        system_instructions = (
            "You are a MySQL Database Analyst repairing an invalid SQL query.\n"
            "Correct the syntax error, unknown column, or missing explicit column projection reported by the validator or MySQL.\n"
            "Rules:\n"
            "- Return ONE safe SELECT query only.\n"
            "- Never use SELECT *; always list explicit columns.\n"
            "- Use ONLY existing authorized tables and columns from the schema.\n"
            "- Maintain the existing named bind parameters.\n"
            "- Never attempt to bypass security policies or access forbidden schemas.\n"
            "- Return your answer strictly in the requested SQLDraft JSON format."
        )

        intent_dump = intent.model_dump() if hasattr(intent, "model_dump") else dict(intent or {})
        input_payload = {
            "task": task.model_dump() if hasattr(task, "model_dump") else str(task),
            "intent": intent_dump,
            "rejected_sql": rejected_sql[:3000],
            "error_message": error_message[:500],
            "available_parameters": available_params or [],
            "schema": schema_context,
        }

        draft, usage = self.selector.parse_structured(
            task_type=TaskType.SQL_GENERATION,
            response_model=SQLDraft,
            system_instructions=system_instructions,
            input_payload=input_payload,
        )

        usage_item = ModelUsageItem(
            agent_role="SQL Agent (Repair)",
            task_type=usage.task_type,
            model=usage.model,
            reasoning_effort=usage.reasoning_effort,
            duration_ms=usage.duration_ms,
            tokens=usage.input_tokens + usage.output_tokens,
            cost_usd=usage.cost_usd,
        )

        return draft.sql, usage_item

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
        """Autonomously explore unfamiliar schemas, inspect samples, execute & repair SQL, and snapshot results."""
        qspec = task.query_spec or QuerySpec(metric="gross_revenue", mode="autonomous")
        objective = qspec.custom_request or task.purpose or task.title
        eff_tables = set(allowed_tables or gateway.approved_tables)
        base_schema = schema_context or gateway.catalog.format_llm_schema_context(sorted(eff_tables))
        dialect_name = gateway.engine.dialect.name if gateway.engine is not None else "mysql"

        loop_budget = budget or AgentLoopBudget.from_settings()
        loop = AutonomousAgentLoop(
            agent_name="sql_research_agent",
            run_id=run_id or "run_sql",
            task_id=task.id,
            objective=objective,
            budget=loop_budget,
            on_event=on_event,
        )

        state_box: dict[str, Any] = {
            "inspected_details": "",
            "last_sql": "",
            "last_error": None,
            "best_result": None,
            "best_sql": "",
            "repair_count": 0,
        }

        sys_prompt = (
            f"You are an Autonomous SQL Research Agent working against a read-only {dialect_name.upper()} database.\n"
            "Rules:\n"
            "1. When table columns are already visible in `schema_overview`, choose action='run_sql' immediately on Step 1 "
            "with ONE safe SELECT query (explicit columns, no SELECT *, no comments, no semicolons). "
            "Only choose action='inspect_schema' if a required table's columns are missing from `schema_overview`.\n"
            "2. Never use JSON_ARRAYAGG, JSON_OBJECTAGG, or GROUP_CONCAT; return clean tabular rows and aggregations.\n"
            "3. If SQL fails or returns 0 rows unexpectedly, issue a corrected query.\n"
        )

        def _decide(step_idx: int, active_loop: AutonomousAgentLoop) -> tuple[SQLStepDecision, ModelUsageItem | None]:
            payload = {
                "objective": objective,
                "task_title": task.title,
                "dialect": dialect_name,
                "period": resolve_period(period_key) if period_key else {},
                "plan_filters": [f.model_dump() for f in (plan_filters or [])],
                "allowed_tables": sorted(eff_tables),
                "schema_overview": base_schema[:1800],
                "detailed_schema_inspected": state_box["inspected_details"][:1500],
                "previous_sql": state_box["last_sql"][:1500] if state_box["last_error"] else "",
                "last_error": state_box["last_error"],
                "has_completed_query": state_box["best_result"] is not None,
                "recent_steps": active_loop.build_compact_history(max_recent=2),
                "step_index": step_idx,
                "remaining_steps": active_loop.remaining_steps(),
            }
            decision, usage = self.selector.parse_structured(
                task_type=TaskType.SQL_RESEARCH,
                response_model=SQLStepDecision,
                system_instructions=sys_prompt,
                input_payload=payload,
                custom_effort="low",
            )
            item = ModelUsageItem(
                agent_role="SQL Research Agent",
                task_type=usage.task_type,
                model=usage.model,
                reasoning_effort=usage.reasoning_effort,
                duration_ms=usage.duration_ms,
                tokens=usage.input_tokens + usage.output_tokens,
                cost_usd=usage.cost_usd,
            )
            return decision, item

        def _execute(step_idx: int, decision: SQLStepDecision, active_loop: AutonomousAgentLoop) -> StepExecutionOutcome:
            if decision.action == "inspect_schema":
                targets = [t for t in decision.tables_to_inspect if t.lower() in eff_tables] or sorted(eff_tables)
                details = gateway.catalog.inspect_table_details(targets, sample_limit=3)
                state_box["inspected_details"] = details
                return StepExecutionOutcome(
                    action="inspect_schema",
                    status="ok",
                    observation_summary=details[:1400],
                )

            if decision.action == "finish":
                if state_box["best_result"] is not None:
                    return StepExecutionOutcome(
                        action="finish",
                        status="completed",
                        observation_summary=decision.observation_notes or "Completed autonomous SQL research.",
                        should_finish=True,
                    )
                if not decision.sql_query.strip():
                    return StepExecutionOutcome(
                        action="finish_rejected",
                        status="error",
                        observation_summary="No SQL query has succeeded yet. Choose action='run_sql' with a valid SELECT query first.",
                    )

            sql = decision.sql_query.strip().rstrip(";")
            if not sql:
                return StepExecutionOutcome(
                    action="run_sql",
                    status="error",
                    observation_summary="No sql_query provided for action='run_sql'.",
                )

            is_repair = state_box["last_error"] is not None
            state_box["last_sql"] = sql
            try:
                qres = gateway.execute(
                    sql,
                    {},
                    trusted_template=False,
                    allowed_tables=eff_tables,
                    run_id=run_id,
                    task_id=task.id,
                )
                state_box["last_error"] = None
                state_box["best_result"] = qres
                state_box["best_sql"] = sql
                row_cnt = len(qres.get("rows", []))
                preview = qres.get("rows", [])[:3]
                obs = (
                    f"Executed SQL successfully in {qres.get('duration_ms', 0)}ms: "
                    f"rows={row_cnt}, columns={qres.get('columns', [])}, preview={preview}"
                )
                if row_cnt == 0:
                    obs += " (Warning: 0 rows returned; check WHERE filters/entity casing or finish if intentionally empty.)"
                auto_done = row_cnt > 0 and type(self.selector) is ModelSelector
                return StepExecutionOutcome(
                    action="run_sql",
                    status="completed" if auto_done else "ok",
                    is_repair=is_repair,
                    observation_summary=obs[:1000],
                    should_finish=auto_done,
                )
            except (ProviderAuthError, ProviderRateLimitError, DatabaseConnectionError):
                raise
            except (QuerySecurityError, Exception) as exc:
                state_box["repair_count"] += 1
                state_box["last_error"] = str(exc)[:800]
                return StepExecutionOutcome(
                    action="run_sql",
                    status="error",
                    is_repair=True,
                    observation_summary=f"SQL validation/execution failed: {exc}. Inspect table details or revise SQL.",
                )

        loop.run(_decide, _execute)

        best_qres = state_box["best_result"]
        if best_qres is None:
            raise QuerySecurityError(state_box["last_error"] or "Autonomous SQL Research failed to execute a valid query.")

        ds_ref = dataset_store.put(
            run_id=run_id,
            task_id=task.id,
            columns=best_qres["columns"],
            rows=best_qres["rows"],
            query_id=best_qres["id"],
            sql_hash=best_qres["query_hash"],
        )
        return (
            state_box["best_sql"],
            {},
            best_qres,
            ds_ref,
            [s.to_dict() for s in loop.steps],
            loop.usages,
            state_box["repair_count"],
        )

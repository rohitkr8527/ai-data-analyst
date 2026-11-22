"""Autonomous Python Research Agent & deterministic analytical utilities."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Callable, Literal
import pandas as pd
from pydantic import BaseModel, Field

from app.agents.runtime import AgentLoopBudget, AutonomousAgentLoop, StepExecutionOutcome
from app.exceptions import SandboxSecurityError
from app.orchestration.contracts import AnalysisTask
from app.schemas import ModelUsageItem
from app.services.analysis_tools import ANALYSIS_TOOL_REGISTRY, run_registered_analysis_tool
from app.services.dataset_broker import AuthorizedDatasetBroker, summarize_dataframe
from app.services.dataset_store import DatasetStore
from app.services.evidence_store import SharedEvidenceStore
from app.services.models import ModelSelector, TaskType
from app.services.sandbox import SandboxExecutionService

logger = logging.getLogger(__name__)

AUTONOMOUS_PYTHON_SYSTEM_PROMPT = """You are an Autonomous Senior Data Scientist and Python Research Agent.
Write and execute Python code inside the isolated sandbox to analyze the staged datasets (`df`, `DATASETS`, `load_dataset(name)`).
Pre-loaded globals: `np`, `pd`, `scipy`, `stats`, `math`, `json`. Installed packages: `sklearn`, `statsmodels.api as sm`, `polars as pl`, `matplotlib.pyplot as plt`, `plotly`.
Always call `save_result(summary_dict, records=..., assumptions=...)` (plus `save_table(title, df)` / `save_chart(title, kind, labels, values, unit)`) so results are recorded.
Never use network/sockets, subprocesses, or access files outside the workspace. Ground all conclusions strictly in observed sandbox outputs.
"""


class ToolSelectionDecision(BaseModel):
    tool: Literal[
        "period_comparison", "revenue_decomposition", "contribution_analysis",
        "trend_analysis", "anomaly_detection", "correlation_analysis",
        "distribution_analysis", "statistical_summary", "data_quality_check", "forecast_linear",
    ]
    reasoning: str = ""


class PythonStepDecision(BaseModel):
    phase: Literal["explore", "analyze", "debug", "request_data", "refine", "conclude"]
    action: Literal["run_python", "request_dataset", "finish"]
    reasoning: str
    chosen_methodology: str = ""
    action_summary: str = ""
    python_code: str = ""
    dataset_request: str = ""
    assumptions_checked: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    conclusion_narrative: str = ""


@dataclass
class AutonomousPythonInvestigation:
    tool_name: str
    methodology: str
    calculation: dict[str, Any]
    generated_code: str
    execution_steps: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    assumptions_checked: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    usages: list[ModelUsageItem] = field(default_factory=list)
    repair_attempts: int = 0
    status: Literal["COMPLETED", "FAILED"] = "COMPLETED"
    error: str | None = None


class PythonAnalysisAgent:
    def __init__(
        self,
        dataset_store: DatasetStore,
        models: ModelSelector | None = None,
        sandbox: SandboxExecutionService | None = None,
        dataset_broker: AuthorizedDatasetBroker | None = None,
        evidence_store: SharedEvidenceStore | None = None,
    ):
        self.dataset_store = dataset_store
        self.models = models or ModelSelector()
        self.sandbox = sandbox or SandboxExecutionService()
        self.dataset_broker = dataset_broker
        self.evidence_store = evidence_store

    def _select_tool_via_llm(self, task: AnalysisTask, df: pd.DataFrame) -> tuple[str, ModelUsageItem | None]:
        spec = task.python_spec
        decision, usage = self.models.parse_structured(
            task_type=TaskType.PYTHON_TOOL_SELECTION,
            response_model=ToolSelectionDecision,
            system_instructions="Select the best deterministic analytical utility for the dataset and objective.",
            input_payload={
                "task_objective": (spec.objective if spec and spec.objective else task.purpose),
                "task_description": task.purpose,
                "data_schema": summarize_dataframe(df),
                "available_tools": [{"name": k, "description": (v.__doc__ or k).strip()} for k, v in ANALYSIS_TOOL_REGISTRY.items()],
            },
            custom_effort="low",
        )
        return decision.tool, ModelUsageItem(
            agent_role="Python Analysis Agent", task_type=usage.task_type, model=usage.model,
            reasoning_effort=usage.reasoning_effort, duration_ms=usage.duration_ms,
            tokens=usage.input_tokens + usage.output_tokens, cost_usd=usage.cost_usd,
        )

    def should_use_autonomous_mode(self, task: AnalysisTask) -> bool:
        spec = task.python_spec
        return bool(spec and (spec.tool in ("autonomous_research", "custom_python") or getattr(spec, "mode", "auto") == "autonomous"))

    def investigate(
        self,
        task: AnalysisTask,
        upstream_dataset_ids: list[str],
        run_id: str = "run_adhoc",
        sql_provider: Callable[[str], tuple[str, dict[str, Any]]] | None = None,
        budget: AgentLoopBudget | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AutonomousPythonInvestigation:
        spec = task.python_spec
        objective = (spec.objective if spec and spec.objective else task.purpose) or task.title
        suggested_methods = list(getattr(spec, "analysis_methods", []) or [])

        workspace = self.sandbox.create_workspace(run_id=run_id, task_id=task.id)
        staged_ids = list(upstream_dataset_ids)
        dataset_profiles: dict[str, Any] = {}

        try:
            for idx, ds_id in enumerate(upstream_dataset_ids):
                df = self.dataset_store.get_dataframe(ds_id)
                ds_name = spec.input_task_ids[idx] if (spec and idx < len(spec.input_task_ids)) else f"dataset_{idx + 1}"
                self.sandbox.stage_dataframe(workspace, ds_name, df)
                dataset_profiles[ds_name] = summarize_dataframe(df, dataset_name=ds_name)

            loop = AutonomousAgentLoop(
                agent_name="python_research_agent", run_id=run_id, task_id=task.id,
                objective=objective, budget=budget or AgentLoopBudget.from_settings(), on_event=on_event,
            )

            working_state: dict[str, Any] = {
                "last_code": "", "last_error": None, "last_completed_result": None, "last_completed_code": "",
                "chosen_methodology": suggested_methods[0] if suggested_methods else "",
                "assumptions": [], "conclusion_narrative": "", "security_blocks": 0,
            }

            def _decide(step_idx: int, active_loop: AutonomousAgentLoop) -> tuple[PythonStepDecision, ModelUsageItem | None]:
                schema_hint = ""
                if self.dataset_broker is not None and not dataset_profiles:
                    try:
                        schema_hint = self.dataset_broker.inspect_schema()[:900]
                    except Exception:
                        pass
                decision, usage = self.models.parse_structured(
                    task_type=TaskType.PYTHON_RESEARCH,
                    response_model=PythonStepDecision,
                    system_instructions=AUTONOMOUS_PYTHON_SYSTEM_PROMPT,
                    input_payload={
                        "objective": objective, "task_title": task.title, "task_description": task.purpose,
                        "suggested_methods": suggested_methods, "step_index": step_idx,
                        "remaining_steps": active_loop.remaining_steps(), "max_steps": active_loop.budget.max_steps,
                        "dataset_requests_used": active_loop.dataset_requests, "max_dataset_requests": active_loop.budget.max_dataset_requests,
                        "staged_datasets": dataset_profiles,
                        "previous_code": working_state["last_code"][:2000] if working_state["last_error"] else "",
                        "last_error": working_state["last_error"],
                        "recent_steps": active_loop.build_compact_history(max_recent=2),
                        "has_completed_analysis": working_state["last_completed_result"] is not None,
                        "authorized_db_schema_overview": schema_hint,
                    },
                    custom_effort="low",
                )
                return decision, ModelUsageItem(
                    agent_role="Python Research Agent", task_type=usage.task_type, model=usage.model,
                    reasoning_effort=usage.reasoning_effort, duration_ms=usage.duration_ms,
                    tokens=usage.input_tokens + usage.output_tokens, cost_usd=usage.cost_usd,
                )

            def _execute(step_idx: int, decision: PythonStepDecision, active_loop: AutonomousAgentLoop) -> StepExecutionOutcome:
                if decision.chosen_methodology:
                    working_state["chosen_methodology"] = decision.chosen_methodology
                for a in decision.assumptions_checked:
                    if a and a not in working_state["assumptions"]:
                        working_state["assumptions"].append(a)

                decision_action = decision.action
                if decision_action == "finish" and working_state["last_completed_result"] is None:
                    if decision.python_code.strip():
                        decision_action = "run_python"
                    else:
                        return StepExecutionOutcome(
                            action="finish_rejected", status="error",
                            observation_summary="Cannot conclude without a completed sandbox execution. Call `save_result(...)` first.",
                        )

                if decision_action == "finish":
                    working_state["conclusion_narrative"] = decision.conclusion_narrative
                    return StepExecutionOutcome(
                        action="finish", status="completed",
                        observation_summary=decision.conclusion_narrative or "Investigation concluded.",
                        should_finish=True,
                    )

                if decision_action == "request_dataset":
                    if active_loop.dataset_requests >= active_loop.budget.max_dataset_requests or self.dataset_broker is None:
                        return StepExecutionOutcome(
                            action="request_dataset", status="error",
                            observation_summary=f"Cannot request dataset. Use staged datasets: {list(dataset_profiles.keys())}.",
                        )
                    try:
                        ds_ref, new_df, meta = self.dataset_broker.request_dataset(
                            run_id=run_id, task_id=task.id, request_text_or_sql=decision.dataset_request.strip(),
                            sql_provider=sql_provider, reason=decision.reasoning or objective,
                        )
                        new_name = f"requested_{active_loop.dataset_requests + 1}"
                        self.sandbox.stage_dataframe(workspace, new_name, new_df)
                        staged_ids.append(ds_ref.dataset_id)
                        dataset_profiles[new_name] = {
                            **summarize_dataframe(new_df, dataset_name=new_name),
                            "dataset_id": ds_ref.dataset_id, "sql": meta.get("sql", ""),
                        }
                        return StepExecutionOutcome(
                            action="request_dataset", status="ok", is_dataset_request=True,
                            observation_summary=f"Staged dataset '{new_name}' ({ds_ref.row_count} rows, columns={ds_ref.columns}).",
                            artifacts=[ds_ref.dataset_id],
                        )
                    except Exception as exc:
                        return StepExecutionOutcome(action="request_dataset", status="error", is_dataset_request=True, observation_summary=f"Dataset request failed: {exc}")

                code = (decision.python_code or "").strip()
                is_repair = working_state["last_error"] is not None
                working_state["last_code"] = code

                try:
                    exec_res = self.sandbox.execute(workspace, code)
                except SandboxSecurityError as sec_exc:
                    working_state["security_blocks"] += 1
                    working_state["last_error"] = f"SecurityViolation: {sec_exc}"
                    if working_state["security_blocks"] >= 2:
                        raise
                    return StepExecutionOutcome(
                        action="run_python", status="blocked", is_repair=is_repair, code_snapshot=code,
                        observation_summary=f"Security violation blocked code: {sec_exc}",
                    )

                if exec_res.status != "COMPLETED":
                    err_text = f"{exec_res.error_type or exec_res.status}: {exec_res.error_message or exec_res.stderr}"
                    working_state["last_error"] = err_text[:1500]
                    return StepExecutionOutcome(
                        action="run_python", status="error", is_repair=is_repair, code_snapshot=code,
                        observation_summary=f"Sandbox status={exec_res.status}. Error:\n{err_text[:1200]}",
                    )

                if not (exec_res.summary or exec_res.records or exec_res.tables or exec_res.charts or exec_res.stdout.strip()):
                    return StepExecutionOutcome(
                        action="run_python", status="error", is_repair=is_repair, code_snapshot=code,
                        observation_summary="Execution produced no stdout, summary, tables, or charts. Call `save_result(...)`.",
                    )

                working_state["last_error"] = None
                working_state["last_completed_result"] = exec_res
                working_state["last_completed_code"] = code
                if decision.conclusion_narrative.strip():
                    working_state["conclusion_narrative"] = decision.conclusion_narrative.strip()
                for a in exec_res.assumptions:
                    if a and a not in working_state["assumptions"]:
                        working_state["assumptions"].append(a)

                if self.evidence_store is not None:
                    self.evidence_store.save_artifact(
                        run_id=run_id, task_id=task.id, agent_name="python_research_agent",
                        artifact_type="python_execution", title=f"Step {step_idx}: {working_state['chosen_methodology'] or task.title}",
                        content={
                            "code": code, "summary": exec_res.summary, "records": exec_res.records[:50],
                            "tables": exec_res.tables, "charts": exec_res.charts,
                            "assumptions": list(working_state["assumptions"]), "stdout": exec_res.stdout[:2000],
                            "environment_versions": exec_res.environment_versions,
                        },
                        code_hash=exec_res.code_hash, dataset_ids=staged_ids,
                    )

                obs_payload = {
                    "stdout": exec_res.stdout[:600], "summary": exec_res.summary,
                    "record_count": len(exec_res.records), "records_preview": exec_res.records[:3],
                }
                auto_done = bool(exec_res.summary or exec_res.tables or exec_res.charts) and type(self.models) is ModelSelector
                return StepExecutionOutcome(
                    action="run_python", status="completed" if auto_done else "ok", is_repair=is_repair,
                    code_snapshot=code, artifacts=[a.name for a in exec_res.artifacts],
                    observation_summary=f"Sandbox COMPLETED in {exec_res.duration_ms}ms:\n{json.dumps(obs_payload, default=str)[:1000]}",
                    should_finish=auto_done,
                )

            loop.run(_decide, _execute)
            last_res = working_state["last_completed_result"]
            if last_res is None:
                return AutonomousPythonInvestigation(
                    tool_name="autonomous_research",
                    methodology=working_state["chosen_methodology"] or "autonomous_python",
                    calculation={"error": working_state["last_error"] or "No completed Python execution."},
                    generated_code=working_state["last_code"],
                    execution_steps=[s.to_dict() for s in loop.steps],
                    assumptions_checked=working_state["assumptions"],
                    dataset_ids=staged_ids, usages=loop.usages, repair_attempts=loop.repair_attempts,
                    status="FAILED", error=working_state["last_error"] or "Autonomous Python agent failed to produce a completed execution.",
                )

            calculation = {
                "tool": "autonomous_research",
                "methodology": working_state["chosen_methodology"] or "custom_autonomous_analysis",
                "objective": objective, "narrative": working_state["conclusion_narrative"],
                "summary": dict(last_res.summary), "records": list(last_res.records),
                "tables": list(last_res.tables), "charts": list(last_res.charts),
                "stdout": last_res.stdout, "assumptions": list(working_state["assumptions"]),
                "warnings": list(last_res.warnings), "environment_versions": dict(last_res.environment_versions),
                "code_hash": last_res.code_hash, "total_steps": len(loop.steps),
                "repair_attempts": loop.repair_attempts, "dataset_ids": staged_ids,
            }
            if self.evidence_store is not None:
                self.evidence_store.save_note(
                    run_id=run_id, task_id=task.id, agent_name="python_research_agent",
                    claim=working_state["conclusion_narrative"] or f"Completed autonomous Python analysis ({calculation['methodology']})",
                    methodology=calculation["methodology"], confidence=0.85 if not last_res.warnings else 0.72,
                    status="completed", metadata={"code_hash": last_res.code_hash, "dataset_ids": staged_ids},
                )

            return AutonomousPythonInvestigation(
                tool_name="autonomous_research", methodology=calculation["methodology"],
                calculation=calculation, generated_code=working_state["last_completed_code"],
                execution_steps=[s.to_dict() for s in loop.steps],
                artifacts=[a.to_dict() for a in last_res.artifacts],
                assumptions_checked=list(working_state["assumptions"]),
                dataset_ids=staged_ids, usages=loop.usages, repair_attempts=loop.repair_attempts,
                status="COMPLETED",
            )
        finally:
            self.sandbox.destroy_workspace(workspace)

    def execute(
        self,
        task: AnalysisTask,
        upstream_dataset_ids: list[str],
    ) -> tuple[str, dict[str, Any], ModelUsageItem | None, int]:
        if not upstream_dataset_ids:
            raise ValueError(f"Python task '{task.id}' has no upstream datasets to inspect.")

        t0 = time.perf_counter()
        if self.should_use_autonomous_mode(task):
            inv = self.investigate(task=task, upstream_dataset_ids=upstream_dataset_ids, run_id="run_exec")
            if inv.status == "FAILED":
                raise RuntimeError(inv.error or "Autonomous Python research failed.")
            return inv.tool_name, inv.calculation, (inv.usages[0] if inv.usages else None), int((time.perf_counter() - t0) * 1000)

        spec = task.python_spec
        tool_name = spec.tool if spec else "statistical_summary"
        datasets = [self.dataset_store.get_dataframe(ds_id) for ds_id in upstream_dataset_ids]
        usage_item: ModelUsageItem | None = None
        if tool_name == "auto":
            tool_name, usage_item = self._select_tool_via_llm(task, datasets[0])

        params = {
            "value_column": spec.value_column if spec else None,
            "label_column": spec.label_column if spec else None,
            "time_column": spec.time_column if spec else None,
            "secondary_value_column": spec.secondary_value_column if spec else None,
            "threshold": spec.threshold if spec else None,
            "horizon": spec.horizon if spec else None,
        }
        result = run_registered_analysis_tool(tool_name=tool_name, datasets=datasets, params=params)
        return tool_name, result, usage_item, int((time.perf_counter() - t0) * 1000)

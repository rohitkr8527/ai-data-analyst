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
from app.services.analysis_tools import (
    ANALYSIS_TOOL_REGISTRY,
    run_registered_analysis_tool,
)
from app.services.dataset_broker import AuthorizedDatasetBroker, summarize_dataframe
from app.services.dataset_store import DatasetStore
from app.services.evidence_store import SharedEvidenceStore
from app.services.models import ModelSelector, TaskType
from app.services.sandbox import SandboxExecutionService

logger = logging.getLogger(__name__)


AUTONOMOUS_PYTHON_SYSTEM_PROMPT = """You are an Autonomous Senior Data Scientist and Python Research Agent.

Mission:
Given an analytical objective and authorized staged datasets, independently determine the right
analytical methodology, inspect schemas/distributions, write clean Python code dynamically, execute
it inside an isolated sandbox, observe stdout/tables/charts/tracebacks, debug any errors, request
additional authorized data if existing data is insufficient, validate statistical assumptions, and
deliver verified numerical evidence.

Available libraries & sandbox helpers (pre-loaded globals inside the sandbox):
- Pre-loaded modules: `np` (numpy), `pd` (pandas), `scipy`, `stats` (scipy.stats), `math`, `json`.
- Additional installed packages you may `import`: `polars as pl`, `statsmodels.api as sm`,
  `statsmodels.tsa.seasonal`, `sklearn` (clustering, regression, decomposition, preprocessing,
  metrics, outlier detection), `matplotlib.pyplot as plt`, `plotly.express as px`,
  `plotly.graph_objects as go`.
  Note: If converting a pandas DataFrame containing string columns to Polars without pyarrow,
  use `pl.DataFrame(df.to_dict(orient="records"))`.
- Data access helpers:
  - `df`: primary DataFrame (copy of the first staged dataset).
  - `DATASETS`: dict `{name: DataFrame}` of all staged datasets.
  - `list_datasets()`: returns `{name: {"rows": int, "columns": [...]}}`.
  - `load_dataset(name=None)`: loads a fresh copy of dataset `name`.
- Output recording helpers (ALWAYS call `save_result` before finishing an analysis step):
  - `save_result(summary_dict, records=None, assumptions=None, warnings_list=None)`:
    Stores structured metrics in `summary_dict` (JSON numbers/strings/lists), optional `records`
    (list of dicts or DataFrame, max 200 rows), `assumptions` (list[str]), `warnings_list` (list[str]).
  - `save_table(title, df_or_rows)`: stores a named structured table for the final report.
  - `save_chart(title, kind, labels, values, unit="USD", x_label="", y_label="", series=None)`:
    stores an interactive chart spec (`kind` in `bar`, `line`, `waterfall`, `pie`, `grouped_bar`).
  - `save_figure(fig, filename="figure.png")`: saves a Matplotlib `Figure` (`.png`) or Plotly figure (`.json`).
  - `save_state(key, value)` / `load_state(key, default=None)`: persists JSON intermediates across steps.

Sandbox Rules (strict):
- No network/HTTP/socket access, no subprocesses, no reading/writing outside the workspace.
- Never fabricate numbers or print placeholder guesses. Every number in `conclusion_narrative` MUST
  come directly from observed stdout, `summary`, or `records` of a COMPLETED sandbox execution step.
- Do not restrict yourself to basic averages or fixed templates when the problem warrants deeper
  statistical analysis (e.g., cohort analysis, RFM/KMeans clustering, OLS/GLM regression,
  seasonal decomposition, change-point detection, bootstrap CIs, permutation tests, Mann-Whitney/t-tests,
  ANOVA, chi-square, outlier isolation, Pareto concentration, mix-vs-rate decomposition, forecast backtesting).

Workflow per step:
1. On Step 1, choose `action="run_python"` and write a complete, self-contained Python script that computes all required metrics, checks assumptions, and calls `save_result(summary_dict, records=...)` (plus `save_table(...)` / `save_chart(...)` as appropriate) in a single pass.
2. Keep `summary_dict` in `save_result(...)` focused on scalar metrics (numbers/strings) rather than embedding huge nested arrays inside `summary_dict` (use `save_table(...)` for tabular breakdowns).
3. If a previous step returned a traceback, choose `phase="debug"` and `action="run_python"` with fixed code addressing the exact error.
4. Only choose `action="request_dataset"` if the staged datasets completely lack a required table/column and budget allows.
"""


class ToolSelectionDecision(BaseModel):
    tool: Literal[
        "period_comparison",
        "revenue_decomposition",
        "contribution_analysis",
        "trend_analysis",
        "anomaly_detection",
        "correlation_analysis",
        "distribution_analysis",
        "statistical_summary",
        "data_quality_check",
        "forecast_linear",
    ]
    reasoning: str = ""


class PythonStepDecision(BaseModel):
    """Closed structured decision schema compatible with OpenAI Responses structured outputs."""

    phase: Literal["explore", "analyze", "debug", "request_data", "refine", "conclude"]
    action: Literal["run_python", "request_dataset", "finish"]
    reasoning: str = Field(description="Why this analytical strategy, revision, data request, or conclusion is chosen.")
    chosen_methodology: str = Field(
        default="",
        description="Name of the analytical/statistical method chosen (e.g., 'RFM KMeans Clustering', 'OLS Regression + Bootstrap CI').",
    )
    action_summary: str = Field(default="", description="Concise one-line summary of the step action.")
    python_code: str = Field(default="", description="Complete self-contained Python script when action='run_python'.")
    dataset_request: str = Field(
        default="",
        description="Validated SELECT SQL or clear natural-language data need when action='request_dataset'.",
    )
    assumptions_checked: list[str] = Field(
        default_factory=list,
        description="Statistical or data-quality assumptions verified or tested in this step.",
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    conclusion_narrative: str = Field(
        default="",
        description="Evidence-backed analytical conclusion citing only observed numbers when action='finish'.",
    )


@dataclass
class AutonomousPythonInvestigation:
    """Rich output returned by Autonomous Python Research Agent."""

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
    """Autonomous Python Research Agent capable of writing, debugging, and validating custom Python analyses."""

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

    def _select_tool_via_llm(
        self,
        task: AnalysisTask,
        df: pd.DataFrame,
    ) -> tuple[str, ModelUsageItem | None]:
        spec = task.python_spec
        objective = spec.objective if (spec and spec.objective) else task.purpose
        tool_list = [
            {"name": k, "description": (v.__doc__ or k).strip()}
            for k, v in ANALYSIS_TOOL_REGISTRY.items()
        ]
        schema_summary = summarize_dataframe(df)
        payload = {
            "task_objective": objective,
            "task_description": task.purpose,
            "data_schema": schema_summary,
            "available_tools": tool_list,
        }
        decision, usage = self.models.parse_structured(
            task_type=TaskType.REASONING,
            response_model=ToolSelectionDecision,
            system_instructions=(
                "You are an expert Python data analyst selecting the best deterministic analytical "
                "utility for the given dataset and analytical objective."
            ),
            input_payload=payload,
            custom_effort="low",
        )
        return decision.tool, ModelUsageItem(
            agent_role="Python Analysis Agent",
            task_type=usage.task_type,
            model=usage.model,
            reasoning_effort=usage.reasoning_effort,
            duration_ms=usage.duration_ms,
            tokens=usage.input_tokens + usage.output_tokens,
            cost_usd=usage.cost_usd,
        )

    def should_use_autonomous_mode(self, task: AnalysisTask) -> bool:
        spec = task.python_spec
        if spec is None:
            return False
        if spec.tool in ("autonomous_research", "custom_python"):
            return True
        if getattr(spec, "mode", "auto") == "autonomous":
            return True
        return False

    def investigate(
        self,
        task: AnalysisTask,
        upstream_dataset_ids: list[str],
        run_id: str = "run_adhoc",
        sql_provider: Callable[[str], tuple[str, dict[str, Any]]] | None = None,
        budget: AgentLoopBudget | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AutonomousPythonInvestigation:
        """Execute a bounded Understand -> Plan -> Act -> Observe -> Reflect -> Finish research loop."""
        spec = task.python_spec
        objective = (spec.objective if spec and spec.objective else task.purpose) or task.title
        suggested_methods = list(getattr(spec, "analysis_methods", []) or [])

        workspace = self.sandbox.create_workspace(run_id=run_id, task_id=task.id)
        staged_ids: list[str] = list(upstream_dataset_ids)
        dataset_profiles: dict[str, Any] = {}
        staged_names: list[str] = []

        try:
            for idx, ds_id in enumerate(upstream_dataset_ids):
                df = self.dataset_store.get_dataframe(ds_id)
                ds_name = spec.input_task_ids[idx] if (spec and idx < len(spec.input_task_ids)) else f"dataset_{idx + 1}"
                self.sandbox.stage_dataframe(workspace, ds_name, df)
                dataset_profiles[ds_name] = summarize_dataframe(df, dataset_name=ds_name)
                staged_names.append(ds_name)

            loop_budget = budget or AgentLoopBudget.from_settings()
            loop = AutonomousAgentLoop(
                agent_name="python_research_agent",
                run_id=run_id,
                task_id=task.id,
                objective=objective,
                budget=loop_budget,
                on_event=on_event,
            )

            working_state: dict[str, Any] = {
                "last_code": "",
                "last_error": None,
                "last_completed_result": None,
                "last_completed_code": "",
                "chosen_methodology": suggested_methods[0] if suggested_methods else "",
                "assumptions": [],
                "conclusion_narrative": "",
                "security_blocks": 0,
            }

            def _decide(step_idx: int, active_loop: AutonomousAgentLoop) -> tuple[PythonStepDecision, ModelUsageItem | None]:
                schema_hint = ""
                if self.dataset_broker is not None and not dataset_profiles:
                    try:
                        schema_hint = self.dataset_broker.inspect_schema()[:900]
                    except Exception:
                        schema_hint = ""
                payload = {
                    "objective": objective,
                    "task_title": task.title,
                    "task_description": task.purpose,
                    "suggested_methods": suggested_methods,
                    "step_index": step_idx,
                    "remaining_steps": active_loop.remaining_steps(),
                    "max_steps": active_loop.budget.max_steps,
                    "dataset_requests_used": active_loop.dataset_requests,
                    "max_dataset_requests": active_loop.budget.max_dataset_requests,
                    "staged_datasets": dataset_profiles,
                    "previous_code": working_state["last_code"][:2000] if working_state["last_error"] else "",
                    "last_error": working_state["last_error"],
                    "recent_steps": active_loop.build_compact_history(max_recent=2),
                    "has_completed_analysis": working_state["last_completed_result"] is not None,
                    "authorized_db_schema_overview": schema_hint,
                }
                decision, usage = self.models.parse_structured(
                    task_type=TaskType.PYTHON_RESEARCH,
                    response_model=PythonStepDecision,
                    system_instructions=AUTONOMOUS_PYTHON_SYSTEM_PROMPT,
                    input_payload=payload,
                    custom_effort="low",
                )
                return decision, ModelUsageItem(
                    agent_role="Python Research Agent",
                    task_type=usage.task_type,
                    model=usage.model,
                    reasoning_effort=usage.reasoning_effort,
                    duration_ms=usage.duration_ms,
                    tokens=usage.input_tokens + usage.output_tokens,
                    cost_usd=usage.cost_usd,
                )

            def _execute(step_idx: int, decision: PythonStepDecision, active_loop: AutonomousAgentLoop) -> StepExecutionOutcome:
                if decision.chosen_methodology:
                    working_state["chosen_methodology"] = decision.chosen_methodology
                for a in decision.assumptions_checked:
                    if a and a not in working_state["assumptions"]:
                        working_state["assumptions"].append(a)

                # Guard against premature finish before any code executed
                if decision.action == "finish" and working_state["last_completed_result"] is None and decision.python_code.strip():
                    decision_action = "run_python"
                elif decision.action == "finish" and working_state["last_completed_result"] is None and not decision.python_code.strip():
                    return StepExecutionOutcome(
                        action="finish_rejected",
                        status="error",
                        observation_summary=(
                            "Cannot conclude without any completed sandbox Python execution. "
                            "Write and execute Python code with `save_result(...)` first."
                        ),
                        should_finish=False,
                    )
                else:
                    decision_action = decision.action

                if decision_action == "finish":
                    working_state["conclusion_narrative"] = decision.conclusion_narrative
                    return StepExecutionOutcome(
                        action="finish",
                        status="completed",
                        observation_summary=decision.conclusion_narrative or "Investigation concluded with verified sandbox results.",
                        should_finish=True,
                    )

                if decision_action == "request_dataset":
                    if active_loop.dataset_requests >= active_loop.budget.max_dataset_requests:
                        return StepExecutionOutcome(
                            action="request_dataset",
                            status="error",
                            observation_summary=(
                                "Dataset request budget exhausted. Proceed using the already staged datasets: "
                                f"{list(dataset_profiles.keys())}."
                            ),
                            should_finish=False,
                        )
                    if self.dataset_broker is None:
                        return StepExecutionOutcome(
                            action="request_dataset",
                            status="error",
                            observation_summary="AuthorizedDatasetBroker is not attached; use currently staged datasets.",
                            should_finish=False,
                        )
                    req_text = decision.dataset_request.strip()
                    try:
                        ds_ref, new_df, meta = self.dataset_broker.request_dataset(
                            run_id=run_id,
                            task_id=task.id,
                            request_text_or_sql=req_text,
                            sql_provider=sql_provider,
                            reason=decision.reasoning or objective,
                        )
                        new_name = f"requested_{active_loop.dataset_requests + 1}"
                        self.sandbox.stage_dataframe(workspace, new_name, new_df)
                        staged_ids.append(ds_ref.dataset_id)
                        staged_names.append(new_name)
                        dataset_profiles[new_name] = summarize_dataframe(new_df, dataset_name=new_name)
                        dataset_profiles[new_name]["dataset_id"] = ds_ref.dataset_id
                        dataset_profiles[new_name]["sql"] = meta.get("sql", "")
                        return StepExecutionOutcome(
                            action="request_dataset",
                            status="ok",
                            is_dataset_request=True,
                            observation_summary=(
                                f"Acquired and staged dataset '{new_name}' ({ds_ref.row_count} rows, "
                                f"columns={ds_ref.columns}, dataset_id={ds_ref.dataset_id}). "
                                f"Sample summary: {json.dumps(dataset_profiles[new_name], default=str)[:900]}"
                            ),
                            artifacts=[ds_ref.dataset_id],
                        )
                    except Exception as exc:
                        return StepExecutionOutcome(
                            action="request_dataset",
                            status="error",
                            is_dataset_request=True,
                            observation_summary=f"Dataset request failed: {exc}",
                        )

                # action == "run_python"
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
                        action="run_python",
                        status="blocked",
                        is_repair=is_repair,
                        code_snapshot=code,
                        observation_summary=(
                            f"Security violation blocked code: {sec_exc}. Use only pre-approved data science "
                            "libraries (pandas, numpy, scipy, sklearn, statsmodels, polars, matplotlib, plotly) "
                            "and workspace data helpers (`load_dataset`, `save_result`, `save_table`, `save_chart`)."
                        ),
                    )

                if exec_res.status != "COMPLETED":
                    err_text = f"{exec_res.error_type or exec_res.status}: {exec_res.error_message or exec_res.stderr}"
                    working_state["last_error"] = err_text[:1500]
                    return StepExecutionOutcome(
                        action="run_python",
                        status="error",
                        is_repair=is_repair,
                        code_snapshot=code,
                        observation_summary=(
                            f"Sandbox status={exec_res.status}. Error:\n{err_text[:1200]}\n"
                            f"Stdout before error:\n{exec_res.stdout[:300]}"
                        ),
                    )

                # Completed execution
                if (
                    not exec_res.summary
                    and not exec_res.records
                    and not exec_res.tables
                    and not exec_res.charts
                    and not exec_res.stdout.strip()
                ):
                    return StepExecutionOutcome(
                        action="run_python",
                        status="error",
                        is_repair=is_repair,
                        code_snapshot=code,
                        observation_summary=(
                            "Execution finished with exit_code=0 but produced no stdout, summary, tables, or charts. "
                            "Call `save_result(summary_dict, records=...)` and/or `print(...)` so outputs are observable."
                        ),
                    )

                working_state["last_error"] = None
                working_state["last_completed_result"] = exec_res
                working_state["last_completed_code"] = code
                if decision.conclusion_narrative.strip():
                    working_state["conclusion_narrative"] = decision.conclusion_narrative.strip()
                for a in exec_res.assumptions:
                    if a and a not in working_state["assumptions"]:
                        working_state["assumptions"].append(a)

                obs_payload = {
                    "stdout": exec_res.stdout[:600],
                    "summary": exec_res.summary,
                    "record_count": len(exec_res.records),
                    "records_preview": exec_res.records[:3],
                    "tables": [{"title": t.get("title"), "rows": len(t.get("rows", []))} for t in exec_res.tables],
                    "charts": [{"title": c.get("title"), "kind": c.get("kind")} for c in exec_res.charts],
                    "warnings": exec_res.warnings[:3],
                    "artifacts": [a.name for a in exec_res.artifacts],
                }
                if self.evidence_store is not None:
                    self.evidence_store.save_artifact(
                        run_id=run_id,
                        task_id=task.id,
                        agent_name="python_research_agent",
                        artifact_type="python_execution",
                        title=f"Step {step_idx}: {working_state['chosen_methodology'] or task.title}",
                        content={
                            "code": code,
                            "summary": exec_res.summary,
                            "records": exec_res.records[:50],
                            "tables": exec_res.tables,
                            "charts": exec_res.charts,
                            "assumptions": list(working_state["assumptions"]),
                            "stdout": exec_res.stdout[:2000],
                            "environment_versions": exec_res.environment_versions,
                        },
                        code_hash=exec_res.code_hash,
                        dataset_ids=staged_ids,
                    )

                auto_done = bool(exec_res.summary or exec_res.tables or exec_res.charts) and type(self.models) is ModelSelector
                return StepExecutionOutcome(
                    action="run_python",
                    status="completed" if auto_done else "ok",
                    is_repair=is_repair,
                    code_snapshot=code,
                    artifacts=[a.name for a in exec_res.artifacts],
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
                    dataset_ids=staged_ids,
                    usages=loop.usages,
                    repair_attempts=loop.repair_attempts,
                    status="FAILED",
                    error=working_state["last_error"] or "Autonomous Python agent failed to produce a completed execution.",
                )

            artifacts_meta = [a.to_dict() for a in last_res.artifacts]
            calculation = {
                "tool": "autonomous_research",
                "methodology": working_state["chosen_methodology"] or "custom_autonomous_analysis",
                "objective": objective,
                "narrative": working_state["conclusion_narrative"],
                "summary": dict(last_res.summary),
                "records": list(last_res.records),
                "tables": list(last_res.tables),
                "charts": list(last_res.charts),
                "stdout": last_res.stdout,
                "assumptions": list(working_state["assumptions"]),
                "warnings": list(last_res.warnings),
                "environment_versions": dict(last_res.environment_versions),
                "code_hash": last_res.code_hash,
                "total_steps": len(loop.steps),
                "repair_attempts": loop.repair_attempts,
                "dataset_ids": staged_ids,
            }
            if self.evidence_store is not None:
                self.evidence_store.save_note(
                    run_id=run_id,
                    task_id=task.id,
                    agent_name="python_research_agent",
                    claim=working_state["conclusion_narrative"] or f"Completed autonomous Python analysis ({calculation['methodology']})",
                    methodology=calculation["methodology"],
                    confidence=0.85 if not last_res.warnings else 0.72,
                    status="completed",
                    metadata={
                        "code_hash": last_res.code_hash,
                        "dataset_ids": staged_ids,
                        "total_steps": len(loop.steps),
                        "repair_attempts": loop.repair_attempts,
                    },
                )

            return AutonomousPythonInvestigation(
                tool_name="autonomous_research",
                methodology=calculation["methodology"],
                calculation=calculation,
                generated_code=working_state["last_completed_code"],
                execution_steps=[s.to_dict() for s in loop.steps],
                artifacts=artifacts_meta,
                assumptions_checked=list(working_state["assumptions"]),
                dataset_ids=staged_ids,
                usages=loop.usages,
                repair_attempts=loop.repair_attempts,
                status="COMPLETED",
            )
        finally:
            self.sandbox.destroy_workspace(workspace)

    def execute(
        self,
        task: AnalysisTask,
        upstream_dataset_ids: list[str],
    ) -> tuple[str, dict[str, Any], ModelUsageItem | None, int]:
        """Backward-compatible entrypoint supporting both autonomous investigation and utility tools."""
        if not upstream_dataset_ids:
            raise ValueError(f"Python task '{task.id}' has no upstream datasets to inspect.")

        t0 = time.perf_counter()
        if self.should_use_autonomous_mode(task):
            inv = self.investigate(
                task=task,
                upstream_dataset_ids=upstream_dataset_ids,
                run_id="run_exec",
            )
            if inv.status == "FAILED":
                raise RuntimeError(inv.error or "Autonomous Python research failed.")
            primary_usage = inv.usages[0] if inv.usages else None
            duration_ms = int((time.perf_counter() - t0) * 1000)
            return inv.tool_name, inv.calculation, primary_usage, duration_ms

        spec = task.python_spec
        tool_name = spec.tool if spec else "statistical_summary"
        datasets = [self.dataset_store.get_dataframe(ds_id) for ds_id in upstream_dataset_ids]
        primary_df = datasets[0]

        usage_item: ModelUsageItem | None = None
        if tool_name == "auto":
            tool_name, usage_item = self._select_tool_via_llm(task, primary_df)

        params = {
            "value_column": spec.value_column if spec else None,
            "label_column": spec.label_column if spec else None,
            "time_column": spec.time_column if spec else None,
            "secondary_value_column": spec.secondary_value_column if spec else None,
            "threshold": spec.threshold if spec else None,
            "horizon": spec.horizon if spec else None,
        }
        result = run_registered_analysis_tool(
            tool_name=tool_name,
            datasets=datasets,
            params=params,
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        return tool_name, result, usage_item, duration_ms

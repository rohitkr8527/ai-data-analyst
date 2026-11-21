"""Hypothesis Agent (gpt-6.1-sol).

Executes only when an investigation requires explaining or testing potential reasons
behind an observed business variance. Distinguishes observed patterns, measured
accounting contributions, and causal explanations, and proposes concrete test tasks.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Literal
import pandas as pd
from pydantic import BaseModel, Field

from app.agents.runtime import AgentLoopBudget, AutonomousAgentLoop, StepExecutionOutcome
from app.orchestration.contracts import (
    AnalysisPlan,
    AnalysisTask,
    HypothesisRecord,
    HypothesisSet,
    TaskDraft,
)
from app.schemas import ModelUsageItem
from app.services.dataset_broker import summarize_dataframe
from app.services.dataset_store import DatasetStore
from app.services.models import ModelSelector, TaskType
from app.services.sandbox import SandboxExecutionService

logger = logging.getLogger(__name__)


class HypothesisStepDecision(BaseModel):
    """Closed structured decision schema for iterative statistical hypothesis testing and revision."""

    hypothesis_id: str = Field(description="ID of the hypothesis currently being tested or evaluated.")
    phase: Literal["plan_test", "run_python_test", "revise", "conclude"]
    action: Literal["run_python_test", "accept_verdict", "reject_and_revise", "mark_inconclusive"]
    reasoning: str = Field(description="Scientific reasoning for the chosen test, rejection, revision, or verdict.")
    python_code: str = Field(
        default="",
        description="Self-contained Python code using scipy.stats/statsmodels/pandas and `save_result({...})` when action='run_python_test'.",
    )
    verdict: Literal["supported", "not_supported", "rejected", "revised", "inconclusive"] = "inconclusive"
    revised_statement: str = Field(
        default="",
        description="Updated falsifiable hypothesis statement when action='reject_and_revise'.",
    )
    revised_dimension: str = Field(default="", description="Dimension/variable for the revised hypothesis.")
    revised_target_entity: str = Field(default="", description="Optional target entity for the revised hypothesis.")
    measured_effect: float | None = Field(default=None, description="Measured numerical effect or correlation/difference.")
    p_value: float | None = Field(default=None, description="Observed statistical p-value if a test was executed.")
    statistical_evidence: str = Field(
        default="",
        description="Concise statistical summary citing exact numbers from observed sandbox results.",
    )
    assessment: str = Field(default="", description="Evidence-backed assessment respecting correlation vs causation.")
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)

HYPOTHESIS_SYSTEM_PROMPT = """You are the Principal Hypothesis & Root-Cause Reasoning Agent for ClassicModels.
Given the user's question, the baseline variance, and initial dimensional observations, formulate 3 to 5 measurable, falsifiable business hypotheses.

### Mandatory Epistemic Guardrails:
1. Distinguish strictly between:
   - `observed_pattern`: An operational or temporal movement in order volume, AOV, or fulfillment status.
   - `accounting_contribution`: A dimensional share of variance (e.g., Classic Cars or USA accounting for X% of the net decline).
   - `causal_candidate`: A plausible business mechanism that requires additional evidence and cannot be proven from correlation alone.
2. Never claim that correlation or accounting attribution proves external causation.
3. For each hypothesis, specify:
   - `id`: unique snake_case ID (e.g. `hyp_classic_cars`, `hyp_order_volume`, `hyp_usa_market`).
   - `statement`: clear, testable statement.
   - `hypothesis_type`: `observed_pattern`, `accounting_contribution`, or `causal_candidate`.
   - `dimension`: one of `product_line`, `product`, `country`, `customer`, `order_status`, `sales_rep`, `office`, `order_volume`, `average_order_value`, `trend`.
   - `target_entity`: specific entity name if applicable (e.g. `"Classic Cars"`, `"USA"`).
   - `expected_direction`: `"decrease"`, `"increase"`, or `"any"`.
   - `evidence_required`: exact measurement needed to support or refute the hypothesis.
   - `priority`: 1 (highest) to 5.
   - `alternative_explanations`: 1 to 3 competing explanations or confounding factors (e.g. cross-dimensional overlap between USA klanten and Classic Cars).
   - `test`: optional `TaskDraft` (SQL or Python task) ONLY if the required evidence is not already covered by existing tasks in `existing_task_ids`.
"""


class HypothesisAgent:
    """Generates measurable hypotheses and evaluates them against verified evidence."""

    def __init__(self, model_selector: ModelSelector | None = None):
        self.selector = model_selector or ModelSelector()

    def generate(
        self,
        task: AnalysisTask,
        plan: AnalysisPlan,
        upstream_summaries: list[dict[str, Any]],
        existing_task_ids: list[str],
    ) -> tuple[list[HypothesisRecord], list[TaskDraft], ModelUsageItem]:
        """Generate structured hypotheses and optional follow-up test tasks via gpt-6.1-sol."""
        payload = {
            "question": plan.question,
            "objective": plan.objective,
            "period": plan.period,
            "focus": task.hypothesis_spec.focus if task.hypothesis_spec else plan.objective,
            "upstream_observations": upstream_summaries,
            "existing_task_ids": existing_task_ids,
        }

        parsed, usage = self.selector.parse_structured(
            task_type=TaskType.HYPOTHESIS_GENERATION,
            response_model=HypothesisSet,
            system_instructions=HYPOTHESIS_SYSTEM_PROMPT,
            input_payload=payload,
            custom_effort="low",
        )

        usage_item = ModelUsageItem(
            agent_role="Hypothesis Agent",
            task_type=usage.task_type,
            model=usage.model,
            reasoning_effort=usage.reasoning_effort,
            duration_ms=usage.duration_ms,
            tokens=usage.input_tokens + usage.output_tokens,
            cost_usd=usage.cost_usd,
        )

        records: list[HypothesisRecord] = []
        proposed_tasks: list[TaskDraft] = []

        dim_to_default_task = {
            "product_line": "product_line_bridge",
            "country": "country_bridge",
            "product": "product_bridge",
            "customer": "customer_bridge",
            "order_status": "order_status",
            "order_volume": "baseline",
            "average_order_value": "baseline",
            "trend": "trend",
        }

        for h in parsed.hypotheses:
            linked_tasks: list[str] = []
            default_tid = dim_to_default_task.get(h.dimension)
            if default_tid and default_tid in existing_task_ids:
                linked_tasks.append(default_tid)
            if "baseline" in existing_task_ids and "baseline" not in linked_tasks:
                linked_tasks.append("baseline")

            if h.test is not None and h.test.id not in existing_task_ids:
                proposed_tasks.append(h.test)
                linked_tasks.append(h.test.id)

            records.append(
                HypothesisRecord(
                    id=h.id,
                    statement=h.statement,
                    hypothesis_type=h.hypothesis_type,
                    dimension=h.dimension,
                    target_entity=h.target_entity,
                    expected_direction=h.expected_direction,
                    evidence_required=h.evidence_required,
                    priority=h.priority,
                    alternative_explanations=list(h.alternative_explanations),
                    test_task_ids=linked_tasks,
                    status="untested",
                )
            )

        return records, proposed_tasks, usage_item

    @staticmethod
    def evaluate_hypotheses(
        hypotheses: list[HypothesisRecord],
        datasets_by_task: dict[str, list[dict[str, Any]]],
        query_ids_by_task: dict[str, str],
        baseline_delta: float | None,
    ) -> list[HypothesisRecord]:
        """Deterministically evaluate hypotheses against validated datasets."""
        evaluated: list[HypothesisRecord] = []

        dim_task_lookup = {
            "product_line": "product_line_bridge",
            "country": "country_bridge",
            "product": "product_bridge",
            "customer": "customer_bridge",
            "order_status": "order_status",
            "trend": "trend",
        }

        # Ensure standard accounting hypotheses for product_line and country exist if none were generated
        existing_dims = {h.dimension for h in hypotheses}
        working_list = list(hypotheses)
        for dim_key, tid, label_title in (
            ("product_line", "product_line_bridge", "Product Line"),
            ("country", "country_bridge", "Country"),
        ):
            if dim_key not in existing_dims and tid in datasets_by_task and datasets_by_task[tid]:
                rows = datasets_by_task[tid]
                worst = min(
                    rows,
                    key=lambda r: float(r.get("current", 0.0) or 0.0) - float(r.get("previous", 0.0) or 0.0),
                )
                ent = str(worst.get("label") or list(worst.values())[0])
                working_list.append(
                    HypothesisRecord(
                        id=f"hyp_{tid}",
                        statement=f"Revenue variance was primarily concentrated in {ent} ({label_title}).",
                        hypothesis_type="accounting_contribution",
                        dimension=dim_key,
                        target_entity=ent,
                        expected_direction="decrease" if (baseline_delta or 0.0) <= 0 else "increase",
                        evidence_required=f"Compare period-over-period variance across {label_title.lower()}s and reconcile with baseline.",
                        priority=1,
                        alternative_explanations=[
                            "Accounting contributions across product lines and countries overlap and are not additive."
                        ],
                        test_task_ids=[tid, "baseline"],
                    )
                )

        base_rows = datasets_by_task.get("baseline", [])
        base_qid = query_ids_by_task.get("baseline")

        for h in working_list:
            ev_qids: list[str] = []
            if base_qid:
                ev_qids.append(base_qid)

            if h.dimension in ("order_volume", "average_order_value") and base_rows:
                brow = base_rows[0]
                prev_rev = float(brow.get("previous_revenue") or 0.0)
                curr_rev = float(brow.get("current_revenue") or 0.0)
                prev_ord = int(brow.get("previous_orders") or 0)
                curr_ord = int(brow.get("current_orders") or 0)
                prev_aov = (prev_rev / prev_ord) if prev_ord > 0 else 0.0
                curr_aov = (curr_rev / curr_ord) if curr_ord > 0 else 0.0

                if h.dimension == "order_volume":
                    effect = round((curr_ord - prev_ord) * prev_aov, 2)
                else:
                    effect = round((curr_aov - prev_aov) * prev_ord, 2)

                share = round((effect / baseline_delta) * 100.0, 2) if baseline_delta and abs(baseline_delta) > 1e-6 else 0.0
                supported = (baseline_delta is not None and effect * baseline_delta > 0 and abs(share) >= 15.0)
                evaluated.append(
                    h.model_copy(
                        update={
                            "status": "supported" if supported else "not_supported",
                            "measured_effect": effect,
                            "share_of_change": share,
                            "evidence_ids": ev_qids,
                            "confidence": 0.9 if supported else 0.75,
                            "assessment": (
                                f"Measured {h.dimension.replace('_', ' ')} effect of ${effect:,.2f} "
                                f"({share:.1f}% of net baseline change)."
                            ),
                        }
                    )
                )
                continue

            tid = dim_task_lookup.get(h.dimension)
            if not tid:
                # Check if any custom test task completed
                tid = next((t for t in h.test_task_ids if t in datasets_by_task and t != "baseline"), None)

            rows = datasets_by_task.get(tid or "", [])
            if tid and tid in query_ids_by_task and query_ids_by_task[tid] not in ev_qids:
                ev_qids.insert(0, query_ids_by_task[tid])

            if not rows:
                evaluated.append(
                    h.model_copy(
                        update={
                            "status": "inconclusive",
                            "evidence_ids": ev_qids,
                            "confidence": 0.4,
                            "assessment": "Required dimensional dataset was empty or not executed.",
                        }
                    )
                )
                continue

            if h.dimension == "order_status":
                canc = [r for r in rows if str(r.get("label", "")).lower() == "cancelled"]
                val_c = float(canc[0].get("total_value") or 0.0) if canc else 0.0
                evaluated.append(
                    h.model_copy(
                        update={
                            "status": "supported" if val_c > 0 else "not_supported",
                            "measured_effect": round(val_c, 2),
                            "evidence_ids": ev_qids,
                            "confidence": 0.85,
                            "assessment": (
                                f"Cancelled orders totaled ${val_c:,.2f} during the period "
                                "(excluded from completed revenue per accounting policy)."
                            ),
                        }
                    )
                )
                continue

            # Bridge table with previous/current
            scored_rows = []
            for r in rows:
                prev_v = float(r.get("previous") or r.get("previous_revenue") or 0.0)
                curr_v = float(r.get("current") or r.get("current_revenue") or r.get("value") or 0.0)
                chg = round(curr_v - prev_v, 2) if ("previous" in r or "previous_revenue" in r) else round(curr_v, 2)
                lbl = str(r.get("label") or r.get("product") or r.get("country") or r.get("customer") or list(r.values())[0])
                scored_rows.append((lbl, prev_v, curr_v, chg))

            scored_rows.sort(key=lambda item: item[3])
            matched = None
            if h.target_entity:
                t_low = h.target_entity.lower()
                matched = next((item for item in scored_rows if t_low in item[0].lower()), None)
            if matched is None and scored_rows:
                matched = scored_rows[0] if (baseline_delta or 0.0) <= 0 else scored_rows[-1]

            if matched is None:
                evaluated.append(h.model_copy(update={"status": "inconclusive", "evidence_ids": ev_qids}))
                continue

            lbl, prev_v, curr_v, chg = matched
            share = round((chg / baseline_delta) * 100.0, 2) if (baseline_delta and abs(baseline_delta) > 1e-6) else None

            if baseline_delta is not None and abs(baseline_delta) > 1e-6:
                same_sign = (chg * baseline_delta) > 0
                is_supported = same_sign and abs(share or 0.0) >= 10.0
            else:
                is_supported = abs(chg) > 0

            evaluated.append(
                h.model_copy(
                    update={
                        "target_entity": h.target_entity or lbl,
                        "status": "supported" if is_supported else "not_supported",
                        "measured_effect": chg,
                        "share_of_change": share,
                        "evidence_ids": ev_qids,
                        "confidence": 0.9 if is_supported else 0.75,
                        "assessment": (
                            f"{lbl} changed by ${chg:,.2f} (from ${prev_v:,.2f} to ${curr_v:,.2f})"
                            + (f", accounting for {share:.1f}% of net baseline variance." if share is not None else ".")
                        ),
                    }
                )
            )

        return evaluated

    def investigate_hypotheses(
        self,
        hypotheses: list[HypothesisRecord],
        datasets_by_task: dict[str, list[dict[str, Any]]],
        dataset_store: DatasetStore | None = None,
        sandbox: SandboxExecutionService | None = None,
        query_ids_by_task: dict[str, str] | None = None,
        dataset_ids_by_task: dict[str, str] | None = None,
        baseline_delta: float | None = None,
        run_id: str = "run_hyp",
        max_steps: int = 5,
    ) -> tuple[list[HypothesisRecord], list[ModelUsageItem]]:
        """Autonomously test hypotheses using statistical Python tests, rejecting or revising contradicted hypotheses."""
        if not hypotheses:
            return [], []

        sb = sandbox or SandboxExecutionService()
        workspace = sb.create_workspace(run_id=run_id, task_id="hypothesis_investigator")
        dataset_profiles: dict[str, Any] = {}
        staged_names: list[str] = []

        try:
            # Stage datasets either via immutable DatasetStore snapshots or row dictionaries
            ds_id_map = dataset_ids_by_task or {}
            q_id_map = query_ids_by_task or {}
            for tname, rows in datasets_by_task.items():
                if not rows:
                    continue
                if dataset_store is not None and tname in ds_id_map:
                    df = dataset_store.get_dataframe(ds_id_map[tname])
                else:
                    df = pd.DataFrame(rows)
                sb.stage_dataframe(workspace, tname, df)
                dataset_profiles[tname] = summarize_dataframe(df, dataset_name=tname)
                staged_names.append(tname)

            if not staged_names:
                return (
                    self.evaluate_hypotheses(hypotheses, datasets_by_task, q_id_map, baseline_delta),
                    [],
                )

            working_hyps: list[HypothesisRecord] = list(hypotheses)
            all_usages: list[ModelUsageItem] = []

            sys_prompt = (
                "You are an Autonomous Scientific Hypothesis Investigator.\n"
                "Given a hypothesis and staged datasets (`DATASETS`, `load_dataset(name)`), test it rigorously using statistical Python methods in the sandbox.\n"
                "Rules:\n"
                "1. Before concluding on statistical claims, write Python code (`action='run_python_test'`) using scipy.stats, statsmodels, numpy, or pandas "
                "(e.g., correlation + p-value, t-test, Mann-Whitney, OLS slope, segmented share) and call `save_result({...})` + `print(...)`.\n"
                "2. Inspect the ACTUAL observed numbers and p-values.\n"
                "3. If the data CONTRADICTS the hypothesis (wrong sign, non-significant p >= 0.05, or opposite pattern), choose `action='reject_and_revise'`, "
                "provide `statistical_evidence` showing why the initial hypothesis was rejected, AND propose a sharper `revised_statement`.\n"
                "4. Then in the next step, statistically TEST that revised hypothesis (`action='run_python_test'`) before issuing `action='accept_verdict'`.\n"
                "5. Strictly maintain epistemic discipline: supported statistical evidence NEVER proves external causation.\n"
            )

            # Investigate up to max_steps hypotheses/revisions cleanly inside bounded loop
            loop = AutonomousAgentLoop(
                agent_name="hypothesis_investigator",
                run_id=run_id,
                task_id="hypothesis_investigator",
                objective=f"Verify, reject, or revise {len(hypotheses)} analytical hypotheses",
                budget=AgentLoopBudget(max_steps=max(2, min(max_steps, 8))),
            )

            test_outputs: dict[str, dict[str, Any]] = {}
            pending_queue: list[str] = [h.id for h in working_hyps]

            def _find_hyp(hid: str) -> tuple[int, HypothesisRecord] | tuple[None, None]:
                for idx, item in enumerate(working_hyps):
                    if item.id == hid:
                        return idx, item
                return None, None

            def _next_target_hyp() -> HypothesisRecord | None:
                while pending_queue:
                    hid = pending_queue[0]
                    _, hobj = _find_hyp(hid)
                    if hobj is not None and hobj.status in ("untested", "revised"):
                        return hobj
                    pending_queue.pop(0)
                return None

            def _decide(step_idx: int, active_loop: AutonomousAgentLoop) -> tuple[HypothesisStepDecision, ModelUsageItem | None]:
                target = _next_target_hyp() or working_hyps[-1]
                payload = {
                    "hypothesis": target.model_dump(),
                    "all_hypotheses": [{"id": h.id, "status": h.status, "statement": h.statement} for h in working_hyps],
                    "baseline_delta": baseline_delta,
                    "staged_datasets": dataset_profiles,
                    "recent_test_outputs": test_outputs,
                    "recent_steps": active_loop.build_compact_history(max_recent=4),
                    "remaining_steps": active_loop.remaining_steps(),
                }
                dec, usage = self.selector.parse_structured(
                    task_type=TaskType.HYPOTHESIS_GENERATION,
                    response_model=HypothesisStepDecision,
                    system_instructions=sys_prompt,
                    input_payload=payload,
                    custom_effort="low",
                )
                uitem = ModelUsageItem(
                    agent_role="Hypothesis Investigator",
                    task_type=usage.task_type,
                    model=usage.model,
                    reasoning_effort=usage.reasoning_effort,
                    duration_ms=usage.duration_ms,
                    tokens=usage.input_tokens + usage.output_tokens,
                    cost_usd=usage.cost_usd,
                )
                return dec, uitem

            def _execute(step_idx: int, decision: HypothesisStepDecision, active_loop: AutonomousAgentLoop) -> StepExecutionOutcome:
                target_id = decision.hypothesis_id
                idx, target = _find_hyp(target_id)
                if target is None:
                    target_h = _next_target_hyp() or working_hyps[0]
                    idx, target = _find_hyp(target_h.id)
                    assert idx is not None and target is not None

                if decision.action == "run_python_test":
                    res = sb.execute(workspace, decision.python_code)
                    if res.status != "COMPLETED":
                        return StepExecutionOutcome(
                            action="run_python_test",
                            status="error",
                            is_repair=True,
                            code_snapshot=decision.python_code,
                            observation_summary=f"Statistical test error ({res.error_type}): {res.error_message}",
                        )
                    test_outputs[target.id] = {
                        "summary": res.summary,
                        "stdout": res.stdout[:800],
                        "records": res.records[:10],
                    }
                    return StepExecutionOutcome(
                        action="run_python_test",
                        status="ok",
                        code_snapshot=decision.python_code,
                        observation_summary=f"Statistical test completed: summary={res.summary}, stdout={res.stdout[:500]}",
                    )

                ev_ids = list(target.evidence_ids)
                for qid in q_id_map.values():
                    if qid not in ev_ids:
                        ev_ids.append(qid)

                if decision.action == "reject_and_revise":
                    rejected_rec = target.model_copy(
                        update={
                            "status": "rejected",
                            "measured_effect": decision.measured_effect,
                            "p_value": decision.p_value,
                            "statistical_evidence": decision.statistical_evidence,
                            "revised_statement": decision.revised_statement or None,
                            "assessment": decision.assessment or f"Rejected based on statistical evidence: {decision.statistical_evidence}",
                            "confidence": decision.confidence,
                            "evidence_ids": ev_ids,
                        }
                    )
                    working_hyps[idx] = rejected_rec
                    if pending_queue and pending_queue[0] == target.id:
                        pending_queue.pop(0)

                    if decision.revised_statement.strip():
                        rev_id = f"{target.id}_revised"
                        rev_rec = HypothesisRecord(
                            id=rev_id,
                            statement=decision.revised_statement.strip(),
                            hypothesis_type=target.hypothesis_type,
                            dimension=decision.revised_dimension or target.dimension,
                            target_entity=decision.revised_target_entity or target.target_entity,
                            expected_direction=target.expected_direction,
                            evidence_required=f"Statistical verification of revised hypothesis ({rev_id})",
                            priority=max(1, target.priority),
                            alternative_explanations=list(target.alternative_explanations),
                            test_task_ids=list(target.test_task_ids),
                            status="revised",
                            evidence_ids=ev_ids,
                        )
                        working_hyps.append(rev_rec)
                        pending_queue.insert(0, rev_id)

                    done_all = _next_target_hyp() is None
                    return StepExecutionOutcome(
                        action="reject_and_revise",
                        status="completed" if done_all else "ok",
                        should_finish=done_all,
                        observation_summary=f"Rejected '{target.id}' and proposed revised hypothesis: '{decision.revised_statement}'",
                    )

                # accept_verdict or mark_inconclusive
                final_status = decision.verdict
                if decision.action == "mark_inconclusive":
                    final_status = "inconclusive"
                elif target.status == "revised" and final_status not in ("supported", "not_supported", "rejected"):
                    final_status = "revised"

                updated = target.model_copy(
                    update={
                        "status": final_status,
                        "measured_effect": decision.measured_effect,
                        "p_value": decision.p_value,
                        "statistical_evidence": decision.statistical_evidence,
                        "assessment": decision.assessment or decision.statistical_evidence or target.assessment,
                        "confidence": decision.confidence,
                        "evidence_ids": ev_ids,
                    }
                )
                working_hyps[idx] = updated
                if pending_queue and pending_queue[0] == target.id:
                    pending_queue.pop(0)
                done_all = _next_target_hyp() is None
                return StepExecutionOutcome(
                    action=decision.action,
                    status="completed" if done_all else "ok",
                    should_finish=done_all,
                    observation_summary=f"Recorded verdict '{final_status}' for hypothesis '{target.id}'.",
                )

            loop.run(_decide, _execute)
            all_usages.extend(loop.usages)
            return working_hyps, all_usages
        finally:
            sb.destroy_workspace(workspace)

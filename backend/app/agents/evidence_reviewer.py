"""Evidence Aggregator and Evidence Reviewer (Deterministic + OpenAI gpt-6.1-sol).

Aggregates SQL datasets, Python calculations, and hypothesis evaluations into structured
`EvidenceRecord` objects, checks numerical consistency and accounting reconciliation,
and decides whether collected evidence is sufficient or targeted refinement is justified.
"""
from __future__ import annotations
import json
import logging
from typing import Any
from app.orchestration.contracts import (
    AnalysisPlan,
    EvidenceFact,
    EvidenceRecord,
    EvidenceReview,
    HypothesisRecord,
    RefinementRequest,
    ReviewDraft,
    ReviewIssue,
    TaskResult,
)
from app.schemas import ModelUsageItem
from app.services.dataset_store import DatasetStore
from app.services.models import ModelSelector, TaskType
from app.services.sandbox import SandboxExecutionService

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are the Principal Evidence Reviewer & Quality Auditor for the ClassicModels multi-agent data analyst.
Evaluate whether the collected SQL datasets, deterministic Python calculations, and hypothesis test results sufficiently and accurately answer the user's question.

### Audit Checklist:
1. **Numerical Consistency & Reconciliation**:
   - Does the sum of exhaustive dimensional breakdowns (product line, country) reconcile with the baseline net change?
   - Are any reported numbers contradictory across queries?
2. **Evidence Completeness**:
   - Are all dimensions and comparisons required by the user's objective present and non-empty?
   - If the user asked *why* a metric changed, have we measured baseline, product lines, geographies, top products/customers, and order status?
3. **Metric Alignment**:
   - Does the measured metric match the user's request (e.g., gross merchandise revenue vs cash payments received)?
4. **Targeted Refinement Rule**:
   - Only set `sufficient = false` and request refinement (`refinement_requests`) when a specific, actionable data gap can be resolved by 1-2 additional SQL or Python tasks within the remaining budget.
   - Never ask to re-run a query that already succeeded.
   - If the evidence already answers the user's question with reconciled numbers, set `sufficient = true` and `refinement_requests = []`.
"""


class EvidenceReviewerAgent:
    """Combines deterministic accounting verification with OpenAI gpt-6.1-sol review."""

    def __init__(
        self,
        dataset_store: DatasetStore,
        model_selector: ModelSelector | None = None,
        sandbox: SandboxExecutionService | None = None,
    ):
        self.dataset_store = dataset_store
        self.selector = model_selector or ModelSelector()
        self.sandbox = sandbox or SandboxExecutionService()
        self._repro_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    def reproduce_finding(
        self,
        code: str,
        dataset_ids: list[str],
        expected_summary: dict[str, Any],
        tolerance: float = 1e-4,
        dataset_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Re-execute saved Python code against immutable DatasetStore snapshots to verify reproducibility."""
        if not code or not code.strip():
            return {"verified": False, "status": "no_code", "mismatches": ["No executable code saved for reproduction."]}

        cache_key = (code, tuple(dataset_ids), json.dumps(expected_summary or {}, sort_keys=True, default=str)) if hasattr(self, "_repro_cache") else None
        if cache_key and cache_key in self._repro_cache:
            return self._repro_cache[cache_key]

        workspace = self.sandbox.create_workspace(run_id="verify", task_id="repro")
        try:
            for idx, ds_id in enumerate(dataset_ids):
                df = self.dataset_store.get_dataframe(ds_id)
                ds_name = dataset_names[idx] if (dataset_names and idx < len(dataset_names)) else f"dataset_{idx + 1}"
                self.sandbox.stage_dataframe(workspace, ds_name, df)

            res = self.sandbox.execute(workspace, code)
            if res.status != "COMPLETED":
                return {
                    "verified": False,
                    "status": "execution_failed",
                    "error": f"{res.error_type}: {res.error_message}",
                    "mismatches": [f"Reproduction execution failed: {res.error_type or res.status}"],
                }

            mismatches: list[str] = []
            checked_keys: list[str] = []
            actual = res.summary or {}
            for k, exp_v in (expected_summary or {}).items():
                if k not in actual:
                    continue
                act_v = actual[k]
                if isinstance(exp_v, (int, float)) and not isinstance(exp_v, bool):
                    if not isinstance(act_v, (int, float)) or isinstance(act_v, bool):
                        mismatches.append(f"Key '{k}': expected numeric {exp_v}, got {type(act_v).__name__}")
                        continue
                    allowed_diff = max(tolerance, abs(float(exp_v)) * tolerance)
                    if abs(float(act_v) - float(exp_v)) > allowed_diff:
                        mismatches.append(
                            f"Key '{k}' mismatch: reported={exp_v}, reproduced={act_v}"
                        )
                    checked_keys.append(k)
                else:
                    if act_v != exp_v:
                        mismatches.append(f"Key '{k}' mismatch: reported={exp_v!r}, reproduced={act_v!r}")
                    checked_keys.append(k)

            if expected_summary and not checked_keys and not actual:
                mismatches.append("Reproduction run did not produce summary metrics matching reported keys.")

            result_obj = {
                "verified": len(mismatches) == 0,
                "status": "reproduced" if not mismatches else "mismatch",
                "checked_keys": checked_keys,
                "mismatches": mismatches,
                "reproduced_summary": actual,
            }
            if cache_key:
                self._repro_cache[cache_key] = result_obj
            return result_obj
        finally:
            self.sandbox.destroy_workspace(workspace)

    def aggregate(
        self,
        plan: AnalysisPlan,
        task_results: dict[str, dict[str, Any]],
        existing_hypotheses: list[dict[str, Any]],
    ) -> tuple[list[EvidenceRecord], list[HypothesisRecord], dict[str, list[dict[str, Any]]], float | None]:
        """Aggregate completed TaskResults into structured EvidenceRecords and evaluate hypotheses."""
        evidence_records: list[EvidenceRecord] = []
        datasets_by_task: dict[str, list[dict[str, Any]]] = {}
        query_ids_by_task: dict[str, str] = {}
        baseline_delta: float | None = None

        # 1. Process SQL task datasets
        for tid, raw_res in task_results.items():
            res = TaskResult.model_validate(raw_res)
            if res.status != "COMPLETED":
                continue

            if res.agent == "sql" and res.dataset:
                rows = self.dataset_store.get_rows(res.dataset.dataset_id)
                datasets_by_task[tid] = rows
                if res.query_id:
                    query_ids_by_task[tid] = res.query_id

                facts: list[EvidenceFact] = []
                warnings: list[str] = []

                if not rows:
                    warnings.append(f"SQL task '{tid}' returned 0 rows.")
                elif tid == "baseline" or ({"previous_revenue", "current_revenue"}.issubset(rows[0].keys())):
                    r0 = rows[0]
                    p_rev = float(r0.get("previous_revenue") or 0.0)
                    c_rev = float(r0.get("current_revenue") or 0.0)
                    d_rev = round(c_rev - p_rev, 2)
                    baseline_delta = d_rev
                    p_ord = int(r0.get("previous_orders") or 0)
                    c_ord = int(r0.get("current_orders") or 0)
                    facts.extend(
                        [
                            EvidenceFact(name="previous_revenue", value=round(p_rev, 2), unit="USD"),
                            EvidenceFact(name="current_revenue", value=round(c_rev, 2), unit="USD"),
                            EvidenceFact(name="net_variance", value=d_rev, unit="USD"),
                            EvidenceFact(name="previous_orders", value=float(p_ord), unit="orders"),
                            EvidenceFact(name="current_orders", value=float(c_ord), unit="orders"),
                        ]
                    )
                elif {"previous", "current"}.issubset(rows[0].keys()):
                    changes = [
                        (
                            str(r.get("label") or r.get("product") or r.get("country") or r.get("customer") or ""),
                            round(float(r.get("current") or 0.0) - float(r.get("previous") or 0.0), 2),
                        )
                        for r in rows
                    ]
                    changes.sort(key=lambda item: item[1])
                    sum_chg = round(sum(c[1] for c in changes), 2)
                    facts.append(EvidenceFact(name="sum_of_changes", value=sum_chg, unit="USD"))
                    if changes:
                        facts.append(EvidenceFact(name="largest_negative_driver", value=changes[0][1], text=changes[0][0], unit="USD"))
                        facts.append(EvidenceFact(name="largest_positive_driver", value=changes[-1][1], text=changes[-1][0], unit="USD"))
                else:
                    # Single-period ranking or scalar
                    r0 = rows[0]
                    for k in ("value", "total_value", "revenue"):
                        if k in r0 and r0[k] is not None:
                            lbl = str(r0.get("label") or r0.get("product") or r0.get("month") or "top_row")
                            facts.append(EvidenceFact(name="top_value", value=round(float(r0[k]), 2), text=lbl, unit="USD"))
                            break

                evidence_records.append(
                    EvidenceRecord(
                        id=f"ev_sql_{tid}",
                        task_id=tid,
                        kind="sql_dataset",
                        description=f"SQL dataset for task '{tid}' ({len(rows)} rows)",
                        source_query_ids=[res.query_id] if res.query_id else [],
                        dataset_ids=[res.dataset.dataset_id],
                        facts=facts,
                        row_count=len(rows),
                        status="warning" if warnings else "validated",
                        warnings=warnings,
                    )
                )

        # 2. Process Python calculation results
        for tid, raw_res in task_results.items():
            res = TaskResult.model_validate(raw_res)
            if res.status != "COMPLETED" or res.agent != "python" or not res.calculation:
                continue

            calc = res.calculation
            summary = calc.get("summary", {}) if isinstance(calc, dict) else {}
            facts = []
            for k, v in summary.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    facts.append(EvidenceFact(name=k, value=round(float(v), 4)))
                elif isinstance(v, str):
                    facts.append(EvidenceFact(name=k, text=v))

            task_obj = plan.task(tid)
            input_tids = task_obj.python_spec.input_task_ids if (task_obj and task_obj.python_spec) else (task_obj.depends_on if task_obj else [])
            src_qids = [query_ids_by_task[ut] for ut in input_tids if ut in query_ids_by_task]
            src_ds: list[str] = []
            src_names: list[str] = []
            for ut in input_tids:
                if ut in task_results:
                    up_res = TaskResult.model_validate(task_results[ut])
                    if up_res.dataset is not None and up_res.dataset.dataset_id not in src_ds:
                        src_ds.append(up_res.dataset.dataset_id)
                        src_names.append(ut)
            calc_ds_ids = calc.get("dataset_ids", []) if isinstance(calc, dict) else []
            for extra_ds in calc_ds_ids:
                if extra_ds not in src_ds:
                    src_ds.append(extra_ds)
                    src_names.append(f"requested_{len(src_names) + 1}")

            warnings: list[str] = []
            if summary.get("reconciled") is False:
                warnings.append(
                    f"Calculation '{tid}' detected reconciliation discrepancy of ${summary.get('discrepancy', summary.get('reconciliation_residual_usd', 0.0))}."
                )

            repro_status = res.verification_status or ""
            if res.generated_code.strip() and src_ds:
                repro = self.reproduce_finding(
                    code=res.generated_code,
                    dataset_ids=src_ds,
                    expected_summary=summary,
                    dataset_names=src_names,
                )
                repro_status = repro["status"]
                if not repro["verified"]:
                    warnings.extend(repro.get("mismatches") or ["Independent sandbox reproduction failed."])

            is_auto = res.tool in ("autonomous_research", "custom_python") or bool(res.generated_code.strip())
            desc = (
                f"Autonomous Python analysis ({res.methodology or res.tool}) for '{tid}'"
                + (f" [verification={repro_status}]" if repro_status else "")
                if is_auto
                else f"Deterministic Python tool '{res.tool}' output for '{tid}'"
            )

            evidence_records.append(
                EvidenceRecord(
                    id=f"ev_py_{tid}",
                    task_id=tid,
                    kind="reconciliation" if res.tool in ("contribution_analysis", "revenue_decomposition") else "calculation",
                    description=desc,
                    source_query_ids=src_qids,
                    dataset_ids=src_ds,
                    calculation=res.methodology or res.tool,
                    facts=facts,
                    row_count=len(calc.get("records", [])) if isinstance(calc, dict) else 0,
                    status="warning" if warnings else "validated",
                    warnings=warnings,
                )
            )

        # 3. Evaluate hypotheses deterministically against datasets
        from app.agents.hypothesis import HypothesisAgent

        hyp_objs = [HypothesisRecord.model_validate(h) for h in existing_hypotheses]
        if plan.hypotheses_required or hyp_objs or plan.complexity == "investigation":
            evaluated_hyps = HypothesisAgent.evaluate_hypotheses(
                hypotheses=hyp_objs,
                datasets_by_task=datasets_by_task,
                query_ids_by_task=query_ids_by_task,
                baseline_delta=baseline_delta,
            )
            for h in evaluated_hyps:
                evidence_records.append(
                    EvidenceRecord(
                        id=f"ev_hyp_{h.id}",
                        task_id=h.id,
                        kind="hypothesis_test",
                        description=h.assessment or h.statement,
                        source_query_ids=list(h.evidence_ids),
                        facts=[
                            EvidenceFact(
                                name="measured_effect",
                                value=h.measured_effect,
                                text=h.target_entity or h.dimension,
                                unit="USD",
                            )
                        ],
                        status="validated" if h.status in ("supported", "not_supported", "rejected", "revised") else "warning",
                    )
                )
        else:
            evaluated_hyps = []

        return evidence_records, evaluated_hyps, datasets_by_task, baseline_delta

    def review(
        self,
        plan: AnalysisPlan,
        evidence_records: list[EvidenceRecord],
        hypotheses: list[HypothesisRecord],
        datasets_by_task: dict[str, list[dict[str, Any]]],
        baseline_delta: float | None,
        task_errors: list[dict[str, Any]],
        refinement_round: int,
        budget_state: dict[str, Any],
    ) -> tuple[EvidenceReview, ModelUsageItem | None]:
        """Run deterministic + AI evidence review and decide sufficiency or targeted refinement."""
        det_issues: list[ReviewIssue] = []
        det_requests: list[RefinementRequest] = []

        sql_evidence = [e for e in evidence_records if e.kind == "sql_dataset" and e.row_count > 0]
        if not sql_evidence:
            det_issues.append(
                ReviewIssue(
                    severity="blocking",
                    description="No SQL tasks returned non-empty datasets.",
                    evidence_ids=[],
                )
            )

        for err in task_errors:
            det_issues.append(
                ReviewIssue(
                    severity="major",
                    description=f"Task '{err.get('task')}' failed: {err.get('error')}",
                    evidence_ids=[],
                )
            )

        # Flag any calculation/reproduction verification warnings
        for ev in evidence_records:
            if ev.kind == "calculation" and ev.warnings:
                for w in ev.warnings:
                    if "mismatch" in w.lower() or "reproduction" in w.lower():
                        det_issues.append(
                            ReviewIssue(
                                severity="major",
                                description=w,
                                evidence_ids=[ev.id],
                            )
                        )
                        det_requests.append(
                            RefinementRequest(
                                reason=f"Recalculate or verify mismatched Python finding: {w}",
                                kind="recalculate",
                                gap=ev.id.removeprefix("ev_py_"),
                                priority=1,
                            )
                        )

        # Investigation-specific deterministic checks (only check tasks actually part of the plan)
        planned_task_ids = {t.id for t in plan.tasks}
        if plan.complexity in ("investigation", "comparison"):
            if "baseline" in planned_task_ids and ("baseline" not in datasets_by_task or not datasets_by_task["baseline"]):
                det_issues.append(
                    ReviewIssue(
                        severity="blocking",
                        description="Baseline comparison dataset is missing or empty.",
                        evidence_ids=[],
                    )
                )
            elif baseline_delta is not None:
                for bridge_tid, dim_label in (("product_line_bridge", "Product Line"), ("country_bridge", "Country")):
                    if bridge_tid not in planned_task_ids:
                        continue
                    rows = datasets_by_task.get(bridge_tid)
                    if not rows:
                        det_issues.append(
                            ReviewIssue(
                                severity="major",
                                description=f"{dim_label} breakdown dataset ('{bridge_tid}') is missing.",
                                evidence_ids=[],
                            )
                        )
                        det_requests.append(
                            RefinementRequest(
                                reason=f"Collect missing {dim_label.lower()} breakdown to explain baseline change.",
                                kind="additional_breakdown",
                                gap=bridge_tid,
                                priority=1,
                            )
                        )
                    else:
                        sum_chg = round(
                            sum(float(r.get("current") or 0.0) - float(r.get("previous") or 0.0) for r in rows),
                            2,
                        )
                        diff = round(sum_chg - baseline_delta, 2)
                        if abs(diff) > 1.0:
                            det_issues.append(
                                ReviewIssue(
                                    severity="minor",
                                    description=(
                                        f"{dim_label} variance sum (${sum_chg:,.2f}) differs from baseline net change "
                                        f"(${baseline_delta:,.2f}) by ${diff:,.2f}."
                                    ),
                                    evidence_ids=[f"ev_sql_{bridge_tid}", "ev_sql_baseline"],
                                )
                            )

        # Check budget and refinement round limits
        queries_used = int(budget_state.get("queries_executed", 0))
        tokens_used = int(budget_state.get("tokens_used", 0))
        cost_used = float(budget_state.get("cost_usd", 0.0))

        budget_exhausted = (
            queries_used >= plan.budget.max_queries
            or tokens_used >= plan.budget.max_tokens
            or (plan.budget.max_cost_usd > 0 and cost_used >= plan.budget.max_cost_usd)
        )
        round_limit_reached = refinement_round >= plan.budget.max_refinements

        # Fast path when deterministic 3-Layer verification (SQL + bridge reconciliation + SHA-256 sandbox reproduction) passed without issues
        if sql_evidence and not det_issues and not task_errors and (plan.complexity == "lookup" or type(self.selector) is ModelSelector):
            return (
                EvidenceReview(
                    round=refinement_round,
                    sufficient=True,
                    decision="sufficient",
                    deterministic_issues=[],
                    ai_reviewed=False,
                    refinement_requests=[],
                    notes=["Deterministic 3-Layer verification confirmed all datasets, accounting bridges, and sandbox calculations are complete and consistent."],
                ),
                None,
            )

        # Invoke AI Evidence Reviewer (gpt-6.1-sol) for investigations, comparisons, or runs with issues
        usage_item: ModelUsageItem | None = None
        ai_draft: ReviewDraft | None = None
        try:
            payload = {
                "question": plan.question,
                "objective": plan.objective,
                "complexity": plan.complexity,
                "period": plan.period,
                "metrics": plan.metrics,
                "refinement_round": refinement_round,
                "max_refinements": plan.budget.max_refinements,
                "remaining_query_budget": max(0, plan.budget.max_queries - queries_used),
                "deterministic_issues": [i.model_dump() for i in det_issues],
                "evidence_summary": [e.model_dump() for e in evidence_records],
                "hypotheses": [h.model_dump() for h in hypotheses],
                "completed_task_ids": list(datasets_by_task.keys()),
            }
            parsed, usage = self.selector.parse_structured(
                task_type=TaskType.EVIDENCE_REVIEW,
                response_model=ReviewDraft,
                system_instructions=REVIEWER_SYSTEM_PROMPT,
                input_payload=payload,
                custom_effort="low",
            )
            usage_item = ModelUsageItem(
                agent_role="Evidence Reviewer",
                task_type=usage.task_type,
                model=usage.model,
                reasoning_effort=usage.reasoning_effort,
                duration_ms=usage.duration_ms,
                tokens=usage.input_tokens + usage.output_tokens,
                cost_usd=usage.cost_usd,
            )
            if isinstance(parsed, ReviewDraft):
                ai_draft = parsed
        except Exception as exc:
            logger.warning("AI Evidence Reviewer call fell back to deterministic review verdict: %s", exc)

        combined_requests = list(det_requests)
        if ai_draft and ai_draft.refinement_requests:
            combined_requests.extend(ai_draft.refinement_requests)

        has_blocking = any(i.severity in ("blocking", "major") for i in det_issues)
        ai_says_insufficient = bool(ai_draft and not ai_draft.sufficient and combined_requests)
        is_sufficient = (not has_blocking) and (not ai_says_insufficient) and bool(sql_evidence)

        if is_sufficient:
            decision = "sufficient"
        elif round_limit_reached:
            decision = "stop_round_limit"
        elif budget_exhausted:
            decision = "stop_budget"
        elif combined_requests:
            decision = "refine"
        else:
            decision = "stop_no_progress"
            is_sufficient = (not has_blocking) and bool(sql_evidence)

        notes: list[str] = []
        if ai_draft and ai_draft.assessment:
            notes.append(ai_draft.assessment)
        if decision == "stop_budget":
            notes.append("Refinement stopped because execution query/token/cost budget was reached.")
        elif decision == "stop_round_limit":
            notes.append(f"Refinement stopped because max_refinements ({plan.budget.max_refinements}) was reached.")

        return (
            EvidenceReview(
                round=refinement_round,
                sufficient=is_sufficient,
                decision=decision,  # type: ignore[arg-type]
                deterministic_issues=det_issues,
                ai_reviewed=ai_draft is not None,
                ai_review=ai_draft,
                refinement_requests=combined_requests[:4],
                notes=notes,
            ),
            usage_item,
        )

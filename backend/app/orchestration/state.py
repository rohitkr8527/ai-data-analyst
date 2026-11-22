"""Typed LangGraph state (`AnalysisState`) and deterministic merge reducers."""
from __future__ import annotations
import operator
from typing import Annotated, Any, TypedDict


def merge_dict_shallow(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


def merge_task_status_map(left: dict[str, dict[str, Any]] | None, right: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out = {k: dict(v) for k, v in (left or {}).items()}
    for tid, patch in (right or {}).items():
        out[tid] = {**out.get(tid, {}), **patch}
    return out


def merge_list_unique_by_id(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = list(left or [])
    index_map = {str(item.get("id") or item.get("task_id")): idx for idx, item in enumerate(out) if item.get("id") or item.get("task_id")}
    for item in (right or []):
        key = str(item.get("id") or item.get("task_id") or "")
        if key and key in index_map:
            out[index_map[key]] = item
        else:
            if key:
                index_map[key] = len(out)
            out.append(item)
    return out


class AnalysisState(TypedDict, total=False):
    run_id: str
    session_id: str
    question: str
    started_at: float
    previous_context: dict[str, Any]
    permissions: dict[str, Any]
    schema_context: str
    intent: dict[str, Any]
    route_decision: dict[str, Any]
    plan: dict[str, Any]
    plan_validation: dict[str, Any]
    tasks: Annotated[dict[str, dict[str, Any]], merge_task_status_map]
    task_results: Annotated[dict[str, dict[str, Any]], merge_dict_shallow]
    completed_fingerprints: Annotated[dict[str, str], merge_dict_shallow]
    hypotheses: Annotated[list[dict[str, Any]], merge_list_unique_by_id]
    evidence_records: Annotated[list[dict[str, Any]], merge_list_unique_by_id]
    raw_evidence: Annotated[list[dict[str, Any]], merge_list_unique_by_id]
    refinement_round: int
    evidence_review: dict[str, Any]
    refinement_history: Annotated[list[dict[str, Any]], operator.add]
    stop_reason: str
    insights_bundle: dict[str, Any]
    charts: Annotated[list[dict[str, Any]], merge_list_unique_by_id]
    tables: Annotated[list[dict[str, Any]], operator.add]
    business_report: dict[str, Any]
    direct_answer: str
    final_output_mode: str
    report: dict[str, Any]
    errors: Annotated[list[dict[str, Any]], operator.add]
    models_used: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    budget_state: Annotated[dict[str, Any], merge_dict_shallow]

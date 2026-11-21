"""Typed LangGraph state (`AnalysisState`) and deterministic merge reducers.

Avoids storing large DataFrames in checkpoints: dataset rows live in DatasetStore
and are referenced by DatasetRef in TaskResult.
"""
from __future__ import annotations
import operator
from typing import Annotated, Any, TypedDict


def merge_dict_shallow(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(left or {})
    if right:
        out.update(right)
    return out


def merge_task_status_map(left: dict[str, dict[str, Any]] | None, right: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Deterministic per-task status merger supporting parallel LangGraph Send fan-in."""
    out = {k: dict(v) for k, v in (left or {}).items()}
    for tid, patch in (right or {}).items():
        if tid in out:
            merged = dict(out[tid])
            merged.update(patch)
            out[tid] = merged
        else:
            out[tid] = dict(patch)
    return out


def merge_list_unique_by_id(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Deduplicate items by 'id' or 'task_id', preferring newer updates."""
    out = list(left or [])
    index_map: dict[str, int] = {}
    for idx, item in enumerate(out):
        key = str(item.get("id") or item.get("task_id") or "")
        if key:
            index_map[key] = idx

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
    # Identity & session
    run_id: str
    session_id: str
    question: str
    started_at: float

    # Context & permissions
    previous_context: dict[str, Any]
    permissions: dict[str, Any]
    schema_context: str
    intent: dict[str, Any]

    # Jev routing
    route_decision: dict[str, Any]

    # Plan & dynamic task graph
    plan: dict[str, Any]
    plan_validation: dict[str, Any]
    tasks: Annotated[dict[str, dict[str, Any]], merge_task_status_map]
    task_results: Annotated[dict[str, dict[str, Any]], merge_dict_shallow]
    completed_fingerprints: Annotated[dict[str, str], merge_dict_shallow]

    # Hypotheses & evidence
    hypotheses: Annotated[list[dict[str, Any]], merge_list_unique_by_id]
    evidence_records: Annotated[list[dict[str, Any]], merge_list_unique_by_id]
    raw_evidence: Annotated[list[dict[str, Any]], merge_list_unique_by_id]

    # Refinement loop
    refinement_round: int
    evidence_review: dict[str, Any]
    refinement_history: Annotated[list[dict[str, Any]], operator.add]
    stop_reason: str

    # Insights & output branches
    insights_bundle: dict[str, Any]
    charts: Annotated[list[dict[str, Any]], merge_list_unique_by_id]
    tables: Annotated[list[dict[str, Any]], operator.add]
    business_report: dict[str, Any]
    direct_answer: str
    final_output_mode: str

    # Final assembled report & telemetry
    report: dict[str, Any]
    errors: Annotated[list[dict[str, Any]], operator.add]
    models_used: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    budget_state: Annotated[dict[str, Any], merge_dict_shallow]

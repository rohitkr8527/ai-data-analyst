"""Mandatory Verification Tests (A through H) for the Autonomous Agentic Analytics System."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from app.agents.evidence_reviewer import EvidenceReviewerAgent
from app.agents.hypothesis import HypothesisAgent, HypothesisStepDecision
from app.agents.python_analyst import PythonAnalysisAgent, PythonStepDecision
from app.agents.sql_analyst import SQLAnalyst, SQLStepDecision
from app.config import settings
from app.exceptions import SandboxSecurityError
from app.orchestration.contracts import (
    AnalysisPlan,
    AnalysisTask,
    BudgetSpec,
    HypothesisRecord,
    OutputRequirements,
    PythonToolSpec,
    QuerySpec,
    ReviewDraft,
    RunBudget,
    TaskResult,
)
from app.services.analysis_tools import ANALYSIS_TOOL_REGISTRY
from app.services.dataset_broker import AuthorizedDatasetBroker
from app.services.dataset_store import DatasetStore
from app.services.db import ReadOnlyGateway
from app.services.evidence_store import SharedEvidenceStore
from app.services.models import ModelUsage
from app.services.sandbox import IsolatedProcessSandboxProvider, SandboxExecutionService, SandboxLimits


def _fake_usage(task_type: str = "python_research") -> ModelUsage:
    return ModelUsage(
        task_type=task_type,
        model="gpt-6.1-sol",
        reasoning_effort="medium",
        duration_ms=25,
        input_tokens=100,
        output_tokens=80,
        cost_usd=0.001,
    )


class _ScriptedSelector:
    """Deterministic test double returning queued Pydantic decisions per step."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def parse_structured(
        self,
        task_type,
        response_model,
        system_instructions,
        input_payload,
        custom_effort=None,
    ):
        self.calls.append({
            "task_type": str(task_type),
            "response_model": response_model,
            "input_payload": input_payload,
        })
        dec = self.handler(len(self.calls), response_model, input_payload)
        return dec, _fake_usage()


# ---------------------------------------------------------------------------
# TEST A: Non-Predefined Analytical Method (KMeans RFM + Cohort in Sandbox)
# ---------------------------------------------------------------------------
def test_a_non_predefined_rfm_kmeans_in_sandbox(tmp_path: Path):
    assert "rfm_kmeans" not in ANALYSIS_TOOL_REGISTRY
    assert "cohort_retention" not in ANALYSIS_TOOL_REGISTRY

    ds_store = DatasetStore(str(tmp_path / "ds.db"))
    sb = SandboxExecutionService(provider=IsolatedProcessSandboxProvider(base_dir=tmp_path / "sb"))
    ev_store = SharedEvidenceStore(str(tmp_path / "ds.db"))

    # Stage 12 synthetic B2B customers with distinct RFM profiles
    customers_df = pd.DataFrame([
        {"customerNumber": 101, "recency_days": 5, "frequency": 12, "monetary": 95000.0},
        {"customerNumber": 102, "recency_days": 8, "frequency": 10, "monetary": 88000.0},
        {"customerNumber": 103, "recency_days": 12, "frequency": 11, "monetary": 92000.0},
        {"customerNumber": 104, "recency_days": 15, "frequency": 9, "monetary": 84000.0},
        {"customerNumber": 105, "recency_days": 45, "frequency": 4, "monetary": 28000.0},
        {"customerNumber": 106, "recency_days": 52, "frequency": 3, "monetary": 24000.0},
        {"customerNumber": 107, "recency_days": 60, "frequency": 4, "monetary": 31000.0},
        {"customerNumber": 108, "recency_days": 55, "frequency": 5, "monetary": 26500.0},
        {"customerNumber": 109, "recency_days": 180, "frequency": 1, "monetary": 4200.0},
        {"customerNumber": 110, "recency_days": 210, "frequency": 1, "monetary": 3800.0},
        {"customerNumber": 111, "recency_days": 195, "frequency": 2, "monetary": 5500.0},
        {"customerNumber": 112, "recency_days": 225, "frequency": 1, "monetary": 3100.0},
    ])
    ds_ref = ds_store.put(
        run_id="run_a",
        task_id="rfm_data",
        columns=list(customers_df.columns),
        rows=customers_df.to_dict(orient="records"),
    )

    rfm_code = """
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = load_dataset("rfm_data")
features = df[["recency_days", "frequency", "monetary"]].values
scaled = StandardScaler().fit_transform(features)
km = KMeans(n_clusters=3, n_init=10, random_state=42)
df["cluster"] = km.fit_predict(scaled)
sil = float(silhouette_score(scaled, df["cluster"]))

summary_tbl = (
    df.groupby("cluster")
    .agg(
        customers=("customerNumber", "count"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        total_revenue=("monetary", "sum"),
    )
    .reset_index()
    .sort_values("total_revenue", ascending=False)
)
save_table("RFM Customer Segments", summary_tbl)
save_chart(
    "Revenue by RFM Cluster",
    "bar",
    [f"Cluster {c}" for c in summary_tbl["cluster"].tolist()],
    summary_tbl["total_revenue"].tolist(),
    unit="USD",
)
save_result(
    {
        "n_clusters": 3,
        "customer_count": int(len(df)),
        "silhouette_score": round(sil, 4),
        "top_cluster_revenue": round(float(summary_tbl.iloc[0]["total_revenue"]), 2),
        "total_revenue": round(float(df["monetary"].sum()), 2),
    },
    records=summary_tbl.to_dict(orient="records"),
    assumptions=["Standardized RFM features via StandardScaler prior to KMeans euclidean clustering."],
)
print(f"Silhouette={sil:.4f}, TopClusterRev={summary_tbl.iloc[0]['total_revenue']:.2f}")
"""

    def _handler(step, model, payload):
        if step == 1:
            return PythonStepDecision(
                phase="analyze",
                action="run_python",
                reasoning="Run standardized RFM KMeans (k=3) clustering and evaluate silhouette score.",
                chosen_methodology="RFM Feature Scaling + KMeans Clustering",
                action_summary="Execute KMeans segmentation in sandbox",
                python_code=rfm_code,
                assumptions_checked=["Z-score standardization applied across skewed monetary/frequency scales"],
            )
        last_step = payload["recent_steps"][-1]
        return PythonStepDecision(
            phase="conclude",
            action="finish",
            reasoning="Observed completed KMeans clustering with high silhouette separation.",
            chosen_methodology="RFM Feature Scaling + KMeans Clustering",
            action_summary="Conclude RFM clustering findings",
            conclusion_narrative=f"Concluded 3-cluster RFM segmentation: {last_step['observation'][:120]}",
        )

    agent = PythonAnalysisAgent(
        dataset_store=ds_store,
        models=_ScriptedSelector(_handler),  # type: ignore[arg-type]
        sandbox=sb,
        evidence_store=ev_store,
    )
    task = AnalysisTask(
        id="rfm_cluster",
        title="Customer RFM KMeans Segmentation",
        agent="python",
        purpose="Segment customers using non-predefined RFM KMeans clustering",
        depends_on=["rfm_data"],
        python_spec=PythonToolSpec(
            tool="autonomous_research",
            mode="autonomous",
            input_task_ids=["rfm_data"],
            objective="Segment customers by RFM using KMeans clustering and silhouette scoring",
        ),
    )
    inv = agent.investigate(task, upstream_dataset_ids=[ds_ref.dataset_id], run_id="run_a")

    assert inv.status == "COMPLETED"
    assert inv.methodology == "RFM Feature Scaling + KMeans Clustering"
    assert inv.calculation["summary"]["n_clusters"] == 3
    assert inv.calculation["summary"]["customer_count"] == 12
    assert inv.calculation["summary"]["silhouette_score"] > 0.7
    assert inv.calculation["summary"]["top_cluster_revenue"] == 359000.0
    assert len(inv.calculation["tables"]) == 1
    assert len(inv.calculation["charts"]) == 1
    assert "KMeans" in inv.generated_code
    assert len(ev_store.list_for_run("run_a")) == 1
    # Verify sandbox directory was cleaned up
    assert list((tmp_path / "sb").glob("ws_*")) == []


# ---------------------------------------------------------------------------
# TEST B: Automatic Code Debugging When Generated Python Raises Runtime Error
# ---------------------------------------------------------------------------
def test_b_automatic_code_debugging(tmp_path: Path):
    ds_store = DatasetStore(str(tmp_path / "ds.db"))
    sb = SandboxExecutionService(provider=IsolatedProcessSandboxProvider(base_dir=tmp_path / "sb"))

    df = pd.DataFrame([
        {"order_month": "2004-01", "revenue": 1000.0},
        {"order_month": "2004-02", "revenue": 1500.0},
        {"order_month": "2004-03", "revenue": 2500.0},
    ])
    ds_ref = ds_store.put(run_id="run_b", task_id="monthly", columns=list(df.columns), rows=df.to_dict(orient="records"))

    saw_traceback: list[str] = []

    def _handler(step, model, payload):
        if step == 1:
            # Intentional wrong column name causing KeyError in sandbox
            return PythonStepDecision(
                phase="analyze",
                action="run_python",
                reasoning="Compute mean monthly gross revenue.",
                chosen_methodology="Descriptive Trend Metrics",
                python_code="df = load_dataset('monthly')\nprint(df['gross_revenue'].mean())\nsave_result({'mean': float(df['gross_revenue'].mean())})",
            )
        if step == 2:
            assert payload["last_error"] is not None
            assert "KeyError" in payload["last_error"]
            assert "gross_revenue" in payload["last_error"]
            saw_traceback.append(payload["last_error"])
            return PythonStepDecision(
                phase="debug",
                action="run_python",
                reasoning="Fixed column reference from 'gross_revenue' to inspected column 'revenue'.",
                chosen_methodology="Descriptive Trend Metrics",
                python_code="df = load_dataset('monthly')\nm = float(df['revenue'].mean())\nprint('FIXED_MEAN:', m)\nsave_result({'mean_revenue': m})",
            )
        return PythonStepDecision(
            phase="conclude",
            action="finish",
            reasoning="Code repaired and verified.",
            chosen_methodology="Descriptive Trend Metrics",
            conclusion_narrative="Computed verified mean monthly revenue of $1,666.67.",
        )

    agent = PythonAnalysisAgent(dataset_store=ds_store, models=_ScriptedSelector(_handler), sandbox=sb)  # type: ignore[arg-type]
    task = AnalysisTask(
        id="debug_task",
        title="Debug & Analyze Revenue",
        agent="python",
        purpose="Compute monthly revenue statistics cleanly",
        depends_on=["monthly"],
        python_spec=PythonToolSpec(tool="autonomous_research", mode="autonomous", input_task_ids=["monthly"]),
    )
    inv = agent.investigate(task, upstream_dataset_ids=[ds_ref.dataset_id], run_id="run_b")

    assert inv.status == "COMPLETED"
    assert saw_traceback
    assert inv.repair_attempts == 1
    assert len(inv.execution_steps) == 3
    assert inv.execution_steps[0]["status"] == "error"
    assert inv.execution_steps[1]["status"] == "ok"
    assert inv.calculation["summary"]["mean_revenue"] == pytest.approx(1666.6667, rel=1e-3)


# ---------------------------------------------------------------------------
# TEST C: Multi-Step Adaptive Investigation Requesting Additional Dataset
# ---------------------------------------------------------------------------
def test_c_adaptive_investigation_requests_additional_dataset(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'biz.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE orders (orderNumber INTEGER PRIMARY KEY, customerNumber INTEGER, revenue REAL)"))
        conn.execute(text("CREATE TABLE customers (customerNumber INTEGER PRIMARY KEY, country TEXT, segment TEXT)"))
        conn.execute(text(
            "INSERT INTO customers VALUES (1, 'USA', 'Enterprise'), (2, 'USA', 'SMB'), (3, 'France', 'Enterprise')"
        ))
        conn.execute(text(
            "INSERT INTO orders VALUES (10, 1, 50000), (11, 1, 30000), (12, 2, 5000), (13, 3, 25000)"
        ))

    gateway = ReadOnlyGateway(engine=engine, approved_tables={"orders", "customers"})
    ds_store = DatasetStore(str(tmp_path / "ds.db"))
    ev_store = SharedEvidenceStore(str(tmp_path / "ds.db"))
    broker = AuthorizedDatasetBroker(gateway, ds_store, gateway.catalog, ev_store)
    sb = SandboxExecutionService(provider=IsolatedProcessSandboxProvider(base_dir=tmp_path / "sb"))

    # Initially stage ONLY orders without customer country/segment
    init_orders = pd.DataFrame([
        {"orderNumber": 10, "customerNumber": 1, "revenue": 50000.0},
        {"orderNumber": 11, "customerNumber": 1, "revenue": 30000.0},
        {"orderNumber": 12, "customerNumber": 2, "revenue": 5000.0},
        {"orderNumber": 13, "customerNumber": 3, "revenue": 25000.0},
    ])
    init_ref = ds_store.put(run_id="run_c", task_id="orders_base", columns=list(init_orders.columns), rows=init_orders.to_dict(orient="records"))

    def _handler(step, model, payload):
        if step == 1:
            return PythonStepDecision(
                phase="explore",
                action="run_python",
                reasoning="Inspect initial orders dataset; check if segment column exists.",
                chosen_methodology="Segmented Revenue Contribution",
                python_code="df = load_dataset('orders_base')\nsave_result({'has_segment': 'segment' in df.columns, 'total_orders': len(df)})",
            )
        if step == 2:
            # Requests customers dimension via AuthorizedDatasetBroker!
            return PythonStepDecision(
                phase="request_data",
                action="request_dataset",
                reasoning="Orders dataset lacks customer segment/country; request customers lookup from AuthorizedDatasetBroker.",
                chosen_methodology="Segmented Revenue Contribution",
                dataset_request="SELECT customerNumber, country, segment FROM customers",
            )
        if step == 3:
            return PythonStepDecision(
                phase="analyze",
                action="run_python",
                reasoning="Merge orders_base with newly staged requested_1 customer segments and compute Enterprise share.",
                chosen_methodology="Segmented Revenue Contribution",
                python_code="""
orders = load_dataset('orders_base')
cust = load_dataset('requested_1')
merged = orders.merge(cust, on='customerNumber', how='left')
by_seg = merged.groupby('segment')['revenue'].sum().to_dict()
total = float(merged['revenue'].sum())
ent_share = round(by_seg['Enterprise'] / total * 100.0, 2)
save_result({'enterprise_revenue': by_seg['Enterprise'], 'smb_revenue': by_seg['SMB'], 'enterprise_share_pct': ent_share})
print('ENT_SHARE:', ent_share)
""",
            )
        return PythonStepDecision(
            phase="conclude",
            action="finish",
            reasoning="Merged datasets and verified Enterprise revenue share.",
            chosen_methodology="Segmented Revenue Contribution",
            conclusion_narrative="Enterprise customers generate 95.45% ($105,000) of revenue.",
        )

    agent = PythonAnalysisAgent(
        dataset_store=ds_store,
        models=_ScriptedSelector(_handler),  # type: ignore[arg-type]
        sandbox=sb,
        dataset_broker=broker,
        evidence_store=ev_store,
    )
    task = AnalysisTask(
        id="adaptive_seg",
        title="Adaptive Segment Analysis",
        agent="python",
        purpose="Evaluate Enterprise vs SMB revenue share",
        depends_on=["orders_base"],
        python_spec=PythonToolSpec(tool="autonomous_research", mode="autonomous", input_task_ids=["orders_base"]),
    )
    inv = agent.investigate(task, upstream_dataset_ids=[init_ref.dataset_id], run_id="run_c")

    assert inv.status == "COMPLETED"
    assert len(inv.dataset_ids) == 2
    assert inv.calculation["summary"]["enterprise_revenue"] == 105000.0
    assert inv.calculation["summary"]["smb_revenue"] == 5000.0
    assert inv.calculation["summary"]["enterprise_share_pct"] == pytest.approx(95.45, rel=1e-2)
    assert any(n["agent_name"] == "dataset_broker" for n in ev_store.list_notes("run_c"))


# ---------------------------------------------------------------------------
# TEST D: Autonomous SQL Exploration on Unfamiliar Schema
# ---------------------------------------------------------------------------
def test_d_sql_exploration_on_unfamiliar_schema(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'unfamiliar.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE warehouse_inventory (sku_id TEXT PRIMARY KEY, warehouse_city TEXT, stock_on_hand INTEGER, reorder_point INTEGER)"
        ))
        conn.execute(text(
            "CREATE TABLE shipment_incidents (incident_id INTEGER PRIMARY KEY, sku_id TEXT, delay_days INTEGER, severity TEXT)"
        ))
        conn.execute(text(
            "INSERT INTO warehouse_inventory VALUES ('SKU1', 'Rotterdam', 120, 50), ('SKU2', 'Rotterdam', 20, 40), ('SKU3', 'Hamburg', 200, 60)"
        ))
        conn.execute(text(
            "INSERT INTO shipment_incidents VALUES (1, 'SKU1', 6, 'high'), (2, 'SKU2', 2, 'low'), (3, 'SKU3', 9, 'high')"
        ))

    gateway = ReadOnlyGateway(engine=engine, approved_tables={"warehouse_inventory", "shipment_incidents"})
    ds_store = DatasetStore(str(tmp_path / "ds.db"))

    observed_schema_in_step2: list[str] = []

    def _handler(step, model, payload):
        if step == 1:
            return SQLStepDecision(
                phase="discover_schema",
                action="inspect_schema",
                reasoning="Inspect unfamiliar warehouse_inventory and shipment_incidents columns and samples first.",
                tables_to_inspect=["warehouse_inventory", "shipment_incidents"],
            )
        if step == 2:
            observed_schema_in_step2.append(payload["detailed_schema_inspected"])
            # Intentional bad column name to test repair
            return SQLStepDecision(
                phase="query",
                action="run_sql",
                reasoning="Aggregate incident delay days by city.",
                sql_query="SELECT wi.city_name, SUM(si.delay_days) AS total_delay FROM warehouse_inventory wi JOIN shipment_incidents si ON wi.sku_id = si.sku_id GROUP BY wi.city_name",
            )
        if step == 3:
            assert payload["last_error"] is not None
            return SQLStepDecision(
                phase="repair",
                action="run_sql",
                reasoning="Fix column name from city_name to inspected warehouse_city.",
                sql_query="SELECT wi.warehouse_city, SUM(si.delay_days) AS total_delay FROM warehouse_inventory wi JOIN shipment_incidents si ON wi.sku_id = si.sku_id GROUP BY wi.warehouse_city ORDER BY total_delay DESC",
            )
        return SQLStepDecision(
            phase="conclude",
            action="finish",
            reasoning="Retrieved verified delay totals per warehouse city.",
            observation_notes="Hamburg had 9 delay days, Rotterdam had 8 delay days.",
        )

    analyst = SQLAnalyst(model_selector=_ScriptedSelector(_handler))  # type: ignore[arg-type]
    task = AnalysisTask(
        id="explore_warehouse",
        title="Warehouse Delay Investigation",
        agent="sql",
        purpose="Measure total shipment delay days by warehouse city",
        query_spec=QuerySpec(metric="gross_revenue", mode="autonomous", custom_request="Total delay days per warehouse city"),
    )
    sql, _, qres, ds_ref, steps, usages, repairs = analyst.investigate_sql(
        task=task,
        gateway=gateway,
        dataset_store=ds_store,
        run_id="run_d",
    )

    assert "warehouse_city" in sql
    assert "warehouse_city" in observed_schema_in_step2[0]
    assert repairs == 1
    assert len(steps) == 4
    rows_by_city = {r["warehouse_city"]: r["total_delay"] for r in qres["rows"]}
    assert rows_by_city == {"Hamburg": 9, "Rotterdam": 8}
    assert ds_ref.row_count == 2


# ---------------------------------------------------------------------------
# TEST E: Hypothesis Rejection and Revision
# ---------------------------------------------------------------------------
def test_e_hypothesis_rejection_and_revision(tmp_path: Path):
    ds_store = DatasetStore(str(tmp_path / "ds.db"))
    sb = SandboxExecutionService(provider=IsolatedProcessSandboxProvider(base_dir=tmp_path / "sb"))

    # Data deliberately contradicts shipping_delay hypothesis and strongly supports discount_rate
    obs_rows = [
        {"month": "2004-01", "revenue": 100000.0, "shipping_delay": 3.0, "discount_pct": 2.0},
        {"month": "2004-02", "revenue": 90000.0, "shipping_delay": 3.0, "discount_pct": 6.0},
        {"month": "2004-03", "revenue": 80000.0, "shipping_delay": 2.9, "discount_pct": 10.0},
        {"month": "2004-04", "revenue": 70000.0, "shipping_delay": 3.1, "discount_pct": 14.0},
        {"month": "2004-05", "revenue": 60000.0, "shipping_delay": 3.0, "discount_pct": 18.0},
        {"month": "2004-06", "revenue": 50000.0, "shipping_delay": 3.0, "discount_pct": 22.0},
    ]

    initial_hyp = HypothesisRecord(
        id="hyp_shipping_delay",
        statement="Revenue decline was associated with rising shipping delays.",
        hypothesis_type="causal_candidate",
        dimension="shipping_delay",
        expected_direction="increase",
        evidence_required="Pearson correlation between shipping_delay and revenue",
        priority=1,
    )

    def _handler(step, model, payload):
        if step == 1:
            return HypothesisStepDecision(
                hypothesis_id="hyp_shipping_delay",
                phase="run_python_test",
                action="run_python_test",
                reasoning="Compute Pearson correlation between shipping_delay and revenue.",
                python_code="""
from scipy import stats
df = load_dataset('monthly_obs')
r, p = stats.pearsonr(df['shipping_delay'], df['revenue'])
save_result({'var': 'shipping_delay', 'pearson_r': round(float(r), 4), 'p_value': round(float(p), 4)})
print(f'shipping_delay r={r:.4f}, p={p:.4f}')
""",
            )
        if step == 2:
            # Observe near-zero correlation -> reject initial hypothesis and propose revised hypothesis
            return HypothesisStepDecision(
                hypothesis_id="hyp_shipping_delay",
                phase="revise",
                action="reject_and_revise",
                reasoning="shipping_delay exhibits near-zero correlation (p > 0.8), contradicting initial hypothesis. Test discount_pct instead.",
                verdict="rejected",
                measured_effect=0.0,
                p_value=0.9,
                statistical_evidence="Pearson r=0.000, p=0.9+ between shipping_delay and revenue.",
                revised_statement="Revenue decline strongly aligns with sharp increases in discount_pct.",
                revised_dimension="discount_pct",
                confidence=0.9,
            )
        if step == 3:
            return HypothesisStepDecision(
                hypothesis_id="hyp_shipping_delay_revised",
                phase="run_python_test",
                action="run_python_test",
                reasoning="Test correlation between discount_pct and revenue.",
                python_code="""
from scipy import stats
df = load_dataset('monthly_obs')
r, p = stats.pearsonr(df['discount_pct'], df['revenue'])
save_result({'var': 'discount_pct', 'pearson_r': round(float(r), 4), 'p_value': round(float(p), 6)})
print(f'discount_pct r={r:.4f}, p={p:.6f}')
""",
            )
        return HypothesisStepDecision(
            hypothesis_id="hyp_shipping_delay_revised",
            phase="conclude",
            action="accept_verdict",
            reasoning="Verified strong negative correlation between discount_pct and revenue (r=-1.0, p<0.001).",
            verdict="supported",
            measured_effect=-1.0,
            p_value=0.0,
            statistical_evidence="Pearson r=-1.000, p<0.0001 between discount_pct and revenue (correlation only, non-causal).",
            confidence=0.95,
        )

    hyp_agent = HypothesisAgent(model_selector=_ScriptedSelector(_handler))  # type: ignore[arg-type]
    evaluated, _ = hyp_agent.investigate_hypotheses(
        hypotheses=[initial_hyp],
        datasets_by_task={"monthly_obs": obs_rows},
        dataset_store=ds_store,
        sandbox=sb,
        run_id="run_e",
        max_steps=5,
    )

    by_id = {h.id: h for h in evaluated}
    assert by_id["hyp_shipping_delay"].status == "rejected"
    assert by_id["hyp_shipping_delay"].revised_statement == "Revenue decline strongly aligns with sharp increases in discount_pct."
    assert "hyp_shipping_delay_revised" in by_id
    assert by_id["hyp_shipping_delay_revised"].status == "supported"
    assert (by_id["hyp_shipping_delay_revised"].p_value or 0.0) < 0.01
    assert "causal mechanism" in by_id["hyp_shipping_delay_revised"].causal_disclaimer


# ---------------------------------------------------------------------------
# TEST F: Independent Verification Reproduces Finding & Catches Mismatches
# ---------------------------------------------------------------------------
def test_f_independent_verification_reproduction(tmp_path: Path):
    ds_store = DatasetStore(str(tmp_path / "ds.db"))
    sb = SandboxExecutionService(provider=IsolatedProcessSandboxProvider(base_dir=tmp_path / "sb"))
    fake_reviewer_model = _ScriptedSelector(
        lambda step, model, payload: ReviewDraft(
            sufficient=False,
            confidence=0.2,
            assessment="Detected reproduction mismatch in Python calculation.",
            metric_alignment_ok=True,
        )
    )
    reviewer = EvidenceReviewerAgent(ds_store, model_selector=fake_reviewer_model, sandbox=sb)  # type: ignore[arg-type]

    df = pd.DataFrame([{"x": 10}, {"x": 20}, {"x": 30}, {"x": 40}])
    ds_ref = ds_store.put(run_id="run_f", task_id="nums", columns=["x"], rows=df.to_dict(orient="records"))

    code = """
df = load_dataset('nums')
save_result({'mean_x': float(df['x'].mean()), 'sum_x': float(df['x'].sum())})
"""

    ok_verdict = reviewer.reproduce_finding(
        code=code,
        dataset_ids=[ds_ref.dataset_id],
        expected_summary={"mean_x": 25.0, "sum_x": 100.0},
        dataset_names=["nums"],
    )
    assert ok_verdict["verified"] is True
    assert ok_verdict["status"] == "reproduced"

    tampered_verdict = reviewer.reproduce_finding(
        code=code,
        dataset_ids=[ds_ref.dataset_id],
        expected_summary={"mean_x": 999.0, "sum_x": 100.0},
        dataset_names=["nums"],
    )
    assert tampered_verdict["verified"] is False
    assert tampered_verdict["status"] == "mismatch"
    assert any("mean_x" in m for m in tampered_verdict["mismatches"])

    # Verify aggregate() + review() surfaces verification mismatch as a major issue
    plan = AnalysisPlan(
        plan_id="plan_f",
        question="Verify custom calculation",
        objective="Check mean and sum",
        complexity="comparison",
        metrics=["gross_revenue"],
        tasks=[
            AnalysisTask(id="nums", title="SQL", agent="sql", purpose="data"),
            AnalysisTask(
                id="py_calc",
                title="Custom Python",
                agent="python",
                purpose="mean and sum",
                depends_on=["nums"],
                python_spec=PythonToolSpec(tool="autonomous_research", mode="autonomous", input_task_ids=["nums"]),
            ),
        ],
        output=OutputRequirements(format="answer_with_visuals"),
        budget=RunBudget(
            max_queries=5,
            max_refinements=1,
            max_seconds=60,
            max_tokens=100000,
            max_cost_usd=1.0,
            max_sql_repairs=2,
            max_parallel_db_queries=3,
            max_parallel_tasks=3,
            max_task_retries=1,
            task_timeout_seconds=30,
        ),
    )
    sql_res = TaskResult(task_id="nums", agent="sql", status="COMPLETED", dataset=ds_ref, query_id="q1")
    tampered_py_res = TaskResult(
        task_id="py_calc",
        agent="python",
        status="COMPLETED",
        tool="autonomous_research",
        generated_code=code,
        calculation={"summary": {"mean_x": 999.0, "sum_x": 100.0}, "records": []},
    )
    ev_records, hyps, ds_map, bdelta = reviewer.aggregate(
        plan,
        {"nums": sql_res.model_dump(), "py_calc": tampered_py_res.model_dump()},
        [],
    )
    review, _ = reviewer.review(plan, ev_records, hyps, ds_map, bdelta, [], 0, {"queries_executed": 1, "tokens_used": 0, "cost_usd": 0.0})
    assert any("mismatch" in iss.description.lower() for iss in review.deterministic_issues)
    assert review.sufficient is False


# ---------------------------------------------------------------------------
# TEST G: Sandbox Security & Secret Isolation
# ---------------------------------------------------------------------------
def test_g_sandbox_security_controls(tmp_path: Path):
    sb = SandboxExecutionService(provider=IsolatedProcessSandboxProvider(base_dir=tmp_path / "sb"))
    ws = sb.create_workspace("run_sec", "sec_task", SandboxLimits(timeout_seconds=10))
    try:
        for bad in (
            "import socket\nprint(1)",
            "import subprocess\nprint(1)",
            "import urllib.request\nprint(1)",
            "eval('2+2')",
            "().__class__.__bases__",
            "open('../../.env').read()",
            "open('../app/config.py').read()",
            "open('../../escape.txt', 'w').write('x')",
        ):
            with pytest.raises(SandboxSecurityError):
                sb.execute(ws, bad)

        res_env = sb.execute(
            ws,
            "import os\nsens = [k for k in os.environ if any(t in k for t in ('KEY','URL','PASS','SECRET'))]\nsave_result({'leaked': sens})",
        )
        assert res_env.status == "COMPLETED"
        assert res_env.summary["leaked"] == []

        res_timeout = sb.execute(ws, "import time\ntime.sleep(5)", timeout_seconds=1)
        assert res_timeout.status == "TIMED_OUT"
    finally:
        sb.destroy_workspace(ws)


# ---------------------------------------------------------------------------
# TEST H: Live End-to-End Autonomous Research (OpenAI + Jev + MySQL)
# ---------------------------------------------------------------------------
def _live_creds_present() -> bool:
    return bool(settings.openai_api_key and settings.openrouter_api_key and settings.database_url)


@pytest.mark.skipif(not _live_creds_present(), reason="Live credentials or DATABASE_URL not configured.")
def test_h_live_open_ended_autonomous_e2e(tmp_path: Path):
    from app.orchestration.workflow import AnalyticalWorkflow

    wf = AnalyticalWorkflow(
        dataset_store=DatasetStore(str(tmp_path / "live_ds.db")),
        checkpoint_path=str(tmp_path / "live_cp.db"),
    )

    prompts = [
        "Segment ClassicModels customers using RFM behavioral metrics and K-Means clustering, and report revenue share of each cluster.",
        "Test whether monthly revenue exhibits seasonal patterns and outlier months across full history using Python statistical analysis.",
    ]

    for idx, q in enumerate(prompts):
        state = wf.run(question=q, session_id=f"live_auto_{idx}")
        rep = state["report"]
        assert rep["status"] in ("complete", "partial")
        assert rep["findings"], f"No findings produced for '{q}'"
        assert rep["evidence"], f"No SQL evidence produced for '{q}'"
        auto_py = [
            p for p in rep.get("python_analyses", [])
            if p.get("tool") in ("autonomous_research", "custom_python") or p.get("generated_code")
        ]
        assert auto_py, f"Expected autonomous Python research task for '{q}', got {rep.get('python_analyses')}"
        assert auto_py[0].get("generated_code"), "Expected non-empty generated Python code."

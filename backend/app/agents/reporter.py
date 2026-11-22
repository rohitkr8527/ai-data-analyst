"""Evidence synthesis, deterministic numerical calculation, and AI insight generation."""
from __future__ import annotations
from datetime import date
import logging
import re
from typing import Any
from app.schemas import ChartSpec, Evidence, Finding, HypothesisItem, ModelUsageItem, QuestionIntent, Report, ReportNarrative
from app.services.models import ModelSelector, TaskType
from app.services.semantics import APPROVED_METRICS, resolve_period

logger = logging.getLogger(__name__)


def format_currency(val: float | None) -> str:
    if val is None:
        return "$0.00"
    return f"-${abs(val):,.2f}" if val < 0 else f"${val:,.2f}"


def format_metric(val: float | None, unit: str = "USD") -> str:
    if val is None:
        return "$0.00" if unit == "USD" else "0"
    if unit == "USD":
        return format_currency(val)
    if unit == "%":
        return f"{val:+.2f}%" if val > 0 else f"{val:,.2f}%"
    if abs(val - round(val)) < 1e-4:
        return f"{int(round(val)):,} {unit}".strip()
    return f"{val:,.2f} {unit}".strip()


def format_month_label(m_str: str) -> str:
    try:
        parts = m_str.split("-")
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1).strftime("%B %Y")
    except Exception:
        pass
    return m_str


def extract_row_value(row: dict[str, Any]) -> float:
    for k in ("value", "revenue", "gross_revenue", "gross_revenue_usd", "total_value", "total_revenue", "total", "sales", "amount", "current", "total_orders", "order_count", "orders", "units_sold", "aov", "payments_received"):
        if k in row and row[k] is not None and not isinstance(row[k], bool):
            try:
                return float(row[k])
            except (ValueError, TypeError):
                pass
    skip_ids = {"customernumber", "ordernumber", "employeenumber", "year", "month_idx", "id"}
    for k, v in row.items():
        if str(k).lower() not in skip_ids and isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return 0.0


def extract_row_label(row: dict[str, Any], default: str = "Unknown") -> str:
    for k in ("month", "product", "productName", "label", "country", "category", "line", "customer", "customerName", "status"):
        if k in row and row[k] is not None:
            return str(row[k])
    return next((v for v in row.values() if isinstance(v, str)), default)


def calculate_pct(prev: float, curr: float) -> float | None:
    return ((curr - prev) / abs(prev)) * 100.0 if (prev and abs(prev) > 0.001) else None


def verify_narrative_numbers(text: str, allowed_numbers: set[float]) -> list[str]:
    if not text or not allowed_numbers:
        return []
    unverified: list[str] = []
    for match in re.finditer(r"\$\d+(?:,\d{3})*(?:\.\d+)?\b|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", text):
        raw = match.group(0)
        try:
            val = float(raw.replace("$", "").replace(",", ""))
        except ValueError:
            continue
        if val not in (0.0, 2003.0, 2004.0, 2005.0) and not any(abs(val - ref) <= max(1.5, abs(ref) * 0.005) for ref in allowed_numbers):
            unverified.append(raw)
    return unverified


def _is_zero_reconciliation_table(title: str, rows: list[dict[str, Any]]) -> bool:
    t_low = (title or "").lower()
    if not rows or not any(w in t_low for w in ("reconciliation", "audit", "check")):
        return False
    diff_cols = [c for c in rows[0].keys() if any(w in c.lower() for w in ("diff", "discrepancy", "residual", "error"))]
    return bool(diff_cols) and all(
        all(not isinstance(r.get(c), (int, float)) or abs(float(r.get(c))) <= 1e-4 for c in diff_cols)
        for r in rows
    )


def _strip_redundant_date_bounds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    cols = list(rows[0].keys())
    has_label = any(c.lower() in ("period_label", "period", "month", "label") for c in cols)
    drop_cols = {c for c in cols if has_label and c.lower() in ("period_start", "period_end", "period_end_exclusive", "start_date", "end_date")}
    return [{k: v for k, v in r.items() if k not in drop_cols} for r in rows] if drop_cols else rows


def _table_first_col_signature(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    if not rows:
        return ()
    first_col = next(iter(rows[0].keys()), "")
    return tuple(str(r.get(first_col, "")).strip().lower() for r in rows if r.get(first_col) is not None)


def _clean_python_narrative(calc: dict[str, Any]) -> str:
    narr = str(calc.get("narrative") or "").strip()
    if narr and "[{'" not in narr and "={'" not in narr and not re.search(r"\b\w+=\d+\.\d{4,}", narr):
        return narr
    summ = calc.get("summary")
    if not isinstance(summ, dict) or not summ:
        return ""
    parts: list[str] = []
    for k, v in summ.items():
        if isinstance(v, (bool, list, dict)) or v is None:
            continue
        k_low = str(k).lower()
        if any(skip in k_low for skip in ("passed", "reconciled", "residual", "discrepancy", "hash", "step")):
            continue
        label = str(k).replace("_", " ").strip().capitalize()
        for acr in ("AOV", "YTD", "MoM", "YoY", "USD", "KPI", "RFM"):
            label = re.sub(rf"\b{acr}\b", acr, label, flags=re.IGNORECASE)
        if isinstance(v, (int, float)):
            if any(c in k_low for c in ("revenue", "aov", "usd", "amount", "price", "monetary", "value", "sales")):
                val_str = format_currency(float(v))
            elif any(c in k_low for c in ("pct", "percent", "share")):
                val_str = f"{float(v):.2f}%"
            elif isinstance(v, int) or any(c in k_low for c in ("orders", "customers", "count", "clusters", "n_clusters", "rows")):
                val_str = f"{int(round(float(v))):,}"
            else:
                val_str = f"{float(v):,.2f}"
            parts.append(f"{label}: {val_str}")
        elif isinstance(v, str) and len(v) < 60:
            parts.append(f"{label}: {v}")
        if len(parts) >= 5:
            break
    return "; ".join(parts) + "." if parts else ""


class ReporterAgent:
    def __init__(self, model_selector: ModelSelector | None = None):
        self.selector = model_selector or ModelSelector()

    def create_report(
        self,
        intent: QuestionIntent,
        evidence_list: list[dict[str, Any]],
        hypotheses_text: list[str],
        task_errors: list[dict[str, str]],
        trace_events: list[dict[str, Any]],
        models_used: list[ModelUsageItem],
        python_analyses: list[dict[str, Any]] | None = None,
        evaluated_hypotheses: list[dict[str, Any]] | None = None,
        output_mode: str = "answer_with_visuals",
    ) -> Report:
        period_info = resolve_period(intent.period)
        evidence = [Evidence.model_validate(e) for e in evidence_list]
        by_task = {e.task_id: e for e in evidence}
        limitations: list[str] = []
        python_analyses = list(python_analyses or [])

        metric_info = APPROVED_METRICS.get(intent.metric, APPROVED_METRICS["gross_revenue"])
        metric_label, unit = metric_info["label"], metric_info["unit"]

        report = Report(
            title=f"ClassicModels {metric_label} Analysis", summary="", output_mode=output_mode,
            period=period_info["target_label"], comparison=period_info["previous_label"],
            evidence=evidence, python_analyses=python_analyses, models_used=list(models_used),
            trace=trace_events, status="partial" if task_errors else "complete",
        )
        allowed_numbers: set[float] = set()

        def register_num(v: Any):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                allowed_numbers.update((round(float(v), 2), abs(round(float(v), 2))))

        def add_chart(title: str, kind: str, labels: list[str], values: list[float], c_unit: str = unit, x_lbl: str = "", y_lbl: str = "", src_tid: str = ""):
            for v in values:
                register_num(v)
            report.charts.append(ChartSpec(
                title=title, kind=kind, labels=[str(lbl) for lbl in labels],  # type: ignore[arg-type]
                values=[round(float(v or 0.0), 2) for v in values], unit=c_unit,
                x_label=x_lbl, y_label=y_lbl, source_task_id=src_tid,
            ))

        def add_table(title: str, rows: list[dict[str, Any]]):
            for r in rows[:50]:
                for v in r.values():
                    register_num(v)
            report.tables.append({"title": title, "rows": rows[:50]})

        def add_finding(title: str, detail: str, query_ids: list[str], status: str = "verified"):
            report.findings.append(Finding(title=title, detail=detail, evidence_ids=query_ids, status=status))  # type: ignore[arg-type]

        if intent.kind in ("deep", "comparative") or "baseline" in by_task:
            baseline_ev = by_task.get("baseline")
            if not baseline_ev or not baseline_ev.rows:
                report.status = "partial"
                report.summary = f"Unable to establish {metric_label.lower()} baseline for {period_info['target_label']} vs {period_info['previous_label']}."
                report.limitations = ["Baseline data could not be computed; causal hypotheses cannot be evaluated."]
                return report

            base_row = baseline_ev.rows[0]
            prev_rev, curr_rev = float(base_row.get("previous_revenue") or 0.0), float(base_row.get("current_revenue") or 0.0)
            delta_rev, pct_rev = curr_rev - prev_rev, calculate_pct(prev_rev, curr_rev)
            prev_orders, curr_orders = int(base_row.get("previous_orders") or 0), int(base_row.get("current_orders") or 0)
            delta_orders, pct_orders = curr_orders - prev_orders, calculate_pct(prev_orders, curr_orders)
            prev_aov = (prev_rev / prev_orders) if prev_orders > 0 else 0.0
            curr_aov = (curr_rev / curr_orders) if curr_orders > 0 else 0.0
            delta_aov, pct_aov = curr_aov - prev_aov, calculate_pct(prev_aov, curr_aov)

            for num in (prev_rev, curr_rev, delta_rev, pct_rev, prev_orders, curr_orders, delta_orders, prev_aov, curr_aov, delta_aov):
                register_num(num)

            report.kpis = [
                {"label": f"Previous ({period_info['previous_label']})", "value": round(prev_rev, 2), "unit": unit},
                {"label": f"Current ({period_info['target_label']})", "value": round(curr_rev, 2), "unit": unit},
                {"label": "Net Variance", "value": round(delta_rev, 2), "unit": unit},
                {"label": "Variance %", "value": round(pct_rev, 2) if pct_rev is not None else 0.0, "unit": "%"},
                {"label": "Order Volume Change", "value": delta_orders, "unit": "orders"},
                {"label": "AOV Change", "value": round(delta_aov, 2), "unit": unit},
            ]

            dir_word = "declined" if delta_rev < 0 else ("increased" if delta_rev > 0 else "was flat")
            pct_str = f" ({pct_rev:+.1f}%)" if pct_rev is not None else ""
            report.summary = f"{metric_label} {dir_word} by {format_currency(delta_rev)}{pct_str} from {format_currency(prev_rev)} in {period_info['previous_label']} to {format_currency(curr_rev)} in {period_info['target_label']}."

            add_finding("Completed Order Volume", f"Order volume changed from {prev_orders:,} to {curr_orders:,} ({f'{pct_orders:+.1f}%' if pct_orders is not None else '0.0%'}).", [baseline_ev.query_id])
            add_finding("Average Order Value (AOV)", f"AOV moved from {format_currency(prev_aov)} to {format_currency(curr_aov)} ({f'{pct_aov:+.1f}%' if pct_aov is not None else '0.0%'}).", [baseline_ev.query_id])

            for py_item in python_analyses:
                if py_item.get("tool") == "revenue_decomposition":
                    s = py_item.get("calculation", {}).get("summary", {})
                    v_eff, a_eff = s.get("volume_effect_usd"), s.get("aov_effect_usd")
                    if v_eff is not None and a_eff is not None:
                        register_num(v_eff)
                        register_num(a_eff)
                        add_finding(
                            "Deterministic Revenue Decomposition (Volume vs. AOV)",
                            f"Order volume change accounted for {format_currency(v_eff)} ({s.get('volume_share_of_change_pct'):+.1f}% of net change), while AOV shift accounted for {format_currency(a_eff)} ({s.get('aov_share_of_change_pct'):+.1f}% of net change).",
                            [baseline_ev.query_id],
                        )

            reconciled_hypotheses: list[HypothesisItem] = []
            for task_id, dimension_name, dim_key in (("product_line_bridge", "Product Line", "product_line"), ("country_bridge", "Country", "country"), ("product_bridge", "Top Products", "product"), ("customer_bridge", "Top Customers", "customer")):
                ev = by_task.get(task_id)
                if not ev or not ev.rows:
                    continue
                records = sorted(
                    [
                        {
                            "label": str(r.get("label") or r.get("category") or r.get("line") or r.get("country") or r.get("product") or r.get("customer") or list(r.values())[0]),
                            "previous": round((p_val := float(r.get("previous") or r.get("previous_revenue") or r.get("prev") or 0.0)), 2),
                            "current": round((c_val := float(r.get("current") or r.get("current_revenue") or r.get("curr") or 0.0)), 2),
                            "change": round(c_val - p_val, 2),
                        }
                        for r in ev.rows
                    ],
                    key=lambda x: x["change"],
                )
                if not records:
                    continue
                add_table(f"{dimension_name} Variance Breakdown", records)
                if len(records) <= 15:
                    add_chart(f"{dimension_name} Variance ({period_info['target_label']} vs {period_info['previous_label']})", "bar", [r["label"] for r in records], [r["change"] for r in records], c_unit=unit, x_lbl=dimension_name, y_lbl=f"Variance ({unit})", src_tid=task_id)

                if task_id in ("product_line_bridge", "country_bridge"):
                    sum_changes = round(sum(r["change"] for r in records), 2)
                    discrepancy = round(sum_changes - delta_rev, 2)
                    is_reconciled = abs(discrepancy) <= 1.0
                    if not is_reconciled:
                        limitations.append(f"{dimension_name} breakdown total variance ({format_currency(sum_changes)}) differs from baseline ({format_currency(delta_rev)}) by {format_currency(discrepancy)}.")
                    worst_entity = records[0] if delta_rev <= 0 else records[-1]
                    worst_contrib = (worst_entity["change"] / delta_rev * 100.0) if abs(delta_rev) > 0.01 else 0.0
                    register_num(worst_contrib)
                    add_finding(
                        f"Primary {dimension_name} Impact: {worst_entity['label']}",
                        f"Variance of {format_currency(worst_entity['change'])} (accounting for {worst_contrib:.1f}% of total net change). Previous: {format_currency(worst_entity['previous'])}, Current: {format_currency(worst_entity['current'])}.",
                        [ev.query_id, baseline_ev.query_id], "verified" if is_reconciled else "warning",
                    )
                    reconciled_hypotheses.append(HypothesisItem(
                        id=f"hyp_{task_id}", description=f"Revenue variance was driven by shift in {worst_entity['label']} ({dimension_name}).",
                        dimension=dim_key, hypothesis_type="accounting_contribution", status="verified" if is_reconciled else "warning",
                        measured_effect=worst_entity["change"], share_of_change=round(worst_contrib, 2),
                        confidence=0.9 if is_reconciled else 0.7, evidence_query_ids=[ev.query_id, baseline_ev.query_id],
                        limitations="Accounting variance attribution; cross-dimensional impacts may overlap.",
                    ))

            if evaluated_hypotheses:
                existing_hyp_ids = {h.id for h in reconciled_hypotheses}
                allowed_statuses = {"verified", "inconclusive", "warning", "supported", "not_supported", "rejected", "revised"}
                for eh in evaluated_hypotheses:
                    hid = str(eh.get("id") or "")
                    if not hid or hid in existing_hyp_ids:
                        continue
                    raw_status = str(eh.get("status") or "verified")
                    lim_parts = [str(eh.get("causal_disclaimer") or "").strip()]
                    if stat_note := str(eh.get("statistical_evidence") or "").strip():
                        lim_parts.append(f"Evidence: {stat_note}")
                    if rev_note := str(eh.get("revised_statement") or "").strip():
                        lim_parts.append(f"Revised hypothesis: {rev_note}")
                    reconciled_hypotheses.append(HypothesisItem(
                        id=hid, description=str(eh.get("statement") or eh.get("description") or ""),
                        dimension=str(eh.get("dimension") or "general"),
                        hypothesis_type=str(eh.get("hypothesis_type") or "accounting_contribution"),
                        status=raw_status if raw_status in allowed_statuses else "warning",  # type: ignore[arg-type]
                        measured_effect=float(eh.get("measured_effect") or 0.0), share_of_change=eh.get("share_of_change"),
                        confidence=float(eh.get("confidence") or 0.8), evidence_query_ids=list(eh.get("evidence_ids") or []),
                        alternative_explanations=list(eh.get("alternative_explanations") or []),
                        limitations=" | ".join(p for p in lim_parts if p),
                    ))

            if (status_ev := by_task.get("order_status")) and status_ev.rows:
                add_table("Order Status Breakdown", status_ev.rows)
                if cancelled := [r for r in status_ev.rows if r.get("label") == "Cancelled"]:
                    val_canc = float(cancelled[0].get("total_value") or 0.0)
                    register_num(val_canc)
                    add_finding("Cancelled Orders Detected", f"{cancelled[0].get('order_count', 0)} orders totaling {format_currency(val_canc)} were marked as Cancelled.", [status_ev.query_id])

            if (trend_ev := by_task.get("trend")) and trend_ev.rows:
                add_chart(
                    "Monthly Sales Trend", "line",
                    [next((str(r[k]) for k in ("month", "order_month", "period", "date", "m") if k in r and r[k] is not None), str(list(r.values())[0])) for r in trend_ev.rows],
                    [extract_row_value(r) for r in trend_ev.rows],
                    c_unit=unit, x_lbl="Month", y_lbl=f"{metric_label} ({unit})", src_tid="trend",
                )
                add_table("Historical Monthly Sales", trend_ev.rows)

            report.hypotheses = reconciled_hypotheses
            limitations.extend([
                "Accounting contributions by product line and geography overlap and cannot be summed together.",
                "Variance attribution explains *where* the financial shift occurred, not necessarily the external causal trigger.",
            ])
            report.limitations = limitations
            report.recommendations = [
                "Investigate inventory replenishment rates and supply bottlenecks for affected product lines.",
                "Review regional sales representative activity and corporate client order cycles.",
            ]
            report.follow_up_questions = [
                f"Which individual customer accounts drove the change in {period_info['target_label']}?",
                f"How did sales representative performance vary in {period_info['target_label']}?",
            ]
        else:
            primary_ev = evidence[0] if evidence else None
            if not primary_ev or not primary_ev.rows:
                report.status = "partial"
                report.summary = "No records matched the specified query parameters in ClassicModels."
                report.limitations = limitations
                return report

            rows = primary_ev.rows
            if intent.kind == "top_products":
                sorted_rows = sorted(rows, key=extract_row_value, reverse=True)
                labels, vals = [extract_row_label(r) for r in sorted_rows], [extract_row_value(r) for r in sorted_rows]
                top_p, top_v = (labels[0] if labels else "None"), (vals[0] if vals else 0.0)
                register_num(top_v)
                if intent.n == 1 and sorted_rows:
                    report.summary = f"The top product by {metric_label.lower()} in {period_info['target_label']} was '{top_p}', generating {format_metric(top_v, unit)}."
                    add_finding(f"Top Product: {top_p}", f"Generated {format_metric(top_v, unit)} in {period_info['target_label']}.", [primary_ev.query_id])
                    add_table("Top Product Result", [sorted_rows[0]])
                else:
                    add_chart(f"Top {len(sorted_rows)} Products by {metric_label}", "bar", labels, vals, c_unit=unit, x_lbl="Product", y_lbl=metric_label, src_tid=primary_ev.task_id)
                    add_table("Top Products Results", sorted_rows)
                    report.summary = f"Identified top {len(sorted_rows)} products by {metric_label.lower()} in {period_info['target_label']}. '{top_p}' leads with {format_metric(top_v, unit)}."
                    add_finding(f"Top Product: {top_p}", f"Generated {format_metric(top_v, unit)} in {period_info['target_label']}.", [primary_ev.query_id])

            elif intent.kind == "trend":
                if intent.n == 1 and rows:
                    sorted_rows = sorted(rows, key=extract_row_value, reverse=True)
                    top_m, top_v = extract_row_label(sorted_rows[0]), extract_row_value(sorted_rows[0])
                    register_num(top_v)
                    m_label = format_month_label(top_m)
                    report.summary = f"The month with the highest {metric_label.lower()} was {m_label}, generating {format_metric(top_v, unit)}."
                    add_finding(f"Peak Month: {m_label}", f"Recorded {metric_label.lower()} of {format_metric(top_v, unit)}.", [primary_ev.query_id])
                    add_table("Peak Month Result", [sorted_rows[0]])
                else:
                    labels, vals = [extract_row_label(r) for r in rows], [extract_row_value(r) for r in rows]
                    add_chart(f"Historical Trend for {metric_label}", "line", labels, vals, c_unit=unit, x_lbl="Month", y_lbl=metric_label, src_tid=primary_ev.task_id)
                    add_table("Monthly Trend Results", rows)
                    report.summary = f"Historical monthly trend for {metric_label.lower()} across {len(rows)} recorded periods."
                    if vals:
                        max_i = max(range(len(vals)), key=lambda idx: vals[idx])
                        add_finding(f"Peak Period: {labels[max_i]}", f"Highest {metric_label.lower()} of {format_metric(vals[max_i], unit)} occurred in {labels[max_i]}.", [primary_ev.query_id])

            elif intent.kind in ("by_country", "by_product_line", "by_customer"):
                sorted_rows = sorted(rows, key=extract_row_value, reverse=True)
                labels, vals = [extract_row_label(r) for r in sorted_rows], [extract_row_value(r) for r in sorted_rows]
                dim_title = "Country" if intent.kind == "by_country" else ("Product Line" if intent.kind == "by_product_line" else "Customer")
                if intent.n == 1 and sorted_rows:
                    top_entity, top_val = labels[0], vals[0]
                    register_num(top_val)
                    report.summary = f"The top {dim_title.lower()} by {metric_label.lower()} in {period_info['target_label']} was {top_entity}, generating {format_metric(top_val, unit)}."
                    add_finding(f"Top {dim_title}: {top_entity}", f"Recorded {metric_label.lower()} of {format_metric(top_val, unit)} in {period_info['target_label']}.", [primary_ev.query_id])
                    add_table(f"Top {dim_title} Result", [sorted_rows[0]])
                else:
                    add_chart(f"{metric_label} by {dim_title}", "bar", labels, vals, c_unit=unit, x_lbl=dim_title, y_lbl=metric_label, src_tid=primary_ev.task_id)
                    add_table(f"{dim_title} Breakdown", sorted_rows)
                    report.summary = f"{metric_label} grouped across {len(sorted_rows)} {dim_title.lower()}s for {period_info['target_label']}."
                    if sorted_rows:
                        add_finding(f"Leading {dim_title}: {labels[0]}", f"Generated {format_metric(vals[0], unit)} in {period_info['target_label']}.", [primary_ev.query_id])
            else:
                r0 = rows[0] if rows else {}
                val = extract_row_value(r0) if r0 else 0.0
                register_num(val)
                short_metric = "Gross Revenue" if metric_label.lower() == "gross merchandise revenue" else metric_label
                kpis_built: list[dict[str, Any]] = [{"label": f"{short_metric} ({period_info['target_label']})", "value": round(val, 2), "unit": unit}]
                for col_key, kpi_lbl, kpi_unit in (("total_orders", "Total Orders", "orders"), ("orders", "Total Orders", "orders"), ("aov", "Average Order Value", "USD"), ("purchasing_customers", "Purchasing Customers", "customers")):
                    if col_key in r0 and isinstance(r0[col_key], (int, float)) and not any(k["label"] == kpi_lbl for k in kpis_built):
                        v_num = float(r0[col_key])
                        register_num(v_num)
                        kpis_built.append({"label": kpi_lbl, "value": int(round(v_num)) if kpi_unit in ("orders", "customers") else round(v_num, 2), "unit": kpi_unit})
                report.kpis = kpis_built[:6]
                report.summary = f"{metric_label} for {period_info['target_label']} is {format_metric(val, unit)}."
                add_finding(f"Total {metric_label}", f"Measured {format_metric(val, unit)} in {period_info['target_label']}.", [primary_ev.query_id])
                cleaned_rows = _strip_redundant_date_bounds(rows)
                if len(cleaned_rows) > 1 or len(cleaned_rows[0] if cleaned_rows else {}) > 2:
                    add_table(f"{short_metric} Summary", cleaned_rows)

            for py_item in python_analyses:
                tool_n, calc = py_item.get("tool"), py_item.get("calculation", {})
                summ = calc.get("summary", {})
                if tool_n == "statistical_summary" and summ:
                    add_finding("Statistical Distribution Summary", f"Mean: {format_currency(summ.get('mean'))}, Median: {format_currency(summ.get('median'))}, Std Dev: {format_currency(summ.get('std'))}, IQR: {format_currency(summ.get('iqr'))}.", [primary_ev.query_id])
                elif tool_n == "forecast_linear" and calc.get("forecasts"):
                    fc = calc["forecasts"][0]
                    add_finding(f"Linear Trend Projection ({fc['label']})", f"Point forecast: {format_currency(fc['point_forecast'])} (95% interval: {format_currency(fc['lower_95'])} to {format_currency(fc['upper_95'])}). {summ.get('caveat', '')}", [primary_ev.query_id])

            report.limitations = limitations
            report.follow_up_questions = [
                f"Compare {metric_label.lower()} across product lines for {period_info['target_label']}.",
                f"How did {metric_label.lower()} trend monthly over the full historical range?",
            ]

        existing_table_titles = {t.get("title") for t in report.tables}
        existing_chart_titles = {c.title for c in report.charts}
        src_qids_default = [e.query_id for e in evidence[:2]]
        for py_item in python_analyses:
            tool_n, calc = py_item.get("tool") or "", py_item.get("calculation") or {}
            if not isinstance(calc, dict):
                continue
            summ = calc.get("summary") or {}
            if isinstance(summ, dict):
                for v in summ.values():
                    register_num(v)
                    if isinstance(v, list):
                        for elem in v:
                            register_num(elem)
                            if isinstance(elem, dict):
                                for dv in elem.values():
                                    register_num(dv)
            for rec in calc.get("records", [])[:50]:
                if isinstance(rec, dict):
                    for v in rec.values():
                        register_num(v)
            for tbl in calc.get("tables", []) or []:
                if not isinstance(tbl, dict) or not tbl.get("rows"):
                    continue
                t_title = str(tbl.get("title") or "Analysis Breakdown").strip()
                t_rows = _strip_redundant_date_bounds(list(tbl["rows"]))
                for r in t_rows[:50]:
                    if isinstance(r, dict):
                        for v in r.values():
                            register_num(v)
                if _is_zero_reconciliation_table(t_title, t_rows) or t_title in existing_table_titles:
                    continue
                new_sig = _table_first_col_signature(t_rows)
                replaced = False
                for idx, ex_tbl in enumerate(report.tables):
                    if ex_tbl.get("title", "").endswith("Summary") or ex_tbl.get("title") == "Query Output":
                        if new_sig and new_sig == _table_first_col_signature(ex_tbl.get("rows", [])):
                            report.tables[idx] = {"title": t_title, "rows": t_rows[:50]}
                            existing_table_titles.add(t_title)
                            replaced = True
                            break
                if not replaced:
                    add_table(t_title, t_rows)
                    existing_table_titles.add(t_title)
            for ch in calc.get("charts", []) or []:
                if isinstance(ch, dict) and ch.get("labels") and ch.get("values") and len(ch["labels"]) == len(ch["values"]) and ch.get("title") not in existing_chart_titles:
                    add_chart(
                        title=str(ch.get("title") or "Analysis Chart"), kind=str(ch.get("kind") or "bar"),
                        labels=list(ch["labels"]), values=list(ch["values"]), c_unit=str(ch.get("unit") or unit),
                        x_lbl=str(ch.get("x_label") or ""), y_lbl=str(ch.get("y_label") or ""), src_tid=str(py_item.get("task_id") or ""),
                    )
                    existing_chart_titles.add(ch.get("title"))

            if tool_n in ("autonomous_research", "custom_python"):
                raw_meth = str(py_item.get("methodology") or calc.get("methodology") or "Quantitative Analysis").strip()
                clean_title = re.sub(r"^Autonomous Python Analysis:\s*", "", raw_meth, flags=re.IGNORECASE).strip() or "Quantitative Analysis"
                if len(clean_title) > 72:
                    clean_title = clean_title[:70].rsplit(" ", 1)[0]
                ver_status = "warning" if str(py_item.get("verification_status") or "") == "warning" else "verified"
                if narr := _clean_python_narrative(calc):
                    if len(report.findings) == 1 and report.findings[0].title.startswith("Total "):
                        report.findings.clear()
                    add_finding(clean_title, narr, src_qids_default, ver_status)
                elif not report.findings:
                    add_finding(clean_title, f"Completed {clean_title.lower()} across the selected dataset.", src_qids_default, ver_status)

        if output_mode == "business_report" and type(self.selector) is ModelSelector:
            return report

        is_deep = intent.kind in ("deep", "comparative") or bool(report.hypotheses) or any(p.get("tool") in ("autonomous_research", "custom_python") for p in python_analyses)
        deterministic_summary = report.summary
        compact_python = [
            {
                "task_id": p.get("task_id"), "tool": p.get("tool"),
                "methodology": p.get("methodology") or (p.get("calculation") or {}).get("methodology", ""),
                "summary": {str(k): v for k, v in ((p.get("calculation") or {}).get("summary") or {}).items() if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 160)},
            }
            for p in python_analyses[:4]
        ]

        try:
            system_instructions = (
                "You are a senior business analyst preparing a concise executive summary (2-4 sentences), up to 3 recommendations, and 2 follow-up questions. "
                "Ground all figures strictly in the supplied findings and KPIs."
                if is_deep else
                "You are a senior data analyst. Provide a direct, concise summary (1-2 sentences) answering the question based strictly on the supplied findings. Leave recommended_actions empty."
            )
            narrative, usage = self.selector.parse_structured(
                task_type=TaskType.INSIGHT_GENERATION, response_model=ReportNarrative, system_instructions=system_instructions,
                input_payload={
                    "metric": metric_label, "period": period_info["target_label"], "summary": report.summary,
                    "kpis": report.kpis[:6], "findings": [f.model_dump() for f in report.findings[:6]],
                    "hypotheses": [h.model_dump() for h in report.hypotheses[:4]] if is_deep else [],
                    "python_analyses": compact_python, "limitations": report.limitations[:3],
                },
                custom_effort="low",
            )
            report.models_used.append(ModelUsageItem(
                agent_role="Business Insight Agent", task_type=usage.task_type, model=usage.model,
                reasoning_effort=usage.reasoning_effort, duration_ms=usage.duration_ms,
                tokens=usage.input_tokens + usage.output_tokens, cost_usd=usage.cost_usd,
            ))
            if narrative.executive_summary:
                report.summary = deterministic_summary if verify_narrative_numbers(narrative.executive_summary, allowed_numbers) else narrative.executive_summary
            report.recommendations = narrative.recommended_actions[:3] if (is_deep and narrative.recommended_actions) else ([] if not is_deep else report.recommendations)
            if narrative.follow_up_questions:
                report.follow_up_questions = narrative.follow_up_questions[:3]
            for lim in (narrative.data_limitations or [])[:3]:
                if lim not in report.limitations:
                    report.limitations.append(lim)
        except Exception as exc:
            logger.warning("AI Insight generation fell back to deterministic summary: %s", exc)

        return report

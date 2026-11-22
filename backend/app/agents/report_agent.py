"""Report Generation Agent and Executive HTML/PDF Renderer."""
from __future__ import annotations
from datetime import datetime, timezone
import html
import io
import logging
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from fpdf import FPDF

from app.orchestration.contracts import AnalysisPlan, BusinessReport, EvidenceRecord, ReportDraft, ReportSection
from app.schemas import ModelUsageItem, Report
from app.services.models import ModelSelector, TaskType

logger = logging.getLogger(__name__)

REPORT_AGENT_SYSTEM_PROMPT = """You are a Senior Business Intelligence Report Writer.
Produce a clear, focused `ReportDraft` grounded strictly in the supplied KPIs, verified findings, tested hypotheses, and evidence records.
1. Every numerical claim MUST come directly from the provided KPIs, findings, or evidence records.
2. Write 2 to 3 substantive analytical sections. Do NOT create a separate "Methodology, Data Scope & Accounting Rules" section inside `sections`—capture scope notes in `limitations`.
3. Avoid filler or internal technical terminology (e.g., sandbox, staged aggregates, refinement rounds).
"""

_ASCII_MAP = {
    "\u2014": " - ", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2022": "-", "\u00b7": " | ", "\u2192": "->", "\u0394": "Delta ", "\u2265": ">=", "\u2264": "<=", "\u00a0": " ",
}
_ACRONYM_MAP = {"aov": "AOV", "ytd": "YTD", "mom": "MoM", "yoy": "YoY", "kpi": "KPI", "kpis": "KPIs", "usd": "USD", "rfm": "RFM", "sql": "SQL", "id": "ID", "pct": "%"}
_COUNT_TOKENS = ("count", "orders", "total_orders", "purchasing_customers", "customers", "units", "qty", "quantity", "cluster", "year", "days", "frequency")
_CURRENCY_TOKENS = ("revenue", "amount", "price", "value", "previous", "current", "baseline", "change", "variance", "effect", "contribution", "aov", "monetary", "sales")
_PCT_TOKENS = ("pct", "percent", "percentage", "share", "rate")


def _ascii_safe(text: str) -> str:
    out = str(text or "")
    for k, v in _ASCII_MAP.items():
        out = out.replace(k, v)
    return out.encode("latin-1", "replace").decode("latin-1")


def _strip_markdown(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\*\*(.*?)\*\*|__(.*?)__|`(.*?)`", lambda m: m.group(1) or m.group(2) or m.group(3) or "", str(text))
    return re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE).strip()


def _truncate_words(text: str, max_len: int) -> str:
    s = str(text or "").strip()
    if len(s) <= max_len:
        return s
    cut = s[: max_len - 3].rsplit(" ", 1)[0].strip(" ,;:-")
    return f"{cut}..." if cut else s[:max_len]


def _format_label(raw: Any) -> str:
    s = str(raw or "").strip().replace("_", " ")
    if not s:
        return ""
    return " ".join(_ACRONYM_MAP.get(w.lower(), w if (w.isupper() and len(w) <= 5) else w.capitalize()) for w in s.split())


def _format_metric_value(val: Any, unit: str = "", col_name: str = "") -> str:
    if val is None or val == "":
        return "-"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, float)):
        num, u, c = float(val), (unit or "").strip(), (col_name or "").lower()
        is_change = "change" in c or "variance" in c
        if u == "%" or any(k in c for k in _PCT_TOKENS):
            return f"{num:+.2f}%" if (is_change or u == "%") and num > 0 else f"{num:,.2f}%"
        if (u.lower() in ("orders", "customers", "units", "count", "days") or any(k in c for k in _COUNT_TOKENS)) and u != "USD":
            if is_change and num > 0:
                return f"+{int(round(num)):,}"
            return f"{int(round(num)):,}" if abs(num - round(num)) < 1e-4 else f"{num:,.2f}"
        if u == "USD" or any(k in c for k in _CURRENCY_TOKENS):
            if is_change and num > 0:
                return f"+${num:,.2f}"
            return f"-${abs(num):,.2f}" if num < 0 else f"${num:,.2f}"
        if isinstance(val, int) or abs(num - round(num)) < 1e-6:
            return f"{int(round(num)):,} {u}".strip()
        return f"{num:,.2f} {u}".strip()
    return str(val)


def _format_cell_value(val: Any, col_name: str = "", row: dict[str, Any] | None = None) -> str:
    if val is None or val == "":
        return "-"
    if isinstance(val, str):
        s = val.strip()
        if s.lower() in _ACRONYM_MAP:
            return _ACRONYM_MAP[s.lower()]
        if "_" in s and not s.startswith(("http", "ds_", "q_", "ev_")):
            return " ".join(_format_label(part) if "_" in part else part for part in s.split())
        return s
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return str(val)

    num, c = float(val), (col_name or "").lower()
    is_change = any(k in c for k in ("change", "variance", "delta"))
    if any(k in c for k in _PCT_TOKENS):
        return f"{num:+.2f}%" if is_change and num > 0 else f"{num:,.2f}%"

    row_metric = next((str(row[mk]).lower() for mk in ("metric", "measure", "kpi", "check", "indicator", "item", "variable") if isinstance(row, dict) and row.get(mk) is not None), "")
    if row_metric and any(k in c for k in ("current", "baseline", "previous", "change", "variance", "delta", "value", "diff")):
        if any(k in row_metric for k in _PCT_TOKENS):
            return f"{num:+.2f}%" if is_change and num > 0 else f"{num:,.2f}%"
        if any(k in row_metric for k in _COUNT_TOKENS):
            if abs(num - round(num)) < 1e-4:
                return f"+{int(round(num)):,}" if (is_change and num > 0) else f"{int(round(num)):,}"
            return f"{num:+,.2f}" if (is_change and num > 0) else f"{num:,.2f}"
        if any(k in row_metric for k in ("revenue", "aov", "amount", "price", "sales", "monetary", "value", "cost", "margin")):
            return f"+${num:,.2f}" if (is_change and num > 0) else (f"-${abs(num):,.2f}" if num < 0 else f"${num:,.2f}")

    return _format_metric_value(val, "", col_name)


def _markdown_to_html(text: str) -> str:
    if not text:
        return ""
    html_parts: list[str] = []
    in_list = False

    def _inline(s: str) -> str:
        esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(s))
        return re.sub(r"`(.+?)`", r"<code>\1</code>", esc)

    for raw_line in str(text).splitlines():
        line = re.sub(r"^#{1,6}\s+", "", raw_line.strip())
        if not line:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue
        if line.startswith(("- ", "* ", "• ")):
            if not in_list:
                html_parts.append('<ul class="rich-list">')
                in_list = True
            html_parts.append(f"<li>{_inline(line[2:].strip())}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p>{_inline(line)}</p>")
    if in_list:
        html_parts.append("</ul>")
    return "\n".join(html_parts)


def _is_raw_dump_text(text: str) -> bool:
    s = str(text or "").strip()
    return not s or "[{'" in s or "={'" in s or "=[{" in s or len(re.findall(r"\b[a-z_]+=[0-9.\-]+", s)) >= 3


def _is_internal_error_or_log_note(note: str) -> bool:
    s = str(note or "").strip().lower()
    if not s:
        return True
    internal_patterns = (
        "only select statements are permitted", "exceeded maximum row limit", "exceeding the row limit",
        "add a limit clause or grouping", "max_refinements", "wall-clock budget", "without repeating successful work",
        "two bounded aggregate queries", "detail requests failed", "ai narrative commentary unavailable",
        "syntaxerror", "querysecurityerror", "statement separators", "sql parsing failed", "unknown column",
    )
    return any(p in s for p in internal_patterns) or bool(re.match(r"^[a-z0-9_]+:\s*(only select|query result|database|syntax|unknown|execution|skipped)", s))


def _clean_report_limitations(raw_limitations: list[Any], section_bodies: list[str] | None = None) -> list[str]:
    cleaned: list[str] = []
    seen_topics: set[str] = set()
    sec_text_low = " ".join(section_bodies or []).lower()

    for item in raw_limitations or []:
        text = _strip_markdown(str(item or "")).strip()
        if not text or _is_internal_error_or_log_note(text):
            continue
        t_low = text.lower()
        topic_key = (
            "revenue_definition" if ("gross ordered quantity" in t_low or "recognized revenue" in t_low)
            else "date_coverage" if ("january 6, 2003" in t_low or "2003-01-06" in t_low)
            else "causation_disclaimer" if ("causation" in t_low or "causal" in t_low)
            else "overlapping_dimensions" if ("overlap" in t_low and ("product line" in t_low or "dimension" in t_low))
            else t_low[:45]
        )
        if topic_key in seen_topics:
            continue
        if (topic_key == "revenue_definition" and "recognized revenue" in sec_text_low) or (topic_key == "date_coverage" and ("january 6, 2003" in sec_text_low or "2003-01-06" in sec_text_low)):
            continue
        seen_topics.add(topic_key)
        cleaned.append(text)
        if len(cleaned) >= 4:
            break
    return cleaned


def _curate_report_tables(raw_tables: list[dict[str, Any]], has_line_chart: bool = False) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    anomaly_tokens = ("diff", "discrepancy", "residual", "error", "excess", "missing", "unmatched", "unknown", "issue", "null_", "invalid_")
    boilerplate_cols = {"coverage_note", "date_definition", "revenue_definition", "kpi_definition", "quality_definition"}

    for tbl in raw_tables or []:
        if not isinstance(tbl, dict) or not isinstance(rows := (tbl.get("rows") or []), list) or not rows:
            continue
        title = _strip_markdown(str(tbl.get("title") or "Supporting Data")).strip()
        t_low = title.lower()
        if any(w in t_low for w in ("reconciliation", "audit", "check", "integrity")):
            diff_cols = [c for c in rows[0].keys() if any(w in c.lower() for w in anomaly_tokens)]
            if diff_cols and all(all(not isinstance(r.get(c), (int, float)) or abs(float(r.get(c))) <= 1e-4 for c in diff_cols) for r in rows):
                continue
        if any(w in t_low for w in ("management actions", "recommended actions", "action plan")) and "action" in {c.lower() for c in rows[0].keys()}:
            continue

        cols = list(rows[0].keys())
        has_period_label = any(c.lower() in ("period_label", "period", "month", "year_month", "label") for c in cols)
        drop_cols = {
            c for c in cols
            if c.lower() in boilerplate_cols
            or (has_period_label and c.lower() in ("period_start", "period_end", "period_end_exclusive", "start_date", "end_date"))
            or (c.lower() in ("residual", "discrepancy") and all(not isinstance(r.get(c), (int, float)) or abs(float(r.get(c))) <= 1e-4 for r in rows))
        }
        is_chronological = any(c.lower() in ("year_month", "month", "order_month") for c in cols)
        sliced_rows = rows[-15:] if (is_chronological and len(rows) > 15) else rows[:25]
        clean_rows = [{k: v for k, v in r.items() if k not in drop_cols} for r in sliced_rows]
        if clean_rows and clean_rows[0]:
            candidates.append({"title": title, "rows": clean_rows, "_is_monthly_history": is_chronological and len(rows) >= 12})

    def _first_col_sig(rows: list[dict[str, Any]]) -> tuple[str, ...]:
        fc = next(iter(rows[0].keys()), "") if rows else ""
        return tuple(str(r.get(fc, "")).strip().lower() for r in rows) if fc else ()

    final_tables: list[dict[str, Any]] = []
    for idx, tbl in enumerate(candidates):
        sig = _first_col_sig(tbl["rows"])
        if tbl["title"].lower() in ("query output", "gross revenue summary", "gross merchandise revenue summary") and sig:
            if any(j != idx and _first_col_sig(other["rows"]) == sig for j, other in enumerate(candidates)):
                continue
        if tbl.get("_is_monthly_history") and has_line_chart and len(candidates) >= 3:
            continue
        final_tables.append({"title": tbl["title"], "rows": tbl["rows"]})
    return final_tables[:4]


def _prepare_report_view_data(report_dict: dict[str, Any]) -> dict[str, Any]:
    biz = report_dict.get("business_report") or {}
    title = _strip_markdown(str(biz.get("title") or report_dict.get("title") or "Analytical Report")).strip()
    summary = str(biz.get("executive_summary") or report_dict.get("summary") or "").strip()
    period = str(report_dict.get("period") or "Current Period").strip()
    comparison = str(report_dict.get("comparison") or "").strip()
    period_str = f"{period} vs. {comparison}" if (comparison and comparison.lower() not in period.lower()) else period

    charts = list(report_dict.get("charts") or [])[:4]
    has_line_chart = any(str(c.get("kind") or "").lower() == "line" for c in charts if isinstance(c, dict))
    tables = _curate_report_tables(report_dict.get("tables") or [], has_line_chart=has_line_chart)

    raw_kpis = [
        {**k, "label": re.sub(r"(?i)^gross merchandise revenue", "Gross Revenue", str(k.get("label") or "").strip())}
        for k in (report_dict.get("kpis") or []) if isinstance(k, dict)
    ]
    if len(raw_kpis) <= 1 and tables and (first_rows := tables[0].get("rows")):
        r0 = first_rows[0]
        for col_k, kpi_lbl, kpi_u in (("gross_revenue", f"Gross Revenue ({period})", "USD"), ("total_orders", "Total Orders", "orders"), ("aov", "Average Order Value", "USD"), ("purchasing_customers", "Purchasing Customers", "customers")):
            if col_k in r0 and isinstance(r0[col_k], (int, float)) and not any(kpi_lbl.split(" (")[0].lower() in str(ek.get("label", "")).lower() for ek in raw_kpis):
                raw_kpis.append({"label": kpi_lbl, "value": r0[col_k], "unit": kpi_u})

    raw_sections = list(biz.get("sections") or [])
    sections: list[dict[str, Any]] = []
    extracted_scope_notes: list[str] = []
    for sec in raw_sections:
        if not isinstance(sec, dict) or not (body := str(sec.get("body") or "").strip()):
            continue
        heading = re.sub(r"^\d+[\.\)]\s*", "", _strip_markdown(str(sec.get("heading") or "")).strip()).strip()
        if len(raw_sections) >= 3 and any(w in heading.lower() for w in ("methodology", "accounting rules", "data scope")):
            extracted_scope_notes.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", _strip_markdown(body)) if len(s.strip()) > 20)
            continue
        sections.append({"heading": heading or f"Section {len(sections) + 1}", "body": body})

    clean_findings = [
        {
            "title": _truncate_words(re.sub(r"(?i)^Autonomous Python Analysis:\s*", "", _strip_markdown(str(f.get("title") or "")).strip()).strip(), 85),
            "detail": str(f.get("detail") or "").strip(),
            "status": str(f.get("status") or "verified").lower(),
        }
        for f in (report_dict.get("findings") or [])
        if isinstance(f, dict) and not _is_raw_dump_text(str(f.get("detail") or ""))
        and not (str(f.get("title") or "").lower().startswith("total gross") and len(_strip_markdown(str(f.get("detail") or ""))) < 65 and (sections or len(report_dict.get("findings") or []) > 1))
    ]

    has_rca_section = any(any(w in s["heading"].lower() for w in ("root-cause", "root cause", "hypothesis")) for s in sections)
    clean_hyps = [] if has_rca_section else [
        {
            "description": _strip_markdown(str(h.get("description") or "")).strip(),
            "dimension": str(h.get("dimension") or "Business Driver"),
            "status": str(h.get("status") or "verified").lower(),
            "measured_effect": float(h.get("measured_effect") or 0.0),
            "limitations": _strip_markdown(str(h.get("limitations") or "")).strip(),
        }
        for h in (report_dict.get("hypotheses") or [])
        if isinstance(h, dict) and _strip_markdown(str(h.get("description") or "")).strip()
    ]

    recs = [_strip_markdown(str(r)).strip() for r in (biz.get("recommendations") or report_dict.get("recommendations") or []) if str(r).strip()][:4]
    raw_lims = list(biz.get("limitations") or []) + list(report_dict.get("limitations") or []) + extracted_scope_notes

    return {
        "title": title, "summary": summary, "period_str": period_str,
        "kpis": raw_kpis[:6], "charts": charts, "sections": sections,
        "findings": clean_findings if not sections else [], "hypotheses": clean_hyps,
        "tables": tables, "recommendations": recs,
        "limitations": _clean_report_limitations(raw_lims, [s["body"] for s in sections]),
    }


class ReportGenerationAgent:
    def __init__(self, model_selector: ModelSelector | None = None):
        self.selector = model_selector or ModelSelector()

    def generate_business_report(
        self,
        plan: AnalysisPlan,
        base_report: Report,
        evidence_records: list[EvidenceRecord],
    ) -> tuple[BusinessReport, ModelUsageItem | None]:
        evidence_index = [
            {
                "id": er.id, "task_id": er.task_id, "kind": er.kind, "description": er.description,
                "source_query_ids": er.source_query_ids, "dataset_ids": er.dataset_ids, "calculation": er.calculation,
            }
            for er in evidence_records
        ]
        usage_item: ModelUsageItem | None = None
        draft: ReportDraft | None = None
        clean_base_lims = _clean_report_limitations(base_report.limitations)

        try:
            parsed, usage = self.selector.parse_structured(
                task_type=TaskType.REPORT_WRITING, response_model=ReportDraft,
                system_instructions=REPORT_AGENT_SYSTEM_PROMPT,
                input_payload={
                    "question": plan.question, "objective": plan.objective, "period": base_report.period,
                    "comparison": base_report.comparison, "kpis": base_report.kpis, "summary": base_report.summary,
                    "findings": [f.model_dump() for f in base_report.findings if not _is_raw_dump_text(f.detail)],
                    "hypotheses": [h.model_dump() for h in base_report.hypotheses],
                    "python_analyses": [
                        {
                            "task_id": p.get("task_id"), "tool": p.get("tool"),
                            "methodology": p.get("methodology") or (p.get("calculation") or {}).get("methodology", ""),
                            "assumptions_checked": (p.get("assumptions_checked") or [])[:4],
                            "verification_status": p.get("verification_status", ""),
                            "summary": {str(k): v for k, v in ((p.get("calculation") or {}).get("summary") or {}).items() if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 160)},
                        }
                        for p in base_report.python_analyses[:4]
                    ],
                    "limitations": clean_base_lims[:3], "recommendations": base_report.recommendations[:4],
                    "evidence_index": evidence_index[:8],
                },
            )
            usage_item = ModelUsageItem(
                agent_role="Report Generation Agent", task_type=usage.task_type, model=usage.model,
                reasoning_effort=usage.reasoning_effort, duration_ms=usage.duration_ms,
                tokens=usage.input_tokens + usage.output_tokens, cost_usd=usage.cost_usd,
            )
            if isinstance(parsed, ReportDraft):
                draft = parsed
        except Exception as exc:
            logger.warning("ReportGenerationAgent fell back to deterministic report assembly: %s", exc)

        if draft is None:
            sections: list[ReportSection] = []
            if valid_findings := [f for f in base_report.findings if not _is_raw_dump_text(f.detail)]:
                sections.append(ReportSection(
                    heading="Key Analytical Findings",
                    body="\n".join(f"- **{f.title}**: {f.detail}" for f in valid_findings),
                    evidence_ids=[qid for f in valid_findings for qid in f.evidence_ids][:8],
                ))
            if base_report.hypotheses:
                sections.append(ReportSection(
                    heading="Driver & Root-Cause Evaluation",
                    body="\n".join(f"- **{h.description}** ({_format_label(h.status)}, Measured Impact: ${h.measured_effect:,.2f}): {h.limitations}" for h in base_report.hypotheses),
                    evidence_ids=[qid for h in base_report.hypotheses for qid in h.evidence_query_ids][:8],
                ))
            if not sections:
                sections.append(ReportSection(heading="Analysis Results", body=base_report.summary, evidence_ids=[e.query_id for e in base_report.evidence][:4]))
            kpi_str = ", ".join(f"{k['label']}: {_format_metric_value(k.get('value'), str(k.get('unit', '')))}" for k in base_report.kpis) or "See key metrics summary."
            return BusinessReport(
                title=base_report.title, executive_summary=base_report.summary,
                business_objective=plan.objective or plan.question, kpi_overview=kpi_str,
                sections=sections, recommendations=list(base_report.recommendations),
                limitations=clean_base_lims, chart_ids=[f"chart_{i+1}" for i in range(len(base_report.charts))],
                evidence_index=evidence_index, generated_at=datetime.now(timezone.utc).isoformat(),
                model=usage_item.model if usage_item else "deterministic", formats=["html", "pdf"],
            ), usage_item

        return BusinessReport(
            title=draft.title or base_report.title,
            executive_summary=draft.executive_summary or base_report.summary,
            business_objective=draft.business_objective or plan.objective,
            kpi_overview=draft.kpi_overview, sections=list(draft.sections),
            recommendations=list(draft.recommendations or base_report.recommendations),
            limitations=_clean_report_limitations(list(draft.limitations or clean_base_lims)),
            chart_ids=[f"chart_{i+1}" for i in range(len(base_report.charts))],
            evidence_index=evidence_index, generated_at=datetime.now(timezone.utc).isoformat(),
            model=usage_item.model if usage_item else "", formats=["html", "pdf"],
        ), usage_item


def _select_tick_indices(n: int, max_ticks: int = 6) -> list[int]:
    if n <= max_ticks:
        return list(range(n))
    k = max(2, max_ticks)
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _render_svg_chart(chart: dict[str, Any]) -> str:
    title = html.escape(_strip_markdown(str(chart.get("title") or "Chart")))
    kind = str(chart.get("kind") or "bar").lower()
    max_pts = 36 if kind == "line" else 15
    labels = [str(x) for x in (chart.get("labels") or [])][:max_pts]
    values = [float(v or 0.0) for v in (chart.get("values") or [])[:max_pts]]
    unit = str(chart.get("unit") or "USD")
    if not labels or not values or len(labels) != len(values):
        return ""

    if kind == "line" and len(values) >= 2:
        w, h, pad_l, pad_r, pad_t, pad_b = 820, 260, 82, 34, 28, 46
        plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
        raw_max = max(values)
        min_v, max_v = min(0.0, min(values)), (raw_max * 1.08 if raw_max > 0 else max(0.0, raw_max) + 1.0)
        span = (max_v - min_v) if abs(max_v - min_v) > 1e-6 else 1.0
        pts = [(pad_l + (idx / max(1, len(values) - 1)) * plot_w, pad_t + plot_h - ((val - min_v) / span) * plot_h) for idx, val in enumerate(values)]
        poly_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        area_pts = f"{pts[0][0]:.1f},{pad_t + plot_h:.1f} {poly_pts} {pts[-1][0]:.1f},{pad_t + plot_h:.1f}"
        grid_svg = "".join(
            f'<line x1="{pad_l}" y1="{pad_t + (g / 4) * plot_h:.1f}" x2="{w - pad_r}" y2="{pad_t + (g / 4) * plot_h:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>'
            f'<text x="{pad_l - 8}" y="{pad_t + (g / 4) * plot_h + 4:.1f}" font-family="Inter, sans-serif" font-size="10" fill="#64748b" text-anchor="end">{html.escape(_format_metric_value(max_v - (g / 4) * span, unit))}</text>'
            for g in range(5)
        )
        tick_set = set(_select_tick_indices(len(labels), max_ticks=7))
        dots_svg = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#059669" stroke="#fff" stroke-width="1.5"/>'
            + (f'<text x="{x:.1f}" y="{h - 14}" font-family="Inter, sans-serif" font-size="10" fill="#475569" text-anchor="middle">{html.escape(lbl[:12])}</text>' if idx in tick_set else "")
            for idx, ((x, y), lbl) in enumerate(zip(pts, labels))
        )
        return f'<div class="chart-card"><div class="chart-card-header"><strong>{title}</strong><span>{html.escape(unit)}</span></div><svg viewBox="0 0 {w} {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg">{grid_svg}<polygon points="{area_pts}" fill="rgba(5, 150, 105, 0.10)"/><polyline fill="none" stroke="#059669" stroke-width="2.5" points="{poly_pts}"/>{dots_svg}</svg></div>'

    row_h, pad_t, pad_b, pad_l, pad_r, w = 30, 18, 22, 195, 95, 820
    h, plot_w = pad_t + pad_b + len(labels) * row_h, w - pad_l - pad_r
    min_v, max_v = min(0.0, min(values)), max(0.0, max(values))
    raw_span = (max_v - min_v) if abs(max_v - min_v) > 1e-6 else 1.0
    eff_min = min_v - (raw_span * 0.25 if min_v < 0 else 0.0)
    eff_max = max_v + (raw_span * 0.22 if max_v > 0 else raw_span * 0.1)
    total_span = (eff_max - eff_min) if abs(eff_max - eff_min) > 1e-6 else 1.0
    zero_x = pad_l + (abs(eff_min) / total_span) * plot_w
    bars_svg = f'<line x1="{zero_x:.1f}" y1="{pad_t - 6}" x2="{zero_x:.1f}" y2="{h - pad_b + 4}" stroke="#94a3b8" stroke-width="1.2"/>'
    for idx, (lbl, val) in enumerate(zip(labels, values)):
        y, bar_w = pad_t + idx * row_h, max(2.0, (abs(val) / total_span) * plot_w)
        bx, color, val_x, anchor = (zero_x, "#059669", zero_x + bar_w + 6, "start") if val >= 0 else (zero_x - bar_w, "#dc2626", zero_x - bar_w - 6, "end")
        bars_svg += (
            f'<text x="{pad_l - 12}" y="{y + 15}" font-family="Inter, sans-serif" font-size="11" fill="#334155" text-anchor="end">{html.escape(_truncate_words(lbl, 27))}</text>'
            f'<rect x="{bx:.1f}" y="{y + 4}" width="{bar_w:.1f}" height="18" rx="4" fill="{color}"/>'
            f'<text x="{val_x:.1f}" y="{y + 16}" font-family="Inter, sans-serif" font-size="10.5" font-weight="600" fill="{color}" text-anchor="{anchor}">{html.escape(_format_metric_value(val, unit))}</text>'
        )
    return f'<div class="chart-card"><div class="chart-card-header"><strong>{title}</strong><span>{html.escape(unit)}</span></div><svg viewBox="0 0 {w} {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg">{bars_svg}</svg></div>'


def render_report_html(report_dict: dict[str, Any], run_id: str = "") -> str:
    view = _prepare_report_view_data(report_dict)
    title = html.escape(view["title"])
    summary_html = _markdown_to_html(view["summary"])
    period_str = html.escape(view["period_str"])
    generated_date = datetime.now(timezone.utc).strftime("%B %d, %Y")

    kpis_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{html.escape(str(k.get("label", "")))}</div>'
        f'<div class="kpi-val {"negative" if isinstance(k.get("value"), (int, float)) and k["value"] < 0 else ("positive" if isinstance(k.get("value"), (int, float)) and k.get("unit") == "%" and k["value"] > 0 else "")}">'
        f'{html.escape(_format_metric_value(k.get("value"), str(k.get("unit", "")), str(k.get("label", ""))))}</div></div>'
        for k in view["kpis"]
    )
    charts_html = "".join(_render_svg_chart(ch) for ch in view["charts"])

    sections_html = ""
    if view["sections"]:
        sec_items = "".join(f'<div class="analysis-section"><h4>{idx}. {html.escape(sec["heading"])}</h4>{_markdown_to_html(sec["body"])}</div>' for idx, sec in enumerate(view["sections"], start=1))
        sections_html = f'<section class="report-block"><div class="block-header"><h3>Analysis &amp; Findings</h3></div>{sec_items}</section>'
    elif view["findings"]:
        f_items = "".join(f'<div class="finding-card"><strong>{html.escape(f["title"])}</strong><div class="finding-detail">{_markdown_to_html(f["detail"])}</div></div>' for f in view["findings"])
        sections_html = f'<section class="report-block"><div class="block-header"><h3>Key Findings</h3></div>{f_items}</section>'

    hyps = view["hypotheses"]
    max_eff = max((abs(float(h.get("measured_effect") or 0.0)) for h in hyps), default=1.0) or 1.0
    hyp_html = ""
    for h in hyps:
        hs, heff, hlim = h["status"], h["measured_effect"], html.escape(h["limitations"])
        bar_pct = min(100, max(6, int(round((abs(heff) / max_eff) * 100)))) if heff != 0 else 15
        bar_col = "#dc2626" if heff < 0 else "#059669"
        hyp_html += (
            f'<div class="finding-card"><div class="finding-top"><strong>{html.escape(h["description"])}</strong>'
            f'<span class="pill {"pill-good" if hs in ("verified", "supported") else "pill-warn"}">{html.escape(_format_label(hs))}</span></div>'
            f'<div class="driver-meta"><span class="meta-tag">Dimension: {html.escape(_format_label(h["dimension"]))}</span>'
            + (f'<span class="meta-tag">Measured Impact: <strong>{html.escape(_format_metric_value(heff, "USD"))}</strong></span>' if heff != 0 else "")
            + "</div>"
            + (f'<div class="impact-track"><div class="impact-fill" style="width:{bar_pct}%;background:{bar_col};"></div></div>' if heff != 0 else "")
            + (f'<p class="driver-note">{hlim}</p>' if hlim else "")
            + "</div>"
        )

    tables_html = ""
    for tbl in view["tables"]:
        rows = tbl["rows"]
        cols = list(rows[0].keys())
        is_num_col = {c: any(isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool) for r in rows) for c in cols}
        thead = "".join(f'<th class="{"num" if is_num_col[c] else ""}">{html.escape(_format_label(c))}</th>' for c in cols)
        tbody = ""
        for r in rows:
            cells = ""
            for c in cols:
                raw_v = r.get(c)
                align = "num" if isinstance(raw_v, (int, float)) and not isinstance(raw_v, bool) else ""
                v_cls = ("neg-cell" if raw_v < 0 else ("pos-cell" if raw_v > 0 else "")) if (isinstance(raw_v, (int, float)) and any(k in c.lower() for k in ("change", "variance", "pct", "delta"))) else ""
                cells += f'<td class="{align} {v_cls}">{html.escape(_format_cell_value(raw_v, c, r))}</td>'
            tbody += f"<tr>{cells}</tr>"
        tables_html += f'<section class="report-block"><div class="block-header"><h3>{html.escape(tbl["title"])}</h3></div><div class="table-wrap"><table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div></section>'

    recs_html = "".join(f'<div class="rec-item"><span class="rec-num">{i}</span><p>{html.escape(r)}</p></div>' for i, r in enumerate(view["recommendations"], start=1))
    lims_html = "".join(f"<li>{html.escape(l)}</li>" for l in view["limitations"])

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/><title>{title}</title>
<style>
  :root {{ --green-100:#d1fae5; --green-600:#059669; --green-700:#047857; --green-800:#065f46; --slate-900:#0f172a; --slate-700:#334155; --slate-500:#64748b; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:'Inter',-apple-system,sans-serif; color:var(--slate-900); background:#f8fafc; margin:0; padding:32px 16px; line-height:1.6; }}
  .page {{ max-width:920px; margin:0 auto; background:#fff; border-radius:12px; box-shadow:0 4px 20px rgba(15,23,42,0.05); overflow:hidden; border:1px solid #e2e8f0; }}
  .toolbar {{ display:flex; justify-content:space-between; align-items:center; padding:12px 36px; background:#f8fafc; border-bottom:1px solid #e2e8f0; font-size:12px; color:var(--slate-500); }}
  .print-btn {{ background:var(--green-600); color:#fff; border:0; border-radius:6px; padding:7px 14px; font-size:12px; font-weight:600; cursor:pointer; }}
  .hero {{ padding:32px 40px 24px; border-bottom:1px solid #e2e8f0; border-top:4px solid var(--green-600); }}
  .hero-overline {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:var(--green-700); margin-bottom:8px; }}
  .hero h1 {{ font-size:26px; margin:0 0 8px; color:var(--slate-900); line-height:1.3; }}
  .hero-meta {{ font-size:12.5px; color:var(--slate-500); }}
  .body-wrap {{ padding:28px 40px 36px; }}
  .summary-callout {{ background:#f8faf9; border:1px solid #d1fae5; border-left:4px solid var(--green-600); border-radius:8px; padding:16px 20px; margin-bottom:24px; }}
  .summary-callout h3 {{ margin:0 0 6px; font-size:11.5px; text-transform:uppercase; color:var(--green-800); }}
  .summary-callout p {{ margin:4px 0 0; font-size:14px; color:#1e293b; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:12px; margin-bottom:26px; }}
  .kpi {{ border:1px solid #e2e8f0; border-radius:8px; padding:14px 16px; }}
  .kpi-label {{ font-size:12px; color:var(--slate-500); font-weight:500; }}
  .kpi-val {{ font-size:20px; font-weight:700; margin-top:4px; color:var(--slate-900); }}
  .kpi-val.positive {{ color:var(--green-700); }} .kpi-val.negative {{ color:#dc2626; }}
  .chart-card {{ border:1px solid #e2e8f0; border-radius:10px; padding:16px 18px; margin-bottom:18px; }}
  .chart-card-header {{ display:flex; justify-content:space-between; align-items:center; font-size:13.5px; font-weight:600; margin-bottom:8px; }}
  .chart-card-header span {{ font-size:11px; color:var(--slate-500); background:#f1f5f9; padding:2px 8px; border-radius:4px; }}
  .report-block {{ margin-top:26px; }}
  .block-header {{ border-bottom:1px solid #e2e8f0; padding-bottom:8px; margin-bottom:14px; }}
  .block-header h3 {{ margin:0; font-size:15.5px; font-weight:700; }}
  .analysis-section {{ margin-bottom:16px; }}
  .analysis-section h4 {{ margin:0 0 6px; font-size:14px; font-weight:600; color:var(--green-800); }}
  .analysis-section p, .finding-detail p {{ margin:4px 0 0; font-size:13.5px; color:var(--slate-700); }}
  .rich-list {{ margin:6px 0; padding-left:20px; color:var(--slate-700); font-size:13.5px; }}
  .finding-card {{ border:1px solid #e2e8f0; border-left:3px solid var(--green-600); border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
  .finding-top {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }}
  .pill {{ font-size:11px; font-weight:600; padding:2px 8px; border-radius:999px; }}
  .pill-good {{ background:var(--green-100); color:var(--green-800); }} .pill-warn {{ background:#fef3c7; color:#b45309; }}
  .driver-meta {{ display:flex; gap:8px; flex-wrap:wrap; margin:6px 0; }}
  .meta-tag {{ font-size:11.5px; background:#f8fafc; border:1px solid #e2e8f0; padding:2px 8px; border-radius:4px; color:var(--slate-700); }}
  .impact-track {{ height:6px; background:#f1f5f9; border-radius:999px; overflow:hidden; margin:6px 0 4px; }}
  .impact-fill {{ height:100%; border-radius:999px; }}
  .driver-note {{ margin:4px 0 0; font-size:12px; color:var(--slate-500); }}
  .table-wrap {{ overflow-x:auto; border:1px solid #e2e8f0; border-radius:8px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  th {{ background:#f8fafc; color:var(--slate-700); font-weight:600; text-align:left; padding:9px 14px; border-bottom:1px solid #e2e8f0; }}
  th.num, td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td {{ padding:8px 14px; border-bottom:1px solid #f1f5f9; color:var(--slate-700); }}
  tr:last-child td {{ border-bottom:0; }}
  td.pos-cell {{ color:var(--green-700); font-weight:600; }} td.neg-cell {{ color:#dc2626; font-weight:600; }}
  .rec-item {{ display:flex; align-items:flex-start; gap:10px; padding:10px 14px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; margin-bottom:8px; }}
  .rec-num {{ width:22px; height:22px; border-radius:50%; background:var(--green-600); color:#fff; font-size:11.5px; font-weight:700; display:grid; place-items:center; flex-shrink:0; }}
  .rec-item p {{ margin:0; font-size:13.5px; }}
  .notes-box {{ border-top:1px solid #e2e8f0; padding-top:14px; margin-top:26px; color:var(--slate-500); }}
  .notes-box h4 {{ margin:0 0 6px; font-size:12px; text-transform:uppercase; color:var(--slate-700); }}
  .notes-box ul {{ margin:0; padding-left:18px; font-size:12px; }}
  @media print {{ body {{ background:#fff; padding:0; }} .page {{ box-shadow:none; border:0; max-width:100%; }} .toolbar {{ display:none; }} }}
</style></head>
<body>
  <div class="page">
    <div class="toolbar"><span>Analytical Report &middot; {html.escape(generated_date)}</span><button class="print-btn" onclick="window.print()">Print / Save PDF</button></div>
    <header class="hero">
      <div class="hero-overline">Executive Business Report</div>
      <h1>{title}</h1>
      <div class="hero-meta">Analysis Period: {period_str} &middot; Date: {html.escape(generated_date)}</div>
    </header>
    <div class="body-wrap">
      {f'<div class="summary-callout"><h3>Executive Summary</h3>{summary_html}</div>' if summary_html else ''}
      {f'<div class="kpi-grid">{kpis_html}</div>' if kpis_html else ''}
      {sections_html}
      {f'<section class="report-block"><div class="block-header"><h3>Performance Charts</h3></div>{charts_html}</section>' if charts_html else ''}
      {f'<section class="report-block"><div class="block-header"><h3>Driver &amp; Hypothesis Evaluation</h3></div>{hyp_html}</section>' if hyp_html else ''}
      {tables_html}
      {f'<section class="report-block"><div class="block-header"><h3>Recommendations</h3></div>{recs_html}</section>' if recs_html else ''}
      {f'<div class="notes-box"><h4>Scope &amp; Methodology Notes</h4><ul>{lims_html}</ul></div>' if lims_html else ''}
    </div>
  </div>
</body></html>"""


class _ExecutivePDF(FPDF):
    def __init__(self, report_title: str, period_str: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_title = _ascii_safe(report_title)
        self.period_str = _ascii_safe(period_str)
        self.generated_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def header(self):
        if self.page_no() == 1:
            return
        self.set_y(10)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(71, 85, 105)
        self.cell(115, 5, _truncate_words(self.report_title, 62), align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, _truncate_words(self.period_str, 36), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(226, 232, 240)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(1.5)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 116, 139)
        self.cell(120, 5, f"{_truncate_words(self.period_str, 48)}  |  {self.generated_str}", align="L")
        self.cell(0, 5, f"Page {self.page_no()}", align="R")


def _render_matplotlib_chart_png(chart: dict[str, Any]) -> io.BytesIO | None:
    try:
        kind = str(chart.get("kind") or "bar").lower()
        max_pts = 36 if kind == "line" else 15
        labels = [str(x) for x in (chart.get("labels") or [])][:max_pts]
        values = [float(v or 0.0) for v in (chart.get("values") or [])][:max_pts]
        if not labels or not values or len(labels) != len(values):
            return None
        title = _ascii_safe(_strip_markdown(str(chart.get("title") or "Performance Chart")))
        unit = str(chart.get("unit") or "USD")

        fig_h = 2.75 if kind == "line" else max(2.0, min(4.2, 0.28 * len(labels) + 0.85))
        fig, ax = plt.subplots(figsize=(7.2, fig_h), dpi=200)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        if kind == "line" and len(values) >= 2:
            x_idx = list(range(len(labels)))
            ax.plot(x_idx, values, color="#059669", linewidth=2.2, marker="o", markersize=4.2, markerfacecolor="#059669")
            ax.fill_between(x_idx, values, min(0.0, min(values)), color="#10b981", alpha=0.10)
            ticks = _select_tick_indices(len(labels), max_ticks=7)
            ax.set_xticks(ticks)
            ax.set_xticklabels([_ascii_safe(labels[i][:12]) for i in ticks], fontsize=8, color="#475569")
            ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
            ax.xaxis.grid(False)
            min_y, max_y = min(0.0, min(values)), max(0.0, max(values))
            ax.set_ylim(min_y * 1.12 if min_y < 0 else 0.0, max_y * 1.15 if max_y > 0 else 1.0)
            if unit == "USD":
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        else:
            y_pos = list(range(len(labels)))
            rev_labels = [_ascii_safe(_truncate_words(lbl, 28)) for lbl in reversed(labels)]
            rev_values = list(reversed(values))
            bars = ax.barh(y_pos, rev_values, color=["#dc2626" if v < 0 else "#059669" for v in rev_values], height=0.58, edgecolor="none")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(rev_labels, fontsize=8, color="#334155")
            ax.axvline(0, color="#94a3b8", linewidth=0.9)
            ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
            ax.yaxis.grid(False)
            if unit == "USD":
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            min_v, max_v = min(0.0, min(rev_values)), max(0.0, max(rev_values))
            total_span = (max_v - min_v) if abs(max_v - min_v) > 1e-6 else 1.0
            ax.set_xlim(min_v - (total_span * 0.30 if min_v < 0 else 0.0), max_v + (total_span * 0.25 if max_v > 0 else total_span * 0.10))
            for bar, v in zip(bars, rev_values):
                offset = total_span * 0.016 if v >= 0 else -total_span * 0.016
                ax.text(v + offset, bar.get_y() + bar.get_height() / 2.0, _ascii_safe(_format_metric_value(v, unit)), va="center", ha="left" if v >= 0 else "right", fontsize=7.5, fontweight="bold", color="#dc2626" if v < 0 else "#065f46")

        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color="#0f172a", pad=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


def _draw_pdf_section_heading(pdf: FPDF, title: str, usable_w: float, min_content_h: float = 22.0):
    if pdf.get_y() + 12.0 + min_content_h > 268:
        pdf.add_page()
    pdf.ln(2.5)
    y = pdf.get_y()
    pdf.set_fill_color(5, 150, 105)
    pdf.rect(pdf.l_margin, y + 1.2, 2.2, 4.8, style="F")
    pdf.set_x(pdf.l_margin + 4.8)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(usable_w - 4.8, 6.5, _ascii_safe(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(226, 232, 240)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + usable_w, pdf.get_y())
    pdf.ln(2.5)


def _compute_pdf_col_widths(pdf: FPDF, cols: list[str], rows: list[dict[str, Any]], usable_w: float) -> list[float]:
    if not cols:
        return []
    pdf.set_font("Helvetica", "B", 8)
    weights = [
        min(62.0, max(18.0, max([pdf.get_string_width(_ascii_safe(_format_label(c))) + 6.0] + [pdf.get_string_width(_ascii_safe(_format_cell_value(r.get(c), c, r))) + 6.0 for r in rows[:20]])))
        for c in cols
    ]
    total_w = sum(weights) or 1.0
    return [(w / total_w) * usable_w for w in weights]


def render_report_pdf(report_dict: dict[str, Any], run_id: str = "") -> bytes:
    view = _prepare_report_view_data(report_dict)
    title, period_str = view["title"], view["period_str"]

    pdf = _ExecutivePDF(report_title=title, period_str=period_str)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_fill_color(5, 150, 105)
    pdf.rect(pdf.l_margin, 12.0, usable_w, 1.8, style="F")
    pdf.set_y(16.0)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(4, 120, 87)
    pdf.cell(usable_w, 4.5, "EXECUTIVE ANALYTICAL REPORT", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(usable_w, 6.8, _ascii_safe(title), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(usable_w, 4.5, _ascii_safe(f"Analysis Period: {period_str}   |   Date: {pdf.generated_str}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + usable_w, pdf.get_y())
    pdf.ln(4)

    if summary_text := _strip_markdown(view["summary"]):
        sum_y = pdf.get_y()
        pdf.set_font("Helvetica", "", 9.5)
        wrapped_lines = pdf.multi_cell(usable_w - 11, 4.8, _ascii_safe(summary_text), align="L", dry_run=True, output="LINES")
        box_h = max(1, len(wrapped_lines)) * 4.8 + 11.5
        pdf.set_fill_color(244, 250, 247)
        pdf.set_draw_color(167, 243, 208)
        pdf.rect(pdf.l_margin, sum_y, usable_w, box_h, style="DF")
        pdf.set_fill_color(5, 150, 105)
        pdf.rect(pdf.l_margin, sum_y, 2.2, box_h, style="F")
        pdf.set_xy(pdf.l_margin + 5.5, sum_y + 2.8)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(6, 95, 70)
        pdf.cell(usable_w - 10, 4.2, "EXECUTIVE SUMMARY", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(pdf.l_margin + 5.5)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.multi_cell(usable_w - 11, 4.8, _ascii_safe(summary_text), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.set_y(sum_y + box_h + 4.5)

    if kpis := view["kpis"]:
        cols = 4 if len(kpis) == 4 else min(3, max(2, len(kpis)))
        gap, card_h = 3.5, 17.0
        card_w = (usable_w - (cols - 1) * gap) / cols
        for row_start in range(0, len(kpis), cols):
            if pdf.get_y() + card_h > 265:
                pdf.add_page()
            row_y = pdf.get_y()
            for c_idx, k in enumerate(kpis[row_start : row_start + cols]):
                cx = pdf.l_margin + c_idx * (card_w + gap)
                pdf.set_fill_color(248, 250, 252)
                pdf.set_draw_color(226, 232, 240)
                pdf.rect(cx, row_y, card_w, card_h, style="DF")
                raw_lbl = _ascii_safe(str(k.get("label", "")))
                if cols == 4:
                    raw_lbl = re.sub(r"\s*\([^)]*\)", "", raw_lbl).strip() or raw_lbl
                lbl = _truncate_words(raw_lbl, 24 if cols == 4 else 34)
                val, unit = k.get("value"), str(k.get("unit", ""))
                pdf.set_xy(cx + 3.0, row_y + 2.4)
                pdf.set_font("Helvetica", "", 7.6)
                pdf.set_text_color(100, 116, 139)
                pdf.cell(card_w - 6, 4.0, lbl)
                pdf.set_xy(cx + 3.0, row_y + 7.8)
                pdf.set_font("Helvetica", "B", 11.0 if cols == 4 else 11.5)
                pdf.set_text_color(*(220, 38, 38) if (isinstance(val, (int, float)) and val < 0) else ((4, 120, 87) if (isinstance(val, (int, float)) and unit == "%" and val > 0) else (15, 23, 42)))
                pdf.cell(card_w - 6, 6.0, _ascii_safe(_format_metric_value(val, unit, lbl))[:22])
            pdf.set_y(row_y + card_h + 3.5)

    if sections := view["sections"]:
        _draw_pdf_section_heading(pdf, "Analysis & Findings", usable_w, min_content_h=24.0)
        for idx, sec in enumerate(sections, start=1):
            if pdf.get_y() > 250:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(6, 95, 70)
            pdf.set_x(pdf.l_margin)
            pdf.cell(usable_w, 5.5, _ascii_safe(f"{idx}. {sec['heading']}"), new_x="LMARGIN", new_y="NEXT")
            for line in str(sec.get("body", "")).splitlines():
                if clean_line := _strip_markdown(line).strip():
                    if clean_line.startswith(("- ", "* ")):
                        clean_line = f"  - {clean_line[2:].strip()}"
                    pdf.set_font("Helvetica", "", 9.2)
                    pdf.set_text_color(51, 65, 85)
                    pdf.set_x(pdf.l_margin + 1.5)
                    pdf.multi_cell(usable_w - 1.5, 4.7, _ascii_safe(clean_line), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2.0)
    elif view["findings"]:
        _draw_pdf_section_heading(pdf, "Key Findings", usable_w, min_content_h=20.0)
        for f in view["findings"]:
            if pdf.get_y() > 252:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(15, 23, 42)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(usable_w, 5.0, _ascii_safe(f["title"]), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9.0)
            pdf.set_text_color(71, 85, 105)
            pdf.set_x(pdf.l_margin + 2)
            pdf.multi_cell(usable_w - 2, 4.6, _ascii_safe(_strip_markdown(f["detail"])), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    if charts := view["charts"]:
        _draw_pdf_section_heading(pdf, "Performance Charts", usable_w, min_content_h=60.0)
        for ch in charts:
            if (png_buf := _render_matplotlib_chart_png(ch)) is not None:
                est_h = 68.0 if str(ch.get("kind") or "bar").lower() == "line" else 56.0
                if pdf.get_y() + est_h > 268:
                    pdf.add_page()
                pdf.image(png_buf, x=pdf.l_margin, w=usable_w)
                pdf.ln(3)

    if hyps := view["hypotheses"]:
        _draw_pdf_section_heading(pdf, "Driver & Hypothesis Evaluation", usable_w, min_content_h=20.0)
        for h in hyps:
            if pdf.get_y() > 248:
                pdf.add_page()
            heff = h["measured_effect"]
            pdf.set_font("Helvetica", "B", 9.2)
            pdf.set_text_color(15, 23, 42)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(usable_w, 4.8, _ascii_safe(f"{h['description']} ({_format_label(h['status'])})"), align="L", new_x="LMARGIN", new_y="NEXT")
            meta_line = f"Dimension: {_format_label(h['dimension'])}" + (f"   |   Measured Impact: {_format_metric_value(heff, 'USD')}" if heff != 0 else "")
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(4, 120, 87)
            pdf.set_x(pdf.l_margin + 2)
            pdf.cell(usable_w - 2, 4.5, _ascii_safe(meta_line), new_x="LMARGIN", new_y="NEXT")
            if h["limitations"]:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(100, 116, 139)
                pdf.set_x(pdf.l_margin + 2)
                pdf.multi_cell(usable_w - 2, 4.2, _ascii_safe(h["limitations"]), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.8)

    for tbl in view["tables"]:
        rows = tbl["rows"][:20]
        if not rows:
            continue
        cols = list(rows[0].keys())[:6]
        _draw_pdf_section_heading(pdf, tbl["title"], usable_w, min_content_h=14.0 + min(len(rows), 4) * 5.8)
        col_widths = _compute_pdf_col_widths(pdf, cols, rows, usable_w)
        is_num_col = [any(isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool) for r in rows) for c in cols]

        def _draw_table_header():
            pdf.set_fill_color(241, 245, 249)
            pdf.set_draw_color(203, 213, 225)
            pdf.set_text_color(30, 41, 59)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_x(pdf.l_margin)
            for c_idx, c in enumerate(cols):
                pdf.cell(col_widths[c_idx], 6.2, _truncate_words(_ascii_safe(_format_label(c)), 26), border="B", fill=True, align="R" if is_num_col[c_idx] else "L")
            pdf.ln(6.2)

        _draw_table_header()
        for r_idx, r in enumerate(rows):
            if pdf.get_y() + 5.8 > 268:
                pdf.add_page()
                _draw_table_header()
            pdf.set_fill_color(*(255, 255, 255) if r_idx % 2 == 0 else (248, 250, 252))
            pdf.set_font("Helvetica", "", 8)
            pdf.set_x(pdf.l_margin)
            for c_idx, c in enumerate(cols):
                raw_v = r.get(c)
                is_var = isinstance(raw_v, (int, float)) and any(k in c.lower() for k in ("change", "variance", "pct", "delta"))
                pdf.set_text_color(*(220, 38, 38) if (is_var and raw_v < 0) else ((4, 120, 87) if (is_var and raw_v > 0) else (51, 65, 85)))
                pdf.cell(col_widths[c_idx], 5.6, _truncate_words(_ascii_safe(_format_cell_value(raw_v, c, r)), 28), border=0, fill=True, align="R" if is_num_col[c_idx] else "L")
            pdf.ln(5.6)
        pdf.ln(2.5)

    if recs := view["recommendations"]:
        _draw_pdf_section_heading(pdf, "Recommendations", usable_w, min_content_h=18.0)
        for idx, rec in enumerate(recs, start=1):
            if pdf.get_y() > 258:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 9.2)
            pdf.set_text_color(5, 150, 105)
            pdf.set_x(pdf.l_margin)
            pdf.cell(6, 5.0, f"{idx}.")
            pdf.set_font("Helvetica", "", 9.2)
            pdf.set_text_color(15, 23, 42)
            pdf.multi_cell(usable_w - 6, 5.0, _ascii_safe(rec), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

    if lims := view["limitations"]:
        _draw_pdf_section_heading(pdf, "Scope & Methodology Notes", usable_w, min_content_h=16.0)
        for lim in lims:
            if pdf.get_y() > 260:
                pdf.add_page()
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(100, 116, 139)
            pdf.set_x(pdf.l_margin + 1.5)
            pdf.multi_cell(usable_w - 1.5, 4.5, _ascii_safe(f"- {lim}"), align="L", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())

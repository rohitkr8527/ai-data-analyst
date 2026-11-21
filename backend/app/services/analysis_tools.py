"""Approved Registry of Deterministic Python Analysis Functions."""
from __future__ import annotations
import math
from typing import Any, Callable
import numpy as np
import pandas as pd
from scipy import stats
from app.exceptions import AnalysisToolError

NUMERIC_CANDIDATE_COLS = (
    "value", "current", "current_revenue", "revenue", "gross_revenue",
    "total_value", "total_revenue", "amount", "sales", "order_count", "units_sold",
)
LABEL_CANDIDATE_COLS = (
    "label", "product", "productName", "month", "country", "category",
    "line", "customer", "customerName", "status",
)


def _clean_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return default


def _pick_value_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in NUMERIC_CANDIDATE_COLS:
        if col in df.columns:
            return col
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        return num_cols[-1]
    raise AnalysisToolError(f"Dataset has no numeric column for analysis (columns={list(df.columns)}).")


def _pick_label_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in LABEL_CANDIDATE_COLS:
        if col in df.columns:
            return col
    str_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return str_cols[0] if str_cols else df.columns[0]


def tool_period_comparison(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute period-over-period absolute and percentage changes across entities or totals."""
    params = params or {}
    if not datasets or datasets[0].empty:
        return {"tool": "period_comparison", "status": "empty", "records": [], "summary": {}}

    df = datasets[0].copy()
    if {"previous_revenue", "current_revenue"}.issubset(df.columns):
        row = df.iloc[0]
        prev_rev, curr_rev = _clean_float(row.get("previous_revenue")), _clean_float(row.get("current_revenue"))
        delta_rev = round(curr_rev - prev_rev, 2)
        pct_rev = round(((curr_rev - prev_rev) / abs(prev_rev)) * 100.0, 2) if abs(prev_rev) > 1e-6 else None
        return {
            "tool": "period_comparison", "status": "ok", "mode": "scalar_baseline",
            "summary": {"previous": prev_rev, "current": curr_rev, "delta": delta_rev, "pct_change": pct_rev},
            "records": [{"label": "Total", "previous": prev_rev, "current": curr_rev, "change": delta_rev, "pct_change": pct_rev}],
        }

    if not {"previous", "current"}.issubset(df.columns):
        val_col = _pick_value_col(df, params.get("value_column"))
        lbl_col = _pick_label_col(df, params.get("label_column"))
        df["value_num"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
        total = _clean_float(df["value_num"].sum())
        return {
            "tool": "period_comparison", "status": "ok", "mode": "single_period_fallback",
            "summary": {"total": round(total, 2), "row_count": int(len(df))},
            "records": [{"label": str(r[lbl_col]), "current": round(_clean_float(r["value_num"]), 2)} for _, r in df.head(20).iterrows()],
        }

    lbl_col = _pick_label_col(df, params.get("label_column"))
    df["previous"] = pd.to_numeric(df["previous"], errors="coerce").fillna(0.0)
    df["current"] = pd.to_numeric(df["current"], errors="coerce").fillna(0.0)
    df["change"] = (df["current"] - df["previous"]).round(2)
    df["pct_change"] = np.where(df["previous"].abs() > 1e-6, ((df["current"] - df["previous"]) / df["previous"].abs() * 100.0).round(2), np.nan)

    prev_total, curr_total = round(_clean_float(df["previous"].sum()), 2), round(_clean_float(df["current"].sum()), 2)
    net_delta = round(curr_total - prev_total, 2)
    net_pct = round((net_delta / abs(prev_total)) * 100.0, 2) if abs(prev_total) > 1e-6 else None

    records = [
        {
            "label": str(r[lbl_col]), "previous": round(_clean_float(r["previous"]), 2),
            "current": round(_clean_float(r["current"]), 2), "change": round(_clean_float(r["change"]), 2),
            "pct_change": None if pd.isna(r["pct_change"]) else _clean_float(r["pct_change"]),
        }
        for _, r in df.sort_values("change", ascending=True).iterrows()
    ]
    return {
        "tool": "period_comparison", "status": "ok", "mode": "dimensional_comparison",
        "summary": {
            "previous_total": prev_total, "current_total": curr_total, "net_change": net_delta, "net_pct_change": net_pct,
            "largest_decline_label": records[0]["label"] if records else None,
            "largest_decline_value": records[0]["change"] if records else 0.0,
            "largest_gain_label": records[-1]["label"] if records else None,
            "largest_gain_value": records[-1]["change"] if records else 0.0,
        },
        "records": records,
    }


def tool_revenue_decomposition(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decompose net revenue variance into order volume effect, AOV effect, and interaction term."""
    if not datasets or datasets[0].empty:
        return {"tool": "revenue_decomposition", "status": "empty", "summary": {}}

    row = datasets[0].iloc[0]
    prev_rev = _clean_float(row.get("previous_revenue", row.get("previous", 0.0)))
    curr_rev = _clean_float(row.get("current_revenue", row.get("current", 0.0)))
    prev_orders, curr_orders = int(_clean_float(row.get("previous_orders", 0))), int(_clean_float(row.get("current_orders", 0)))

    delta_rev = round(curr_rev - prev_rev, 2)
    pct_rev = round((delta_rev / abs(prev_rev)) * 100.0, 2) if abs(prev_rev) > 1e-6 else None
    delta_orders = curr_orders - prev_orders
    pct_orders = round((delta_orders / abs(prev_orders)) * 100.0, 2) if prev_orders > 0 else None

    prev_aov = (prev_rev / prev_orders) if prev_orders > 0 else 0.0
    curr_aov = (curr_rev / curr_orders) if curr_orders > 0 else 0.0
    delta_aov = curr_aov - prev_aov
    pct_aov = round((delta_aov / abs(prev_aov)) * 100.0, 2) if abs(prev_aov) > 1e-6 else None

    volume_effect = round(delta_orders * prev_aov, 2)
    aov_effect = round(delta_aov * prev_orders, 2)
    interaction_effect = round(delta_orders * delta_aov, 2)
    residual = round(delta_rev - round(volume_effect + aov_effect + interaction_effect, 2), 2)

    vol_share = round((volume_effect / delta_rev) * 100.0, 2) if abs(delta_rev) > 1e-6 else 0.0
    aov_share = round((aov_effect / delta_rev) * 100.0, 2) if abs(delta_rev) > 1e-6 else 0.0
    inter_share = round((interaction_effect / delta_rev) * 100.0, 2) if abs(delta_rev) > 1e-6 else 0.0

    return {
        "tool": "revenue_decomposition", "status": "ok",
        "summary": {
            "previous_revenue": round(prev_rev, 2), "current_revenue": round(curr_rev, 2),
            "net_revenue_change": delta_rev, "net_revenue_pct_change": pct_rev,
            "previous_orders": prev_orders, "current_orders": curr_orders,
            "order_volume_change": delta_orders, "order_volume_pct_change": pct_orders,
            "previous_aov": round(prev_aov, 2), "current_aov": round(curr_aov, 2),
            "aov_change": round(delta_aov, 2), "aov_pct_change": pct_aov,
            "volume_effect_usd": volume_effect, "aov_effect_usd": aov_effect,
            "interaction_effect_usd": interaction_effect, "volume_share_of_change_pct": vol_share,
            "aov_share_of_change_pct": aov_share, "interaction_share_of_change_pct": inter_share,
            "reconciled": abs(residual) <= 1.0, "reconciliation_residual_usd": residual,
            "primary_driver": "order_volume" if abs(volume_effect) >= abs(aov_effect) else "average_order_value",
        },
    }


def tool_contribution_analysis(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Calculate dimensional variance contributions and reconcile against baseline net change."""
    params = params or {}
    if not datasets or datasets[0].empty:
        return {"tool": "contribution_analysis", "status": "empty", "records": [], "summary": {}}

    df = datasets[0].copy()
    lbl_col = _pick_label_col(df, params.get("label_column"))
    baseline_delta: float | None = None
    if len(datasets) > 1 and not datasets[1].empty and {"previous_revenue", "current_revenue"}.issubset(datasets[1].columns):
        brow = datasets[1].iloc[0]
        baseline_delta = round(_clean_float(brow.get("current_revenue")) - _clean_float(brow.get("previous_revenue")), 2)

    if {"previous", "current"}.issubset(df.columns):
        df["previous"] = pd.to_numeric(df["previous"], errors="coerce").fillna(0.0)
        df["current"] = pd.to_numeric(df["current"], errors="coerce").fillna(0.0)
        df["change"] = (df["current"] - df["previous"]).round(2)
        sum_changes = round(_clean_float(df["change"].sum()), 2)
        denom = baseline_delta if (baseline_delta is not None and abs(baseline_delta) > 1e-6) else sum_changes

        records = []
        for _, r in df.sort_values("change", ascending=True).iterrows():
            chg, prev_v, curr_v = round(_clean_float(r["change"]), 2), round(_clean_float(r["previous"]), 2), round(_clean_float(r["current"]), 2)
            records.append({
                "label": str(r[lbl_col]), "previous": prev_v, "current": curr_v, "change": chg,
                "pct_change": round(((curr_v - prev_v) / abs(prev_v)) * 100.0, 2) if abs(prev_v) > 1e-6 else None,
                "share_of_net_change_pct": round((chg / denom) * 100.0, 2) if abs(denom) > 1e-6 else 0.0,
            })

        discrepancy = round(sum_changes - baseline_delta, 2) if baseline_delta is not None else 0.0
        is_reconciled = abs(discrepancy) <= 1.0 if baseline_delta is not None else True
        top_neg, top_pos = (records[0] if records else None), (records[-1] if records else None)
        return {
            "tool": "contribution_analysis", "status": "ok",
            "summary": {
                "sum_of_changes": sum_changes, "baseline_net_change": baseline_delta,
                "discrepancy": discrepancy, "reconciled": is_reconciled,
                "primary_contributor_label": top_neg["label"] if top_neg else None,
                "primary_contributor_change": top_neg["change"] if top_neg else 0.0,
                "primary_contributor_share_pct": top_neg["share_of_net_change_pct"] if top_neg else 0.0,
                "largest_positive_offset_label": top_pos["label"] if top_pos else None,
                "largest_positive_offset_change": top_pos["change"] if top_pos else 0.0,
            },
            "records": records,
        }

    val_col = _pick_value_col(df, params.get("value_column"))
    df["val"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
    total = round(_clean_float(df["val"].sum()), 2)
    records = [
        {"label": str(r[lbl_col]), "value": (v := round(_clean_float(r["val"]), 2)), "share_of_total_pct": round((v / total) * 100.0, 2) if abs(total) > 1e-6 else 0.0}
        for _, r in df.sort_values("val", ascending=False).iterrows()
    ]
    return {
        "tool": "contribution_analysis", "status": "ok",
        "summary": {
            "total": total, "top_entity_label": records[0]["label"] if records else None,
            "top_entity_value": records[0]["value"] if records else 0.0,
            "top_entity_share_pct": records[0]["share_of_total_pct"] if records else 0.0, "reconciled": True,
        },
        "records": records,
    }


def tool_statistical_summary(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute descriptive statistics using NumPy and SciPy."""
    params = params or {}
    if not datasets or datasets[0].empty:
        return {"tool": "statistical_summary", "status": "empty", "summary": {}}
    df = datasets[0]
    val_col = _pick_value_col(df, params.get("value_column"))
    series = pd.to_numeric(df[val_col], errors="coerce").dropna()
    if series.empty:
        return {"tool": "statistical_summary", "status": "empty", "summary": {}}
    arr = series.to_numpy(dtype=float)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    skew_val = float(stats.skew(arr)) if len(arr) >= 3 else 0.0
    return {
        "tool": "statistical_summary", "status": "ok", "column": val_col,
        "summary": {
            "count": int(len(arr)), "sum": round(_clean_float(arr.sum()), 2), "mean": round(_clean_float(arr.mean()), 2),
            "std": round(_clean_float(arr.std(ddof=1) if len(arr) > 1 else 0.0), 2), "min": round(_clean_float(arr.min()), 2),
            "q25": round(_clean_float(q25), 2), "median": round(_clean_float(q50), 2), "q75": round(_clean_float(q75), 2),
            "iqr": round(_clean_float(q75 - q25), 2), "max": round(_clean_float(arr.max()), 2), "skewness": round(_clean_float(skew_val), 3),
        },
    }


def tool_trend_analysis(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Perform linear trend regression, MoM change computation, and peak/trough detection."""
    params = params or {}
    if not datasets or datasets[0].empty:
        return {"tool": "trend_analysis", "status": "empty", "summary": {}}
    df = datasets[0].copy()
    lbl_col = _pick_label_col(df, params.get("time_column") or params.get("label_column"))
    val_col = _pick_value_col(df, params.get("value_column"))
    df["val"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
    n, y = len(df), df["val"].to_numpy(dtype=float)
    x = np.arange(n, dtype=float)
    if n >= 2 and np.std(y) > 1e-9:
        reg = stats.linregress(x, y)
        slope, r_squared, p_value = _clean_float(reg.slope), _clean_float(reg.rvalue ** 2), _clean_float(reg.pvalue)
    else:
        slope, r_squared, p_value = 0.0, 0.0, 1.0
    max_idx, min_idx = int(np.argmax(y)), int(np.argmin(y))
    direction = "upward" if slope > 0 and p_value < 0.1 else ("downward" if slope < 0 and p_value < 0.1 else "seasonal_or_flat")
    return {
        "tool": "trend_analysis", "status": "ok",
        "summary": {
            "periods": n, "slope_per_period": round(slope, 2), "r_squared": round(r_squared, 4), "p_value": round(p_value, 4),
            "trend_direction": direction, "peak_period": str(df.iloc[max_idx][lbl_col]), "peak_value": round(_clean_float(y[max_idx]), 2),
            "trough_period": str(df.iloc[min_idx][lbl_col]), "trough_value": round(_clean_float(y[min_idx]), 2), "mean_value": round(_clean_float(y.mean()), 2),
        },
    }


def tool_anomaly_detection(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Detect statistical outliers using Z-score and Tukey IQR fences."""
    params = params or {}
    if not datasets or datasets[0].empty:
        return {"tool": "anomaly_detection", "status": "empty", "anomalies": [], "summary": {}}
    df = datasets[0].copy()
    lbl_col, val_col = _pick_label_col(df, params.get("label_column")), _pick_value_col(df, params.get("value_column"))
    df["val"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
    arr, threshold = df["val"].to_numpy(dtype=float), float(params.get("threshold") or 2.0)
    mean = float(np.mean(arr)) if len(arr) else 0.0
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    q25, q75 = np.percentile(arr, [25, 75]) if len(arr) >= 4 else (mean, mean)
    iqr = float(q75 - q25)
    lower_fence, upper_fence = q25 - 1.5 * iqr, q75 + 1.5 * iqr

    anomalies = []
    for _, row in df.iterrows():
        v = float(row["val"])
        z = ((v - mean) / std) if std > 1e-6 else 0.0
        if abs(z) >= threshold or (len(arr) >= 4 and (v < lower_fence or v > upper_fence)):
            anomalies.append({"label": str(row[lbl_col]), "value": round(v, 2), "z_score": round(_clean_float(z), 2), "direction": "high" if v >= mean else "low"})
    return {
        "tool": "anomaly_detection", "status": "ok",
        "summary": {"evaluated_rows": int(len(arr)), "mean": round(_clean_float(mean), 2), "std": round(_clean_float(std), 2), "anomaly_count": len(anomalies), "threshold_z": threshold},
        "anomalies": anomalies,
    }


def tool_correlation_analysis(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute Pearson and Spearman correlation between two numeric variables."""
    params = params or {}
    if not datasets or datasets[0].empty:
        return {"tool": "correlation_analysis", "status": "empty", "summary": {}}
    df = datasets[0].copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        for col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() >= 3 and col not in num_cols:
                df[col] = coerced
                num_cols.append(col)
    if len(num_cols) < 2 or len(df) < 3:
        return {"tool": "correlation_analysis", "status": "insufficient_columns", "summary": {"message": "At least 2 numeric columns and 3 observations are required for correlation analysis."}}

    col_x = params.get("value_column") if params.get("value_column") in num_cols else num_cols[0]
    col_y = params.get("secondary_value_column") if (params.get("secondary_value_column") in num_cols and params.get("secondary_value_column") != col_x) else next(c for c in num_cols if c != col_x)
    clean = df[[col_x, col_y]].dropna()
    if len(clean) < 3:
        return {"tool": "correlation_analysis", "status": "insufficient_data", "summary": {}}
    x, y = clean[col_x].to_numpy(dtype=float), clean[col_y].to_numpy(dtype=float)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return {"tool": "correlation_analysis", "status": "constant_series", "summary": {"pearson_r": 0.0, "spearman_rho": 0.0}}
    p_r, p_p = stats.pearsonr(x, y)
    s_r, s_p = stats.spearmanr(x, y)
    return {
        "tool": "correlation_analysis", "status": "ok",
        "summary": {
            "x_column": col_x, "y_column": col_y, "n": int(len(clean)),
            "pearson_r": round(_clean_float(p_r), 4), "pearson_p_value": round(_clean_float(p_p), 4),
            "spearman_rho": round(_clean_float(s_r), 4), "spearman_p_value": round(_clean_float(s_p), 4),
            "note": "Correlation measures statistical association only and does not prove causation.",
        },
    }


def tool_distribution_analysis(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyze entity concentration (top-1/top-3 share, Herfindahl-Hirschman Index, quantiles)."""
    params = params or {}
    if not datasets or datasets[0].empty:
        return {"tool": "distribution_analysis", "status": "empty", "summary": {}}
    df = datasets[0].copy()
    val_col, lbl_col = _pick_value_col(df, params.get("value_column")), _pick_label_col(df, params.get("label_column"))
    df["val"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
    pos = df[df["val"] > 0].sort_values("val", ascending=False)
    total = float(pos["val"].sum()) if not pos.empty else 0.0
    if total <= 1e-6:
        return {"tool": "distribution_analysis", "status": "empty", "summary": {}}
    shares = (pos["val"] / total).to_numpy(dtype=float)
    hhi = float(np.sum((shares * 100.0) ** 2))
    return {
        "tool": "distribution_analysis", "status": "ok",
        "summary": {
            "entity_count": int(len(pos)), "total": round(_clean_float(total), 2),
            "top_1_entity": str(pos.iloc[0][lbl_col]),
            "top_1_share_pct": round(_clean_float(float(shares[0] * 100.0) if len(shares) >= 1 else 0.0), 2),
            "top_3_share_pct": round(_clean_float(float(shares[:3].sum() * 100.0) if len(shares) >= 1 else 0.0), 2),
            "hhi_index": round(_clean_float(hhi), 1),
            "concentration_level": "high" if hhi >= 2500 else ("moderate" if hhi >= 1500 else "diversified"),
        },
    }


def tool_data_quality_check(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run deterministic data quality checks on a dataset (nulls, duplicates, non-positive values)."""
    if not datasets or datasets[0].empty:
        return {"tool": "data_quality_check", "status": "empty", "summary": {"row_count": 0, "null_cells": 0, "duplicate_rows": 0, "quality_ok": False}}
    df = datasets[0]
    null_cells, dup_rows = int(df.isna().sum().sum()), int(df.duplicated().sum())
    neg_counts = {c: cnt for c in df.select_dtypes(include=[np.number]).columns if c != "change" and (cnt := int((df[c] < 0).sum())) > 0}
    return {
        "tool": "data_quality_check", "status": "ok",
        "summary": {
            "row_count": int(len(df)), "column_count": int(len(df.columns)), "null_cells": null_cells,
            "duplicate_rows": dup_rows, "unexpected_negative_columns": neg_counts,
            "quality_ok": null_cells == 0 and dup_rows == 0 and not neg_counts,
        },
    }


def tool_forecast_linear(datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Produce a bounded linear trend forecast when explicitly requested and sample size >= 6."""
    params = params or {}
    if not datasets or datasets[0].empty:
        return {"tool": "forecast_linear", "status": "empty", "forecasts": [], "summary": {}}
    df = datasets[0].copy()
    val_col = _pick_value_col(df, params.get("value_column"))
    lbl_col = _pick_label_col(df, params.get("time_column") or params.get("label_column"))
    df["val"] = pd.to_numeric(df[val_col], errors="coerce").dropna()
    n = len(df)
    if n < 6:
        return {
            "tool": "forecast_linear", "status": "insufficient_history", "forecasts": [],
            "summary": {"periods_available": n, "minimum_required": 6, "warning": "At least 6 historical periods are required for a statistically meaningful forecast."},
        }
    horizon = min(6, max(1, int(params.get("horizon") or 3)))
    y, x = df["val"].to_numpy(dtype=float), np.arange(n, dtype=float)
    reg = stats.linregress(x, y)
    resid_std = float(np.std(y - (reg.intercept + reg.slope * x), ddof=2)) if n > 2 else 0.0
    forecasts = [
        {
            "step": step, "label": f"t+{step}",
            "point_forecast": round(_clean_float(max(0.0, (pt := float(reg.intercept + reg.slope * float(n - 1 + step))))), 2),
            "lower_95": round(_clean_float(max(0.0, pt - 1.96 * resid_std)), 2),
            "upper_95": round(_clean_float(max(0.0, pt + 1.96 * resid_std)), 2),
        }
        for step in range(1, horizon + 1)
    ]
    return {
        "tool": "forecast_linear", "status": "ok",
        "summary": {
            "historical_periods": n, "last_observed_label": str(df.iloc[-1][lbl_col]),
            "last_observed_value": round(_clean_float(y[-1]), 2), "horizon": horizon,
            "slope_per_period": round(_clean_float(reg.slope), 2), "r_squared": round(_clean_float(reg.rvalue ** 2), 4),
            "caveat": "Linear trend projection does not model holiday seasonality or inventory constraints.",
        },
        "forecasts": forecasts,
    }


ANALYSIS_TOOL_REGISTRY: dict[str, Callable[[list[pd.DataFrame], dict[str, Any] | None], dict[str, Any]]] = {
    "period_comparison": tool_period_comparison,
    "revenue_decomposition": tool_revenue_decomposition,
    "contribution_analysis": tool_contribution_analysis,
    "statistical_summary": tool_statistical_summary,
    "trend_analysis": tool_trend_analysis,
    "anomaly_detection": tool_anomaly_detection,
    "correlation_analysis": tool_correlation_analysis,
    "distribution_analysis": tool_distribution_analysis,
    "data_quality_check": tool_data_quality_check,
    "forecast_linear": tool_forecast_linear,
}


def run_registered_analysis_tool(tool_name: str, datasets: list[pd.DataFrame], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute an approved analysis function from `ANALYSIS_TOOL_REGISTRY`."""
    if tool_name not in ANALYSIS_TOOL_REGISTRY:
        raise AnalysisToolError(f"Unapproved Python analysis tool: '{tool_name}'.")
    return ANALYSIS_TOOL_REGISTRY[tool_name](datasets, params)

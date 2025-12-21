"""Feature monitoring utilities for drift detection and stability tracking."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
FEATURE_DIR = PUBLIC_DIR / "ml_features"
MONITOR_DIR = FEATURE_DIR / "monitoring"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_float_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series([], dtype=float)


def calculate_psi(reference: Iterable[float], target: Iterable[float], bins: int = 10) -> Optional[float]:
    """Calculates the Population Stability Index between two samples."""

    ref = np.asarray(list(reference), dtype=float)
    tgt = np.asarray(list(target), dtype=float)
    ref = ref[np.isfinite(ref)]
    tgt = tgt[np.isfinite(tgt)]
    if ref.size == 0 or tgt.size == 0:
        return None

    bins = max(3, min(bins, ref.size))
    try:
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(ref, quantiles)
    except Exception:
        return None

    edges = np.unique(edges)
    if edges.size <= 2:
        return None

    edges[0] = -math.inf
    edges[-1] = math.inf
    hist_ref, _ = np.histogram(ref, bins=edges)
    hist_tgt, _ = np.histogram(tgt, bins=edges)
    ref_total = hist_ref.sum()
    tgt_total = hist_tgt.sum()
    if ref_total == 0 or tgt_total == 0:
        return None

    psi = 0.0
    eps = 1e-6
    for ref_count, tgt_count in zip(hist_ref, hist_tgt):
        ref_pct = max(ref_count / ref_total, eps)
        tgt_pct = max(tgt_count / tgt_total, eps)
        psi += (tgt_pct - ref_pct) * math.log(tgt_pct / ref_pct)
    return float(psi)


def update_feature_drift_report(
    asset: str,
    log_path: Path,
    latest_features: Optional[Dict[str, float]] = None,
    baseline_window: int = 500,
    recent_window: int = 120,
) -> None:
    """Generates a drift snapshot for a given asset's feature history."""

    if not log_path.exists():
        return

    try:
        df = pd.read_csv(log_path)
    except Exception:
        return
    if df.empty:
        return

    total_rows = len(df)
    if total_rows < 2:
        return

    if total_rows >= baseline_window + recent_window:
        baseline = df.iloc[-(baseline_window + recent_window) : -recent_window]
        recent = df.iloc[-recent_window:]
    else:
        split = max(1, total_rows // 2)
        baseline = df.iloc[:split]
        recent = df.iloc[split:]

    if recent.empty:
        recent = df.tail(min(total_rows, recent_window))
    if baseline.empty:
        baseline = df.head(min(total_rows, baseline_window))

    feature_report: Dict[str, Dict[str, Optional[float]]] = {}
    alerts = []

    for column in df.columns:
        series_recent = _to_float_series(recent[column])
        series_base = _to_float_series(baseline[column])
        if series_recent.empty:
            continue

        psi_value = None
        if not series_base.empty:
            psi_value = calculate_psi(series_base, series_recent)
        recent_mean = float(series_recent.mean()) if not series_recent.empty else None
        recent_std = float(series_recent.std(ddof=0)) if series_recent.size > 1 else 0.0
        base_mean = float(series_base.mean()) if not series_base.empty else None
        base_std = float(series_base.std(ddof=0)) if series_base.size > 1 else 0.0
        recent_min = float(series_recent.min()) if not series_recent.empty else None
        recent_max = float(series_recent.max()) if not series_recent.empty else None

        drift_flag = bool(psi_value is not None and psi_value >= 0.2)
        severity = "warning"
        if psi_value is not None and psi_value >= 0.5:
            severity = "critical"
        if drift_flag:
            alerts.append({
                "feature": column,
                "psi": psi_value,
                "severity": severity,
            })

        feature_report[column] = {
            "recent_mean": recent_mean,
            "recent_std": recent_std,
            "recent_min": recent_min,
            "recent_max": recent_max,
            "baseline_mean": base_mean,
            "baseline_std": base_std,
            "psi": psi_value,
            "drift_flag": drift_flag,
        }

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MONITOR_DIR / f"{asset.upper()}_monitor.json"
    payload = {
        "asset": asset.upper(),
        "generated_utc": _now().isoformat(),
        "total_samples": total_rows,
        "baseline_window": len(baseline),
        "recent_window": len(recent),
        "features": feature_report,
        "alerts": alerts,
        "latest_snapshot": latest_features or {},
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # Also maintain a compact CSV with headline metrics.
    summary_rows = []
    for name, metrics in feature_report.items():
        row = {
            "asset": asset.upper(),
            "feature": name,
            "generated_utc": payload["generated_utc"],
            "recent_mean": metrics.get("recent_mean"),
            "baseline_mean": metrics.get("baseline_mean"),
            "psi": metrics.get("psi"),
            "drift_flag": metrics.get("drift_flag"),
        }
        summary_rows.append(row)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = MONITOR_DIR / f"{asset.upper()}_monitor.csv"
        if csv_path.exists():
            summary_df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(csv_path, index=False)


__all__ = [
    "calculate_psi",
    "update_feature_drift_report",
]

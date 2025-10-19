"""Centralised monitoring utilities for signal health and data quality."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
MONITOR_DIR = PUBLIC_DIR / "monitoring"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _determine_status(payload: Dict[str, Any]) -> str:
    if not payload.get("ok", True):
        return "error"
    diagnostics = payload.get("diagnostics") or {}
    latency_flags = diagnostics.get("latency_flags") or []
    if latency_flags:
        return "warning"
    if payload.get("signal") in {"no entry", "no-entry", "NO ENTRY"}:
        gates = payload.get("gates") or {}
        missing = gates.get("missing") or []
        if missing:
            return "blocked"
    return "ok"


def update_signal_health_report(public_dir: Path = PUBLIC_DIR, summary: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Produces a consolidated health snapshot across all analysed assets."""

    if summary is None:
        summary_path = public_dir / "analysis_summary.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as fh:
                    summary = json.load(fh)
            except Exception:
                summary = None
    if not summary or "assets" not in summary:
        return None

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)

    now = _now()
    rows: List[Dict[str, Any]] = []
    alerts: List[str] = []

    for asset, payload in summary.get("assets", {}).items():
        if not isinstance(payload, dict):
            continue
        signal_time = _parse_iso(payload.get("retrieved_at_utc"))
        signal_age_minutes = None
        if signal_time is not None:
            signal_age_minutes = (now - signal_time).total_seconds() / 60.0
        diagnostics = payload.get("diagnostics") or {}
        timeframes = diagnostics.get("timeframes") or {}
        spot_latency = None
        expected_spot_latency = None
        if isinstance(timeframes, dict):
            spot_meta = timeframes.get("spot")
            if isinstance(spot_meta, dict):
                spot_latency = spot_meta.get("latency_seconds")
                expected_spot_latency = spot_meta.get("expected_max_delay_seconds")
        status = _determine_status(payload)
        if isinstance(diagnostics.get("latency_flags"), list):
            alerts.extend(str(flag) for flag in diagnostics["latency_flags"])
        rows.append(
            {
                "asset": asset,
                "status": status,
                "signal": payload.get("signal"),
                "probability": payload.get("probability"),
                "spot_latency_seconds": spot_latency,
                "expected_spot_latency_seconds": expected_spot_latency,
                "signal_age_minutes": signal_age_minutes,
                "last_error": payload.get("error"),
            }
        )

    report_path = MONITOR_DIR / "health.json"
    payload = {
        "generated_utc": now.isoformat(),
        "assets": rows,
        "alerts": alerts,
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = MONITOR_DIR / "health.csv"
        if csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False)
    return report_path


__all__ = ["update_signal_health_report"]

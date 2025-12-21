"""Centralised monitoring utilities for signal health and data quality."""

from __future__ import annotations

import json
import math
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


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            result = float(value)
        else:
            text = str(value).strip()
            if not text:
                return None
            result = float(text)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
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


def update_data_latency_report(public_dir: Path = PUBLIC_DIR, summary: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Generate a per-asset latency snapshot to surface stale feeds quickly."""

    if summary is None:
        summary_path = Path(public_dir) / "analysis_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = None
    if not summary or "assets" not in summary:
        return None

    monitor_dir = Path(public_dir) / "monitoring"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    assets = summary.get("assets")
    if not isinstance(assets, dict):
        return None

    rows: List[Dict[str, Any]] = []
    alerts: List[str] = []
    now = _now()

    for asset, payload in assets.items():
        if not isinstance(payload, dict):
            continue
        diagnostics = payload.get("diagnostics") or {}
        timeframes = diagnostics.get("timeframes") or {}
        if not isinstance(timeframes, dict):
            timeframes = {}

        spot_meta = timeframes.get("spot") or {}
        spot_latency = _coerce_float(spot_meta.get("latency_seconds"))
        spot_limit = _coerce_float(
            spot_meta.get("freshness_limit_seconds")
            or spot_meta.get("expected_max_delay_seconds")
        )
        spot_violation = bool(spot_meta.get("freshness_violation"))
        if not spot_violation and spot_latency is not None and spot_limit is not None:
            spot_violation = spot_latency > spot_limit

        stale_frames: List[str] = []
        critical_frames: List[str] = []
        max_frame_latency = spot_latency
        for name, meta in timeframes.items():
            if not isinstance(meta, dict) or name == "spot":
                continue
            latency_value = _coerce_float(meta.get("latency_seconds"))
            if latency_value is not None:
                if max_frame_latency is None or latency_value > max_frame_latency:
                    max_frame_latency = latency_value
            if meta.get("stale_for_signals"):
                stale_frames.append(name)
            if meta.get("critical_stale"):
                critical_frames.append(name)

        spot_payload = payload.get("spot") if isinstance(payload.get("spot"), dict) else {}
        fallback_provider = None
        if isinstance(spot_payload, dict):
            fallback_provider = spot_payload.get("fallback_provider")

        row = {
            "asset": asset,
            "signal": payload.get("signal"),
            "spot_latency_seconds": spot_latency,
            "spot_limit_seconds": spot_limit,
            "spot_violation": bool(spot_violation),
            "fallback_provider": fallback_provider,
            "stale_timeframes": stale_frames,
            "critical_timeframes": critical_frames,
            "max_timeframe_latency_seconds": max_frame_latency,
        }
        rows.append(row)

        if spot_violation and spot_latency is not None:
            if spot_limit is not None:
                alerts.append(
                    f"{asset}: spot latency {spot_latency / 60.0:.1f} min (limit {spot_limit / 60.0:.1f} min)"
                )
            else:
                alerts.append(f"{asset}: spot latency {spot_latency / 60.0:.1f} min")
        if critical_frames:
            alerts.append(
                f"{asset}: critical stale frames {', '.join(sorted(set(critical_frames)))}"
            )

    report_path = monitor_dir / "data_latency.json"
    payload = {
        "generated_utc": now.isoformat(),
        "assets": rows,
        "alerts": alerts,
    }
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    csv_path = monitor_dir / "data_latency.csv"
    if rows:
        serializable_rows = []
        for row in rows:
            csv_row = dict(row)
            csv_row["stale_timeframes"] = ",".join(row.get("stale_timeframes", []))
            csv_row["critical_timeframes"] = ",".join(row.get("critical_timeframes", []))
            serializable_rows.append(csv_row)
        pd.DataFrame(serializable_rows).to_csv(csv_path, index=False)
    elif csv_path.exists():
        csv_path.unlink()

    return report_path


def record_latency_alert(
    asset: str,
    feed: str,
    message: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    public_dir: Path = PUBLIC_DIR,
) -> Path:
    """Persistálja a latency riasztást monitorozási célból."""

    monitor_dir = Path(public_dir) / "monitoring"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    alert_path = monitor_dir / "latency_alerts.json"
    log_path = monitor_dir / "latency_alerts.log"

    now = _now().astimezone(timezone.utc)
    payload: Dict[str, Any] = {
        "asset": str(asset),
        "feed": str(feed),
        "message": str(message),
        "created_utc": now.isoformat(),
    }
    if metadata:
        try:
            payload["metadata"] = dict(metadata)
        except Exception:
            payload["metadata"] = {"repr": repr(metadata)}

    try:
        existing_raw = json.loads(alert_path.read_text(encoding="utf-8"))
        existing = existing_raw[-199:] if isinstance(existing_raw, list) else []
    except FileNotFoundError:
        existing = []
    except Exception:
        existing = []

    existing.append(payload)
    alert_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Naplózási hiba esetén sem bukjon a stratégia.
        pass

    return alert_path


__all__ = ["update_signal_health_report", "update_data_latency_report", "record_latency_alert"]

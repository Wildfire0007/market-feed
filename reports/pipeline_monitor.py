"""Utilities for recording pipeline timing checkpoints between trading and analysis."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
MONITOR_DIR = Path(os.getenv("PIPELINE_MONITOR_DIR", str(PUBLIC_DIR / "monitoring")))
PIPELINE_MONITOR_PATH = Path(
    os.getenv("PIPELINE_MONITOR_PATH", str(MONITOR_DIR / "pipeline_timing.json"))
)
PIPELINE_LOG_PATH = Path(
    os.getenv("PIPELINE_MONITOR_LOG", str(MONITOR_DIR / "pipeline.log"))
)
DEFAULT_MAX_LAG_SECONDS = int(os.getenv("PIPELINE_MAX_LAG_SECONDS", "240"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _load_payload(path: Optional[Path] = None) -> Dict[str, Any]:
    target = Path(path or PIPELINE_MONITOR_PATH)
    if target.exists():
        try:
            with target.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return data
        except Exception:
            return {}
    return {}


def _save_payload(payload: Dict[str, Any], path: Optional[Path] = None) -> Path:
    target = Path(path or PIPELINE_MONITOR_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(target)
    return target


def record_trading_run(
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    duration_seconds: Optional[float] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Persist metadata about the most recent trading pipeline execution."""

    completed = completed_at or _now()
    if started_at is None and duration_seconds is not None:
        started = completed - timedelta(seconds=float(duration_seconds))
    else:
        started = started_at or completed
    computed_duration = duration_seconds
    if computed_duration is None:
        computed_duration = max((completed - started).total_seconds(), 0.0)
    payload = _load_payload(path)
    payload["trading"] = {
        "started_utc": _to_iso(started),
        "completed_utc": _to_iso(completed),
        "duration_seconds": round(float(computed_duration), 3),
    }
    payload["updated_utc"] = _to_iso(_now())
    _save_payload(payload, path)
    return payload


def record_analysis_run(
    started_at: Optional[datetime] = None,
    max_lag_seconds: Optional[int] = None,
    path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[float], Optional[int], bool]:
    """Persist metadata about the current analysis execution and compute lag."""

    started = started_at or _now()
    payload = _load_payload(path)
    trading_meta = payload.get("trading") if isinstance(payload, dict) else {}
    trading_completed = None
    if isinstance(trading_meta, dict):
        trading_completed = _parse_iso(trading_meta.get("completed_utc"))
    lag_seconds: Optional[float] = None
    if trading_completed is not None:
        lag_seconds = max((started - trading_completed).total_seconds(), 0.0)
    threshold = max_lag_seconds if max_lag_seconds is not None else DEFAULT_MAX_LAG_SECONDS
    breach = False
    if lag_seconds is not None and threshold is not None:
        breach = lag_seconds > float(threshold)
    payload["analysis"] = {
        "started_utc": _to_iso(started),
        "lag_from_trading_seconds": round(float(lag_seconds), 3) if lag_seconds is not None else None,
        "lag_threshold_seconds": threshold,
        "lag_breached": breach,
    }
    payload["updated_utc"] = _to_iso(_now())
    _save_payload(payload, path)
    return payload, lag_seconds, threshold, breach


def get_pipeline_log_path(path: Optional[Path] = None) -> Path:
    """Return the file path used for pipeline level logging."""

    target = Path(path or PIPELINE_LOG_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def summarize_pipeline_warnings(path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse the pipeline log and compute warning/client-error ratios."""

    target = Path(path or PIPELINE_LOG_PATH)
    summary: Dict[str, Any] = {
        "total_lines": 0,
        "warning_lines": 0,
        "client_error_lines": 0,
        "client_error_ratio": 0.0,
        "updated_utc": _to_iso(_now()),
    }
    if not target.exists():
        return summary

    try:
        with target.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                summary["total_lines"] += 1
                lower = line.lower()
                if "warning" in lower:
                    summary["warning_lines"] += 1
                    if "404" in lower or "400" in lower or "client error" in lower:
                        summary["client_error_lines"] += 1
    except Exception:
        return summary

    warnings = summary.get("warning_lines") or 0
    client_errors = summary.get("client_error_lines") or 0
    if warnings:
        summary["client_error_ratio"] = round(client_errors / warnings, 3)
    return summary
    
    
__all__ = [
    "DEFAULT_MAX_LAG_SECONDS",
    "PIPELINE_MONITOR_PATH",
    "PIPELINE_LOG_PATH",
    "get_pipeline_log_path",
    "record_analysis_run",
    "record_trading_run",
    "summarize_pipeline_warnings",
]

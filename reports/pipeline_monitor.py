"""Utilities for recording pipeline timing checkpoints between trading and analysis."""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
MONITOR_DIR = Path(os.getenv("PIPELINE_MONITOR_DIR", str(PUBLIC_DIR / "monitoring")))
PIPELINE_MONITOR_PATH = Path(
    os.getenv("PIPELINE_MONITOR_PATH", str(MONITOR_DIR / "pipeline_timing.json"))
)
PIPELINE_LOG_PATH = Path(
    os.getenv("PIPELINE_MONITOR_LOG", str(MONITOR_DIR / "pipeline.log"))
)
DEFAULT_MAX_LAG_SECONDS = int(os.getenv("PIPELINE_MAX_LAG_SECONDS", "240"))
ML_MODEL_REMINDER_DAYS = int(os.getenv("ML_MODEL_REMINDER_DAYS", "7"))
SYMBOL_HISTORY_WINDOW = max(int(os.getenv("PIPELINE_SYMBOL_HISTORY_WINDOW", "20")), 0)
WARNING_TREND_BUCKET_MINUTES = max(
    int(os.getenv("PIPELINE_WARNING_TREND_BUCKET_MINUTES", "15")), 1
)


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


_KNOWN_SYMBOLS: Optional[set[str]] = None
_SYMBOL_IGNORE = {
    "WARNING",
    "CLIENT",
    "ERROR",
    "TRACEBACK",
    "HTTP",
    "HTTPS",
    "API",
    "SENTIMENT",
    "EXIT",
    "THROTTLING",
    "THROTTLE",
    "RATE",
    "LIMIT",
    "REQUESTS",
}


def _load_known_symbols() -> set[str]:
    """Return the configured asset universe when available."""

    global _KNOWN_SYMBOLS
    if _KNOWN_SYMBOLS is None:
        try:
            from config.analysis_settings import load_config

            cfg = load_config()
            assets = cfg.get("assets")
            if isinstance(assets, Iterable) and not isinstance(assets, (str, bytes)):
                _KNOWN_SYMBOLS = {str(asset).upper() for asset in assets if str(asset).strip()}
            else:
                _KNOWN_SYMBOLS = set()
        except Exception:
            _KNOWN_SYMBOLS = set()
    return _KNOWN_SYMBOLS


SYMBOL_PATTERN = re.compile(r"\b([A-Z][A-Z0-9_]{1,14})\b")


def _extract_symbols(message: Optional[str]) -> List[str]:
    """Best effort extraction of instrument symbols from a log message."""

    if not message:
        return []
    candidates = set(match.group(1).upper() for match in SYMBOL_PATTERN.finditer(message))
    if not candidates:
        return []
    known = _load_known_symbols()
    if known:
        filtered = {token for token in candidates if token in known}
    else:
        filtered = {
            token
            for token in candidates
            if token not in _SYMBOL_IGNORE and any(char.isalpha() for char in token)
        }
    return sorted(filtered)


def _classify_warning(message: str) -> str:
    """Classify a warning message into client_error/throttling/other buckets."""

    lowered = message.lower()
    if any(keyword in lowered for keyword in ("429", "rate limit", "ratelimit", "throttle")):
        return "throttling"
    if any(keyword in lowered for keyword in ("client error", "http 4", "status 4")):
        return "client_error"
    return "other"


def _bucket_timestamp(timestamp: datetime) -> datetime:
    """Return the start of the trend bucket for ``timestamp``."""

    minutes = (timestamp.minute // WARNING_TREND_BUCKET_MINUTES) * WARNING_TREND_BUCKET_MINUTES
    return timestamp.replace(minute=minutes, second=0, microsecond=0)
    

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


def record_ml_model_status(
    missing: Optional[Iterable[str]] = None,
    placeholders: Optional[Iterable[str]] = None,
    remind_after_days: Optional[int] = None,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Persist ML model availability metadata and determine reminder cadence."""

    current = now or _now()
    try:
        remind_days = int(remind_after_days) if remind_after_days is not None else ML_MODEL_REMINDER_DAYS
    except (TypeError, ValueError):
        remind_days = ML_MODEL_REMINDER_DAYS
    remind_days = max(remind_days, 1)

    missing_list = sorted({str(item).upper() for item in (missing or []) if str(item).strip()})
    placeholder_list = sorted({str(item).upper() for item in (placeholders or []) if str(item).strip()})

    payload = _load_payload(path)
    ml_section = payload.get("ml_models") if isinstance(payload, dict) else None
    if not isinstance(ml_section, dict):
        ml_section = {}

    status_payload = {
        "missing": missing_list,
        "placeholders": placeholder_list,
        "updated_utc": _to_iso(current),
    }

    last_reminder = _parse_iso(ml_section.get("last_reminder_utc")) if isinstance(ml_section, dict) else None
    reminder_due = False
    if missing_list or placeholder_list:
        due_since = None
        if last_reminder is not None:
            delta = current - last_reminder
            if delta >= timedelta(days=remind_days):
                due_since = last_reminder
        else:
            due_since = current
        if due_since is not None:
            reminder_due = True
            ml_section["last_reminder_utc"] = _to_iso(current)

    ml_section["status"] = status_payload
    ml_section["reminder_period_days"] = remind_days
    payload["ml_models"] = ml_section
    payload["updated_utc"] = _to_iso(_now())
    _save_payload(payload, path)
    return payload, reminder_due


def _parse_log_line(line: str) -> Tuple[Optional[datetime], Optional[str], str]:
    parts = line.split(" ", 3)
    if len(parts) >= 4:
        timestamp_str = f"{parts[0]} {parts[1]}"
        level = parts[2]
        message = parts[3]
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%fZ")
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        except ValueError:
            timestamp = None
        return timestamp, level, message
    return None, None, line


def summarize_pipeline_warnings(path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse the pipeline log and compute warning/client-error ratios."""

    target = Path(path or PIPELINE_LOG_PATH)
    summary: Dict[str, Any] = {
        "total_lines": 0,
        "warning_lines": 0,
        "client_error_lines": 0,
        "error_lines": 0,
        "exception_lines": 0,
        "exception_types": {},
        "client_error_ratio": 0.0,
        "last_timestamp_utc": None,
        "last_error": None,
        "last_exception": None,
        "sentiment_exit_events": [],
        "recent_warning_symbols": {
            "window_events": SYMBOL_HISTORY_WINDOW,
            "total_events": 0,
            "symbols": [],
        },
        "warning_trend": {
            "bucket_minutes": WARNING_TREND_BUCKET_MINUTES,
            "series": [],
        },
        "updated_utc": _to_iso(_now()),
    }
    if not target.exists():
        return summary

    latest_timestamp: Optional[datetime] = None
    sentiment_marker = "[sentiment_exit]"
    in_traceback = False
    traceback_timestamp: Optional[datetime] = None
    if SYMBOL_HISTORY_WINDOW > 0:
        symbol_window: Deque[Tuple[Optional[datetime], str, str]] = deque(
            maxlen=SYMBOL_HISTORY_WINDOW
        )
    else:
        symbol_window = deque()
    trend_buckets: Dict[datetime, Dict[str, int]] = defaultdict(
        lambda: {"client_error": 0, "throttling": 0, "other": 0}
    )
    
    try:
        with target.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if not line.strip():
                    continue
                summary["total_lines"] += 1

                timestamp, level, message = _parse_log_line(line.strip())
                if timestamp and (latest_timestamp is None or timestamp > latest_timestamp):
                    latest_timestamp = timestamp
                    
                lower = line.lower()
                if "warning" in lower:
                    summary["warning_lines"] += 1
                    classification = _classify_warning(line)
                    if classification == "client_error":
                        summary["client_error_lines"] += 1
                    if timestamp:
                        bucket = _bucket_timestamp(timestamp)
                        trend_buckets[bucket][classification] += 1
                    if SYMBOL_HISTORY_WINDOW:
                        relevant_symbols: List[str] = []
                        if classification in {"client_error", "throttling"}:
                            relevant_symbols = _extract_symbols(message or line)
                        elif "symbol" in lower or "asset" in lower:
                            relevant_symbols = _extract_symbols(message or line)
                        for sym in relevant_symbols:
                            symbol_window.append((timestamp, classification, sym))

                if level and level.upper() == "ERROR":
                    summary["error_lines"] += 1
                    summary["last_error"] = {
                        "timestamp_utc": _to_iso(timestamp) if timestamp else None,
                        "level": level,
                        "message": message,
                    }

                if sentiment_marker in line:
                    detail = line.split(sentiment_marker, 1)[1].strip()
                    summary["sentiment_exit_events"].append(
                        {
                            "timestamp_utc": _to_iso(timestamp) if timestamp else None,
                            "level": level,
                            "detail": detail,
                        }
                    )

                if "traceback (most recent call last" in lower:
                    in_traceback = True
                    traceback_timestamp = timestamp
                    summary["exception_lines"] += 1
                    continue

                if in_traceback:
                    summary["exception_lines"] += 1
                    stripped = line.strip()
                    if not stripped:
                        continue
                    exception_match = re.match(
                        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*(Error|Exception))", stripped
                    )
                    if exception_match:
                        exc_name = exception_match.group("name")
                        summary["exception_types"][exc_name] = (
                            summary["exception_types"].get(exc_name, 0) + 1
                        )
                        summary["last_exception"] = {
                            "timestamp_utc": _to_iso(traceback_timestamp)
                            if traceback_timestamp
                            else None,
                            "type": exc_name,
                            "message": stripped,
                        }
                        in_traceback = False
                        traceback_timestamp = None
                    elif not stripped.startswith("File "):
                        in_traceback = False
                        traceback_timestamp = None
                    continue

                if "error" in lower and "warning" not in lower and "traceback" not in lower:
                    exception_match = re.search(
                        r"([A-Za-z_][A-Za-z0-9_]*(Error|Exception))", line
                    )
                    if exception_match:
                        exc_name = exception_match.group(1)
                        summary["exception_types"][exc_name] = (
                            summary["exception_types"].get(exc_name, 0) + 1
                        )
                        summary["last_exception"] = {
                            "timestamp_utc": _to_iso(timestamp) if timestamp else None,
                            "type": exc_name,
                            "message": message if message else line,
                        }
    except Exception:
        return summary

    warnings = summary.get("warning_lines") or 0
    client_errors = summary.get("client_error_lines") or 0
    if warnings:
        summary["client_error_ratio"] = round(client_errors / warnings, 3)
    if latest_timestamp is not None:
        summary["last_timestamp_utc"] = _to_iso(latest_timestamp)

    if symbol_window:
        counts: Dict[str, int] = {}
        last_seen: Dict[str, Optional[datetime]] = {}
        categories: Dict[str, set[str]] = defaultdict(set)
        for ts, category, symbol in symbol_window:
            counts[symbol] = counts.get(symbol, 0) + 1
            if ts is not None:
                previous = last_seen.get(symbol)
                if previous is None or ts > previous:
                    last_seen[symbol] = ts
            categories[symbol].add(category)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        summary["recent_warning_symbols"]["symbols"] = [
            {
                "symbol": symbol,
                "occurrences": count,
                "categories": sorted(categories[symbol]),
                "last_seen_utc": _to_iso(last_seen.get(symbol))
                if last_seen.get(symbol)
                else None,
            }
            for symbol, count in ordered
        ]
        summary["recent_warning_symbols"]["total_events"] = sum(counts.values())

    if trend_buckets:
        summary["warning_trend"]["series"] = [
            {
                "bucket_start_utc": _to_iso(bucket_time),
                "client_errors": counts.get("client_error", 0),
                "throttling": counts.get("throttling", 0),
                "other": counts.get("other", 0),
            }
            for bucket_time, counts in sorted(trend_buckets.items())
        ]
        
    return summary
    
    
__all__ = [
    "DEFAULT_MAX_LAG_SECONDS",
    "ML_MODEL_REMINDER_DAYS",
    "PIPELINE_MONITOR_PATH",
    "PIPELINE_LOG_PATH",
    "get_pipeline_log_path",
    "record_ml_model_status",
    "record_analysis_run",
    "record_trading_run",
    "summarize_pipeline_warnings",
]

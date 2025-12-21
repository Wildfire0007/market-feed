"""Utilities for recording pipeline timing checkpoints between trading and analysis."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
MONITOR_DIR = Path(os.getenv("PIPELINE_MONITOR_DIR", str(PUBLIC_DIR / "monitoring")))
PIPELINE_MONITOR_PATH = Path(
    os.getenv("PIPELINE_MONITOR_PATH", str(MONITOR_DIR / "pipeline_timing.json"))
)
PIPELINE_LOG_PATH = Path(
    os.getenv("PIPELINE_MONITOR_LOG", str(MONITOR_DIR / "pipeline.log"))
)
DEFAULT_ARTIFACTS = (
    PUBLIC_DIR / "analysis_summary.json",
    PUBLIC_DIR / "status.json",
    MONITOR_DIR / "pipeline_timing.json",
)
DEFAULT_MAX_LAG_SECONDS = int(os.getenv("PIPELINE_MAX_LAG_SECONDS", "240"))
ML_MODEL_REMINDER_DAYS = int(os.getenv("ML_MODEL_REMINDER_DAYS", "7"))
SYMBOL_HISTORY_WINDOW = max(int(os.getenv("PIPELINE_SYMBOL_HISTORY_WINDOW", "20")), 0)
WARNING_TREND_BUCKET_MINUTES = max(
    int(os.getenv("PIPELINE_WARNING_TREND_BUCKET_MINUTES", "15")), 1
)
RUN_ID_ENV_VARS = ("PIPELINE_RUN_ID", "GITHUB_RUN_ID")
RUN_CAPTURED_AT_UTC = datetime.now(timezone.utc)
LOGGER = logging.getLogger("market_feed.pipeline_monitor")
LOCAL_TIMEZONE = os.getenv("PIPELINE_LOCAL_TIMEZONE", "Europe/Budapest")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_local_iso(dt: datetime, tz: ZoneInfo) -> str:
    localized = dt.astimezone(tz).replace(microsecond=0)
    return localized.isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    except Exception:
        return None


def _normalize_section_timestamps(
    section: Dict[str, Any],
    *,
    keys: Iterable[str],
    section_label: str,
    errors: List[str],
) -> None:
    for key in keys:
        raw_value = section.get(key)
        if raw_value is None:
            continue
        parsed = _parse_iso(raw_value) if isinstance(raw_value, str) else None
        if parsed is None:
            section.pop(key, None)
            errors.append(f"{section_label}.{key}")
            continue
        section[key] = _to_iso(parsed)


def _validate_timestamp_payload(
    payload: Dict[str, Any], *, path: Optional[Path] = None
) -> Dict[str, Any]:
    errors: List[str] = []
    run_section = payload.get("run") if isinstance(payload, dict) else None
    if isinstance(run_section, dict):
        _normalize_section_timestamps(
            run_section,
            keys=("captured_at_utc", "started_at_utc"),
            section_label="run",
            errors=errors,
        )

    trading_section = payload.get("trading") if isinstance(payload, dict) else None
    if isinstance(trading_section, dict):
        _normalize_section_timestamps(
            trading_section,
            keys=("started_utc", "completed_utc"),
            section_label="trading",
            errors=errors,
        )

    analysis_section = payload.get("analysis") if isinstance(payload, dict) else None
    if isinstance(analysis_section, dict):
        _normalize_section_timestamps(
            analysis_section,
            keys=("started_utc", "completed_utc", "updated_utc"),
            section_label="analysis",
            errors=errors,
        )

    if isinstance(payload, dict):
        _normalize_section_timestamps(
            payload,
            keys=("updated_utc",),
            section_label="payload",
            errors=errors,
        )

    if errors:
        LOGGER.error(
            "Érvénytelen időbélyeg",
            extra={"mezok": sorted(errors), "forras": str(path or PIPELINE_MONITOR_PATH)},
        )
    return payload


def _append_localized_timestamps(payload: Dict[str, Any]) -> None:
    """Attach Europe/Budapest timestamp másolatok a payloadhoz."""

    try:
        tz = ZoneInfo(LOCAL_TIMEZONE)
    except Exception:
        LOGGER.warning(
            "Nem sikerült betölteni a helyi időzónát, UTC marad.",
            extra={"timezone": LOCAL_TIMEZONE},
        )
        return

    localized: Dict[str, Any] = {"timezone": LOCAL_TIMEZONE}

    def _copy_section(section_key: str, keys: Iterable[str]) -> None:
        section = payload.get(section_key)
        if not isinstance(section, dict):
            return
        local_section: Dict[str, str] = {}
        for key in keys:
            value = section.get(key)
            parsed = _parse_iso(value) if isinstance(value, str) else None
            if parsed is None:
                continue
            local_section[key.replace("_utc", "_local")] = _to_local_iso(parsed, tz)
        if local_section:
            localized[section_key] = local_section

    _copy_section("run", ("captured_at_utc", "started_at_utc"))
    _copy_section("trading", ("started_utc", "completed_utc"))
    _copy_section("analysis", ("started_utc", "completed_utc", "updated_utc"))

    if isinstance(payload.get("updated_utc"), str):
        parsed_updated = _parse_iso(payload["updated_utc"])
        if parsed_updated is not None:
            localized["updated_local"] = _to_local_iso(parsed_updated, tz)

    if len(localized) > 1:
        payload["localized_times"] = localized


def load_pipeline_payload(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and normalize the pipeline monitor payload (best effort)."""

    return _validate_timestamp_payload(_load_payload(path), path=path)
    

def _resolve_run_id() -> Optional[str]:
    for env_var in RUN_ID_ENV_VARS:
        value = os.getenv(env_var)
        if value is None:
            continue
        stripped = str(value).strip()
        if stripped:
            return stripped
    return None


def _current_run_metadata(now: Optional[datetime] = None) -> Dict[str, Any]:
    captured = now or RUN_CAPTURED_AT_UTC
    started_env = _parse_iso(os.getenv("PIPELINE_RUN_STARTED_AT_UTC"))
    started = started_env or captured
    return {
        "run_id": _resolve_run_id(),
        "started_at_utc": _to_iso(started),
        "captured_at_utc": _to_iso(captured),
    }


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


def _ensure_run_metadata(payload: Dict[str, Any], *, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Attach run_id and capture timestamps to the payload."""

    run_section = payload.get("run") if isinstance(payload, dict) else None
    if not isinstance(run_section, dict):
        run_section = {}
    meta = _current_run_metadata(now)
    if run_section.get("run_id") is None:
        run_section["run_id"] = meta.get("run_id")
    run_section["captured_at_utc"] = meta.get("captured_at_utc")
    run_section["started_at_utc"] = meta.get("started_at_utc")
    payload["run"] = run_section
    return payload


def _artifact_paths_from_env() -> List[Path]:
    raw = os.getenv("PIPELINE_ARTIFACTS")
    if raw:
        candidates = []
        for entry in raw.split(","):
            cleaned = entry.strip()
            if cleaned:
                candidates.append(Path(cleaned))
        if candidates:
            return candidates
    return list(DEFAULT_ARTIFACTS)


def _hash_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        stat = path.stat()
        return {"sha256": digest.hexdigest(), "size": stat.st_size}
    except (FileNotFoundError, PermissionError, OSError):
        return None


def collect_artifact_hashes(paths: Optional[Iterable[Path]] = None) -> Dict[str, Any]:
    """Return a mapping of artefact path to hash/size metadata."""

    target_paths = list(paths) if paths is not None else _artifact_paths_from_env()
    hashes: Dict[str, Any] = {}
    for path in target_paths:
        meta = _hash_file(path)
        hashes[str(Path(path))] = meta
    return hashes
    

def _record_execution_order(
    payload: Dict[str, Any],
    *,
    analysis_started: Optional[datetime],
    trading_completed: Optional[datetime],
) -> None:
    """Persist ordering metadata between trading and analysis runs."""

    order_section = payload.get("execution_order") if isinstance(payload, dict) else None
    if not isinstance(order_section, dict):
        order_section = {}
    if analysis_started is not None:
        order_section["analysis_started_utc"] = _to_iso(analysis_started)
    else:
        order_section["analysis_started_utc"] = None
    if trading_completed is not None:
        order_section["trading_completed_utc"] = _to_iso(trading_completed)
    else:
        order_section["trading_completed_utc"] = None
    if analysis_started is not None and trading_completed is not None:
        order_section["analysis_after_trading"] = analysis_started >= trading_completed
    else:
        order_section["analysis_after_trading"] = None
    payload["execution_order"] = order_section


def compute_run_timing_deltas(
    payload: Dict[str, Any], *, now: Optional[datetime] = None
) -> Dict[str, Optional[float]]:
    """Compute basic timestamp differences for pipeline stages."""

    current = now or _now()

    def _diff(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
        if start is None or end is None:
            return None
        return round(float((end - start).total_seconds()), 3)

    run_section = payload.get("run") if isinstance(payload, dict) else {}
    trading_section = payload.get("trading") if isinstance(payload, dict) else {}
    analysis_section = payload.get("analysis") if isinstance(payload, dict) else {}

    trading_started = _parse_iso(trading_section.get("started_utc")) if isinstance(trading_section, dict) else None
    trading_completed = _parse_iso(trading_section.get("completed_utc")) if isinstance(trading_section, dict) else None
    analysis_started = _parse_iso(analysis_section.get("started_utc")) if isinstance(analysis_section, dict) else None
    analysis_completed = _parse_iso(analysis_section.get("completed_utc")) if isinstance(analysis_section, dict) else None
    run_started = _parse_iso(run_section.get("started_at_utc")) if isinstance(run_section, dict) else None
    captured = _parse_iso(run_section.get("captured_at_utc")) if isinstance(run_section, dict) else None

    return {
        "trading_duration_seconds": _diff(trading_started, trading_completed),
        "analysis_duration_seconds": _diff(analysis_started, analysis_completed),
        "trading_to_analysis_gap_seconds": _diff(trading_completed, analysis_started),
        "analysis_age_seconds": _diff(analysis_completed, current),
        "run_capture_offset_seconds": _diff(run_started, captured),
    }
    

def get_run_logging_context(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Return static logging fields describing the current pipeline run."""

    meta = _current_run_metadata(now)
    return {key: value for key, value in meta.items() if value is not None}
    

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
    payload = _ensure_run_metadata(
        _validate_timestamp_payload(_load_payload(path), path=path), now=completed
    )
    payload["trading"] = {
        "started_utc": _to_iso(started),
        "completed_utc": _to_iso(completed),
        "duration_seconds": round(float(computed_duration), 3),
    }
    payload["updated_utc"] = _to_iso(_now())
    _append_localized_timestamps(payload)
    _save_payload(payload, path)
    return payload


def record_analysis_run(
    started_at: Optional[datetime] = None,
    max_lag_seconds: Optional[int] = None,
    path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[float], Optional[int], bool]:
    """Persist metadata about the current analysis execution and compute lag."""

    started = started_at or _now()
    payload = _ensure_run_metadata(
        _validate_timestamp_payload(_load_payload(path), path=path), now=started
    )
    trading_meta = payload.get("trading") if isinstance(payload, dict) else {}
    trading_completed = None
    if isinstance(trading_meta, dict):
        trading_completed = _parse_iso(trading_meta.get("completed_utc"))
        
    analysis_started = started
    if trading_completed is not None and started < trading_completed:
        analysis_started = trading_completed

    _record_execution_order(
        payload, analysis_started=analysis_started, trading_completed=trading_completed
    )
    lag_seconds: Optional[float] = None
    if trading_completed is not None:
        lag_seconds = max((analysis_started - trading_completed).total_seconds(), 0.0)
    threshold = max_lag_seconds if max_lag_seconds is not None else DEFAULT_MAX_LAG_SECONDS
    breach = False
    if lag_seconds is not None and threshold is not None:
        breach = lag_seconds > float(threshold)
    payload["analysis"] = {
        "started_utc": _to_iso(analysis_started),
        "lag_from_trading_seconds": round(float(lag_seconds), 3) if lag_seconds is not None else None,
        "lag_threshold_seconds": threshold,
        "lag_breached": breach,
    }
    payload["updated_utc"] = _to_iso(_now())
    _append_localized_timestamps(payload)
    _save_payload(payload, path)
    return payload, lag_seconds, threshold, breach


def finalize_analysis_run(
    completed_at: Optional[datetime] = None,
    duration_seconds: Optional[float] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Update the pipeline timing payload when the analysis stage finishes."""

    completed = completed_at or _now()
    payload = _ensure_run_metadata(
        _validate_timestamp_payload(_load_payload(path), path=path), now=completed
    )
    analysis_meta = payload.get("analysis") if isinstance(payload, dict) else {}
    if not isinstance(analysis_meta, dict):
        analysis_meta = {}
    started = _parse_iso(analysis_meta.get("started_utc"))
    trading_meta = payload.get("trading") if isinstance(payload, dict) else {}
    trading_completed = None
    if isinstance(trading_meta, dict):
        trading_completed = _parse_iso(trading_meta.get("completed_utc"))

    computed_duration = duration_seconds
    if computed_duration is None and started is not None:
        computed_duration = max((completed - started).total_seconds(), 0.0)

    if started is not None and trading_completed is not None:
        lag_seconds = max((started - trading_completed).total_seconds(), 0.0)
        analysis_meta["lag_from_trading_seconds"] = round(float(lag_seconds), 3)
        threshold_raw = analysis_meta.get("lag_threshold_seconds")
        try:
            threshold = float(threshold_raw) if threshold_raw is not None else DEFAULT_MAX_LAG_SECONDS
        except (TypeError, ValueError):
            threshold = float(DEFAULT_MAX_LAG_SECONDS)
        analysis_meta["lag_threshold_seconds"] = threshold
        analysis_meta["lag_breached"] = bool(lag_seconds > threshold)

    analysis_meta["completed_utc"] = _to_iso(completed)
    analysis_meta["duration_seconds"] = (
        round(float(computed_duration), 3) if computed_duration is not None else None
    )

    _record_execution_order(payload, analysis_started=started, trading_completed=trading_completed)
    payload["analysis"] = analysis_meta
    payload["updated_utc"] = _to_iso(_now())
    payload["artifacts"] = {
        "hashes": collect_artifact_hashes(),
        "updated_utc": _to_iso(_now()),
    }
    _append_localized_timestamps(payload)
    written_path = _save_payload(payload, path)
    written_key = str(Path(written_path))
    artifacts = payload.get("artifacts", {})
    hashes = artifacts.get("hashes") if isinstance(artifacts, dict) else None
    if isinstance(hashes, dict) and written_key in hashes:
        hashes[written_key] = _hash_file(Path(written_path))
        artifacts["hashes"] = hashes
        payload["artifacts"] = artifacts
        _append_localized_timestamps(payload)
        _save_payload(payload, path)
    return payload


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


def _parse_log_line(line: str) -> Tuple[Optional[datetime], Optional[str], str, Dict[str, Any]]:
    stripped = line.strip()
    if not stripped:
        return None, None, "", {}

    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return None, None, stripped, {}
        timestamp_value = data.get("timestamp") or data.get("@timestamp") or data.get("time")
        timestamp = _parse_iso(str(timestamp_value)) if timestamp_value is not None else None
        level_value = data.get("level")
        level = str(level_value) if level_value is not None else None
        message_value = data.get("message")
        message = str(message_value) if message_value is not None else ""
        return timestamp, level, message, data if isinstance(data, dict) else {}

    parts = stripped.split(" ", 3)
    if len(parts) >= 4:
        timestamp_str = f"{parts[0]} {parts[1]}"
        level = parts[2]
        message = parts[3]
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%fZ")
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        except ValueError:
            timestamp = None
        return timestamp, level, message, {}
    return None, None, stripped, {}


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

                timestamp, level, message, payload = _parse_log_line(line.strip())
                message_text = message or line
                message_lower = message_text.lower()
                line_lower = line.lower()
                level_upper = level.upper() if isinstance(level, str) else ""
                combined_lower = f"{level_upper} {message_lower}".strip().lower()

                if timestamp and (latest_timestamp is None or timestamp > latest_timestamp):
                    latest_timestamp = timestamp
                    
                is_warning = (
                    level_upper == "WARNING"
                    or "warning" in message_lower
                    or "warning" in line_lower
                )
                if is_warning:
                    summary["warning_lines"] += 1
                    classification = _classify_warning(message_text)
                    if classification == "client_error":
                        summary["client_error_lines"] += 1
                    if timestamp:
                        bucket = _bucket_timestamp(timestamp)
                        trend_buckets[bucket][classification] += 1
                    if SYMBOL_HISTORY_WINDOW:
                        relevant_symbols: List[str] = []
                        if classification in {"client_error", "throttling"}:
                            relevant_symbols = _extract_symbols(message_text)
                        elif "symbol" in combined_lower or "asset" in combined_lower:
                            relevant_symbols = _extract_symbols(message_text)
                        for sym in relevant_symbols:
                            symbol_window.append((timestamp, classification, sym))

                if level_upper == "ERROR":
                    summary["error_lines"] += 1
                    summary["last_error"] = {
                        "timestamp_utc": _to_iso(timestamp) if timestamp else None,
                        "level": level,
                        "message": message_text,
                    }

                source_for_marker = message_text or line
                if sentiment_marker in source_for_marker:
                    detail = source_for_marker.split(sentiment_marker, 1)[1].strip()
                    summary["sentiment_exit_events"].append(
                        {
                            "timestamp_utc": _to_iso(timestamp) if timestamp else None,
                            "level": level,
                            "detail": detail,
                        }
                    )

                exc_info_text = payload.get("exc_info") if isinstance(payload, dict) else None
                if isinstance(exc_info_text, str) and exc_info_text.strip():
                    exc_lines = [segment.strip() for segment in exc_info_text.splitlines() if segment.strip()]
                    count = max(len(exc_lines), 1)
                    summary["exception_lines"] += count
                    for exc_line in reversed(exc_lines):
                        exception_match = re.match(
                            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*(Error|Exception))",
                            exc_line,
                        )
                        if exception_match:
                            exc_name = exception_match.group("name")
                            summary["exception_types"][exc_name] = (
                                summary["exception_types"].get(exc_name, 0) + 1
                            )
                            summary["last_exception"] = {
                                "timestamp_utc": _to_iso(timestamp) if timestamp else None,
                                "type": exc_name,
                                "message": exc_line,
                            }
                            break
                    in_traceback = False
                    traceback_timestamp = timestamp

                if not payload and "traceback (most recent call last" in line_lower:
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

                if (
                    "error" in combined_lower
                    and "warning" not in combined_lower
                    and "traceback" not in combined_lower
                ):
                    exception_match = re.search(
                        r"([A-Za-z_][A-Za-z0-9_]*(Error|Exception))", message_text
                    )
                    if exception_match:
                        exc_name = exception_match.group(1)
                        summary["exception_types"][exc_name] = (
                            summary["exception_types"].get(exc_name, 0) + 1
                        )
                        summary["last_exception"] = {
                            "timestamp_utc": _to_iso(timestamp) if timestamp else None,
                            "type": exc_name,
                            "message": message_text,
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
    "collect_artifact_hashes",
    "compute_run_timing_deltas",
    "get_pipeline_log_path",
    "get_run_logging_context",
    "load_pipeline_payload",
    "finalize_analysis_run",
    "record_ml_model_status",
    "record_analysis_run",
    "record_trading_run",
    "summarize_pipeline_warnings",
]


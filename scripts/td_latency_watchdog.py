#!/usr/bin/env python3
"""Watchdog utility that restarts the trading data collector when feeds go stale.

The watchdog inspects ``public/analysis_summary.json`` (or a custom path) and
looks for critical latency on the one and five minute timeframes.  When the
latency exceeds the configured threshold – or when the analysis pipeline flags a
critical staleness condition – the watchdog executes a restart command.  By
default the restart command simply reruns ``Trading.py`` to refresh the caches,
but operators can override it with ``--restart-cmd``.

The script keeps a state file under ``public/monitoring`` to avoid spamming the
restart command and to provide observability for operators.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("td_latency_watchdog")
FLAG_PATTERN = re.compile(r"\b(k\d+m)\b.*?(\d+)\s+perc", re.IGNORECASE)
DEFAULT_TIMEFRAMES = ("k1m", "k5m")
DEFAULT_THRESHOLD_MINUTES = float(os.getenv("TD_WATCHDOG_THRESHOLD_MINUTES", "30"))
DEFAULT_COOLDOWN_MINUTES = float(os.getenv("TD_WATCHDOG_COOLDOWN_MINUTES", "10"))


@dataclass
class LatencyIssue:
    """Structured description of a latency breach discovered by the watchdog."""

    timeframe: str
    latency_seconds: Optional[float]
    message: str
    asset: Optional[str] = None
    source: str = "diagnostics"

    @property
    def latency_minutes(self) -> Optional[float]:
        if self.latency_seconds is None:
            return None
        return round(float(self.latency_seconds) / 60.0, 3)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "timeframe": self.timeframe,
            "latency_seconds": self.latency_seconds,
            "latency_minutes": self.latency_minutes,
            "message": self.message,
            "source": self.source,
        }


__all__ = ["LatencyIssue", "collect_latency_issues", "load_state", "save_state"]

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_iso(dt: datetime) -> str:
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


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        result = float(text)
        if not (result == result):  # NaN guard
            return None
        return result
    except Exception:
        return None


def _monitored_timeframes(items: Iterable[str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for name in items:
        trimmed = name.strip()
        if not trimmed:
            continue
        normalized[trimmed.lower()] = trimmed
    return normalized


def collect_latency_issues(
    summary: Dict[str, Any],
    threshold_seconds: float,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
) -> List[LatencyIssue]:
    """Return latency issues breaching ``threshold_seconds`` for the timeframes."""

    issues: List[LatencyIssue] = []
    seen: set[Tuple[Optional[str], str, str]] = set()
    tf_lookup = _monitored_timeframes(timeframes)
    if not tf_lookup:
        return issues

    def _register(issue: LatencyIssue) -> None:
        key = (issue.asset, issue.timeframe, issue.message)
        if key in seen:
            return
        seen.add(key)
        issues.append(issue)

    # Global latency flags from the summary header.
    for entry in summary.get("latency_flags", []) or []:
        if not isinstance(entry, str):
            continue
        match = FLAG_PATTERN.search(entry)
        if not match:
            continue
        tf_norm = match.group(1).lower()
        minutes = _coerce_float(match.group(2))
        if tf_norm not in tf_lookup:
            continue
        latency_seconds = None
        if minutes is not None:
            latency_seconds = float(minutes) * 60.0
        if latency_seconds is not None and latency_seconds < threshold_seconds:
            continue
        _register(
            LatencyIssue(
                asset=None,
                timeframe=tf_lookup[tf_norm],
                latency_seconds=latency_seconds,
                message=entry.strip(),
                source="summary_flag",
            )
        )

    assets = summary.get("assets")
    if isinstance(assets, dict):
        for asset_name, asset_meta in assets.items():
            if not isinstance(asset_meta, dict):
                continue
            diagnostics = asset_meta.get("diagnostics")
            if not isinstance(diagnostics, dict):
                continue
            timeframes_meta = diagnostics.get("timeframes")
            if not isinstance(timeframes_meta, dict):
                continue
            for tf_name, tf_meta in timeframes_meta.items():
                tf_norm = str(tf_name).lower()
                if tf_norm not in tf_lookup:
                    continue
                if not isinstance(tf_meta, dict):
                    continue
                latency_seconds = _coerce_float(tf_meta.get("latency_seconds"))
                critical_stale = bool(tf_meta.get("critical_stale"))
                stale_for_signals = bool(tf_meta.get("stale_for_signals"))
                expected_max = _coerce_float(tf_meta.get("expected_max_delay_seconds"))

                should_flag = False
                reason_parts: List[str] = []

                if latency_seconds is not None and latency_seconds >= threshold_seconds:
                    should_flag = True
                    minutes = latency_seconds / 60.0
                    reason_parts.append(
                        f"latency {minutes:.1f} perc >= küszöb {threshold_seconds/60.0:.1f} perc"
                    )
                elif critical_stale:
                    should_flag = True
                    if latency_seconds is not None:
                        minutes = latency_seconds / 60.0
                        reason_parts.append(
                            f"critical_stale (latency {minutes:.1f} perc)"
                        )
                    else:
                        reason_parts.append("critical_stale flag aktív")
                elif stale_for_signals and latency_seconds is not None and expected_max is not None:
                    if latency_seconds >= expected_max:
                        should_flag = True
                        minutes = latency_seconds / 60.0
                        reason_parts.append(
                            f"jelzéshez túl régi ({minutes:.1f} perc, limit {expected_max/60.0:.1f} perc)"
                        )

                if should_flag:
                    message = f"{asset_name} {tf_lookup[tf_norm]}: " + ", ".join(reason_parts)
                    _register(
                        LatencyIssue(
                            asset=str(asset_name),
                            timeframe=tf_lookup[tf_norm],
                            latency_seconds=latency_seconds,
                            message=message,
                            source="diagnostics",
                        )
                    )

    return issues


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def save_state(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)
    return path


def _default_restart_command() -> List[str]:
    python = sys.executable or "python3"
    return [python, "Trading.py"]


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restart data collection when klines go stale")
    parser.add_argument(
        "--summary",
        default=os.getenv("TD_WATCHDOG_SUMMARY", "public/analysis_summary.json"),
        help="Path to analysis_summary.json (default: public/analysis_summary.json)",
    )
    parser.add_argument(
        "--threshold-minutes",
        type=float,
        default=DEFAULT_THRESHOLD_MINUTES,
        help="Latency threshold in minutes before triggering a restart (default: 30)",
    )
    parser.add_argument(
        "--timeframes",
        default="k1m,k5m",
        help="Comma separated list of timeframes to monitor (default: k1m,k5m)",
    )
    parser.add_argument(
        "--cooldown-minutes",
        type=float,
        default=DEFAULT_COOLDOWN_MINUTES,
        help="Minimum minutes between restart attempts (default: 10)",
    )
    parser.add_argument(
        "--state-path",
        help="Optional path to persist watchdog state (default: <public>/monitoring/td_latency_watchdog.json)",
    )
    parser.add_argument(
        "--restart-cmd",
        nargs="+",
        help="Command to execute when a restart is required (default: python Trading.py)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not execute restart command")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / candidate).resolve()
    return candidate


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    summary_path = _resolve_path(args.summary)
    if not summary_path.exists():
        LOGGER.error("analysis_summary.json not found at %s", summary_path)
        return 1

    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception as exc:
        LOGGER.error("Failed to load analysis summary %s: %s", summary_path, exc)
        return 1

    if not isinstance(summary, dict):
        LOGGER.error("Unexpected analysis summary format: expected JSON object")
        return 1

    timeframes = [item for item in args.timeframes.split(",") if item.strip()]
    threshold_seconds = max(float(args.threshold_minutes), 0.0) * 60.0
    cooldown_seconds = max(float(args.cooldown_minutes), 0.0) * 60.0

    issues = collect_latency_issues(summary, threshold_seconds, timeframes)
    now = _now()

    state_path = (
        _resolve_path(args.state_path)
        if args.state_path
        else (summary_path.parent / "monitoring" / "td_latency_watchdog.json")
    )
    state = load_state(state_path)
    state["last_run_utc"] = _format_iso(now)

    if not issues:
        LOGGER.info(
            "No latency issues detected (threshold=%.1f perc, timeframes=%s)",
            threshold_seconds / 60.0,
            ",".join(timeframes) or "<none>",
        )
        state["last_result"] = "noop"
        state["last_issues"] = []
        save_state(state_path, state)
        return 0

    for issue in issues:
        scope = f"{issue.asset} {issue.timeframe}" if issue.asset else issue.timeframe
        LOGGER.warning("Latency issue detected: %s — %s", scope, issue.message)

    state["last_issues"] = [issue.as_dict() for issue in issues]

    if cooldown_seconds > 0:
        last_restart = _parse_iso(state.get("last_restart_utc"))
        if last_restart is not None:
            elapsed = (now - last_restart).total_seconds()
            if elapsed < cooldown_seconds:
                remaining = max(cooldown_seconds - elapsed, 0.0)
                LOGGER.info(
                    "Skipping restart (cooldown %.0f sec remaining)",
                    remaining,
                )
                state["last_result"] = "cooldown_skip"
                state["cooldown_remaining_seconds"] = remaining
                save_state(state_path, state)
                return 0

    restart_cmd = list(args.restart_cmd) if args.restart_cmd else _default_restart_command()
    pretty_cmd = " ".join(restart_cmd)

    if args.dry_run:
        LOGGER.info("Dry-run mode active — would execute: %s", pretty_cmd)
        state["last_result"] = "dry_run"
        state["last_restart_command"] = restart_cmd
        save_state(state_path, state)
        return 0

    LOGGER.info("Triggering restart via command: %s", pretty_cmd)
    result = subprocess.run(restart_cmd, cwd=_REPO_ROOT)
    exit_code = int(result.returncode)

    state["last_restart_utc"] = _format_iso(_now())
    state["last_restart_command"] = restart_cmd
    state["last_returncode"] = exit_code
    state["restart_count"] = int(state.get("restart_count", 0)) + 1
    state["last_result"] = "restart_ok" if exit_code == 0 else "restart_failed"
    save_state(state_path, state)

    if exit_code != 0:
        LOGGER.error("Restart command exited with code %s", exit_code)
    else:
        LOGGER.info("Restart command completed successfully")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

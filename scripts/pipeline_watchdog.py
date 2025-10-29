#!/usr/bin/env python3
"""Cron-friendly watchdog for pipeline log anomalies."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from reports.pipeline_monitor import get_pipeline_log_path, summarize_pipeline_warnings


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _build_summary_text(summary: Dict[str, Any]) -> str:
    lines = []
    warnings = summary.get("warning_lines") or 0
    client_errors = summary.get("client_error_lines") or 0
    ratio = summary.get("client_error_ratio") or 0.0
    errors = summary.get("error_lines") or 0
    exception_types = summary.get("exception_types") or {}
    last_exception = summary.get("last_exception") or {}
    sentiment_events = summary.get("sentiment_exit_events") or []
    lines.append(f"warnings={warnings} client_errors={client_errors} ratio={ratio:.3f}")
    lines.append(f"error_lines={errors}")
    if exception_types:
        formatted = ", ".join(
            f"{name}:{count}" for name, count in sorted(exception_types.items())
        )
        lines.append(f"exceptions={formatted}")
    if last_exception.get("message"):
        lines.append(f"last_exception={last_exception.get('message')}")
    if sentiment_events:
        latest = sentiment_events[-1]
        detail = latest.get("detail") or "sentiment exit"
        lines.append(f"latest_sentiment={detail}")
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline log health and emit alert-friendly output."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Path to the pipeline log file (defaults to PIPELINE_MONITOR_LOG).",
    )
    parser.add_argument(
        "--client-error-threshold",
        type=float,
        default=0.5,
        help="Alert when client error ratio across warnings meets or exceeds this value.",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=int,
        default=None,
        help="Alert when the last log entry is older than this many minutes.",
    )
    parser.add_argument(
        "--fail-on-sentiment",
        action="store_true",
        help="Escalate when sentiment exit events are present in the log.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON summary instead of a human readable format.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    log_path: Optional[Path]
    if args.log_path is not None:
        log_path = args.log_path
    else:
        try:
            log_path = get_pipeline_log_path()
        except Exception:
            log_path = None

    summary = summarize_pipeline_warnings(log_path)
    exit_code = 0
    alerts = []

    warnings = summary.get("warning_lines") or 0
    client_errors = summary.get("client_error_lines") or 0
    ratio = summary.get("client_error_ratio") or 0.0
    if warnings and client_errors and ratio >= args.client_error_threshold:
        alerts.append(
            f"client error ratio {ratio:.3f} exceeds threshold {args.client_error_threshold:.3f}"
        )
        exit_code = max(exit_code, 1)

    exception_types = summary.get("exception_types") or {}
    if exception_types:
        formatted = ", ".join(
            f"{name}:{count}" for name, count in sorted(exception_types.items())
        )
        alerts.append(f"exceptions detected: {formatted}")
        exit_code = max(exit_code, 2)

    last_timestamp = _parse_iso8601(summary.get("last_timestamp_utc"))
    if args.max_age_minutes and args.max_age_minutes > 0:
        if last_timestamp is None:
            alerts.append("log has no parsable timestamps")
            exit_code = max(exit_code, 1)
        else:
            age = datetime.now(timezone.utc) - last_timestamp
            if age > timedelta(minutes=args.max_age_minutes):
                alerts.append(
                    f"log is stale ({age.total_seconds()/60:.1f} minutes without entries)"
                )
                exit_code = max(exit_code, 1)

    sentiment_events = summary.get("sentiment_exit_events") or []
    if args.fail_on_sentiment and sentiment_events:
        detail = sentiment_events[-1].get("detail") or "sentiment exit"
        alerts.append(f"sentiment exit detected: {detail}")
        exit_code = max(exit_code, 1)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        log_display = str(log_path) if log_path is not None else "(default)"
        print(f"Pipeline log summary for {log_display}:")
        print(_build_summary_text(summary))

    if alerts:
        print("\nALERTS:")
        for alert in alerts:
            print(f"- {alert}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

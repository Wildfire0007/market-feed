#!/usr/bin/env python3
"""Market opening watchdog that resets dashboard snapshots automatically.

This utility is intended to run shortly after the exchange opens on working
days.  It refreshes the monitoring state files so that the dashboard does not
display stale "market closed" errors accumulated during the weekend or public
holidays.  The script is idempotent – once the reset is performed for the
current trading day it will skip subsequent invocations until the next day.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple

from scripts.reset_dashboard_state import (
    DEFAULT_BACKUP_DIR,
    DEFAULT_STATUS_MESSAGE,
    ResetResult,
    reset_anchor_state_file,
    reset_notify_state_file,
    reset_status_file,
)
from scripts.reset_dashboard_state import to_utc_iso

LOGGER = logging.getLogger("market_open_watchdog")

DEFAULT_MARKET_OPEN_UTC = os.getenv("MARKET_OPEN_UTC", "13:30")
DEFAULT_ALLOWED_DAYS = os.getenv("MARKET_OPEN_WEEKDAYS", "mon,tue,wed,thu,fri")
DEFAULT_LEEWAY_MINUTES = float(os.getenv("MARKET_OPEN_LEEWAY_MINUTES", "0"))
DEFAULT_ANCHOR_MAX_AGE_HOURS = float(os.getenv("MARKET_WATCHDOG_ANCHOR_MAX_AGE", "36"))
DEFAULT_STATE_FILENAME = os.getenv(
    "MARKET_WATCHDOG_STATE_FILENAME", "market_watchdog_state.json"
)


DAY_ALIASES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


@dataclass(frozen=True)
class WatchdogConfig:
    """Configuration describing the trading calendar."""

    market_open_time: time
    allowed_weekdays: Set[int]
    pre_open_leeway: timedelta
    anchor_max_age_hours: float


@dataclass
class WatchdogState:
    """Serializable state for remembering the last reset date."""

    last_reset_date: Optional[date] = None


def _parse_market_open(value: str) -> time:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid market open time: {value!r}")
    hour, minute = (int(parts[0]), int(parts[1]))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid market open time: {value!r}")
    return time(hour=hour, minute=minute, tzinfo=timezone.utc)


def _parse_weekdays(items: Iterable[str]) -> Set[int]:
    weekdays: Set[int] = set()
    for item in items:
        key = item.strip().lower()
        if not key:
            continue
        if key not in DAY_ALIASES:
            raise ValueError(f"Unknown weekday label: {item!r}")
        weekdays.add(DAY_ALIASES[key])
    return weekdays


def _load_calendar(path: Optional[Path]) -> Set[date]:
    if not path:
        return set()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        LOGGER.warning("Holiday calendar %s not found", path)
        return set()
    except Exception as exc:  # pragma: no cover - unexpected IO failure
        LOGGER.error("Failed to parse holiday calendar %s: %s", path, exc)
        return set()

    holidays: Set[date] = set()
    candidates: Sequence[str]
    if isinstance(raw, dict):
        candidates = raw.get("holidays") or []
    elif isinstance(raw, list):
        candidates = raw
    else:
        candidates = []

    for entry in candidates:
        try:
            holidays.add(datetime.fromisoformat(str(entry)).date())
        except Exception:
            LOGGER.warning("Ignoring invalid holiday entry %r", entry)
    return holidays


def _state_path(public_dir: Path, state_filename: str = DEFAULT_STATE_FILENAME) -> Path:
    monitor_dir = Path(public_dir) / "monitoring"
    return monitor_dir / state_filename


def _load_state(path: Path) -> WatchdogState:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return WatchdogState()
    except Exception:
        return WatchdogState()
    raw_date = payload.get("last_reset_date") if isinstance(payload, dict) else None
    if isinstance(raw_date, str):
        try:
            return WatchdogState(last_reset_date=datetime.fromisoformat(raw_date).date())
        except Exception:
            return WatchdogState()
    return WatchdogState()


def _write_state(path: Path, state: WatchdogState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if state.last_reset_date is not None:
        payload["last_reset_date"] = state.last_reset_date.isoformat()
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _make_backup(path: Path, backup_dir: Optional[Path]) -> Optional[Path]:
    if not path.exists():
        return None
    target_dir = backup_dir or DEFAULT_BACKUP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = target_dir / f"{path.name}.{timestamp}.bak"
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def _reset_health_file(
    public_dir: Path,
    *,
    now: datetime,
    reason: str,
    dry_run: bool = False,
    backup_dir: Optional[Path] = None,
) -> ResetResult:
    monitor_dir = Path(public_dir) / "monitoring"
    health_path = monitor_dir / "health.json"

    existing_assets = 0
    try:
        payload = json.loads(health_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("assets"), list):
            existing_assets = len(payload["assets"])
    except Exception:
        payload = None

    timestamp = to_utc_iso(now)
    refreshed = {
        "generated_utc": timestamp,
        "assets": [],
        "alerts": [],
        "notes": [
            {
                "type": "reset",
                "message": reason,
                "reset_utc": timestamp,
            }
        ],
    }

    changed = payload != refreshed
    backup_path = None

    if changed and not dry_run:
        backup_path = _make_backup(health_path, backup_dir)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        with health_path.open("w", encoding="utf-8") as fh:
            json.dump(refreshed, fh, ensure_ascii=False, indent=2)

    message = "health snapshot reset" if changed else "health already reset"

    return ResetResult(
        path=health_path,
        before=existing_assets,
        after=0,
        removed=existing_assets,
        changed=changed and not dry_run,
        backup_path=backup_path,
        message=message,
    )


def should_trigger_reset(
    now: datetime,
    *,
    config: WatchdogConfig,
    last_reset: Optional[date],
    holidays: Set[date],
    force: bool = False,
) -> bool:
    if force:
        return True

    today = now.date()

    if last_reset is not None and last_reset == today:
        return False

    if now.weekday() not in config.allowed_weekdays:
        return False

    if today in holidays:
        LOGGER.info("Skipping reset because %s is a holiday", today)
        return False

    open_dt = datetime.combine(today, config.market_open_time)
    leeway = config.pre_open_leeway
    if now + leeway < open_dt:
        LOGGER.info(
            "Skipping reset because current time %s is before market open %s", now, open_dt
        )
        return False

    return True


def perform_reset(
    public_dir: Path,
    *,
    now: datetime,
    config: WatchdogConfig,
    reason: str,
    dry_run: bool = False,
    backup_dir: Optional[Path] = None,
) -> Tuple[ResetResult, ResetResult, ResetResult, ResetResult]:
    backup_dir = backup_dir or DEFAULT_BACKUP_DIR

    anchor_result = reset_anchor_state_file(
        max_age_hours=config.anchor_max_age_hours,        
        now=now,
        dry_run=dry_run,
        backup_dir=backup_dir,
    )
    notify_result = reset_notify_state_file(
        path=Path(public_dir) / "_notify_state.json",
        now=now,
        dry_run=dry_run,
        backup_dir=backup_dir,
        reason=reason,
    )
    status_result = reset_status_file(
        path=Path(public_dir) / "status.json",
        now=now,
        dry_run=dry_run,
        backup_dir=backup_dir,
        reason=reason,
    )
    health_result = _reset_health_file(
        Path(public_dir),
        now=now,
        reason=reason,
        dry_run=dry_run,
        backup_dir=backup_dir,
    )
    return anchor_result, notify_result, status_result, health_result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", default=os.getenv("PUBLIC_DIR", "public"))
    parser.add_argument(
        "--market-open",
        default=DEFAULT_MARKET_OPEN_UTC,
        help="Market opening time in HH:MM UTC (default: %(default)s).",
    )
    parser.add_argument(
        "--weekdays",
        default=DEFAULT_ALLOWED_DAYS,
        help="Comma separated list of active trading weekdays (default: %(default)s).",
    )
    parser.add_argument(
        "--pre-open-leeway-minutes",
        type=float,
        default=DEFAULT_LEEWAY_MINUTES,
        help="Allow resets to run this many minutes before the configured open time.",
    )
    parser.add_argument(
        "--holiday-calendar",
        default=os.getenv("MARKET_WATCHDOG_HOLIDAYS"),
        help="Optional JSON file containing a list or object with a 'holidays' list of ISO dates.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Override the state tracking file path (defaults to monitoring/market_watchdog_state.json).",
    )
    parser.add_argument(
        "--reason",
        default=DEFAULT_STATUS_MESSAGE,
        help="Human readable reason recorded in the reset snapshots.",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Custom directory for backup JSON files.",
    )
    parser.add_argument(
        "--anchor-max-age-hours",
        type=float,
        default=DEFAULT_ANCHOR_MAX_AGE_HOURS,
        help="Max age in hours for anchor entries (<=0 clears all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a reset even if conditions are not met.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    public_dir = Path(args.public_dir).expanduser().resolve()
    now = datetime.now(timezone.utc)
    try:
        market_open = _parse_market_open(str(args.market_open))
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    try:
        weekdays = _parse_weekdays(str(args.weekdays).split(","))
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    config = WatchdogConfig(
        market_open_time=market_open,
        allowed_weekdays=weekdays or {0, 1, 2, 3, 4},
        pre_open_leeway=timedelta(minutes=float(args.pre_open_leeway_minutes)),
        anchor_max_age_hours=float(args.anchor_max_age_hours),
    )

    calendar_path = Path(args.holiday_calendar).expanduser() if args.holiday_calendar else None
    holidays = _load_calendar(calendar_path)

    state_file = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else _state_path(public_dir)
    )
    state = _load_state(state_file)

    if not should_trigger_reset(
        now,
        config=config,
        last_reset=state.last_reset_date,
        holidays=holidays,
        force=args.force,
    ):
        LOGGER.info("No reset required at %s", now.isoformat())
        return 0

    backup_dir = Path(args.backup_dir).expanduser() if args.backup_dir else DEFAULT_BACKUP_DIR

    results = perform_reset(
        public_dir,
        now=now,
        config=config,
        reason=str(args.reason),
        dry_run=args.dry_run,
        backup_dir=backup_dir,
    )

    for label, result in zip(("anchor", "notify", "status", "health"), results):
        changed = "changed" if result.changed else "unchanged"
        LOGGER.info(
            "%s: %s | before=%s | after=%s | removed=%s%s | %s",
            label,
            changed,
            result.before,
            result.after,
            result.removed,
            f" | backup={result.backup_path}" if result.backup_path else "",
            result.message,
        )

    if not args.dry_run:
        state.last_reset_date = now.date()
        _write_state(state_file, state)
        LOGGER.info("Updated watchdog state at %s", state_file)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

#!/usr/bin/env python3
"""Watchdog utility that alerts when spot quotes grow stale.

The script inspects the ``public/<asset>/spot.json`` payloads emitted by
``Trading.py`` and compares their timestamps against the freshness limits defined
in ``analysis_settings``.  When an asset breaches the configured threshold the
watchdog marks it as stale, writes a summary state file and optionally triggers
an alert command.

Run the watchdog periodically (e.g. from cron) to ensure the realtime spot feed
keeps up with the expectations of the analysis pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config.analysis_settings import SPOT_MAX_AGE_SECONDS, load_config

LOGGER = logging.getLogger("td_spot_watchdog")

DEFAULT_MARGIN_SECONDS = float(os.getenv("TD_SPOT_WATCHDOG_MARGIN_SECONDS", "0"))
DEFAULT_LOOP_SECONDS = float(os.getenv("TD_SPOT_WATCHDOG_LOOP_SECONDS", "0"))
DEFAULT_STATE_DIR = os.getenv("TD_SPOT_WATCHDOG_STATE_DIR")
DEFAULT_ALERT_CMD = os.getenv("TD_SPOT_WATCHDOG_ALERT_CMD", "")


@dataclass
class SpotStatus:
    """Structured representation of a spot quote freshness check."""

    asset: str
    path: Path
    limit_seconds: float
    threshold_seconds: float
    age_seconds: Optional[float]
    retrieved_utc: Optional[str]
    stale: bool
    reason: str
    freshness_violation: bool = False

    def age_minutes(self) -> Optional[float]:
        if self.age_seconds is None:
            return None
        return round(float(self.age_seconds) / 60.0, 3)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "path": str(self.path),
            "limit_seconds": self.limit_seconds,
            "threshold_seconds": self.threshold_seconds,
            "age_seconds": self.age_seconds,
            "age_minutes": self.age_minutes(),
            "retrieved_utc": self.retrieved_utc,
            "stale": self.stale,
            "reason": self.reason,
            "freshness_violation": self.freshness_violation,
        }


__all__ = [
    "SpotStatus",
    "collect_spot_statuses",
    "write_state",
    "parse_args",
    "main",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso8601(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except Exception:
        return None


def _load_spot_payload(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "missing_file"
    except Exception as exc:  # pragma: no cover - unexpected IO failure
        LOGGER.error("Failed to read spot payload for %s: %s", path, exc)
        return None, "read_error"

    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        LOGGER.warning("Invalid JSON payload at %s", path)
        return None, "invalid_json"


def _effective_threshold(limit_seconds: float, margin_seconds: float) -> float:
    if margin_seconds <= 0:
        return float(limit_seconds)
    return max(float(limit_seconds) - float(margin_seconds), 0.0)


def collect_spot_statuses(
    assets: Iterable[str],
    public_dir: Path,
    *,
    limits: Optional[Dict[str, Any]] = None,
    margin_seconds: float = 0.0,
    now: Optional[datetime] = None,
) -> List[SpotStatus]:
    """Return the freshness status for every requested asset."""

    public_dir = Path(public_dir)
    snapshot_time = now or _now()

    limits_map: Dict[str, float] = {}
    if limits is None:
        limits_map.update({str(k): float(v) for k, v in SPOT_MAX_AGE_SECONDS.items()})
    else:
        limits_map.update({str(k): float(v) for k, v in dict(limits).items()})

    default_limit = float(limits_map.get("default", 900.0))

    statuses: List[SpotStatus] = []
    for asset in assets:
        asset_name = str(asset)
        asset_dir = public_dir / asset_name
        spot_path = asset_dir / "spot.json"
        limit_seconds = float(limits_map.get(asset_name, default_limit))
        threshold_seconds = _effective_threshold(limit_seconds, margin_seconds)

        payload, error = _load_spot_payload(spot_path)
        if error is not None or payload is None:
            statuses.append(
                SpotStatus(
                    asset=asset_name,
                    path=spot_path,
                    limit_seconds=limit_seconds,
                    threshold_seconds=threshold_seconds,
                    age_seconds=None,
                    retrieved_utc=None,
                    stale=True,
                    reason=error or "unknown_error",
                )
            )
            continue

        retrieved_utc = payload.get("utc") or payload.get("retrieved_at_utc")
        freshness_violation = bool(payload.get("freshness_violation"))

        timestamp = _parse_iso8601(retrieved_utc)
        age_seconds: Optional[float]
        reason = "ok"
        stale = False

        if timestamp is None:
            age_seconds = None
            stale = True
            reason = "missing_timestamp"
        else:
            age_seconds = max((snapshot_time - timestamp).total_seconds(), 0.0)
            if freshness_violation:
                stale = True
                reason = "freshness_violation_flag"
            elif age_seconds > threshold_seconds:
                stale = True
                reason = "age_exceeds_limit"

        statuses.append(
            SpotStatus(
                asset=asset_name,
                path=spot_path,
                limit_seconds=limit_seconds,
                threshold_seconds=threshold_seconds,
                age_seconds=age_seconds,
                retrieved_utc=retrieved_utc,
                stale=stale,
                reason=reason,
                freshness_violation=freshness_violation,
            )
        )

    return statuses


def write_state(statuses: Sequence[SpotStatus], state_path: Path, generated_at: Optional[datetime] = None) -> Dict[str, Any]:
    """Persist the watchdog status to ``state_path`` and return the payload."""

    state_path = Path(state_path)
    generated_at = generated_at or _now()

    payload = {
        "generated_utc": _format_iso(generated_at),
        "stale_assets": [status.asset for status in statuses if status.stale],
        "assets": {status.asset: status.as_dict() for status in statuses},
    }

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _trigger_alert(cmd: str) -> None:
    if not cmd:
        return
    LOGGER.warning("Executing alert command: %s", cmd)
    try:
        subprocess.run(cmd, shell=True, check=False)
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.exception("Alert command failed")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor spot quote freshness against analysis limits")
    parser.add_argument("--public-dir", default=os.getenv("OUT_DIR", "public"), help="Base directory containing <asset>/spot.json files")
    parser.add_argument("--assets", nargs="*", help="Explicit asset symbols to inspect (defaults to analysis configuration)")
    parser.add_argument("--margin-seconds", type=float, default=DEFAULT_MARGIN_SECONDS, help="Alert earlier by subtracting this margin from the freshness limit")
    parser.add_argument("--state-file", help="Optional path for persisting watchdog state")
    parser.add_argument("--alert-cmd", default=DEFAULT_ALERT_CMD, help="Shell command to execute when any asset is stale")
    parser.add_argument("--loop-seconds", type=float, default=DEFAULT_LOOP_SECONDS, help="When >0 the watchdog re-runs on this interval")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def _resolve_assets(explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return [str(asset) for asset in explicit]
    cfg = load_config()
    return list(cfg.get("assets", []))


def _resolve_state_path(args: argparse.Namespace) -> Optional[Path]:
    if args.state_file:
        return Path(args.state_file)
    if DEFAULT_STATE_DIR:
        return Path(DEFAULT_STATE_DIR)
    public_dir = Path(args.public_dir)
    return public_dir / "monitoring" / "spot_watchdog_state.json"


def _run_once(args: argparse.Namespace) -> int:
    now = _now()
    assets = _resolve_assets(args.assets)
    if not assets:
        LOGGER.error("No assets configured for spot watchdog")
        return 2

    statuses = collect_spot_statuses(
        assets,
        Path(args.public_dir),
        margin_seconds=float(args.margin_seconds),
        now=now,
    )

    for status in statuses:
        if status.stale:
            LOGGER.error(
                "Spot feed stale for %s – reason=%s age=%s limit=%s path=%s",
                status.asset,
                status.reason,
                status.age_seconds,
                status.threshold_seconds,
                status.path,
            )
        else:
            LOGGER.info(
                "Spot feed healthy for %s – age=%.1fs limit=%.1fs",
                status.asset,
                status.age_seconds or 0.0,
                status.threshold_seconds,
            )

    state_path = _resolve_state_path(args)
    if state_path:
        write_state(statuses, state_path, generated_at=now)

    stale_assets = [status for status in statuses if status.stale]
    if stale_assets:
        _trigger_alert(args.alert_cmd)
        return 1

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    loop_seconds = max(float(args.loop_seconds), 0.0)
    if loop_seconds <= 0:
        return _run_once(args)

    LOGGER.info("Starting spot watchdog loop with %.1f second interval", loop_seconds)
    exit_code = 0
    try:
        while True:
            exit_code = _run_once(args)
            time.sleep(loop_seconds)
    except KeyboardInterrupt:
        LOGGER.info("Spot watchdog loop interrupted by user")
        return exit_code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

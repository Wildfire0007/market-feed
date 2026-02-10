"""Utility helpers to normalise the Discord notify state cache.

The notifier keeps per-asset counters inside ``public/_notify_state.json``.  When
those counters grow unbounded the dashboard stops surfacing new "entry" events.
This module introduces a daily normalisation that clears stale counters while
preserving the latest qualitative status for each asset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config.analysis_settings import ASSETS

__all__ = [
    "NotifyStateUpdate",
    "normalise_notify_state",
    "normalise_notify_state_file",
    "main",
]


@dataclass
class NotifyStateUpdate:
    """Summary of a normalisation run."""

    path: Path
    changed: bool
    reset_assets: List[str]
    cleared_counts: int
    message: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_asset_state() -> Dict[str, Any]:
    return {
        "last": "no entry",
        "count": 0,
        "last_sent": None,
        "last_sent_decision": None,
        "last_sent_mode": None,
        "last_sent_known": False,
        "cooldown_until": None,
    }


def build_default_state(*, now: Optional[datetime] = None, reason: str = "initialise") -> Dict[str, Any]:
    now = now or _now()
    payload: Dict[str, Any] = {
        "_meta": {
            "last_reset_utc": _to_iso(now),
            "last_reset_reason": reason,
        }
    }
    for asset in ASSETS:
        payload[asset] = _default_asset_state()
    return payload


def _reset_asset_state(asset: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    state = _default_asset_state()
    if payload:
        last_value = payload.get("last")
        if isinstance(last_value, str) and last_value.strip():
            state["last"] = last_value.strip()
    return state


def normalise_notify_state(
    state: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    reset_counts: bool = True,
    max_cooldown_age_minutes: Optional[float] = 1440.0,
    reason: str = "scheduled_normalise",
) -> Dict[str, Any]:
    """Return a normalised state dictionary without mutating the input."""

    now = now or _now()
    result: Dict[str, Any] = {}
    meta = dict(state.get("_meta") or {})
    meta["last_reset_reason"] = reason
    meta["last_reset_utc"] = _to_iso(now)
    result["_meta"] = meta

    cleared = []
    cleared_total = 0
    cooldown_cutoff: Optional[datetime] = None
    if max_cooldown_age_minutes and max_cooldown_age_minutes > 0:
        cooldown_cutoff = now - timedelta(minutes=float(max_cooldown_age_minutes))

    for asset, payload in state.items():
        if asset == "_meta":
            continue
        if not isinstance(payload, dict):
            result[asset] = _reset_asset_state(asset, None)
            cleared.append(asset)
            continue
        asset_state = dict(payload)
        original_count = asset_state.get("count")
        if reset_counts:
            if isinstance(original_count, (int, float)) and original_count != 0:
                asset_state["count"] = 0
                cleared.append(asset)
                cleared_total += 1
            else:
                asset_state["count"] = 0
        cooldown_until = asset_state.get("cooldown_until")
        if cooldown_until and cooldown_cutoff:
            try:
                cooldown_ts = datetime.fromisoformat(str(cooldown_until).replace("Z", "+00:00"))
            except Exception:
                cooldown_ts = None
            if cooldown_ts is None or cooldown_ts <= cooldown_cutoff:
                asset_state["cooldown_until"] = None
        result[asset] = asset_state

    # ensure all configured assets exist even if missing from disk
    defaults = build_default_state(now=now, reason=reason)
    for asset, payload in defaults.items():
        if asset == "_meta":
            continue
        if asset not in result:
            result[asset] = payload
            cleared.append(asset)

    meta["assets_reset"] = sorted(set(cleared))
    meta["cleared_counts"] = cleared_total
    return result


def normalise_notify_state_file(
    *,
    path: Path,
    now: Optional[datetime] = None,
    reset_counts: bool = True,
    max_cooldown_age_minutes: Optional[float] = 1440.0,
    reason: str = "scheduled_normalise",
) -> NotifyStateUpdate:
    now = now or _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                current = json.load(fh)
        except Exception:
            current = {}
    else:
        current = {}
    if not isinstance(current, dict) or not current:
        current = build_default_state(now=now, reason=reason)
        changed = True
    else:
        changed = False
    normalised = normalise_notify_state(
        current,
        now=now,
        reset_counts=reset_counts,
        max_cooldown_age_minutes=max_cooldown_age_minutes,
        reason=reason,
    )
    if normalised != current:
        changed = True
        with path.open("w", encoding="utf-8") as fh:
            json.dump(normalised, fh, ensure_ascii=False, indent=2)
    meta = normalised.get("_meta", {})
    reset_assets = meta.get("assets_reset") or []
    cleared_counts = int(meta.get("cleared_counts") or 0)
    message = f"reset {len(reset_assets)} assets" if reset_assets else "state unchanged"
    return NotifyStateUpdate(
        path=path,
        changed=changed,
        reset_assets=list(reset_assets),
        cleared_counts=cleared_counts,
        message=message,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", type=Path, default=Path("public/_notify_state.json"))
    parser.add_argument(
        "--no-reset-counts",
        action="store_true",
        help="Keep existing counts (only refresh metadata).",
    )
    parser.add_argument(
        "--max-cooldown-age-minutes",
        type=float,
        default=1440.0,
        help="Cooldown entries older than this many minutes are cleared (<=0 disables).",
    )
    parser.add_argument("--reason", type=str, default="scheduled_normalise")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    result = normalise_notify_state_file(
        path=args.state_path,
        reset_counts=not args.no_reset_counts,
        max_cooldown_age_minutes=args.max_cooldown_age_minutes,
        reason=args.reason,
    )
    print(result.message)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

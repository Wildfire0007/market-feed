"""Background SL/TP2 watchdog to enforce deterministic position lifecycle."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

from config import analysis_settings as settings

import position_tracker


LOGGER = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_close(asset: str) -> Tuple[Any, Any]:
    kline_path = Path("public") / asset / "klines_5m.json"
    data = _load_json(kline_path)
    rows = data.get("values") if isinstance(data, dict) else data if isinstance(data, list) else []
    fallback_price = None
    fallback_ts = None
    for row in rows or []:
        ts_raw = row.get("datetime") or row.get("t")
        close_raw = row.get("close") or row.get("c")
        if ts_raw is None or close_raw in (None, ""):
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
            close_val = float(close_raw)
        except Exception:
            continue
        if fallback_ts is None or ts > fallback_ts:
            fallback_ts = ts
            fallback_price = close_val
    return fallback_price, fallback_ts


def _resolve_spot(asset: str) -> Tuple[Any, Any]:
    signal_path = Path("public") / asset / "signal.json"
    spot_payload = _load_json(signal_path).get("spot", {}) if signal_path.exists() else {}
    price = spot_payload.get("price") or spot_payload.get("price_usd")
    utc = spot_payload.get("utc") or spot_payload.get("timestamp")

    if price is None:
        spot_path = Path("public") / asset / "spot.json"
        payload = _load_json(spot_path)
        price = payload.get("price") or payload.get("price_usd")
        utc = utc or payload.get("utc") or payload.get("timestamp")

    try:
        ts = datetime.fromisoformat(str(utc).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        ts = None

    fallback_price, fallback_ts = _latest_close(asset)
    if fallback_price is not None and (price is None or (fallback_ts and ts and fallback_ts > ts)):
        return fallback_price, fallback_ts

    return price, ts


def _cooldown_minutes(tracking_cfg: Dict[str, Any], asset: str, default: int) -> int:
    cooldown_map = tracking_cfg.get("post_exit_cooldown_minutes") or {}
    try:
        return int(cooldown_map.get(asset, cooldown_map.get("default", default)))
    except Exception:
        return int(default)


def main() -> None:
    position_tracker.set_audit_context(source="watchdog", run_id=str(uuid4()))
    config = settings.load_config().get("signal_stability") or {}
    tracking_cfg = config.get("manual_position_tracking") or {}
    if not tracking_cfg.get("enabled"):
        LOGGER.info("Manual tracking disabled; watchdog exiting")
        return

    manual_writer = str(tracking_cfg.get("writer") or "dual").lower()
    redundant_guard = bool(tracking_cfg.get("redundant_write_guard", False))
    can_write = manual_writer in {"analysis", "dual"} or redundant_guard
    positions_path = tracking_cfg.get("positions_file") or "public/_manual_positions.json"
    pending_exit_path = (
        tracking_cfg.get("pending_exit_file")
        or "public/_manual_positions_pending_exit.json"
    )
    treat_missing = bool(tracking_cfg.get("treat_missing_file_as_flat", False))
    cooldown_default = _cooldown_minutes(tracking_cfg, "default", 20)

    now_dt = datetime.now(timezone.utc)
    manual_positions = position_tracker.load_positions(positions_path, treat_missing)
    changed_assets = []

    for asset in settings.ASSETS:
        manual_state = position_tracker.compute_state(asset, tracking_cfg, manual_positions, now_dt)
        if not manual_state.get("has_position"):
            continue

        spot_price, _ = _resolve_spot(asset)
        changed, reason, manual_positions = position_tracker.check_close_by_levels(
            asset,
            manual_positions,
            spot_price,
            now_dt,
            _cooldown_minutes(tracking_cfg, asset, cooldown_default),
        )
        if not changed:
            continue

        if can_write:
            changed_assets.append((asset, reason))
        else:
            position_tracker.record_pending_exit(
                pending_exit_path,
                asset,
                reason=reason or "auto_close",
                closed_at_utc=position_tracker._to_utc_iso(now_dt),
                cooldown_minutes=_cooldown_minutes(tracking_cfg, asset, cooldown_default),
                source="watchdog",
            )

    if can_write and changed_assets:
        position_tracker.save_positions_atomic(positions_path, manual_positions)
        position_tracker.clear_pending_exits(
            pending_exit_path, [asset for asset, _ in changed_assets]
        )
        for asset, reason in changed_assets:
            position_tracker.log_audit_event(
                "watchdog auto-close committed",
                event="WATCHDOG_CLOSE_COMMIT",
                asset=asset,
                reason=reason,
            )


if __name__ == "__main__":
    main()

"""Helpers for deterministic manual/assumed position tracking.

This module centralizes the persistence and state derivation logic for
``manual_positions.json`` so both analysis and Discord notification layers
share the same behavior.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from logging_utils import ensure_json_file_handler, ensure_json_stream_handler


LOGGER = logging.getLogger("manual_positions")
ensure_json_stream_handler(LOGGER, static_fields={"component": "manual_positions"})

_AUDIT_CONTEXT: Dict[str, Any] = {
    "source": None,
    "run_id": None,
    "tz_name": "Europe/Budapest",
}
_FILE_LOGGER_ATTACHED = False


def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate repository root by walking upwards until markers are found."""

    start_path = start or Path(__file__).resolve()
    cursor = start_path if start_path.is_dir() else start_path.parent

    while True:
        if (cursor / ".git").is_dir() or (cursor / "public").is_dir():
            return cursor
        if cursor.parent == cursor:
            return cursor
        cursor = cursor.parent   


def resolve_repo_path(path: str, start: Optional[Path] = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    repo_root = _find_repo_root(start=start)
    return (repo_root / candidate).resolve()


def _parse_utc_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "00:00")).astimezone(
            timezone.utc
        )
    except Exception:
        return None


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("00:00", "Z")


def set_audit_context(source: str, run_id: str, tz_name: str = "Europe/Budapest") -> None:
    """Configure audit context fields for downstream log entries."""

    _AUDIT_CONTEXT["source"] = source
    _AUDIT_CONTEXT["run_id"] = run_id
    _AUDIT_CONTEXT["tz_name"] = tz_name or "Europe/Budapest"
    _maybe_attach_file_handler()


def _audit_fields(now_dt_utc: datetime) -> Dict[str, Any]:
    tz = ZoneInfo(str(_AUDIT_CONTEXT.get("tz_name") or "Europe/Budapest"))
    ts_utc = now_dt_utc.astimezone(timezone.utc)
    ts_local = now_dt_utc.astimezone(tz)
    fields: Dict[str, Any] = {
        "ts_utc": _to_utc_iso(ts_utc),
        "ts_budapest": ts_local.replace(microsecond=0).isoformat(),
        "source": _AUDIT_CONTEXT.get("source"),
        "run_id": _AUDIT_CONTEXT.get("run_id"),
        "component": "manual_positions",
    }
    gh_run_id = os.getenv("GITHUB_RUN_ID")
    if gh_run_id:
        fields["gh_run_id"] = gh_run_id
    return fields
    

def _audit_log(message: str, *, event: str, now_dt: Optional[datetime] = None, **fields: Any) -> None:
    now_dt = now_dt or datetime.now(timezone.utc)
    payload = {**_audit_fields(now_dt), **fields, "event": event}
    LOGGER.info(message, extra=payload)


def log_audit_event(message: str, *, event: str, **fields: Any) -> None:
    """Public helper so callers emit audit logs with consistent fields."""

    _audit_log(message, event=event, **fields)


def _should_log_to_file() -> bool:
    flag = os.getenv("MANUAL_POS_AUDIT_TO_FILE")
    if flag is not None:
        return str(flag).strip().lower() in {"1", "true", "yes", "on"}

    cfg_path = resolve_repo_path("config/analysis_settings.json")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return bool(cfg.get("manual_pos_audit_to_file", False))
    except Exception:
        return False


def _maybe_attach_file_handler() -> None:
    global _FILE_LOGGER_ATTACHED
    if _FILE_LOGGER_ATTACHED:
        return
    if not _should_log_to_file():
        return

    try:
        path = resolve_repo_path("public/_manual_positions_audit.jsonl")
        ensure_json_file_handler(
            LOGGER, path, static_fields={"component": "manual_positions"}
        )
        _FILE_LOGGER_ATTACHED = True
    except Exception:
        # Best-effort; keep pipeline running even if audit file logging fails.
        _FILE_LOGGER_ATTACHED = False


def load_positions(path: str, treat_missing_as_flat: bool) -> Dict[str, Any]:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    
    if not resolved.exists():
        _audit_log(
            "positions missing; returning empty state",
            event="LOAD_POSITIONS",
            positions_file=str(resolved),
            entries=0,
            missing_file=True,
            treat_missing_as_flat=bool(treat_missing_as_flat),
        )
        return {}

    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    positions = data if isinstance(data, dict) else {}
    _audit_log(
        "positions loaded",
        event="LOAD_POSITIONS",
        positions_file=str(resolved),
        entries=len(positions),
    )
    return positions


def save_positions_atomic(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path = resolved.with_suffix(resolved.suffix  ".tmp")
    tmp_path.write_text(payload  "\n", encoding="utf-8")
    _audit_log(
        "persisting positions to disk",
        event="SAVE_BEGIN",
        positions_file=str(resolved),
        tmp_file=str(tmp_path),
        entries=len(data),
        assets=sorted(data.keys()),
    )
    os.replace(tmp_path, resolved)

    stat = resolved.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    _audit_log(
        "positions saved",
        event="SAVE_COMMIT",
        positions_file=str(resolved),
        size=stat.st_size,
        mtime=mtime,
        entries=len(data),
    )

    return {
        "positions_file": str(resolved),
        "size": stat.st_size,
        "mtime": mtime,
        "written_bytes": len(payload.encode("utf-8"))  1,  # account for trailing newline
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def positions_file_snapshot(path: str) -> Dict[str, Any]:
    """Return size/mtime/sha256 for diagnostics; never raises."""

    resolved = resolve_repo_path(path)
    payload: Dict[str, Any] = {"positions_file": str(resolved)}
    try:
        stat = resolved.stat()
        payload.update(
            {
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        payload["error"] = str(exc)
    return payload


def compute_state(
    asset: str,
    cfg: Dict[str, Any],
    positions: Dict[str, Any],
    now_dt: datetime,
) -> Dict[str, Any]:
    tracking_enabled = bool((cfg or {}).get("enabled"))
    if not tracking_enabled:
        return {
            "enabled": False,
            "tracking_enabled": False,
            "has_position": False,
            "is_flat": True,
            "side": None,
            "cooldown_active": False,
            "cooldown_until_utc": None,
            "opened_at_utc": None,
            "entry": None,
            "sl": None,
            "tp2": None,
            "position": None,
        }

    asset_entry = positions.get(asset) if isinstance(positions, dict) else None
    side_raw = None
    cooldown_raw = None
    if isinstance(asset_entry, dict):
        side_raw = str(asset_entry.get("side") or "").strip().lower()
        cooldown_raw = asset_entry.get("cooldown_until_utc")

    cooldown_until = _parse_utc_timestamp(cooldown_raw)
    cooldown_active = bool(cooldown_until and now_dt < cooldown_until)

    side_map = {"long": "buy", "short": "sell"}
    side = side_map.get(side_raw)

    has_position = bool(side) and not cooldown_active
    is_flat = not side and not cooldown_active
    opened_at = asset_entry.get("opened_at_utc") if isinstance(asset_entry, dict) else None
    entry_level = asset_entry.get("entry") if isinstance(asset_entry, dict) else None
    sl_level = asset_entry.get("sl") if isinstance(asset_entry, dict) else None
    tp2_level = asset_entry.get("tp2") if isinstance(asset_entry, dict) else None

    return {
        "enabled": tracking_enabled,
        "tracking_enabled": tracking_enabled,
        "has_position": has_position,
        "is_flat": is_flat,
        "side": side,
        "cooldown_active": cooldown_active,
        "cooldown_until_utc": _to_utc_iso(cooldown_until) if cooldown_until else None,
        "opened_at_utc": opened_at,
        "entry": entry_level,
        "sl": sl_level,
        "tp2": tp2_level,
        "position": deepcopy(asset_entry) if isinstance(asset_entry, dict) else None,
    }


def open_position(
    asset: str,
    side: Optional[str],
    entry: Optional[float],
    sl: Optional[float],
    tp2: Optional[float],
    opened_at_utc: str,
    positions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    side_map = {"buy": "long", "sell": "short", "long": "long", "short": "short"}
    norm_side = side_map.get(str(side or "").lower())
    updated = deepcopy(positions) if isinstance(positions, dict) else {}
    previous_entry = deepcopy(updated.get(asset)) if isinstance(updated.get(asset), dict) else None
    updated[asset] = {
        "side": norm_side,
        "opened_at_utc": opened_at_utc,
        "entry": entry,
        "sl": sl,
        "tp2": tp2,
        "closed_at_utc": None,
        "close_reason": None,
        "cooldown_until_utc": None,
    }
    _audit_log(
        "open_position applied",
        event="OPEN_APPLIED",
        asset=asset,
        requested_side=side,
        normalized_side=norm_side,
        entry=entry,
        sl=sl,
        tp2=tp2,
        opened_at_utc=opened_at_utc,
        previous_position=previous_entry,
        updated_position=deepcopy(updated.get(asset)),
        positions_count=len(updated),
    )
    return updated


def close_position(
    asset: str,
    reason: str,
    closed_at_utc: str,
    cooldown_minutes: int,
    positions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    updated = deepcopy(positions) if isinstance(positions, dict) else {}
    entry = updated.get(asset) if isinstance(updated, dict) else None
    cooldown_dt = _parse_utc_timestamp(closed_at_utc) or datetime.now(timezone.utc)
    cooldown_until = cooldown_dt  timedelta(minutes=max(0, int(cooldown_minutes)))

    if not isinstance(entry, dict):
        entry = {"side": None}

    entry.update(
        {
            "side": None,
            "closed_at_utc": closed_at_utc,
            "close_reason": reason,
            "cooldown_until_utc": _to_utc_iso(cooldown_until),
        }
    )
    updated[asset] = entry
    _audit_log(
        "close_position applied",
        event="CLOSE_APPLIED",
        asset=asset,
        reason=reason,
        closed_at_utc=closed_at_utc,
        cooldown_until_utc=entry.get("cooldown_until_utc"),
        previous_position=positions.get(asset) if isinstance(positions, dict) else None,
        updated_position=deepcopy(entry),
    )
    return updated


def _levels_hit(side: Optional[str], spot_price: Optional[float], sl: Any, tp2: Any) -> Tuple[bool, Optional[str]]:
    try:
        price = float(spot_price)
    except (TypeError, ValueError):
        return False, None

    side_norm = str(side or "").lower()
    sl_val = None if sl is None else float(sl)
    tp2_val = None if tp2 is None else float(tp2)

    if side_norm == "long":
        if sl_val is not None and price <= sl_val:
            return True, "sl_hit"
        if tp2_val is not None and price >= tp2_val:
            return True, "tp2_hit"
    elif side_norm == "short":
        if sl_val is not None and price >= sl_val:
            return True, "sl_hit"
        if tp2_val is not None and price <= tp2_val:
            return True, "tp2_hit"
    return False, None


def check_close_by_levels(
    asset: str,
    positions: Dict[str, Any],
    spot_price: Optional[float],
    now_dt: datetime,
    cooldown_minutes: int,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    entry = positions.get(asset) if isinstance(positions, dict) else None
    if not isinstance(entry, dict):
        return False, None, positions if isinstance(positions, dict) else {}

    side = entry.get("side")
    if side not in {"long", "short"}:
        return False, None, positions if isinstance(positions, dict) else {}

    hit, reason = _levels_hit(side, spot_price, entry.get("sl"), entry.get("tp2"))
    if not hit or not reason:
        return False, None, positions if isinstance(positions, dict) else {}

    updated = close_position(
        asset,
        reason=reason,
        closed_at_utc=_to_utc_iso(now_dt),
        cooldown_minutes=cooldown_minutes,
        positions=positions,
    )
    _audit_log(
        "auto close triggered by levels",
        event="AUTO_CLOSE_BY_LEVELS",
        asset=asset,
        reason=reason,
        spot_price=spot_price,
    )
    return True, reason, updated
  

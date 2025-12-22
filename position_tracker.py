"""Helpers for deterministic manual/assumed position tracking.

This module centralizes the persistence and state derivation logic for
``manual_positions.json`` so both analysis and Discord notification layers
share the same behavior.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate repository root by walking upwards until markers are found."""

    def _search(path: Path) -> Optional[Path]:
        while True:
            if (path / ".git").is_dir() or (path / "public").is_dir():
                return path
            if path.parent == path:
                return None
            path = path.parent

    candidates = []
    if start:
        candidates.append(start)
    candidates.extend([Path(__file__).resolve().parent, Path.cwd().resolve()])

    for candidate in candidates:
        resolved_candidate = candidate if candidate.is_dir() else candidate.parent
        root = _search(resolved_candidate)
        if root:
            return root

    return Path.cwd().resolve()


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    repo_root = _find_repo_root()
    return (repo_root / candidate).resolve()


def _parse_utc_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except Exception:
        return None


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_positions(path: str, treat_missing_as_flat: bool) -> Dict[str, Any]:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    
    if not resolved.exists():
        resolved.write_text("{}\n", encoding="utf-8")
        
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    positions = data if isinstance(data, dict) else {}
    print(
        "[manual_positions] positions_file=%s entries=%d"
        % (resolved, len(positions)),
        flush=True,
    )
    return positions


def save_positions_atomic(path: str, data: Dict[str, Any]) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
    tmp_path.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp_path, resolved)

    stat = resolved.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    print(
        "[manual_positions] saved positions_file=%s size=%d mtime=%s"
        % (resolved, stat.st_size, mtime),
        flush=True,
    )


def compute_state(
    asset: str,
    cfg: Dict[str, Any],
    positions: Dict[str, Any],
    now_dt: datetime,
) -> Dict[str, Any]:
    tracking_enabled = bool((cfg or {}).get("enabled"))
    if not tracking_enabled:
        return {
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
    cooldown_until = cooldown_dt + timedelta(minutes=max(0, int(cooldown_minutes)))

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
    return True, reason, updated
  

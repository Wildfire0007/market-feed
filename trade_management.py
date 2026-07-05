"""Manual TP1 partial-close and breakeven operator instructions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from profit_target import _round_trip_cost


def _session_force_close_utc(session_meta: Dict[str, Any], buffer_min: int) -> Optional[str]:
    close_raw = session_meta.get("current_entry_window_end_utc") or session_meta.get("next_close_utc")
    if not close_raw:
        return None
    try:
        close_ts = datetime.fromisoformat(str(close_raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if close_ts.tzinfo is None:
        close_ts = close_ts.replace(tzinfo=timezone.utc)
    return (close_ts.astimezone(timezone.utc) - timedelta(minutes=max(0, int(buffer_min)))).isoformat().replace("+00:00", "Z")


def build_management_plan(
    *,
    asset: str,
    side: str,
    entry: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
    config: Dict[str, Any],
    asset_cost_model: Dict[str, Any],
    default_cost_model: Optional[Dict[str, Any]],
    session_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return explicit manual operator instructions for TP1/BE management."""
    enabled = {str(item).upper() for item in config.get("enabled_assets", []) if item}
    asset_key = str(asset).upper()
    if asset_key not in enabled or side not in {"buy", "sell"} or entry is None or tp1 is None:
        return None
    partial = max(0.0, min(1.0, float(config.get("partial_close_pct", 0.5) or 0.5)))
    round_trip = _round_trip_cost(asset_key, asset_cost_model, default_cost_model)
    offset = abs(float(entry)) * round_trip if config.get("breakeven_offset_covers_costs", True) else 0.0
    be_sl = float(entry) + offset if side == "buy" else float(entry) - offset
    force_close_at = _session_force_close_utc(session_meta or {}, int(config.get("session_force_close_buffer_min", 20) or 20))
    instructions = [
        f"TP1 elérésénél zárd a pozíció {partial:.0%}-át manuálisan.",
        f"TP1 után húzd a stopot költségekkel korrigált breakevenre: {be_sl:.5f}.",
        "A maradék runner célára TP2; nincs automatikus végrehajtás, operátori megerősítés szükséges.",
    ]
    if force_close_at:
        instructions.append(f"Napon túl ne tartsd: force-close legkésőbb {force_close_at} UTC.")
    return {
        "type": "manual_tp1_partial_breakeven",
        "partial_close_pct": partial,
        "move_sl_to_breakeven": bool(config.get("move_sl_to_breakeven", True)),
        "breakeven_sl": round(be_sl, 5),
        "round_trip_cost_pct": round(round_trip, 6),
        "tp1": float(tp1),
        "tp2": float(tp2) if tp2 is not None else None,
        "force_close_at_utc": force_close_at,
        "operator_instructions": instructions,
    }

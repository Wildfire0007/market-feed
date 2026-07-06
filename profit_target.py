"""Deterministic TP/SL builder for manual profit targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ProfitTargetResult:
    feasible: bool
    reason: Optional[str]
    entry: float
    sl: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    rr_tp1: Optional[float]
    meta: Dict[str, Any]


def _round_trip_cost(asset: str, cost_model: Dict[str, Any], default_model: Optional[Dict[str, Any]] = None) -> float:
    model = cost_model.get(asset) or cost_model.get(asset.upper()) or default_model or {}
    if str(model.get("type") or "pct").lower() != "pct":
        return 0.0
    value = model.get("round_trip_pct", model.get("pct", model.get("value", 0.0)))
    return max(0.0, float(value or 0.0))


def build_profit_target_levels(
    *,
    asset: str,
    side: str,
    entry: float,
    leverage: float,
    config: Dict[str, Any],
    asset_cost_model: Dict[str, Any],
    default_cost_model: Optional[Dict[str, Any]] = None,
    min_stoploss_pct: float = 0.0,
    atr5: Optional[float] = None,
    atr1h: Optional[float] = None,
    atr5_noise_mult: float = 0.4,
) -> ProfitTargetResult:
    margin = float(config.get("margin_usd", 100.0) or 100.0)
    net_min = float(config.get("net_tp1_usd_min", 10.0) or 10.0)
    lev = max(float(leverage or 0.0), 1e-9)
    entry = float(entry)
    required_net_move = net_min / (margin * lev)
    round_trip = _round_trip_cost(asset, asset_cost_model, default_cost_model)
    required_gross_move = required_net_move + round_trip
    tp2_rr_multiple = float(config.get("tp2_rr_multiple", 2.0) or 2.0)
    sl_rr_min = float(config.get("sl_rr_min", 1.5) or 1.5)
    max_atr1h_mult = float(config.get("max_required_move_atr1h_mult", 1.2) or 1.2)
    liquidation_cap_pct = min(0.95 / lev, 0.049)

    meta: Dict[str, Any] = {
        "required_net_move": required_net_move,
        "round_trip_cost_pct": round_trip,
        "required_gross_move": required_gross_move,
        "margin_usd": margin,
        "net_tp1_usd_min": net_min,
        "leverage": lev,
        "sl_rr_min": sl_rr_min,
        "liquidation_cap_pct": liquidation_cap_pct,
    }

    if entry <= 0:
        return ProfitTargetResult(False, "profit_target_infeasible", entry, None, None, None, None, meta)
    if atr1h is not None and atr1h > 0:
        atr1h_pct = float(atr1h) / entry
        ceiling = max_atr1h_mult * atr1h_pct
        meta.update({
            "atr1h_pct": atr1h_pct,
            "max_required_move_atr1h_mult": max_atr1h_mult,
            "required_move_atr1h_ceiling": ceiling,
            "required_move_over_ceiling": required_gross_move - ceiling,
        })
        if required_gross_move > ceiling:
            return ProfitTargetResult(False, "profit_target_infeasible", entry, None, None, None, None, meta)        
    if atr1h is not None and atr1h > 0 and required_gross_move > max_atr1h_mult * (float(atr1h) / entry):
        meta["atr1h_pct"] = float(atr1h) / entry
        meta["max_required_move_atr1h_mult"] = max_atr1h_mult
        return ProfitTargetResult(False, "profit_target_infeasible", entry, None, None, None, None, meta)

    tp1_dist = entry * required_gross_move
    max_sl_dist = tp1_dist / max(sl_rr_min, 1e-9)
    noise_dist = max(entry * float(min_stoploss_pct or 0.0), float(atr5 or 0.0) * float(atr5_noise_mult or 0.0))
    cap_dist = entry * liquidation_cap_pct
    min_sl_dist = max(noise_dist, 1e-9)
    if min_sl_dist > max_sl_dist or min_sl_dist > cap_dist:
        meta.update({"min_sl_distance": min_sl_dist, "max_sl_distance": max_sl_dist, "cap_sl_distance": cap_dist})
        return ProfitTargetResult(False, "profit_target_infeasible", entry, None, None, None, None, meta)

    sl_dist = min_sl_dist
    if side == "buy":
        tp1 = entry + tp1_dist
        tp2 = entry + tp1_dist * tp2_rr_multiple
        sl = entry - sl_dist
    elif side == "sell":
        tp1 = entry - tp1_dist
        tp2 = entry - tp1_dist * tp2_rr_multiple
        sl = entry + sl_dist
    else:
        return ProfitTargetResult(False, "profit_target_infeasible", entry, None, None, None, None, meta)
    rr = tp1_dist / sl_dist
    meta.update({"tp1_distance": tp1_dist, "sl_distance": sl_dist, "tp2_rr_multiple": tp2_rr_multiple})
    return ProfitTargetResult(True, None, entry, sl, tp1, tp2, rr, meta)

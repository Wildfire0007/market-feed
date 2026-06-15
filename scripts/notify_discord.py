#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — Enterprise Entry Only (v8)
Golyóálló, tiszta belépési jelzések pozíciómenedzsment blokkolás nélkül,
beépített SL/TP Auto-Correctorral és eToro kockázatkezeléssel.
"""
from __future__ import annotations
import importlib
import importlib.util
import json
import os
import sys
import fcntl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

requests = importlib.import_module("requests") if importlib.util.find_spec("requests") else None
BUDAPEST_TZ = ZoneInfo("Europe/Budapest")
from config import analysis_settings as settings

DRY_RUN = os.getenv("NOTIFY_DRY_RUN", "").lower() in {"1", "true", "yes"}
ENTRY_COOLDOWN_MINUTES = 30
DISCORD_WEBHOOK_URLS = [url.strip() for url in os.getenv("DISCORD_WEBHOOK_URL", "").replace("\\n", ",").split(",") if url.strip()]
DISCORD_NOTIFY_ASSETS = {p.strip().upper() for p in os.getenv("DISCORD_NOTIFY_ASSETS", "").split(",") if p.strip()}

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = Path(os.getenv("NOTIFY_PUBLIC_DIR", "")) if os.getenv("NOTIFY_PUBLIC_DIR") else BASE_DIR / "public"
if not PUBLIC_DIR.exists() and (BASE_DIR.parent / "public").exists():
    PUBLIC_DIR = BASE_DIR.parent / "public"

NOTIFY_LOCK_PATH = PUBLIC_DIR / ".notify_discord.lock"
COLOR_GREEN, COLOR_RED, COLOR_YELLOW = 0x2ECC71, 0xE74C3C, 0xF1C40F
LIFECYCLE_INBOX_PATH = PUBLIC_DIR / "_position_lifecycle_inbox.jsonl"


def _append_lifecycle_entry_event(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as h:
            data = json.load(h)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, payload: Dict[str, Any]):
    try:
        with path.open("w", encoding="utf-8") as h:
            json.dump(payload, h, ensure_ascii=False, indent=2)
    except Exception:
        pass


def safe_float(value: Any) -> Optional[float]:
    try:
        res = float(value)
        return res if res == res else None
    except Exception:
        return None


def format_price(price: Any) -> str:
    val = safe_float(price)
    if val is None:
        return "N/A"
    return f"{val:,.1f}" if val > 1000 else f"{val:.2f}" if val > 10 else f"{val:.5f}"


def to_utc_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_budapest_time(dt: datetime) -> str:
    return dt.astimezone(BUDAPEST_TZ).strftime("%Y-%m-%d %H:%M:%S CET/CEST")


def _asset_emoji(asset: str) -> str:
    key = str(asset or "").upper()
    if key in {"XAU", "XAUUSD", "GOLD_CFD", "GOLDCFD"}:
        return "🟡"
    if key in {"XAG", "XAGUSD", "SILVER", "SILVER_CFD"}:
        return "⚪"
    if key in {"USOIL", "OIL", "BRENT"}:
        return "🛢️"
    return "📌"


def _hu_reason(reason: str) -> str:
    reason_map = {
        "tp1_hit": "TP1 szint elérve.",
        "regime_shift": "Piaci rezsimváltás érzékelve.",
        "momentum_loss": "Lendület gyengül.",
        "structure_break": "Szerkezeti törés.",
        "volatility_spike": "Megugrott volatilitás.",
    }
    return reason_map.get(str(reason or "").strip().lower(), str(reason or "N/A"))


def _format_minutes(minutes: Optional[float]) -> str:
    if minutes is None:
        return "N/A"
    total = max(1, int(round(minutes)))
    if total < 60:
        return f"{total} perc"
    hours, mins = divmod(total, 60)
    return f"{hours}ó {mins}p" if mins else f"{hours} óra"


def _median(values: List[float]) -> Optional[float]:
    clean = sorted(v for v in values if v > 0)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _percentile(values: List[float], pct: float) -> Optional[float]:
    clean = sorted(v for v in values if v > 0)
    if not clean:
        return None
    idx = max(0, min(len(clean) - 1, int(round((len(clean) - 1) * pct))))
    return clean[idx]


def _load_close_series(asset_dir: Path, filename: str) -> Tuple[List[float], int]:
    payload = load_json(asset_dir / filename)
    rows = payload.get("values") or (payload.get("raw") or {}).get("values") or []
    parsed: List[Tuple[str, float]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        close = safe_float(row.get("close"))
        stamp = str(row.get("datetime") or row.get("timestamp") or "")
        if close is not None and stamp:
            parsed.append((stamp, close))
    parsed.sort(key=lambda item: item[0])
    interval = 5 if "5m" in filename else 1
    return [close for _, close in parsed[-120:]], interval


def _estimate_tp_eta_minutes(asset_dir: Path, direction: str, entry: float, tp1: float) -> Dict[str, Any]:
    closes, interval_minutes = _load_close_series(asset_dir, "klines_1m.json")
    if len(closes) < 12:
        closes, interval_minutes = _load_close_series(asset_dir, "klines_5m.json")
    if len(closes) < 3:
        return {"available": False, "reason": "missing_price_history"}

    favorable: List[float] = []
    absolute: List[float] = []
    for prev, cur in zip(closes, closes[1:]):
        delta = cur - prev
        absolute.append(abs(delta) / interval_minutes)
        if direction == "buy" and delta > 0:
            favorable.append(delta / interval_minutes)
        elif direction == "sell" and delta < 0:
            favorable.append(abs(delta) / interval_minutes)

    distance = abs(tp1 - entry)
    fast_speed = _percentile(favorable, 0.75)
    base_speed = _median(favorable) or _median(absolute)
    conservative_speed = _percentile(favorable, 0.25) or _median(absolute)
    if not base_speed:
        return {"available": False, "reason": "flat_price_history"}

    return {
        "available": True,
        "fast_minutes": distance / fast_speed if fast_speed else None,
        "base_minutes": distance / base_speed,
        "conservative_minutes": distance / conservative_speed if conservative_speed else None,
        "source_interval_minutes": interval_minutes,
    }


def _manual_trade_model_for_asset(asset_name: str, manual_trade_model: Dict[str, Any]) -> Dict[str, Any]:
    model = dict(manual_trade_model)
    overrides = manual_trade_model.get("asset_overrides")
    asset_override = overrides.get(asset_name.upper()) if isinstance(overrides, dict) else None
    if isinstance(asset_override, dict):
        model.update(asset_override)
    leverage_map = getattr(settings, "LEVERAGE", {}) or {}
    asset_leverage = safe_float(leverage_map.get(asset_name.upper()))
    if asset_leverage is not None and "leverage" not in (asset_override or {}):
        model["leverage"] = asset_leverage
    return model


def build_expected_trade_outcome(
    asset_dir: Path,
    asset_name: str,
    data: Dict[str, Any],
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    manual_trade_model: Dict[str, Any],
) -> Dict[str, Any]:
    model = _manual_trade_model_for_asset(asset_name, manual_trade_model)
    equity_usd = safe_float(model.get("equity_usd")) or 100.0
    leverage = safe_float(model.get("leverage")) or 20.0
    tp1_close_fraction = safe_float(model.get("tp1_close_fraction")) or 1.0
    min_net_usd = safe_float(model.get("tp1_min_net_usd")) or 10.0
    eta_min = safe_float(model.get("eta_min_minutes")) or 5.0
    eta_max = safe_float(model.get("eta_max_minutes")) or 240.0
    max_chase_r = safe_float(model.get("max_chase_r")) or 0.2
    valid_for = safe_float(model.get("signal_valid_minutes")) or 10.0

    cost_pct = float((settings.ASSET_COST_MODEL.get(asset_name) or {}).get("round_trip_pct", 0.0))
    gross_pct = abs(entry - tp1) / entry if entry else 0.0
    net_pct = gross_pct - cost_pct
    notional = equity_usd * leverage
    tp1_net_usd = net_pct * notional * tp1_close_fraction
    risk = abs(entry - sl)
    spot = safe_float((data.get("spot") or {}).get("price")) or entry
    chase_r = 0.0
    if risk > 0:
        if direction == "buy" and spot > entry:
            chase_r = (spot - entry) / risk
        elif direction == "sell" and spot < entry:
            chase_r = (entry - spot) / risk

    eta = _estimate_tp_eta_minutes(asset_dir, direction, entry, tp1)
    eta_base = safe_float(eta.get("base_minutes"))
    eta_gate = bool(eta.get("available") and eta_base is not None and eta_min <= eta_base <= eta_max)
    profit_gate = tp1_net_usd >= min_net_usd
    no_chase_gate = chase_r <= max_chase_r

    return {
        "equity_usd": round(equity_usd, 2),
        "leverage": round(leverage, 2),
        "notional_usd": round(notional, 2),
        "tp1_net_usd": round(tp1_net_usd, 2),
        "min_required_net_usd": round(min_net_usd, 2),
        "tp1_net_pct": round(net_pct, 6),
        "eta_minutes_fast": round(eta["fast_minutes"], 1) if eta.get("fast_minutes") else None,
        "eta_minutes_base": round(eta_base, 1) if eta_base is not None else None,
        "eta_minutes_conservative": round(eta["conservative_minutes"], 1) if eta.get("conservative_minutes") else None,
        "eta_source_interval_minutes": eta.get("source_interval_minutes"),
        "eta_available": bool(eta.get("available")),
        "eta_unavailable_reason": eta.get("reason"),
        "valid_for_minutes": round(valid_for, 1),
        "max_chase_r": round(max_chase_r, 3),
        "current_chase_r": round(chase_r, 3),
        "max_entry_price": round(entry + risk * max_chase_r, 6) if direction == "buy" and risk > 0 else None,
        "min_entry_price": round(entry - risk * max_chase_r, 6) if direction == "sell" and risk > 0 else None,
        "profit_gate_pass": profit_gate,
        "eta_gate_pass": eta_gate,
        "no_chase_pass": no_chase_gate,
        "passes": profit_gate and eta_gate and no_chase_gate,
    }


def check_and_notify() -> None:
    if not PUBLIC_DIR.exists():
        return
    manual_trade_model = settings.MANUAL_TRADE_MODEL or {}
    tp1_min_net_usd = safe_float(manual_trade_model.get("tp1_min_net_usd")) or 10.0
    sl_risk_usd = safe_float(manual_trade_model.get("sl_risk_usd")) or 50.0
    
    notify_state_path = PUBLIC_DIR / "_notify_state.json"
    notify_state = load_json(notify_state_path)
    notify_changed = False

    for asset_dir in [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]:    
        asset_name = asset_dir.name
        if DISCORD_NOTIFY_ASSETS and asset_name.upper() not in DISCORD_NOTIFY_ASSETS:
            continue

        data = load_json(asset_dir / "signal.json")
        if not data:
            continue

        signal = str(data.get("signal") or "no entry").lower()
        if signal not in {"buy", "sell", "precision_arming"}:
            continue

        entry, sl, tp1, tp2 = safe_float(data.get("entry")), safe_float(data.get("sl")), safe_float(data.get("tp1")), safe_float(data.get("tp2"))
        order_type, direction = str(data.get("order_type") or "MARKET").upper(), signal
        
        if signal == "precision_arming":
            plan = data.get("precision_plan") or {}
            direction = str(plan.get("direction") or "buy").lower()
            order_type = str(plan.get("order_type") or "LIMIT").upper()
            entry = safe_float(plan.get("entry") or entry)
            sl = safe_float(plan.get("stop_loss") or sl)
            tp1 = safe_float(plan.get("take_profit_1") or tp1)
            tp2 = safe_float(plan.get("take_profit_2") or tp2)

        if direction not in {"buy", "sell"}:
            continue

        # AUTO-CORRECT (No Max Delta Limits)
        if entry is not None and sl is not None:
            if direction == "sell" and sl < entry:
                sl = entry + (entry - sl)
            elif direction == "buy" and sl > entry:
                sl = entry - (sl - entry)
        if entry is not None and tp1 is not None:
            if direction == "sell" and tp1 > entry:
                tp1 = entry - (tp1 - entry)
            elif direction == "buy" and tp1 < entry:
                tp1 = entry + (entry - tp1)

        if None in (entry, sl, tp1):
            continue

        expected = build_expected_trade_outcome(asset_dir, asset_name, data, direction, entry, sl, tp1, manual_trade_model)
        tp1_net_usd = safe_float(expected.get("tp1_net_usd")) or 0.0
        if not expected.get("passes"):
            continue

        units_text = f"{sl_risk_usd / abs(entry - sl):.2f} Egység (Units)" if abs(entry - sl) > 0 else "N/A"

        now_dt = datetime.now(timezone.utc)
        asset_state = notify_state.get(asset_name) or {}
        entry_sig = f"{direction}_{order_type}"

        if asset_state.get("last_entry_signature") == entry_sig and asset_state.get("last_entry_sent_utc"):
            try:
                if now_dt - datetime.fromisoformat(asset_state["last_entry_sent_utc"].replace("Z", "+00:00")) < timedelta(minutes=ENTRY_COOLDOWN_MINUTES):
                    continue
            except Exception:
                pass

        prefix, color = ("🟡", COLOR_YELLOW) if order_type in ["LIMIT", "STOP"] else (("🟢", COLOR_GREEN) if direction == "buy" else ("🔴", COLOR_RED))
        title = f"{prefix} NYISS {'LONG' if direction == 'buy' else 'SHORT'} – {'BUY' if direction == 'buy' else 'SELL'} {order_type} @ {format_price(entry)}"

        p_score = safe_float(data.get("probability_raw"))
        reasons = "\n".join([f"• {_hu_reason(r)}" for r in (data.get("reasons") or [])[:2]]) or "• Rendszer jelzés"
        reasons_text = f"P Score: **{p_score:.1f}** (Erősség)\n{reasons}" if p_score else reasons
        eta_text = (
            f"Gyors: `{_format_minutes(safe_float(expected.get('eta_minutes_fast')))}`\n"
            f"Normál: `{_format_minutes(safe_float(expected.get('eta_minutes_base')))}`\n"
            f"Konzervatív: `{_format_minutes(safe_float(expected.get('eta_minutes_conservative')))}`\n"
            f"Jel érvényessége: `{_format_minutes(safe_float(expected.get('valid_for_minutes')))}`"
        )
        entry_limit_text = (
            f"Ne nyiss, ha spot > `{format_price(expected.get('max_entry_price'))}`"
            if direction == "buy"
            else f"Ne nyiss, ha spot < `{format_price(expected.get('min_entry_price'))}`"
        )
        
        embed = {
            "title": title,
            "description": f"{_asset_emoji(asset_name)} Eszköz: `{asset_name}`",
            "color": color,
            "fields": [
                {"name": "📊 Árfolyam", "value": f"Spot ár: `{format_price(safe_float((data.get('spot') or {}).get('price')))}`\nBelépő: `{format_price(entry)}`", "inline": False},
                {"name": "🎯 Profit cél", "value": f"Várható nettó TP1: `+${tp1_net_usd:.2f}`\nMinimum: `${tp1_min_net_usd:.2f}`\nTőkeáttételes méret: `${expected.get('notional_usd'):.2f}`", "inline": False},
                {"name": "⏱️ Várható idő TP1-ig", "value": eta_text, "inline": False},
                {"name": "⚙️ Paraméterek az eToro-hoz", "value": f"MÉRET: `{units_text}` ({sl_risk_usd} USD kockázat)\nSL: `{format_price(sl)}`\nTP1: `{format_price(tp1)}`" + (f"\nTP2: `{format_price(tp2)}`" if tp2 else ""), "inline": False},
                {"name": "🎯 Belépési pontosság", "value": f"Aktuális chase: `{expected.get('current_chase_r')}R`\n{entry_limit_text}", "inline": False},                
                {"name": "💡 Indoklás", "value": reasons_text, "inline": False},
                {"name": "🕒 Időbélyeg", "value": f"`{format_budapest_time(now_dt)}` (Budapest)", "inline": False},
            ],
            "footer": {"text": f"Signal • Budapest: {format_budapest_time(now_dt)} • Várakozás (30 perc csend indítva)"},
        }

        if not DRY_RUN and requests and DISCORD_WEBHOOK_URLS:
            for url in DISCORD_WEBHOOK_URLS:
                try:
                    requests.post(url, json={"embeds": [embed]}, timeout=5)
                except Exception:
                    pass

        asset_state.update({"last_entry_signature": entry_sig, "last_entry_sent_utc": to_utc_iso(now_dt)})
        notify_state[asset_name] = asset_state
        notify_changed = True
        _append_lifecycle_entry_event(LIFECYCLE_INBOX_PATH, {
            "event": "entry_signal",
            "ts_utc": to_utc_iso(now_dt),
            "asset": asset_name,
            "signal": signal,
            "direction": direction,
            "order_type": order_type,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "entry_signature": entry_sig,
            "expected_trade_outcome": expected,    
        })

    if notify_changed and not DRY_RUN:
        save_json(notify_state_path, notify_state)


if __name__ == "__main__":
    if not PUBLIC_DIR.exists():
        sys.exit(0)
    with NOTIFY_LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(0)
        check_and_notify()

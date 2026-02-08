#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — Manual Trader Friendly (v7)
Belépési jelzések átlátható, kézi kereskedő-barát összegzése.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from zoneinfo import ZoneInfo

requests = None
if importlib.util.find_spec("requests") is not None:
    requests = importlib.import_module("requests")

BUDAPEST_TZ = ZoneInfo("Europe/Budapest")

from config import analysis_settings as settings


DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
DRY_RUN = os.getenv("NOTIFY_DRY_RUN", "").lower() in {"1", "true", "yes"}
ENTRY_COOLDOWN_MINUTES = 5
EXIT_NOTIFY_COOLDOWN_MINUTES = 10

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = Path(os.getenv("NOTIFY_PUBLIC_DIR", "")) if os.getenv("NOTIFY_PUBLIC_DIR") else None
if not PUBLIC_DIR:
    if (BASE_DIR / "public").exists():
        PUBLIC_DIR = BASE_DIR / "public"
    elif (BASE_DIR.parent / "public").exists():
        PUBLIC_DIR = BASE_DIR.parent / "public"
    else:
        PUBLIC_DIR = BASE_DIR / "public"


ICON_BUY_MARKET = "🟢"
ICON_SELL_MARKET = "🔴"
ICON_BUY_LIMIT = "🔵"
ICON_SELL_LIMIT = "🟠"

COLOR_GREEN = 0x2ECC71
COLOR_RED = 0xE74C3C
COLOR_BLUE = 0x3498DB
COLOR_ORANGE = 0xE67E22
COLOR_YELLOW = 0xF1C40F


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Hiba: {exc}")


def safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def format_price(price: Any) -> str:
    value = safe_float(price)
    if value is None:
        return "N/A"
    if value > 1000:
        return f"{value:,.1f}"
    if value > 10:
        return f"{value:.2f}"
    return f"{value:.5f}"


def to_utc_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    

def get_budapest_time(utc_iso_string: Optional[str]) -> str:
    if not utc_iso_string or utc_iso_string == "-":
        return "N/A"
    try:
        dt_utc = datetime.fromisoformat(str(utc_iso_string).replace("Z", "+00:00"))
        dt_bp = dt_utc.astimezone(BUDAPEST_TZ)
        return dt_bp.strftime("%H:%M:%S")
    except Exception:
        return "Idő?"


def send_discord_embed(embed_data: Dict[str, Any]) -> None:
    if DRY_RUN or not DISCORD_WEBHOOK_URL:
        print(f"[DRY RUN] Embed title: {embed_data.get('title')}")
        return
    if requests is None:
        print("Hiba: a 'requests' modul hiányzik; webhook küldés kihagyva.")
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed_data]}, timeout=5)
    except Exception as exc:
        print(f"Hiba: {exc}")


def round_trip_pct(asset: str) -> float:
    model = settings.ASSET_COST_MODEL.get(asset) or settings.DEFAULT_COST_MODEL
    if str(model.get("type") or "pct").lower() == "pip":
        return 0.0
    return float(model.get("round_trip_pct", 0.0) or 0.0)


def calc_gross_pct(entry: float, target: float, side: str) -> Optional[float]:
    if entry <= 0:
        return None
    if side == "sell":
        return (entry - target) / entry
    return (target - entry) / entry


def calc_points(entry: float, target: float) -> float:
    return abs(target - entry)


def calc_rr(entry: float, sl: float, target: float) -> Optional[float]:
    risk = abs(entry - sl)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    return reward / risk


def _entry_signature(signal: str, entry: Optional[float], order_type: str) -> Dict[str, Any]:
    return {
        "signal": signal,
        "entry": round(entry, 2) if entry is not None else None,
        "order_type": order_type,
    }


def _exit_signature(exit_signal: Dict[str, Any]) -> Dict[str, Any]:
    state = str(exit_signal.get("state") or exit_signal.get("action") or "").lower()
    direction = str(exit_signal.get("direction") or "").lower()
    reasons = [str(r) for r in (exit_signal.get("reasons") or []) if r]
    return {
        "state": state,
        "direction": direction,
        "reasons": reasons[:4],
    }


def _format_biases(biases: Dict[str, Any]) -> str:
    bias_4h = str(biases.get("adjusted_4h") or "n/a")
    bias_1h = str(biases.get("adjusted_1h") or "n/a")
    bias_5m = str(biases.get("adjusted_5m") or "n/a")
    return f"4H: `{bias_4h}` | 1H: `{bias_1h}` | 5m: `{bias_5m}`"


def _spread_gate_status(gates: Dict[str, Any]) -> str:
    missing = set(str(item) for item in (gates.get("missing") or []) if item)
    return "BLOCK" if "spread_guard" in missing else "OK"


def check_and_notify() -> None:
    if not PUBLIC_DIR.exists():
        return

    manual_trade_model = settings.MANUAL_TRADE_MODEL or {}
    equity_usd = safe_float(manual_trade_model.get("equity_usd")) or 100.0
    leverage = safe_float(manual_trade_model.get("leverage")) or 20.0
    tp1_close_fraction = safe_float(manual_trade_model.get("tp1_close_fraction")) or 1.0
    tp1_min_net_usd = safe_float(manual_trade_model.get("tp1_min_net_usd")) or 5.0
    tp2_min_net_usd = safe_float(manual_trade_model.get("tp2_min_net_usd")) or 10.0
    sl_risk_usd = safe_float(manual_trade_model.get("sl_risk_usd")) or 10.0
    
    notify_state_path = PUBLIC_DIR / "_notify_state.json"
    notify_state = load_json(notify_state_path)
    if not isinstance(notify_state, dict):
        notify_state = {}
    notify_state_changed = False

    assets = [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    
    for asset_dir in assets:
        asset_name = asset_dir.name
        signal_path = asset_dir / "signal.json"        
        data = load_json(signal_path)
        if not data:
            continue

        signal = str(data.get("signal") or "no entry").lower()
        exit_signal = data.get("position_exit_signal") or data.get("active_position_meta", {}).get("exit_signal")
        if signal not in {"buy", "sell", "precision_arming"} and not exit_signal:
            continue

        biases = data.get("biases") if isinstance(data.get("biases"), dict) else {}
        gates = data.get("gates") if isinstance(data.get("gates"), dict) else {}
        effective_thresholds = (
            data.get("effective_thresholds")
            if isinstance(data.get("effective_thresholds"), dict)
            else {}
        )
        alignment_state = str(data.get("alignment_state") or "MIXED").upper()
        alignment_gate_note = "OK"
        if alignment_state == "COUNTER":
            alignment_gate_note = "BLOCK"

        atr1h = safe_float(data.get("atr1h"))
        probability = safe_float(data.get("probability"))
        p_score = safe_float(data.get("probability_raw"))
        reasons = [str(r) for r in (data.get("reasons") or []) if r][:4]

        entry = safe_float(data.get("entry"))
        sl = safe_float(data.get("sl"))
        tp1 = safe_float(data.get("tp1"))
        tp2 = safe_float(data.get("tp2"))

        order_type = "MARKET"
        direction = signal
        if signal == "precision_arming":
            plan = data.get("precision_plan") if isinstance(data.get("precision_plan"), dict) else {}
            direction = str(plan.get("direction") or data.get("signal") or "buy").lower()
            order_type = "LIMIT"
            entry = safe_float(plan.get("entry") or entry)
            sl = safe_float(plan.get("stop_loss") or sl)
            tp1 = safe_float(plan.get("take_profit_1") or tp1)
            tp2 = safe_float(plan.get("take_profit_2") or tp2)

        if direction not in {"buy", "sell"}:
            direction = "buy"

        if exit_signal:
            exit_state = str(exit_signal.get("state") or exit_signal.get("action") or "").lower()            
            exit_signature = _exit_signature(exit_signal)            
            now_dt = datetime.now(timezone.utc)
            asset_state = notify_state.get(asset_name) if isinstance(notify_state.get(asset_name), dict) else {}
            last_exit_signature = asset_state.get("last_exit_signature")
            last_exit_sent_utc = asset_state.get("last_exit_sent_utc")
            if last_exit_signature == exit_signature and last_exit_sent_utc:
                try:
                    last_exit_dt = datetime.fromisoformat(str(last_exit_sent_utc).replace("Z", "+00:00"))
                except Exception:
                    last_exit_dt = None
                if last_exit_dt and now_dt - last_exit_dt < timedelta(minutes=EXIT_NOTIFY_COOLDOWN_MINUTES):                    
                    print(f" -> EXIT DEDUP: {asset_name} ({EXIT_NOTIFY_COOLDOWN_MINUTES}m)")
                    continue
            state_label = {
                "hard_exit": "HARD EXIT",
                "scale_out": "RÉSZLEGES ZÁRÁS",
                "tighten_stop": "SL SZŰKÍTÉS",
            }.get(exit_state, "POZÍCIÓ MENEDZSMENT")
            embed = {
                "title": f"⚠️ {state_label}: {asset_name}",
                "description": "**Pozíció menedzsment jelzés**",
                "color": COLOR_ORANGE if exit_state != "hard_exit" else COLOR_RED,
                "fields": [
                    {"name": "Irány", "value": str(exit_signal.get("direction") or "n/a").upper(), "inline": True},
                    {"name": "Okok", "value": "\n".join(f"• {r}" for r in reasons) or "N/A", "inline": False},
                ],
                "footer": {"text": f"Exit • {asset_name}"},
            }
            send_discord_embed(embed)
            asset_state["last_exit_signature"] = exit_signature
            asset_state["last_exit_sent_utc"] = to_utc_iso(now_dt)
            notify_state[asset_name] = asset_state
            notify_state_changed = True
            continue

        if entry is None or sl is None or tp1 is None:
            print(f"{asset_name}: hiányzó entry/SL/TP1 — jelzés eldobva.")
            continue

        notional = equity_usd * leverage
        rt_pct = round_trip_pct(asset_name)
        tp1_gross_pct = calc_gross_pct(entry, tp1, direction) or 0.0
        tp2_gross_pct = calc_gross_pct(entry, tp2, direction) if tp2 is not None else None
        tp1_net_pct = tp1_gross_pct - rt_pct
        tp2_net_pct = (tp2_gross_pct - rt_pct) if tp2_gross_pct is not None else None

        tp1_net_usd = tp1_net_pct * notional * tp1_close_fraction
        tp2_net_usd = (tp2_net_pct * notional) if tp2_net_pct is not None else None
        sl_gross_pct = calc_gross_pct(entry, sl, "sell" if direction == "buy" else "buy") or 0.0
        sl_usd = sl_gross_pct * notional

        rr_tp1 = calc_rr(entry, sl, tp1)
        rr_tp2 = calc_rr(entry, sl, tp2) if tp2 is not None else None
        rr_tp1_display = f"{rr_tp1:.2f}" if rr_tp1 is not None else "n/a"
        rr_tp2_display = f"{rr_tp2:.2f}" if rr_tp2 is not None else "n/a"

        counter_min_net = max(tp2_min_net_usd, tp1_min_net_usd * 1.5)
        allow_entry = tp1_net_usd >= tp1_min_net_usd
        if alignment_state == "COUNTER":
            allow_entry = tp1_net_usd >= counter_min_net

        asset_state = notify_state.get(asset_name) if isinstance(notify_state.get(asset_name), dict) else {}
        now_dt = datetime.now(timezone.utc)
        entry_signature = _entry_signature(signal, entry, order_type)
        last_entry_signature = asset_state.get("last_entry_signature")
        last_entry_sent_utc = asset_state.get("last_entry_sent_utc")
        if allow_entry and last_entry_signature == entry_signature and last_entry_sent_utc:
            try:
                last_entry_dt = datetime.fromisoformat(str(last_entry_sent_utc).replace("Z", "+00:00"))
            except Exception:
                last_entry_dt = None
            if last_entry_dt and now_dt - last_entry_dt < timedelta(minutes=ENTRY_COOLDOWN_MINUTES):
                print(f" -> ENTRY DEDUP: {asset_name} ({ENTRY_COOLDOWN_MINUTES}m)")
                allow_entry = False

        if not allow_entry:
            print(
                f"{asset_name}: belépő blokkolva (TP1 net USD {tp1_net_usd:.2f} / min {tp1_min_net_usd}, alignment {alignment_state})."
            )
            continue

        icon = ICON_BUY_MARKET if direction == "buy" else ICON_SELL_MARKET
        color = COLOR_GREEN if direction == "buy" else COLOR_RED
        if order_type == "LIMIT":
            icon = ICON_BUY_LIMIT if direction == "buy" else ICON_SELL_LIMIT
            color = COLOR_BLUE if direction == "buy" else COLOR_ORANGE
        if alignment_state == "COUNTER":
            color = COLOR_YELLOW

        spot = data.get("spot") if isinstance(data.get("spot"), dict) else {}
        spot_price = spot.get("price")
        time_label = get_budapest_time(spot.get("utc"))
        spread_status = _spread_gate_status(gates)

        fields = [
            {
                "name": "Setup",
                "value": (
                    f"Entry: `{format_price(entry)}`\n"
                    f"SL: `{format_price(sl)}`\n"
                    f"TP1: `{format_price(tp1)}`\n"
                    f"TP2: `{format_price(tp2)}`"
                ),
                "inline": True,
            },
            {
                "name": "Δ ár (points)",
                "value": (
                    f"TP1: `{calc_points(entry, tp1):.4f}`\n"
                    f"TP2: `{calc_points(entry, tp2):.4f}`" if tp2 is not None else "TP2: `n/a`"
                ),
                "inline": True,
            },
            {
                "name": "TP1/TP2 % (gross/net)",
                "value": (
                    f"TP1: `{tp1_gross_pct*100:.2f}% / {tp1_net_pct*100:.2f}%`\n"
                    f"TP2: `{(tp2_gross_pct or 0)*100:.2f}% / {(tp2_net_pct or 0)*100:.2f}%`"
                    if tp2 is not None
                    else f"TP1: `{tp1_gross_pct*100:.2f}% / {tp1_net_pct*100:.2f}%`"
                ),
                "inline": True,
            },
            {
                "name": "PnL (USD) & R",
                "value": (
                    f"TP1: `${tp1_net_usd:.2f}` | R: `{rr_tp1_display}`\n"
                    f"TP2: `${tp2_net_usd:.2f}` | R: `{rr_tp2_display}`\n"
                    f"SL: `-${sl_usd:.2f}` | Risk cap: `${sl_risk_usd:.2f}`"
                    if tp2_net_usd is not None
                    else f"TP1: `${tp1_net_usd:.2f}` | R: `{rr_tp1_display}`\nSL: `-${sl_usd:.2f}` | Risk cap: `${sl_risk_usd:.2f}`"
                ),
                "inline": False,
            },
            {
                "name": "Bias / ATR / Spread",
                "value": (
                    f"{_format_biases(biases)}\n"
                    f"ATR(1H): `{atr1h:.4f}`\n"
                    f"Spread gate: `{spread_status}`"
                    if atr1h is not None
                    else f"{_format_biases(biases)}\nATR(1H): `n/a`\nSpread gate: `{spread_status}`"
                ),
                "inline": False,
            },
            {
                "name": "Alignment",
                "value": f"State: `{alignment_state}` | Gate: `{alignment_gate_note}`",
                "inline": True,
            },
            {
                "name": "Probability / P-score",
                "value": (
                    f"Prob: `{probability:.0f}%`\nP-score: `{p_score:.0f}`"
                    if probability is not None and p_score is not None
                    else f"Prob: `{probability}` | P-score: `{p_score}`"
                ),
                "inline": True,
            },
        ]

        if reasons:
            fields.append(
                {
                    "name": "Miért",
                    "value": "\n".join(f"• {reason}" for reason in reasons),
                    "inline": False,
                }
            )

        embed = {
            "title": f"{icon} {order_type} {direction.upper()} — {asset_name}",
            "description": f"Spot: `{format_price(spot_price)}` • 🕒 {time_label}",
            "color": color,
            "fields": fields,
            "footer": {"text": f"{asset_name} • Manual trade model"},
        }

        send_discord_embed(embed)
        asset_state["last_entry_signature"] = entry_signature
        asset_state["last_entry_sent_utc"] = to_utc_iso(now_dt)
        notify_state[asset_name] = asset_state
        notify_state_changed = True

    if notify_state_changed and not DRY_RUN:  
        save_json(notify_state_path, notify_state)


if __name__ == "__main__":
    check_and_notify()

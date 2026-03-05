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
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

requests = None
if importlib.util.find_spec("requests") is not None:
    requests = importlib.import_module("requests")

BUDAPEST_TZ = ZoneInfo("Europe/Budapest")

from config import analysis_settings as settings


DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
DRY_RUN = os.getenv("NOTIFY_DRY_RUN", "").lower() in {"1", "true", "yes"}
ENTRY_COOLDOWN_MINUTES = 30
EXIT_NOTIFY_COOLDOWN_MINUTES = 30
DISCORD_NOTIFY_ASSETS = {"GOLD_CFD", "XAGUSD", "USOIL"}

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
COLOR_HARD_EXIT = 0xB71C1C
COLOR_BLUE = 0x3498DB
COLOR_ORANGE = 0xE67E22
COLOR_YELLOW = 0xF1C40F

HARD_GATE_NAMES = {
    "spread_guard",
    "volatility_guard",
    "atr_guard",
    "order_flow_guard",
    "structure_guard",
    "breakout_guard",
    "false_breakout_guard",
    "metal_bias_5m_alignment",
}


def _is_hard_gate_blocker(gate_name: str) -> bool:
    key = str(gate_name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return key in HARD_GATE_NAMES or key.startswith((
        "spread_",
        "volatility_",
        "atr_",
        "order_flow_",
        "structure_",
        "breakout_",
        "false_breakout_",
    ))


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


def _entry_signature(direction: str, order_type: str) -> Dict[str, Any]:
    return {
        "direction": direction,
        "order_type": order_type,
    }


def _entry_levels_signature(
    entry: Optional[float],
    sl: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
) -> Dict[str, Any]:
    def _bucket(value: Optional[float]) -> Optional[float]:
        return round(value, 5) if value is not None else None

    return {
        "entry": _bucket(entry),
        "sl": _bucket(sl),
        "tp1": _bucket(tp1),
        "tp2": _bucket(tp2),
    }


def _exit_signature(exit_signal: Dict[str, Any]) -> Dict[str, Any]:
    state = str(exit_signal.get("state") or exit_signal.get("action") or "").lower()
    direction = str(exit_signal.get("direction") or "").lower()
    category = str(exit_signal.get("category") or "").lower()
    trigger_price = safe_float(exit_signal.get("trigger_price"))
    trigger_bucket = round(trigger_price, 3) if trigger_price is not None else None
    return {
        "state": state,
        "direction": direction,
        "category": category,
        "trigger_bucket": trigger_bucket,
    }


def _format_biases(biases: Dict[str, Any]) -> str:
    bias_4h = str(biases.get("adjusted_4h") or "n/a")
    bias_1h = str(biases.get("adjusted_1h") or "n/a")
    bias_5m = str(biases.get("adjusted_5m") or "n/a")
    return f"4H: `{bias_4h}` | 1H: `{bias_1h}` | 5m: `{bias_5m}`"


def _spread_gate_status(gates: Dict[str, Any]) -> str:
    missing = set(str(item) for item in (gates.get("missing") or []) if item)
    return "BLOCK" if "spread_guard" in missing else "OK"


def _hu_reason(reason: str) -> str:
    reason_map = {
        "tp1_hit": "TP1 szint elérve.",
        "regime_shift": "Piaci rezsimváltás érzékelve.",
        "momentum_loss": "A lendület gyengül.",
        "structure_break": "Szerkezeti törés történt.",
        "volatility_spike": "Megugrott a volatilitás.",
    }
    key = str(reason or "").strip().lower()
    return reason_map.get(key, str(reason or "N/A"))


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

    # assets = [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    assets = [
        d
        for d in PUBLIC_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name in DISCORD_NOTIFY_ASSETS
    ]
    
    
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

        if alignment_state in {"MIXED", "COUNTER"} and signal != "precision_arming" and not exit_signal:
            print(
                f"{asset_name}: {alignment_state} piac — jelzés némítva (csak precision_arming mehet át)."
            )
            continue

        missing_gates = [str(item) for item in (gates.get("missing") or []) if item]
        hard_missing_gates = [gate for gate in missing_gates if _is_hard_gate_blocker(gate)]
        soft_missing_gates = [gate for gate in missing_gates if gate not in hard_missing_gates]
        if hard_missing_gates and not exit_signal:
            print(
                f"{asset_name}: belépő blokkolva hard védelmi kapun ({', '.join(hard_missing_gates)})."
            )
            continue
        if soft_missing_gates and not exit_signal:
            print(
                f"{asset_name}: soft gate figyelmeztetés ({', '.join(soft_missing_gates)}), jelzés továbbengedve."
            )
        
        atr1h = safe_float(data.get("atr1h"))
        probability = safe_float(data.get("probability"))
        p_score = safe_float(data.get("probability_raw"))
        reasons = [str(r) for r in (data.get("reasons") or []) if r][:4]

        entry = safe_float(data.get("entry"))
        sl = safe_float(data.get("sl"))
        tp1 = safe_float(data.get("tp1"))
        tp2 = safe_float(data.get("tp2"))

        order_type = str(data.get("entry_order_type") or data.get("order_type") or "MARKET").upper()
        direction = signal
        if signal == "precision_arming":
            plan = data.get("precision_plan") if isinstance(data.get("precision_plan"), dict) else {}
            direction = str(plan.get("direction") or data.get("signal") or "buy").lower()
            order_type = str(plan.get("order_type") or order_type or "LIMIT").upper()
            entry = safe_float(plan.get("entry") or entry)
            sl = safe_float(plan.get("stop_loss") or sl)
            tp1 = safe_float(plan.get("take_profit_1") or tp1)
            tp2 = safe_float(plan.get("take_profit_2") or tp2)

        if order_type not in {"LIMIT", "MARKET", "STOP"}:
            order_type = "MARKET"
        
        if direction not in {"buy", "sell"}:
            print(f"{asset_name}: bizonytalan irány ({direction}) — jelzés némítva.")
            continue

        spot = data.get("spot") if isinstance(data.get("spot"), dict) else {}
        spot_price = spot.get("price")
        
        if exit_signal:
            exit_state = str(exit_signal.get("state") or exit_signal.get("action") or "").lower()            
            exit_signature = _exit_signature(exit_signal)
            exit_reasons = [str(r) for r in (exit_signal.get("reasons") or []) if r][:4]
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
                "scale_out": "🟠 TEENDŐ MOST: RÉSZZÁRÁS 50% ÉS STOP NULLÁBA",
                "hard_exit": "🔴 TEENDŐ MOST: AZONNAL ZÁRD A TELJES POZÍCIÓT!",
                "tighten_stop": "🟠 TEENDŐ MOST: SZŰKÍTSD A STOP-LOSST!",
            }.get(exit_state, "🟠 TEENDŐ MOST: POZÍCIÓ MENEDZSMENT")
            embed = {
                "title": state_label,
                "description": f"Eszköz: `{asset_name}`",
                "color": COLOR_ORANGE if exit_state != "hard_exit" else COLOR_HARD_EXIT,
                "fields": [                    
                    {
                        "name": "📊 Árfolyam & Szintek",    
                        "value": (
                            f"Spot: `{format_price(spot_price)}`\n"
                            f"Eredeti Entry: `{format_price(entry)}`"
                        ),
                        "inline": False,
                    },       
                    {
                        "name": "💡 Javaslat oka",
                        "value": "\n".join(f"• {_hu_reason(r)}" for r in exit_reasons) or "• N/A",
                        "inline": False,
                    },
                ],
                "footer": {"text": f"Exit • {asset_name}"},
            }
            send_discord_embed(embed)
            asset_state["last_exit_signature"] = exit_signature
            asset_state["last_exit_sent_utc"] = to_utc_iso(now_dt)
            notify_state[asset_name] = asset_state
            notify_state_changed = True
            continue

        tp1_net_usd = 0.0
        if entry is None or sl is None or tp1 is None:
            print(f"{asset_name}: hiányzó entry/SL/TP1 — jelzés eldobva.")
            continue

        notional = equity_usd * leverage
        rt_pct = round_trip_pct(asset_name)
        tp1_gross_pct = calc_gross_pct(entry, tp1, direction) or 0.0
        tp2_gross_pct = calc_gross_pct(entry, tp2, direction) if tp2 is not None else None
        tp1_net_pct = tp1_gross_pct - rt_pct
        tp1_net_usd = tp1_net_pct * notional * tp1_close_fraction

        counter_min_net = max(tp2_min_net_usd, tp1_min_net_usd * 1.5)
        allow_entry = tp1_net_usd >= tp1_min_net_usd
        if alignment_state == "COUNTER":
            allow_entry = tp1_net_usd >= counter_min_net

        asset_state = notify_state.get(asset_name) if isinstance(notify_state.get(asset_name), dict) else {}
        now_dt = datetime.now(timezone.utc)
        entry_signature = _entry_signature(direction, order_type)
        entry_levels_signature = _entry_levels_signature(entry, sl, tp1, tp2)
        last_entry_signature = asset_state.get("last_entry_signature")
        last_entry_levels_signature = asset_state.get("last_entry_levels_signature")
        last_entry_sent_utc = asset_state.get("last_entry_sent_utc")
        if (
            allow_entry
            and last_entry_signature == entry_signature
            and last_entry_levels_signature == entry_levels_signature
            and last_entry_sent_utc
        ):    
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

        budapest_time = datetime.now(ZoneInfo("Europe/Budapest")).strftime("%H:%M")
        valid_until = datetime.now(BUDAPEST_TZ) + timedelta(minutes=30)
        mode_label = "🔵 RANGE" if alignment_state == "COUNTER" else "🟢 TREND"

        if order_type == "LIMIT":
            instruction = (
                f"NYISS LONG – BUY LIMIT @ {format_price(entry)}"
                if direction == "buy"
                else f"NYISS SHORT – SELL LIMIT @ {format_price(entry)}"
            )
            prefix = "🟡"
            color = COLOR_YELLOW
            order_card_hint = "Limit (függő): csak a megadott vagy jobb áron teljesül."
        elif order_type == "STOP":
            instruction = (
                f"NYISS LONG – BUY STOP @ {format_price(entry)}"
                if direction == "buy"
                else f"NYISS SHORT – SELL STOP @ {format_price(entry)}"
            )
            prefix = "🔵"
            color = COLOR_BLUE
            order_card_hint = "Stop (függő): kitörésre aktiválódik a trigger árnál."
        else:
            instruction = (
                f"NYISS LONG – BUY MARKET @ {format_price(entry)}"
                if direction == "buy"
                else f"NYISS SHORT – SELL MARKET @ {format_price(entry)}"
            )
            if direction == "buy":
                prefix = "🟢"
                color = COLOR_GREEN
            else:
                prefix = "🔴"
                color = COLOR_RED
            order_card_hint = "Market (azonnali): azonnali végrehajtás piaci áron."
            
        title = f"{prefix} TEENDŐ MOST: {instruction}"
        entry_text = format_price(entry)
        sl_text = format_price(sl)
        tp_text = f"TP1: {format_price(tp1)}" + (f" | TP2: {format_price(tp2)}" if tp2 is not None else "")

        fields = [
            {
                "name": "📊 Aktuális Piaci Állapot",
                "value": f"Spot ár: {format_price(spot_price)} | Időpont: {budapest_time}",
                "inline": False,
            }
        ]
        fields.extend(
            [
                {
                    "name": "⚙️ Paraméterek a brókerhez",
                    "value": (
                        f"ESZKÖZ: `{asset_name}`\n"
                        f"MODE: `{mode_label}`\n"
                        f"ENTRY: `{entry_text}`\n"
                        f"STOP (SL): `{sl_text}`\n"
                        f"CÉL (TP): `{tp_text}`"
                    ),
                    "inline": False,
                },
                {
                    "name": "🧭 Megbízás típus (kártya jelölés)",
                    "value": (
                        f"TÍPUS: `{order_type}`\n"
                        f"JELÖLÉS: `{prefix}`\n"
                        f"LEÍRÁS: {order_card_hint}"
                    ),
                    "inline": False,
                },
                {
                    "name": "⏳ Érvényesség & Törlés",
                    "value": (
                        f"LEJÁRAT (Valid Until): `{valid_until.strftime('%H:%M')}` – Ha addig nem aktiválódik, töröld a megbízást!\n"
                        "TÖRLÉS FELTÉTEL: Ha az árfolyam eléri a Stop-Loss szintet az aktiválódás előtt, töröld!"
                    ),
                    "inline": False,
                },
                {
                    "name": "🛠️ Menedzsment",
                    "value": "TP1 elérésekor automatikus jelzés érkezik a részleges zárásra és a Stop nullába (Breakeven) húzására.",
                    "inline": False,
                },
                {
                    "name": "🧠 Rövid indoklás",
                    "value": f"• {reasons[0]}" if reasons else "• N/A",
                    "inline": False,
                },
            ]
        )
        embed = {
            "title": title,
            "description": f"Eszköz: `{asset_name}`",
            "color": color,
            "fields": fields,
            "footer": {"text": f"{asset_name} • Manual trade model"},
        }

        send_discord_embed(embed)
        asset_state["last_entry_signature"] = entry_signature
        asset_state["last_entry_levels_signature"] = entry_levels_signature
        asset_state["last_entry_sent_utc"] = to_utc_iso(now_dt)
        notify_state[asset_name] = asset_state
        notify_state_changed = True

    if notify_state_changed and not DRY_RUN:  
        save_json(notify_state_path, notify_state)


if __name__ == "__main__":
    check_and_notify()

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
import fcntl
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
import position_tracker
import state_db


DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
DRY_RUN = os.getenv("NOTIFY_DRY_RUN", "").lower() in {"1", "true", "yes"}
ENTRY_COOLDOWN_MINUTES = 30
ACTIVATION_REMINDER_MINUTES = max(1, int(os.getenv("NOTIFY_ACTIVATION_REMINDER_MINUTES", "240") or "240"))
DISCORD_NOTIFY_ASSETS = {
    str(asset).upper()
    for asset in getattr(settings, "ASSETS", [])
    if str(asset).strip() and str(asset).upper() != "BTCUSD"
}

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
NOTIFY_LOCK_PATH = PUBLIC_DIR / ".notify_discord.lock"
NOTIFY_EVENT_LOG_PATH = PUBLIC_DIR / "monitoring" / "discord_notify_events.jsonl"

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
    
def _append_notify_event(event: Dict[str, Any]) -> None:
    try:
        NOTIFY_EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with NOTIFY_EVENT_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Hiba: notify event log írás sikertelen: {exc}")


def _log_notify_skip(asset: str, event: str, reason: str, **extra: Any) -> None:
    payload = {
        "timestamp_utc": to_utc_iso(datetime.now(timezone.utc)),
        "asset": str(asset or "").upper(),
        "event": str(event or "skip"),
        "skipped": True,
        "reason": str(reason or "unspecified"),
    }
    if extra:
        payload.update(extra)
    _append_notify_event(payload)



def get_budapest_time(utc_iso_string: Optional[str]) -> str:
    if not utc_iso_string or utc_iso_string == "-":
        return "N/A"
    try:
        dt_utc = datetime.fromisoformat(str(utc_iso_string).replace("Z", "+00:00"))
        dt_bp = dt_utc.astimezone(BUDAPEST_TZ)
        return dt_bp.strftime("%H:%M:%S")
    except Exception:
        return "Idő?"


def _format_ts(timestamp: Any, fallback: str = "nincs adat") -> str:
    raw = str(timestamp or "").strip()
    if not raw or raw == "-":
        return fallback
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(BUDAPEST_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return fallback


def _asset_emoji(asset: str) -> str:
    key = str(asset or "").upper()
    if key in {"XAU", "XAUUSD", "GOLD_CFD", "GOLDCFD"}:
        return "🟡"
    if key in {"XAG", "XAGUSD", "SILVER", "SILVER_CFD"}:
        return "⚪"
    if key in {"USOIL", "OIL", "BRENT"}:
        return "🛢️"
    return "📌"


def _close_reason_hu(reason: str) -> str:
    return {
        "tp2_hit": "TP2 elérve",
        "sl_hit": "SL elérve",
        "hard_exit": "Hard exit",
    }.get(str(reason or "").strip().lower(), "nincs adat")


def _position_status_hu(status: str, tp1_hit_ts: str) -> str:
    state = str(status or "").strip().lower()
    if state == "closed":
        return "Lezárt"
    if tp1_hit_ts != "még nem":
        return "Részben zárt"
    return "Nyitott"


def _resolve_tp1_hit_ts(position: Dict[str, Any]) -> str:
    if not isinstance(position, dict):
        return "még nem"
    explicit_hit = _format_ts(position.get("tp1_hit_at_utc"), fallback="")
    if explicit_hit:
        return explicit_hit
    last_signal = position.get("last_management_signal")
    if isinstance(last_signal, dict) and str(last_signal.get("state") or "").lower() == "scale_out":
        return _format_ts(last_signal.get("triggered_at"), fallback="még nem")
    return "még nem"


def _pick_first_nonempty(payload: Dict[str, Any], *keys: str) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _resolve_entry_type_label(event_type: str, signal_data: Dict[str, Any], tracked_entry: Dict[str, Any]) -> str:
    tracked = tracked_entry if isinstance(tracked_entry, dict) else {}
    signal = signal_data if isinstance(signal_data, dict) else {}
    resolved = _pick_first_nonempty(
        tracked,
        "entry_type",
        "entryType",
        "order_type",
        "orderType",
        "signal_type",
        "signalType",
        "trigger_type",
        "triggerType",
        "entry_order_type",
    )
    if not resolved:
        resolved = _pick_first_nonempty(
            signal,
            "entry_type",
            "entryType",
            "order_type",
            "orderType",
            "signal_type",
            "signalType",
            "trigger_type",
            "triggerType",
            "entry_order_type",
        )
    if not resolved:
        return "Automatikus aktiválás" if event_type == "activation" else "N/A"
    return resolved.upper()


def send_discord_embed(embed_data: Dict[str, Any]) -> bool:
     event_base = {
         "timestamp_utc": to_utc_iso(datetime.now(timezone.utc)),
         "title": str(embed_data.get("title") or ""),
     }
     if DRY_RUN:
         print(f"[DRY RUN] Embed title: {embed_data.get('title')}")
         _append_notify_event({**event_base, "success": False, "skipped": True, "reason": "dry_run"})
         return False
     if not DISCORD_WEBHOOK_URL:
         print("Hiba: DISCORD_WEBHOOK_URL nincs beállítva; webhook küldés kihagyva.")
         _append_notify_event({**event_base, "success": False, "skipped": True, "reason": "missing_webhook"})        
         return False
     if requests is None:
         print("Hiba: a 'requests' modul hiányzik; webhook küldés kihagyva.")
         _append_notify_event({**event_base, "success": False, "skipped": True, "reason": "requests_missing"})    
         return False
     try:
         response = requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed_data]}, timeout=5)
         ok = 200 <= int(response.status_code) < 300
         _append_notify_event({
             **event_base,
             "success": ok,
             "status_code": int(response.status_code),
             "reason": "ok" if ok else "http_error",
         })
         if not ok:
             print(f"Hiba: Discord webhook HTTP {response.status_code}")
         return ok
     except Exception as exc:
         _append_notify_event({**event_base, "success": False, "reason": "request_exception", "error": str(exc)[:300]})
         print(f"Hiba: {exc}")
         return False


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


def levels_match_direction(
    direction: str,
    entry: Optional[float],
    sl: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
) -> bool:
    if direction not in {"buy", "sell"}:
        return False
    if entry is None or sl is None or tp1 is None:
        return False
    if direction == "buy":
        return sl < entry < tp1 and (tp2 is None or tp1 <= tp2)
    return tp1 < entry < sl and (tp2 is None or tp2 <= tp1)


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
    triggered_at = str(
        exit_signal.get("triggered_at")
        or exit_signal.get("closed_at_utc")
        or exit_signal.get("timestamp")
        or ""
    ).strip()
    return {
        "state": state,
        "direction": direction,
        "category": category,
        "trigger_bucket": trigger_bucket,
        "triggered_at": triggered_at,
    }


def _position_event_key(position: Dict[str, Any], event_state: str) -> Optional[str]:
    if not isinstance(position, dict):
        return None
    opened_at = str(position.get("opened_at_utc") or "").strip()
    side = str(position.get("side") or "").strip().lower()
    entry = safe_float(position.get("entry"))
    if not opened_at or side not in {"long", "short"} or entry is None:
        return None
    return f"{opened_at}|{side}|{round(entry, 5)}|{event_state}"


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


def _resolve_tracking_context() -> Tuple[Dict[str, Any], str, bool]:
    stability_cfg = settings.load_config().get("signal_stability") or {}
    tracking_cfg = stability_cfg.get("manual_position_tracking") or {}
    positions_path = str(tracking_cfg.get("positions_file") or state_db.DEFAULT_DB_PATH)
    treat_missing = bool(tracking_cfg.get("treat_missing_file_as_flat", True))
    return tracking_cfg, positions_path, treat_missing


def _normalize_position_side(side: Any) -> str:
    raw = str(side or "").strip().lower()
    if raw in {"long", "buy"}:
        return "buy"
    if raw in {"short", "sell"}:
        return "sell"
    return ""


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
    tracking_cfg, positions_path, treat_missing = _resolve_tracking_context()
    tracking_enabled = bool(tracking_cfg.get("enabled"))
    manual_positions = position_tracker.load_positions(positions_path, treat_missing)
    
    # assets = [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    assets = [
        d
        for d in PUBLIC_DIR.iterdir()
        if d.is_dir()
        and not d.name.startswith("_")
        and (not DISCORD_NOTIFY_ASSETS or d.name.upper() in DISCORD_NOTIFY_ASSETS)
    ]
    
    
    for asset_dir in assets:
        asset_name = asset_dir.name
        signal_path = asset_dir / "signal.json"        
        data = load_json(signal_path)
        if not data:
            continue

        signal = str(data.get("signal") or "no entry").lower()
        plan_for_signal = data.get("precision_plan") if isinstance(data.get("precision_plan"), dict) else {}
        trigger_state = str(plan_for_signal.get("trigger_state") or "").lower()
        if signal == "no entry" and trigger_state in {"arming", "fire"}:
            signal = "precision_arming"
        exit_signal = data.get("position_exit_signal") or data.get("active_position_meta", {}).get("exit_signal")
        notify_meta = data.get("notify") if isinstance(data.get("notify"), dict) else {}
        position_state = data.get("position_state") if isinstance(data.get("position_state"), dict) else {}        
        tracked_entry = manual_positions.get(asset_name) if isinstance(manual_positions, dict) else None
        tracked_status = str(tracked_entry.get("status") or "").lower() if isinstance(tracked_entry, dict) else ""
        has_lifecycle_event = tracking_enabled and tracked_status in {"open", "closed"}
        if signal not in {"buy", "sell", "precision_arming"} and not exit_signal and not has_lifecycle_event:
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

        if alignment_state in {"MIXED", "COUNTER"} and signal != "precision_arming" and not exit_signal and not has_lifecycle_event:
            print(
                f"{asset_name}: {alignment_state} piac — jelzés némítva (csak precision_arming mehet át)."
            )
            continue

        missing_gates = [str(item) for item in (gates.get("missing") or []) if item]
        hard_missing_gates = [gate for gate in missing_gates if _is_hard_gate_blocker(gate)]
        soft_missing_gates = [gate for gate in missing_gates if gate not in hard_missing_gates]
        if hard_missing_gates and not exit_signal and not has_lifecycle_event:
            print(
                f"{asset_name}: belépő blokkolva hard védelmi kapun ({', '.join(hard_missing_gates)})."
            )
            continue
        if soft_missing_gates and not exit_signal and not has_lifecycle_event:
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

        order_type = str(
            data.get("entry_order_type")
            or data.get("entry_type")
            or data.get("entryType")
            or data.get("order_type")
            or data.get("orderType")
            or data.get("signal_type")
            or data.get("signalType")
            or "MARKET"
        ).upper()
        direction = signal
        if signal == "precision_arming":
            plan = plan_for_signal
            direction = str(plan.get("direction") or data.get("signal") or "buy").lower()
            order_type = str(plan.get("order_type") or "LIMIT").upper()
            entry = safe_float(plan.get("entry") or entry)
            sl = safe_float(plan.get("stop_loss") or sl)
            tp1 = safe_float(plan.get("take_profit_1") or tp1)
            tp2 = safe_float(plan.get("take_profit_2") or tp2)

        if order_type not in {"LIMIT", "MARKET", "STOP"}:
            order_type = "MARKET"


        if direction not in {"buy", "sell"} and not exit_signal and not has_lifecycle_event:
            print(f"{asset_name}: bizonytalan irány ({direction}) — jelzés némítva.")
            continue

        loop_now_dt = datetime.now(timezone.utc)        
        payload_has_position = bool(position_state.get("has_position"))
        if payload_has_position and not exit_signal and not has_lifecycle_event and direction in {"buy", "sell"}:
            allow_market_with_tracking = order_type == "MARKET" and tracking_enabled
            allow_precision_limit_info_card = signal == "precision_arming" and order_type == "LIMIT"
            if allow_market_with_tracking:
                print(
                    f"{asset_name}: payload has_position=true, de MARKET jelzés tracking mellett továbbengedve (dedup/tracking gate dönt)."
                )
            elif allow_precision_limit_info_card:
                print(
                    f"{asset_name}: payload has_position=true, precision_arming LIMIT kártya információs módban továbbengedve."
                )
            else:
                print(f"{asset_name}: belépő jelzés némítva (payload position_state.has_position=true).")
                continue

        should_notify_entry = notify_meta.get("should_notify")
        if (
            should_notify_entry is False
            and signal in {"buy", "sell"}
            and not exit_signal
            and not has_lifecycle_event
            and direction in {"buy", "sell"}
        ):
            reason = str(notify_meta.get("reason") or "unspecified")
            print(f"{asset_name}: belépő jelzés némítva (notify.should_notify=false, reason={reason}).")
            continue

        has_same_direction_tracked_position = False
        if (
            tracking_enabled
            and tracked_status in {"open", "pending"}
            and not exit_signal
            and signal in {"buy", "sell", "precision_arming"}
            and direction in {"buy", "sell"}
        ):
            tracked_side = _normalize_position_side(tracked_entry.get("side") if isinstance(tracked_entry, dict) else "")
            if tracked_side == direction:
                if signal == "precision_arming":
                    has_same_direction_tracked_position = True
                    print("Azonos irányú pozíció már fut, precision_arming LIMIT kártya információs módban küldve.")
                else:
                    print("Azonos irányú pozíció már fut, új belépő blokkolva.")
                    continue
            if tracked_side and tracked_side != direction:
                exit_signal = {
                    "state": "hard_exit",
                    "direction": "long" if tracked_side == "buy" else "short",
                    "reasons": ["Ellenirányú erős belépési jel (Trendforduló)"],
                    "synthetic_reverse": True,
                    "timestamp": to_utc_iso(loop_now_dt),
                }
                closed_at_utc = to_utc_iso(loop_now_dt)
                manual_positions = position_tracker.close_position(
                    asset_name,
                    reason="hard_exit",
                    closed_at_utc=closed_at_utc,
                    cooldown_minutes=0,
                    positions=manual_positions,
                )
                tracked_entry = manual_positions.get(asset_name) if isinstance(manual_positions, dict) else None
                tracked_status = str(tracked_entry.get("status") or "").lower() if isinstance(tracked_entry, dict) else ""
                
        if tracking_enabled and not exit_signal and not has_lifecycle_event:
            tracked_state = position_tracker.compute_state(
                asset_name,
                tracking_cfg,
                manual_positions,
                loop_now_dt,
            )
            has_position = bool(tracked_state.get("has_position"))
            pending_active = bool(tracked_state.get("pending_active"))
            if (has_position or pending_active) and not has_same_direction_tracked_position:
                tracked_status = "OPEN" if has_position else "PENDING"
                print(f"{asset_name}: belépő blokkolva (pozíció státusz: {tracked_status}).")
                continue
            if tracked_state.get("cooldown_active"):
                print(
                    f"{asset_name}: belépő blokkolva (cooldown aktív: {tracked_state.get('cooldown_until_utc')})."
                )
                continue

        spot = data.get("spot") if isinstance(data.get("spot"), dict) else {}
        spot_price = spot.get("price")

        asset_state = notify_state.get(asset_name) if isinstance(notify_state.get(asset_name), dict) else {}
        if tracking_enabled and isinstance(tracked_entry, dict):
            if tracked_status == "open":
                activation_key = _position_event_key(tracked_entry, "activated")
                activation_due = False
                if activation_key:
                    last_activation_sent_utc = str(asset_state.get("last_activation_sent_utc") or "").strip()
                    last_activation_dt = None
                    if last_activation_sent_utc:
                        try:
                            last_activation_dt = datetime.fromisoformat(last_activation_sent_utc.replace("Z", "+00:00"))
                        except Exception:
                            last_activation_dt = None
                    activation_due = asset_state.get("last_position_event_key") != activation_key
                    if not activation_due and last_activation_dt is not None:
                        activation_due = (loop_now_dt - last_activation_dt) >= timedelta(minutes=ACTIVATION_REMINDER_MINUTES)
                if activation_key and activation_due:
                    activation_side = str(tracked_entry.get("side") or "").lower()
                    irany = "Vétel" if activation_side == "long" else "Eladás"
                    order_type = _resolve_entry_type_label("activation", data, tracked_entry)
                    tp1_hit_ts = _resolve_tp1_hit_ts(tracked_entry)
                    print(
                        f"{asset_name}: activation entry type resolved "
                        f"event_type=activation tracked_entry_type={tracked_entry.get('entry_type')!r} "
                        f"tracked_order_type={tracked_entry.get('order_type')!r} "
                        f"tracked_signal_type={tracked_entry.get('signal_type')!r} "
                        f"tracked_trigger_type={tracked_entry.get('trigger_type')!r} "
                        f"signal_entry_type={data.get('entry_type')!r} "
                        f"signal_order_type={data.get('order_type')!r} "
                        f"signal_signal_type={data.get('signal_type')!r} "
                        f"signal_trigger_type={data.get('trigger_type')!r} "
                        f"displayed={order_type!r}"
                    )
                    sent_ok = send_discord_embed({
                        "title": f"{_asset_emoji(asset_name)} {asset_name}",
                        "description": (
                            f"Irány: `{irany}`\n"
                            f"Belépő típus: `{order_type}`\n"
                            f"Belépő: `{format_price(tracked_entry.get('entry'))}`\n"
                            f"Aktiválva: `{_format_ts(tracked_entry.get('opened_at_utc'))}`\n"
                            f"SL: `{format_price(tracked_entry.get('sl'))}`\n"
                            f"TP1: `{format_price(tracked_entry.get('tp1'))}`\n"
                            f"Spot: `{format_price(spot_price)}`\n"
                            f"TP1 elérve: `{tp1_hit_ts}`\n"
                            f"Állapot: `{_position_status_hu(tracked_status, tp1_hit_ts)}`\n"
                            f"Kártya időbélyeg: `{_format_ts(to_utc_iso(loop_now_dt))}`"
                        ),
                        "color": COLOR_GREEN if activation_side == "long" else COLOR_RED,
                        "fields": [],
                        "footer": {"text": f"Activation • {asset_name}"},
                    })
                    if sent_ok:
                        asset_state["last_position_event_key"] = activation_key
                        asset_state["last_activation_sent_utc"] = to_utc_iso(loop_now_dt)
                        notify_state[asset_name] = asset_state
                        notify_state_changed = True
                elif activation_key:
                    print(f" -> ACTIVATION DEDUP: {asset_name} (position event key)")
                    _log_notify_skip(asset_name, "activation", "dedup_position_event_key")
                management_signal = tracked_entry.get("last_management_signal")
                management_state = (
                    str(management_signal.get("state") or "").lower()
                    if isinstance(management_signal, dict)
                    else ""
                )
                if bool(tracked_entry.get("tp1_scaled")) and management_state == "scale_out":
                    scale_out_key = _position_event_key(tracked_entry, "scale_out")
                    if scale_out_key and asset_state.get("last_position_event_key") != scale_out_key:
                        sent_ok = send_discord_embed(
                            {
                                "title": "🟠 RÉSZZÁRÁS (TP1) - Húzd a Stop-ot Nullába!",
                                "description": f"Eszköz: `{asset_name}`",
                                "color": COLOR_ORANGE,
                                "fields": [
                                    {
                                        "name": "🕒 Kártya időbélyeg",
                                        "value": f"`{_format_ts(management_signal.get('triggered_at'))}`",
                                        "inline": False,
                                    },
                                    {
                                        "name": "📊 Árfolyam & Szintek",
                                        "value": (
                                            f"Spot: `{format_price(spot_price)}`\n"
                                            f"Eredeti Entry: `{format_price(tracked_entry.get('entry'))}`\n"
                                            f"Új SL (breakeven): `{format_price(tracked_entry.get('sl'))}`"
                                        ),
                                        "inline": False,
                                    },
                                ],
                                "footer": {"text": f"Menedzsment • {asset_name}"},
                            }
                        )
                        if sent_ok:
                            asset_state["last_position_event_key"] = scale_out_key
                            notify_state[asset_name] = asset_state
                            notify_state_changed = True
                    elif scale_out_key:
                        _log_notify_skip(
                            asset_name,
                            "scale_out",
                            "dedup_position_event_key",
                            key=scale_out_key,
                        )         
            elif tracked_status == "closed":
                close_reason = str(tracked_entry.get("close_reason") or "").lower()
                closed_at = str(tracked_entry.get("closed_at_utc") or "")
                close_key = f"{closed_at}|{close_reason}|closed" if closed_at and close_reason else None
                if close_key and asset_state.get("last_position_event_key") != close_key:
                    activation_side = str(tracked_entry.get("side") or "").lower()
                    irany = "Vétel" if activation_side == "long" else "Eladás"
                    tp1_hit_ts = _resolve_tp1_hit_ts(tracked_entry)
                    close_color = COLOR_GREEN if close_reason == "tp2_hit" else COLOR_HARD_EXIT if close_reason == "hard_exit" else COLOR_RED
                    sent_ok = send_discord_embed({
                        "title": f"{_asset_emoji(asset_name)} {asset_name}",
                        "description": (
                            f"Irány: `{irany}`\n"
                            f"Belépő: `{format_price(tracked_entry.get('entry'))}`\n"
                            f"Aktiválva: `{_format_ts(tracked_entry.get('opened_at_utc'))}`\n"
                            f"SL: `{format_price(tracked_entry.get('sl'))}`\n"
                            f"TP1: `{format_price(tracked_entry.get('tp1'))}`\n"
                            f"Spot: `{format_price(spot_price)}`\n"
                            f"TP1 elérve: `{tp1_hit_ts}`\n"
                            "Állapot: `Lezárt`"
                        ),
                        "color": close_color,
                        "fields": [
                            {
                                "name": "❗ Lezárás oka",
                                "value": f"`{_close_reason_hu(close_reason)}`",
                                "inline": False,
                            },
                            {
                                "name": "🕒 Lezárás ideje",
                                "value": f"`{_format_ts(closed_at)}`",
                                "inline": False,
                            }
                        ],
                        "footer": {"text": f"Close • {asset_name}"},
                    })
                    if sent_ok:
                        asset_state["last_position_event_key"] = close_key
                        notify_state[asset_name] = asset_state
                        notify_state_changed = True
                elif close_key:
                    _log_notify_skip(
                        asset_name,
                        "close",
                        "dedup_position_event_key",
                        key=close_key,
                        close_reason=close_reason,
                    )
    
        if exit_signal:
            exit_state = str(exit_signal.get("state") or exit_signal.get("action") or "").lower()            
            exit_signature = _exit_signature(exit_signal)
            exit_reasons = [str(r) for r in (exit_signal.get("reasons") or []) if r][:4]
            is_synthetic_reverse = bool(exit_signal.get("synthetic_reverse"))
            now_dt = loop_now_dt
            asset_state = notify_state.get(asset_name) if isinstance(notify_state.get(asset_name), dict) else {}
            tracked_entry = manual_positions.get(asset_name) if isinstance(manual_positions, dict) else None
            tracked_status_for_exit = str(tracked_entry.get("status") or "").lower() if isinstance(tracked_entry, dict) else ""
            if tracking_enabled and tracked_status_for_exit != "open" and not is_synthetic_reverse:
                print(f"{asset_name}: exit jelzés figyelmen kívül hagyva (nincs követett nyitott pozíció).")
                _log_notify_skip(asset_name, "exit", "no_tracked_open_position", exit_state=exit_state)    
                continue
            event_key = _position_event_key(tracked_entry, exit_state)
            last_exit_signature = asset_state.get("last_exit_signature")
            if last_exit_signature == exit_signature:
                print(f" -> EXIT DEDUP: {asset_name} (event signature)")
                _log_notify_skip(asset_name, "exit", "dedup_exit_signature", exit_state=exit_state)
                continue
            state_label = {
                "scale_out": "🟠 RÉSZZÁRÁS (TP1) - Húzd a Stop-ot Nullába!",
                "hard_exit": "🔴 AZONNAL ZÁRD A POZÍCIÓT! (Hard Exit)",
                "tighten_stop": "🟠 SZŰKÍTSD A STOP-LOSST!",
                "stop_loss_hit": "🔴 STOP LOSS ELÉRVE - Pozíció Lezárva",
                "take_profit_hit": "🟢 CÉLÁR ELÉRVE - Pozíció Lezárva",
                "take_profit_2_hit": "🟢 CÉLÁR (TP2) ELÉRVE - Pozíció Lezárva",
                "closed": "⚪ POZÍCIÓ LEZÁRVA",
            }.get(exit_state, f"🟠 POZÍCIÓ MENEDZSMENT ({exit_state})")

            if exit_state in ["hard_exit", "stop_loss_hit"]:
                color = COLOR_RED
            elif exit_state in ["take_profit_hit", "take_profit_2_hit"]:
                color = COLOR_GREEN
            elif exit_state == "closed":
                color = 0x95A5A6
            else:
                color = COLOR_ORANGE

            close_direction = _normalize_position_side(exit_signal.get("direction"))
            if not close_direction and isinstance(tracked_entry, dict):
                close_direction = _normalize_position_side(tracked_entry.get("side"))
            close_direction_text = "LONG" if close_direction == "buy" else "SHORT" if close_direction == "sell" else "N/A"

            fields = [
                {
                    "name": "🕒 Kártya időbélyeg",
                    "value": f"`{_format_ts(to_utc_iso(now_dt))}`",
                    "inline": False,
                },
            ]
            if exit_state == "hard_exit":
                fields.append(
                    {
                        "name": "🎯 Zárandó irány",
                        "value": f"`{close_direction_text}`",
                        "inline": False,
                    }
                )
            fields.extend([
                {
                    "name": "📊 Árfolyam & Szintek",
                    "value": f"Spot: `{format_price(spot_price)}`\nEredeti Entry: `{format_price(entry)}`",
                    "inline": False,
                },
                {
                    "name": "💡 Javaslat oka",
                    "value": "\n".join(f"• {_hu_reason(r)}" for r in exit_reasons) or "• N/A",
                    "inline": False,
                }
            ])

            embed = {
                "title": state_label,
                "description": f"Eszköz: `{asset_name}`",
                "color": color,
                "fields": fields,
                "footer": {"text": f"Menedzsment • {asset_name}"},
            }
            sent_ok = send_discord_embed(embed)
            if sent_ok:
                asset_state["last_exit_signature"] = exit_signature
                asset_state["last_exit_sent_utc"] = to_utc_iso(now_dt)
                if event_key:
                    asset_state["last_position_event_key"] = event_key
                notify_state[asset_name] = asset_state
                notify_state_changed = True
            if not is_synthetic_reverse:
                continue

        tp1_net_usd = 0.0
        if entry is None or sl is None or tp1 is None:
            print(f"{asset_name}: hiányzó entry/SL/TP1 — jelzés eldobva.")
            continue

        if not levels_match_direction(direction, entry, sl, tp1, tp2):
            print(
                f"{asset_name}: inkonzisztens irány/szintek ({direction}) — jelzés eldobva. "
                f"entry={format_price(entry)} sl={format_price(sl)} tp1={format_price(tp1)} tp2={format_price(tp2)}"
            )
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
        now_dt = loop_now_dt
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

        if order_type in ["LIMIT", "STOP"]:
            prefix = "🟡"
            color = COLOR_YELLOW
        else:
            prefix = "🟢"
            color = COLOR_GREEN

        direction_hu = "LONG" if direction == "buy" else "SHORT"
        direction_en = "BUY" if direction == "buy" else "SELL"
        instruction = f"NYISS {direction_hu} – {direction_en} {order_type} @ {format_price(entry)}"
            
        title = f"{prefix} {instruction}"

        risk_usd = sl_risk_usd if sl_risk_usd and sl_risk_usd > 0 else 50.0
        units_text = "N/A"
        if entry and sl and abs(entry - sl) > 0:
            units = risk_usd / abs(entry - sl)
            units_text = f"{units:.2f} Egység (Units)"
        
        p_score_text = f"P Score: **{p_score:.1f}** (Erősség)" if p_score is not None else ""
        reasons_text = "\n".join([f"• {_hu_reason(r)}" for r in reasons[:2]]) if reasons else "• Rendszer jelzés"
        final_reasons = f"{p_score_text}\n{reasons_text}" if p_score_text else reasons_text
        
        fields = [
            {
                "name": "📊 Árfolyam",
                "value": f"Spot ár: `{format_price(spot_price)}`\nBelépő (Entry): `{format_price(entry)}`",
                "inline": False,
            },
            {
                "name": "⚙️ Paraméterek az eToro-hoz",
                "value": f"MÉRET: `{units_text}` ({risk_usd} USD kockázat)\nStop Loss (SL): `{format_price(sl)}`\nTake Profit 1 (TP1): `{format_price(tp1)}`"
                + (f"\nTake Profit 2 (TP2): `{format_price(tp2)}`" if tp2 else ""),
                "inline": False,
            },
            {
                "name": "💡 Indoklás",
                "value": final_reasons,
                "inline": False,
            },
        ]
        embed = {
            "title": title,
            "description": f"Eszköz: `{_asset_emoji(asset_name)} {asset_name}`",
            "color": color,
            "fields": fields + [
                {
                    "name": "🕒 Időbélyeg (Budapest)",
                    "value": f"`{now_dt.astimezone(BUDAPEST_TZ).strftime('%Y-%m-%d %H:%M:%S')}`",
                    "inline": False,
                }
            ],
            "footer": {"text": f"Signal • Várakozás (30 perc csend indítva)"},
        }

        sent_ok = send_discord_embed(embed)
        if sent_ok:
            asset_state["last_entry_signature"] = entry_signature
            asset_state["last_entry_levels_signature"] = entry_levels_signature
            asset_state["last_entry_sent_utc"] = to_utc_iso(now_dt)
            notify_state[asset_name] = asset_state
            notify_state_changed = True

        if tracking_enabled and direction in {"buy", "sell"} and not has_same_direction_tracked_position:
            if order_type in {"LIMIT", "STOP"}:
                manual_positions = position_tracker.register_precision_pending_position(
                    asset_name,
                    data,
                    now_dt,
                    manual_positions,
                )
            elif order_type == "MARKET":    
                manual_positions = position_tracker.open_position(
                    asset_name,
                    side="long" if direction == "buy" else "short",
                    entry=entry,
                    sl=sl,
                    tp1=tp1,
                    tp2=tp2,
                    opened_at_utc=to_utc_iso(now_dt),
                    order_type=order_type,
                    positions=manual_positions,
                )
                if asset_name in manual_positions:
                    manual_positions[asset_name]["status"] = "open"

            if not DRY_RUN:
                position_tracker.save_positions_atomic(positions_path, manual_positions)

    if notify_state_changed and not DRY_RUN:  
        save_json(notify_state_path, notify_state)


if __name__ == "__main__":
    with NOTIFY_LOCK_PATH.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Másik notify_discord folyamat már fut; duplikált küldés elkerülve.")
            sys.exit(0)
        check_and_notify()

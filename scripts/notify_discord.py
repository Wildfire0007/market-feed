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
import time
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


def _collect_webhook_urls() -> list[str]:
    raw = os.getenv("DISCORD_WEBHOOK_URL", "")
    urls: list[str] = []
    seen: set[str] = set()
    for part in raw.replace("\n", ",").split(","):
        url = part.strip()
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


DISCORD_WEBHOOK_URLS = _collect_webhook_urls()
DRY_RUN = os.getenv("NOTIFY_DRY_RUN", "").lower() in {"1", "true", "yes"}
MANUAL_RUN = "--manual" in sys.argv
ENTRY_COOLDOWN_MINUTES = 30
EXIT_NOTIFY_COOLDOWN_MINUTES = 30
_notify_assets_raw = os.getenv("DISCORD_NOTIFY_ASSETS", "").strip()
DISCORD_NOTIFY_ASSETS = {
    part.strip().upper()
    for part in _notify_assets_raw.replace("\n", ",").split(",")
    if part.strip()
}
_notify_blocked_assets_raw = os.getenv("DISCORD_NOTIFY_BLOCKED_ASSETS", "BTCUSD,EURUSD").strip()
DISCORD_NOTIFY_BLOCKED_ASSETS = {
    part.strip().upper()
    for part in _notify_blocked_assets_raw.replace("\n", ",").split(",")
    if part.strip()
}
NOTIFY_ATTEMPTS = 0
NOTIFY_SUCCESSES = 0
NOTIFY_FAILURES = 0
_WEBHOOK_COOLDOWN_UNTIL: Dict[str, float] = {}


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

def post_batches(webhook_url: str, content: str, embeds: list[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
    if requests is None:
        return {"attempted": False, "success": False, "http_status": None, "error": "requests_missing", "message_id": None, "batch_results": []}
    now = time.time()
    if _WEBHOOK_COOLDOWN_UNTIL.get(webhook_url, 0.0) > now:
        return {"attempted": False, "success": False, "http_status": None, "error": "cooldown", "message_id": None, "batch_results": []}

    payload = {"embeds": embeds[:batch_size]}
    if content:
        payload["content"] = content
    batch_results = []
    for _ in range(2):
        try:
            response = requests.post(webhook_url, json=payload, timeout=5)
            status = int(getattr(response, "status_code", 0) or 0)
            if status == 429:
                retry_after = safe_float(getattr(response, "headers", {}).get("Retry-After")) or 1.0
                _WEBHOOK_COOLDOWN_UNTIL[webhook_url] = time.time() + retry_after
                time.sleep(retry_after)
                batch_results.append({"attempted": True, "success": False, "http_status": status, "error": "rate_limited", "message_id": None, "batch_index": 0, "embed_count": len(payload.get("embeds") or [])})
                continue
            ok = 200 <= status < 300
            batch_results.append({"attempted": True, "success": ok, "http_status": status, "error": None if ok else f"http_{status}", "message_id": None, "batch_index": 0, "embed_count": len(payload.get("embeds") or [])})
            return {"attempted": True, "success": ok, "http_status": status, "error": None if ok else f"http_{status}", "message_id": None, "batch_results": batch_results}
        except Exception as exc:
            batch_results.append({"attempted": True, "success": False, "http_status": None, "error": str(exc), "message_id": None, "batch_index": 0, "embed_count": len(payload.get("embeds") or [])})
            break
    last = batch_results[-1] if batch_results else {"http_status": None, "error": "unknown", "success": False}
    return {"attempted": True, "success": False, "http_status": last.get("http_status"), "error": last.get("error"), "message_id": None, "batch_results": batch_results}


def send_discord_embed(embed_data: Dict[str, Any]) -> bool:
     global NOTIFY_ATTEMPTS, NOTIFY_SUCCESSES, NOTIFY_FAILURES
     if DRY_RUN or not DISCORD_WEBHOOK_URLS:
         print(f"[DRY RUN] Embed title: {embed_data.get('title')}")
         return True
     if requests is None:
         print("Hiba: a 'requests' modul hiányzik; webhook küldés kihagyva.")
         NOTIFY_FAILURES += 1
         return False

     NOTIFY_ATTEMPTS += 1
     payload_candidates = [{"embeds": [embed_data]}]
     title = str(embed_data.get("title") or "").strip()
     if title:
         payload_candidates.append({"content": title})

     print(f"[notify] webhook_count={len(DISCORD_WEBHOOK_URLS)} title={title or '(no-title)'}")    
     for webhook_url in DISCORD_WEBHOOK_URLS:
         for payload in payload_candidates:
             try:
                 response = requests.post(webhook_url, json=payload, timeout=5)
                 status = int(getattr(response, "status_code", 0) or 0)
                 _append_notify_event({"url": webhook_url, "status": status, "payload": list(payload.keys())})
                 if 200 <= status < 300:
                     NOTIFY_SUCCESSES += 1
                     return True
                 print(f"[notify] webhook HTTP status={status} url={webhook_url[:48]}...")
             except Exception as exc:
                 _append_notify_event({"url": webhook_url, "status": None, "error": str(exc)})

     NOTIFY_FAILURES += 1
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
    skip_reasons: Dict[str, int] = {}

    def _mark_skip(reason: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    
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
        if asset_name.upper() in DISCORD_NOTIFY_BLOCKED_ASSETS:
            _mark_skip("asset_opt_out")
            continue
        signal_path = asset_dir / "signal.json"        
        data = load_json(signal_path)
        if not data:
            _mark_skip("missing_signal")
            continue

        signal = str(data.get("signal") or "no entry").lower()
        exit_signal = data.get("position_exit_signal") or data.get("active_position_meta", {}).get("exit_signal")
        tracked_entry = manual_positions.get(asset_name) if isinstance(manual_positions, dict) else None
        tracked_status = str(tracked_entry.get("status") or "").lower() if isinstance(tracked_entry, dict) else ""
        has_lifecycle_event = tracking_enabled and tracked_status in {"open", "closed"}
        if signal not in {"buy", "sell", "precision_arming"} and not exit_signal and not has_lifecycle_event:
            _mark_skip("unsupported_signal")            
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
            _mark_skip(f"alignment_{alignment_state.lower()}")
            continue

        missing_gates = [str(item) for item in (gates.get("missing") or []) if item]
        hard_missing_gates = [gate for gate in missing_gates if _is_hard_gate_blocker(gate)]
        soft_missing_gates = [gate for gate in missing_gates if gate not in hard_missing_gates]
        if hard_missing_gates and not exit_signal and not has_lifecycle_event:
            print(
                f"{asset_name}: belépő blokkolva hard védelmi kapun ({', '.join(hard_missing_gates)})."
            )
            _mark_skip("hard_gate_block")    
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
        
        if direction not in {"buy", "sell"} and not exit_signal and not has_lifecycle_event:
            print(f"{asset_name}: bizonytalan irány ({direction}) — jelzés némítva.")
            _mark_skip("invalid_direction")
            continue

        if tracking_enabled and not exit_signal and not has_lifecycle_event:
            tracked_state = position_tracker.compute_state(
                asset_name,
                tracking_cfg,
                manual_positions,
                datetime.now(timezone.utc),
            )
            if tracked_state.get("has_position") or tracked_state.get("pending_active"):
                tracked_status = "OPEN" if tracked_state.get("has_position") else "PENDING"
                print(f"{asset_name}: belépő blokkolva (pozíció státusz: {tracked_status}).")
                _mark_skip("position_active")
                continue
            if tracked_state.get("cooldown_active"):
                print(
                    f"{asset_name}: belépő blokkolva (cooldown aktív: {tracked_state.get('cooldown_until_utc')})."
                )
                _mark_skip("position_cooldown")
                continue

        spot = data.get("spot") if isinstance(data.get("spot"), dict) else {}
        spot_price = spot.get("price")

        asset_state = notify_state.get(asset_name) if isinstance(notify_state.get(asset_name), dict) else {}
        if tracking_enabled and isinstance(tracked_entry, dict):
            if tracked_status == "open":
                activation_key = _position_event_key(tracked_entry, "activated")
                if activation_key and asset_state.get("last_position_event_key") != activation_key:
                    activation_side = str(tracked_entry.get("side") or "").lower()
                    irany = "Vétel" if activation_side == "long" else "Eladás"
                    tp1_hit_ts = _resolve_tp1_hit_ts(tracked_entry)                     
                    send_discord_embed({
                        "title": f"{_asset_emoji(asset_name)} {asset_name}",
                        "description": (
                            f"Irány: `{irany}`\n"
                            f"Belépő: `{format_price(tracked_entry.get('entry'))}`\n"
                            f"Aktiválva: `{_format_ts(tracked_entry.get('opened_at_utc'))}`\n"
                            f"TP1: `{format_price(tracked_entry.get('tp1'))}`\n"
                            f"TP1 elérve: `{tp1_hit_ts}`\n"
                            f"Állapot: `{_position_status_hu(tracked_status, tp1_hit_ts)}`"
                        ),
                        "color": COLOR_GREEN if activation_side == "long" else COLOR_RED,
                        "fields": [],
                        "footer": {"text": f"Activation • {asset_name}"},
                    })
                    asset_state["last_position_event_key"] = activation_key
                    notify_state[asset_name] = asset_state
                    notify_state_changed = True
            elif tracked_status == "closed":
                close_reason = str(tracked_entry.get("close_reason") or "").lower()
                closed_at = str(tracked_entry.get("closed_at_utc") or "")
                close_key = f"{closed_at}|{close_reason}|closed" if closed_at and close_reason else None
                if close_key and asset_state.get("last_position_event_key") != close_key:
                    activation_side = str(tracked_entry.get("side") or "").lower()
                    irany = "Vétel" if activation_side == "long" else "Eladás"
                    tp1_hit_ts = _resolve_tp1_hit_ts(tracked_entry)
                    close_color = COLOR_GREEN if close_reason == "tp2_hit" else COLOR_HARD_EXIT if close_reason == "hard_exit" else COLOR_RED
                    send_discord_embed({
                        "title": f"{_asset_emoji(asset_name)} {asset_name}",
                        "description": (
                            f"Irány: `{irany}`\n"
                            f"Belépő: `{format_price(tracked_entry.get('entry'))}`\n"
                            f"Aktiválva: `{_format_ts(tracked_entry.get('opened_at_utc'))}`\n"
                            f"TP1: `{format_price(tracked_entry.get('tp1'))}`\n"
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
                    asset_state["last_position_event_key"] = close_key
                    notify_state[asset_name] = asset_state
                    notify_state_changed = True
    
        if exit_signal:
            exit_state = str(exit_signal.get("state") or exit_signal.get("action") or "").lower()            
            exit_signature = _exit_signature(exit_signal)
            exit_reasons = [str(r) for r in (exit_signal.get("reasons") or []) if r][:4]
            now_dt = datetime.now(timezone.utc)
            asset_state = notify_state.get(asset_name) if isinstance(notify_state.get(asset_name), dict) else {}
            tracked_entry = manual_positions.get(asset_name) if isinstance(manual_positions, dict) else None
            tracked_status_for_exit = str(tracked_entry.get("status") or "").lower() if isinstance(tracked_entry, dict) else ""
            if tracking_enabled and tracked_status_for_exit not in {"open", "closed"}:
                print(f"{asset_name}: exit jelzés figyelmen kívül hagyva (nincs követett nyitott/zárt pozíció).")
                continue
            event_key = _position_event_key(tracked_entry, exit_state)
            if event_key and asset_state.get("last_position_event_key") == event_key:
                print(f" -> EXIT POSITION DEDUP: {asset_name} ({exit_state})")
                continue
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

            p_score_text = f"P Score: **{p_score:.1f}** (Erősség)" if p_score is not None else ""
            reason_lines = [f"• {_hu_reason(r)}" for r in exit_reasons] or ["• N/A"]
            if p_score_text:
                reason_lines.insert(0, p_score_text)
                
            embed = {
                "title": state_label,
                "description": f"{_asset_emoji(asset_name)} Eszköz: `{asset_name}`",
                "color": color,
                "fields": [
                    {
                        "name": "📊 Árfolyam & Szintek",
                        "value": f"Spot: `{format_price(spot_price)}`\nEredeti Entry: `{format_price(entry)}`",
                        "inline": False,
                    },
                    {
                        "name": "💡 Javaslat oka",
                        "value": "\n".join(reason_lines),
                        "inline": False,
                    }
                ],
                "footer": {"text": f"Menedzsment • {asset_name}"},
            }
            send_discord_embed(embed)
            asset_state["last_exit_signature"] = exit_signature
            asset_state["last_exit_sent_utc"] = to_utc_iso(now_dt)
            if event_key:
                asset_state["last_position_event_key"] = event_key
            notify_state[asset_name] = asset_state
            notify_state_changed = True
            continue

        tp1_net_usd = 0.0
        if entry is None or sl is None or tp1 is None:
            print(f"{asset_name}: hiányzó entry/SL/TP1 — jelzés eldobva.")
            _mark_skip("missing_levels")
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
            _mark_skip("tp1_net_too_low")
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

        p_score_text = f"P Score: **{p_score:.1f}** (Erősség)" if p_score is not None else ""
        reason_lines = [f"• {_hu_reason(r)}" for r in reasons[:2]] if reasons else ["• Rendszer jelzés"]
        if p_score_text:
            reason_lines.insert(0, p_score_text)
        reasons_text = "\n".join(reason_lines)
        
        fields = [
            {
                "name": "📊 Árfolyam",
                "value": f"Spot ár: `{format_price(spot_price)}`\nBelépő (Entry): `{format_price(entry)}`",
                "inline": False,
            },
            {
                "name": "⚙️ Paraméterek",
                "value": f"Stop Loss (SL): `{format_price(sl)}`\nTake Profit 1 (TP1): `{format_price(tp1)}`"
                + (f"\nTake Profit 2 (TP2): `{format_price(tp2)}`" if tp2 else ""),
                "inline": False,
            },
            {
                "name": "💡 Indoklás",
                "value": reasons_text,
                "inline": False,
            },
        ]
        embed = {
            "title": title,
            "description": f"{_asset_emoji(asset_name)} Eszköz: `{asset_name}`",
            "color": color,
            "fields": fields,
            "footer": {"text": "Signal • Várakozás (30 perc csend indítva)"},
        }

        send_discord_embed(embed)
        asset_state["last_entry_signature"] = entry_signature
        asset_state["last_entry_levels_signature"] = entry_levels_signature
        asset_state["last_entry_sent_utc"] = to_utc_iso(now_dt)
        notify_state[asset_name] = asset_state
        notify_state_changed = True

        if (
            tracking_enabled
            and signal == "precision_arming"
            and order_type in {"LIMIT", "STOP"}
        ):
            manual_positions = position_tracker.register_precision_pending_position(
                asset_name,
                data,
                now_dt,
                manual_positions,
            )
            if not DRY_RUN:
                position_tracker.save_positions_atomic(positions_path, manual_positions)
        
    print(
        f"[notify] manual={MANUAL_RUN} attempts={NOTIFY_ATTEMPTS} "
        f"success={NOTIFY_SUCCESSES} failures={NOTIFY_FAILURES}"
    )

    if notify_state_changed and not DRY_RUN:  
        save_json(notify_state_path, notify_state)


if __name__ == "__main__":
    if not PUBLIC_DIR.exists():
        check_and_notify()
        sys.exit(0)
    with NOTIFY_LOCK_PATH.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Másik notify_discord folyamat már fut; duplikált küldés elkerülve.")
            sys.exit(0)
        check_and_notify()

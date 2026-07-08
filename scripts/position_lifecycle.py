#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Position lifecycle worker for manual Discord trading.

ATR-adaptive pending expiry formula: when enabled, the configured base validity
window is multiplied by ``median_atr5m_20d / current_atr5m`` and clamped to
25%-200% of the base. Higher-than-normal ATR therefore shortens stale entry
validity; lower ATR extends it without letting pending orders live indefinitely.
"""
from __future__ import annotations

import json
import os
import sys
import fcntl
import importlib
import importlib.util
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import analysis_settings as settings
from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response

requests = importlib.import_module("requests") if importlib.util.find_spec("requests") else None

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = Path(os.getenv("NOTIFY_PUBLIC_DIR", "")) if os.getenv("NOTIFY_PUBLIC_DIR") else BASE_DIR / "public"
if not PUBLIC_DIR.exists() and (BASE_DIR.parent / "public").exists():
    PUBLIC_DIR = BASE_DIR.parent / "public"

LOCK_PATH = PUBLIC_DIR / ".position_lifecycle.lock"
INBOX_PATH = PUBLIC_DIR / "_position_lifecycle_inbox.jsonl"
STATE_PATH = PUBLIC_DIR / "_position_lifecycle_state.json"
EXPIRY_NOTIFY_STATE_PATH = PUBLIC_DIR / "_position_expiry_notify_state.json"
CLOSE_NOTIFY_STATE_PATH = PUBLIC_DIR / "_position_close_notify_state.json"
STALE_OPEN_POSITION_NOTIFY_STATE_PATH = PUBLIC_DIR / "_position_stale_open_notify_state.json"
CLOSE_STATES = {"hard_exit", "stop_loss_hit", "take_profit_hit", "take_profit_2_hit", "closed"}


def _cfg() -> Dict[str, Any]:
    return getattr(settings, "POSITION_LIFECYCLE", None) or getattr(settings, "position_lifecycle", None) or {}


def to_utc_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None
        

def safe_float(value: Any) -> Optional[float]:
    try:
        n = float(value)
        return n if n == n else None
    except Exception:
        return None


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as h:
            data = json.load(h)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as h:
            json.dump(payload, h, ensure_ascii=False, indent=2)
    except Exception:
        pass



def _webhook_urls() -> list[str]:
    raw = os.getenv("DISCORD_WEBHOOK_URL_ACTIONABLE") or os.getenv("DISCORD_WEBHOOK_URL", "")
    return [url.strip() for url in raw.replace("\\n", ",").replace("\n", ",").split(",") if url.strip()]


def _position_key(asset: str, pos: Dict[str, Any]) -> str:
    return f"{asset}|{pos.get('pending_since_utc') or pos.get('updated_at_utc') or ''}|{pos.get('entry')}|{pos.get('order_type')}"


def _format_price(value: Any) -> str:
    n = safe_float(value)
    if n is None:
        return "N/A"
    return f"{n:.5f}".rstrip("0").rstrip(".")


def _pnl_usd(side: str, entry: Any, exit_level: Any, units: Any) -> Optional[float]:
    entry_f, exit_f, units_f = safe_float(entry), safe_float(exit_level), safe_float(units)
    if entry_f is None or exit_f is None or units_f is None:
        return None
    return (exit_f - entry_f) * units_f * (1 if side == "long" else -1)


def _send_expiry_cancel_alert(asset: str, pos: Dict[str, Any], now: datetime) -> bool:
    side = str(pos.get("side") or "").lower()
    direction = "BUY" if side == "long" else "SELL"
    side_label = "LONG" if side == "long" else "SHORT"
    order_type = str(pos.get("order_type") or "LIMIT").upper()
    embed = {
        "title": "❌ JEL LEJÁRT – TÖRÖLD A MEGBÍZÁST",
        "color": 0xE74C3C,
        "fields": [
            {"name": "Eszköz", "value": f"`{asset}`", "inline": True},
            {"name": "Irány", "value": f"`{side_label}`", "inline": True},
            {"name": "Megbízás", "value": f"`{order_type} @ {pos.get('entry')}`", "inline": True},
            {"name": "Utasítás", "value": f"Töröld a függő {asset} {direction} {order_type} megbízást a brókernél most — a jel érvényessége lejárt.", "inline": False},
            {"name": "🕒 Időbélyeg", "value": f"`{to_utc_iso(now)}` UTC", "inline": False},
        ],
    }
    sent = False
    for url in _webhook_urls():
        if requests is None:
            continue
        try:
            resp = requests.post(url, json={"embeds": [embed]}, timeout=8)
            sent = _webhook_log_response(__import__("logging").getLogger(__name__), "position_lifecycle", "actionable", resp) or sent
        except Exception as exc:
            _webhook_log_exception(__import__("logging").getLogger(__name__), "position_lifecycle", "actionable", exc)
    return sent


def _notify_expired_once(asset: str, pos: Dict[str, Any], now: datetime) -> None:
    st = load_json(EXPIRY_NOTIFY_STATE_PATH)
    key = _position_key(asset, pos)
    if st.get(key):
        return
    if _send_expiry_cancel_alert(asset, pos, now):
        st[key] = to_utc_iso(now)
        save_json(EXPIRY_NOTIFY_STATE_PATH, st)


def _exit_level_for_reason(pos: Dict[str, Any], reason: str) -> Any:
    if reason == "take_profit_hit":
        return pos.get("tp1")
    if reason == "take_profit_2_hit":
        return pos.get("tp2")
    if reason == "stop_loss_hit":
        return pos.get("sl")
    if reason in {"hard_exit", "session_force_close"}:
        return pos.get("close_spot") or pos.get("last_spot") or pos.get("entry")        
    return pos.get("entry")


def _send_close_alert(asset: str, pos: Dict[str, Any], reason: str, now: datetime) -> bool:
    side = str(pos.get("side") or "").lower()
    side_label = "LONG" if side == "long" else "SHORT"
    exit_level = _exit_level_for_reason(pos, reason)
    pnl = _pnl_usd(side, pos.get("entry"), exit_level, pos.get("size_units"))
    title = {
        "take_profit_hit": "🟢 TP1 ELÉRVE – ZÁRD A TELJES POZÍCIÓT" if bool(pos.get("tp1_closes_position", True)) else "🟠 TP1 ELÉRVE – RÉSZZÁRÁS + BE",
        "take_profit_2_hit": "🟢 CÉLÁR ELÉRVE – ZÁRD A POZÍCIÓT",
        "stop_loss_hit": "🔴 STOP LOSS SZINT ELÉRVE – ZÁRD A POZÍCIÓT",
        "hard_exit": "🔴 AZONNAL ZÁRD A POZÍCIÓT – HARD EXIT",
        "session_force_close": "🟠 SESSION ZÁRÁS – ZÁRD A POZÍCIÓT MOST",        
    }.get(reason, "🔴 ZÁRD A POZÍCIÓT")
    detail = str(pos.get("close_detail") or "hard_exit")
    fields = [
        {"name": "Eszköz", "value": f"`{asset}`", "inline": True},
        {"name": "Irány", "value": f"`{side_label}`", "inline": True},
    ]
    if reason == "session_force_close":
        fields.extend([
            {"name": "Aktuális spot", "value": f"`{_format_price(exit_level)}`", "inline": True},
            {"name": "Becsült PnL", "value": f"`${pnl:.2f}`" if pnl is not None else "`N/A`", "inline": True},
            {"name": "Utasítás", "value": f"Zárd a teljes {asset} {side_label} pozíciót piaci áron most.", "inline": False},
        ])
    elif reason == "hard_exit":
        fields.extend([
            {"name": "Aktuális spot", "value": f"`{_format_price(exit_level)}`", "inline": True},
            {"name": "Becsült PnL", "value": f"`${pnl:.2f}`" if pnl is not None else "`N/A`", "inline": True},
        ])
    else:
        fields.extend([    
            {"name": "Belépő", "value": f"`{_format_price(pos.get('entry'))}`", "inline": True},
            {"name": "Kilépési szint", "value": f"`{_format_price(exit_level)}`", "inline": True},
            {"name": "Méret", "value": f"`{_format_price(pos.get('size_units'))}`", "inline": True},
            {"name": "Becsült PnL", "value": f"`${pnl:.2f}`" if pnl is not None else "`N/A`", "inline": True},
            {"name": "Utasítás", "value": f"Zárd a teljes {asset} {side_label} pozíciót piaci áron most.", "inline": False},
        ])
    fields.append({"name": "🕒 Időbélyeg", "value": f"`{to_utc_iso(now)}` UTC", "inline": False})
    embed = {
        "title": title,
        "description": "A kereskedési ablak zárul / max. tartási idő letelt — zárd piaci áron. Becsült PnL: " + (f"${pnl:.2f}" if pnl is not None else "N/A") + "." if reason == "session_force_close" else (f"Ok: {detail}" if reason == "hard_exit" else ""),
        "color": 0xF39C12 if reason == "session_force_close" else (0x2ECC71 if reason.startswith("take_profit") else 0xE74C3C),
        "fields": fields,
    }
    sent = False
    for url in _webhook_urls():
        if requests is None:
            continue
        try:
            resp = requests.post(url, json={"embeds": [embed]}, timeout=8)
            sent = _webhook_log_response(__import__("logging").getLogger(__name__), "position_lifecycle", "actionable", resp) or sent
        except Exception as exc:
            _webhook_log_exception(__import__("logging").getLogger(__name__), "position_lifecycle", "actionable", exc)
    return sent


def _notify_close_once(asset: str, pos: Dict[str, Any], reason: str, now: datetime) -> None:
    if reason not in {"take_profit_hit", "take_profit_2_hit", "stop_loss_hit", "hard_exit", "session_force_close"}:    
        return
    st = load_json(CLOSE_NOTIFY_STATE_PATH)
    key = f"{asset}|{pos.get('opened_at_utc') or pos.get('pending_since_utc') or pos.get('updated_at_utc') or ''}|{reason}|{pos.get('entry')}|{_exit_level_for_reason(pos, reason)}"
    if st.get(key):
        return
    if _send_close_alert(asset, pos, reason, now):
        st[key] = to_utc_iso(now)
        save_json(CLOSE_NOTIFY_STATE_PATH, st)


def _spot_timestamp(asset_dir: Path, signal: Dict[str, Any]) -> Optional[datetime]:
    spot = load_json(asset_dir / "spot.json")
    if not spot and isinstance(signal.get("spot"), dict):
        spot = signal.get("spot") or {}
    return parse_utc(spot.get("utc") or spot.get("timestamp") or spot.get("retrieved_at_utc"))


def _send_stale_open_position_alert(asset: str, age_minutes: float, now: datetime) -> bool:
    embed = {
        "title": "⚠️ ADATKIESÉS NYITOTT POZÍCIÓ MELLETT – FIGYELD KÉZZEL",
        "color": 0xF39C12,
        "fields": [
            {"name": "Eszköz", "value": f"`{asset}`", "inline": True},
            {"name": "Utolsó adat kora", "value": f"`{age_minutes:.1f} perc`", "inline": True},
            {"name": "Utasítás", "value": "Kezeld az SL/TP-t kézzel a brókernél, amíg az adatfolyam helyre nem áll.", "inline": False},
            {"name": "🕒 Időbélyeg", "value": f"`{to_utc_iso(now)}` UTC", "inline": False},
        ],
    }
    sent = False
    for url in _webhook_urls():
        if requests is None:
            continue
        try:
            resp = requests.post(url, json={"embeds": [embed]}, timeout=8)
            sent = _webhook_log_response(__import__("logging").getLogger(__name__), "position_lifecycle", "actionable", resp) or sent
        except Exception as exc:
            _webhook_log_exception(__import__("logging").getLogger(__name__), "position_lifecycle", "actionable", exc)
    return sent


def _notify_stale_open_position_once(asset: str, pos: Dict[str, Any], age_minutes: float, now: datetime) -> None:
    st = load_json(STALE_OPEN_POSITION_NOTIFY_STATE_PATH)
    key = f"{asset}|{pos.get('opened_at_utc') or pos.get('pending_since_utc') or pos.get('updated_at_utc') or ''}"
    if st.get(key):
        return
    if _send_stale_open_position_alert(asset, age_minutes, now):
        st[key] = to_utc_iso(now)
        save_json(STALE_OPEN_POSITION_NOTIFY_STATE_PATH, st)


def _clear_stale_open_position_episode(asset: str, pos: Dict[str, Any]) -> None:
    st = load_json(STALE_OPEN_POSITION_NOTIFY_STATE_PATH)
    key = f"{asset}|{pos.get('opened_at_utc') or pos.get('pending_since_utc') or pos.get('updated_at_utc') or ''}"
    if key in st:
        st.pop(key, None)
        save_json(STALE_OPEN_POSITION_NOTIFY_STATE_PATH, st)


def _session_force_close_due(now: datetime, cfg: Dict[str, Any]) -> bool:
    raw = str(cfg.get("session_force_close_utc") or "").strip()
    if not raw or now.weekday() >= 5:
        return False
    try:
        hour, minute = [int(part) for part in raw.split(":", 1)]
    except Exception:
        return False
    cutoff = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return now >= cutoff


def _max_hold_due(pos: Dict[str, Any], now: datetime, cfg: Dict[str, Any]) -> bool:
    max_hold = safe_float(cfg.get("max_hold_minutes"))
    opened = parse_utc(pos.get("opened_at_utc"))
    return bool(max_hold and opened and now - opened >= timedelta(minutes=max_hold))

def _read_inbox_new_lines(path: Path, last_line: int) -> tuple[list[Dict[str, Any]], int]:
    if not path.exists():
        return [], last_line
    events, current_line = [], 0
    with path.open("r", encoding="utf-8") as h:
        for raw in h:
            current_line += 1
            if current_line <= last_line:
                continue
            try:
                evt = json.loads(raw)
            except Exception:
                continue
            if isinstance(evt, dict):
                events.append(evt)
    return events, current_line


def _latest_bar(asset_dir: Path) -> Dict[str, Any]:
    for name in ("klines_5m.json", "k1m.json", "klines_1m.json"):
        data = load_json(asset_dir / name)
        rows = data.get("values") or data.get("data") or data.get("candles") or []
        if isinstance(rows, list) and rows:
            return rows[0] if isinstance(rows[0], dict) else {}
    return {}


def _atr_values(signal: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    atr = signal.get("atr") if isinstance(signal.get("atr"), dict) else {}
    current = safe_float(signal.get("atr5m") or atr.get("atr5m") or atr.get("current_5m"))
    median = safe_float(signal.get("atr5m_median_20d") or atr.get("atr5m_median_20d") or atr.get("median_20d_5m"))
    return current, median


def _validity_minutes(signal: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    base = float(cfg.get("entry_validity_minutes", 120) or 120)
    if not cfg.get("entry_validity_atr_adaptive", True):
        return base
    current, median = _atr_values(signal)
    if not current or not median or current <= 0 or median <= 0:
        return base
    return max(base * 0.25, min(base * 2.0, base * (median / current)))


def _pending_filled(pos: Dict[str, Any], price: Optional[float]) -> bool:
    entry = safe_float(pos.get("entry"))
    if price is None or entry is None:
        return False
    side = str(pos.get("side") or "").lower()
    order_type = str(pos.get("order_type") or "MARKET").upper()
    if order_type == "LIMIT":
        return price <= entry if side == "long" else price >= entry
    return price >= entry if side == "long" else price <= entry


def _bar_hits(side: str, pos: Dict[str, Any], bar: Dict[str, Any]) -> tuple[bool, bool, bool, Optional[float]]:
    high = safe_float(bar.get("high") or bar.get("h"))
    low = safe_float(bar.get("low") or bar.get("l"))
    open_ = safe_float(bar.get("open") or bar.get("o"))
    sl, tp1, tp2 = pos.get("sl"), pos.get("tp1"), pos.get("tp2")
    if side == "long":
        return low is not None and safe_float(sl) is not None and low <= safe_float(sl), high is not None and safe_float(tp1) is not None and high >= safe_float(tp1), high is not None and safe_float(tp2) is not None and high >= safe_float(tp2), open_
    return high is not None and safe_float(sl) is not None and high >= safe_float(sl), low is not None and safe_float(tp1) is not None and low <= safe_float(tp1), low is not None and safe_float(tp2) is not None and low <= safe_float(tp2), open_


def _close(pos: Dict[str, Any], reason: str, now: datetime, *, outcome: Optional[str] = None, detail: str = "") -> Dict[str, Any]:
    was_open = str(pos.get("status") or "").lower() == "open"    
    pos.update({"status": "closed", "close_reason": reason, "outcome": outcome or reason, "close_detail": detail, "closed_at_utc": to_utc_iso(now), "updated_at_utc": to_utc_iso(now)})
    if was_open:
        pos["_notify_close_reason"] = reason    
    return pos


def process() -> None:
    if not PUBLIC_DIR.exists():
        return
    cfg = _cfg()
    state = load_json(STATE_PATH)
    meta = state.get("_meta") if isinstance(state.get("_meta"), dict) else {}
    positions = state.get("positions") if isinstance(state.get("positions"), dict) else {}
    events, new_last_line = _read_inbox_new_lines(INBOX_PATH, int(meta.get("last_inbox_line") or 0))
    now_dt = datetime.now(timezone.utc)

    for evt in events:
        if str(evt.get("event") or "") != "entry_signal":
            continue
        asset = str(evt.get("asset") or "").strip()
        if not asset or str((positions.get(asset) or {}).get("status") or "").lower() in {"pending", "open"}:
            continue
        order_type = str(evt.get("order_type") or "MARKET").upper()
        direction = str(evt.get("direction") or "buy").lower()
        if direction not in {"buy", "sell"}:
            continue
        ts = str(evt.get("ts_utc") or to_utc_iso(now_dt))
        mgmt = evt.get("management") if isinstance(evt.get("management"), dict) else {}
        positions[asset] = {"status": "open" if order_type == "MARKET" else "pending", "side": "long" if direction == "buy" else "short", "entry": safe_float(evt.get("entry")), "sl": safe_float(evt.get("sl")), "tp1": safe_float(evt.get("tp1")), "tp2": safe_float(evt.get("tp2")), "order_type": order_type, "source_signal": str(evt.get("signal") or ""), "entry_signature": str(evt.get("entry_signature") or ""), "pending_since_utc": ts if order_type != "MARKET" else None, "updated_at_utc": ts, "opened_at_utc": ts if order_type == "MARKET" else None, "size_units": safe_float(evt.get("size_units")), "breakeven_sl": safe_float(mgmt.get("breakeven_sl")), "partial_close_pct": safe_float(mgmt.get("partial_close_pct")), "tp1_closes_position": bool(cfg.get("tp1_closes_position", True))}          

    hard_cfg = cfg.get("hard_exit", {}) if isinstance(cfg.get("hard_exit"), dict) else {}
    immediate_on = set(hard_cfg.get("immediate_on") or [])
    ambiguous_as = str(cfg.get("ambiguous_bar_counts_as", "sl")).lower()
    for asset_dir in sorted([d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")], key=lambda p: p.name):
        asset, pos = asset_dir.name, positions.get(asset_dir.name)
        if not isinstance(pos, dict) or str(pos.get("status") or "").lower() in {"", "closed"}:
            continue
        signal = load_json(asset_dir / "signal.json")
        side = str(pos.get("side") or "").lower()
        bar = _latest_bar(asset_dir)
        status = str(pos.get("status") or "").lower()
        if status == "open" and (_session_force_close_due(now_dt, cfg) or _max_hold_due(pos, now_dt, cfg)):
            pos["close_spot"] = safe_float((signal.get("spot") or {}).get("price")) or safe_float(bar.get("close") or bar.get("c")) or pos.get("entry")
            pos = _close(pos, "session_force_close", now_dt, outcome="force_closed")
            close_reason = pos.pop("_notify_close_reason", None)
            if close_reason:
                _notify_close_once(asset, pos, close_reason, now_dt)
            positions[asset] = pos
            continue
        spot_ts = _spot_timestamp(asset_dir, signal)
        stale_limit = safe_float(cfg.get("open_position_data_stale_minutes")) or 10.0
        if spot_ts and now_dt - spot_ts > timedelta(minutes=stale_limit):
            _notify_stale_open_position_once(asset, pos, (now_dt - spot_ts).total_seconds() / 60.0, now_dt)
        else:
            _clear_stale_open_position_episode(asset, pos)        
        if pos.get("status") == "pending":
            price = safe_float((signal.get("spot") or {}).get("price"))
            if _pending_filled(pos, price):
                pos.update({"status": "open", "opened_at_utc": to_utc_iso(now_dt), "updated_at_utc": to_utc_iso(now_dt)})
            else:
                since = parse_utc(pos.get("pending_since_utc") or pos.get("updated_at_utc")) or now_dt
                if now_dt - since >= timedelta(minutes=_validity_minutes(signal, cfg)):
                    _notify_expired_once(asset, pos, now_dt)                    
                    pos.update({"status": "closed", "close_reason": "expired", "outcome": "expired", "closed_at_utc": to_utc_iso(now_dt), "updated_at_utc": to_utc_iso(now_dt)})
            positions[asset] = pos
            continue
        exit_signal = signal.get("position_exit_signal") if isinstance(signal.get("position_exit_signal"), dict) else signal.get("exit_signal")
        exit_state = str((exit_signal or {}).get("state") or (exit_signal or {}).get("action") or "").lower()
        exit_reason = str((exit_signal or {}).get("reason") or (exit_signal or {}).get("comment") or exit_state)
        atr_now, atr_med = _atr_values(signal)
        shock = atr_now and atr_med and atr_now > float(hard_cfg.get("volatility_shock_atr5m_median_mult", 3.0) or 3.0) * atr_med
        if (exit_state == "hard_exit" and exit_reason in immediate_on) or ("volatility_shock" in immediate_on and shock):
            pos["close_spot"] = safe_float((signal.get("spot") or {}).get("price")) or safe_float(bar.get("close") or bar.get("c")) or safe_float(bar.get("open") or bar.get("o")) or pos.get("entry")            
            pos = _close(pos, "hard_exit", now_dt, outcome="hard_exit", detail="volatility_shock" if shock else exit_reason)
        elif exit_state == "hard_exit":
            key = f"hard_exit_counter_{asset}"
            meta[key] = int(meta.get(key) or 0) + 1
            if meta[key] >= int(((hard_cfg.get("trend_reversal_requires") or {}).get("consecutive_runs")) or 2):
                pos["close_spot"] = safe_float((signal.get("spot") or {}).get("price")) or safe_float(bar.get("close") or bar.get("c")) or safe_float(bar.get("open") or bar.get("o")) or pos.get("entry")                
                pos = _close(pos, "hard_exit", now_dt, outcome="hard_exit", detail=exit_reason or "trend_reversal")
        else:
            meta[f"hard_exit_counter_{asset}"] = 0
            sl_hit, tp1_hit, tp2_hit, open_ = _bar_hits(side, pos, bar)
            if sl_hit and tp1_hit and ambiguous_as == "sl":
                tp1_hit = tp2_hit = False
            gap = ""
            sl = safe_float(pos.get("sl"))
            if sl_hit and open_ is not None and sl is not None and ((side == "long" and open_ < sl) or (side == "short" and open_ > sl)):
                gap = f"azonnali piaci zárás — gap az SL alatt/felett ({abs(open_ - sl):.5g})"
            if sl_hit or exit_state == "stop_loss_hit":
                pos = _close(pos, "stop_loss_hit", now_dt, outcome="stopped", detail=gap)
            elif tp2_hit or exit_state == "take_profit_2_hit":
                pos = _close(pos, "take_profit_2_hit", now_dt, outcome="tp_hit")
            elif tp1_hit or exit_state == "take_profit_hit":
                if cfg.get("tp1_closes_position", True):
                    pos = _close(pos, "take_profit_hit", now_dt, outcome="tp1_closed")
        close_reason = pos.pop("_notify_close_reason", None)
        if close_reason:
            _notify_close_once(asset, pos, close_reason, now_dt)                    
        positions[asset] = pos

    state["_meta"] = {**meta, "last_inbox_line": new_last_line}
    state["positions"] = positions
    save_json(STATE_PATH, state)


if __name__ == "__main__":
    if any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
        print('usage: position_lifecycle.py [--help]')
        raise SystemExit(0)    
    if not PUBLIC_DIR.exists():
        sys.exit(0)
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(0)
        process()

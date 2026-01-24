#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — V6.0 (PRO Dashboard - Psychology Free)
Ez a verzió minden döntést meghoz helyetted.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Requests modul ellenőrzése
try:
    import requests
except ImportError:
    sys.exit(1)

# Időzóna (Budapest)
try:
    from zoneinfo import ZoneInfo
    BUDAPEST_TZ = ZoneInfo("Europe/Budapest")
except ImportError:
    BUDAPEST_TZ = timezone(timedelta(hours=1)) 

# --- KONFIGURÁCIÓ ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
SNIPER_MIN_P_SCORE = 35.0
SNIPER_RR_REQUIRED = 2.0
SNIPER_TP_MIN_PROFIT_PCT = 0.0035

# --- MAPPÁK ---
BASE_DIR = Path(__file__).resolve().parent
if (BASE_DIR / "public").exists():
    PUBLIC_DIR = BASE_DIR / "public"
elif (BASE_DIR.parent / "public").exists():
    PUBLIC_DIR = BASE_DIR.parent / "public"
else:
    PUBLIC_DIR = BASE_DIR / "public"

# Színek és Ikonok
ICON_BUY_MARKET  = "🟢"
ICON_SELL_MARKET = "🔴"
ICON_BUY_LIMIT   = "🔵" 
ICON_SELL_LIMIT  = "🟠"

COLOR_GREEN  = 0x2ecc71
COLOR_RED    = 0xe74c3c
COLOR_BLUE   = 0x3498db
COLOR_ORANGE = 0xe67e22

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return {}

def send_discord_embed(webhook_url, embed_data):
    if not webhook_url: return
    try:
        requests.post(webhook_url, json={"embeds": [embed_data]}, timeout=5)
    except Exception as e: print(f"Hiba: {e}")

def format_price(price):
    if price is None: return "N/A"
    try:
        p = float(price)
        if p > 1000: return f"{p:,.1f}"
        if p > 10: return f"{p:.2f}"
        return f"{p:.5f}"
    except: return str(price)

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_budapest_time(utc_iso_string):
    if not utc_iso_string or utc_iso_string == "-": return "N/A"
    try:
        dt_utc = datetime.fromisoformat(utc_iso_string.replace('Z', '+00:00'))
        dt_bp = dt_utc.astimezone(BUDAPEST_TZ)
        return dt_bp.strftime("%H:%M:%S")
    except: return "Idő?"

def calculate_fallback_levels(entry_price, atr, direction):
    if not entry_price or not atr: return None, None, None
    entry, vol = float(entry_price), float(atr)
    stop_dist = 1.5 * vol
    tp1_dist  = 1.0 * vol
    tp2_dist  = 2.5 * vol
    if direction == "buy":
        return entry - stop_dist, entry + tp1_dist, entry + tp2_dist
    else:
        return entry + stop_dist, entry - tp1_dist, entry - tp2_dist

def get_trend_icon(bias_str):
    if not bias_str: return "⚪"
    b = bias_str.lower()
    if "long" in b: return "⬆️ BULL (Emelkedő)"
    if "short" in b: return "⬇️ BEAR (Eső)"
    return "➡️ RANGE (Oldalazás)"

def check_and_notify():
    if not PUBLIC_DIR.exists(): return

    assets = [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    
    for asset_dir in assets:
        asset_name = asset_dir.name
        signal_path = asset_dir / "signal.json"
        
        data = load_json(signal_path)
        if not data: continue

        signal = data.get("signal", "no entry")
        prob = data.get("probability", 0)
        prob_raw = data.get("probability_raw")
        spot_obj = data.get("spot", {})
        spot_price = spot_obj.get("price")
        bp_time = get_budapest_time(spot_obj.get("utc"))
        time_label = bp_time or "N/A"
        
        atr = 0
        try: atr = data.get("intervention_watch", {}).get("metrics", {}).get("atr5_usd", 0)
        except: pass

        # --- EXTRA ADATOK KINYERÉSE ---
        leverage = data.get("leverage", "N/A")
        
        # Trend (Bias)
        bias_4h = data.get("biases", {}).get("adjusted_4h", "neutral")
        trend_display = get_trend_icon(bias_4h)
        bias_label = str(bias_4h or "").lower()
        if any(token in bias_label for token in ("choppy", "range", "neutral")):
            prev_side = data.get("position_state", {}).get("side")
            if not prev_side:
                prev_side = data.get("active_position_meta", {}).get("anchor_side")
            prev_map = {"long": "BUY", "short": "SELL", "buy": "BUY", "sell": "SELL"}
            prev_signal = prev_map.get(str(prev_side or "").lower())
            if prev_signal:
                trend_display = f"{trend_display}\n↩️ Előző jel: {prev_signal}"
        
        # Okok (Reasons) - Csak az első 3
        reasons_list = data.get("reasons", [])
        if not reasons_list: reasons_list = ["Nincs részletezve."]
        reasons_display = "\n".join([f"• {r}" for r in reasons_list[:3]])

        # Precision Score
        p_score = 0
        p_threshold = 0
        try:
            pg = data.get("entry_thresholds", {}).get("precision_gate_state", {})
            p_score = pg.get("score", 0)
            p_threshold = pg.get("threshold", 0)
        except: pass
        effective_thresholds = data.get("effective_thresholds", {})
        if not isinstance(effective_thresholds, dict):
            effective_thresholds = {}
        sniper_guard_reasons = []
        prob_raw_value = safe_float(prob_raw)
        if prob_raw_value is None or prob_raw_value < SNIPER_MIN_P_SCORE:
            sniper_guard_reasons.append("P-score < 35")
        rr_required = safe_float(effective_thresholds.get("rr_required"))
        if rr_required is None or rr_required < SNIPER_RR_REQUIRED:
            sniper_guard_reasons.append("RR < 2.0")
        tp_min_profit_pct = safe_float(effective_thresholds.get("tp_min_profit_pct"))
        if tp_min_profit_pct is None or tp_min_profit_pct < SNIPER_TP_MIN_PROFIT_PCT:
            sniper_guard_reasons.append("TP min profit < 0.35%")
        tp_net_min = safe_float(effective_thresholds.get("tp_net_min"))
        if tp_net_min is None or tp_net_min < SNIPER_TP_MIN_PROFIT_PCT:
            sniper_guard_reasons.append("TP net min < 0.35%")
        if sniper_guard_reasons:
            print(f"{asset_name}: Sniper guard blokkolta: {', '.join(sniper_guard_reasons)}")
            continue

        should_notify = False
        embed = {}
        notify_payload = data.get("notify") if isinstance(data, dict) else None

        # ---------------------------
        # 1. MARKET SIGNAL
        # ---------------------------
        if signal in ["buy", "sell"]:
            should_notify = True
            is_buy = (signal == "buy")
            embed = {
                "title": f"{ICON_BUY_MARKET if is_buy else ICON_SELL_MARKET} MARKET {'BUY' if is_buy else 'SELL'}: {asset_name} • {time_label}",
                "description": f"**VÉGREHAJTÁS AZONNAL!**\n\n**Miért?**\n{reasons_display}",
                "color": COLOR_GREEN if is_buy else COLOR_RED,
                "fields": [
                    {"name": "Árfolyam & Idő", "value": f"`{format_price(spot_price)}`\n🕒 {bp_time}", "inline": True},
                    {"name": "Minőség & Trend", "value": f"🎲 Esély: `{prob}%`\n{trend_display}", "inline": True},
                    {"name": "⚙️ Setup", "value": f"Kar: `x{leverage}`\nScore: `{p_score}/{p_threshold}`", "inline": True},
                    
                    {"name": "🛑 STOP LOSS", "value": f"`{format_price(data.get('sl'))}`", "inline": True},
                    {"name": "🎯 CÉLÁRAK (TP)", "value": f"TP1: `{format_price(data.get('tp1'))}`\nTP2: `{format_price(data.get('tp2'))}`", "inline": True}
                ],
                "footer": {"text": f"Market Order • {asset_name} • Kockázatkezelés: Kötelező!"}
            }

        # ---------------------------
        # 2. LIMIT SIGNAL (Precision)
        # ---------------------------
        elif signal == "precision_arming":
            plan = data.get("precision_plan", {})
            playbook = data.get("execution_playbook", [])
            trigger_state = "unknown"
            if playbook: trigger_state = playbook[-1].get("state", "unknown")
            
            if trigger_state == "fire":
                should_notify = True
                
                limit_price = plan.get("entry")
                sl_val = plan.get("stop_loss")
                tp1_val = plan.get("take_profit_1")
                tp2_val = plan.get("take_profit_2")
                direction = plan.get("direction", "buy")
                
                if not limit_price:
                    limit_price = playbook[-1].get("trigger_levels", {}).get("fire")
                    if not sl_val:
                        sl_val, tp1_val, tp2_val = calculate_fallback_levels(limit_price, atr, direction)

                rr_display = "N/A"
                try:
                    risk = abs(limit_price - sl_val)
                    reward = abs(tp1_val - limit_price)
                    if risk > 0: rr_display = f"1:{reward/risk:.1f}"
                except: pass

                if direction == "buy":
                    title_text = f"{ICON_BUY_LIMIT} LIMIT BUY: {asset_name} • {time_label}"
                    color_code = COLOR_BLUE
                else:
                    title_text = f"{ICON_SELL_LIMIT} LIMIT SELL: {asset_name} • {time_label}"
                    color_code = COLOR_ORANGE

                embed = {
                    "title": title_text,
                    "description": f"**LIMIT MEGBÍZÁS ELHELYEZÉSE!**\n(Visszateszt Zóna)\n\n**Miért?**\n{reasons_display}",
                    "color": color_code,
                    "fields": [
                        {"name": "🔵 Limit Ár (Entry)", "value": f"`{format_price(limit_price)}`", "inline": False},
                        {"name": "🛑 STOP LOSS", "value": f"`{format_price(sl_val)}`", "inline": True},
                        {"name": "🎯 CÉLÁRAK (TP)", "value": f"TP1: `{format_price(tp1_val)}`\nTP2: `{format_price(tp2_val)}`", "inline": True},
                        
                        {"name": "Jelenlegi Ár", "value": f"{format_price(spot_price)}\n🕒 {bp_time}", "inline": True},
                        {"name": "Minőség", "value": f"Score: `{p_score}`\nR/R: `{rr_display}`", "inline": True},
                        {"name": "Trend", "value": f"{trend_display}", "inline": True}
                    ],
                    "footer": {"text": f"Limit Order • {asset_name} • 'Set & Forget' Mód"}
                }
                
        if isinstance(notify_payload, dict) and notify_payload.get("should_notify") is False:
            reason = notify_payload.get("reason")
            print(f" -> BLOKKOLVA: {asset_name} ({reason})")
            should_notify = False

        if isinstance(notify_payload, dict) and notify_payload.get("should_notify") is False:
            reason = notify_payload.get("reason")
            print(f" -> BLOKKOLVA: {asset_name} ({reason})")
            should_notify = False

        if should_notify:
            print(f" -> KÜLDÉS: {asset_name}")
            send_discord_embed(DISCORD_WEBHOOK_URL, embed)

if __name__ == "__main__":
    check_and_notify()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — AGRESSZÍV Értesítő (Path-Fixed Verzió)
Javítva: Automatikusan megtalálja a public mappát, bárhol is fut a script.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# 1. Requests modul ellenőrzése
try:
    import requests
except ImportError:
    print("HIBA: Hiányzik a 'requests' modul! (pip install requests)")
    sys.exit(1)

# --- KONFIGURÁCIÓ ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# --- MAPPÁK INTELLIGENS BEÁLLÍTÁSA ---
BASE_DIR = Path(__file__).resolve().parent

# Megkeressük a 'public' mappát:
# 1. Először megnézzük a script mellett (ha root-ban fut)
if (BASE_DIR / "public").exists():
    PUBLIC_DIR = BASE_DIR / "public"
# 2. Ha ott nincs, megnézzük eggyel feljebb (ha a script a /scripts mappában van)
elif (BASE_DIR.parent / "public").exists():
    PUBLIC_DIR = BASE_DIR.parent / "public"
# 3. Végső esetben feltételezzük a relatív utat (de logolunk, ha nincs meg)
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
    if not webhook_url: 
        print(" [!] FIGYELEM: Nincs beállítva Discord Webhook URL (környezeti változó)!")
        return
    try:
        requests.post(webhook_url, json={"embeds": [embed_data]}, timeout=5)
    except Exception as e: print(f"Hiba a küldéskor: {e}")

def format_price(price):
    if price is None: return "N/A"
    try:
        p = float(price)
        if p > 1000: return f"{p:,.1f}"
        if p > 10: return f"{p:.2f}"
        return f"{p:.5f}"
    except: return str(price)

def calculate_smart_levels(entry_price, atr, direction):
    """Kiszámolja az SL/TP-t, ha a JSON-ben nincs benne"""
    if not entry_price or not atr: return None, None, None
    entry, vol = float(entry_price), float(atr)
    
    stop_dist = 1.5 * vol
    tp1_dist  = 1.0 * vol
    tp2_dist  = 2.5 * vol
    
    if direction == "buy":
        return entry - stop_dist, entry + tp1_dist, entry + tp2_dist
    else:
        return entry + stop_dist, entry - tp1_dist, entry - tp2_dist

def check_and_notify():
    # Végső ellenőrzés
    if not PUBLIC_DIR.exists():
        print(f"KRITIKUS HIBA: Nem találom a 'public' mappát!")
        print(f"Keresési helyek:\n 1. {BASE_DIR / 'public'}\n 2. {BASE_DIR.parent / 'public'}")
        return

    print(f"Adatok olvasása innen: {PUBLIC_DIR}")
    assets = [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    print(f"--- Ellenőrzés: {len(assets)} eszköz ---")
    
    sent_count = 0
    
    for asset_dir in assets:
        asset_name = asset_dir.name
        signal_path = asset_dir / "signal.json"
        
        data = load_json(signal_path)
        if not data: continue

        signal = data.get("signal", "no entry")
        prob = data.get("probability", 0)
        spot = data.get("spot", {}).get("price")
        
        # ATR kinyerése
        atr = 0
        try: atr = data.get("intervention_watch", {}).get("metrics", {}).get("atr5_usd", 0)
        except: pass

        should_notify = False
        embed = {}

        # --- 1. MARKET SIGNAL ---
        if signal in ["buy", "sell"]:
            should_notify = True
            is_buy = (signal == "buy")
            embed = {
                "title": f"{ICON_BUY_MARKET if is_buy else ICON_SELL_MARKET} MARKET {'BUY' if is_buy else 'SELL'}: {asset_name}",
                "description": "**AZONNALI BELÉPŐ!**",
                "color": COLOR_GREEN if is_buy else COLOR_RED,
                "fields": [
                    {"name": "Ár", "value": f"`{format_price(spot)}`", "inline": True},
                    {"name": "Esély", "value": f"`{prob}%`", "inline": True},
                    {"name": "🛑 SL", "value": f"`{format_price(data.get('sl'))}`", "inline": True},
                    {"name": "🎯 TP1 / TP2", "value": f"`{format_price(data.get('tp1'))}`\n`{format_price(data.get('tp2'))}`", "inline": True}
                ],
                "footer": {"text": "Market Order"}
            }

        # --- 2. LIMIT SIGNAL ---
        elif signal == "precision_arming":
            playbook = data.get("execution_playbook", [])
            last_step = playbook[-1] if playbook else {}
            state = last_step.get("state", "unknown")
            trigger_levels = last_step.get("trigger_levels", {})
            
            if state == "fire":
                should_notify = True
                limit_price = trigger_levels.get("fire")
                
                direction = "buy"
                if limit_price and spot and limit_price > spot:
                    direction = "sell"
                
                c_sl, c_tp1, c_tp2 = calculate_smart_levels(limit_price, atr, direction)
                
                title_text = f"{ICON_BUY_LIMIT} LIMIT BUY: {asset_name}" if direction == "buy" else f"{ICON_SELL_LIMIT} LIMIT SELL: {asset_name}"
                color_code = COLOR_BLUE if direction == "buy" else COLOR_ORANGE

                embed = {
                    "title": title_text,
                    "description": f"Státusz: **{state.upper()}** (Tüzelés)",
                    "color": color_code,
                    "fields": [
                        {"name": "🔵 Limit Ár (Entry)", "value": f"`{format_price(limit_price)}`", "inline": False},
                        {"name": "🛑 SL (Becsült)", "value": f"`{format_price(c_sl)}`", "inline": True},
                        {"name": "🎯 TP1 / TP2", "value": f"`{format_price(c_tp1)}`\n`{format_price(c_tp2)}`", "inline": True},
                        {"name": "Spot Ár", "value": f"{format_price(spot)}", "inline": True},
                        {"name": "Esély", "value": f"{prob}%", "inline": True}
                    ],
                    "footer": {"text": "Limit Order Setup"}
                }

        if should_notify:
            print(f" -> ÉRTESÍTÉS KÜLDÉSE: {asset_name}")
            send_discord_embed(DISCORD_WEBHOOK_URL, embed)
            sent_count += 1

    if sent_count == 0:
        print("Nincs aktív jelzés egyik eszközön sem.")

if __name__ == "__main__":
    check_and_notify()

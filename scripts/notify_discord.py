#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — INTELLIGENS Értesítő v3.0
Javítva: Limit megbízásoknál kiolvassa a "precision_plan" pontos értékeit a JSON-ből.
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
    if not webhook_url: 
        print(" [!] FIGYELEM: Nincs beállítva Discord Webhook URL!")
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

# Fallback számítás, ha a JSON-ben mégsem lenne benne a terv
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

def check_and_notify():
    if not PUBLIC_DIR.exists():
        print(f"HIBA: Nem találom a 'public' mappát: {PUBLIC_DIR}")
        return

    print(f"Adatok olvasása innen: {PUBLIC_DIR}")
    assets = [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    
    sent_count = 0
    
    for asset_dir in assets:
        asset_name = asset_dir.name
        signal_path = asset_dir / "signal.json"
        
        data = load_json(signal_path)
        if not data: continue

        signal = data.get("signal", "no entry")
        prob = data.get("probability", 0)
        spot = data.get("spot", {}).get("price")
        
        # ATR kinyerése fallback esetére
        atr = 0
        try: atr = data.get("intervention_watch", {}).get("metrics", {}).get("atr5_usd", 0)
        except: pass

        should_notify = False
        embed = {}

        # --- 1. MARKET SIGNAL (Azonnali) ---
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

        # --- 2. LIMIT SIGNAL (Precision Arming) ---
        elif signal == "precision_arming":
            # Megnézzük a precíziós tervet (ITT VAN A KINCS!)
            plan = data.get("precision_plan", {})
            trigger_state = "unknown"
            
            # Státusz ellenőrzése a playbook-ból
            playbook = data.get("execution_playbook", [])
            if playbook:
                trigger_state = playbook[-1].get("state", "unknown")
            
            # Ha TÜZELÉS van (fire)
            if trigger_state == "fire":
                should_notify = True
                
                # 1. Próbáljuk meg kivenni a pontos adatokat a precision_plan-ből
                limit_price = plan.get("entry")
                sl_val = plan.get("stop_loss")
                tp1_val = plan.get("take_profit_1")
                tp2_val = plan.get("take_profit_2")
                direction = plan.get("direction", "buy") # buy vagy sell
                
                # Ha véletlenül üres a plan, akkor fallback a trigger levels-re
                if not limit_price:
                    limit_price = playbook[-1].get("trigger_levels", {}).get("fire")
                    # És számolunk ATR alapon
                    if not sl_val:
                        sl_val, tp1_val, tp2_val = calculate_fallback_levels(limit_price, atr, direction)

                # Cím és Szín beállítása
                if direction == "buy":
                    title_text = f"{ICON_BUY_LIMIT} LIMIT BUY: {asset_name}"
                    desc_text = "**Vételi Limit (Pullback)**"
                    color_code = COLOR_BLUE
                else:
                    title_text = f"{ICON_SELL_LIMIT} LIMIT SELL: {asset_name}"
                    desc_text = "**Eladási Limit (Pullback)**"
                    color_code = COLOR_ORANGE

                embed = {
                    "title": title_text,
                    "description": f"{desc_text}\nStátusz: **FIRE** (Aktív)",
                    "color": color_code,
                    "fields": [
                        {"name": "🔵 Limit Ár (Entry)", "value": f"`{format_price(limit_price)}`", "inline": False},
                        {"name": "🛑 SL", "value": f"`{format_price(sl_val)}`", "inline": True},
                        {"name": "🎯 TP1 / TP2", "value": f"`{format_price(tp1_val)}`\n`{format_price(tp2_val)}`", "inline": True},
                        {"name": "Spot Ár", "value": f"{format_price(spot)}", "inline": True},
                        {"name": "Esély", "value": f"{prob}%", "inline": True}
                    ],
                    "footer": {"text": "Limit Order Setup (Precision Plan)"}
                }

        if should_notify:
            print(f" -> ÉRTESÍTÉS KÜLDÉSE: {asset_name}")
            send_discord_embed(DISCORD_WEBHOOK_URL, embed)
            sent_count += 1

    if sent_count == 0:
        print("Nincs aktív jelzés.")

if __name__ == "__main__":
    check_and_notify()

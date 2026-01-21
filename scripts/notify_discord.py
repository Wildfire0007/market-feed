#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — AGRESSZÍV Értesítő (Limit Order Fix)
Ez a verzió KIKAPCSOL minden stabilitási szűrőt.
Ha a rendszer azt mondja "FIRE", ez azonnal küld.
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime, timezone

# --- KONFIGURÁCIÓ ---
# Ha nincs ENV változó, ide írd be a Webhook URL-edet:
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "") 

# Mappák
BASE_DIR = Path(__file__).resolve().parent
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

def calculate_smart_levels(entry_price, atr, direction):
    """Kiszámolja az SL/TP-t, ha a JSON-ben nincs benne"""
    if not entry_price or not atr: return None, None, None
    entry, vol = float(entry_price), float(atr)
    
    # 1.5x ATR Stop, 1.0x ATR TP1, 2.5x ATR TP2
    stop_dist = 1.5 * vol
    tp1_dist  = 1.0 * vol
    tp2_dist  = 2.5 * vol
    
    if direction == "buy":
        return entry - stop_dist, entry + tp1_dist, entry + tp2_dist
    else:
        return entry + stop_dist, entry - tp1_dist, entry - tp2_dist

def check_and_notify():
    assets = [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    print(f"--- Ellenőrzés: {len(assets)} eszköz ---")
    
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
            playbook = data.get("execution_playbook", [])
            last_step = playbook[-1] if playbook else {}
            state = last_step.get("state", "unknown")
            trigger_levels = last_step.get("trigger_levels", {})
            
            # HA A STÁTUSZ 'FIRE', AZONNAL KÜLDÜNK!
            if state == "fire":
                should_notify = True
                limit_price = trigger_levels.get("fire")
                
                # Irány: Ha a Limit ár kisebb mint a mostani -> Buy Limit
                direction = "buy"
                if limit_price and spot and limit_price > spot:
                    direction = "sell"
                
                # Számítás (Smart Levels)
                c_sl, c_tp1, c_tp2 = calculate_smart_levels(limit_price, atr, direction)
                
                title_text = f"{ICON_BUY_LIMIT} LIMIT BUY: {asset_name}" if direction == "buy" else f"{ICON_SELL_LIMIT} LIMIT SELL: {asset_name}"
                desc_text = "**Vételi Limit (Pullback)**" if direction == "buy" else "**Eladási Limit (Pullback)**"
                color_code = COLOR_BLUE if direction == "buy" else COLOR_ORANGE

                embed = {
                    "title": title_text,
                    "description": desc_text + f"\nStátusz: **{state.upper()}** (Tüzelés)",
                    "color": color_code,
                    "fields": [
                        {"name": "🔵 Limit Ár (Entry)", "value": f"`{format_price(limit_price)}`", "inline": False},
                        {"name": "🛑 SL (Becsült)", "value": f"`{format_price(c_sl)}`", "inline": True},
                        {"name": "🎯 TP1 / TP2", "value": f"`{format_price(c_tp1)}`\n`{format_price(c_tp2)}`", "inline": True},
                        {"name": "Jelenlegi Ár", "value": f"{format_price(spot)}", "inline": True},
                        {"name": "Esély", "value": f"{prob}%", "inline": True}
                    ],
                    "footer": {"text": "Limit Order Setup"}
                }

        if should_notify:
            print(f" -> ÉRTESÍTÉS KÜLDÉSE: {asset_name}")
            send_discord_embed(DISCORD_WEBHOOK_URL, embed)

if __name__ == "__main__":
    check_and_notify()

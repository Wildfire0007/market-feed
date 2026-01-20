#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — SMART Értesítő
Támogatja: Market Orders, Limit Orders (SL/TP számítással).
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

# --- KONFIGURÁCIÓ ---
# Ha nincs ENV változó, ide írd be a Webhook URL-edet:
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "") 

# Mappák
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"

# Ikonok és Színek
ICON_BUY_MARKET  = "🟢"
ICON_SELL_MARKET = "🔴"
ICON_BUY_LIMIT   = "🔵"  # Kék a Limit Buy-hoz
ICON_SELL_LIMIT  = "🟠"  # Narancs a Limit Sell-hez

COLOR_GREEN  = 0x2ecc71
COLOR_RED    = 0xe74c3c
COLOR_BLUE   = 0x3498db
COLOR_ORANGE = 0xe67e22

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def send_discord_embed(webhook_url, embed_data):
    if not webhook_url:
        print(" [!] Nincs DISCORD_WEBHOOK_URL beállítva!")
        return
    try:
        payload = {"embeds": [embed_data]}
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        print(f" [!] Hiba a küldéskor: {e}")

def format_price(price):
    if price is None: return "N/A"
    try:
        p = float(price)
        if p > 1000: return f"{p:,.1f}"
        if p > 10: return f"{p:.2f}"
        return f"{p:.5f}"
    except:
        return str(price)

def calculate_smart_levels(entry_price, atr, direction):
    """Kiszámolja a hiányzó SL/TP szinteket Limit megbízáshoz"""
    if not entry_price or not atr:
        return None, None, None
    
    entry = float(entry_price)
    vol = float(atr)
    
    # Stratégia: 1.5x ATR Stop, 1.0x ATR TP1, 2.5x ATR TP2
    stop_dist = 1.5 * vol
    tp1_dist  = 1.0 * vol
    tp2_dist  = 2.5 * vol
    
    if direction == "buy":
        sl  = entry - stop_dist
        tp1 = entry + tp1_dist
        tp2 = entry + tp2_dist
    else: # sell
        sl  = entry + stop_dist
        tp1 = entry - tp1_dist
        tp2 = entry - tp2_dist
        
    return sl, tp1, tp2

def check_and_notify():
    assets = [d for d in PUBLIC_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    print(f"🔍 Ellenőrzés: {len(assets)} eszköz...")
    
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
        try:
            atr = data.get("intervention_watch", {}).get("metrics", {}).get("atr5_usd", 0)
        except: pass

        should_notify = False
        embed = {}

        # 1. MARKET JEL
        if signal in ["buy", "sell"]:
            should_notify = True
            is_buy = (signal == "buy")
            color = COLOR_GREEN if is_buy else COLOR_RED
            icon  = ICON_BUY_MARKET if is_buy else ICON_SELL_MARKET
            action = "VÉTEL (Market)" if is_buy else "ELADÁS (Market)"
            
            sl = data.get("sl")
            tp1 = data.get("tp1")
            tp2 = data.get("tp2")

            embed = {
                "title": f"{icon} {action}: {asset_name}",
                "description": "**Azonnali belépés piaci áron!**",
                "color": color,
                "fields": [
                    {"name": "Ár", "value":f"`{format_price(spot)}`", "inline": True},
                    {"name": "Esély", "value":f"`{prob}%`", "inline": True},
                    {"name": "🛑 SL", "value":f"`{format_price(sl)}`", "inline": True},
                    {"name": "🎯 TP1 / TP2", "value":f"`{format_price(tp1)}`\n`{format_price(tp2)}`", "inline": True},
                ],
                "footer": {"text": f"Market Order • {datetime.now(timezone.utc).strftime('%H:%M')} UTC"}
            }

        # 2. LIMIT JEL (Smart)
        elif signal == "precision_arming":
            playbook = data.get("execution_playbook", [])
            state = "unknown"
            trigger_levels = {}
            if playbook:
                state = playbook[-1].get("state")
                trigger_levels = playbook[-1].get("trigger_levels", {})
            
            if state == "fire":
                should_notify = True
                limit_price = trigger_levels.get("fire")
                
                # Irány
                direction = "buy"
                if limit_price and spot and limit_price > spot:
                    direction = "sell"
                
                c_sl, c_tp1, c_tp2 = calculate_smart_levels(limit_price, atr, direction)
                
                if direction == "buy":
                    color = COLOR_BLUE
                    title = f"{ICON_BUY_LIMIT} LIMIT BUY: {asset_name}"
                    desc = "**Tegyél be Vételi Limit megbízást!**"
                else:
                    color = COLOR_ORANGE
                    title = f"{ICON_SELL_LIMIT} LIMIT SELL: {asset_name}"
                    desc = "**Tegyél be Eladási Limit megbízást!**"

                embed = {
                    "title": title,
                    "description": desc,
                    "color": color,
                    "fields": [
                        {"name": "🔵 Limit Ár", "value":f"`{format_price(limit_price)}`", "inline": False},
                        {"name": "🛑 SL (Becsült)", "value":f"`{format_price(c_sl)}`", "inline": True},
                        {"name": "🎯 TP1 / TP2", "value":f"`{format_price(c_tp1)}`\n`{format_price(c_tp2)}`", "inline": True},
                        {"name": "Jelenlegi Ár", "value":f"{format_price(spot)}", "inline": True},
                    ],
                    "footer": {"text": f"Limit Setup • {datetime.now(timezone.utc).strftime('%H:%M')} UTC"}
                }

        if should_notify:
            print(f" -> Üzenet küldése: {asset_name}")
            send_discord_embed(DISCORD_WEBHOOK_URL, embed)

if __name__ == "__main__":
    print("--- Discord Notifier v2.0 (Smart Mode) ---")
    check_and_notify()

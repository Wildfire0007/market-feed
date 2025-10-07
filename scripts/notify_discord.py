#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, requests
from datetime import datetime, timezone

PUBLIC_DIR = "public"
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]

# ---- Debounce/stabilitás beállítások ----
STATE_PATH = f"{PUBLIC_DIR}/_notify_state.json"
STABILITY_RUNS = 2    # ennyi egymás utáni körben legyen BUY/SELL, hogy "aktívnak" számítson
COOLDOWN_MIN   = 0    # ha akarsz, tegyél ide pl. 10-15-öt (perc), hogy ritkábban értesítsen

def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st):
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def fmt_num(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"

def spot_from_sig_or_file(asset: str, sig: dict):
    spot = (sig or {}).get("spot") or {}
    price = spot.get("price") or spot.get("price_usd")
    utc = spot.get("utc") or spot.get("timestamp")
    if price is None:
        js = load(f"{PUBLIC_DIR}/{asset}/spot.json") or {}
        price = js.get("price") or js.get("price_usd")
        utc = utc or js.get("utc") or js.get("timestamp")
    return price, utc

def missing_from_sig(sig: dict):
    gates = (sig or {}).get("gates") or {}
    miss = gates.get("missing") or []
    if not miss:
        return ""
    # kedvesebb megjelenítés
    pretty = {
        "bos5m": "BOS (5m)",
        "fib79": "Fib",
        "atr": "ATR",
        "bias": "Bias",
        "rr_math": "RR≥1.5",
        "liquidity": "liquidity",
        "tp_min_profit": "tp_min_profit",
    }
    names = []
    for k in miss:
        key = k.replace("rr_math", "RR≥1.5")
        names.append(pretty.get(k, key))
    return ", ".join(names)

def fmt_sig(asset: str, sig: dict):
    dec = (sig.get("signal") or "no entry").upper()
    p   = sig.get("probability", 0)
    entry = sig.get("entry"); sl = sig.get("sl"); t1 = sig.get("tp1"); t2 = sig.get("tp2")
    rr = sig.get("rr")

    price, utc = spot_from_sig_or_file(asset, sig)
    spot_s = fmt_num(price)
    utc_s  = utc or "-"

    base = f"• {asset}: {dec} | Spot: {spot_s} | P={p}% | UTC: {utc_s}"

    if dec in ("BUY", "SELL") and all(v is not None for v in (entry, sl, t1, t2)):
        base += (f" | @ {fmt_num(entry)} | SL {fmt_num(sl)} | "
                 f"TP1 {fmt_num(t1)} | TP2 {fmt_num(t2)} | RR≈{rr}")

    miss = missing_from_sig(sig)
    if (dec not in ("BUY", "SELL")) and miss:
        base += f" | Hiányzó: {miss}"
    return base

def main():
    hook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not hook:
        print("No DISCORD_WEBHOOK_URL, skipping notify.")
        return

    state = load_state()

    lines = []
    actionable = False
    for asset in ASSETS:
        sig = load(f"{PUBLIC_DIR}/{asset}/signal.json")
        if not sig:
            summ = load(f"{PUBLIC_DIR}/analysis_summary.json") or {}
            sig = (summ.get("assets") or {}).get(asset)
        if not sig:
            sig = {"asset": asset, "signal": "no entry", "probability": 0}

        # Stabilitás-számláló frissítése
        key = asset
        prev = state.get(key, {"last": None, "count": 0, "last_sent": None})

        curr = (sig.get("signal") or "no entry").lower()
        curr_effective = curr if curr in ("buy", "sell") else "no entry"

        if curr_effective == prev.get("last"):
            prev["count"] = int(prev.get("count", 0)) + 1
        else:
            prev["last"] = curr_effective
            prev["count"] = 1

        state[key] = prev

        # Stabil BUY/SELL-e?
        is_stable_actionable = (curr_effective in ("buy","sell") and prev["count"] >= STABILITY_RUNS)
        if is_stable_actionable:
            actionable = True

        # Sor render
        line = fmt_sig(asset, sig)
        if curr in ("buy","sell") and not is_stable_actionable:
            line += " | Állapot: stabilizálás alatt"
        lines.append(line)

    save_state(state)

    title = "📣 TD Jelentés — Automatikus Discord értesítés"
    header = (f"{title}\nAktív jelzés(ek) találhatók:\n"
              if actionable else f"{title}\nÖsszefoglaló (no entry / várakozás):\n")
    content = header + "\n".join(lines)

    if len(content) > 1900:
        content = content[:1900] + "\n…"

    try:
        r = requests.post(hook, json={"content": content}, timeout=20)
        r.raise_for_status()
        print("Discord notify OK.")
    except Exception as e:
        print("Discord notify FAILED:", e)

if __name__ == "__main__":
    main()

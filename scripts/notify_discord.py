#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, requests, pathlib

PUBLIC_DIR = "public"

def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def fmt_sig(sig):
    dec = sig.get("signal","no entry").upper()
    p   = sig.get("probability")
    rr  = sig.get("rr")
    e   = sig.get("entry"); sl = sig.get("sl"); t1 = sig.get("tp1"); t2 = sig.get("tp2")
    if dec == "NO ENTRY":
        return f"• {sig['asset']}: no entry (P={p}%)"
    return (f"• {sig['asset']}: {dec} @ {e:.4f} | SL {sl:.4f} | "
            f"TP1 {t1:.4f} | TP2 {t2:.4f} | P={p}% | RR≈{rr}")

def main():
    hook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not hook:
        print("No DISCORD_WEBHOOK_URL, skipping notify.")
        return

    summ = load(f"{PUBLIC_DIR}/analysis_summary.json") or {}
    assets = summ.get("assets", {})
    lines = []
    actionable = False

    # Elsődlegesen az assetenkénti signal.json-ból olvassunk (részletesebb lehet)
    for asset in ["SOL","NSDQ100","GOLD_CFD"]:
        sig = load(f"{PUBLIC_DIR}/{asset}/signal.json")
        if not sig:
            sig = assets.get(asset) or {"asset":asset, "signal":"no entry", "probability":0}
        lines.append(fmt_sig(sig))
        if sig.get("signal") in ("buy","sell"):
            actionable = True

    title = "📣 TD Jelentés — Automatikus Discord értesítés"
    if actionable:
        header = f"{title}\nAktív jelzés(ek) találhatók:\n"
    else:
        header = f"{title}\nNincs aktív jelzés (összefoglaló):\n"

    content = header + "\n".join(lines)
    # Discord 2000 karakter limit
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

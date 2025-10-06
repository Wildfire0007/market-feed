#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, requests

PUBLIC_DIR = "public"
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]

def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def fmt_num(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"

def spot_from_sig_or_file(asset: str, sig: dict):
    # 1) signal.json-ból próbáljuk
    spot = (sig or {}).get("spot") or {}
    price = spot.get("price") or spot.get("price_usd")
    utc = spot.get("utc") or spot.get("timestamp")

    # 2) ha nincs, olvassuk a public/<ASSET>/spot.json-t
    if price is None:
        js = load(f"{PUBLIC_DIR}/{asset}/spot.json") or {}
        price = js.get("price") or js.get("price_usd")
        utc = utc or js.get("utc") or js.get("timestamp")

    return price, utc

# --- hiányzó feltételek kinyerése és formázása -------------------------------

def extract_missing(sig: dict):
    """Visszaadja a hiányzó kapukat listában.
       Először gates.missing, ha nincs, akkor a reasons 'missing: ...' sort parsolja."""
    if not isinstance(sig, dict):
        return []
    gates = sig.get("gates") or {}
    missing = gates.get("missing") or []
    if missing:
        return [str(x).strip() for x in missing if str(x).strip()]

    # fallback: reasons között keres 'missing: a, b, c' sort
    for r in sig.get("reasons") or []:
        if isinstance(r, str) and r.lower().startswith("missing:"):
            rest = r.split(":", 1)[1]
            return [p.strip() for p in rest.split(",") if p.strip()]
    return []

def pretty_missing(keys):
    """Kulcsokból emberi olvasmányos címkék."""
    labels = {
        "bias": "Bias (4H→1H)",
        "bos5m": "BOS (5m)",
        "fib79": "Fib 79%",
        "atr": "ATR",
        "rr_math": "RR≥1.5",
        "rr_math>=1.5": "RR≥1.5",
    }
    return ", ".join(labels.get(k, k) for k in keys)

# --- Discord sor formázó ------------------------------------------------------

def fmt_sig(asset: str, sig: dict):
    dec = (sig.get("signal") or "no entry").upper()
    p   = sig.get("probability", 0)
    entry = sig.get("entry"); sl = sig.get("sl"); t1 = sig.get("tp1"); t2 = sig.get("tp2")
    rr = sig.get("rr")

    price, utc = spot_from_sig_or_file(asset, sig)
    spot_s = fmt_num(price)
    utc_s  = utc or "-"

    # Mindig legyen Spot + P%, még no entry esetén is
    base = f"• {asset}: {dec} | Spot: {spot_s} | P={p}% | UTC: {utc_s}"

    if dec in ("BUY", "SELL") and all(v is not None for v in (entry, sl, t1, t2)):
        return (base +
                f" | @ {fmt_num(entry)} | SL {fmt_num(sl)} | "
                f"TP1 {fmt_num(t1)} | TP2 {fmt_num(t2)} | RR≈{rr}")

    # NO ENTRY eset: hiányzó feltételek kiírása
    miss = extract_missing(sig)
    if miss:
        base += f" | Hiányzó: {pretty_missing(miss)}"
    return base

# --- fő -----------------------------------------------------------------------

def main():
    hook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not hook:
        print("No DISCORD_WEBHOOK_URL, skipping notify.")
        return

    lines = []
    actionable = False
    for asset in ASSETS:
        sig = load(f"{PUBLIC_DIR}/{asset}/signal.json")
        # Ha nincs külön signal.json, próbáljuk a summary-t
        if not sig:
            summ = load(f"{PUBLIC_DIR}/analysis_summary.json") or {}
            sig = (summ.get("assets") or {}).get(asset)
        if not sig:
            sig = {"asset": asset, "signal": "no entry", "probability": 0}

        lines.append(fmt_sig(asset, sig))
        if (sig.get("signal") or "").lower() in ("buy", "sell"):
            actionable = True

    title = "📣 TD Jelentés — Automatikus Discord értesítés"
    header = (f"{title}\nAktív jelzés(ek) találhatók:\n"
              if actionable else f"{title}\nÖsszefoglaló (no entry / várakozás):\n")
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

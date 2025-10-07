#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, requests
from datetime import datetime, timezone

PUBLIC_DIR = "public"
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD", "BNB", "GER40"]

# ---- Debounce/stabilitás ----
STATE_PATH = f"{PUBLIC_DIR}/_notify_state.json"
STABILITY_RUNS = 2     # ennyi körben legyen BUY/SELL, hogy "aktívnak" számítson
COOLDOWN_MIN   = 0     # (ha kell, tegyél ide 10–15-öt)

# ---- Megjelenés / emoji / színek ----
EMOJI = {
    "SOL": "💵",
    "NSDQ100": "📈",
    "GOLD_CFD": "💰",
    "BNB": "🪙",
    "GER40": "🇩🇪",
}
COLOR = {
    "BUY":  0x2ecc71,  # zöld
    "SELL": 0x2ecc71,  # zöld (ha külön akarod: 0x00b894)
    "NO":   0xe74c3c,  # piros
    "WAIT": 0xf1c40f,  # sárga (stabilizálás)
}

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
    pretty = {
        "bos5m": "BOS (5m)",
        "atr": "ATR",
        "bias": "Bias",
        "liquidity": "liquidity",
        "tp_min_profit": "tp_min_profit",
        "RR≥1.5": "RR≥1.5",
        "rr_math>=2.0": "RR≥2.0",
    }
    out = []
    for k in miss:
        key = "RR≥2.0" if k.startswith("rr_math") else k
        out.append(pretty.get(k, key))
    return ", ".join(out)

def build_embed_for_asset(asset: str, sig: dict, is_stable: bool):
    emoji = EMOJI.get(asset, "📊")
    dec_raw = (sig.get("signal") or "no entry").upper()
    dec = dec_raw
    if dec not in ("BUY", "SELL"):
        dec = "NO ENTRY"

    p   = int(sig.get("probability", 0) or 0)
    entry = sig.get("entry"); sl = sig.get("sl"); t1 = sig.get("tp1"); t2 = sig.get("tp2")
    rr = sig.get("rr")

    price, utc = spot_from_sig_or_file(asset, sig)
    spot_s = fmt_num(price)
    utc_s  = utc or "-"

    # státusz sor (színezett jelöléssel)
    status_emoji = "🟢" if dec in ("BUY","SELL") else "🔴"
    status_bold  = f"{status_emoji} **{dec}**"

    lines = [
        f"{status_bold} • P={p}%",
        f"Spot: `{spot_s}` • UTC: `{utc_s}`",
    ]

    if dec in ("BUY", "SELL") and all(v is not None for v in (entry, sl, t1, t2, rr)):
        lines.append(f"@ `{fmt_num(entry)}` • SL `{fmt_num(sl)}` • TP1 `{fmt_num(t1)}` • TP2 `{fmt_num(t2)}` • RR≈`{rr}`")
        if not is_stable:
            lines.append("⏳ Állapot: *stabilizálás alatt*")

    if dec == "NO ENTRY":
        miss = missing_from_sig(sig)
        if miss:
            lines.append(f"Hiányzó: *{miss}*")

    color = COLOR["WAIT"] if (dec in ("BUY","SELL") and not is_stable) else (COLOR["BUY"] if dec in ("BUY","SELL") else COLOR["NO"])

    return {
        "title": f"{emoji} **{asset}**",
        "description": "\n".join(lines),
        "color": color,
    }

def main():
    hook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not hook:
        print("No DISCORD_WEBHOOK_URL, skipping notify.")
        return

    state = load_state()
    embeds = []
    actionable = False

    for asset in ASSETS:
        sig = load(f"{PUBLIC_DIR}/{asset}/signal.json")
        if not sig:
            summ = load(f"{PUBLIC_DIR}/analysis_summary.json") or {}
            sig = (summ.get("assets") or {}).get(asset)
        if not sig:
            sig = {"asset": asset, "signal": "no entry", "probability": 0}

        # Stabilitás számláló
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

        is_stable_actionable = (curr_effective in ("buy","sell") and prev["count"] >= STABILITY_RUNS)
        if is_stable_actionable:
            actionable = True

        embeds.append(build_embed_for_asset(asset, sig, is_stable_actionable))

    save_state(state)

    title = "📣 eToro-Riasztás"
    header = "Aktív jelzés(ek) találhatók:" if actionable else "Összefoglaló (no entry / várakozás):"
    content = f"**{title}**\n{header}"

    # Webhook küldés (content + embeds)
    try:
        r = requests.post(hook, json={"content": content, "embeds": embeds[:10]}, timeout=20)
        r.raise_for_status()
        print("Discord notify OK.")
    except Exception as e:
        print("Discord notify FAILED:", e)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — Esemény-alapú Discord riasztó

Mikor küld üzenetet?
- STABIL (>= STABILITY_RUNS) BUY/SELL jelzésnél  ➜ "normal"
- Ellenirányú stabil jel flipnél                 ➜ "flip"
- Ha a korábban küldött BUY/SELL stabilan NO ENTRY-be fordul ➜ "invalidate"
- COOLDOWN_MIN perc védi a spammelt (ENV: DISCORD_COOLDOWN_MIN)

Kimenet: színes embedek, asset-enként külön kártya, emoji + félkövér cím,
hiányzó kapuk szépen formázva.
"""

import os, json, requests
from datetime import datetime, timezone

PUBLIC_DIR = "public"

# --- VÉGSŐ ASSET LISTA (GER40 -> USOIL) ---
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD", "BNB", "USOIL"]

# ---- Debounce / stabilitás / cooldown ----
STATE_PATH = f"{PUBLIC_DIR}/_notify_state.json"
STABILITY_RUNS = 2                                   # ennyi egymás utáni körben legyen BUY/SELL/NO ENTRY, hogy "stabil"
COOLDOWN_MIN   = int(os.getenv("DISCORD_COOLDOWN_MIN", "10"))  # perc; 0 = kikapcsolva

# ---- Megjelenés / emoji / színek ----
EMOJI = {
    "SOL": "💵",
    "NSDQ100": "📈",
    "GOLD_CFD": "💰",
    "BNB": "🪙",
    "USOIL": "🛢️",
}
COLOR = {
    "BUY":   0x2ecc71,  # zöld
    "SELL":  0x2ecc71,  # zöld (ha külön akarod: 0x00b894)
    "NO":    0xe74c3c,  # piros (no entry / invalid)
    "WAIT":  0xf1c40f,  # sárga (stabilizálás)
    "FLIP":  0x3498db,  # kék (ellenirányú flip)
}

# ---------------- util ----------------

def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def utcnow_epoch():
    return int(datetime.now(timezone.utc).timestamp())

def iso_to_epoch(s: str) -> int:
    try:
        return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp())
    except Exception:
        return 0

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
        "session": "Session",
        "regime": "Regime",
        "bias": "Bias",
        "bos5m": "BOS (5m)",
        "liquidity": "Liquidity",
        "liquidity(fib_zone|sweep)": "Liquidity",
        "atr": "ATR",
        "tp_min_profit": "TP min. profit",
        "RR≥1.5": "RR≥1.5",
        "rr_math>=2.0": "RR≥2.0",
        # momentum
        "momentum(ema9x21)": "Momentum (EMA9×21)",
        "bos5m|struct_break": "BOS/Structure",
    }
    out = []
    for k in miss:
        key = "RR≥2.0" if k.startswith("rr_math") else k
        out.append(pretty.get(k, key))
    # uniq + formázott
    return ", ".join(dict.fromkeys(out).keys())

def gates_mode(sig: dict) -> str:
    return ((sig or {}).get("gates") or {}).get("mode") or "-"

def decision_of(sig: dict) -> str:
    d = (sig or {}).get("signal", "no entry")
    d = (d or "").lower()
    if d not in ("buy","sell"):
        return "no entry"
    return d

# ------------- embed-renderek -------------

def build_embed_for_asset(asset: str, sig: dict, is_stable: bool, kind: str = "normal", prev_decision: str = None):
    """
    kind: "normal" | "invalidate" | "flip"
    """
    emoji = EMOJI.get(asset, "📊")
    dec_raw = (sig.get("signal") or "no entry").upper()
    dec = dec_raw if dec_raw in ("BUY","SELL") else "NO ENTRY"

    p   = int(sig.get("probability", 0) or 0)
    entry = sig.get("entry"); sl = sig.get("sl"); t1 = sig.get("tp1"); t2 = sig.get("tp2")
    rr = sig.get("rr")
    mode = gates_mode(sig)

    price, utc = spot_from_sig_or_file(asset, sig)
    spot_s = fmt_num(price)
    utc_s  = utc or "-"

    # státusz
    status_emoji = "🟢" if dec in ("BUY","SELL") else "🔴"
    status_bold  = f"{status_emoji} **{dec}**"

    lines = [f"{status_bold} • P={p}% • mód: `{mode}`",
             f"Spot: `{spot_s}` • UTC: `{utc_s}`"]

    if dec in ("BUY", "SELL") and all(v is not None for v in (entry, sl, t1, t2, rr)):
        lines.append(f"@ `{fmt_num(entry)}` • SL `{fmt_num(sl)}` • TP1 `{fmt_num(t1)}` • TP2 `{fmt_num(t2)}` • RR≈`{rr}`")
        if not is_stable and kind == "normal":
            lines.append("⏳ Állapot: *stabilizálás alatt*")

    if dec == "NO ENTRY":
        miss = missing_from_sig(sig)
        if miss:
            lines.append(f"Hiányzó: *{miss}*")

    # cím + szín
    title = f"{emoji} **{asset}**"
    if kind == "invalidate":
        title += " • ❌ Invalidate"
        color = COLOR["NO"]
    elif kind == "flip":
        arrow = "→"
        title += f" • 🔁 Flip ({(prev_decision or '').upper()} {arrow} {dec})"
        color = COLOR["FLIP"]
    else:
        color = COLOR["WAIT"] if (dec in ("BUY","SELL") and not is_stable) else (COLOR["BUY"] if dec in ("BUY","SELL") else COLOR["NO"])

    return {
        "title": title,
        "description": "\n".join(lines),
        "color": color,
    }

# ---------------- főlogika ----------------

def main():
    hook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not hook:
        print("No DISCORD_WEBHOOK_URL, skipping notify.")
        return

    state = load_state()
    embeds = []
    actionable_any = False
    now_iso = utcnow_iso()
    now_ep  = utcnow_epoch()

    for asset in ASSETS:
        sig = load(f"{PUBLIC_DIR}/{asset}/signal.json")
        if not sig:
            summ = load(f"{PUBLIC_DIR}/analysis_summary.json") or {}
            sig = (summ.get("assets") or {}).get(asset)
        if not sig:
            sig = {"asset": asset, "signal": "no entry", "probability": 0}

        # --- stabilitás számítása (buy/sell/no entry effektív) ---
        eff = decision_of(sig)  # 'buy' | 'sell' | 'no entry'

        # per-asset state init
        st = state.get(asset, {
            "last": None, "count": 0,
            "last_sent": None,               # ISO
            "last_sent_decision": None,      # 'buy'|'sell'|'no entry'
            "last_sent_mode": None,          # 'core'|'momentum'|None
            "cooldown_until": None,          # ISO
        })

        # stabil számláló
        if eff == st.get("last"):
            st["count"] = int(st.get("count", 0)) + 1
        else:
            st["last"]  = eff
            st["count"] = 1

        # flags
        is_stable = st["count"] >= STABILITY_RUNS
        is_actionable_now = (eff in ("buy","sell")) and is_stable
        actionable_any = actionable_any or is_actionable_now

        cooldown_until_iso = st.get("cooldown_until")
        cooldown_active = False
        if COOLDOWN_MIN > 0 and cooldown_until_iso:
            cooldown_active = now_ep < iso_to_epoch(cooldown_until_iso)

        prev_sent_decision = st.get("last_sent_decision")

        # --- küldési döntés ---
        send_kind = None  # None | "normal" | "invalidate" | "flip"

        if is_actionable_now:
            if prev_sent_decision in ("buy","sell"):
                if eff != prev_sent_decision:
                    # Ellenirányú stabil jelzés — FLIP mindig mehet
                    send_kind = "flip"
                else:
                    # Ugyanaz az irány stabilan — cooldown védi a spammelt
                    if not cooldown_active:
                        send_kind = "normal"
            else:
                # Először lesz stabilan actionable
                if not cooldown_active:
                    send_kind = "normal"
        else:
            # Nem actionable: ha korábban actionable-t küldtünk, és most stabil "no entry", INVALIDATE
            if prev_sent_decision in ("buy","sell") and eff == "no entry" and is_stable:
                send_kind = "invalidate"

        # --- embed + állapot frissítés ---
        if send_kind:
            embeds.append(build_embed_for_asset(asset, sig, is_stable, kind=send_kind, prev_decision=prev_sent_decision))
            if send_kind in ("normal","flip"):
                # új akció bejelentve → cooldown indítás / frissítés
                if COOLDOWN_MIN > 0:
                    st["cooldown_until"] = datetime.fromtimestamp(
                        now_ep + COOLDOWN_MIN*60, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                st["last_sent"] = now_iso
                st["last_sent_decision"] = eff
                st["last_sent_mode"] = gates_mode(sig)
            elif send_kind == "invalidate":
                # érvénytelenítettük a korábbi jelet
                st["last_sent"] = now_iso
                st["last_sent_decision"] = "no entry"
                st["last_sent_mode"] = None
                # cooldownt nem piszkáljuk
        state[asset] = st

    save_state(state)

    if not embeds:
        print("Discord notify: nothing to send (no embeds after cooldown/invalidate logic).")
        return

    title = "📣 eToro-Riasztás"
    header = "Aktív jelzés(ek):" if actionable_any else "Változás / érvénytelenítés:"
    content = f"**{title}**\n{header}"

    try:
        r = requests.post(hook, json={"content": content, "embeds": embeds[:10]}, timeout=20)
        r.raise_for_status()
        print("Discord notify OK.")
    except Exception as e:
        print("Discord notify FAILED:", e)

if __name__ == "__main__":
    main()

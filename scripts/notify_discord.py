#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notify_discord.py — Esemény-alapú Discord riasztó + óránkénti összefoglaló (per-eszköz panelek)

Stílus:
- Külön embed minden eszköznek, saját emojival.
- Félkövér eszköznév a címben. A leírás elején 🟢/🔴 ikon.
- BUY/SELL = zöld sáv, NO ENTRY = piros sáv, stabilizálás alatt = sárga sáv.
- RR/TP/SL/Entry számok backtick-ben.

Küldés:
- STABIL (>= STABILITY_RUNS) BUY/SELL ➜ "normal"
- Ellenirányú stabil jel flip ➜ "flip"
- Korábban küldött BUY/SELL stabilan NO ENTRY ➜ "invalidate"
- ÓRÁNKÉNTI HEARTBEAT (minden órában), akkor is, ha nincs riasztás.
  Ha adott órában már ment event (normal/flip/invalidate), külön heartbeat nem megy ki.
  --force / --manual kapcsolóval (vagy DISCORD_FORCE_NOTIFY=1) kézi futtatáskor is kimegy az összefoglaló.
  Kézi futtatáskor elfogadjuk a "manual"/"force" kulcsszavakat is flag nélkül.

ENV:
- DISCORD_WEBHOOK_URL
- DISCORD_COOLDOWN_MIN (perc, default 10)
- DISCORD_FORCE_NOTIFY=1 ➜ cooldown figyelmen kívül hagyása + összefoglaló kényszerítése
- DISCORD_FORCE_HEARTBEAT=1 ➜ csak az összefoglalót kényszerítjük (cooldown marad)
"""

import os, json, sys, requests
from typing import Iterable, Set
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Py3.9+

PUBLIC_DIR = "public"
ASSETS = ["EURUSD", "NSDQ100", "GOLD_CFD", "USDJPY", "USOIL"]

# ---- Debounce / stabilitás / cooldown ----
STATE_PATH = f"{PUBLIC_DIR}/_notify_state.json"
STABILITY_RUNS = 2
def int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(
            f"WARN: {name}='{raw}' nem értelmezhető egész számként, {default}-ot használunk.",
            file=sys.stderr,
        )
        return default


COOLDOWN_MIN   = int_env("DISCORD_COOLDOWN_MIN", 10)  # perc; 0 = off
MOMENTUM_COOLDOWN_MIN = int_env("DISCORD_COOLDOWN_MOMENTUM_MIN", 8)

def env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if not raw:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def normalize_cli_flags(argv: Iterable[str]) -> Set[str]:
    """Egyszerű (argparse nélküli) flag normalizálás kézi futtatáshoz."""
    norm: Set[str] = set()
    for arg in argv:
        if not arg:
            continue
        if arg.startswith("--"):
            norm.add(arg[2:].lower())
        elif arg.startswith("-"):
            # jelenleg csak -f/-F érdekes, de legyen rugalmas
            if arg.lower() in {"-f", "-force"}:
                norm.add("force")
            else:
                norm.add(arg[1:].lower())
        else:
            norm.add(arg.lower())
    return norm

# ---- Időzóna a fejlécben / órakulcshoz ----
try:
    HB_TZ = ZoneInfo("Europe/Budapest")
except Exception as exc:  # pragma: no cover - környezeti hiányosságokra
    print(
        "WARN: Europe/Budapest időzóna nem elérhető, UTC-re esünk vissza.",
        f"({exc})",
        file=sys.stderr,
    )
    HB_TZ = timezone.utc

# ---- Megjelenés / emoji / színek ----
EMOJI = {
    "EURUSD": "💶",
    "NSDQ100": "📈",
    "GOLD_CFD": "💰",
    "USDJPY": "💴",
    "USOIL": "🛢️",
}
COLOR = {
    "BUY":   0x2ecc71,  # zöld
    "SELL":  0x2ecc71,  # zöld
    "NO":    0xe74c3c,  # piros
    "WAIT":  0xf1c40f,  # sárga
    "FLIP":  0x3498db,  # kék
    "INFO":  0x95a5a6,  # semleges
}

# ---------------- util ----------------

def bud_now():
    return datetime.now(HB_TZ)

def bud_hh_key(dt=None) -> str:
    dt = dt or bud_now()
    return dt.strftime("%Y%m%d%H")

def bud_time_str(dt=None) -> str:
    dt = dt or bud_now()
    return dt.strftime("%Y-%m-%d %H:%M ") + (dt.tzname() or "CET")

def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def utcnow_epoch():
    return int(datetime.now(timezone.utc).timestamp())

def iso_to_epoch(s: str) -> int:
    try:
        return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp())
    except Exception:
        return 0

def parse_utc(value):
    if value is None or value == "-":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                return None
    return None


def to_utc_iso(dt):
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

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
    spot_ts = parse_utc(utc)

    kline_data = load(f"{PUBLIC_DIR}/{asset}/klines_5m.json") or {}
    rows = []
    if isinstance(kline_data, dict):
        rows = kline_data.get("values") or []
    elif isinstance(kline_data, list):
        rows = kline_data

    fallback_price = None
    fallback_ts = None
    for row in rows:
        ts_raw = row.get("datetime") or row.get("t")
        close_raw = row.get("close") or row.get("c")
        if ts_raw is None or close_raw in (None, ""):
            continue
        ts = parse_utc(ts_raw)
        if ts is None:
            continue
        try:
            close_val = float(close_raw)
        except (TypeError, ValueError):
            continue
        if fallback_ts is None or ts > fallback_ts:
            fallback_ts = ts
            fallback_price = close_val

    use_fallback = False
    if fallback_price is not None:
        if price is None or spot_ts is None:
            use_fallback = True
        elif fallback_ts and spot_ts and fallback_ts > spot_ts:
            use_fallback = True

    if use_fallback:
        price = fallback_price
        utc = to_utc_iso(fallback_ts) or utc
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
        "liquidity(fib|sweep|ema21|retest)": "Liquidity",
        "atr": "ATR",
        "tp_min_profit": "TP min. profit",
        "tp1_net>=+1.0%": "TP1 nettó ≥ +1.0%",
        "min_stoploss": "Minimum stoploss",
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
    uniq = list(dict.fromkeys(out))
    return ", ".join(uniq)

def gates_mode(sig: dict) -> str:
    return ((sig or {}).get("gates") or {}).get("mode") or "-"

def decision_of(sig: dict) -> str:
    d = (sig or {}).get("signal", "no entry")
    d = (d or "").lower()
    if d not in ("buy","sell"):
        return "no entry"
    return d

# ------------- embed-renderek -------------

def card_color(dec: str, is_stable: bool, kind: str) -> int:
    if kind == "flip":
        return COLOR["FLIP"]
    if kind == "invalidate":
        return COLOR["NO"]
    if dec in ("BUY","SELL"):
        return COLOR["BUY"] if is_stable else COLOR["WAIT"]
    return COLOR["NO"]

def build_embed_for_asset(asset: str, sig: dict, is_stable: bool, kind: str = "normal", prev_decision: str = None):
    """
    kind: "normal" | "invalidate" | "flip" | "heartbeat"
    """
    emoji = EMOJI.get(asset, "📊")
    dec_raw = (sig.get("signal") or "no entry").upper()
    dec = dec_raw if dec_raw in ("BUY","SELL") else "NO ENTRY"

    p   = int(sig.get("probability", 0) or 0)
    entry = sig.get("entry"); sl = sig.get("sl"); t1 = sig.get("tp1"); t2 = sig.get("tp2")
    rr = sig.get("rr")
    mode = gates_mode(sig)
    missing_list = ((sig.get("gates") or {}).get("missing") or [])
    core_bos_pending = (mode == "core") and ("bos5m" in missing_list)

    price, utc = spot_from_sig_or_file(asset, sig)
    spot_s = fmt_num(price)
    utc_s  = utc or "-"

    # státusz
    if core_bos_pending and dec in ("BUY","SELL"):
        status_emoji = "🟡"
    else:
        status_emoji = "🟢" if dec in ("BUY","SELL") else "🔴"
    status_bold  = f"{status_emoji} **{dec}**"

    lines = [
        f"{status_bold} • P={p}% • mód: `{mode}`",
        f"Spot: `{spot_s}` • UTC: `{utc_s}`",
    ]

    # RR/TP/SL sor (ha minden adat megvan)
    if dec in ("BUY", "SELL") and all(v is not None for v in (entry, sl, t1, t2, rr)):
        lines.append(f"@ `{fmt_num(entry)}` • SL `{fmt_num(sl)}` • TP1 `{fmt_num(t1)}` • TP2 `{fmt_num(t2)}` • RR≈`{rr}`")
    # Stabilizálás információ
    if dec in ("BUY","SELL") and kind in ("normal","heartbeat"):
        if core_bos_pending:
            lines.append("⏳ Állapot: *stabilizálás alatt (5m BOS megerősítésre várunk)*")
        elif not is_stable:
            lines.append("⏳ Állapot: *stabilizálás alatt*")

    # Hiányzó feltételek — ha vannak, mindig mutatjuk
    miss = missing_from_sig(sig)
    if miss:
        lines.append(f"Hiányzó: *{miss}*")

    # cím + szín
    title = f"{emoji} **{asset}**"
    if kind == "invalidate":
        title += " • ❌ Invalidate"
    elif kind == "flip":
        arrow = "→"
        title += f" • 🔁 Flip ({(prev_decision or '').upper()} {arrow} {dec})"
    elif kind == "heartbeat":
        title += " • ℹ️ Állapot"

    color = card_color(dec, is_stable, kind)

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

    argv = sys.argv[1:]
    flags = normalize_cli_flags(argv)
  
    force_env = env_flag("DISCORD_FORCE_NOTIFY")
    force_heartbeat_env = env_flag("DISCORD_FORCE_HEARTBEAT")
  
    manual_flag = any(key in flags for key in {"manual", "m", "man"})
    force_flag = any(key in flags for key in {"force", "f"})
    heartbeat_flag = any(key in flags for key in {"force-heartbeat", "heartbeat", "hb"})
    skip_cooldown_flag = any(key in flags for key in {"skip-cooldown", "no-cooldown", "nocooldown"})

    # Ha TTY-ból futtatjuk kézzel és nincs külön flag, tekintsük manuális kényszerítésnek.
    if not flags and (sys.stdin.isatty() or sys.stdout.isatty()):
        manual_flag = True

    force_send = force_flag or manual_flag or force_env or skip_cooldown_flag
    force_heartbeat = force_send or heartbeat_flag or force_heartbeat_env

    state = load_state()
    meta  = state.get("_meta", {})
    embeds = []
    actionable_any = False
    now_iso = utcnow_iso()
    now_ep  = utcnow_epoch()
    bud_dt  = bud_now()
    bud_key = bud_hh_key(bud_dt)

    per_asset_sigs = {}
    per_asset_is_stable = {}

    for asset in ASSETS:
        sig = load(f"{PUBLIC_DIR}/{asset}/signal.json")
        if not sig:
            summ = load(f"{PUBLIC_DIR}/analysis_summary.json") or {}
            sig = (summ.get("assets") or {}).get(asset)
        if not sig:
            sig = {"asset": asset, "signal": "no entry", "probability": 0}
        per_asset_sigs[asset] = sig

        # --- stabilitás számítása ---
        mode_current = gates_mode(sig)
        eff = decision_of(sig)  # 'buy' | 'sell' | 'no entry'

        st = state.get(asset, {
            "last": None, "count": 0,
            "last_sent": None,
            "last_sent_decision": None,
            "last_sent_mode": None,
            "cooldown_until": None,
        })

        if eff == st.get("last"):
            st["count"] = int(st.get("count", 0)) + 1
        else:
            st["last"]  = eff
            st["count"] = 1

        missing_list = ((sig.get("gates") or {}).get("missing") or [])
        core_bos_pending = (mode_current == "core") and ("bos5m" in missing_list)

        is_stable = st["count"] >= STABILITY_RUNS
        display_stable = is_stable and not core_bos_pending
        per_asset_is_stable[asset] = display_stable
        is_actionable_now = (eff in ("buy","sell")) and is_stable and not core_bos_pending
        actionable_any = actionable_any or is_actionable_now

        cooldown_until_iso = st.get("cooldown_until")
        cooldown_active = False
        if COOLDOWN_MIN > 0 and cooldown_until_iso:
            cooldown_active = now_ep < iso_to_epoch(cooldown_until_iso)
        if force_send:
            cooldown_active = False

        prev_sent_decision = st.get("last_sent_decision")

        # --- küldési döntés ---
        send_kind = None  # None | "normal" | "invalidate" | "flip"

        if is_actionable_now:
            if prev_sent_decision in ("buy","sell"):
                if eff != prev_sent_decision:
                    send_kind = "flip"
                else:
                    if not cooldown_active:
                        send_kind = "normal"
            else:
                if not cooldown_active:
                    send_kind = "normal"
        else:
            if prev_sent_decision in ("buy","sell") and eff == "no entry" and is_stable:
                send_kind = "invalidate"

        # --- embed + állapot frissítés ---
        if send_kind:
            embeds.append(build_embed_for_asset(asset, sig, display_stable, kind=send_kind, prev_decision=prev_sent_decision))
            if send_kind in ("normal","flip"):
                cooldown_minutes = COOLDOWN_MIN
                if COOLDOWN_MIN > 0 and mode_current == "momentum":
                    cooldown_minutes = MOMENTUM_COOLDOWN_MIN
                if cooldown_minutes > 0:
                    st["cooldown_until"] = datetime.fromtimestamp(
                        now_ep + cooldown_minutes*60, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    st["cooldown_until"] = None
                st["last_sent"] = now_iso
                st["last_sent_decision"] = eff
                st["last_sent_mode"] = mode_current
                meta["last_heartbeat_key"] = bud_key
            elif send_kind == "invalidate":
                st["last_sent"] = now_iso
                st["last_sent_decision"] = "no entry"
                st["last_sent_mode"] = None
                meta["last_heartbeat_key"] = bud_key

        state[asset] = st

    # --- Heartbeat: MINDEN órában, ha az órában még nem ment ki event ---
    last_hb_key = meta.get("last_heartbeat_key")
    heartbeat_due = last_hb_key != bud_key
    want_heartbeat = force_heartbeat or (not embeds and heartbeat_due)
    heartbeat_embeds = []
    if want_heartbeat:
        for asset in ASSETS:
            sig = per_asset_sigs.get(asset) or {"asset": asset, "signal": "no entry", "probability": 0}
            is_stable = per_asset_is_stable.get(asset, True)
            heartbeat_embeds.append(
                build_embed_for_asset(asset, sig, is_stable=is_stable, kind="heartbeat")
            )

        if heartbeat_embeds:
            if embeds and not force_heartbeat:
                # már van küldendő embed, de nem force-olt heartbeat -> kihagyjuk
                heartbeat_embeds = []
            else:
                embeds.extend(heartbeat_embeds)
                meta["last_heartbeat_key"] = bud_key

    state["_meta"] = meta
    save_state(state)

    if not embeds:
        print("Discord notify: nothing to send.")
        return

    # Fejléc: Budapest-idővel
    bud_str = bud_time_str(bud_dt)
    title  = f"📣 eToro-Riasztás • Budapest: {bud_str}"
    header = "Aktív jelzés(ek):" if actionable_any else "Összefoglaló / változás:"
    content = f"**{title}**\n{header}"

    try:
        r = requests.post(hook, json={"content": content, "embeds": embeds[:10]}, timeout=20)
        r.raise_for_status()
        print("Discord notify OK.")
    except Exception as e:
        print("Discord notify FAILED:", e)

if __name__ == "__main__":
    main()

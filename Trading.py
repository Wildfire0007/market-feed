#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trading.py – GitHub Actions alatt fut. 
Feladat: friss adatok lekérése a TwelveData API-ról és kimentés a GitHub Pages 'public/' mappába.

Eszközök (alias → TwelveData ticker):
  - SOL       → SOL/USD           (crypto)
  - QQQ       → QQQ               (ETF, NASDAQ-100 tracker)
  - GOLD_CFD  → XAU/USD           (arany spot; mappanév és "asset" mező: GOLD_CFD)

Kimenetek (mindig létrejönnek – hiba esetén helykitöltő JSON-ok):
  public/[ALIAS]/spot.json
  public/[ALIAS]/klines_5m.json
  public/[ALIAS]/klines_1h.json
  public/[ALIAS]/klines_4h.json
  public/[ALIAS]/signal.json
"""

import os
import time
import json
import math
import typing as t
from datetime import datetime, timezone
from pathlib import Path

import requests


# ======= KONFIG =======

# GitHub Pages gyökér
PUBLIC_DIR = Path("public")

# TwelveData API kulcs (Actions → Secrets → TWELVEDATA_API_KEY)
TD_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "").strip()

# TwelveData alap URL-ek
TD_QUOTE_URL = "https://api.twelvedata.com/quote"
TD_SERIES_URL = "https://api.twelvedata.com/time_series"

# Asset definíciók: alias → { id, folder, td_symbol }
ASSETS: dict[str, dict[str, str]] = {
    "SOL": {
        "id": "SOL",         # JSON 'asset'
        "folder": "SOL",     # public mappa
        "td_symbol": "SOL/USD",
    },
    "QQQ": {
        "id": "QQQ",
        "folder": "QQQ",
        "td_symbol": "QQQ",
    },
    "GOLD_CFD": {
        "id": "GOLD_CFD",
        "folder": "GOLD_CFD",
        "td_symbol": "XAU/USD",
    },
}

# Aliások (ha később NSDQ100-at, NASDAQ100-at, stb. kap a worker)
ASSET_ALIASES = {
    "NSDQ100": "QQQ",
    "NASDAQ100": "QQQ",
    "GOLD": "GOLD_CFD",
    "XAUUSD": "GOLD_CFD",
}

# Milyen timeframe-eket kérjünk a time_series-ből:
TIMEFRAMES = [
    ("k5m", "5min"),
    ("k1h", "1h"),
    ("k4h", "4h"),
]

# time_series mennyit kérjen vissza (ingyenes csomagnál tartsuk ésszerűn)
OUTPUTSIZE = 180  # kb. 15h a 5min-nál; 180h az 1h-nál; 30 nap a 4h-nál


# ======= SEGÉDEK =======

def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def placeholder_error(asset_id: str, msg: str) -> dict:
    return {"asset": asset_id, "ok": False, "error": msg, "retrieved_at_utc": utc_now_str()}

def ok_wrap(payload: dict) -> dict:
    payload.setdefault("ok", True)
    payload.setdefault("retrieved_at_utc", utc_now_str())
    return payload

def td_request(url: str, params: dict, max_retries: int = 3, backoff: float = 1.2) -> t.Tuple[bool, t.Any, str]:
    """
    Egyszerű retry/backoff wrapper TwelveData hívásokhoz.
    Vissza: (ok, data, err_msg)
    """
    if not TD_API_KEY:
        return False, None, "Missing TWELVEDATA_API_KEY"

    p = dict(params or {})
    p["apikey"] = TD_API_KEY

    err = ""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=p, timeout=15)
            if r.status_code == 200:
                j = r.json()
                # TwelveData hibák JSON-ban:
                if isinstance(j, dict) and j.get("status") == "error":
                    err = f"TwelveData error: {j.get('message')}"
                else:
                    return True, j, ""
            else:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            err = f"Exception: {e}"

        if attempt < max_retries:
            time.sleep(backoff ** attempt)

    return False, None, err


def fetch_spot(td_symbol: str) -> t.Tuple[bool, dict, str]:
    ok, data, err = td_request(TD_QUOTE_URL, {"symbol": td_symbol})
    if not ok:
        return False, {}, err

    # Várható mezők: price, symbol, timestamp stb.
    # Biztonság kedvéért csak a fontosabbakat emeljük ki.
    try:
        price = float(data.get("price"))
        symbol = data.get("symbol", td_symbol)
        out = {
            "symbol": symbol,
            "price": price,
            "raw": data,
        }
        return True, out, ""
    except Exception as e:
        return False, {}, f"Parse spot failed: {e}"


def fetch_series(td_symbol: str, interval: str, outputsize: int = OUTPUTSIZE) -> t.Tuple[bool, dict, str]:
    """
    TwelveData time_series → {meta, values[]}
    JSON mentéshez a teljes payloadot visszük (érintetlenül),
    de ellenőrizzük, hogy van-e "values".
    """
    ok, data, err = td_request(TD_SERIES_URL, {
        "symbol": td_symbol,
        "interval": interval,
        "outputsize": outputsize,
        "order": "ASC",   # időben növekvő sorrend kényelmesebb a jelfeldolgozáshoz
    })
    if not ok:
        return False, {}, err

    # Megpróbálunk normalizálni egy kicsit
    if not isinstance(data, dict) or "values" not in data:
        return False, {}, "time_series response has no 'values'"

    return True, data, ""


def simple_signal_from_series(series_payload: dict) -> t.Tuple[str, list[str]]:
    """
    Nagyon egyszerű, konzervatív "trend" jel:
      - ha utolsó close > előző close → 'long attempt'
      - ha utolsó close < előző close → 'short attempt'
      - különben 'no entry'
    Ha nincs elég adat, 'no signal'.
    """
    reasons: list[str] = []
    try:
        values = series_payload.get("values", [])
        if not values or len(values) < 2:
            return "no signal", ["not enough candles"]

        # utolsó két gyertya:
        # TwelveData: values ASC vagy DESC – mi ASC-et kértünk
        last = values[-1]
        prev = values[-2]
        c_last = float(last.get("close"))
        c_prev = float(prev.get("close"))

        if math.isfinite(c_last) and math.isfinite(c_prev):
            if c_last > c_prev:
                reasons.append("close up vs previous")
                return "long attempt", reasons
            elif c_last < c_prev:
                reasons.append("close down vs previous")
                return "short attempt", reasons
            else:
                return "no entry", ["equal closes"]
        else:
            return "no signal", ["non-finite close"]
    except Exception as e:
        return "no signal", [f"calc error: {e}"]


# ======= FŐ FOLYAM =======

def resolve_asset(key: str) -> dict:
    k = key.upper()
    if k in ASSETS:
        return ASSETS[k]
    if k in ASSET_ALIASES:
        return ASSETS[ASSET_ALIASES[k]]
    raise ValueError(f"Ismeretlen asset: {key}")


def main() -> int:
    order = ["SOL", "QQQ", "GOLD_CFD"]

    # Biztosítsuk a public gyökeret
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    for key in order:
        conf = resolve_asset(key)
        asset_id = conf["id"]
        folder = conf["folder"]
        td_symbol = conf["td_symbol"]

        out_dir = PUBLIC_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Spot
        ok, spot, err = fetch_spot(td_symbol)
        if not ok:
            write_json(out_dir / "spot.json", placeholder_error(asset_id, f"spot: {err}"))
        else:
            write_json(out_dir / "spot.json", ok_wrap(spot))

        # 2) Series (5m / 1h / 4h)
        last_good_series: dict[str, dict] = {}   # utolsó sikeres time_series payload, a jelhez jól jön
        for name, interval in TIMEFRAMES:
            ok, series, err = fetch_series(td_symbol, interval)
            if not ok:
                write_json(out_dir / f"klines_{name.split('k')[-1]}.json", placeholder_error(asset_id, f"{interval}: {err}"))
            else:
                write_json(out_dir / f"klines_{name.split('k')[-1]}.json", ok_wrap(series))
                last_good_series[name] = series

            # A TwelveData free limitjei miatt legyen köztük egy kis pihenő
            time.sleep(1.0)

        # 3) Egyszerű signal – ha bármelyik timeframe megvan, abból készítünk
        signal = "no entry"
        reasons: list[str] = ["no signal"]
        for name in ("k5m", "k1h", "k4h"):
            if name in last_good_series:
                s, r = simple_signal_from_series(last_good_series[name])
                signal = s
                reasons = r
                break

        signal_payload = ok_wrap({
            "asset": asset_id,
            "signal": signal,
            "reasons": reasons,
        })
        write_json(out_dir / "signal.json", signal_payload)

        print(f"[OK] {asset_id}: fájlok frissítve a {out_dir} mappában.")

    print("[KÉSZ] Minden eszköz feldolgozva.")
    return 0


if __name__ == "__main__":
    exit(main())

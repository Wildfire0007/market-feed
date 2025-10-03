#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trading.py — market-feed generator
- SOL: Coinbase (public REST)
- NSDQ100 (QQQ) + GOLD_CFD (XAU/USD): TwelveData (API key szükséges)
Kimenetek eszközönként: public/<ASSET>/
  - spot.json (price_usd, source, ok, retrieved_at_utc)
  - klines_5m.json, klines_1h.json, klines_4h.json
  - signal.json (egyszerű EMA9/21 trend jelzés)
Extra:
  - public/all_<ASSET>.json (aggregált)
  - public/analysis_summary.json + public/analysis.html
Env: ASSET_ONLY = "SOL,GOLD_CFD" stb. a szűréshez.
"""

import os, json, time
from datetime import datetime, timezone
from typing import Dict, Any, List

import requests
import pandas as pd
import numpy as np

OUT_DIR = "public"

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_BASE = "https://api.twelvedata.com"
TD_PAUSE = float(os.getenv("TD_PAUSE", "1.5"))

CB_BASE = "https://api.exchange.coinbase.com"

ASSETS = [
    {"name": "SOL",      "source": "coinbase",   "cb_pair": "SOL-USD"},
    {"name": "NSDQ100",  "source": "twelvedata", "td_symbol": "QQQ"},
    {"name": "GOLD_CFD", "source": "twelvedata", "td_symbol": "XAU/USD"},
]

# --- env alapú szűrés ---
_only = [x.strip().upper() for x in os.getenv("ASSET_ONLY", "").split(",") if x.strip()]
if _only:
    ASSETS = [a for a in ASSETS if a["name"].upper() in _only]

# ----------------- segédek -----------------

def nowiso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:  # csak akkor hozunk létre könyvtárat, ha van megadott mappa
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def with_status_ok(payload: Dict[str, Any], ok: bool, error: str | None = None) -> Dict[str, Any]:
    payload = dict(payload) if payload else {}
    payload.setdefault("retrieved_at_utc", nowiso())
    payload["ok"] = bool(ok)
    if not ok and error:
        payload["error"] = error
    return payload

# ----------------- Coinbase (SOL) -----------------

def cb_get(path: str, params: dict | None = None, timeout: int = 15):
    url = f"{CB_BASE}{path}"
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "market-feed/1.0"})
    r.raise_for_status()
    return r.json()

def coinbase_spot(pair: str) -> dict:
    try:
        j = cb_get(f"/products/{pair}/ticker")
        price = None
        if isinstance(j, dict):
            if j.get("price") not in (None, "None"):
                price = float(j["price"])
            elif j.get("bid") and j.get("ask"):
                price = (float(j["bid"]) + float(j["ask"])) / 2.0
        return with_status_ok(
            {"asset": "SOL", "price_usd": price, "source": "coinbase", "raw": j},
            price is not None
        )
    except Exception as e:
        return with_status_ok({"asset": "SOL", "source": "coinbase"}, False, f"spot error: {e}")

# candles: [ time, low, high, open, close, volume ] (max ~300), granu: 60/300/900/3600/21600/86400
def coinbase_candles(pair: str, granularity_sec: int = 300, start_iso: str | None = None, end_iso: str | None = None) -> list:
    params = {"granularity": granularity_sec}
    if start_iso: params["start"] = start_iso
    if end_iso:   params["end"] = end_iso
    rows = cb_get(f"/products/{pair}/candles", params=params)
    rows = sorted(rows, key=lambda x: x[0])
    return rows

def cb_rows_to_df(rows: list) -> pd.DataFrame:
    idx = pd.to_datetime([int(r[0]) for r in rows], unit="s", utc=True).tz_convert(None)
    return pd.DataFrame({
        "open":  [float(r[3]) for r in rows],
        "high":  [float(r[2]) for r in rows],
        "low":   [float(r[1]) for r in rows],
        "close": [float(r[4]) for r in rows],
    }, index=idx)

def df_to_values(df: pd.DataFrame) -> list[dict]:
    out = []
    for ts, row in df.iterrows():
        out.append({
            "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open":  f"{row['open']:.2f}",
            "high":  f"{row['high']:.2f}",
            "low":   f"{row['low']:.2f}",
            "close": f"{row['close']:.2f}",
        })
    return out

# ----------------- TwelveData (QQQ, GOLD_CFD) -----------------

def td_get(endpoint: str, params: dict, timeout: int = 20):
    if not TD_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")
    url = f"{TD_BASE}/{endpoint}"
    p = dict(params); p["apikey"] = TD_API_KEY
    r = requests.get(url, params=p, timeout=timeout, headers={"User-Agent": "market-feed/1.0"})
    r.raise_for_status()
    time.sleep(TD_PAUSE)  # kvóta kímélés
    return r.json()

def td_spot(symbol: str, asset_name: str) -> dict:
    try:
        j = td_get("price", {"symbol": symbol})
        price = None
        if isinstance(j, dict) and j.get("price") not in (None, "None"):
            try: price = float(j["price"])
            except Exception: price = None
        if price is None and isinstance(j, dict) and j.get("status") == "error":
            return with_status_ok({"asset": asset_name, "source": "twelvedata"}, False, f"spot: {j.get('message')}")
        return with_status_ok({"asset": asset_name, "price_usd": price, "source": "twelvedata", "raw": j}, price is not None)
    except Exception as e:
        return with_status_ok({"asset": asset_name, "source": "twelvedata"}, False, f"spot error: {e}")

def td_series(symbol: str, interval: str, asset_name: str) -> dict:
    try:
        j = td_get("time_series", {"symbol": symbol, "interval": interval, "outputsize": 300, "timezone": "UTC"})
        if isinstance(j, dict) and j.get("status") == "error":
            return {"ok": False, "status": "error", "error": j.get("message"), "meta": {"symbol": symbol, "interval": interval}, "values": [], "retrieved_at_utc": nowiso()}
        meta = j.get("meta", {}) if isinstance(j, dict) else {}
        vals = list(reversed(j.get("values", []) or []))
        return {"ok": True, "status": "ok",
                "meta": {"symbol": meta.get("symbol", symbol), "interval": interval,
                         "currency": meta.get("currency"), "exchange": meta.get("exchange"),
                         "exchange_timezone": meta.get("exchange_timezone"), "type": meta.get("type"),
                         "source": "twelvedata"},
                "values": vals, "retrieved_at_utc": nowiso()}
    except Exception as e:
        return {"ok": False, "status": "error", "error": f"TwelveData {interval} fetch failed: {e}", "meta": {"symbol": symbol, "interval": interval}, "values": [], "retrieved_at_utc": nowiso()}

# ----------------- jelzés (egyszerű EMA9/21) -----------------

def simple_signal_from_df(df: pd.DataFrame) -> tuple[str, list[str]]:
    if df is None or df.empty or len(df) < 30:
        return "no entry", ["no signal"]
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    last, prev = -1, -5
    if ema9.iloc[last] > ema21.iloc[last] and ema9.iloc[prev] > ema21.iloc[prev]:
        return "uptrend", ["ema9 > ema21 (5 bars)"]
    if ema9.iloc[last] < ema21.iloc[last] and ema9.iloc[prev] < ema21.iloc[prev]:
        return "downtrend", ["ema9 < ema21 (5 bars)"]
    return "no entry", ["no signal"]

def dump_signal(asset_dir: str, asset_name: str, df_for_signal: pd.DataFrame | None):
    sig, reasons = simple_signal_from_df(df_for_signal)
    save_json(os.path.join(asset_dir, "signal.json"),
              {"asset": asset_name, "ok": True, "retrieved_at_utc": nowiso(), "signal": sig, "reasons": reasons})

# ----------------- eszköz feldolgozók -----------------

def process_SOL_coinbase(pair: str):
    asset = "SOL"
    adir = os.path.join(OUT_DIR, asset); ensure_dir(adir)

    spot = coinbase_spot(pair); save_json(os.path.join(adir, "spot.json"), spot)

    # 5m
    try:
        k5_rows = coinbase_candles(pair, 300)
        df5 = cb_rows_to_df(k5_rows)
        save_json(os.path.join(adir, "klines_5m.json"),
                  {"meta": {"symbol": pair.replace("-", "/"), "interval": "5min", "source": "coinbase"},
                   "values": df_to_values(df5), "status": "ok", "ok": True, "retrieved_at_utc": nowiso()})
    except Exception as e:
        save_json(os.path.join(adir, "klines_5m.json"), with_status_ok({"meta": {"symbol": pair, "interval": "5min"}, "values": []}, False, str(e)))
        df5 = None

    # 1h
    try:
        k1_rows = coinbase_candles(pair, 3600)
        df1 = cb_rows_to_df(k1_rows)
        save_json(os.path.join(adir, "klines_1h.json"),
                  {"meta": {"symbol": pair.replace("-", "/"), "interval": "1h", "source": "coinbase"},
                   "values": df_to_values(df1), "status": "ok", "ok": True, "retrieved_at_utc": nowiso()})
    except Exception as e:
        save_json(os.path.join(adir, "klines_1h.json"), with_status_ok({"meta": {"symbol": pair, "interval": "1h"}, "values": []}, False, str(e)))
        df1 = None

    # 4h (1h->4h)
    try:
        if df1 is None:
            k1_rows = coinbase_candles(pair, 3600); df1 = cb_rows_to_df(k1_rows)
        k4 = df1.resample("4H").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        save_json(os.path.join(adir, "klines_4h.json"),
                  {"meta": {"symbol": pair.replace("-", "/"), "interval": "4h", "source": "coinbase (1h->4h)"},
                   "values": df_to_values(k4), "status": "ok", "ok": True, "retrieved_at_utc": nowiso()})
    except Exception as e:
        save_json(os.path.join(adir, "klines_4h.json"), with_status_ok({"meta": {"symbol": pair, "interval": "4h"}, "values": []}, False, str(e)))

    dump_signal(adir, asset, df1 if isinstance(df1, pd.DataFrame) else None)

def process_TD_generic(asset_name: str, symbol: str):
    adir = os.path.join(OUT_DIR, asset_name); ensure_dir(adir)

    spot = td_spot(symbol, asset_name); save_json(os.path.join(adir, "spot.json"), spot)

    for interval, fname in [("5min", "klines_5m.json"), ("1h", "klines_1h.json"), ("4h", "klines_4h.json")]:
        save_json(os.path.join(adir, fname), td_series(symbol, interval, asset_name))

    # jelzés 1h alapján
    vals = read_json(os.path.join(adir, "klines_1h.json")).get("values", [])
    df = None
    if vals:
        idx = pd.to_datetime([v["datetime"] for v in vals])
        df = pd.DataFrame({
            "open":  [float(v["open"]) for v in vals],
            "high":  [float(v["high"]) for v in vals],
            "low":   [float(v["low"]) for v in vals],
            "close": [float(v["close"]) for v in vals],
        }, index=idx)
    dump_signal(adir, asset_name, df)

# ----------------- aggregátorok -----------------

def write_all_asset(asset: str):
    adir = os.path.join(OUT_DIR, asset)
    obj = {
        "asset": asset,
        "spot":   read_json(os.path.join(adir, "spot.json")),
        "k5m":    read_json(os.path.join(adir, "klines_5m.json")),
        "k1h":    read_json(os.path.join(adir, "klines_1h.json")),
        "k4h":    read_json(os.path.join(adir, "klines_4h.json")),
        "signal": read_json(os.path.join(adir, "signal.json")),
        "retrieved_at_utc": nowiso(),
        "ok": True,
    }
    save_json(os.path.join(OUT_DIR, f"all_{asset}.json"), obj)
    save_json(os.path.join(f"all_{asset}.json"), obj)  # kompatibilitás

def write_summary(assets: List[dict]):
    summ = {"ok": True, "retrieved_at_utc": nowiso(), "assets": {}}
    for a in assets:
        name = a["name"]
        spot = read_json(os.path.join(OUT_DIR, name, "spot.json"))
        sig  = read_json(os.path.join(OUT_DIR, name, "signal.json"))
        summ["assets"][name] = {"spot": spot, "signal": sig}
    save_json(os.path.join(OUT_DIR, "analysis_summary.json"), summ)
    html = "<!doctype html><meta charset='utf-8'><title>Analysis Summary</title><h1>Analysis Summary</h1><pre>" \
           + json.dumps(summ, ensure_ascii=False, indent=2) + "</pre>"
    with open(os.path.join(OUT_DIR, "analysis.html"), "w", encoding="utf-8") as f:
        f.write(html)

# ----------------- main -----------------

def main():
    ensure_dir(OUT_DIR)
    for a in ASSETS:
        if a["source"] == "coinbase":
            process_SOL_coinbase(a["cb_pair"])
        elif a["source"] == "twelvedata":
            process_TD_generic(a["name"], a["td_symbol"])
    for a in ASSETS:
        write_all_asset(a["name"])
    write_summary(ASSETS)

if __name__ == "__main__":
    main()


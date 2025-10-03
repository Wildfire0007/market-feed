#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trading.py
- SOL: Coinbase (ingyenes), QQQ + GOLD_CFD: TwelveData
- Kimenetek: public/<ASSET>/{spot.json,klines_5m.json,klines_1h.json,klines_4h.json,signal.json}
"""

import os, json, time, math
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np

# -----------------------------
# Beállítások
# -----------------------------
OUT_DIR = "public"

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_BASE = "https://api.twelvedata.com"
TD_PAUSE = float(os.getenv("TD_PAUSE", "1.5"))  # mp szünet TD hívások között (kvóta-kímélés)

CB_BASE = "https://api.exchange.coinbase.com"

ASSETS = [
    # SOL: teljesen Coinbase
    {"name": "SOL", "source": "coinbase", "cb_pair": "SOL-USD"},
    # QQQ (NSDQ100): TwelveData – ingest ETF (QQQ)
    {"name": "NSDQ100", "source": "twelvedata", "td_symbol": "QQQ"},
    # Arany (CFD/spot): TwelveData – XAU/USD (forex)
    {"name": "GOLD_CFD", "source": "twelvedata", "td_symbol": "XAU/USD"},
]

# -----------------------------
# Segédek/IO
# -----------------------------

def nowiso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_json(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def with_status_ok(payload: dict, ok: bool, error: str | None = None) -> dict:
    payload = dict(payload) if payload else {}
    payload.setdefault("retrieved_at_utc", nowiso())
    payload["ok"] = bool(ok)
    if not ok and error:
        payload["error"] = error
    return payload

# -----------------------------
# Coinbase (SOL)
# -----------------------------

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
            if j.get("price") is not None:
                price = float(j["price"])
            elif j.get("bid") and j.get("ask"):
                price = (float(j["bid"]) + float(j["ask"])) / 2.0
        return with_status_ok({"asset": "SOL", "price": price, "raw": j}, price is not None)
    except Exception as e:
        return with_status_ok({"asset": "SOL"}, False, f"coinbase spot error: {e}")

# Coinbase candle: [ time, low, high, open, close, volume ] – UNIX sec
def coinbase_candles(pair: str, granularity_sec: int = 300,
                     start_iso: str | None = None, end_iso: str | None = None) -> list:
    params = {"granularity": granularity_sec}
    if start_iso:
        params["start"] = start_iso
    if end_iso:
        params["end"] = end_iso
    rows = cb_get(f"/products/{pair}/candles", params=params)
    rows = sorted(rows, key=lambda x: x[0])  # idő szerint növekvő
    return rows

def cb_rows_to_values(rows: list) -> list[dict]:
    """ Coinbase sorokból (time, low, high, open, close, vol) standard values tömb """
    out = []
    for row in rows:
        t, low, high, opn, cls, vol = row
        dt = datetime.utcfromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M:%S")
        out.append({
            "datetime": dt,
            "open": f"{float(opn):.2f}",
            "high": f"{float(high):.2f}",
            "low": f"{float(low):.2f}",
            "close": f"{float(cls):.2f}",
        })
    return out

def cb_rows_to_df(rows: list) -> pd.DataFrame:
    # time sec -> pandas index (UTC-naive)
    idx = pd.to_datetime([int(r[0]) for r in rows], unit="s", utc=True).tz_convert(None)
    df = pd.DataFrame({
        "open": [float(r[3]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low":  [float(r[1]) for r in rows],
        "close":[float(r[4]) for r in rows],
    }, index=idx)
    return df

def df_to_values(df: pd.DataFrame) -> list[dict]:
    out = []
    for ts, row in df.iterrows():
        dt = ts.strftime("%Y-%m-%d %H:%M:%S")
        out.append({
            "datetime": dt,
            "open": f"{row['open']:.2f}",
            "high": f"{row['high']:.2f}",
            "low": f"{row['low']:.2f}",
            "close": f"{row['close']:.2f}",
        })
    return out

# -----------------------------
# TwelveData (QQQ, GOLD_CFD)
# -----------------------------

def td_get(endpoint: str, params: dict, timeout: int = 15):
    if not TD_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")
    url = f"{TD_BASE}/{endpoint}"
    p = dict(params)
    p["apikey"] = TD_API_KEY
    r = requests.get(url, params=p, timeout=timeout)
    r.raise_for_status()
    # kvótakímélő a következő hívás előtt
    time.sleep(TD_PAUSE)
    return r.json()

def td_spot(symbol: str, asset_name: str) -> dict:
    try:
        j = td_get("price", {"symbol": symbol})
        # siker esetén pl. {"price":"123.45","symbol":"QQQ"}
        price = None
        if isinstance(j, dict) and j.get("price") not in (None, "None"):
            try:
                price = float(j["price"])
            except Exception:
                price = None
        ok = price is not None
        # TwelveData hibaüzenetek standard kulccsal jöhetnek
        if not ok and isinstance(j, dict) and j.get("status") == "error" and j.get("message"):
            return with_status_ok({"asset": asset_name}, False, f"spot: {j['message']}")
        return with_status_ok({"asset": asset_name, "price": price, "raw": j}, ok)
    except Exception as e:
        return with_status_ok({"asset": asset_name}, False, f"spot: TwelveData error: {e}")

def td_series(symbol: str, interval: str, asset_name: str) -> dict:
    """
    interval: '5min' | '1h' | '4h'
    """
    try:
        j = td_get("time_series", {
            "symbol": symbol,
            "interval": interval,
            "outputsize": 300,
            "timezone": "UTC",
            # adjclose nem kell
        })
        # hiba?
        if isinstance(j, dict) and j.get("status") == "error":
            return {
                "ok": False,
                "status": "error",
                "error": f"Error: {j.get('message')}",
                "meta": {"symbol": symbol, "interval": interval},
                "values": [],
                "retrieved_at_utc": nowiso(),
            }
        # siker (lista a "values" kulcs alatt)
        meta = j.get("meta", {})
        vals = j.get("values", []) or []
        # reverse legfrissebb -> legrégebbi helyett
        vals = list(reversed(vals))
        out = {
            "ok": True,
            "status": "ok",
            "meta": {
                "symbol": meta.get("symbol", symbol),
                "interval": interval,
                "currency": meta.get("currency"),
                "exchange": meta.get("exchange"),
                "exchange_timezone": meta.get("exchange_timezone"),
                "type": meta.get("type"),
            },
            "values": vals,
            "retrieved_at_utc": nowiso(),
        }
        return out
    except Exception as e:
        return {
            "ok": False,
            "status": "error",
            "error": f"Error: TwelveData {interval} fetch failed: {e}",
            "meta": {"symbol": symbol, "interval": interval},
            "values": [],
            "retrieved_at_utc": nowiso(),
        }

# -----------------------------
# Alap „jelzés” (placeholder)
# -----------------------------

def simple_signal_from_df(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Pofonegyszerű placeholder: ha 5 gyertya EMA(9) > EMA(21) trendben, akkor 'up', ha fordítva 'down', különben 'no entry'.
    """
    if df is None or df.empty or len(df) < 30:
        return "no entry", ["no signal"]

    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()

    last = -1
    prev = -5
    if ema9.iloc[last] > ema21.iloc[last] and ema9.iloc[prev] > ema21.iloc[prev]:
        return "uptrend", ["ema9 > ema21 (5 bars)"]
    if ema9.iloc[last] < ema21.iloc[last] and ema9.iloc[prev] < ema21.iloc[prev]:
        return "downtrend", ["ema9 < ema21 (5 bars)"]
    return "no entry", ["no signal"]

def dump_signal(asset_dir: str, asset_name: str, df_for_signal: pd.DataFrame | None):
    sig, reasons = simple_signal_from_df(df_for_signal)
    save_json(os.path.join(asset_dir, "signal.json"), {
        "asset": asset_name,
        "ok": True,
        "retrieved_at_utc": nowiso(),
        "signal": sig,
        "reasons": reasons
    })

# -----------------------------
# Eszköz feldolgozók
# -----------------------------

def process_SOL_coinbase(pair: str):
    asset = "SOL"
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    # Spot
    spot = coinbase_spot(pair)
    save_json(os.path.join(adir, "spot.json"), spot)

    # 5m
    try:
        k5 = coinbase_candles(pair, 300)
        vals5 = cb_rows_to_values(k5)
        out5 = {
            "meta": {"symbol": pair.replace("-", "/"), "interval": "5min", "source": "coinbase"},
            "values": vals5,
            "status": "ok",
            "ok": True,
            "retrieved_at_utc": nowiso(),
        }
        save_json(os.path.join(adir, "klines_5m.json"), out5)
    except Exception as e:
        save_json(os.path.join(adir, "klines_5m.json"), with_status_ok(
            {"meta": {"symbol": pair, "interval": "5min"}, "values": []}, False, str(e)
        ))
        k5 = []

    # 1h
    try:
        k1 = coinbase_candles(pair, 3600)
        vals1 = cb_rows_to_values(k1)
        out1 = {
            "meta": {"symbol": pair.replace("-", "/"), "interval": "1h", "source": "coinbase"},
            "values": vals1,
            "status": "ok",
            "ok": True,
            "retrieved_at_utc": nowiso(),
        }
        save_json(os.path.join(adir, "klines_1h.json"), out1)
    except Exception as e:
        save_json(os.path.join(adir, "klines_1h.json"), with_status_ok(
            {"meta": {"symbol": pair, "interval": "1h"}, "values": []}, False, str(e)
        ))
        k1 = []

    # 4h – 1h DF-ből összevonva
    try:
        if not k1:
            k1 = coinbase_candles(pair, 3600)
        df1 = cb_rows_to_df(k1)
        k4 = df1.resample("4H").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna()
        vals4 = df_to_values(k4)
        out4 = {
            "meta": {"symbol": pair.replace("-", "/"), "interval": "4h", "source": "coinbase (1h->4h)"},
            "values": vals4,
            "status": "ok",
            "ok": True,
            "retrieved_at_utc": nowiso(),
        }
        save_json(os.path.join(adir, "klines_4h.json"), out4)
    except Exception as e:
        save_json(os.path.join(adir, "klines_4h.json"), with_status_ok(
            {"meta": {"symbol": pair, "interval": "4h"}, "values": []}, False, str(e)
        ))
        k4 = None

    # jelzéshez 1h DF (egyszerű)
    try:
        df_for_sig = cb_rows_to_df(k1) if k1 else None
    except Exception:
        df_for_sig = None
    dump_signal(adir, asset, df_for_sig)

def process_TD_generic(asset_name: str, symbol: str):
    adir = os.path.join(OUT_DIR, asset_name)
    ensure_dir(adir)

    # spot
    spot = td_spot(symbol, asset_name)
    save_json(os.path.join(adir, "spot.json"), spot)

    # 5m / 1h / 4h
    for interval, fname in [("5min", "klines_5m.json"), ("1h", "klines_1h.json"), ("4h", "klines_4h.json")]:
        data = td_series(symbol, interval, asset_name)
        save_json(os.path.join(adir, fname), data)

    # jelzés (1h alapján)
    try:
        d1 = td_series(symbol, "1h", asset_name)
        vals = d1.get("values", [])
        if not vals:
            df = None
        else:
            # TwelveData values: [{"datetime": "...", "open":"", "high":"", "low":"", "close":""}, ...]
            idx = pd.to_datetime([v["datetime"] for v in vals])
            df = pd.DataFrame({
                "open": [float(v["open"]) for v in vals],
                "high": [float(v["high"]) for v in vals],
                "low": [float(v["low"]) for v in vals],
                "close": [float(v["close"]) for v in vals],
            }, index=idx)
    except Exception:
        df = None

    dump_signal(adir, asset_name, df)

# -----------------------------
# main
# -----------------------------

def main():
    ensure_dir(OUT_DIR)
    for a in ASSETS:
        if a["source"] == "coinbase":
            process_SOL_coinbase(a["cb_pair"])
        elif a["source"] == "twelvedata":
            process_TD_generic(a["name"], a["td_symbol"])
        else:
            print(f"[WARN] Unknown source for {a}")

if __name__ == "__main__":
    main()

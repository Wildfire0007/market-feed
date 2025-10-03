#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds JSON feeds under ./public for: SOL, NSDQ100 (QQQ), GOLD_CFD
- SOL: Coinbase public API
- GOLD_CFD (XAU/USD): Yahoo Finance (yfinance)
- NSDQ100 (QQQ): TwelveData (needs TWELVEDATA_API_KEY env)
Includes TTL-based throttling to spare API quotas.

Output structure per asset (unchanged):
  public/<ASSET>/
    spot.json
    klines_5m.json
    klines_1h.json
    klines_4h.json
    signal.json
"""

import os, json, time, math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# yfinance for GOLD (XAUUSD)
import yfinance as yf

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()

ROOT = Path("public")
ROOT.mkdir(parents=True, exist_ok=True)

# --- TTLs (cache windows) ---
TTL = {
    "spot": 5 * 60,       # 5 perc
    "k5m": 15 * 60,       # 15 perc
    "k1h": 60 * 60,       # 60 perc
    "k4h": 4 * 60 * 60,   # 4 óra
    "signal": 5 * 60,     # jelzés, ha input frissült, úgyis újraírjuk
}

def nowiso() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def fresh_enough(path: Path, ttl_sec: int) -> bool:
    if not path.exists():
        return False
    try:
        mtime = path.stat().st_mtime
        return (time.time() - mtime) < ttl_sec
    except Exception:
        return False

def ensure(path: Path, ttl_sec: int, producer_fn):
    """If path is fresh, return False. Otherwise call producer_fn() and write -> True."""
    if fresh_enough(path, ttl_sec):
        return False
    try:
        data = producer_fn()
    except Exception as e:
        data = {"ok": False, "error": f"{type(e).__name__}: {e}", "retrieved_at_utc": nowiso()}
    write_json(path, data)
    return True

# ---- helpers to shape OHLC output ----
def df_to_values(df: pd.DataFrame) -> list:
    # Expect df index as datetime (UTC) and columns: open/high/low/close
    out = []
    if df is None or df.empty:
        return out
    for ts, row in df.iterrows():
        out.append({
            "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open": f"{row['open']:.2f}",
            "high": f"{row['high']:.2f}",
            "low":  f"{row['low']:.2f}",
            "close":f"{row['close']:.2f}",
        })
    return out

# =========================
# SOL (Coinbase)
# =========================
def sol_spot_coinbase():
    url = "https://api.exchange.coinbase.com/products/SOL-USD/ticker"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return {"asset": "SOL", "ok": False, "error": f"HTTP {r.status_code}", "retrieved_at_utc": nowiso()}
    j = r.json()
    price = j.get("price")
    if price is None:
        return {"asset": "SOL", "ok": False, "error": "no price", "retrieved_at_utc": nowiso()}
    return {
        "asset": "SOL",
        "ok": True,
        "price": float(price),
        "raw": j,
        "symbol": "SOL/USD",
        "retrieved_at_utc": nowiso(),
    }

def sol_klines_coinbase(granularity: int, label: str):
    """
    Coinbase candles:
      GET /products/SOL-USD/candles?granularity=300|3600|14400
      returns [[time, low, high, open, close, volume], ...] (time = Unix)
    """
    url = "https://api.exchange.coinbase.com/products/SOL-USD/candles"
    r = requests.get(url, params={"granularity": granularity}, timeout=20)
    if r.status_code != 200:
        return {"ok": False, "status": "error", "meta": {"symbol": "SOL/USD", "interval": label}, "error": f"HTTP {r.status_code}", "retrieved_at_utc": nowiso()}
    arr = r.json()
    if not isinstance(arr, list):
        return {"ok": False, "status": "error", "meta": {"symbol": "SOL/USD", "interval": label}, "error": "bad response", "retrieved_at_utc": nowiso()}

    # Coinbase returns latest first; sort ascending by time
    arr.sort(key=lambda x: x[0])

    idx = pd.to_datetime([x[0] for x in arr], unit="s", utc=True)
    df = pd.DataFrame({
        "open":  [float(x[3]) for x in arr],
        "high":  [float(x[2]) for x in arr],
        "low":   [float(x[1]) for x in arr],
        "close": [float(x[4]) for x in arr],
    }, index=idx)

    return {
        "meta": {
            "symbol": "SOL/USD",
            "interval": label,
            "currency_base": "Solana",
            "currency_quote": "US Dollar",
            "exchange": "Coinbase",
            "type": "Digital Currency",
        },
        "values": df_to_values(df),
        "status": "ok",
        "ok": True,
        "retrieved_at_utc": nowiso(),
    }

# =========================
# GOLD (XAU/USD via yfinance)
# =========================
XAU_TICKER = "XAUUSD=X"  # Yahoo Finance symbol for spot XAU/USD

def gold_spot_yf():
    try:
        t = yf.Ticker(XAU_TICKER)
        # fast path: use info['regularMarketPrice'] if available; else last close from 1m
        price = None
        info = t.fast_info if hasattr(t, "fast_info") else None
        if info and getattr(info, "last_price", None) is not None:
            price = float(info.last_price)
        if price is None:
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if price is None:
            return {"asset": "GOLD_CFD", "ok": False, "error": "no price", "retrieved_at_utc": nowiso()}
        return {
            "asset": "GOLD_CFD",
            "ok": True,
            "price": price,
            "raw": {"source": "yfinance", "ticker": XAU_TICKER},
            "symbol": "XAU/USD",
            "retrieved_at_utc": nowiso(),
        }
    except Exception as e:
        return {"asset": "GOLD_CFD", "ok": False, "error": f"{type(e).__name__}: {e}", "retrieved_at_utc": nowiso()}

def gold_klines_yf(interval: str, label: str):
    """
    interval: '5m' | '60m' | '240m' (yfinance does not have 4h, ezért 15m-ből aggregálunk)
    """
    try:
        if interval == "240m":
            # build 4h from 15m
            raw = yf.download(XAU_TICKER, period="60d", interval="15m", progress=False)
            if raw.empty:
                raise RuntimeError("empty history")
            # resample to 4H UTC
            df = raw.tz_convert("UTC").resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
        else:
            period = "30d" if interval == "5m" else "365d"
            raw = yf.download(XAU_TICKER, period=period, interval=interval, progress=False)
            if raw.empty:
                raise RuntimeError("empty history")
            df = raw.tz_convert("UTC")[["Open","High","Low","Close"]]
            df.columns = ["open","high","low","close"]
        if "open" not in df.columns:
            df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
        return {
            "meta": {
                "symbol": "XAU/USD",
                "interval": label,
                "currency_base": "Gold",
                "currency_quote": "US Dollar",
                "exchange": "YahooFinance",
                "type": "Commodity",
            },
            "values": df_to_values(df),
            "status": "ok",
            "ok": True,
            "retrieved_at_utc": nowiso(),
        }
    except Exception as e:
        return {"ok": False, "status": "error", "meta": {"symbol":"XAU/USD","interval":label}, "error": f"{type(e).__name__}: {e}", "retrieved_at_utc": nowiso()}

# =========================
# QQQ (TwelveData) – throttled
# =========================
def qqq_spot_td():
    if not TD_API_KEY:
        return {"asset":"NSDQ100","ok":False,"error":"no TWELVEDATA_API_KEY","retrieved_at_utc":nowiso()}
    url = "https://api.twelvedata.com/price"
    r = requests.get(url, params={"symbol":"QQQ","apikey":TD_API_KEY}, timeout=15)
    j = r.json() if r.ok else {}
    if "price" not in j:
        return {"asset":"NSDQ100","ok":False,"error":f"spot: TwelveData error: {j}", "retrieved_at_utc": nowiso()}
    return {"asset":"NSDQ100","ok":True,"price":float(j["price"]),"raw":j,"symbol":"QQQ","retrieved_at_utc":nowiso()}

def qqq_klines_td(interval: str, label: str):
    if not TD_API_KEY:
        return {"ok": False, "status":"error", "meta":{"symbol":"QQQ","interval":label}, "error":"no TWELVEDATA_API_KEY", "retrieved_at_utc":nowiso()}
    url = "https://api.twelvedata.com/time_series"
    r = requests.get(url, params={
        "symbol": "QQQ",
        "interval": interval,   # '5min' | '1h' | '4h'
        "outputsize": 500,
        "apikey": TD_API_KEY,
        "timezone": "UTC",
        "order": "ASC"
    }, timeout=20)
    j = r.json() if r.ok else {}
    if "values" not in j:
        return {"ok": False, "status":"error", "meta":{"symbol":"QQQ","interval":label}, "error": f"TwelveData error: {j}", "retrieved_at_utc":nowiso()}
    # normalize
    vals = j["values"]
    idx = pd.to_datetime([v["datetime"] for v in vals], utc=True)
    df = pd.DataFrame({
        "open":  [float(v["open"]) for v in vals],
        "high":  [float(v["high"]) for v in vals],
        "low":   [float(v["low"]) for v in vals],
        "close": [float(v["close"]) for v in vals],
    }, index=idx)
    return {
        "meta": {
            "symbol": "QQQ",
            "interval": label,
            "currency": "USD",
            "exchange": "NASDAQ",
            "type": "ETF",
        },
        "values": df_to_values(df),
        "status": "ok",
        "ok": True,
        "retrieved_at_utc": nowiso(),
    }

# =========================
# SIGNAL (dummy – no entry unless you want logic)
# =========================
def make_signal(asset: str) -> dict:
    # Helytartó jelzés: "no entry" – ide később jöhet stratégiád
    return {
        "asset": asset,
        "ok": True,
        "retrieved_at_utc": nowiso(),
        "signal": "no entry",
        "reasons": ["no signal"],
    }

# =========================
# BUILD PIPELINES
# =========================
def build_SOL():
    base = ROOT / "SOL"
    spot_p = base / "spot.json"
    k5m_p  = base / "klines_5m.json"
    k1h_p  = base / "klines_1h.json"
    k4h_p  = base / "klines_4h.json"
    sig_p  = base / "signal.json"

    changed = False
    changed |= ensure(spot_p, TTL["spot"], lambda: sol_spot_coinbase())
    changed |= ensure(k5m_p, TTL["k5m"], lambda: sol_klines_coinbase(300, "5min"))
    changed |= ensure(k1h_p, TTL["k1h"], lambda: sol_klines_coinbase(3600, "1h"))
    changed |= ensure(k4h_p, TTL["k4h"], lambda: sol_klines_coinbase(14400, "4h"))
    if changed or not sig_p.exists() or not fresh_enough(sig_p, TTL["signal"]):
        write_json(sig_p, make_signal("SOL"))

def build_GOLD():
    base = ROOT / "GOLD_CFD"
    spot_p = base / "spot.json"
    k5m_p  = base / "klines_5m.json"
    k1h_p  = base / "klines_1h.json"
    k4h_p  = base / "klines_4h.json"
    sig_p  = base / "signal.json"

    changed = False
    changed |= ensure(spot_p, TTL["spot"], lambda: gold_spot_yf())
    changed |= ensure(k5m_p, TTL["k5m"], lambda: gold_klines_yf("5m", "5min"))
    changed |= ensure(k1h_p, TTL["k1h"], lambda: gold_klines_yf("60m", "1h"))
    changed |= ensure(k4h_p, TTL["k4h"], lambda: gold_klines_yf("240m", "4h"))
    if changed or not sig_p.exists() or not fresh_enough(sig_p, TTL["signal"]):
        write_json(sig_p, make_signal("GOLD_CFD"))

def build_QQQ():
    base = ROOT / "NSDQ100"
    spot_p = base / "spot.json"
    k5m_p  = base / "klines_5m.json"
    k1h_p  = base / "klines_1h.json"
    k4h_p  = base / "klines_4h.json"
    sig_p  = base / "signal.json"

    changed = False
    changed |= ensure(spot_p, TTL["spot"], lambda: qqq_spot_td())
    changed |= ensure(k5m_p, TTL["k5m"], lambda: qqq_klines_td("5min","5min"))
    changed |= ensure(k1h_p, TTL["k1h"], lambda: qqq_klines_td("1h","1h"))
    changed |= ensure(k4h_p, TTL["k4h"], lambda: qqq_klines_td("4h","4h"))
    if changed or not sig_p.exists() or not fresh_enough(sig_p, TTL["signal"]):
        write_json(sig_p, make_signal("NSDQ100"))

def main():
    build_SOL()
    build_GOLD()
    build_QQQ()
    # simple index for convenience
    idx = ROOT / "index.html"
    idx.write_text(
        "<!doctype html><meta charset='utf-8'><title>market-feed</title>"
        "<h1>market-feed</h1><ul>"
        "<li><a href='./SOL/signal.json'>SOL/signal.json</a></li>"
        "<li><a href='./NSDQ100/signal.json'>NSDQ100/signal.json</a></li>"
        "<li><a href='./GOLD_CFD/signal.json'>GOLD_CFD/signal.json</a></li>"
        "<li><a href='./analysis.html'>analysis.html</a></li>"
        "</ul>", encoding="utf-8"
    )

if __name__ == "__main__":
    main()

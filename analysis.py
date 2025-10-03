# -*- coding: utf-8 -*-
"""
analysis.py — Intraday jelzésképző script az Ekereskedő (eToro Ügynök) számára.
Forrás: a CF Worker /h/all alias-szal szinkronban lévő, frissített adatokat használja.
Kimenet:
  public/<ASSET>/signal.json — a jelzés ("buy", "sell", "no entry") 
                                és annak okai, minden eszközhöz
  public/analysis_summary.json — összefoglaló minden eszközre
  public/analysis.html       — egyszerű HTML index a jelentéshez
Szabályok (példa): 4H→1H bias, 79% fib ("swing hi-lo"), 5M breakout, RR≥1.5, P≥60% → jelzés.
"""

import os, json, re, time, traceback
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

# --- Elemzendő eszközök ---
ASSETS = ["SOL", "NSDQ100", "GOLD_CFD"]

# Segédfüggvény a JSON fájlok mentéséhez
def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def nowiso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --- OHLC normalizáló (támogatja az ohlc_utc_ms listát) ---
def df_from_ohlc_obj(obj) -> pd.DataFrame:
    if not obj:
        return pd.DataFrame(columns=["open","high","low","close"])
    if "ohlc_utc_ms" in obj and isinstance(obj["ohlc_utc_ms"], list):
        arr = obj["ohlc_utc_ms"]
        if not arr:
            return pd.DataFrame(columns=["open","high","low","close"])
        idx = pd.to_datetime([x[0] for x in arr], unit="ms", utc=True)
        df = pd.DataFrame({
            "open":  [float(x[1]) for x in arr],
            "high":  [float(x[2]) for x in arr],
            "low":   [float(x[3]) for x in arr],
            "close": [float(x[4]) for x in arr],
        }, index=idx)
        return df
    if "ohlc" in obj and isinstance(obj["ohlc"], list):
        # (Nem használt formátum itt, de támogatva van.)
        data = obj["ohlc"]
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df["date"], utc=True)
        return df[["open","high","low","close"]]
    return pd.DataFrame(columns=["open","high","low","close"])

# --- Kockázat-számító segédfüggvények (példák) ---
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi14(series):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rs = ma_up/ma_down
    rsi = 100 - (100/(1+rs))
    return rsi

def macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def bb(series, n=20):
    m = series.rolling(n).mean()
    s = series.rolling(n).std()
    upper = m + 2*s
    lower = m - 2*s
    return m, upper, lower

# --- fő függvény egy eszközre ---
def analyze(asset: str) -> dict:
    outdir = os.path.join("public", asset)
    os.makedirs(outdir, exist_ok=True)
    result = {"asset": asset, "ok": True, "retrieved_at_utc": nowiso(), "errors": []}

    # 1) Lokális all_<asset>.json beolvasása
    try:
        with open(f"all_{asset}.json", "r", encoding="utf-8") as f:
            all_json = json.load(f)
    except Exception as e:
        msg = {"asset": asset, "ok": False, "error": f"Nem sikerült beolvasni all_{asset}.json: {e}"}
        save_json(os.path.join(outdir, "signal.json"), msg)
        return msg

    # 2) Adatok előkészítése
    spot_obj = all_json.get("spot") or {}
    spot_price = spot_obj.get("price_usd")
    # OHLC idősorok DataFrame-mé konvertálva
    df5 = df_from_ohlc_obj(all_json.get("k5m"))
    df1 = df_from_ohlc_obj(all_json.get("k1h"))
    df4 = df_from_ohlc_obj(all_json.get("k4h"))

    # 3) Érvelés/indikátorok (példa: trend bias)
    b4 = "bull" if not df4.empty and df4["close"].iloc[-1] >= df4["open"].iloc[-1] else "bear"
    b1 = "bull" if not df1.empty and df1["close"].iloc[-1] >= df1["open"].iloc[-1] else "bear"

    # 4) 79% Fibonacci szintek kiszámítása (kötött fej-fej paraméterekkel példa)
    near79 = False
    fib_info = {}
    if not df1.empty:
        hi = df1["close"].max()
        lo = df1["close"].min()
        up79 = lo + 0.79 * (hi - lo)
        dn79 = hi - 0.79 * (hi - lo)
        last5 = df5["close"].iloc[-1] if not df5.empty else df1["close"].iloc[-1]
        if abs(last5 - up79) < 0.0035 * last5 or abs(last5 - dn79) < 0.0035 * last5:
            near79 = True
        fib_info = {"swing_hi": hi, "swing_lo": lo, "up79": up79, "dn79": dn79, "last": float(last5)}

    # 5) MACD, RSI, Bollinger feltételek (példa)
    decision = "no entry"
    reasons = []
    if not df1.empty:
        closes = df1["close"]
        ema20 = ema(closes, 20).iloc[-1]
        ema50 = ema(closes, 50).iloc[-1]
        rsi_val = rsi14(closes).iloc[-1]
        macd_line, macd_signal, _ = macd(closes)
        macd_val = macd_line.iloc[-1] - macd_signal.iloc[-1]
        bb_mid, bb_up, bb_lo = bb(closes)
        price = df1["close"].iloc[-1]
        # Egyszerű példa szabályok:
        if b4 == "bull" and b1 == "bull" and near79 and price > bb_mid.iloc[-1] and (price - ema20) > 0:
            decision = "buy"
            reasons.append("4H és 1H bull bias, közel 79% fib, fölötte a Bollinger középértéknek")
        elif b4 == "bear" and b1 == "bear" and near79 and price < bb_mid.iloc[-1]:
            decision = "sell"
            reasons.append("4H és 1H bear bias, közel 79% fib, alatta a Bollinger középértéknek")

    # 6) Eredmény mentése JSON-ba
    decision_obj = {
        "asset": asset,
        "ok": True,
        "retrieved_at_utc": nowiso(),
        "signal": decision,
        "reasons": reasons or ["no signal"],
    }
    save_json(os.path.join(outdir, "signal.json"), decision_obj)
    return decision_obj

def main():
    summary = {"ok": True, "assets": {}}
    for asset in ASSETS:
        try:
            res = analyze(asset)
            summary["assets"][asset] = res
        except Exception as e:
            summary["assets"][asset] = {"asset": asset, "ok": False, "error": str(e)}
    # Összefoglaló mentése és elemzési HTML generálása
    save_json(os.path.join("public", "analysis_summary.json"), summary)
    # Egyszerű HTML oldal (opcionális)
    html = "<!doctype html><meta charset='utf-8'><title>Analysis Summary</title><h1>Analysis Summary</h1>"
    html += "<pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre>"
    save_json(os.path.join("public", "analysis.html"), {"html": html})
    with open("public/analysis.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()

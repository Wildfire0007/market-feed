# Trading.py
# --- Adatgyűjtés + jel-generálás Pages publikáláshoz -------------------------
# SOL: Cloudflare Worker (ingyenes), 1h -> 4h aggregálás Pythonban
# GOLD_CFD, NSDQ100: TwelveData (csak 1 kérés/asset: 5min -> 1h aggregálás)

import os, json, math, time
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

# ----------------------------- Beállítások -----------------------------------

CF_WORKER_BASE = os.getenv(
    "CF_WORKER_BASE",
    "https://market-feed-proxy.czipo-agnes.workers.dev"
).rstrip("/")

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

OUTDIR = "public"

# Mit publikálunk
ASSETS = ["SOL", "GOLD_CFD", "NSDQ100"]

# Mennyi historika elég
TD_OUTPUTSIZE_5M = 300   # ~25 óra (kvótakímélő, de elég 1h resample-hez)
SOL_OUTPUTSIZE_1H = 400  # worker default bőven elég szignálhoz

# ------------------------------ Segédfüggvények ------------------------------

def nowiso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def http_get_json(url: str, params=None, timeout=30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def df_from_values(values):
    """TwelveData/Worker féle values -> pandas OHLC DF UTC index-szel"""
    if not values:
        return pd.DataFrame(columns=["open","high","low","close"])
    rows = []
    for it in values:
        dt = pd.to_datetime(it.get("datetime") or it.get("time") or it.get("timestamp"), utc=True)
        rows.append([
            dt,
            to_float(it.get("open")),
            to_float(it.get("high")),
            to_float(it.get("low")),
            to_float(it.get("close")),
        ])
    df = pd.DataFrame(rows, columns=["datetime","open","high","low","close"]).set_index("datetime").sort_index()
    return df

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Pandas OHLC resample (right-closed), utolsó csonka gyertyát eldobjuk"""
    if df.empty:
        return df.copy()
    rs = df.resample(rule, label="right", closed="right").agg({
        "open":"first","high":"max","low":"min","close":"last"
    })
    # Drop csonka utolsó bar, ha épp most képződik
    if not rs.empty and rs.index[-1] > df.index[-1]:
        rs = rs.iloc[:-1]
    return rs.dropna()

# --- indikátorok ---

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, n=20, k=2.0):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std()
    upper = ma + k*sd
    lower = ma - k*sd
    return ma, upper, lower

# --- jelgenerálás (egyszerű, óvatos): ---------------------------------------
def basic_signal(df: pd.DataFrame):
    """
    Egyszerű szabály:
    - BUY: close > EMA(21) és MACD hist > 0 és RSI > 50
    - SELL: close < EMA(21) és MACD hist < 0 és RSI < 50
    - különben: no entry
    """
    if df is None or df.empty or len(df) < 30:
        return "no entry", ["insufficient data"]

    close = df["close"]
    e21 = ema(close, 21)
    macd_line, sig, hist = macd(close)
    r = rsi(close, 14)

    last = df.index[-1]
    cond_buy  = (close.loc[last] > e21.loc[last]) and (hist.loc[last] > 0) and (r.loc[last] > 50)
    cond_sell = (close.loc[last] < e21.loc[last]) and (hist.loc[last] < 0) and (r.loc[last] < 50)

    if cond_buy:
        return "buy",  ["close>ema21", "macd_hist>0", "rsi>50"]
    if cond_sell:
        return "sell", ["close<ema21", "macd_hist<0", "rsi<50"]
    return "no entry", ["no signal"]

# ----------------------------- Források --------------------------------------

# --- SOL: Cloudflare Worker ---
def worker_spot(asset="SOL"):
    url = f"{CF_WORKER_BASE}/spot"
    try:
        data = http_get_json(url, {"asset": asset})
        ok = bool(data.get("ok"))
        if not ok:
            return {"asset": asset, "ok": False, "error": data.get("error","worker spot error"), "retrieved_at_utc": nowiso()}
        # Átnevezés és vékonyítás
        out = {
            "asset": asset,
            "ok": True,
            "symbol": data.get("symbol","SOL/USD"),
            "price": to_float(data.get("price")),
            "bid": to_float(data.get("nav",{}).get("bid")),
            "ask": to_float(data.get("nav",{}).get("ask")),
            "volume": to_float(data.get("volume")),
            "time": data.get("time"),
            "retrieved_at_utc": nowiso(),
        }
        return out
    except Exception as e:
        return {"asset": asset, "ok": False, "error": f"spot: {e}", "retrieved_at_utc": nowiso()}

def worker_k(asset="SOL", interval="1h"):
    url = f"{CF_WORKER_BASE}/k{interval}"
    try:
        data = http_get_json(url, {"asset": asset})
        values = data.get("values") or []
        df = df_from_values(values)
        return {"ok": True, "df": df, "meta": data.get("meta")}
    except Exception as e:
        return {"ok": False, "error": f"k{interval}: {e}"}

# --- TwelveData: 5m idősor -> resample 1h ---
def td_timeseries_5m(symbol: str):
    if not TD_API_KEY:
        return {"ok": False, "error": "Missing TWELVEDATA_API_KEY"}
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "5min",
        "outputsize": TD_OUTPUTSIZE_5M,
        "timezone": "UTC",
        "order": "ASC",
        "format": "JSON",
        "apikey": TD_API_KEY,
    }
    try:
        data = http_get_json(url, params=params)
        if "status" in data and data["status"] == "error":
            return {"ok": False, "error": data.get("message", "TwelveData error")}
        values = data.get("values") or data.get("data") or []
        df = df_from_values(values)
        return {"ok": True, "df5m": df, "meta": data.get("meta") or {}}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ------------------------------ Pipeline-ek ----------------------------------

def build_SOL():
    # Spot
    spot = worker_spot("SOL")
    save_json(f"{OUTDIR}/SOL/spot.json", spot)

    # 1h (worker), 4h aggregálás
    h1 = worker_k("SOL", "1h")
    if not h1.get("ok"):
        save_json(f"{OUTDIR}/SOL/k1h.json", {
            "asset":"SOL","ok":False,"error":h1.get("error","worker k1h error"),"retrieved_at_utc":nowiso()
        })
        # 4h is bukik
        save_json(f"{OUTDIR}/SOL/k4h.json", {
            "asset":"SOL","ok":False,"error":"no 1h data","retrieved_at_utc":nowiso()
        })
        # jel
        save_json(f"{OUTDIR}/SOL/signal.json", {
            "asset":"SOL","ok":True,"retrieved_at_utc":nowiso(),"signal":"no entry","reasons":["no signal"]
        })
        return

    df1h = h1["df"]
    # mentés k1h
    k1h_out = {
        "asset":"SOL","ok":True,"retrieved_at_utc":nowiso(),
        "meta": h1.get("meta"),
        "values":[
            {
                "datetime": ts.isoformat(),
                "open": float(o), "high": float(h), "low": float(l), "close": float(c)
            }
            for ts,(o,h,l,c) in zip(df1h.index, df1h[["open","high","low","close"]].to_numpy())
        ]
    }
    save_json(f"{OUTDIR}/SOL/k1h.json", k1h_out)

    # 4h aggregálás
    df4h = resample_ohlc(df1h, "4H")
    k4h_out = {
        "asset":"SOL","ok":True,"retrieved_at_utc":nowiso(),
        "meta":{"source":"worker 1h → resample 4h"},
        "values":[
            {
                "datetime": ts.isoformat(),
                "open": float(o), "high": float(h), "low": float(l), "close": float(c)
            }
            for ts,(o,h,l,c) in zip(df4h.index, df4h[["open","high","low","close"]].to_numpy())
        ]
    }
    save_json(f"{OUTDIR}/SOL/k4h.json", k4h_out)

    # Jel (4h alapján, konzervatívabb)
    signal, reasons = basic_signal(df4h)
    save_json(f"{OUTDIR}/SOL/signal.json", {
        "asset":"SOL","ok":True,"retrieved_at_utc":nowiso(),"signal":signal,"reasons":reasons
    })

def build_TD_asset(asset: str, symbol: str):
    """
    Általános TD pipeline: 5m → mentjük, majd 1h resample → jel
    """
    td = td_timeseries_5m(symbol)
    if not td.get("ok"):
        # legalább egy error-json legyen kint
        save_json(f"{OUTDIR}/{asset}/k5m.json", {
            "asset":asset,"ok":False,"error":td.get("error","twelvedata error"),"retrieved_at_utc":nowiso()
        })
        save_json(f"{OUTDIR}/{asset}/k1h.json", {
            "asset":asset,"ok":False,"error":"no 5m data","retrieved_at_utc":nowiso()
        })
        save_json(f"{OUTDIR}/{asset}/signal.json", {
            "asset":asset,"ok":True,"retrieved_at_utc":nowiso(),"signal":"no entry","reasons":["no data"]
        })
        return

    df5 = td["df5m"]
    # Mentsük a 5m sort
    k5m_out = {
        "asset":asset,"ok":True,"retrieved_at_utc":nowiso(),
        "meta": td.get("meta",{}),
        "values":[
            {
                "datetime": ts.isoformat(),
                "open": float(o), "high": float(h), "low": float(l), "close": float(c)
            }
            for ts,(o,h,l,c) in zip(df5.index, df5[["open","high","low","close"]].to_numpy())
        ]
    }
    save_json(f"{OUTDIR}/{asset}/k5m.json", k5m_out)

    # 1h resample
    df1 = resample_ohlc(df5, "1H")
    k1h_out = {
        "asset":asset,"ok":True,"retrieved_at_utc":nowiso(),
        "meta":{"source":"twelvedata 5m → resample 1h"},
        "values":[
            {
                "datetime": ts.isoformat(),
                "open": float(o), "high": float(h), "low": float(l), "close": float(c)
            }
            for ts,(o,h,l,c) in zip(df1.index, df1[["open","high","low","close"]].to_numpy())
        ]
    }
    save_json(f"{OUTDIR}/{asset}/k1h.json", k1h_out)

    # Jel (1h alapján)
    signal, reasons = basic_signal(df1)
    save_json(f"{OUTDIR}/{asset}/signal.json", {
        "asset":asset,"ok":True,"retrieved_at_utc":nowiso(),"signal":signal,"reasons":reasons
    })

def main():
    ensure_dir(OUTDIR)

    # --- SOL (worker) ---
    try:
        build_SOL()
        print("[SOL] done.")
    except Exception as e:
        print("[SOL] ERROR:", e)
        save_json(f"{OUTDIR}/SOL/signal.json", {
            "asset":"SOL","ok":False,"error":str(e),"retrieved_at_utc":nowiso()
        })

    # --- GOLD_CFD (TD: XAU/USD) ---
    # TwelveData-hoz a "XAU/USD" a legstabilabb jelölés az aranyra
    try:
        build_TD_asset("GOLD_CFD", "XAU/USD")
        print("[GOLD_CFD] done.")
    except Exception as e:
        print("[GOLD_CFD] ERROR:", e)
        save_json(f"{OUTDIR}/GOLD_CFD/signal.json", {
            "asset":"GOLD_CFD","ok":False,"error":str(e),"retrieved_at_utc":nowiso()
        })

    # --- NSDQ100 (TD: QQQ vagy NDX100USD) ---
    # Itt sok brókernél a QQQ ETF ára közkedvelt proxy — TwelveData-n QQQ működik.
    try:
        build_TD_asset("NSDQ100", "QQQ")
        print("[NSDQ100] done.")
    except Exception as e:
        print("[NSDQ100] ERROR:", e)
        save_json(f"{OUTDIR}/NSDQ100/signal.json", {
            "asset":"NSDQ100","ok":False,"error":str(e),"retrieved_at_utc":nowiso()
        })

if __name__ == "__main__":
    main()

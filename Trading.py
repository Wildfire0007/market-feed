# -*- coding: utf-8 -*-
"""
Trading.py — all_*.json feed generátor 1M triggerrel (SOL, NSDQ100, GOLD_CFD)
Kimenetek (repo gyökerébe):
  - all_SOL.json
  - all_NSDQ100.json
  - all_GOLD_CFD.json

Adatforrások:
  Spot:
    - SOL: CoinGecko simple/price (kulcs nélkül)
    - NSDQ100 (QQQ): Twelve Data quote (TWELVEDATA_API_KEY szükséges)
    - GOLD_CFD (XAU/USD): Twelve Data quote (TWELVEDATA_API_KEY szükséges)
  OHLC:
    - SOL: Binance klines (SOLUSDT, kulcs nélkül)
    - NSDQ100 (QQQ): Twelve Data time_series
    - GOLD_CFD (XAU/USD): Twelve Data time_series

A JSON-ba bekerül:
  signal.trigger_1m (true/false) + trigger_meta{reason, side, trigger_price, trigger_ts_utc}

Futtatás:
  export TWELVEDATA_API_KEY="…"
  python Trading.py
"""

import os
import time
import json
import math
from typing import Optional, Callable, Any, Dict, List
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np


# ================================
# KÖZÖS SEGÉDFÜGGVÉNYEK
# ================================

TD_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()

def _ts_utc_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")

def _ensure_ok_json(resp: requests.Response) -> Dict[str, Any]:
    """Twelve Data hibák egységes kezelése, JSON biztosítása."""
    try:
        data = resp.json()
    except Exception:
        resp.raise_for_status()
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(f"TwelveData error: code={data.get('code')} msg={data.get('message')}")
    return data

def _retry_fetch(fn: Callable[[], Any], tries: int = 3, sleep: float = 0.7) -> Any:
    last_exc = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(sleep)
    # ha itt vagyunk, végleg hibás
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown fetch error")

def _price_tick_for(asset: str) -> float:
    asset = asset.upper()
    if asset in ("NSDQ100", "GOLD_CFD"):
        return 0.01
    return 0.0001


# ================================
# 1M TRIGGER + 5M ELŐFELTÉTELEK
# ================================

def last_swing_levels_1m(df1m: pd.DataFrame, lookback: int = 20):
    """Utolsó lookback×1M ablakból max high / min low (UTC indexet és OHLC oszlopokat vár)."""
    if df1m is None or df1m.empty or len(df1m) < max(lookback, 5):
        return (None, None)
    win = df1m.tail(lookback)
    return float(win['high'].max()), float(win['low'].min())

def compute_trigger_1m(signal_dict: Dict[str, Any],
                       df5m: pd.DataFrame,
                       df1m: pd.DataFrame,
                       price_tick: float = 0.0001) -> Dict[str, Any]:
    """
    Trigger logika:
      - Előfeltételek: bias_5m in {long,short} AND bos_5m AND continuation_5m AND fib79_ok
      - Mikró-BOS (1M): utolsó zárt 1M close > swing_high + buffer (long) / < swing_low - buffer (short)
    """
    out = {"trigger_1m": False,
           "trigger_meta": {"reason": None, "side": None, "trigger_price": None, "trigger_ts_utc": None}}

    side = signal_dict.get("bias_5m")
    if side not in ("long", "short"):
        out["trigger_meta"]["reason"] = "no_bias_5m"; return out
    if not signal_dict.get("bos_5m", False):
        out["trigger_meta"]["reason"] = "no_bos_5m"; return out
    if not signal_dict.get("continuation_5m", False):
        out["trigger_meta"]["reason"] = "no_continuation_5m"; return out
    if not signal_dict.get("fib79_ok", False):
        out["trigger_meta"]["reason"] = "no_fib79"; return out

    if df1m is None or df1m.empty:
        out["trigger_meta"]["reason"] = "k1m_missing"; out["trigger_meta"]["side"] = side
        return out

    last_row = df1m.tail(1).iloc[0]
    last_close = float(last_row["close"])
    swing_hi, swing_lo = last_swing_levels_1m(df1m, lookback=20)
    if swing_hi is None or swing_lo is None:
        out["trigger_meta"]["reason"] = "not_enough_1m"; out["trigger_meta"]["side"] = side
        return out

    pct_buffer = 0.0002  # 0.02%
    price_buffer = max(last_close * pct_buffer, 2.0 * price_tick)
    ts_utc = df1m.tail(1).index[0].to_pydatetime().replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z")

    if side == "long":
        needed = swing_hi + price_buffer
        if last_close > needed:
            out["trigger_1m"] = True
            out["trigger_meta"].update({"reason":"ok","side":"long","trigger_price":round(last_close,8),"trigger_ts_utc":ts_utc})
        else:
            out["trigger_meta"].update({"reason":"micro_bos_fail","side":"long","trigger_price":round(last_close,8),"trigger_ts_utc":ts_utc})
    else:  # short
        needed = swing_lo - price_buffer
        if last_close < needed:
            out["trigger_1m"] = True
            out["trigger_meta"].update({"reason":"ok","side":"short","trigger_price":round(last_close,8),"trigger_ts_utc":ts_utc})
        else:
            out["trigger_meta"].update({"reason":"micro_bos_fail","side":"short","trigger_price":round(last_close,8),"trigger_ts_utc":ts_utc})

    return out

def simple_bias_bos_continuation_fib79(df5m: pd.DataFrame) -> Dict[str, Any]:
    """
    Konzervatív 5M előfeltétel detektor (ideiglenes). Ha van saját BOS/FVG/OB/Breaker/79% Fib logikád, cseréld arra.
    """
    out = {"bias_5m": None, "bos_5m": False, "continuation_5m": False, "fib79_ok": False}
    if df5m is None or len(df5m) < 40:  # min. minta
        return out
    win = df5m.tail(40)
    highs = win['high'].values
    lows  = win['low'].values
    closes = win['close'].values

    # primitív bias: utolsó 20 close trendje
    last20 = closes[-20:]
    bias = "long" if last20[-1] > last20[0] else "short"
    out["bias_5m"] = bias

    swin_hi = float(highs[-20:].max())
    swin_lo = float(lows[-20:].min())
    last_close = float(closes[-1])

    if bias == "long" and last_close > swin_hi:
        out["bos_5m"] = True
        mid = 0.5 * (swin_hi + swin_lo)
        out["continuation_5m"] = bool((win['close'].tail(3) > mid).all())
        leg = swin_hi - swin_lo
        if leg > 0:
            recent_mean = float(pd.Series(closes[-5:]).mean())
            out["fib79_ok"] = (recent_mean >= swin_lo + 0.62*leg) and (recent_mean <= swin_lo + 0.79*leg)
    elif bias == "short" and last_close < swin_lo:
        out["bos_5m"] = True
        mid = 0.5 * (swin_hi + swin_lo)
        out["continuation_5m"] = bool((win['close'].tail(3) < mid).all())
        leg = swin_hi - swin_lo
        if leg > 0:
            recent_mean = float(pd.Series(closes[-5:]).mean())
            out["fib79_ok"] = (recent_mean <= swin_hi - 0.62*leg) and (recent_mean >= swin_hi - 0.79*leg)

    return out


# ================================
# SPOT FORRÁSOK (SOL/QQQ/XAU)
# ================================

def fetch_spot(asset: str) -> Dict[str, Any]:
    """
    Visszaadott forma:
    {
      "price_usd": <float>,
      "last_updated_at": "YYYY-MM-DDTHH:MM:SSZ",
      "age_sec": <int>
    }
    """
    asset = asset.upper()

    if asset == "SOL":
        # CoinGecko simple/price — kulcs nélkül
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "solana", "vs_currencies": "usd", "include_last_updated_at": "true"}
        def _do():
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            price = float(data["solana"]["usd"])
            ts = int(data["solana"].get("last_updated_at", time.time()))
            return {
                "price_usd": price,
                "last_updated_at": _ts_utc_iso(ts),
                "age_sec": max(0, int(time.time() - ts))
            }
        return _retry_fetch(_do)

    elif asset == "NSDQ100":
        # Twelve Data quote (QQQ)
        if not TD_KEY:
            raise RuntimeError("Missing env TWELVEDATA_API_KEY for NSDQ100 spot")
        url = "https://api.twelvedata.com/quote"
        params = {"symbol": "QQQ", "apikey": TD_KEY}
        def _do():
            r = requests.get(url, params=params, timeout=15)
            data = _ensure_ok_json(r)
            price = float(data["price"])
            ts_iso = data.get("timestamp")  # pl. "2025-10-01 10:42:39"
            if ts_iso and " " in ts_iso:
                ts_iso = ts_iso.replace(" ", "T") + "Z"
            return {
                "price_usd": price,
                "last_updated_at": ts_iso or datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                "age_sec": 0
            }
        return _retry_fetch(_do)

    elif asset == "GOLD_CFD":
        # Twelve Data quote (XAU/USD) — proxy a GOLD_CFD-hez
        if not TD_KEY:
            raise RuntimeError("Missing env TWELVEDATA_API_KEY for GOLD_CFD spot")
        url = "https://api.twelvedata.com/quote"
        params = {"symbol": "XAU/USD", "apikey": TD_KEY}
        def _do():
            r = requests.get(url, params=params, timeout=15)
            data = _ensure_ok_json(r)
            price = float(data["price"])
            ts_iso = data.get("timestamp")
            if ts_iso and " " in ts_iso:
                ts_iso = ts_iso.replace(" ", "T") + "Z"
            return {
                "price_usd": price,
                "last_updated_at": ts_iso or datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                "age_sec": 0
            }
        return _retry_fetch(_do)

    else:
        raise ValueError(f"Unknown asset for spot: {asset}")


# ================================
# OHLC FORRÁSOK (SOL/QQQ/XAU)
# ================================

def _td_interval(interval: str) -> str:
    m = {"1m": "1min", "5m": "5min", "1h": "1h", "4h": "4h"}
    if interval not in m:
        raise ValueError(f"Unsupported interval: {interval}")
    return m[interval]

def _make_df_from_td(values: List[Dict[str, str]]) -> pd.DataFrame:
    """Twelve Data time_series -> Pandas DF (UTC index, float OHLC)."""
    if not values:
        return pd.DataFrame(columns=["open","high","low","close"])
    df = pd.DataFrame(values)
    # Twelve Data gyakran új->régi sorrendben adja: fordítsuk meg
    df = df.iloc[::-1].copy()
    dt = pd.to_datetime(df["datetime"].str.replace(" ", "T"), utc=True)
    df = df.assign(open=df["open"].astype(float),
                   high=df["high"].astype(float),
                   low=df["low"].astype(float),
                   close=df["close"].astype(float)).drop(columns=["datetime"])
    df.index = dt
    return df[["open","high","low","close"]]

def _binance_interval(interval: str) -> str:
    m = {"1m":"1m", "5m":"5m", "1h":"1h", "4h":"4h"}
    if interval not in m:
        raise ValueError(f"Unsupported interval: {interval}")
    return m[interval]

def _make_df_from_binance(klines: List[List[Any]]) -> pd.DataFrame:
    """
    Binance kline sor: [openTime, open, high, low, close, volume, closeTime, ...]
    """
    if not klines:
        return pd.DataFrame(columns=["open","high","low","close"])
    T = []
    for k in klines:
        T.append({
            "ts": pd.to_datetime(int(k[0]), unit="ms", utc=True),
            "open": float(k[1]),
            "high": float(k[2]),
            "low":  float(k[3]),
            "close":float(k[4])
        })
    df = pd.DataFrame(T).set_index("ts")
    return df[["open","high","low","close"]]

def fetch_ohlc(asset: str, interval: str, periods: int = 240) -> pd.DataFrame:
    """
    Vissza: Pandas DataFrame (UTC index), oszlopok: open/high/low/close (float)
      - SOL → Binance klines (SOLUSDT)
      - NSDQ100 → Twelve Data time_series (QQQ)
      - GOLD_CFD → Twelve Data time_series (XAU/USD)
    """
    asset = asset.upper()

    if asset == "SOL":
        sym = "SOLUSDT"
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": sym, "interval": _binance_interval(interval), "limit": max(200, periods)}
        def _do():
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            return _make_df_from_binance(data)
        return _retry_fetch(_do)

    elif asset == "NSDQ100":
        if not TD_KEY:
            raise RuntimeError("Missing env TWELVEDATA_API_KEY for NSDQ100 OHLC")
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": "QQQ", "interval": _td_interval(interval), "outputsize": max(200, periods), "apikey": TD_KEY}
        def _do():
            r = requests.get(url, params=params, timeout=20)
            data = _ensure_ok_json(r)
            return _make_df_from_td(data.get("values", []))
        return _retry_fetch(_do)

    elif asset == "GOLD_CFD":
        if not TD_KEY:
            raise RuntimeError("Missing env TWELVEDATA_API_KEY for GOLD_CFD OHLC")
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": "XAU/USD", "interval": _td_interval(interval), "outputsize": max(200, periods), "apikey": TD_KEY}
        def _do():
            r = requests.get(url, params=params, timeout=20)
            data = _ensure_ok_json(r)
            return _make_df_from_td(data.get("values", []))
        return _retry_fetch(_do)

    else:
        raise ValueError(f"Unknown asset for OHLC: {asset}")


# ================================
# JSON ÖSSZEÁLLÍTÁS
# ================================

def df_to_ohlc_list(df: pd.DataFrame, limit: int = 300) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    dfx = df.tail(limit).copy().reset_index().rename(columns={"index": "ts"})
    dfx["ts"] = pd.to_datetime(dfx["ts"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dfx[["ts", "open", "high", "low", "close"]].astype(
        {k: float for k in ["open", "high", "low", "close"]}
    ).to_dict(orient="records")

def assemble_all_json_for_asset(asset: str) -> Dict[str, Any]:
    # 0) spot
    spot = fetch_spot(asset)

    # 1) OHLC (1m/5m/1h/4h) — 1m a triggerhez, 5m/1h/4h a top-down elemzéshez
    df1m = fetch_ohlc(asset, "1m")
    df5m = fetch_ohlc(asset, "5m")
    df1h = fetch_ohlc(asset, "1h")
    df4h = fetch_ohlc(asset, "4h")

    # 2) 5M előfeltételek (konzervatív detektor; ha van profibb, cseréld)
    signal_dict = simple_bias_bos_continuation_fib79(df5m)

    # 3) 1M trigger
    price_tick = _price_tick_for(asset)
    trig = compute_trigger_1m(signal_dict, df5m, df1m, price_tick=price_tick)
    signal_dict["trigger_1m"] = trig["trigger_1m"]
    signal_dict["trigger_meta"] = trig["trigger_meta"]

    # 4) Kimeneti JSON (k1m nem kötelező; a trigger a signal mezőben van)
    bundle = {
        "asset": asset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "spot": spot,
        "k5m": {"interval": "5m", "ohlc": df_to_ohlc_list(df5m)},
        "k1h": {"interval": "1h", "ohlc": df_to_ohlc_list(df1h)},
        "k4h": {"interval": "4h", "ohlc": df_to_ohlc_list(df4h)},
        "signal": signal_dict
    }
    return bundle

def write_all_json(asset: str, bundle: Dict[str, Any], out_dir: str = ".") -> None:
    path = os.path.join(out_dir, f"all_{asset}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"[OK] írás: {path}")


# ================================
# MAIN
# ================================

def main():
    for asset in ("SOL", "NSDQ100", "GOLD_CFD"):
        bundle = assemble_all_json_for_asset(asset)
        write_all_json(asset, bundle)
    print("Kész: all_SOL.json, all_NSDQ100.json, all_GOLD_CFD.json")

if __name__ == "__main__":
    # kis random mag a reálisabb szimulációhoz (ha np-t használsz máshol)
    np.random.seed(int(time.time()) % 2_000_000_000)
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD-only market-feed / Trading.py
Minden adat CSAK a Twelve Data REST API-ból jön.

Kimenetek (változatlan könyvtárstruktúra):
  public/<ASSET>/spot.json
  public/<ASSET>/klines_5m.json
  public/<ASSET>/klines_1h.json
  public/<ASSET>/klines_4h.json
  public/<ASSET>/signal.json  (5m EMA9–EMA21 5 bar szabály)

Környezeti változók:
  TWELVEDATA_API_KEY = "<api key>"
  OUT_DIR            = "public" (alapértelmezés)
  TD_PAUSE           = "0.3"    (kímélő szünet hívások közt, sec)
"""

import os, json, time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import requests

OUT_DIR = os.getenv("OUT_DIR", "public")
API_KEY = os.environ["TWELVEDATA_API_KEY"].strip()
TD_BASE = "https://api.twelvedata.com"
TD_PAUSE = float(os.getenv("TD_PAUSE", "0.3"))

ASSETS = {
    # exchange nem kötelező, de SOL-nál tipikusan van Binance feed is
    "SOL":      {"symbol": "SOL/USD",  "exchange": "Binance"},
    "NSDQ100":  {"symbol": "QQQ",      "exchange": None},       # ETF proxy
    "GOLD_CFD": {"symbol": "XAU/USD",  "exchange": None},       # XAU/USD proxy
}

# ---------- segédek ----------
def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def td_get(path: str, **params) -> Dict[str, Any]:
    params["apikey"] = API_KEY
    r = requests.get(f"{TD_BASE}/{path}", params=params, timeout=30,
                     headers={"User-Agent": "market-feed/td-only/1.0"})
    r.raise_for_status()
    data = r.json()
    # TD hibaformátum: {"status":"error","message":...}
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(data.get("message", "TD error"))
    return data

def td_time_series(symbol: str, interval: str, outputsize: int = 500,
                   exchange: Optional[str] = None, order: str = "asc") -> Dict[str, Any]:
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize,
              "order": order, "timezone": "UTC"}
    if exchange:
        params["exchange"] = exchange
    j = td_get("time_series", **params)
    ok = bool(j.get("values"))
    return {
        "asset": symbol,
        "interval": interval,
        "source": "twelvedata:time_series",
        "ok": ok,
        "retrieved_at_utc": now_utc(),
        "raw": j if ok else {"values": []}
    }

def td_quote(symbol: str) -> Dict[str, Any]:
    j = td_get("quote", symbol=symbol)
    price = float(j["price"])
    ts = j.get("datetime") or j.get("timestamp") or now_utc()
    return {
        "asset": symbol,
        "source": "twelvedata:quote",
        "ok": True,
        "retrieved_at_utc": now_utc(),
        "price_usd": price,
        "raw": {"datetime": ts}
    }

def td_last_close(symbol: str, interval: str = "1min", exchange: Optional[str] = None) -> Optional[float]:
    params = {"symbol": symbol, "interval": interval, "outputsize": 1, "order": "desc", "timezone": "UTC"}
    if exchange:
        params["exchange"] = exchange
    j = td_get("time_series", **params)
    vals = j.get("values") or []
    if not vals:
        return None
    return float(vals[0]["close"])

# ---------- jelzés: 5m EMA9–EMA21 (5 bar) ----------
def ema(series: List[Optional[float]], period: int) -> List[Optional[float]]:
    if not series or period <= 1:
        return [None]*len(series)
    k = 2.0 / (period + 1.0)
    out: List[Optional[float]] = []
    prev: Optional[float] = None
    for v in series:
        if v is None:
            out.append(prev)
            continue
        prev = v if prev is None else v * k + prev * (1.0 - k)
        out.append(prev)
    return out

def last_n_true(flags: List[bool], n: int) -> bool:
    return len(flags) >= n and all(flags[-n:])

def closes_from_ts(payload: Dict[str, Any]) -> List[Optional[float]]:
    try:
        vals = payload["raw"]["values"]
        out: List[Optional[float]] = []
        for row in vals:
            try:
                out.append(float(row["close"]))
            except Exception:
                out.append(None)
        return out
    except Exception:
        return []

def signal_from_5m(klines_5m: Dict[str, Any]) -> Dict[str, Any]:
    if not klines_5m.get("ok"):
        return {"ok": False, "signal": "no entry", "reasons": ["missing 5m data"]}
    closes = closes_from_ts(klines_5m)
    if not closes:
        return {"ok": False, "signal": "no entry", "reasons": ["empty 5m data"]}
    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    gt = [False if (a is None or b is None) else (a > b) for a, b in zip(e9, e21)]
    lt = [False if (a is None or b is None) else (a < b) for a, b in zip(e9, e21)]
    if last_n_true(gt, 5):
        return {"ok": True, "signal": "uptrend", "reasons": ["ema9 > ema21 (5 bars)"]}
    if last_n_true(lt, 5):
        return {"ok": True, "signal": "downtrend", "reasons": ["ema9 < ema21 (5 bars)"]}
    return {"ok": True, "signal": "no entry", "reasons": ["no consistent ema bias"]}

# ---------- egy eszköz feldolgozása ----------
def process_asset(asset: str, cfg: Dict[str, Any]) -> None:
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    symbol = cfg["symbol"]
    exch = cfg.get("exchange")

    # 1) Spot
    try:
        if asset == "NSDQ100":
            spot = td_quote(symbol)                 # QQQ -> quote
        else:
            px = td_last_close(symbol, "1min", exch) # SOL, XAU/USD -> 1m last close
            spot = {
                "asset": asset,
                "source": "twelvedata:1min_close",
                "ok": px is not None,
                "retrieved_at_utc": now_utc(),
                "price_usd": float(px) if px is not None else None,
                "raw": {"note": "derived from time_series last close (1min)"}
            }
    except Exception as e:
        spot = {"asset": asset, "source": "twelvedata", "ok": False,
                "retrieved_at_utc": now_utc(), "error": str(e), "price_usd": None}
    save_json(os.path.join(adir, "spot.json"), spot)
    time.sleep(TD_PAUSE)

    # 2) OHLC (5m / 1h / 4h)
    k5  = td_time_series(symbol, "5min", 500, exch, "asc");  save_json(os.path.join(adir, "klines_5m.json"), k5);  time.sleep(TD_PAUSE)
    k1  = td_time_series(symbol, "1h",   500, exch, "asc");  save_json(os.path.join(adir, "klines_1h.json"),  k1); time.sleep(TD_PAUSE)
    k4  = td_time_series(symbol, "4h",   500, exch, "asc");  save_json(os.path.join(adir, "klines_4h.json"),  k4); time.sleep(TD_PAUSE)

    # 3) 5m jelzés
    sig = signal_from_5m(k5)
    save_json(os.path.join(adir, "signal.json"), {
        "asset": asset,
        "ok": bool(sig.get("ok")),
        "retrieved_at_utc": now_utc(),
        "signal": sig.get("signal", "no entry"),
        "reasons": sig.get("reasons", [])
    })

def main():
    if not API_KEY:
        raise SystemExit("TWELVEDATA_API_KEY hiányzik (GitHub Secret).")
    ensure_dir(OUT_DIR)
    for a, cfg in ASSETS.items():
        process_asset(a, cfg)

if __name__ == "__main__":
    main()

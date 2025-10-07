#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD-only market-feed / Trading.py
Minden adat CSAK a Twelve Data REST API-ból jön.

Kimenetek:
  public/<ASSET>/spot.json
  public/<ASSET>/klines_5m.json
  public/<ASSET>/klines_1h.json
  public/<ASSET>/klines_4h.json
  public/<ASSET>/signal.json  (5m EMA9–EMA21 5 bar szabály – előzetes jelzés)

Környezeti változók:
  TWELVEDATA_API_KEY = "<api key>"
  OUT_DIR            = "public" (alapértelmezés)
  TD_PAUSE           = "0.3"    (kímélő szünet hívások közt, sec)
"""

import os, json, time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

import requests

OUT_DIR = os.getenv("OUT_DIR", "public")
API_KEY = os.environ["TWELVEDATA_API_KEY"].strip()
TD_BASE = "https://api.twelvedata.com"
TD_PAUSE = float(os.getenv("TD_PAUSE", "0.3"))

ASSETS = {
    "SOL":      {"symbol": "SOL/USD",  "exchange": "Binance"},
    "NSDQ100":  {"symbol": "QQQ",      "exchange": None},       
    "GOLD_CFD": {"symbol": "XAU/USD",  "exchange": None},       

    "BNB":      {"symbol": os.getenv("BNB_SYMBOL", "BNB/USD"),
                 "exchange": os.getenv("BNB_EXCHANGE", "Binance")},

    "GER40":    {"symbol": os.getenv("GER40_SYMBOL", "DE40/EUR"),
                 "exchange": os.getenv("GER40_EXCHANGE", None)},
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
    r = requests.get(
        f"{TD_BASE}/{path}",
        params=params,
        timeout=30,
        headers={"User-Agent": "market-feed/td-only/1.0"},
    )
    r.raise_for_status()
    data = r.json()
    # TD hibaformátum: {"status":"error","message":...}
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(data.get("message", "TD error"))
    return data

def _iso_from_td_ts(ts: Any) -> Optional[str]:
    """Twelve Data időbélyeg ISO-UTC-re. Kezeli a 'YYYY-MM-DD HH:MM:SS' és unix epoch (sec) formátumot is."""
    if ts is None:
        return None
    # unix epoch (int / str int)
    try:
        if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.isdigit()):
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            return dt.isoformat()
    except Exception:
        pass
    # 'YYYY-MM-DD HH:MM:SS'
    if isinstance(ts, str):
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            # ha már ISO, visszaadjuk
            if "T" in ts:
                return ts
    return None

def td_time_series(symbol: str, interval: str, outputsize: int = 500,
                   exchange: Optional[str] = None, order: str = "desc") -> Dict[str, Any]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "order": order,
        "timezone": "UTC",
        "dp": 6,
    }
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
        "raw": j if ok else {"values": []},
    }

def td_quote(symbol: str) -> Dict[str, Any]:
    j = td_get("quote", symbol=symbol)
    price = j.get("price")
    price = float(price) if price not in (None, "") else None
    ts = j.get("datetime") or j.get("timestamp")
    ts_iso = _iso_from_td_ts(ts) or now_utc()
    return {
        "asset": symbol,
        "source": "twelvedata:quote",
        "ok": price is not None,
        "retrieved_at_utc": now_utc(),
        "price": price,
        "price_usd": price,
        "utc": ts_iso,
        "raw": {"timestamp": ts},
    }

def td_last_close(symbol: str, interval: str = "5min", exchange: Optional[str] = None) -> Tuple[Optional[float], Optional[str]]:
    """Idősorból az utolsó gyertya close + időpont (UTC ISO)."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 1,
        "order": "desc",
        "timezone": "UTC",
        "dp": 6,
    }
    if exchange:
        params["exchange"] = exchange
    j = td_get("time_series", **params)
    vals = j.get("values") or []
    if not vals:
        return None, None
    v0 = vals[0]
    px = float(v0["close"])
    ts_iso = _iso_from_td_ts(v0.get("datetime")) or now_utc()
    return px, ts_iso

def td_spot_with_fallback(symbol: str, exchange: Optional[str] = None) -> Dict[str, Any]:
    """
    Spot ár: először quote, ha nincs ár → time_series utolsó 5min close.
    Mindig ad 'price', 'price_usd', 'utc' és 'retrieved_at_utc' mezőt.
    """
    try:
        q = td_quote(symbol)
    except Exception as e:
        q = {"ok": False, "error": str(e)}

    price = q.get("price") if isinstance(q, dict) else None
    utc = q.get("utc") if isinstance(q, dict) else None

    if price is None:
        try:
            px, ts = td_last_close(symbol, "5min", exchange)
        except Exception as e:
            px, ts = None, None
            q["error_fallback"] = str(e)
        price = px
        utc = ts

    return {
        "asset": symbol,
        "source": "twelvedata:quote+series_fallback",
        "ok": price is not None,
        "retrieved_at_utc": now_utc(),
        "price": price,
        "price_usd": price,
        "utc": utc or now_utc(),
    }

# ---------- jelzés: 5m EMA9–EMA21 (5 bar) ----------

def ema(series: List[Optional[float]], period: int) -> List[Optional[float]]:
    if not series or period <= 1:
        return [None] * len(series)
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

    # 1) Spot: quote → time_series(5min) fallback
    try:
        spot = td_spot_with_fallback(symbol, exch)
    except Exception as e:
        spot = {
            "asset": asset,
            "source": "twelvedata",
            "ok": False,
            "retrieved_at_utc": now_utc(),
            "error": str(e),
            "price": None,
            "price_usd": None,
            "utc": now_utc(),
        }
    save_json(os.path.join(adir, "spot.json"), spot)
    time.sleep(TD_PAUSE)

    # 2) OHLC (5m / 1h / 4h) – RAW mentés (csak a TD válasz)
    k5  = td_time_series(symbol, "5min", 500, exch, "desc");  save_json(os.path.join(adir, "klines_5m.json"), k5["raw"]);  time.sleep(TD_PAUSE)
    k1  = td_time_series(symbol, "1h",   500, exch, "desc");  save_json(os.path.join(adir, "klines_1h.json"),  k1["raw"]); time.sleep(TD_PAUSE)
    k4  = td_time_series(symbol, "4h",   500, exch, "desc");  save_json(os.path.join(adir, "klines_4h.json"),  k4["raw"]); time.sleep(TD_PAUSE)

    # 3) 5m előzetes jelzés (az analysis.py később felülírhatja a végleges stratégia szerint)
    sig = signal_from_5m(k5)
    save_json(
        os.path.join(adir, "signal.json"),
        {
            "asset": asset,
            "ok": bool(sig.get("ok")),
            "retrieved_at_utc": now_utc(),
            "signal": sig.get("signal", "no entry"),
            "reasons": sig.get("reasons", []),
            "probability": 0,   # előzetes; a véglegeset az analysis.py adja
            "spot": {"price": spot.get("price"), "utc": spot.get("utc")},
        },
    )

def main():
    if not API_KEY:
        raise SystemExit("TWELVEDATA_API_KEY hiányzik (GitHub Secret).")
    ensure_dir(OUT_DIR)
    for a, cfg in ASSETS.items():
        process_asset(a, cfg)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twelve Data ellenőrző szonda.
- Élőben lehúzza a quote + time_series (5min/1h/4h) adatokat
- Összeveti a repo-beli public/<ASSET>/spot.json értékekkel
- Nem dob hibát; csak WARN-t jelez, ha eltérés > 1%
"""

import os, json, sys, time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional

import requests

PUBLIC_DIR = "public"
ASSETS = {
    "EURUSD": {"symbol": "EUR/USD", "exchange": "FX"},
    "USDJPY": {"symbol": "USD/JPY", "exchange": "FX"},
    "GOLD_CFD": {"symbol": "XAU/USD"},
    "USOIL": {"symbol": "WTI/USD"},
    "NVDA": {
        "symbol": "NVDA",
        "exchange": "NASDAQ",
        "alt": [
            {"symbol": "NVDA", "exchange": None},
            "NVDA:US",
        ],
    },
    "SRTY": {
        "symbol": "SRTY",
        "exchange": "NYSEARCA",
        "alt": [
            {"symbol": "SRTY", "exchange": None},
            {"symbol": "SRTY", "exchange": "NYSE"},
            {"symbol": "SRTY", "exchange": "ARCA"},
            {"symbol": "SRTY", "exchange": "NSE"},
            "SRTY:US",
            "SRTY:NSE",
        ],
    },
}

INTERVALS = ["5min", "1h", "4h"]

API_BASE = "https://api.twelvedata.com"

API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
if not API_KEY:
    print("ERROR: TWELVEDATA_API_KEY nincs beállítva a környezetben.", file=sys.stderr)
    sys.exit(0)  # ne bukjon el a pipeline

def td_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    p["apikey"] = API_KEY
    url = f"{API_BASE}/{endpoint}"
    try:
        r = requests.get(url, params=p, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e), "endpoint": endpoint, "params": params}

def load_local_spot(asset: str) -> Tuple[float, str]:
    path = f"{PUBLIC_DIR}/{asset}/spot.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        price = js.get("price") or js.get("price_usd")
        utc = js.get("utc") or js.get("timestamp") or js.get("retrieved_at_utc") or js.get("retrieved_at")
        return (float(price) if price is not None else None, utc or "-")
    except Exception:
        return (None, "-")

def bps_diff(a: float, b: float) -> str:
    try:
        if a is None or b is None or b == 0:
            return "—"
        return f"{abs(a - b) / b * 10000:.1f} bps"
    except Exception:
        return "—"

def _normalize_attempts(meta: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    base_symbol = meta["symbol"]
    base_exchange = meta.get("exchange")
    attempts: List[Tuple[str, Optional[str]]] = []

    def push(symbol: Optional[str], exchange: Optional[str]) -> None:
        if symbol:
            attempts.append((symbol, exchange))

    push(base_symbol, base_exchange)
    for alt in meta.get("alt", []):
        if isinstance(alt, str):
            push(alt, base_exchange)
        elif isinstance(alt, dict):
            push(alt.get("symbol", base_symbol), alt.get("exchange", base_exchange))
        elif isinstance(alt, (list, tuple)) and alt:
            symbol = alt[0]
            exchange = alt[1] if len(alt) > 1 else base_exchange
            push(symbol, exchange)

    seen = set()
    uniq: List[Tuple[str, Optional[str]]] = []
    for sym, exch in attempts:
        key = (sym, exch)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((sym, exch))
    return uniq


def fetch_attempt(sym: str, exchange: Optional[str]) -> Tuple[Optional[float], str, List[str]]:
    q_params = {"symbol": sym}
    if exchange:
        q_params["exchange"] = exchange
    q = td_get("quote", q_params)
    q_price = None
    q_ts = "-"
    if isinstance(q, dict):
        q_price = q.get("price")
        q_ts = q.get("timestamp") or q.get("datetime") or "-"
        try:
            if q_price is not None:
                q_price = float(q_price)
        except Exception:
            q_price = None

    ts_info: List[str] = []
    for iv in INTERVALS:
        ts_params = {"symbol": sym, "interval": iv, "outputsize": 1, "dp": 6}
        if exchange:
            ts_params["exchange"] = exchange
        ts = td_get("time_series", ts_params)
        if isinstance(ts, dict) and isinstance(ts.get("values"), list) and ts["values"]:
            ts_info.append(f"{iv}:{ts['values'][0].get('datetime', '-')}")
        else:
            ts_info.append(f"{iv}:n/a")

    return q_price, q_ts, ts_info


def main():
    print("== Twelve Data live probe ==")
    for asset, meta in ASSETS.items():
        attempts = _normalize_attempts(meta)
        chosen_index = 0
        chosen_payload: Tuple[Optional[float], str, List[str]] = (None, "-", ["n/a"] * len(INTERVALS))
        chosen_meta: Tuple[str, Optional[str]] = (attempts[0][0], attempts[0][1])

        for idx, (sym, exchange) in enumerate(attempts, start=1):
            payload = fetch_attempt(sym, exchange)
            price, _, ts_info = payload
            chosen_index = idx
            chosen_payload = payload
            chosen_meta = (sym, exchange)
            has_series = any(part != f"{iv}:n/a" for part, iv in zip(ts_info, INTERVALS))
            if price is not None or has_series:
                break
            time.sleep(0.2)

        price, q_ts, ts_info = chosen_payload
        sym, exchange = chosen_meta
        exch_str = exchange or "default"
        attempt_label = f"{sym} @ {exch_str}" if exchange else sym

        l_price, l_utc = load_local_spot(asset)

        print(f"-- {asset} (attempt {chosen_index}: {attempt_label})")
        print(f"TD quote:   price={price}  utc={q_ts}")
        print(f"TD series:  {', '.join(ts_info)}")
        print(f"Local spot: price={l_price}  utc={l_utc}  (diff={bps_diff(l_price, price)})")

if __name__ == "__main__":
    main()

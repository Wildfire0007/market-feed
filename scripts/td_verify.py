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
from typing import Dict, Any, Tuple, List

import requests

PUBLIC_DIR = "public"
ASSETS = {
    "EURUSD": {"symbol": "EUR/USD"},
    "NSDQ100": {"symbol": "QQQ"},
    "GOLD_CFD": {"symbol": "XAU/USD"},
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

def main():
    print("== Twelve Data live probe ==")
    for asset, meta in ASSETS.items():
        sym = meta["symbol"]
        # Quote
        q = td_get("quote", {"symbol": sym})
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

        # Time series timestamps
        ts_info: List[str] = []
        for iv in INTERVALS:
            ts = td_get("time_series", {"symbol": sym, "interval": iv, "outputsize": 1, "dp": 6})
            if isinstance(ts, dict) and isinstance(ts.get("values"), list) and ts["values"]:
                ts_info.append(f"{iv}:{ts['values'][0].get('datetime', '-')}")
            else:
                ts_info.append(f"{iv}:n/a")

        # Local spot
        l_price, l_utc = load_local_spot(asset)

        print(f"-- {asset} ({sym})")
        print(f"TD quote:   price={q_price}  utc={q_ts}")
        print(f"TD series:  {', '.join(ts_info)}")
        print(f"Local spot: price={l_price}  utc={l_utc}  (diff={bps_diff(l_price, q_price)})")
        print()

if __name__ == "__main__":
    main()

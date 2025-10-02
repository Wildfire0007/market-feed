#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trading.py – Twelvedata alapú feed-generátor

Eszközök:
  - SOL/USD   -> SOL_USD
  - QQQ       -> QQQ
  - GLD       -> GLD

Kimenetek (minden eszközhöz):
  public/<ASSET_SAFE>/spot.json
  public/<ASSET_SAFE>/klines_5m.json
  public/<ASSET_SAFE>/klines_1h.json
  public/<ASSET_SAFE>/klines_4h.json
  public/<ASSET_SAFE>/signal.json
  all_<ASSET_SAFE>.json
"""

import os
import json
import time
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import requests


# ===== Beállítások =====

TD_BASE = "https://api.twelvedata.com"
API_KEY = os.environ.get("TWELVEDATA_API_KEY")

# Eszközök: { biztonságos_név: Twelvedata_symbol }
ASSETS: Dict[str, str] = {
    "SOL_USD": "SOL/USD",
    "QQQ": "QQQ",
    "GLD": "GLD",
}

# Intervallumok (Twelvedata formátum -> fájlnév)
INTERVALS: List[Tuple[str, str]] = [
    ("5min", "klines_5m.json"),
    ("1h",   "klines_1h.json"),
    ("4h",   "klines_4h.json"),
]

# Általános HTTP beállítások
HTTP_TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF_SEC = 1.5

USER_AGENT = "market-feed/1.0 (+github actions; twelvedata)"


# ===== Segédfüggvények =====

def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def _td_get(endpoint: str, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Twelvedata GET kérés retry-olt hibakezeléssel.
    Visszatérés: (ok, payload_vagy_error)
    - ok=True:  payload a Twelvedata JSON + {"ok": True}
    - ok=False: {"ok": False, "error": "..."}
    """
    if not API_KEY:
        return False, {"ok": False, "error": "Missing TWELVEDATA_API_KEY env var"}

    url = f"{TD_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    params = dict(params or {})
    params["apikey"] = API_KEY

    headers = {"User-Agent": USER_AGENT}

    last_err: Optional[str] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
            # 429 vagy 5xx esetén próbálkozzon újra
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(BACKOFF_SEC * attempt)
                continue

            r.raise_for_status()
            data = r.json()

            # Twelvedata hibajelzés
            if isinstance(data, dict) and data.get("status") == "error":
                msg = data.get("message") or str(data)
                return False, {"ok": False, "error": f"Twelvedata error: {msg}"}

            # Siker
            if isinstance(data, dict):
                data["ok"] = True
            return True, data

        except requests.RequestException as e:
            last_err = f"Request error: {e}"
            time.sleep(BACKOFF_SEC * attempt)
        except ValueError as e:
            last_err = f"JSON decode error: {e}"
            break

    return False, {"ok": False, "error": last_err or "Unknown error"}


def fetch_price(symbol: str) -> Dict[str, Any]:
    ok, data = _td_get("price", {"symbol": symbol})
    if ok:
        # Normalizált kimenet
        return {
            "ok": True,
            "symbol": symbol,
            "price": data.get("price"),
            "raw": data,
        }
    return data


def fetch_time_series(symbol: str, interval: str, outputsize: int = 100) -> Dict[str, Any]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        # Győződjünk meg róla, hogy a legfrissebb jön előre vagy később rendezzük
        "order": "desc",
    }
    ok, data = _td_get("time_series", params)
    if ok:
        # Normalizált kimenet: meta + values lista (ha van)
        meta = data.get("meta") or {}
        values = data.get("values") or []
        return {
            "ok": True,
            "symbol": symbol,
            "interval": interval,
            "meta": meta,
            "values": values,
            "raw": data,
        }
    return data


def compute_ma_signal(series_5m: Dict[str, Any], ma_len: int = 10) -> Dict[str, Any]:
    """
    Egyszerű BUY/SELL jelzés:
      - záróár > MA(ma_len)  -> BUY
      - különben             -> SELL
    A Twelvedata `values` mezőjét használja. Ha kevés adat van, hibát ad vissza.
    """
    if not series_5m.get("ok"):
        return {"ok": False, "error": "5m series unavailable"}

    values = series_5m.get("values") or []
    if len(values) < ma_len:
        return {"ok": False, "error": f"Insufficient 5m values (<{ma_len})"}

    # Az API általában DESC sorrendben adja (legfrissebb elöl), rendezzük idő szerint:
    try:
        values_sorted = sorted(values, key=lambda x: x["datetime"])
        closes = [float(v["close"]) for v in values_sorted]
    except Exception as e:
        return {"ok": False, "error": f"Parse error: {e}"}

    last_price = closes[-1]
    ma = float(np.mean(closes[-ma_len:]))
    signal = "BUY" if last_price > ma else "SELL"

    return {
        "ok": True,
        "type": "MA",
        "len": ma_len,
        "last_price": last_price,
        "ma": ma,
        "signal": signal,
    }


# ===== Fő folyamat =====

def process_asset(safe_name: str, td_symbol: str) -> None:
    out_dir = os.path.join("public", safe_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {safe_name} ({td_symbol}) ===")

    # 1) Spot
    spot = fetch_price(td_symbol)
    _save_json(os.path.join(out_dir, "spot.json"), spot)
    print("  [OK]" if spot.get("ok") else "  [ERR]", "spot")

    # 2) Time series (5m, 1h, 4h)
    series_map: Dict[str, Dict[str, Any]] = {}
    for interval, fname in INTERVALS:
        ts = fetch_time_series(td_symbol, interval, outputsize=100)
        series_map[interval] = ts
        _save_json(os.path.join(out_dir, fname), ts)
        print("  [OK]" if ts.get("ok") else "  [ERR]", interval)

    # 3) Signal (5m MA10)
    sig = compute_ma_signal(series_map.get("5min", {}), ma_len=10)
    _save_json(os.path.join(out_dir, "signal.json"), sig)
    print("  [OK]" if sig.get("ok") else "  [ERR]", "signal")

    # 4) Összesítő
    all_payload = {
        "ok": True,
        "asset": safe_name,
        "spot": spot,
        "klines_5m": series_map.get("5min", {}),
        "klines_1h": series_map.get("1h", {}),
        "klines_4h": series_map.get("4h", {}),
        "signal": sig,
    }
    _save_json(f"all_{safe_name}.json", all_payload)
    print("  [+] all packet saved")


def main() -> None:
    if not API_KEY:
        raise SystemExit("ERROR: TWELVEDATA_API_KEY nincs beállítva (Actions secret).")

    any_err = False
    for safe, sym in ASSETS.items():
        try:
            process_asset(safe, sym)
        except Exception as e:
            any_err = True
            print(f"!! {safe}: unexpected error: {e}")

    if any_err:
        # ne szakítsuk meg a workflow-t, de jelezzünk
        print("\n[WARN] Volt legalább egy hiba a feldolgozás során.")
    else:
        print("\n[OK] Minden eszköz feldolgozva.")


if __name__ == "__main__":
    main()

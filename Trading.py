# Trading.py
# Robusztus feed-generátor: decision.json minden assethez a public/<ASSET>/ alatt

import os
import json
import time
import math
import pathlib
from typing import Any, Dict, Optional

import requests


# --------- Beállítások ---------

OUT_DIR = pathlib.Path("public")

# TwelveData API kulcs (Settings → Secrets → Actions → TWELVEDATA_API_KEY)
TD_API = os.environ.get("TWELVEDATA_API_KEY", "").strip()

# Eszközök és források
ASSETS = {
    "SOL": {
        "source": "binance",
        "binance_symbol": "SOLUSDT",
        "note": "Spot ár Binance-ről (SOL/USDT).",
    },
    "NSDQ100": {
        "source": "twelvedata",
        "td_symbol": "NDX",  # Nasdaq-100 index
        "note": "Spot ár TwelveData-ról (fallback price→quote).",
    },
    "GOLD_CFD": {
        "source": "twelvedata",
        "td_symbol": "XAU/USD",  # Arany CFD
        "note": "Spot ár TwelveData-ról (fallback price→quote).",
    },
}

# HTTP időzítések
TIMEOUT = 15
RETRIES = 3
RETRY_SLEEP = 1.2


# --------- Közös segédek ---------

def _http_get_json(url: str, timeout: int = TIMEOUT) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _retry(fn, *args, **kwargs):
    last_err = None
    for _ in range(RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP)
    if last_err:
        raise last_err


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------- Spot lekérők ---------

def fetch_spot_binance(symbol: str) -> float:
    """
    Binance spot ár (ticker price). Példa: symbol='SOLUSDT'
    """
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    j = _retry(_http_get_json, url)
    price = _safe_float(j.get("price"))
    if price is None:
        raise KeyError(f"Binance response doesn't have numeric price for {symbol}: {j}")
    return price


def fetch_spot_twelvedata(symbol: str) -> float:
    """
    TwelveData robusztus spot:
    1) /price → 'price'
    2) fallback /quote → 'close' | 'previous_close' | 'last' | 'open'
    """
    if not TD_API:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")

    # 1) /price
    url_price = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TD_API}"
    try:
        j = _retry(_http_get_json, url_price)
        price = _safe_float(j.get("price"))
        if price is not None:
            return price
    except Exception:
        pass  # esünk vissza /quote-ra

    # 2) /quote
    url_quote = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={TD_API}"
    j2 = _retry(_http_get_json, url_quote)
    for k in ("close", "previous_close", "last", "open"):
        v = _safe_float(j2.get(k))
        if v is not None:
            return v

    raise KeyError(f"No price-like field in TwelveData quote for '{symbol}': {j2}")


def fetch_quote_twelvedata(symbol: str) -> Dict[str, Any]:
    """
    TwelveData /quote – kiegészítő infókhoz (pl. percent_change).
    Hibatűrő, üres dictet ad vissza, ha nem sikerül.
    """
    if not TD_API:
        return {}
    try:
        url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={TD_API}"
        return _retry(_http_get_json, url)
    except Exception:
        return {}


# --------- Döntés (egyszerű szabály) ---------

def simple_decision(asset: str, spot: float, extra: Dict[str, Any]) -> str:
    """
    Minimál jelzés.
    - NSDQ100/GOLD_CFD: ha van percent_change a TwelveData quote-ból, abból képezünk irányt.
    - SOL: nincs előző adat → 'neutral'
    """
    if asset in ("NSDQ100", "GOLD_CFD"):
        pc = _safe_float(extra.get("percent_change"))
        if pc is None:
            return "neutral"
        if pc > 0.2:
            return "buy"
        if pc < -0.2:
            return "sell"
        return "neutral"

    # kriptó – nincs quote, marad semleges
    return "neutral"


# --------- Fő folyamat ---------

def generate_for_asset(asset: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lekérdezi a spot árat, készít egy egyszerű decision.json-t.
    Nem dob kivételt: hiba esetén 'ok': False + hibatxt kerül a JSON-ba.
    """
    out = {
        "ok": True,
        "asset": asset,
        "ts": int(time.time()),
        "spot": None,
        "source": cfg.get("source"),
        "note": cfg.get("note", ""),
        "signal": "neutral",
    }

    try:
        if cfg.get("source") == "binance":
            sym = cfg["binance_symbol"]
            spot = fetch_spot_binance(sym)
            out["spot"] = spot
            out["signal"] = simple_decision(asset, spot, {})
        elif cfg.get("source") == "twelvedata":
            sym = cfg["td_symbol"]
            spot = fetch_spot_twelvedata(sym)
            quote = fetch_quote_twelvedata(sym)
            out["spot"] = spot
            out["twelvedata_quote"] = quote
            out["signal"] = simple_decision(asset, spot, quote)
        else:
            raise ValueError(f"Unknown source for {asset}: {cfg.get('source')}")
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)

    # mentés
    path = OUT_DIR / asset / "decision.json"
    _write_json(path, out)
    print(f"[OK] írás: {path}")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {"ok": True, "generated": [], "errors": []}

    for asset, cfg in ASSETS.items():
        res = generate_for_asset(asset, cfg)
        summary["generated"].append({asset: res.get("ok", False)})
        if not res.get("ok"):
            summary["ok"] = False
            summary["errors"].append({asset: res.get("error", "unknown error")})

    # opcionális státusz a gyökérben
    _write_json(OUT_DIR / "status.json", summary)
    print("[KÉSZ] Fejlécek/decisions legenerálva.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market-feed / Trading.py
Egységes adatelőkészítő és jelzésképző script GitHub Pages publikáláshoz.

Eszközök:
- SOL           (kripto, spot: coinbase→coingecko→kraken fallback; OHLC: TwelveData)
- NSDQ100       (QQQ proxy; spot: Yahoo;       OHLC: TwelveData)
- GOLD_CFD      (XAU/USD;   spot: TwelveData;  OHLC: TwelveData)

Kimenetek (mindig a public/<ASSET>/ alatt):
- spot.json
- klines_5m.json
- klines_1h.json
- klines_4h.json
- signal.json   (EMA9/EMA21 alapján, 5 bar következetességi feltétellel)

Megjegyzés:
- A Pages workflow most már egy külön lépésben MERGE-li a 3 eszköz signal.json-ját
  analysis_summary.json-ná; ez a script NEM írja felül azt.
"""

import os
import json
import time
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

# ---- Konfiguráció / környezeti változók -------------------------------------

OUT_DIR = os.getenv("OUT_DIR", "public")

# TwelveData
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_PAUSE = float(os.getenv("TD_PAUSE", "1.5"))  # kímélő késleltetés hívások közt (sec)

# OHLC frissítési életkor-minimumok (másodperc)
TD_M5_MIN_AGE = int(os.getenv("TD_M5_MIN_AGE", "900"))     # 15 perc
TD_H1_MIN_AGE = int(os.getenv("TD_H1_MIN_AGE", "3600"))    # 1 óra
TD_H4_MIN_AGE = int(os.getenv("TD_H4_MIN_AGE", "14400"))   # 4 óra

# Eszközszűkítés: pl. "SOL,NSDQ100" vagy "SOL,GOLD_CFD" vagy "NSDQ100"
ASSET_ONLY = [a.strip() for a in os.getenv("ASSET_ONLY", "").split(",") if a.strip()]

# Debug log
DEBUG = os.getenv("DEBUG", "0") == "1"

# Teszt: kényszerített SOL fallback (Coinbase->hibát szimulálunk)
FORCE_SOL_FALLBACK = os.getenv("FORCE_SOL_FALLBACK", "0") == "1"


# ---- Közművek ----------------------------------------------------------------

def log(msg: str) -> None:
    if DEBUG:
        print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def now_utc_str() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def with_status_ok(payload: Dict[str, Any],
                   ok: bool,
                   error: Optional[str] = None) -> Dict[str, Any]:
    out = dict(payload)
    out["ok"] = bool(ok)
    out["retrieved_at_utc"] = now_utc_str()
    if error:
        out["error"] = str(error)
    return out


def age_seconds(path: str) -> Optional[int]:
    """Visszaadja, hogy a file mióta friss (sec)."""
    try:
        st = os.stat(path)
        return int(time.time() - st.st_mtime)
    except Exception:
        return None


# ---- Külső API-k: Coinbase / CoinGecko / Kraken / Yahoo / TwelveData --------

def coinbase_spot(product_id: str = "SOL-USD", timeout: int = 15) -> Dict[str, Any]:
    """
    Coinbase Exchange public REST: snapshot ticker.
    Doc: https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductticker
    """
    try:
        if FORCE_SOL_FALLBACK:
            raise RuntimeError("FORCE_SOL_FALLBACK=1 (test) → skip coinbase")

        url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "market-feed/1.0"})
        r.raise_for_status()
        j = r.json()
        # prefer "price" ha van, egyébként (bid+ask)/2
        price = None
        if "price" in j:
            try:
                price = float(j["price"])
            except Exception:
                price = None
        if price is None and "bid" in j and "ask" in j:
            try:
                price = (float(j["bid"]) + float(j["ask"])) / 2.0
            except Exception:
                price = None
        payload = {
            "asset": "SOL",
            "source": "coinbase",
            "price_usd": price,
            "raw": j
        }
        return with_status_ok(payload, price is not None)
    except Exception as e:
        return with_status_ok({"asset": "SOL", "source": "coinbase"}, False, f"spot error: {e}")


def coingecko_sol_spot(timeout: int = 15) -> Dict[str, Any]:
    """
    CoinGecko simple price (SOL→USD).
    Doc: https://docs.coingecko.com/reference/simple-price
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "solana",
            "vs_currencies": "usd",
            "include_last_updated_at": "true"
        }
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "market-feed/1.0"})
        r.raise_for_status()
        j = r.json()
        price = j.get("solana", {}).get("usd")
        payload = {
            "asset": "SOL",
            "source": "coingecko",
            "price_usd": float(price) if price is not None else None,
            "raw": j
        }
        return with_status_ok(payload, payload["price_usd"] is not None)
    except Exception as e:
        return with_status_ok({"asset": "SOL", "source": "coingecko"}, False, f"spot error: {e}")


def kraken_sol_spot(timeout: int = 15) -> Dict[str, Any]:
    """
    Kraken public Ticker (SOLUSD).
    Doc: https://docs.kraken.com/api/docs/rest-api/get-ticker-information/
    """
    try:
        url = "https://api.kraken.com/0/public/Ticker"
        r = requests.get(url, params={"pair": "SOLUSD"}, timeout=timeout,
                         headers={"User-Agent": "market-feed/1.0"})
        r.raise_for_status()
        data = r.json()
        res = data.get("result", {})
        if not res:
            return with_status_ok({"asset": "SOL", "source": "kraken"}, False, "no result")
        pair = next(iter(res.keys()))
        row = res[pair]
        price = None
        # last trade price: c[0]; ha az nincs, átlag ask/bid
        if isinstance(row.get("c"), list) and row["c"]:
            try:
                price = float(row["c"][0])
            except Exception:
                price = None
        if price is None:
            try:
                if isinstance(row.get("a"), list) and row["a"] and isinstance(row.get("b"), list) and row["b"]:
                    price = (float(row["a"][0]) + float(row["b"][0])) / 2.0
            except Exception:
                price = None
        payload = {
            "asset": "SOL",
            "source": "kraken",
            "price_usd": price,
            "raw": row
        }
        return with_status_ok(payload, price is not None)
    except Exception as e:
        return with_status_ok({"asset": "SOL", "source": "kraken"}, False, f"spot error: {e}")


def yahoo_spot(symbol: str, timeout: int = 15) -> Dict[str, Any]:
    """
    Yahoo finance quick quote (pl. QQQ). Egyszerű JSON.
    """
    try:
        url = "https://query1.finance.yahoo.com/v7/finance/quote"
        r = requests.get(url, params={"symbols": symbol}, timeout=timeout,
                         headers={"User-Agent": "market-feed/1.0"})
        r.raise_for_status()
        j = r.json()
        res = j.get("quoteResponse", {}).get("result", [])
        price = None
        if res:
            row = res[0]
            # prefer regularMarketPrice
            for key in ("regularMarketPrice", "postMarketPrice", "preMarketPrice"):
                v = row.get(key)
                if isinstance(v, (int, float)):
                    price = float(v)
                    break
        payload = {
            "asset": symbol,
            "source": "yahoo",
            "price_usd": price,
            "raw": j
        }
        return with_status_ok(payload, price is not None)
    except Exception as e:
        return with_status_ok({"asset": symbol, "source": "yahoo"}, False, f"spot error: {e}")


def twelvedata_ohlc(symbol: str, interval: str, timeout: int = 20, count: int = 400) -> Dict[str, Any]:
    """
    TwelveData time_series OHLC lekérés.
    """
    try:
        if not TWELVEDATA_API_KEY:
            raise RuntimeError("TWELVEDATA_API_KEY missing")

        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": str(count),
            "format": "JSON",
            "timezone": "UTC",
            "order": "asc",
            "apikey": TWELVEDATA_API_KEY
        }
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "market-feed/1.0"})
        r.raise_for_status()
        j = r.json()
        if "values" not in j:
            return with_status_ok({"asset": symbol, "interval": interval, "source": "twelvedata"}, False,
                                  f"bad response: {j.get('message') or 'no values'}")
        return with_status_ok({"asset": symbol, "interval": interval, "source": "twelvedata", "raw": j}, True)
    except Exception as e:
        return with_status_ok({"asset": symbol, "interval": interval, "source": "twelvedata"}, False, f"ohlc error: {e}")


# ---- Jelzésképzés (EMA-k) ----------------------------------------------------

def compute_ema(series: List[float], period: int) -> List[Optional[float]]:
    """
    Egyszerű EMA  (nem pandas; GitHub runneren pandas nélküli fallback esetére is jó).
    """
    if not series or period <= 1:
        return [None] * len(series)
    k = 2.0 / (period + 1.0)
    ema: List[Optional[float]] = []
    prev: Optional[float] = None
    for i, v in enumerate(series):
        if v is None:
            ema.append(prev)
            continue
        if prev is None:
            prev = v  # seed
        else:
            prev = v * k + prev * (1.0 - k)
        ema.append(prev)
    return ema


def last_n_consistent(cond: List[bool], n: int) -> bool:
    """Igaz-e az utolsó n elem mindegyike? Ha nincs elég adat, False."""
    if len(cond) < n:
        return False
    return all(cond[-n:])


def klines_to_closes(raw: Dict[str, Any]) -> List[Optional[float]]:
    """TwelveData JSON -> close lista (float v. None)."""
    try:
        vals = raw["raw"]["values"]
        closes: List[Optional[float]] = []
        for row in vals:
            try:
                closes.append(float(row["close"]))
            except Exception:
                closes.append(None)
        return closes
    except Exception:
        return []


def build_signal_from_ema(klines_5m: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Egyszerű trend-bias: EMA9 vs EMA21 az utolsó 5 gyertya alapján.
    """
    if not klines_5m or not klines_5m.get("ok"):
        return {
            "ok": False,
            "signal": "no entry",
            "reasons": ["missing 5m data"]
        }

    closes = klines_to_closes(klines_5m)
    if not closes:
        return {"ok": False, "signal": "no entry", "reasons": ["empty 5m data"]}

    ema9 = compute_ema(closes, 9)
    ema21 = compute_ema(closes, 21)
    gt = [False if (a is None or b is None) else (a > b) for a, b in zip(ema9, ema21)]
    lt = [False if (a is None or b is None) else (a < b) for a, b in zip(ema9, ema21)]

    if last_n_consistent(gt, 5):
        return {"ok": True, "signal": "uptrend", "reasons": ["ema9 > ema21 (5 bars)"]}
    if last_n_consistent(lt, 5):
        return {"ok": True, "signal": "downtrend", "reasons": ["ema9 < ema21 (5 bars)"]}
    return {"ok": True, "signal": "no entry", "reasons": ["no consistent ema bias"]}


# ---- Mentési segédek az OHLC-hez (életkor-gátlókkal) -------------------------

def should_refresh(path: str, min_age_sec: int) -> bool:
    a = age_seconds(path)
    if a is None:
        return True
    return a >= min_age_sec


def save_series_with_guard(path: str, payload: Dict[str, Any], ok: bool) -> None:
    """
    Csak akkor írjuk felül, ha OK az új adat, különben meghagyjuk a korábbit.
    """
    if ok:
        save_json(path, payload)
    else:
        # ha nincs korábbi, akkor is érdemes lementeni a hibát diagnosztikának
        if not os.path.exists(path):
            save_json(path, payload)


# ---- Eszköz-specifikus feldolgozók -------------------------------------------

def process_SOL() -> None:
    """
    SOL spot: coinbase → coingecko → kraken → (utolsó jó)
    OHLC: TwelveData (5m/1h/4h) életkor-gátlókkal.
    Jelzés: EMA9/EMA21 (5 bar).
    """
    asset = "SOL"
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    # --- SPOT (fallback lánc)
    spot = coinbase_spot("SOL-USD")
    if (not spot.get("ok")) or (spot.get("price_usd") is None):
        cg = coingecko_sol_spot()
        if cg.get("ok") and (cg.get("price_usd") is not None):
            spot = cg
        else:
            kk = kraken_sol_spot()
            if kk.get("ok") and (kk.get("price_usd") is not None):
                spot = kk
            else:
                prev = read_json(os.path.join(adir, "spot.json"))
                if prev:
                    spot = prev  # utolsó jó megtartása
    log(f"SOL spot source={spot.get('source')} price={spot.get('price_usd')}")
    save_json(os.path.join(adir, "spot.json"), spot)

    # --- OHLC (5m/1h/4h) – csak ha elég régi a file
    path_5m = os.path.join(adir, "klines_5m.json")
    if should_refresh(path_5m, TD_M5_MIN_AGE):
        time.sleep(TD_PAUSE)
        k5 = twelvedata_ohlc("SOL/USD", "5min")
        save_series_with_guard(path_5m, k5, k5.get("ok", False))

    path_1h = os.path.join(adir, "klines_1h.json")
    if should_refresh(path_1h, TD_H1_MIN_AGE):
        time.sleep(TD_PAUSE)
        k1 = twelvedata_ohlc("SOL/USD", "1h")
        save_series_with_guard(path_1h, k1, k1.get("ok", False))

    path_4h = os.path.join(adir, "klines_4h.json")
    if should_refresh(path_4h, TD_H4_MIN_AGE):
        time.sleep(TD_PAUSE)
        k4 = twelvedata_ohlc("SOL/USD", "4h")
        save_series_with_guard(path_4h, k4, k4.get("ok", False))

    # --- SIGNAL
    k5m = read_json(path_5m)
    sig = build_signal_from_ema(k5m)
    sig_out = {
        "asset": asset,
        "ok": bool(sig.get("ok")),
        "retrieved_at_utc": now_utc_str(),
        "signal": sig.get("signal", "no entry"),
        "reasons": sig.get("reasons", [])
    }
    save_json(os.path.join(adir, "signal.json"), sig_out)


def process_NSDQ100() -> None:
    """
    NSDQ100 (QQQ proxy)
    Spot: Yahoo (QQQ)
    OHLC: TwelveData (QQQ / NDQ100 index -> a projektben "NSDQ100" aliasra publikálunk)
    """
    asset = "NSDQ100"
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    # --- SPOT (Yahoo: QQQ)
    spot = yahoo_spot("QQQ")
    log(f"NSDQ100 spot source={spot.get('source')} price={spot.get('price_usd')}")
    save_json(os.path.join(adir, "spot.json"), spot)

    # --- OHLC
    # TwelveData szimbólum – ha az API kulcsod QQQ-ra ad idősort, használd QQQ-t.
    # (Alternatíva: ^NDX, NDX100 - TwelveData támogatástól függ.)
    path_5m = os.path.join(adir, "klines_5m.json")
    if should_refresh(path_5m, TD_M5_MIN_AGE):
        time.sleep(TD_PAUSE)
        k5 = twelvedata_ohlc("QQQ", "5min")
        save_series_with_guard(path_5m, k5, k5.get("ok", False))

    path_1h = os.path.join(adir, "klines_1h.json")
    if should_refresh(path_1h, TD_H1_MIN_AGE):
        time.sleep(TD_PAUSE)
        k1 = twelvedata_ohlc("QQQ", "1h")
        save_series_with_guard(path_1h, k1, k1.get("ok", False))

    path_4h = os.path.join(adir, "klines_4h.json")
    if should_refresh(path_4h, TD_H4_MIN_AGE):
        time.sleep(TD_PAUSE)
        k4 = twelvedata_ohlc("QQQ", "4h")
        save_series_with_guard(path_4h, k4, k4.get("ok", False))

    # --- SIGNAL
    k5m = read_json(path_5m)
    sig = build_signal_from_ema(k5m)
    sig_out = {
        "asset": asset,
        "ok": bool(sig.get("ok")),
        "retrieved_at_utc": now_utc_str(),
        "signal": sig.get("signal", "no entry"),
        "reasons": sig.get("reasons", [])
    }
    save_json(os.path.join(adir, "signal.json"), sig_out)


def process_GOLD_CFD() -> None:
    """
    GOLD_CFD (XAU/USD)
    Spot: TwelveData (XAU/USD) – egyszerű és egységes
    OHLC: TwelveData (5m/1h/4h)
    """
    asset = "GOLD_CFD"
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    # --- SPOT (TwelveData; ha kell, itt is tehetsz Yahoo fallbacket GC=F-re)
    spot_td = twelvedata_ohlc("XAU/USD", "1min", count=2)  # proxy a legfrissebb zárásra
    price = None
    if spot_td.get("ok"):
        try:
            vals = spot_td["raw"]["values"]
            if vals:
                price = float(vals[-1]["close"])
        except Exception:
            price = None
    spot = with_status_ok({"asset": asset, "source": "twelvedata", "price_usd": price, "raw": spot_td.get("raw")}, price is not None)
    log(f"GOLD_CFD spot source={spot.get('source')} price={spot.get('price_usd')}")
    save_json(os.path.join(adir, "spot.json"), spot)

    # --- OHLC
    path_5m = os.path.join(adir, "klines_5m.json")
    if should_refresh(path_5m, TD_M5_MIN_AGE):
        time.sleep(TD_PAUSE)
        k5 = twelvedata_ohlc("XAU/USD", "5min")
        save_series_with_guard(path_5m, k5, k5.get("ok", False))

    path_1h = os.path.join(adir, "klines_1h.json")
    if should_refresh(path_1h, TD_H1_MIN_AGE):
        time.sleep(TD_PAUSE)
        k1 = twelvedata_ohlc("XAU/USD", "1h")
        save_series_with_guard(path_1h, k1, k1.get("ok", False))

    path_4h = os.path.join(adir, "klines_4h.json")
    if should_refresh(path_4h, TD_H4_MIN_AGE):
        time.sleep(TD_PAUSE)
        k4 = twelvedata_ohlc("XAU/USD", "4h")
        save_series_with_guard(path_4h, k4, k4.get("ok", False))

    # --- SIGNAL
    k5m = read_json(path_5m)
    sig = build_signal_from_ema(k5m)
    sig_out = {
        "asset": asset,
        "ok": bool(sig.get("ok")),
        "retrieved_at_utc": now_utc_str(),
        "signal": sig.get("signal", "no entry"),
        "reasons": sig.get("reasons", [])
    }
    save_json(os.path.join(adir, "signal.json"), sig_out)


# ---- Belépési pont -----------------------------------------------------------

def main() -> None:
    ensure_dir(OUT_DIR)

    want = {"SOL", "NSDQ100", "GOLD_CFD"}
    if ASSET_ONLY:
        want = {a for a in want if a in set(ASSET_ONLY)}

    log(f"Start Trading.py (assets={','.join(sorted(want))})")

    if "SOL" in want:
        process_SOL()
    if "NSDQ100" in want:
        process_NSDQ100()
    if "GOLD_CFD" in want:
        process_GOLD_CFD()

    log("Done Trading.py")


if __name__ == "__main__":
    main()

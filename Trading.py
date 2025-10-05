#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market-feed / Trading.py
Egységes adatelőkészítő és jelzésképző script GitHub Pages publikáláshoz.

Eszközök:
- SOL           (kripto, spot: coinbase→coingecko→kraken fallback; OHLC: TwelveData)
+ NSDQ100       (QQQ proxy; spot: TwelveData;  OHLC: TwelveData)
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
import re
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

# --- Finnhub ---------------------------------------------------------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
# Forex szimbólum Finnhub-on (OANDA feeden: "OANDA:XAU_USD")
FINNHUB_XAU_SYMBOL = os.getenv("FINNHUB_XAU_SYMBOL", "OANDA:XAU_USD")

def _res_to_secs(res: str) -> int:
    """Finnhub resolution -> másodperc."""
    return {"1": 60, "5": 300, "15": 900, "30": 1800, "60": 3600,
            "240": 14400, "D": 86400}.get(res, 60)

def _label_from_res(res: str) -> str:
    return {"1": "1min", "5": "5min", "60": "1h", "240": "4h"}.get(res, f"{res}")

def finnhub_forex_candles(symbol: str, res: str, bars: int = 400, timeout: int = 20) -> Dict[str, Any]:
    """
    Finnhub FOREX gyertyák (XAU/USD stb.) → TD-kompatibilis 'raw.values' szerkezet.
    API: /forex/candle (symbol, resolution, from, to, token)
    """
    try:
        if not FINNHUB_API_KEY:
            raise RuntimeError("FINNHUB_API_KEY missing")

        now = int(time.time())
        span = _res_to_secs(res) * bars + 60
        frm = now - span
        url = "https://finnhub.io/api/v1/forex/candle"
        params = {"symbol": symbol, "resolution": res, "from": frm, "to": now, "token": FINNHUB_API_KEY}
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "market-feed/1.0"})
        r.raise_for_status()
        j = r.json()
        if j.get("s") != "ok" or not j.get("t") or not j.get("c"):
            return with_status_ok({"asset": symbol, "interval": _label_from_res(res), "source": "finnhub"}, False,
                                  f"bad response: {j.get('s')}")
        vals = []
        for ts, close in zip(j.get("t", []), j.get("c", [])):
            try:
                vals.append({"time": int(ts), "close": float(close)})
            except Exception:
                continue
        payload = {"asset": symbol, "interval": _label_from_res(res), "source": "finnhub", "raw": {"values": vals}}
        return with_status_ok(payload, True)
    except Exception as e:
        # ne szivárogjon token/URL a hibalogba
        msg = re.sub(r'(token=)[A-Za-z0-9_-]+', r'\1***', str(e))
        msg = re.sub(r'https?://\S+', '[redacted-url]', msg)
        safe_payload = {"asset": symbol, "interval": _label_from_res(res), "source": "finnhub"}
        return with_status_ok(safe_payload, False, f"ohlc error: {msg}")

def finnhub_fx_spot(symbol: str, timeout: int = 15) -> Dict[str, Any]:
    """
    'Spot' = az utolsó 1 perces gyertya záróára a Finnhub /forex/candle végpontjáról.
    Biztonság: a hibaszövegből elfedjük a tokeneket/URL-eket.
    """
    try:
        k1 = finnhub_forex_candles(symbol, "1", bars=2, timeout=timeout)  # REST candles
        price = None
        if k1.get("ok"):
            values = k1.get("raw", {}).get("values", [])
            if values:
                price = float(values[-1]["close"])
        payload = {
            "asset": "GOLD_CFD",
            "source": "finnhub",
            "price_usd": price,
            "raw": k1.get("raw")  # ez a valódi JSON, nem a hibaszöveg
        }
        return with_status_ok(payload, price is not None)
    except Exception as e:
        # ne szivárogjon API key / teljes URL
        msg = str(e)
        # nagyon egyszerű maszkolás: token param és query string kivágása
        msg = re.sub(r'(token=)[A-Za-z0-9_-]+', r'\1***', msg)
        msg = re.sub(r'https?://\S+', '[redacted-url]', msg)
        return with_status_ok(
            {"asset": "GOLD_CFD", "source": "finnhub"},
            False,
            f"spot error: {msg}"
        )


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

def twelvedata_quote(symbol: str, timeout: int = 15) -> Dict[str, Any]:
    """TwelveData /quote – megbízható spot forrás QQQ-hoz."""
    try:
        if not TWELVEDATA_API_KEY:
            raise RuntimeError("TWELVEDATA_API_KEY missing")
        url = "https://api.twelvedata.com/quote"
        params = {"symbol": symbol, "apikey": TWELVEDATA_API_KEY}
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "market-feed/1.0"})
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict) and j.get("status") != "error":
            price = float(j["price"])
            return with_status_ok({"asset": symbol, "source": "twelvedata:quote",
                                   "price_usd": price, "raw": j}, True)
        return with_status_ok({"asset": symbol, "source": "twelvedata:quote", "raw": j},
                              False, j.get("message") or "status=error")
    except Exception as e:
        return with_status_ok({"asset": symbol, "source": "twelvedata:quote"},
                              False, f"spot error: {e}")


def twelvedata_last_close(symbol: str, interval: str = "5min", timeout: int = 15) -> Optional[float]:
    """Utolsó záróár lehúzása 1 db gyertyából – fallback spothoz."""
    try:
        if not TWELVEDATA_API_KEY:
            return None
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol, "interval": interval, "outputsize": "1",
            "order": "desc", "timezone": "UTC", "apikey": TWELVEDATA_API_KEY
        }
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "market-feed/1.0"})
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "error" or not j.get("values"):
            return None
        return float(j["values"][0]["close"])
    except Exception:
        return None

def twelvedata_xauusd_spot(timeout: int = 15) -> Dict[str, Any]:
    """
    XAU/USD 'spot': az utolsó 1 perces gyertya záróára TwelveData time_series-ből.
    Fallbacknek direkt a 1m utolsó zárót kérjük (outputsize=1).
    """
    try:
        price = twelvedata_last_close("XAU/USD", "1min", timeout=timeout)
        payload = {
            "asset": "GOLD_CFD",
            "source": "twelvedata:1min_close",
            "price_usd": float(price) if price is not None else None,
            "raw": {"note": "derived from time_series last close (1min)"}
        }
        return with_status_ok(payload, payload["price_usd"] is not None)
    except Exception as e:
        return with_status_ok({"asset": "GOLD_CFD", "source": "twelvedata:1min_close"}, False, f"spot error: {e}")


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
    Spot: TwelveData (quote → 5m time_series → utolsó 5m close fallback)
    OHLC: TwelveData (QQQ)
    """
    asset = "NSDQ100"
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    # --- OHLC ELŐBB (legyen last_close fallbackhoz)
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

    # last_close a friss (vagy korábbi) 5m-ből
    last_close = None
    try:
        k5m_json = read_json(path_5m)
        if k5m_json and k5m_json.get("ok"):
            vals = k5m_json["raw"]["values"]
            if vals:
                last_close = float(vals[-1]["close"])
    except Exception:
        last_close = None

    # --- SPOT: TwelveData quote → time_series(5m,1) → last_close → előző spot
    spot_q = twelvedata_quote("QQQ")
    if spot_q.get("ok") and (spot_q.get("price_usd") is not None):
        spot = spot_q
    else:
        # próbáljuk a legutóbbi 5m gyertyát közvetlenül TD-ből
        close_5m = twelvedata_last_close("QQQ", "5min")
        if close_5m is not None:
            spot = with_status_ok({
                "asset": "QQQ", "source": "twelvedata:ts_5m",
                "price_usd": float(close_5m), "raw": {"note": "1 bar time_series"}
            }, True)
        elif last_close is not None:
            spot = with_status_ok({
                "asset": "QQQ", "source": "fallback:kline_5m_close",
                "price_usd": float(last_close)
            }, True)
        else:
            prev = read_json(os.path.join(adir, "spot.json"))
            spot = prev if prev else with_status_ok(
                {"asset": "QQQ", "source": "twelvedata"}, False,
                "no spot from TD and no kline fallback"
            )

    # a Pages-en az eszköz neve NSDQ100 legyen
    spot["asset"] = asset
    log(f"NSDQ100 spot source={spot.get('source')} price={spot.get('price_usd')}")
    save_json(os.path.join(adir, "spot.json"), spot)

    # --- SIGNAL (5m EMA)
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
    Spot: TwelveData 1m close → Finnhub 1m candles fallback → előző érték
    OHLC: TwelveData (5m / 1h / 4h) az eddigi életkor-gátlókkal
    """
    asset = "GOLD_CFD"
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    # --- SPOT (TD -> Finnhub fallback -> előző érték)
    spot = twelvedata_xauusd_spot()
    if (not spot.get("ok")) or (spot.get("price_usd") is None):
        fh = finnhub_fx_spot(FINNHUB_XAU_SYMBOL)  # pl. "OANDA:XAU_USD"
        if fh.get("ok") and fh.get("price_usd") is not None:
            spot = fh
        else:
            prev = read_json(os.path.join(adir, "spot.json"))
            if prev:
                spot = prev
    log(f"GOLD_CFD spot source={spot.get('source')} price={spot.get('price_usd')}")
    save_json(os.path.join(adir, "spot.json"), spot)

    # --- OHLC (5m/1h/4h) – TwelveData
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

    # --- SIGNAL (változatlan)
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






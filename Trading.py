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
  public/<ASSET>/signal.json  (5m EMA9–EMA21 7 bar szabály – előzetes jelzés)

Környezeti változók:
  TWELVEDATA_API_KEY = "<api key>"
  OUT_DIR            = "public" (alapértelmezés)
  TD_PAUSE           = "0.3"    (kímélő szünet hívások közt, sec)
"""

import os, json, time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

import requests

OUT_DIR  = os.getenv("OUT_DIR", "public")
API_KEY  = (os.getenv("TWELVEDATA_API_KEY") or "").strip()
TD_BASE  = "https://api.twelvedata.com"
TD_PAUSE = float(os.getenv("TD_PAUSE", "0.3"))

# Csak akkor hívunk API-t, ha a lokális fájl elég régi
STALE_SPOT = 120        
STALE_5M   = 300        
STALE_1H   = 3600       
STALE_4H   = 14400    

# ───────────────────────────────── ASSETS ────────────────────────────────
# GER40 helyett USOIL. A fő ticker a WTI/USD, de adunk több fallbackot.
ASSETS = {
    "SOL":      {"symbol": "SOL/USD",  "exchange": "Binance"},
    "NSDQ100":  {"symbol": "QQQ",      "exchange": None},
    "GOLD_CFD": {"symbol": "XAU/USD",  "exchange": None},
    "BNB":      {"symbol": "BNB/USD",  "exchange": "Binance"},

    # ÚJ: WTI kőolaj. A Twelve Data-n a hivatalos jelölés: WTI/USD.
    # Biztonság kedvéért próbálunk alternatív jelöléseket is.
    "USOIL": {
        "symbol": "WTI/USD",
        "exchange": None,
        "alt": ["USOIL", "WTICOUSD", "WTIUSD"]  # ha a fő nem menne, próbáljuk ezeket
    },
}

# ─────────────────────────────── Segédek ─────────────────────────────────

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

def file_age_seconds(path: str) -> Optional[int]:
    try:
        return int(max(0, time.time() - os.path.getmtime(path)))
    except FileNotFoundError:
        return None

def should_refresh(path: str, min_age_sec: int) -> bool:
    age = file_age_seconds(path)
    return (age is None) or (age >= min_age_sec)

def values_ok(raw: Dict[str, Any]) -> bool:
    vals = (raw or {}).get("values") or []
    return isinstance(vals, list) and len(vals) > 0

def guard_write_timeseries(path: str, td_payload: Dict[str, Any]) -> bool:
    """Csak nem üres time_series választ írunk ki."""
    raw = td_payload.get("raw") if isinstance(td_payload, dict) else None
    if not isinstance(raw, dict) or not values_ok(raw):
        return False
    save_json(path, raw);  return True

def guard_write_spot(path: str, spot_obj: Dict[str, Any]) -> bool:
    """Csak akkor írjuk a spotot, ha van érvényes ár."""
    price = spot_obj.get("price")
    if price is None:
        return False
    save_json(path, spot_obj);  return True


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
    if isinstance(data, dict) and data.get("status") == "error":
        # a hívó oldalon hibatűrően kezeljük
        raise RuntimeError(data.get("message", "TD error"))
    return data

def _iso_from_td_ts(ts: Any) -> Optional[str]:
    """TD időbélyeg ISO-UTC-re (kezeli a 'YYYY-MM-DD HH:MM:SS' és epoch sec formátumot)."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.isdigit()):
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            return dt.isoformat()
    except Exception:
        pass
    if isinstance(ts, str):
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            if "T" in ts:
                return ts
    return None

def td_time_series(symbol: str, interval: str, outputsize: int = 500,
                   exchange: Optional[str] = None, order: str = "desc") -> Dict[str, Any]:
    """Hibatűrő time_series (hiba esetén ok:false + üres values)."""
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
    try:
        j = td_get("time_series", **params)
        ok = bool(j.get("values"))
        return {
            "used_symbol": symbol,
            "asset": symbol,
            "interval": interval,
            "source": "twelvedata:time_series",
            "ok": ok,
            "retrieved_at_utc": now_utc(),
            "raw": j if ok else {"values": []},
        }
    except Exception as e:
        return {
            "used_symbol": symbol,
            "asset": symbol,
            "interval": interval,
            "source": "twelvedata:time_series",
            "ok": False,
            "retrieved_at_utc": now_utc(),
            "error": str(e),
            "raw": {"values": []},
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
    """Spot ár: quote → ha nincs, 5m utolsó close (time_series) fallback."""
    try:
        q = td_quote(symbol)
    except Exception as e:
        q = {"ok": False, "error": str(e)}

    price = q.get("price") if isinstance(q, dict) else None
    utc   = q.get("utc") if isinstance(q, dict) else None

    if price is None:
        try:
            px, ts = td_last_close(symbol, "5min", exchange)
        except Exception as e:
            px, ts = None, None
            q["error_fallback"] = str(e)
        price, utc = px, ts

    return {
        "asset": symbol,
        "source": "twelvedata:quote+series_fallback",
        "ok": price is not None,
        "retrieved_at_utc": now_utc(),
        "price": price,
        "price_usd": price,
        "utc": utc or now_utc(),
    }

# ─────────────────────── több-szimbólumos fallback ───────────────────────

def try_symbols(symbols: List[str], fetch_fn):
    """Próbálkozik több tickerrel sorban; az első ok:true visszatér. Máskülönben az utolsó eredményt adja."""
    last = None
    for s in symbols:
        try:
            r = fetch_fn(s)
            last = r
            if isinstance(r, dict) and r.get("ok"):
                return r
        except Exception as e:
            last = {"ok": False, "error": str(e)}
        time.sleep(TD_PAUSE * 0.5)
    return last or {"ok": False}

# ───────────────────── 5m EMA9–EMA21 (előzetes jelzés) ───────────────────

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
    if last_n_true(gt, 7):
        return {"ok": True, "signal": "uptrend", "reasons": ["ema9 > ema21 (5 bars)"]}
    if last_n_true(lt, 7):
        return {"ok": True, "signal": "downtrend", "reasons": ["ema9 < ema21 (5 bars)"]}
    return {"ok": True, "signal": "no entry", "reasons": ["no consistent ema bias"]}

# ─────────────────────────── Egy eszköz feldolgozása ──────────────────────

def process_asset(asset: str, cfg: Dict[str, Any]) -> None:
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    symbols = [cfg["symbol"]] + list(cfg.get("alt", []))
    exch = cfg.get("exchange")

    # --- elérési utak
    spot_path = os.path.join(adir, "spot.json")
    k5_path   = os.path.join(adir, "klines_5m.json")
    k1_path   = os.path.join(adir, "klines_1h.json")
    k4_path   = os.path.join(adir, "klines_4h.json")

    # 1) SPOT (quote → 5m close fallback) — csak ha elég régi
    if should_refresh(spot_path, STALE_SPOT):
        spot = try_symbols(symbols, lambda s: td_spot_with_fallback(s, exch))
        guard_write_spot(spot_path, spot if isinstance(spot, dict) else {"ok": False})
        time.sleep(TD_PAUSE)

    # 2) OHLC: 5m / 1h / 4h — stale + guard
    def ts(s: str, iv: str):
        return td_time_series(s, iv, 500, exch, "desc")

    if should_refresh(k5_path, STALE_5M):
        k5 = try_symbols(symbols, lambda s: ts(s, "5min"))
        guard_write_timeseries(k5_path, k5)
        time.sleep(TD_PAUSE)

    if should_refresh(k1_path, STALE_1H):
        k1 = try_symbols(symbols, lambda s: ts(s, "1h"))
        guard_write_timeseries(k1_path, k1)
        time.sleep(TD_PAUSE)

    if should_refresh(k4_path, STALE_4H):
        k4 = try_symbols(symbols, lambda s: ts(s, "4h"))
        guard_write_timeseries(k4_path, k4)
        time.sleep(TD_PAUSE)

    # 3) 5m előzetes jelzés – csak akkor írjuk, ha van 5m adat
    k5_raw = (json.load(open(k5_path, "r", encoding="utf-8")) if os.path.exists(k5_path) else {"values": []})
    if values_ok(k5_raw):
        # Készítünk egy "ál-payloadot", hogy a meglévő signal_from_5m() fel tudja dolgozni
        sig = signal_from_5m({"ok": True, "raw": k5_raw})
        spot_now = json.load(open(spot_path, "r", encoding="utf-8")) if os.path.exists(spot_path) else {}
        save_json(
            os.path.join(adir, "signal.json"),
            {
                "asset": asset,
                "ok": bool(sig.get("ok")),
                "retrieved_at_utc": now_utc(),
                "signal": sig.get("signal", "no entry"),
                "reasons": sig.get("reasons", []),
                "probability": 0,  # előzetes; a véglegeset az analysis.py adja
                "spot": {"price": spot_now.get("price"), "utc": spot_now.get("utc")},
            },
        )

# ─────────────────────────────── main ─────────────────────────────────────

def main():
    if not API_KEY:
        raise SystemExit("TWELVEDATA_API_KEY hiányzik (GitHub Secret).")
    ensure_dir(OUT_DIR)
    for a, cfg in ASSETS.items():
        try:
            process_asset(a, cfg)
        except Exception as e:
            # Ne álljon meg a pipeline – írjunk minimális signal/spot-ot.
            adir = os.path.join(OUT_DIR, a)
            ensure_dir(adir)
            now = now_utc()
            save_json(os.path.join(adir, "spot.json"), {
                "asset": a, "ok": False, "retrieved_at_utc": now, "price": None, "price_usd": None, "utc": now
            })
            save_json(os.path.join(adir, "signal.json"), {
                "asset": a, "ok": False, "retrieved_at_utc": now,
                "signal": "no entry", "probability": 0, "reasons": [f"fetch error: {e}"],
                "spot": {"price": None, "utc": now}
            })

if __name__ == "__main__":
    main()


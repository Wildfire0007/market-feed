#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD-only market-feed / Trading.py  —  FINOMHANGOLT (2025-10-09)
Minden adat CSAK a Twelve Data REST API-ból jön.

Kimenetek (guard-olt mentés):
  public/<ASSET>/spot.json
  public/<ASSET>/klines_5m.json
  public/<ASSET>/klines_1h.json
  public/<ASSET>/klines_4h.json
  public/<ASSET>/signal.json   (5m EMA9–EMA21 7 bar — előzetes jelzés, csak ha k5m OK)
  public/status.json           (összefoglaló: ok/age/refresh/saved per asset)

Stale-check (kor-alapú frissítés; csak ha elég régi a lokális fájl):
  5m:   ≥ 900 s
  1h:   ≥ 3600 s
  4h:   ≥ 14400 s
  spot: ≥ 120 s

Guard-elv:
  - Ha a TD-hívás hibás vagy üres ("values": []), NEM írjuk felül a korábbi jó fájlt.
  - Csak sikeres, nem üres válasszal történik felülírás.
  - Előzetes signal.json-t csak k5m OK esetén írunk (különben nem nyúlunk hozzá).
"""

import os, json, time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

import requests

OUT_DIR  = os.getenv("OUT_DIR", "public")
API_KEY  = os.environ["TWELVEDATA_API_KEY"].strip()
TD_BASE  = "https://api.twelvedata.com"
TD_PAUSE = float(os.getenv("TD_PAUSE", "0.3"))

# ───────────────────────────────── ASSETS ────────────────────────────────
ASSETS = {
    "SOL":      {"symbol": "SOL/USD",  "exchange": "Binance"},
    "NSDQ100":  {"symbol": "QQQ",      "exchange": None},
    "GOLD_CFD": {"symbol": "XAU/USD",  "exchange": None},
    "BNB":      {"symbol": "BNB/USD",  "exchange": "Binance"},

    # WTI (USOIL) — több alternatív tickerrel
    "USOIL": {
        "symbol": "WTI/USD",
        "exchange": None,
        "alt": ["USOIL", "WTICOUSD", "WTIUSD"]
    },
}

# ─────────────────────────────── Stale policy ────────────────────────────
STALE_SPOT = 120        # sec
STALE_5M   = 900        # sec
STALE_1H   = 3600       # sec
STALE_4H   = 14400      # sec

# ─────────────────────────────── Segédek ─────────────────────────────────
def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def file_age_seconds(path: str) -> Optional[int]:
    try:
        mtime = os.path.getmtime(path)
        return int(max(0, time.time() - mtime))
    except FileNotFoundError:
        return None

def should_refresh(path: str, min_age_sec: int) -> bool:
    age = file_age_seconds(path)
    return (age is None) or (age >= min_age_sec)

def values_ok(raw: Dict[str, Any]) -> bool:
    try:
        vals = raw.get("values") or []
        return isinstance(vals, list) and len(vals) > 0
    except Exception:
        return False

def guard_write_timeseries(path: str, td_payload: Dict[str, Any]) -> bool:
    """
    Csak akkor írjuk felül a klines fájlt, ha a TD válaszban van legalább 1 value.
    Visszatérés: True ha írtunk, False ha nem.
    """
    raw = td_payload.get("raw") if isinstance(td_payload, dict) else None
    if not isinstance(raw, dict) or not values_ok(raw):
        return False
    save_json_atomic(path, raw)
    return True

def guard_write_spot(path: str, spot_obj: Dict[str, Any]) -> bool:
    """
    Csak akkor írjuk felül a spot fájlt, ha van price (nem None).
    """
    price = spot_obj.get("price")
    if price is None:
        return False
    save_json_atomic(path, spot_obj)
    return True

# ───────────────────────────── Twelve Data hívók ─────────────────────────
def td_get(path: str, **params) -> Dict[str, Any]:
    params["apikey"] = API_KEY
    r = requests.get(
        f"{TD_BASE}/{path}",
        params=params,
        timeout=30,
        headers={"User-Agent": "market-feed/td-only/1.1"},
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
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
            if isinstance(q, dict):
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

def closes_from_ts(payload_raw: Dict[str, Any]) -> List[Optional[float]]:
    try:
        vals = payload_raw["values"]
        out: List[Optional[float]] = []
        for row in vals:
            try:
                out.append(float(row["close"]))
            except Exception:
                out.append(None)
        return out
    except Exception:
        return []

def pre_signal_from_k5m(raw_5m: Dict[str, Any]) -> Dict[str, Any]:
    # Csak akkor hívjuk, ha a raw_5m nem üres.
    closes = closes_from_ts(raw_5m)
    if not closes:
        return {"ok": False, "signal": "no entry", "reasons": ["empty 5m data"]}
    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    gt = [False if (a is None or b is None) else (a > b) for a, b in zip(e9, e21)]
    lt = [False if (a is None or b is None) else (a < b) for a, b in zip(e9, e21)]
    # Momentum új szabály: 7 bar
    if last_n_true(gt, 7):
        return {"ok": True, "signal": "uptrend", "reasons": ["ema9 > ema21 (7 bars)"]}
    if last_n_true(lt, 7):
        return {"ok": True, "signal": "downtrend", "reasons": ["ema9 < ema21 (7 bars)"]}
    return {"ok": True, "signal": "no entry", "reasons": ["no consistent ema bias"]}

# ─────────────────────────── Egy eszköz feldolgozása ──────────────────────
def process_asset(asset: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    symbols = [cfg["symbol"]] + list(cfg.get("alt", []))
    exch = cfg.get("exchange")

    status_entry: Dict[str, Any] = {
        "asset": asset,
        "paths": {},
        "refreshed": {},
        "saved": {},
        "ok": {},
        "ages_sec": {},
        "used_symbol": None,
        "run_utc": now_utc(),
    }

    # ── 1) SPOT ───────────────────────────────────────────────────────────
    spot_path = os.path.join(adir, "spot.json")
    status_entry["ages_sec"]["spot"] = file_age_seconds(spot_path)
    need_spot = should_refresh(spot_path, STALE_SPOT)

    if need_spot:
        r_spot = try_symbols(symbols, lambda s: td_spot_with_fallback(s, exch))
        status_entry["used_symbol"] = r_spot.get("asset")
        status_entry["refreshed"]["spot"] = True
        saved = guard_write_spot(spot_path, r_spot if isinstance(r_spot, dict) else {"ok": False})
        status_entry["saved"]["spot"] = saved
    else:
        status_entry["refreshed"]["spot"] = False
        status_entry["saved"]["spot"] = False

    spot_now = load_json(spot_path) or {}
    status_entry["ok"]["spot"] = bool(spot_now.get("price") is not None)
    time.sleep(TD_PAUSE)

    # ── 2) KLINES 5m / 1h / 4h  (csak ha elég régi a fájl) ────────────────
    def fetch_ts(iv: str) -> Dict[str, Any]:
        return try_symbols(symbols, lambda s: td_time_series(s, iv, 500, exch, "desc"))

    # 5m
    k5_path = os.path.join(adir, "klines_5m.json")
    status_entry["ages_sec"]["k5m"] = file_age_seconds(k5_path)
    need_k5 = should_refresh(k5_path, STALE_5M)
    if need_k5:
        r5 = fetch_ts("5min")
        status_entry["refreshed"]["k5m"] = True
        status_entry["saved"]["k5m"] = guard_write_timeseries(k5_path, r5)
    else:
        status_entry["refreshed"]["k5m"] = False
        status_entry["saved"]["k5m"] = False
    k5_raw = load_json(k5_path) or {"values": []}
    status_entry["ok"]["k5m"] = values_ok(k5_raw)
    time.sleep(TD_PAUSE)

    # 1h
    k1_path = os.path.join(adir, "klines_1h.json")
    status_entry["ages_sec"]["k1h"] = file_age_seconds(k1_path)
    need_k1 = should_refresh(k1_path, STALE_1H)
    if need_k1:
        r1 = fetch_ts("1h")
        status_entry["refreshed"]["k1h"] = True
        status_entry["saved"]["k1h"] = guard_write_timeseries(k1_path, r1)
    else:
        status_entry["refreshed"]["k1h"] = False
        status_entry["saved"]["k1h"] = False
    k1_raw = load_json(k1_path) or {"values": []}
    status_entry["ok"]["k1h"] = values_ok(k1_raw)
    time.sleep(TD_PAUSE)

    # 4h
    k4_path = os.path.join(adir, "klines_4h.json")
    status_entry["ages_sec"]["k4h"] = file_age_seconds(k4_path)
    need_k4 = should_refresh(k4_path, STALE_4H)
    if need_k4:
        r4 = fetch_ts("4h")
        status_entry["refreshed"]["k4h"] = True
        status_entry["saved"]["k4h"] = guard_write_timeseries(k4_path, r4)
    else:
        status_entry["refreshed"]["k4h"] = False
        status_entry["saved"]["k4h"] = False
    k4_raw = load_json(k4_path) or {"values": []}
    status_entry["ok"]["k4h"] = values_ok(k4_raw)
    time.sleep(TD_PAUSE)

    # ── 3) Előzetes 5m EMA jelzés (csak ha k5m OK) ────────────────────────
    sig_path = os.path.join(adir, "signal.json")
    if status_entry["ok"]["k5m"]:
        pre = pre_signal_from_k5m(k5_raw)
        save_json_atomic(sig_path, {
            "asset": asset,
            "ok": bool(pre.get("ok")),
            "retrieved_at_utc": now_utc(),
            "signal": pre.get("signal", "no entry"),
            "reasons": pre.get("reasons", []),
            "probability": 0,   # előzetes; a véglegeset az analysis.py adja
            "spot": {"price": spot_now.get("price"), "utc": spot_now.get("utc")},
        })
        status_entry["saved"]["signal"] = True
    else:
        # nem nyúlunk a meglévő signal.json-hoz (guard)
        status_entry["saved"]["signal"] = False

    # fájl elérési utak a státuszba
    status_entry["paths"] = {
        "spot":   os.path.relpath(spot_path, OUT_DIR),
        "k5m":    os.path.relpath(k5_path, OUT_DIR),
        "k1h":    os.path.relpath(k1_path, OUT_DIR),
        "k4h":    os.path.relpath(k4_path, OUT_DIR),
        "signal": os.path.relpath(sig_path, OUT_DIR),
    }
    return status_entry

# ─────────────────────────────── main ─────────────────────────────────────
def main():
    if not API_KEY:
        raise SystemExit("TWELVEDATA_API_KEY hiányzik (GitHub Secret).")
    ensure_dir(OUT_DIR)

    status: Dict[str, Any] = {
        "ok": True,
        "generated_utc": now_utc(),
        "td_base": TD_BASE,
        "assets": {},
        "notes": {
            "stale_policy_sec": {"spot": STALE_SPOT, "k5m": STALE_5M, "k1h": STALE_1H, "k4h": STALE_4H},
            "guard": "no overwrite on error/empty values",
            "pre_signal_rule": "5m ema9×ema21 7 bars",
        }
    }

    for a, cfg in ASSETS.items():
        try:
            st = process_asset(a, cfg)
            status["assets"][a] = st
        except Exception as e:
            status["assets"][a] = {
                "asset": a,
                "error": str(e),
                "run_utc": now_utc(),
            }
            # Guard: hiba esetén sem írunk üres fájlokat.

    # status.json mentése
    save_json_atomic(os.path.join(OUT_DIR, "status.json"), status)

if __name__ == "__main__":
    main()


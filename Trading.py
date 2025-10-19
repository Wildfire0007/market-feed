#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD-only market-feed / Trading.py
Minden adat CSAK a Twelve Data REST API-ból jön.

Kimenetek:
  public/<ASSET>/spot.json
  public/<ASSET>/klines_1m.json
  public/<ASSET>/klines_5m.json
  public/<ASSET>/klines_1h.json
  public/<ASSET>/klines_4h.json
  public/<ASSET>/signal.json  (5m EMA9–EMA21 5 bar szabály – előzetes jelzés)

Környezeti változók:
  TWELVEDATA_API_KEY = "<api key>"
  OUT_DIR            = "public" (alapértelmezés)
  TD_PAUSE           = "0.3"    (kímélő szünet hívások közt, sec)
  TD_MAX_RETRIES     = "4"      (újrapróbálkozások száma)
  TD_BACKOFF_BASE    = "0.3"    (exponenciális visszavárás alapja)
  TD_BACKOFF_MAX     = "8"      (exponenciális visszavárás plafonja, sec)
  TD_REALTIME_SPOT   = "0/1"    (bekapcsolja a valós idejű spot-frissítést)
  TD_REALTIME_INTERVAL = "5"    (realtime ciklus közötti szünet sec)
  TD_REALTIME_DURATION = "60"   (realtime poll futási ideje sec)
  TD_REALTIME_ASSETS = ""       (komma-szeparált lista, üres = mind)
"""

import os, json, time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

import requests

try:
    import websocket  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    websocket = None

OUT_DIR  = os.getenv("OUT_DIR", "public")
API_KEY_RAW = os.getenv("TWELVEDATA_API_KEY", "")
API_KEY  = API_KEY_RAW.strip() if API_KEY_RAW else ""
TD_BASE  = "https://api.twelvedata.com"
TD_PAUSE = float(os.getenv("TD_PAUSE", "0.3"))
TD_PAUSE_MIN = float(os.getenv("TD_PAUSE_MIN", str(max(TD_PAUSE * 0.5, 0.1))))
TD_PAUSE_MAX = float(os.getenv("TD_PAUSE_MAX", str(max(TD_PAUSE * 6, 3.0))))
TD_MAX_RETRIES = int(os.getenv("TD_MAX_RETRIES", "4"))
TD_BACKOFF_BASE = float(os.getenv("TD_BACKOFF_BASE", str(max(TD_PAUSE, 0.3))))
TD_BACKOFF_MAX = float(os.getenv("TD_BACKOFF_MAX", "8.0"))
REALTIME_FLAG = os.getenv("TD_REALTIME_SPOT", "").lower() in {"1", "true", "yes", "on"}
REALTIME_INTERVAL = float(os.getenv("TD_REALTIME_INTERVAL", "5"))
REALTIME_DURATION = float(os.getenv("TD_REALTIME_DURATION", "60"))
REALTIME_ASSETS_ENV = os.getenv("TD_REALTIME_ASSETS", "")
REALTIME_WS_URL = os.getenv("TD_REALTIME_WS_URL", "")
REALTIME_WS_SUBSCRIBE = os.getenv("TD_REALTIME_WS_SUBSCRIBE", "")
REALTIME_WS_PRICE_FIELD = os.getenv("TD_REALTIME_WS_PRICE_FIELD", "price")
REALTIME_WS_TS_FIELD = os.getenv("TD_REALTIME_WS_TS_FIELD", "timestamp")
REALTIME_WS_MAX_SAMPLES = int(os.getenv("TD_REALTIME_WS_MAX_SAMPLES", "120"))
REALTIME_WS_TIMEOUT = float(os.getenv("TD_REALTIME_WS_TIMEOUT", str(max(REALTIME_DURATION, 30.0))))
REALTIME_WS_ENABLED = bool(REALTIME_WS_URL and websocket is not None)


class AdaptiveRateLimiter:
    def __init__(self, base_pause: float, min_pause: float, max_pause: float) -> None:
        self.base = max(base_pause, 0.0)
        self.min_pause = max(min_pause, 0.0)
        self.max_pause = max(max_pause, self.min_pause if self.min_pause else 0.0)
        self.current = max(self.base, self.min_pause)

    @property
    def current_delay(self) -> float:
        return self.current

    def wait(self) -> None:
        if self.current > 0:
            time.sleep(self.current)

    def record_success(self) -> None:
        if self.current <= 0:
            return
        self.current = max(self.min_pause, self.current * 0.85)

    def record_failure(self, throttled: bool = False) -> None:
        growth = 1.6 if throttled else 1.3
        baseline = self.base if self.base > 0 else 0.3
        self.current = min(self.max_pause, max(self.current, baseline) * growth)

    def backoff_seconds(self, attempt: int, retry_after: Optional[float] = None) -> float:
        baseline = self.base if self.base > 0 else TD_BACKOFF_BASE
        wait = baseline * (2 ** max(attempt - 1, 0))
        if retry_after is not None:
            wait = max(wait, retry_after)
        return min(wait, TD_BACKOFF_MAX)


TD_RATE_LIMITER = AdaptiveRateLimiter(TD_PAUSE, TD_PAUSE_MIN, TD_PAUSE_MAX)

# ───────────────────────────────── ASSETS ────────────────────────────────␊
# GER40 helyett USOIL. A fő ticker a WTI/USD, de adunk több fallbackot.␊
ASSETS = {
    "EURUSD":   {"symbol": "EUR/USD", "exchange": "FX", "alt": ["EURUSD", "EURUSD:CUR"]},
    "USDJPY":   {"symbol": "USD/JPY",  "exchange": "FX", "alt": ["USDJPY", "USDJPY:CUR"]},
    "GOLD_CFD": {"symbol": "XAU/USD",  "exchange": None},

    # ÚJ: WTI kőolaj. A Twelve Data-n a hivatalos jelölés: WTI/USD.
    # Biztonság kedvéért próbálunk alternatív jelöléseket is.
    "USOIL": {
        "symbol": "WTI/USD",
        "exchange": None,
        "alt": ["USOIL", "WTICOUSD", "WTIUSD"]  # ha a fő nem menne, próbáljuk ezeket
    },

    # Egyedi részvény és ETF kiterjesztések
    "NVDA": {
        "symbol": "NVDA",
        "exchange": "NASDAQ",
        "alt": ["NVDA", "NVDA:US"]
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

def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        return None


def td_get(path: str, **params) -> Dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY hiányzik")
    params["apikey"] = API_KEY
    last_error: Optional[Exception] = None
    last_status: Optional[int] = None
    for attempt in range(1, TD_MAX_RETRIES + 1):
        TD_RATE_LIMITER.wait()
        response: Optional[requests.Response] = None
        try:
            response = requests.get(
                f"{TD_BASE}/{path}",
                params=params,
                timeout=30,
                headers={"User-Agent": "market-feed/td-only/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("status") == "error":
                message = data.get("message", "TD error")
                err = RuntimeError(message)
                last_error = err
                last_status = response.status_code if response is not None else None
                throttled = "limit" in message.lower()
                TD_RATE_LIMITER.record_failure(throttled=throttled)
                if attempt == TD_MAX_RETRIES:
                    raise err
            else:
                TD_RATE_LIMITER.record_success()
                return data
        except requests.HTTPError as exc:
            last_error = exc
            last_status = exc.response.status_code if exc.response else None
            TD_RATE_LIMITER.record_failure(throttled=last_status in {429, 503})
            if attempt == TD_MAX_RETRIES:
                raise
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            last_status = None
            TD_RATE_LIMITER.record_failure()
            if attempt == TD_MAX_RETRIES:
                raise RuntimeError(f"TD request failed: {exc}") from exc

        retry_after: Optional[float] = None
        if response is not None:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
        backoff = TD_RATE_LIMITER.backoff_seconds(attempt, retry_after=retry_after)
        time.sleep(backoff)

    status_str = f" status={last_status}" if last_status else ""
    raise RuntimeError(f"TD request failed after {TD_MAX_RETRIES} attempts{status_str}: {last_error}")

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


def _parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
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

def td_quote(symbol: str, exchange: Optional[str] = None) -> Dict[str, Any]:
    params = {"symbol": symbol, "timezone": "UTC"}
    if exchange:
        params["exchange"] = exchange
    j = td_get("quote", **params)
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
        q = td_quote(symbol, exchange)
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


def _extract_field(payload: Any, path: str) -> Optional[Any]:
    if not path:
        return None
    current = payload
    for segment in path.split('.'):
        if segment == "":
            continue
        while '[' in segment:
            bracket = segment.find('[')
            attr = segment[:bracket]
            remainder = segment[bracket + 1:]
            if attr:
                if not isinstance(current, dict):
                    return None
                current = current.get(attr)
            if not isinstance(current, (list, tuple)):
                return None
            end = remainder.find(']')
            if end == -1:
                return None
            index_str = remainder[:end]
            try:
                index = int(index_str)
            except ValueError:
                return None
            if index >= len(current) or index < -len(current):
                return None
            current = current[index]
            segment = remainder[end + 1:]
            if not segment:
                break
        if segment:
            if not isinstance(current, dict):
                return None
            current = current.get(segment)
        if current is None:
            return None
    return current


def _collect_http_frames(
    symbol_cycle: List[Tuple[str, Optional[str]]],
    deadline: float,
    interval: float,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    while time.time() < deadline:
        for symbol, exchange in symbol_cycle:
            try:
                quote = td_quote(symbol, exchange)
            except Exception:
                continue
            price = quote.get("price")
            if price is None:
                continue
            utc_ts = quote.get("utc") or quote.get("retrieved_at_utc") or now_utc()
            retrieved = now_utc()
            latency: Optional[float] = None
            parsed_spot = _parse_iso_utc(utc_ts)
            parsed_retrieved = _parse_iso_utc(retrieved)
            if parsed_spot and parsed_retrieved:
                latency = max(0.0, (parsed_retrieved - parsed_spot).total_seconds())
            frames.append(
                {
                    "price": price,
                    "utc": utc_ts,
                    "retrieved_at_utc": retrieved,
                    "symbol": symbol,
                    "exchange": exchange,
                    "latency_seconds": latency,
                }
            )
            break
        time.sleep(interval)
    return frames


def _collect_ws_frames(
    asset: str,
    symbol_cycle: List[Tuple[str, Optional[str]]],
    deadline: float,
) -> List[Dict[str, Any]]:
    if not REALTIME_WS_ENABLED:
        return []
    symbol = symbol_cycle[0][0] if symbol_cycle else asset
    exchange = symbol_cycle[0][1] if symbol_cycle else None
    frames: List[Dict[str, Any]] = []
    timeout = max(1.0, min(REALTIME_WS_TIMEOUT, 300.0))
    try:
        ws = websocket.create_connection(REALTIME_WS_URL, timeout=timeout)
    except Exception:
        return []
    try:
        subscribe = REALTIME_WS_SUBSCRIBE.strip()
        if subscribe:
            try:
                ws.send(subscribe.format(asset=asset, symbol=symbol, exchange=exchange or ""))
            except Exception:
                pass
        ws.settimeout(max(1.0, min(REALTIME_INTERVAL, 15.0)))
        while time.time() < deadline and len(frames) < REALTIME_WS_MAX_SAMPLES:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                break
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            price_val = _extract_field(payload, REALTIME_WS_PRICE_FIELD)
            if price_val is None:
                continue
            try:
                price = float(price_val)
            except (TypeError, ValueError):
                continue
            ts_val = _extract_field(payload, REALTIME_WS_TS_FIELD)
            utc_ts: Optional[str] = None
            if ts_val is not None:
                if isinstance(ts_val, (int, float)):
                    try:
                        utc_ts = datetime.fromtimestamp(float(ts_val), timezone.utc).replace(microsecond=0).isoformat()
                    except Exception:
                        utc_ts = None
                else:
                    utc_ts = str(ts_val)
            retrieved = now_utc()
            latency: Optional[float] = None
            parsed_spot = _parse_iso_utc(utc_ts) if utc_ts else None
            parsed_retrieved = _parse_iso_utc(retrieved)
            if parsed_spot and parsed_retrieved:
                latency = max(0.0, (parsed_retrieved - parsed_spot).total_seconds())
            frames.append(
                {
                    "price": price,
                    "utc": utc_ts or retrieved,
                    "retrieved_at_utc": retrieved,
                    "symbol": symbol,
                    "exchange": exchange,
                    "latency_seconds": latency,
                    "raw": payload,
                }
            )
    finally:
        try:
            ws.close()
        except Exception:
            pass
    return frames


def collect_realtime_spot(
    asset: str,
    attempts: List[Tuple[str, Optional[str]]],
    out_dir: str,
) -> None:
    if not REALTIME_FLAG:
        return
    allowed_assets = {
        a.strip().upper()
        for a in REALTIME_ASSETS_ENV.split(",")
        if a.strip()
    }
    if allowed_assets and asset.upper() not in allowed_assets:
        return

    ensure_dir(out_dir)
    symbol_cycle = list(attempts) if attempts else []
    if not symbol_cycle:
        return

    interval = max(0.5, REALTIME_INTERVAL)
    frames: List[Dict[str, Any]] = []
    transport: Optional[str] = None

    deadline = time.time() + max(REALTIME_DURATION, REALTIME_INTERVAL)
    if REALTIME_WS_ENABLED:
        frames = _collect_ws_frames(asset, symbol_cycle, deadline)
        if frames:
            transport = "websocket"

    if not frames:
        deadline_http = time.time() + max(REALTIME_DURATION, REALTIME_INTERVAL)
        frames = _collect_http_frames(symbol_cycle, deadline_http, interval)
        if frames:
            transport = "http"

    if not frames:
        return

    prices = [float(frame["price"]) for frame in frames if frame.get("price") is not None]
    latencies = [float(lat) for lat in (frame.get("latency_seconds") for frame in frames) if lat is not None]
    stats: Dict[str, Any] = {
        "samples": len(frames),
        "min_price": min(prices) if prices else None,
        "max_price": max(prices) if prices else None,
        "mean_price": sum(prices) / len(prices) if prices else None,
        "latency_avg": sum(latencies) / len(latencies) if latencies else None,
        "latency_max": max(latencies) if latencies else None,
    }
    if latencies:
        lat_sorted = sorted(latencies)
        stats["latency_median"] = lat_sorted[len(lat_sorted) // 2]
        stats["latency_latest"] = frames[-1].get("latency_seconds")

    last_frame = frames[-1]
    payload = {
        "asset": asset,
        "ok": True,
        "transport": transport or ("websocket" if REALTIME_WS_ENABLED else "http"),
        "retrieved_at_utc": last_frame.get("retrieved_at_utc") or now_utc(),
        "price": last_frame.get("price"),
        "utc": last_frame.get("utc"),
        "frames": frames,
        "statistics": stats,
        "source": "websocket" if transport == "websocket" else "rest",
    }
    if transport == "websocket" and isinstance(last_frame.get("raw"), dict):
        payload["raw_last_frame"] = last_frame["raw"]
    save_json(os.path.join(out_dir, "spot_realtime.json"), payload)

# ─────────────────────── több-szimbólumos fallback ───────────────────────

def _normalize_symbol_attempts(cfg: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    base_symbol = cfg["symbol"]
    base_exchange = cfg.get("exchange")
    attempts: List[Tuple[str, Optional[str]]] = []

    def push(symbol: Optional[str], exchange: Optional[str]) -> None:
        if symbol:
            attempts.append((symbol, exchange))

    push(base_symbol, base_exchange)
    for alt in cfg.get("alt", []):
        if isinstance(alt, str):
            push(alt, base_exchange)
        elif isinstance(alt, dict):
            push(alt.get("symbol", base_symbol), alt.get("exchange", base_exchange))
        elif isinstance(alt, (list, tuple)) and alt:
            symbol = alt[0]
            exchange = alt[1] if len(alt) > 1 else base_exchange
            push(symbol, exchange)

    seen = set()
    unique: List[Tuple[str, Optional[str]]] = []
    for sym, exch in attempts:
        key = (sym, exch)
        if key in seen:
            continue
        seen.add(key)
        unique.append((sym, exch))
    return unique


def try_symbols(attempts: List[Tuple[str, Optional[str]]], fetch_fn):
    """Próbálkozik több tickerrel (szimbólum, tőzsde) sorban; az első ok:true visszatér."""
    last = None
    for sym, exch in attempts:
        try:
            r = fetch_fn(sym, exch)
            last = r
            if isinstance(r, dict) and r.get("ok"):
                return r
        except Exception as e:
            last = {"ok": False, "error": str(e)}
        delay = max(TD_RATE_LIMITER.current_delay * 0.5, 0.1)
        time.sleep(delay)
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
    if last_n_true(gt, 5):
        return {"ok": True, "signal": "uptrend", "reasons": ["ema9 > ema21 (5 bars)"]}
    if last_n_true(lt, 5):
        return {"ok": True, "signal": "downtrend", "reasons": ["ema9 < ema21 (5 bars)"]}
    return {"ok": True, "signal": "no entry", "reasons": ["no consistent ema bias"]}

# ─────────────────────────── Egy eszköz feldolgozása ──────────────────────

def process_asset(asset: str, cfg: Dict[str, Any]) -> None:
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)

    attempts = _normalize_symbol_attempts(cfg)

    # 1) Spot (quote → 5m close fallback), több tickerrel
    spot = try_symbols(attempts, lambda s, ex: td_spot_with_fallback(s, ex))
    save_json(os.path.join(adir, "spot.json"), spot)
    collect_realtime_spot(asset, attempts, adir)
    time.sleep(TD_PAUSE)

    # 2) OHLC (1m / 5m / 1h / 4h) – az első sikeres tickerrel
    def ts(s: str, ex: Optional[str], iv: str):
        return td_time_series(s, iv, 500, ex, "desc")

    k1m = try_symbols(attempts, lambda s, ex: ts(s, ex, "1min"))
    save_json(os.path.join(adir, "klines_1m.json"), k1m.get("raw", {"values": []}))
    time.sleep(TD_PAUSE)

    k5 = try_symbols(attempts, lambda s, ex: ts(s, ex, "5min"))
    save_json(os.path.join(adir, "klines_5m.json"), k5.get("raw", {"values": []}))
    time.sleep(TD_PAUSE)

    k1 = try_symbols(attempts, lambda s, ex: ts(s, ex, "1h"))
    save_json(os.path.join(adir, "klines_1h.json"), k1.get("raw", {"values": []}))
    time.sleep(TD_PAUSE)

    k4 = try_symbols(attempts, lambda s, ex: ts(s, ex, "4h"))
    save_json(os.path.join(adir, "klines_4h.json"), k4.get("raw", {"values": []}))
    time.sleep(TD_PAUSE)

    # 3) 5m előzetes jelzés (analysis.py később felülírhatja)
    sig = signal_from_5m(k5 if isinstance(k5, dict) else {"ok": False})
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









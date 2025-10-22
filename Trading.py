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
  TD_REALTIME_SPOT   = "0/1"    (alapértelmezés: 1 — bekapcsolja a valós idejű spot-frissítést)
  TD_REALTIME_INTERVAL = "5"    (realtime ciklus közötti szünet sec)
  TD_REALTIME_DURATION = "60"   (realtime poll futási ideje sec)
  TD_REALTIME_ASSETS = ""       (komma-szeparált lista, üres = mind)
  TD_REALTIME_HTTP_MAX_SAMPLES = "6" (HTTP fallback minták plafonja)
  TD_REALTIME_WS_IDLE_GRACE = "15"  (WebSocket késlekedési türelem sec)
  TD_MAX_WORKERS      = "3"      (max. párhuzamos eszköz feldolgozás)
  TD_REQUEST_CONCURRENCY = "3"   (egyszerre futó TD hívások plafonja)
  PIPELINE_MAX_LAG_SECONDS = "240" (Trading → Analysis log figyelmeztetés küszöbe)
"""

import os, json, time, math, logging
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, List, Tuple, Callable

import requests

try:
    from active_anchor import update_anchor_metrics
except Exception:  # pragma: no cover - optional dependency path
    update_anchor_metrics = None

try:
    import websocket  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    websocket = None

try:
    from reports.pipeline_monitor import record_trading_run, get_pipeline_log_path
except Exception:  # pragma: no cover - optional helper
    def record_trading_run(*_args, **_kwargs):
        return None

    def get_pipeline_log_path(*_args, **_kwargs):
        return None

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
REALTIME_FLAG = os.getenv("TD_REALTIME_SPOT", "1").lower() in {"1", "true", "yes", "on"}
REALTIME_INTERVAL = float(os.getenv("TD_REALTIME_INTERVAL", "5"))
REALTIME_DURATION = float(os.getenv("TD_REALTIME_DURATION", "60"))
REALTIME_ASSETS_ENV = os.getenv("TD_REALTIME_ASSETS", "")
REALTIME_WS_URL = os.getenv("TD_REALTIME_WS_URL", "")
REALTIME_WS_SUBSCRIBE = os.getenv("TD_REALTIME_WS_SUBSCRIBE", "")
REALTIME_WS_PRICE_FIELD = os.getenv("TD_REALTIME_WS_PRICE_FIELD", "price")
REALTIME_WS_TS_FIELD = os.getenv("TD_REALTIME_WS_TS_FIELD", "timestamp")
REALTIME_WS_MAX_SAMPLES = int(os.getenv("TD_REALTIME_WS_MAX_SAMPLES", "120"))
REALTIME_WS_IDLE_GRACE = float(os.getenv("TD_REALTIME_WS_IDLE_GRACE", "15"))
REALTIME_WS_TIMEOUT = float(os.getenv("TD_REALTIME_WS_TIMEOUT", str(max(REALTIME_DURATION, 30.0))))
REALTIME_WS_ENABLED = bool(REALTIME_WS_URL and websocket is not None)
REALTIME_HTTP_MAX_SAMPLES = int(os.getenv("TD_REALTIME_HTTP_MAX_SAMPLES", "6"))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw_str = str(raw).strip()
    if not raw_str:
        return default
    try:
        return int(float(raw_str))
    except Exception:
        return default

# ───────────────────────────── Data freshness guards ──────────────────────
# Align the freshness limits with ``analysis.py`` tolerances so we can fall
# back to alternative symbols whenever the primary feed lags behind.  The
# values are intentionally generous – if every symbol is stale we still return
# the least-delayed payload instead of failing the pipeline.
SERIES_FRESHNESS_LIMITS = {
    "1min": 180.0 + 240.0,  # 1m candle + 4m tolerance
    "5min": 300.0 + 900.0,  # 5m candle + 15m tolerance
    "1h": 3600.0 + 5400.0,  # 1h candle + 90m tolerance
    "4h": 4 * 3600.0 + 21600.0,  # 4h candle + 6h tolerance
}
SPOT_FRESHNESS_LIMIT = float(os.getenv("TD_SPOT_FRESHNESS_LIMIT", "900"))


class AdaptiveRateLimiter:
    def __init__(self, base_pause: float, min_pause: float, max_pause: float) -> None:
        self.base = max(base_pause, 0.0)
        self.min_pause = max(min_pause, 0.0)
        self.max_pause = max(max_pause, self.min_pause if self.min_pause else 0.0)
        self._current = max(self.base, self.min_pause)
        self._lock = threading.Lock()

    @property
    def current_delay(self) -> float:
        with self._lock:
            return self._current

    def wait(self) -> None:
        delay = self.current_delay
        if delay > 0:
            time.sleep(delay)

    def record_success(self) -> None:
        with self._lock:
            if self._current <= 0:
                return
            self._current = max(self.min_pause, self._current * 0.85)

    def record_failure(self, throttled: bool = False) -> None:
        growth = 1.6 if throttled else 1.3
        baseline = self.base if self.base > 0 else 0.3
        with self._lock:
            self._current = min(self.max_pause, max(self._current, baseline) * growth)

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

TD_MAX_WORKERS = max(1, min(len(ASSETS), _env_int("TD_MAX_WORKERS", 3)))
_DEFAULT_REQ_CONCURRENCY = max(1, min(TD_MAX_WORKERS, 3))
TD_REQUEST_CONCURRENCY = max(1, min(len(ASSETS), _env_int("TD_REQUEST_CONCURRENCY", _DEFAULT_REQ_CONCURRENCY)))
_REQUEST_SEMAPHORE = threading.Semaphore(TD_REQUEST_CONCURRENCY)
ANCHOR_LOCK = threading.Lock()

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


def save_series_payload(out_dir: str, name: str, payload: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    raw: Dict[str, Any] = {"values": []}
    meta: Dict[str, Any] = {}
    if isinstance(payload, dict):
        raw_candidate = payload.get("raw")
        if isinstance(raw_candidate, dict):
            raw = raw_candidate
        meta = {k: v for k, v in payload.items() if k != "raw"}
    save_json(os.path.join(out_dir, f"{name}.json"), raw)
    if meta:
        save_json(os.path.join(out_dir, f"{name}_meta.json"), meta)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(value, int):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        result = float(text)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except Exception:
        return None


def _td_error_details(payload: Any) -> Tuple[Optional[str], Optional[int]]:
    """Extract Twelve Data error information from a JSON payload."""

    if not isinstance(payload, dict):
        return None, None

    def _message_from(data: Dict[str, Any]) -> Optional[str]:
        message = data.get("message") or data.get("error")
        return str(message) if message else None

    status = str(payload.get("status") or "").lower()
    if status == "error":
        message = _message_from(payload) or "Twelve Data error"
        return message, _safe_int(payload.get("code"))

    code = _safe_int(payload.get("code"))
    if code and code not in {0, 200}:
        message = _message_from(payload) or f"Twelve Data error (code {code})"
        return message, code

    meta = payload.get("meta")
    if isinstance(meta, dict):
        meta_status = str(meta.get("status") or "").lower()
        meta_code = _safe_int(meta.get("code"))
        if meta_status == "error" or (meta_code and meta_code not in {0, 200}):
            message = _message_from(meta) or _message_from(payload) or "Twelve Data error"
            return message, meta_code

    return None, None


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
            with _REQUEST_SEMAPHORE:
                response = requests.get(
                    f"{TD_BASE}/{path}",
                    params=params,
                    timeout=30,
                    headers={"User-Agent": "market-feed/td-only/1.0"},
                )
            response.raise_for_status()
            data = response.json()
            error_message, error_code = _td_error_details(data)
            if error_message:
                if error_code and str(error_code) not in error_message:
                    message = f"{error_message} (code {error_code})"
                else:
                    message = error_message
                err = RuntimeError(message)
                last_error = err
                effective_status = error_code if error_code is not None else (
                    response.status_code if response is not None else None
                )
                last_status = effective_status
                throttled = (error_code == 429) or ("limit" in message.lower())
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
        latest_iso: Optional[str] = None
        latency_seconds: Optional[float] = None
        values = j.get("values") or []
        if values:
            ts_raw = None
            top = values[0]
            if isinstance(top, dict):
                ts_raw = (
                    top.get("datetime")
                    or top.get("timestamp")
                    or top.get("time")
                )
            ts_iso = _iso_from_td_ts(ts_raw)
            latest_iso = ts_iso
            parsed = _parse_iso_utc(ts_iso) if ts_iso else None
            if parsed:
                latency = (datetime.now(timezone.utc) - parsed).total_seconds()
                latency_seconds = max(0.0, latency)
        return {
            "used_symbol": symbol,
            "asset": symbol,
            "interval": interval,
            "source": "twelvedata:time_series",
            "ok": ok,
            "retrieved_at_utc": now_utc(),
            "latest_utc": latest_iso,
            "latency_seconds": latency_seconds,
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
    price: Optional[float] = None
    price_candidates = [
        j.get("price"),
        j.get("close"),
        j.get("last"),
        j.get("last_price"),
        j.get("bid"),
        j.get("ask"),
        j.get("previous_close"),
    ]
    for candidate in price_candidates:
        value = _coerce_float(candidate)
        if value is not None:
            price = value
            break
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
    utc = q.get("utc") if isinstance(q, dict) else None
    fallback_used = False

    if price is None:
        try:
            px, ts = td_last_close(symbol, "5min", exchange)
        except Exception as e:
            px, ts = None, None
            if isinstance(q, dict):
                q["error_fallback"] = str(e)
        price, utc = px, ts
        fallback_used = True

    retrieved = now_utc()
    utc_iso = utc or retrieved
    latency_seconds: Optional[float] = None
    parsed = _parse_iso_utc(utc_iso)
    if parsed:
        latency_seconds = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())

    source = "twelvedata:quote"
    if fallback_used:
        source = "twelvedata:quote+series_fallback"

    result: Dict[str, Any] = {
        "asset": symbol,
        "source": source,
        "ok": price is not None,
        "retrieved_at_utc": retrieved,
        "price": price,
        "price_usd": price,
        "utc": utc_iso,
        "latency_seconds": latency_seconds,
        "fallback_used": fallback_used,
        "used_symbol": symbol,
        "used_exchange": exchange,
    }

    if isinstance(q, dict):
        if q.get("error"):
            result["error"] = q.get("error")
        if q.get("error_fallback"):
            result["error_fallback"] = q.get("error_fallback")
        quote_ts = _parse_iso_utc(q.get("utc") if isinstance(q.get("utc"), str) else q.get("utc"))
        retrieved_ts = _parse_iso_utc(q.get("retrieved_at_utc")) if isinstance(q.get("retrieved_at_utc"), str) else None
        if quote_ts and retrieved_ts:
            result["quote_latency_seconds"] = max(0.0, (retrieved_ts - quote_ts).total_seconds())

    return result


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
    max_samples: int,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    sample_cap = max(1, int(max_samples))
    failure_cycles = 0
    max_failures = max(2, len(symbol_cycle)) if symbol_cycle else 2
    while time.time() < deadline and len(frames) < sample_cap:
        cycle_success = False
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
            cycle_success = True
            break
        if len(frames) >= sample_cap:
            break
        if not cycle_success:
            failure_cycles += 1
            if not frames and failure_cycles >= max_failures:
                break
        else:
            failure_cycles = 0
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
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
        idle_grace = max(0.0, REALTIME_WS_IDLE_GRACE)
        last_progress = time.time()
        while time.time() < deadline and len(frames) < REALTIME_WS_MAX_SAMPLES:
            try:
                raw = ws.recv()
                last_progress = time.time()
            except websocket.WebSocketTimeoutException:
                if idle_grace == 0.0:
                    break
                if time.time() - last_progress >= idle_grace:
                    break
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
            last_progress = time.time()
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
    force: bool = False,
    reason: Optional[str] = None,
) -> None:
    realtime_enabled = REALTIME_FLAG or force
    if not realtime_enabled:
        return
    allowed_assets = {
        a.strip().upper()
        for a in REALTIME_ASSETS_ENV.split(",")
        if a.strip()
    }
    if allowed_assets and asset.upper() not in allowed_assets and not force:
        return

    ensure_dir(out_dir)
    symbol_cycle = list(attempts) if attempts else []
    if not symbol_cycle:
        return

    interval = max(0.5, REALTIME_INTERVAL)
    http_max_samples = REALTIME_HTTP_MAX_SAMPLES
    duration = max(REALTIME_DURATION, interval)

    # Forced realtime collection (pl. spot fallback) should be quick – if the
    # regular websocket polling is disabled we fall back to a much shorter HTTP
    # sampling window so the trading pipeline does not block for a full minute
    # on every asset.  This was the primary reason behind the ~8 perces
    # futásidő: minden instrumentum 60 másodpercig próbálkozott, miközben a
    # quote továbbra sem frissült.
    if force and not REALTIME_FLAG:
        duration = min(duration, max(interval * 2.0, 6.0))
        interval = min(interval, 2.0)
        http_max_samples = max(1, min(http_max_samples, 2))

    frames: List[Dict[str, Any]] = []
    transport: Optional[str] = None

    deadline = time.time() + duration
    use_ws = REALTIME_WS_ENABLED and REALTIME_FLAG and not force
    if use_ws:
        frames = _collect_ws_frames(asset, symbol_cycle, deadline)
        if frames:
            transport = "websocket"

    if not frames:
        remaining = max(0.0, deadline - time.time())
        if remaining > 0:
            deadline_http = time.time() + remaining
            frames = _collect_http_frames(
                symbol_cycle,
                deadline_http,
                interval,
                http_max_samples,
            )
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
    if force:
        payload["forced"] = True
        if reason:
            payload["force_reason"] = reason
        payload["sampling_window_seconds"] = round(duration, 3)
        payload["sampling_max_samples"] = http_max_samples
    if transport == "websocket" and isinstance(last_frame.get("raw"), dict):
        payload["raw_last_frame"] = last_frame["raw"]

    if update_anchor_metrics and payload.get("price") is not None:
        anchor_extras = {
            "current_price": payload.get("price"),
            "spot_price": payload.get("price"),
            "analysis_timestamp": payload.get("retrieved_at_utc"),
            "realtime_transport": payload.get("transport"),
            "realtime_samples": stats.get("samples") if isinstance(stats, dict) else None,
            "realtime_latency": stats.get("latency_latest") if isinstance(stats, dict) else None,
            "realtime_source": payload.get("source"),
            "realtime_price_utc": payload.get("utc"),
        }
        try:
            with ANCHOR_LOCK:
                update_anchor_metrics(asset, anchor_extras)
        except Exception:
            pass
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


def try_symbols(
    attempts: List[Tuple[str, Optional[str]]],
    fetch_fn,
    freshness_limit: Optional[float] = None,
):
    """Iterate over symbol candidates and prefer the freshest successful payload."""

    last: Optional[Dict[str, Any]] = None
    best: Optional[Dict[str, Any]] = None
    best_latency: Optional[float] = None

    for sym, exch in attempts:
        try:
            result = fetch_fn(sym, exch)
        except Exception as exc:
            last = {"ok": False, "error": str(exc)}
            time.sleep(max(TD_RATE_LIMITER.current_delay * 0.5, 0.1))
            continue

        if not isinstance(result, dict):
            last = {"ok": False, "error": "invalid response"}
            time.sleep(max(TD_RATE_LIMITER.current_delay * 0.5, 0.1))
            continue

        last = result
        result.setdefault("used_symbol", sym)
        if exch is not None and not result.get("used_exchange"):
            result["used_exchange"] = exch

        latency = _coerce_float(result.get("latency_seconds"))
        if latency is not None:
            result["latency_seconds"] = latency

        violation = False
        if freshness_limit is not None:
            result["freshness_limit_seconds"] = freshness_limit
        if freshness_limit is not None and latency is not None:
            violation = latency > freshness_limit
            result["freshness_violation"] = bool(violation)
        elif freshness_limit is not None and "freshness_violation" not in result:
            result["freshness_violation"] = False

        if result.get("ok"):
            if not violation:
                return result
            if best is None:
                best = result
                best_latency = latency
            elif latency is not None:
                if best_latency is None or latency < best_latency:
                    best = result
                    best_latency = latency

        time.sleep(max(TD_RATE_LIMITER.current_delay * 0.5, 0.1))

    if best is not None:
        return best
    return last or {"ok": False}


def fetch_with_freshness(
    attempts: List[Tuple[str, Optional[str]]],
    fetch_fn: Callable[[str, Optional[str]], Dict[str, Any]],
    freshness_limit: Optional[float] = None,
    max_refreshes: int = 1,
):
    result = try_symbols(attempts, fetch_fn, freshness_limit=freshness_limit)
    retries = 0
    while (
        freshness_limit is not None
        and isinstance(result, dict)
        and result.get("ok")
        and result.get("freshness_violation")
        and retries < max_refreshes
    ):
        retries += 1
        time.sleep(max(TD_RATE_LIMITER.current_delay, 0.2))
        result = try_symbols(attempts, fetch_fn, freshness_limit=freshness_limit)
    if isinstance(result, dict):
        result.setdefault("freshness_violation", bool(result.get("freshness_violation")))
        result["freshness_retries"] = retries
    return result

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
    spot = fetch_with_freshness(
        attempts,
        lambda s, ex: td_spot_with_fallback(s, ex),
        freshness_limit=SPOT_FRESHNESS_LIMIT,
        max_refreshes=1,
    )
    if isinstance(spot, dict):
        spot.setdefault("freshness_limit_seconds", SPOT_FRESHNESS_LIMIT)
    else:
        spot = {"ok": False, "freshness_limit_seconds": SPOT_FRESHNESS_LIMIT}
    save_json(os.path.join(adir, "spot.json"), spot)

    spot_violation = bool(spot.get("freshness_violation")) if isinstance(spot, dict) else False
    force_reason: Optional[str] = None
    if not spot.get("ok"):
        force_reason = "spot_error"
    elif spot_violation:
        force_reason = "spot_stale"
    elif spot.get("fallback_used"):
        force_reason = "spot_fallback"
    collect_realtime_spot(asset, attempts, adir, force=bool(force_reason), reason=force_reason)
    time.sleep(TD_PAUSE)

    # 2) OHLC (1m / 5m / 1h / 4h) – az első sikeres tickerrel
    def ts(s: str, ex: Optional[str], iv: str):
        return td_time_series(s, iv, 500, ex, "desc")

    k1m = fetch_with_freshness(
        attempts,
        lambda s, ex: ts(s, ex, "1min"),
        freshness_limit=SERIES_FRESHNESS_LIMITS.get("1min"),
        max_refreshes=1,
    )
    save_series_payload(adir, "klines_1m", k1m if isinstance(k1m, dict) else {"ok": False, "raw": {"values": []}})
    time.sleep(TD_PAUSE)

    k5 = fetch_with_freshness(
        attempts,
        lambda s, ex: ts(s, ex, "5min"),
        freshness_limit=SERIES_FRESHNESS_LIMITS.get("5min"),
        max_refreshes=1,
    )
    save_series_payload(adir, "klines_5m", k5 if isinstance(k5, dict) else {"ok": False, "raw": {"values": []}})
    time.sleep(TD_PAUSE)

    k1 = fetch_with_freshness(
        attempts,
        lambda s, ex: ts(s, ex, "1h"),
        freshness_limit=SERIES_FRESHNESS_LIMITS.get("1h"),
        max_refreshes=1,
    )
    save_series_payload(adir, "klines_1h", k1 if isinstance(k1, dict) else {"ok": False, "raw": {"values": []}})
    time.sleep(TD_PAUSE)

    k4 = fetch_with_freshness(
        attempts,
        lambda s, ex: ts(s, ex, "4h"),
        freshness_limit=SERIES_FRESHNESS_LIMITS.get("4h"),
        max_refreshes=1,
    )
    save_series_payload(adir, "klines_4h", k4 if isinstance(k4, dict) else {"ok": False, "raw": {"values": []}})
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

def _write_error_payload(asset: str, error: Exception) -> None:
    adir = os.path.join(OUT_DIR, asset)
    ensure_dir(adir)
    now = now_utc()
    reason = f"fetch error: {error}"
    reason = " ".join(reason.split())
    save_json(
        os.path.join(adir, "spot.json"),
        {
            "asset": asset,
            "ok": False,
            "retrieved_at_utc": now,
            "price": None,
            "price_usd": None,
            "utc": now,
        },
    )
    save_json(
        os.path.join(adir, "signal.json"),
        {
            "asset": asset,
            "ok": False,
            "retrieved_at_utc": now,
            "signal": "no entry",
            "probability": 0,
            "reasons": [reason],
            "spot": {"price": None, "utc": now},
        },
    )


def _process_asset_guard(asset: str, cfg: Dict[str, Any]) -> None:
    try:
        process_asset(asset, cfg)
    except Exception as error:
        _write_error_payload(asset, error)


def main():
    if not API_KEY:
        raise SystemExit("TWELVEDATA_API_KEY hiányzik (GitHub Secret).")
    ensure_dir(OUT_DIR)
    logger = logging.getLogger("market_feed.trading")
    pipeline_log_path = None
    try:
        pipeline_log_path = get_pipeline_log_path()
    except Exception:
        pipeline_log_path = None
    if pipeline_log_path:
        ensure_dir(str(pipeline_log_path.parent))
        if not any(getattr(handler, "_pipeline_log", False) for handler in logger.handlers):
            handler = logging.FileHandler(pipeline_log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s"))
            handler._pipeline_log = True  # type: ignore[attr-defined]
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    started_at_dt = datetime.now(timezone.utc)
    logger.info("Trading run started for %d assets", len(ASSETS))
    if REALTIME_FLAG:
        workers = 1
    else:
        workers = max(1, min(len(ASSETS), TD_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_asset_guard, asset, cfg): asset
            for asset, cfg in ASSETS.items()
        }
        for future in as_completed(futures):
            future.result()
    completed_at_dt = datetime.now(timezone.utc)
    duration_seconds = max((completed_at_dt - started_at_dt).total_seconds(), 0.0)
    logger.info("Trading run completed in %.1f seconds", duration_seconds)
    try:
        record_trading_run(started_at=started_at_dt, completed_at=completed_at_dt, duration_seconds=duration_seconds)
    except Exception as exc:
        logger.warning("Failed to record trading pipeline metrics: %s", exc)

if __name__ == "__main__":
    main()


















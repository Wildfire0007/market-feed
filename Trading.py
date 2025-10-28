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
  TD_PAUSE           = "0.15"   (kímélő szünet hívások közt, sec)
  TD_MAX_RETRIES     = "3"      (újrapróbálkozások száma)
  TD_BACKOFF_BASE    = "0.25"   (exponenciális visszavárás alapja)
  TD_BACKOFF_MAX     = "8"      (exponenciális visszavárás plafonja, sec)
  TD_REQUESTS_PER_MINUTE = "55" (Twelve Data perces hívás plafon)
  TD_REQUEST_BURST   = "10"     (engedélyezett azonnali burst ablak)
  TD_REALTIME_SPOT   = "0/1"    (alapértelmezés: 0 — külön job kezeli a realtime spotot)
  TD_REALTIME_INTERVAL = "5"    (realtime ciklus közötti szünet sec)
  TD_REALTIME_DURATION = "20"   (realtime poll futási ideje sec)
  TD_REALTIME_HTTP_DURATION = "8"  (HTTP fallback mintavételezés hossza sec)
  TD_REALTIME_ASSETS = ""       (komma-szeparált lista, üres = mind)
  TD_REALTIME_HTTP_MAX_SAMPLES = "6" (HTTP fallback minták plafonja)
  TD_REALTIME_HTTP_BACKGROUND = "auto/1" (HTTP realtime gyűjtés háttérszálon fusson-e)
  TD_REALTIME_HTTP_BUDGET = "auto" (HTTP realtime mintavételek per-run limitje)
  TD_REALTIME_WS_IDLE_GRACE = "15"  (WebSocket késlekedési türelem sec)
  TD_MAX_WORKERS      = "5"      (max. párhuzamos eszköz feldolgozás)
  TD_REQUEST_CONCURRENCY = "4"   (egyszerre futó TD hívások plafonja)
  PIPELINE_MAX_LAG_SECONDS = "240" (Trading → Analysis log figyelmeztetés küszöbe)
"""

import os, json, time, math, logging
import threading
from datetime import datetime, timezone, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, List, Tuple, Callable, Set

import requests
from requests.adapters import HTTPAdapter

try:
    from active_anchor import update_anchor_metrics
except Exception:  # pragma: no cover - optional dependency path
    update_anchor_metrics = None

try:
    import websocket  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    websocket = None

try:
    from reports.pipeline_monitor import (
        record_trading_run,
        get_pipeline_log_path,
        summarize_pipeline_warnings,
    )
except Exception:  # pragma: no cover - optional helper
    def record_trading_run(*_args, **_kwargs):
        return None

    def get_pipeline_log_path(*_args, **_kwargs):
        return None

    def summarize_pipeline_warnings(*_args, **_kwargs):
        return {}


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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw_str = str(raw).strip()
    if not raw_str:
        return default
    try:
        return float(raw_str)
    except Exception:
        return default


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw_str = str(raw).strip().lower()
    if not raw_str or raw_str == "auto":
        return None
    if raw_str in {"1", "true", "yes", "on"}:
        return True
    if raw_str in {"0", "false", "no", "off"}:
        return False
    return None

OUT_DIR  = os.getenv("OUT_DIR", "public")
API_KEY_RAW = os.getenv("TWELVEDATA_API_KEY", "")
API_KEY  = API_KEY_RAW.strip() if API_KEY_RAW else ""
TD_BASE  = "https://api.twelvedata.com"
TD_PAUSE = float(os.getenv("TD_PAUSE", "0.15"))
TD_PAUSE_MIN = float(os.getenv("TD_PAUSE_MIN", str(max(TD_PAUSE * 0.6, 0.1))))
TD_PAUSE_MAX = float(os.getenv("TD_PAUSE_MAX", str(max(TD_PAUSE * 6, 4.0))))
TD_MAX_RETRIES = int(os.getenv("TD_MAX_RETRIES", "3"))
TD_BACKOFF_BASE = float(os.getenv("TD_BACKOFF_BASE", str(max(TD_PAUSE, 0.25))))
TD_BACKOFF_MAX = float(os.getenv("TD_BACKOFF_MAX", "8.0"))
TD_REQUESTS_PER_MINUTE = max(10, _env_int("TD_REQUESTS_PER_MINUTE", 55))
TD_REQUEST_BURST = max(1, _env_int("TD_REQUEST_BURST", min(10, TD_REQUESTS_PER_MINUTE)))
REALTIME_FLAG = os.getenv("TD_REALTIME_SPOT", "0").lower() in {"1", "true", "yes", "on"}
REALTIME_INTERVAL = float(os.getenv("TD_REALTIME_INTERVAL", "5"))
REALTIME_DURATION = float(os.getenv("TD_REALTIME_DURATION", "20"))
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
REALTIME_HTTP_DURATION = max(1.0, _env_float("TD_REALTIME_HTTP_DURATION", 8.0))
_REALTIME_HTTP_BACKGROUND_FLAG = _env_flag("TD_REALTIME_HTTP_BACKGROUND")

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

# Reduce the volume of large timeframe requests to keep latency in check.  The
# short-term series now default to a lighter 300 bar history, while the heavier
# 1h and 4h payloads stay lean as well.  All values can be overridden from the
# environment if necessary.
SERIES_OUTPUT_SIZES = {
    "1min": max(50, _env_int("TD_OUTPUTSIZE_1MIN", 300)),
    "5min": max(50, _env_int("TD_OUTPUTSIZE_5MIN", 300)),
    "1h": max(50, _env_int("TD_OUTPUTSIZE_1H", 240)),
    "4h": max(50, _env_int("TD_OUTPUTSIZE_4H", 180)),
}

SERIES_FETCH_PLAN = [
    ("klines_1m", "1min"),
    ("klines_5m", "5min"),
    ("klines_1h", "1h"),
    ("klines_4h", "4h"),
]


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
            target = self._current * 0.7
            if self.base > 0:
                target = max(target, self.base * 0.6)
            self._current = max(self.min_pause, target)

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


class TokenBucketLimiter:
    def __init__(
        self,
        rate_per_minute: int,
        burst_size: int,
    ) -> None:
        rate_per_minute = max(1, rate_per_minute)
        burst_size = max(1, burst_size)
        self.rate_per_second = rate_per_minute / 60.0
        self.capacity = float(max(burst_size, 1))
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        delta = now - self.updated
        if delta <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + delta * self.rate_per_second)
        self.updated = now

    def wait(self) -> None:
        while True:
            with self._lock:
                self._refill_locked()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                needed = (1.0 - self.tokens) / self.rate_per_second if self.rate_per_second else 0.2
            time.sleep(max(needed, 0.05))

    def penalize(self, seconds: float = 2.0) -> None:
        seconds = max(seconds, 0.0)
        if seconds <= 0:
            return
        with self._lock:
            self._refill_locked()
            penalty = max(self.rate_per_second * seconds, 1.0)
            self.tokens = max(0.0, self.tokens - penalty)
            self.updated = time.monotonic()


class RequestGovernor:
    def __init__(
        self,
        base_pause: float,
        min_pause: float,
        max_pause: float,
        rate_per_minute: int,
        burst_size: int,
    ) -> None:
        self._adaptive = AdaptiveRateLimiter(base_pause, min_pause, max_pause)
        self._bucket = TokenBucketLimiter(rate_per_minute, burst_size)

    @property
    def current_delay(self) -> float:
        return self._adaptive.current_delay

    def wait(self) -> None:
        self._bucket.wait()
        self._adaptive.wait()

    def record_success(self) -> None:
        self._adaptive.record_success()

    def record_failure(
        self,
        throttled: bool = False,
        *,
        retry_after: Optional[float] = None,
    ) -> None:
        self._adaptive.record_failure(throttled=throttled)
        if throttled:
            penalty = retry_after if retry_after is not None else 5.0
            self._bucket.penalize(max(penalty, 1.0))

    def backoff_seconds(self, attempt: int, retry_after: Optional[float] = None) -> float:
        return self._adaptive.backoff_seconds(attempt, retry_after=retry_after)


TD_RATE_LIMITER = RequestGovernor(
    TD_PAUSE,
    TD_PAUSE_MIN,
    TD_PAUSE_MAX,
    TD_REQUESTS_PER_MINUTE,
    TD_REQUEST_BURST,
)

# ────────────────────────────── Exceptions ───────────────────────────────


class TDError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        throttled: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.throttled = throttled


# ───────────────────────────────── ASSETS ────────────────────────────────
# GER40 helyett USOIL. A fő ticker a WTI/USD, fallbackokra már nincs szükség.
LOGGER = logging.getLogger("market_feed.trading")


ASSETS = {
    "EURUSD": {
        "symbol": "EUR/USD",
        "name": "Euro / US Dollar",
        "asset_class": "FX",
        "exchange": "PHYSICAL CURRENCY",
        "exchange_display": "Physical Currency",
        "currency": "USD",
    },
    "BTCUSD": {
        "symbol": "BTC/USD",
        "name": "Bitcoin / US Dollar (Coinbase Pro)",
        "asset_class": "Crypto",
        "exchange": "Coinbase Pro",
        "exchange_display": "Coinbase Pro",
        "currency": "USD",
    },
    "GOLD_CFD": {
        "symbol": "XAU/USD",
        "name": "Gold / US Dollar",
        "asset_class": "Commodity",
        "exchange": "PHYSICAL METAL",
        "exchange_display": "Physical Metal",
        "currency": "USD",
    },

    # ÚJ: WTI kőolaj. A Twelve Data-n a hivatalos jelölés: WTI/USD, de több
    # alternatív ticker is forgalomban van – tartsuk meg a ténylegesen szükséges
    # fallbackokat, hogy elkerüljük a fölös próbálkozásokat.
    "USOIL": {
        "symbol": "WTI/USD",
        "name": "Crude Oil WTI Spot / US Dollar",
        "asset_class": "Commodity",
        "exchange": "COMMODITY",
        "exchange_display": "Commodity",
        "currency": "USD",
        "disable_compact_variants": True,
        "alt": [
            {"symbol": "WTI/USD", "exchange": None, "disable_compact_variants": True},
        ],
    },

# Egyedi részvény és ETF kiterjesztések
    "NVDA": {
        "symbol": "NVDA",
        "name": "NVIDIA Corporation",
        "asset_class": "Equity",
        "exchange": "NASDAQ",
        "exchange_display": "Nasdaq",
        "mic": "XNGS",
        "currency": "USD",
        "supports_prepost": True,
    },
    "XAGUSD": {
        "symbol": "XAG/USD",
        "name": "Silver / US Dollar",
        "asset_class": "Commodity",
        "exchange": "PHYSICAL METAL",
        "exchange_display": "Physical Metal",
        "currency": "USD",
        "disable_compact_variants": True,
        "alt": [
            {"symbol": "XAG/USD", "exchange": None, "disable_compact_variants": True},
            {"symbol": "XAGUSD", "exchange": None, "disable_compact_variants": True},
            {"symbol": "XAGUSD", "exchange": "FOREX", "disable_compact_variants": True},
            "XAG/USD:FOREX",
        ],
    },
}


def _normalize_symbol_token(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").replace("-", "").strip().upper()


_ASSET_SYMBOL_INDEX: Dict[str, str] = {}
for _asset_key, _cfg in ASSETS.items():
    if not isinstance(_cfg, dict):
        continue
    if isinstance(_asset_key, str) and _asset_key:
        _ASSET_SYMBOL_INDEX.setdefault(_normalize_symbol_token(_asset_key), _asset_key)
    symbol_text = _cfg.get("symbol")
    if isinstance(symbol_text, str) and symbol_text:
        _ASSET_SYMBOL_INDEX.setdefault(_normalize_symbol_token(symbol_text), _asset_key)
    for alt in _cfg.get("alt", []) or []:
        if isinstance(alt, str) and alt:
            _ASSET_SYMBOL_INDEX.setdefault(_normalize_symbol_token(alt), _asset_key)
        elif isinstance(alt, dict):
            alt_symbol = alt.get("symbol")
            if isinstance(alt_symbol, str) and alt_symbol:
                _ASSET_SYMBOL_INDEX.setdefault(_normalize_symbol_token(alt_symbol), _asset_key)


def _resolve_asset_from_payload(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    candidates: List[str] = []
    for key in (
        "asset",
        "asset_key",
        "symbol",
        "requested_symbol",
        "base_symbol",
        "symbol_requested",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    for candidate in candidates:
        normalized = _normalize_symbol_token(candidate)
        asset_key = _ASSET_SYMBOL_INDEX.get(normalized)
        if not asset_key:
            continue
        cfg = ASSETS.get(asset_key)
        if isinstance(cfg, dict):
            return asset_key, cfg
    return None, None


def _is_us_equity_asset(cfg: Dict[str, Any]) -> bool:
    asset_class = str(cfg.get("asset_class") or "").strip().lower()
    if asset_class != "equity":
        return False
    exchange = str(cfg.get("exchange") or cfg.get("exchange_display") or "").upper()
    return any(tag in exchange for tag in ("NASDAQ", "NYSE", "ARCA", "BATS"))


def _asset_market_closed_reason(
    asset_key: Optional[str],
    cfg: Optional[Dict[str, Any]],
    latest_dt: datetime,
    now_dt: datetime,
) -> Optional[str]:
    if now_dt.weekday() >= 5 or latest_dt.weekday() >= 5:
        return "weekend"
    if not cfg:
        return None
    if _is_us_equity_asset(cfg):
        now_time = now_dt.time()
        if now_time < US_EQUITY_OPEN_UTC or now_time >= US_EQUITY_CLOSE_UTC:
            return "outside_hours"
    return None

_BASE_REQUESTS_PER_ASSET = 1 + len(SERIES_FETCH_PLAN)
_BASE_REQUESTS_TOTAL = len(ASSETS) * _BASE_REQUESTS_PER_ASSET
_DEFAULT_HTTP_BUDGET = max(1, TD_REQUESTS_PER_MINUTE - _BASE_REQUESTS_TOTAL)
REALTIME_HTTP_BUDGET_TOTAL = max(
    1,
    _env_int("TD_REALTIME_HTTP_BUDGET", _DEFAULT_HTTP_BUDGET),
)
REALTIME_HTTP_BUDGET_PER_ASSET = max(
    1,
    math.ceil(REALTIME_HTTP_BUDGET_TOTAL / max(len(ASSETS), 1)),
)

MARKET_CLOSED_GRACE_SECONDS = max(0.0, float(os.getenv("TD_MARKET_CLOSED_GRACE", "43200")))
MARKET_CLOSED_MAX_AGE_SECONDS = max(0.0, float(os.getenv("TD_MARKET_CLOSED_MAX_AGE", "79200")))
MAX_CONSECUTIVE_FALLBACKS = max(1, int(os.getenv("TD_MAX_CONSECUTIVE_FALLBACKS", "3")))
US_EQUITY_OPEN_UTC = dt_time(13, 30)
US_EQUITY_CLOSE_UTC = dt_time(20, 0)

_TRADING_STATUS: Dict[str, Dict[str, Any]] = {}
_TRADING_STATUS_LOCK = threading.Lock()

TD_MAX_WORKERS = max(1, min(len(ASSETS), _env_int("TD_MAX_WORKERS", len(ASSETS))))
_DEFAULT_REQ_CONCURRENCY = max(
    1,
    min(len(ASSETS) * max(len(SERIES_FETCH_PLAN), 1), 12),
)
TD_REQUEST_CONCURRENCY = max(
    1,
    min(
        len(ASSETS) * max(len(SERIES_FETCH_PLAN), 1),
        _env_int("TD_REQUEST_CONCURRENCY", _DEFAULT_REQ_CONCURRENCY),
    ),
)
_REQUEST_SEMAPHORE = threading.Semaphore(TD_REQUEST_CONCURRENCY)
_REQUEST_SESSION = requests.Session()
_REQUEST_ADAPTER = HTTPAdapter(
    pool_connections=TD_REQUEST_CONCURRENCY,
    pool_maxsize=max(TD_REQUEST_CONCURRENCY * 2, 4),
)
_REQUEST_SESSION.mount("https://", _REQUEST_ADAPTER)
_REQUEST_SESSION.mount("http://", _REQUEST_ADAPTER)
_REQUEST_SESSION.headers.update({"User-Agent": "market-feed/td-only/1.0"})
_ORIGINAL_REQUESTS_GET = requests.get
ANCHOR_LOCK = threading.Lock()
_REALTIME_BACKGROUND_LOCK = threading.Lock()
REALTIME_BACKGROUND_THREADS: List[threading.Thread] = []

# ───────────────────────────── Attempt memory ─────────────────────────────


class AttemptMemory:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hard_failures: Dict[Tuple[str, Optional[str]], str] = {}
        self._skip_logged: Set[Tuple[str, Optional[str]]] = set()

    def record_hard_failure(
        self,
        symbol: str,
        exchange: Optional[str],
        reason: str,
    ) -> None:
        key = (symbol, exchange)
        normalized_reason = " ".join(str(reason).split())
        with self._lock:
            if key not in self._hard_failures:
                self._hard_failures[key] = normalized_reason
            else:
                # Update the reason if the new message is more descriptive.
                existing = self._hard_failures[key]
                if normalized_reason and normalized_reason not in existing:
                    self._hard_failures[key] = normalized_reason
            self._skip_logged.discard(key)

    def should_skip(self, symbol: str, exchange: Optional[str]) -> Tuple[bool, Optional[str], bool]:
        key = (symbol, exchange)
        with self._lock:
            reason = self._hard_failures.get(key)
            if reason is None:
                return False, None, False
            first_skip = key not in self._skip_logged
            if first_skip:
                self._skip_logged.add(key)
        return True, reason, first_skip

    def snapshot(self) -> Dict[Tuple[str, Optional[str]], str]:
        with self._lock:
            return dict(self._hard_failures)

    def is_blacklisted(self, symbol: str, exchange: Optional[str]) -> Optional[str]:
        with self._lock:
            return self._hard_failures.get((symbol, exchange))

# HTTP státuszkódok, amelyeknél a kliens oldali hibát tartósnak tekintjük.
CLIENT_ERROR_STATUS_CODES: Set[Optional[int]] = {
    400,
    401,
    402,
    403,
    404,
    410,
    422,
    451,
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
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def _should_preserve_cache(paths: List[str], payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if not payload.get("fallback_previous_payload"):
        return False
    return all(os.path.exists(path) for path in paths)


def _fallback_state_path(adir: str) -> str:
    return os.path.join(adir, "_fallback_state.json")


def _load_fallback_state(adir: str) -> Dict[str, Any]:
    state = load_json(_fallback_state_path(adir))
    return state if isinstance(state, dict) else {}


def _store_fallback_state(adir: str, key: str, state: Optional[Dict[str, Any]]) -> None:
    path = _fallback_state_path(adir)
    current = _load_fallback_state(adir)
    if state:
        current[key] = state
    else:
        current.pop(key, None)
    if current:
        save_json(path, current)
    else:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _fallback_state_info(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if not payload.get("fallback_previous_payload"):
        return None
    reuse_count = payload.get("fallback_reuse_count")
    try:
        reuse_count_int = max(0, int(reuse_count or 0))
    except Exception:
        reuse_count_int = 0
    info: Dict[str, Any] = {
        "reuse_count": reuse_count_int,
        "updated_at_utc": payload.get("retrieved_at_utc") or now_utc(),
    }
    reason = payload.get("fallback_reason") or payload.get("error") or payload.get("message")
    if reason:
        info["reason"] = str(reason)
    interval = payload.get("interval")
    if interval:
        info["interval"] = interval
    return info


def _write_spot_payload(adir: str, asset: str, payload: Dict[str, Any]) -> None:
    path = os.path.join(adir, "spot.json")
    info = _fallback_state_info(payload)
    if _should_preserve_cache([path], payload):
        if info:
            _store_fallback_state(adir, "spot", info)
        LOGGER.debug("Preserving existing spot cache for %s due to fallback reuse", asset)
        return
    if info:
        _store_fallback_state(adir, "spot", info)
    else:
        _store_fallback_state(adir, "spot", None)
    save_json(path, payload)


def load_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _format_attempts(attempts: List[Tuple[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
    formatted: List[Dict[str, Optional[str]]] = []
    for symbol, exchange in attempts:
        formatted.append({"symbol": symbol, "exchange": exchange})
    return formatted


def _reuse_previous_spot(adir: str, payload: Dict[str, Any], freshness_limit: float) -> Dict[str, Any]:
    if payload.get("ok") and payload.get("price") is not None:
        payload.pop("fallback_reuse_count", None)
        return payload

    previous = load_json(os.path.join(adir, "spot.json"))
    if not isinstance(previous, dict) or not previous.get("ok"):
        return payload

    reuse_count = 0
    try:
        reuse_count = max(0, int(previous.get("fallback_reuse_count", 0)))
    except Exception:
        reuse_count = 0
    state_entry = _load_fallback_state(adir).get("spot")
    if isinstance(state_entry, dict):
        try:
            state_count = int(state_entry.get("reuse_count", reuse_count))
            reuse_count = max(reuse_count, max(0, state_count))
        except Exception:
            pass
    if reuse_count >= MAX_CONSECUTIVE_FALLBACKS:
        asset_name = os.path.basename(adir) or "<asset>"
        LOGGER.warning(
            "Skipping spot fallback for %s after %d consecutive reuses",
            asset_name,
            reuse_count,
        )
        payload.pop("fallback_reuse_count", None)
        return payload

    price = previous.get("price")
    utc_iso = previous.get("utc")
    if price is None or not isinstance(utc_iso, str):
        return payload

    age = _age_seconds_from_iso(utc_iso)
    if age is None or age > freshness_limit:
        return payload

    reused = dict(previous)
    reused["retrieved_at_utc"] = now_utc()
    reused["latency_seconds"] = age
    reused["fallback_previous_payload"] = True
    reason = payload.get("error") or payload.get("error_fallback")
    if reason:
        reused["fallback_reason"] = str(reason)
    reused.setdefault("freshness_limit_seconds", freshness_limit)
    reused["fallback_reuse_count"] = reuse_count + 1
    return reused


def _record_asset_status(
    asset: str,
    attempts: List[Tuple[str, Optional[str]]],
    spot: Dict[str, Any],
    series_payloads: Dict[str, Dict[str, Any]],
    attempt_memory: Optional[AttemptMemory] = None,
) -> None:
    summary_series: Dict[str, Dict[str, Any]] = {}
    for name, payload in series_payloads.items():
        if not isinstance(payload, dict):
            continue
        values = []
        raw = payload.get("raw")
        if isinstance(raw, dict):
            raw_values = raw.get("values")
            if isinstance(raw_values, list):
                values = raw_values
        summary_series[name] = {
            "ok": bool(payload.get("ok")),
            "values": len(values),
            "freshness_violation": bool(payload.get("freshness_violation")),
            "used_symbol": payload.get("used_symbol"),
            "used_exchange": payload.get("used_exchange"),
            "error": payload.get("error"),
        }

    spot_info = spot if isinstance(spot, dict) else {"ok": False}
    spot_summary = {
        "ok": bool(spot_info.get("ok")),
        "freshness_violation": bool(spot_info.get("freshness_violation")),
        "used_symbol": spot_info.get("used_symbol") or spot_info.get("asset"),
        "used_exchange": spot_info.get("used_exchange"),
        "error": spot_info.get("error"),
    }

    payload: Dict[str, Any] = {
        "asset": asset,
        "updated_at_utc": now_utc(),
        "attempts": _format_attempts(attempts),
        "spot": spot_summary,
        "series": summary_series,
    }

    if attempt_memory is not None:
        recorded = attempt_memory.snapshot()
        if recorded:
            payload["hard_failures"] = [
                {"symbol": sym, "exchange": exch, "reason": reason}
                for (sym, exch), reason in recorded.items()
            ]

    with _TRADING_STATUS_LOCK:
        _TRADING_STATUS[asset] = payload


def _record_asset_failure(asset: str, reason: str) -> None:
    payload = {
        "asset": asset,
        "updated_at_utc": now_utc(),
        "attempts": [],
        "spot": {"ok": False, "error": reason},
        "series": {},
    }
    with _TRADING_STATUS_LOCK:
        _TRADING_STATUS[asset] = payload


def _write_trading_status_summary(out_dir: str) -> None:
    with _TRADING_STATUS_LOCK:
        assets = list(_TRADING_STATUS.values())

    assets.sort(key=lambda item: item.get("asset", ""))
    all_ready = True if assets else False
    for item in assets:
        spot_ok = bool(item.get("spot", {}).get("ok"))
        series_info = item.get("series", {})
        frames_ok = all(
            bool(frame.get("ok")) and frame.get("values", 0) > 0
            for frame in series_info.values()
        ) if series_info else False
        if not (spot_ok and frames_ok):
            all_ready = False
            break

    pipeline_dir = os.path.join(out_dir, "pipeline")
    ensure_dir(pipeline_dir)
    payload = {
        "generated_at_utc": now_utc(),
        "assets": assets,
        "all_assets_ready": all_ready,
    }
    save_json(os.path.join(pipeline_dir, "trading_status.json"), payload)


def save_series_payload(out_dir: str, name: str, payload: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    raw_path = os.path.join(out_dir, f"{name}.json")
    meta_path = os.path.join(out_dir, f"{name}_meta.json")
    info = _fallback_state_info(payload)
    state_key = f"series:{name}"
    if _should_preserve_cache([raw_path, meta_path], payload):
        if info:
            _store_fallback_state(out_dir, state_key, info)
        else:
            _store_fallback_state(out_dir, state_key, None)
        asset_name = os.path.basename(out_dir) or "<asset>"
        LOGGER.debug(
            "Preserving existing %s cache for %s due to fallback reuse",
            name,
            asset_name,
        )
        return
    if info:
        _store_fallback_state(out_dir, state_key, info)
    else:
        _store_fallback_state(out_dir, state_key, None)
    raw: Dict[str, Any] = {"values": []}
    meta: Dict[str, Any] = {}
    if isinstance(payload, dict):
        raw_candidate = payload.get("raw")
        if isinstance(raw_candidate, dict):
            raw = raw_candidate
        meta = {k: v for k, v in payload.items() if k != "raw"}
    save_json(raw_path, raw)
    if meta:
        save_json(meta_path, meta)


def _load_existing_series(out_dir: str, name: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    meta_path = os.path.join(out_dir, f"{name}_meta.json")
    raw_path = os.path.join(out_dir, f"{name}.json")
    meta = load_json(meta_path)
    raw = load_json(raw_path)
    if not isinstance(meta, dict) or not isinstance(raw, dict):
        return None, None
    return meta, raw


def _age_seconds_from_iso(ts: Optional[str]) -> Optional[float]:
    parsed = _parse_iso_utc(ts)
    if not parsed:
        return None
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _reuse_previous_series_payload(
    out_dir: str,
    name: str,
    payload: Dict[str, Any],
    freshness_limit: Optional[float],
) -> Dict[str, Any]:
    if freshness_limit is None:
        return payload

    ok = bool(payload.get("ok"))
    raw_values: List[Any] = []
    raw = payload.get("raw")
    if isinstance(raw, dict):
        raw_candidate = raw.get("values")
        if isinstance(raw_candidate, list):
            raw_values = raw_candidate
    if ok and raw_values:
        return payload

    meta_prev, raw_prev = _load_existing_series(out_dir, name)
    if not meta_prev or not raw_prev:
        return payload

    reuse_count = 0
    try:
        reuse_count = max(0, int(meta_prev.get("fallback_reuse_count", 0)))
    except Exception:
        reuse_count = 0
    state_entry = _load_fallback_state(out_dir).get(f"series:{name}")
    if isinstance(state_entry, dict):
        try:
            state_count = int(state_entry.get("reuse_count", reuse_count))
            reuse_count = max(reuse_count, max(0, state_count))
        except Exception:
            pass
    if reuse_count >= MAX_CONSECUTIVE_FALLBACKS:
        asset_name = os.path.basename(out_dir) or "<asset>"
        LOGGER.warning(
            "Skipping %s fallback for %s after %d consecutive reuses",
            name,
            asset_name,
            reuse_count,
        )
        return payload

    prev_values = raw_prev.get("values") if isinstance(raw_prev, dict) else None
    if not prev_values:
        return payload

    latest_iso = None
    if isinstance(meta_prev, dict):
        latest_iso = meta_prev.get("latest_utc") or meta_prev.get("utc")
    age = _age_seconds_from_iso(latest_iso)
    if age is None or age > freshness_limit:
        return payload

    reused = dict(meta_prev)
    reused["raw"] = raw_prev
    reused.setdefault("ok", True)
    reused["freshness_violation"] = False
    reused.setdefault("freshness_limit_seconds", freshness_limit)
    reused["latency_seconds"] = age
    reused["retrieved_at_utc"] = now_utc()
    reused["fallback_previous_payload"] = True
    reason = payload.get("error") or payload.get("message")
    if reason:
        reused["fallback_reason"] = str(reason)
    reused["fallback_reuse_count"] = reuse_count + 1
    return reused


def _series_fetcher(interval: str, outputsize: int) -> Callable[[str, Optional[str]], Dict[str, Any]]:
    def _inner(symbol: str, exchange: Optional[str]) -> Dict[str, Any]:
        return td_time_series(symbol, interval, outputsize, exchange, "desc")

    return _inner


def _finalize_series_payload(
    attempts: List[Tuple[str, Optional[str]]],
    out_dir: str,
    name: str,
    interval: str,
    freshness_limit: Optional[float],
    payload: Dict[str, Any],
    attempt_memory: Optional[AttemptMemory] = None,
) -> Dict[str, Any]:
    data: Dict[str, Any] = dict(payload) if isinstance(payload, dict) else {"ok": False}
    if "raw" not in data or not isinstance(data.get("raw"), dict):
        data["raw"] = payload.get("raw") if isinstance(payload, dict) else {"values": []}
        if not isinstance(data["raw"], dict):
            data["raw"] = {"values": []}

    if freshness_limit is not None:
        data.setdefault("freshness_limit_seconds", freshness_limit)

    original_retries = data.get("freshness_retries")

    if data.get("ok"):
        data = _refresh_series_if_stale(
            attempts,
            interval,
            data,
            freshness_limit,
            attempt_memory=attempt_memory,
        )
        data = _prefer_existing_series(out_dir, name, data, freshness_limit)

    if original_retries is not None:
        data.setdefault("freshness_retries", original_retries)

    if not data.get("ok") or not data.get("raw", {}).get("values"):
        data = _reuse_previous_series_payload(out_dir, name, data, freshness_limit)

    if not data.get("fallback_previous_payload"):
        data.pop("fallback_reuse_count", None)

    save_series_payload(out_dir, name, data)
    return data


def _collect_series_payloads(
    attempts: List[Tuple[str, Optional[str]]],
    attempt_memory: Optional[AttemptMemory],
    out_dir: str,
) -> Dict[str, Dict[str, Any]]:
    if not SERIES_FETCH_PLAN:
        return {}

    workers = max(1, min(len(SERIES_FETCH_PLAN), TD_REQUEST_CONCURRENCY))
    results: Dict[str, Dict[str, Any]] = {}
    future_map = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, interval in SERIES_FETCH_PLAN:
            outputsize = SERIES_OUTPUT_SIZES.get(interval, 500)
            fetch_fn = _series_fetcher(interval, outputsize)
            freshness_limit = SERIES_FRESHNESS_LIMITS.get(interval)
            future = pool.submit(
                fetch_with_freshness,
                attempts,
                fetch_fn,
                freshness_limit=freshness_limit,
                max_refreshes=1,
                attempt_memory=attempt_memory,
            )
            future_map[future] = (name, interval, freshness_limit)

        for future in as_completed(future_map):
            name, interval, freshness_limit = future_map[future]
            try:
                payload = future.result()
            except Exception as exc:
                payload = {
                    "ok": False,
                    "error": f"series fetch failed: {exc}",
                    "retrieved_at_utc": now_utc(),
                    "raw": {"values": []},
                }
            results[name] = _finalize_series_payload(
                attempts,
                out_dir,
                name,
                interval,
                freshness_limit,
                payload,
                attempt_memory=attempt_memory,
            )
            result_payload = results[name]
            if isinstance(result_payload, dict):
                ok = bool(result_payload.get("ok"))
                values = []
                raw = result_payload.get("raw")
                if isinstance(raw, dict):
                    raw_values = raw.get("values")
                    if isinstance(raw_values, list):
                        values = raw_values
                if not ok or not values:
                    attempts_fmt = ", ".join(
                        f"{sym}@{exch}" if exch else sym for sym, exch in attempts
                    ) or "<none>"
                    LOGGER.warning(
                        "Series fetch degraded for %s (%s): ok=%s values=%d attempts=[%s] error=%s",
                        name,
                        interval,
                        ok,
                        len(values),
                        attempts_fmt,
                        result_payload.get("error") or result_payload.get("message"),
                    )

    return results


def _refresh_series_if_stale(
    attempts: List[Tuple[str, Optional[str]]],
    interval: str,
    payload: Dict[str, Any],
    freshness_limit: Optional[float],
    *,
    extra_attempts: int = 2,
    attempt_memory: Optional[AttemptMemory] = None,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    if not payload.get("ok"):
        return payload
    latency = _coerce_float(payload.get("latency_seconds"))
    if freshness_limit is None or latency is None or latency <= freshness_limit:
        return payload

    best_payload = dict(payload)
    best_latency = latency
    attempts_made = 0

    for _ in range(max(0, extra_attempts)):
        time.sleep(max(TD_RATE_LIMITER.current_delay, 0.2))
        refreshed = try_symbols(
            attempts,
            lambda s, ex: td_time_series(s, interval, 180, ex, "desc"),
            freshness_limit=None,
            attempt_memory=attempt_memory,
        )
        attempts_made += 1
        if not isinstance(refreshed, dict) or not refreshed.get("ok"):
            continue
        refreshed_latency = _coerce_float(refreshed.get("latency_seconds"))
        if refreshed_latency is not None:
            refreshed["latency_seconds"] = refreshed_latency
        if (
            refreshed_latency is not None
            and (best_latency is None or refreshed_latency < best_latency)
        ):
            best_payload = dict(refreshed)
            best_latency = refreshed_latency
        if (
            freshness_limit is not None
            and refreshed_latency is not None
            and refreshed_latency <= freshness_limit
        ):
            best_payload = dict(refreshed)
            best_payload["freshness_violation"] = False
            best_payload["stale_refresh_attempts"] = attempts_made
            return best_payload

    best_payload["stale_refresh_attempts"] = attempts_made
    return best_payload


def _prefer_existing_series(
    out_dir: str,
    name: str,
    payload: Dict[str, Any],
    freshness_limit: Optional[float],
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    if freshness_limit is None:
        return payload

    latency = _coerce_float(payload.get("latency_seconds"))
    violation = bool(payload.get("freshness_violation"))
    if latency is not None and freshness_limit is not None:
        violation = violation or latency > freshness_limit
    if not violation:
        return payload

    meta_path = os.path.join(out_dir, f"{name}_meta.json")
    raw_path = os.path.join(out_dir, f"{name}.json")
    existing_meta = load_json(meta_path)
    existing_raw = load_json(raw_path)
    if not isinstance(existing_meta, dict) or not isinstance(existing_raw, dict):
        return payload

    prev_latency = _coerce_float(existing_meta.get("latency_seconds"))
    existing_latest_iso = None
    if isinstance(existing_meta, dict):
        existing_latest_iso = existing_meta.get("latest_utc") or existing_meta.get("utc")
    recomputed_latency = _age_seconds_from_iso(existing_latest_iso)
    if recomputed_latency is None:
        recomputed_latency = prev_latency

    if recomputed_latency is None:
        return payload

    if freshness_limit is not None and recomputed_latency > freshness_limit:
        # The cached payload is also outside the allowed freshness budget – keep
        # the violation flag so downstream diagnostics see the stale state.
        violation = True

    if prev_latency is None or (freshness_limit is not None and prev_latency > freshness_limit):
        return payload
    values = existing_raw.get("values") if isinstance(existing_raw, dict) else None
    if not values:
        return payload

    merged = dict(payload)
    merged["raw"] = existing_raw
    merged["latency_seconds"] = recomputed_latency
    merged["latest_utc"] = existing_meta.get("latest_utc")
    merged["freshness_violation"] = bool(violation)
    merged["fallback_previous_payload"] = True
    merged.setdefault("freshness_limit_seconds", freshness_limit)
    if "used_symbol" not in merged and existing_meta.get("used_symbol"):
        merged["used_symbol"] = existing_meta.get("used_symbol")
    if "used_exchange" not in merged and existing_meta.get("used_exchange"):
        merged["used_exchange"] = existing_meta.get("used_exchange")
    return merged


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

    def _addon_message(source: Dict[str, Any]) -> Optional[Tuple[str, int]]:
        addon_value = None
        for key in ("request_access_via_add_on", "request_access_via_plan"):
            if source.get(key):
                addon_value = str(source.get(key))
                break
        if not addon_value:
            return None

        symbol = source.get("symbol")
        data_field = source.get("data")
        if isinstance(data_field, dict):
            symbol = data_field.get("symbol", symbol)
        elif isinstance(data_field, list):
            for item in data_field:
                if isinstance(item, dict) and item.get("symbol"):
                    symbol = item.get("symbol")
                    break

        message = source.get("message") or source.get("note") or _message_from(source)
        if not message:
            details = f" '{addon_value}'" if addon_value else ""
            target = f" {symbol}" if symbol else ""
            message = f"Twelve Data add-on{details} required{target}".strip()
        return message, 451

    addon_details = _addon_message(payload)
    if addon_details:
        return addon_details

    if isinstance(meta, dict):
        addon_details = _addon_message(meta)
        if addon_details:
            return addon_details

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
            request_fn = _REQUEST_SESSION.get
            if getattr(requests, "get", None) is not _ORIGINAL_REQUESTS_GET:
                request_fn = requests.get
            with _REQUEST_SEMAPHORE:
                response = request_fn(
                    f"{TD_BASE}/{path}",
                    params=params,
                    timeout=30,
                )
            response.raise_for_status()
            data = response.json()
            error_message, error_code = _td_error_details(data)
            if error_message:
                if error_code and str(error_code) not in error_message:
                    message = f"{error_message} (code {error_code})"
                else:
                    message = error_message
                throttled = (error_code == 429) or ("limit" in message.lower())
                retry_after_hint = None
                if response is not None:
                    retry_after_hint = _parse_retry_after(response.headers.get("Retry-After"))
                TD_RATE_LIMITER.record_failure(
                    throttled=throttled,
                    retry_after=retry_after_hint,
                )
                td_error = TDError(message, status_code=error_code, throttled=throttled)
                last_error = td_error
                effective_status = error_code if error_code is not None else (
                    response.status_code if response is not None else None
                )
                last_status = effective_status
                if error_code in {400, 404, 422}:
                    raise td_error
                if attempt == TD_MAX_RETRIES:
                    raise td_error
            else:
                TD_RATE_LIMITER.record_success()
                return data
        except requests.HTTPError as exc:
            last_error = exc
            last_status = exc.response.status_code if exc.response else None
            retry_after_hint = None
            if exc.response is not None:
                retry_after_hint = _parse_retry_after(exc.response.headers.get("Retry-After"))
            throttled = last_status in {429, 503}
            TD_RATE_LIMITER.record_failure(
                throttled=throttled,
                retry_after=retry_after_hint,
            )
            if last_status and last_status < 500 and not throttled:
                raise TDError(str(exc), status_code=last_status, throttled=throttled) from exc
            if attempt == TD_MAX_RETRIES:
                if last_status and last_status < 500:
                    raise TDError(str(exc), status_code=last_status, throttled=throttled) from exc
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
    raise TDError(
        f"TD request failed after {TD_MAX_RETRIES} attempts{status_str}: {last_error}",
        status_code=last_status,
    )

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
    except TDError as exc:
        return {
            "used_symbol": symbol,
            "asset": symbol,
            "interval": interval,
            "source": "twelvedata:time_series",
            "ok": False,
            "retrieved_at_utc": now_utc(),
            "error": str(exc),
            "error_code": exc.status_code,
            "error_throttled": exc.throttled,
            "raw": {"values": []},
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
    except TDError as exc:
        q = {
            "ok": False,
            "error": str(exc),
            "error_code": exc.status_code,
            "error_throttled": exc.throttled,
        }
    except Exception as e:
        q = {"ok": False, "error": str(e)}

    price = q.get("price") if isinstance(q, dict) else None
    utc = q.get("utc") if isinstance(q, dict) else None
    fallback_used = False

    if price is None:
        try:
            px, ts = td_last_close(symbol, "5min", exchange)
        except TDError as exc:
            px, ts = None, None
            if isinstance(q, dict):
                q["error_fallback"] = str(exc)
                q["error_code"] = exc.status_code or q.get("error_code")
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
    *,
    force: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    frames: List[Dict[str, Any]] = []
    sample_cap = max(1, int(max_samples))
    failure_cycles = 0
    max_failures = max(2, len(symbol_cycle)) if symbol_cycle else 2
    consecutive_client_errors = 0
    abort_reason: Optional[str] = None
    while time.time() < deadline and len(frames) < sample_cap:
        cycle_success = False
        for symbol, exchange in symbol_cycle:
            try:
                quote = td_quote(symbol, exchange)
            except TDError as exc:
                if exc.status_code in {400, 404}:
                    consecutive_client_errors += 1
                    if force and consecutive_client_errors >= 2 and not frames:
                        abort_reason = f"client_error_{exc.status_code or 'unknown'}"
                        LOGGER.info(
                            "Realtime HTTP sampling aborted for %s (exchange=%s) after repeated %s errors",
                            symbol,
                            exchange or "default",
                            exc.status_code,
                        )
                        break
                else:
                    consecutive_client_errors = 0
                continue
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
            consecutive_client_errors = 0
            break
        if abort_reason:
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
    return frames, abort_reason


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


def _collect_realtime_spot_impl(
    asset: str,
    symbol_cycle: List[Tuple[str, Optional[str]]],
    out_dir: str,
    force: bool = False,
    reason: Optional[str] = None,
) -> None:
    if not symbol_cycle:
        return

    ensure_dir(out_dir)
    interval = max(0.5, REALTIME_INTERVAL)
    http_max_samples = REALTIME_HTTP_MAX_SAMPLES
    initial_http_max_samples = http_max_samples
    http_budget_cap: Optional[int] = None
    duration = max(REALTIME_DURATION, interval)
    http_duration = max(interval, REALTIME_HTTP_DURATION)

    use_ws = REALTIME_WS_ENABLED and REALTIME_FLAG and not force
    if not use_ws:
        duration = min(duration, http_duration)
        if interval > 0:
            expected_cycles = max(1, int(math.ceil(http_duration / interval)))
            http_max_samples = max(1, min(http_max_samples, expected_cycles + 1))

    # Forced realtime collection (pl. spot fallback) should be quick – if we
    # cannot rely on the websocket path we fall back to a much shorter HTTP
    # sampling window so the trading pipeline does not block for a long
    # interval on every asset.  Korábban a 60 másodperces ablak minden kényszer
    # esetén extra perceket adott a futáshoz; az új 20 másodperces alap és a
    # még rövidebb force-ablak ezt hivatott megszüntetni.
    if force and not use_ws:
        interval = min(interval, 2.0)
        quick_window = max(interval * 2.0, 6.0)
        duration = min(duration, quick_window)
        http_max_samples = max(1, min(http_max_samples, 2))

    if not use_ws:
        budget_cap = REALTIME_HTTP_BUDGET_PER_ASSET
        if budget_cap is not None and budget_cap > 0:
            http_budget_cap = budget_cap
            http_max_samples = max(1, min(http_max_samples, budget_cap))

    frames: List[Dict[str, Any]] = []
    transport: Optional[str] = None
    abort_reason: Optional[str] = None

    deadline = time.time() + duration
    if use_ws:
        frames = _collect_ws_frames(asset, symbol_cycle, deadline)
        if frames:
            transport = "websocket"

    if not frames:
        remaining = max(0.0, deadline - time.time())
        if remaining > 0:
            deadline_http = time.time() + remaining
            frames, abort_reason = _collect_http_frames(
                symbol_cycle,
                deadline_http,
                interval,
                http_max_samples,
                force=force,
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
    if abort_reason:
        stats["http_abort_reason"] = abort_reason
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
    if http_budget_cap is not None:
        payload["sampling_budget_cap"] = http_budget_cap
        payload["sampling_capped"] = http_max_samples < initial_http_max_samples
    if abort_reason:
        payload["http_abort_reason"] = abort_reason
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


def collect_realtime_spot(
    asset: str,
    attempts: List[Tuple[str, Optional[str]]],
    out_dir: str,
    *,
    attempt_memory: Optional[AttemptMemory] = None,
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

    base_cycle = list(attempts) if attempts else []
    if attempt_memory is not None:
        filtered_cycle = [
            (sym, exch)
            for sym, exch in base_cycle
            if attempt_memory.is_blacklisted(sym, exch) is None
        ]
    else:
        filtered_cycle = base_cycle
    symbol_cycle = filtered_cycle or base_cycle
    if not symbol_cycle:
        return

    use_ws = REALTIME_WS_ENABLED and REALTIME_FLAG and not force
    http_background_enabled = (
        _REALTIME_HTTP_BACKGROUND_FLAG
        if _REALTIME_HTTP_BACKGROUND_FLAG is not None
        else True
    )
    run_async = http_background_enabled and not use_ws

    if run_async:
        thread = threading.Thread(
            target=_collect_realtime_spot_impl,
            args=(asset, symbol_cycle, out_dir, force, reason),
            name=f"td-realtime-{asset.lower()}",
            daemon=False,
        )
        thread.start()
        with _REALTIME_BACKGROUND_LOCK:
            REALTIME_BACKGROUND_THREADS.append(thread)
        return

    _collect_realtime_spot_impl(asset, symbol_cycle, out_dir, force=force, reason=reason)


def wait_for_realtime_background() -> None:
    while True:
        with _REALTIME_BACKGROUND_LOCK:
            if not REALTIME_BACKGROUND_THREADS:
                break
            threads = list(REALTIME_BACKGROUND_THREADS)
            REALTIME_BACKGROUND_THREADS.clear()
        for thread in threads:
            thread.join()

# ─────────────────────── több-szimbólumos fallback ───────────────────────

def _symbol_attempt_variants(
    symbol: str,
    exchange: Optional[str],
    *,
    allow_compact: bool = True,
) -> List[Tuple[str, Optional[str]]]:
    """Generate fallback symbol/exchange combinations for Twelve Data requests."""

    variants: List[Tuple[str, Optional[str]]] = []
    if not symbol:
        return variants

    base = (symbol, exchange)
    variants.append(base)

    if exchange:
        variants.append((symbol, None))
        if ":" not in symbol:
            exchange_str = str(exchange)
            if exchange_str and not any(ch.isspace() for ch in exchange_str):
                variants.append((f"{symbol}:{exchange_str}", None))
    elif ":" not in symbol and "/" not in symbol:
        variants.append((symbol, None))

    if allow_compact and "/" in symbol:
        compact = symbol.replace("/", "")
        if compact:
            variants.append((compact, exchange if exchange else None))
            if exchange is not None:
                variants.append((compact, None))

    return variants


def _normalize_symbol_attempts(cfg: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    base_symbol = cfg["symbol"]
    base_exchange = cfg.get("exchange")
    attempts: List[Tuple[str, Optional[str]]] = []
    allow_compact = not bool(cfg.get("disable_compact_variants"))

    def push(
        symbol: Optional[str],
        exchange: Optional[str],
        *,
        allow_compact_override: Optional[bool] = None,
    ) -> None:
        if not symbol:
            return
        compact_allowed = (
            allow_compact if allow_compact_override is None else allow_compact_override
        )
        attempts.extend(
            _symbol_attempt_variants(symbol, exchange, allow_compact=compact_allowed)
        )

    push(base_symbol, base_exchange)
    for alt in cfg.get("alt", []):
        if isinstance(alt, str):
            push(alt, base_exchange)
        elif isinstance(alt, dict):
            push(
                alt.get("symbol", base_symbol),
                alt.get("exchange", base_exchange),
                allow_compact_override=None
                if alt.get("disable_compact_variants") is None
                else not bool(alt.get("disable_compact_variants")),
            )
        elif isinstance(alt, (list, tuple)) and alt:
            symbol = alt[0]
            exchange = alt[1] if len(alt) > 1 else base_exchange
            push(symbol, exchange)

    seen: set[Tuple[str, Optional[str]]] = set()
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
    *,
    attempt_memory: Optional[AttemptMemory] = None,
):
    """Iterate over symbol candidates and prefer the freshest successful payload."""

    last: Optional[Dict[str, Any]] = None
    best: Optional[Dict[str, Any]] = None
    best_latency: Optional[float] = None

    for sym, exch in attempts:
        skip = False
        skip_reason: Optional[str] = None
        if attempt_memory is not None:
            skip, skip_reason, first_skip = attempt_memory.should_skip(sym, exch)
            if skip:
                if first_skip:
                    LOGGER.info(
                        "Skipping %s (exchange=%s) due to previous client error: %s",
                        sym,
                        exch or "default",
                        skip_reason or "",
                    )
                last = {
                    "ok": False,
                    "error": skip_reason or "client error blacklist",
                    "used_symbol": sym,
                    "used_exchange": exch,
                }
                continue
        try:
            result = fetch_fn(sym, exch)
        except TDError as exc:
            last = {"ok": False, "error": str(exc), "error_code": exc.status_code}
            LOGGER.warning(
                "Twelve Data request error for %s (exchange=%s): %s",
                sym,
                exch or "default",
                exc,
            )
            if (
                attempt_memory is not None
                and exc.status_code in CLIENT_ERROR_STATUS_CODES
            ):
                attempt_memory.record_hard_failure(sym, exch, str(exc))
            time.sleep(max(TD_RATE_LIMITER.current_delay * 0.5, 0.1))
            continue
        except Exception as exc:
            last = {"ok": False, "error": str(exc)}
            LOGGER.warning(
                "Twelve Data request error for %s (exchange=%s): %s",
                sym,
                exch or "default",
                exc,
            )
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

        if attempt_memory is not None and isinstance(result, dict):
            error_code = _safe_int(result.get("error_code"))
            if error_code in CLIENT_ERROR_STATUS_CODES:
                reason = result.get("error") or result.get("message") or f"code {error_code}"
                attempt_memory.record_hard_failure(sym, exch, str(reason))

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

        if not result.get("ok"):
            LOGGER.warning(
                "Attempt returned no data for %s (exchange=%s, interval=%s): error=%s",
                sym,
                exch or "default",
                result.get("interval"),
                result.get("error") or result.get("message"),
            )

        time.sleep(max(TD_RATE_LIMITER.current_delay * 0.5, 0.1))

    if best is not None:
        return best
    return last or {"ok": False}


def _accept_market_closed_staleness(
    payload: Optional[Dict[str, Any]],
    freshness_limit: Optional[float],
) -> bool:
    if not isinstance(payload, dict):
        return False
    if not payload.get("ok") or not payload.get("freshness_violation"):
        return False
    if freshness_limit is None:
        return False
    source = str(payload.get("source") or "")
    if "time_series" not in source:
        return False
    latest_iso = payload.get("latest_utc") or payload.get("utc")
    age = _age_seconds_from_iso(latest_iso)
    asset_key, asset_cfg = _resolve_asset_from_payload(payload)
    if age is None:
        return False
    if MARKET_CLOSED_MAX_AGE_SECONDS and age > MARKET_CLOSED_MAX_AGE_SECONDS:
        if asset_key:
            LOGGER.warning(
                "Rejecting market-closed fallback for %s: latency %.1f min exceeds hard cap %.1f min",
                asset_key,
                age / 60.0,
                MARKET_CLOSED_MAX_AGE_SECONDS / 60.0,
            )
        return False
    latest_dt = _parse_iso_utc(latest_iso)
    now_dt = datetime.now(timezone.utc)
    if latest_dt is None:
        return False
    closed_reason = _asset_market_closed_reason(asset_key, asset_cfg, latest_dt, now_dt)
    if closed_reason is None:
        # Fall back to the historical behaviour for assets without explicit
        # trading hours metadata: treat multi-hour gaps around the weekend as
        # acceptable market closures.
        if age < max(MARKET_CLOSED_GRACE_SECONDS, 0.0):
            return False
        if latest_dt.weekday() < 5 and now_dt.weekday() < 5:
            return False
        closed_reason = "weekend"
    payload["freshness_violation"] = False
    payload.setdefault("freshness_limit_seconds", freshness_limit)
    payload["market_closed_assumed"] = True
    if age is not None:
        payload["latency_seconds"] = age
    if closed_reason == "weekend":
        payload["freshness_note"] = "market_closed_weekend"
    else:
        payload["freshness_note"] = "market_closed_outside_hours"
    if asset_key and "asset" not in payload:
        payload.setdefault("asset", asset_key)
    payload["market_closed_reason"] = closed_reason
    return True


def fetch_with_freshness(
    attempts: List[Tuple[str, Optional[str]]],
    fetch_fn: Callable[[str, Optional[str]], Dict[str, Any]],
    freshness_limit: Optional[float] = None,
    max_refreshes: int = 1,
    *,
    attempt_memory: Optional[AttemptMemory] = None,
):
    result = try_symbols(
        attempts,
        fetch_fn,
        freshness_limit=freshness_limit,
        attempt_memory=attempt_memory,
    )
    if _accept_market_closed_staleness(result, freshness_limit):
        if isinstance(result, dict):
            result.setdefault("freshness_retries", 0)
        return result
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
        result = try_symbols(
            attempts,
            fetch_fn,
            freshness_limit=freshness_limit,
            attempt_memory=attempt_memory,
        )
        if _accept_market_closed_staleness(result, freshness_limit):
            break
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
    attempt_memory = AttemptMemory()

    # 1) Spot (quote → 5m close fallback), több tickerrel
    spot = fetch_with_freshness(
        attempts,
        lambda s, ex: td_spot_with_fallback(s, ex),
        freshness_limit=SPOT_FRESHNESS_LIMIT,
        max_refreshes=1,
        attempt_memory=attempt_memory,
    )
    if isinstance(spot, dict):
        spot.setdefault("freshness_limit_seconds", SPOT_FRESHNESS_LIMIT)
    else:
        spot = {"ok": False, "freshness_limit_seconds": SPOT_FRESHNESS_LIMIT}
    if isinstance(spot, dict):
        spot = _reuse_previous_spot(adir, spot, SPOT_FRESHNESS_LIMIT)
    _write_spot_payload(adir, asset, spot)

    spot_violation = bool(spot.get("freshness_violation")) if isinstance(spot, dict) else False
    force_reason: Optional[str] = None
    if not spot.get("ok"):
        force_reason = "spot_error"
    elif spot_violation:
        force_reason = "spot_stale"
    elif spot.get("fallback_used") or spot.get("fallback_previous_payload"):
        force_reason = "spot_fallback"
    series_payloads = _collect_series_payloads(attempts, attempt_memory, adir)
    k5 = series_payloads.get("klines_5m")
    if not isinstance(k5, dict):
        k5 = {
            "ok": False,
            "raw": {"values": []},
            "freshness_limit_seconds": SERIES_FRESHNESS_LIMITS.get("5min"),
        }

    _record_asset_status(asset, attempts, spot, series_payloads, attempt_memory=attempt_memory)

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

    collect_realtime_spot(
        asset,
        attempts,
        adir,
        attempt_memory=attempt_memory,
        force=bool(force_reason),
        reason=force_reason,
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
        _record_asset_failure(asset, str(error))


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
    workers = max(1, min(len(ASSETS), TD_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_asset_guard, asset, cfg): asset
            for asset, cfg in ASSETS.items()
        }
        for future in as_completed(futures):
            future.result()
    wait_for_realtime_background()
    completed_at_dt = datetime.now(timezone.utc)
    duration_seconds = max((completed_at_dt - started_at_dt).total_seconds(), 0.0)
    logger.info("Trading run completed in %.1f seconds", duration_seconds)
    try:
        _write_trading_status_summary(OUT_DIR)
    except Exception as exc:
        logger.warning("Failed to write trading status summary: %s", exc)
    try:
        record_trading_run(started_at=started_at_dt, completed_at=completed_at_dt, duration_seconds=duration_seconds)
    except Exception as exc:
        logger.warning("Failed to record trading pipeline metrics: %s", exc)
    try:
        warning_summary = summarize_pipeline_warnings()
        warning_lines = warning_summary.get("warning_lines") or 0
        client_lines = warning_summary.get("client_error_lines") or 0
        ratio = warning_summary.get("client_error_ratio") or 0.0
        if warning_lines:
            logger.info(
                "Pipeline warnings=%d client_errors=%d ratio=%.3f",
                warning_lines,
                client_lines,
                ratio,
            )
    except Exception as exc:
        logger.warning("Failed to summarize pipeline warnings: %s", exc)

if __name__ == "__main__":
    main()












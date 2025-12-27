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

import os, json, time, math, logging, shutil
import threading
from datetime import datetime, timezone, time as dt_time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, List, Tuple, Callable, Set

import requests
from requests.adapters import HTTPAdapter

from logging_utils import ensure_json_file_handler

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
        get_run_logging_context,
        record_trading_run,
        get_pipeline_log_path,
        summarize_pipeline_warnings,
    )
except Exception:  # pragma: no cover - optional helper
    def get_run_logging_context(*_args, **_kwargs):
        return {}

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
TD_RESPONSE_CACHE_TTL = max(5.0, float(os.getenv("TD_RESPONSE_CACHE_TTL", "90")))
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
FORCED_SNAPSHOT_MAX_AGE = 300.0
RESET_PUBLIC_ON_TRADING_START = os.getenv("RESET_PUBLIC_ON_TRADING_START", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_SYMBOL_META_DISABLED = os.getenv("TD_DISABLE_SYMBOL_META", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
FINNHUB_BASE_URL = os.getenv("FINNHUB_BASE_URL", "https://finnhub.io/api/v1").rstrip("/")
FINNHUB_TIMEOUT = max(1.0, _env_float("FINNHUB_TIMEOUT", 4.0))
_FINNHUB_DISABLED_RAW = os.getenv("DISABLE_FINNHUB_FALLBACK", "")
FINNHUB_SYMBOL_MAP = {
    "EURUSD": "OANDA:EUR_USD",
    "GOLD_CFD": "OANDA:XAU_USD",
    "XAGUSD": "OANDA:XAG_USD",
    "BTCUSD": "COINBASE:BTC-USD",
    "USOIL": "OANDA:WTICO_USD",
    "NVDA": "NVDA",
}
_FINNHUB_SESSION = requests.Session()
_FINNHUB_SESSION.headers.update({"User-Agent": "market-feed/finnhub-fallback/1.0"})

# ───────────────────────────── Data freshness guards ──────────────────────
# Align the freshness limits with ``analysis.py`` tolerances so we can fall
# back to alternative symbols whenever the primary feed lags behind.  The
# values are intentionally generous – if every symbol is stale we still return
# the least-delayed payload instead of failing the pipeline.
SERIES_FRESHNESS_LIMITS = {
    "1min": 240.0,  # 1m candle  4m tolerance
    "5min": 900.0,  # 5m candle  15m tolerance
    "1h": 5400.0,  # 1h candle  90m tolerance
    "4h": 21600.0,  # 4h candle  6h tolerance
}
try:
    from config.analysis_settings import (
        SPOT_MAX_AGE_SECONDS as _ANALYSIS_SPOT_MAX_AGE_SECONDS,
        resolve_session_status_for_asset as _resolve_session_status_for_asset,
    )
except Exception:  # pragma: no cover - optional dependency during tests
    _ANALYSIS_SPOT_MAX_AGE_SECONDS = {}

    def _resolve_session_status_for_asset(asset: str):
        return "default", {}

_SPOT_FRESHNESS_ENV = os.getenv("TD_SPOT_FRESHNESS_LIMIT")
_SPOT_FRESHNESS_DEFAULT = float(_SPOT_FRESHNESS_ENV) if _SPOT_FRESHNESS_ENV else 900.0
_SPOT_FRESHNESS_OVERRIDES: Dict[str, float] = {}

if isinstance(_ANALYSIS_SPOT_MAX_AGE_SECONDS, dict):
    cfg_default = _ANALYSIS_SPOT_MAX_AGE_SECONDS.get("default")
    if cfg_default is not None and _SPOT_FRESHNESS_ENV is None:
        try:
            _SPOT_FRESHNESS_DEFAULT = float(cfg_default)
        except (TypeError, ValueError):
            pass
    for name, value in _ANALYSIS_SPOT_MAX_AGE_SECONDS.items():
        if name is None:
            continue
        key = str(name).strip()
        if not key:
            continue
        if key.lower() == "default":
            continue
        try:
            _SPOT_FRESHNESS_OVERRIDES[key.upper()] = float(value)
        except (TypeError, ValueError):
            continue


def _spot_freshness_limit(asset: str) -> float:
    key = str(asset or "").upper()
    return _SPOT_FRESHNESS_OVERRIDES.get(key, _SPOT_FRESHNESS_DEFAULT)


# Backwards compatibility: retain the module level constant for external callers
# that previously imported ``Trading.SPOT_FRESHNESS_LIMIT``.  The default value
# continues to mirror the synchronised analysis threshold while per-asset
# overrides are served through ``_spot_freshness_limit``.
SPOT_FRESHNESS_LIMIT = _SPOT_FRESHNESS_DEFAULT

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

_SERIES_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_SERIES_REPLAY_DIR_RAW = os.getenv("TD_SERIES_REPLAY_DIR")
_SERIES_REPLAY_DIR = (
    Path(_SERIES_REPLAY_DIR_RAW).expanduser() if _SERIES_REPLAY_DIR_RAW else None
)
# Only enable replay mode when the directory is explicitly configured and exists.
_SERIES_REPLAY_ENABLED = bool(_SERIES_REPLAY_DIR and _SERIES_REPLAY_DIR.exists())


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
        "trading_schedule": "24/7",
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
        "exchange_timezone": "Australia/Sydney",
        "mic": "COMMODITY",
        "currency": "USD",
        "disable_compact_variants": True,
        "alt": [
            {
                "symbol": "WTI/USD",
                "exchange": "NYMEX",
                "note": "Twelve Data NYMEX crude fallback",
            },
            {
                "symbol": "WTI/USD",
                "exchange": "ICE",
                "note": "Twelve Data ICE crude fallback",
            },
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
        "alt": [
            {"symbol": "NVDA", "exchange": "XNGS"},
            {"symbol": "NVDA", "exchange": "XNAS"},
            "NVDA:XNGS",
            "NVDA:XNAS",
        ],
    },
    "XAGUSD": {
        # Lásd: https://twelvedata.com/symbols (Metals) – a dokumentált ticker a
        # "XAG/USD" formátum, amelyet a Twelve Data REST API ismer.
        "symbol": "XAG/USD",
        "name": "Silver / US Dollar",
        "asset_class": "Commodity",
        "exchange": "COMMODITY",
        "exchange_display": "Commodity",
        "exchange_timezone": "Australia/Sydney",
        "mic": "COMMODITY",
        "currency": "USD",
        "disable_compact_variants": True,
        "disable_exchange_fallbacks": True,
        "alt": [
            {
                "symbol": "XAG/USD",
                "exchange": None,
                "disable_compact_variants": True,
                "disable_exchange_fallbacks": True,
            },
            {
                "symbol": "XAG/USD",
                "exchange": "FOREXCOM",
                "note": "Twelve Data metals fallback",
            },
            {
                "symbol": "XAG/USD",
                "exchange": "METAL",
                "note": "Twelve Data metals fallback",
            },
        ],
    },
}

_ASSET_FILTER_ENV = os.getenv("TD_ASSET_FILTER", "").strip()
if _ASSET_FILTER_ENV:
    _asset_filter = {
        item.strip().upper()
        for item in _ASSET_FILTER_ENV.split(",")
        if item and item.strip()
    }
    if _asset_filter:
        filtered_assets = {
            key: value
            for key, value in ASSETS.items()
            if key.upper() in _asset_filter
        }
        missing_assets = sorted(_asset_filter - {key.upper() for key in ASSETS})
        if missing_assets:
            LOGGER.warning(
                "TD_ASSET_FILTER tartalmaz ismeretlen eszközöket: %s",
                ", ".join(missing_assets),
            )
        if not filtered_assets:
            raise SystemExit("TD_ASSET_FILTER nem tartalmaz érvényes eszközt")
        LOGGER.info(
            "TD_ASSET_FILTER aktív, %d eszköz feldolgozása: %s",
            len(filtered_assets),
            ", ".join(sorted(filtered_assets.keys())),
        )
        ASSETS = filtered_assets


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


def _is_continuous_trading_asset(cfg: Dict[str, Any]) -> bool:
    trading_schedule = str(cfg.get("trading_schedule") or "").strip().lower()
    if trading_schedule in {"24/7", "247", "continuous"}:
        return True
    asset_class = str(cfg.get("asset_class") or "").strip().lower()
    return asset_class == "crypto"


def _asset_market_closed_reason(
    asset_key: Optional[str],
    cfg: Optional[Dict[str, Any]],
    latest_dt: datetime,
    now_dt: datetime,
) -> Optional[str]:
    if cfg and _is_continuous_trading_asset(cfg):
        return None
    if now_dt.weekday() >= 5 or latest_dt.weekday() >= 5:
        return "weekend"
    if not cfg:
        return None
    if _is_us_equity_asset(cfg):
        now_time = now_dt.time()
        if now_time < US_EQUITY_OPEN_UTC or now_time >= US_EQUITY_CLOSE_UTC:
            return "outside_hours"
    return None


def _market_closed_skip_reason(asset_key: str, cfg: Dict[str, Any]) -> Optional[str]:
    """Return a market-closed reason when requests should be skipped."""

    if not asset_key or not isinstance(cfg, dict):
        return None
    if not _is_us_equity_asset(cfg):
        return None
    now_dt = datetime.now(timezone.utc)
    return _asset_market_closed_reason(asset_key, cfg, now_dt, now_dt)


def _status_profile_for_asset(asset: str) -> Tuple[str, Dict[str, Any]]:
    now_dt = datetime.now(timezone.utc)
    name, profile = _resolve_session_status_for_asset(asset, when=now_dt)
    if not isinstance(profile, dict):
        profile = {}
    return name, dict(profile)


def _status_profile_skip_reason(profile: Dict[str, Any]) -> Optional[str]:
    if not isinstance(profile, dict) or not profile.get("force_session_closed"):
        return None
    reason = profile.get("market_closed_reason") or profile.get("status")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return "status_profile_forced"


def _apply_status_profile_metadata(
    payload: Dict[str, Any], profile_name: str, profile: Dict[str, Any]
) -> None:
    if not isinstance(payload, dict) or not isinstance(profile, dict):
        return

    payload["status_profile"] = profile_name or "default"
    if profile.get("force_session_closed"):
        payload["status_profile_forced"] = True

    tags = profile.get("tags")
    if isinstance(tags, list) and tags:
        clean_tags: List[str] = []
        for tag in tags:
            if isinstance(tag, str):
                cleaned = tag.strip()
                if cleaned and cleaned not in clean_tags:
                    clean_tags.append(cleaned)
        if clean_tags:
            payload["status_profile_tags"] = clean_tags

    context = profile.get("context")
    if isinstance(context, dict) and context:
        payload["status_profile_context"] = {str(key): value for key, value in context.items()}

    notes = profile.get("notes")
    if isinstance(notes, list) and notes:
        bucket = payload.setdefault("notes", [])
        for note in notes:
            if isinstance(note, str):
                cleaned = note.strip()
                if cleaned and cleaned not in bucket:
                    bucket.append(cleaned)

    if "market_closed_reason" in profile:
        payload["market_closed_reason"] = str(profile.get("market_closed_reason"))
    if "market_closed_assumed" in profile:
        payload["market_closed_assumed"] = bool(profile.get("market_closed_assumed"))

    next_open = profile.get("next_open_utc")
    if isinstance(next_open, str) and next_open.strip():
        payload["next_open_utc"] = next_open.strip()

    status_note = profile.get("status_note")
    if isinstance(status_note, str) and status_note.strip():
        payload["status_note"] = status_note.strip()

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
# Proxy-k kikapcsolása: a Twelve Data hívások egyes hálózatokban 403-as
# proxy hibát kaptak (miközben böngészőből közvetlenül működtek), ezért
# a requests Session ne vegye át az esetleges HTTP(S)_PROXY változókat.
_REQUEST_SESSION.trust_env = False
_REQUEST_SESSION.headers.update({"User-Agent": "market-feed/td-only/1.0"})
_ORIGINAL_REQUESTS_GET = requests.get
_TD_RESPONSE_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, Dict[str, Any]]] = {}
ANCHOR_LOCK = threading.Lock()
_REALTIME_BACKGROUND_LOCK = threading.Lock()
REALTIME_BACKGROUND_THREADS: List[threading.Thread] = []

# ───────────────────────────── Attempt memory ─────────────────────────────


_GLOBAL_SYMBOL_FAILURES: Dict[Tuple[str, Optional[str]], str] = {}
_GLOBAL_SYMBOL_FAILURE_LOCK = threading.Lock()


def _normalize_failure_reason(reason: str) -> str:
    return " ".join(str(reason).split())


def _remember_global_symbol_failure(
    symbol: str, exchange: Optional[str], reason: str
) -> None:
    key = (symbol, exchange)
    normalized = _normalize_failure_reason(reason)
    with _GLOBAL_SYMBOL_FAILURE_LOCK:
        existing = _GLOBAL_SYMBOL_FAILURES.get(key)
        if existing is None:
            _GLOBAL_SYMBOL_FAILURES[key] = normalized
        elif normalized and normalized not in existing:
            _GLOBAL_SYMBOL_FAILURES[key] = normalized


def _global_symbol_failure_reason(symbol: str, exchange: Optional[str]) -> Optional[str]:
    with _GLOBAL_SYMBOL_FAILURE_LOCK:
        return _GLOBAL_SYMBOL_FAILURES.get((symbol, exchange))


def _reset_global_symbol_failure_cache() -> None:
    with _GLOBAL_SYMBOL_FAILURE_LOCK:
        _GLOBAL_SYMBOL_FAILURES.clear()


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
        normalized_reason = _normalize_failure_reason(reason)
        with self._lock:
            if key not in self._hard_failures:
                self._hard_failures[key] = normalized_reason
            else:
                # Update the reason if the new message is more descriptive.
                existing = self._hard_failures[key]
                if normalized_reason and normalized_reason not in existing:
                    self._hard_failures[key] = normalized_reason
            self._skip_logged.discard(key)
        _remember_global_symbol_failure(symbol, exchange, normalized_reason)

    def should_skip(self, symbol: str, exchange: Optional[str]) -> Tuple[bool, Optional[str], bool]:
        key = (symbol, exchange)
        with self._lock:
            reason = self._hard_failures.get(key)
            if reason is None:
                global_reason = _global_symbol_failure_reason(symbol, exchange)
                if global_reason is not None:
                    self._hard_failures[key] = global_reason
                    reason = global_reason
                else:
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


class SymbolAttemptsExhausted(RuntimeError):
    """Raised when every Twelve Data attempt ends with client-side errors."""

    def __init__(self, failures: List[Dict[str, Any]]) -> None:
        self.failures = failures
        formatted = ", ".join(
            f"{entry.get('symbol')}@{entry.get('exchange') or 'default'} ({entry.get('status_code')})"
            for entry in failures
        )
        message = "All Twelve Data symbol attempts failed with client errors"
        if formatted:
            message = f"{message}: {formatted}"
        super().__init__(message)

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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _parse_iso_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def update_system_heartbeat(out_dir: str) -> None:
    heartbeat_path = Path(out_dir) / "system_heartbeat.json"
    payload = {
        "last_update_utc": now_utc(),
        "status": "running",
    }
    save_json(str(heartbeat_path), payload)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(str(target.parent))
    tmp_path = target.with_suffix(target.suffix + ".tmp")

    # ENV flags (defaults: fast  safe enough for CI)
    durable_flag = os.getenv("TD_DURABLE_WRITES", "0").strip().lower() in {"1", "true", "yes", "on"}
    skip_unchanged_flag = (
        os.getenv("TD_SKIP_UNCHANGED_KLINES_WRITES", "1").strip().lower() in {"1", "true", "yes", "on"}
    )

    # Serialize once
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # Only skip unchanged for heavy raw series files (preserve freshness semantics for spot/meta/signal)
    name = target.name
    is_heavy_raw_series = name.startswith("klines_") and name.endswith(".json") and (not name.endswith("_meta.json"))
    if skip_unchanged_flag and is_heavy_raw_series and target.exists():
        try:
            st = target.stat()
            if st.st_size == len(data):
                with target.open("rb") as rf:
                    if rf.read() == data:
                        return
        except Exception:
            pass

    with tmp_path.open("wb") as f:
        f.write(data)
        if durable_flag:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp_path, target)


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
    seen: Set[Tuple[str, Optional[str]]] = set()
    formatted: List[Dict[str, Optional[str]]] = []
    for symbol, exchange in attempts:
        key = (symbol, exchange)
        if key in seen:
            continue
        seen.add(key)
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
    sorted_series_items = sorted(series_payloads.items(), key=lambda item: item[0])
    for name, payload in sorted_series_items:
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
            for _, frame in sorted(series_info.items(), key=lambda kv: kv[0])
        ) if series_info else False
        if not (spot_ok and frames_ok):
            all_ready = False
            break

    pipeline_dir = os.path.join(out_dir, "pipeline")
    ensure_dir(pipeline_dir)
    payload = {
        "generated_at_utc": now_utc(),
        "run": get_run_logging_context(),
        "assets": assets,
        "all_assets_ready": all_ready,
    }
    save_json(os.path.join(pipeline_dir, "trading_status.json"), payload)


def _write_public_refresh_marker(
    out_dir: str,
    *,
    started_at: datetime,
    completed_at: datetime,
    duration_seconds: float,
) -> None:
    pipeline_dir = os.path.join(out_dir, "pipeline")
    ensure_dir(pipeline_dir)
    save_json(
        os.path.join(pipeline_dir, "public_refresh.json"),
        {
            "trading_started_at_utc": started_at.isoformat(),
            "trading_completed_at_utc": completed_at.isoformat(),
            "duration_seconds": duration_seconds,
            "out_dir": out_dir,
        },
    )


def _series_row_timestamp(row: Dict[str, Any]) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    for field in ("datetime", "time", "timestamp"):
        if field not in row:
            continue
        value = row.get(field)
        iso = _iso_from_td_ts(value)
        if iso:
            parsed = _parse_iso_utc(iso)
            if parsed is not None:
                return parsed.timestamp()
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    continue
                return float(stripped)
        except Exception:
            continue
    return None


def _sort_series_values(raw: Dict[str, Any]) -> Dict[str, Any]:
    values = raw.get("values") if isinstance(raw, dict) else None
    if not isinstance(values, list) or len(values) <= 1:
        return raw
    keyed: List[Tuple[float, int, Any]] = []
    has_timestamp = False
    for idx, row in enumerate(values):
        ts = _series_row_timestamp(row)
        if ts is not None:
            has_timestamp = True
            keyed.append((ts, idx, row))
        else:
            keyed.append((float("inf"), idx, row))
    if not has_timestamp:
        return raw
    keyed.sort(key=lambda item: (item[0], item[1]))
    sorted_values = [item[2] for item in keyed]
    new_raw = dict(raw)
    new_raw["values"] = sorted_values
    return new_raw


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
            raw = _sort_series_values(raw_candidate)
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
    outputsize: int,
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
        retries_done = 0
        try:
            retries_done = int(data.get("freshness_retries") or 0)
        except Exception:
            retries_done = 0
        if retries_done <= 0:
            data = _refresh_series_if_stale(
                attempts,
                interval,
                data,
                freshness_limit,
                attempt_memory=attempt_memory,
            )
        data = _prefer_existing_series(out_dir, name, data, freshness_limit)

    asset_name = os.path.basename(out_dir) or ""
    if asset_name:
        data = _maybe_use_secondary_series(
            asset_name,
            interval,
            data,
            freshness_limit,
            outputsize=outputsize,
        )

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
                raise_on_all_client_errors=True,
            )
            future_map[future] = (name, interval, freshness_limit, outputsize)

        for future in as_completed(future_map):
            name, interval, freshness_limit, outputsize = future_map[future]
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
                outputsize,
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
                  
    if results:
        ordered = {key: results[key] for key in sorted(results)}
        return ordered
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
            raise_on_all_client_errors=True,
        )
        attempts_made = 1
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


def _td_cache_key(path: str, params: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    normalized_params = []
    for key, value in params.items():
        if key == "apikey":
            continue
        normalized_params.append((str(key), str(value)))
    normalized_params.sort(key=lambda item: (item[0], item[1]))
    return path, tuple(normalized_params)


def _get_cached_td_response(path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = _td_cache_key(path, params)
    cached = _TD_RESPONSE_CACHE.get(key)
    if not cached:
        return None
    recorded_at, payload = cached
    age = max(0.0, time.monotonic() - recorded_at)
    if age > TD_RESPONSE_CACHE_TTL:
        _TD_RESPONSE_CACHE.pop(key, None)
        return None
    cached_payload = dict(payload)
    cached_payload.setdefault("ok", True)
    cached_payload.setdefault("from_cache", True)
    cached_payload["cache_age_seconds"] = age
    return cached_payload


def _store_cached_td_response(path: str, params: Dict[str, Any], payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    key = _td_cache_key(path, params)
    payload_copy = dict(payload)
    payload_copy.setdefault("retrieved_at_utc", now_utc())
    _TD_RESPONSE_CACHE[key] = (time.monotonic(), payload_copy)


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
    cached_response = _get_cached_td_response(path, params)
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
                _store_cached_td_response(path, params, data)
                return data
        except requests.HTTPError as exc:
            last_error = exc
            last_status = exc.response.status_code if exc.response else None
            retry_after_hint = None
            if exc.response is not None:
                retry_after_hint = _parse_retry_after(exc.response.headers.get("Retry-After"))
            throttled = last_status in {429, 503}
            cache_ready = bool(throttled and cached_response)
            TD_RATE_LIMITER.record_failure(
                throttled=throttled,
                retry_after=retry_after_hint,
            )
            if last_status and last_status < 500 and not throttled:
                raise TDError(str(exc), status_code=last_status, throttled=throttled) from exc
            if attempt == TD_MAX_RETRIES:
                if last_status and last_status < 500 and not throttled:
                    raise TDError(str(exc), status_code=last_status, throttled=throttled) from exc
                if last_status and last_status < 500 and not cache_ready:
                    raise TDError(str(exc), status_code=last_status, throttled=throttled) from exc
                if not cache_ready:
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
    if cached_response and (last_status is None or last_status >= 500 or last_status == 429):
        cached_response.setdefault("from_cache", True)
        cached_response.setdefault("ok", True)
        cached_response["cache_fallback_reason"] = status_str.strip() or "unknown"
        return cached_response
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


def _finnhub_available() -> bool:
    flag = _FINNHUB_DISABLED_RAW.strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    return bool(FINNHUB_API_KEY)


def _resolve_finnhub_symbol(asset: str, preferred_symbol: Optional[str] = None) -> Optional[str]:
    key = (asset or "").strip().upper()
    if not key:
        return None
    env_override = os.getenv(f"FINNHUB_SYMBOL_{key}", "").strip()
    if env_override:
        return env_override
    mapping = FINNHUB_SYMBOL_MAP.get(key)
    if mapping:
        return mapping
    if preferred_symbol:
        symbol = preferred_symbol.strip()
        if not symbol:
            return None
        if ":" in symbol:
            return symbol
        normalized = symbol.replace("/", "_").replace("-", "_")
        if key.endswith("USD"):
            return f"OANDA:{normalized}"
        return symbol
    return None


def _fetch_finnhub_spot(asset: str, *, preferred_symbol: Optional[str] = None) -> Dict[str, Any]:
    if not _finnhub_available():
        if not FINNHUB_API_KEY:
            return {"ok": False, "error": "FINNHUB_API_KEY not configured"}
        return {"ok": False, "error": "finnhub fallback disabled"}

    symbol = _resolve_finnhub_symbol(asset, preferred_symbol)
    if not symbol:
        return {"ok": False, "error": "finnhub symbol mapping missing"}
    if not FINNHUB_API_KEY:
        return {"ok": False, "error": "FINNHUB_API_KEY not configured"}

    params = {"symbol": symbol, "token": FINNHUB_API_KEY}
    url = f"{FINNHUB_BASE_URL}/quote"
    try:
        response = _FINNHUB_SESSION.get(url, params=params, timeout=FINNHUB_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Finnhub request failed: {exc}"}

    retrieved_at = now_utc()
    try:
        payload = response.json()
    except ValueError:
        return {"ok": False, "error": "Finnhub returned invalid JSON"}

    price = _coerce_float(
        (payload.get("c") if isinstance(payload, dict) else None)
        or (payload.get("p") if isinstance(payload, dict) else None)
        or (payload.get("lastPrice") if isinstance(payload, dict) else None)
        or (payload.get("last") if isinstance(payload, dict) else None)
        or (payload.get("close") if isinstance(payload, dict) else None)
    )
    utc_iso = retrieved_at
    latency: Optional[float] = None
    if isinstance(payload, dict):
        ts_value = payload.get("t")
        if isinstance(ts_value, (int, float)) and ts_value > 0:
            try:
                stamp = datetime.fromtimestamp(float(ts_value), timezone.utc).replace(microsecond=0)
            except (OverflowError, ValueError):
                stamp = None
            if stamp is not None:
                utc_iso = stamp.isoformat()
                latency = max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds())

    result: Dict[str, Any] = {
        "asset": asset,
        "source": "finnhub:quote",
        "ok": price is not None,
        "retrieved_at_utc": retrieved_at,
        "utc": utc_iso,
        "price": price,
        "price_usd": price,
        "latency_seconds": latency,
        "used_symbol": symbol,
    }
    if isinstance(payload, dict):
        exchange = payload.get("exchange")
        if exchange:
            result["used_exchange"] = exchange
        compact_raw = {k: payload.get(k) for k in ("c", "h", "l", "o", "pc", "t") if k in payload}
        if compact_raw:
            result["raw"] = compact_raw
    return result


def _maybe_use_secondary_spot(asset: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not asset:
        return payload

    if not isinstance(payload, dict):
        placeholder: Dict[str, Any] = {"ok": False}
        if payload is None:
            placeholder["error"] = "primary spot payload missing"
        else:
            placeholder["error"] = "primary spot payload invalid"
            placeholder["primary_payload_type"] = type(payload).__name__
        payload = placeholder

    if not _finnhub_available():
        return payload

    limit = _coerce_float(payload.get("freshness_limit_seconds"))
    latency = _coerce_float(payload.get("latency_seconds"))
    needs_fallback = False
    reason: Optional[str] = None

    if not payload.get("ok") or payload.get("price") is None:
        needs_fallback = True
        reason = payload.get("error") or payload.get("message") or "primary feed unavailable"
    elif bool(payload.get("freshness_violation")):
        needs_fallback = True
        reason = "primary feed stale"
    elif limit is not None and latency is not None and latency > limit:
        needs_fallback = True
        reason = "primary feed stale"

    if not needs_fallback:
        return payload

    fallback = _fetch_finnhub_spot(asset, preferred_symbol=payload.get("used_symbol"))
    if not fallback.get("ok"):
        attempts = payload.setdefault("fallback_attempts", [])
        if isinstance(attempts, list):
            attempts.append(
                {
                    "provider": "finnhub",
                    "ok": False,
                    "error": fallback.get("error"),
                }
            )
        LOGGER.warning(
            "Finnhub fallback failed for %s: %s",
            asset,
            fallback.get("error") or "unknown error",
        )
        return payload

    fallback_latency = _coerce_float(fallback.get("latency_seconds"))
    primary_latency = latency
    fallback_limit = limit if limit is not None else _coerce_float(fallback.get("freshness_limit_seconds"))
    if fallback_limit is not None:
        fallback["freshness_limit_seconds"] = fallback_limit
        if fallback_latency is not None:
            fallback["freshness_violation"] = fallback_latency > fallback_limit
        else:
            fallback.setdefault("freshness_violation", False)
    else:
        fallback.setdefault("freshness_violation", False)

    fallback_violation = bool(fallback.get("freshness_violation"))
    if payload.get("ok") and primary_latency is not None and fallback_latency is not None:
        if fallback_violation and fallback_latency >= primary_latency:
            LOGGER.info(
                "Finnhub fallback skipped for %s — latency %.1fs >= primary %.1fs",
                asset,
                fallback_latency,
                primary_latency,
            )
            return payload

    if not fallback.get("used_exchange") and payload.get("used_exchange"):
        fallback["used_exchange"] = payload.get("used_exchange")

    fallback["fallback_used"] = True
    fallback["fallback_provider"] = "finnhub"
    if reason:
        fallback["fallback_reason"] = reason
    fallback["primary_source"] = payload.get("source")
    fallback["primary_latency_seconds"] = primary_latency
    fallback["primary_freshness_violation"] = bool(payload.get("freshness_violation"))
    fallback.setdefault("price_usd", fallback.get("price"))

    latency_display = fallback_latency if fallback_latency is not None else float("nan")
    LOGGER.info(
        "Finnhub fallback engaged for %s (reason: %s, latency %.1fs)",
        asset,
        reason or "unspecified",
        latency_display,
    )
    return fallback


def _finnhub_series_resolution(interval: str) -> Optional[Tuple[str, int]]:
    key = (interval or "").strip().lower()
    mapping = {
        "1min": ("1", 60),
        "1m": ("1", 60),
        "5min": ("5", 300),
        "5m": ("5", 300),
        "1h": ("60", 3600),
        "60min": ("60", 3600),
        "4h": ("240", 14400),
        "240min": ("240", 14400),
    }
    return mapping.get(key)


def _finnhub_series_endpoint(symbol: Optional[str]) -> str:
    if not symbol:
        return "stock/candle"
    prefix = symbol.split(":", 1)[0].strip().upper()
    if prefix in {"OANDA", "FXCM", "FOREX", "ICMARKETS", "SAXO"}:
        return "forex/candle"
    if prefix in {"COINBASE", "COINBASEPRO", "BINANCE", "BITSTAMP", "KRAKEN", "GEMINI", "HUOBI"}:
        return "crypto/candle"
    return "stock/candle"


def _fetch_finnhub_series(
    asset: str,
    interval: str,
    *,
    preferred_symbol: Optional[str] = None,
    limit: int = 300,
    freshness_limit: Optional[float] = None,
) -> Dict[str, Any]:
    if not _finnhub_available():
        return {"ok": False, "error": "finnhub fallback disabled"}
    symbol = _resolve_finnhub_symbol(asset, preferred_symbol)
    if not symbol:
        return {"ok": False, "error": "finnhub symbol mapping missing"}
    if not FINNHUB_API_KEY:
        return {"ok": False, "error": "FINNHUB_API_KEY not configured"}

    resolution_info = _finnhub_series_resolution(interval)
    if not resolution_info:
        return {"ok": False, "error": f"unsupported interval {interval}"}
    resolution, interval_seconds = resolution_info
    limit = max(int(limit), 1)
    now_epoch = int(time.time())
    span = max(interval_seconds, interval_seconds * limit)
    start_epoch = max(0, now_epoch - span - interval_seconds)
    endpoint = _finnhub_series_endpoint(symbol)
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": start_epoch,
        "to": now_epoch,
        "token": FINNHUB_API_KEY,
    }
    url = f"{FINNHUB_BASE_URL}/{endpoint}"
    try:
        response = _FINNHUB_SESSION.get(url, params=params, timeout=FINNHUB_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Finnhub request failed: {exc}"}

    retrieved_at = now_utc()
    try:
        data = response.json()
    except ValueError:
        return {"ok": False, "error": "Finnhub returned invalid JSON"}

    if not isinstance(data, dict):
        return {"ok": False, "error": "Finnhub returned malformed payload"}

    status = str(data.get("s") or data.get("status") or "").lower()
    if status and status != "ok":
        return {"ok": False, "error": f"Finnhub returned status {status or 'unknown'}"}

    timestamps = data.get("t") or []
    closes = data.get("c") or []
    opens = data.get("o") or []
    highs = data.get("h") or []
    lows = data.get("l") or []
    volumes = data.get("v") or []

    rows: List[Dict[str, Any]] = []
    count = min(
        len(timestamps),
        len(closes),
        len(opens),
        len(highs),
        len(lows),
        len(volumes),
    )
    for idx in range(count):
        ts = timestamps[idx]
        try:
            stamp = datetime.fromtimestamp(float(ts), timezone.utc).replace(microsecond=0)
        except (ValueError, OSError, OverflowError, TypeError):
            continue
        iso = stamp.isoformat()
        open_v = _coerce_float(opens[idx])
        high_v = _coerce_float(highs[idx])
        low_v = _coerce_float(lows[idx])
        close_v = _coerce_float(closes[idx])
        volume_v = _coerce_float(volumes[idx])
        row: Dict[str, Any] = {"datetime": iso}
        if open_v is not None:
            row["open"] = f"{open_v:.6f}"
        if high_v is not None:
            row["high"] = f"{high_v:.6f}"
        if low_v is not None:
            row["low"] = f"{low_v:.6f}"
        if close_v is not None:
            row["close"] = f"{close_v:.6f}"
        if volume_v is not None:
            row["volume"] = f"{max(volume_v, 0.0):.6f}"
        rows.append(row)

    if not rows:
        return {"ok": False, "error": "Finnhub returned empty candles"}

    # Finnhub lists candles oldest → newest; convert to desc ordering to match TD.
    rows.sort(key=lambda item: item.get("datetime") or "")
    rows_desc = list(reversed(rows))

    latest_iso = rows_desc[0].get("datetime") if rows_desc else None
    latency_seconds: Optional[float] = None
    if latest_iso:
        parsed = _parse_iso_utc(latest_iso)
        if parsed:
            latency_seconds = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())

    payload: Dict[str, Any] = {
        "asset": asset,
        "interval": interval,
        "source": "finnhub:candle",
        "ok": True,
        "retrieved_at_utc": retrieved_at,
        "latest_utc": latest_iso,
        "latency_seconds": latency_seconds,
        "raw": {"values": rows_desc},
        "used_symbol": symbol,
    }

    if ":" in symbol:
        payload["used_exchange"] = symbol.split(":", 1)[0]

    if freshness_limit is not None:
        payload["freshness_limit_seconds"] = freshness_limit
        if latency_seconds is not None:
            payload["freshness_violation"] = latency_seconds > freshness_limit
        else:
            payload.setdefault("freshness_violation", False)
    else:
        payload.setdefault("freshness_violation", False)

    return payload


def _maybe_use_secondary_series(
    asset: str,
    interval: str,
    payload: Dict[str, Any],
    freshness_limit: Optional[float],
    *,
    outputsize: int,
) -> Dict[str, Any]:
    if interval not in {"1min", "5min"}:
        return payload
    if not _finnhub_available():
        return payload

    if not isinstance(payload, dict):
        payload = {"ok": False, "raw": {"values": []}}
    else:
        payload = dict(payload)

    if payload.get("market_closed_assumed"):
        return payload

    reason: Optional[str] = None
    raw_block = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
    values = raw_block.get("values") if isinstance(raw_block, dict) else None

    if not payload.get("ok"):
        reason = payload.get("error") or payload.get("message") or "primary feed unavailable"
    elif not values:
        reason = "primary feed empty"
    else:
        violation = bool(payload.get("freshness_violation"))
        latency = _coerce_float(payload.get("latency_seconds"))
        if not violation and freshness_limit is not None and latency is not None:
            violation = latency > freshness_limit
        if violation:
            reason = "primary feed stale"

    if not reason:
        return payload

    fallback = _fetch_finnhub_series(
        asset,
        interval,
        preferred_symbol=payload.get("used_symbol"),
        limit=outputsize,
        freshness_limit=freshness_limit,
    )

    if not fallback.get("ok") or not isinstance(fallback.get("raw"), dict) or not fallback["raw"].get("values"):
        attempts = payload.setdefault("fallback_attempts", [])
        if isinstance(attempts, list):
            attempts.append(
                {
                    "provider": "finnhub",
                    "ok": False,
                    "error": fallback.get("error"),
                }
            )
        LOGGER.warning(
            "Finnhub series fallback failed for %s (%s): %s",
            asset,
            interval,
            fallback.get("error") or "unknown error",
        )
        return payload

    fallback_latency = _coerce_float(fallback.get("latency_seconds"))
    primary_latency = _coerce_float(payload.get("latency_seconds"))
    fallback_violation = bool(fallback.get("freshness_violation"))

    if reason == "primary feed stale" and fallback_violation:
        if primary_latency is None or (
            fallback_latency is not None and primary_latency is not None and fallback_latency >= primary_latency
        ):
            LOGGER.info(
                "Finnhub series fallback skipped for %s (%s) — latency %.1fs >= primary %.1fs",
                asset,
                interval,
                fallback_latency if fallback_latency is not None else float("nan"),
                primary_latency if primary_latency is not None else float("nan"),
            )
            return payload

    if payload.get("used_exchange") and not fallback.get("used_exchange"):
        fallback["used_exchange"] = payload.get("used_exchange")

    if freshness_limit is not None:
        fallback.setdefault("freshness_limit_seconds", freshness_limit)

    fallback["fallback_used"] = True
    fallback["fallback_provider"] = "finnhub"
    fallback["fallback_reason"] = reason
    fallback["primary_source"] = payload.get("source")
    fallback["primary_latency_seconds"] = primary_latency
    fallback["primary_freshness_violation"] = bool(payload.get("freshness_violation"))
    fallback.setdefault("interval", interval)

    attempts_source = payload.get("fallback_attempts")
    if isinstance(attempts_source, list) and not fallback.get("fallback_attempts"):
        fallback["fallback_attempts"] = list(attempts_source)
    if "freshness_retries" in payload and "freshness_retries" not in fallback:
        fallback["freshness_retries"] = payload.get("freshness_retries")

    LOGGER.info(
        "Finnhub series fallback engaged for %s (%s) — reason: %s, latency %.1fs",
        asset,
        interval,
        reason,
        fallback_latency if fallback_latency is not None else float("nan"),
    )

    return fallback
def td_time_series(symbol: str, interval: str, outputsize: int = 500,
                   exchange: Optional[str] = None, order: str = "desc") -> Dict[str, Any]:
    """Hibatűrő time_series (hiba esetén ok:false  üres values)."""

    cache_key = (symbol, interval, exchange or "", int(outputsize), order or "desc")
    if cache_key in _SERIES_CACHE:
        return _SERIES_CACHE[cache_key]

    if _SERIES_REPLAY_ENABLED:
        replay_path = _SERIES_REPLAY_DIR / f"{symbol}_{interval}.json"
        try:
            if replay_path.exists():
                payload = load_json(str(replay_path)) or {}
                payload.setdefault("source", "replay")
                payload.setdefault("ok", bool(payload.get("raw")))
                payload.setdefault("interval", interval)
                payload.setdefault("used_symbol", symbol)
                payload.setdefault("retrieved_at_utc", now_utc())
                _SERIES_CACHE[cache_key] = payload
                return payload
        except Exception:
            LOGGER.debug("replay_series_load_failed", exc_info=True)
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
        payload = {
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
        _SERIES_CACHE[cache_key] = payload
        return payload
    except TDError as exc:
        payload = {
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
        _SERIES_CACHE[cache_key] = payload
        return payload
    except Exception as e:
        payload = {
            "used_symbol": symbol,
            "asset": symbol,
            "interval": interval,
            "source": "twelvedata:time_series",
            "ok": False,
            "retrieved_at_utc": now_utc(),
            "error": str(e),
            "raw": {"values": []},
        }
        _SERIES_CACHE[cache_key] = payload
        return payload
      
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
    """Idősorból az utolsó gyertya close  időpont (UTC ISO)."""
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
    fallback_interval = "5min"

    if price is None:
        try:
            px, ts = td_last_close(symbol, fallback_interval, exchange)
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
        source = "twelvedata:quotetime_series_fallback"

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

    if fallback_used:
        result["interval"] = fallback_interval

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
            failure_cycles = 1
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


def _extract_bid_ask(frame: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    bid_candidates = ("bid", "best_bid", "bid_price", "b")
    ask_candidates = ("ask", "best_ask", "ask_price", "a")
    bid_val: Optional[float] = None
    ask_val: Optional[float] = None
    for key in bid_candidates:
        if key in frame and frame.get(key) is not None:
            try:
                bid_val = float(frame[key])
            except (TypeError, ValueError):
                bid_val = None
            if bid_val is not None:
                break
    for key in ask_candidates:
        if key in frame and frame.get(key) is not None:
            try:
                ask_val = float(frame[key])
            except (TypeError, ValueError):
                ask_val = None
            if ask_val is not None:
                break
    return bid_val, ask_val


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
    ws_attempted = False
    http_attempted = False

    deadline = time.time() + duration
    if use_ws:
        ws_attempted = True
        frames = _collect_ws_frames(asset, symbol_cycle, deadline)
        if frames:
            transport = "websocket"

    if not frames:
        remaining = max(0.0, deadline - time.time())
        if remaining > 0:
            deadline_http = time.time() + remaining
            http_attempted = True
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
        attempted: List[str] = []
        if transport:
            attempted.append(transport)
        else:
            if ws_attempted:
                attempted.append("websocket")
            if http_attempted:
                attempted.append("http")
        transport_display = "".join(attempted) if attempted else "none"
        LOGGER.info(
            "Realtime spot unavailable for %s transport=%s abort_reason=%s forced=%s",
            asset,
            transport_display,
            abort_reason or "none",
            "yes" if force else "no",
        )
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

    stale_reason: Optional[str] = None
    if force:
        last_bid, last_ask = _extract_bid_ask(last_frame)
        if last_bid is None or last_ask is None:
            stale_reason = "missing_bid_ask"
        snapshot_ts = _parse_iso_utc(payload.get("utc") or payload.get("retrieved_at_utc"))
        if snapshot_ts is None:
            stale_reason = stale_reason or "missing_timestamp"
        else:
            age = (datetime.now(timezone.utc) - snapshot_ts).total_seconds()
            if age > FORCED_SNAPSHOT_MAX_AGE:
                stale_reason = "older_than_300s"
        if stale_reason:
            payload["ok"] = False
            payload["stale_reason"] = stale_reason
            payload["stale_marked_at_utc"] = now_utc()
            payload["suggested_refresh"] = "now"
            for field in ("price", "frames", "statistics", "forced", "force_reason"):
                if field in payload:
                    payload.pop(field, None)

    transport_label = payload.get("transport") or (
        "websocket" if ws_attempted and not http_attempted else "http" if http_attempted else "unknown"
    )
    latency_avg = stats.get("latency_avg") if isinstance(stats, dict) else None
    latency_display = (
        f"{float(latency_avg):.3f}"
        if isinstance(latency_avg, (int, float))
        else "n/a"
    )
    LOGGER.info(
        "Realtime spot collected for %s transport=%s samples=%s latency_avg=%s abort_reason=%s forced=%s",
        asset,
        transport_label,
        stats.get("samples") if isinstance(stats, dict) else None,
        latency_display,
        payload.get("http_abort_reason") or "none",
        "yes" if force else "no",
    )

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

    attempt_cycle_for_log = list(attempts or [])

    def _log_plan(
        transport: str,
        *,
        note: Optional[str] = None,
        cycle=None,
        http_background: Optional[bool] = None,
    ) -> None:
        cycle_items = [
            {"symbol": sym, "exchange": exch} for sym, exch in (cycle or [])
        ]
        payload: Dict[str, Any] = {
            "event": "realtime_plan",
            "asset": asset,
            "transport": transport,
            "force": bool(force),
            "reason": reason,
            "realtime_enabled": bool(realtime_enabled),
            "attempt_cycle": cycle_items,
        }
        if note:
            payload["note"] = note
        if cycle is not None:
            payload["cycle_size"] = len(cycle_items)
        if attempt_cycle_for_log:
            payload["configured_cycle_size"] = len(attempt_cycle_for_log)
        if REALTIME_WS_ENABLED:
            payload["websocket_preferred"] = True
        if http_background is not None:
            payload["http_background_enabled"] = bool(http_background)
        LOGGER.debug("realtime_plan", extra=payload)

    if not realtime_enabled:
        _log_plan("disabled", note="realtime_flag_off", cycle=attempt_cycle_for_log)
        return

    allowed_assets = {
        a.strip().upper()
        for a in REALTIME_ASSETS_ENV.split(",")
        if a.strip()
    }
    if allowed_assets and asset.upper() not in allowed_assets and not force:
        _log_plan(
            "skipped",
            note="asset_not_allowlisted",
            cycle=attempt_cycle_for_log,
        )
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
        _log_plan("skipped", note="empty_symbol_cycle", cycle=attempt_cycle_for_log)
        return

    use_ws = REALTIME_WS_ENABLED and REALTIME_FLAG and not force
    http_background_enabled = (
        _REALTIME_HTTP_BACKGROUND_FLAG
        if _REALTIME_HTTP_BACKGROUND_FLAG is not None
        else True
    )
    run_async = http_background_enabled and not use_ws

    planned_transport = "websocket" if use_ws else "http_background" if run_async else "http_sync"

    _log_plan(
        planned_transport,
        note="forced_realtime" if force else None,
        cycle=symbol_cycle,
        http_background=http_background_enabled,
    )

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
    allow_exchange_fallback: bool = True,
) -> List[Tuple[str, Optional[str]]]:
    """Generate fallback symbol/exchange combinations for Twelve Data requests."""

    variants: List[Tuple[str, Optional[str]]] = []
    if not symbol:
        return variants

    base = (symbol, exchange)
    variants.append(base)

    if allow_exchange_fallback:
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
            if allow_exchange_fallback and exchange is not None:
                variants.append((compact, None))

    return variants


def _normalize_symbol_attempts(cfg: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    base_symbol = cfg["symbol"]
    base_exchange = cfg.get("exchange")
    attempts: List[Tuple[str, Optional[str]]] = []
    allow_compact = not bool(cfg.get("disable_compact_variants"))
    allow_exchange = not bool(cfg.get("disable_exchange_fallbacks"))

    skipped_variants: List[Tuple[str, Optional[str], Optional[str]]] = []

    def push(
        symbol: Optional[str],
        exchange: Optional[str],
        *,
        allow_compact_override: Optional[bool] = None,
        allow_exchange_override: Optional[bool] = None,
        skip: bool = False,
        note: Optional[str] = None,
    ) -> None:
        if not symbol:
            return
        if skip:
            skipped_variants.append((symbol, exchange, note))
            return
        compact_allowed = (
            allow_compact if allow_compact_override is None else allow_compact_override
        )
        exchange_allowed = (
            allow_exchange if allow_exchange_override is None else allow_exchange_override
        )
        attempts.extend(
            _symbol_attempt_variants(
                symbol,
                exchange,
                allow_compact=compact_allowed,
                allow_exchange_fallback=exchange_allowed,
            )
        )

    push(
        base_symbol,
        base_exchange,
        allow_exchange_override=allow_exchange,
    )
    for alt in cfg.get("alt", []):
        if isinstance(alt, str):
            push(alt, base_exchange, allow_exchange_override=allow_exchange)
        elif isinstance(alt, dict):
            push(
                alt.get("symbol", base_symbol),
                alt.get("exchange", base_exchange),
                allow_compact_override=None
                if alt.get("disable_compact_variants") is None
                else not bool(alt.get("disable_compact_variants")),
                allow_exchange_override=None
                if alt.get("disable_exchange_fallbacks") is None
                else not bool(alt.get("disable_exchange_fallbacks")),
                skip=bool(alt.get("skip")),
                note=str(alt.get("note")) if alt.get("note") is not None else None,
            )
        elif isinstance(alt, (list, tuple)) and alt:
            symbol = alt[0]
            exchange = alt[1] if len(alt) > 1 else base_exchange
            push(symbol, exchange, allow_exchange_override=allow_exchange)

    seen: set[Tuple[str, Optional[str]]] = set()
    unique: List[Tuple[str, Optional[str]]] = []
    for sym, exch in attempts:
        key = (sym, exch)
        if key in seen:
            continue
        seen.add(key)
        unique.append((sym, exch))

    if skipped_variants and LOGGER.isEnabledFor(logging.DEBUG):
        for sym, exch, note in skipped_variants:
            if note:
                LOGGER.debug(
                    "Skipping configured symbol variant %s (exchange=%s): %s",
                    sym,
                    exch or "default",
                    note,
                )
            else:
                LOGGER.debug(
                    "Skipping configured symbol variant %s (exchange=%s)",
                    sym,
                    exch or "default",
                )
    return unique


_SYMBOL_META_LOCK = threading.Lock()
_SYMBOL_META_CACHE: Dict[str, Optional[List[Dict[str, Any]]]] = {}


def _reset_symbol_catalog_cache() -> None:
    with _SYMBOL_META_LOCK:
        _SYMBOL_META_CACHE.clear()


def _split_symbol_variant(symbol: str) -> Tuple[str, Optional[str]]:
    if not symbol:
        return "", None
    if ":" not in symbol:
        return symbol.strip(), None
    base, suffix = symbol.split(":", 1)
    suffix = suffix.strip()
    return base.strip(), suffix or None


def _symbol_catalog_for(symbol: str) -> Optional[List[Dict[str, Any]]]:
    key = symbol.strip().upper()
    if not key:
        return None
    with _SYMBOL_META_LOCK:
        if key in _SYMBOL_META_CACHE:
            return _SYMBOL_META_CACHE[key]
    try:
        response = td_get("symbols", symbol=symbol)
    except TDError as exc:
        log_level = logging.INFO if exc.status_code == 404 else logging.WARNING
        LOGGER.log(
            log_level,
            "Failed to fetch Twelve Data symbol catalog for %s: %s",
            symbol,
            exc,
        )
        with _SYMBOL_META_LOCK:
            _SYMBOL_META_CACHE[key] = None
        return None
    except Exception as exc:
        LOGGER.warning(
            "Failed to fetch Twelve Data symbol catalog for %s: %s",
            symbol,
            exc,
        )
        with _SYMBOL_META_LOCK:
            _SYMBOL_META_CACHE[key] = None
        return None

    entries: List[Dict[str, Any]] = []
    data_block = response.get("data") if isinstance(response, dict) else None
    if isinstance(data_block, list):
        for item in data_block:
            if isinstance(item, dict):
                entries.append(item)

    with _SYMBOL_META_LOCK:
        _SYMBOL_META_CACHE[key] = entries
    return entries


def _catalog_exchange_map(entries: List[Dict[str, Any]], symbol: str) -> Dict[str, str]:
    allowed: Dict[str, str] = {}
    if not entries:
        return allowed
    symbol_norm = symbol.strip().upper()
    compact_norm = symbol_norm.replace("/", "")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_symbol = str(entry.get("symbol") or "").strip().upper()
        if entry_symbol not in {symbol_norm, compact_norm}:
            continue
        for key in ("exchange", "mic_code", "mic"):
            value = entry.get(key)
            if not value:
                continue
            value_str = str(value).strip()
            if not value_str:
                continue
            allowed.setdefault(value_str.upper(), value_str)
    return allowed


def _select_preferred_exchange(options: Dict[str, str], *candidates: Optional[str]) -> Optional[str]:
    if not options:
        return None
    for candidate in candidates:
        if not candidate:
            continue
        key = str(candidate).strip().upper()
        if not key:
            continue
        if key in options:
            return options[key]
    first_key = sorted(options.keys())[0]
    return options[first_key]


def _apply_symbol_catalog_filter(
    asset: str,
    cfg: Dict[str, Any],
    attempts: List[Tuple[str, Optional[str]]],
) -> List[Tuple[str, Optional[str]]]:
    if not attempts:
        return attempts
    if _SYMBOL_META_DISABLED or not API_KEY:
        return attempts

    preferred_exchange = str(cfg.get("exchange") or "").strip()
    filtered: List[Tuple[str, Optional[str]]] = []
    seen: Set[Tuple[str, Optional[str]]] = set()
    skipped: List[Tuple[str, Optional[str]]] = []
    adjustments: Set[Tuple[str, Optional[str], str]] = set()

    for symbol, exchange in attempts:
        base_symbol, colon_exchange = _split_symbol_variant(symbol)
        query_symbol = base_symbol or symbol
        catalog = _symbol_catalog_for(query_symbol)
        if catalog is None:
            pair = (symbol, exchange)
            if pair not in seen:
                filtered.append(pair)
                seen.add(pair)
            continue

        exchange_map = _catalog_exchange_map(catalog, base_symbol or symbol)
        if not exchange_map:
            pair = (symbol, exchange)
            if pair not in seen:
                filtered.append(pair)
                seen.add(pair)
            continue

        selected_exchange = _select_preferred_exchange(
            exchange_map,
            colon_exchange,
            exchange,
            preferred_exchange,
        )

        if selected_exchange:
            normalized_symbol = base_symbol or symbol
            pair = (normalized_symbol, selected_exchange)
            if pair not in seen:
                filtered.append(pair)
                seen.add(pair)
            orig_exchange = colon_exchange or exchange
            orig_key = str(orig_exchange).strip().upper() if orig_exchange else None
            if not exchange or colon_exchange or (
                orig_key and selected_exchange.strip().upper() != orig_key
            ):
                adjustments.add((symbol, exchange, selected_exchange))
        else:
            skipped.append((symbol, exchange))

    if skipped:
        skipped_label = ", ".join(
            f"{sym}@{exch or 'default'}" for sym, exch in skipped
        )
        LOGGER.warning(
            "Skipping unsupported Twelve Data symbol variants for %s: %s",
            asset,
            skipped_label,
        )

    if adjustments:
        adjustment_label = ", ".join(
            f"{sym}@{exch or 'default'}→{new}" for sym, exch, new in sorted(
                adjustments, key=lambda item: (item[0], item[1] or "", item[2])
            )
        )
        LOGGER.info(
            "Adjusted Twelve Data symbol mappings for %s: %s",
            asset,
            adjustment_label,
        )

    return filtered or attempts


def try_symbols(
    attempts: List[Tuple[str, Optional[str]]],
    fetch_fn,
    freshness_limit: Optional[float] = None,
    *,
    attempt_memory: Optional[AttemptMemory] = None,
    raise_on_all_client_errors: bool = False,
):
    """Iterate over symbol candidates and prefer the freshest successful payload."""

    last: Optional[Dict[str, Any]] = None
    best: Optional[Dict[str, Any]] = None
    best_latency: Optional[float] = None
    attempts_made = 0
    client_failures: List[Dict[str, Any]] = []

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
        attempts_made = 1
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
            if exc.status_code in CLIENT_ERROR_STATUS_CODES:
                client_failures.append(
                    {
                        "symbol": sym,
                        "exchange": exch,
                        "status_code": exc.status_code,
                        "reason": str(exc),
                    }
                )
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
                client_failures.append(
                    {
                        "symbol": sym,
                        "exchange": exch,
                        "status_code": error_code,
                        "reason": str(reason),
                    }
                )

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
    if (
        raise_on_all_client_errors
        and attempts_made
        and client_failures
        and len(client_failures) >= attempts_made
        and all(item.get("status_code") == 404 for item in client_failures if item.get("status_code") is not None)
    ):
        LOGGER.error(
            "All symbol attempts returned 404 client errors",
            extra={
                "event": "symbol_attempts_exhausted",
                "attempts": client_failures,
            },
        )
        raise SymbolAttemptsExhausted(client_failures)
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
    raise_on_all_client_errors: bool = False,
):
    result = try_symbols(
        attempts,
        fetch_fn,
        freshness_limit=freshness_limit,
        attempt_memory=attempt_memory,
        raise_on_all_client_errors=raise_on_all_client_errors,
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
        retries = 1
        time.sleep(max(TD_RATE_LIMITER.current_delay, 0.2))
        result = try_symbols(
            attempts,
            fetch_fn,
            freshness_limit=freshness_limit,
            attempt_memory=attempt_memory,
            raise_on_all_client_errors=raise_on_all_client_errors,
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
                value = float(row["close"])
                if value != value or math.isinf(value):
                    out.append(None)
                else:
                    out.append(value)
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
    attempts = _apply_symbol_catalog_filter(asset, cfg, attempts)
    attempt_memory = AttemptMemory()

    spot_limit = _spot_freshness_limit(asset)
    status_profile_name, status_profile = _status_profile_for_asset(asset)
    skip_reason = _status_profile_skip_reason(status_profile) or _market_closed_skip_reason(
        asset, cfg
    )
    series_payloads: Dict[str, Dict[str, Any]]

    if skip_reason:
        LOGGER.info(
            "Skipping Twelve Data fetches for %s: market closed (%s%s)",
            asset,
            skip_reason,
            f", profile={status_profile_name}" if status_profile else "",
        )
        note = (
            "market_closed_weekend"
            if skip_reason == "weekend"
            else "market_closed_outside_hours"
        )
        base_symbol: Optional[str] = attempts[0][0] if attempts else cfg.get("symbol")
        base_exchange: Optional[str] = attempts[0][1] if attempts else cfg.get("exchange")
        placeholder_spot: Dict[str, Any] = {
            "ok": False,
            "asset": asset,
            "retrieved_at_utc": now_utc(),
            "freshness_limit_seconds": spot_limit,
            "freshness_violation": False,
            "market_closed_assumed": True,
            "market_closed_reason": skip_reason,
            "freshness_note": note,
            "error": f"market closed ({skip_reason})",
        }
        _apply_status_profile_metadata(placeholder_spot, status_profile_name, status_profile)
        if base_symbol:
            placeholder_spot["used_symbol"] = base_symbol
        if base_exchange:
            placeholder_spot["used_exchange"] = base_exchange
        spot = _reuse_previous_spot(adir, placeholder_spot, spot_limit)
        if not isinstance(spot, dict):
            spot = dict(placeholder_spot)
        else:
            spot.setdefault("market_closed_assumed", True)
            spot.setdefault("market_closed_reason", skip_reason)
            spot.setdefault("freshness_note", note)
            spot.setdefault("freshness_limit_seconds", spot_limit)
            _apply_status_profile_metadata(spot, status_profile_name, status_profile)
        series_payloads = {}
        for name, interval in SERIES_FETCH_PLAN:
            freshness_limit = SERIES_FRESHNESS_LIMITS.get(interval)
            outputsize = SERIES_OUTPUT_SIZES.get(interval, 500)
            placeholder_series: Dict[str, Any] = {
                "ok": False,
                "retrieved_at_utc": now_utc(),
                "freshness_violation": False,
                "market_closed_assumed": True,
                "market_closed_reason": skip_reason,
                "freshness_note": note,
                "error": f"market closed ({skip_reason})",
                "raw": {"values": []},
            }
            _apply_status_profile_metadata(placeholder_series, status_profile_name, status_profile)
            payload = _finalize_series_payload(
                attempts,
                adir,
                name,
                interval,
                freshness_limit,
                placeholder_series,
                outputsize,
                attempt_memory=attempt_memory,
            )
            if isinstance(payload, dict):
                payload.setdefault("market_closed_assumed", True)
                payload.setdefault("market_closed_reason", skip_reason)
                payload.setdefault("freshness_note", note)
                _apply_status_profile_metadata(payload, status_profile_name, status_profile)
            series_payloads[name] = payload
        spot_violation = False
        force_reason: Optional[str] = None
    else:
        spot = fetch_with_freshness(
            attempts,
            lambda s, ex: td_spot_with_fallback(s, ex),
            freshness_limit=spot_limit,
            max_refreshes=1,
            attempt_memory=attempt_memory,
            raise_on_all_client_errors=True,
        )
        if isinstance(spot, dict):
            spot = _maybe_use_secondary_spot(asset, spot)
            spot.setdefault("freshness_limit_seconds", spot_limit)
        else:
            spot = {"ok": False, "freshness_limit_seconds": spot_limit}
        if isinstance(spot, dict):
            spot = _reuse_previous_spot(adir, spot, spot_limit)
        spot_violation = bool(spot.get("freshness_violation")) if isinstance(spot, dict) else False
        force_reason = None
        if not spot.get("ok"):
            force_reason = "spot_error"
        elif spot_violation:
            force_reason = "spot_stale"
        elif spot.get("fallback_used") or spot.get("fallback_previous_payload"):
            if not spot.get("market_closed_assumed"):
                force_reason = "spot_fallback"
        series_payloads = _collect_series_payloads(attempts, attempt_memory, adir)

    _write_spot_payload(adir, asset, spot)
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

    spot_payload = spot if isinstance(spot, dict) else {}
    sorted_series_items = sorted(series_payloads.items(), key=lambda item: item[0])
    series_ok = {
        name: bool(payload.get("ok")) if isinstance(payload, dict) else False
        for name, payload in sorted_series_items
    }
    series_fallback = {
        name: bool(
            isinstance(payload, dict)
            and (
                payload.get("fallback_used")
                or payload.get("fallback_previous_payload")
            )
        )
        for name, payload in sorted_series_items
    }
    LOGGER.info(
        "fetch_completed",
        extra={
            "event": "fetch_completed",
            "asset": asset,
            "used_symbol": spot_payload.get("used_symbol"),
            "used_exchange": spot_payload.get("used_exchange"),
            "spot_ok": bool(spot_payload.get("ok")),
            "spot_price": spot_payload.get("price"),
            "spot_freshness_violation": bool(spot_payload.get("freshness_violation")),
            "spot_fallback": bool(
                spot_payload.get("fallback_used")
                or spot_payload.get("fallback_previous_payload")
            ),
            "market_closed": bool(spot_payload.get("market_closed_assumed")),
            "force_reason": force_reason,
            "series_ok": series_ok,
            "series_fallback": series_fallback,
            "attempt_cycle": [
                {"symbol": sym, "exchange": exch}
                for sym, exch in (attempts or [])
            ],
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
    except SymbolAttemptsExhausted as error:
        _write_error_payload(asset, error)
        _record_asset_failure(asset, str(error))
        raise
    except Exception as error:
        _write_error_payload(asset, error)
        _record_asset_failure(asset, str(error))


def _reset_out_dir_if_requested(out_dir: str, logger: logging.Logger) -> None:
    if not RESET_PUBLIC_ON_TRADING_START:
        return

    manual_state_snapshots = []
    out_dir_path = Path(out_dir)

    for state_name in ("_manual_positions.json", "_manual_positions_audit.jsonl"):
        state_path = out_dir_path / state_name
        try:
            if state_path.exists():
                manual_state_snapshots.append((state_name, state_path.read_bytes()))
        except Exception as exc:
            logger.warning("Failed to snapshot manual state before reset (%s): %s", state_path, exc)

    try:
        shutil.rmtree(out_dir)
        logger.info("Reset public artefacts before trading run: %s", out_dir)
    except FileNotFoundError:
        logger.info("Public artefact directory missing, creating: %s", out_dir)
    except Exception as exc:
        logger.warning("Failed to reset public artefacts (%s): %s", out_dir, exc)

    ensure_dir(out_dir)

    for state_name, snapshot in manual_state_snapshots:
        try:
            target = out_dir_path / state_name
            ensure_dir(str(target.parent))
            target.write_bytes(snapshot)
            logger.info(
                "Restored manual positions artefact after reset: %s (%d bytes)",
                target,
                len(snapshot),
            )
        except Exception as exc:
            logger.warning("Failed to restore manual state after reset (%s): %s", state_name, exc)


def main():
    if not API_KEY:
        raise SystemExit("TWELVEDATA_API_KEY hiányzik (GitHub Secret).")
    logger = logging.getLogger("market_feed.trading")
    _reset_out_dir_if_requested(OUT_DIR, logger)
    ensure_dir(OUT_DIR)
    pipeline_log_path = None
    try:
        pipeline_log_path = get_pipeline_log_path()
    except Exception:
        pipeline_log_path = None
    if pipeline_log_path:
        ensure_json_file_handler(
            logger,
            pipeline_log_path,
            static_fields={"component": "trading", **(get_run_logging_context() or {})},
        )
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
        exception_types = warning_summary.get("exception_types") or {}
        if exception_types:
            formatted = ", ".join(
                f"{name}:{count}" for name, count in sorted(exception_types.items())
            )
            logger.warning("Pipeline exceptions detected: %s", formatted)
        sentiment_events = warning_summary.get("sentiment_exit_events") or []
        if sentiment_events:
            latest_event = sentiment_events[-1]
            detail = latest_event.get("detail") or "sentiment exit triggered"
            logger.info("Latest sentiment exit event: %s", detail)
    except Exception as exc:
        logger.warning("Failed to summarize pipeline warnings: %s", exc)

    try:
        update_system_heartbeat(OUT_DIR)
    except Exception as exc:
        logger.warning("Failed to update system heartbeat: %s", exc)

    try:
        _write_public_refresh_marker(
            OUT_DIR,
            started_at=started_at_dt,
            completed_at=completed_at_dt,
            duration_seconds=duration_seconds,
        )
    except Exception as exc:
        logger.warning("Failed to write public refresh marker: %s", exc)

if __name__ == "__main__":
    main()




























"""Lightweight sentiment loader for the market-feed assets.

The production environment drops a ``news_sentiment.json`` file into the
``public/BTCUSD`` directory whenever crypto news flashes occur.  The same
machinery can be extended to additional instruments by dropping sentiment
snapshots into ``public/<ASSET>/sentiment.json`` (or ``events.json``) with an
optional ``severity`` field so that higher impact events influence the trading
score more aggressively.

Single snapshot example:

.. code-block:: json

    {
        "score": 0.75,
        "bias": "btc_bullish",
        "headline": "ETF inflows accelerate, bitcoin rallies",
        "severity": 0.6,
        "expires_at": "2024-05-01T12:30:00Z"
    }

Event list example (the loader picks the most severe, non-expired entry):

.. code-block:: json

    {
        "events": [
            {
                "score": 0.3,
                "bias": "bullish",
                "headline": "NVDA earnings whisper",
                "severity": 0.4,
                "expires_at": "2024-05-22T20:00:00Z"
            }
        ]
    }

The ``score`` lies within ``[-1, 1]`` while ``severity`` is clipped to
``[0, 1]``.  ``expires_at`` invalidates stale sentiment snapshots.  The module
performs no network access besides the optional BTCUSD auto-refresh; it simply
parses the JSON, validates the timestamp and returns a structured payload to
``analysis.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import dateutil.parser

from btcusd_sentiment import DEFAULT_MIN_INTERVAL, SENTIMENT_FILENAME, refresh_btcusd_sentiment

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))

# Optional override configuration for static or scheduled sentiment events.
_OVERRIDE_ENV = "SENTIMENT_OVERRIDE_FILE"
_BASE_DIR = Path(__file__).resolve().parent

# Disable automatic sentiment refresh by default; operators can opt-in via
# ``BTCUSD_SENTIMENT_AUTO=1`` when a live news API is available.
_AUTO_REFRESH_DEFAULT = "0"
_AUTO_REFRESH_FLAG = os.getenv("BTCUSD_SENTIMENT_AUTO", _AUTO_REFRESH_DEFAULT).lower()
AUTO_REFRESH_ENABLED = _AUTO_REFRESH_FLAG not in {"0", "false", "off", "no"}

def _auto_interval(default: int) -> int:
    raw = os.getenv("BTCUSD_SENTIMENT_MIN_INTERVAL")
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(0, value)


AUTO_REFRESH_MIN_INTERVAL = _auto_interval(DEFAULT_MIN_INTERVAL)


DEFAULT_SENTIMENT_FILENAME = "sentiment.json"


def _override_config_path() -> Path:
    raw = os.getenv(_OVERRIDE_ENV)
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = _BASE_DIR / raw
        return path
    return _BASE_DIR / "config" / "sentiment_overrides.json"


@lru_cache(maxsize=1)
def _load_override_config() -> Dict[str, Any]:
    path = _override_config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class SentimentSignal:
    score: float
    bias: str
    headline: Optional[str]
    expires_at: Optional[datetime]
    severity: Optional[float] = None
    source: Optional[Path] = None
    category: Optional[str] = None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.expires_at is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now > self.expires_at

    @property
    def effective_severity(self) -> float:
        if self.severity is None:
            return 1.0
        return max(0.0, min(1.0, float(self.severity)))


def _normalise_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not (-1.5 <= score <= 1.5):
        return None
    return max(-1.0, min(1.0, score))
    

def _normalise_severity(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        sev = float(value)
    except (TypeError, ValueError):
        return None
    if sev < 0 and sev > -1e-6:
        sev = 0.0
    if sev > 1 and sev < 1 + 1e-6:
        sev = 1.0
    return max(0.0, min(1.0, sev))


def _parse_expires(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return dateutil.parser.isoparse(str(value)).astimezone(timezone.utc)
    except Exception:
        return None


def _select_event(events: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates: list[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for raw in events:
        if not isinstance(raw, dict):
            continue
        score = _normalise_score(raw.get("score"))
        bias = raw.get("bias")
        if score is None or not bias:
            continue
        expires = _parse_expires(raw.get("expires_at"))
        if expires and expires <= now:
            continue
        candidates.append(dict(raw, score=score, bias=str(bias)))
    if not candidates:
        return None
    candidates.sort(key=lambda e: _normalise_severity(e.get("severity")) or 0.0, reverse=True)
    return candidates[0]


def _load_btcusd_sentiment(base: Path) -> Optional[SentimentSignal]:
    if AUTO_REFRESH_ENABLED:
        try:
            refresh_btcusd_sentiment(
                api_key=os.getenv("NEWSAPI_KEY", ""),
                output_dir=base,
                min_interval=AUTO_REFRESH_MIN_INTERVAL,
            )
        except Exception:
            # Auto-refresh must never break signal loading; fall back to existing snapshot.
            pass
            
    path = base / SENTIMENT_FILENAME
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return None
        
    score = _normalise_score(raw.get("score"))
    bias = raw.get("bias")
    if score is None or bias is None:
        return None
    expires_at = _parse_expires(raw.get("expires_at"))
    signal = SentimentSignal(
        score=score,
        bias=str(bias),
        headline=raw.get("headline"),
        expires_at=expires_at,
        severity=_normalise_severity(raw.get("severity")),
        source=path,
        category=str(raw.get("category")) if raw.get("category") else None,
    )
    if signal.is_expired():
        return None
    return signal


def _time_token_to_minutes(token: Any) -> Optional[int]:
    if token is None:
        return None
    if isinstance(token, (int, float)):
        value = int(token)
        if 0 <= value <= 1440:
            return value
        return None
    if isinstance(token, str):
        token = token.strip()
        if not token:
            return None
        if ":" in token:
            try:
                hour, minute = token.split(":", 1)
                value = int(hour) * 60 + int(minute)
            except ValueError:
                return None
            return max(0, min(1440, value))
        try:
            value_int = int(token)
        except ValueError:
            return None
        if 0 <= value_int <= 1440:
            return value_int
    return None


def _parse_intraday_window(window: Any) -> Optional[Tuple[int, int]]:
    if window is None:
        return None
    if isinstance(window, (list, tuple)) and len(window) >= 2:
        start = _time_token_to_minutes(window[0])
        end = _time_token_to_minutes(window[1])
        if start is None or end is None:
            return None
        return start, end
    if isinstance(window, str):
        parts = window.split("-", 1)
        if len(parts) == 2:
            start = _time_token_to_minutes(parts[0])
            end = _time_token_to_minutes(parts[1])
            if start is None or end is None:
                return None
            return start, end
    return None


def _override_event_matches(event: Dict[str, Any], asset: str, now: datetime, defaults: Dict[str, Any]) -> bool:
    assets = event.get("assets")
    if isinstance(assets, (list, tuple)):
        asset_keys = {str(item).upper() for item in assets if isinstance(item, (str, bytes))}
        if asset.upper() not in asset_keys and "*" not in asset_keys:
            return False
    elif assets is not None:
        if str(assets).upper() not in {asset.upper(), "*"}:
            return False
    start = _parse_expires(event.get("start") or event.get("valid_from"))
    if start and now < start:
        return False
    end = _parse_expires(event.get("end") or event.get("valid_until") or event.get("expires_at"))
    if end and now > end:
        return False
    weekdays = event.get("weekdays")
    if weekdays is None:
        weekdays = defaults.get("weekdays")
    if isinstance(weekdays, (list, tuple)) and weekdays:
        try:
            weekday_set = {int(day) % 7 for day in weekdays}
        except (TypeError, ValueError):
            weekday_set = set()
        if weekday_set and now.weekday() not in weekday_set:
            return False
    window = (
        event.get("intraday_window")
        or event.get("window")
        or event.get("intraday")
        or defaults.get("intraday_window")
    )
    parsed_window = _parse_intraday_window(window)
    if parsed_window:
        start_minute, end_minute = parsed_window
        minute_of_day = now.hour * 60 + now.minute
        if start_minute <= end_minute:
            if not (start_minute <= minute_of_day <= end_minute):
                return False
        else:  # Wrap-around window
            if not (minute_of_day >= start_minute or minute_of_day <= end_minute):
                return False
    return True


def _build_override_signal(
    event: Dict[str, Any],
    asset: str,
    now: datetime,
    defaults: Dict[str, Any],
) -> Optional[SentimentSignal]:
    score = _normalise_score(event.get("score"))
    bias = event.get("bias")
    if score is None or not bias:
        return None
    severity = _normalise_severity(event.get("severity"))
    if severity is None:
        severity = _normalise_severity(defaults.get("severity")) or 0.5
    expires = _parse_expires(event.get("expires_at") or event.get("end"))
    if expires is None:
        ttl_minutes = event.get("ttl_minutes")
        if ttl_minutes is None:
            ttl_minutes = defaults.get("ttl_minutes") or defaults.get("expires_ttl_minutes")
        try:
            ttl_minutes = float(ttl_minutes) if ttl_minutes is not None else None
        except (TypeError, ValueError):
            ttl_minutes = None
        if ttl_minutes and ttl_minutes > 0:
            expires = now + timedelta(minutes=float(ttl_minutes))
    signal = SentimentSignal(
        score=score,
        bias=str(bias),
        headline=event.get("headline") or defaults.get("headline"),
        expires_at=expires,
        severity=severity,
        source=_override_config_path(),
        category=str(event.get("category") or defaults.get("category") or asset),
    )
    if signal.is_expired(now):
        return None
    return signal


def _load_override_sentiment(asset: str) -> Optional[SentimentSignal]:
    config = _load_override_config()
    events = config.get("events")
    if not isinstance(events, list) or not events:
        return None
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    now = datetime.now(timezone.utc)
    candidates: List[SentimentSignal] = []
    for raw_event in events:
        if not isinstance(raw_event, dict):
            continue
        if not _override_event_matches(raw_event, asset, now, defaults):
            continue
        signal = _build_override_signal(raw_event, asset, now, defaults)
        if signal is not None:
            candidates.append(signal)
    if not candidates:
        return None
    candidates.sort(key=lambda s: abs(s.score) * s.effective_severity, reverse=True)
    return candidates[0]


def _select_sentiment_signal(*signals: Optional[SentimentSignal]) -> Optional[SentimentSignal]:
    best: Optional[SentimentSignal] = None
    best_score = -1.0
    for signal in signals:
        if signal is None:
            continue
        strength = abs(signal.score) * signal.effective_severity
        if strength > best_score:
            best = signal
            best_score = strength
    return best


def reload_sentiment_overrides() -> None:
    """Clear the override configuration cache (useful for tests)."""

    _load_override_config.cache_clear()


def _load_generic_sentiment(asset: str, base: Path) -> Optional[SentimentSignal]:
    candidates = [base / DEFAULT_SENTIMENT_FILENAME, base / "events.json"]
    payload: Optional[Dict[str, Any]] = None
    chosen_path: Optional[Path] = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        if isinstance(payload, dict):
            chosen_path = path
            break
        payload = None
    if payload is None:
        return None

    event: Optional[Dict[str, Any]]
    if isinstance(payload.get("events"), list):
        event = _select_event(payload["events"])  # type: ignore[arg-type]
    else:
        event = payload if isinstance(payload, dict) else None
    if event is None:
        return None

    score = _normalise_score(event.get("score"))
    bias = event.get("bias")
    if score is None or not bias:
        return None
    expires = _parse_expires(event.get("expires_at"))
    signal = SentimentSignal(
        score=score,
        bias=str(bias),
        headline=event.get("headline") or payload.get("headline"),
        expires_at=expires,
        severity=_normalise_severity(event.get("severity") or payload.get("severity")),
        source=chosen_path,
        category=str(event.get("category") or payload.get("category") or asset),
    )
    if signal.is_expired():
        return None
    return signal


def load_sentiment(asset: str, outdir: Optional[Path] = None) -> Optional[SentimentSignal]:
    asset_key = asset.upper()
    base = outdir if outdir is not None else PUBLIC_DIR / asset_key
    if not isinstance(base, Path):
        base = Path(base)

    override_signal = _load_override_sentiment(asset_key)
    if asset_key == "BTCUSD":
        primary = _load_btcusd_sentiment(base)
        return _select_sentiment_signal(primary, override_signal)
    generic_signal = _load_generic_sentiment(asset_key, base)
    return _select_sentiment_signal(generic_signal, override_signal)


__all__ = [
    "SentimentSignal",
    "load_sentiment",
    "SENTIMENT_FILENAME",
    "DEFAULT_SENTIMENT_FILENAME",
    "reload_sentiment_overrides",
]

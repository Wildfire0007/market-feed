"""Lightweight sentiment loader for the USDJPY intervention watch.

The production environment drops a ``news_sentiment.json`` file into the
``public/USDJPY`` directory whenever macro news flashes occur.  The same
machinery can be extended to additional instruments by dropping sentiment
snapshots into ``public/<ASSET>/sentiment.json`` (or ``events.json``) with an
optional ``severity`` field so that higher impact events influence the trading
score more aggressively.

Single snapshot example:

.. code-block:: json

    {
        "score": 0.75,
        "bias": "usd_bullish",
        "headline": "MoF reassures markets, yen weakens",
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
performs no network access besides the optional USDJPY auto-refresh; it simply
parses the JSON, validates the timestamp and returns a structured payload to
``analysis.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import dateutil.parser

from usdjpy_sentiment import DEFAULT_MIN_INTERVAL, SENTIMENT_FILENAME, refresh_usdjpy_sentiment

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))

# Disable automatic sentiment refresh by default; operators can opt-in via
# ``USDJPY_SENTIMENT_AUTO=1`` when a live news API is available.
_AUTO_REFRESH_DEFAULT = "0"
_AUTO_REFRESH_FLAG = os.getenv("USDJPY_SENTIMENT_AUTO", _AUTO_REFRESH_DEFAULT).lower()
AUTO_REFRESH_ENABLED = _AUTO_REFRESH_FLAG not in {"0", "false", "off", "no"}

def _auto_interval(default: int) -> int:
    raw = os.getenv("USDJPY_SENTIMENT_MIN_INTERVAL")
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(0, value)


AUTO_REFRESH_MIN_INTERVAL = _auto_interval(DEFAULT_MIN_INTERVAL)


DEFAULT_SENTIMENT_FILENAME = "sentiment.json"


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


def _load_usdjpy_sentiment(base: Path) -> Optional[SentimentSignal]:
    if AUTO_REFRESH_ENABLED:
        try:
            refresh_usdjpy_sentiment(
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

    if asset_key == "USDJPY":
        return _load_usdjpy_sentiment(base)
    return _load_generic_sentiment(asset_key, base)


__all__ = [
    "SentimentSignal",
    "load_sentiment",
    "SENTIMENT_FILENAME",
    "DEFAULT_SENTIMENT_FILENAME",
]

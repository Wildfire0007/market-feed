"""Lightweight sentiment loader for the USDJPY intervention watch.

The production environment drops a ``news_sentiment.json`` file into the
``public/USDJPY`` directory whenever macro news flashes occur.  The structure
is intentionally simple so that human operators (or a separate micro-service)
can maintain it without tight coupling:

.. code-block:: json

    {
        "score": 0.75,
        "bias": "usd_bullish",
        "headline": "MoF reassures markets, yen weakens",
        "expires_at": "2024-05-01T12:30:00Z"
    }

The ``score`` lies within ``[-1, 1]`` and the optional ``expires_at`` field
invalidates stale sentiment snapshots.  The module performs no network access;
it simply parses the JSON, validates the timestamp and returns a structured
payload to ``analysis.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import dateutil.parser

PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR", "public"))
SENTIMENT_FILENAME = "news_sentiment.json"


@dataclass
class SentimentSignal:
    score: float
    bias: str
    headline: Optional[str]
    expires_at: Optional[datetime]

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.expires_at is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now > self.expires_at


def load_sentiment(asset: str, outdir: Optional[Path] = None) -> Optional[SentimentSignal]:
    if asset.upper() != "USDJPY":
        return None
    base = outdir if outdir is not None else PUBLIC_DIR / asset.upper()
    if not isinstance(base, Path):
        base = Path(base)
    path = base / SENTIMENT_FILENAME
    if not path.exists():
        return None
    try:
        raw: Dict[str, Any] = {}  # type: ignore[assignment]
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)  # type: ignore[name-defined]
    except Exception:
        return None
    score = raw.get("score")
    bias = raw.get("bias")
    if score is None or bias is None:
        return None
    try:
        score = max(-1.0, min(1.0, float(score)))
    except Exception:
        return None
    headline = raw.get("headline")
    expires_at = None
    if raw.get("expires_at"):
        try:
            expires_at = dateutil.parser.isoparse(raw["expires_at"]).astimezone(timezone.utc)
        except Exception:
            expires_at = None
    signal = SentimentSignal(score=score, bias=str(bias), headline=headline, expires_at=expires_at)
    if signal.is_expired():
        return None
    return signal


__all__ = ["SentimentSignal", "load_sentiment", "SENTIMENT_FILENAME"]

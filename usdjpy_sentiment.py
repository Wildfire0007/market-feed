"""Utilities for fetching and maintaining USDJPY macro sentiment snapshots."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import requests

SENTIMENT_FILENAME = "news_sentiment.json"

DEFAULT_QUERY = "USDJPY intervention"
DEFAULT_LANGUAGE = "en"
DEFAULT_COUNTRY: Optional[str] = "jp"
DEFAULT_PAGE_SIZE = 20
DEFAULT_TIMEOUT = 10
DEFAULT_EXPIRY_MINUTES = 90
MAX_PAGES = 2
DEFAULT_MIN_INTERVAL = 600

POSITIVE_KEYWORDS = {
    "intervention",
    "warns",
    "concern",
    "sell",
    "weak",
    "weakens",
    "record",
    "support",
    "defend",
    "meeting",
    "pressure",
    "surge",
    "boost",
    "hawkish",
    "rate",
    "tighten",
}
NEGATIVE_KEYWORDS = {
    "relief",
    "calm",
    "stabilize",
    "stable",
    "strengthens",
    "strong",
    "buy",
    "bullish",
    "dovish",
    "ease",
    "cuts",
    "cut",
}


@dataclass
class Article:
    title: str
    description: Optional[str]
    url: str
    published_at: Optional[datetime]

    @property
    def text(self) -> str:
        parts: List[str] = [self.title]
        if self.description:
            parts.append(self.description)
        return " ".join(parts).lower()


def fetch_articles(
    query: str,
    *,
    api_key: str,
    language: str = DEFAULT_LANGUAGE,
    country: Optional[str] = DEFAULT_COUNTRY,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = MAX_PAGES,
    timeout: int = DEFAULT_TIMEOUT,
) -> List[Article]:
    session = requests.Session()
    url = "https://newsapi.org/v2/top-headlines"
    articles: List[Article] = []
    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "page": page,
        }
        if country:
            params["country"] = country
        headers = {"X-Api-Key": api_key}
        backoff = 1.0
        for attempt in range(5):
            try:
                response = session.get(url, params=params, headers=headers, timeout=timeout)
            except requests.RequestException:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            if response.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            if response.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            if response.status_code != 200:
                raise RuntimeError(
                    f"News API request failed with status {response.status_code}: {response.text[:200]}"
                )
            payload = response.json()
            for item in payload.get("articles", []):
                published_at = item.get("publishedAt")
                dt = None
                if published_at:
                    try:
                        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    except ValueError:
                        dt = None
                article = Article(
                    title=item.get("title") or "",
                    description=item.get("description"),
                    url=item.get("url") or "",
                    published_at=dt,
                )
                if article.title:
                    articles.append(article)
            break
        else:
            raise RuntimeError("Failed to retrieve headlines after multiple retries")
    return articles


def score_text(text: str) -> float:
    tokens = text.lower().split()
    pos_hits = sum(tokens.count(word) for word in POSITIVE_KEYWORDS)
    neg_hits = sum(tokens.count(word) for word in NEGATIVE_KEYWORDS)
    if pos_hits == 0 and neg_hits == 0:
        return 0.0
    raw = pos_hits - neg_hits
    total = pos_hits + neg_hits
    score = raw / total
    return max(-1.0, min(1.0, score))


def aggregate_score(articles: Iterable[Article]) -> Optional[float]:
    scores = [score_text(article.text) for article in articles]
    if not scores:
        return None
    return max(-1.0, min(1.0, sum(scores) / len(scores)))


def determine_bias(score: float) -> str:
    if score > 0.2:
        return "usd_bullish"
    if score < -0.2:
        return "usd_bearish"
    return "neutral"


def choose_representative(articles: Iterable[Article], score: float) -> Optional[Article]:
    best: Optional[Article] = None
    best_delta = float("inf")
    for article in articles:
        article_score = score_text(article.text)
        delta = abs(article_score - score)
        if best is None or delta < best_delta:
            best = article
            best_delta = delta
    return best


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_sentiment(
    *,
    score: float,
    bias: str,
    article: Optional[Article],
    output_path: Path,
    expires_minutes: int,
) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {
        "score": round(score, 4),
        "bias": bias,
        "headline": article.title if article else None,
        "source_url": article.url if article else None,
        "published_at": article.published_at.isoformat() if article and article.published_at else None,
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _existing_payload(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _expires_at(payload: Optional[dict]) -> Optional[datetime]:
    if not payload:
        return None
    expires = payload.get("expires_at")
    if not isinstance(expires, str):
        return None
    try:
        return datetime.fromisoformat(expires.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def refresh_usdjpy_sentiment(
    *,
    api_key: str,
    output_dir: Path,
    query: str = DEFAULT_QUERY,
    language: str = DEFAULT_LANGUAGE,
    country: Optional[str] = DEFAULT_COUNTRY,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = MAX_PAGES,
    expires_minutes: int = DEFAULT_EXPIRY_MINUTES,
    min_interval: int = DEFAULT_MIN_INTERVAL,
) -> Optional[dict]:
    api_key = api_key.strip()
    if not api_key:
        return None

    output_dir = Path(output_dir)
    ensure_output_dir(output_dir)
    output_path = output_dir / SENTIMENT_FILENAME

    payload = _existing_payload(output_path)
    expires_at = _expires_at(payload)
    now = datetime.now(timezone.utc)
    should_refresh = True
    if output_path.exists():
        mtime = output_path.stat().st_mtime
        age = time.time() - mtime
        expired = expires_at is not None and now >= expires_at
        recent_enough = age < max(min_interval, 0) if min_interval > 0 else False
        if not expired and recent_enough:
            should_refresh = False
    if not should_refresh:
        return None

    articles = fetch_articles(
        query,
        api_key=api_key,
        language=language,
        country=country,
        page_size=page_size,
        max_pages=max_pages,
    )
    score = aggregate_score(articles)
    if score is None:
        return None
    bias = determine_bias(score)
    representative = choose_representative(articles, score)
    write_sentiment(
        score=score,
        bias=bias,
        article=representative,
        output_path=output_path,
        expires_minutes=expires_minutes,
    )
    return {
        "score": score,
        "bias": bias,
        "headline": representative.title if representative else None,
        "output_path": str(output_path),
    }


__all__ = [
    "Article",
    "SENTIMENT_FILENAME",
    "fetch_articles",
    "aggregate_score",
    "determine_bias",
    "choose_representative",
    "score_text",
    "ensure_output_dir",
    "write_sentiment",
    "refresh_usdjpy_sentiment",
    "DEFAULT_QUERY",
    "DEFAULT_LANGUAGE",
    "DEFAULT_COUNTRY",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_EXPIRY_MINUTES",
    "MAX_PAGES",
    "DEFAULT_MIN_INTERVAL",
]

"""Automate maintenance of USDJPY macro news sentiment snapshots.

This script queries a configurable headline API (defaults to NewsAPI.org) for
recent USDJPY / Japanese intervention news, heuristically scores the
headlines, and writes the ``news_sentiment.json`` file consumed by
``news_feed.py``.  It is intentionally dependency-light so operators can run it
from cron or ad-hoc when macro wires start moving.

Usage example::

    NEWSAPI_KEY=... python scripts/update_usdjpy_sentiment.py \
        --query "USDJPY intervention" --country jp --expires-minutes 90

The command will create or update ``public/USDJPY/news_sentiment.json`` with the
most recent article summary and a normalized score in ``[-1, 1]``.  The file is
expired automatically after the configured TTL so stale guidance cannot leak
into the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import requests

from news_feed import PUBLIC_DIR, SENTIMENT_FILENAME

DEFAULT_QUERY = "USDJPY intervention"
DEFAULT_LANGUAGE = "en"
DEFAULT_COUNTRY = "jp"
DEFAULT_PAGE_SIZE = 20
DEFAULT_TIMEOUT = 10
DEFAULT_EXPIRY_MINUTES = 90
MAX_PAGES = 2

POSITIVE_KEYWORDS = {
    "intervention", "warns", "concern", "sell", "weak", "weakens",
    "record low", "support", "defend", "meeting", "pressure", "surge",
    "boost", "hawkish", "rate hike", "tighten",
}
NEGATIVE_KEYWORDS = {
    "relief", "calm", "stabilize", "stable", "strengthens", "strong",
    "buy", "bullish yen", "dovish", "ease", "cuts", "cut", "dovish",
}


@dataclass
class Article:
    title: str
    description: Optional[str]
    url: str
    published_at: Optional[datetime]

    @property
    def text(self) -> str:
        bits: List[str] = [self.title]
        if self.description:
            bits.append(self.description)
        return " ".join(bits).lower()


def _env_api_key() -> Optional[str]:
    return os.getenv("NEWSAPI_KEY")


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
    counts = Counter(tokens)
    pos_hits = sum(counts.get(word, 0) for word in POSITIVE_KEYWORDS)
    neg_hits = sum(counts.get(word, 0) for word in NEGATIVE_KEYWORDS)
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
    best_delta = -1.0
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update USDJPY macro sentiment snapshot")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search query for the news API")
    parser.add_argument(
        "--language", default=DEFAULT_LANGUAGE, help="ISO language code for filtering headlines"
    )
    parser.add_argument(
        "--country",
        default=DEFAULT_COUNTRY,
        help="ISO country code to scope headlines (omit to disable)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PUBLIC_DIR / "USDJPY"),
        help="Directory where news_sentiment.json should be written",
    )
    parser.add_argument(
        "--expires-minutes",
        type=int,
        default=DEFAULT_EXPIRY_MINUTES,
        help="How long the sentiment snapshot remains valid",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=MAX_PAGES,
        help="Maximum number of pages to fetch from the news API",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Number of articles to request per page",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    api_key = _env_api_key()
    if not api_key:
        print("NEWSAPI_KEY is not set; cannot refresh sentiment", file=sys.stderr)
        return 1
    try:
        articles = fetch_articles(
            args.query,
            api_key=api_key,
            language=args.language,
            country=args.country or None,
            page_size=args.page_size,
            max_pages=args.max_pages,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Failed to download headlines: {exc}", file=sys.stderr)
        return 2
    score = aggregate_score(articles)
    if score is None:
        print("No relevant headlines returned; sentiment not updated", file=sys.stderr)
        return 3
    bias = determine_bias(score)
    representative = choose_representative(articles, score)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    output_path = output_dir / SENTIMENT_FILENAME
    write_sentiment(score=score, bias=bias, article=representative, output_path=output_path, expires_minutes=args.expires_minutes)
    print(f"Wrote sentiment snapshot to {output_path} (score={score:.2f}, bias={bias})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

""Utilities for fetching and maintaining BTCUSD intraday sentiment snapshots."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - fallback shim exercised in CI without requests
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in CI without requests
    import json as _json
    from typing import Any as _Any
    from urllib import error as _urlerror, request as _urlrequest
    from urllib.parse import urlencode as _urlencode

    class _RequestException(Exception):
        """Fallback replacement for requests.RequestException."""

    class _Response:
        def __init__(self, *, status_code: int, body: bytes, headers: _Any = None) -> None:
            self.status_code = status_code
            self._body = body
            self.headers = headers

        @property
        def text(self) -> str:
            return self._body.decode("utf-8", errors="replace")

        def json(self) -> _Any:
            return _json.loads(self.text)

    class _Session:
        def get(
            self,
            url: str,
            *,
            params: Optional[dict] = None,
            headers: Optional[dict] = None,
            timeout: Optional[int] = None,
        ) -> _Response:
            if params:
                query = _urlencode(params, doseq=True)
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}{query}"
            req = _urlrequest.Request(url, headers=headers or {})
            try:
                with _urlrequest.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
                    return _Response(status_code=resp.getcode(), body=body, headers=resp.headers)
            except _urlerror.HTTPError as exc:
                body = exc.read()
                return _Response(status_code=exc.code, body=body, headers=exc.headers)
            except _urlerror.URLError as exc:  # pragma: no cover - network errors are rare in tests
                raise _RequestException(str(exc)) from exc

    class _RequestsShim:
        Session = _Session
        RequestException = _RequestException

    requests = _RequestsShim()  # type: ignore[assignment]


SENTIMENT_FILENAME = "news_sentiment.json"

DEFAULT_QUERY = "bitcoin OR btcusd OR \"btc price\" OR \"bitcoin etf\""
DEFAULT_LANGUAGE = "en"
DEFAULT_COUNTRY: Optional[str] = None
DEFAULT_PAGE_SIZE = 30
DEFAULT_TIMEOUT = 10
DEFAULT_EXPIRY_MINUTES = 60
MAX_PAGES = 3
DEFAULT_MIN_INTERVAL = 300
 
# Simple lexicon tuned for crypto headlines.  We bias towards liquidity and
# policy keywords because they often foreshadow intraday volatility in BTC.
POSITIVE_KEYWORDS = {
    "adoption",
    "approval",
    "bullish",
    "etf",
    "fidelity",
    "halving",
    "inflow",
    "institutional",
    "listing",
    "longs",
    "record",
    "rally",
    "surge",
}
NEGATIVE_KEYWORDS = {
    "ban",
    "bearish",
    "crackdown",
    "dump",
    "hack",
    "liquidation",
    "outflow",
    "regulatory",
    "selloff",
    "shorts",
    "slump",
    "tax",
}
WEIGHTED_PHRASES = {
    "spot etf": 1.5,
    "rate cut": 0.6,
    "risk-off": -0.8,
    "risk on": 0.5,
    "margin call": -1.2,
    "leverage wipeout": -1.5,
    "all-time high": 1.2,
    "halving": 0.8,
    "miners capitulate": -1.0,
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


def _weighted_term_score(text: str) -> float:
    score = 0.0
    for phrase, weight in WEIGHTED_PHRASES.items():
        if phrase in text:
            score += weight
    return score


def score_text(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    base = 0.0
    for token in tokens:
        if token in POSITIVE_KEYWORDS:
            base += 1.0
        elif token in NEGATIVE_KEYWORDS:
            base -= 1.0
    base += _weighted_term_score(" ".join(tokens))
    # Normalise by headline length but avoid excessive damping.
    normaliser = max(len(tokens) / 12.0, 1.0)
    score = base / normaliser
    return max(-1.0, min(1.0, score))


def aggregate_score(articles: Iterable[Article]) -> Optional[float]:
    scores = [score_text(article.text) for article in articles]
    if not scores:
        return None
    return max(-1.0, min(1.0, sum(scores) / len(scores)))


def determine_bias(score: float) -> str:
    if score > 0.25:
        return "btc_bullish"
    if score < -0.25:
        return "btc_bearish"
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


def refresh_btcusd_sentiment(
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
    "refresh_btcusd_sentiment",
    "DEFAULT_QUERY",
    "DEFAULT_LANGUAGE",
    "DEFAULT_COUNTRY",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_EXPIRY_MINUTES",
    "MAX_PAGES",
    "DEFAULT_MIN_INTERVAL",
]

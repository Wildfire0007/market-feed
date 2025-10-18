"""Automate maintenance of USDJPY macro news sentiment snapshots."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from usdjpy_sentiment import (
    DEFAULT_COUNTRY,
    DEFAULT_EXPIRY_MINUTES,
    DEFAULT_LANGUAGE,
    DEFAULT_PAGE_SIZE,
    DEFAULT_QUERY,
    MAX_PAGES,
    SENTIMENT_FILENAME,
    aggregate_score,
    choose_representative,
    determine_bias,
    ensure_output_dir,
    fetch_articles,
    write_sentiment,
)


def _env_api_key() -> Optional[str]:
    api_key = os.getenv("NEWSAPI_KEY")
    return api_key.strip() if api_key else None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_output_dir = Path(os.getenv("PUBLIC_DIR", "public")) / "USDJPY"

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
        default=str(default_output_dir),
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
    write_sentiment(
        score=score,
        bias=bias,
        article=representative,
        output_path=output_path,
        expires_minutes=args.expires_minutes,
    )
    print(f"Wrote sentiment snapshot to {output_path} (score={score:.2f}, bias={bias})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

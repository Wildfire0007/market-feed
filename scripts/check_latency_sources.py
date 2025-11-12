#!/usr/bin/env python3
"""Összegző eszköz a Twelve Data / Finnhub késleltetés monitorozásához.

A script a ``public/<ASSET>/klines_*_meta.json`` állományokat olvassa ki és
egységes JSON riportot generál ``public/monitoring/data_latency.json`` néven.

Felhasználás::

    python scripts/check_latency_sources.py \
        --public-dir public \
        --output public/monitoring/data_latency.json

A kimenet a monitoring dashboardon grafikon inputként is felhasználható.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


TIMEFRAME_FILES: Dict[str, str] = {
    "k1m": "klines_1m_meta.json",
    "k5m": "klines_5m_meta.json",
    "k1h": "klines_1h_meta.json",
    "k4h": "klines_4h_meta.json",
}


@dataclass
class LatencySample:
    provider: str
    asset: str
    timeframe: str
    latency_seconds: Optional[float]
    freshness_limit_seconds: Optional[float]
    latest_utc: Optional[str]
    retrieved_at_utc: Optional[str]


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        result = float(value)
        if not math.isfinite(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _detect_provider(source: Optional[str]) -> str:
    if not source:
        return "unknown"
    if isinstance(source, str):
        token = source.split(":", 1)[0].strip().lower()
        if token:
            return token
    return "unknown"


def _load_latency_samples(public_dir: Path, assets: Iterable[str]) -> Iterable[LatencySample]:
    for asset in assets:
        asset_dir = public_dir / asset
        if not asset_dir.exists():
            continue
        for timeframe, filename in TIMEFRAME_FILES.items():
            path = asset_dir / filename
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            provider = _detect_provider(payload.get("source"))
            latency = _parse_float(payload.get("latency_seconds"))
            freshness = _parse_float(payload.get("freshness_limit_seconds"))
            latest_utc = payload.get("latest_utc") if isinstance(payload.get("latest_utc"), str) else None
            retrieved_at = (
                payload.get("retrieved_at_utc")
                if isinstance(payload.get("retrieved_at_utc"), str)
                else None
            )
            yield LatencySample(
                provider=provider,
                asset=asset,
                timeframe=timeframe,
                latency_seconds=latency,
                freshness_limit_seconds=freshness,
                latest_utc=latest_utc,
                retrieved_at_utc=retrieved_at,
            )


def _collect_assets(public_dir: Path) -> Iterable[str]:
    cfg_path = Path("config/analysis_settings.json")
    assets: Iterable[str] = []
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    asset_list = data.get("assets") if isinstance(data, dict) else None
    if isinstance(asset_list, list):
        assets = [str(item) for item in asset_list if str(item).strip()]
    else:
        # Fallback: list directories under public
        assets = [path.name for path in public_dir.iterdir() if path.is_dir()]
    return sorted(dict.fromkeys(assets))


def generate_latency_report(public_dir: Path) -> Dict[str, Any]:
    assets = _collect_assets(public_dir)
    samples = list(_load_latency_samples(public_dir, assets))

    provider_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "latency_avg_seconds": None,
        "latency_max_seconds": None,
        "stale_count": 0,
    })

    provider_latency_accumulator: Dict[str, float] = defaultdict(float)
    provider_latency_counts: Dict[str, int] = defaultdict(int)

    asset_payload: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        asset_section = asset_payload.setdefault(sample.asset, {"timeframes": {}})
        asset_section["timeframes"][sample.timeframe] = {
            "provider": sample.provider,
            "latency_seconds": sample.latency_seconds,
            "freshness_limit_seconds": sample.freshness_limit_seconds,
            "latest_utc": sample.latest_utc,
            "retrieved_at_utc": sample.retrieved_at_utc,
        }

        stats = provider_stats[sample.provider]
        stats["count"] += 1
        if sample.latency_seconds is not None:
            provider_latency_accumulator[sample.provider] += sample.latency_seconds
            provider_latency_counts[sample.provider] += 1
            current_max = stats.get("latency_max_seconds")
            stats["latency_max_seconds"] = (
                max(float(current_max), sample.latency_seconds)
                if current_max is not None
                else sample.latency_seconds
            )
            if (
                sample.freshness_limit_seconds is not None
                and sample.latency_seconds > sample.freshness_limit_seconds
            ):
                stats["stale_count"] += 1

    for provider, count in provider_latency_counts.items():
        if count:
            avg_latency = provider_latency_accumulator[provider] / count
            provider_stats[provider]["latency_avg_seconds"] = round(float(avg_latency), 3)

    generated_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "generated_utc": generated_utc,
        "public_dir": str(public_dir),
        "assets": asset_payload,
        "providers": provider_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency monitor összegzés")
    default_public = Path(os.getenv("PUBLIC_DIR", "public"))
    default_output = default_public / "monitoring" / "data_latency.json"
    parser.add_argument("--public-dir", type=Path, default=default_public, help="A public könyvtár útvonala")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="A generált riport célfájlja",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Formázott (indentált) JSON kimenet mentése",
    )
    args = parser.parse_args()

    report = generate_latency_report(args.public_dir)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2 if args.pretty else None)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

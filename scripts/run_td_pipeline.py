#!/usr/bin/env python3
"""Orchestrate the TD-only pipeline with optional preprocessing stages."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent


def _ensure_repo_on_path() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def _run_step(label: str, cmd: Sequence[str], optional: bool = False) -> int:
    """Execute ``cmd`` inside the repository root and report failures."""

    pretty_cmd = " ".join(cmd)
    print(f"\n▶ {label}: {pretty_cmd}")
    result = subprocess.run(cmd, cwd=_REPO_ROOT)
    if result.returncode != 0:
        message = f"{label} exited with code {result.returncode}"
        if optional:
            print(f"⚠️  {message}")
        else:
            raise SystemExit(message)
    return int(result.returncode)


def _news_available(force: bool) -> bool:
    if force:
        return True
    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    return bool(api_key)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trading → Analysis pipeline")
    parser.add_argument("--skip-trading", action="store_true", help="Skip Trading.py")
    parser.add_argument(
        "--skip-volatility",
        action="store_true",
        help="Skip volatility overlay refresh",
    )
    parser.add_argument(
        "--volatility-optional",
        action="store_true",
        help="Do not abort the pipeline when the overlay refresh fails",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=120,
        help="Volatility overlay window size in minutes",
    )
    parser.add_argument(
        "--vol-output",
        default="vol_overlay.json",
        help="Filename to store the overlay snapshot (per asset)",
    )
    parser.add_argument(
        "--assets",
        nargs="*",
        help="Optional asset filter for the overlay generator",
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Skip BTCUSD sentiment refresh step",
    )
    parser.add_argument(
        "--force-news",
        action="store_true",
        help="Attempt the news refresh even without NEWSAPI_KEY",
    )
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis.py")
    parser.add_argument(
        "--skip-discord",
        action="store_true",
        help="Skip Discord notification dispatch",
    )
    parser.add_argument(
        "--auto-train-models",
        action="store_true",
        help="Run the Feature→Label→Train automation after analysis",
    )
    parser.add_argument(
        "--auto-train-arg",
        action="append",
        default=[],
        help="Additional arguments forwarded to auto_train_models.py",
    )
    parser.add_argument(
        "--skip-watchdog",
        action="store_true",
        help="Skip latency watchdog enforcement",
    )
    parser.add_argument(
        "--watchdog-threshold",
        type=float,
        default=15.0,
        help="Latency threshold in minutes before triggering restart (default: 15)",
    )
    parser.add_argument(
        "--watchdog-verify-seconds",
        type=float,
        default=180.0,
        help="How long to wait for analysis summary refresh after restart (default: 180)",
    )
    parser.add_argument(
        "--watchdog-loop-seconds",
        type=float,
        default=0.0,
        help="Optional watchdog polling interval in seconds (default: run once)",
    )
    parser.add_argument(
        "--notify-arg",
        action="append",
        default=[],
        help="Additional arguments forwarded to notify_discord.py",
    )
    parser.add_argument(
        "--public-dir",
        default=os.getenv("PUBLIC_DIR", "public"),
        help="Public artefact directory shared across steps",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    _ensure_repo_on_path()
    args = parse_args(argv)
    python = sys.executable or "python3"

    if not args.skip_trading:
        _run_step("Trading", [python, "Trading.py"])

    if not args.skip_volatility:
        vol_cmd: List[str] = [
            python,
            "-m",
            "volatility_metrics",
            "--public-dir",
            args.public_dir,
            "--window",
            str(max(1, args.vol_window)),
            "--output",
            args.vol_output,
        ]
        if args.assets:
            vol_cmd.append("--assets")
            vol_cmd.extend(asset.upper() for asset in args.assets)
        _run_step(
            "Volatility overlay",
            vol_cmd,
            optional=args.volatility_optional,
        )

    if not args.skip_news:
        if _news_available(args.force_news):
            _run_step(
                "BTCUSD sentiment",
                [python, "scripts/update_btcusd_sentiment.py"],
                optional=True,
            )
        else:
            print(
                "⚠️  Skipping BTCUSD sentiment refresh (NEWSAPI_KEY not configured)",
                file=sys.stderr,
            )

    if not args.skip_analysis:
        _run_step("Analysis", [python, "analysis.py"])

    if args.auto_train_models:
        auto_cmd = [python, "scripts/auto_train_models.py", "--public-dir", args.public_dir]
        auto_cmd.extend(args.auto_train_arg)
        _run_step("Auto-train models", auto_cmd)
        
    if not args.skip_discord:
        notify_cmd = [python, "scripts/notify_discord.py"]
        notify_cmd.extend(args.notify_arg)
        _run_step("Discord notify", notify_cmd, optional=True)

    if not args.skip_watchdog:
        public_dir = Path(args.public_dir)
        public_dir.mkdir(parents=True, exist_ok=True)
        summary_path = public_dir / "analysis_summary.json"
        monitoring_dir = public_dir / "monitoring"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        state_path = monitoring_dir / "td_latency_watchdog.json"
        cache_bust_path = monitoring_dir / "cache_bust.json"
        watchdog_loop = max(float(args.watchdog_loop_seconds), 0.0)

        watchdog_cmd: List[str] = [
            python,
            "scripts/td_latency_watchdog.py",
            "--summary",
            str(summary_path),
            "--threshold-minutes",
            f"{max(float(args.watchdog_threshold), 0.0):g}",
            "--verify-refresh-seconds",
            f"{max(float(args.watchdog_verify_seconds), 0.0):g}",
            "--loop-seconds",
            f"{watchdog_loop:g}",
            "--cache-bust",
            "--cache-bust-path",
            str(cache_bust_path),
            "--state-path",
            str(state_path),
        ]

        watchdog_cmd.extend(["--restart-cmd", python, "Trading.py"])
        _run_step("Latency watchdog", watchdog_cmd)
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

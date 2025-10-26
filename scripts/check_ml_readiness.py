#!/usr/bin/env python3
"""Utility to verify ML scoring readiness for all configured assets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.analysis_settings import ASSETS
from ml_model import inspect_model_artifact, runtime_dependency_issues


def _normalise_assets(assets: Sequence[str]) -> List[str]:
    return sorted({asset.upper() for asset in assets})


def _format_status(status: str) -> str:
    if status == "ok":
        return "OK"
    return status.upper()


def run_diagnostics(assets: Iterable[str]) -> int:
    exit_code = 0

    issues = runtime_dependency_issues()
    if issues:
        exit_code = 1
        print("⚠️  Függőségi problémák észlelve:")
        for name, detail in issues.items():
            print(f"  - {name}: {detail}")
        print()

    print("ML modell diagnosztika:")
    for asset in assets:
        info = inspect_model_artifact(asset)
        status = info.get("status", "unknown")
        if status != "ok":
            exit_code = 1
        status_label = _format_status(status)
        print(f"- {asset}: {status_label}")
        extra = {
            key: value
            for key, value in info.items()
            if key not in {"asset", "status"} and value is not None
        }
        if extra:
            formatted = json.dumps(extra, ensure_ascii=False, indent=2)
            for line in formatted.splitlines():
                print(f"    {line}")

    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "assets",
        nargs="*",
        help="Csak a felsorolt eszközöket ellenőrzi (alapértelmezés: összes konfigurált)",
    )
    args = parser.parse_args(argv)
    assets = _normalise_assets(args.assets or ASSETS)
    return run_diagnostics(assets)


if __name__ == "__main__":
    sys.exit(main())

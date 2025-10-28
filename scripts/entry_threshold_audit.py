"""Audit helper for entry threshold profiles and P-score gates.

The script inspects the active entry threshold profile together with the
``public/analysis_summary.json`` export and highlights where the
probability gate blocks signals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import analysis_settings as settings


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Analysis summary not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or "assets" not in data:
        raise ValueError("Invalid analysis summary format: missing 'assets'")

    return data


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _collect_blockers(meta: Dict[str, Any]) -> List[str]:
    blockers: List[str] = []

    def _extend(container: Dict[str, Any]) -> None:
        raw = container.get("blockers_raw") or container.get("blockers")
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            blockers.extend(str(item) for item in raw)

    action_plan = meta.get("action_plan")
    if isinstance(action_plan, dict):
        _extend(action_plan)

    active_meta = meta.get("active_position_meta")
    if isinstance(active_meta, dict):
        plan = active_meta.get("action_plan")
        if isinstance(plan, dict):
            _extend(plan)

    return blockers


def _extract_p_score(meta: Dict[str, Any]) -> float | None:
    active_meta = meta.get("active_position_meta")
    if isinstance(active_meta, dict):
        score = _safe_float(active_meta.get("p_score"))
        if score is not None:
            return score

    return _safe_float(meta.get("probability"))


def _extract_snapshot_threshold(meta: Dict[str, Any]) -> float | None:
    thresholds = meta.get("entry_thresholds")
    if isinstance(thresholds, dict):
        return _safe_float(thresholds.get("p_score_min_effective"))
    return None


def audit(summary_path: Path, suggest_buffer: float) -> None:
    summary = _load_summary(summary_path)
    profile = settings.describe_entry_threshold_profile()
    p_score_section = profile["p_score_min"]
    thresholds_by_asset: Dict[str, float] = {
        asset: float(value)
        for asset, value in p_score_section["by_asset"].items()
    }
    profile_default = float(p_score_section["default"])

    print(f"Active profile: {profile['name']}")
    print(
        "Using P-score defaults: "
        f"default={profile_default:.1f}; overrides={len(thresholds_by_asset)}"
    )

    header = (
        f"{'Asset':<8}{'P-score':>9}{'CfgThr':>9}{'SnapThr':>9}"
        f"{'Δ':>8}{'Gate':>8}{'Suggest':>10}"
    )
    print(header)
    print("-" * len(header))

    below_threshold = 0
    gate_blocked = 0

    for asset in settings.ASSETS:
        meta = summary.get("assets", {}).get(asset, {})
        p_score = _extract_p_score(meta)
        cfg_threshold = thresholds_by_asset.get(asset, profile_default)
        snap_threshold = _extract_snapshot_threshold(meta)
        blockers = _collect_blockers(meta)
        gate_flag = any("p_score" in blocker.lower() for blocker in blockers)
        if gate_flag:
            gate_blocked += 1

        delta = (
            None
            if p_score is None or cfg_threshold is None
            else p_score - cfg_threshold
        )
        if delta is not None and delta < 0:
            below_threshold += 1

        suggested = (
            None
            if p_score is None
            else max(20.0, round(p_score - suggest_buffer, 1))
        )

        def _fmt(value: float | None) -> str:
            return "   n/a" if value is None else f"{value:9.1f}"

        def _fmt_delta(value: float | None) -> str:
            return "   n/a" if value is None else f"{value:8.1f}"

        suggest_repr = "   n/a" if suggested is None else f"{suggested:10.1f}"
        gate_repr = "   yes" if gate_flag else "    no"

        print(
            f"{asset:<8}"
            f"{_fmt(p_score)}"
            f"{_fmt(cfg_threshold)}"
            f"{_fmt(snap_threshold)}"
            f"{_fmt_delta(delta)}"
            f"{gate_repr}"
            f"{suggest_repr}"
        )

    print()
    assets_considered = len(settings.ASSETS)
    print(
        f"Assets below configured threshold: {below_threshold}/{assets_considered}"
    )
    print(f"Assets with P-score gate blocker: {gate_blocked}/{assets_considered}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose entry threshold profile against analysis summary",
    )
    parser.add_argument(
        "summary",
        nargs="?",
        default="public/analysis_summary.json",
        type=Path,
        help="Path to analysis_summary.json (default: public/analysis_summary.json)",
    )
    parser.add_argument(
        "--suggest-buffer",
        type=float,
        default=5.0,
        help="Buffer (in score points) subtracted when suggesting relaxed thresholds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit(Path(args.summary), suggest_buffer=args.suggest_buffer)


if __name__ == "__main__":
    main()

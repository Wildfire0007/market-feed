#!/usr/bin/env python3
"""
Diagnostic helper that aggregates entry gate rejection reasons per asset.

Data sources (Step 1 discovery):
- public/monitoring/pipeline.log → JSON lines emitted by the full TD pipeline. Each
  line with message="gate_summary" contains a "gate_summary" payload with the
  evaluated asset, the active profile/mode and, crucially, a "missing" list that
  enumerates which gates blocked an entry on that run.
- public/analysis_summary.json → Latest pipeline snapshot. The "assets" object
  mirrors the per-asset decision and repeats the gate verdict via
  assets[SYMBOL]["gates"]["missing"], while action_plan["blockers_raw"] and the
  top-level "reasons" array provide textual explanations for the active
  blockages.

The script reads the log history to quantify how often every gate prevents
entries per symbol and enriches the report with the current snapshot context.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional, Sequence


def _parse_iso8601(value: str) -> Optional[datetime]:
    """Parse ISO-8601 timestamps that optionally end with "Z"."""
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class GateEventStats:
    total_events: int = 0
    gate_counts: Counter[str] = field(default_factory=Counter)
    profile_counts: Counter[str] = field(default_factory=Counter)
    mode_counts: Counter[str] = field(default_factory=Counter)
    p_scores: list[float] = field(default_factory=list)
    p_score_min: list[float] = field(default_factory=list)
    atr_rel: list[float] = field(default_factory=list)
    recent_events: Deque[Mapping[str, Any]] = field(default_factory=lambda: deque(maxlen=5))

    def register_event(self, summary: Mapping[str, Any], raw_event: Mapping[str, Any]) -> None:
        self.total_events += 1
        missing = summary.get("missing") or []
        for gate in set(missing):
            self.gate_counts[gate] += 1
        profile = summary.get("profile")
        if profile:
            self.profile_counts[profile] += 1
        mode = summary.get("mode")
        if mode:
            self.mode_counts[mode] += 1
        for key, bucket in (("p_score", self.p_scores), ("p_score_min", self.p_score_min), ("atr_rel", self.atr_rel)):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                bucket.append(float(value))
        self.recent_events.append(
            {
                "timestamp": raw_event.get("timestamp"),
                "missing": list(missing),
                "p_score": summary.get("p_score"),
                "p_score_min": summary.get("p_score_min"),
            }
        )


def analyze_pipeline_log(log_path: Path, *, since: Optional[datetime], assets: Optional[Sequence[str]]) -> dict[str, GateEventStats]:
    stats: dict[str, GateEventStats] = defaultdict(GateEventStats)
    if not log_path.exists():
        return {}
    asset_filter = {a.upper() for a in assets} if assets else None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("message") != "gate_summary":
                continue
            asset = (event.get("asset") or "").upper()
            if not asset:
                continue
            if asset_filter and asset not in asset_filter:
                continue
            event_ts = _parse_iso8601(event.get("timestamp"))
            if since and event_ts and event_ts < since:
                continue
            summary = event.get("gate_summary") or {}
            stats[asset].register_event(summary, event)
    return stats


def load_latest_snapshot(snapshot_path: Path) -> dict[str, dict[str, Any]]:
    if not snapshot_path.exists():
        return {}
    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    assets_payload = data.get("assets")
    if not isinstance(assets_payload, Mapping):
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for asset, payload in assets_payload.items():
        if not isinstance(payload, Mapping):
            continue
        gates = payload.get("gates")
        missing: list[str] = []
        if isinstance(gates, Mapping):
            missing = list(gates.get("missing") or [])
        action_plan = payload.get("action_plan")
        blockers_raw: list[str] = []
        blockers_pretty: list[str] = []
        if isinstance(action_plan, Mapping):
            blockers_raw = list(action_plan.get("blockers_raw") or [])
            blockers_pretty = list(action_plan.get("blockers") or [])
        reasons = payload.get("reasons")
        reason_list = list(reasons or []) if isinstance(reasons, list) else []
        latest[asset.upper()] = {
            "missing": missing,
            "blockers_raw": blockers_raw,
            "blockers": blockers_pretty,
            "reasons": reason_list,
            "signal": payload.get("signal"),
            "probability": payload.get("probability"),
            "retrieved_at_utc": payload.get("retrieved_at_utc"),
        }
    return latest


def render_text_report(
    stats: Mapping[str, GateEventStats],
    latest: Mapping[str, Mapping[str, Any]],
    *,
    assets: Optional[Sequence[str]],
    top_n: int,
) -> None:
    all_assets = {asset.upper() for asset in assets} if assets else set(stats.keys()) | set(latest.keys())
    for asset in sorted(all_assets):
        print(f"\n=== {asset} ===")
        asset_stats = stats.get(asset)
        if not asset_stats or asset_stats.total_events == 0:
            print("No gate_summary observations in log.")
        else:
            print(f"Observed gate_summary events: {asset_stats.total_events}")
            for gate, count in asset_stats.gate_counts.most_common(top_n):
                share = (count / asset_stats.total_events) * 100 if asset_stats.total_events else 0.0
                print(f"  - {gate:<24} {count:>5} events  ({share:5.1f}% of runs)")
            if len(asset_stats.gate_counts) > top_n:
                remaining = len(asset_stats.gate_counts) - top_n
                print(f"  ... {remaining} additional gate(s) with lower frequency")
            if asset_stats.p_scores:
                mean_p = statistics.fmean(asset_stats.p_scores)
                min_p = min(asset_stats.p_scores)
                max_p = max(asset_stats.p_scores)
                mean_min = statistics.fmean(asset_stats.p_score_min) if asset_stats.p_score_min else None
                line = f"Average P-score: {mean_p:.1f} (min {min_p:.1f}, max {max_p:.1f})"
                if mean_min is not None:
                    line += f" — required avg {mean_min:.1f}"
                print(f"  {line}")
            if asset_stats.atr_rel:
                mean_atr = statistics.fmean(asset_stats.atr_rel)
                print(f"  Average ATR ratio vs threshold: {mean_atr:.4f}")
            recent = list(asset_stats.recent_events)
            if recent:
                print("  Recent missing gates (most recent last):")
                for event in recent:
                    ts = event.get("timestamp") or "?"
                    missing = ", ".join(event.get("missing") or []) or "<none>"
                    print(f"    • {ts}: {missing}")
        latest_snapshot = latest.get(asset)
        if latest_snapshot:
            missing = ", ".join(latest_snapshot.get("missing") or []) or "<none>"
            print(f"Latest snapshot missing gates: {missing}")
            blockers = ", ".join(latest_snapshot.get("blockers_raw") or []) or "<none>"
            print(f"Action plan blockers (raw): {blockers}")
            reasons = latest_snapshot.get("reasons") or []
            if reasons:
                print("Sample textual reasons:")
                for reason in reasons[:top_n]:
                    print(f"  • {reason}")
                extra = len(reasons) - top_n
                if extra > 0:
                    print(f"  • ... ({extra} more reason(s) omitted)")
            else:
                print("No textual reasons recorded in latest snapshot.")
        else:
            print("Latest snapshot unavailable for this asset.")


def render_json_report(
    stats: Mapping[str, GateEventStats],
    latest: Mapping[str, Mapping[str, Any]],
    *,
    assets: Optional[Sequence[str]],
) -> None:
    asset_filter = {asset.upper() for asset in assets} if assets else None
    output: dict[str, Any] = {"assets": {}}
    asset_names = asset_filter or (set(stats.keys()) | set(latest.keys()))
    for asset in sorted(asset_names):
        entry: dict[str, Any] = {}
        asset_stats = stats.get(asset)
        if asset_stats and asset_stats.total_events:
            entry["events"] = asset_stats.total_events
            entry["gate_counts"] = dict(asset_stats.gate_counts)
            if asset_stats.profile_counts:
                entry["profile_counts"] = dict(asset_stats.profile_counts)
            if asset_stats.mode_counts:
                entry["mode_counts"] = dict(asset_stats.mode_counts)
            if asset_stats.p_scores:
                entry["p_score"] = {
                    "mean": statistics.fmean(asset_stats.p_scores),
                    "min": min(asset_stats.p_scores),
                    "max": max(asset_stats.p_scores),
                }
            if asset_stats.p_score_min:
                entry["p_score_min"] = {
                    "mean": statistics.fmean(asset_stats.p_score_min),
                    "min": min(asset_stats.p_score_min),
                    "max": max(asset_stats.p_score_min),
                }
            if asset_stats.atr_rel:
                entry["atr_rel"] = {
                    "mean": statistics.fmean(asset_stats.atr_rel),
                    "min": min(asset_stats.atr_rel),
                    "max": max(asset_stats.atr_rel),
                }
            entry["recent_events"] = list(asset_stats.recent_events)
        if asset in latest:
            entry["latest_snapshot"] = latest[asset]
        output["assets"][asset] = entry
    print(json.dumps(output, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate pipeline gate rejection frequencies per asset."
    )
    parser.add_argument(
        "--pipeline-log",
        default="public/monitoring/pipeline.log",
        type=Path,
        help="Path to the pipeline JSONL log (defaults to public/monitoring/pipeline.log).",
    )
    parser.add_argument(
        "--analysis-summary",
        default="public/analysis_summary.json",
        type=Path,
        help="Path to the latest analysis summary snapshot (defaults to public/analysis_summary.json).",
    )
    parser.add_argument(
        "--since",
        help="Only include gate_summary log lines at or after this ISO8601 timestamp.",
    )
    parser.add_argument(
        "--assets",
        nargs="*",
        help="Optional list of asset symbols to include (defaults to all discovered).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of gate/reason entries to display per asset in text mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of human-readable text.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    since_dt = _parse_iso8601(args.since) if args.since else None
    stats = analyze_pipeline_log(args.pipeline_log, since=since_dt, assets=args.assets)
    latest = load_latest_snapshot(args.analysis_summary)
    if args.json:
        render_json_report(stats, latest, assets=args.assets)
    else:
        render_text_report(stats, latest, assets=args.assets, top_n=args.top)


if __name__ == "__main__":
    main()

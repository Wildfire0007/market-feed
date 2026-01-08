from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_entry_gates import (
    SymbolStats,
    apply_summary_deltas,
    build_daily_visualization,
    build_json_summary,
)


def test_entry_gate_stats_summary_is_deterministic():
    stats = {
        "ZSYM": SymbolStats(
            total_candidates=2,
            total_rejected=1,
            by_reason=Counter({"b_reason": 1, "a_reason": 2}),
        ),
        "ASYM": SymbolStats(total_candidates=1, total_rejected=0),
    }
    stats["ZSYM"].total_candidates_by_tod.update({"open": 2})
    summary = build_json_summary(stats)

    assert list(summary.keys()) == ["ASYM", "ZSYM"]
    assert list(summary["ZSYM"]["by_reason"].keys()) == ["a_reason", "b_reason"]
    assert summary["ZSYM"]["by_time_of_day"]["open"]["total_candidates"] == 2


def test_daily_visualization_is_bucketised_and_sorted():
    stats = {
        "EURUSD": SymbolStats(total_candidates=0, total_rejected=0),
    }
    stats["EURUSD"].daily_candidates.update({("2024-01-01", "open"): 2, ("2024-01-02", "mid"): 1})
    stats["EURUSD"].daily_rejections.update({("2024-01-02", "mid"): 1})

    daily = build_daily_visualization(stats)
    assert list(daily.keys()) == ["2024-01-01", "2024-01-02"]
    assert daily["2024-01-01"]["open"]["total_candidates"] == 2
    assert daily["2024-01-01"]["open"].get("total_rejected", 0) == 0
    assert daily["2024-01-02"]["mid"]["total_rejected"] == 1


def test_summary_delta_tracking():
    summary = {"EURUSD": {"total_candidates": 3, "total_rejected": 2}}
    previous = {"EURUSD": {"total_candidates": 1, "total_rejected": 1}}

    updated = apply_summary_deltas(summary, previous)

    assert updated["EURUSD"]["total_candidates_delta"] == 2
    assert updated["EURUSD"]["total_rejected_delta"] == 1

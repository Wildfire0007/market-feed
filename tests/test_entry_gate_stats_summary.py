from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_entry_gates import SymbolStats, build_json_summary


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

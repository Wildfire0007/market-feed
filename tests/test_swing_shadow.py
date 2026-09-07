import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.swing_shadow import (
    MAX_BARS_HELD,
    compute_atr,
    detect_signal,
    net_r,
    process_asset,
)


def mkbars(closes, start_day=1, spread=0.5):
    bars = []
    for i, c in enumerate(closes):
        day = start_day + i // 6
        hour = (i % 6) * 4
        bars.append(
            {
                "dt": f"2026-01-{day:02d} {hour:02d}:00:00",
                "open": c,
                "high": c + spread,
                "low": c - spread,
                "close": c,
            }
        )
    return bars


def flat_then(base, n_flat, tail):
    return [base] * n_flat + tail


def test_signal_long_breakout_no_lookahead():
    closes = [100.0] * 20 + [101.0]
    assert detect_signal(closes, 20) == 1
    assert detect_signal(closes, 19) == 0


def test_signal_none_at_equal_close():
    closes = [100.0] * 20 + [100.0]
    assert detect_signal(closes, 20) == 0


def test_signal_short_breakdown():
    closes = [100.0] * 20 + [99.0]
    assert detect_signal(closes, 20) == -1


def test_atr_warmup_and_positive():
    bars = mkbars([100 + (i % 3) for i in range(40)])
    atr = compute_atr(bars)
    assert atr[13] is None and atr[14] is not None and atr[14] > 0


def test_entry_applies_floor_risk():
    closes = flat_then(100.0, 30, [101.0, 101.0])
    bars = mkbars(closes)
    st, events = process_asset("GOLD_CFD", bars, {})
    entries = [e for e in events if e["event"] == "entry"]
    assert len(entries) == 1
    e = entries[0]
    assert e["risk"] >= 0.015 * e["entry"] - 1e-9


def test_no_second_entry_while_open():
    closes = flat_then(100.0, 30, [101.0, 102.0, 103.0, 104.0])
    bars = mkbars(closes)
    st, events = process_asset("GOLD_CFD", bars, {})
    assert len([e for e in events if e["event"] == "entry"]) == 1


def test_trailing_stop_only_tightens_and_exits_on_close_cross():
    closes = flat_then(100.0, 30, [101.0, 110.0, 120.0, 80.0, 80.0])
    bars = mkbars(closes)
    st, events = process_asset("GOLD_CFD", bars, {})
    exits = [e for e in events if e["event"] == "exit"]
    assert len(exits) == 1
    assert exits[0]["reason"] == "trail_stop"
    assert st["position"] is None


def test_time_limit_exit():
    closes = flat_then(100.0, 30, [101.0] + [101.0] * (MAX_BARS_HELD + 2))
    bars = mkbars(closes)
    st, events = process_asset("GOLD_CFD", bars, {})
    exits = [e for e in events if e["event"] == "exit"]
    assert len(exits) == 1 and exits[0]["reason"] == "time_limit"


def test_net_r_cost_math():
    r = net_r(1, 100.0, 103.0, 1.5, "GOLD_CFD", 5)
    gross = 3.0 / 1.5
    cost = (0.0006 * 100.0 + 0.0002 * 100.0 * 5) / 1.5
    assert abs(r - (gross - cost)) < 1e-12


def test_idempotent_state_advance():
    closes = flat_then(100.0, 30, [101.0, 102.0])
    bars = mkbars(closes)
    st1, ev1 = process_asset("GOLD_CFD", bars, {})
    st2, ev2 = process_asset("GOLD_CFD", bars, st1)
    assert ev2 == []
    assert st2["last_bar"] == st1["last_bar"]

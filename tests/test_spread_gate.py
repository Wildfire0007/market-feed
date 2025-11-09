"""Tests for realtime spread guard helpers."""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from analysis import _extract_bid_ask_spread, _resolve_spread_for_entry, _should_use_realtime_spot


def test_extract_bid_ask_spread_parses_strings() -> None:
    snapshot = {"bid": "100.25", "ask": "100.55"}
    assert _extract_bid_ask_spread(snapshot) == pytest.approx(0.30, rel=1e-6)


def test_extract_bid_ask_spread_rejects_inverted() -> None:
    snapshot = {"bid": 101.0, "ask": 100.0}
    assert _extract_bid_ask_spread(snapshot) is None


def test_resolve_spread_ignores_stale_realtime() -> None:
    now = datetime(2025, 11, 9, 10, 0, tzinfo=timezone.utc)
    rt_ts = datetime(2025, 10, 25, 7, 58, tzinfo=timezone.utc)
    last_spot = now - timedelta(minutes=5)
    use_rt, meta = _should_use_realtime_spot(111_511.51, rt_ts, last_spot, now, max_age_seconds=300)
    assert not use_rt
    assert meta["reason"] == "stale"
    spread = _resolve_spread_for_entry({"price": 111_511.51}, 101_594.94, 111_511.51, use_rt)
    assert spread is None


def test_resolve_spread_uses_bid_ask_when_realtime_active() -> None:
    now = datetime(2025, 11, 9, 10, 0, tzinfo=timezone.utc)
    rt_ts = now
    last_spot = now - timedelta(minutes=5)
    use_rt, _ = _should_use_realtime_spot(101_600.0, rt_ts, last_spot, now, max_age_seconds=300)
    assert use_rt
    spread = _resolve_spread_for_entry({"best_bid": 101_599.5, "best_ask": 101_600.5}, 101_550.0, 101_600.0, use_rt)
    assert spread == pytest.approx(1.0)


def test_resolve_spread_falls_back_to_price_gap() -> None:
    now = datetime(2025, 11, 9, 10, 0, tzinfo=timezone.utc)
    rt_ts = now
    last_spot = now - timedelta(minutes=5)
    use_rt, _ = _should_use_realtime_spot(101_600.0, rt_ts, last_spot, now, max_age_seconds=300)
    assert use_rt
    spread = _resolve_spread_for_entry({}, 101_550.0, 101_600.0, use_rt)
    assert spread == pytest.approx(50.0)

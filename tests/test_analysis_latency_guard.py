import datetime as dt

import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from analysis import _apply_latency_guard_adjustment


@pytest.mark.parametrize(
    "latency_seconds,retrieved_delta_minutes,expected_guard,expected_effective,expected_adjustment",
    [
        (28 * 60, 24, 4 * 60, 4 * 60, 24 * 60),
        (10 * 60, None, 10 * 60, 10 * 60, None),
        (None, 10, None, None, None),
    ],
)
def test_latency_guard_adjustment(
    latency_seconds, retrieved_delta_minutes, expected_guard, expected_effective, expected_adjustment
):
    now = dt.datetime(2025, 11, 11, 18, 0, tzinfo=dt.timezone.utc)
    retrieved_iso = None
    if retrieved_delta_minutes is not None:
        retrieved_iso = (now - dt.timedelta(minutes=retrieved_delta_minutes)).isoformat()

    guard_latency, guard_adjustment, retrieved_ts, effective_latency = _apply_latency_guard_adjustment(
        latency_seconds, retrieved_iso, now=now
    )

    assert guard_latency == expected_guard
    assert effective_latency == expected_effective
    assert guard_adjustment == expected_adjustment
    if retrieved_delta_minutes is None:
        assert retrieved_ts is None
    else:
        assert retrieved_ts.isoformat() == retrieved_iso

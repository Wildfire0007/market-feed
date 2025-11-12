from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402


def test_latency_guard_status_handles_nan_latency():
    status = analysis._latency_guard_status(
        "BTCUSD",
        {"k1m": float("nan")},
        {"BTCUSD": {"latency_k1m_sec_max": 600}},
    )
    assert status is None


def test_latency_guard_status_accepts_string_latency():
    status = analysis._latency_guard_status(
        "XAGUSD",
        {"k1m": "601"},
        {"DEFAULT": {"latency_k1m_sec_max": "600"}},
    )
    assert status is not None
    assert status["asset"] == "XAGUSD"
    assert status["age_seconds"] == 601
    assert status["limit_seconds"] == 600

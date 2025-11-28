from datetime import datetime, timezone
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


def test_latency_guard_recovery_logs_and_resets(tmp_path, caplog):
    outdir = tmp_path / "BTCUSD"
    outdir.mkdir(parents=True)
    guard_state = {
        "active": True,
        "feed": "k1m",
        "triggered_utc": "2024-01-01 00:00:00",
        "triggered_cet": "2024-01-01 01:00:00",
        "limit_seconds": 300,
    }
    analysis.save_latency_guard_state(str(outdir), guard_state)
    entry_meta: dict = {}

    caplog.set_level("INFO", logger=analysis.LOGGER.name)
    recovery = analysis._log_latency_guard_recovery(
        "BTCUSD",
        str(outdir),
        guard_state,
        datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
        "suppressed",
        entry_meta,
    )

    assert recovery is not None
    assert entry_meta.get("latency_guard_recovery", {}).get("profile") == "suppressed"
    saved_state = analysis.load_latency_guard_state(str(outdir))
    assert saved_state.get("active") is False
    assert any(record.message == "Latency guard feloldva" for record in caplog.records)

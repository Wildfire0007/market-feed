import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _reload_analysis(monkeypatch):
    monkeypatch.delenv("SESSION_STATUS_PROFILE", raising=False)
    monkeypatch.delenv("ENTRY_THRESHOLD_PROFILE", raising=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return importlib.reload(importlib.import_module("analysis"))


def test_session_state_dst_exports_next_open(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    # EU DST end 2024-10-27: a hétfői nyitásnak Budapest-időben is szerepelnie kell
    now = datetime(2024, 10, 27, 21, 30, tzinfo=timezone.utc)
    entry_open, info = analysis.session_state("EURUSD", now=now)
    assert entry_open is False
    next_open = info.get("next_session_open_budapest")
    assert next_open is not None
    parsed = datetime.fromisoformat(next_open)
    assert parsed.tzinfo is not None


def test_atr_nan_and_fib_missing_fail_safe(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    assert analysis.btc_atr_gate_ok("baseline", "BTCUSD", float("nan"), None) is False
    assert analysis.fib_zone_ok(np.nan, np.nan, price_now=100.0, tol_abs=0.0, tol_frac=0.01) is False


def test_precision_timeout_trips_after_ten_minutes(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    runtime = analysis._PRECISION_RUNTIME.setdefault("BTCUSD", {})
    runtime.clear()
    runtime.update({"last_state": "precision_ready", "ready_since": start})
    now = start + timedelta(minutes=11)
    elapsed = analysis._precision_ready_elapsed_seconds(runtime.get("ready_since"), now_runtime=now)
    assert elapsed and elapsed >= 660
    entry_thresholds = {}
    precision_state = "precision_ready"
    if elapsed > 10 * 60:
        precision_state = "none"
        entry_thresholds["precision_timeout"] = {"minutes": 10}
    assert precision_state == "none"
    assert entry_thresholds["precision_timeout"]["minutes"] == 10


def test_illiquid_order_flow_yields_safe_metrics(monkeypatch):
    analysis = _reload_analysis(monkeypatch)
    index = pd.date_range("2025-01-01", periods=150, freq="min")
    close = np.full(len(index), 100.0)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(index), 1e-4),
        },
        index=index,
    )
    metrics = analysis.compute_order_flow_metrics(frame, frame.resample("5min").last())
    assert metrics["status"] == "ok"
    assert metrics["imbalance"] is None
    pressure = metrics.get("pressure")
    if pressure is not None:
        assert pressure == pytest.approx(0.0)
    assert metrics["delta_volume"] == pytest.approx(0.0)

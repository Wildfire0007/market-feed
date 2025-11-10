import numpy as np
import pandas as pd

from analysis import compute_order_flow_metrics, compute_precision_entry


def _make_series(length: int = 150) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=length, freq="min")
    close = np.linspace(100.0, 101.0, length)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.zeros(length),
        },
        index=index,
    )
    return frame


def test_compute_order_flow_metrics_marks_volume_unavailable():
    k1m = pd.DataFrame({"close": [1.0, 1.01, 1.02]})
    metrics = compute_order_flow_metrics(k1m, pd.DataFrame())
    assert metrics["status"] == "volume_unavailable"
    assert metrics["imbalance"] is None
    assert metrics["pressure"] is None


def test_precision_entry_treats_missing_order_flow_as_optional():
    k1m = _make_series()
    k5m = _make_series(60).resample("5min").last().dropna()
    order_flow_metrics = {"status": "volume_unavailable"}
    plan = compute_precision_entry(
        asset="BTCUSD",
        direction="buy",
        k1m=k1m,
        k5m=k5m,
        price_now=100.5,
        atr5=1.2,
        order_flow_metrics=order_flow_metrics,
        score_threshold=50.0,
    )
    assert plan["order_flow_optional"] is True
    assert plan["order_flow_ready"] is True
    assert plan["order_flow_blockers"] == []
    assert any("order flow optional" in reason for reason in plan["trigger_reasons"])


def test_precision_entry_overrides_stalled_flow_when_trigger_fires():
    index = pd.date_range("2025-01-01", periods=150, freq="min")
    close = np.linspace(100.0, 101.0, len(index))
    k1m = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.ones(len(index)),
        },
        index=index,
    )
    k5m = k1m.resample("5min").last().dropna()
    order_flow_metrics = {"status": "ok", "imbalance": 0.0, "pressure": 0.0, "delta_volume": 0.0}
    plan = compute_precision_entry(
        asset="BTCUSD",
        direction="buy",
        k1m=k1m,
        k5m=k5m,
        price_now=float(close[-1]) - 0.3,
        atr5=1.2,
        order_flow_metrics=order_flow_metrics,
        score_threshold=50.0,
    )
    assert plan["trigger_state"] == "fire"
    assert plan["order_flow_stalled"] is True
    assert plan["order_flow_ready"] is True
    assert plan.get("order_flow_fallback") == "stalled_flow_trigger_fire"
    assert any("stalled" in reason for reason in plan["trigger_reasons"])

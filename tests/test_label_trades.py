import pandas as pd
import pytest

from scripts import label_trades as lt


def _make_price_frame(*rows):
    timestamps = [pd.Timestamp(ts, tz="UTC") for ts, *_ in rows]
    opens = [row[1] for row in rows]
    highs = [row[2] for row in rows]
    lows = [row[3] for row in rows]
    closes = [row[4] for row in rows]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def test_market_order_hits_take_profit():
    analysis_ts = pd.Timestamp("2024-01-01T10:00:00Z")
    trade = pd.Series(
        {
            "asset": "EURUSD",
            "journal_id": "t-1",
            "analysis_timestamp": analysis_ts.isoformat(),
            "entry_price": 1.1000,
            "stop_loss": 1.0950,
            "take_profit_1": 1.1100,
            "signal": "buy",
            "spot_price": 1.1000,
            "precision_state": "fire",
        }
    )
    prices = _make_price_frame(
        ("2024-01-01T10:00:00Z", 1.1000, 1.1050, 1.0970, 1.1005),
        ("2024-01-01T10:01:00Z", 1.1005, 1.1150, 1.1000, 1.1110),
        ("2024-01-01T10:02:00Z", 1.1110, 1.1120, 1.1080, 1.1090),
    )

    result = lt._evaluate_trade(
        trade,
        prices,
        horizon_minutes=60,
        entry_grace_minutes=5,
        entry_tolerance=0.0005,
        precision_state_column="precision_state",
        executed_precision_states={"fire", "executed"},
    )

    assert result.outcome == lt.OUTCOME_PROFIT
    assert result.label == 1
    assert result.entry_kind == "market"
    assert result.fill_timestamp == analysis_ts
    assert result.exit_timestamp == pd.Timestamp("2024-01-01T10:01:00Z", tz="UTC")
    assert result.risk_r_multiple == pytest.approx((1.1100 - 1.1000) / (1.1000 - 1.0950))


def test_limit_order_not_filled_within_grace():
    analysis_ts = pd.Timestamp("2024-01-01T10:00:00Z")
    trade = pd.Series(
        {
            "asset": "EURUSD",
            "journal_id": "t-2",
            "analysis_timestamp": analysis_ts.isoformat(),
            "entry_price": 1.0900,
            "stop_loss": 1.0850,
            "take_profit_1": 1.1050,
            "signal": "buy",
            "spot_price": 1.1000,
            "precision_state": "fire",
        }
    )
    prices = _make_price_frame(
        ("2024-01-01T10:00:00Z", 1.1000, 1.1020, 1.0950, 1.0990),
        ("2024-01-01T10:01:00Z", 1.0990, 1.1010, 1.0945, 1.0955),
        ("2024-01-01T10:02:00Z", 1.0955, 1.1000, 1.0950, 1.0980),
    )

    result = lt._evaluate_trade(
        trade,
        prices,
        horizon_minutes=60,
        entry_grace_minutes=2,
        entry_tolerance=0.0005,
        precision_state_column="precision_state",
        executed_precision_states={"fire", "executed"},
    )

    assert result.outcome == lt.OUTCOME_NO_FILL
    assert result.label is None
    assert result.entry_kind == "limit"
    assert result.fill_timestamp is None
    assert result.exit_timestamp is None


def test_precision_pending_skips_evaluation():
    analysis_ts = pd.Timestamp("2024-01-01T10:00:00Z")
    trade = pd.Series(
        {
            "asset": "EURUSD",
            "journal_id": "t-3",
            "analysis_timestamp": analysis_ts.isoformat(),
            "entry_price": 1.1000,
            "stop_loss": 1.0950,
            "take_profit_1": 1.1100,
            "signal": "buy",
            "spot_price": 1.1000,
            "precision_state": "precision_arming",
        }
    )
    prices = _make_price_frame(
        ("2024-01-01T10:00:00Z", 1.1000, 1.1050, 1.0950, 1.1005),
        ("2024-01-01T10:01:00Z", 1.1005, 1.1150, 1.1000, 1.1110),
    )

    result = lt._evaluate_trade(
        trade,
        prices,
        horizon_minutes=60,
        entry_grace_minutes=5,
        entry_tolerance=0.0005,
        precision_state_column="precision_state",
        executed_precision_states={"fire", "executed"},
    )

    assert result.outcome == lt.OUTCOME_PRECISION_PENDING
    assert result.fill_timestamp is None
    assert result.exit_timestamp is None
    assert result.label is None


def test_ambiguous_exit_when_same_bar_hits_tp_and_sl():
    analysis_ts = pd.Timestamp("2024-01-01T10:00:00Z")
    trade = pd.Series(
        {
            "asset": "EURUSD",
            "journal_id": "t-4",
            "analysis_timestamp": analysis_ts.isoformat(),
            "entry_price": 1.1000,
            "stop_loss": 1.0950,
            "take_profit_1": 1.1100,
            "signal": "buy",
            "spot_price": 1.1000,
            "precision_state": "fire",
        }
    )
    prices = _make_price_frame(
        ("2024-01-01T10:00:00Z", 1.1000, 1.1200, 1.0900, 1.1000),
        ("2024-01-01T10:01:00Z", 1.1000, 1.1050, 1.0950, 1.1000),
    )

    result = lt._evaluate_trade(
        trade,
        prices,
        horizon_minutes=60,
        entry_grace_minutes=5,
        entry_tolerance=0.0005,
        precision_state_column="precision_state",
        executed_precision_states={"fire", "executed"},
    )

    assert result.outcome == lt.OUTCOME_AMBIGUOUS
    assert result.label is None
    assert result.exit_timestamp == pd.Timestamp("2024-01-01T10:00:00Z", tz="UTC")

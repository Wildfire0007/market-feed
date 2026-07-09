import pandas as pd

from reports import trade_journal as tj


def test_suppressed_momentum_journal_row_has_tag_and_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(tj, "JOURNAL_DIR", tmp_path)
    monkeypatch.setattr(tj, "JOURNAL_FILE", tmp_path / "trade_journal.csv")
    monkeypatch.setattr(tj, "SUMMARY_FILE", tmp_path / "summary.json")

    tj.record_signal_event("EURUSD", {
        "retrieved_at_utc": "2024-01-01T10:00:00Z",
        "signal": "buy",
        "probability": 76,
        "entry": 1.1,
        "sl": 1.095,
        "tp1": 1.11,
        "tp2": 1.12,
        "gates": {"mode": "suppressed_momentum"},
    })

    df = pd.read_csv(tj.JOURNAL_FILE)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["mode"] == "suppressed_momentum"
    assert row["entry_price"] == 1.1
    assert row["stop_loss"] == 1.095
    assert row["take_profit_1"] == 1.11
    assert row["take_profit_2"] == 1.12

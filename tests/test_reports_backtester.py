import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from reports.backtester import update_live_validation
from scripts import label_trades as lt


def _write_price_file(path: Path, values: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"values": values}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_update_live_validation_prefers_high_resolution_data(tmp_path):
    public_dir = tmp_path / "public"
    reports_dir = tmp_path / "reports"
    journal_dir = public_dir / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = public_dir / "TEST"
    asset_dir.mkdir(parents=True, exist_ok=True)

    journal_df = pd.DataFrame(
        [
            {
                "journal_id": "t-1",
                "asset": "TEST",
                "analysis_timestamp": "2024-01-01T10:00:00Z",
                "signal": "buy",
                "entry_price": 1.1,
                "stop_loss": 1.095,
                "take_profit_1": 1.115,
                "spot_price": 1.1,
                "precision_state": "fire",
            }
        ]
    )
    journal_path = journal_dir / "trade_journal.csv"
    journal_df.to_csv(journal_path, index=False)

    _write_price_file(
        asset_dir / "klines_1m.json",
        [
            {"datetime": "2024-01-01T10:00:00Z", "open": 1.1, "high": 1.13, "low": 1.09, "close": 1.11},
            {"datetime": "2024-01-01T10:01:00Z", "open": 1.11, "high": 1.12, "low": 1.094, "close": 1.095},
        ],
    )
    _write_price_file(
        asset_dir / "klines_15s.json",
        [
            {"datetime": "2024-01-01T10:00:00Z", "open": 1.1, "high": 1.116, "low": 1.099, "close": 1.115},
            {"datetime": "2024-01-01T10:00:15Z", "open": 1.115, "high": 1.118, "low": 1.114, "close": 1.117},
            {"datetime": "2024-01-01T10:00:30Z", "open": 1.117, "high": 1.118, "low": 1.094, "close": 1.095},
        ],
    )

    summary_path = update_live_validation(
        public_dir=public_dir,
        reports_dir=reports_dir,
        lookahead_hours=1.0,
    )

    assert summary_path is not None
    updated_journal = pd.read_csv(journal_path)
    row = updated_journal.iloc[0]
    assert row["validation_outcome"] == lt.OUTCOME_PROFIT
    assert row["validation_source"] == "auto"
    assert row["entry_kind"] == "market"
    assert row["fill_timestamp"].startswith("2024-01-01T10:00:00")

    live_validation_path = reports_dir / "live_validation.csv"
    assert live_validation_path.exists()
    live_df = pd.read_csv(live_validation_path)
    assert not live_df.empty
    assert live_df.loc[0, "validation_outcome"] == lt.OUTCOME_PROFIT


def test_update_live_validation_applies_manual_overrides(tmp_path):
    public_dir = tmp_path / "public"
    reports_dir = tmp_path / "reports"
    journal_dir = public_dir / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = public_dir / "TEST"
    asset_dir.mkdir(parents=True, exist_ok=True)

    journal_df = pd.DataFrame(
        [
            {
                "journal_id": "t-2",
                "asset": "TEST",
                "analysis_timestamp": "2024-01-01T10:00:00Z",
                "signal": "buy",
                "entry_price": 1.1,
                "stop_loss": 1.095,
                "take_profit_1": 1.115,
                "spot_price": 1.1,
                "precision_state": "fire",
            }
        ]
    )
    journal_path = journal_dir / "trade_journal.csv"
    journal_df.to_csv(journal_path, index=False)

    _write_price_file(
        asset_dir / "klines_1m.json",
        [
            {"datetime": "2024-01-01T10:00:00Z", "open": 1.1, "high": 1.11, "low": 1.094, "close": 1.095},
            {"datetime": "2024-01-01T10:01:00Z", "open": 1.095, "high": 1.096, "low": 1.09, "close": 1.091},
        ],
    )

    manual_fills = pd.DataFrame(
        [
            {
                "journal_id": "t-2",
                "validation_outcome": "manual_tp",
                "validation_rr": 2.5,
                "max_favorable_excursion": 2.75,
                "max_adverse_excursion": 0.3,
                "time_to_outcome_minutes": 45.0,
                "fill_timestamp": "2024-01-01T10:05:00Z",
                "exit_timestamp": "2024-01-01T10:50:00Z",
                "validation_source": "audited",
            }
        ]
    )
    manual_fills.to_csv(journal_dir / "manual_fills.csv", index=False)

    summary_path = update_live_validation(
        public_dir=public_dir,
        reports_dir=reports_dir,
        lookahead_hours=1.0,
    )

    assert summary_path is not None
    updated_journal = pd.read_csv(journal_path)
    row = updated_journal.iloc[0]
    assert row["validation_outcome"] == "manual_tp"
    assert row["validation_source"] == "audited"
    assert row["validation_rr"] == 2.5
    assert row["fill_timestamp"] == "2024-01-01T10:05:00Z"
    assert row["exit_timestamp"] == "2024-01-01T10:50:00Z"

    live_validation_path = reports_dir / "live_validation.csv"
    live_df = pd.read_csv(live_validation_path)
    assert not live_df.empty
    assert set(live_df["validation_outcome"]) == {"manual_tp"}
    assert set(live_df["validation_source"]) == {"audited"}

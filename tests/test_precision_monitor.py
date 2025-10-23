import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from reports.precision_monitor import update_precision_gate_report


def test_precision_monitor_generates_reports(tmp_path):
    rows = [
        {
            "journal_id": "1",
            "asset": "EURUSD",
            "analysis_timestamp": "2025-01-01T10:00:00Z",
            "signal": "no entry",
            "notes": "missing: precision_score>=70, precision_flow_alignment",
        },
        {
            "journal_id": "2",
            "asset": "EURUSD",
            "analysis_timestamp": "2025-01-01T11:00:00Z",
            "signal": "buy",
            "notes": "",
        },
        {
            "journal_id": "3",
            "asset": "USDJPY",
            "analysis_timestamp": "2025-01-02T09:00:00Z",
            "signal": "no entry",
            "notes": "missing: precision_trigger_sync",
        },
        {
            "journal_id": "4",
            "asset": "USDJPY",
            "analysis_timestamp": "2024-12-20T09:00:00Z",
            "signal": "no entry",
            "notes": "missing: precision_flow_alignment",
        },
        {
            "journal_id": "5",
            "asset": "GOLD_CFD",
            "analysis_timestamp": "",
            "signal": "no entry",
            "notes": "missing: precision_score>=70",
        },
    ]
    journal_path = tmp_path / "journal.csv"
    pd.DataFrame(rows).to_csv(journal_path, index=False)

    monitor_dir = tmp_path / "monitor"
    now = datetime(2025, 1, 2, 12, tzinfo=timezone.utc)
    summary_path = update_precision_gate_report(
        journal_path=journal_path,
        monitor_dir=monitor_dir,
        lookback_days=2,
        now=now,
    )

    assert summary_path is not None
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["total"]["total_signals"] == 5
    assert payload["total"]["precision_blocked"] == 4
    assert payload["total"]["metrics"]["precision_score"]["count"] == 2
    assert payload["total"]["metrics"]["precision_flow_alignment"]["count"] == 2
    assert payload["total"]["metrics"]["precision_trigger_sync"]["count"] == 1

    assert payload["recent"]["total_signals"] == 3
    assert payload["recent"]["precision_blocked"] == 2
    assert payload["recent"]["metrics"]["precision_score"]["count"] == 1
    assert payload["recent"]["metrics"]["precision_flow_alignment"]["count"] == 1
    assert payload["recent"]["metrics"]["precision_trigger_sync"]["count"] == 1

    assets = {entry["asset"]: entry for entry in payload["assets"]}
    assert assets["EURUSD"]["precision_blocked"] == 1
    assert assets["USDJPY"]["precision_blocked"] == 2
    assert assets["GOLD_CFD"]["precision_blocked"] == 1

    daily_csv = monitor_dir / "precision_gates_daily.csv"
    assert daily_csv.exists()
    daily_df = pd.read_csv(daily_csv)
    daily_totals = {
        row["analysis_day"]: row for row in daily_df.to_dict(orient="records")
    }
    assert daily_totals["2025-01-01"]["total_signals"] == 2
    assert daily_totals["2025-01-01"]["precision_blocked"] == 1
    assert daily_totals["2025-01-02"]["precision_trigger_sync"] == 1

    asset_csv = monitor_dir / "precision_gates_by_asset.csv"
    assert asset_csv.exists()
    asset_df = pd.read_csv(asset_csv)
    asset_totals = {
        row["asset"]: row for row in asset_df.to_dict(orient="records")
    }
    assert asset_totals["EURUSD"]["precision_score"] == 1
    assert asset_totals["USDJPY"]["precision_flow_alignment"] == 1
    assert asset_totals["USDJPY"]["precision_trigger_sync"] == 1
    assert asset_totals["GOLD_CFD"]["precision_score"] == 1

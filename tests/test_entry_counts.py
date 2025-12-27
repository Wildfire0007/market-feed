import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import analysis


def test_entry_count_summary_includes_journal_and_results(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "PUBLIC_DIR", tmp_path)

    journal_dir = tmp_path / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    rows = [
        {
            "analysis_timestamp": (now - timedelta(days=1)).isoformat(),
            "signal": "buy",
            "asset": "BTCUSD",
        },
        {
            "analysis_timestamp": (now - timedelta(days=3)).isoformat(),
            "signal": "sell",
            "asset": "EURUSD",
        },
        {
            "analysis_timestamp": (now - timedelta(days=20)).isoformat(),
            "signal": "buy",
            "asset": "OUTSIDE",
        },
    ]

    journal_path = journal_dir / "trade_journal.csv"
    with journal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["analysis_timestamp", "signal", "asset"])
        writer.writeheader()
        writer.writerows(rows)

    asset_results = {
        "xagusd": {"signal": "sell", "retrieved_at_utc": now.isoformat()},
        "nvdia": {"decision": "hold", "retrieved_at_utc": now.isoformat()},
    }

    summary = analysis._build_entry_count_summary(asset_results, window_days=7)

    by_day = summary["by_day_asset"]
    first_day = (now - timedelta(days=1)).date().isoformat()
    third_day = (now - timedelta(days=3)).date().isoformat()
    today = now.date().isoformat()

    assert by_day[first_day]["BTCUSD"] == 1
    assert by_day[third_day]["EURUSD"] == 1
    assert by_day[today]["XAGUSD"] == 1

    old_day = (now - timedelta(days=20)).date().isoformat()
    assert old_day not in by_day

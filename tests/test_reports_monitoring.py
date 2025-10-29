import json
from pathlib import Path

import pandas as pd

from reports.monitoring import update_data_latency_report


def test_update_data_latency_report_generates_alerts(tmp_path: Path) -> None:
    summary = {
        "assets": {
            "EURUSD": {
                "signal": "no entry",
                "spot": {"fallback_provider": "finnhub"},
                "diagnostics": {
                    "timeframes": {
                        "spot": {
                            "latency_seconds": 1800.0,
                            "freshness_limit_seconds": 600.0,
                            "freshness_violation": True,
                        },
                        "k5m": {
                            "latency_seconds": 900.0,
                            "stale_for_signals": True,
                            "critical_stale": True,
                        },
                        "k1m": {
                            "latency_seconds": 120.0,
                            "stale_for_signals": False,
                        },
                    }
                },
            },
            "NVDA": {
                "signal": "buy",
                "diagnostics": {
                    "timeframes": {
                        "spot": {
                            "latency_seconds": 120.0,
                            "freshness_limit_seconds": 900.0,
                        },
                        "k5m": {
                            "latency_seconds": 300.0,
                            "stale_for_signals": False,
                        },
                    }
                },
            },
        }
    }

    report_path = update_data_latency_report(tmp_path, summary)
    assert report_path is not None
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assets = {row["asset"]: row for row in payload["assets"]}

    assert assets["EURUSD"]["spot_violation"] is True
    assert assets["EURUSD"]["fallback_provider"] == "finnhub"
    assert "EURUSD" in " ".join(payload.get("alerts", []))

    csv_path = tmp_path / "monitoring" / "data_latency.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert set(df["asset"]) == {"EURUSD", "NVDA"}
    eurusd_row = df[df["asset"] == "EURUSD"].iloc[0]
    assert eurusd_row["critical_timeframes"] == "k5m"

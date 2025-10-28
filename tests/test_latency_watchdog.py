import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.td_latency_watchdog import (
    _wait_for_summary_refresh,
    _write_cache_bust_token,
    collect_latency_issues,
)


def test_collect_latency_from_diagnostics():
    summary = {
        "assets": {
            "BTCUSD": {
                "diagnostics": {
                    "timeframes": {
                        "k1m": {
                            "latency_seconds": 1900,
                            "critical_stale": False,
                            "stale_for_signals": False,
                        },
                        "k5m": {
                            "latency_seconds": 200,
                            "critical_stale": False,
                            "stale_for_signals": False,
                        },
                    }
                }
            }
        }
    }

    issues = collect_latency_issues(summary, threshold_seconds=900, timeframes=("k1m", "k5m"))
    assert len(issues) == 1
    issue = issues[0]
    assert issue.asset == "BTCUSD"
    assert issue.timeframe == "k1m"
    assert math.isclose(issue.latency_seconds, 1900)


def test_collect_latency_when_critical_stale_flagged():
    summary = {
        "assets": {
            "EURUSD": {
                "diagnostics": {
                    "timeframes": {
                        "k1m": {
                            "latency_seconds": None,
                            "critical_stale": True,
                            "stale_for_signals": True,
                        }
                    }
                }
            }
        }
    }

    issues = collect_latency_issues(summary, threshold_seconds=10_000, timeframes=("k1m",))
    assert issues
    issue = issues[0]
    assert issue.asset == "EURUSD"
    assert issue.timeframe == "k1m"
    assert issue.latency_seconds is None


def test_collect_latency_from_summary_flags():
    summary = {
        "latency_flags": [
            "k5m: utolsó zárt gyertya 640 perc késésben van",
            "spot: realtime feed aktív",
        ]
    }

    issues = collect_latency_issues(summary, threshold_seconds=600 * 60, timeframes=("k1m", "k5m"))
    assert len(issues) == 1
    issue = issues[0]
    assert issue.asset is None
    assert issue.timeframe == "k5m"
    assert math.isclose(issue.latency_minutes or 0.0, 640.0)


def test_wait_for_summary_refresh_detects_new_signature(tmp_path):
    summary_path = tmp_path / "analysis_summary.json"
    initial = {"generated_utc": "2024-01-01T00:00:00Z"}
    summary_path.write_text(json.dumps(initial), encoding="utf-8")
    previous_signature = initial["generated_utc"]

    updated = {"generated_utc": "2024-01-01T00:05:00Z"}
    summary_path.write_text(json.dumps(updated), encoding="utf-8")

    refreshed, payload = _wait_for_summary_refresh(
        summary_path,
        previous_signature,
        None,
        timeout_seconds=0.1,
        poll_seconds=0.05,
    )

    assert refreshed
    assert payload == updated


def test_wait_for_summary_refresh_timeout(tmp_path):
    summary_path = tmp_path / "analysis_summary.json"
    payload = {"generated_utc": "2024-01-01T00:00:00Z"}
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    previous_signature = payload["generated_utc"]
    previous_mtime = summary_path.stat().st_mtime

    refreshed, refreshed_payload = _wait_for_summary_refresh(
        summary_path,
        previous_signature,
        previous_mtime,
        timeout_seconds=0.15,
        poll_seconds=0.05,
    )

    assert not refreshed
    assert refreshed_payload == {"generated_utc": "2024-01-01T00:00:00Z"}


def test_write_cache_bust_token(tmp_path):
    summary_payload = {"generated_utc": "2024-01-01T00:10:00Z"}
    cache_path = tmp_path / "cache" / "cache_bust.json"

    payload = _write_cache_bust_token(cache_path, summary_payload)

    assert cache_path.exists()
    on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
    assert on_disk["generated_utc"] == summary_payload["generated_utc"]
    assert on_disk["token"] == payload["token"]
    assert on_disk["created_utc"] == payload["created_utc"]
    assert payload["token"]

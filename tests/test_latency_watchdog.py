import math

from scripts.td_latency_watchdog import collect_latency_issues


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

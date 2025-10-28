"""Regression tests for the human-facing intraday report outputs."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.intraday_report import spot_display_metadata


def _build_signal(
    latency_seconds: float,
    expected_seconds: float,
    original_issue: str | None = None,
) -> dict:
    diagnostics = {
        "timeframes": {
            "spot": {
                "latency_seconds": latency_seconds,
                "expected_max_delay_seconds": expected_seconds,
            }
        }
    }
    if original_issue:
        diagnostics["timeframes"]["spot"]["original_issue"] = original_issue
    return {
        "spot": {
            "price": 1.2345,
            "utc": "2023-10-10T12:00:00Z",
            "retrieved_at_utc": "2023-10-12T12:00:00Z",
        },
        "diagnostics": diagnostics,
        "reasons": [],
    }


def test_spot_display_metadata_masks_latency_breach() -> None:
    signal = _build_signal(latency_seconds=7200, expected_seconds=900)
    meta = spot_display_metadata(signal)

    assert meta["stale"] is True
    assert meta["price"] is None
    assert meta["timestamp"] == "2023-10-12T12:00:00Z"
    assert "latency" in (meta["reason"] or "").lower()


def test_spot_display_metadata_uses_original_issue_reason() -> None:
    signal = _build_signal(
        latency_seconds=100, expected_seconds=900, original_issue="Spot data stale: 5h"
    )
    meta = spot_display_metadata(signal)

    assert meta["stale"] is True
    assert meta["reason"] == "Spot data stale: 5h"

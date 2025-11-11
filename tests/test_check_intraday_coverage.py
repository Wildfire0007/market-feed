import json

from scripts.check_intraday_coverage import compute_coverage


def test_compute_coverage_detects_gaps(tmp_path):
    public_dir = tmp_path / "public"
    asset_dir = public_dir / "NVDA"
    asset_dir.mkdir(parents=True)
    rows = [
        {"datetime": "2025-11-11T00:00:00Z", "close": 100},
        {"datetime": "2025-11-11T00:05:00Z", "close": 101},
        {"datetime": "2025-11-11T00:20:00Z", "close": 102},
        {"datetime": "2025-11-11T23:55:00Z", "close": 103},
    ]
    (asset_dir / "klines_5m.json").write_text(json.dumps({"values": rows}), encoding="utf-8")

    summary = compute_coverage("NVDA", "2025-11-11", public_dir=public_dir)

    assert summary.asset == "NVDA"
    assert summary.date == "2025-11-11"
    assert summary.bar_count == 4
    assert summary.first_bar_utc == "2025-11-11T00:00:00Z"
    assert summary.last_bar_utc == "2025-11-11T23:55:00Z"
    assert summary.expected_bars == 288
    assert summary.missing_bars == 284
    assert summary.session_coverage_pct == round((4 / 288) * 100.0, 2)
    assert any("00:10" in gap for gap in summary.gaps)

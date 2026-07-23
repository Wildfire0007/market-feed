import json
from pathlib import Path

import Trading


def test_revision_event_logged(tmp_path: Path):
    public = tmp_path / "public"
    asset = public / "USOIL"
    asset.mkdir(parents=True)
    old = {"values": [
        {"datetime": "2026-07-23 07:28:00", "close": "88.378"},
        {"datetime": "2026-07-23 07:29:00", "close": "88.341"},
        {"datetime": "2026-07-23 07:30:00", "close": "88.456"},
    ]}
    (asset / "klines_1m.json").write_text(json.dumps(old), encoding="utf-8")
    new_raw = {"values": [
        {"datetime": "2026-07-23 07:28:00", "close": "88.378"},
        {"datetime": "2026-07-23 07:29:00", "close": "89.292"},
        {"datetime": "2026-07-23 07:30:00", "close": "89.396"},
        {"datetime": "2026-07-23 07:31:00", "close": "89.436"},
    ]}
    Trading._append_revision_telemetry(str(asset), "klines_1m", new_raw)
    lines = (public / "debug" / "revision_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 1
    row = rows[0]
    assert row["asset"] == "USOIL"
    assert row["series"] == "klines_1m"
    assert row["revised_count"] == 1
    assert row["max_delta_ts"] == "2026-07-23 07:29:00"
    assert abs(row["max_abs_delta"] - 0.951) < 1e-6


def test_no_event_when_unchanged(tmp_path: Path):
    public = tmp_path / "public"
    asset = public / "GOLD_CFD"
    asset.mkdir(parents=True)
    vals = {"values": [
        {"datetime": "2026-07-23 07:28:00", "close": "4060.0"},
        {"datetime": "2026-07-23 07:29:00", "close": "4061.0"},
    ]}
    (asset / "klines_1m.json").write_text(json.dumps(vals), encoding="utf-8")
    Trading._append_revision_telemetry(str(asset), "klines_1m", vals)
    assert not (public / "debug" / "revision_telemetry.jsonl").exists()

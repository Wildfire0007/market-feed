import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "log_candle_revisions.py"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(repo: Path):
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")


def test_revision_event_logged_against_head(tmp_path: Path):
    _init_repo(tmp_path)
    asset = tmp_path / "public" / "USOIL"
    asset.mkdir(parents=True)
    old = {"values": [
        {"datetime": "2026-07-23 07:28:00", "close": "88.378"},
        {"datetime": "2026-07-23 07:29:00", "close": "88.341"},
        {"datetime": "2026-07-23 07:30:00", "close": "88.456"},
    ]}
    (asset / "klines_1m.json").write_text(json.dumps(old), encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "baseline")
    new = {"values": [
        {"datetime": "2026-07-23 07:28:00", "close": "88.378"},
        {"datetime": "2026-07-23 07:29:00", "close": "89.292"},
        {"datetime": "2026-07-23 07:30:00", "close": "89.396"},
        {"datetime": "2026-07-23 07:31:00", "close": "89.436"},
    ]}
    (asset / "klines_1m.json").write_text(json.dumps(new), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0
    lines = (tmp_path / "public" / "debug" / "revision_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 1
    assert rows[0]["asset"] == "USOIL"
    assert rows[0]["revised_count"] == 1
    assert rows[0]["max_delta_ts"] == "2026-07-23 07:29:00"
    assert abs(rows[0]["max_abs_delta"] - 0.951) < 1e-6


def test_no_event_when_head_matches(tmp_path: Path):
    _init_repo(tmp_path)
    asset = tmp_path / "public" / "GOLD_CFD"
    asset.mkdir(parents=True)
    vals = {"values": [
        {"datetime": "2026-07-23 07:28:00", "close": "4060.0"},
        {"datetime": "2026-07-23 07:29:00", "close": "4061.0"},
    ]}
    (asset / "klines_1m.json").write_text(json.dumps(vals), encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "baseline")
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0
    assert not (tmp_path / "public" / "debug" / "revision_telemetry.jsonl").exists()

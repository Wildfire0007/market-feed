import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_macro_schema import validate_calendar, validate_lockout


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validate_calendar_happy_path(tmp_path: Path) -> None:
    calendar_path = _write(
        tmp_path / "calendar.json",
        {
            "generated_utc": "2024-01-01T00:00:00Z",
            "d1": "2024-01-01",
            "d2": "2024-01-02",
            "items": [
                {
                    "id": "TE:1",
                    "provider": "tradingeconomics",
                    "type": "SCHEDULED",
                    "country": "US",
                    "category": "INFLATION",
                    "event": "CPI",
                    "importance": 3,
                    "ts_release_utc": "2024-01-01T13:30:00Z",
                    "lockout": {"pre_seconds": 900, "post_seconds": 1800},
                }
            ],
        },
    )
    errors = validate_calendar(str(calendar_path))
    assert errors == []


def test_validate_calendar_rejects_non_iso(tmp_path: Path) -> None:
    calendar_path = _write(
        tmp_path / "calendar.json",
        {
            "generated_utc": "not-a-date",
            "d1": "2024-01-01",
            "d2": "2024-01-02",
            "items": [
                {
                    "id": "TE:1",
                    "provider": "tradingeconomics",
                    "type": "SCHEDULED",
                    "country": "US",
                    "category": "INFLATION",
                    "event": "CPI",
                    "importance": 3,
                    "ts_release_utc": "bad-ts",
                    "lockout": {"pre_seconds": 900, "post_seconds": 1800},
                }
            ],
        },
    )
    errors = validate_calendar(str(calendar_path))
    assert any("generated_utc" in err for err in errors)
    assert any("ts_release_utc" in err for err in errors)


def test_validate_lockout_detects_missing_fields(tmp_path: Path) -> None:
    lockout_path = _write(
        tmp_path / "lockout.json",
        {"BTCUSD": [{"id": "TE:1", "ts_release_utc": "2024-01-01T00:00:00Z"}]},
    )
    errors = validate_lockout(str(lockout_path))
    assert any("pre_seconds" in err for err in errors)
    assert any("post_seconds" in err for err in errors)
    assert any("label" in err for err in errors)


def test_validate_lockout_happy_path(tmp_path: Path) -> None:
    lockout_path = _write(
        tmp_path / "lockout.json",
        {
            "BTCUSD": [
                {
                    "id": "TE:1",
                    "ts_release_utc": "2024-01-01T00:00:00Z",
                    "pre_seconds": 600,
                    "post_seconds": 1200,
                    "label": "CPI",
                }
            ]
        },
    )
    errors = validate_lockout(str(lockout_path))
    assert errors == []

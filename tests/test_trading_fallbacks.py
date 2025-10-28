import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Trading


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reuse_previous_spot_respects_limit(tmp_path: Path) -> None:
    asset_dir = tmp_path / "EURUSD"
    asset_dir.mkdir()
    previous = {
        "ok": True,
        "price": 1.2345,
        "utc": "2024-01-01T00:00:00Z",
        "fallback_reuse_count": Trading.MAX_CONSECUTIVE_FALLBACKS,
    }
    _write(asset_dir / "spot.json", previous)

    fresh_payload = {"ok": False, "error": "unavailable"}
    reused = Trading._reuse_previous_spot(str(asset_dir), dict(fresh_payload), 900.0)

    assert reused.get("ok") is False
    assert not reused.get("fallback_previous_payload")
    assert "fallback_reuse_count" not in reused


def test_reuse_previous_spot_increments_counter(tmp_path: Path) -> None:
    asset_dir = tmp_path / "USOIL"
    asset_dir.mkdir()
    previous = {
        "ok": True,
        "price": 80.5,
        "utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "fallback_reuse_count": Trading.MAX_CONSECUTIVE_FALLBACKS - 1,
    }
    _write(asset_dir / "spot.json", previous)

    degraded = {"ok": False, "error": "timeout"}
    reused = Trading._reuse_previous_spot(str(asset_dir), degraded, 3600.0)

    assert reused.get("fallback_previous_payload")
    assert reused.get("fallback_reuse_count") == Trading.MAX_CONSECUTIVE_FALLBACKS


def test_reuse_previous_series_respects_limit(tmp_path: Path) -> None:
    asset_dir = tmp_path / "NVDA"
    asset_dir.mkdir()
    name = "klines_1m"
    previous_meta = {
        "ok": True,
        "latest_utc": "2024-01-01T00:00:00Z",
        "fallback_reuse_count": Trading.MAX_CONSECUTIVE_FALLBACKS,
    }
    previous_raw = {"values": [["2024-01-01T00:00:00Z", "1.0"]]}
    _write(asset_dir / f"{name}_meta.json", previous_meta)
    _write(asset_dir / f"{name}.json", previous_raw)

    degraded = {"ok": False, "error": "rate limit"}
    reused = Trading._reuse_previous_series_payload(
        str(asset_dir),
        name,
        dict(degraded),
        freshness_limit=900.0,
    )

    assert reused.get("ok") is False
    assert not reused.get("fallback_previous_payload")


def test_market_closed_staleness_enforces_hard_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    hard_cap = 60.0
    monkeypatch.setattr(Trading, "MARKET_CLOSED_MAX_AGE_SECONDS", hard_cap)

    now = datetime.now(timezone.utc)
    payload = {
        "asset": "NVDA",
        "ok": True,
        "freshness_violation": True,
        "source": "twelvedata:time_series",
        "latest_utc": (now - timedelta(seconds=hard_cap + 30)).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }

    accepted = Trading._accept_market_closed_staleness(dict(payload), freshness_limit=300.0)

    assert not accepted

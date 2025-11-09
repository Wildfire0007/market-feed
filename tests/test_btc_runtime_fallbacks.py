import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

# Ensure repository root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _reload_analysis():
    import analysis  # noqa: F401

    return importlib.reload(sys.modules['analysis'])


def test_btc_core_triggers_ok_relaxes_when_components_missing():
    analysis = _reload_analysis()

    analysis._BTC_MOMENTUM_RUNTIME = {"ofi_z": analysis.BTC_OFI_Z["trigger"] + 0.05}
    analysis._BTC_STRUCT_MICRO = {}
    analysis._BTC_STRUCT_VWAP = {}

    ok, meta = analysis.btc_core_triggers_ok("BTCUSD", "long")

    assert ok is True
    assert meta["hits"] == 1
    assert set(meta["missing"]) == {"bos", "vwap"}
    assert "ofi" in meta["available"]


def test_btc_core_triggers_ok_blocks_when_no_positive_components():
    analysis = _reload_analysis()

    analysis._BTC_MOMENTUM_RUNTIME = {"ofi_z": 0.0}
    analysis._BTC_STRUCT_MICRO = {}
    analysis._BTC_STRUCT_VWAP = {}

    ok, meta = analysis.btc_core_triggers_ok("BTCUSD", "long")

    assert ok is False
    assert meta["hits"] == 0
    assert "ofi" in meta["available"] or "ofi" in meta["missing"]


def test_should_use_realtime_spot_rejects_stale_snapshot():
    analysis = _reload_analysis()

    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    rt_ts = now - timedelta(minutes=15)

    use, meta = analysis._should_use_realtime_spot(50000.0, rt_ts, None, now, max_age_seconds=300)

    assert use is False
    assert meta["reason"] == "stale"
    assert meta["age_seconds"] == pytest.approx(900.0)


def test_should_use_realtime_spot_accepts_fresh_newer_snapshot():
    analysis = _reload_analysis()

    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    spot_ts = now - timedelta(minutes=10)
    rt_ts = now - timedelta(minutes=2)

    use, meta = analysis._should_use_realtime_spot(50100.0, rt_ts, spot_ts, now, max_age_seconds=600)

    assert use is True
    assert meta["reason"] == "ok"
    assert meta["age_seconds"] == pytest.approx(120.0)

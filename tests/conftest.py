import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _reload_analysis(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SESSION_STATUS_PROFILE", raising=False)
    monkeypatch.delenv("ENTRY_THRESHOLD_PROFILE", raising=False)
    if "analysis" in sys.modules:
        return importlib.reload(sys.modules["analysis"])
    return importlib.import_module("analysis")


@pytest.fixture
def fixed_now():
    return datetime(2024, 1, 10, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def analysis_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixed_now: datetime):
    analysis = _reload_analysis(monkeypatch)
    real_datetime = analysis.datetime

    class FixedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    monkeypatch.setattr(analysis, "datetime", FixedDateTime)
    monkeypatch.setattr(analysis, "PUBLIC_DIR", str(tmp_path))
    return analysis


def apply_common_analysis_stubs(
    analysis: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_models: Mapping[str, Any] | None = None,
    record_run_result: tuple | None = None,
    record_status_result: tuple | None = None,
):
    monkeypatch.setattr(analysis, "evaluate_news_lockout", lambda asset, now: (False, None))
    monkeypatch.setattr(analysis, "load_funding_snapshot", lambda asset: {})
    monkeypatch.setattr(analysis, "load_tick_order_flow", lambda asset, outdir: {})
    monkeypatch.setattr(analysis, "compute_order_flow_metrics", lambda *a, **k: {})
    monkeypatch.setattr(analysis, "current_anchor_state", lambda: {})
    monkeypatch.setattr(analysis, "log_feature_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "inspect_model_artifact", lambda asset: {})
    monkeypatch.setattr(
        analysis,
        "missing_model_artifacts",
        lambda assets=None: missing_models if missing_models is not None else {},
    )
    monkeypatch.setattr(analysis, "predict_signal_probability", lambda *a, **k: (0.42, {"model": "stub"}))
    monkeypatch.setattr(analysis, "runtime_dependency_issues", lambda: [])
    monkeypatch.setattr(analysis, "load_sentiment", lambda asset, now: ([], None))
    monkeypatch.setattr(analysis, "load_volatility_overlay", lambda *a, **k: {})
    monkeypatch.setattr(analysis, "update_precision_gate_report", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_signal_health_report", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_data_latency_report", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_live_validation", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "record_signal_event", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "record_ml_model_status", lambda *a, **k: record_status_result or (None, False))
    monkeypatch.setattr(analysis, "load_anchor_state", lambda: {})
    monkeypatch.setattr(analysis, "record_anchor", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "update_anchor_metrics", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "ensure_closed_candles", lambda df, now, tolerance_seconds=0: df)
    monkeypatch.setattr(analysis, "file_mtime", lambda path: None)
    monkeypatch.setattr(analysis, "record_analysis_run", lambda *a, **k: record_run_result or (None, None, None, False))


def make_raw_klines(final_time: datetime, periods: int, step: timedelta, base: float):
    rows = []
    start = final_time - step * (periods - 1)
    for idx in range(periods):
        ts = start + step * idx
        rows.append(
            {
                "datetime": ts.isoformat(),
                "open": base + idx * 0.1,
                "high": base + idx * 0.1 + 0.2,
                "low": base + idx * 0.1 - 0.2,
                "close": base + idx * 0.1 + 0.05,
                "volume": 100 + idx,
            }
        )
    return rows


def prime_data_registry(
    analysis: Any, monkeypatch: pytest.MonkeyPatch, registry: Dict[str, Dict[str, Any]]
):
    def fake_load_json(path: str):
        p = Path(path)
        asset = p.parent.name
        asset_map = registry.get(asset)
        if asset_map and p.name in asset_map:
            return asset_map[p.name]
        return {}

    monkeypatch.setattr(analysis, "load_json", fake_load_json)
    return registry


@pytest.fixture
def asset_registry(monkeypatch: pytest.MonkeyPatch, analysis_module: Any):
    registry: Dict[str, Dict[str, Any]] = {}
    prime_data_registry(analysis_module, monkeypatch, registry)
    return registry

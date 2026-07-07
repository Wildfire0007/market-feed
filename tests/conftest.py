import importlib
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.pop("CI", None)
os.environ.pop("GITHUB_ACTIONS", None)

# Offline fallback: expose lightweight stubs (tests/_stubs) ONLY when the real
# package is not installed. The real packages from requirements.lock always
# take precedence, so schema validation and time freezing behave identically
# to production when dependencies are available.
_STUBS_DIR = Path(__file__).resolve().parent / "_stubs"
for _stub_pkg in ("jsonschema", "freezegun"):
    try:
        importlib.import_module(_stub_pkg)
    except ModuleNotFoundError:
        if str(_STUBS_DIR) not in sys.path:
            sys.path.append(str(_STUBS_DIR))
        break

def _reload_analysis(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SESSION_STATUS_PROFILE", raising=False)
    monkeypatch.delenv("ENTRY_THRESHOLD_PROFILE", raising=False)
    if "analysis" in sys.modules:
        return importlib.reload(sys.modules["analysis"])
    return importlib.import_module("analysis")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_public_paths: allow a test to use repository public/ path constants",
    )


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    import_targets = {
        "numpy": ("import numpy", "from numpy", "import analysis", "from analysis", "import Trading", "import ml_model", "from ml_model", "scripts.intraday_report"),
        "pandas": ("import pandas", "from pandas"),
        "requests": ("import requests", "from requests", "import Trading"),
    }
    for module, markers in import_targets.items():
        if any(marker in text for marker in markers):
            try:
                importlib.import_module(module)
            except ModuleNotFoundError:
                return True
    return False




def _public_snapshot(root: Path):
    if not root.exists():
        return None
    return {
        path.relative_to(root): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file()
    }


def pytest_sessionstart(session):
    session.config._public_guard_tmp = session.config.cache.mkdir("isolated_public")
    import os
    os.environ.setdefault("NOTIFY_PUBLIC_DIR", str(session.config._public_guard_tmp))
    os.environ.setdefault("PUBLIC_DIR", str(session.config._public_guard_tmp))
    os.environ.setdefault("MANUAL_POS_AUDIT_FILE", str(session.config._public_guard_tmp / "_manual_positions_audit.jsonl"))
    session.config._public_guard_root = PROJECT_ROOT / "public"
    session.config._public_guard_snapshot = _public_snapshot(session.config._public_guard_root)


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_public_guard_snapshot", None)
    if before is None:
        return
    root = getattr(session.config, "_public_guard_root", PROJECT_ROOT / "public")
    after = _public_snapshot(root) or {}
    created = sorted(after.keys() - before.keys())
    deleted = sorted(before.keys() - after.keys())
    modified = sorted(path for path in before.keys() & after.keys() if before[path] != after[path])
    offenders = [*(f"created: {path}" for path in created), *(f"modified: {path}" for path in modified), *(f"deleted: {path}" for path in deleted)]
    if offenders:
        raise pytest.UsageError("Tests changed files under public/:\n" + "\n".join(offenders))


@pytest.fixture(autouse=True)
def isolate_public_dirs(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    public_dir = tmp_path / "public"
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)    
    monkeypatch.setenv("NOTIFY_PUBLIC_DIR", str(public_dir))
    monkeypatch.setenv("PUBLIC_DIR", str(public_dir))
    monkeypatch.setenv("MANUAL_POS_AUDIT_FILE", str(public_dir / "_manual_positions_audit.jsonl"))
    try:
        import state_db
        monkeypatch.setattr(state_db, "DEFAULT_DB_PATH", tmp_path / "trading.db")
    except Exception:
        pass
    def _isolate_analysis_paths(module):
        monkeypatch.setattr(module, "PUBLIC_DIR", str(public_dir), raising=False)
        monkeypatch.setattr(module, "ENTRY_GATE_LOG_DIR", public_dir / "debug" / "entry_gates", raising=False)
        monkeypatch.setattr(module, "ENTRY_GATE_STATS_PATH", public_dir / "debug" / "entry_gate_stats.json", raising=False)
        monkeypatch.setattr(module, "ENTRY_GATE_GAP_LOG_PATH", public_dir / "debug" / "entry_gate_gap_log.jsonl", raising=False)
        return module

    if request.node.get_closest_marker("real_public_paths") is None:
        real_reload = importlib.reload

        def _isolating_reload(module):
            reloaded = real_reload(module)
            if getattr(reloaded, "__name__", None) == "analysis":
                _isolate_analysis_paths(reloaded)
            return reloaded

        monkeypatch.setattr(importlib, "reload", _isolating_reload)
        _isolate_analysis_paths(sys.modules.get("analysis") or importlib.import_module("analysis"))        
    for module_name in ("scripts.notify_discord", "scripts.notify_management_discord"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        monkeypatch.setattr(module, "PUBLIC_DIR", public_dir, raising=False)
        monkeypatch.setattr(module, "ENTRY_GATE_STATS_PATH", public_dir / "monitoring" / "entry_gate_stats.json", raising=False)
        monkeypatch.setattr(module, "ENTRY_GATE_LOG_DIR", public_dir / "debug" / "entry_gates", raising=False)
        monkeypatch.setattr(module, "MANAGEMENT_DIAG_PATH", public_dir / "debug" / "management_notify_events.jsonl", raising=False)
        monkeypatch.setattr(module, "LIFECYCLE_INBOX_PATH", public_dir / "_position_lifecycle_inbox.jsonl", raising=False)

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

    public_dir = tmp_path    
    monkeypatch.setattr(analysis, "datetime", FixedDateTime)
    monkeypatch.setattr(analysis, "PUBLIC_DIR", str(public_dir))
    monkeypatch.setattr(analysis, "ENTRY_GATE_LOG_DIR", public_dir / "debug" / "entry_gates", raising=False)
    monkeypatch.setattr(analysis, "ENTRY_GATE_STATS_PATH", public_dir / "debug" / "entry_gate_stats.json", raising=False)
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", public_dir / "debug" / "entry_gate_gap_log.jsonl", raising=False)
    lifecycle = dict(getattr(analysis, "POSITION_LIFECYCLE", {}) or {})
    lifecycle["positions_file"] = str(public_dir / "trading.db")
    lifecycle["pending_exit_file"] = str(public_dir / "trading.db")
    monkeypatch.setattr(analysis, "POSITION_LIFECYCLE", lifecycle, raising=False)
    if isinstance(getattr(analysis, "SETTINGS", None), dict):
        settings_copy = dict(analysis.SETTINGS)
        settings_copy["position_lifecycle"] = lifecycle
        monkeypatch.setattr(analysis, "SETTINGS", settings_copy, raising=False)    
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

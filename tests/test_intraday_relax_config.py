import importlib
import json
import sys
from pathlib import Path

import pytest


def _base_payload() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "config" / "analysis_settings.json"
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    cfg_path = tmp_path / "analysis_settings.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return cfg_path


def _load_settings(monkeypatch, request, cfg_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import config.analysis_settings as settings

    monkeypatch.setenv("ANALYSIS_CONFIG_FILE", str(cfg_path))
    settings.reload_config(path=str(cfg_path))
    module = importlib.reload(settings)

    # Clear the override immediately so any later reload uses the default path.
    monkeypatch.delenv("ANALYSIS_CONFIG_FILE", raising=False)

    # Restore the default configuration so subsequent tests see the canonical
    # settings module state.
    def _reset_defaults():
        settings.reload_config()
        importlib.reload(settings)

    request.addfinalizer(_reset_defaults)

    return module


def test_intraday_relax_toggle_and_defaults(tmp_path, monkeypatch, request):
    payload = _base_payload()
    payload.update(
        {
            "enable_intraday_relax": {"default": True, "BTCUSD": False},
            "intraday_relax_size_scale": {"default": 0.5, "BTCUSD": 0.9},
        }
    )
    cfg_path = _write_config(tmp_path, payload)

    settings = _load_settings(monkeypatch, request, cfg_path)

    assert settings.is_intraday_relax_enabled("BTCUSD") is False
    assert settings.is_intraday_relax_enabled("EURUSD") is True

    assert settings.get_intraday_relax_size_scale("BTCUSD") == pytest.approx(0.9)
    assert settings.get_intraday_relax_size_scale("EURUSD") == pytest.approx(0.5)


def test_intraday_relax_size_scale_clamping_and_fallback(tmp_path, monkeypatch, request):
    payload = _base_payload()
    payload.update(
        {
            "enable_intraday_relax": True,
            "intraday_relax_size_scale": {"default": 2.5, "BTCUSD": "invalid"},
        }
    )
    cfg_path = _write_config(tmp_path, payload)

    settings = _load_settings(monkeypatch, request, cfg_path)
    
    # Default value is clamped to the [0.1, 1.0] interval
    assert settings.get_intraday_relax_size_scale("EURUSD") == pytest.approx(1.0)
    # Invalid per-asset value should fall back to the default mapping
    assert settings.get_intraday_relax_size_scale("BTCUSD") == pytest.approx(1.0)

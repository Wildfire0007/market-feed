"""Validation behaviour for analysis settings assets."""

import importlib
import json
import sys
from pathlib import Path


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


def _load_settings(monkeypatch, cfg_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import config.analysis_settings as settings

    monkeypatch.setenv("ANALYSIS_CONFIG_FILE", str(cfg_path))
    settings.reload_config(path=str(cfg_path))
    return importlib.reload(settings)


def test_asset_missing_fields_filtered_with_hints(tmp_path, monkeypatch, caplog):
    payload = _base_payload()
    payload["assets"] = ["BTCUSD", "EURUSD"]
    payload["leverage"].pop("EURUSD", None)
    payload["session_windows_utc"].pop("EURUSD", None)

    cfg_path = _write_config(tmp_path, payload)

    with caplog.at_level("ERROR"):
        settings = _load_settings(monkeypatch, cfg_path)

    assert "Asset EURUSD missing leverage" in " ".join(caplog.messages)
    assert settings.ASSETS == ["BTCUSD"]


def test_asset_missing_time_rules(tmp_path, monkeypatch, caplog):
    payload = _base_payload()
    payload["assets"] = ["BTCUSD", "EURUSD"]
    payload["session_time_rules"].pop("EURUSD", None)

    cfg_path = _write_config(tmp_path, payload)

    with caplog.at_level("ERROR"):
        settings = _load_settings(monkeypatch, cfg_path)

    message_blob = " ".join(caplog.messages)
    assert "Add session_time_rules for EURUSD" in message_blob
    assert settings.ASSETS == ["BTCUSD"]


def test_min_intraday_assets_fallback(tmp_path, monkeypatch, caplog):
    payload = _base_payload()
    payload["assets"] = ["EURUSD"]
    payload["leverage"].pop("EURUSD", None)
    payload["session_windows_utc"].pop("EURUSD", None)
    payload["session_time_rules"].pop("EURUSD", None)

    cfg_path = _write_config(tmp_path, payload)

    with caplog.at_level("ERROR"):
        settings = _load_settings(monkeypatch, cfg_path)

    assert "No valid assets after validation" in " ".join(caplog.messages)
    assert settings.ASSETS == ["BTCUSD", "EURUSD"]


def test_momentum_assets_follow_validated_assets(tmp_path, monkeypatch):
    payload = _base_payload()
    payload["assets"] = ["BTCUSD", "EURUSD"]
    payload["leverage"].pop("EURUSD", None)
    payload["session_windows_utc"].pop("EURUSD", None)
    payload["session_time_rules"].pop("EURUSD", None)

    cfg_path = _write_config(tmp_path, payload)

    settings = _load_settings(monkeypatch, cfg_path)

    assert settings.ASSETS == ["BTCUSD"]
    assert settings.ENABLE_MOMENTUM_ASSETS == {"BTCUSD"}

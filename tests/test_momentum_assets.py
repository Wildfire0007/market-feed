"""Tests for the momentum asset helpers."""
import importlib


def _fresh_settings():
    import config.analysis_settings as settings

    settings.reload_config()
    return importlib.reload(settings)


def test_is_momentum_asset_uses_config():
    settings = _fresh_settings()

    assert settings.is_momentum_asset("NVDA") is True
    assert settings.is_momentum_asset("EURUSD") is True


def test_is_momentum_asset_fallback(monkeypatch):
    settings = _fresh_settings()

    def _raise(*args, **kwargs):
        raise settings.AnalysisConfigError("boom")

    monkeypatch.setattr(settings, "load_config", _raise)

    assert settings.is_momentum_asset("BTCUSD") is True
    assert settings.is_momentum_asset("EURUSD") is False

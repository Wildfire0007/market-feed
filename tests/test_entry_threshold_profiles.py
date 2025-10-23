import importlib
import sys
from pathlib import Path

import pytest


def _reload_settings(monkeypatch, profile=None):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import config.analysis_settings as settings

    if profile is None:
        monkeypatch.delenv("ENTRY_THRESHOLD_PROFILE", raising=False)
    else:
        monkeypatch.setenv("ENTRY_THRESHOLD_PROFILE", profile)
    settings.reload_config()
    return importlib.reload(settings)


def test_baseline_profile_configuration(monkeypatch):
    settings = _reload_settings(monkeypatch)

    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "baseline"
    assert settings.get_p_score_min("EURUSD") == pytest.approx(60.0)
    assert settings.get_atr_threshold_multiplier("EURUSD") == pytest.approx(1.0)
    # Baseline profile should remain unchanged for non-overridden assets.
    assert settings.get_atr_threshold_multiplier("USOIL") == pytest.approx(1.0)


def test_relaxed_profile_override(monkeypatch):
    settings = _reload_settings(monkeypatch, profile="relaxed")

    assert settings.ENTRY_THRESHOLD_PROFILE_NAME == "relaxed"
    assert settings.get_p_score_min("GOLD_CFD") == pytest.approx(55.0)
    assert settings.get_atr_threshold_multiplier("USOIL") == pytest.approx(0.9)

    # Restore the default profile for subsequent tests.
    _reload_settings(monkeypatch)

from datetime import datetime, timezone
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import Trading
from config import analysis_settings as settings


def test_status_profile_respects_weekday(monkeypatch):
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 11, 28, 18, 8, tzinfo=tz or timezone.utc)

    monkeypatch.setattr(Trading, "datetime", _FixedDatetime)

    monkeypatch.setattr(settings, "SESSION_STATUS_PROFILE_NAME", "weekend", raising=False)
    monkeypatch.setattr(
        settings, "SESSION_STATUS_PROFILE", settings.SESSION_STATUS_PROFILES["weekend"], raising=False
    )

    name, profile = Trading._status_profile_for_asset("GOLD_CFD")

    assert name == "default"
    assert profile == settings.SESSION_STATUS_PROFILES["default"]

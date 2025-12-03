from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.notify_discord import (  # noqa: E402
    _default_asset_state,
    to_utc_iso,
    update_asset_send_state,
)


def test_update_asset_send_state_sets_last_sent_and_cooldown():
    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    base = _default_asset_state()

    updated = update_asset_send_state(
        base,
        decision="buy",
        now=now,
        cooldown_minutes=15,
        mode="core",
    )

    assert updated["last_sent"] == to_utc_iso(now)
    assert updated["last_sent_decision"] == "buy"
    assert updated["last_sent_mode"] == "core"
    assert updated["last_sent_known"] is True
    assert updated["cooldown_until"] == to_utc_iso(now + timedelta(minutes=15))



def test_update_asset_send_state_clears_cooldown_when_zero():
    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    base = _default_asset_state()
    base["cooldown_until"] = to_utc_iso(now + timedelta(minutes=30))

    updated = update_asset_send_state(
        base,
        decision="sell",
        now=now,
        cooldown_minutes=0,
        mode=None,
    )

    assert updated["cooldown_until"] is None
    assert updated["last_sent"] == to_utc_iso(now)
    assert updated["last_sent_decision"] == "sell"
    assert updated["last_sent_mode"] is None
    assert updated["last_sent_known"] is True

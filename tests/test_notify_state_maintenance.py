from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from scripts.notify_discord import (
    _default_asset_state,
    _sanitize_last_sent,
    LAST_SENT_RETENTION_DAYS,
    to_utc_iso,
)


def test_sanitize_last_sent_invalid_timestamp():
    record = _default_asset_state()
    record.update({
        "last_sent": "not-a-timestamp",
        "last_sent_decision": None,
    })
    archived = []

    _sanitize_last_sent("BTCUSD", record, archived, now=datetime(2025, 1, 1, tzinfo=timezone.utc))

    assert record["last_sent"] is None
    assert record["last_sent_known"] is True
    assert archived and archived[0]["reason"] == "invalid-format"


def test_sanitize_last_sent_future_timestamp_clamped():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    future = now + timedelta(minutes=30)

    record = _default_asset_state()
    record.update({
        "last_sent": to_utc_iso(future),
        "last_sent_decision": "buy",
    })
    archived = []

    _sanitize_last_sent("EURUSD", record, archived, now=now)

    assert record["last_sent"] is None
    assert record["last_sent_known"] is True
    assert archived and archived[0]["reason"] == "future"


def test_sanitize_last_sent_within_retention_kept():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    recent = now - timedelta(days=min(5, LAST_SENT_RETENTION_DAYS // 2 or 1))

    record = _default_asset_state()
    record.update({
        "last_sent": to_utc_iso(recent),
        "last_sent_decision": "sell",
    })
    archived = []

    _sanitize_last_sent("XAGUSD", record, archived, now=now)

    assert record["last_sent"] == to_utc_iso(recent)
    assert record["last_sent_known"] is True
    assert archived == []

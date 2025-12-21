from datetime import datetime, timedelta, timezone
from pathlib import Path

from analysis import apply_signal_stability_layer, save_json, to_utc_iso
from config import analysis_settings as settings


def _default_config():
    cfg = settings.load_config().get("signal_stability") or {}
    return cfg


def test_entry_cooldown_blocks_recent_signal(tmp_path):
    asset = "EURUSD"
    outdir = tmp_path / "public" / asset
    outdir.mkdir(parents=True)

    now = datetime.now(timezone.utc)
    state = {
        "last_notified_intent": "entry",
        "last_notified_side": "buy",
        "last_notified_at_utc": to_utc_iso(now - timedelta(minutes=5)),
        "entry_direction_history": ["buy"],
    }
    save_json(outdir / "signal_state.json", state)

    payload = {
        "asset": asset,
        "signal": "buy",
        "gates": {"missing": []},
        "retrieved_at_utc": to_utc_iso(now),
    }

    result = apply_signal_stability_layer(
        asset,
        payload,
        decision="buy",
        action_plan={},
        exit_signal=None,
        gates_missing=[],
        analysis_timestamp=payload["retrieved_at_utc"],
        outdir=outdir,
        stability_config=_default_config(),
        manual_positions={},
    )

    assert result["notify"]["should_notify"] is False
    assert result["notify"]["reason"] == "entry_cooldown_active"
    assert result["notify"].get("cooldown_until_utc")


def test_manual_tracking_blocks_entry_when_open(tmp_path):
    asset = "EURUSD"
    outdir = tmp_path / "public" / asset
    outdir.mkdir(parents=True)

    now = datetime.now(timezone.utc)
    payload = {
        "asset": asset,
        "signal": "buy",
        "gates": {"missing": []},
        "retrieved_at_utc": to_utc_iso(now),
    }

    manual_positions = {asset: {"side": "long", "opened_at_utc": to_utc_iso(now - timedelta(hours=1))}}

    result = apply_signal_stability_layer(
        asset,
        payload,
        decision="buy",
        action_plan={},
        exit_signal=None,
        gates_missing=[],
        analysis_timestamp=payload["retrieved_at_utc"],
        outdir=outdir,
        stability_config=_default_config(),
        manual_positions=manual_positions,
    )

    assert result["actionable"] is False
    assert result["notify"]["should_notify"] is False
    assert result["notify"]["reason"] == "position_already_open"

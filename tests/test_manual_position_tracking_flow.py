import os
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import analysis
import position_tracker
import scripts.notify_discord as notify_discord
from scripts.notify_discord import build_mobile_embed_for_asset
from unittest import mock


class ManualPositionFlowTests(unittest.TestCase):
    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def test_embed_carries_manual_position_line(self) -> None:
        now = datetime.now(timezone.utc)
        now_iso = self._now_iso()
        manual_positions = position_tracker.open_position(
            "BTCUSD",
            side="long",
            entry=100.0,
            sl=95.0,
            tp2=120.0,
            opened_at_utc=now_iso,
            positions={},
        )
        manual_state = position_tracker.compute_state(
            "BTCUSD", {"enabled": True}, manual_positions, now
        )
        signal = {
            "asset": "BTCUSD",
            "signal": "no entry",
            "reasons": ["monitoring"],
            "retrieved_at_utc": now_iso,
            "position_state": manual_state,
            "intent": "entry",
            "notify": {"reason": "position_already_open", "should_notify": False},
            "tracked_levels": {
                "entry": 100.0,
                "sl": 95.0,
                "tp2": 120.0,
                "opened_at_utc": now_iso,
            },
        }

        embed = build_mobile_embed_for_asset(
            "BTCUSD",
            state={},
            signal_data=signal,
            decision="no entry",
            mode="core",
            is_stable=True,
            is_flip=False,
            is_invalidate=False,
            manual_positions=manual_positions,
        )

        description = embed.get("description") or ""
        assert "Pozíciómenedzsment" in description
        assert "TP2" in description

    def test_tp2_hit_triggers_cooldown_and_blocks_entry(self) -> None:
        now = datetime.now(timezone.utc)
        now_iso = self._now_iso()
        manual_positions = position_tracker.open_position(
            "BTCUSD",
            side="long",
            entry=100.0,
            sl=90.0,
            tp2=110.0,
            opened_at_utc=now_iso,
            positions={},
        )

        changed, reason, updated_positions = position_tracker.check_close_by_levels(
            "BTCUSD", manual_positions, 111.0, now, cooldown_minutes=30
        )
        assert changed is True
        assert reason == "tp2_hit"

        cooldown_state = position_tracker.compute_state(
            "BTCUSD", {"enabled": True}, updated_positions, now
        )
        assert cooldown_state["cooldown_active"] is True
        assert cooldown_state["has_position"] is False

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = {"asset": "BTCUSD", "reasons": []}
            result = analysis.apply_signal_stability_layer(
                "BTCUSD",
                payload,
                decision="buy",
                action_plan=None,
                exit_signal=None,
                gates_missing=[],
                analysis_timestamp=now_iso,
                outdir=Path(tmpdir),
                manual_positions=updated_positions,
            )

        notify = result.get("notify") or {}
        assert notify.get("reason") == "cooldown_active"
        assert notify.get("should_notify") is False

    def test_entry_and_hard_exit_drive_position_lifecycle(self) -> None:
        now = datetime.now(timezone.utc)
        now_iso = self._now_iso()
        with tempfile.TemporaryDirectory() as tmpdir:
            positions_path = Path(tmpdir) / "positions.json"
            stability_cfg = {
                "enabled": False,
                "manual_position_tracking": {
                    "enabled": True,
                    "positions_file": str(positions_path),
                    "treat_missing_file_as_flat": True,
                    "post_exit_cooldown_minutes": {"default": 15},
                },
            }

            entry_payload = {
                "asset": "BTCUSD",
                "signal": "buy",
                "setup_grade": "A",
                "notify": {"should_notify": True},
                "reasons": [],
            }

            analysis.apply_signal_stability_layer(
                "BTCUSD",
                entry_payload,
                decision="buy",
                action_plan=None,
                exit_signal=None,
                gates_missing=[],
                analysis_timestamp=now_iso,
                outdir=Path(tmpdir),
                stability_config=stability_cfg,
            )

            positions_after_entry = position_tracker.load_positions(
                str(positions_path), True
            )
            manual_state = position_tracker.compute_state(
                "BTCUSD",
                stability_cfg["manual_position_tracking"],
                positions_after_entry,
                now,
            )
            assert manual_state["has_position"] is True

            analysis.apply_signal_stability_layer(
                "BTCUSD",
                {"asset": "BTCUSD", "signal": "no entry"},
                decision="no entry",
                action_plan=None,
                exit_signal={"state": "hard_exit"},
                gates_missing=[],
                analysis_timestamp=now_iso,
                outdir=Path(tmpdir),
                stability_config=stability_cfg,
                manual_positions=positions_after_entry,
            )

            positions_after_exit = position_tracker.load_positions(
                str(positions_path), True
            )
            cooldown_state = position_tracker.compute_state(
                "BTCUSD",
                stability_cfg["manual_position_tracking"],
                positions_after_exit,
                now,
            )
            assert cooldown_state["cooldown_active"] is True
            assert cooldown_state["has_position"] is False

            cooldown_payload = {
                "asset": "BTCUSD",
                "signal": "buy",
                "setup_grade": "A",
                "notify": {"should_notify": True},
                "reasons": [],
            }
            cooldown_result = analysis.apply_signal_stability_layer(
                "BTCUSD",
                cooldown_payload,
                decision="buy",
                action_plan=None,
                exit_signal=None,
                gates_missing=[],
                analysis_timestamp=now_iso,
                outdir=Path(tmpdir),
                stability_config=stability_cfg,
                manual_positions=positions_after_exit,
            )

            notify_meta = cooldown_result.get("notify") or {}
            assert notify_meta.get("should_notify") is False
            assert notify_meta.get("reason") == "cooldown_active"
            
    def test_analysis_is_read_only_when_writer_notify(self) -> None:
        now_iso = self._now_iso()
        with tempfile.TemporaryDirectory() as tmpdir:
            positions_path = Path(tmpdir) / "positions.json"
            stability_cfg = {
                "enabled": True,
                "manual_position_tracking": {
                    "enabled": True,
                    "writer": "notify",
                    "positions_file": str(positions_path),
                    "treat_missing_file_as_flat": True,
                },
            }

            payload = {"asset": "BTCUSD", "signal": "buy", "setup_grade": "A", "reasons": []}
            with mock.patch.object(position_tracker, "save_positions_atomic") as save_mock, \
                mock.patch.object(position_tracker, "open_position") as open_mock, \
                mock.patch.object(position_tracker, "close_position") as close_mock:
                analysis.apply_signal_stability_layer(
                    "BTCUSD",
                    payload,
                    decision="buy",
                    action_plan=None,
                    exit_signal=None,
                    gates_missing=[],
                    analysis_timestamp=now_iso,
                    outdir=Path(tmpdir),
                    stability_config=stability_cfg,
                    manual_positions={},
                )

            save_mock.assert_not_called()
            open_mock.assert_not_called()
            close_mock.assert_not_called()

    def test_notify_entry_emits_open_events(self) -> None:
        now = datetime.now(timezone.utc)
        now_iso = self._now_iso()
        tracking_cfg = {"enabled": True}
        manual_positions: dict = {}
        manual_state = position_tracker.compute_state(
            "BTCUSD", tracking_cfg, manual_positions, now
        )
        sig = {"entry": 101.0, "sl": 99.0, "tp2": 110.0}
        events = []

        def _capture(message: str, *, event: str, **fields: object) -> None:
            events.append({"event": event, **fields})

        with mock.patch.object(position_tracker, "log_audit_event", side_effect=_capture), \
            mock.patch.object(position_tracker, "save_positions_atomic"):
            manual_positions, manual_state, positions_changed, entry_opened = notify_discord._apply_manual_position_transitions(
                asset="BTCUSD",
                intent="entry",
                decision="buy",
                setup_grade="A",
                notify_meta={"should_notify": True},
                signal_payload=sig,
                manual_tracking_enabled=True,
                can_write_positions=True,
                manual_state=manual_state,
                manual_positions=manual_positions,
                tracking_cfg=tracking_cfg,
                now_dt=now,
                now_iso=now_iso,
                send_kind="normal",
                display_stable=True,
                missing_list=[],
                cooldown_map={},
                cooldown_default=20,
            )

            if positions_changed:
                position_tracker.save_positions_atomic("/tmp/ignore.json", manual_positions)
            if positions_changed and entry_opened:
                entry_level, sl_level, tp2_level = notify_discord.extract_trade_levels(sig)
                position_tracker.log_audit_event(
                    "entry open committed",
                    event="OPEN_COMMIT",
                    asset="BTCUSD",
                    intent="entry",
                    decision="buy",
                    entry_side="buy",
                    setup_grade="A",
                    entry=entry_level,
                    sl=sl_level,
                    tp2=tp2_level,
                    positions_file="/tmp/ignore.json",
                    send_kind="normal",
                )

        event_kinds = {entry.get("event") for entry in events}
        assert "OPEN_ATTEMPT" in event_kinds
        assert "OPEN_COMMIT" in event_kinds


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))

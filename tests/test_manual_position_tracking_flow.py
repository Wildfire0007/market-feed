import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import analysis
import position_tracker
from scripts.notify_discord import build_mobile_embed_for_asset


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
            

if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import analysis
import position_tracker


def test_entry_event_persists_and_is_reloaded():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    now_iso = now.isoformat().replace("+00:00", "Z")

    with tempfile.TemporaryDirectory() as tmpdir:
        positions_path = Path(tmpdir) / "public" / "_manual_positions.json"
        stability_cfg = {
            "enabled": False,
                "manual_position_tracking": {
                    "enabled": True,
                    "writer": "analysis",
                    "positions_file": str(positions_path),
                    "treat_missing_file_as_flat": True,
                    "post_exit_cooldown_minutes": {"default": 15},
            },
        }

        payload = {
            "asset": "BTCUSD",
            "signal": "buy",
            "setup_grade": "B",
            "notify": {"should_notify": True},
            "reasons": [],
        }

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
        )

        positions_after_entry = position_tracker.load_positions(
            str(positions_path), treat_missing_as_flat=True
        )
        manual_state = position_tracker.compute_state(
            "BTCUSD",
            stability_cfg["manual_position_tracking"],
            positions_after_entry,
            now,
        )

        assert manual_state["has_position"] is True
        assert manual_state["side"] == "buy"
        assert positions_path.exists()

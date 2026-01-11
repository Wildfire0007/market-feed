import tempfile
from datetime import datetime, timezone
from pathlib import Path

import analysis
import position_tracker
import state_db


def test_entry_event_persists_and_is_reloaded():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    now_iso = now.isoformat().replace("+00:00", "Z")

    with tempfile.TemporaryDirectory() as tmpdir:
        positions_path = Path(tmpdir) / "trading.db"
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


def test_manual_state_reads_open_position_from_db(monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    now_iso = now.isoformat().replace("+00:00", "Z")

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.chdir(tmpdir)
        db_path = Path(tmpdir) / "trading.db"
        state_db.initialize(db_path)
        connection = state_db.connect(db_path)
        with connection:
            connection.execute(
                """
                INSERT INTO positions (
                    asset,
                    entry_price,
                    size,
                    sl,
                    tp,
                    status,
                    strategy_metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "BTCUSD",
                    100.0,
                    1.0,
                    90.0,
                    120.0,
                    "OPEN",
                    '{"side":"long","opened_at_utc":"%s"}' % now_iso,
                ),
            )
        connection.close()

        positions_path = db_path
        stability_cfg = {
            "manual_position_tracking": {
                "enabled": True,
                "positions_file": str(positions_path),
                "treat_missing_file_as_flat": True,
            }
        }

        manual_state = analysis._manual_position_state(
            "BTCUSD",
            stability_cfg,
            now,
        )

        assert manual_state["has_position"] is True
        assert manual_state["side"] == "buy"

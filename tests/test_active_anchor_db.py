import json
import sqlite3
from pathlib import Path

import active_anchor
import state_db


def test_anchor_update_syncs_to_db(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "trading_state.db"
    anchor_path = tmp_path / "_active_anchor.json"
    monkeypatch.setattr(state_db, "DEFAULT_DB_PATH", db_path)
    active_anchor._DB_INITIALIZED = False

    active_anchor.record_anchor(
        "BTCUSD",
        "buy",
        price=101.5,
        timestamp="2024-03-12T14:05:00Z",
        path=str(anchor_path),
        extras={"note": "first"},
    )
    active_anchor.update_anchor_metrics(
        "BTCUSD",
        extras={"current_price": 103.0},
        path=str(anchor_path),
    )

    connection = state_db.connect()
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT * FROM anchors WHERE asset = ?",
            ("BTCUSD",),
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    assert row["anchor_type"] == "buy"
    meta = json.loads(row["meta_json"])
    assert meta["side"] == "buy"
    assert meta["current_price"] == 103.0

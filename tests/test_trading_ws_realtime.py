import json
import sqlite3

import state_db
import Trading


def test_ws_tick_updates_spot_price(tmp_path, monkeypatch):
    db_path = tmp_path / "trading.db"
    monkeypatch.setattr(state_db, "DEFAULT_DB_PATH", db_path)
    Trading._SPOT_DB_INITIALIZED = False
    with Trading._REALTIME_LAST_PRICE_LOCK:
        Trading._REALTIME_LAST_PRICE.clear()
    state_db.initialize(db_path)

    payload = {
        "symbol": "BTC/USD",
        "price": "43000.5",
        "timestamp": 1_700_000_000,
    }
    asset = Trading._handle_realtime_ws_message(json.dumps(payload))

    assert asset == "BTCUSD"
    assert Trading._REALTIME_LAST_PRICE["BTCUSD"]["price"] == 43000.5

    connection = state_db.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT price, utc, retrieved_at_utc, source FROM spot_prices WHERE asset = ?",
            ("BTCUSD",),
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    assert row["price"] == 43000.5
    assert row["source"] == "websocket"

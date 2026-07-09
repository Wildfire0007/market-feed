import sqlite3
import threading
import time

import state_db


def test_parallel_state_db_writers_wait_for_locks(tmp_path):
    db_path = tmp_path / "t.db"
    state_db.initialize(db_path)
    errors: list[sqlite3.OperationalError] = []

    def write_signals(asset: str) -> None:
        connection = state_db.connect(db_path)
        try:
            for index in range(50):
                try:
                    with connection:
                        cursor = connection.execute(
                            """
                            INSERT INTO signals (asset, direction, p_score, timestamp)
                            VALUES (?, ?, ?, ?)
                            """,
                            (asset, "pending", index / 100, f"2026-07-09T14:{index:02d}:00Z"),
                        )
                        connection.execute(
                            "UPDATE signals SET direction = ? WHERE id = ?",
                            ("long", cursor.lastrowid),
                        )
                    time.sleep(0.001)
                except sqlite3.OperationalError as exc:
                    errors.append(exc)
        finally:
            connection.close()

    threads = [
        threading.Thread(target=write_signals, args=("EURUSD",)),
        threading.Thread(target=write_signals, args=("XAUUSD",)),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []

    connection = state_db.connect(db_path)
    try:
        total_rows = connection.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        long_rows = connection.execute(
            "SELECT COUNT(*) FROM signals WHERE direction = 'long'"
        ).fetchone()[0]
        per_asset_rows = dict(
            connection.execute(
                "SELECT asset, COUNT(*) FROM signals GROUP BY asset"
            ).fetchall()
        )
    finally:
        connection.close()

    assert total_rows == 100
    assert long_rows == 100
    assert per_asset_rows == {"EURUSD": 50, "XAUUSD": 50}

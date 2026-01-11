import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import analysis
import state_db


def _set_market_update(db_path: Path, timestamp: str) -> None:
    connection = state_db.connect(db_path)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO market_data (id, last_updated_at)
                VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_updated_at=excluded.last_updated_at
                """,
                (timestamp,),
            )
    finally:
        connection.close()


def test_analysis_triggers_on_market_update(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    state_db.initialize(db_path)
    _set_market_update(db_path, datetime.now(timezone.utc).isoformat())

    triggered = threading.Event()

    def run_cycle() -> None:
        triggered.set()

    def update_later() -> None:
        time.sleep(0.02)
        new_ts = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
        _set_market_update(db_path, new_ts)

    updater = threading.Thread(target=update_later)
    updater.start()

    analysis.run_on_market_updates(
        run_cycle=run_cycle,
        db_path=db_path,
        poll_interval=0.01,
        max_cycles=1,
        max_wait_seconds=1.0,
    )

    updater.join(timeout=1.0)
    assert triggered.is_set()
 
EOF
)

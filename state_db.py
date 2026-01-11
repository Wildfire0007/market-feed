"""SQLite state storage helpers for trading state."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

DEFAULT_DB_PATH = Path("trading.db")


SCHEMA_STATEMENTS: Iterable[str] = (
    """
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        entry_price REAL,
        size REAL,
        sl REAL,
        tp REAL,
        status TEXT NOT NULL,
        strategy_metadata TEXT
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_positions_asset_status
        ON positions(asset, status);
    """,
    """
    CREATE TABLE IF NOT EXISTS anchors (
        asset TEXT NOT NULL,
        anchor_type TEXT NOT NULL,
        price REAL,
        timestamp TEXT NOT NULL,
        meta_json TEXT,
        PRIMARY KEY (asset, anchor_type)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_anchors_asset_type
        ON anchors(asset, anchor_type);
    """,
    """
    CREATE TABLE IF NOT EXISTS pending_exits (
        asset TEXT PRIMARY KEY,
        reason TEXT NOT NULL,
        closed_at_utc TEXT NOT NULL,
        cooldown_minutes INTEGER NOT NULL,
        source TEXT,
        run_id TEXT,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pending_exits_asset
        ON pending_exits(asset);
    """,
    """
    CREATE TABLE IF NOT EXISTS spot_prices (
        asset TEXT PRIMARY KEY,
        price REAL,
        utc TEXT,
        retrieved_at_utc TEXT,
        source TEXT,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS market_data (
        symbol TEXT PRIMARY KEY,
        price REAL,
        timestamp TEXT NOT NULL,
        source TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        direction TEXT NOT NULL,
        p_score REAL,
        timestamp TEXT NOT NULL
    );
    """,
)


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with WAL configured."""

    path = Path(db_path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    return connection


def initialize(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Initialize the trading state database and schema."""

    connection = connect(db_path)
    try:
        with connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
    finally:
        connection.close()


if __name__ == "__main__":
    initialize()

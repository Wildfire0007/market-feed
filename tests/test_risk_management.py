from datetime import datetime, timezone
from pathlib import Path

from risk_limits import evaluate_daily_lockout


def test_daily_lockout_two_losing_trades(tmp_path: Path):
    journal = tmp_path / "trade_journal.csv"
    journal.write_text(
        "asset,analysis_timestamp,validation_outcome,validation_rr\n"
        "GOLD_CFD,2026-07-04T01:00:00Z,stopped,-1\n"
        "XAGUSD,2026-07-04T02:00:00Z,stopped,-1\n",
        encoding="utf-8",
    )
    cfg = {"risk_limits": {"enabled": True, "daily_loss_limit_usd": 15, "daily_max_losing_trades": 2, "day_boundary_utc": "00:00", "count_ambiguous_as_loss": True, "lockout_scope": ["GOLD_CFD", "XAGUSD", "USOIL"]}}
    state = evaluate_daily_lockout(cfg, journal, now=datetime(2026, 7, 4, 3, tzinfo=timezone.utc))
    assert state["locked"] is True
    assert state["losing_trades"] == 2

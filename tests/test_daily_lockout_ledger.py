import csv
from datetime import datetime, timezone
from pathlib import Path

from risk_limits import evaluate_daily_lockout_from_ledger

CONFIG = {
    "risk_limits": {
        "enabled": True,
        "daily_loss_limit_usd": 15,
        "daily_max_losing_trades": 2,
        "day_boundary_utc": "00:00",
        "lockout_scope": ["GOLD_CFD", "XAGUSD", "USOIL"],
    }
}

FIELDS = [
    "ledger_id", "asset", "side", "order_type", "entry", "sl", "tp1", "tp2",
    "size_units", "opened_at_utc", "closed_at_utc", "close_reason", "outcome",
    "est_pnl_usd", "source_signal", "entry_signature", "trigger_bar_utc",
    "voided", "void_reason", "truth_verified_utc", "verify_note",
]


def _write_ledger(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: "" for k in FIELDS}, **row})


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def test_locked_after_daily_loss_cap_replay_2026_07_24(tmp_path):
    ledger = tmp_path / "trade_ledger.csv"
    _write_ledger(ledger, [
        {"asset": "USOIL", "outcome": "stopped", "voided": "false",
         "closed_at_utc": "2026-07-24T09:36:39Z", "est_pnl_usd": "-15.83"},
    ])
    state = evaluate_daily_lockout_from_ledger(CONFIG, ledger, now=_utc("2026-07-24T13:16:00"))
    assert state["locked"] is True
    assert state["realized_pnl_usd"] == -15.83
    assert state["losing_trades"] == 1


def test_not_locked_before_loss_row_closes(tmp_path):
    ledger = tmp_path / "trade_ledger.csv"
    _write_ledger(ledger, [
        {"asset": "USOIL", "outcome": "stopped", "voided": "false",
         "closed_at_utc": "2026-07-24T09:36:39Z", "est_pnl_usd": "-15.83"},
    ])
    state = evaluate_daily_lockout_from_ledger(CONFIG, ledger, now=_utc("2026-07-24T09:00:00"))
    assert state["locked"] is False
    assert state["labeled_trades"] == 0


def test_locked_on_two_losing_trades_below_cap(tmp_path):
    ledger = tmp_path / "trade_ledger.csv"
    _write_ledger(ledger, [
        {"asset": "USOIL", "outcome": "stopped", "voided": "false",
         "closed_at_utc": "2026-07-24T09:00:00Z", "est_pnl_usd": "-6.00"},
        {"asset": "GOLD_CFD", "outcome": "force_closed", "voided": "false",
         "closed_at_utc": "2026-07-24T10:00:00Z", "est_pnl_usd": "-5.00"},
    ])
    state = evaluate_daily_lockout_from_ledger(CONFIG, ledger, now=_utc("2026-07-24T11:00:00"))
    assert state["locked"] is True
    assert state["losing_trades"] == 2


def test_voided_and_expired_rows_excluded(tmp_path):
    ledger = tmp_path / "trade_ledger.csv"
    _write_ledger(ledger, [
        {"asset": "USOIL", "outcome": "stopped", "voided": "true",
         "closed_at_utc": "2026-07-24T09:00:00Z", "est_pnl_usd": "-50.00"},
        {"asset": "USOIL", "outcome": "expired", "voided": "false",
         "closed_at_utc": "2026-07-24T09:30:00Z", "est_pnl_usd": "-0.00"},
    ])
    state = evaluate_daily_lockout_from_ledger(CONFIG, ledger, now=_utc("2026-07-24T12:00:00"))
    assert state["locked"] is False
    assert state["labeled_trades"] == 0

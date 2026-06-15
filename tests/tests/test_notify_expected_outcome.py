import json

from scripts import notify_discord


def _write_klines(path, closes):
    values = [
        {
            "datetime": f"2026-06-15 10:{idx:02d}:00",
            "close": str(close),
        }
        for idx, close in enumerate(closes)
    ]
    (path / "klines_1m.json").write_text(
        json.dumps({"values": values}), encoding="utf-8"
    )


def test_expected_trade_outcome_requires_ten_usd_and_eta(tmp_path, monkeypatch):
    _write_klines(tmp_path, [100 + i * 0.1 for i in range(30)])
    monkeypatch.setattr(notify_discord.settings, "ASSET_COST_MODEL", {"GOLD_CFD": {"round_trip_pct": 0.0}})
    monkeypatch.setattr(notify_discord.settings, "LEVERAGE", {"GOLD_CFD": 20.0})

    outcome = notify_discord.build_expected_trade_outcome(
        tmp_path,
        "GOLD_CFD",
        {"spot": {"price": 100.0}},
        "buy",
        100.0,
        99.0,
        100.6,
        {
            "equity_usd": 100,
            "leverage": 20,
            "tp1_close_fraction": 1.0,
            "tp1_min_net_usd": 10,
            "eta_min_minutes": 1,
            "eta_max_minutes": 20,
            "max_chase_r": 0.2,
        },
    )

    assert outcome["tp1_net_usd"] == 12.0
    assert outcome["profit_gate_pass"] is True
    assert outcome["eta_gate_pass"] is True
    assert outcome["no_chase_pass"] is True
    assert outcome["passes"] is True


def test_expected_trade_outcome_blocks_late_chase(tmp_path, monkeypatch):
    _write_klines(tmp_path, [100 + i * 0.1 for i in range(30)])
    monkeypatch.setattr(notify_discord.settings, "ASSET_COST_MODEL", {"GOLD_CFD": {"round_trip_pct": 0.0}})
    monkeypatch.setattr(notify_discord.settings, "LEVERAGE", {"GOLD_CFD": 20.0})

    outcome = notify_discord.build_expected_trade_outcome(
        tmp_path,
        "GOLD_CFD",
        {"spot": {"price": 100.5}},
        "buy",
        100.0,
        99.0,
        100.6,
        {
            "equity_usd": 100,
            "leverage": 20,
            "tp1_close_fraction": 1.0,
            "tp1_min_net_usd": 10,
            "eta_min_minutes": 1,
            "eta_max_minutes": 20,
            "max_chase_r": 0.2,
        },
    )

    assert outcome["profit_gate_pass"] is True
    assert outcome["no_chase_pass"] is False
    assert outcome["passes"] is False


def test_expected_trade_outcome_uses_asset_leverage(tmp_path, monkeypatch):
    _write_klines(tmp_path, [100 + i * 0.1 for i in range(30)])
    monkeypatch.setattr(notify_discord.settings, "ASSET_COST_MODEL", {"BTCUSD": {"round_trip_pct": 0.0}})
    monkeypatch.setattr(notify_discord.settings, "LEVERAGE", {"BTCUSD": 2.0})

    outcome = notify_discord.build_expected_trade_outcome(
        tmp_path,
        "BTCUSD",
        {"spot": {"price": 100.0}},
        "buy",
        100.0,
        99.0,
        100.6,
        {
            "equity_usd": 100,
            "leverage": 20,
            "tp1_close_fraction": 1.0,
            "tp1_min_net_usd": 1,
            "eta_min_minutes": 1,
            "eta_max_minutes": 20,
            "max_chase_r": 0.2,
        },
    )

    assert outcome["leverage"] == 2.0
    assert outcome["notional_usd"] == 200.0

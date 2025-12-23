import os
from datetime import datetime, timezone

import position_tracker
import scripts.notify_discord as notify_discord


def test_management_channel_routing_excludes_market_scan(monkeypatch):
    # use traded assets only
    monkeypatch.setattr(
        notify_discord,
        "ASSETS",
        ["BTCUSD", "XAGUSD", "GOLD_CFD", "USOIL", "NVDA", "EURUSD"],  # deterministic order
    )
    
    asset_embeds = {
        "BTCUSD": {"title": "btc"},
        "XAGUSD": {"title": "xag"},
    }
    asset_channels = {"BTCUSD": "management", "XAGUSD": "market_scan"}

    live, management, market_scan = notify_discord._collect_channel_embeds(
        asset_embeds=asset_embeds,
        asset_channels=asset_channels,
        watcher_embeds=[{"title": "watcher"}],
        auto_close_embeds=[],
        heartbeat_snapshots=[],
        gate_embed=None,
        pipeline_embed=None,
    )

    management_embeds = [embed for _, embed in management]
    market_scan_embeds = [embed for _, embed in market_scan]
    live_embeds = [embed for _, embed in live]

    assert asset_embeds["BTCUSD"] in management_embeds
    assert asset_embeds["BTCUSD"] not in market_scan_embeds
    assert not live_embeds


def test_hard_exit_embed_uses_closed_state(tmp_path):
    now = datetime.now(timezone.utc)
    now_iso = notify_discord.to_utc_iso(now)

    tracking_cfg = {"enabled": True, "writer": "notify"}
    manual_positions = position_tracker.open_position(
        "BTCUSD",
        side="long",
        entry=100.0,
        sl=95.0,
        tp2=110.0,
        opened_at_utc=now_iso,
        positions={},
    )
    manual_state = position_tracker.compute_state(
        "BTCUSD", tracking_cfg, manual_positions, now
    )

    sig = {
        "asset": "BTCUSD",
        "signal": "no entry",
        "intent": "hard_exit",
        "setup_grade": "A",
        "notify": {"should_notify": True},
        "position_state": manual_state,
    }

    positions_path = str(tmp_path / "positions.json")
    open_commits = set()

    manual_positions, manual_state, _, _, _ = notify_discord._apply_and_persist_manual_transitions(
        asset="BTCUSD",
        intent="hard_exit",
        decision="no entry",
        setup_grade="A",
        notify_meta=sig.get("notify"),
        signal_payload=sig,
        manual_tracking_enabled=True,
        can_write_positions=True,
        manual_state=manual_state,
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now,
        now_iso=now_iso,
        send_kind="normal",
        display_stable=True,
        missing_list=[],
        cooldown_map={},
        cooldown_default=20,
        positions_path=positions_path,
        entry_level=None,
        sl_level=None,
        tp2_level=None,
        open_commits_this_run=open_commits,
        sig=sig,
    )

    embed = notify_discord.build_mobile_embed_for_asset(
        "BTCUSD",
        state={},
        signal_data=sig,
        decision="no entry",
        mode="core",
        is_stable=True,
        is_flip=False,
        is_invalidate=False,
        manual_positions=manual_positions,
    )

    description = embed.get("description") or ""
    assert manual_state.get("has_position") is False
    assert "Pozíciómenedzsment: aktív" not in description
    # cooldown info should be persisted after transition
    assert os.path.exists(positions_path)


def test_entry_embed_uses_pre_dispatch_manual_state():
    now = datetime.now(timezone.utc)
    
    tracking_cfg = {"enabled": True, "writer": "notify"}    
    manual_positions = {}
    manual_state = position_tracker.compute_state("BTCUSD", tracking_cfg, manual_positions, now)

    sig = {
        "asset": "BTCUSD",
        "signal": "buy",
        "intent": "entry",
        "setup_grade": "A",
        "entry": 100.0,
        "sl": 95.0,
        "tp2": 110.0,
        "notify": {"should_notify": True},
        "position_state": manual_state,
    }
    
    embed = notify_discord.build_mobile_embed_for_asset(
        "BTCUSD",
        state={},
        signal_data=sig,
        decision="buy",
        mode="core",
        is_stable=True,
        is_flip=False,
        is_invalidate=False,
        kind="normal",
        manual_positions=manual_positions,
    )

    description = embed.get("description") or ""
    assert manual_state.get("has_position") is False
    assert "Pozíciómenedzsment" not in description


def test_market_scan_and_heartbeat_suppress_manual_position_line():
    manual_state = {
        "has_position": True,
        "side": "buy",
        "entry": 100.0,
        "sl": 95.0,
        "tp2": 110.0,
    }
    sig = {
        "asset": "BTCUSD",
        "signal": "buy",
        "intent": "entry",
        "setup_grade": "A",
        "position_state": manual_state,
    }

    market_scan_embed = notify_discord.build_mobile_embed_for_asset(
        "BTCUSD",
        state={},
        signal_data=sig,
        decision="buy",
        mode="core",
        is_stable=True,
        is_flip=False,
        is_invalidate=False,
        kind="normal",
        manual_positions={"BTCUSD": {"entry": 100.0, "side": "buy"}},
        include_manual_position=False,
    )

    heartbeat_embed = notify_discord.build_mobile_embed_for_asset(
        "BTCUSD",
        state={},
        signal_data=sig,
        decision="buy",
        mode="core",
        is_stable=True,
        is_flip=False,
        is_invalidate=False,
        kind="heartbeat",
        manual_positions={"BTCUSD": {"entry": 100.0, "side": "buy"}},
        include_manual_position=False,
    )

    market_description = market_scan_embed.get("description") or ""
    heartbeat_description = heartbeat_embed.get("description") or ""

    assert "Pozíciómenedzsment" not in market_description
    assert "Pozíciómenedzsment" not in heartbeat_description

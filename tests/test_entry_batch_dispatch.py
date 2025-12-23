from datetime import datetime, timezone

import position_tracker
import scripts.notify_discord as notify_discord


def _base_pending(asset: str, manual_state: dict, signal_payload: dict, entry_record: notify_discord.EntryAuditRecord):
    return {
        "intent": "entry",
        "decision": "buy",
        "setup_grade": "A",
        "entry_side": "buy",
        "send_kind": "normal",
        "display_stable": True,
        "notify_meta": signal_payload.get("notify"),
        "manual_state_pre": manual_state,
        "manual_tracking_enabled": True,
        "can_write_positions": True,
        "state_loaded": True,
        "levels": {"entry": 100.0, "sl": 95.0, "tp2": 110.0},
        "gates_missing": [],
        "signal_payload": signal_payload,
        "audit": entry_record,
        "channel": "live",
    }


def _make_entry_record(asset: str, manual_state: dict, positions_path: str) -> notify_discord.EntryAuditRecord:
    return notify_discord.EntryAuditRecord(
        asset=asset,
        intent="entry",
        decision="buy",
        setup_grade="A",
        stable=True,
        send_kind="normal",
        should_notify=True,
        manual_state=manual_state,
        manual_tracking_enabled=True,
        can_write_positions=True,
        state_loaded=True,
        positions_file=positions_path,
        gates_missing=[],
        notify_reason=None,
        display_stable=True,
    )


def test_batch_failure_only_blocks_embeds_in_failed_batch(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    now_iso = notify_discord.to_utc_iso(now)
    positions_path = tmp_path / "positions.json"
    tracking_cfg = {
        "enabled": True,
        "writer": "notify",
        "positions_file": str(positions_path),
        "treat_missing_file_as_flat": True,
    }

    manual_positions: dict = {}
    manual_state_btc = position_tracker.compute_state("BTCUSD", tracking_cfg, manual_positions, now)
    manual_state_eur = position_tracker.compute_state("EURUSD", tracking_cfg, manual_positions, now)

    sig_template = {
        "signal": "buy",
        "intent": "entry",
        "setup_grade": "A",
        "entry": 100.0,
        "sl": 95.0,
        "tp2": 110.0,
        "notify": {"should_notify": True},
    }

    sig_btc = {"asset": "BTCUSD", **sig_template}
    sig_eur = {"asset": "EURUSD", **sig_template}

    entry_record_btc = _make_entry_record("BTCUSD", manual_state_btc, str(positions_path))
    entry_record_eur = _make_entry_record("EURUSD", manual_state_eur, str(positions_path))

    pending_btc = _base_pending("BTCUSD", manual_state_btc, sig_btc, entry_record_btc)
    pending_eur = _base_pending("EURUSD", manual_state_eur, sig_eur, entry_record_eur)

    asset_pairs = [("BTCUSD", {"title": "btc"}), ("EURUSD", {"title": "eur"})]

    def fake_post_batches(hook, content, embeds, batch_size=10):
        batches = [embeds[i : i + batch_size] for i in range(0, len(embeds), batch_size)]
        batch_results = []
        for idx, batch in enumerate(batches):
            success = idx == 0
            batch_results.append(
                {
                    "attempted": True,
                    "success": success,
                    "http_status": 204 if success else 500,
                    "error": None if success else "boom",
                    "message_id": None,
                    "batch_index": idx,
                    "embed_count": len(batch),
                }
            )

        return {
            "attempted": any(br.get("attempted") for br in batch_results),
            "success": all(br.get("success") for br in batch_results),
            "http_status": batch_results[-1]["http_status"],
            "error": batch_results[-1]["error"],
            "message_id": None,
            "batch_results": batch_results,
        }

    monkeypatch.setattr(notify_discord, "post_batches", fake_post_batches)

    dispatch_result = notify_discord.post_batches("https://hook", "content", [e for _, e in asset_pairs], batch_size=1)
    asset_results = notify_discord._map_batch_results_to_assets(asset_pairs, dispatch_result, batch_size=1)

    open_commits: set = set()
    manual_positions, manual_state_btc_after, commit_btc = notify_discord._finalize_entry_commit(
        "BTCUSD",
        pending_btc,
        asset_results["BTCUSD"],
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now,
        now_iso=now_iso,
        cooldown_map={},
        cooldown_default=20,
        positions_path=str(positions_path),
        open_commits_this_run=open_commits,
    )

    manual_positions, manual_state_eur_after, commit_eur = notify_discord._finalize_entry_commit(
        "EURUSD",
        pending_eur,
        asset_results["EURUSD"],
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now,
        now_iso=now_iso,
        cooldown_map={},
        cooldown_default=20,
        positions_path=str(positions_path),
        open_commits_this_run=open_commits,
    )

    persisted = position_tracker.load_positions(str(positions_path), treat_missing_as_flat=True)
    assert asset_results["BTCUSD"]["success"] is True
    assert asset_results["EURUSD"]["success"] is False
    assert commit_btc.get("committed") is True
    assert manual_state_btc_after.get("has_position") is True
    assert commit_eur.get("committed") is False
    assert manual_state_eur_after.get("has_position") is False
    assert "BTCUSD" in persisted
    assert "EURUSD" not in persisted


def test_commit_exception_after_dispatch_is_audited(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    now_iso = notify_discord.to_utc_iso(now)
    positions_path = tmp_path / "positions.json"
    tracking_cfg = {
        "enabled": True,
        "writer": "notify",
        "positions_file": str(positions_path),
        "treat_missing_file_as_flat": True,
    }

    manual_positions: dict = {}
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
    }

    entry_record = _make_entry_record("BTCUSD", manual_state, str(positions_path))
    pending = _base_pending("BTCUSD", manual_state, sig, entry_record)

    def boom_save(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(position_tracker, "save_positions_atomic", boom_save)
    events = []

    def capture_audit(message: str, *, event: str, **fields):
        events.append(event)

    monkeypatch.setattr(position_tracker, "log_audit_event", capture_audit)

    dispatch_result = {
        "attempted": True,
        "success": True,
        "http_status": 204,
        "error": None,
        "message_id": None,
        "batch_results": [
            {
                "attempted": True,
                "success": True,
                "http_status": 204,
                "error": None,
                "message_id": None,
                "batch_index": 0,
                "embed_count": 1,
            }
        ],
    }

    manual_positions, manual_state_after, commit_result = notify_discord._finalize_entry_commit(
        "BTCUSD",
        pending,
        dispatch_result,
        manual_positions=manual_positions,
        tracking_cfg=tracking_cfg,
        now_dt=now,
        now_iso=now_iso,
        cooldown_map={},
        cooldown_default=20,
        positions_path=str(positions_path),
        open_commits_this_run=set(),
    )

    entry_record.commit_result = commit_result
    entry_record.log_commit_decision()

    assert commit_result.get("committed") is False
    assert commit_result.get("exception") is not None
    assert manual_state_after.get("has_position") is True
    assert "ENTRY_DISPATCHED_BUT_NOT_COMMITTED" in events
 
EOF
)

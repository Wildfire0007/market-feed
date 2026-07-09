import pytest

from profit_target import build_profit_target_levels
import analysis

COSTS = {"GOLD_CFD": {"type": "pct", "round_trip_pct": 0.0006}}
CFG = {"margin_usd": 100, "net_tp1_usd_min": 10, "tp2_rr_multiple": 2.0, "sl_rr_min": 1.5, "max_required_move_atr1h_mult": 1.2}


def test_profit_target_20x_cost_adjustment():
    r = build_profit_target_levels(asset="GOLD_CFD", side="buy", entry=100.0, leverage=20, config=CFG, asset_cost_model=COSTS, min_stoploss_pct=0.001, atr5=0.01, atr1h=1.0)
    assert r.feasible
    assert r.meta["required_move_atr1h_ceiling"] == pytest.approx(0.012)    
    assert r.meta["required_net_move"] == pytest.approx(0.005)
    assert r.meta["required_gross_move"] == pytest.approx(0.0056)
    assert r.tp1 == pytest.approx(100.56)


def test_profit_target_10x_requires_one_percent_before_cost():
    r = build_profit_target_levels(asset="GOLD_CFD", side="sell", entry=100.0, leverage=10, config=CFG, asset_cost_model=COSTS, min_stoploss_pct=0.001, atr5=0.01, atr1h=2.0)
    assert r.feasible
    assert r.meta["required_net_move"] == pytest.approx(0.01)
    assert r.tp1 == pytest.approx(98.94)


def test_profit_target_infeasible_when_atr1h_target_unrealistic():
    r = build_profit_target_levels(asset="GOLD_CFD", side="buy", entry=100.0, leverage=20, config=CFG, asset_cost_model=COSTS, min_stoploss_pct=0.001, atr5=0.01, atr1h=0.1)
    assert not r.feasible
    assert r.reason == "profit_target_infeasible"
    assert r.meta["required_move_atr1h_ceiling"] == pytest.approx(0.0012)
    assert r.meta["required_move_over_ceiling"] == pytest.approx(0.0044)    


def test_soft_penalty_cap():
    assert analysis._cap_soft_penalty(50, 20, 12) == pytest.approx(38)


def test_bos_direction_consistency():
    assert analysis._bos_direction_to_bias("bos_up") == "long"
    assert analysis._bos_direction_to_bias("bos_down") == "short"


def test_profit_target_feasibility_gap_record_pass_fields():
    cfg = dict(CFG, max_required_move_atr1h_mult=2.4)
    r = build_profit_target_levels(
        asset="GOLD_CFD",
        side="buy",
        entry=100.0,
        leverage=20,
        config=cfg,
        asset_cost_model=COSTS,
        min_stoploss_pct=0.001,
        atr5=0.01,
        atr1h=0.3,
    )

    record = analysis._profit_target_feasibility_gap_record(
        "GOLD_CFD", r.meta, r.feasible, analysis.parse_utc_timestamp("2026-07-07T08:00:00Z")
    )

    assert r.feasible
    assert record == {
        "ts_utc": "2026-07-07T08:00:00Z",
        "asset": "GOLD_CFD",
        "gate": "profit_target_feasibility",
        "result": "pass",
        "required_gross_move_pct": pytest.approx(0.56),
        "atr1h_pct": pytest.approx(0.3),
        "ceiling_pct": pytest.approx(0.72),
        "mult": 2.4,
    }    


def test_log_profit_target_feasibility_always_blocked_path(tmp_path, monkeypatch):
    gap_path = tmp_path / "entry_gate_gap_log.jsonl"
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", gap_path)
    payload = {
        "signal": "no entry",
        "spot": {"price": 4000.0},
        "atr1h": 12.0,
        "entry_thresholds_meta": {},
        "missing": ["choppy_hard_block"],
    }

    analysis._log_profit_target_feasibility_always("GOLD_CFD", payload)

    rows = [analysis.json.loads(line) for line in gap_path.read_text(encoding="utf-8").splitlines()]
    lev = max(float(analysis.LEVERAGE.get("GOLD_CFD", 1.0) or 1.0), 1e-9)
    margin = float(analysis.PROFIT_TARGET_CONFIG.get("margin_usd", 100.0) or 100.0)
    net_min = float(analysis.PROFIT_TARGET_CONFIG.get("net_tp1_usd_min", 10.0) or 10.0)
    required = net_min / (margin * lev) + analysis.pt_round_trip_cost(
       "GOLD_CFD", analysis.ASSET_COST_MODEL, analysis.DEFAULT_COST_MODEL
    )
    required_pct = round(required * 100, 4)

    assert len(rows) == 1
    assert rows[0]["gate"] == "profit_target_feasibility"
    assert rows[0]["evaluated_in_flow"] is False
    assert rows[0]["blocked_by"] == "choppy_hard_block"
    assert rows[0]["atr1h_pct"] == pytest.approx(0.3)
    assert rows[0]["ceiling_pct"] == pytest.approx(0.72)
    assert rows[0]["required_gross_move_pct"] == pytest.approx(required_pct)
    assert rows[0]["result"] == ("pass" if required_pct <= 0.72 else "reject")


def test_log_profit_target_feasibility_always_skips_entry_window_closed(tmp_path, monkeypatch):
    gap_path = tmp_path / "entry_gate_gap_log.jsonl"
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", gap_path)
    payload = {
        "signal": "entry window closed",
        "spot": {"price": 4000.0},
        "atr1h": 12.0,
    }
    analysis._log_profit_target_feasibility_always("GOLD_CFD", payload)

    assert not gap_path.exists()


def test_log_profit_target_feasibility_always_dedupes_per_asset(tmp_path, monkeypatch):
    gap_path = tmp_path / "entry_gate_gap_log.jsonl"
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", gap_path)
    analysis._PT_FEAS_LOGGED_ASSETS.clear()
    payload = {
        "signal": "no entry",
        "spot": {"price": 4000.0},
        "atr1h": 12.0,
    }

    analysis._log_profit_target_feasibility_always("GOLD_CFD", payload)
    analysis._log_profit_target_feasibility_always("GOLD_CFD", payload)

    rows = gap_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_backfill_signal_probability_metadata_logs_feasibility_for_no_entry(tmp_path, monkeypatch):
    public_dir = tmp_path / "public"
    signal_dir = public_dir / "GOLD_CFD"
    signal_dir.mkdir(parents=True)
    gap_path = tmp_path / "entry_gate_gap_log.jsonl"
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", gap_path)
    analysis._PT_FEAS_LOGGED_ASSETS.clear()
    analysis.save_json(signal_dir / "signal.json", {
        "signal": "no entry",
        "spot": {"price": 4000.0},
        "atr1h": 12.0,
        "missing": ["choppy_hard_block"],
    })

    analysis._backfill_signal_probability_metadata("gold_cfd", base_dir=public_dir)

    rows = [analysis.json.loads(line) for line in gap_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["asset"] == "GOLD_CFD"
    assert rows[0]["gate"] == "profit_target_feasibility"
    assert rows[0]["blocked_by"] == "choppy_hard_block"


def test_entry_gate_payload_writes_ts_utc_and_bud_ts():
    row = analysis._entry_gate_log_payload("GOLD_CFD", analysis.parse_utc_timestamp("2026-07-07T08:00:00Z"), ["session"])
    assert row["ts_utc"] == "2026-07-07T08:00:00Z"
    assert row["ts_utc"].endswith("Z")
    assert row["bud_ts"]


def test_log_profit_target_feasibility_entry_missing_meta_flags_invariant(tmp_path, monkeypatch, caplog):
    gap_path = tmp_path / "entry_gate_gap_log.jsonl"
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", gap_path)
    analysis._PT_FEAS_LOGGED_ASSETS.clear()
    payload = {"signal": "buy", "spot": {"price": 4000.0}, "atr1h": 12.0, "entry_thresholds_meta": {}}

    analysis._log_profit_target_feasibility_always("GOLD_CFD", payload)

    row = analysis.json.loads(gap_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["invariant_violation"] is True
    assert "pt_feasibility_invariant_violation" in caplog.text


def test_log_profit_target_feasibility_entry_with_meta_has_no_invariant(tmp_path, monkeypatch):
    gap_path = tmp_path / "entry_gate_gap_log.jsonl"
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", gap_path)
    analysis._PT_FEAS_LOGGED_ASSETS.clear()
    payload = {"signal": "sell", "spot": {"price": 4000.0}, "atr1h": 12.0, "entry_thresholds_meta": {"profit_target": {"tp1": 1}}}

    analysis._log_profit_target_feasibility_always("GOLD_CFD", payload)

    row = analysis.json.loads(gap_path.read_text(encoding="utf-8").splitlines()[0])
    assert not row.get("invariant_violation", False)


def test_log_profit_target_feasibility_precision_arming_with_levels_flags_invariant(tmp_path, monkeypatch, caplog):
    gap_path = tmp_path / "entry_gate_gap_log.jsonl"
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", gap_path)
    analysis._PT_FEAS_LOGGED_ASSETS.clear()
    payload = {
        "signal": "precision_arming",
        "spot": {"price": 4000.0},
        "atr1h": 12.0,
        "entry_thresholds_meta": {},
        "precision_plan": {"entry": 4000.0, "sl": 3990.0, "tp1": 4020.0},
    }

    analysis._log_profit_target_feasibility_always("GOLD_CFD", payload)

    row = analysis.json.loads(gap_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["invariant_violation"] is True
    assert "pt_feasibility_invariant_violation" in caplog.text


def test_log_profit_target_feasibility_precision_arming_with_meta_has_no_invariant(tmp_path, monkeypatch):
    gap_path = tmp_path / "entry_gate_gap_log.jsonl"
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", gap_path)
    analysis._PT_FEAS_LOGGED_ASSETS.clear()
    payload = {
        "signal": "precision_arming",
        "spot": {"price": 4000.0},
        "atr1h": 12.0,
        "entry_thresholds_meta": {"profit_target": {"tp1": 1}},
        "precision_plan": {"entry": 4000.0, "sl": 3990.0, "tp1": 4020.0},
    }

    analysis._log_profit_target_feasibility_always("GOLD_CFD", payload)

    row = analysis.json.loads(gap_path.read_text(encoding="utf-8").splitlines()[0])
    assert not row.get("invariant_violation", False)


def test_log_profit_target_feasibility_precision_arming_without_levels_unchanged(tmp_path, monkeypatch):
    gap_path = tmp_path / "entry_gate_gap_log.jsonl"
    monkeypatch.setattr(analysis, "ENTRY_GATE_GAP_LOG_PATH", gap_path)
    analysis._PT_FEAS_LOGGED_ASSETS.clear()
    payload = {"signal": "precision_arming", "spot": {"price": 4000.0}, "atr1h": 12.0, "entry_thresholds_meta": {}}

    analysis._log_profit_target_feasibility_always("GOLD_CFD", payload)

    row = analysis.json.loads(gap_path.read_text(encoding="utf-8").splitlines()[0])
    assert not row.get("invariant_violation", False)

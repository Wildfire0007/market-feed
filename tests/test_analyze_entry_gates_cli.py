import importlib.util
import json
from pathlib import Path


def load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "analyze_entry_gates.py"
    spec = importlib.util.spec_from_file_location("analyze_entry_gates", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_cli_falls_back_to_recent_json(tmp_path, capsys):
    module = load_cli_module()

    public_dir = tmp_path / "public"
    public_dir.mkdir()
    sample = {
        "symbol": "btcusd",
        "timestamp": "2024-08-05T09:15:00Z",
        "entry_checks": {
            "spread_gate": False,
            "p_score_too_low": {"status": "failed"},
        },
    }
    (public_dir / "fallback.json").write_text(json.dumps(sample), encoding="utf-8")

    output_path = public_dir / "debug" / "entry_gate_stats.json"
    exit_code = module.main(
        [
            "--root",
            str(tmp_path),
            "--limit-runs",
            "5",
            "--output-json",
            str(output_path),
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "No dedicated entry gate log files found" in captured.err
    assert "Analysing 1 JSON file" in captured.err

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["BTCUSD"]["total_candidates"] == 1
    assert summary["BTCUSD"]["total_rejected"] == 1
    assert summary["BTCUSD"]["by_reason"] == {
        "p_score_too_low": 1,
        "spread_gate": 1,
    }
    assert summary["BTCUSD"]["by_time_of_day"]["mid"]["total_candidates"] == 1


def test_cli_prefers_entry_gate_logs(tmp_path, capsys):
    module = load_cli_module()

    logs_dir = tmp_path / "public" / "debug" / "entry_gates"
    logs_dir.mkdir(parents=True)
    log_payload = [
        {"symbol": "eurusd", "timestamp": "2024-08-05T19:10:00Z", "reasons": ["spread_gate"]},
        {"symbol": "btcusd", "timestamp": "2024-08-05T01:05:00Z", "reasons": []},
    ]
    log_lines = "\n".join(json.dumps(item) for item in log_payload)
    (logs_dir / "entry_gates_sample.jsonl").write_text(log_lines, encoding="utf-8")

    output_path = tmp_path / "public" / "debug" / "entry_gate_stats.json"
    exit_code = module.main([
        "--root",
        str(tmp_path),
        "--limit-runs",
        "5",
        "--output-json",
        str(output_path),
    ])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Processing 1 entry gate log file" in captured.err
    assert "Analysing" not in captured.err  # ensure fallback message absent

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["EURUSD"]["total_candidates"] == 1
    assert summary["EURUSD"]["total_rejected"] == 1
    assert summary["EURUSD"]["by_reason"] == {"spread_gate": 1}
    assert summary["BTCUSD"]["total_rejected"] == 0

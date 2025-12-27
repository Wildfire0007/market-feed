import json

import Trading


def test_refresh_entry_gate_stats_recent(tmp_path):
    source = tmp_path / "entry_gate_stats_recent.json"
    source.write_text(json.dumps({"sample": 1}), encoding="utf-8")

    out_dir = tmp_path / "public"
    out_dir.mkdir()

    Trading._refresh_entry_gate_stats_recent(str(out_dir), source_path=source)

    target = out_dir / "reports" / "entry_gate_stats_recent.json"
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"sample": 1}

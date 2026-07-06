from pathlib import Path


def test_choppy_hard_block_missing_merged_after_missing_constructed():
    src = Path("analysis.py").read_text(encoding="utf-8")
    start = src.index('if not regime_ok and "regime" in conds_core:')
    branch = src[start:src.index('if not atr_ok:', start)]
    assert 'critical_missing.append("choppy_hard_block")' in branch
    assert not any(line.strip().startswith('missing.append("choppy_hard_block")') for line in branch.splitlines())
    merge = src[src.index('missing = list(missing_core)'):src.index('required_list: List[str] = list(core_required)')]
    assert '"choppy_hard_block" in critical_missing' in merge
    assert 'missing.append("choppy_hard_block")' in merge

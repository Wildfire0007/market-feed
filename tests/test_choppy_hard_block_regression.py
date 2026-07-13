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


def test_choppy_hard_block_h1_adx_exemption_source_contract():
    src = Path("analysis.py").read_text(encoding="utf-8")
    start = src.index('if not regime_ok and "regime" in conds_core:')
    branch = src[start:src.index('if not atr_ok:', start)]
    assert 'latest_adx(k1h_closed, period=14)' in branch
    assert 'h1_adx_value >= CHOPPY_BLOCK_H1_ADX_EXEMPT' in branch
    assert 'P -= 10.0' in branch
    assert 'Regime: 5m chop 1h-trendben — soft büntetés (hard block felmentve)' in branch
    assert branch.index('P -= 10.0') < branch.index('Regime: 5m chop 1h-trendben')
    hard_block_tail = branch[branch.index('else:', branch.index('CHOPPY_BLOCK_H1_ADX_EXEMPT')):]
    assert 'critical_missing.append("choppy_hard_block")' in hard_block_tail    

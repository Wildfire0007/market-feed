# Gate tuning frontier

The repository currently contains only the frozen public snapshot, so the replayable sample is too small for a statistically meaningful frontier. The implemented grid-search support should be run against archived `public/` snapshots when available.

Recommended starting profile: `precision_metal_oil`.

| profile | assets | p_score_min | penalty_cap | measured signals/day | measured TP1-before-SL | note |
|---|---:|---:|---:|---:|---:|---|
| precision_metal_oil | GOLD_CFD, XAGUSD, USOIL | 32 / 30 / 31 | 12 | n/a | n/a | configured starting point pending archive replay |

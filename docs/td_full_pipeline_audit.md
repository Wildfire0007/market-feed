# TD Full Pipeline (5m) audit snapshot — 2025-11-09 07:54 UTC

## Pipeline findings
- GitHub Actions workflow `td-pipeline.yml` runs every 6 minutes with four jobs (`trading`, `analysis`, `predeploy`, `notify`) chained via `needs`, each force-resetting to `origin/main` before work; this deletes in-progress fixes and prevents testing feature branches.
- Trading and analysis steps overwrite `public/**` directly on `main` and push from CI; repeated retries collide because the job cancels in-flight runs yet uses shared working tree state.
- Logs show persistent Twelve Data 404 responses across all assets and repeated `NameError`/`UnboundLocalError` exceptions, yet the workflow reports success; failures are masked because `analysis.py` exceptions are caught after emitting partial outputs, and `pytest` never exercises live paths.
- Pipeline timing indicates ~74 s lag between trading and analysis; no explicit cache invalidation for stale spot/klines means weekend runs load weeks-old spot data, inflating spread guards.

## Strategy gating diagnosis
- BTC core trigger chain requires `structure_ok_5m`, `vwap_retest_ok`, and OFI, but optional modules are absent so the stub implementations always return `False`, making `structure(2of3)` impossible. Momentum overrides depend on the same unavailable state.
- ATR guardrails combine profile floors (120 USD in suppressed), 20-day percentile minima, and volatility overlays; with BTC ATR≈98 USD the floor alone blocks entry.
- Precision flow alignment blocks ~46% of recent signals because order-flow samples never populate (`order_flow_signals = 0`).
- Spread gate divides stale forced realtime spot (2025-10-25) by fresh 5m closes, yielding 98×ATR ratios and hard blocks.

## Parameter retune outline
- Activate a new "normal" profile targeting 1–2 entries/day by relaxing BTC ATR floor to 85 USD, reducing P-score min to 50, and downgrading structure gate to require at least one live confirmation when optional feeds are missing.
- Introduce per-asset suppressed adjustments for calm markets: e.g. BTC ATR floor 70 USD, precision threshold 48, OFI trigger 0.6; EURUSD/GOLD keep ATR multipliers ≥0.9 but drop P-score min by 5 points.
- Extend session entry windows for NVDA to include early liquidity (13:00–20:00 UTC) and enforce cooldowns (45 min default, 75 min suppressed) to stay within risk appetite.

## JSON/state hygiene
- Prune or age-out stale `spot_realtime.json` payloads older than 5 minutes; missing bid/ask should nullify `spread_ratio` instead of reusing forced price snapshots.
- Remove unused `_notify_state.json` keys `cooldown_override` & `manual_last_trigger` (never read by notifier) and archive `public/pipeline/trading_status.json` fields `exceptions`, `retries` which accumulate dead counters.

## Proposed remediation steps
1. Guard optional microstructure dependencies and allow single-source confirmation fallback when unavailable.
2. Parameterise ATR floors, precision gates, and P-score via config and switch active profile from `suppressed` to `normal` once telemetry confirms stability.
3. Add regression fixtures covering zero-order-flow, stale spot, and weekend closures to prevent silent "no entry" loops.
4. Implement JSON ageing for realtime spot data and remove unused notifier state fields.

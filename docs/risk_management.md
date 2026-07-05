# Risk management

## Daily risk lockout

`risk_limits` is config-driven. When enabled, the analysis reads labeled journal outcomes for the UTC trading day defined by `day_boundary_utc`. For assets in `lockout_scope`, a hard entry gate is triggered when either:

- realized daily PnL is at or below `-daily_loss_limit_usd`, or
- losing labeled trades reach `daily_max_losing_trades`.

`ambiguous` outcomes count as losses when `count_ambiguous_as_loss` is true. If there are no labeled trades for the day, no lockout is applied.

Operator-facing block reason: `Napi kockázati limit elérve — belépés tiltva a nap végéig`.

## TP1 partial close and breakeven

`tp1_management` emits manual operator instructions for GOLD_CFD, XAGUSD, and USOIL. At TP1, the operator should close `partial_close_pct` of the position and move the stop to breakeven adjusted by configured round-trip costs so the stop is not a small net-loss after costs. The remaining runner targets TP2.

Execution is manual: the system emits instructions only and never assumes automatic order execution.

## Session force-close rule

The management plan includes a force-close timestamp when the session close can be resolved: entry-window end minus `session_force_close_buffer_min`. This is intended to avoid overnight holding.

## Probabilistic targets

Targets, hit rates, and expectancy statistics are probabilistic validation aids. No output text encodes or implies an accuracy guarantee.

## Notification lifecycle configuration

`position_lifecycle.tp1_closes_position` closes tracked manual trades on TP1 with `tp1_closed`.
`entry_validity_minutes` is the base pending-entry lifetime. When `entry_validity_atr_adaptive` is true, the worker scales it by median/current ATR5m, clamped to 25%-200%.
`hard_exit.immediate_on` lists reasons that bypass hysteresis; `volatility_shock_atr5m_median_mult` defines ATR shock; `trend_reversal_requires.consecutive_runs` persists the confirmation counter.
`ambiguous_bar_counts_as` controls same-bar SL/TP handling and defaults to `sl`.
`state_unknown_max_age_minutes` controls stale heartbeat alerts, and `state_unknown_include_pending` can include pending orders.
`daily_digest_utc` controls the one-per-day actionable digest time.

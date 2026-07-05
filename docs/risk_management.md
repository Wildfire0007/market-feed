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

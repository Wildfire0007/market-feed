# Profit target configuration

`config/analysis_settings.json` contains a `profit_target` block used by live analysis and backtests:

- `margin_usd`: manual margin assumed per trade.
- `net_tp1_usd_min`: minimum net USD profit at TP1.
- `use_config_leverage`: use per-asset leverage from the config.
- `tp2_rr_multiple`: TP2 distance multiple relative to TP1 distance.
- `sl_rr_min`: minimum reward/risk using TP1.
- `max_required_move_atr1h_mult`: rejects targets that are too large versus current 1h ATR.

Required net move is `net_tp1_usd_min / (margin_usd * leverage)`. Round-trip percentage cost from `asset_cost_model` is added to produce the gross TP1 distance.

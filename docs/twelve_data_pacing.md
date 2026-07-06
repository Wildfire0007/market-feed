# Twelve Data pacing

`td_calls_per_minute_max` in `config/analysis_settings.json` caps outbound Twelve Data calls independently of `TD_PAUSE`. The current default is `45`, sized below the Grow 55 plan limit (55 credits/min). With `TD_PAUSE=2` and `TD_ASSET_FILTER=GOLD_CFD,XAGUSD,USOIL`, expected demand is about 17 calls/run, roughly 25/min peak, with 45/min as a safety ceiling.

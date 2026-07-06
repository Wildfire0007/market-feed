# Discord notification lifecycle

Manual operators act only on actionable cards; silence means no action is required.

```
flat --ENTRY--> pending --price trades through entry--> open
  ^              |                                      |
  |              +--validity expires (silent)-----------+
  |                                                     |
  +-- CLOSE / HARD EXIT / TP1 closes position <---------+
```

ATR-adaptive pending validity uses `base_minutes * median_atr5m_20d / current_atr5m`, clamped to 25%-200% of the configured base.

## Event → channel matrix

| Event | Meaning | Channel |
|---|---|---|
| ENTRY | Open position card | ACTIONABLE |
| BE / `tp1_hit` | Close/partial close and move SL to breakeven | ACTIONABLE |
| CLOSE | SL, TP2 or session force-close | ACTIONABLE |
| HARD EXIT | Macro lockout, volatility shock, or confirmed trend reversal | ACTIONABLE |
| STATE UNKNOWN | Open position with stale heartbeat/data | ACTIONABLE |
| Daily risk lockout | Daily lockout alert | ACTIONABLE |
| Market scan, gates, heartbeat diagnostics, pipeline diagnostics | Informational only | DIAGNOSTIC |

`DISCORD_WEBHOOK_URL_ACTIONABLE` and `DISCORD_WEBHOOK_URL_DIAGNOSTIC` are optional; if unset, both fall back to `DISCORD_WEBHOOK_URL`.

## Manual position audit log retention

`config/analysis_settings.json` controls append-only manual position audit file retention with `audit_log.max_mb` (rotate once the active JSONL file exceeds this size in MiB) and `audit_log.keep_files` (number of rotated `*.jsonl` files to retain).

## Notify state preservation and webhook audit
The TD notify job refreshes `public/` from the analysis artifact, so notify-owned state is saved before `rm -rf public` and restored after artifact download. The saved paths include the daily digest dedup file, state-unknown guard state, risk lockout notify state, management/lifecycle state, lifecycle inbox, and notify lock files.

Webhook delivery attempts from notification scripts append JSON lines to `public/monitoring/webhook_delivery.jsonl` with `ts_utc`, `script`, `channel_kind`, `status`, and `ok`. The file is rotated according to `webhook_delivery_log.max_mb` (MiB threshold, default 5) and `webhook_delivery_log.keep_files` (rotated files retained, default 1).

# Discord notification lifecycle

Manual operators act only on actionable cards; silence means no action is required.

```
flat --ENTRY--> pending --price trades through entry--> open
  ^              |                                      |
  |              +--validity expires -> CANCEL alert----+
  |                                                     |
  +-- CLOSE / HARD EXIT / TP1 closes position <---------+
```

ATR-adaptive pending validity uses `base_minutes * median_atr5m_20d / current_atr5m`, clamped to 25%-200% of the configured base.

## Event → channel matrix

| Event | Meaning | Channel |
|---|---|---|
| ENTRY | Open position card | ACTIONABLE |
| ENTRY EXPIRED | Pending limit/stop signal expired; delete broker order | ACTIONABLE |
| BE / `tp1_hit` | Close/partial close and move SL to breakeven | ACTIONABLE |
| CLOSE | SL, TP2 or session force-close | ACTIONABLE |
| HARD EXIT | Macro lockout, volatility shock, or confirmed trend reversal | ACTIONABLE |
| STATE UNKNOWN | Open position with stale heartbeat/data | ACTIONABLE |
| Daily risk lockout | Daily lockout alert | ACTIONABLE |
| Weekly measurement report | Heti mérési összefoglaló a határ utáni első futáson | ACTIONABLE |
| Market scan, gates, heartbeat diagnostics, pipeline diagnostics | Informational only | DIAGNOSTIC |

`DISCORD_WEBHOOK_URL_ACTIONABLE` and `DISCORD_WEBHOOK_URL_DIAGNOSTIC` are optional; if unset, both fall back to `DISCORD_WEBHOOK_URL`.

## Heartbeat watchdog quiet hours

The standalone heartbeat watchdog is context-aware for the overnight UTC quiet window (`22:00`–`05:00`). When no lifecycle position is `open` or `pending`, stale heartbeat alerts use the quiet-hours threshold (`150` minutes) and route to DIAGNOSTIC. If any position is open or pending, the override keeps the normal `30` minute threshold and routes to ACTIONABLE even during quiet hours.

The watchdog is stateless, so it self-deduplicates stale episodes by sending only near the first detection edge and hourly escalation edges derived from the active threshold. Alert cards include stale-since time in UTC and Europe/Budapest, whether quiet hours are active, and the open/pending lifecycle position count.

## Manual position audit log retention

`config/analysis_settings.json` controls append-only manual position audit file retention with `audit_log.max_mb` (rotate once the active JSONL file exceeds this size in MiB) and `audit_log.keep_files` (number of rotated `*.jsonl` files to retain).

## Notify state preservation and webhook audit
The TD notify job refreshes `public/` from the analysis artifact, so notify-owned state is saved before `rm -rf public` and restored after artifact download. The saved paths include the daily digest dedup file, weekly report dedup file, state-unknown guard state, risk lockout notify state, management/lifecycle state, lifecycle inbox, and notify lock files.
Webhook delivery attempts from notification scripts append JSON lines to `public/monitoring/webhook_delivery.jsonl` with `ts_utc`, `script`, `channel_kind`, `status`, and `ok`. The file is rotated according to `webhook_delivery_log.max_mb` (MiB threshold, default 5) and `webhook_delivery_log.keep_files` (rotated files retained, default 1).

## 2026-07-08 momentum override entry gate

`config/analysis_settings.json` keeps `momentum_override_entries.enabled` off by default during the live measurement phase. If operators enable it later, `respect_p_score_min: true` preserves the effective `p_score_min` gate so micro-bias/momentum override entries cannot dispatch below the configured P-score minimum.

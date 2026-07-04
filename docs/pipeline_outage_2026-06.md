# TD pipeline freeze diagnosis — 2026-06

The frozen public heartbeat (`2026-06-11T21:11:43Z`) is consistent with the TD workflow failing before the final `public/` commit step. The workflow ran every 2 minutes while fetching 6 assets across multiple intervals, which can exceed the Twelve Data free-tier credit budget (8 credits/minute). A single hard failure in `Trading.py` or downstream freshness checks prevented analysis outputs from reaching the commit/push step, while independent macro updates could still commit.

Mitigations in this change:

- schedule reduced to `*/5 * * * *`;
- TD pause increased to a sustainable value;
- the Trading step now retries with exponential backoff;
- a heartbeat watchdog sends a Discord alert when `public/system_heartbeat.json` is older than 30 minutes.

If the API key is exhausted or invalid, the retry still fails the data step, but the watchdog produces an operator-visible alert rather than silently leaving stale signals unnoticed.

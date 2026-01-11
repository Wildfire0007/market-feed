# Monitoring JSON séma (status.json és anchor állapot)

## status.json
Kimeneti mezők (flattenelt, extra mezők tiltva):
- `ok` (bool): össz pipeline egészségi állapota.
- `status` ("ok" | "error" | "reset"): állapotcímke.
- `generated_utc` (ISO 8601 UTC): a snapshot generálásának ideje.
- `assets` (objektum): minden eszközhöz `ok`, `signal`, opcionálisan `latency_seconds`, `expected_latency_seconds`, `notes`.
- `notes` (lista): reset/hiba bejegyzések `type`, `message`, `reset_utc` kulcsokkal.

A korábbi `td_base` mező nem kerül kiírásra (nem volt fogyasztó), így elkerülhető a félrevezető, elavult URL-ek tárolása.

## Anchor állapot (SQLite `anchors` tábla)
Aktív pozíció-metainformációk eszközönkénti rekordokban:
- Kötelező: `side`, `price` vagy `entry_price`, `timestamp`, `last_update`.
- Ajánlott: `initial_risk_abs`, `trail_log`, `partial_exits`, `max_favorable_excursion`, `max_adverse_excursion`, `current_pnl_abs`, `current_pnl_r`.
- Opcionális kontextus: `analysis_timestamp`, `precision_plan`, `p_score`, `realtime_confidence`, `order_flow_*`, stb.

A reset logika minden olyan bejegyzést töröl, amely nem tartalmaz érvényes irányt (`side`) vagy kizárólag meta kulcsokat tartalmaz. Így nem maradnak üres, nem használt kulcsok a táblában.

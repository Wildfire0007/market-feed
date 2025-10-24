# Intraday riport (Twelve Data-only)

Generálva (UTC): `2025-10-12T15:31:17Z`

### EURUSD

Spot (USD): **1.1613** • UTC: `2025-10-10T21:40:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Spot data stale: 2511 min behind (limit 20 min)
Hiányzó kapuk: data_integrity

### NSDQ100

Spot (USD): **589.4800** • UTC: `2025-10-10T19:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Spot data stale: 2616 min behind (limit 20 min)
Hiányzó kapuk: data_integrity

### GOLD_CFD

Spot (USD): **3997.5200** • UTC: `2025-10-11T12:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Spot data stale: 1596 min behind (limit 45 min)
Hiányzó kapuk: data_integrity

### BTCUSD

Spot (USD): **48250.00** • UTC: `2025-10-12T14:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Spot data stale: 36 min behind (limit 5 min)
Hiányzó kapuk: data_integrity


### USOIL

Spot (USD): **58.2400** • UTC: `2025-10-10T20:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Spot data stale: 2556 min behind (limit 45 min)
Hiányzó kapuk: data_integrity

#### Elemzés & döntés checklist
- 4H→1H trend bias + EMA21 rezsim
- Likviditás: HTF sweep / Fib zóna / EMA21 visszateszt / szerkezet
- 5M BOS vagy szerkezeti retest a trend irányába (micro BOS támogatás)
- ATR filter + TP minimum (költség + ATR alapú) + nettó TP1 ≥ +1.0%
- RR küszöb: core ≥2.0R, momentum ≥1.6R; kockázat ≤ 1.8%; minimum stoploss ≥1%
- Momentum override: EMA9×21 + ATR + BOS + micro-BOS (csak engedélyezett eszközöknél)

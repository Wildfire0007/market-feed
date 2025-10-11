# Intraday riport (Twelve Data-only)

Generálva (UTC): `2025-10-11T09:11:37Z`

### SOL

Spot (USD): **187.9400** • UTC: `2025-10-11T09:05:00+00:00`
Valószínűség: **P = 75%**
Forrás: Twelve Data (lokális JSON)

[BUY @ 187.5500; SL: 185.5563; TP1: 191.5374; TP2: 193.5311; mód: momentum; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Regime ok (EMA21 slope)
- Fib zóna konfluencia (0.618–0.886)
- ATR rendben
- Momentum override (5m EMA + ATR + BOS)
- Momentum: rész-realizálás javasolt 2.5R-n
- Momentum: micro BOS elfogadva (1m szerkezet)

### NSDQ100

Spot (USD): **589.4800** • UTC: `2025-10-10T19:55:00+00:00`
Valószínűség: **P = 90%**
Forrás: Twelve Data (lokális JSON)

[BUY @ 590.0900; SL: 588.0247; TP1: 594.2207; TP2: 596.2860; mód: core; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Bias(4H→1H)=long
- Regime ok (EMA21 slope)
- HTF sweep ok
- 5M BOS trendirányba
- ATR rendben

### GOLD_CFD

Spot (USD): **3992.9900** • UTC: `2025-10-11T09:05:00+00:00`
Valószínűség: **P = 49%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Fib zóna konfluencia (0.618–0.886); ATR rendben; missing: regime, bias, bos5m
Hiányzó kapuk: Regime (EMA21 slope), Bias, 5m BOS

### BNB

Spot (USD): **1126.5900** • UTC: `2025-10-11T09:05:00+00:00`
Valószínűség: **P = 75%**
Forrás: Twelve Data (lokális JSON)

[BUY @ 1125.8600; SL: 1108.7626; TP1: 1160.0549; TP2: 1177.1523; mód: momentum; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Regime ok (EMA21 slope)
- Fib zóna konfluencia (0.618–0.886)
- ATR rendben
- Momentum override (5m EMA + ATR + BOS)
- Momentum: rész-realizálás javasolt 2.5R-n
- Momentum: micro BOS elfogadva (1m szerkezet)

### USOIL

Spot (USD): **58.2400** • UTC: `2025-10-10T20:55:00+00:00`
Valószínűség: **P = 75%**
Forrás: Twelve Data (lokális JSON)

[SELL @ 58.4000; SL: 58.9107; TP1: 57.3786; TP2: 56.8679; mód: core; Ajánlott tőkeáttétel: 2.0×; RR≈3.00]
Indoklás:
- Bias(4H→1H)=short
- Regime ok (EMA21 slope)
- 5M BOS trendirányba
- ATR rendben

#### Elemzés & döntés checklist
- 4H→1H trend bias + EMA21 rezsim
- Likviditás: HTF sweep / Fib zóna / EMA21 visszateszt / szerkezet
- 5M BOS vagy szerkezeti retest a trend irányába (micro BOS támogatás)
- ATR filter + TP minimum (költség + ATR alapú)
- RR küszöb: core ≥2.0R, momentum ≥1.6R; kockázat ≤ 1.8%
- Momentum override: EMA9×21 + ATR + BOS + micro-BOS (csak kriptók)

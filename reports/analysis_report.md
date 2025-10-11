# Intraday riport (Twelve Data-only)

Generálva (UTC): `2025-10-11T09:56:37Z`

### SOL

Spot (USD): **184.1000** • UTC: `2025-10-11T09:50:00+00:00`
Valószínűség: **P = 95%**
Forrás: Twelve Data (lokális JSON)

[SELL @ 185.7900; SL: 186.9996; TP1: 183.3709; TP2: 182.1613; mód: core; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Bias override: 1h trend short + momentum támogatás
- Regime ok (EMA21 slope)
- 5M BOS trendirányba
- Fib zóna konfluencia (0.618–0.886)
- ATR rendben

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

Spot (USD): **3995.9900** • UTC: `2025-10-11T09:55:00+00:00`
Valószínűség: **P = 49%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Fib zóna konfluencia (0.618–0.886); ATR rendben; missing: regime, bias, bos5m
Hiányzó kapuk: Regime (EMA21 slope), Bias, 5m BOS

### BNB

Spot (USD): **1121.9800** • UTC: `2025-10-11T09:55:00+00:00`
Valószínűség: **P = 57%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Regime ok (EMA21 slope); Fib zóna konfluencia (0.618–0.886); ATR rendben; missing: bos5m|struct_break
Hiányzó kapuk: BOS/Structure

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

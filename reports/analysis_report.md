# Intraday riport (Twelve Data-only)

Generálva (UTC): `2025-10-11T14:41:51Z`

### SOL

Spot (USD): **182.7500** • UTC: `2025-10-11T14:40:00+00:00`
Valószínűség: **P = 95%**
Forrás: Twelve Data (lokális JSON)

[SELL @ 182.8700; SL: 184.3180; TP1: 179.9740; TP2: 178.5260; mód: core; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Bias(4H→1H)=short
- Regime ok (EMA21 slope)
- 5M BOS trendirányba
- Fib zóna konfluencia (0.618–0.886)
- ATR rendben
- Diagnosztika: k1h: utolsó zárt gyertya 101 perc késésben van
- Diagnosztika: k4h: utolsó zárt gyertya 401 perc késésben van

### NSDQ100

Spot (USD): **589.4800** • UTC: `2025-10-10T19:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Piac zárva (hétvége); Bias(4H→1H)=long; Regime ok (EMA21 slope); HTF sweep ok; 5M BOS trendirányba; ATR rendben; Diagnosztika: k1m: utolsó zárt gyertya 1123 perc késésben van; Diagnosztika: k5m: utolsó zárt gyertya 1131 perc késésben van; Diagnosztika: k1h: utolsó zárt gyertya 1211 perc késésben van; Diagnosztika: k4h: utolsó zárt gyertya 1511 perc késésben van; missing: session
Hiányzó kapuk: Session

### GOLD_CFD

Spot (USD): **3997.5200** • UTC: `2025-10-11T12:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Piac zárva (hétvége); Fib zóna konfluencia (0.618–0.886); Diagnosztika: k1m: utolsó zárt gyertya 103 perc késésben van; Diagnosztika: k5m: utolsó zárt gyertya 111 perc késésben van; Diagnosztika: k1h: utolsó zárt gyertya 221 perc késésben van; Diagnosztika: k4h: utolsó zárt gyertya 401 perc késésben van; missing: session, regime, bias, bos5m, atr
Hiányzó kapuk: Session, Regime (EMA21 slope), Bias, 5m BOS, ATR

### BNB

Spot (USD): **1132.6900** • UTC: `2025-10-11T14:40:00+00:00`
Valószínűség: **P = 95%**
Forrás: Twelve Data (lokális JSON)

[SELL @ 1131.2800; SL: 1135.2395; TP1: 1123.3610; TP2: 1119.4016; mód: core; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Bias override: 1h trend short + momentum támogatás
- Regime ok (EMA21 slope)
- 5M BOS trendirányba
- Fib zóna konfluencia (0.618–0.886)
- ATR rendben
- Diagnosztika: k1h: utolsó zárt gyertya 101 perc késésben van
- Diagnosztika: k4h: utolsó zárt gyertya 401 perc késésben van

### USOIL

Spot (USD): **58.2400** • UTC: `2025-10-10T20:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Piac zárva (hétvége); Bias(4H→1H)=short; Regime ok (EMA21 slope); 5M BOS trendirányba; ATR rendben; Diagnosztika: k1m: utolsó zárt gyertya 1063 perc késésben van; Diagnosztika: k5m: utolsó zárt gyertya 1071 perc késésben van; Diagnosztika: k1h: utolsó zárt gyertya 1181 perc késésben van; Diagnosztika: k4h: utolsó zárt gyertya 1361 perc késésben van; missing: session
Hiányzó kapuk: Session

#### Elemzés & döntés checklist
- 4H→1H trend bias + EMA21 rezsim
- Likviditás: HTF sweep / Fib zóna / EMA21 visszateszt / szerkezet
- 5M BOS vagy szerkezeti retest a trend irányába (micro BOS támogatás)
- ATR filter + TP minimum (költség + ATR alapú)
- RR küszöb: core ≥2.0R, momentum ≥1.6R; kockázat ≤ 1.8%
- Momentum override: EMA9×21 + ATR + BOS + micro-BOS (csak kriptók)

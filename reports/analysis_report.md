# Intraday riport (Twelve Data-only)

Generálva (UTC): `2025-10-11T14:51:36Z`

### SOL

Spot (USD): **182.9000** • UTC: `2025-10-11T14:45:00+00:00`
Valószínűség: **P = 95%**
Forrás: Twelve Data (lokális JSON)

[SELL @ 182.7500; SL: 183.3896; TP1: 181.4708; TP2: 180.8311; mód: core; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Bias(4H→1H)=short
- Regime ok (EMA21 slope)
- 5M BOS trendirányba
- Fib zóna konfluencia (0.618–0.886)
- ATR rendben
- Diagnosztika: k5m: utolsó zárt gyertya 11 perc késésben van
- Diagnosztika: k1h: utolsó zárt gyertya 111 perc késésben van
- Diagnosztika: k4h: utolsó zárt gyertya 411 perc késésben van

### NSDQ100

Spot (USD): **589.4800** • UTC: `2025-10-10T19:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Piac zárva (hétvége); Bias(4H→1H)=long; Regime ok (EMA21 slope); HTF sweep ok; 5M BOS trendirányba; ATR rendben; Diagnosztika: k1m: utolsó zárt gyertya 1133 perc késésben van; Diagnosztika: k5m: utolsó zárt gyertya 1141 perc késésben van; Diagnosztika: k1h: utolsó zárt gyertya 1221 perc késésben van; Diagnosztika: k4h: utolsó zárt gyertya 1521 perc késésben van; missing: session
Hiányzó kapuk: Session

### GOLD_CFD

Spot (USD): **3997.5200** • UTC: `2025-10-11T12:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Piac zárva (hétvége); Fib zóna konfluencia (0.618–0.886); Diagnosztika: k1m: utolsó zárt gyertya 113 perc késésben van; Diagnosztika: k5m: utolsó zárt gyertya 121 perc késésben van; Diagnosztika: k1h: utolsó zárt gyertya 231 perc késésben van; Diagnosztika: k4h: utolsó zárt gyertya 411 perc késésben van; missing: session, regime, bias, bos5m, atr
Hiányzó kapuk: Session, Regime (EMA21 slope), Bias, 5m BOS, ATR

### BNB

Spot (USD): **1129.8800** • UTC: `2025-10-11T14:50:00+00:00`
Valószínűség: **P = 95%**
Forrás: Twelve Data (lokális JSON)

[SELL @ 1131.6700; SL: 1135.6308; TP1: 1123.7483; TP2: 1119.7875; mód: core; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Bias override: 1h trend short + momentum támogatás
- Regime ok (EMA21 slope)
- 5M BOS trendirányba
- Fib zóna konfluencia (0.618–0.886)
- ATR rendben
- Diagnosztika: k1h: utolsó zárt gyertya 111 perc késésben van
- Diagnosztika: k4h: utolsó zárt gyertya 411 perc késésben van

### USOIL

Spot (USD): **58.2400** • UTC: `2025-10-10T20:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Piac zárva (hétvége); Bias(4H→1H)=short; Regime ok (EMA21 slope); 5M BOS trendirányba; ATR rendben; Diagnosztika: k1m: utolsó zárt gyertya 1073 perc késésben van; Diagnosztika: k5m: utolsó zárt gyertya 1081 perc késésben van; Diagnosztika: k1h: utolsó zárt gyertya 1191 perc késésben van; Diagnosztika: k4h: utolsó zárt gyertya 1371 perc késésben van; missing: session
Hiányzó kapuk: Session

#### Elemzés & döntés checklist
- 4H→1H trend bias + EMA21 rezsim
- Likviditás: HTF sweep / Fib zóna / EMA21 visszateszt / szerkezet
- 5M BOS vagy szerkezeti retest a trend irányába (micro BOS támogatás)
- ATR filter + TP minimum (költség + ATR alapú)
- RR küszöb: core ≥2.0R, momentum ≥1.6R; kockázat ≤ 1.8%
- Momentum override: EMA9×21 + ATR + BOS + micro-BOS (csak kriptók)

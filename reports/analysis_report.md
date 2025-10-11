# Intraday riport (Twelve Data-only)

Generálva (UTC): `2025-10-11T14:22:10Z`

### SOL

Spot (USD): **183.8700** • UTC: `2025-10-11T14:20:00+00:00`
Valószínűség: **P = 95%**
Forrás: Twelve Data (lokális JSON)

[SELL @ 184.0500; SL: 184.7233; TP1: 182.7034; TP2: 182.0301; mód: core; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Bias(4H→1H)=short
- Regime ok (EMA21 slope)
- 5M BOS trendirányba
- Fib zóna konfluencia (0.618–0.886)
- ATR rendben
- Diagnosztika: k1h: utolsó zárt gyertya 82 perc késésben van
- Diagnosztika: k4h: utolsó zárt gyertya 382 perc késésben van

### NSDQ100

Spot (USD): **589.4800** • UTC: `2025-10-10T19:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Piac zárva (hétvége); Bias(4H→1H)=long; Regime ok (EMA21 slope); HTF sweep ok; 5M BOS trendirányba; ATR rendben; Diagnosztika: k1m: utolsó zárt gyertya 1104 perc késésben van; Diagnosztika: k5m: utolsó zárt gyertya 1112 perc késésben van; Diagnosztika: k1h: utolsó zárt gyertya 1192 perc késésben van; Diagnosztika: k4h: utolsó zárt gyertya 1492 perc késésben van; missing: session
Hiányzó kapuk: Session

### GOLD_CFD

Spot (USD): **3997.5200** • UTC: `2025-10-11T12:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Piac zárva (hétvége); Fib zóna konfluencia (0.618–0.886); Diagnosztika: k1m: utolsó zárt gyertya 84 perc késésben van; Diagnosztika: k5m: utolsó zárt gyertya 92 perc késésben van; Diagnosztika: k1h: utolsó zárt gyertya 202 perc késésben van; Diagnosztika: k4h: utolsó zárt gyertya 382 perc késésben van; missing: session, regime, bias, bos5m, atr
Hiányzó kapuk: Session, Regime (EMA21 slope), Bias, 5m BOS, ATR

### BNB

Spot (USD): **1136.1700** • UTC: `2025-10-11T14:20:00+00:00`
Valószínűség: **P = 75%**
Forrás: Twelve Data (lokális JSON)

[BUY @ 1136.6300; SL: 1130.7053; TP1: 1148.4794; TP2: 1154.4041; mód: momentum; Ajánlott tőkeáttétel: 3.0×; RR≈3.00]
Indoklás:
- Regime ok (EMA21 slope)
- Fib zóna konfluencia (0.618–0.886)
- ATR rendben
- Momentum override (5m EMA + ATR + BOS)
- Momentum: rész-realizálás javasolt 2.5R-n
- Momentum: micro BOS elfogadva (1m szerkezet)
- Diagnosztika: k1h: utolsó zárt gyertya 82 perc késésben van
- Diagnosztika: k4h: utolsó zárt gyertya 382 perc késésben van

### USOIL

Spot (USD): **58.2400** • UTC: `2025-10-10T20:55:00+00:00`
Valószínűség: **P = 0%**
Forrás: Twelve Data (lokális JSON)

**Állapot:** no entry — Piac zárva (hétvége); Bias(4H→1H)=short; Regime ok (EMA21 slope); 5M BOS trendirányba; ATR rendben; Diagnosztika: k1m: utolsó zárt gyertya 1044 perc késésben van; Diagnosztika: k5m: utolsó zárt gyertya 1052 perc késésben van; Diagnosztika: k1h: utolsó zárt gyertya 1162 perc késésben van; Diagnosztika: k4h: utolsó zárt gyertya 1342 perc késésben van; missing: session
Hiányzó kapuk: Session

#### Elemzés & döntés checklist
- 4H→1H trend bias + EMA21 rezsim
- Likviditás: HTF sweep / Fib zóna / EMA21 visszateszt / szerkezet
- 5M BOS vagy szerkezeti retest a trend irányába (micro BOS támogatás)
- ATR filter + TP minimum (költség + ATR alapú)
- RR küszöb: core ≥2.0R, momentum ≥1.6R; kockázat ≤ 1.8%
- Momentum override: EMA9×21 + ATR + BOS + micro-BOS (csak kriptók)

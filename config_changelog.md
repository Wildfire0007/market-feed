# Config changelog

## 2026-07-07
- A heti mérési riport ACTIONABLE csatornára kerül, ISO-hét deduplikációval és vasárnap 20:30 UTC határ utáni első futásos küldéssel.
- Az entry gate log új sorai `ts_utc` mezőt is írnak a `bud_ts` átmeneti megtartása mellett.
- A profit-target feasibility mérősor hangos invariant violation jelzést kap ENTRY jel nélküli deep meta esetén.

## 2026-07-06
- A lejárt pending jelzések ACTIONABLE törlési kártyát kapnak deduplikált állapottal.
- A napi digest a `public/journal/trade_journal.csv` címkézett napló alapján számol kimeneteket és realizált napi PnL-t.
- A TP1/TP2/SL/hard-exit operátori szövegek pontos, számszerű kézi végrehajtási utasítások.

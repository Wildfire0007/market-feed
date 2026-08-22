# Config changelog

- Terv-feasibility kapu az ENTRY-kártya kibocsátási pontján (`evaluate_plan_feasibility`
  + notify_discord kapu): min_stoploss_pct floor, TP1 nettó-minimum (RR=2 rögzített) és
  2,4×ATR1h plafon — a precision_arming út is a floor/feasibility szabályok alá tartozik,
  hiányzó ATR1h esetén fail-closed. (Bizonyított hiba javítása: P2 feasibility-bypass +
  P3 floor-bypass — audit: claude_audit_P2_P3_2026-08-13.md; a 10 éles trade-ből 4 floor
  alatt ment ki.)

## 2026-08-01
- USOIL felfüggesztve az aktív roster-ből (feed-integritási karantén): a TD WTI-feed
  2026-07-20 óta dokumentáltan revizionista (396+ esemény, Δmax 1.77, fantom-SL 07-30),
  a fémek feedje tiszta. Visszavétel: docs/operator_handbook.md szerinti próbahét-eljárással.

- Napi kockázati lockout: új ledger-alapú kiértékelés (`evaluate_daily_lockout_from_ledger`)
  és ENTRY-kártya kapu a notify_discord kibocsátási pontján — a precision_arming út is a
  −$15 / 2 vesztes napi sapka alá tartozik. (Bizonyított hiba javítása: a 2026-07-24-i
  sapka-átlépés — második ENTRY-kártya −$15,83 realizált napi veszteség után.)


## 2026-07-07
- A heti mérési riport ACTIONABLE csatornára kerül, ISO-hét deduplikációval és vasárnap 20:30 UTC határ utáni első futásos küldéssel.
- Az entry gate log új sorai `ts_utc` mezőt is írnak a `bud_ts` átmeneti megtartása mellett.
- A profit-target feasibility mérősor hangos invariant violation jelzést kap ENTRY jel nélküli deep meta esetén.

## 2026-07-06
- A lejárt pending jelzések ACTIONABLE törlési kártyát kapnak deduplikált állapottal.
- A napi digest a `public/journal/trade_journal.csv` címkézett napló alapján számol kimeneteket és realizált napi PnL-t.
- A TP1/TP2/SL/hard-exit operátori szövegek pontos, számszerű kézi végrehajtási utasítások.

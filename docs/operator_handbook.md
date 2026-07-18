# OPERÁTORI KÉZIKÖNYV ÉS DÖNTÉSI KERETRENDSZER
## Wildfire0007/market-feed — v1.0, 2026-07-17
### Ez a dokumentum a rendszer üzemeltetésének teljes leírása. Minden szabály mögött mért indok áll; a változtatás rendje a 6. fejezetben.

---

## 1. NAPI ÜZEMELTETÉS (a teljes protokoll)

**Session:** hétköznap 09:00–18:30 CEST (belépési ablak); 18:25 CEST-kor kényszerzárás-kártya jön minden nyitott pozícióra. Éjjel/hétvégén a rendszer önjáró, a riasztások némítva (kivéve: nyitott pozíció mellett bármikor hangos).

**Amikor ENTRY-kártya jön (🟢 zöld = LONG/Buy fül, 🔴 piros = SHORT/Short fül):**
1. HÁROM KÉRDÉS: (a) Érvényes még? (⏳ sor az eToro-blokkban) (b) A spot a "Ne nyiss" határon belül? (c) Van margó és ~1 óra jelenlét? — Bármelyik NEM → kihagyod, és a jegyzőkönyvbe írod az okát. Mind IGEN → nyitsz, gépiesen.
2. VÉGREHAJTÁS (divergencia-adapter protokoll): AZONNALI piaci nyitás a kártya iránya szerint → Amount = a kártya "eToro Amount" sora (minimál fázisban ×0,1–0,2; teljes fázisban natúr) → tőkeáttét a kártyáról → SL/TP beállítása Rate-nézetben OFFSET-tEL: TP = kártya-TP1 mínusz offset, SL = kártya-SL plusz offset (offszetek: GOLD ±1,5$ · SILVER ±0,08$ · OIL ±0,10$).
3. ZÁRÓ KÁRTYA = AZONNALI ZÁRÁS az eToro-ártól függetlenül (TP1/SL/HARD EXIT/SESSION — mind felszólító módban mondja a teendőt). Ha a bróker-TP/SL már zárt, a kártya csak megerősítés.
4. JEGYZŐKÖNYV trade-enként: `kártya-idő | fill-idő | fill-ár | rendszer-kimenet | bróker-kimenet | bróker-P&L` — kihagyásnál: `kihagyva: <ok>`.

**Vas-szabályok:** TP1-en teljes zárás (nincs runner, nincs kézi SL-húzás — a TP2-kérdésre a kontrafaktuális elemzés válaszol majd); P Score alapján NEM válogatsz (minden kártya azonos téttel — különben a statisztika az ízlésedet méri); napi −$15 lockout és minden kártya-utasítás feltétel nélkül követendő; paraméterhez csak a 6. fejezet rendje szerint nyúlunk.

**Kártya-szótár:** 🟢/🔴 NYISS = nyitás a fenti lépésekkel · ❌ JEL LEJÁRT = függő megbízás törlése · 🟢 TP1 ELÉRVE / 🔴 SL ELÉRVE = zárás/ellenőrzés · 🔴 HARD EXIT / 🟠 SESSION ZÁRÁS = azonnali kézi piaci zárás · ⚠️ ADATKIESÉS = kézi felügyelet, amíg új kártya nem jön · 📋/📊 összefoglalók = olvasnivaló. Az "Ár-forrás: TD/OANDA" lábjegyzet emlékeztet: a rendszer és a bróker árfolyama eltérhet — ezért zárol kártyára, nem árra.

## 2. HETI RITUÁLÉ (vasárnap este, ~30 perc)

1. 22:30 CEST után megjön a Heti mérési riport. Ellenőrzőlista: (a) a "Címkézett ügyletek" egyezik-e a ledger érvényes soraival? (b) a Wilson-sorok N-je nő-e? (c) a "Kapu-vétók top 5" összetétele változott-e drasztikusan az előző héthez képest? (d) van-e `invariant_violation` vagy truth-checker hiba a héten? (bármelyikre gyanú → 5. fejezet).
2. Jegyzőkönyv-összegzés: heti bróker-P&L vs rendszer-P&L; medián kártya→fill késés; kihagyások és okaik.
3. Döntési kapuk ellenőrzése (3. fejezet): elértünk-e N-küszöböt? Ha igen, a hozzá rendelt elemzés lefuttatása/lefuttattatása.

## 3. ELŐRE RÖGZÍTETT DÖNTÉSI KAPUK (ezek a rendszer "végrendelete" — nem hangulat, hanem N dönt)

- **N=10 éles trade → latencia/költség-kapu:** ha medián kártya→fill ≤ 90 s ÉS a mért csúszás+költség a modellen belül → átállás teljes tétre (Amount natúr). Ha nem: maradj minimálon, és a költségmodell frissítendő a mért értékre (dokumentált kalibráció).
- **N=30 ledger-trade → első Wilson-pillantás + TP2-runner kontrafaktuális + P-sáv tábla.** Még nem döntés — irány.
- **N=50–100 → GO/NO-GO/SCALE:** ha a TP1-találati arány 95%-os Wilson-intervallumának ALSÓ széle > fedezeti szint (~36–38%, a mért költséggel) → tét-skálázás lépcsőkben (stake.multiplier 1.0→2.0→…) + API-bróker kiértékelés (MT5/cTrader/IBKR; kulcs: a jelzés-adat = bróker-adat legyen). Ha az intervallum átfedi a fedezetit → további 50 trade. Ha a FELSŐ szél is alatta → NO-GO: leállás, post-mortem a naplókból, tőke épségben.
- **Eszköz-kapuk (eszközönként N≥10-nél):** ha egy eszköz Wilson-felső széle a fedezeti alatt VAGY a bróker-vs-rendszer divergenciája rendszeres → az eszköz kivétele a rosterből (assets lista) — az ezüst az első számú megfigyelt (mért gyanú: legalacsonyabb küszöb + legvadabb vol + legdrágább költség kombináció; állás 2026-07-17: ezüst 0/3, arany 2/2).

## 4. ELŐJEGYZETT ELEMZÉSEK (kritériumokkal — bármely jövőbeli asszisztens/Codex-session lefuttathatja a naplókból)

1. **Trigger-boncolás** (telemetria N≥50 armed-sor után): melyik részfeltétel bukik a leggyakrabban nem-tüzelő armed futásokon? Ha egyetlen feltétel felel a bukások >60%-áért ÉS a kihagyott jelöltek árnyék-kimenete pozitív várható értékű → az a feltétel kalibrálandó (egyetlen paraméter, egyszerre).
2. **Kiterjedtség-kapu** (N≥20 trade+árnyék): a belépéskor már megtett mozgás (ATR-ben) prediktálja-e a kimenetet? Ha a vesztesek mediánja ≥2× a nyerteseké → max-kiterjedtség kapu tervezhető.
3. **Fém-egyezés szűrő** (N≥20 fém-jelölt): az egyezés-szűrő a vesztesek ≥50%-át szűrné a nyertesek <25%-ának árán? → beépíthető. (Állás: 1 mellette, 1 ellene.)
4. **Korai exit / MAE-MFE** (N≥30): a vesztesek ≥60%-ánál van 0,5R-es korai ellenjel, ami a nyertesek <20%-ánál? → órán belüli fordulat-trigger tervezhető.
5. **Graze-szabály** (folyamatos): a TP/SL-érintések hány %-a "hajszál" (<0,05%)? Ha >20% → záróár-megerősítés megfontolandó érintés helyett.
6. **P-kalibráció** (N≥30): P-sávonkénti találati arány — ha monoton nő, a P árazható (méret-szorzó); ha nem, a küszöb az egyetlen érvényes használat.
7. **ADX-késés friss trendben:** a choppy-vétók hány %-a esik olyan órákra, ahol az 1h-ADX 20–25 közt van ÉS a nap iránya egyértelmű? Ha jelentős → a felmentési küszöb 25→22 kalibráció mérlegelhető.

## 5. HIBAELHÁRÍTÁS (tünet → teendő)

- **"CF-őr: heartbeat elavult" (hangos, napközben):** nézd a kártya "Open/pending positions" sorát. 0 pozíció → figyeld 10 percig, általában öngyógyul (a hét során minden elakadás magától helyreállt); ha 30+ percig áll, Actions → td-pipeline → utolsó futás hibája → a piros lépés naplójának utolsó sorai + képernyőkép a Codexnek/asszisztensnek. Nyitott pozíció mellett: a pozíciót KÉZZEL kezeled a brókernél (SL/TP be van állítva — a bróker véd), a rendszert utána javítod.
- **CI gate piros kártya:** a legutóbbi commit tört. A futás linkjén a piros lépés → jellemzően szintaktikai hiba, a naplóban a fájl:sor → Codexszel javíttatod ("Fix SyntaxError in <fájl> line <N>"). A pipeline addig a legutóbbi jó kódon fut tovább.
- **Kártya-anomália (fantom-gyanú):** a te módszered a hiteles: kártya-ár vs eToro-grafikon ugyanabban a percben; plusz `python scripts/verify_ledger_against_klines.py` — ha exit≠0, a kiírt sor mutatja a hibás trade-et és az okot.
- **Duplakártya / hiányzó záró kártya / P-küszöb alatti kártya:** mindhárom osztály javított és tesztelt; ha mégis előfordul, az regresszió → az adott napi delivery-log + gate-log sorai a bizonyíték, Codex-feladat.
- **Codex-sajátosságok:** a naplófájl-csatolás üresen érkezik az asszisztenshez → képernyőkép vagy szöveg-beillesztés; a Codex commit-üzenetei néha félrevezetők ("Hello→Goodbye") → a diff számít; nagy diff = nagy törési kockázat → nagy feladatot csak mechanikus, horgonyzott promptban.

## 6. A VÁLTOZTATÁS RENDJE (fagyasztási fegyelem)

**Azonnal javítható:** bizonyított hiba (a rendszer mást csinál, mint amit a saját szabályai mondanak) — bizonyíték: napló+gyertya. **Kalibrálható dokumentáltan:** mért külső adat (pl. bróker-költség) eltér a configtól. **Minden más** (küszöbök, új szűrők, új eszköz, runner-mód): CSAK a 3–4. fejezet N-kapuin és kritériumain át, egyszerre EGY változtatás, changelog-bejegyzéssel, és a változtatás utáni 2 hét új mérési ablak. A kísértés elleni mondat: *"N=1-ből szabályt írni a rendszerhalál receptje."*

## 7. KARBANTARTÁSI NAPTÁR

- **PAT-lejárat:** a GitHub tokened lejárati dátuma a naptáradban legyen; lejárat előtt új token → Secrets frissítés (GH_TOKEN a repóban ÉS a CF worker env-ben!).
- **Fagyasztás utáni nagy ablak (a GO-döntés után):** 1) dead-code gyomlálás a dormant-leltárból (BTC/EURUSD/NVDA), 2) analysis.py modularizálás lépésenként, 3) timestamp-helper egységesítés, 4) bud_ts elhagyása, 5) optimizer-hibataxonómia (ha valaha visszakapcsoljuk), 6) PR-flow + branch protection.
- **Heti:** vasárnapi rituálé. **Havi:** backup-workflow kézi próbafuttatás; a CF worker és a webhookok szúrópróbája (/heartbeat).

## 8. A RENDSZER ÁLLAPOTA E DOKUMENTUM ÍRÁSAKOR (2026-07-17)

Érvényes ledger: 4 trade — USOIL tp1_closed +$19,19 (TD-univerzum; bróker-oldalon vitatott — a divergencia-eset), XAGUSD stopped −$6,37, XAGUSD stopped −$5,67 (ÉLES, jegyzőkönyvezett), GOLD tp1_closed +$11,94 (mindkét univerzumban igazolt). Minden sor gyertya-pecsétes. Teszt-bázis: 441 passed / 5 skipped mindkét környezetben. Ismert nyitott kérdések: a 4. fejezet elemzései + a min_stoploss_pct konzisztencia-audit. A rendszer célfüggvénye változatlan: pontos, ritka, sapkázott kockázatú jelzések — és minden állítás gyertyáig visszavezethető bizonyítéka.

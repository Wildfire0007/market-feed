# Pipeline diagnosztika Discord-kártya

Ez a kártya a teljes trading→analysis pipeline időzítését és az utolsó futásból
származó artefaktokat sűríti össze. Minden mező arra szolgál, hogy gyorsan
eldöntsd: aktuális, friss és használható-e az elemzés a pozíciónyitáshoz.

## Mezők és jelentésük

- **run_id**: egyedi azonosító a futáshoz. A logokban és a tárolt
  `analysis_summary.json`/`pipeline_timing.json` fájlokban erre tudsz rákeresni
  a megfelelő artefaktokhoz.

- **Trading időtartam**: a `trading` szakasz indulása és befejezése közötti idő
  (mp vagy perc). Ha ez hirtelen megnyúlik, külső API-lassulásra vagy
  adatletöltési problémára gyanakodhatsz.

- **Trading→analysis késés**: mennyi idő telt el a trading szakasz vége és az
  analysis szakasz indulása között. Nagy érték esetén az elemzés már nem a
  legfrissebb tickeken futott, így a jel lehet elavult.

- **Analysis futásidő**: maga az elemző pipeline runtime-ja. Jelentős növekedés
  géperőforrás-szűk keresztmetszetre vagy túl nagy input batchre utalhat.

- **Utolsó analysis kora**: mennyi idő telt el az analysis befejezése óta. Ez
  mutatja az output frissességét; ha több tíz perc, a jel már gyengülhet.

- **Run start-capture eltérés**: mennyi idő telt el a teljes pipeline indulása
  és a diagnosztikai snapshot felvétele között. 0 mp körül van rendben; nagy
  eltérés esetén az idők „elmászhatnak”, így óvatosabban értelmezd a többi
  mezőt.

- **Artefakt-hash blokk**: a legfontosabb kimeneti fájlok (pl.
  `analysis_summary.json`, `status.json`, `pipeline_timing.json`) első 8
  karakteres SHA-256 hash-e és mérete. Ezzel gyorsan ellenőrizheted, melyik
  verziót láttad, és van-e hiányzó vagy sérült artefakt.

## Hogyan olvasd a kártyát pozíciónyitás előtt?

1. **Frissesség**: nézd meg az „Utolsó analysis kora” és a
   „Trading→analysis késés” értékeit. Ha bármelyik 10–15 percnél nagyobb, a jel
   könnyen elavult lehet, érdemes megerősítő forrást (order book/volumen)
   keresni, vagy megvárni a következő friss futást.

2. **Runtime anomáliák**: ha a „Trading időtartam” vagy az „Analysis futásidő”
   szokatlanul nagy (pl. a megszokott 20–60 mp helyett percek), gyanakodj
   erőforrás- vagy API-problémára. Ilyenkor konzervatívabb pozícióméretezés
   javasolt, mert a jel számítási útja nem volt „normál” terhelésen.

3. **Késés a szakaszok között**: magas „Trading→analysis késés” esetén a piaci
   állapot a futás alatt már változhatott. Ha az ár gyorsan mozog, csak
   megerősítéssel (pl. rövid távú trendfolytatás) nyiss pozíciót.

4. **Artefakt ellenőrzés**: ha hiányzik hash vagy méret (`hiányzik`), akkor a
   kimenet nem teljes. Ilyenkor ne indíts pozíciót az adott jel alapján, amíg a
   következő futás nem pótolja.

5. **Kombinált jelzés**: ha egyszerre magas a „Trading→analysis késés” és nagy
   az „Utolsó analysis kora”, tekintsd a jelzést **magas kockázatúnak**: a
   piac könnyen „kifuthatott” a számolt setupból. Csak kisméretű vagy paper
   trade-del próbálkozz, vagy várd meg a friss futást.

## Mit tekints normál tartománynak?

- **Trading időtartam** és **Analysis futásidő**: jellemzően 20–90 mp között.
- **Trading→analysis késés**: néhány másodperc; 1 perc felett már késői adatot
  jelenthet.
- **Utolsó analysis kora**: ideálisan <5–10 perc. 30+ perc már erősen kockázatos
  jel.
- **Run start-capture eltérés**: 0–5 mp; ha 30+ mp, akkor a többi értéket
  fenntartással kezeld.

Ha a fenti tartományokon belül vannak az értékek, a pipeline friss és stabil,
így a jel alapjául szolgáló elemzés megbízhatóbb a pozícionyitáshoz.

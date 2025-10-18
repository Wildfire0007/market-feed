# Intraday reports

- **latest/** – mindig a legutóbbi futás

## Strategy implementation notes

A stratégiák minden döntési logikája az `analysis.py`-ben található. A `Trading.py` kizárólag
az adatgyűjtést és az előzetes (EMA-alapú) jelzést végzi, így az új thresholdok és
momentum/likviditási módosítások életbe lépéséhez további módosítás a `Trading.py`-ben nem
szükséges. Amint a frissített `analysis.py` fut, automatikusan az új feltételek alapján készíti
el a `public/<ASSET>/signal.json` és a kapcsolódó riportokat.

## Makro hangulatfolyam (news_sentiment.json) frissítése

### Automatikus frissítés scripts/update_usdjpy_sentiment.py-vel

1. **API-kulcs beállítása.** Szerezz NewsAPI.org (vagy kompatibilis) kulcsot, majd add ki a
   `export NEWSAPI_KEY="<kulcs>"` parancsot. A script az `os.getenv`-en keresztül tölti be a
   titkot, így hiányzó kulcs esetén biztonságosan kilép.
2. **Script futtatása.** Példa parancs: `python scripts/update_usdjpy_sentiment.py \
   --query "USDJPY intervention" --country jp --expires-minutes 90`. A futás több oldalt is
   lekér a NewsAPI-tól, exponenciális visszavárással kezeli a rate-limit válaszokat, majd a
   kulcsszavas heurisztika alapján számolja ki a −1…1 tartományú sentiment pontszámot.
3. **Fájl frissítése.** A script automatikusan létrehozza/frissíti a
   `public/USDJPY/news_sentiment.json` állományt a következő mezőkkel:

   ```json
   {
     "score": 0.42,
     "bias": "usd_bullish",
     "headline": "MoF warns about rapid yen moves",
     "source_url": "https://...",
     "published_at": "2024-05-01T11:05:00+00:00",
     "expires_at": "2024-05-01T12:35:00Z"
   }
   ```

   Az `expires_at` mező automatikusan a megadott időtartammal (alapértelmezetten 90 perc)
   tolódik ki, így a `news_feed.py` nem dolgozik lejárt sentimenttel.
4. **Eredmény validálása.** A terminálon megjelenik az új score/bias páros; opcionálisan nézd meg
   a fájlt, hogy a headline releváns-e. A pipeline következő `analysis.py` futása már ezt használja.

### Manuális override

1. **Forrás beazonosítása.** Amint beérkezik egy releváns USDJPY makrohír (pl. japán pénzügyi
   minisztériumi kommunikáció, váratlan kamatdöntés vagy USD-érzékeny headline), az operátor
   eldönti, hogy a hír befolyásolja-e az intervenciós kockázatot. A döntéshez használhatók a
   belső hírfeedek, Bloomberg/Reuters jelzések vagy egy dedikált newsroom bot kivonatai.
2. **JSON fájl szerkesztése.** Navigálj a `public/USDJPY/` könyvtárba, és nyisd meg a
   `news_sentiment.json` állományt. Ha nem létezik, hozz létre egy új, UTF-8 kódolású fájlt.
3. **Értékek kitöltése.** A fájl szerkezete megegyezik a fenti példával; manuális frissítésnél is
   tartsd be a −1…1 tartományú `score` értéket és az ISO8601 formátumú `expires_at` mezőt.
4. **Mentés és verziózás.** Mentsd a fájlt (opcionálisan commitold). Az `analysis.py` a következő
   futáskor már a frissített sentimentet használja.
5. **Lejárat figyelése.** Ha a hír aktualitása megszűnt, frissítsd a `score` értékét 0-ra vagy
   töröld a fájlt; ellenkező esetben az `expires_at` elérése után a rendszer magától eldobja a
   jelzést.

## Pipeline futtatási sorrend

1. **`Trading.py`** – minden eszközre szekvenciálisan lekéri a spot és OHLC adatokat. A folyamat
   rugalmas rate-limitet és exponenciális visszavárást használ, így rövid hálózati hibák esetén
   is stabilan elkészülnek a `public/<ASSET>/` fájlok.
2. **`analysis.py`** – a Trading által előállított JSON-okra építve számolja a trend- és
   momentum-jelzéseket, alkalmazza a frissített RR/TP profilokat, valamint a real-time spot feed
   és késésgátak alapján dönti el, hogy érvényes-e a setup.
3. **Riportok / notifier scriptek** – igény szerint futtathatók (pl. `scripts/intraday_report.py`,
   `scripts/notify_discord.py`), amelyek a `public/` alatti legfrissebb `signal.json` állapotot
   használják fel. Ezek csak olvasnak, ezért a Trading → Analysis sorrend teljes lefutása után
   indítsuk őket.

A pipeline így garantálja, hogy az elemző modul mindig a legfrissebb, validált adatállapotot
dolgozza fel, és a későbbi riportolás már az elemzett kimenetekre támaszkodik.

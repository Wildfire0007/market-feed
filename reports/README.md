# Intraday reports

- **latest/** – mindig a legutóbbi futás

## Strategy implementation notes

A stratégiák minden döntési logikája az `analysis.py`-ben található. A `Trading.py` kizárólag
az adatgyűjtést és az előzetes (EMA-alapú) jelzést végzi, így az új thresholdok és
momentum/likviditási módosítások életbe lépéséhez további módosítás a `Trading.py`-ben nem
szükséges. Amint a frissített `analysis.py` fut, automatikusan az új feltételek alapján készíti
el a `public/<ASSET>/signal.json` és a kapcsolódó riportokat.

## Makro hangulatfolyam (news_sentiment.json) frissítése

### Automatikus frissítés scripts/update_btcusd_sentiment.py-vel

#### Beépített automatikus futtatás

Amint `analysis.py` BTCUSD eszközhöz fut, a `news_feed.load_sentiment()` automatikusan meghívja a
NewsAPI-alapú lekérdező modult. Ha a `NEWSAPI_KEY` környezeti változó be van állítva és a
`BTCUSD_SENTIMENT_AUTO` nincs kikapcsolva (`0`/`false`), a rendszer:

1. ellenőrzi, hogy a meglévő `news_sentiment.json` snapshot lejárt-e, vagy régebbi, mint a
   `BTCUSD_SENTIMENT_MIN_INTERVAL` (alapértelmezés: 600 s),
2. szükség esetén lekéri a legfrissebb címeket a NewsAPI-ról, kiszámolja a kulcsszavas
   sentiment-átlagot, majd frissíti a fájlt új `expires_at` időbélyeggel.

Így a pipeline futtatása önmagában elegendő a makro hangulatfolyam naprakészen tartásához – nincs
szükség kézi szerkesztésre, ha a kulcs és az internetkapcsolat rendelkezésre áll. A folyamat
hibatűrő: hálózati vagy API-hiba esetén a meglévő snapshot marad érvényben, az elemzés pedig a
legutóbbi érvényes sentimenttel dolgozik tovább.

#### Kézi script futtatása (opcionális)

1. **API-kulcs beállítása.** Szerezz NewsAPI.org (vagy kompatibilis) kulcsot, majd add ki a
   `export NEWSAPI_KEY="<kulcs>"` parancsot. A script az `os.getenv`-en keresztül tölti be a
   titkot, így hiányzó kulcs esetén biztonságosan kilép.
2. **Script futtatása.** Példa parancs: `python scripts/update_btcusd_sentiment.py \
   --query "bitcoin OR btcusd" --expires-minutes 60`. A futás több oldalt is
   lekér a NewsAPI-tól, exponenciális visszavárással kezeli a rate-limit válaszokat, majd a
   kulcsszavas heurisztika alapján számolja ki a −1…1 tartományú sentiment pontszámot.
3. **Fájl frissítése.** A script automatikusan létrehozza/frissíti a
   `public/BTCUSD/news_sentiment.json` állományt a következő mezőkkel:

   ```json
   {
     "score": 0.42,
     "bias": "btc_bullish",
     "headline": "ETF inflows accelerate bitcoin rally",
     "source_url": "https://...",
     "published_at": "2024-05-01T11:05:00+00:00",
     "expires_at": "2024-05-01T12:35:00Z"
   }
   ```

   Az `expires_at` mező automatikusan a megadott időtartammal (alapértelmezetten 90 perc)
   tolódik ki, így a `news_feed.py` nem dolgozik lejárt sentimenttel.
4. **Eredmény validálása.** A terminálon megjelenik az új score/bias páros; opcionálisan nézd meg
   a fájlt, hogy a headline releváns-e. A pipeline következő `analysis.py` futása már ezt használja.

#### Hogyan tedd automatává

- **Ütemezett futtatás.** A scriptet futtathatod cronból vagy bármelyik schedulerből (pl. systemd
  timer). Példa cron bejegyzés, ami 15 percenként frissít, ha elérhető új hír::

      */15 * * * * NEWSAPI_KEY=... /usr/bin/python /path/to/repo/scripts/update_BTCUSD_sentiment.py --expires-minutes 60 >> /var/log/BTCUSD_sentiment.log 2>&1

- **Hibafigyelés.** A script nem írja felül a meglévő fájlt, ha nincs találat; ilyenkor kilép
  hiba nélkül (exit code 3) és a korábbi sentiment marad érvényben egészen a lejáratig.
- **Kézi override kompatibilitás.** Ha az automata futás közben manuálisan kell módosítanod a
  sentimentet (például mert egy headline-at másképp értékelsz), egyszerűen szerkeszd a
  `news_sentiment.json` fájlt; a következő automatikus futás csak akkor írja felül, ha új
  releváns cikket talál.

### Manuális override

1. **Forrás beazonosítása.** Amint beérkezik egy releváns BTCUSD hír (pl. ETF-beáramlások,
   szabályozói lépések vagy hálózati esemény), az operátor eldönti, hogy a hír befolyásolja-e a
   rövid távú kockázati környezetet. A döntéshez használhatók a belső hírfeedek, Bloomberg/Reuters
   jelzések vagy kripto specifikus monitoring botok kivonatai.
2. **JSON fájl szerkesztése.** Navigálj a `public/BTCUSD/` könyvtárba, és nyisd meg a
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


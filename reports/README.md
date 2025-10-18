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

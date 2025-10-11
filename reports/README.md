# Intraday reports

- **latest/** – mindig a legutóbbi futás

## Strategy implementation notes

A stratégiák minden döntési logikája az `analysis.py`-ben található. A `Trading.py` kizárólag
az adatgyűjtést és az előzetes (EMA-alapú) jelzést végzi, így az új thresholdok és
momentum/likviditási módosítások életbe lépéséhez további módosítás a `Trading.py`-ben nem
szükséges. Amint a frissített `analysis.py` fut, automatikusan az új feltételek alapján készíti
el a `public/<ASSET>/signal.json` és a kapcsolódó riportokat.

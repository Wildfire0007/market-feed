# Belépési küszöb profilok áttekintése

Ez a jegyzet összefoglalja, hogyan lehet a P-score és ATR küszöböket
profilokkal paraméterezni, illetve hogyan ellenőrizhető a beállított profil.

## Konfiguráció

A profilokat a `config/analysis_settings.json` fájl `entry_threshold_profiles`
blokkjában tartjuk karban. Egy profil két térképet tartalmaz:

- `p_score_min`: az eszköz-specifikus minimum P-score érték.
- `atr_threshold_multiplier`: az ATR kapu szorzója.

Minden térkép tartalmaz egy `default` kulcsot, ami a nem felülírt eszközökre
vonatkozik. Az aktív profilt két módon lehet megadni:

1. `active_entry_threshold_profile` mező a konfigurációban.
2. `ENTRY_THRESHOLD_PROFILE` környezeti változó futásidőben.

## Gyors állapotlekérdezés

A `config.analysis_settings` modul a következő, magától értetődő segédeket
biztosítja:

```python
from config import analysis_settings as settings

settings.ENTRY_THRESHOLD_PROFILE_NAME  # Aktív profil neve
settings.list_entry_threshold_profiles()  # Elérhető profilok listája
settings.describe_entry_threshold_profile()  # Az aktív profil összes küszöbe
settings.describe_entry_threshold_profile("relaxed")  # Másik profil megtekintése
```

A `describe_entry_threshold_profile` egy könnyen olvasható szótárat ad vissza,
amelynek `p_score_min` és `atr_threshold_multiplier` szekciója tartalmazza a
`default`, az esetleges `overrides`, illetve a `by_asset` (összes eszköz) bontást.
Így nem kell a JSON-t kézzel bogarászni, és a profil könnyen megjeleníthető egy
riportban vagy debug logban is.

## Tesztelési lépések

1. **Baseline profil** – futtasd:
   ```bash
   pytest tests/test_entry_threshold_profiles.py::test_baseline_profile_configuration
   ```
   A teszt azt ellenőrzi, hogy a baseline profil 60-as P-score-t és 1.0-s ATR
   szorzót használ, és hogy a helper visszaadja a teljes profil leírást.

2. **Relaxed profil** – futtasd:
   ```bash
   pytest tests/test_entry_threshold_profiles.py::test_relaxed_profile_override
   ```
   A teszt a `ENTRY_THRESHOLD_PROFILE=relaxed` környezeti váltással 55-ös
   P-score-t és 0.9-es USOIL ATR szorzót vár.

3. **Manuális ellenőrzés** – állítsd be a kívánt profilt, futtasd az
   `analysis.py` pipeline-t, majd a generált `signal.json` fájlban a
   `entry_thresholds` mezőben ellenőrizheted, hogy melyik profil milyen
   küszöböket alkalmazott.

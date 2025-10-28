# ML valószínűségi score státusz

A rövid távú prioritások mellett továbbra sincs kapacitás a Gradient Boosting alapú
valószínűségi modellek újratanítására, ezért a pipeline-ban marad a manuális
lekapcsolás. Az `analysis.py` már induláskor letiltja az ML-alapú scoringot a
`ML_PROBABILITY_MANUAL_OVERRIDE` kapcsolóval, így a döntési logika a
heurisztikus `fallback` pontozást használja minden olyan eszközre, amelyhez nem
áll rendelkezésre tényleges modell. 【F:analysis.py†L121-L147】【F:ml_model.py†L39-L118】

## Fallback működése

A fallback becslés a `ml_model._fallback_probability` segédfüggvényre épül, ahol
az alap `p_score` értéket különböző súlyozott indikátorokkal módosítjuk. A
kimenet továbbra is korlátozottan 5–95% közé kerül, így elkerüljük a szélsőséges
értékeket. Ez a logika változatlanul lefut, még akkor is, ha a diszkre mentett
`<asset>_gbm.pkl` fájl hiányzik. 【F:ml_model.py†L118-L233】

## Logolás és zajszűrés

Ha a környezetben nincs energia a modellek pótlására, az
`SUPPRESS_ML_MODEL_WARNINGS=1` környezeti változóval ki lehet kapcsolni a
hiányzó- és placeholder-modellekre vonatkozó WARN szintű bejegyzéseket. A
troubleshooting blokk ettől függetlenül megjelenít minden releváns információt,
csak a pipeline log marad tisztább. 【F:analysis.py†L5749-L5792】

## Heti emlékeztető

A `reports/pipeline_monitor.record_ml_model_status` tartósítja a legutóbbi ML
modell státuszt, és legalább hetente egyszer visszajelzést küld, ha továbbra is
hiányoznak vagy placeholder státuszban vannak a fájlok. A periódus az
`ML_MODEL_REMINDER_DAYS` értékével paraméterezhető, alapértelmezetten 7 nap.
Amikor az emlékeztető esedékes, az analysis összefoglalóban külön sor jelzi a
teendőket. 【F:reports/pipeline_monitor.py†L15-L18】【F:reports/pipeline_monitor.py†L66-L109】【F:analysis.py†L5757-L5792】

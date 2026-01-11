# Ticket: Weekend & Holiday Dashboard Watchdog Reset

## Összefoglaló
A monitoring dashboard `status.json` és `monitoring/health.json` fájljai hétvégi vagy ünnepnapi leállások után tartósan hibás állapotban maradnak, amíg a kézi `reset_dashboard_state.py` scriptet nem futtatjuk. Ez a ticket automatizált watchdogot kér, amely a piacnyitási eseménykor visszaállítja a snapshotokat, hogy a felügyeleti felület ne mutasson fals "error" státuszt munkanap kezdetén. 【F:scripts/reset_dashboard_state.py†L28-L213】【F:public/status.json†L1-L13】【F:public/monitoring/health.json†L1-L66】

## Háttér
- A `reset_dashboard_state.py` parancs jelenleg ad-hoc módon fut, és manuális felügyeletet igényel a status, notify és anchor snapshotokra. Automatizmus hiányában a dashboard hétvégén felgyülemlett "market closed" hibákat továbbviszi hétfőn is. 【F:scripts/reset_dashboard_state.py†L28-L213】【F:tests/test_dashboard_reset.py†L22-L185】
- A `public/status.json` és `public/monitoring/health.json` állományok a legutóbbi reset időpontjával maradnak a fájlban, ezért piacnyitáskor is "error"/"market closed" jelzések látszanak, ami hamis riasztásokat okoz. 【F:public/status.json†L1-L13】【F:public/monitoring/health.json†L1-L66】

## Feladat
Implementálj egy időzített watchdog komponenst, amely:
1. Felismeri a hétvégi/ünnepnapi zárást követő legelső instrumentum-kereskedési időszak kezdetét (pl. exchange nyitási ablak vagy előre konfigurált cron marker alapján).
2. A nyitás pillanatában automatikusan futtatja a dashboard reset logikát, biztosítva, hogy a `status.json` és `health.json` (és opcionálisan a `_notify_state.json`/anchor tábla) frissek legyenek anélkül, hogy emberi beavatkozásra lenne szükség. 【F:scripts/reset_dashboard_state.py†L72-L213】
3. Naplózza és opcionálisan riasztja (pl. Slack/Discord log) a reset eredményét, hogy visszakövethető legyen, mikor történt az automatikus tisztítás.

## Javasolt megoldási irányok
- Új watchdog script a `scripts/` könyvtárban, amely a `reset_dashboard_state` modul függvényeit importálva futtatja a reset lépéseket, majd frissíti a `health.json`-t a reset után (üres vagy "standby" állapotba).
- Cron/systemd timer vagy meglévő scheduler integráció, amely a nyitási időpontot figyeli; ünnepnapokat konfigurációból (pl. `config/market_calendar.json`) lehet olvasni, ha rendelkezésre áll, különben fallback: hétvégi logika + manuális override lista.
- A reset után futtasson egy rövid egészség-ellenőrzést, hogy megerősítse a fájlok létezését és alapértelmezett mezőit, minimalizálva a félkész snapshotok kockázatát. 【F:tests/test_dashboard_reset.py†L112-L185】

## Kézbesítési kritériumok
- A watchdog automatikusan lefut a következő munkanap-piaczárás után, és igazoltan visszaállítja a `status.json` és `health.json` fájlokat alaphelyzetbe.
- Napló vagy riasztás készül a reset futásáról (siker/hiba), beleértve a backup path információt, ha készül mentés.
- Új tesztek fedik a watchdog időzítési logikáját (pl. szintetikus "piacnyitás" esemény szimulálása) és ellenőrzik, hogy a reset hívások a megfelelő fájlokra lefutnak.
- Dokumentáció frissül a scheduler beállításához (cron/systemd snippet vagy pipeline workflow update).

## Kockázatok és megjegyzések
- Figyelni kell az időzónákra: a piacnyitás UTC időkódjának egyeztetése a watchdog futtató host lokális idejével.
- Ünnepnapi naptár hiányában szükség lehet kézi override-ra vagy rugalmas fallbackre, hogy a watchdog ne reseteljen túl korán (pl. hosszú hétvégék).
- A reset futtatásakor ügyelni kell a backup könyvtár méretére (`public/monitoring/reset_backups`), érdemes rotációt vagy cleanup lépést hozzáadni. 【F:scripts/reset_dashboard_state.py†L59-L119】

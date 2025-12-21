# BTCUSD long jelzés hiánya — 2025-10-26 12:08 UTC

## Idősoros kontextus
- **5 perces gyertyák**: Az elmúlt 30 percben egymást követően magasabb záróárak születtek, a 12:00 UTC-s gyertya 113 437,78 USD-n zárt, miközben a teljes sáv 113 284–113 495 USD között mozgott. A 11:35–11:55 közötti gyertyák mind 112 792–113 348 USD tartományban zártak, ami meredek rövid távú emelkedést jelez és a lokális szerkezetet a napi csúcshoz tolta.【99a8b0†L1-L20】
- **1 órás és 4 órás idősík**: Az utolsó két 1 órás gyertya 112 500→113 303→113 437 USD záróárakkal egymást erősítve új csúcsra futott, miközben a 4 órás idősíkon még csak a korábbi 111 750→113 303→113 437 USD sávon belüli kitörési kísérlet zajlik, vagyis a magasabb idősík trendje még nem fordult egyértelműen longba.【99a8b0†L20-L27】
- **Napon belüli helyzet**: A rendszer intraday profilja 0,97-es range-pozíciót, 1,5× feletti ATR-expanziót és kimerült long oldali range-et jelez, ezért a long belépés üldözésnek minősülne a napi tető közelében.【ea6d94†L370-L400】

## Miért hiányzik a long belépő?
Az automata kapurendszer több kötelező feltétel teljesülését várja el, amelyek közül több is hiányzik:

| Kapu | Hiány oka |
| --- | --- |
| Regime | A "regime_ok" jelző hamis, mert a magasabb idősík (4h) csak neutrális sávban mozog, ezért a trendfilter nem engedi a long irányt.【7290e9†L528-L535】 |
| Bias | Az "effective_bias" értéke neutrális, mert a 4 órás bias továbbra is "neutral", így nincs elegendő többszintű megerősítés egy long setuphoz.【0e194c†L41-L45】【7290e9†L528-L535】 |
| BOS 5m | Bár a struktúra "bos_up", a kapu azért marad zárva, mert a rendszer retestre és likviditásra vár; a szerkezeti flip jelző ugyan aktiválódott, de a belépési kritériumhoz hiányzik a likviditási igazolás.【0e194c†L63-L68】【7290e9†L514-L533】 |
| Likviditás (Fibonacci zóna) | A napi tartomány felső 3%-ában lévő árfolyam nem szolgáltat vonzó RR-t és nincs megerősített likviditási támasz, ezért a likviditási kapu zárolva marad.【ea6d94†L370-L399】【0e194c†L63-L68】 |
| ATR | Az 1 órás ATR ≈ 372 USD, miközben az effektív küszöb 40 USD – a volatilitás túl nagy, ezért a rendszer kiszűri a belépőt, nehogy széles stopra kényszerüljön.【ea6d94†L358-L409】【7290e9†L521-L536】 |
| P_score ≥ 60 | A pontszám mindössze 6, így az algoritmus a valószínűségi kaput sem nyitja meg.【7290e9†L536-L541】 |

A fenti hiányok egyenként is tiltók, együtt pedig teljesen lezárják a long oldal megjátszását, ráadásul a diagnosztika hard exitet írt elő az aktív short pozícióra is, így a fókusz a kockázatcsökkentésen van.【0e194c†L188-L197】

## Miért csak P = 5%?
A futtatáskor a scikit-learn csomag hiánya miatt a gradient boosting modell nem tudott betöltődni, így a rendszer automatikus fallback logit pontozásra váltott (``reason = "sklearn_missing"``).【F:public/BTCUSD/signal.json†L261-L309】【F:public/analysis_summary.json†L275-L340】 A kiinduló 6%-os alapvalószínűséget több negatív korrekció is csökkentette:

- A bias semlegessége -0,55 deltat jelentett, mert nincs konzisztens trend (bias_neutral_penalty).【ac41d6†L205-L223】
- A momentum/volatilitás arány -0,35 ponttal rontott, mert a spike-szerű rally nem támogatott struktúra (momentum_vol_ratio).【ac41d6†L218-L223】
- A precision és trigger komponensek további -0,8435 levonást eredményeztek, mert a rendszer múltbeli teljesítménye ebben a setupban gyenge.【ac41d6†L237-L247】
- A magas volatilitás (rel_atr) ugyan pozitív 0,21 korrekciót adott, de ez nem tudta ellensúlyozni a negatív súlyokat.【ac41d6†L213-L217】

Az eredmény egy -3,53-as logit és 2,8%-os kalibrált valószínűség lett, amit a rendszer lefelé kerekít 5%-ra a felhasználói felületen, messze a 60%-os küszöb alatt.【ac41d6†L200-L276】【7290e9†L536-L541】

### Miért panaszkodik a dashboard a „hiányzó ML modellekre”?
A pipeline összegző nézete most már külön kiírja, ha nem a modellek hiányoznak, hanem a futtatási függőségek: a `troubleshooting` blokkban megjelenik az „ML függőségek hiányoznak: joblib, scikit-learn …” üzenet, a `ml_runtime_issues` mező pedig részletes telepítési útmutatót tartalmaz a scikit-learn és a joblib csomagokra.【F:analysis.py†L5410-L5428】【F:public/analysis_summary.json†L3113-L3126】 A scikit-learn + joblib páros feltelepítése (vagy konténerben történő szállítása) után a rendszer automatikusan visszatér a gradient boosting alapú pontozáshoz.

## Következtetés
A jelenlegi környezetben a piac ugyan felfelé száguld, de ezt szélsőséges volatilitás és kimerült intraday range kíséri. A trendfilter és a bias sem támogatja a longot, a likviditási és ATR kapuk zárva vannak, így a rendszerünk helyesen nem kínált long belépőt, hanem a meglévő short pozíció azonnali lezárását követelte meg a strukturális invalidáció miatt.【0e194c†L188-L197】【ea6d94†L370-L399】

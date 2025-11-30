# Entry gate Discord kártya – pontos értelmezés

Ez a segéd a Discordon megjelenő „Entry gate toplista (24h)” kártya számait
magyarázza el, különösen az **ATR**, **regime** és **session** jelzőket.
A jelölések 30 napos alapvonalhoz képest mutatják az aktuális állapotot, így
minden érték relatív (sokszor „4x”, „5x” formában jelenik meg).

## Metodika röviden
- **ATR szorzó**: a 24 órás átlagos true range (ATR) osztva a 30 napos medián
  ATR-rel. Így egy „5x” felirat azt jelenti, hogy az elmúlt 24 óra
  volatilitása az elmúlt hónap tipikus értékének ötszöröse.
- **Regime szint**: a kockázati rezsimet az 1h EMA21 lejtőjének abszolút értéke
  és a volatilitás-overlay alapján pontozzuk. Ha az EMA21 meredekebb, mint a
  profilban engedett küszöb (alap: 0.08–0.10%/óra), akkor „elevated/extrém”
  rezsimet jelzünk, és a kártyán ez szorzóként jelenik meg.
- **Session aktivitás**: az adott kereskedési idősáv (session) forgalma és
  range-e az elmúlt 30 nap ugyanezen idősávjához képest. A „4x” azt jelzi,
  hogy a mostani session négyszer akkora range-et vagy forgalmat hoz, mint a
  megszokott medián.

## Konkrét sávok
Az alábbi táblázat mutatja, hogy mikortól tekintjük „magasnak” vagy „extrémnek”
az értékeket. A Discord-kártyán látott szorzók közvetlenül ezekből a sávokból
származnak.

| Mutató              | Normál                     | Emelkedett (high)        | Extrém                 |
|---------------------|----------------------------|--------------------------|------------------------|
| **ATR szorzó**      | < 2.0×                     | 2.0–4.9×                 | >= 5.0×                |
| **Regime szorzó**   | < 1.5× (küszöb alatt)      | 1.5–2.9× (küszöb felett) | >= 3.0× (erősen tiltó) |
| **Session szorzó**  | < 1.5×                     | 1.5–3.9×                 | >= 4.0×                |

## Mit jelent ez a gyakorlatban?
- **Magas ATR (2–4.9×)**: a piac tágabban mozog. Javasolt kisebb méretezés és
  konzervatívabb belépési szintek.
- **Extrém ATR (>=5×)**: a jel könnyen túlkorrekció vagy whipsaw lehet. Csak
  megerősítéssel vagy csökkentett kitettséggel érdemes lépni.
- **Emelkedett/Extrém regime (>=1.5× / >=3×)**: az 1h EMA21 túl meredek a
  profilhoz képest; a stratégia stop- és target-paramétereit érdemes
  feszesebbre venni, vagy a jelzést kihagyni.
- **Magas session (>=1.5×)**: gyors végrehajtás, csúszás (slippage) és
  kiszélesedő spread kockázata. 4× felett a jel jellemzően rövid élettartamú
  momentumra utal.

## Gyors „mikor aggódjak?” ellenőrzés
- **ATR >=5× vagy regime >=3×**: kezeljük extrémként → csak kis méret vagy
  kihagyás.
- **Session >=4×**: momentum gyorsan kifuthat, érdemes real-time likviditást és
  order book-ot nézni a végrehajtás előtt.
- **Több mutató egyszerre magas**: ha ATR és regime is extrém, a jel nagy
  valószínűséggel túl kockázatos; akkor is csökkentsünk méretet, ha a toplista
  egyébként kedvező.

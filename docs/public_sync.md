# Public artefakt szinkronizálás

A `Trading.py` futása után automatikusan lefut egy szinkronizáló lépés a
CI pipeline-ban, ami összegyűjti a friss `data/` és `reports/` tartalmakat,
majd frissíti a `public` könyvtárat. A szinkronról metaadat készül
(`public/.sync-metadata.json`, `public/.last_sync`), amely tartalmazza a
GitHub run ID-t, a commit hash-t és egy checksumot. Ezeket a workflow
ellenőrzései használják, és hiba esetén a job failre fut.

## Futtatás
- Automatikusan fut minden `Trading.py` lépés után a `td-pipeline.yml` és
  a `predeploy-verification.yml` workflow-ban.
- Kézi futtatáskor (`workflow_dispatch`) a `force_public_sync` inputtal
  szabályozható, hogy lefusson-e a szinkron, alapértelmezetten "true".
- Hibaelhárításkor a `scripts/update_public.py` lokálisan is futtatható:

```bash
python scripts/update_public.py --target public --sources data reports --run-id debug-run
```

## Deploy
A `td-pipeline.yml` új `deploy_public` jobja a szinkronizált `public`
tartalmat GitHub Pages artefaktként tölti fel és deployolja. A Discord
értesítő csak a sikeres deploy után fut le.

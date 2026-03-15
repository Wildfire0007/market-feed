#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "python3 nem található." >&2
  exit 1
fi

if ! command -v crontab >/dev/null 2>&1; then
  echo "crontab parancs nem található. Telepítsd a cron/csomagot, majd futtasd újra." >&2
  exit 1
fi

if [[ -z "${DISCORD_WEBHOOK_URL:-}" ]]; then
  echo "DISCORD_WEBHOOK_URL nincs beállítva. Exportáld (vagy add hozzá a crontab környezethez), majd futtasd újra." >&2
  exit 1
fi

if ! compgen -G "${REPO_DIR}/public/*/signal.json" >/dev/null; then
  echo "Nem található frissülő public/<ASSET>/signal.json input. Generáld a jeleket, majd futtasd újra." >&2
  exit 1
fi

CRON_BLOCK=$(cat <<CRON
# === market-feed begin ===
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PYTHON_BIN=${PYTHON_BIN}
DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}

*/5 * * * * cd ${REPO_DIR} && ./scripts/run_signal_cycle.sh
* * * * * cd ${REPO_DIR} && flock -n ${REPO_DIR}/public/.position_lifecycle_cron.lock ${PYTHON_BIN} scripts/position_lifecycle.py >> ${REPO_DIR}/public/_cron_position_lifecycle.log 2>&1
* * * * * cd ${REPO_DIR} && flock -n ${REPO_DIR}/public/.notify_mgmt_cron.lock ${PYTHON_BIN} scripts/notify_management_discord.py >> ${REPO_DIR}/public/_cron_notify_management.log 2>&1
# === market-feed end ===
CRON
)

TMP_FILE=$(mktemp)
(crontab -l 2>/dev/null || true) \
  | sed '/# === market-feed begin ===/,/# === market-feed end ===/d' \
  > "${TMP_FILE}"
printf '\n%s\n' "${CRON_BLOCK}" >> "${TMP_FILE}"
crontab "${TMP_FILE}"
rm -f "${TMP_FILE}"

echo "Cron beállítva a market-feed ütemezéshez."
echo "Log fájlok: ${REPO_DIR}/public/_cron_position_lifecycle.log és ${REPO_DIR}/public/_cron_notify_management.log"

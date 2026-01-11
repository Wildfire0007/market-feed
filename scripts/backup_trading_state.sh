#!/usr/bin/env bash
set -euo pipefail

STATE_DB_PATH="${STATE_DB_PATH:-trading_state.db}"
BACKUP_DIR="${BACKUP_DIR:-backups}"

if [[ ! -f "${STATE_DB_PATH}" ]]; then
  echo "State DB not found at ${STATE_DB_PATH}" >&2
  exit 1
fi

mkdir -p "${BACKUP_DIR}"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_file="${BACKUP_DIR}/trading_state_${stamp}.db"

cp "${STATE_DB_PATH}" "${backup_file}"

echo "Backup created at ${backup_file}"

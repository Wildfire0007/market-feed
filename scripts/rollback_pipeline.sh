#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THRESHOLD_FILE="${REPO_ROOT}/config/analysis_settings.json"
THRESHOLD_BACKUP="${REPO_ROOT}/config/analysis_settings.backup.json"
LOCKFILE="${REPO_ROOT}/requirements.lock"
LOCK_BACKUP="${REPO_ROOT}/requirements.lock.backup"

log() {
  printf '[rollback] %s\n' "$1"
}

restore_file() {
  local target="$1"
  local backup="$2"
  if [[ -f "${backup}" ]]; then
    cp "${backup}" "${target}"
  else
    git -C "${REPO_ROOT}" checkout -- "${target}"
  fi
}

log "Resetting entry thresholds to repository defaults"
restore_file "${THRESHOLD_FILE}" "${THRESHOLD_BACKUP}"

log "Restoring Python lockfile"
restore_file "${LOCKFILE}" "${LOCK_BACKUP}"

log "Purging generated public artefacts"
rm -rf "${REPO_ROOT}/public"
mkdir -p "${REPO_ROOT}/public"

log "Rollback steps finished. Pipeline can be re-run with clean thresholds."

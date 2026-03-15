#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${REPO_DIR}/public/.signal_cycle.lock"
LOG_FILE="${REPO_DIR}/public/_cron_signal_cycle.log"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ANALYSIS_TIMEOUT="${ANALYSIS_TIMEOUT:-240}"
NOTIFY_TIMEOUT="${NOTIFY_TIMEOUT:-60}"

mkdir -p "${REPO_DIR}/public"

{
  flock -n 9 || exit 0
  cd "${REPO_DIR}"
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] cycle start"
  timeout "${ANALYSIS_TIMEOUT}" "${PYTHON_BIN}" analysis.py
  timeout "${NOTIFY_TIMEOUT}" "${PYTHON_BIN}" scripts/notify_discord.py
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] cycle ok"
} 9>"${LOCK_FILE}" >>"${LOG_FILE}" 2>&1

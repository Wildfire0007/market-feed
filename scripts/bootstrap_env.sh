#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_VERSION_FILE="${REPO_ROOT}/.python-version"
if [[ -f "${PYTHON_VERSION_FILE}" ]]; then
  PY_VERSION="$(<"${PYTHON_VERSION_FILE}")"
else
  PY_VERSION=""
fi
VENV_PATH="${REPO_ROOT}/.venv"
activate_venv() {
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
}
if command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] using uv for environment management" >&2
  if [[ ! -d "${VENV_PATH}" ]]; then
    if [[ -n "${PY_VERSION}" ]]; then
      uv venv "${VENV_PATH}" --python "${PY_VERSION}"
    else
      uv venv "${VENV_PATH}"
    fi
  fi
  activate_venv
  if [[ -f "${REPO_ROOT}/requirements.lock" ]]; then
    uv pip sync "${REPO_ROOT}/requirements.lock"
  else
    uv pip install -r "${REPO_ROOT}/requirements.txt"
  fi
elif command -v pipenv >/dev/null 2>&1; then
  echo "[bootstrap] using pipenv fallback" >&2
  if [[ -n "${PY_VERSION}" ]]; then
    PIPENV_PYTHON="${PY_VERSION}" pipenv --python "${PY_VERSION}" >/dev/null
  else
    default_minor="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    pipenv --python "${default_minor}" >/dev/null
  fi
  if [[ -f "${REPO_ROOT}/requirements.lock" ]]; then
    pipenv run pip install -r "${REPO_ROOT}/requirements.lock"
  else
    pipenv run pip install -r "${REPO_ROOT}/requirements.txt"
  fi
else
  echo "[bootstrap] using python -m venv fallback" >&2
  python_bin="python"
  if [[ -n "${PY_VERSION}" ]]; then
    if command -v "python${PY_VERSION%.*}" >/dev/null 2>&1; then
      python_bin="python${PY_VERSION%.*}"
    fi
  fi
  if [[ ! -d "${VENV_PATH}" ]]; then
    "${python_bin}" -m venv "${VENV_PATH}"
  fi
  activate_venv
  python -m pip install --upgrade pip
  if [[ -f "${REPO_ROOT}/requirements.lock" ]]; then
    pip install -r "${REPO_ROOT}/requirements.lock"
  else
    pip install -r "${REPO_ROOT}/requirements.txt"
  fi
fi

#!/usr/bin/env python3
"""
Download all `entry-gate-stats.zip` artifacts from GitHub Actions runs for the
"TD Full Pipeline (5m)" workflow on the main branch using the GitHub API only.
Extracts the contained `entry_gate_stats.json` files and merges them into a
single summary JSON.

Requirements:
- requests
- tqdm
- zipfile (stdlib)
- json (stdlib)

Assumptions:
- GitHub token is stored in env var GITHUB_TOKEN.
- You are a collaborator on the repo and can access artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from zipfile import ZipFile

import requests
from tqdm import tqdm

from github_utils import build_api_headers, detect_repo, load_github_token

API_ROOT = "https://api.github.com"
WORKFLOW_NAME = "TD Full Pipeline (5m)"
BRANCH = "main"
ARTIFACT_NAME = "entry-gate-stats"
ARTIFACT_FILE_SUFFIX = ".zip"
ARTIFACT_CONTENT = "entry_gate_stats.json"
DOWNLOAD_DIR = Path("artifacts")
SUMMARY_PATH = Path("entry_gate_stats_summary.json")


def _api_get(url: str, token: str, params: Optional[dict] = None) -> requests.Response:
    headers = build_api_headers(token)
    headers["X-GitHub-Api-Version"] = "2022-11-28"
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def get_workflow_id(owner: str, repo: str, token: str) -> int:
    url = f"{API_ROOT}/repos/{owner}/{repo}/actions/workflows"
    resp = _api_get(url, token)
    for workflow in resp.json().get("workflows", []):
        if workflow.get("name") == WORKFLOW_NAME:
            return workflow["id"]
    raise SystemExit(f"Workflow '{WORKFLOW_NAME}' not found in repository {owner}/{repo}.")


def list_runs(owner: str, repo: str, workflow_id: int, token: str) -> List[dict]:
    params = {"branch": BRANCH, "per_page": 100}
    url = f"{API_ROOT}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
    runs: List[dict] = []
    while url:
        resp = _api_get(url, token, params=params)
        payload = resp.json()
        runs.extend(payload.get("workflow_runs", []))
        url = payload.get("next") or resp.links.get("next", {}).get("url")
        params = None  # Only needed for first request.
    return runs


def _format_download_path(run_id: int) -> Path:
    return DOWNLOAD_DIR / f"{run_id}-{ARTIFACT_NAME}{ARTIFACT_FILE_SUFFIX}"


def list_artifacts(owner: str, repo: str, run_id: int, token: str) -> List[dict]:
    url = f"{API_ROOT}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    artifacts: List[dict] = []
    params: Optional[dict] = {"per_page": 100}
    while url:
        resp = _api_get(url, token, params=params)
        payload = resp.json()
        artifacts.extend(payload.get("artifacts", []))
        url = payload.get("next") or resp.links.get("next", {}).get("url")
        params = None
    return artifacts


def find_artifact(artifacts: List[dict], name: str) -> Optional[dict]:
    def _normalize(value: str) -> str:
        return value.lower().replace("_", "-").replace(" ", "-")

    target = _normalize(name)

    # Exact normalized match
    for artifact in artifacts:
        if _normalize(artifact.get("name", "")) == target:
            return artifact

    # Fallback: substring match to catch suffixes (e.g., entry-gate-stats-public)
    for artifact in artifacts:
        if target in _normalize(artifact.get("name", "")):
            return artifact

    return None


def download_artifact_archive(artifact: dict, token: str, destination: Path) -> Optional[Path]:
    if destination.exists():
        return destination

    download_url = artifact.get("archive_download_url")
    if not download_url:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/octet-stream",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with requests.get(download_url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as fp:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fp.write(chunk)
    return destination


def extract_entry_gate_stats(zip_paths: Iterable[Path]) -> Dict[str, dict]:
    extracted: Dict[str, dict] = {}
    for path in zip_paths:
        if not path or not path.exists():
            continue
        run_id = path.stem.split("-")[0]
        with ZipFile(path) as archive:
            if ARTIFACT_CONTENT not in archive.namelist():
                continue
            with archive.open(ARTIFACT_CONTENT) as file_obj:
                extracted[run_id] = json.load(file_obj)
    return extracted


def write_summary(data: Dict[str, dict], runs: List[dict]) -> None:
    run_lookup = {str(run["id"]): run for run in runs}
    summary = []
    for run_id, payload in data.items():
        run_info = run_lookup.get(run_id, {})
        summary.append(
            {
                "run_id": run_id,
                "created_at": run_info.get("created_at"),
                "updated_at": run_info.get("updated_at"),
                "html_url": run_info.get("html_url"),
                "data": payload,
            }
        )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def main():
    token = load_github_token()
    owner, repo = detect_repo()
    workflow_id = get_workflow_id(owner, repo, token)
    runs = list_runs(owner, repo, workflow_id, token)
    if not runs:
        raise SystemExit("No workflow runs found.")

    downloaded: List[Path] = []
    missing_artifacts: List[dict] = []

    for run in tqdm(runs, desc="Downloading artifacts"):
        artifacts = list_artifacts(owner, repo, run["id"], token)
        artifact = find_artifact(artifacts, ARTIFACT_NAME)
        if not artifact:
            missing_artifacts.append(
                {
                    "run_id": run.get("id"),
                    "run_number": run.get("run_number"),
                    "branch": run.get("head_branch"),
                    "created_at": run.get("created_at"),
                    "available_artifacts": [a.get("name") for a in artifacts],
                }
            )
            continue
        path = download_artifact_archive(artifact, token, _format_download_path(run["id"]))
        if path:
            downloaded.append(path)

    extracted = extract_entry_gate_stats(downloaded)
    write_summary(extracted, runs)
    if missing_artifacts:
        diagnostics_path = Path("missing_entry_gate_stats.json")
        diagnostics_path.write_text(json.dumps(missing_artifacts, indent=2))
        print(
            "Saved diagnostics for runs without matching artifacts to"
            f" {diagnostics_path}"
        )
    print(f"Extracted {len(extracted)} JSON files into {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

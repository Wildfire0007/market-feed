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
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from zipfile import ZipFile

import requests
from tqdm import tqdm

API_ROOT = "https://api.github.com"
WORKFLOW_NAME = "TD Full Pipeline (5m)"
BRANCH = "main"
ARTIFACT_NAME = "entry-gate-stats"
ARTIFACT_FILE_SUFFIX = ".zip"
ARTIFACT_CONTENT = "entry_gate_stats.json"
DOWNLOAD_DIR = Path("artifacts")
SUMMARY_PATH = Path("entry_gate_stats_summary.json")


def _require_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN environment variable is required.")
    return token


def _detect_repo() -> tuple[str, str]:
    try:
        url = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                text=True,
            )
            .strip()
            .rstrip(".git")
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit("Unable to detect Git repository origin.") from exc

    if url.startswith("git@"):
        path = url.split(":", 1)[-1]
    elif url.startswith("https://") or url.startswith("http://"):
        path = url.split("github.com/")[-1]
    else:
        raise SystemExit(f"Unrecognized remote URL format: {url}")

    owner, repo = path.split("/", 1)
    return owner, repo


def _api_get(url: str, token: str, params: Optional[dict] = None) -> requests.Response:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
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
    name_lower = name.lower()
    for artifact in artifacts:
        if artifact.get("name", "").lower() == name_lower:
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
    token = _require_token()
    owner, repo = _detect_repo()
    workflow_id = get_workflow_id(owner, repo, token)
    runs = list_runs(owner, repo, workflow_id, token)
    if not runs:
        raise SystemExit("No workflow runs found.")

    downloaded: List[Path] = []
    for run in tqdm(runs, desc="Downloading artifacts"):
        artifacts = list_artifacts(owner, repo, run["id"], token)
        artifact = find_artifact(artifacts, ARTIFACT_NAME)
        if not artifact:
            continue
        path = download_artifact_archive(artifact, token, _format_download_path(run["id"]))
        if path:
            downloaded.append(path)

    extracted = extract_entry_gate_stats(downloaded)
    write_summary(extracted, runs)
    print(f"Extracted {len(extracted)} JSON files into {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

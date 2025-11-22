#!/usr/bin/env python3
"""Collect entry gate stats artifacts from GitHub Actions runs.

This script queries the GitHub REST API for runs of a target workflow, downloads
artifacts that match the entry gate stats naming convention, extracts the JSON
payloads, and aggregates them into a single summary JSON file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

GITHUB_API = "https://api.github.com"
DEFAULT_SINCE_DATE = "2025-11-14"
DEFAULT_WORKFLOW_NAME = "TD Full Pipeline (5m)"
DEFAULT_OWNER = "Wildfire0007"
DEFAULT_REPO = "market-feed"
DEFAULT_BRANCH = "main"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="Repository owner")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository name")
    parser.add_argument(
        "--workflow-name", default=DEFAULT_WORKFLOW_NAME, help="Workflow name"
    )
    parser.add_argument(
        "--since-date",
        default=DEFAULT_SINCE_DATE,
        help="ISO date (YYYY-MM-DD) interpreted as UTC midnight",
    )
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Branch name")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output path. Defaults to public/debug/entry_gate_stats_aggregate_since_<DATE>.json"
        ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=200,
        help="Maximum number of workflow runs to consider",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def load_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
    if not token:
        sys.exit(
            "GitHub token is required. Set the GITHUB_TOKEN or GITHUB_PAT environment variable."
        )
    return token


def parse_utc_date(date_str: str) -> dt.datetime:
    try:
        base_date = dt.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"Invalid --since-date value '{date_str}': {exc}") from exc
    return base_date.replace(tzinfo=dt.timezone.utc)


def parse_github_timestamp(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc
    )


def github_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    response = session.get(url, timeout=30, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub API request failed ({response.status_code}) for {url}: {response.text}"
        )
    return response


def find_workflow_id(
    session: requests.Session, owner: str, repo: str, workflow_name: str
) -> int:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows"
    workflows = github_get(session, url).json().get("workflows", [])
    for wf in workflows:
        if wf.get("name") == workflow_name:
            workflow_id = wf.get("id")
            if workflow_id is None:
                break
            logger.info("Found workflow '%s' with id %s", workflow_name, workflow_id)
            return workflow_id
    raise SystemExit(f"Workflow named '{workflow_name}' not found in {owner}/{repo}.")


def iter_workflow_runs(
    session: requests.Session,
    owner: str,
    repo: str,
    workflow_id: int,
    branch: str,
    since: dt.datetime,
    max_runs: int,
) -> Iterable[Dict]:
    page = 1
    collected = 0
    while collected < max_runs:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
        params = {
            "branch": branch,
            "status": "completed",
            "per_page": 100,
            "page": page,
        }
        response = github_get(session, url, params=params)
        payload = response.json()
        runs = payload.get("workflow_runs", [])
        if not runs:
            break
        for run in runs:
            created_at = parse_github_timestamp(run["created_at"])
            if created_at < since:
                logger.info("Encountered run older than since-date; stopping pagination.")
                return
            yield run
            collected += 1
            if collected >= max_runs:
                logger.info("Reached max_runs limit (%s).", max_runs)
                return
        page += 1


def find_matching_artifacts(artifacts: List[Dict]) -> List[Dict]:
    matches = []
    for artifact in artifacts:
        name = (artifact.get("name") or "").lower()
        if (
            name == "entry-gate-stats"
            or "entry-gate-stats" in name
            or "entry_gate_stats" in name
        ):
            matches.append(artifact)
    return matches


def list_run_artifacts(
    session: requests.Session, owner: str, repo: str, run_id: int
) -> List[Dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    response = github_get(session, url)
    return response.json().get("artifacts", [])


def download_artifact(session: requests.Session, artifact: Dict, dest_dir: Path) -> Path:
    download_url = artifact.get("archive_download_url")
    if not download_url:
        raise RuntimeError(f"Artifact {artifact.get('name')} missing download URL")
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = dest_dir / f"artifact_{artifact['id']}.zip"
    headers = {"Accept": "application/zip"}
    with session.get(download_url, headers=headers, timeout=60, stream=True) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Failed to download artifact {artifact.get('name')} ({resp.status_code}): {resp.text}"
            )
        with tmp_zip.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    return tmp_zip


def select_json_file(extracted_dir: Path) -> Optional[Path]:
    json_files = sorted(path for path in extracted_dir.rglob("*.json") if path.is_file())
    if not json_files:
        return None
    if len(json_files) == 1:
        return json_files[0]
    for candidate in json_files:
        if candidate.name.lower() == "entry_gate_stats.json":
            return candidate
    return json_files[0]


def extract_stats_from_artifact(zip_path: Path) -> Optional[Dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_path)
        except zipfile.BadZipFile:
            logger.warning("Skipping artifact %s due to invalid ZIP format", zip_path)
            return None
        json_file = select_json_file(tmp_path)
        if not json_file:
            logger.warning("No JSON files found in artifact %s", zip_path)
            return None
        try:
            with json_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse JSON file %s: %s", json_file, exc)
            return None
        return {"data": data, "file_name": json_file.name}


def aggregate_runs(
    session: requests.Session,
    owner: str,
    repo: str,
    workflow_id: int,
    branch: str,
    since: dt.datetime,
    max_runs: int,
) -> List[Dict]:
    aggregated: List[Dict] = []
    for run in iter_workflow_runs(session, owner, repo, workflow_id, branch, since, max_runs):
        run_id = run.get("id")
        if run_id is None:
            logger.warning("Skipping run with missing id: %s", run)
            continue
        artifacts = list_run_artifacts(session, owner, repo, run_id)
        matches = find_matching_artifacts(artifacts)
        if not matches:
            logger.info("No entry gate stats artifacts for run %s", run_id)
            continue
        for artifact in matches:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                try:
                    zip_path = download_artifact(session, artifact, tmp_path)
                except RuntimeError as exc:
                    logger.warning("%s", exc)
                    continue
                stats_payload = extract_stats_from_artifact(zip_path)
                if not stats_payload:
                    continue
                aggregated.append(
                    {
                        "run_id": run.get("id"),
                        "run_number": run.get("run_number"),
                        "status": run.get("status"),
                        "conclusion": run.get("conclusion"),
                        "created_at": run.get("created_at"),
                        "updated_at": run.get("updated_at"),
                        "html_url": run.get("html_url"),
                        "artifact_name": artifact.get("name"),
                        "artifact_id": artifact.get("id"),
                        "artifact_size_in_bytes": artifact.get("size_in_bytes"),
                        "stats_file": stats_payload["file_name"],
                        "stats": stats_payload["data"],
                    }
                )
    return aggregated


def build_output_path(output: Optional[Path], since: dt.datetime) -> Path:
    date_str = since.strftime("%Y-%m-%d")
    if output:
        return output
    return Path(f"public/debug/entry_gate_stats_aggregate_since_{date_str}.json")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    token = load_token()
    since = parse_utc_date(args.since_date)

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "entry-gate-stats-collector",
        }
    )

    workflow_id = find_workflow_id(session, args.owner, args.repo, args.workflow_name)
    logger.info(
        "Collecting runs for workflow '%s' on branch '%s' since %s",
        args.workflow_name,
        args.branch,
        since.isoformat(),
    )

    runs = aggregate_runs(
        session,
        args.owner,
        args.repo,
        workflow_id,
        args.branch,
        since,
        args.max_runs,
    )

    output_path = build_output_path(args.output, since)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at_utc": dt.datetime.now(tz=dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "since_date_utc": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "owner": args.owner,
        "repo": args.repo,
        "workflow_name": args.workflow_name,
        "branch": args.branch,
        "run_count": len(runs),
        "runs": runs,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    logger.info("Wrote aggregate stats to %s", output_path)


if __name__ == "__main__":
    main()

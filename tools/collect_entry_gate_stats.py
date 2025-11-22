#!/usr/bin/env python3
"""
Collect entry-gate-stats artifacts from TD Full Pipeline (5m) runs.

- A repo összes workflow-runját lekérdezi (branch = main, status = completed),
- ezek közül csak a "TD Full Pipeline (5m)" nevű runokat tartja meg,
- minden ilyen runhoz lekéri az artifactokat,
- a név alapján kiszűri az entry-gate-stats ZIP-eket,
- kibontja belőlük az entry_gate_stats.json fájlt,
- és mindent egyetlen összesítő JSON-ba gyűjt.

Kimenet (példa):

{
  "owner": "Wildfire0007",
  "repo": "market-feed",
  "workflow_name": "TD Full Pipeline (5m)",
  "branch": "main",
  "since_date_utc": "2025-11-14T00:00:00Z",
  "generated_at_utc": "...",
  "artifact_count": 7,
  "entries": [
    {
      "run_id": 5968,
      "run_number": 123,
      "run_name": "TD Full Pipeline (5m)",
      "run_created_at": "2025-11-22T09:55:12Z",
      "run_conclusion": "success",
      "run_html_url": "https://github.com/…",
      "artifact_id": 987654321,
      "artifact_name": "entry-gate-stats",
      "artifact_size_in_bytes": 350,
      "stats": { … az adott entry_gate_stats.json teljes tartalma … }
    },
    …
  ]
}
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

DEFAULT_OWNER = "Wildfire0007"
DEFAULT_REPO = "market-feed"
DEFAULT_BRANCH = "main"
DEFAULT_WORKFLOW_NAME = "TD Full Pipeline (5m)"
DEFAULT_SINCE_DATE = "2025-11-14"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argumentumok, logging, token
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect entry-gate-stats artifacts.")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--workflow-name",
        default=DEFAULT_WORKFLOW_NAME,
        help="Run name to filter for (e.g. 'TD Full Pipeline (5m)')",
    )
    parser.add_argument(
        "--since-date",
        default=DEFAULT_SINCE_DATE,
        help="Start date (UTC, YYYY-MM-DD). Runs older than this are ignored.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=200,
        help="Maximum number of runs to inspect (across all workflows).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit output path for the aggregate JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
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
            "GitHub token is required. Set GITHUB_TOKEN or GITHUB_PAT in the environment."
        )
    return token


def parse_utc_date(value: str) -> dt.datetime:
    try:
        d = dt.datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"Invalid --since-date '{value}': {exc}") from exc
    return d.replace(tzinfo=dt.timezone.utc)


def parse_github_timestamp(value: str) -> dt.datetime:
    # formátum: "2025-11-22T09:55:12Z"
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc
    )


# ---------------------------------------------------------------------------
# GitHub API helper függvények
# ---------------------------------------------------------------------------

def github_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    try:
        resp = session.get(url, timeout=30, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"GitHub API request failed for {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise RuntimeError(
            f"GitHub API request failed ({resp.status_code}) for {url}: {resp.text}"
        )
    return resp


def iter_repo_runs(
    session: requests.Session,
    owner: str,
    repo: str,
    branch: str,
    since: dt.datetime,
    max_runs: int,
    workflow_name: str,
) -> Iterable[Dict]:
    """
    Az egész repo runjait listázza (`/actions/runs`), és szűr run-név és idő szerint.
    """
    page = 1
    collected = 0
    stop = False

    while not stop and collected < max_runs:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs"
        params = {
            "branch": branch,
            "status": "completed",
            "per_page": 100,
            "page": page,
        }
        logger.debug("Requesting runs page %s with params %s", page, params)
        resp = github_get(session, url, params=params)
        payload = resp.json()
        runs = payload.get("workflow_runs", [])
        if not runs:
            logger.debug("No more runs on page %s, stopping.", page)
            break

        for run in runs:
            created_str = run.get("created_at")
            if not created_str:
                continue
            try:
                created_at = parse_github_timestamp(created_str)
            except ValueError:
                logger.debug("Skipping run with invalid created_at: %s", created_str)
                continue

            # idő szerinti vágás – ha már túl régi, vége
            if created_at < since:
                logger.info(
                    "Run %s (%s) is older than since-date (%s), stopping pagination.",
                    run.get("id"),
                    created_str,
                    since.isoformat(),
                )
                stop = True
                break

            # csak a megadott nevű runok (TD Full Pipeline (5m))
            if run.get("name") != workflow_name:
                continue

            yield run
            collected += 1
            if collected >= max_runs:
                logger.info("Reached max_runs limit (%s).", max_runs)
                stop = True
                break

        page += 1


def list_run_artifacts(
    session: requests.Session, owner: str, repo: str, run_id: int
) -> List[Dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    resp = github_get(session, url)
    artifacts = resp.json().get("artifacts", [])
    logger.debug("Run %s has %d artifacts", run_id, len(artifacts))
    return artifacts


def find_matching_artifacts(artifacts: List[Dict]) -> List[Dict]:
    matches: List[Dict] = []
    for a in artifacts:
        name = (a.get("name") or "").lower()
        if (
            name == "entry-gate-stats"
            or "entry-gate-stats" in name
            or "entry_gate_stats" in name
        ):
            matches.append(a)
    return matches


def download_artifact(
    session: requests.Session, artifact: Dict, dest_dir: Path
) -> Path:
    download_url = artifact.get("archive_download_url")
    if not download_url:
        raise RuntimeError(f"Artifact {artifact.get('id')} has no archive_download_url")

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"artifact_{artifact['id']}.zip"

    headers = {"Accept": "application/zip"}
    try:
        with session.get(download_url, headers=headers, timeout=60, stream=True) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Failed to download artifact {artifact.get('name')} "
                    f"({resp.status_code}): {resp.text}"
                )
            with zip_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Error downloading artifact {artifact.get('id')} from {download_url}: {exc}"
        ) from exc

    return zip_path


def select_json_file(extracted_dir: Path) -> Optional[Path]:
    json_files = sorted(p for p in extracted_dir.rglob("*.json") if p.is_file())
    if not json_files:
        return None
    # ha csak egy van, azt választjuk
    if len(json_files) == 1:
        return json_files[0]
    # ha van név szerint entry_gate_stats.json, azt preferáljuk
    for p in json_files:
        if p.name.lower() == "entry_gate_stats.json":
            return p
    return json_files[0]


def extract_stats_from_artifact(zip_path: Path) -> Optional[Dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_root)
        except zipfile.BadZipFile:
            logger.warning("Invalid ZIP, skipping: %s", zip_path)
            return None

        json_file = select_json_file(tmp_root)
        if not json_file:
            logger.warning("No JSON found in artifact %s", zip_path)
            return None

        try:
            with json_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse JSON %s: %s", json_file, exc)
            return None

        return data


# ---------------------------------------------------------------------------
# Aggregálás és kimenet
# ---------------------------------------------------------------------------

def aggregate_entry_gate_stats(
    session: requests.Session,
    owner: str,
    repo: str,
    branch: str,
    workflow_name: str,
    since: dt.datetime,
    max_runs: int,
) -> List[Dict]:
    entries: List[Dict] = []

    for run in iter_repo_runs(
        session=session,
        owner=owner,
        repo=repo,
        branch=branch,
        since=since,
        max_runs=max_runs,
        workflow_name=workflow_name,
    ):
        run_id = run.get("id")
        if run_id is None:
            continue

        artifacts = list_run_artifacts(session, owner, repo, run_id)
        matches = find_matching_artifacts(artifacts)
        if not matches:
            logger.info("Run %s: no entry-gate-stats artifacts", run_id)
            continue

        logger.info(
            "Run %s (%s): found %d matching artifacts",
            run_id,
            run.get("name"),
            len(matches),
        )

        for artifact in matches:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_root = Path(tmpdir)
                try:
                    zip_path = download_artifact(session, artifact, tmp_root)
                except RuntimeError as exc:
                    logger.warning("%s", exc)
                    continue

                stats = extract_stats_from_artifact(zip_path)
                if stats is None:
                    continue

                entries.append(
                    {
                        "run_id": run.get("id"),
                        "run_number": run.get("run_number"),
                        "run_name": run.get("name"),
                        "run_created_at": run.get("created_at"),
                        "run_updated_at": run.get("updated_at"),
                        "run_status": run.get("status"),
                        "run_conclusion": run.get("conclusion"),
                        "run_html_url": run.get("html_url"),
                        "artifact_id": artifact.get("id"),
                        "artifact_name": artifact.get("name"),
                        "artifact_size_in_bytes": artifact.get("size_in_bytes"),
                        "stats": stats,
                    }
                )

    logger.info("Collected %d artifacts in total.", len(entries))
    return entries


def build_output_path(output: Optional[Path], since: dt.datetime) -> Path:
    if output is not None:
        return output
    date_str = since.strftime("%Y-%m-%d")
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

    logger.info(
        "Collecting entry-gate-stats artifacts for '%s' on branch '%s' since %s",
        args.workflow_name,
        args.branch,
        since.isoformat(),
    )

    entries = aggregate_entry_gate_stats(
        session=session,
        owner=args.owner,
        repo=args.repo,
        branch=args.branch,
        workflow_name=args.workflow_name,
        since=since,
        max_runs=args.max_runs,
    )

    output_path = build_output_path(args.output, since)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "owner": args.owner,
        "repo": args.repo,
        "workflow_name": args.workflow_name,
        "branch": args.branch,
        "since_date_utc": since.strftime("%Y-%m-%dT00:00:00Z"),
        "generated_at_utc": dt.datetime.now(tz=dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "artifact_count": len(entries),
        "entries": entries,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    logger.info("Wrote aggregate JSON to %s", output_path)


if __name__ == "__main__":
    main()

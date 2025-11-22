#!/usr/bin/env python3
"""
Collect entry-gate-stats artifacts from TD Full Pipeline (5m) runs.

- Lekéri a repo összes completed runját (branch = main),
- ezek közül csak a "TD Full Pipeline (5m)" nevű workflow-runokat tartja meg,
- minden ilyen runhoz lekéri az artifactokat,
- név alapján kiszűri az entry-gate-stats ZIP-eket,
- kibontja belőlük az entry_gate_stats*.json fájlt,
- és mindent egyetlen összesítő JSON-ba gyűjt.

A script csak a standard könyvtárat + requests-et használja.
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


# ---------------------------------------------------------------------------
# CLI és alap beállítások
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="Repository owner")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository name")
    parser.add_argument(
        "--workflow-name",
        default=DEFAULT_WORKFLOW_NAME,
        help="Target workflow run name (pl. 'TD Full Pipeline (5m)')",
    )
    parser.add_argument(
        "--since-date",
        default=DEFAULT_SINCE_DATE,
        help="ISO date (YYYY-MM-DD) interpreted as UTC midnight",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Branch filter (pl. 'main')",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1000,
        help="Maximum number of matching runs to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output path. "
            "Default: public/debug/entry_gate_stats_aggregate_since_<DATE>.json"
        ),
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
    # GitHub: "2025-11-22T10:30:53Z"
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc
    )


def github_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    try:
        response = session.get(url, timeout=30, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"GitHub API request failed for {url}: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub API request failed ({response.status_code}) for {url}: {response.text}"
        )
    return response


# ---------------------------------------------------------------------------
# Run-listázás repó szinten
# ---------------------------------------------------------------------------

def workflow_name_matches(run_name: str, target: str) -> bool:
    """Laza, de determinisztikus egyezés a workflow névre."""
    rn = (run_name or "").strip()
    tn = (target or "").strip()
    if not rn or not tn:
        return False
    # case-insensitive exact, plus fallback: target substring a run_name-ben
    rn_cf = rn.casefold()
    tn_cf = tn.casefold()
    return rn_cf == tn_cf or tn_cf in rn_cf


def iter_repo_runs(
    session: requests.Session,
    owner: str,
    repo: str,
    branch: str,
    since: dt.datetime,
    workflow_name: str,
    max_runs: int,
) -> Iterable[Dict]:
    """
    Végigmegy a repó összes completed runján (branch filterrel),
    és csak azokat yieldeli, amelyek neve egyezik a kívánt workflow névvel.
    """
    page = 1
    collected = 0
    while collected < max_runs:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs"
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
            logger.info("No more workflow runs returned (page %s).", page)
            break

        for run in runs:
            created_at_str = run.get("created_at")
            if not created_at_str:
                continue
            try:
                created_at = parse_github_timestamp(created_at_str)
            except ValueError:
                logger.debug("Skipping run with invalid created_at: %s", created_at_str)
                continue

            # GitHub API runok időrendben visszafelé jönnek -> ha ennél régebbi, megállhatunk.
            if created_at < since:
                logger.info(
                    "Encountered run older than since-date (%s < %s); stopping pagination.",
                    created_at.isoformat(),
                    since.isoformat(),
                )
                return

            run_name = run.get("name") or ""
            if not workflow_name_matches(run_name, workflow_name):
                continue

            collected += 1
            logger.debug(
                "Matched run_id=%s run_number=%s name=%r created_at=%s",
                run.get("id"),
                run.get("run_number"),
                run_name,
                created_at_str,
            )
            yield run

            if collected >= max_runs:
                logger.info("Reached max_runs limit (%s).", max_runs)
                return

        page += 1


# ---------------------------------------------------------------------------
# Artifactok letöltése, kibontása
# ---------------------------------------------------------------------------

def list_run_artifacts(
    session: requests.Session, owner: str, repo: str, run_id: int
) -> List[Dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    response = github_get(session, url)
    artifacts = response.json().get("artifacts", []) or []
    logger.debug(
        "run_id=%s: received %s artifacts: %s",
        run_id,
        len(artifacts),
        [a.get("name") for a in artifacts],
    )
    return artifacts



def find_matching_artifacts(artifacts: List[Dict]) -> List[Dict]:
    """entry-gate-stats / entry_gate_stats nevű artifactok szűrése."""
    matches: List[Dict] = []
    for artifact in artifacts:
        name = (artifact.get("name") or "").lower()
        expired = artifact.get("expired", False)
        logger.debug(
            "  artifact id=%s name=%r expired=%s size=%s",
            artifact.get("id"),
            name,
            expired,
            artifact.get("size_in_bytes"),
        )
        # MÉG NEM szűrjük ki az expired-et, csak logoljuk:
        if (
            name == "entry-gate-stats"
            or "entry-gate-stats" in name
            or "entry_gate_stats" in name
        ):
            matches.append(artifact)
    logger.debug("  matched %s artifacts by name", len(matches))
    return matches



def download_artifact(
    session: requests.Session, artifact: Dict, dest_dir: Path
) -> Path:
    download_url = artifact.get("archive_download_url")
    if not download_url:
        raise RuntimeError(f"Artifact {artifact.get('name')} missing download URL")
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = dest_dir / f"artifact_{artifact['id']}.zip"
    headers = {"Accept": "application/zip"}
    try:
        with session.get(download_url, headers=headers, timeout=60, stream=True) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Failed to download artifact {artifact.get('name')} "
                    f"({resp.status_code}): {resp.text}"
                )
            with tmp_zip.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download artifact {artifact.get('name') or artifact.get('id')} "
            f"from {download_url}: {exc}"
        ) from exc
    return tmp_zip


def select_stats_json(extracted_dir: Path) -> Optional[Path]:
    """Kiválasztja a releváns JSON-t az artifactból."""
    json_files = sorted(p for p in extracted_dir.rglob("*.json") if p.is_file())
    if not json_files:
        return None

    # Prefer entry_gate_stats*.json, ha van ilyen.
    for candidate in json_files:
        name = candidate.name.lower()
        if "entry_gate_stats" in name:
            return candidate
    return json_files[0]


def extract_stats_from_artifact(zip_path: Path) -> Optional[Dict]:
    """ZIP -> entry_gate_stats JSON payload (dict)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_path)
        except zipfile.BadZipFile:
            logger.warning("Skipping artifact %s due to invalid ZIP format", zip_path)
            return None

        json_file = select_stats_json(tmp_path)
        if not json_file:
            logger.warning("No JSON files found in artifact %s", zip_path)
            return None

        try:
            with json_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse JSON file %s: %s", json_file, exc)
            return None

        return data


# ---------------------------------------------------------------------------
# Aggregálás
# ---------------------------------------------------------------------------

def aggregate(
    session: requests.Session,
    owner: str,
    repo: str,
    workflow_name: str,
    branch: str,
    since: dt.datetime,
    max_runs: int,
) -> Dict:
    entries: List[Dict] = []
    run_counter = 0
    artifact_counter = 0

    for run in iter_repo_runs(
        session=session,
        owner=owner,
        repo=repo,
        workflow_name=workflow_name,
        branch=branch,
        since=since,
        max_runs=max_runs,
    ):
        run_id = run.get("id")
        if run_id is None:
            continue

        run_counter += 1
        logger.debug(
            "Processing run_id=%s run_number=%s name=%r created_at=%s",
            run_id,
            run.get("run_number"),
            run.get("name"),
            run.get("created_at"),
        )

        artifacts = list_run_artifacts(session, owner, repo, run_id)
        if not artifacts:
            logger.debug("  -> no artifacts returned for this run")
        matches = find_matching_artifacts(artifacts)

        if not matches:
            logger.debug("  -> no entry-gate-stats artifacts matched for this run")
            continue

        for artifact in matches:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_dir = Path(tmpdir)
                try:
                    zip_path = download_artifact(session, artifact, tmp_dir)
                except RuntimeError as exc:
                    logger.warning("%s", exc)
                    continue

                stats_data = extract_stats_from_artifact(zip_path)
                if stats_data is None:
                    continue

                artifact_counter += 1
                entries.append(
                    {
                        "run_id": run.get("id"),
                        "run_number": run.get("run_number"),
                        "run_name": run.get("name"),
                        "status": run.get("status"),
                        "conclusion": run.get("conclusion"),
                        "created_at": run.get("created_at"),
                        "updated_at": run.get("updated_at"),
                        "html_url": run.get("html_url"),
                        "head_sha": run.get("head_sha"),
                        "artifact_id": artifact.get("id"),
                        "artifact_name": artifact.get("name"),
                        "artifact_size_in_bytes": artifact.get("size_in_bytes"),
                        "stats": stats_data,
                    }
                )

    entries.sort(
        key=lambda item: (item.get("created_at") or "", item.get("artifact_id") or 0),
        reverse=True,
    )

    summary = {
        "owner": owner,
        "repo": repo,
        "workflow_name": workflow_name,
        "branch": branch,
        "since_date_utc": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_at_utc": dt.datetime.now(tz=dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "run_count": run_counter,
        "artifact_count": artifact_counter,
        "entries": entries,
    }
    return summary


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

    logger.info(
        "Collecting runs for workflow_name=%r, branch=%r, since=%s",
        args.workflow_name,
        args.branch,
        since.isoformat(),
    )

    summary = aggregate(
        session=session,
        owner=args.owner,
        repo=args.repo,
        workflow_name=args.workflow_name,
        branch=args.branch,
        since=since,
        max_runs=args.max_runs,
    )

    output_path = build_output_path(args.output, since)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    logger.info(
        "Wrote aggregate stats to %s (runs=%s, artifacts=%s)",
        output_path,
        summary["run_count"],
        summary["artifact_count"],
    )


if __name__ == "__main__":
    main()

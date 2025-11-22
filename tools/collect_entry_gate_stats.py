#!/usr/bin/env python3
"""Download and aggregate entry gate stats artifacts from GitHub Actions."""
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
from typing import Any, Dict, List, Optional

import requests
from requests import RequestException

GITHUB_API = "https://api.github.com"
DEFAULT_OWNER = "Wildfire0007"
DEFAULT_REPO = "market-feed"
DEFAULT_SINCE_DATE = "2025-11-14"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="Repository owner")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository name")
    parser.add_argument(
        "--since-date",
        default=DEFAULT_SINCE_DATE,
        help="ISO date (YYYY-MM-DD) interpreted as UTC midnight",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output path. Defaults to public/debug/entry_gate_stats_"
            "aggregate_since_<DATE>.json"
        ),
    )
    parser.add_argument(
        "--max-artifacts",
        type=int,
        default=500,
        help="Maximum number of artifacts to process",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
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


def parse_since_date(date_str: str) -> dt.datetime:
    try:
        parsed = dt.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"Invalid --since-date value '{date_str}': {exc}") from exc
    return parsed.replace(tzinfo=dt.timezone.utc)


def parse_github_timestamp(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def github_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    try:
        response = session.get(url, timeout=30, **kwargs)
    except RequestException as exc:
        raise RuntimeError(f"GitHub API request failed for {url}: {exc}") from exc
    if response.status_code >= 400:
        snippet = response.text[:200]
        raise RuntimeError(
            f"GitHub API request failed ({response.status_code}) for {url}: {snippet}"
        )
    return response


def list_repo_artifacts(
    session: requests.Session, owner: str, repo: str, max_artifacts: int
) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    page = 1
    while len(artifacts) < max_artifacts:
        params = {"per_page": 100, "page": page}
        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/artifacts"
        response = github_get(session, url, params=params)
        payload = response.json()
        page_artifacts = payload.get("artifacts", [])
        if not page_artifacts:
            break
        artifacts.extend(page_artifacts)
        logger.debug("Fetched %s artifacts on page %s", len(page_artifacts), page)
        if len(page_artifacts) < 100:
            break
        page += 1
    return artifacts[:max_artifacts]


def download_artifact_zip(
    session: requests.Session, artifact: Dict[str, Any], dest_dir: Path
) -> Path:
    download_url = artifact.get("archive_download_url")
    if not download_url:
        raise RuntimeError(f"Artifact {artifact.get('id')} missing download URL")
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"artifact_{artifact['id']}.zip"
    headers = {"Accept": "application/zip"}
    try:
        with session.get(download_url, headers=headers, timeout=30, stream=True) as resp:
            if resp.status_code >= 400:
                snippet = resp.text[:200]
                raise RuntimeError(
                    f"Failed to download artifact {artifact.get('id')} ({resp.status_code}): {snippet}"
                )
            with zip_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
    except RequestException as exc:
        raise RuntimeError(
            f"Failed to download artifact {artifact.get('id')} from {download_url}: {exc}"
        ) from exc
    return zip_path


def extract_stats_from_zip(zip_path: Path) -> Optional[Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_path)
        except zipfile.BadZipFile:
            logger.warning("Invalid ZIP file: %s", zip_path)
            return None
        json_files = sorted(path for path in tmp_path.rglob("*.json") if path.is_file())
        if not json_files:
            logger.warning("No JSON files found in %s", zip_path)
            return None
        if len(json_files) == 1:
            target = json_files[0]
        else:
            preferred = [p for p in json_files if p.name.lower() == "entry_gate_stats.json"]
            target = preferred[0] if preferred else json_files[0]
        try:
            with target.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse JSON from %s: %s", target, exc)
            return None


def filter_entry_gate_artifacts(
    artifacts: List[Dict[str, Any]], since_date: dt.datetime
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for artifact in artifacts:
        name = (artifact.get("name") or "").lower()
        created_at_raw = artifact.get("created_at")
        if not created_at_raw:
            logger.debug("Skipping artifact without created_at: %s", artifact.get("id"))
            continue
        try:
            created_at = parse_github_timestamp(created_at_raw)
        except ValueError:
            logger.debug("Skipping artifact with invalid timestamp: %s", created_at_raw)
            continue
        if created_at < since_date:
            continue
        if "entry-gate-stats" in name or "entry_gate_stats" in name:
            matches.append(artifact)
    return matches


def build_output_path(output: Optional[Path], since: dt.datetime) -> Path:
    date_str = since.strftime("%Y-%m-%d")
    if output:
        return output
    return Path(f"public/debug/entry_gate_stats_aggregate_since_{date_str}.json")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    
    token = load_token()
    since_date = parse_since_date(args.since_date)

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "entry-gate-stats-collector",
        }
    )

    try:
        artifacts = list_repo_artifacts(
            session, args.owner, args.repo, max_artifacts=args.max_artifacts
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    filtered = filter_entry_gate_artifacts(artifacts, since_date)
    logger.info("Found %s matching artifacts.", len(filtered))
    if filtered:
        for artifact in filtered:
            logger.info("Artifact %s: %s", artifact.get("id"), artifact.get("name"))

    entries: List[Dict[str, Any]] = []
    for artifact in filtered:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            try:
                zip_path = download_artifact_zip(session, artifact, tmp_path)
                stats = extract_stats_from_zip(zip_path)
            except RuntimeError as exc:
                logger.warning("%s", exc)
                continue
            if stats is None:
                continue
        workflow_run = artifact.get("workflow_run") or {}
        entry = {
            "artifact_id": artifact.get("id"),
            "artifact_name": artifact.get("name"),
            "artifact_size_in_bytes": artifact.get("size_in_bytes"),
            "artifact_created_at": artifact.get("created_at"),
            "artifact_expired": artifact.get("expired"),
            "workflow_run_id": workflow_run.get("id"),
            "workflow_run_number": workflow_run.get("run_number"),
            "workflow_run_html_url": workflow_run.get("html_url"),
            "stats": stats,
        }
        entries.append(entry)

    output_path = build_output_path(args.output, since_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "owner": args.owner,
        "repo": args.repo,
        "since_date_utc": since_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_at_utc": dt.datetime.now(tz=dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "artifact_count": len(entries),
        "entries": entries,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    logger.info("Wrote aggregate stats to %s", output_path)


if __name__ == "__main__":
    main()

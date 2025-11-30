import argparse
import datetime as dt
import json
import logging
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import requests


GITHUB_API = "https://api.github.com"
DEFAULT_WORKFLOW_ID = 195596686
DEFAULT_SINCE_DATE = "2024-11-14"
DEFAULT_OUTPUT_TEMPLATE = (
    "public/debug/entry_gate_stats_aggregate_since_{since}.json"
)
REQUEST_TIMEOUT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate entry-gate-stats artifacts for a workflow since a given date."
        )
    )
    parser.add_argument(
        "--since-date",
        default=DEFAULT_SINCE_DATE,
        help=(
            "ISO date (YYYY-MM-DD) in UTC for filtering workflow runs (default:"
            f" {DEFAULT_SINCE_DATE})."
        ),
    )
    parser.add_argument(
        "--workflow-id",
        type=int,
        default=DEFAULT_WORKFLOW_ID,
        help=f"Workflow ID to scan (default: {DEFAULT_WORKFLOW_ID}).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional maximum number of runs to scan.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path for aggregated JSON. Defaults to"
            " public/debug/entry_gate_stats_aggregate_since_<date>.json"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="Optional run ID to inspect directly instead of scanning a workflow.",
    )
    parser.add_argument(
        "--artifact-name",
        default="entry-gate-stats",
        help="Artifact name to collect when not using regex filtering.",
    )
    parser.add_argument(
        "--artifact-regex",
        help="Optional regex to match artifact names.",
    )
    parser.add_argument(
        "--artifact-case-insensitive",
        action="store_true",
        help="Perform case-insensitive artifact name matching.",
    )
    return parser.parse_args()


def get_repo_slug() -> str:
    try:
        url = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                text=True,
            )
            .strip()
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Unable to determine git remote origin URL") from exc

    if url.startswith("git@"):
        _, path = url.split(":", 1)
        repo_part = path
    elif url.startswith("https://") or url.startswith("http://"):
        repo_part = url.split("//", 1)[-1]
        if "/" in repo_part:
            repo_part = repo_part.split("/", 1)[1]
    else:
        raise RuntimeError(f"Unsupported remote URL format: {url}")

    if repo_part.endswith(".git"):
        repo_part = repo_part[:-4]
    if repo_part.count("/") != 1:
        raise RuntimeError(f"Unable to parse owner/repo from URL: {url}")

    return repo_part


def get_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN (or GH_TOKEN) environment variable is required")
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def parse_iso_date(date_str: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(date_str)
    except ValueError as exc:
        raise ValueError(f"Invalid date format for --since-date: {date_str}") from exc
    if parsed.tzinfo:
        parsed = parsed.astimezone(dt.timezone.utc)
    else:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    if parsed > now:
        logging.warning(
            "--since-date %s is in the future; clamping to current UTC date %s",
            parsed.date().isoformat(),
            now.date().isoformat(),
        )
        parsed = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return parsed


def fetch_paginated(session: requests.Session, url: str, params: dict):
    page = 1
    while True:
        page_params = {**params, "page": page}
        logging.debug("Requesting %s with params %s", url, page_params)
        response = session.get(url, params=page_params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        items = data.get("workflow_runs") or data.get("artifacts") or []
        logging.debug("Received %s items on page %s", len(items), page)
        if not items:
            break
        yield from items
        if len(items) < params.get("per_page", 100):
            break
        page += 1


def fetch_run(session: requests.Session, repo_slug: str, run_id: int) -> dict:
    url = f"{GITHUB_API}/repos/{repo_slug}/actions/runs/{run_id}"
    logging.info("Fetching run details for run %s", run_id)
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def download_artifact_zip(session: requests.Session, download_url: str, dest_dir: Path) -> Path:
    logging.info("Downloading artifact archive from %s", download_url)
    response = session.get(download_url, stream=True, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    dest_dir.mkdir(parents=True, exist_ok=True)
    temp_path = dest_dir / "artifact.zip"
    with temp_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logging.info("Artifact archive saved to %s (%s bytes)", temp_path, temp_path.stat().st_size)
    return temp_path


def extract_stats_from_zip(zip_path: Path) -> tuple[str | None, dict | None, list[str], bool]:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            logging.info("ZIP %s contains files: %s", zip_path, ", ".join(namelist))
            stats_file: str | None = None
            stats_data: dict | None = None
            json_error = False
            for name in namelist:
                if Path(name).name == "entry_gate_stats.json":
                    stats_file = name
                    logging.info("Found entry_gate_stats.json at %s", name)
                    with zf.open(name) as stats_file_handle:
                        try:
                            stats_data = json.load(stats_file_handle)
                        except json.JSONDecodeError as exc:
                            json_error = True
                            logging.error(
                                "Malformed JSON in %s within %s: %s", name, zip_path, exc
                            )
                    break
            if not stats_file:
                logging.warning("entry_gate_stats.json not found in %s", zip_path)
            return stats_file, stats_data, namelist, json_error
    except zipfile.BadZipFile:
        logging.error("Invalid ZIP archive: %s", zip_path)
    return None, None, [], False


def aggregate(args: argparse.Namespace) -> dict:
    since_dt = parse_iso_date(args.since_date)
    since_iso = since_dt.date().isoformat()
    output_path = Path(
        args.output
        or DEFAULT_OUTPUT_TEMPLATE.format(since=since_iso)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(get_headers())
    repo_slug = get_repo_slug()
    
    entries = []
    artifact_success_count = 0
    artifact_error_count = 0
    run_count = 0
    artifact_count = 0
    if args.run_id:
        logging.info(
            "Inspecting single run %s (since-date filters are skipped)", args.run_id
        )
        runs = [fetch_run(session, repo_slug, args.run_id)]
    else:
        runs_url = f"{GITHUB_API}/repos/{repo_slug}/actions/workflows/{args.workflow_id}/runs"
        params = {
            "branch": "main",
            "status": "completed",
            "per_page": 100,
        }

        logging.info("Fetching workflow runs for %s since %s", repo_slug, since_iso)
        runs = fetch_paginated(session, runs_url, params)

    def artifact_matches(name: str) -> bool:
        if args.artifact_regex:
            flags = re.IGNORECASE if args.artifact_case_insensitive else 0
            return re.search(args.artifact_regex, name, flags) is not None
        if args.artifact_case_insensitive:
            return name.casefold() == args.artifact_name.casefold()
        return name == args.artifact_name

    for run in runs:
        run_id = run.get("id")
        created_at_raw = run.get("created_at")
        if not args.run_id:
            created_at = dt.datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            if created_at.tzinfo:
                created_at = created_at.astimezone(dt.timezone.utc)
            else:
                created_at = created_at.replace(tzinfo=dt.timezone.utc)
            if created_at < since_dt:
                logging.info(
                    "Encountered run %s before since-date (%s), stopping pagination",
                    run_id,
                    since_iso,
                )
                break
            if args.max_runs and run_count >= args.max_runs:
                logging.info("Reached max-runs limit (%s)", args.max_runs)
                break

        run_count += 1
        
        logging.info("Processing run %s (number %s)", run_id, run.get("run_number"))
        artifacts_url = run.get("artifacts_url") or (
            f"{GITHUB_API}/repos/{repo_slug}/actions/runs/{run_id}/artifacts"
        )
        artifact_params = {"per_page": 100}
        run_artifact_total = 0
        run_artifact_matched = 0
        run_artifact_names: list[str] = []
        for artifact in fetch_paginated(session, artifacts_url, artifact_params):
            run_artifact_total += 1
            artifact_name = artifact.get("name", "")
            run_artifact_names.append(artifact_name)
            if not artifact_matches(artifact_name):
                continue
            run_artifact_matched += 1            
            logging.info(
                "Downloading artifact %s (%s) for run %s",
                artifact.get("id"),
                artifact_name,
                run_id,
            )
            download_url = artifact.get("archive_download_url")
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = download_artifact_zip(
                    session, download_url, Path(tmpdir)
                )
                stats_file, stats_data, zip_contents, json_error = extract_stats_from_zip(
                    zip_path
                )
                logging.info(
                    "Artifact %s ZIP contents: %s",
                    artifact.get("id"),
                    ", ".join(zip_contents) if zip_contents else "<empty>",
                )
                if stats_data is None:
                    artifact_error_count += 1
                    logging.warning(
                        "Skipping artifact %s due to missing or invalid stats file",
                        artifact.get("id"),
                    )
                    continue
                artifact_success_count += 1
                artifact_count += 1
                entries.append(
                    {
                        "run_id": run_id,
                        "run_number": run.get("run_number"),
                        "created_at": run.get("created_at"),
                        "conclusion": run.get("conclusion"),
                        "artifact_id": artifact.get("id"),
                        "artifact_name": artifact_name,
                        "stats_file": Path(stats_file).as_posix() if stats_file else None,
                        "stats": stats_data,
                    }
                )

        filtered_out = run_artifact_total - run_artifact_matched
        logging.info(
            "Run %s: %s artifacts found, %s matched filter, %s filtered out",
            run_id,
            run_artifact_total,
            run_artifact_matched,
            filtered_out,
        )
        if run_artifact_names:
            logging.info(
                "Run %s available artifact names: %s",
                run_id,
                ", ".join(sorted(set(run_artifact_names))),
            )
            
    aggregated = {
        "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat()
        + "Z",
        "since_date_utc": since_iso,
        "workflow_id": args.workflow_id,
        "run_count": run_count,
        "artifact_count": artifact_count,
        "artifact_success_count": artifact_success_count,
        "artifact_error_count": artifact_error_count,
        "entries": entries,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    logging.info(
        "Aggregation complete: %s runs scanned, %s artifacts collected, output saved to %s",
        run_count,
        artifact_count,
        output_path,
    )
    return aggregated


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        aggregate(args)
    except Exception as exc:  # noqa: BLE001
        logging.error("Aggregation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

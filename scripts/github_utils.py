"""Shared helpers for interacting with the GitHub API."""
from __future__ import annotations

import os
import subprocess
from typing import Iterable, Tuple


def load_github_token(env_names: Iterable[str] | None = None) -> str:
    """Return the first available GitHub token from the provided env names."""

    candidates = tuple(env_names) if env_names is not None else (
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_PAT",
        "ENTRY_GATE_STATS_PAT",
    )
    for env_name in candidates:
        token = os.getenv(env_name)
        if token:
            return token
    joined = ", ".join(candidates)
    raise RuntimeError(f"GitHub token is required. Set one of: {joined}.")


def _parse_remote_url(url: str) -> Tuple[str, str]:
    if url.startswith("git@"):
        path = url.split(":", 1)[1]
    elif url.startswith("https://") or url.startswith("http://"):
        path = url.split("//", 1)[1]
        if "/" in path:
            path = path.split("/", 1)[1]
    else:
        raise RuntimeError(f"Unsupported remote URL format: {url}")

    if path.endswith(".git"):
        path = path[:-4]
    if path.count("/") != 1:
        raise RuntimeError(f"Unable to parse owner/repo from URL: {url}")
    owner, repo = path.split("/", 1)
    return owner, repo


def detect_repo(remote: str = "origin") -> Tuple[str, str]:
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", f"remote.{remote}.url"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Unable to determine git remote '{remote}' URL") from exc
    return _parse_remote_url(url)


def detect_repo_slug(remote: str = "origin") -> str:
    owner, repo = detect_repo(remote)
    return f"{owner}/{repo}"


def build_api_headers(token: str, *, accept: str = "application/vnd.github+json") -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": accept}


__all__ = [
    "build_api_headers",
    "detect_repo",
    "detect_repo_slug",
    "load_github_token",
]

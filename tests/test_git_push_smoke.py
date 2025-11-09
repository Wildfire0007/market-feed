from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _run(cmd, cwd: Path, *, check: bool = True, capture_output: bool = False):
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        capture_output=capture_output,
        text=True,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
    )


def _configure_identity(repo: Path) -> None:
    _run(["git", "config", "user.name", "TD Pipeline"], repo)
    _run(["git", "config", "user.email", "td@example.com"], repo)


def test_git_push_conflict_smoke(tmp_path):
    remote = tmp_path / "remote.git"
    _run(["git", "init", "--bare", str(remote)], tmp_path)

    primary = tmp_path / "primary"
    _run(["git", "clone", str(remote), str(primary)], tmp_path)
    _configure_identity(primary)

    (primary / "payload.txt").write_text("alpha\n", encoding="utf-8")
    _run(["git", "add", "payload.txt"], primary)
    _run(["git", "commit", "-m", "initial"], primary)
    branch = (
        _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], primary, capture_output=True)
        .stdout.strip()
        or "master"
    )
    _run(["git", "push", "origin", branch], primary)

    secondary = tmp_path / "secondary"
    _run(["git", "clone", str(remote), str(secondary)], tmp_path)
    _configure_identity(secondary)

    (secondary / "payload.txt").write_text("beta\n", encoding="utf-8")
    _run(["git", "add", "payload.txt"], secondary)
    _run(["git", "commit", "-m", "upstream"], secondary)
    _run(["git", "push", "origin", branch], secondary)

    (primary / "payload.txt").write_text("gamma\n", encoding="utf-8")
    _run(["git", "add", "payload.txt"], primary)
    _run(["git", "commit", "-m", "local"], primary)
    result = _run(["git", "push", "origin", branch], primary, check=False, capture_output=True)

    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "non-fast-forward" in combined or "failed to push" in combined or "fetch first" in combined

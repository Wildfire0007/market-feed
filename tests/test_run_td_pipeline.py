from __future__ import annotations

from typing import Any, Sequence

from scripts import run_td_pipeline


def test_pipeline_runs_position_watchdog_before_discord(monkeypatch):
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    def _fake_run(cmd: Sequence[str], cwd: Any = None):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(run_td_pipeline.subprocess, "run", _fake_run)

    exit_code = run_td_pipeline.main(
        [
            "--skip-env-sync",
            "--skip-trading",
            "--skip-volatility",
            "--skip-news",
            "--skip-analysis",
            "--skip-watchdog",
            "--skip-spot-watchdog",
        ]
    )

    assert exit_code == 0
    command_texts = [" ".join(cmd) for cmd in calls]
    watchdog_idx = next(i for i, cmd in enumerate(command_texts) if "scripts/position_watchdog.py" in cmd)
    notify_idx = next(i for i, cmd in enumerate(command_texts) if "scripts/notify_discord.py" in cmd)
    assert watchdog_idx < notify_idx



def test_pipeline_runs_watchdog_at_end(monkeypatch):
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    def _fake_run(cmd: Sequence[str], cwd: Any = None):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(run_td_pipeline.subprocess, "run", _fake_run)

    exit_code = run_td_pipeline.main(
        [
            "--skip-env-sync",
            "--skip-trading",
            "--skip-volatility",
            "--skip-news",
            "--skip-analysis",
            "--skip-watchdog",
            "--skip-spot-watchdog",
            "--skip-discord",
        ]
    )

    assert exit_code == 0
    assert calls
    assert calls[-1][1:] == ["scripts/position_watchdog.py"]

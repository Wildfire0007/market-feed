import subprocess
import sys
from pathlib import Path


ENTRYPOINTS = [
    "scripts/heartbeat_watchdog.py",
    "scripts/state_unknown_guard.py",
    "scripts/daily_actionable_digest.py",
    "scripts/weekly_report_discord.py",    
    "scripts/notify_discord.py",
    "scripts/notify_management_discord.py",
    "scripts/position_lifecycle.py",
    "scripts/label_trades.py",
]


def test_standalone_script_entrypoints_support_help():
    repo_root = Path(__file__).resolve().parents[1]
    for entrypoint in ENTRYPOINTS:
        result = subprocess.run(
            [sys.executable, entrypoint, "--help"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, f"{entrypoint} failed: stdout={result.stdout!r} stderr={result.stderr!r}"

import os
import subprocess
from pathlib import Path


def test_backup_script_creates_backup(tmp_path: Path) -> None:
    state_db = tmp_path / "trading_state.db"
    state_db.write_text("seed", encoding="utf-8")
    backup_dir = tmp_path / "backups"

    env = os.environ.copy()
    env["STATE_DB_PATH"] = str(state_db)
    env["BACKUP_DIR"] = str(backup_dir)

    result = subprocess.run(
        ["scripts/backup_trading_state.sh"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert "Backup created" in result.stdout
    backups = list(backup_dir.glob("trading_state_*.db"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "seed"

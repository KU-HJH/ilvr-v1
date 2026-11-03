import re
import subprocess
from pathlib import Path

from src.multinode import _RUN_ID_PATTERN


_OPERATOR_DOCUMENTS = tuple(sorted(Path("docs").rglob("*.md")))


def test_operator_timestamp_commands_produce_valid_run_id_components():
    """Catches documented timestamps producing run IDs rejected by launchers."""
    formats = []
    for document in _OPERATOR_DOCUMENTS:
        formats.extend(
            re.findall(r"date -u \+([^\s\)\"]+)", document.read_text())
        )

    assert formats
    for date_format in formats:
        completed = subprocess.run(
            ["date", "-u", f"+{date_format}"],
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = f"operator-{completed.stdout.strip()}"
        assert _RUN_ID_PATTERN.fullmatch(candidate), (
            f"{date_format!r} produced invalid run ID {candidate!r}"
        )

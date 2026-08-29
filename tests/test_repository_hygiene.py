import subprocess
from pathlib import Path


def test_repository_tracks_no_python_bytecode():
    """Catches generated Python bytecode making clean-tree preflight unstable."""
    repository = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.pyc"],
        cwd=repository,
        check=True,
        capture_output=True,
    )

    assert completed.stdout == b""

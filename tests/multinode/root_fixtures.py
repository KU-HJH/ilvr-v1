"""Explicit same-root fixtures for legacy CPU command tests (no remote execution)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT_KWARGS = dict(remote_repo_root=ROOT,
    remote_host_repo_root=Path('/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/stage1-v3'))
ROOT_ARGUMENTS = ('--remote-repo-root', str(ROOT), '--remote-host-repo-root',
                  str(REMOTE_ROOT_KWARGS['remote_host_repo_root']))

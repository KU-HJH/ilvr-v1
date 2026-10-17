"""Explicit same-root fixtures for historical coordinator tests in native checkouts.

Managed-root/default-guard tests opt out with ``native_roots``; production APIs
always retain their real requirement for explicit remote roots.
"""
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]

def pytest_configure(config):
    config.addinivalue_line('markers','native_roots: exercise actual repository root defaults without legacy fixture')

@pytest.fixture(autouse=True)
def legacy_same_root_cluster(request,monkeypatch):
    if ROOT.is_relative_to('/root/workspace/ILVR') or request.node.get_closest_marker('native_roots'):
        return
    from src.multinode import ClusterSpec
    original=ClusterSpec.default.__func__
    def explicit_same_root(cls,**kwargs):
        if not any(value is not None for value in kwargs.values()):
            kwargs=dict(remote_repo_root=ROOT,
                remote_host_repo_root=Path('/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/native-test'))
        return original(cls,**kwargs)
    monkeypatch.setattr(ClusterSpec,'default',classmethod(explicit_same_root))

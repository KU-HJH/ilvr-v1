"""Node roots must not weaken topology or silently relocate a saved launch."""
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src import multinode as mn
from src import multinode_runtime as rt

LOCAL = Path(__file__).resolve().parents[2]
REMOTE = Path('/root/workspace/ILVR/.worktrees/stage1-v3')
HOST = Path('/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/stage1-v3')


def native_plan(**kwargs):
    return mn.build_launch_plan(stage='stage1', tuning_method='dora', zero_stage=1,
        run_id='stage1-v3-roots-test', main_process_port=29681,
        local_repo_root=LOCAL, remote_repo_root=REMOTE, remote_host_repo_root=HOST, **kwargs)


@pytest.mark.parametrize('suffix', ['', '.worktrees/example'])
def test_canonical_roots_preserve_the_remote_host_suffix(suffix):
    local = Path('/root/workspace/ILVR') / suffix
    spec = mn.ClusterSpec.default(local_repo_root=local)
    assert spec.local_repo_root == spec.remote_repo_root == local
    assert spec.remote_host_repo_root == Path('/home/korea/Workspace/hjh/workspace/ILVR') / suffix


def test_native_root_requires_explicit_remote_pair():
    with pytest.raises(ValueError, match='remote.*root'):
        mn.ClusterSpec.default(local_repo_root=LOCAL)
    with pytest.raises(ValueError, match='remote.*root'):
        mn.ClusterSpec.default(local_repo_root=LOCAL, remote_repo_root=REMOTE)


def test_root_identity_survives_serialization(tmp_path):
    plan = native_plan()
    serialized = rt.serialize_launch_plan(plan, log_root=tmp_path)
    raw = serialized.path.read_bytes()
    payload = json.loads(raw)
    assert payload['cluster']['local_repo_root'] == str(LOCAL)
    assert payload['cluster']['remote_repo_root'] == str(REMOTE)
    assert payload['cluster']['remote_host_repo_root'] == str(HOST)
    assert mn.decode_launch_plan(raw) == plan


@pytest.mark.parametrize('field,value', [
    ('remote_host','someone@172.16.11.41'), ('remote_container','foreign'),
    ('master_addr','127.0.0.1'), ('socket_ifname','eth0'),
    ('remote_rank',1), ('local_rank',0), ('num_machines',1),
    ('gpus_per_node',4), ('num_processes',2), ('remote_rank',False),
])
def test_root_support_still_rejects_each_topology_drift(field,value):
    plan = native_plan()
    with pytest.raises(ValueError, match='topology'):
        replace(plan, cluster=replace(plan.cluster, **{field:value}))


@pytest.mark.parametrize('field,value', [
    ('local_repo_root',Path('relative')), ('remote_repo_root',Path('/root/../foreign')),
    ('remote_host_repo_root',Path('relative')), ('remote_repo_root',Path('/tmp/bad\nroot')),
])
def test_invalid_repository_roots_are_rejected(field,value):
    spec = native_plan().cluster
    with pytest.raises(ValueError, match='root'):
        replace(native_plan(), cluster=replace(spec, **{field:value}))


def test_launch_plan_cannot_claim_a_different_local_checkout():
    with pytest.raises(ValueError, match='local.*checkout'):
        mn.build_launch_plan(stage='stage1', tuning_method='dora', zero_stage=1,
            run_id='stage1-v3-other-checkout', main_process_port=29681,
            local_repo_root=Path('/root/workspace/ILVR'),
            remote_repo_root=REMOTE, remote_host_repo_root=HOST)


def test_plan_cli_accepts_explicit_roots(capsys):
    result = mn._main(['plan','--stage','stage1','--tuning-method','dora','--zero-stage','1',
        '--run-id','stage1-v3-cli-roots','--main-process-port','29681',
        '--local-repo-root',str(LOCAL),'--remote-repo-root',str(REMOTE),
        '--remote-host-repo-root',str(HOST)])
    assert result == 0
    assert json.loads(capsys.readouterr().out)['cluster']['remote_repo_root'] == str(REMOTE)


def test_legacy_same_root_is_preserved_but_native_is_not_guessed(tmp_path):
    plan = native_plan()
    raw = json.loads(rt.serialize_launch_plan(plan, log_root=tmp_path).path.read_text())
    del raw['cluster']['local_repo_root']
    canonical = '/root/workspace/ILVR'
    raw['cluster']['remote_repo_root'] = canonical
    raw['cluster']['remote_host_repo_root'] = '/home/korea/Workspace/hjh/workspace/ILVR'
    raw['deepspeed_config'] = canonical + '/configs/deepspeed_zero1.json'
    decoded = mn.decode_launch_plan(json.dumps(raw).encode())
    assert decoded.cluster.local_repo_root == Path(canonical)
    raw['cluster']['remote_repo_root'] = str(LOCAL)
    with pytest.raises(ValueError, match='legacy.*root'):
        mn.decode_launch_plan(json.dumps(raw).encode())


def test_receive_rejects_foreign_node_before_publishing(tmp_path, monkeypatch):
    plan = native_plan()
    raw = rt.serialize_launch_plan(plan, log_root=tmp_path/'sender').path.read_bytes()
    monkeypatch.setattr(rt, '_REPOSITORY_ROOT', tmp_path/'foreign')
    target = tmp_path/'foreign/logs/multinode'/plan.run_id/plan.attempt_id/'launch-plan.json'
    with pytest.raises(ValueError, match='receiver.*root'):
        rt.receive_launch_plan(path=target, raw=raw, expected_size=len(raw),
            expected_sha256=hashlib.sha256(raw).hexdigest())
    assert not target.exists()


def test_receive_preserves_all_three_roots(tmp_path, monkeypatch):
    remote = tmp_path/'remote'
    plan = replace(native_plan(), cluster=replace(native_plan().cluster, remote_repo_root=remote))
    raw = rt.serialize_launch_plan(plan, log_root=tmp_path/'sender').path.read_bytes()
    monkeypatch.setattr(rt, '_REPOSITORY_ROOT', remote)
    target = remote/'logs/multinode'/plan.run_id/plan.attempt_id/'launch-plan.json'
    rt.prepare_attempt_directory(log_root=remote/'logs/multinode',run_id=plan.run_id,attempt_id=plan.attempt_id)
    rt.receive_launch_plan(path=target, raw=raw, expected_size=len(raw),
        expected_sha256=hashlib.sha256(raw).hexdigest())
    assert mn.decode_launch_plan(target.read_bytes()) == plan


def test_native_checkout_can_import_artifact_module_without_remote_configuration():
    result = subprocess.run([sys.executable,'-c','import src.multinode_artifacts'],
        cwd=LOCAL, capture_output=True,text=True)
    assert result.returncode == 0, result.stderr

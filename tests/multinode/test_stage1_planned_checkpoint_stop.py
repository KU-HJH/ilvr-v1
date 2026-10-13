"""Stage1 producer/coordinator checks bind the invocation to the saved budget."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import multinode as mn
from src.multinode_artifacts import ArtifactEntry, validate_collection
from stage1.training_evidence import build_stage1_training_evidence, decode_stage1_training_evidence
from stage1.v3_config import Stage1V3Config
from stage1.v3_resume import V3_FRESH_PROFILE, build_v3_checkpoint_contract, build_v3_schedule_identity
from tests.multinode.test_artifacts import strict_manifest
from tests.multinode.test_planned_checkpoint_stop import coordinator_paths
from tests.multinode.test_v3_roots import REMOTE, HOST

STOP = {'global_step': 2, 'max_steps': 3, 'budget_complete': False}


def evidence(checkpoint, **extra):
    return build_stage1_training_evidence(run_id=checkpoint.parent.name, method='dora', zero_stage=1,
        world_size=4, micro_batch=1, gradient_accumulation_steps=4, ema_tau=1., seed=42,
        global_step=2, checkpoint_path=checkpoint, loss=.125, **extra)


def decode(payload, checkpoint, **extra):
    return decode_stage1_training_evidence(json.dumps(payload).encode(), expected_run_id=checkpoint.parent.name,
        expected_method='dora', expected_zero_stage=1, expected_world_size=4,
        expected_checkpoint=checkpoint, expected_global_step=2, **extra)


def write_budget(checkpoint, *, max_steps=3, global_step=2, contract_steps=3):
    (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': global_step, 'max_steps': max_steps}))
    schedule = build_v3_schedule_identity(retained_ids=tuple(map(str, range(48))), sampler_identity={'world_size': 4},
        microsteps_per_rank=4 * contract_steps, accumulation_steps=4)
    contract = build_v3_checkpoint_contract(Stage1V3Config(), schedule, dict(V3_FRESH_PROFILE, frozen_teacher_sha256='a' * 64))
    (checkpoint / 'stage1_v3.json').write_text(json.dumps(contract))


def test_evidence_requires_explicit_matching_stop_expectation(tmp_path):
    checkpoint = tmp_path / 'run/checkpoint-2'
    checkpoint.mkdir(parents=True)
    payload = evidence(checkpoint, planned_stop=STOP)
    assert payload['schema_version'] == 3 and payload['planned_stop'] == STOP
    assert decode(payload, checkpoint, expected_planned_stop=2)['planned_stop'] == STOP
    with pytest.raises(ValueError, match='planned stop'):
        decode(payload, checkpoint)
    with pytest.raises(ValueError, match='planned stop'):
        decode(payload, checkpoint, expected_planned_stop=1)
    normal = evidence(checkpoint)
    assert normal['schema_version'] == 2 and 'planned_stop' not in normal
    assert decode(normal, checkpoint) == normal
    with pytest.raises(ValueError, match='planned stop'):
        decode(normal, checkpoint, expected_planned_stop=2)


@pytest.mark.parametrize('field,value', [('global_step', 1), ('global_step', True), ('max_steps', 2),
    ('max_steps', 3.), ('budget_complete', True), ('budget_complete', 0), ('unknown', 1)])
def test_stop_evidence_rejects_unobserved_or_malformed_claims(tmp_path, field, value):
    checkpoint = tmp_path / 'run/checkpoint-2'
    checkpoint.mkdir(parents=True)
    invalid = dict(STOP, **{field: value})
    with pytest.raises(ValueError, match='planned stop'):
        evidence(checkpoint, planned_stop=invalid)
    payload = evidence(checkpoint, planned_stop=STOP)
    payload['planned_stop'] = invalid
    with pytest.raises(ValueError, match='planned stop'):
        decode(payload, checkpoint, expected_planned_stop=2)


@pytest.mark.parametrize('problem', [None, 'missing', 'wrong-step', 'failed-exit', 'producer-step',
    'producer-budget', 'missing-stop', 'complete-claim', 'trainer-budget', 'trainer-step', 'contract-budget', 'missing-contract'])
def test_coordinator_binds_stop_to_all_rank_save_and_original_budget(coordinator_paths, monkeypatch, capsys, problem):
    run_dir, result_root = coordinator_paths
    checkpoint = run_dir / 'checkpoint-2'
    events = []
    commit = mn._repository_git_stdout(('rev-parse', 'HEAD'))
    monkeypatch.setattr(mn, 'collect_node_snapshot', lambda *a, **kw: None)
    monkeypatch.setattr(mn, 'validate_preflight', lambda *a: SimpleNamespace(commit=commit))
    monkeypatch.setattr(mn, 'start_rank0_then_rank1', lambda *a, **kw: events.append('start'))
    def monitor(*a, **kw):
        events.append('exit')
        return {0: 0, 1: 1 if problem == 'failed-exit' else 0}
    monkeypatch.setattr(mn, 'monitor_owned_jobs', monitor)
    class Runner:
        def __init__(self, plan, **kw):
            self.manifests = {}
            checkpoint.mkdir()
            (checkpoint / 'adapter_model.safetensors').write_bytes(b'fixture')
        def read_stage1_training_evidence(self, path):
            # Literal producer input leaves strict decoding in the product path.
            payload = evidence(checkpoint)
            payload.update(schema_version=3, planned_stop=deepcopy(STOP))
            payload['checkpoint_path'] = str(REMOTE / 'checkpoints' / run_dir.name / 'checkpoint-2')
            if problem == 'producer-step':
                payload['global_step'] = 1
            if problem == 'producer-budget':
                payload['planned_stop']['max_steps'] = 4
            if problem == 'missing-stop':
                payload.pop('planned_stop')
                payload['schema_version'] = 2
            if problem == 'complete-claim':
                payload['planned_stop']['budget_complete'] = True
            return json.dumps(payload).encode()
        def read_artifact_manifest(self, rank, path, identity):
            events.append(f'manifest{rank}:{path.name}')
            assert events[:2] == ['start', 'exit']
            assert path == checkpoint and identity['global_step'] == 2
            if problem == 'missing':
                raise FileNotFoundError(path)
            manifest = strict_manifest(rank, changes=identity)
            entries = [e for e in manifest.entries if not e.relative_path.name.endswith('_model_states.pt')]
            if rank == 0:
                entries += [ArtifactEntry(Path('global_step2/mp_rank_00_model_states.pt'), 42, 0),
                            ArtifactEntry(Path('stage1_v3.json'), 43, 0)]
            if problem == 'wrong-step':
                entries = [replace(e, relative_path=Path(e.relative_path.as_posix().replace('global_step2/', 'global_step3/'))) for e in entries]
            self.manifests[rank] = replace(manifest, entries=tuple(entries))
            return self.manifests[rank]
        def validate_push_destination(self, *a):
            pass
        def push_artifacts(self, *a):
            # Shared state is only available on this node after actual collection.
            write_budget(checkpoint, max_steps=4 if problem == 'trainer-budget' else 3,
                global_step=1 if problem == 'trainer-step' else 2,
                contract_steps=4 if problem == 'contract-budget' else 3)
            if problem == 'missing-contract':
                (checkpoint / 'stage1_v3.json').unlink()
        def read_canonical_artifact_manifest(self, path, identity):
            events.append('canonical-validation')
            entries = validate_collection(tuple(self.manifests.values()), world_size=4)
            return replace(self.manifests[0], process_ranks=(0, 1, 2, 3),
                entries=tuple(replace(e, node_rank=0) for e in entries))
    monkeypatch.setattr(mn, 'SubprocessCoordinatorRunner', Runner)
    argv = ['--run-id', run_dir.name, '--zero-stage', '1', '--main-process-port', '29689',
        '--result-root', str(result_root), '--remote-repo-root', str(REMOTE), '--remote-host-repo-root', str(HOST),
        '--stage1-variant', 'v3', '--method', 'dora', '--no-final-save', '--stage1-v3-stop-after-checkpoint', '2']
    witness = result_root / run_dir.name / 'initial-train-witness.json'
    if problem:
        with pytest.raises(SystemExit) as failure:
            mn._training_main('stage1', argv)
        assert failure.value.code == 2
        assert not witness.exists()
        if problem == 'failed-exit':
            assert not any(item.startswith('manifest') for item in events)
    else:
        assert mn._training_main('stage1', argv) == 0
        payload = json.loads(witness.read_text())
        assert payload['planned_stop'] == STOP
        assert payload['world_size'] == 4 and payload['exit_codes'] == {'0': 0, '1': 0}
        assert events[-1] == 'canonical-validation'

"""Real coordinator parsing, collection validation and witness publication."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest
import src.multinode as coordinator
from src.multinode_artifacts import ArtifactEntry, validate_collection
from stage2.evidence import build_stage2_producer_evidence
from tests.multinode.test_artifacts import strict_manifest


@pytest.fixture
def coordinator_paths():
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='planned-stop-test-', dir=repo / 'checkpoints') as state, tempfile.TemporaryDirectory(prefix='planned-stop-test-', dir=repo / 'results') as results:
        yield Path(state), Path(results)


@pytest.mark.parametrize('stop,problem,expected', [
    (2, None, 0), (None, None, 0), (2, 'missing', 1),
    (2, 'wrong-step', 1), (2, 'failed-exit', 1), (2, 'producer-step', 1),
    (None, 'producer-step', 1),
])
def test_coordinator_collects_observed_target_only_after_success(coordinator_paths, monkeypatch, stop, problem, expected):
    # Removing planned-target derivation collects 3, dropping expected-step
    # validation accepts a step3 producer for step2, bypassing exits collects early.
    checkpoint_run, result_root = coordinator_paths
    run_id = checkpoint_run.name
    step = 2 if stop else 3
    events = []
    commit = coordinator._repository_git_stdout(('rev-parse', 'HEAD'))
    monkeypatch.setattr(coordinator, 'collect_node_snapshot', lambda *a, **k: None)
    monkeypatch.setattr(coordinator, 'validate_preflight', lambda *a: SimpleNamespace(commit=commit))
    monkeypatch.setattr(coordinator, 'start_rank0_then_rank1', lambda *a, **k: events.append('start'))
    def monitor(*a, **k):
        events.extend(['exit0=0', 'exit1=1' if problem == 'failed-exit' else 'exit1=0'])
        return {0: 0, 1: 1 if problem == 'failed-exit' else 0}
    monkeypatch.setattr(coordinator, 'monitor_owned_jobs', monitor)
    for key, value in {'STAGE1_CHECKPOINT': '/models/base', 'STAGE1_ADAPTER_PATH': '/models/stage1',
        'DATA_PATH': '/data/train', 'IMAGE_ROOT': '/data', 'TUNING_METHOD': 'dora',
        'STEP_PREFERENCE_ENABLED': 'true', 'STEP_PREFERENCE_MODE': 'stage2b',
        'STEP_PREFERENCE_OBJECTIVE': 'context_preference', 'STEP_PREFERENCE_LOSS_TYPE': 'dil_bce_rf',
        'STAGE2B_DIRECT_FROM_STAGE1': 'true'}.items():
        monkeypatch.setenv(key, value)
    class Runner:
        def __init__(self, plan, *, commands, **kwargs):
            self.manifests = {}
            checkpoint = checkpoint_run / f'checkpoint-{step}'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'adapter_model.safetensors').write_bytes(b'weights')
            producer_step = (3 if stop else 2) if problem == 'producer-step' else step
            producer = commands.finite_loss_evidence_path
            producer.parent.mkdir(parents=True)
            producer.write_text(json.dumps(build_stage2_producer_evidence(
                run_id=run_id, method='dora', zero_stage=2,
                checkpoint_path=checkpoint.parent / f'checkpoint-{producer_step}', artifact_kind='peft',
                rank_records=[dict(rank=i, global_step=producer_step, optimizer_step=producer_step,
                    loss=.25, gradient_norm=1., worker_error=None) for i in range(4)])))
        def read_artifact_manifest(self, rank, checkpoint, identity):
            events.append(f'manifest{rank}:{checkpoint.name}')
            assert events[:3] == ['start', 'exit0=0', 'exit1=0']
            assert checkpoint.name == f'checkpoint-{step}'
            assert identity['global_step'] == step
            if problem == 'missing':
                raise FileNotFoundError(checkpoint)
            manifest = strict_manifest(rank, changes=identity)
            entries = []
            for entry in manifest.entries:
                name = entry.relative_path.as_posix()
                if name.endswith('_model_states.pt'):
                    continue
                name = name.replace('global_step2/', f'global_step{step}/').replace('stage1_peft_metadata', 'stage2_peft_metadata')
                entries.append(replace(entry, relative_path=Path(name)))
            if rank == 0:
                entries.append(ArtifactEntry(Path(f'global_step{step}/mp_rank_00_model_states.pt'), 42, 0))
                entries.append(ArtifactEntry(Path('stage2-sampler.json'), 43, 0))
            if problem == 'wrong-step':
                entries = [replace(e, relative_path=Path(e.relative_path.as_posix().replace('global_step2/', 'global_step3/'))) for e in entries]
            self.manifests[rank] = replace(manifest, entries=tuple(entries), checkpoint_relative=Path(f'checkpoint-{step}'))
            return self.manifests[rank]
        def validate_push_destination(self, manifest, entries):
            events.append('validate-destination')
        def push_artifacts(self, manifest, entries):
            events.append('push')
        def read_canonical_artifact_manifest(self, checkpoint, identity):
            events.append('canonical-validation')
            entries = validate_collection(tuple(self.manifests.values()), world_size=4)
            return replace(self.manifests[0], process_ranks=(0, 1, 2, 3),
                           entries=tuple(replace(e, node_rank=0) for e in entries))
    monkeypatch.setattr(coordinator, 'SubprocessCoordinatorRunner', Runner)
    argv = ['--run-id', run_id, '--zero-stage', '2', '--main-process-port', '29631',
            '--max_steps', '3', '--save_steps', '3', '--result-root', str(result_root)]
    if stop:
        argv += ['--stage2_stop_after_checkpoint', str(stop)]
    if expected:
        with pytest.raises(SystemExit) as failure:
            coordinator._training_main('stage2', argv)
        assert failure.value.code == 2
    else:
        assert coordinator._training_main('stage2', argv) == 0
    witness = result_root / run_id / 'initial-train-witness.json'
    if expected == 0:
        payload = json.loads(witness.read_text())
        assert events[-1] == 'canonical-validation'
        if stop:
            assert payload['planned_stop'] == {'global_step': 2, 'max_steps': 3, 'budget_complete': False}
        else:
            assert 'planned_stop' not in payload
    else:
        assert not witness.exists()
        if problem == 'failed-exit':
            assert not any(event.startswith('manifest') for event in events)

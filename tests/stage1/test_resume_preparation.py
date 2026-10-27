"""Selected historical checkpoints are copied, never overwritten or relabelled."""
import importlib.util
import json
from pathlib import Path

import pytest

from tests.stage1.test_matched_resume import checkpoint_with_resume_contract


def api():
    assert importlib.util.find_spec('stage1.resume_preparation'), 'resume preparation is missing'
    from stage1 import resume_preparation
    return resume_preparation


def state_files(checkpoint):
    for name in ('stage1_ema_teacher.pt', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                 'training_args.bin', 'adapter_model.safetensors'):
        (checkpoint/name).write_bytes(b'state bytes')
    (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step':216, 'max_steps':2160}))


def test_copy_into_fresh_run_preserves_every_source_byte(tmp_path):
    module = api()
    checkpoint, contract = checkpoint_with_resume_contract(tmp_path)
    state_files(checkpoint)
    before = {p.relative_to(checkpoint): p.read_bytes() for p in checkpoint.rglob('*') if p.is_file()}
    destination = tmp_path/'continued/checkpoint-216'
    module.copy_resume_checkpoint(checkpoint, destination, profile=contract['profile'],
                                  source=contract['source'], checkpoint_root=tmp_path)
    assert {p.relative_to(checkpoint): p.read_bytes() for p in checkpoint.rglob('*') if p.is_file()} == before
    for relative, raw in before.items():
        assert (destination/relative).read_bytes() == raw


@pytest.mark.parametrize('collision', ['source', 'existing', 'outside', 'symlink', 'incomplete'])
def test_copy_refuses_unsafe_or_incomplete_destination(tmp_path, collision):
    module = api()
    checkpoint, contract = checkpoint_with_resume_contract(tmp_path)
    state_files(checkpoint)
    destination = tmp_path/'continued/checkpoint-216'
    if collision == 'source': destination = checkpoint
    if collision == 'existing': destination.mkdir(parents=True)
    if collision == 'outside': destination = tmp_path.parent/'outside/checkpoint-216'
    if collision == 'symlink':
        (tmp_path/'continued').symlink_to(tmp_path/'source', target_is_directory=True)
    if collision == 'incomplete': (checkpoint/'rng_state.pth').unlink()
    with pytest.raises((ValueError, FileExistsError)):
        module.copy_resume_checkpoint(checkpoint, destination, profile=contract['profile'],
                                      source=contract['source'], checkpoint_root=tmp_path)


def test_resume_contract_is_training_only_transfer_metadata():
    from dataclasses import replace
    from tests.multinode.test_evaluation_coordinator import _manifest
    from src.multinode_artifacts import ArtifactEntry, select_common_metadata_entries, select_inference_entries
    entry = ArtifactEntry(Path('stage1_matched_resume.json'), 1024, 0)
    manifest = _manifest()
    manifest = replace(manifest, entries=manifest.entries+(entry,))
    assert entry in select_common_metadata_entries(manifest)
    assert entry not in select_inference_entries(manifest)


def test_collection_requires_genuine_successful_supervisors(tmp_path):
    module = api()
    class Runner:
        def poll_exit(self, rank): return None if rank == 1 else 0
        def read_canonical_artifact_manifest(self, *args):
            pytest.fail('must not inspect/collect checkpoint of a running source')
    with pytest.raises(ValueError, match='successful'):
        module.assemble_source_checkpoint(Runner(), tmp_path/'checkpoint-216', {})


@pytest.mark.parametrize('complete', [False, True])
def test_split_source_is_collected_but_complete_final_is_reused(tmp_path, monkeypatch, complete):
    module = api()
    from types import SimpleNamespace
    from src import multinode_artifacts, multinode_runtime
    ranks = (0,1,2,3) if complete else (0,1)
    manifest = SimpleNamespace(entries=tuple(SimpleNamespace(process_rank=r) for r in ranks))
    calls = []
    class Runner:
        def poll_exit(self, rank): return 0
        def read_canonical_artifact_manifest(self, *args): return manifest
    monkeypatch.setattr(multinode_artifacts, 'validate_canonical_checkpoint_manifest',
                        lambda *a, **kw: calls.append('validate'))
    monkeypatch.setattr(multinode_runtime, 'collect_checkpoint_artifacts',
                        lambda *a, **kw: calls.append(('collect', kw['exit_codes'])))
    module.assemble_source_checkpoint(Runner(), tmp_path/'checkpoint-216', {})
    assert calls == (['validate'] if complete else [('collect', {0:0,1:0})])


@pytest.mark.parametrize('drift', [None, 'data', 'workers', 'commit', 'entropy', 'seed'])
def test_legacy_import_cross_checks_saved_args_and_both_node_records(tmp_path, drift):
    module = api()
    import torch
    from copy import deepcopy
    from types import SimpleNamespace
    from stage1.matched_resume import capture_matched_resume_profile
    checkpoint, contract = checkpoint_with_resume_contract(tmp_path)
    (checkpoint/'stage1_matched_resume.json').unlink()
    data = tmp_path/'TRAIN.jsonl'; data.write_bytes(b'samples')
    base = tmp_path/'base'; base.mkdir()
    (base/'config.json').write_bytes(b'{}')
    (base/'model.safetensors').write_bytes(b'base')
    worker = SimpleNamespace(**contract['profile']['worker'],
                             model=str(base), data_path=str(data), seed=42)
    training = SimpleNamespace(**contract['profile']['training'])
    torch.save(training, checkpoint/'training_args.bin')
    expected = capture_matched_resume_profile(worker, training)
    record = {'commit': '8'*40, 'training_data_sha256': expected['data_sha256'],
              'base_metadata_sha256': {'config.json': expected['base_sha256']['config.json']},
              'base_weights_sha256': {'model.safetensors': expected['base_sha256']['model.safetensors']},
              'profile': dict(expected['training']), 'execution_args': {},
              'ce_entropy_top_percent': 20., 'ce_weight': 1., 'sim_weight': 1.}
    records = [record, deepcopy(record)]
    if drift == 'data': records[0]['training_data_sha256'] = '0'*64
    if drift == 'workers': records[0]['profile']['dataloader_num_workers'] = 4
    if drift == 'commit': records[0]['commit'] = '7'*40
    if drift == 'entropy': worker.stage1_ce_entropy_top_percent = None
    if drift == 'seed': worker.seed = 43
    if drift:
        with pytest.raises(ValueError, match='saved|source|record|profile'):
            module.import_legacy_profile(checkpoint, worker, records, source_commit='8'*40)
    else:
        assert module.import_legacy_profile(checkpoint, worker, records, source_commit='8'*40) == expected
    assert not (checkpoint/'stage1_matched_resume.json').exists()


def test_existing_complete_replica_is_read_with_all_four_rank_ownership(tmp_path):
    from dataclasses import replace
    from src.multinode_runtime import redistribute_resume_checkpoint
    from tests.multinode.test_artifacts import FakeProductionTransferRunner, identity
    class Runner(FakeProductionTransferRunner):
        def read_artifact_manifest(self, machine_rank, checkpoint, runtime_identity, *, canonical_owner=False):
            assert canonical_owner, 'a complete local replica must admit ranks0/1 as well'
            return replace(self.canonical, node_rank=1,
                           entries=tuple(replace(e, node_rank=1) for e in self.canonical.entries))
    checkpoint = tmp_path/identity()['run_id']/'checkpoint-2'
    checkpoint.mkdir(parents=True)
    (checkpoint/'rng_state_0.pth').write_bytes(b'present')
    runner = Runner()
    redistribute_resume_checkpoint(runner, checkpoint_path=checkpoint,
        checkpoint_identity=identity(), requested_identity=identity(), destination=checkpoint)

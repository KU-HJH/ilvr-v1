import json
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer, TrainingArguments

from tests.stage1.test_run_cache import make_cache


def test_incomplete_epoch_never_enters_replay(tmp_path):
    from stage1.cache_lifecycle import Stage1CacheEpochCallback
    cache = make_cache(tmp_path)
    callback = Stage1CacheEpochCallback(cache, expected={'preprocess': (0,)}, control_dir=tmp_path / 'attempt')
    callback.on_epoch_end(SimpleNamespace(process_index=0, local_process_index=0),
                          SimpleNamespace(epoch=0.5), SimpleNamespace(should_training_stop=True))
    assert cache.phase.value == 0
    assert not (tmp_path / 'attempt/ready.json').exists()


class CacheCollator:
    def __init__(self, cache):
        self.cache = cache
    def __call__(self, rows):
        result = []
        for row in rows:
            if self.cache.phase.value == 0:
                self.cache.publish('preprocess', row['id'], {'value': torch.tensor(float(row['id']))})
            result.append(self.cache._load('preprocess', row['id'])['value'])
        return {'x': torch.stack(result), 'phase': torch.tensor(self.cache.phase.value)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.observed = []
    def forward(self, x, phase):
        self.observed.append(int(phase))
        return {'loss': ((self.weight * (x + 1)) ** 2).mean()}


def test_native_fractional_epoch_tail_rejects_before_second_iterator(tmp_path):
    from stage1.cache_lifecycle import Stage1CacheEpochCallback
    cache = make_cache(tmp_path)
    callback = Stage1CacheEpochCallback(cache, expected={'preprocess': tuple(range(7))},
                                       control_dir=tmp_path / 'attempt')
    model = TinyModel()
    args = TrainingArguments(output_dir=str(tmp_path / 'trainer'), use_cpu=True, report_to=[],
        max_steps=2, gradient_accumulation_steps=4, per_device_train_batch_size=1,
        save_strategy='no', dataloader_num_workers=4, remove_unused_columns=False,
        disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=[{'id': i} for i in range(7)],
                      data_collator=CacheCollator(cache), callbacks=[callback])
    # Represent the existing four-rank sharded-count geometry, retaining the
    # real Trainer/Accelerate iterator, accumulation and callback dispatch.
    trainer.num_examples = lambda loader: 26
    with pytest.raises(RuntimeError, match='incomplete first epoch'):
        trainer.train()
    assert model.observed == [0] * 6
    assert trainer.state.global_step == 1
    assert not (tmp_path / 'attempt/ready.json').exists()


@pytest.mark.parametrize('persistent', [False, True])
def test_epoch_two_waits_for_both_node_manifests(tmp_path, persistent):
    from stage1.cache_lifecycle import Stage1CacheEpochCallback, publish_release
    cache = make_cache(tmp_path)
    control = tmp_path / 'attempt-2'
    callback = Stage1CacheEpochCallback(cache, expected={'preprocess': (0, 1)}, control_dir=control)
    # A stale attempt must not release this iterator.
    (tmp_path / 'attempt-1').mkdir()
    (tmp_path / 'attempt-1/release.json').write_text('{}')
    model = TinyModel()
    errors = []
    def coordinator():
        try:
            deadline = time.monotonic() + 30
            while not (control / 'ready.json').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('no epoch boundary')
                time.sleep(.01)
            assert cache.phase.value == 0
            assert model.observed == [0, 0]
            # This simulates the coordinator completing node-wide coverage.
            manifest = cache.validate_coverage({'preprocess': (0, 1)})
            publish_release(control, manifest)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=coordinator, daemon=True)
    thread.start()
    args = TrainingArguments(output_dir=str(tmp_path / 'trainer'), use_cpu=True, report_to=[],
        num_train_epochs=2, per_device_train_batch_size=1, save_strategy='no',
        dataloader_num_workers=4, dataloader_persistent_workers=persistent,
        remove_unused_columns=False, disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=[{'id': 0}, {'id': 1}],
                      data_collator=CacheCollator(cache), callbacks=[callback])
    trainer.train()
    thread.join(timeout=2)
    assert not errors
    assert model.observed == [0, 0, 1, 1]


def test_shuffled_remote_record_is_available_to_workers(tmp_path):
    from stage1.cache_transport import exchange_stage1_cache, LocalCacheExchange
    a, b = (make_cache(tmp_path / node) for node in ('a', 'b'))
    a.publish('preprocess', 0, {'value': torch.tensor(10.)})
    b.publish('preprocess', 1, {'value': torch.tensor(11.)})
    exchange_stage1_cache(None, LocalCacheExchange({0: a, 1: b}), expected={'preprocess': (0, 1)})
    a.phase.value = 1
    loader = torch.utils.data.DataLoader([{'id': 1}, {'id': 0}], batch_size=1,
                                         num_workers=4, collate_fn=CacheCollator(a))
    assert [float(batch['x']) for batch in loader] == [11., 10.]


def test_replay_missing_record_fails_without_recompute(tmp_path):
    cache = make_cache(tmp_path)
    cache.phase.value = 1
    with pytest.raises(FileNotFoundError):
        cache.read('preprocess', 99)

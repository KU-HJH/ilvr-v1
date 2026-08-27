import json
from dataclasses import dataclass

import pytest
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler
from transformers import Trainer

from stage2.sampler import (
    STAGE2_SAMPLER_FILENAME,
    LatentBlockBalancedSampler,
)
from stage2.trainer import Stage2Trainer


@dataclass(frozen=True)
class CountedExample:
    latent_block_count: int


def examples_with_counts(counts):
    return [CountedExample(count) for count in counts]


def test_global_groups_have_one_latent_count_across_ranks():
    dataset = examples_with_counts([1, 2, 1, 3, 2, 3, 3])
    sampler = LatentBlockBalancedSampler(dataset, num_replicas=2, seed=42)

    indices = list(sampler)
    groups = [indices[offset : offset + 2] for offset in range(0, len(indices), 2)]

    assert all(
        len({dataset[index].latent_block_count for index in group}) == 1
        for group in groups
    )


def test_sampler_padding_and_identity_are_explicit():
    dataset = examples_with_counts([1, 1, 1, 2, 2])
    sampler = LatentBlockBalancedSampler(dataset, num_replicas=2, seed=42)

    assert len(sampler) == 6
    assert dict(sampler.identity.bucket_sizes) == {1: 3, 2: 2}
    assert sampler.identity.padded_samples == 1
    assert sampler.identity.to_dict() == {
        "schema_version": 1,
        "seed": 42,
        "world_size": 2,
        "dataset_size": 5,
        "bucket_sizes": {"1": 3, "2": 2},
        "padded_samples": 1,
    }


def test_epoch_is_deterministic_but_changes_shuffle_order():
    sampler = LatentBlockBalancedSampler(
        examples_with_counts([1, 1, 1, 1, 2, 2, 2, 2]),
        num_replicas=2,
        seed=42,
    )

    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    sampler.set_epoch(0)

    assert list(sampler) == epoch0
    assert epoch1 != epoch0


def test_batch_sampler_shard_assigns_aligned_global_groups_to_two_ranks():
    dataset = examples_with_counts([1, 2, 1, 2, 3, 3, 3])
    sampler = LatentBlockBalancedSampler(dataset, num_replicas=2, seed=42)
    batches = BatchSampler(sampler, batch_size=1, drop_last=False)
    rank_batches = [
        list(
            BatchSamplerShard(
                batches,
                num_processes=2,
                process_index=rank,
                split_batches=False,
            )
        )
        for rank in range(2)
    ]

    assert len(rank_batches[0]) == len(rank_batches[1])
    for left, right in zip(*rank_batches):
        assert (
            dataset[left[0]].latent_block_count
            == dataset[right[0]].latent_block_count
        )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([], "empty dataset"),
        ([0], "between 1 and 6"),
        ([7], "between 1 and 6"),
        ([True], "integer"),
    ],
)
def test_sampler_rejects_invalid_latent_count_datasets(counts, message):
    with pytest.raises(ValueError, match=message):
        LatentBlockBalancedSampler(
            examples_with_counts(counts), num_replicas=2, seed=42
        )


def test_sampler_rejects_negative_epoch():
    sampler = LatentBlockBalancedSampler(
        examples_with_counts([1, 1]), num_replicas=2, seed=42
    )

    with pytest.raises(ValueError, match="epoch must be non-negative"):
        sampler.set_epoch(-1)


def test_multigpu_trainer_publishes_sampler_identity(tmp_path, monkeypatch):
    trainer = object.__new__(Stage2Trainer)
    trainer.args = type(
        "Args",
        (),
        {
            "world_size": 2,
            "seed": 42,
            "output_dir": str(tmp_path),
            "should_save": True,
        },
    )()
    trainer.train_dataset = examples_with_counts([1, 1, 1, 2, 2])
    trainer.accelerator = type(
        "Accelerator",
        (),
        {
            "device": "cpu",
            "reduce": staticmethod(lambda value, reduction: value),
            "wait_for_everyone": staticmethod(lambda: None),
        },
    )()
    monkeypatch.setattr(
        Trainer,
        "get_train_dataloader",
        lambda selected: selected._get_train_sampler(),
    )

    dataloader = trainer.get_train_dataloader()
    payload = json.loads(
        (tmp_path / STAGE2_SAMPLER_FILENAME).read_text(encoding="utf-8")
    )

    assert isinstance(dataloader, LatentBlockBalancedSampler)
    assert payload["world_size"] == 2
    assert payload["dataset_size"] == 5
    assert payload["padded_samples"] == 1
    assert payload["global_step"] == 0

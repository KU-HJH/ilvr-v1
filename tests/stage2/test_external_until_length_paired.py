"""RAID Sept-17 length-paired grouping-two stream and PEFT DoRA norm selection."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from stage2 import sampler as sampling

RAID_SAMPLER = Path("/raid/workspace/temp2/ILVR/.worktrees/ours1/src/stage2/sampler.py")
RETAINED_MANIFEST = Path(
    "/root/workspace/ILVR/.worktrees/preference-alignment-completion-20260923/results/"
    "until-b05-offckpt-20260924-013155/stage2-length-manifest.json"
)
# Synthetic buckets with tied lengths; expected streams were produced by the
# unchanged RAID LatentBlockBalancedSampler(num_replicas=2, seed=42) given
# per-item sequence_length (stable sort keeps shuffled order among ties).
BLOCKS = [1, 2, 1, 3, 2, 1, 1, 3, 2, 2, 1, 3, 1, 2, 1, 1, 3, 2, 1, 2, 3, 1, 1]
LENGTHS = [50, 40, 50, 90, 41, 30, 70, 90, 39, 40, 10, 95, 50, 40, 60, 20, 80, 45, 50, 41, 90, 33, 50]
RAID_EPOCH_STREAMS = {
    0: [11, 11, 17, 17, 20, 7, 12, 14, 10, 15, 16, 3, 13, 9, 6, 6, 8, 1, 22, 0, 4, 19, 18, 2, 5, 21],
    1: [20, 3, 8, 9, 17, 17, 19, 4, 6, 6, 2, 12, 5, 21, 16, 7, 10, 15, 18, 22, 11, 11, 0, 14, 13, 1],
}


def global_order(sampler, processes=4):
    shards = [list(BatchSamplerShard(BatchSampler(sampler, 1, False), num_processes=processes,
              process_index=rank, even_batches=False)) for rank in range(processes)]
    return [batch[0] for row in zip(*shards) for batch in row]


@pytest.mark.parametrize("epoch", [0, 1])
def test_length_sorted_grouping_matches_raid_stream(epoch):
    # Fails if bucket sort, tie order, longest padding or generator order drifts.
    data = [SimpleNamespace(latent_block_count=block) for block in BLOCKS]
    sampler = sampling.LatentBlockBalancedSampler(
        data, num_replicas=2, seed=42, sequence_lengths=tuple(LENGTHS))
    sampler.set_epoch(epoch)
    assert list(sampler) == RAID_EPOCH_STREAMS[epoch]


# Produced by the base-commit (8e366e0) sampler: existing external_until_balanced
# checkpoints must keep exactly this unsorted, wrap-padded stream.
BASE_EPOCH_STREAMS = {
    0: [7, 3, 19, 1, 11, 16, 6, 12, 14, 18, 3, 20, 8, 4, 10, 14, 1, 17, 0, 15, 13, 9, 2, 22, 21, 5],
    1: [11, 3, 8, 17, 1, 8, 9, 13, 0, 6, 14, 22, 10, 5, 7, 20, 6, 21, 18, 15, 16, 7, 2, 12, 19, 4],
}


@pytest.mark.parametrize("epoch", [0, 1])
def test_unsorted_grouping_is_unchanged_without_lengths(epoch):
    data = [SimpleNamespace(latent_block_count=block) for block in BLOCKS]
    sampler = sampling.LatentBlockBalancedSampler(data, num_replicas=2, seed=42)
    sampler.set_epoch(epoch)
    assert list(sampler) == BASE_EPOCH_STREAMS[epoch]


def test_length_paired_sampler_requires_one_positive_length_per_item():
    data = [SimpleNamespace(latent_block_count=1)] * 8
    for lengths in (None, (5,) * 7, (5,) * 7 + (0,)):
        with pytest.raises(ValueError, match="sequence lengths"):
            sampling.ExternalUntilLengthPairedSampler(
                data, num_replicas=4, seed=42, gradient_accumulation_steps=4,
                sequence_lengths=lengths)


@pytest.mark.skipif(not (RAID_SAMPLER.is_file() and RETAINED_MANIFEST.is_file()),
                    reason="RAID sampler source or retained length manifest unavailable")
@pytest.mark.parametrize("epoch", [0, 1])
def test_four_rank_length_paired_stream_equals_raid_on_retained_cohort(epoch):
    manifest = json.loads(RETAINED_MANIFEST.read_text())
    retained = manifest["retained_indices"]
    blocks = [manifest["latent_block_counts"][index] for index in retained]
    lengths = tuple(manifest["lengths"][index] for index in retained)
    # The RAID sampler imports its own tree's modules; run it isolated there.
    program = (
        "import json, sys\nfrom types import SimpleNamespace as N\n"
        "from stage2.sampler import LatentBlockBalancedSampler as S\n"
        "blocks, lengths, epoch = json.load(sys.stdin)\n"
        "s = S([N(latent_block_count=b, sequence_length=l) for b, l in zip(blocks, lengths)],"
        " num_replicas=2, seed=42)\ns.set_epoch(epoch)\nprint(json.dumps(list(s)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], input=json.dumps([blocks, list(lengths), epoch]),
        capture_output=True, text=True, check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join(
            (str(RAID_SAMPLER.parents[1]), str(RAID_SAMPLER.parents[2] / "transformers" / "src")))},
    )
    local = sampling.ExternalUntilLengthPairedSampler(
        [SimpleNamespace(latent_block_count=b) for b in blocks], num_replicas=4, seed=42,
        gradient_accumulation_steps=4, sequence_lengths=lengths)
    local.set_epoch(epoch)
    expected = json.loads(completed.stdout)
    assert len(expected) == 3380
    assert global_order(local) == expected


def test_length_paired_identity_round_trips_and_rejects_unsorted_resume(tmp_path):
    data = [SimpleNamespace(latent_block_count=1)] * 3379
    paired = sampling.ExternalUntilLengthPairedSampler(
        data, num_replicas=4, seed=42, gradient_accumulation_steps=4,
        sequence_lengths=tuple(range(1, 3380)))
    unsorted = sampling.ExternalUntilBalancedSampler(
        data, num_replicas=4, seed=42, gradient_accumulation_steps=4)
    assert paired.identity.to_dict() == {
        "schema_version": 3, "seed": 42, "world_size": 4, "dataset_size": 3379,
        "bucket_sizes": {"1": 3379}, "padded_samples": 1,
        "policy": "external_until_length_paired", "gradient_accumulation_steps": 4,
        "update_accounting": "ceil", "padding_weight": 1, "reference_grouping": 2,
        "length_order": "sorted_adjacent_pairs", "padding_source": "longest",
        "lengths_sha256": sampling.sequence_lengths_sha256(tuple(range(1, 3380))),
    }
    sampling.write_stage2_sampler_sidecar(tmp_path / "paired", paired.identity, global_step=1)
    assert sampling.read_stage2_sampler_sidecar(tmp_path / "paired") == paired.identity
    sampling.validate_stage2_sampler_resume_identity(tmp_path / "paired", paired.identity)
    with pytest.raises(ValueError, match="policy"):
        sampling.validate_stage2_sampler_resume_identity(tmp_path / "paired", unsorted.identity)
    sampling.write_stage2_sampler_sidecar(tmp_path / "unsorted", unsorted.identity, global_step=1)
    with pytest.raises(ValueError, match="policy"):
        sampling.validate_stage2_sampler_resume_identity(tmp_path / "unsorted", paired.identity)
    # Same cohort geometry but different lengths would silently change the resumed stream.
    shifted = sampling.ExternalUntilLengthPairedSampler(
        data, num_replicas=4, seed=42, gradient_accumulation_steps=4,
        sequence_lengths=tuple(range(2, 3381)))
    with pytest.raises(ValueError, match="lengths_sha256"):
        sampling.validate_stage2_sampler_resume_identity(tmp_path / "paired", shifted.identity)


def test_trainer_builds_length_paired_sampler_from_supplied_lengths(tmp_path, monkeypatch):
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    from stage2.trainer import Stage2Trainer
    from step_preference.config import StepPreferenceConfig
    from tests.stage2.test_reference_random_sampler import ScalarModel
    from transformers import TrainingArguments

    monkeypatch.setattr(TrainingArguments, "world_size", property(lambda self: 4))
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
        per_device_train_batch_size=1, gradient_accumulation_steps=4, disable_tqdm=True)
    data = [SimpleNamespace(latent_block_count=1)] * 20
    lengths = tuple(LENGTHS[:20])
    trainer = Stage2Trainer(model=ScalarModel(), args=args, train_dataset=data,
        stage2_sampler_policy="external_until_length_paired",
        stage2_execution_profile="latest_external_until",
        stage2_sampler_lengths=lengths,
        step_preference_config=StepPreferenceConfig(),
        deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True, 2, None, None, 4))
    sampler = trainer._get_train_sampler()
    assert isinstance(sampler, sampling.ExternalUntilLengthPairedSampler)
    assert sampler.sequence_lengths == lengths
    assert trainer.model_accepts_loss_kwargs is False
    with pytest.raises(ValueError, match="sequence lengths"):
        Stage2Trainer(model=ScalarModel(), args=args, train_dataset=data,
            stage2_sampler_policy="external_until_length_paired",
            stage2_execution_profile="latest_external_until",
            step_preference_config=StepPreferenceConfig(),
            deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True, 2, None, None, 4))


@pytest.mark.parametrize("setting, expected", [("on", [True]), ("off", [False])])
def test_training_step_applies_dora_norm_cache_setting(tmp_path, monkeypatch, setting, expected):
    # Fails if "off" still installs the scope (i.e. RAID's PEFT norm path is not used).
    from contextlib import contextmanager
    import stage2.trainer as trainer_module
    from tests.stage2.test_reference_random_sampler import cpu_trainer

    observed = []

    @contextmanager
    def recording_scope(model, *, enabled=True):
        observed.append(enabled)
        yield

    monkeypatch.setattr(trainer_module, "dora_microbatch_scope", recording_scope)
    trainer = cpu_trainer(tmp_path, 1, policy="external_until_balanced", max_steps=1)
    trainer.stage2_execution_profile = "latest_external_until"
    trainer.stage2_dora_norm_cache = setting
    trainer.train()
    assert observed == expected


def test_cli_profile_keeps_explicit_length_paired_policy_and_dora_setting():
    from utils_deepseed import parse_args

    base = ["--stage", "stage2", "--stage2_execution_profile", "latest_external_until"]
    default = parse_args(base)
    assert default.stage2_sampler_policy == "external_until_balanced"
    assert default.stage2_dora_norm_cache == "on"
    synced = parse_args(base + ["--stage2_sampler_policy", "external_until_length_paired",
                                "--stage2_dora_norm_cache", "off"])
    assert synced.stage2_sampler_policy == "external_until_length_paired"
    assert synced.stage2_dora_norm_cache == "off"
    with pytest.raises(ValueError, match="external_until_length_paired requires latest_external_until"):
        parse_args(["--stage", "stage2", "--stage2_sampler_policy", "external_until_length_paired"])


def test_retained_sequence_lengths_follow_retained_dataset_order():
    from stage2.length_manifest import retained_sequence_lengths

    manifest = SimpleNamespace(lengths=(7, 9, 11, 13, 15), retained_indices=(0, 2, 4))
    assert retained_sequence_lengths(manifest, available=5) == (7, 11, 15)
    assert retained_sequence_lengths(manifest, available=3) == (7, 11)


def test_main_forwards_lengths_only_for_length_paired_policy():
    from main import stage2_sampler_trainer_kwargs

    assert stage2_sampler_trainer_kwargs("latent_balanced", train_sequence_lengths=(1,)) == {}
    assert stage2_sampler_trainer_kwargs("external_until_balanced", train_sequence_lengths=None) == {
        "stage2_sampler_policy": "external_until_balanced"}
    assert stage2_sampler_trainer_kwargs("external_until_length_paired", train_sequence_lengths=(3, 4)) == {
        "stage2_sampler_policy": "external_until_length_paired", "stage2_sampler_lengths": (3, 4)}
    with pytest.raises(ValueError, match="length manifest"):
        stage2_sampler_trainer_kwargs("external_until_length_paired", train_sequence_lengths=None)


def test_launcher_and_multinode_forward_sync_settings():
    from src.multinode import build_launch_plan
    from src.multinode_runtime import build_owned_tmux_commands, validated_stage2_environment
    from tests.stage2.test_launcher import dry_run_tokens, option_value, run_launcher

    tokens = dry_run_tokens(run_launcher(env_overrides={
        "STAGE2_EXECUTION_PROFILE": "latest_external_until",
        "STAGE2_SAMPLER_POLICY": "external_until_length_paired",
        "STAGE2_DORA_NORM_CACHE": "off",
    }))
    assert option_value(tokens, "--stage2_sampler_policy") == "external_until_length_paired"
    assert option_value(tokens, "--stage2_dora_norm_cache") == "off"
    assert "--stage2_dora_norm_cache" not in dry_run_tokens(run_launcher())
    bad = run_launcher(env_overrides={"STAGE2_DORA_NORM_CACHE": "maybe"})
    assert bad.returncode != 0 and "STAGE2_DORA_NORM_CACHE" in bad.stderr

    environment = validated_stage2_environment({
        "STAGE1_CHECKPOINT": "/checkpoint", "DATA_PATH": "/data", "IMAGE_ROOT": "/images",
        "STAGE2_EXECUTION_PROFILE": "latest_external_until",
        "STAGE2_SAMPLER_POLICY": "external_until_length_paired",
        "STAGE2_DORA_NORM_CACHE": "off",
    }, output_dir="/output", log_file="/log", deepspeed_config="/zero2")
    plan = build_launch_plan(stage="stage2", tuning_method="dora", zero_stage=2,
        run_id="length-paired-sync-test", main_process_port=29632)
    commands = build_owned_tmux_commands(plan, commit="a" * 40, stage_environment=environment)
    assert [dict(spec.environment)["STAGE2_DORA_NORM_CACHE"] for spec in commands.node_specs] == ["off", "off"]
    assert [dict(spec.environment)["STAGE2_SAMPLER_POLICY"] for spec in commands.node_specs] == [
        "external_until_length_paired"] * 2

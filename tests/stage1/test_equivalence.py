import copy
import hashlib
from dataclasses import replace
from itertools import islice
import subprocess

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard

from stage1.data import PreparedStage1Example
from stage1.equivalence import (
    CROSS_ARM_V1_POINTERS,
    SAME_ARM_V1_POINTERS,
    Stage1CanonicalBatchSampler,
    Stage1CanonicalDataset,
    Stage1ResolvedManifest,
    Stage1SampleOccurrence,
    canonical_json_bytes,
    canonical_tensor_sha256,
    compare_stage1_manifests,
    expected_stage1_assignment,
    materialize_stage1_schedule,
    runtime_source_sha256,
    validate_stage1_schedule,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64


@pytest.fixture
def resolved_manifest_mapping():
    return {
        "schema_version": "stage1-resolved-manifest-v1",
        "model": {
            "model_path": "checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct",
            "processor_path": "checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct",
            "model_revision": "immutable-model-revision",
            "processor_revision": "immutable-processor-revision",
            "model_files_sha256": _SHA_A,
            "processor_files_sha256": _SHA_B,
            "config_sha256": _SHA_A,
        },
        "code": {
            "git_commit": "1" * 40,
            "runtime_source_digest": _SHA_A,
            "software_versions": {
                "python": "3.11.13",
                "pytorch": "2.7.1+cu128",
                "transformers": "4.52.4",
                "peft": "0.15.2",
                "deepspeed": "0.17.1",
                "cuda": "12.8",
                "flash_attention": "2.8.3",
            },
        },
        "data": {
            "train_jsonl_path": "data/TRAIN.jsonl",
            "train_sha256": "5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8",
            "test_jsonl_path": "data/TEST.jsonl",
            "test_sha256": "6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618",
            "image_root": "data/images_comt",
            "image_files_sha256": _SHA_B,
            "source_rows": 3450,
            "canonical_epoch_slots": 3456,
            "padding_rule": "cyclic-prefix-v1",
            "ordering_rule": "source-jsonl-order-v1",
        },
        "batch": {
            "world_size": 4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_global_batch_size": 16,
            "rank_local_microbatch_grouping": [[0, 0], [1, 0], [2, 0], [3, 0]],
        },
        "schedule": {
            "epochs": 3,
            "seed": 42,
            "permutation_rule": "torch-randperm-seed-plus-epoch-v1",
            "optimizer_steps_per_epoch": 216,
            "total_optimizer_steps": 648,
            "logging_steps": 20,
            "checkpoint_steps": [216, 432, 648],
        },
        "tuning": {
            "method": "dora",
            "dora": {
                "rank": 32,
                "alpha": 64,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                    "up_proj", "down_proj",
                ],
                "dropout": 0.05,
            },
            "losses": {"ce_weight": 1.0, "similarity_weight": 1.0},
            "targets": {
                "coverage_probability": 0.9,
                "image_pool_size": 8,
                "helper_group_size": 256,
                "latent_size": 8,
                "alignment_noise": 0.01,
            },
            "maximum_sequence_length": 32768,
        },
        "optimizer": {
            "name": "adamw_torch_fused",
            "learning_rate": 1e-5,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        },
        "scheduler": {
            "name": "linear",
            "warmup_optimizer_steps": 10,
            "total_optimizer_steps": 648,
        },
        "ema": {
            "tau": 0.999,
            "initialization": "copy-trainable-parameters-v1",
            "update_rule": "post-optimizer-fp32-v1",
        },
        "precision": {
            "training_dtype": "bfloat16",
            "loss_reduction_dtype": "float32",
            "attention": "flash_attention_2",
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
        },
        "stochastic": {
            "rng_schema": "sample-occurrence-v1",
            "site_allowlist": ["dora_dropout", "alignment_target_noise"],
            "dropout_schema": "sample-keyed-dora-v1",
            "noise_schema": "sample-keyed-alignment-v1",
        },
        "deepspeed": {
            "source_digest": "c0922c8640778ae928c503f5bcacb070f2efaf34b7122431d9b4c2989e25debd",
            "settings": {
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 1,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 500000000,
                },
                "gradient_clipping": 1.0,
                "steps_per_print": 10,
                "wall_clock_breakdown": False,
            },
        },
        "evaluation": {
            "dataset_path": "data/TEST.jsonl",
            "dataset_sha256": "6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618",
            "shard_count": 4,
            "max_new_tokens": 4096,
            "temperature": 0.0,
            "top_p": 1.0,
            "scorer_identity": "stage1-eval-scorer-v1",
        },
        "diagnostics": {
            "bounded_optimizer_steps": None,
            "evidence_schema": "stage1-equivalence-evidence-v1",
        },
        "artifacts": {
            "run_id": "b1-main",
            "checkpoints": "checkpoints/b1-main",
            "results": "results/b1-main",
            "logs": "logs/b1-main",
        },
        "initialization": {
            "trainable_parameters": [
                {"name": "adapter.a", "dtype": "torch.float32", "shape": [2], "sha256": _SHA_A},
                {"name": "adapter.b", "dtype": "torch.float32", "shape": [2, 2], "sha256": _SHA_B},
            ],
            "ema_parameters": [
                {"name": "adapter.a", "dtype": "torch.float32", "shape": [2], "sha256": _SHA_A},
                {"name": "adapter.b", "dtype": "torch.float32", "shape": [2, 2], "sha256": _SHA_B},
            ],
        },
    }


@pytest.fixture
def rows_3450():
    return tuple(
        PreparedStage1Example(
            source_index=index,
            source_id=str(index),
            conversation=[],
            original_image_paths=(),
            helper_image_paths=(),
        )
        for index in range(3450)
    )


def test_canonical_view_pads_3450_rows_to_3456_without_losing_provenance(rows_3450):
    """Catches padding that changes source-row provenance or omits final slots."""
    view = Stage1CanonicalDataset(rows_3450)

    assert len(view) == 3456
    assert [view.source_index_for_slot(index) for index in range(3448, 3456)] == [
        3448,
        3449,
        0,
        1,
        2,
        3,
        4,
        5,
    ]


@pytest.mark.parametrize(("micro_batch", "ga"), ((1, 4), (2, 2)))
def test_schedule_has_exact_216_updates_and_closed_rank_grouping(
    rows_3450, micro_batch, ga
):
    """Catches a schedule whose rank-local accumulation no longer consumes four rows."""
    schedule = materialize_stage1_schedule(
        rows_3450,
        rank=2,
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=ga,
    )

    assert schedule.updates_per_epoch == 216
    assert schedule.total_updates == 648
    first = schedule.update(0)
    assert [item.permutation_offset for item in first.samples] == [2, 6, 10, 14]
    expected_grouping = (
        [(0, 0), (1, 0), (2, 0), (3, 0)]
        if micro_batch == 1
        else [(0, 0), (0, 1), (1, 0), (1, 1)]
    )
    assert [
        (item.rank_local_microbatch_ordinal, item.row_ordinal_within_microbatch)
        for item in first.samples
    ] == expected_grouping


def test_canonical_dataset_propagates_scheduled_occurrence(rows_3450):
    """Catches canonical indices being reduced to source indices before collation."""
    schedule = materialize_stage1_schedule(
        rows_3450,
        rank=2,
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
    )
    occurrence = schedule.update(0).samples[0]
    view = Stage1CanonicalDataset(rows_3450)

    example = view[replace(schedule.scheduled_index(occurrence), occurrence=occurrence)]

    assert example.source_index == occurrence.source_index
    assert example.occurrence == occurrence


def test_schedule_validation_rejects_source_provenance_drift(rows_3450):
    """Catches a schedule whose slot identity is retained but source provenance changes."""
    schedule = materialize_stage1_schedule(
        rows_3450,
        rank=2,
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
    )
    bad_schedule = replace(
        schedule,
        source_indices=(999, *schedule.source_indices[1:]),
    )

    with pytest.raises(ValueError, match="source_index"):
        validate_stage1_schedule(bad_schedule)


@pytest.mark.parametrize("start_global_step", (0, 3))
def test_schedule_validation_rejects_truncated_full_or_resumed_schedule(
    rows_3450, start_global_step
):
    """Catches validation accepting a partial canonical horizon as complete."""
    schedule = materialize_stage1_schedule(
        rows_3450,
        rank=2,
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
        start_global_step=start_global_step,
    )

    with pytest.raises(ValueError, match="exactly"):
        validate_stage1_schedule(replace(schedule, updates=schedule.updates[:-1]))


def test_canonical_schedule_rejects_noncanonical_seed_and_source_rows(rows_3450):
    """Catches mutable seed or source-row ordering from redefining the fixed protocol."""
    with pytest.raises(ValueError, match="seed=42"):
        materialize_stage1_schedule(
            rows_3450,
            rank=2,
            world_size=4,
            seed=99,
            epochs=3,
            micro_batch_size=1,
            gradient_accumulation_steps=4,
        )
    with pytest.raises(ValueError, match="exactly 3450"):
        Stage1CanonicalDataset(rows_3450[:-1])
    reordered = list(rows_3450)
    reordered[0] = replace(reordered[0], source_index=1)
    with pytest.raises(ValueError, match="zero-based source_index"):
        Stage1CanonicalDataset(reordered)
    with pytest.raises(ValueError, match="invalid epoch"):
        expected_stage1_assignment(
            epoch=3,
            update_in_epoch=0,
            rank=0,
            micro_batch_size=1,
        )


@pytest.mark.parametrize(
    "values",
    (
        (0, 0, 0, 0, 0, 0, 3, 3, 0),
        (0, 3450, 0, 3450, 1, 0, 0, 0, 0),
    ),
)
def test_occurrence_rejects_offset_ordinal_and_source_slot_mismatches(values):
    """Catches occurrence identity that is locally shaped but not protocol-canonical."""
    with pytest.raises(ValueError, match="canonical"):
        Stage1SampleOccurrence(*values)


@pytest.mark.parametrize(("micro_batch", "ga"), ((1, 4), (2, 2)))
def test_accelerate_shard_selects_each_rank_global_batches_once(
    rows_3450, micro_batch, ga
):
    """Catches an Accelerate shard changing the prescribed rank-local row order."""
    sampler = Stage1CanonicalBatchSampler(
        Stage1CanonicalDataset(rows_3450),
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=ga,
    )

    rank_two = BatchSamplerShard(
        sampler,
        num_processes=4,
        process_index=2,
        split_batches=False,
        even_batches=False,
    )
    selected = list(islice(rank_two, ga))

    assert rank_two.split_batches is False
    assert rank_two.even_batches is False
    assert [
        index.occurrence.permutation_offset for batch in selected for index in batch
    ] == [2, 6, 10, 14]


def test_canonical_json_uses_sorted_compact_utf8_bytes_and_stable_digest():
    """Catches locale, insertion order, or pretty-printing changing manifest identity."""
    encoded = canonical_json_bytes({"z": [3, 1.5], "a": "é"})

    assert encoded == b'{"a":"\\u00e9","z":[3,1.5]}'
    assert hashlib.sha256(encoded).hexdigest() == (
        "bfa83d4067052b477afeceea3dd278e933897951a733451a04f1e3232e78b268"
    )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_canonical_json_rejects_nonfinite_values(value):
    """Catches non-standard NaN or Infinity tokens entering a sealed JSON identity."""
    with pytest.raises((TypeError, ValueError), match="finite|JSON"):
        canonical_json_bytes({"value": value})


def test_canonical_tensor_hash_has_literal_dtype_shape_and_little_endian_identity():
    """Catches tensor hashing that omits metadata or hashes native-endian values."""
    tensor = torch.tensor([1, 256], dtype=torch.int16)

    assert canonical_tensor_sha256(tensor) == (
        "26702d8b9f19e7a40dbe4533aa4f4d9c1f4a408074a5ea6125fa9b4d0d434db5"
    )


@pytest.mark.parametrize(
    ("tensor", "expected"),
    (
        (
            torch.tensor([1.0, -2.0], dtype=torch.float32),
            "2e19b3139969771ce2390d505f6aae30c1bf255f2fa4ba7c47923497c05646f3",
        ),
        (
            torch.tensor([[1, 256]], dtype=torch.int16),
            "ca5d1a099077b602d1cda98748a8b8b2a3c6858aa11ea5e7af538f0ebb43ce00",
        ),
        (
            torch.tensor([1, 256], dtype=torch.int32),
            "02288f5aa34e0d7df9cd2047602b52fbd1f0a7b819a1e197c16e2acb1fa96a07",
        ),
    ),
)
def test_canonical_tensor_hash_changes_for_content_shape_and_dtype(tensor, expected):
    """Catches collisions caused by ignoring tensor content, shape, or dtype."""
    assert canonical_tensor_sha256(tensor) == expected


@pytest.mark.parametrize(
    "tensor",
    (
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (1,)),
    ),
)
def test_canonical_tensor_hash_rejects_nonfinite_and_nonstrided_tensors(tensor):
    """Catches unstable sparse layouts or nonfinite payloads receiving an identity."""
    with pytest.raises(ValueError, match="finite|strided"):
        canonical_tensor_sha256(tensor)


def test_manifest_round_trip_is_closed_and_has_literal_digest(resolved_manifest_mapping):
    """Catches manifest construction dropping fields or hashing a noncanonical form."""
    manifest = Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)

    assert manifest.to_mapping() == resolved_manifest_mapping
    assert manifest.digest == (
        "fdd29bd042ab227dfe5a79a0c2e41101f8e0aab489aa95d83af702190f545a7d"
    )


@pytest.mark.parametrize(
    ("section", "field"),
    (("model", "model_path"), ("deepspeed", "settings"), ("batch", "world_size")),
)
def test_manifest_rejects_missing_child_keys(resolved_manifest_mapping, section, field):
    """Catches absent schema fields being silently treated like defaults or nulls."""
    del resolved_manifest_mapping[section][field]

    with pytest.raises(ValueError, match="missing"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("model", "host_model_cache", "/tmp/model"),
        ("batch", "derived_batch_copy", 16),
        ("deepspeed", "train_micro_batch_size_per_gpu", 1),
        ("deepspeed", "gradient_accumulation_steps", 4),
    ),
)
def test_manifest_rejects_extra_and_duplicated_batch_child_keys(
    resolved_manifest_mapping, section, field, value
):
    """Catches unknown fields and duplicated DeepSpeed batch settings evading comparison."""
    resolved_manifest_mapping[section][field] = value

    with pytest.raises(ValueError, match="extra"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


@pytest.mark.parametrize(
    ("pointer", "value"),
    (
        ("/schedule/epochs", 3.0),
        ("/batch/world_size", True),
        ("/optimizer/learning_rate", "1e-5"),
        ("/diagnostics/bounded_optimizer_steps", 0),
        ("/optimizer/epsilon", float("nan")),
    ),
)
def test_manifest_rejects_wrong_types_and_nonfinite_values(
    resolved_manifest_mapping, pointer, value
):
    """Catches JSON-compatible but semantically invalid scalar substitutions."""
    target = resolved_manifest_mapping
    parts = pointer.removeprefix("/").split("/")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value

    with pytest.raises((TypeError, ValueError), match="type|integer|finite|bounded"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


def test_manifest_rejects_duplicate_json_keys_before_schema_validation():
    """Catches a JSON parser accepting the last of two conflicting semantic values."""
    duplicate = b'{"schema_version":"first","schema_version":"second"}'

    with pytest.raises(ValueError, match="duplicate JSON key"):
        Stage1ResolvedManifest.from_mapping(duplicate)


@pytest.mark.parametrize(
    "grouping",
    (
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [2, 0], [9, 0]],
    ),
)
def test_manifest_rejects_invalid_b1_rank_local_grouping(
    resolved_manifest_mapping, grouping
):
    """Catches arbitrary or B=2 grouping being relabeled as a valid B=1 manifest."""
    resolved_manifest_mapping["batch"]["rank_local_microbatch_grouping"] = grouping

    with pytest.raises(ValueError, match="grouping"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


def test_cross_arm_manifest_accepts_exact_batch_and_artifact_differences(
    resolved_manifest_mapping,
):
    """Catches omission of any approved B=1/B=2 pointer from cross-arm normalization."""
    right = copy.deepcopy(resolved_manifest_mapping)
    right["batch"].update(
        {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "rank_local_microbatch_grouping": [[0, 0], [0, 1], [1, 0], [1, 1]],
        }
    )
    right["artifacts"] = {
        "run_id": "b2-main",
        "checkpoints": "checkpoints/b2-main",
        "results": "results/b2-main",
        "logs": "logs/b2-main",
    }

    comparison = compare_stage1_manifests(
        resolved_manifest_mapping, right, profile="cross_arm_v1"
    )

    assert CROSS_ARM_V1_POINTERS == (
        "/batch/per_device_train_batch_size",
        "/batch/gradient_accumulation_steps",
        "/batch/rank_local_microbatch_grouping",
        "/artifacts/run_id",
        "/artifacts/checkpoints",
        "/artifacts/results",
        "/artifacts/logs",
    )
    assert comparison.equal
    assert comparison.left_normalized_digest == comparison.right_normalized_digest


@pytest.mark.parametrize(
    "pointer",
    (
        "/optimizer/learning_rate",
        "/ema/tau",
        "/stochastic/rng_schema",
        "/data/train_sha256",
        "/code/runtime_source_digest",
        "/deepspeed/settings/zero_optimization/stage",
    ),
)
def test_cross_arm_manifest_rejects_semantic_drift(
    resolved_manifest_mapping, pointer
):
    """Catches a semantic field being accidentally added to the cross-arm whitelist."""
    right = copy.deepcopy(resolved_manifest_mapping)
    target = right
    parts = pointer.removeprefix("/").split("/")
    for part in parts[:-1]:
        target = target[part]
    old = target[parts[-1]]
    target[parts[-1]] = (
        "b" * 64
        if pointer.endswith("sha256") or pointer.endswith("digest")
        else "changed"
        if isinstance(old, str)
        else old + 0.0001
        if isinstance(old, float)
        else 2
    )

    with pytest.raises(ValueError, match="non-whitelisted"):
        compare_stage1_manifests(
            resolved_manifest_mapping, right, profile="cross_arm_v1"
        )


def test_same_arm_manifest_accepts_only_diagnostic_bound_and_artifact_differences(
    resolved_manifest_mapping,
):
    """Catches same-arm replay comparison normalizing batch fields or missing replay fields."""
    replay = copy.deepcopy(resolved_manifest_mapping)
    replay["diagnostics"]["bounded_optimizer_steps"] = 1
    replay["artifacts"] = {
        "run_id": "b1-replay",
        "checkpoints": "checkpoints/b1-replay",
        "results": "results/b1-replay",
        "logs": "logs/b1-replay",
    }

    comparison = compare_stage1_manifests(
        resolved_manifest_mapping, replay, profile="same_arm_v1"
    )

    assert SAME_ARM_V1_POINTERS == (
        "/artifacts/run_id",
        "/artifacts/checkpoints",
        "/artifacts/results",
        "/artifacts/logs",
        "/diagnostics/bounded_optimizer_steps",
    )
    assert comparison.equal


def test_manifest_comparison_rejects_unknown_profile(resolved_manifest_mapping):
    """Catches a typo selecting an empty or permissive normalization profile."""
    with pytest.raises(ValueError, match="unknown manifest profile"):
        compare_stage1_manifests(
            resolved_manifest_mapping,
            resolved_manifest_mapping,
            profile="cross-arm-v2",
        )


def test_manifest_runtime_source_hashes_sorted_relative_paths_and_rejects_dirty_files(
    tmp_path,
):
    """Catches path-order dependence or hashing tracked runtime bytes unlike recorded HEAD."""
    repository = tmp_path / "repo"
    repository.mkdir()
    first = repository / "a.py"
    second = repository / "src" / "b.sh"
    second.parent.mkdir()
    first.write_bytes(b"print('a')\n")
    second.write_bytes(b"echo b\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "a.py", "src/b.sh"], check=True)
    subprocess.run(
        [
            "git", "-C", str(repository), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    expected = "5e9a3ea6994148e6fff42763e174a1b61c89336eedd5265fecd547227872e96d"

    assert runtime_source_sha256([second, first]) == expected
    first.write_bytes(b"print('dirty')\n")
    with pytest.raises(ValueError, match="dirty tracked runtime source"):
        runtime_source_sha256([first, second])

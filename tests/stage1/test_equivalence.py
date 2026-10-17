import ast
from collections.abc import MutableMapping
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from itertools import islice
from pathlib import Path
import subprocess

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.checkpoint import checkpoint

from stage1.data import PreparedStage1Example
from stage1.equivalence import (
    CROSS_ARM_V1_POINTERS,
    SAME_ARM_V1_POINTERS,
    Stage1CanonicalBatchSampler,
    Stage1CanonicalDataset,
    Stage1ResolvedManifest,
    Stage1SampleOccurrence,
    Stage1StochasticContext,
    SampleStableStage1Dropout,
    activate_stage1_stochastic_context,
    canonical_json_bytes,
    canonical_tensor_sha256,
    compare_stage1_manifests,
    configure_stage1_backend_settings,
    expected_stage1_assignment,
    materialize_stage1_resolved_manifest,
    materialize_stage1_schedule,
    normalized_manifest_bytes,
    read_stage1_backend_settings,
    runtime_source_sha256,
    stage1_checkpoint_context_fn,
    stage1_sample_seed,
    validate_stage1_schedule,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_FIXED_BACKEND_SETTINGS = {
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
}


def _raw_backend_settings():
    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _set_raw_backend_settings(settings):
    torch.backends.cudnn.deterministic = settings["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = settings["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = settings["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(settings["float32_matmul_precision"])


@pytest.fixture(autouse=True)
def restore_backend_settings_after_test():
    before = _raw_backend_settings()
    try:
        yield
    finally:
        _set_raw_backend_settings(before)


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
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
        "stochastic": {
            "rng_schema": "stage1-sample-rng-v1",
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


def test_canonical_tensor_hash_accepts_finite_scalar_parameter():
    """Catches dtype-byte views rejecting rank-zero trainable parameters."""
    tensor = torch.tensor(1.5, dtype=torch.float32)

    assert canonical_tensor_sha256(tensor) == (
        "f30b2bed8d34527c8c9541fb867942e85c6eaf81b07b73430f0303b6ec3bdc1c"
    )


def test_stage1_sample_seed_has_literal_schema_digest_and_key_separation():
    """Catches process hash(), field omission, or non-little-endian seed decoding."""
    assert stage1_sample_seed(42, 17, 3450, "site.a") == 480346063979829922
    assert stage1_sample_seed(42, 17, 0, "site.a") == 2497506465054421059
    assert stage1_sample_seed(42, 18, 3450, "site.a") != 480346063979829922
    assert stage1_sample_seed(42, 17, 3450, "site.b") != 480346063979829922


def test_stage1_backend_helpers_observe_configure_and_restore_fixed_settings():
    """Catches manifest backend fields being constants disconnected from runtime state."""
    noncanonical = {
        "cudnn_deterministic": False,
        "cudnn_benchmark": True,
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
        "float32_matmul_precision": "high",
    }
    _set_raw_backend_settings(noncanonical)

    assert read_stage1_backend_settings() == noncanonical
    assert configure_stage1_backend_settings() == _FIXED_BACKEND_SETTINGS
    assert _raw_backend_settings() == _FIXED_BACKEND_SETTINGS


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("sample-stable Stage 1 dropout requires a CUDA witness")
    return torch.device("cuda", 0)


_DROPOUT_SITE = "base_model.model.model.layers.0.self_attn.q_proj.lora_dropout.default"


def _independent_dropout_mask(*, slot, length, width, hidden, p, device):
    payload = json.dumps(
        {
            "base_seed": 42,
            "optimizer_global_step": 17,
            "canonical_slot_id": slot,
            "stable_site_name": _DROPOUT_SITE,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    seed &= (1 << 63) - 1
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    mask = torch.ones((width, hidden), device=device)
    draws = torch.rand(
        (length, hidden),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    mask[:length] = draws.ge(p).float() / (1.0 - p)
    return mask


def _run_dropout_rows(*, batch_sizes, lengths, device):
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25).to(device)
    dropout.train()
    slots = (9, 3450)
    results = []
    cursor = 0
    for batch_size in batch_sizes:
        row_lengths = lengths[cursor : cursor + batch_size]
        row_slots = slots[cursor : cursor + batch_size]
        width = max(row_lengths)
        values = torch.ones((batch_size, width, 4), device=device)
        context = Stage1StochasticContext(
            base_seed=42,
            optimizer_global_step=17,
            rows=tuple(zip(row_slots, row_lengths)),
            expected_sites=(_DROPOUT_SITE,),
        )
        with activate_stage1_stochastic_context(context):
            output = dropout(values)
        results.extend(output.unbind(0))
        cursor += batch_size
    return tuple(results)


def test_sample_dropout_is_grouping_and_right_padding_invariant(cuda_device):
    """Catches batch grouping or a neighbor's padding shifting a row RNG stream."""
    separate = _run_dropout_rows(
        batch_sizes=(1, 1), lengths=(5, 8), device=cuda_device
    )
    packed = _run_dropout_rows(
        batch_sizes=(2,), lengths=(5, 8), device=cuda_device
    )

    torch.testing.assert_close(packed[0][:5], separate[0][:5], rtol=0, atol=0)
    torch.testing.assert_close(packed[1][:8], separate[1][:8], rtol=0, atol=0)
    assert torch.equal(packed[0][5:], torch.ones_like(packed[0][5:]))
    torch.testing.assert_close(
        packed[0],
        _independent_dropout_mask(
            slot=9,
            length=5,
            width=8,
            hidden=4,
            p=0.25,
            device=cuda_device,
        ),
        rtol=0,
        atol=0,
    )


def test_checkpoint_recompute_regenerates_same_dropout_and_rejects_second_call(
    cuda_device,
):
    """Catches checkpoint recompute consuming RNG or accepting a duplicate site call."""
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25).to(cuda_device)
    dropout.train()
    values = torch.ones((1, 6, 4), device=cuda_device, requires_grad=True)
    seen = []
    context = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 6),),
        expected_sites=(_DROPOUT_SITE,),
    )

    def block(tensor):
        dropped = dropout(tensor)
        seen.append(dropped.detach().clone())
        return dropped * dropped

    with activate_stage1_stochastic_context(context):
        checkpointed = checkpoint(
            block,
            values,
            use_reentrant=False,
            context_fn=stage1_checkpoint_context_fn,
        )
        checkpointed.sum().backward()
        assert len(seen) == 2
        torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
        with pytest.raises(AttributeError):
            context.phase
        # Creating another pair proves recompute restored its nested token and
        # left the training-step parent token active in this original context.
        _, recompute_context = stage1_checkpoint_context_fn()
        with recompute_context:
            with pytest.raises(RuntimeError, match="second.*recompute"):
                dropout(values)


def test_stochastic_context_does_not_expose_mutable_invocation_bookkeeping():
    """Catches the immutable public spec retaining a path to mutable counts."""
    context = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25)
    dropout.train()
    values = torch.ones((1, 5, 4))

    assert set(type(context).__slots__) == {
        "base_seed",
        "optimizer_global_step",
        "rows",
        "expected_sites",
        "__weakref__",
    }

    def assert_immutable_identity_tree(value):
        assert not isinstance(value, MutableMapping)
        if type(value) is tuple:
            for child in value:
                assert_immutable_identity_tree(child)
        else:
            assert type(value) in (int, str)

    for slot_name in type(context).__slots__:
        if slot_name != "__weakref__":
            assert_immutable_identity_tree(getattr(context, slot_name))

    with activate_stage1_stochastic_context(context):
        dropout(values)
        with pytest.raises(AttributeError):
            context._invocation_state._Stage1InvocationState__original.clear()
        with pytest.raises(RuntimeError, match="second.*original"):
            dropout(values)


def test_stochastic_context_rejects_nested_but_not_equal_fresh_spec():
    """Catches lifecycle tracking conflating nested use with equal spec values."""
    outer = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )
    equal_fresh = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25)
    dropout.train()
    values = torch.ones((1, 5, 4))
    assert outer == equal_fresh

    with activate_stage1_stochastic_context(outer):
        dropout(values)
        with pytest.raises(RuntimeError, match="nested"):
            with activate_stage1_stochastic_context(equal_fresh):
                pass

    with activate_stage1_stochastic_context(equal_fresh):
        dropout(values)


def _assert_checkpoint_failure_left_no_stochastic_context(context):
    with pytest.raises(AttributeError):
        context.phase
    with pytest.raises(RuntimeError, match="missing.*stochastic context"):
        stage1_checkpoint_context_fn()
    with pytest.raises(RuntimeError, match="stale"):
        with activate_stage1_stochastic_context(context):
            pass

    fresh_context = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=18,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )
    fresh_dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25)
    fresh_dropout.train()
    with activate_stage1_stochastic_context(fresh_context):
        fresh_dropout(torch.ones((1, 5, 4)))


def test_nonreentrant_checkpoint_original_exception_cleans_stochastic_context():
    """Catches an original-forward exception leaking its token, phase, or counts."""
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25)
    dropout.train()
    values = torch.ones((1, 5, 4), requires_grad=True)
    context = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )

    def failing_block(tensor):
        dropout(tensor)
        raise RuntimeError("original checkpoint failure")

    with pytest.raises(RuntimeError, match="original checkpoint failure"):
        with activate_stage1_stochastic_context(context):
            checkpoint(
                failing_block,
                values,
                use_reentrant=False,
                context_fn=stage1_checkpoint_context_fn,
            )

    _assert_checkpoint_failure_left_no_stochastic_context(context)


def test_nonreentrant_checkpoint_recompute_exception_cleans_stochastic_context():
    """Catches a recompute exception leaking its nested token, phase, or counts."""
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25)
    dropout.train()
    values = torch.ones((1, 5, 4), requires_grad=True)
    calls = []
    context = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )

    def failing_recompute_block(tensor):
        dropped = dropout(tensor)
        calls.append(dropped.detach().clone())
        if len(calls) == 2:
            raise RuntimeError("recompute checkpoint failure")
        return dropped * dropped

    with pytest.raises(RuntimeError, match="recompute checkpoint failure"):
        with activate_stage1_stochastic_context(context):
            checkpointed = checkpoint(
                failing_recompute_block,
                values,
                use_reentrant=False,
                context_fn=stage1_checkpoint_context_fn,
            )
            checkpointed.sum().backward()

    assert len(calls) == 2
    torch.testing.assert_close(calls[0], calls[1], rtol=0, atol=0)
    _assert_checkpoint_failure_left_no_stochastic_context(context)


def test_sample_dropout_rejects_missing_stale_shape_and_site_context():
    """Catches silent global-RNG fallback or inferred row/layout ownership."""
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25)
    dropout.train()
    values = torch.ones((1, 5, 4))

    with pytest.raises(RuntimeError, match="missing.*stochastic context"):
        dropout(values)

    stale_context = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )
    with activate_stage1_stochastic_context(stale_context):
        dropout(values)
    with pytest.raises(RuntimeError, match="missing.*stochastic context"):
        dropout(values)
    with pytest.raises(RuntimeError, match="stale"):
        with activate_stage1_stochastic_context(stale_context):
            pass

    wrong_rank = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )
    with pytest.raises(ValueError, match=r"\[B,T,H\]"):
        with activate_stage1_stochastic_context(wrong_rank):
            dropout(torch.ones((5, 4)))

    wrong_rows = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5), (10, 5)),
        expected_sites=(_DROPOUT_SITE,),
    )
    with pytest.raises(ValueError, match="row count"):
        with activate_stage1_stochastic_context(wrong_rows):
            dropout(values)

    wrong_length = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 6),),
        expected_sites=(_DROPOUT_SITE,),
    )
    with pytest.raises(ValueError, match="valid length"):
        with activate_stage1_stochastic_context(wrong_length):
            dropout(values)

    wrong_site = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=("different.site",),
    )
    with pytest.raises(RuntimeError, match="unregistered.*site"):
        with activate_stage1_stochastic_context(wrong_site):
            dropout(values)


def test_sample_dropout_rejects_sparse_layout_before_invocation_accounting():
    """Catches a rank-three sparse activation consuming the site's one call."""
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25)
    dropout.train()
    context = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [0]]),
        torch.tensor([1.0]),
        size=(1, 5, 4),
    )
    dense = torch.ones((1, 5, 4))

    with activate_stage1_stochastic_context(context):
        with pytest.raises(ValueError, match="strided"):
            dropout(sparse)
        dropout(dense)


def test_zero_probability_dropout_preserves_identity_and_context_counts():
    """Catches zero-dropout PEFT behavior changing or bypassing topology counts."""
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.0)
    dropout.train()
    values = torch.randn((1, 5, 4))

    assert dropout(values) is values
    context = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE,),
    )
    with activate_stage1_stochastic_context(context):
        assert dropout(values) is values


def test_stochastic_context_rejects_duplicate_sites_and_missing_original_call():
    """Catches ambiguous site identity or a supported topology silently going unused."""
    with pytest.raises(ValueError, match="duplicate.*site"):
        Stage1StochasticContext(
            base_seed=42,
            optimizer_global_step=17,
            rows=((9, 5),),
            expected_sites=(_DROPOUT_SITE, _DROPOUT_SITE),
        )

    context = Stage1StochasticContext(
        base_seed=42,
        optimizer_global_step=17,
        rows=((9, 5),),
        expected_sites=(_DROPOUT_SITE, "missing.site"),
    )
    with pytest.raises(FrozenInstanceError):
        context.rows = ((10, 5),)
    dropout = SampleStableStage1Dropout(_DROPOUT_SITE, 0.25)
    dropout.train()
    with pytest.raises(RuntimeError, match="original count exactly one"):
        with activate_stage1_stochastic_context(context):
            dropout(torch.ones((1, 5, 4)))


def test_padded_duplicate_receives_different_sample_dropout_seed():
    """Catches a padded occurrence reusing source-row rather than slot identity."""
    original = stage1_sample_seed(42, 17, 0, _DROPOUT_SITE)
    padded_duplicate = stage1_sample_seed(42, 17, 3450, _DROPOUT_SITE)

    assert original != padded_duplicate


def test_stage1_stochastic_inventory_identifies_only_registered_and_deferred_sources():
    """Catches a new direct random API entering Stage 1 without source ownership."""
    random_apis = {
        "rand",
        "rand_like",
        "randn",
        "randn_like",
        "randint",
        "randperm",
        "normal",
        "bernoulli",
        "multinomial",
        "poisson",
        "dropout",
    }
    observed = set()
    for path in sorted((Path("src") / "stage1").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in random_apis
            ):
                continue
            owners = []
            parent = node
            while parent in parents:
                parent = parents[parent]
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    owners.append(parent.name)
                elif isinstance(parent, ast.ClassDef):
                    owners.append(parent.name)
                    break
            observed.add(
                (
                    path.relative_to(Path("src")).as_posix(),
                    ".".join(reversed(owners)),
                    ast.unparse(node.func),
                )
            )

    assert observed == {
        (
            "stage1/equivalence.py",
            "canonical_epoch_permutation",
            "torch.randperm",
        ),
        (
            "stage1/equivalence.py",
            "SampleStableStage1Dropout.forward",
            "torch.rand",
        ),
        # Opt-in v5 owns F/packed source-coordinate Bernoulli draws.
        (
            "stage1/v5_execution.py",
            "V5Execution.dropout_mask",
            "torch.rand",
        ),
        # Task 7 must replace this exact interim legacy source with
        # alignment_noise_bank before the final three-source inventory gate.
        (
            "stage1/trainer.py",
            "CustomTrainerStage1._alignment_loss",
            "torch.randn_like",
        ),
    }


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
        "1807d864c942ef2d01400f186a4a263bf78f7c7709b43c9ecf1f00ce4745c5c4"
    )


def test_manifest_rejects_non_stage1_sample_rng_schema(resolved_manifest_mapping):
    """Catches a caller sealing a mutable-stream or stale occurrence RNG identity."""
    resolved_manifest_mapping["stochastic"]["rng_schema"] = "sample-occurrence-v1"

    with pytest.raises(ValueError, match="RNG schema"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


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


@pytest.fixture
def manifest_runtime_inputs(tmp_path, resolved_manifest_mapping):
    configure_stage1_backend_settings()
    repository = tmp_path / "runtime-repository"
    files = {
        "src/core.py": b"VALUE = 1\n",
        "eval.py": b"print('eval')\n",
        "run_stage1_training.sh": b"#!/bin/sh\necho train\n",
        "run_multinode_evaluation.sh": b"#!/bin/sh\necho eval\n",
        "qwen/modeling.py": b"MODEL = 1\n",
        "qwen/README.txt": b"not runtime Python\n",
        "model/config.json": b'{"model":"tiny"}\n',
        "model/tokenizer.json": b"{}\n",
        "model/weights.bin": b"weights\n",
        "data/TRAIN.jsonl": b'{"train":1}\n',
        "data/TEST.jsonl": b'{"test":1}\n',
        "data/images/a.png": b"image-a\n",
        "configs/deepspeed.json": (
            b'{"train_micro_batch_size_per_gpu":"auto","train_batch_size":"auto",'
            b'"gradient_accumulation_steps":"auto","bf16":{"enabled":true},'
            b'"zero_optimization":{"stage":1,"overlap_comm":true,'
            b'"contiguous_gradients":true,"reduce_bucket_size":500000000},'
            b'"gradient_clipping":1.0,"steps_per_print":10,'
            b'"wall_clock_breakdown":false}\n'
        ),
    }
    for relative_path, content in files.items():
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(repository), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    semantic_values = copy.deepcopy(resolved_manifest_mapping)
    semantic_values["model"] = {
        key: "fabricated"
        for key in semantic_values["model"]
    }
    semantic_values["code"]["git_commit"] = "f" * 40
    semantic_values["code"]["runtime_source_digest"] = "f" * 64
    semantic_values["code"]["software_versions"] = {
        key: "fabricated"
        for key in semantic_values["code"]["software_versions"]
    }
    for key in ("train_sha256", "test_sha256", "image_files_sha256"):
        semantic_values["data"][key] = "f" * 64
    semantic_values["deepspeed"]["source_digest"] = "f" * 64
    semantic_values["initialization"] = {
        "trainable_parameters": [], "ema_parameters": []
    }
    return {
        "semantic_values": semantic_values,
        "repository_root": repository,
        "git_commit": commit,
        "qwen_source_path": repository / "qwen/modeling.py",
        "model_path": repository / "model",
        "processor_path": repository / "model",
        "model_config_path": repository / "model/config.json",
        "model_revision": "model-revision",
        "processor_revision": "processor-revision",
        "train_jsonl_path": repository / "data/TRAIN.jsonl",
        "test_jsonl_path": repository / "data/TEST.jsonl",
        "image_root": repository / "data/images",
        "referenced_image_paths": [repository / "data/images/a.png"],
        "deepspeed_config_path": repository / "configs/deepspeed.json",
        "world_size": 4,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "trainable_parameters": {
            "adapter.z": torch.tensor([2.0]),
            "adapter.a": torch.tensor(1.5),
        },
        "ema_parameters": {
            "adapter.z": torch.tensor([2.0]),
            "adapter.a": torch.tensor(1.5),
        },
    }


def test_manifest_materializer_derives_runtime_identities_instead_of_trusting_mapping(
    manifest_runtime_inputs,
):
    """Catches schema-valid fabricated identity strings being sealed as runtime truth."""
    payload = materialize_stage1_resolved_manifest(**manifest_runtime_inputs).to_mapping()

    assert payload["model"] == {
        "model_path": "model",
        "processor_path": "model",
        "model_revision": "model-revision",
        "processor_revision": "processor-revision",
        "model_files_sha256": "5a2ecbd170dc12a285e956dfacaa0302fece68324c6795a1a149cc94d9fd710f",
        "processor_files_sha256": "5a2ecbd170dc12a285e956dfacaa0302fece68324c6795a1a149cc94d9fd710f",
        "config_sha256": "0b01122dba8fdc9845d4c7bd9509b3684e47545b8684e7ce9c8df5ccfa714984",
    }
    assert payload["data"]["train_sha256"] == (
        "3d44981f1b4cae2b8e7a51e8b041e4c269a71e41fca6f11b5fc68c82cb39e115"
    )
    assert payload["data"]["test_sha256"] == (
        "76ee56ae9188eecf6269efd1d512d2fb3711524325959737507ea87047355a64"
    )
    assert payload["data"]["image_files_sha256"] == (
        "68e99a243f571f27c35618aa0ac1f54732810b9edf76d38b3604487778386938"
    )
    assert payload["batch"] == {
        "world_size": 4,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch_size": 16,
        "rank_local_microbatch_grouping": [[0, 0], [0, 1], [1, 0], [1, 1]],
    }
    assert {
        key: payload["precision"][key] for key in _FIXED_BACKEND_SETTINGS
    } == _FIXED_BACKEND_SETTINGS
    assert payload["deepspeed"]["source_digest"] == (
        "ac9adfe99efe8c21745014ee696b490be1e680379077babefc9111186dac68ec"
    )
    assert "train_micro_batch_size_per_gpu" not in payload["deepspeed"]["settings"]
    assert [item["name"] for item in payload["initialization"]["trainable_parameters"]] == [
        "adapter.a", "adapter.z"
    ]
    assert payload["initialization"]["trainable_parameters"][0]["sha256"] == (
        "f30b2bed8d34527c8c9541fb867942e85c6eaf81b07b73430f0303b6ec3bdc1c"
    )
    assert payload["initialization"]["trainable_parameters"][1]["sha256"] == (
        "504d91fad0e0c894e56d89102dac38a3754a718b06fe4f493d3a80ab8aa1eafc"
    )
    assert payload["evaluation"]["dataset_sha256"] == payload["data"]["test_sha256"]
    assert payload["code"]["software_versions"]["pytorch"] != "fabricated"


def test_manifest_materializer_overwrites_fabricated_backend_settings(
    manifest_runtime_inputs,
):
    """Catches semantic_values being trusted for observed backend state."""
    manifest_runtime_inputs["semantic_values"]["precision"].update(
        {
            "cudnn_deterministic": False,
            "cudnn_benchmark": True,
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
            "float32_matmul_precision": "high",
        }
    )

    payload = materialize_stage1_resolved_manifest(
        **manifest_runtime_inputs
    ).to_mapping()

    assert {
        key: payload["precision"][key] for key in _FIXED_BACKEND_SETTINGS
    } == _FIXED_BACKEND_SETTINGS


def test_manifest_materializer_derives_fixed_sample_rng_schema(
    manifest_runtime_inputs,
):
    """Catches semantic_values selecting the stochastic seed identity."""
    manifest_runtime_inputs["semantic_values"]["stochastic"]["rng_schema"] = (
        "caller-selected-schema"
    )

    payload = materialize_stage1_resolved_manifest(
        **manifest_runtime_inputs
    ).to_mapping()

    assert payload["stochastic"]["rng_schema"] == "stage1-sample-rng-v1"


def test_manifest_materializer_rejects_observed_backend_drift(
    manifest_runtime_inputs,
):
    """Catches a launch sealing a manifest before fixed backends are configured."""
    torch.backends.cudnn.allow_tf32 = True

    with pytest.raises(RuntimeError, match="backend settings"):
        materialize_stage1_resolved_manifest(**manifest_runtime_inputs)


def test_manifest_materializer_binds_fixed_source_closure_to_recorded_commit(
    manifest_runtime_inputs,
):
    """Catches production source identity omitting required scripts or Qwen source files."""
    payload = materialize_stage1_resolved_manifest(**manifest_runtime_inputs).to_mapping()

    assert payload["code"]["git_commit"] == manifest_runtime_inputs["git_commit"]
    assert payload["code"]["runtime_source_digest"] == (
        "7b541743f0eb21b5b3cab7498afb841c4a73223ce790623072c3498a1c2c38ac"
    )
    surprise = manifest_runtime_inputs["repository_root"] / "src/surprise.py"
    surprise.write_bytes(b"SURPRISE = True\n")
    with pytest.raises(ValueError, match="fixed runtime source closure"):
        materialize_stage1_resolved_manifest(**manifest_runtime_inputs)


def test_manifest_materializer_rejects_recorded_commit_mismatch(manifest_runtime_inputs):
    """Catches clean runtime bytes from a different HEAD being bound to an old commit."""
    repository = manifest_runtime_inputs["repository_root"]
    note = repository / "note.txt"
    note.write_text("next commit\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "note.txt"], check=True)
    subprocess.run(
        [
            "git", "-C", str(repository), "-c", "user.name=Test",
            "-c", "user.email=test@example.com", "commit", "-qm", "next",
        ],
        check=True,
    )

    with pytest.raises(ValueError, match="recorded git_commit.*HEAD"):
        materialize_stage1_resolved_manifest(**manifest_runtime_inputs)


def test_manifest_materializer_rejects_deleted_tracked_runtime_source(
    manifest_runtime_inputs,
):
    """Catches a deleted recorded source file disappearing from current-path discovery."""
    (manifest_runtime_inputs["repository_root"] / "src/core.py").unlink()

    with pytest.raises(ValueError, match="deleted|closure|dirty"):
        materialize_stage1_resolved_manifest(**manifest_runtime_inputs)


def test_manifest_materializer_rejects_missing_explicit_qwen_source(
    manifest_runtime_inputs,
):
    """Catches a missing loaded Qwen file falling through to an empty prefix closure."""
    manifest_runtime_inputs["qwen_source_path"] = (
        manifest_runtime_inputs["repository_root"] / "qwen/missing.py"
    )

    with pytest.raises(ValueError, match="Qwen source.*Python file"):
        materialize_stage1_resolved_manifest(**manifest_runtime_inputs)


@pytest.mark.parametrize("relative_path", ("qwen", "qwen/README.txt"))
def test_manifest_materializer_rejects_non_python_file_or_directory_qwen_source(
    manifest_runtime_inputs, relative_path
):
    """Catches directory-prefix or non-Python membership replacing one loaded Qwen file."""
    manifest_runtime_inputs["qwen_source_path"] = (
        manifest_runtime_inputs["repository_root"] / relative_path
    )

    with pytest.raises(ValueError, match="Qwen source.*Python file"):
        materialize_stage1_resolved_manifest(**manifest_runtime_inputs)


def test_manifest_materializer_rejects_noninteger_resolved_deepspeed_batch(
    manifest_runtime_inputs,
):
    """Catches float-equal DeepSpeed batch values bypassing exact integer resolution."""
    path = manifest_runtime_inputs["deepspeed_config_path"]
    source = json.loads(path.read_text(encoding="utf-8"))
    source["train_micro_batch_size_per_gpu"] = 2.0
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="train_micro_batch_size_per_gpu"):
        materialize_stage1_resolved_manifest(**manifest_runtime_inputs)


def test_manifest_materializer_rejects_noncanonical_world_size(
    manifest_runtime_inputs,
):
    """Catches a schema-valid effective batch being derived for a nonprotocol world size."""
    manifest_runtime_inputs["world_size"] = 2

    with pytest.raises(ValueError, match="world_size=4"):
        materialize_stage1_resolved_manifest(**manifest_runtime_inputs)


# Independent schema oracle: this table is deliberately maintained without reading
# the production _MANIFEST_SCHEMA so a missing production entry fails these tests.
MANIFEST_OBJECT_POINTERS = (
    "/",
    "/model",
    "/code",
    "/code/software_versions",
    "/data",
    "/batch",
    "/schedule",
    "/tuning",
    "/tuning/dora",
    "/tuning/losses",
    "/tuning/targets",
    "/optimizer",
    "/scheduler",
    "/ema",
    "/precision",
    "/stochastic",
    "/deepspeed",
    "/deepspeed/settings",
    "/deepspeed/settings/bf16",
    "/deepspeed/settings/zero_optimization",
    "/evaluation",
    "/diagnostics",
    "/artifacts",
    "/initialization",
    "/initialization/trainable_parameters/0",
    "/initialization/ema_parameters/0",
)
CROSS_ARM_POINTER_ORACLE = (
    "/batch/per_device_train_batch_size",
    "/batch/gradient_accumulation_steps",
    "/batch/rank_local_microbatch_grouping",
    "/artifacts/run_id",
    "/artifacts/checkpoints",
    "/artifacts/results",
    "/artifacts/logs",
)
SAME_ARM_POINTER_ORACLE = (
    "/artifacts/run_id",
    "/artifacts/checkpoints",
    "/artifacts/results",
    "/artifacts/logs",
    "/diagnostics/bounded_optimizer_steps",
)
MANIFEST_ARRAY_ITEM_POINTERS = (
    "/batch/rank_local_microbatch_grouping/0",
    "/batch/rank_local_microbatch_grouping/0/0",
    "/schedule/checkpoint_steps/0",
    "/tuning/dora/target_modules/0",
    "/optimizer/betas/0",
    "/stochastic/site_allowlist/0",
    "/initialization/trainable_parameters/0",
    "/initialization/trainable_parameters/0/shape/0",
    "/initialization/ema_parameters/0",
    "/initialization/ema_parameters/0/shape/0",
)
MANIFEST_REQUIRED_POINTERS = (
    "/schema_version",
    "/model", "/code", "/data", "/batch", "/schedule", "/tuning",
    "/optimizer", "/scheduler", "/ema", "/precision", "/stochastic",
    "/deepspeed", "/evaluation", "/diagnostics", "/artifacts", "/initialization",
    *(
        f"/model/{field}"
        for field in (
            "model_path", "processor_path", "model_revision", "processor_revision",
            "model_files_sha256", "processor_files_sha256", "config_sha256",
        )
    ),
    "/code/git_commit", "/code/runtime_source_digest", "/code/software_versions",
    *(
        f"/code/software_versions/{field}"
        for field in (
            "python", "pytorch", "transformers", "peft", "deepspeed", "cuda",
            "flash_attention",
        )
    ),
    *(
        f"/data/{field}"
        for field in (
            "train_jsonl_path", "train_sha256", "test_jsonl_path", "test_sha256",
            "image_root", "image_files_sha256", "source_rows",
            "canonical_epoch_slots", "padding_rule", "ordering_rule",
        )
    ),
    *(
        f"/batch/{field}"
        for field in (
            "world_size", "per_device_train_batch_size",
            "gradient_accumulation_steps", "effective_global_batch_size",
            "rank_local_microbatch_grouping",
        )
    ),
    *(
        f"/schedule/{field}"
        for field in (
            "epochs", "seed", "permutation_rule", "optimizer_steps_per_epoch",
            "total_optimizer_steps", "logging_steps", "checkpoint_steps",
        )
    ),
    "/tuning/method", "/tuning/dora", "/tuning/losses", "/tuning/targets",
    "/tuning/maximum_sequence_length",
    *(f"/tuning/dora/{field}" for field in ("rank", "alpha", "target_modules", "dropout")),
    *(f"/tuning/losses/{field}" for field in ("ce_weight", "similarity_weight")),
    *(
        f"/tuning/targets/{field}"
        for field in (
            "coverage_probability", "image_pool_size", "helper_group_size",
            "latent_size", "alignment_noise",
        )
    ),
    *(
        f"/optimizer/{field}"
        for field in (
            "name", "learning_rate", "betas", "epsilon", "weight_decay",
            "max_grad_norm",
        )
    ),
    *(f"/scheduler/{field}" for field in ("name", "warmup_optimizer_steps", "total_optimizer_steps")),
    *(f"/ema/{field}" for field in ("tau", "initialization", "update_rule")),
    *(
        f"/precision/{field}"
        for field in (
            "training_dtype", "loss_reduction_dtype", "attention",
            "gradient_checkpointing", "gradient_checkpointing_use_reentrant",
            "cudnn_deterministic", "cudnn_benchmark",
            "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
            "float32_matmul_precision",
        )
    ),
    *(
        f"/stochastic/{field}"
        for field in ("rng_schema", "site_allowlist", "dropout_schema", "noise_schema")
    ),
    "/deepspeed/source_digest", "/deepspeed/settings",
    "/deepspeed/settings/bf16", "/deepspeed/settings/zero_optimization",
    "/deepspeed/settings/gradient_clipping", "/deepspeed/settings/steps_per_print",
    "/deepspeed/settings/wall_clock_breakdown",
    "/deepspeed/settings/bf16/enabled",
    *(
        f"/deepspeed/settings/zero_optimization/{field}"
        for field in ("stage", "overlap_comm", "contiguous_gradients", "reduce_bucket_size")
    ),
    *(
        f"/evaluation/{field}"
        for field in (
            "dataset_path", "dataset_sha256", "shard_count", "max_new_tokens",
            "temperature", "top_p", "scorer_identity",
        )
    ),
    "/diagnostics/bounded_optimizer_steps", "/diagnostics/evidence_schema",
    *(f"/artifacts/{field}" for field in ("run_id", "checkpoints", "results", "logs")),
    "/initialization/trainable_parameters", "/initialization/ema_parameters",
    *(
        f"/initialization/trainable_parameters/0/{field}"
        for field in ("name", "dtype", "shape", "sha256")
    ),
    *(
        f"/initialization/ema_parameters/0/{field}"
        for field in ("name", "dtype", "shape", "sha256")
    ),
)


def _manifest_pointer_parent(payload, pointer):
    parts = pointer.removeprefix("/").split("/")
    target = payload
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    return target, parts[-1]


def _manifest_pointer_value(payload, pointer):
    if pointer == "/":
        return payload
    parent, leaf = _manifest_pointer_parent(payload, pointer)
    return parent[int(leaf)] if isinstance(parent, list) else parent[leaf]


@pytest.mark.parametrize("pointer", MANIFEST_REQUIRED_POINTERS)
def test_manifest_closed_schema_rejects_every_required_node_deletion(
    resolved_manifest_mapping, pointer
):
    """Catches any independently specified owned node becoming optional."""
    parent, leaf = _manifest_pointer_parent(resolved_manifest_mapping, pointer)
    if isinstance(parent, list):
        del parent[int(leaf)]
    else:
        del parent[leaf]

    with pytest.raises((TypeError, ValueError), match="missing|exactly|must|grouping"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


@pytest.mark.parametrize("pointer", MANIFEST_OBJECT_POINTERS)
def test_manifest_closed_schema_rejects_extra_key_at_every_owned_object(
    resolved_manifest_mapping, pointer
):
    """Catches any owned object silently becoming an arbitrary grouping bucket."""
    target = _manifest_pointer_value(resolved_manifest_mapping, pointer)
    target["review_extra_key"] = "forbidden"

    with pytest.raises(ValueError, match="extra"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


@pytest.mark.parametrize(
    "pointer",
    tuple(
        pointer
        for pointer in MANIFEST_REQUIRED_POINTERS
        if pointer != "/diagnostics/bounded_optimizer_steps"
    ),
)
def test_manifest_closed_schema_rejects_null_for_every_nonnullable_node(
    resolved_manifest_mapping, pointer
):
    """Catches absent-versus-null substitution at every nonnullable owned node."""
    parent, leaf = _manifest_pointer_parent(resolved_manifest_mapping, pointer)
    if isinstance(parent, list):
        parent[int(leaf)] = None
    else:
        parent[leaf] = None

    with pytest.raises((TypeError, ValueError), match="type|must|schema_version"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


def test_manifest_closed_schema_rejects_wrong_type_for_nullable_bounded_steps(
    resolved_manifest_mapping,
):
    """Catches the one nullable field accepting values beyond null or integer one."""
    resolved_manifest_mapping["diagnostics"]["bounded_optimizer_steps"] = "1"

    with pytest.raises(ValueError, match="null or integer 1"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


@pytest.mark.parametrize("pointer", MANIFEST_ARRAY_ITEM_POINTERS)
def test_manifest_closed_schema_rejects_null_for_every_array_item_schema(
    resolved_manifest_mapping, pointer
):
    """Catches list element schemas accepting null or the wrong structural kind."""
    parent, leaf = _manifest_pointer_parent(resolved_manifest_mapping, pointer)
    if isinstance(parent, list):
        parent[int(leaf)] = None
    else:
        parent[leaf] = None

    with pytest.raises((TypeError, ValueError), match="type|must|grouping"):
        Stage1ResolvedManifest.from_mapping(resolved_manifest_mapping)


@pytest.mark.parametrize("pointer", CROSS_ARM_POINTER_ORACLE)
def test_cross_arm_profile_exercises_each_permitted_pointer_independently(
    resolved_manifest_mapping, pointer
):
    """Catches one of the seven cross-arm pointers being omitted from normalization."""
    right = copy.deepcopy(resolved_manifest_mapping)
    if pointer.startswith("/batch/"):
        right["batch"].update(
            {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "rank_local_microbatch_grouping": [[0, 0], [0, 1], [1, 0], [1, 1]],
            }
        )
    else:
        parent, leaf = _manifest_pointer_parent(right, pointer)
        parent[leaf] = f"changed-{leaf}"

    assert _manifest_pointer_value(resolved_manifest_mapping, pointer) != (
        _manifest_pointer_value(right, pointer)
    )
    assert compare_stage1_manifests(
        resolved_manifest_mapping, right, profile="cross_arm_v1"
    ).equal
    normalized = json.loads(
        normalized_manifest_bytes(resolved_manifest_mapping, "cross_arm_v1")
    )
    assert _manifest_pointer_value(normalized, pointer) == "<approved-difference>"


@pytest.mark.parametrize("pointer", SAME_ARM_POINTER_ORACLE)
def test_same_arm_profile_exercises_each_permitted_pointer_independently(
    resolved_manifest_mapping, pointer
):
    """Catches one of the five same-arm pointers being omitted from normalization."""
    right = copy.deepcopy(resolved_manifest_mapping)
    parent, leaf = _manifest_pointer_parent(right, pointer)
    parent[leaf] = 1 if pointer.startswith("/diagnostics/") else f"changed-{leaf}"

    assert _manifest_pointer_value(resolved_manifest_mapping, pointer) != (
        _manifest_pointer_value(right, pointer)
    )
    assert compare_stage1_manifests(
        resolved_manifest_mapping, right, profile="same_arm_v1"
    ).equal
    normalized = json.loads(
        normalized_manifest_bytes(resolved_manifest_mapping, "same_arm_v1")
    )
    assert _manifest_pointer_value(normalized, pointer) == "<approved-difference>"

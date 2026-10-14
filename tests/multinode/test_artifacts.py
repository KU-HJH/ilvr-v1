import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _source_root in (
    _REPOSITORY_ROOT / "transformers" / "src",
    _REPOSITORY_ROOT / "src",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from src.multinode_artifacts import (
    ArtifactEntry,
    NodeArtifactManifest,
    build_inference_pull_command,
    build_inference_manifest,
    build_node_manifest,
    build_pull_command,
    build_push_command,
    prepare_inference_copy,
    prepare_resume_transfer,
    publish_files_list,
    publish_node_manifest,
    reserve_inference_destination,
    derive_model_state_layout,
    observe_checkpoint_identity,
    select_inference_entries,
    select_resume_entries,
    validate_canonical_destination,
    validate_collection,
    validate_resume_identity,
    validate_transfer_destination,
    validate_reserved_inference_destination,
    validate_checkpoint_directory,
    validate_loader_artifact,
)
from src.multinode_runtime import collect_checkpoint_artifacts
from step_preference.config import StepPreferenceConfig
from step_preference.dil_identity import build_dil_identity, read_dil_identity
from src.multinode_runtime import (
    prepare_inference_artifacts,
    redistribute_resume_checkpoint,
    validate_resume_path,
)
from src.multinode import _checkpoint_artifact_identity, build_launch_plan
from src.multinode_runtime import SubprocessCoordinatorRunner, build_owned_tmux_commands


def identity(**changes):
    values = {
        "stage": "stage1",
        "tuning_method": "lora",
        "zero_stage": 3,
        "deepspeed_config_digest": "a" * 64,
        "world_size": 4,
        "rank_mapping": {"0": [0, 1], "1": [2, 3]},
        "global_step": 2,
        "model_state_layout": "partitioned",
        "optimizer": True,
        "scheduler": True,
        "rng_ranks": [0, 1, 2, 3],
        "stage_sidecar": "stage1_peft_metadata.json",
        "commit": "b" * 40,
        "run_id": "stage1-lora-zero3-four-gpu",
    }
    values.update(changes)
    return values


def full_identity(**changes):
    values = identity(tuning_method="full", stage_sidecar="stage1_ema_teacher.pt")
    values.update(changes)
    return values


def write_observed_full_checkpoint(checkpoint, *, values=None, model_rank=2):
    values = full_identity() if values is None else values
    tag = checkpoint / f"global_step{values['global_step']}"
    tag.mkdir(parents=True, exist_ok=True)
    if values["model_state_layout"] == "partitioned":
        (tag / f"zero_pp_rank_{model_rank}_mp_rank_00_model_states.pt").write_bytes(
            b"model-state"
        )
    else:
        (tag / "mp_rank_00_model_states.pt").write_bytes(b"model-state")
    (checkpoint / "latest").write_text(
        f"global_step{values['global_step']}", encoding="utf-8"
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": values["global_step"]}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "deepspeed-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enabled": True,
                "zero_stage": values["zero_stage"],
                "config_path": (
                    f"/root/workspace/ILVR/configs/"
                    f"deepspeed_zero{values['zero_stage']}.json"
                ),
                "config_digest": values["deepspeed_config_digest"],
                "world_size": values["world_size"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "config.json").write_text(
        json.dumps(standard_qwen_config()) + "\n", encoding="utf-8"
    )
    for name in (
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        (checkpoint / name).write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"full-model")
    (checkpoint / "stage1_ema_teacher.pt").write_bytes(b"ema")
    return values


def strict_manifest(node_rank, *, prefix="bf16_", omit=(), changes=None):
    owned = (0, 1) if node_rank == 0 else (2, 3)
    values = identity(**(changes or {}))
    entries = []
    for rank in owned:
        entries.extend(
            (
                ArtifactEntry(
                    Path(
                        f"global_step2/{prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                    ),
                    10 + rank,
                    node_rank,
                ),
                ArtifactEntry(Path(f"rng_state_{rank}.pth"), 20 + rank, node_rank),
                ArtifactEntry(
                    Path(
                        f"global_step2/zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
                    ),
                    30 + rank,
                    node_rank,
                ),
            )
        )
    if node_rank == 0:
        entries.extend(
            ArtifactEntry(Path(name), size, 0)
            for name, size in (
                ("latest", 12),
                ("trainer_state.json", 40),
                ("scheduler.pt", 50),
                ("deepspeed-runtime.json", 60),
                ("stage1_ema_teacher.pt", 70),
                ("stage1_peft_metadata.json", 80),
                ("adapter_config.json", 81),
                ("adapter_model.safetensors", 82),
                ("ilvr_token_rows.safetensors", 83),
                ("config.json", 84),
                ("preprocessor_config.json", 85),
                ("tokenizer_config.json", 86),
                ("tokenizer.json", 87),
            )
        )
    omitted = set(omit)
    entries = tuple(entry for entry in entries if entry.relative_path.as_posix() not in omitted)
    return NodeArtifactManifest(
        schema_version=1,
        run_id=values["run_id"],
        node_rank=node_rank,
        process_ranks=owned,
        entries=entries,
        runtime_identity=values,
        completion_marker=True,
    )


def strict_manifests(**kwargs):
    return strict_manifest(0, **kwargs), strict_manifest(1, **kwargs)


def entry(rank, *, size=8, node_rank=None):
    owner = rank // 2 if node_rank is None else node_rank
    return ArtifactEntry(
        Path(f"checkpoint-2/global_step2/zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"),
        size,
        owner,
    )


def stage1_inference_entries(node_rank=0):
    return (
        ArtifactEntry(Path("adapter_config.json"), 31, node_rank),
        ArtifactEntry(Path("adapter_model.safetensors"), 32, node_rank),
        ArtifactEntry(Path("stage1_peft_metadata.json"), 33, node_rank),
        ArtifactEntry(Path("ilvr_token_rows.safetensors"), 34, node_rank),
        ArtifactEntry(Path("preprocessor_config.json"), 35, node_rank),
        ArtifactEntry(Path("config.json"), 36, node_rank),
        ArtifactEntry(Path("tokenizer_config.json"), 37, node_rank),
        ArtifactEntry(Path("tokenizer.json"), 38, node_rank),
    )


def loader_asset_entries(node_rank=0):
    return (
        ArtifactEntry(Path("config.json"), 36, node_rank),
        ArtifactEntry(Path("preprocessor_config.json"), 35, node_rank),
        ArtifactEntry(Path("tokenizer_config.json"), 37, node_rank),
        ArtifactEntry(Path("tokenizer.json"), 38, node_rank),
    )


def qwen_test_architecture():
    return {
        "architectures": [],
        "decoder": {
            "attention_dropout": 0.0,
            "head_dim": 4,
            "hidden_act": "silu",
            "hidden_size": 8,
            "intermediate_size": 16,
            "max_position_embeddings": 32768,
            "max_window_layers": 80,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rope_scaling": {
                "mrope_section": [1, 1, 0],
                "rope_type": "default",
                "type": "default",
            },
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-5,
            "sliding_window": 4096,
            "tie_word_embeddings": False,
            "use_sliding_window": False,
            "vocab_size": 128,
        },
        "model_type": "qwen2_5_vl",
        "multimodal_token_ids": {
            "image_token_id": 80,
            "video_token_id": 82,
            "vision_end_token_id": None,
            "vision_start_token_id": 81,
        },
        "vision": {
            "depth": 1,
            "fullatt_block_indexes": [0],
            "hidden_act": "silu",
            "hidden_size": 8,
            "in_channels": 3,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 8,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "tokens_per_second": 4,
            "window_size": 2,
        },
    }


def standard_qwen_config():
    return {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.49.0",
    }


def loader_identity(stage, method):
    sidecar = {
        ("stage1", "full"): "stage1_ema_teacher.pt",
        ("stage1", "lora"): "stage1_peft_metadata.json",
        ("stage1", "dora"): "stage1_peft_metadata.json",
        ("stage2", "full"): "stage2-sampler.json",
        ("stage2", "lora"): "stage2_peft_metadata.json",
        ("stage2", "dora"): "stage2_peft_metadata.json",
    }[(stage, method)]
    return identity(stage=stage, tuning_method=method, stage_sidecar=sidecar)


def write_loader_artifact(directory, *, stage, method, sparse_stage2=False):
    from stage1.checkpointing import Stage1PeftMetadata
    from stage2.checkpointing import Stage2PeftMetadata

    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "config.json": json.dumps(standard_qwen_config()).encode("utf-8") + b"\n",
        "preprocessor_config.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
    }.items():
        (directory / name).write_bytes(payload)
    if method == "full":
        (directory / "model.safetensors").write_bytes(b"full")
        if stage == "stage1":
            (directory / "stage1_ema_teacher.pt").write_bytes(b"ema")
        else:
            (directory / "stage2-sampler.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "seed": 7,
                        "world_size": 4,
                        "dataset_size": 4,
                        "bucket_sizes": {"1": 4},
                        "padded_samples": 0,
                        "global_step": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return
    (directory / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"adapter")
    if stage == "stage1":
        metadata = Stage1PeftMetadata.from_dict(
            {
                "schema_version": 2,
                "stage": "stage1",
                "base_model_reference": "tiny/base",
                "base_model_architecture": qwen_test_architecture(),
                "base_vocab_size": 125,
                "pretrained_model_vocab_size": 128,
                "resized_vocab_size": 128,
                "latent_token_ids": [125, 126, 127],
                "latent_token_strings": [
                    "<|latent_pad|>",
                    "<|latent_start|>",
                    "<|latent_end|>",
                ],
                "latent_size": 8,
                "ce_weight": 1.0,
                "sim_weight": 0.5,
                "coverage_p": 0.9,
                "image_pool_k": 8,
                "tuning_method": method,
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "requested_target_suffixes": ["q_proj"],
                "resolved_target_names": ["model.layers.0.self_attn.q_proj"],
                "input_row_format": "peft_trainable_tokens_v1",
                "output_row_format": "ilvr_sparse_output_rows_v1",
                "output_row_shape": [3, 8],
                "peft_version": "0.17.1",
                "repository_format_version": 1,
            }
        )
        (directory / "stage1_peft_metadata.json").write_text(
            json.dumps(metadata.to_dict()) + "\n", encoding="utf-8"
        )
        (directory / "ilvr_token_rows.safetensors").write_bytes(b"rows")
        return
    token_row_source = "sparse_adapter" if sparse_stage2 else "base"
    metadata = Stage2PeftMetadata.from_dict(
        {
            "schema_version": 4,
            "stage": "stage2",
            "tuning_method": method,
            "base_model_reference": "tiny/stage1",
            "base_model_architecture": qwen_test_architecture(),
            "tokenizer_base_vocab_size": 125,
            "pretrained_model_vocab_size": 125,
            "resized_vocab_size": 128,
            "latent_size": 8,
            "latent_start_token_id": 126 if sparse_stage2 else 91,
            "latent_pad_token_id": 125 if sparse_stage2 else 90,
            "latent_end_token_id": 127 if sparse_stage2 else 92,
            "supervise_latent_boundaries": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "requested_target_modules": ["q_proj"],
            "resolved_target_module_names": ["model.layers.0.self_attn.q_proj"],
            "peft_version": "0.17.1",
            "token_row_source": token_row_source,
            "stage1_adapter_reference": (
                "tiny/stage1-adapter" if sparse_stage2 else None
            ),
        }
    )
    (directory / "stage2_peft_metadata.json").write_text(
        json.dumps(metadata.to_dict()) + "\n", encoding="utf-8"
    )
    if sparse_stage2:
        import torch
        from safetensors.torch import save_file

        save_file(
            {
                "token_ids": torch.tensor([125, 126, 127], dtype=torch.int64),
                "output_delta": torch.zeros((3, 8), dtype=torch.float32),
            },
            directory / "ilvr_token_rows.safetensors",
            metadata={"format_version": "1", "training_dtype": "float32"},
        )


def loader_manifest(directory, *, stage, method):
    return NodeArtifactManifest(
        schema_version=1,
        run_id=identity()["run_id"],
        node_rank=0,
        process_ranks=(0, 1, 2, 3),
        entries=tuple(
            ArtifactEntry(path.relative_to(directory), path.stat().st_size, 0)
            for path in sorted(directory.iterdir())
            if path.is_file()
        ),
        runtime_identity=loader_identity(stage, method),
        completion_marker=True,
    )


def manifest_for(
    ranks,
    *,
    node_rank=None,
    runtime_identity=None,
    run_id="stage1-lora-zero3-four-gpu",
    completion_marker=True,
    extra_entries=(),
):
    ranks = tuple(ranks)
    node = ranks[0] // 2 if node_rank is None else node_rank
    return NodeArtifactManifest(
        schema_version=1,
        run_id=run_id,
        node_rank=node,
        process_ranks=(0, 1) if node == 0 else (2, 3),
        entries=tuple(entry(rank, node_rank=node) for rank in ranks)
        + tuple(extra_entries),
        runtime_identity=identity() if runtime_identity is None else runtime_identity,
        completion_marker=completion_marker,
    )


def complete_manifests(**changes):
    return (
        manifest_for((0, 1), **changes),
        manifest_for((2, 3), **changes),
    )


def launch_plan():
    return build_launch_plan(
        stage="stage1",
        tuning_method="lora",
        zero_stage=3,
        run_id="stage1-lora-zero3-four-gpu",
        main_process_port=29631,
    )


def test_manifest_records_size_and_rank_without_weight_hash(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / identity()["run_id"] / "checkpoint-2"
    shard = checkpoint / "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"rank-two")
    values = write_observed_full_checkpoint(checkpoint)

    original_read_bytes = Path.read_bytes

    def reject_weight_read(path, *args, **kwargs):
        if path.suffix in {".pt", ".pth", ".safetensors", ".bin"}:
            raise AssertionError("checkpoint weights must never be opened")
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", reject_weight_read)
    manifest = build_node_manifest(
        checkpoint,
        node_rank=1,
        process_ranks=(2, 3),
        run_id="stage1-lora-zero3-four-gpu",
        runtime_identity=values,
        checkpoint_root=checkpoint_root,
    )

    assert ArtifactEntry(
        relative_path=Path(
            "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"
        ),
        byte_size=8,
        node_rank=1,
    ) in manifest.entries
    assert next(
        item for item in manifest.entries if item.relative_path == shard.relative_to(checkpoint)
    ).process_rank == 2


def test_noncanonical_node_manifest_accepts_only_its_deepspeed_shards(tmp_path):
    """Catches requiring rank-0 common metadata on the rank-1 filesystem."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / identity()["run_id"] / "checkpoint-2"
    step = checkpoint / "global_step2"
    step.mkdir(parents=True)
    for rank in (2, 3):
        (step / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(
            f"optimizer-{rank}".encode()
        )
        (step / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt").write_bytes(
            f"model-{rank}".encode()
        )
        (checkpoint / f"rng_state_{rank}.pth").write_bytes(
            f"rng-{rank}".encode()
        )

    manifest = build_node_manifest(
        checkpoint,
        node_rank=1,
        process_ranks=(2, 3),
        run_id=identity()["run_id"],
        runtime_identity=identity(),
        checkpoint_root=checkpoint_root,
    )

    assert manifest.runtime_identity == identity()
    assert {entry.process_rank for entry in manifest.entries} == {2, 3}


def test_manifest_rejects_symlink_and_nonregular_file(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / identity()["run_id"] / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    values = write_observed_full_checkpoint(checkpoint)
    outside = tmp_path.parent / "outside-shard"
    outside.write_bytes(b"outside")
    (checkpoint / "linked.pt").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        build_node_manifest(
            checkpoint,
            node_rank=1,
            process_ranks=(2, 3),
                runtime_identity=values,
            checkpoint_root=checkpoint_root,
        )

    (checkpoint / "linked.pt").unlink()
    os.mkfifo(checkpoint / "fifo")
    with pytest.raises(ValueError, match="regular"):
        build_node_manifest(
            checkpoint,
            node_rank=1,
            process_ranks=(2, 3),
            runtime_identity=values,
            checkpoint_root=checkpoint_root,
        )


def test_manifest_rejects_shard_owned_by_another_node(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / identity()["run_id"] / "checkpoint-2"
    shard = checkpoint / "global_step2/zero_pp_rank_0_mp_rank_00_optim_states.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"foreign")
    values = write_observed_full_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="assigned process ranks"):
        build_node_manifest(
            checkpoint,
            node_rank=1,
            process_ranks=(2, 3),
            runtime_identity=values,
            checkpoint_root=checkpoint_root,
        )


def test_manifest_rejects_noncanonical_run_or_checkpoint_directory(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    wrong = checkpoint_root / identity()["run_id"] / "checkpoint-3"
    wrong.mkdir(parents=True)
    with pytest.raises(ValueError, match="exact canonical"):
        build_node_manifest(
            wrong,
            node_rank=1,
            process_ranks=(2, 3),
            runtime_identity=identity(),
            checkpoint_root=checkpoint_root,
        )


def test_manifest_run_id_is_bound_to_observed_canonical_ancestor(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / full_identity()["run_id"] / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    values = write_observed_full_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="run ID"):
        build_node_manifest(
            checkpoint,
            node_rank=1,
            process_ranks=(2, 3),
            run_id="another-valid-run",
            runtime_identity=values,
            checkpoint_root=checkpoint_root,
        )


def test_observed_identity_rejects_latest_trainer_step_disagreement(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    values = write_observed_full_checkpoint(checkpoint)
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step":3}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="steps disagree"):
        observe_checkpoint_identity(checkpoint, values)


def test_observed_identity_rejects_runtime_sidecar_relabel(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    values = write_observed_full_checkpoint(checkpoint)
    runtime = json.loads((checkpoint / "deepspeed-runtime.json").read_text())
    runtime["world_size"] = 8
    (checkpoint / "deepspeed-runtime.json").write_text(
        json.dumps(runtime) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="world_size"):
        observe_checkpoint_identity(checkpoint, values)


def test_observed_layout_must_agree_with_derived_zero_layout(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    values = full_identity(zero_stage=2, model_state_layout="replicated")
    write_observed_full_checkpoint(checkpoint, values=values)
    tag = checkpoint / "global_step2"
    (tag / "mp_rank_00_model_states.pt").unlink()
    (tag / "zero_pp_rank_2_mp_rank_00_model_states.pt").write_bytes(b"partition")

    with pytest.raises(ValueError, match="model_state_layout"):
        observe_checkpoint_identity(checkpoint, values)


def test_observed_stage2_full_identity_rejects_invalid_sampler_sidecar(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    values = full_identity(stage="stage2", stage_sidecar="stage2-sampler.json")
    write_observed_full_checkpoint(checkpoint, values=values)
    (checkpoint / "stage1_ema_teacher.pt").unlink()
    (checkpoint / "stage2-sampler.json").write_text(
        '{"schema_version":1,"global_step":2}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Stage 2 sampler"):
        observe_checkpoint_identity(checkpoint, values)


def test_observed_stage2_full_identity_accepts_strict_sampler_sidecar(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    values = full_identity(stage="stage2", stage_sidecar="stage2-sampler.json")
    write_observed_full_checkpoint(checkpoint, values=values)
    (checkpoint / "stage1_ema_teacher.pt").unlink()
    (checkpoint / "stage2-sampler.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 7,
                "world_size": 4,
                "dataset_size": 4,
                "bucket_sizes": {"1": 4},
                "padded_samples": 0,
                "global_step": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert observe_checkpoint_identity(checkpoint, values) == values


def test_resume_path_is_the_same_immediate_logical_checkpoint_on_each_node(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.symlink_to(storage, target_is_directory=True)
    logical = checkpoint_root / identity()["run_id"] / "checkpoint-2"
    logical.mkdir(parents=True)

    assert validate_resume_path(
        identity()["run_id"], logical, checkpoint_root=checkpoint_root
    ) == logical
    nested = checkpoint_root / identity()["run_id"] / "old" / "checkpoint-2"
    nested.mkdir(parents=True)
    with pytest.raises(ValueError, match="immediate"):
        validate_resume_path(
            identity()["run_id"], nested, checkpoint_root=checkpoint_root
        )


def test_manifest_json_round_trip_preserves_path_and_identity():
    original = manifest_for((2, 3))
    assert NodeArtifactManifest.from_json(original.to_json()) == original

    payload = original.entries[0].to_json()
    payload["relative_path"] = "../escape.pt"
    with pytest.raises(ValueError, match="relative path"):
        ArtifactEntry.from_json(payload)


def test_collection_requires_all_four_process_ranks():
    with pytest.raises(ValueError, match="ranks 0 through 3"):
        validate_collection((manifest_for((0, 1)),), world_size=4)


@pytest.mark.parametrize(
    ("manifests", "message"),
    [
        (
            (
                dataclasses.replace(
                    manifest_for((0, 1)),
                    entries=(ArtifactEntry(Path("../escape.pt"), 8, 0),),
                ),
                manifest_for((2, 3)),
            ),
            "relative path",
        ),
        (
            (
                manifest_for(
                    (0, 1),
                    extra_entries=(ArtifactEntry(Path("trainer_state.json"), 8, 0),),
                ),
                manifest_for(
                    (2, 3),
                    extra_entries=(ArtifactEntry(Path("trainer_state.json"), 9, 1),),
                ),
            ),
            "different sizes",
        ),
        (
            (
                manifest_for((0, 1), run_id="run-a"),
                manifest_for((2, 3), run_id="run-b"),
            ),
            "run ID",
        ),
        (
            (
                manifest_for((0, 1), completion_marker=False),
                manifest_for((2, 3)),
            ),
            "completion marker",
        ),
        (
            (
                manifest_for((0, 1)),
                manifest_for((2, 3), runtime_identity=identity(commit="c" * 40)),
            ),
            "runtime identity",
        ),
    ],
)
def test_collection_rejects_invalid_manifests(manifests, message):
    with pytest.raises(ValueError, match=message):
        validate_collection(manifests, world_size=4)


def test_collection_accepts_identical_small_common_metadata_once():
    remote, local = strict_manifests()
    trainer = next(
        entry
        for entry in remote.entries
        if entry.relative_path == Path("trainer_state.json")
    )
    local = dataclasses.replace(
        local,
        entries=local.entries + (dataclasses.replace(trainer, node_rank=1),),
    )
    entries = validate_collection(
        (remote, local),
        world_size=4,
    )
    assert [item.relative_path for item in entries].count(Path("trainer_state.json")) == 1


def test_collection_never_treats_equal_sized_weights_as_common_metadata():
    remote_weight = ArtifactEntry(Path("adapter_model.safetensors"), 31, 0)
    local_weight = ArtifactEntry(Path("adapter_model.safetensors"), 31, 1)
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        validate_collection(
            (
                dataclasses.replace(
                    strict_manifest(0),
                    entries=strict_manifest(0).entries + (remote_weight,),
                ),
                dataclasses.replace(
                    strict_manifest(1),
                    entries=strict_manifest(1).entries + (local_weight,),
                ),
            ),
            world_size=4,
        )


def test_transfer_refuses_preexisting_destination(tmp_path):
    destination = tmp_path / "global_step2/model.safetensors"
    destination.parent.mkdir()
    destination.write_bytes(b"owned-by-earlier-run")
    with pytest.raises(FileExistsError, match="destination"):
        validate_transfer_destination((destination,))


def test_transfer_allows_only_validated_identical_common_metadata(tmp_path):
    destination = tmp_path / "trainer_state.json"
    destination.write_bytes(b"small")
    validate_transfer_destination(
        (destination,),
        validated_identical_common_metadata={destination: 5},
    )
    with pytest.raises(FileExistsError, match="destination"):
        validate_transfer_destination(
            (destination,),
            validated_identical_common_metadata={destination: 6},
        )
    weight = tmp_path / "adapter_model.safetensors"
    weight.write_bytes(b"small")
    with pytest.raises(FileExistsError, match="destination"):
        validate_transfer_destination(
            (weight,), validated_identical_common_metadata={weight: 5}
        )


def test_push_never_overwrites_and_is_scoped_to_the_run():
    command = build_push_command(manifest_for((2, 3)))
    assert "--archive" in command
    assert "--ignore-existing" in command
    assert "--files-from" in command
    assert "--from0" in command
    assert "--checksum" not in command
    assert command[command.index("--rsync-path") + 1] == (
        "/usr/bin/docker exec -i llm_kd /usr/bin/rsync"
    )
    assert command[-2] == (
        f"{_REPOSITORY_ROOT}/checkpoints/"
        "stage1-lora-zero3-four-gpu/checkpoint-2/"
    )
    assert command[-1] == (
        "korea@10.1.102.41:"
        f"{_REPOSITORY_ROOT}/checkpoints/"
        "stage1-lora-zero3-four-gpu/checkpoint-2/"
    )
    assert not any("venv" in argument or "/data/" in argument for argument in command)


def test_pull_selects_only_local_process_ranks_and_common_metadata(tmp_path):
    common = ArtifactEntry(Path("trainer_state.json"), 31, 0)
    combined = dataclasses.replace(
        manifest_for((0, 1), extra_entries=(common,)),
        process_ranks=(0, 1, 2, 3),
        entries=tuple(entry(rank, node_rank=0) for rank in range(4))
        + (common,)
        + stage1_inference_entries(),
    )

    selected = select_resume_entries(combined)
    assert {item.process_rank for item in selected} == {None, 0, 1, 2, 3}
    command = build_pull_command(combined, tmp_path / "checkpoint-copy")
    assert command[command.index("--rsync-path") + 1] == (
        "/usr/bin/docker exec -i llm_kd /usr/bin/rsync"
    )
    assert command[-2].startswith(
        f"korea@10.1.102.41:{_REPOSITORY_ROOT}/checkpoints/"
    )
    assert command[-1] == f"{tmp_path}/checkpoint-copy/"
    assert "--checksum" not in command


def test_resume_selection_retains_loader_artifacts_and_only_local_ds_shards():
    canonical = dataclasses.replace(
        strict_manifest(0),
        process_ranks=(0, 1, 2, 3),
        entries=tuple(
            dataclasses.replace(item, node_rank=0)
            for item in validate_collection(strict_manifests(), world_size=4)
        ),
    )

    selected = select_resume_entries(canonical)
    names = {item.relative_path.name for item in selected}
    assert {
        "adapter_config.json",
        "adapter_model.safetensors",
        "stage1_peft_metadata.json",
        "ilvr_token_rows.safetensors",
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "rng_state_2.pth",
        "rng_state_3.pth",
    } <= names
    assert {item.process_rank for item in selected} == {None, 0, 1, 2, 3}


@pytest.mark.parametrize(
    "field", ("stage", "tuning_method", "zero_stage", "world_size", "run_id")
)
def test_resume_rejects_identity_change_before_transfer(field):
    replacements = {
        "stage": "stage2",
        "tuning_method": "dora",
        "zero_stage": 2,
        "world_size": 8,
        "run_id": "another-run",
    }
    requested = identity(**{field: replacements[field]})
    with pytest.raises(ValueError, match=field):
        validate_resume_identity(identity(), requested)


@pytest.mark.parametrize(
    "field",
    (
        "deepspeed_config_digest",
        "rank_mapping",
        "global_step",
        "optimizer",
        "scheduler",
        "rng_ranks",
        "stage_sidecar",
        "commit",
    ),
)
def test_resume_rejects_changed_checkpoint_state_field(field):
    changed = {
        "deepspeed_config_digest": "d" * 64,
        "rank_mapping": {"0": [0], "1": [1, 2, 3]},
        "global_step": 3,
        "optimizer": False,
        "scheduler": False,
        "rng_ranks": [0, 1],
        "stage_sidecar": "stage2_peft_metadata.json",
        "commit": "e" * 40,
    }
    with pytest.raises(ValueError, match=field):
        validate_resume_identity(identity(), identity(**{field: changed[field]}))


def test_inference_copy_requires_fresh_nonsymlink_local_destination(tmp_path):
    remote = manifest_for((0, 1))
    destination = tmp_path / "inference-copy"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="fresh inference destination"):
        build_inference_pull_command(remote, destination)

    destination.rmdir()
    destination.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError, match="fresh inference destination"):
        build_inference_pull_command(remote, destination)


def test_inference_pull_transfers_only_manifest_listed_files(tmp_path):
    remote = dataclasses.replace(
        manifest_for((0, 1)),
        entries=stage1_inference_entries(),
    )
    destination = tmp_path / "fresh-inference"
    command = build_inference_pull_command(remote, destination)
    assert "--files-from" in command
    assert "--ignore-existing" in command
    assert command[-1] == f"{destination}/"
    assert not destination.exists()


def test_inference_pull_rejects_untrusted_manifest_path(tmp_path):
    remote = dataclasses.replace(
        manifest_for((0, 1)),
        entries=(ArtifactEntry(Path("../escape.safetensors"), 99, 0),),
    )
    with pytest.raises(ValueError, match="relative path"):
        build_inference_pull_command(remote, tmp_path / "fresh-inference")


def test_small_manifest_and_files_list_publish_exclusively(tmp_path):
    manifest = manifest_for((2, 3))
    manifest_path = tmp_path / "control" / "node-manifest.json"
    files_path = tmp_path / "control" / "push.files"

    publish_node_manifest(manifest_path, manifest)
    publish_files_list(files_path, manifest.entries)

    assert json.loads(manifest_path.read_text()) == manifest.to_json()
    assert files_path.read_bytes().split(b"\0") == [
        b"checkpoint-2/global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt",
        b"checkpoint-2/global_step2/zero_pp_rank_3_mp_rank_00_optim_states.pt",
        b"",
    ]
    with pytest.raises(FileExistsError):
        publish_node_manifest(manifest_path, manifest)
    with pytest.raises(FileExistsError):
        publish_files_list(files_path, manifest.entries)


class FakeArtifactRunner:
    def __init__(self):
        self.events = []
        self.manifests = {
            0: strict_manifest(0),
            1: strict_manifest(1),
        }

    def read_artifact_manifest(self, machine_rank, checkpoint_path, runtime_identity):
        self.events.append(f"manifest-rank{machine_rank}")
        assert checkpoint_path.name == "checkpoint-2"
        assert runtime_identity == identity()
        return self.manifests[machine_rank]

    def validate_push_destination(self, manifest, entries):
        self.events.append("destination-precondition")
        assert manifest.node_rank == 1
        assert {item.process_rank for item in entries} == {2, 3}

    def push_artifacts(self, manifest, entries):
        self.events.append("push")

    def read_canonical_artifact_manifest(self, checkpoint_path, runtime_identity):
        self.events.append("post-push-manifest")
        expected = validate_collection(tuple(self.manifests.values()), world_size=4)
        return dataclasses.replace(
            self.manifests[0],
            process_ranks=(0, 1, 2, 3),
            entries=tuple(dataclasses.replace(entry, node_rank=0) for entry in expected),
        )

def test_collection_runs_after_both_zero_exits_and_checks_destination_first(tmp_path):
    runner = FakeArtifactRunner()
    checkpoint = tmp_path / "stage1-lora-zero3-four-gpu" / "checkpoint-2"

    entries = collect_checkpoint_artifacts(
        runner,
        checkpoint_path=checkpoint,
        runtime_identity=identity(),
        exit_codes={0: 0, 1: 0},
        world_size=4,
    )

    assert {item.process_rank for item in entries if item.process_rank is not None} == {
        0,
        1,
        2,
        3,
    }
    assert runner.events == [
        "manifest-rank0",
        "manifest-rank1",
        "destination-precondition",
        "push",
        "post-push-manifest",
    ]


def test_failed_node_exit_never_fabricates_collection(tmp_path):
    runner = FakeArtifactRunner()
    with pytest.raises(ValueError, match="zero exits"):
        collect_checkpoint_artifacts(
            runner,
            checkpoint_path=tmp_path / "stage1-lora-zero3-four-gpu" / "checkpoint-2",
            runtime_identity=identity(),
            exit_codes={0: 0, 1: 7},
            world_size=4,
        )
    assert runner.events == []


class FakeTransferRunner:
    def __init__(self):
        self.events = []

    def validate_resume_destination(self, entries, destination):
        self.events.append("resume-destination")

    def pull_resume_artifacts(self, manifest, entries, destination):
        self.events.append("resume-pull")

    def validate_inference_manifest(self, manifest):
        self.events.append("inference-manifest")

    def pull_inference_artifacts(self, manifest, destination):
        self.events.append("inference-pull")


def test_resume_validates_identity_before_destination_or_transfer(tmp_path):
    runner = FakeTransferRunner()
    expected = validate_collection(strict_manifests(), world_size=4)
    remote = dataclasses.replace(
        strict_manifest(0),
        process_ranks=(0, 1, 2, 3),
        entries=tuple(dataclasses.replace(entry, node_rank=0) for entry in expected),
    )
    with pytest.raises(ValueError, match="zero_stage"):
        prepare_resume_transfer(
            identity(),
            identity(zero_stage=2),
            remote,
            tmp_path / "resume",
            runner,
        )
    assert runner.events == []

    prepare_resume_transfer(
        identity(), identity(), remote, tmp_path / "resume", runner
    )
    assert runner.events == ["resume-destination", "resume-pull"]


def test_inference_preparation_validates_manifest_before_one_pull(tmp_path):
    runner = FakeTransferRunner()
    remote = dataclasses.replace(
        manifest_for((0, 1)),
        entries=stage1_inference_entries(),
    )
    destination = tmp_path / "fresh-inference"

    prepare_inference_copy(remote, destination, runner)

    assert runner.events == ["inference-manifest", "inference-pull"]
    assert destination.is_dir()


def test_subprocess_runner_uses_bounded_node_json_then_exact_rsync_list(tmp_path):
    plan = launch_plan()
    commands = build_owned_tmux_commands(plan, commit="b" * 40)
    remote_manifest = strict_manifest(0)
    local_manifest = strict_manifest(1)
    combined = validate_collection((remote_manifest, local_manifest), world_size=4)
    canonical_manifest = dataclasses.replace(
        remote_manifest,
        process_ranks=(0, 1, 2, 3),
        entries=tuple(dataclasses.replace(entry, node_rank=0) for entry in combined),
    )
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        argv = tuple(command)
        calls.append((argv, input_bytes, timeout_seconds))
        display = " ".join(argv)
        if "artifact-manifest" in display:
            if "--canonical-owner" in display:
                payload = canonical_manifest
            else:
                payload = remote_manifest if argv[0] == "/usr/bin/ssh" else local_manifest
            stdout = json.dumps(payload.to_json())
        else:
            stdout = ""
        return __import__("subprocess").CompletedProcess(
            command, 0, stdout=stdout, stderr=""
        )

    runner = SubprocessCoordinatorRunner(
        plan,
        commands=commands,
        commit="b" * 40,
        log_root=tmp_path / "logs",
        command_runner=fake_command,
    )
    checkpoint = _REPOSITORY_ROOT / "checkpoints" / plan.run_id / "checkpoint-2"

    collected = collect_checkpoint_artifacts(
        runner,
        checkpoint_path=checkpoint,
        runtime_identity=identity(),
        exit_codes={0: 0, 1: 0},
        world_size=4,
    )

    assert len(collected) == len(combined)
    displays = [" ".join(call[0]) for call in calls]
    assert sum("artifact-manifest" in display for display in displays) == 3
    destination_call = next(
        call for call in calls if "artifact-destination-check" in " ".join(call[0])
    )
    assert destination_call[0][0] == "/usr/bin/ssh"
    assert destination_call[1] is not None
    rsync = next(call[0] for call in calls if call[0][0] == "/usr/bin/rsync")
    assert "--ignore-existing" in rsync
    assert "--files-from" in rsync
    assert "--checksum" not in rsync
    files_path = Path(rsync[rsync.index("--files-from") + 1])
    transferred = set(files_path.read_bytes().split(b"\0"))
    assert b"global_step2/bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt" in transferred
    assert b"rng_state_3.pth" in transferred
    assert b"" in transferred
    assert all(
        call[2] == (3600 if call[0][0] == "/usr/bin/rsync" else 30)
        for call in calls
    )


def test_checkpoint_transfer_has_long_bounded_timeout_and_propagates_timeout(
    tmp_path,
):
    plan = launch_plan()
    manifest = strict_manifest(1)
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        calls.append((tuple(command), timeout_seconds))
        if command[0] == "/usr/bin/rsync":
            assert timeout_seconds > 30
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner = SubprocessCoordinatorRunner(
        plan,
        commands=build_owned_tmux_commands(plan, commit="b" * 40),
        commit="b" * 40,
        log_root=tmp_path / "logs",
        command_runner=fake_command,
    )

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        runner.push_artifacts(manifest, manifest.entries[:1])

    assert raised.value.timeout == 3600
    assert len(calls) == 1
    assert calls[0][0][0] == "/usr/bin/rsync"
    assert calls[0][1] == 3600


def test_artifact_manifest_cli_emits_only_small_json_metadata(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    run_directory = checkpoint_root / "stage1-lora-zero3-four-gpu" / "checkpoint-2"
    shard = run_directory / "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"rank-two")
    values = write_observed_full_checkpoint(run_directory)

    completed = subprocess.run(
        (
            Path(__file__).resolve().parents[2] / "venv/ilvr-venv/bin/python",
            "-m",
            "src.multinode",
            "artifact-manifest",
            "--run-directory",
            str(run_directory),
            "--run-id",
            values["run_id"],
            "--checkpoint-root",
            str(checkpoint_root),
            "--node-rank",
            "1",
            "--process-rank",
            "2",
            "--process-rank",
            "3",
            "--runtime-identity-json",
            json.dumps(values),
            "--completion-marker",
        ),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert {
        "relative_path": "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt",
        "byte_size": 8,
        "node_rank": 1,
        "process_rank": 2,
    } in payload["entries"]
    assert "sha256" not in completed.stdout


def test_destination_cli_rejects_ranked_collision_but_allows_identical_common(tmp_path):
    run_directory = tmp_path / "stage1-lora-zero3-four-gpu" / "checkpoint-2"
    ranked = run_directory / "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"
    common = run_directory / "trainer_state.json"
    ranked.parent.mkdir(parents=True)
    ranked.write_bytes(b"old-rank")
    common.write_bytes(b"small")
    payload = {
        "entries": [
            ArtifactEntry(
                Path("global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"),
                8,
                1,
            ).to_json(),
            ArtifactEntry(Path("trainer_state.json"), 5, 1).to_json(),
        ],
        "runtime_identity": identity(),
    }
    command = (
        Path(__file__).resolve().parents[2] / "venv/ilvr-venv/bin/python",
        "-m",
        "src.multinode",
        "artifact-destination-check",
        "--run-directory",
        str(run_directory),
        "--run-id",
        identity()["run_id"],
        "--checkpoint-root",
        str(tmp_path),
    )

    completed = subprocess.run(
        command,
        input=json.dumps(payload),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "destination" in completed.stderr

    ranked.unlink()
    completed = subprocess.run(
        command,
        input=json.dumps(payload),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("prefix", ("bf16_", "fp16_", ""))
def test_collection_accepts_all_canonical_optimizer_prefixes(prefix):
    entries = validate_collection(strict_manifests(prefix=prefix), world_size=4)
    optimizer_ranks = {
        entry.process_rank
        for entry in entries
        if entry.relative_path.name.endswith("_optim_states.pt")
    }
    assert optimizer_ranks == {0, 1, 2, 3}


def test_collection_rejects_rng_only_rank_completeness():
    manifests = []
    for node_rank in (0, 1):
        manifest = strict_manifest(node_rank)
        manifests.append(
            dataclasses.replace(
                manifest,
                entries=tuple(
                    entry
                    for entry in manifest.entries
                    if not entry.relative_path.name.endswith("_optim_states.pt")
                ),
            )
        )
    with pytest.raises(ValueError, match="optimizer ranks 0 through 3"):
        validate_collection(tuple(manifests), world_size=4)


def test_collection_rejects_more_than_one_optimizer_shard_per_rank():
    remote, local = strict_manifests()
    duplicate = ArtifactEntry(
        Path("global_step2/fp16_zero_pp_rank_0_mp_rank_00_optim_states.pt"),
        10,
        0,
    )
    remote = dataclasses.replace(remote, entries=remote.entries + (duplicate,))
    with pytest.raises(ValueError, match="optimizer ranks 0 through 3"):
        validate_collection((remote, local), world_size=4)


@pytest.mark.parametrize(
    ("omitted", "message"),
    [
        ("rng_state_3.pth", "RNG ranks 0 through 3"),
        ("scheduler.pt", "scheduler.pt"),
        ("trainer_state.json", "trainer_state.json"),
        ("deepspeed-runtime.json", "deepspeed-runtime.json"),
        ("stage1_ema_teacher.pt", "stage1_ema_teacher.pt"),
        ("stage1_peft_metadata.json", "stage1_peft_metadata.json"),
        (
            "global_step2/zero_pp_rank_3_mp_rank_00_model_states.pt",
            "model-state ranks 0 through 3",
        ),
    ],
)
def test_collection_requires_every_checkpoint_component(omitted, message):
    remote = strict_manifest(0, omit=(omitted,))
    local = strict_manifest(1, omit=(omitted,))
    with pytest.raises(ValueError, match=message):
        validate_collection((remote, local), world_size=4)


def test_collection_rejects_missing_stage1_adapter_config():
    remote = strict_manifest(0, omit=("adapter_config.json",))
    local = strict_manifest(1)

    with pytest.raises(ValueError, match="adapter_config|PEFT"):
        validate_collection((remote, local), world_size=4)


def test_collection_rejects_checkpoint_and_global_step_mismatch():
    remote, local = strict_manifests()
    local = dataclasses.replace(
        local,
        entries=tuple(
            dataclasses.replace(
                entry,
                relative_path=Path(
                    str(entry.relative_path).replace("global_step2", "global_step3")
                ),
            )
            for entry in local.entries
        ),
    )
    with pytest.raises(ValueError, match="global_step"):
        validate_collection((remote, local), world_size=4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("zero_stage", True),
        ("world_size", True),
        ("global_step", True),
        ("zero_stage", "3"),
        ("world_size", "4"),
        ("rank_mapping", {0: [0, 1], 1: [2, 3]}),
        ("rank_mapping", {"0": [False, True], "1": [2, 3]}),
        ("rng_ranks", [False, True, 2, 3]),
        ("model_state_layout", "unknown"),
    ],
)
def test_manifest_identity_rejects_coercive_or_wrong_types(field, value):
    remote, local = strict_manifests(changes={field: value})
    with pytest.raises(ValueError, match=field):
        validate_collection((remote, local), world_size=4)


def test_common_metadata_uses_explicit_identity_allowlist():
    remote, local = strict_manifests()
    remote = dataclasses.replace(
        remote,
        entries=remote.entries + (ArtifactEntry(Path("invented.json"), 5, 0),),
    )
    local = dataclasses.replace(
        local,
        entries=local.entries + (ArtifactEntry(Path("invented.json"), 5, 1),),
    )
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_collection((remote, local), world_size=4)


def test_post_push_requires_exact_fresh_remote_manifest():
    expected = validate_collection(strict_manifests(), world_size=4)
    observed = dataclasses.replace(
        strict_manifest(0),
        process_ranks=(0, 1, 2, 3),
        entries=tuple(dataclasses.replace(entry, node_rank=0) for entry in expected),
    )
    validate_canonical_destination(
        expected,
        observed,
        expected_identity=identity(),
        expected_run_id=identity()["run_id"],
    )
    corrupted = dataclasses.replace(
        observed,
        entries=observed.entries[:-1]
        + (dataclasses.replace(observed.entries[-1], byte_size=999),),
    )
    with pytest.raises(ValueError, match="exact path and size"):
        validate_canonical_destination(
            expected,
            corrupted,
            expected_identity=identity(),
            expected_run_id=identity()["run_id"],
        )


def test_inference_selection_excludes_all_training_state():
    manifest = dataclasses.replace(
        strict_manifest(0),
        entries=strict_manifest(0).entries,
    )
    selected = select_inference_entries(manifest)
    assert {entry.relative_path.name for entry in selected} == {
        "adapter_model.safetensors",
        "adapter_config.json",
        "config.json",
        "stage1_peft_metadata.json",
        "ilvr_token_rows.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }


def test_full_inference_manifest_parses_index_and_rejects_unlisted_sibling(tmp_path):
    checkpoint = tmp_path / "full-artifact"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(standard_qwen_config()) + "\n", encoding="utf-8"
    )
    for name in (
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        (checkpoint / name).write_text("{}\n", encoding="utf-8")
    (checkpoint / "stage1_ema_teacher.pt").write_bytes(b"ema")
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    (checkpoint / first).write_bytes(b"one")
    (checkpoint / second).write_bytes(b"two")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": first}}) + "\n",
        encoding="utf-8",
    )
    manifest = NodeArtifactManifest(
        schema_version=1,
        run_id=full_identity()["run_id"],
        node_rank=0,
        process_ranks=(0, 1, 2, 3),
        entries=tuple(
            ArtifactEntry(path.relative_to(checkpoint), path.stat().st_size, 0)
            for path in sorted(checkpoint.iterdir())
        ),
        runtime_identity=full_identity(),
        completion_marker=True,
    )

    with pytest.raises(ValueError, match="shard set does not match index"):
        build_inference_manifest(checkpoint, manifest)


def test_inference_destination_is_securely_reserved(tmp_path):
    destination = tmp_path / "fresh" / "model"
    assert reserve_inference_destination(destination) == destination
    assert destination.is_dir()
    with pytest.raises(FileExistsError, match="fresh inference destination"):
        reserve_inference_destination(destination)


def test_reserved_inference_destination_detects_pretransfer_race(tmp_path):
    destination = reserve_inference_destination(tmp_path / "fresh" / "model")
    (destination / "raced-file").write_bytes(b"foreign")
    with pytest.raises(FileExistsError, match="no longer empty"):
        validate_reserved_inference_destination(destination)


class FakeProductionTransferRunner(FakeTransferRunner):
    def __init__(self):
        super().__init__()
        expected = validate_collection(strict_manifests(), world_size=4)
        self.canonical = dataclasses.replace(
            strict_manifest(0),
            process_ranks=(0, 1, 2, 3),
            entries=tuple(dataclasses.replace(item, node_rank=0) for item in expected),
        )
        self.pulled = False

    def read_canonical_artifact_manifest(self, checkpoint_path, runtime_identity):
        self.events.append("remote-canonical-manifest")
        return self.canonical

    def read_inference_artifact_manifest(self, checkpoint_path, runtime_identity):
        self.events.append("remote-inference-manifest")
        return dataclasses.replace(
            self.canonical,
            entries=select_inference_entries(self.canonical),
        )

    def read_artifact_manifest(
        self, machine_rank, checkpoint_path, runtime_identity, *, canonical_owner=False
    ):
        assert machine_rank == 1
        self.events.append("local-post-manifest" if self.pulled else "local-manifest")
        if not self.pulled:
            assert not canonical_owner
            return strict_manifest(1)
        assert canonical_owner
        selected = select_resume_entries(self.canonical)
        return dataclasses.replace(
            strict_manifest(1),
            process_ranks=(0, 1, 2, 3),
            entries=tuple(dataclasses.replace(item, node_rank=1) for item in selected),
        )

    def pull_resume_artifacts(self, manifest, entries, destination):
        super().pull_resume_artifacts(manifest, entries, destination)
        self.pulled = True

    def mirror_prepared_inference(self, manifest, destination):
        self.events.append("remote-inference-mirror")
        return manifest


def test_production_resume_redistribution_uses_same_immediate_canonical_path():
    runner = FakeProductionTransferRunner()
    checkpoint = (
        Path("/root/workspace/ILVR/checkpoints")
        / identity()["run_id"]
        / "checkpoint-2"
    )

    entries = redistribute_resume_checkpoint(
        runner,
        checkpoint_path=checkpoint,
        checkpoint_identity=identity(),
        requested_identity=identity(),
        destination=checkpoint,
    )

    assert {entry.process_rank for entry in entries} == {None, 0, 1, 2, 3}
    assert runner.events == [
        "remote-canonical-manifest",
        "local-manifest",
        "resume-destination",
        "resume-pull",
        "local-post-manifest",
    ]


def test_inference_preparation_is_callable_before_evaluation(
    tmp_path, monkeypatch
):
    import src.multinode_artifacts as artifacts_module

    repository = tmp_path / "repo"
    repository.mkdir()
    storage = tmp_path / "checkpoint-storage"
    storage.mkdir()
    (repository / "checkpoints").symlink_to(storage, target_is_directory=True)
    monkeypatch.setattr(artifacts_module, "_LOCAL_REPOSITORY", repository)
    runner = FakeProductionTransferRunner()
    runner.canonical = dataclasses.replace(
        runner.canonical,
        entries=runner.canonical.entries
        + tuple(
            item
            for item in stage1_inference_entries()
            if item.relative_path
            not in {entry.relative_path for entry in runner.canonical.entries}
        ),
    )
    destination = (
        repository
        / "checkpoints"
        / "evaluation-staging"
        / identity()["run_id"]
        / "prepared-model"
    )

    prepared = prepare_inference_artifacts(
        runner,
        checkpoint_path=Path("/root/workspace/ILVR/checkpoints")
        / identity()["run_id"]
        / "checkpoint-2",
        runtime_identity=identity(),
        destination=destination,
    )

    assert prepared == destination
    assert destination.is_dir()
    assert runner.events == [
        "remote-canonical-manifest",
        "remote-inference-manifest",
        "inference-manifest",
        "inference-pull",
        "remote-inference-mirror",
    ]


def test_inference_preparation_writes_through_canonical_checkpoint_link(
    tmp_path, monkeypatch
):
    import src.multinode_artifacts as artifacts_module

    repository = tmp_path / "repo"
    repository.mkdir()
    storage = tmp_path / "checkpoint-storage"
    storage.mkdir()
    (repository / "checkpoints").symlink_to(storage, target_is_directory=True)
    runner = FakeProductionTransferRunner()
    runner.canonical = dataclasses.replace(
        runner.canonical,
        entries=runner.canonical.entries
        + tuple(
            item
            for item in stage1_inference_entries()
            if item.relative_path
            not in {entry.relative_path for entry in runner.canonical.entries}
        ),
    )
    destination = (
        repository
        / "checkpoints"
        / "evaluation-staging"
        / identity()["run_id"]
        / "prepared-model"
    )
    monkeypatch.setattr(artifacts_module, "_LOCAL_REPOSITORY", repository)

    prepared = prepare_inference_artifacts(
        runner,
        checkpoint_path=(
            repository
            / "checkpoints"
            / identity()["run_id"]
            / "checkpoint-2"
        ),
        runtime_identity=identity(),
        destination=destination,
    )

    assert prepared == destination
    assert (
        storage
        / "evaluation-staging"
        / identity()["run_id"]
        / "prepared-model"
    ).is_dir()


def test_inference_preparation_binds_filtered_remote_manifest_identity(tmp_path):
    runner = FakeProductionTransferRunner()
    existing = {entry.relative_path for entry in runner.canonical.entries}
    runner.canonical = dataclasses.replace(
        runner.canonical,
        entries=runner.canonical.entries
        + tuple(
            entry
            for entry in stage1_inference_entries()
            if entry.relative_path not in existing
        ),
    )
    original = runner.read_inference_artifact_manifest

    def relabeled(checkpoint_path, runtime_identity):
        manifest = original(checkpoint_path, runtime_identity)
        return dataclasses.replace(
            manifest,
            runtime_identity=identity(commit="c" * 40),
        )

    runner.read_inference_artifact_manifest = relabeled
    with pytest.raises(ValueError, match="identity"):
        prepare_inference_artifacts(
            runner,
            checkpoint_path=Path("/root/workspace/ILVR/checkpoints")
            / identity()["run_id"]
            / "checkpoint-2",
            runtime_identity=identity(),
            destination=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_collection_rejects_post_rsync_manifest_mismatch(tmp_path):
    runner = FakeArtifactRunner()
    original = runner.read_canonical_artifact_manifest

    def corrupted(checkpoint_path, runtime_identity):
        manifest = original(checkpoint_path, runtime_identity)
        return dataclasses.replace(manifest, entries=manifest.entries[:-1])

    runner.read_canonical_artifact_manifest = corrupted
    with pytest.raises(ValueError, match="exact path and size"):
        collect_checkpoint_artifacts(
            runner,
            checkpoint_path=tmp_path
            / identity()["run_id"]
            / "checkpoint-2",
            runtime_identity=identity(),
            exit_codes={0: 0, 1: 0},
            world_size=4,
        )
    assert runner.events[-2:] == ["push", "post-push-manifest"]


def test_prepare_inference_mirrors_exact_artifact_to_remote_and_revalidates(
    tmp_path, monkeypatch
):
    import src.multinode_artifacts as artifacts_module
    import src.multinode_runtime as runtime_module

    repository = tmp_path / "repo"
    destination = (
        repository
        / "checkpoints"
        / "evaluation-staging"
        / identity()["run_id"]
        / "prepared-model"
    )
    destination.mkdir(parents=True)
    manifest = dataclasses.replace(
        strict_manifest(0),
        process_ranks=(0, 1, 2, 3),
        entries=stage1_inference_entries(),
    )
    publish_node_manifest(
        artifacts_module.inference_manifest_path(destination), manifest
    )
    monkeypatch.setattr(artifacts_module, "_LOCAL_REPOSITORY", repository)
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        calls.append(tuple(command))
        display = " ".join(command)
        if command[0] == "/usr/bin/rsync" and "--rsync-path" not in command:
            return subprocess.CompletedProcess(
                command,
                3,
                stdout="",
                stderr="remote container result parent is not host-traversable",
            )
        if "prepared-inference-show" in display:
            stdout = json.dumps(manifest.to_json())
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    plan = launch_plan()
    log_root = tmp_path / "logs"
    runner = SubprocessCoordinatorRunner(
        plan,
        commands=build_owned_tmux_commands(plan, commit="b" * 40),
        commit="b" * 40,
        log_root=log_root,
        command_runner=fake_command,
    )
    mirror = getattr(runner, "mirror_prepared_inference", None)
    assert mirror is not None

    assert mirror(manifest, destination) == manifest

    python = str(_REPOSITORY_ROOT / "venv/ilvr-venv/bin/python")
    fixed_prefix = (
        "/usr/bin/env",
        f"PYTHONPATH={_REPOSITORY_ROOT}:{_REPOSITORY_ROOT / 'src'}",
        python,
    )
    assert calls[0] == tuple(runtime_module.build_remote_exec(
        (
            *fixed_prefix,
            "-m", "src.multinode", "prepare-inference-destination",
            "--destination", str(destination),
        ),
        working_directory=_REPOSITORY_ROOT,
    ))
    files_path = (
        log_root / plan.run_id / plan.attempt_id
        / "artifact-inference-mirror.files"
    )
    build_push = getattr(
        artifacts_module, "build_prepared_inference_push_command", None
    )
    assert build_push is not None
    assert calls[1] == build_push(
        manifest, destination, files_from=files_path, cluster=plan.cluster
    )
    assert calls[2] == tuple(runtime_module.build_remote_exec(
        (
            *fixed_prefix,
            "-m", "src.multinode", "prepared-inference-show",
            "--model-dir", str(destination),
        ),
        working_directory=_REPOSITORY_ROOT,
    ))
    names = set(files_path.read_bytes().split(b"\0")) - {b""}
    assert names == {
        f"prepared-model/{entry.relative_path.as_posix()}".encode()
        for entry in manifest.entries
    } | {b"prepared-model.manifest.json"}


def test_subprocess_runner_executes_resume_and_inference_file_lists(
    tmp_path, monkeypatch
):
    import src.multinode_artifacts as artifacts_module
    import src.multinode_runtime as runtime_module

    repository = tmp_path / "repo"
    checkpoint_root = repository / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    monkeypatch.setattr(artifacts_module, "_LOCAL_REPOSITORY", repository)
    monkeypatch.setattr(runtime_module, "_CHECKPOINT_ROOT", checkpoint_root)

    plan = launch_plan()
    commands = build_owned_tmux_commands(plan, commit="b" * 40)
    expected = validate_collection(strict_manifests(), world_size=4)
    canonical = dataclasses.replace(
        strict_manifest(0),
        process_ranks=(0, 1, 2, 3),
        entries=tuple(dataclasses.replace(item, node_rank=0) for item in expected)
        + tuple(
            item
            for item in stage1_inference_entries()
            if item.relative_path
            not in {entry.relative_path for entry in expected}
        ),
    )
    local = strict_manifest(1)
    selected_resume = select_resume_entries(canonical)
    hydrated_local = dataclasses.replace(
        local,
        process_ranks=(0, 1, 2, 3),
        entries=tuple(dataclasses.replace(item, node_rank=1) for item in selected_resume),
    )
    inference_manifest = dataclasses.replace(
        canonical, entries=select_inference_entries(canonical)
    )
    sizes = {item.relative_path.as_posix(): item.byte_size for item in canonical.entries}
    calls = []
    resume_pulled = False

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        nonlocal resume_pulled
        argv = tuple(command)
        calls.append(argv)
        if argv[0] == "/usr/bin/rsync":
            files_path = Path(argv[argv.index("--files-from") + 1])
            if argv[-1].startswith("korea@10.1.102.41:"):
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr=""
                )
            destination = Path(argv[-1].removesuffix("/"))
            for encoded in files_path.read_bytes().split(b"\0"):
                if not encoded:
                    continue
                relative = encoded.decode("utf-8")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"x" * sizes[relative])
            if destination == canonical_checkpoint:
                resume_pulled = True
            stdout = ""
        else:
            display = " ".join(argv)
            if "artifact-inference-manifest" in display:
                payload = inference_manifest
            elif "prepared-inference-show" in display:
                payload = inference_manifest
            elif "--canonical-owner" in display and argv[0] == "/usr/bin/ssh":
                payload = canonical
            else:
                payload = hydrated_local if resume_pulled else local
            stdout = json.dumps(payload.to_json())
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    runner = SubprocessCoordinatorRunner(
        plan,
        commands=commands,
        commit="b" * 40,
        log_root=tmp_path / "logs",
        checkpoint_root=checkpoint_root,
        command_runner=fake_command,
    )
    canonical_checkpoint = checkpoint_root / plan.run_id / "checkpoint-2"
    canonical_checkpoint.mkdir(parents=True)
    for item in local.entries:
        target = canonical_checkpoint / item.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * item.byte_size)
    redistribute_resume_checkpoint(
        runner,
        checkpoint_path=canonical_checkpoint,
        checkpoint_identity=identity(),
        requested_identity=identity(),
        destination=canonical_checkpoint,
    )
    inference_destination = (
        repository
        / "checkpoints"
        / "evaluation-staging"
        / plan.run_id
        / "prepared-model"
    )
    prepare_inference_artifacts(
        runner,
        checkpoint_path=canonical_checkpoint,
        runtime_identity=identity(),
        destination=inference_destination,
    )

    rsync_calls = [call for call in calls if call[0] == "/usr/bin/rsync"]
    assert len(rsync_calls) == 3
    resume_names = {
        path.relative_to(canonical_checkpoint).as_posix()
        for path in canonical_checkpoint.rglob("*")
        if path.is_file()
    }
    assert {f"rng_state_{rank}.pth" for rank in range(4)} <= resume_names
    assert any("rank_0_" in name for name in resume_names)
    inference_names = {
        path.relative_to(inference_destination).as_posix()
        for path in inference_destination.rglob("*")
        if path.is_file()
    }
    assert inference_names == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "stage1_peft_metadata.json",
        "ilvr_token_rows.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }


@pytest.mark.parametrize(
    ("zero_stage", "layout"),
    ((1, "replicated"), (2, "replicated"), (3, "partitioned")),
)
def test_checkpoint_layout_is_derived_from_zero_stage(zero_stage, layout):
    assert derive_model_state_layout(zero_stage) == layout

    plan = build_launch_plan(
        stage="stage1",
        tuning_method="full",
        zero_stage=zero_stage,
        run_id=f"stage1-full-zero{zero_stage}-four-gpu",
        main_process_port=29631 + zero_stage,
    )
    observed = _checkpoint_artifact_identity(
        plan,
        commit="b" * 40,
        global_step=2,
    )
    assert observed["model_state_layout"] == layout


def test_observed_checkpoint_identity_rejects_caller_relabelled_sidecar(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    tag = checkpoint / "global_step2"
    tag.mkdir(parents=True)
    (tag / "zero_pp_rank_0_mp_rank_00_model_states.pt").write_bytes(b"model")
    (checkpoint / "latest").write_text("global_step2", encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step":2}\n', encoding="utf-8"
    )
    (checkpoint / "deepspeed-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enabled": True,
                "zero_stage": 3,
                "config_path": "/root/workspace/ILVR/configs/deepspeed_zero3.json",
                "config_digest": "a" * 64,
                "world_size": 4,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "stage1_peft_metadata.json").write_text(
        '{"schema_version":2,"stage":"stage2","tuning_method":"lora"}\n',
        encoding="utf-8",
    )
    for entry in loader_asset_entries():
        (checkpoint / entry.relative_path).write_bytes(b"{}\n")
    (checkpoint / "config.json").write_text(
        json.dumps(standard_qwen_config()) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Stage 1 PEFT|metadata"):
        observe_checkpoint_identity(checkpoint, identity())


def test_checkpoint_directory_rejects_symlinked_run_ancestor(tmp_path):
    storage = tmp_path / "storage"
    logical_root = tmp_path / "checkpoints"
    outside = tmp_path / "outside" / identity()["run_id"] / "checkpoint-2"
    outside.mkdir(parents=True)
    storage.mkdir()
    logical_root.symlink_to(storage, target_is_directory=True)
    (storage / identity()["run_id"]).symlink_to(
        outside.parent, target_is_directory=True
    )

    with pytest.raises(ValueError, match="symlink|trusted"):
        validate_checkpoint_directory(
            logical_root / identity()["run_id"] / "checkpoint-2",
            checkpoint_root=logical_root,
            run_id=identity()["run_id"],
            global_step=2,
        )


def test_manifest_inspection_stays_bound_when_checkpoint_path_is_swapped(
    tmp_path, monkeypatch
):
    import src.multinode_artifacts as artifacts_module

    checkpoint_root = tmp_path / "checkpoints"
    logical = checkpoint_root / full_identity()["run_id"] / "checkpoint-2"
    logical.mkdir(parents=True)
    values = write_observed_full_checkpoint(logical, model_rank=0)
    original_shard = (
        logical / "global_step2/zero_pp_rank_0_mp_rank_00_optim_states.pt"
    )
    original_shard.write_bytes(b"old-owned")

    replacement = tmp_path / "replacement" / "checkpoint-2"
    replacement.mkdir(parents=True)
    replacement_values = write_observed_full_checkpoint(replacement, model_rank=0)
    runtime_path = replacement / "deepspeed-runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["config_digest"] = "c" * 64
    runtime_path.write_text(json.dumps(runtime) + "\n", encoding="utf-8")
    assert replacement_values["deepspeed_config_digest"] == "a" * 64

    original_observe = artifacts_module.observe_checkpoint_identity
    swapped = False

    def swap_after_open(checkpoint, trusted_identity):
        nonlocal swapped
        if not swapped:
            displaced = tmp_path / "displaced-checkpoint"
            logical.rename(displaced)
            replacement.rename(logical)
            swapped = True
        return original_observe(checkpoint, trusted_identity)

    monkeypatch.setattr(
        artifacts_module, "observe_checkpoint_identity", swap_after_open
    )
    manifest = build_node_manifest(
        logical,
        node_rank=0,
        process_ranks=(0, 1),
        runtime_identity=values,
        checkpoint_root=checkpoint_root,
    )

    assert swapped
    assert next(
        entry
        for entry in manifest.entries
        if entry.relative_path == original_shard.relative_to(logical)
    ).byte_size == len(b"old-owned")


def test_small_sidecar_read_stays_bound_when_final_component_is_swapped(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "stage2-full"
    write_loader_artifact(artifact, stage="stage2", method="full")
    sampler = artifact / "stage2-sampler.json"
    sampler.write_text("not-json\n", encoding="utf-8")
    attacker = tmp_path / "attacker-sampler.json"
    attacker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 7,
                "world_size": 4,
                "dataset_size": 4,
                "bucket_sizes": {"1": 4},
                "padded_samples": 0,
                "global_step": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    displaced = tmp_path / "original-sampler.json"
    original_lstat = Path.lstat
    original_open = os.open
    swapped = False
    sampler_lstats = 0

    def swap_final_component():
        nonlocal swapped
        if swapped:
            return
        sampler.rename(displaced)
        sampler.symlink_to(attacker)
        swapped = True

    def swapping_lstat(path, *args, **kwargs):
        nonlocal sampler_lstats
        metadata = original_lstat(path, *args, **kwargs)
        if path == sampler:
            sampler_lstats += 1
            if sampler_lstats == 2:
                swap_final_component()
        return metadata

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == sampler.name and dir_fd is not None:
            swap_final_component()
        return descriptor

    monkeypatch.setattr(Path, "lstat", swapping_lstat)
    monkeypatch.setattr(os, "open", swapping_open)

    with pytest.raises(ValueError, match="sampler.*JSON"):
        validate_loader_artifact(artifact, loader_identity("stage2", "full"))
    assert swapped


def test_inference_peft_selection_requires_complete_stage_metadata_and_rows():
    incomplete = dataclasses.replace(
        strict_manifest(0),
        entries=(
            ArtifactEntry(Path("adapter_config.json"), 10, 0),
            ArtifactEntry(Path("adapter_model.safetensors"), 11, 0),
        )
        + loader_asset_entries(),
    )
    with pytest.raises(ValueError, match="metadata|token-row|complete"):
        select_inference_entries(incomplete)

    complete = dataclasses.replace(
        incomplete,
        entries=incomplete.entries
        + (
            ArtifactEntry(Path("stage1_peft_metadata.json"), 12, 0),
            ArtifactEntry(Path("ilvr_token_rows.safetensors"), 13, 0),
        ),
    )
    assert {entry.relative_path.name for entry in select_inference_entries(complete)} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "stage1_peft_metadata.json",
        "ilvr_token_rows.safetensors",
        "preprocessor_config.json",
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }


def test_inference_full_selection_rejects_partial_or_mixed_shards():
    base = strict_manifest(
        0,
        changes={
            "tuning_method": "full",
            "stage_sidecar": "stage1_ema_teacher.pt",
        },
    )
    partial = dataclasses.replace(
        base,
        entries=(
            ArtifactEntry(Path("model.safetensors.index.json"), 10, 0),
            ArtifactEntry(Path("model-00001-of-00002.safetensors"), 11, 0),
        )
        + loader_asset_entries(),
    )
    with pytest.raises(ValueError, match="complete|shard"):
        select_inference_entries(partial)

    mixed = dataclasses.replace(
        base,
        entries=(
            ArtifactEntry(Path("model.safetensors"), 10, 0),
            ArtifactEntry(Path("pytorch_model.bin"), 11, 0),
        )
        + loader_asset_entries(),
    )
    with pytest.raises(ValueError, match="mixed"):
        select_inference_entries(mixed)


@pytest.mark.parametrize(
    ("stage", "method"),
    (
        ("stage1", "full"),
        ("stage1", "lora"),
        ("stage1", "dora"),
        ("stage2", "full"),
        ("stage2", "lora"),
        ("stage2", "dora"),
    ),
)
def test_resume_requires_real_loader_complete_artifact_for_all_six_rows(
    tmp_path, stage, method
):
    artifact = tmp_path / f"{stage}-{method}"
    write_loader_artifact(artifact, stage=stage, method=method)
    manifest = loader_manifest(artifact, stage=stage, method=method)

    validate_loader_artifact(artifact, manifest.runtime_identity)
    if stage == "stage1":
        from stage1.checkpointing import (
            Stage1ArtifactKind,
            classify_stage1_artifact,
        )

        observed_kind = classify_stage1_artifact(artifact)
        expected_kind = (
            Stage1ArtifactKind.FULL
            if method == "full"
            else Stage1ArtifactKind.PEFT
        )
    else:
        from stage2.checkpointing import (
            Stage2ArtifactKind,
            classify_stage2_artifact,
        )

        observed_kind = classify_stage2_artifact(artifact)
        expected_kind = (
            Stage2ArtifactKind.FULL
            if method == "full"
            else Stage2ArtifactKind.PEFT
        )
    assert observed_kind is expected_kind
    selected = select_resume_entries(manifest)

    names = {entry.relative_path.name for entry in selected}
    assert {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    } <= names


@pytest.mark.parametrize(
    ("stage", "method"),
    (
        ("stage1", "full"),
        ("stage1", "lora"),
        ("stage1", "dora"),
        ("stage2", "full"),
        ("stage2", "lora"),
        ("stage2", "dora"),
    ),
)
def test_resume_rejects_missing_processor_marker_for_all_six_rows(
    tmp_path, stage, method
):
    artifact = tmp_path / f"{stage}-{method}"
    write_loader_artifact(artifact, stage=stage, method=method)
    (artifact / "preprocessor_config.json").unlink()
    manifest = loader_manifest(artifact, stage=stage, method=method)

    with pytest.raises(ValueError, match="preprocessor_config"):
        validate_loader_artifact(artifact, manifest.runtime_identity)
    with pytest.raises(ValueError, match="preprocessor_config"):
        select_resume_entries(manifest)


@pytest.mark.parametrize(
    "missing",
    (
        "adapter_config.json",
        "adapter_model.safetensors",
        "ilvr_token_rows.safetensors",
    ),
)
def test_stage1_lora_resume_rejects_each_required_peft_payload(tmp_path, missing):
    artifact = tmp_path / "stage1-lora"
    write_loader_artifact(artifact, stage="stage1", method="lora")
    (artifact / missing).unlink()

    with pytest.raises(ValueError, match="incomplete|missing|adapter|token-row"):
        validate_loader_artifact(artifact, loader_identity("stage1", "lora"))


@pytest.mark.parametrize("stage", ("stage1", "stage2"))
def test_peft_loader_rejects_unsupported_legacy_adapter_marker(tmp_path, stage):
    artifact = tmp_path / f"{stage}-lora"
    write_loader_artifact(artifact, stage=stage, method="lora")
    (artifact / "adapter_model.bin").write_bytes(b"legacy")

    with pytest.raises(ValueError, match="unsupported|legacy"):
        validate_loader_artifact(artifact, loader_identity(stage, "lora"))


def test_stage2_sparse_rows_are_required_only_by_strict_metadata(tmp_path):
    sparse = tmp_path / "stage2-sparse"
    write_loader_artifact(
        sparse, stage="stage2", method="lora", sparse_stage2=True
    )
    validate_loader_artifact(sparse, loader_identity("stage2", "lora"))
    (sparse / "ilvr_token_rows.safetensors").unlink()
    with pytest.raises(ValueError, match="token-row|sparse"):
        validate_loader_artifact(sparse, loader_identity("stage2", "lora"))

    base = tmp_path / "stage2-base"
    write_loader_artifact(base, stage="stage2", method="lora")
    validate_loader_artifact(base, loader_identity("stage2", "lora"))


def test_stage2_peft_production_validation_never_reads_weight_bytes(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "stage2-sparse"
    write_loader_artifact(
        artifact, stage="stage2", method="lora", sparse_stage2=True
    )
    original_read_bytes = Path.read_bytes

    def reject_weight_read(path, *args, **kwargs):
        if path.suffix in {".pt", ".pth", ".safetensors", ".bin"}:
            raise AssertionError("checkpoint weights must never be opened")
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", reject_weight_read)

    validate_loader_artifact(artifact, loader_identity("stage2", "lora"))


def test_loader_assets_accept_vocab_and_merges_tokenizer_alternative(tmp_path):
    artifact = tmp_path / "stage1-full"
    write_loader_artifact(artifact, stage="stage1", method="full")
    (artifact / "tokenizer.json").unlink()
    (artifact / "vocab.json").write_text("{}\n", encoding="utf-8")
    (artifact / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")

    validate_loader_artifact(artifact, loader_identity("stage1", "full"))


@pytest.mark.parametrize("stage", ("stage1", "stage2"))
def test_full_loader_matches_real_classifier_with_standard_qwen_config_without_stage(
    tmp_path, stage
):
    from transformers import Qwen2_5_VLConfig

    artifact = tmp_path / f"{stage}-full"
    write_loader_artifact(artifact, stage=stage, method="full")
    Qwen2_5_VLConfig().to_json_file(artifact / "config.json")
    assert "stage" not in json.loads(
        (artifact / "config.json").read_text(encoding="utf-8")
    )

    validate_loader_artifact(artifact, loader_identity(stage, "full"))
    if stage == "stage1":
        from stage1.checkpointing import Stage1ArtifactKind, classify_stage1_artifact

        assert classify_stage1_artifact(artifact) is Stage1ArtifactKind.FULL
    else:
        from stage2.checkpointing import Stage2ArtifactKind, classify_stage2_artifact

        assert classify_stage2_artifact(artifact) is Stage2ArtifactKind.FULL


def test_inference_rejects_weights_without_complete_config_processor_assets():
    manifest = dataclasses.replace(
        strict_manifest(
            0,
            changes={
                "tuning_method": "full",
                "stage_sidecar": "stage1_ema_teacher.pt",
            },
        ),
        entries=(ArtifactEntry(Path("model.safetensors"), 10, 0),),
    )

    with pytest.raises(ValueError, match="config.json|processor|tokenizer"):
        select_inference_entries(manifest)


@pytest.mark.parametrize("stage", ("stage1", "stage2"))
def test_full_inference_accepts_standard_qwen_config_without_custom_stage(
    tmp_path, stage
):
    from transformers import Qwen2_5_VLConfig

    artifact = tmp_path / f"{stage}-full"
    write_loader_artifact(artifact, stage=stage, method="full")
    Qwen2_5_VLConfig().to_json_file(artifact / "config.json")
    manifest = loader_manifest(artifact, stage=stage, method="full")

    filtered = build_inference_manifest(artifact, manifest)

    assert "config.json" in {
        entry.relative_path.name for entry in filtered.entries
    }


def test_stage2_inference_manifest_retains_dil_config_without_optimizer_state(tmp_path):
    artifact = tmp_path / "stage2-lora"
    write_loader_artifact(artifact, stage="stage2", method="lora")
    config_path = artifact / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    objective = StepPreferenceConfig(
        enabled=True, mode="stage2b", objective="context_preference",
        preference_loss_type="dil_bce_rf", beta=0.5,
        lambda_entropy=0.0, direct_from_stage1=True,
        shared_prefix_enabled=True,
        visual_off_suppress_latent_boundaries=False,
    )
    identity_payload = build_dil_identity(objective)
    payload["ilvr_context_preference"] = identity_payload
    config_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (artifact / "optimizer.pt").write_bytes(b"optimizer")
    manifest = loader_manifest(artifact, stage="stage2", method="lora")
    selected = build_inference_manifest(artifact, manifest)

    class LocalTransfer:
        def validate_inference_manifest(self, candidate):
            assert candidate == selected

        def pull_inference_artifacts(self, candidate, destination):
            for entry in candidate.entries:
                target = destination / entry.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(artifact / entry.relative_path, target)

    prepared = prepare_inference_copy(selected, tmp_path / "prepared", LocalTransfer())

    names = {entry.relative_path.name for entry in selected.entries}
    assert "config.json" in names
    assert "optimizer.pt" not in names
    assert not (prepared / "optimizer.pt").exists()
    copied = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    assert read_dil_identity(copied) == identity_payload


def test_prepare_inference_help_is_local_and_side_effect_free(tmp_path):
    marker = tmp_path / "remote-called"
    completed = subprocess.run(
        (
            Path(__file__).resolve().parents[2] / "venv/ilvr-venv/bin/python",
            "-m",
            "src.multinode",
            "prepare-inference",
            "--help",
        ),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "ILVR_TEST_REMOTE_MARKER": str(marker),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "--checkpoint-step" in completed.stdout
    assert not marker.exists()


def test_prepared_inference_destination_rejects_result_tree(
    tmp_path, monkeypatch, capsys
):
    import src.multinode as multinode_module

    repository = tmp_path / "repo"
    destination = repository / "results" / "trained-model-run" / "prepared-model"
    monkeypatch.setattr(multinode_module, "_REPOSITORY_ROOT", repository)

    with pytest.raises(SystemExit):
        multinode_module._main(
            [
                "prepare-inference-destination",
                "--destination",
                str(destination),
            ]
        )

    assert "checkpoint evaluation-staging" in capsys.readouterr().err
    assert not destination.exists()


def test_prepared_inference_destination_resolves_the_canonical_checkpoint_link(
    tmp_path, monkeypatch
):
    import src.multinode as multinode_module
    import src.multinode_artifacts as artifacts_module

    repository = tmp_path / "repo"
    repository.mkdir()
    storage = tmp_path / "checkpoint-storage"
    storage.mkdir()
    (repository / "checkpoints").symlink_to(storage, target_is_directory=True)
    destination = (
        repository
        / "checkpoints"
        / "evaluation-staging"
        / identity()["run_id"]
        / "prepared-model"
    )
    monkeypatch.setattr(multinode_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(artifacts_module, "_LOCAL_REPOSITORY", repository)

    result = multinode_module._main(
        ["prepare-inference-destination", "--destination", str(destination)]
    )

    assert result == 0
    assert (
        storage
        / "evaluation-staging"
        / identity()["run_id"]
        / "prepared-model"
    ).is_dir()


def test_prepared_inference_show_rejects_result_tree_before_read(
    tmp_path, monkeypatch, capsys
):
    import src.multinode as multinode_module
    import src.multinode_runtime as runtime_module

    repository = tmp_path / "repo"
    model_dir = repository / "results" / "trained-model-run" / "prepared-model"
    read_called = False

    def fake_read(*args, **kwargs):
        nonlocal read_called
        read_called = True
        return "{}"

    monkeypatch.setattr(multinode_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(runtime_module, "_read_regular_file", fake_read)

    with pytest.raises(SystemExit):
        multinode_module._main(
            ["prepared-inference-show", "--model-dir", str(model_dir)]
        )

    assert "checkpoint evaluation-staging" in capsys.readouterr().err
    assert not read_called


def test_prepared_inference_show_allows_only_the_canonical_checkpoint_link(
    tmp_path, monkeypatch, capsys
):
    import src.multinode as multinode_module
    import src.multinode_artifacts as artifacts_module

    repository = tmp_path / "repo"
    repository.mkdir()
    storage = tmp_path / "checkpoint-storage"
    storage.mkdir()
    (repository / "checkpoints").symlink_to(storage, target_is_directory=True)
    logical_model = (
        repository
        / "checkpoints"
        / "evaluation-staging"
        / identity()["run_id"]
        / "prepared-model"
    )
    physical_model = (
        storage
        / "evaluation-staging"
        / identity()["run_id"]
        / "prepared-model"
    )
    physical_model.mkdir(parents=True)
    manifest = dataclasses.replace(
        strict_manifest(0),
        process_ranks=(0, 1, 2, 3),
        entries=stage1_inference_entries(),
    )
    for entry in manifest.entries:
        path = physical_model / entry.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * entry.byte_size)
    publish_node_manifest(
        artifacts_module.inference_manifest_path(physical_model), manifest
    )
    monkeypatch.setattr(multinode_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(artifacts_module, "_LOCAL_REPOSITORY", repository)

    result = multinode_module._main(
        ["prepared-inference-show", "--model-dir", str(logical_model)]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == manifest.to_json()


@pytest.mark.parametrize(
    ("stage", "method", "zero_stage", "layout"),
    (("stage1", "lora", "3", "partitioned"),
     ("stage2", "full", "1", "replicated")),
)
def test_prepare_inference_cli_dispatches_without_building_training_commands(
    tmp_path, monkeypatch, capsys, stage, method, zero_stage, layout
):
    import src.multinode as multinode_module
    import src.multinode_runtime as runtime_module

    repository = tmp_path / "repo"
    destination = (
        repository
        / "checkpoints"
        / "evaluation-staging"
        / identity()["run_id"]
        / "prepared-model"
    )
    observed = {}

    class FakeCliRunner:
        def __init__(self, plan, *, commands, commit, log_root):
            observed["plan"] = plan
            observed["commands"] = commands
            observed["commit"] = commit
            observed["log_root"] = log_root

    def fake_prepare(
        runner, *, checkpoint_path, runtime_identity, destination
    ):
        observed["runner"] = runner
        observed["checkpoint_path"] = checkpoint_path
        observed["runtime_identity"] = runtime_identity
        observed["destination"] = destination
        return destination

    monkeypatch.setattr(multinode_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        multinode_module, "_REMOTE_CONTAINER_REPOSITORY_BASE", repository
    )
    monkeypatch.setattr(
        multinode_module, "_command_stdout", lambda *args, **kwargs: "b" * 40
    )
    monkeypatch.setattr(
        multinode_module, "SubprocessCoordinatorRunner", FakeCliRunner
    )
    monkeypatch.setattr(
        runtime_module, "prepare_inference_artifacts", fake_prepare
    )

    result = multinode_module._main(
        [
            "prepare-inference",
            "--stage",
            stage,
            "--tuning-method",
            method,
            "--zero-stage",
            zero_stage,
            "--run-id",
            identity()["run_id"],
            "--main-process-port",
            "29631",
            "--checkpoint-step",
            "2",
            "--destination",
            str(destination),
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "inference_artifact": str(destination)
    }
    assert observed["checkpoint_path"] == (
        repository / "checkpoints" / identity()["run_id"] / "checkpoint-2"
    )
    assert observed["commands"] is None
    assert observed["runtime_identity"]["model_state_layout"] == layout
    assert observed["destination"] == destination

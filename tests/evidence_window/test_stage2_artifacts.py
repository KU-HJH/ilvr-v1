from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import trainer_stage2 as trainer_stage2_module
import precompute_stage2_reference as precompute_module
import evidence_window.artifacts as artifacts_module
from main import resolve_stage2_reference_identity
from evidence_window.artifacts import (
    ArtifactIdentity,
    ReferenceBatchTargets,
    ReferenceSampleTargets,
    PrecomputedReferenceProvider,
    publish_reference_manifest,
    write_reference_shard,
)
from evidence_window.losses import stage2_ew_dpo_loss
from evidence_window.stage2_config import (
    Stage2EvidenceWindowConfig,
    Stage2PreferenceConfig,
    Stage2ReferenceConfig,
)
from evidence_window.stage2_effects import Stage2ConditionOutput
from evidence_window.stage2_engine import Stage2EvidenceWindowEngine
from evidence_window.stage2_layout import Stage2InterleavedLayout
from trainer_stage2 import CustomTrainerStage2


def _identity(**changes):
    value = ArtifactIdentity(
        format_version=1,
        dataset_sha256="a" * 64,
        checkpoint_id="stage1-run-42",
        checkpoint_witness="approved-stage1-step-1200",
        tokenizer_id="qwen-tokenizer-rev-a",
        tokenizer_special_tokens=(0, 90, 91, 92, 99),
        parser_version="stage2-marker-v1",
        contribution_reduction="mean",
        intervention="window_attention_block",
        attention_backend="sdpa",
        max_evidence_edges_per_window=0,
        model_config_id="qwen2.5-vl-ilvr-stage2-v1",
    )
    return replace(value, **changes)


def _sample(contributions=None, pair_mask=None, hidden_residuals=None):
    if contributions is None:
        contributions = torch.tensor([[0.7], [0.2]], dtype=torch.float32)
    if pair_mask is None:
        pair_mask = torch.tensor([[True], [True]])
    return ReferenceSampleTargets(contributions, pair_mask, hidden_residuals)


def _layout(pair_mask=None):
    if pair_mask is None:
        pair_mask = torch.tensor([[[True], [True]]])
    return Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor(
            [[[1, 0, 0, 0], [0, 1, 0, 0]]], dtype=torch.bool
        ),
        window_token_mask=torch.tensor([[[0, 0, 1, 1]]], dtype=torch.bool),
        evidence_valid=torch.tensor([[True, True]]),
        window_valid=torch.tensor([[True]]),
        causal_pair_mask=pair_mask,
    )


def _write_single(tmp_path, identity=None, sample=None):
    identity = _identity() if identity is None else identity
    sample = _sample() if sample is None else sample
    metadata = write_reference_shard(
        tmp_path,
        identity,
        {7: sample},
        rank=0,
        publish_manifest=False,
    )
    publish_reference_manifest(tmp_path, identity, [metadata])


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("format_version", 2),
        ("dataset_sha256", "b" * 64),
        ("checkpoint_id", "other-checkpoint"),
        ("checkpoint_witness", "other-witness"),
        ("tokenizer_id", "other-tokenizer"),
        ("tokenizer_special_tokens", (0, 90, 91)),
        ("parser_version", "other-parser"),
        ("contribution_reduction", "sum"),
        ("intervention", "not-the-intervention"),
        ("attention_backend", "flash_attention_2"),
        ("max_evidence_edges_per_window", 1),
        ("model_config_id", "other-model-config"),
    ],
)
def test_identity_mismatch_rejects_every_semantic_field(tmp_path, field, wrong):
    """Changing any scoring identity field must reject an otherwise valid artifact."""
    _write_single(tmp_path)

    with pytest.raises(ValueError, match=field):
        PrecomputedReferenceProvider(tmp_path, replace(_identity(), **{field: wrong}))


def test_provider_loads_primitive_tensor_shards_with_weights_only_and_preserves_order(
    tmp_path, monkeypatch
):
    """Unsafe pickle loading or sorting the requested batch would violate the contract."""
    identity = _identity()
    metadata = write_reference_shard(
        tmp_path,
        identity,
        {9: _sample(torch.tensor([[9.0], [9.5]])), 7: _sample()},
        rank=3,
        publish_manifest=False,
    )
    publish_reference_manifest(tmp_path, identity, [metadata])
    real_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_load(*args, **kwargs)

    monkeypatch.setattr("evidence_window.artifacts.torch.load", checked_load)
    provider = PrecomputedReferenceProvider(tmp_path, identity)
    layout = Stage2InterleavedLayout(
        evidence_token_mask=_layout().evidence_token_mask.expand(2, -1, -1),
        window_token_mask=_layout().window_token_mask.expand(2, -1, -1),
        evidence_valid=torch.tensor([[True, True], [True, True]]),
        window_valid=torch.tensor([[True], [True]]),
        causal_pair_mask=torch.tensor([[[True], [True]], [[True], [True]]]),
    )
    batch = provider.get_batch(torch.tensor([9, 7]), layout)

    assert calls and all(call.get("weights_only") is True for call in calls)
    assert batch.sample_indices == (9, 7)
    torch.testing.assert_close(batch.contributions[:, 0, 0], torch.tensor([9.0, 0.7]))
    assert batch.hidden_residuals is None


def test_provider_rejects_duplicate_sample_across_shards(tmp_path):
    """A duplicated source row must never be selected by filesystem iteration order."""
    identity = _identity()
    first = write_reference_shard(
        tmp_path, identity, {7: _sample()}, rank=0, publish_manifest=False
    )
    second = write_reference_shard(
        tmp_path, identity, {7: _sample()}, rank=1, publish_manifest=False
    )

    with pytest.raises(ValueError, match="duplicate.*sample.*7"):
        publish_reference_manifest(tmp_path, identity, [first, second])


def test_provider_rejects_missing_sample_and_pair_layout_mismatch(tmp_path):
    """Stale or incomplete targets must hard-fail instead of silently falling back."""
    _write_single(tmp_path)
    provider = PrecomputedReferenceProvider(tmp_path, _identity())

    with pytest.raises(KeyError, match="sample.*8"):
        provider.get_batch([8], _layout())
    mismatched = _layout(torch.tensor([[[True], [False]]]))
    with pytest.raises(ValueError, match="pair_mask.*sample 7"):
        provider.get_batch([7], mismatched)


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (
            ReferenceSampleTargets(
                torch.ones(2, 1, 1), torch.ones(2, 1, dtype=torch.bool), None
            ),
            "contributions",
        ),
        (
            ReferenceSampleTargets(torch.ones(2, 1), torch.ones(2, 1), None),
            "pair_mask.*bool",
        ),
        (
            ReferenceSampleTargets(
                torch.tensor([[float("nan")], [0.0]]),
                torch.ones(2, 1, dtype=torch.bool),
                None,
            ),
            "finite",
        ),
        (
            ReferenceSampleTargets(
                torch.ones(2, 1),
                torch.ones(2, 1, dtype=torch.bool),
                torch.ones(2, 2, 3),
            ),
            "hidden_residuals",
        ),
    ],
)
def test_writer_rejects_malformed_tensor_schema(tmp_path, sample, message):
    """Malformed dimensions, dtypes, or values must not enter an immutable shard."""
    with pytest.raises(ValueError, match=message):
        write_reference_shard(tmp_path, _identity(), {7: sample})


def test_optional_residual_absence_is_explicit(tmp_path):
    """Contribution-only artifacts load, but a residual consumer gets a hard error."""
    _write_single(tmp_path)
    provider = PrecomputedReferenceProvider(tmp_path, _identity())

    assert provider.get_batch([7], _layout()).hidden_residuals is None
    with pytest.raises(ValueError, match="hidden residual"):
        provider.get_batch([7], _layout(), require_hidden_residuals=True)


def test_mixed_batch_requires_hidden_only_for_rows_with_selected_edges(tmp_path):
    """Edge-free rows may omit hidden targets without creating fake residual edges."""
    identity = _identity()
    edge_hidden = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    edge_shard = write_reference_shard(
        tmp_path,
        identity,
        {7: _sample(hidden_residuals=edge_hidden)},
        rank=0,
        publish_manifest=False,
    )
    empty_shard = write_reference_shard(
        tmp_path,
        identity,
        {
            8: ReferenceSampleTargets(
                contributions=torch.zeros((0, 1)),
                pair_mask=torch.zeros((0, 1), dtype=torch.bool),
                hidden_residuals=None,
            )
        },
        rank=1,
        publish_manifest=False,
    )
    publish_reference_manifest(tmp_path, identity, [edge_shard, empty_shard])
    provider = PrecomputedReferenceProvider(tmp_path, identity)
    layout = Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.bool,
        ),
        window_token_mask=torch.tensor(
            [[[0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=torch.bool
        ),
        evidence_valid=torch.tensor([[True, True], [False, False]]),
        window_valid=torch.tensor([[True], [True]]),
        causal_pair_mask=torch.tensor(
            [[[True], [True]], [[False], [False]]], dtype=torch.bool
        ),
    )

    batch = provider.get_batch(
        [7, 8], layout, require_hidden_residuals=True
    )

    assert batch.hidden_residuals is not None
    assert batch.hidden_residuals.shape == (2, 2, 1, 3)
    torch.testing.assert_close(batch.hidden_residuals[0], edge_hidden)
    assert torch.count_nonzero(batch.hidden_residuals[1]).item() == 0
    assert torch.equal(batch.pair_mask, layout.causal_pair_mask)
    assert not batch.pair_mask[1].any()


def test_manifest_missing_or_malformed_shard_is_rejected(tmp_path):
    """The manifest must not authorize absent or structurally altered shard payloads."""
    identity = _identity()
    metadata = write_reference_shard(
        tmp_path, identity, {7: _sample()}, rank=0, publish_manifest=False
    )
    publish_reference_manifest(tmp_path, identity, [metadata])
    shard = tmp_path / metadata.filename
    shard.unlink()
    with pytest.raises(ValueError, match="missing shard"):
        PrecomputedReferenceProvider(tmp_path, identity)

    torch.save({"wrong": "layout"}, shard)
    with pytest.raises(ValueError, match="shard.*schema"):
        PrecomputedReferenceProvider(tmp_path, identity)


def test_concurrent_shard_writers_have_one_immutable_winner(tmp_path):
    """Concurrent producers must never overwrite a complete final shard."""
    identity = _identity()
    first = _sample(torch.tensor([[1.0], [1.5]]))
    second = _sample(torch.tensor([[2.0], [2.5]]))

    def write(value):
        return write_reference_shard(
            tmp_path,
            identity,
            {7: value},
            rank=0,
            publish_manifest=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write, value) for value in (first, second)]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception() is not None]

    assert len(successes) == 1
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)
    payload = torch.load(
        tmp_path / "shard-rank-00000.pt", map_location="cpu", weights_only=True
    )
    stored = payload["records"][0]["contributions"]
    assert torch.equal(stored, first.contributions) or torch.equal(
        stored, second.contributions
    )
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_atomic_publish_failure_exposes_no_partial_final_and_cleans_temp(
    tmp_path, monkeypatch
):
    """Failure before the atomic link must leave the final name retryable."""
    def fail_link(_source, _destination):
        raise OSError("injected publish failure")

    monkeypatch.setattr(artifacts_module.os, "link", fail_link)

    with pytest.raises(OSError, match="injected publish failure"):
        write_reference_shard(
            tmp_path,
            _identity(),
            {7: _sample()},
            rank=0,
            publish_manifest=False,
        )

    assert not (tmp_path / "shard-rank-00000.pt").exists()
    assert not (tmp_path / "shard-rank-00000.metadata.json").exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_atomic_manifest_publish_never_overwrites_existing_final(tmp_path):
    """A second publisher cannot replace an already sealed manifest."""
    identity = _identity()
    metadata = write_reference_shard(
        tmp_path, identity, {7: _sample()}, rank=0, publish_manifest=False
    )
    publish_reference_manifest(tmp_path, identity, [metadata])
    original = (tmp_path / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="manifest.json"):
        publish_reference_manifest(tmp_path, identity, [metadata])

    assert (tmp_path / "manifest.json").read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_provider_rejects_duplicate_manifest_rank_and_filename_entries(tmp_path):
    """A malicious manifest must not make shard ownership iteration-dependent."""
    identity = _identity()
    metadata = write_reference_shard(
        tmp_path, identity, {7: _sample()}, rank=0, publish_manifest=False
    )
    manifest_path = publish_reference_manifest(tmp_path, identity, [metadata])
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"].append(dict(manifest["shards"][0]))
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="duplicate.*rank.*filename"):
        PrecomputedReferenceProvider(tmp_path, identity)


@pytest.mark.parametrize(
    ("location", "field", "value", "message"),
    [
        ("identity", "format_version", 1.0, "format_version.*integer"),
        ("metadata", "rank", 0.0, "rank.*integer"),
        ("metadata", "record_count", 1.0, "record_count.*integer"),
        ("metadata", "has_hidden_residuals", 0, "residual flag.*boolean"),
    ],
)
def test_provider_rejects_equal_but_wrong_manifest_primitive_types(
    tmp_path, location, field, value, message
):
    """Python equality must not let floats or ints impersonate schema primitives."""
    identity = _identity()
    metadata = write_reference_shard(
        tmp_path, identity, {7: _sample()}, rank=0, publish_manifest=False
    )
    manifest_path = publish_reference_manifest(tmp_path, identity, [metadata])
    manifest = json.loads(manifest_path.read_text())
    target = manifest["identity"] if location == "identity" else manifest["shards"][0]
    target[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        PrecomputedReferenceProvider(tmp_path, identity)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rank", 0.0, "rank.*integer"),
        ("sample_indices", [7.0], "sample index.*integer"),
        ("record_sample_index", 7.0, "sample index.*integer"),
    ],
)
def test_provider_rejects_equal_but_wrong_shard_primitive_types(
    tmp_path, field, value, message
):
    """Tensor-shard primitive schema is strict before equality comparisons."""
    identity = _identity()
    metadata = write_reference_shard(
        tmp_path, identity, {7: _sample()}, rank=0, publish_manifest=False
    )
    publish_reference_manifest(tmp_path, identity, [metadata])
    shard_path = tmp_path / metadata.filename
    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    if field == "record_sample_index":
        payload["records"][0]["sample_index"] = value
    else:
        payload[field] = value
    shard_path.unlink()
    torch.save(payload, shard_path)

    with pytest.raises(ValueError, match=message):
        PrecomputedReferenceProvider(tmp_path, identity)


def test_manifest_is_published_from_metadata_without_loading_shards(tmp_path, monkeypatch):
    """Rank zero must publish the small manifest without rereading tensor shards."""
    identity = _identity()
    metadata = write_reference_shard(
        tmp_path, identity, {7: _sample()}, rank=0, publish_manifest=False
    )
    monkeypatch.setattr(
        "evidence_window.artifacts.torch.load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("reread shard")),
    )

    publish_reference_manifest(tmp_path, identity, [metadata])

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["shards"][0]["sample_indices"] == [7]


def test_precompute_identity_hashes_dataset_once_and_uses_explicit_checkpoint_witness(
    tmp_path, monkeypatch
):
    """Checkpoint trees must never be hashed, and dataset content gets one producer pass."""
    dataset = tmp_path / "rows.jsonl"
    dataset.write_bytes(b'{"row":0}\n{"row":1}\n')
    calls = []
    real_sha = precompute_module.sha256_file

    def one_hash(path):
        calls.append(str(path))
        return real_sha(path)

    monkeypatch.setattr(precompute_module, "sha256_file", one_hash)
    args = precompute_module.parse_args(
        [
            "--data-path",
            str(dataset),
            "--reference-checkpoint",
            str(tmp_path / "large-checkpoint-directory"),
            "--reference-checkpoint-id",
            "stage1-run-42",
            "--reference-checkpoint-witness",
            "approved-step-1200",
            "--artifact-dir",
            str(tmp_path / "artifact"),
            "--tokenizer-id",
            "tokenizer-rev-a",
            "--model-config-id",
            "model-config-rev-a",
        ]
    )

    identity = precompute_module.build_artifact_identity(
        args,
        tokenizer_special_tokens=(0, 90, 91, 92),
        dataset_sha256=None,
    )

    assert calls == [str(dataset)]
    assert identity.checkpoint_id == "stage1-run-42"
    assert identity.checkpoint_witness == "approved-step-1200"
    assert identity.dataset_sha256 == real_sha(dataset)


def test_precompute_identity_reuses_supplied_dataset_witness_without_rehash(
    tmp_path, monkeypatch
):
    """Distributed ranks consume the producer witness instead of rereading the dataset."""
    args = precompute_module.parse_args(
        [
            "--data-path",
            str(tmp_path / "unread.jsonl"),
            "--dataset-sha256-witness",
            "c" * 64,
            "--reference-checkpoint",
            "/checkpoint",
            "--reference-checkpoint-id",
            "stage1-run-42",
            "--reference-checkpoint-witness",
            "approved-step-1200",
            "--artifact-dir",
            str(tmp_path / "artifact"),
            "--tokenizer-id",
            "tokenizer-rev-a",
            "--model-config-id",
            "model-config-rev-a",
        ]
    )
    monkeypatch.setattr(
        precompute_module,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("dataset rehash")),
    )

    identity = precompute_module.build_artifact_identity(
        args,
        tokenizer_special_tokens=(0, 90, 91, 92),
        dataset_sha256=args.dataset_sha256_witness,
    )

    assert identity.dataset_sha256 == "c" * 64


def test_precompute_cli_rejects_model_batch_larger_than_one():
    """The heterogeneous Stage-2 model path must not silently score batched rows."""
    with pytest.raises(ValueError, match="batch size 1"):
        precompute_module.parse_args(
            [
                "--data-path", "/data.jsonl",
                "--reference-checkpoint", "/checkpoint",
                "--reference-checkpoint-id", "stage1",
                "--reference-checkpoint-witness", "sealed",
                "--artifact-dir", "/artifact",
                "--tokenizer-id", "tokenizer",
                "--model-config-id", "model",
                "--batch-size", "2",
            ]
        )


@pytest.mark.parametrize("entry_name", ["partial.pt", "shard-rank-00000.pt"])
def test_nonempty_artifact_destination_fails_before_model_load(
    tmp_path, monkeypatch, entry_name
):
    """A partial/reserved artifact directory must fail before loading a model."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / entry_name).write_bytes(b"partial")
    args = SimpleNamespace(
        artifact_dir=str(artifact_dir),
        dataset_sha256_witness="c" * 64,
    )
    loads = []
    monkeypatch.setattr(precompute_module, "_initialize_distributed_if_requested", lambda: None)
    monkeypatch.setattr(precompute_module.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        precompute_module,
        "_produce_rank_shard",
        lambda *args, **kwargs: loads.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="artifact precheck.*nonempty"):
        precompute_module.precompute(args)

    assert loads == []


def test_dangling_artifact_symlink_fails_before_model_load(tmp_path, monkeypatch):
    """A dangling destination symlink must not bypass the cheap artifact gate."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    args = SimpleNamespace(
        artifact_dir=str(artifact_dir),
        dataset_sha256_witness=None,
        data_path=str(tmp_path / "data.jsonl"),
    )
    loads = []
    monkeypatch.setattr(precompute_module, "_initialize_distributed_if_requested", lambda: None)
    monkeypatch.setattr(precompute_module.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        precompute_module,
        "sha256_file",
        lambda path: loads.append(("dataset", path)) or "c" * 64,
    )
    monkeypatch.setattr(
        precompute_module,
        "_produce_rank_shard",
        lambda *args, **kwargs: loads.append(("model", args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="artifact precheck.*symlink"):
        precompute_module.precompute(args)

    assert loads == []


@pytest.mark.parametrize(
    ("phase", "local_failure", "remote_failure"),
    [
        ("artifact precheck", "manifest already exists", None),
        ("shard production", None, "rank 1: data decode failed"),
        ("manifest publication", "rank 0 publish failed", None),
    ],
)
def test_distributed_precompute_phase_propagates_local_or_remote_failure(
    monkeypatch, phase, local_failure, remote_failure
):
    """Every rank must make the same next-collective decision after each phase."""
    monkeypatch.setattr(precompute_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(precompute_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(precompute_module.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(precompute_module.distributed, "get_backend", lambda: "gloo")
    collectives = []

    def reduce_failure(value, op):
        collectives.append(("reduce", op))
        if remote_failure is not None:
            value.fill_(1)

    def gather_errors(output, local):
        collectives.append(("gather", local))
        output[:] = [local, remote_failure]

    monkeypatch.setattr(precompute_module.distributed, "all_reduce", reduce_failure)
    monkeypatch.setattr(
        precompute_module.distributed, "all_gather_object", gather_errors
    )
    monkeypatch.setattr(
        precompute_module.distributed,
        "barrier",
        lambda: (_ for _ in ()).throw(AssertionError("unsafe barrier")),
    )

    def operation():
        if local_failure is not None:
            raise OSError(local_failure)
        return "completed"

    expected = local_failure if local_failure is not None else remote_failure
    with pytest.raises(RuntimeError, match=expected):
        precompute_module.run_distributed_phase(
            phase, operation, device=torch.device("cpu")
        )

    assert [name for name, _ in collectives] == ["reduce", "gather"]


def test_dataset_hash_failure_is_coordinated_before_witness_broadcast(
    tmp_path, monkeypatch
):
    """Rank-zero hash failure must not strand peers inside broadcast_object_list."""
    args = type(
        "Args",
        (),
        {"dataset_sha256_witness": None, "data_path": str(tmp_path / "missing")},
    )()
    monkeypatch.setattr(precompute_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(precompute_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(precompute_module.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(precompute_module.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        precompute_module.distributed,
        "all_reduce",
        lambda value, op: None,
    )

    def gather_errors(output, local):
        output[:] = [local, None]

    monkeypatch.setattr(
        precompute_module.distributed, "all_gather_object", gather_errors
    )
    monkeypatch.setattr(
        precompute_module.distributed,
        "broadcast_object_list",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("broadcast entered after hash failure")
        ),
    )

    with pytest.raises(RuntimeError, match="dataset witness.*FileNotFoundError"):
        precompute_module._distributed_dataset_witness(args, rank=0)


def test_reference_effect_rows_are_unpadded_and_detached():
    """Shard rows must not retain batch padding or an autograd graph."""
    effects = type(
        "Effects",
        (),
        {
            "contributions": torch.tensor(
                [[[0.7], [0.2]], [[1.0], [0.0]]], requires_grad=True
            ),
            "pair_mask": torch.tensor([[[True], [True]], [[True], [False]]]),
            "hidden_residuals": None,
        },
    )()
    layout = Stage2InterleavedLayout(
        evidence_token_mask=torch.zeros((2, 2, 4), dtype=torch.bool),
        window_token_mask=torch.zeros((2, 1, 4), dtype=torch.bool),
        evidence_valid=torch.tensor([[True, True], [True, False]]),
        window_valid=torch.tensor([[True], [True]]),
        causal_pair_mask=torch.tensor([[[True], [True]], [[True], [False]]]),
    )

    rows = precompute_module.reference_rows_from_effects(
        torch.tensor([9, 7]), layout, effects
    )

    assert rows[9].contributions.shape == (2, 1)
    assert rows[7].contributions.shape == (1, 1)
    assert rows[9].contributions.requires_grad is False
    assert rows[7].pair_mask.tolist() == [[True]]


def test_dpo_stops_reference_gradient():
    """Reference targets are constants in the edge-local DPO margin."""
    policy = torch.tensor([0.7], requires_grad=True)
    reference = torch.tensor([0.2], requires_grad=True)

    loss = stage2_ew_dpo_loss(policy, reference, torch.tensor([True]), 0.4)
    loss.backward()

    expected = -F.logsigmoid(torch.tensor(0.4 * (0.7 - 0.2)))
    torch.testing.assert_close(loss, expected)
    assert policy.grad is not None
    assert reference.grad is None


def test_precomputed_engine_uses_policy_minus_detached_reference():
    """Precomputed mode must change only the local margin, not policy effect computation."""
    config = Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(mode="precomputed", artifact="/artifact"),
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=2.0, beta=0.4),
    )
    engine = Stage2EvidenceWindowEngine(config)
    full = torch.tensor([[0.0, 0.0, 3.0, 3.0]], requires_grad=True)
    loo_values = iter(
        [
            torch.tensor([[0.0, 0.0, 1.0, 1.0]], requires_grad=True),
            torch.tensor([[0.0, 0.0, 2.0, 2.0]], requires_grad=True),
        ]
    )

    def policy_forward(block_mask, need_hidden):
        assert need_hidden is False
        return Stage2ConditionOutput(full if block_mask is None else next(loo_values), None)

    reference = torch.tensor([[[0.5], [0.25]]], requires_grad=True)
    targets = ReferenceBatchTargets(
        sample_indices=(7,),
        contributions=reference,
        pair_mask=torch.tensor([[[True], [True]]]),
        hidden_residuals=None,
    )
    result = engine.compute(_layout(), policy_forward, 1.0, reference_targets=targets)

    policy_values = result.effects.contributions[result.effects.pair_mask]
    expected = -F.logsigmoid(0.4 * (policy_values - reference.detach().flatten())).mean()
    torch.testing.assert_close(result.losses["ew_pref"], expected)
    contribution_sum, contribution_count = result.metric_sums["ew/mean_contribution"]
    margin_sum, margin_count = result.metric_sums["ew/mean_margin"]
    global_margin_sum, global_margin_count = result.metric_sums["ew/global_margin"]
    torch.testing.assert_close(contribution_sum / contribution_count, torch.tensor(1.5))
    torch.testing.assert_close(margin_sum / margin_count, torch.tensor(1.125))
    torch.testing.assert_close(global_margin_sum / global_margin_count, torch.tensor(2.25))
    result.auxiliary_loss.backward()
    assert full.grad is not None
    assert reference.grad is None


class _RecordingProvider:
    def __init__(self, targets):
        self.targets = targets
        self.calls = []

    def get_batch(self, sample_indices, layout, *, require_hidden_residuals=False):
        self.calls.append((tuple(sample_indices.tolist()), layout, require_hidden_residuals))
        return self.targets


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 4)
        self.head = torch.nn.Linear(4, 128)
        self.calls = 0

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        attention_block_mask=None,
        attention_backend_override=None,
        output_hidden_states=False,
        return_dict=True,
        use_cache=None,
    ):
        del return_dict, use_cache, attention_backend_override
        self.calls += 1
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        if attention_block_mask is not None:
            hidden = hidden - attention_block_mask.any(dim=-1).unsqueeze(-1) * 0.1
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return type("Output", (), {"loss": loss, "logits": logits, "hidden_states": hidden})()


def test_trainer_looks_up_precomputed_targets_by_stable_sample_index(
    monkeypatch, tmp_path
):
    """Batch position must never be mistaken for the deterministic dataset row index."""
    def fake_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = type("Args", (), {"output_dir": str(tmp_path)})()
        instance.state = type("State", (), {"global_step": 1, "max_steps": 1})()
        instance.accelerator = type("Accelerator", (), {"process_index": 0})()

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_init)
    targets = ReferenceBatchTargets(
        sample_indices=(17,),
        contributions=torch.tensor([[[0.2]]]),
        pair_mask=torch.tensor([[[True]]]),
        hidden_residuals=None,
    )
    provider = _RecordingProvider(targets)
    config = Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(mode="precomputed", artifact="/artifact"),
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=1.0, beta=0.4),
    )
    model = _TinyModel()
    trainer = CustomTrainerStage2(
        model=model,
        evidence_window_config=config,
        latent_token_ids={
            "start": 90,
            "end": 91,
            "pad": 92,
            "padding": 0,
            "special": (0, 90, 91, 92, 99),
        },
        reference_provider=provider,
    )
    batch = {
        "input_ids": torch.tensor([[90, 92, 91, 5, 6]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 5, 6]]),
        "ew_sample_indices": torch.tensor([17]),
        "ew_plan_types": (("latent", "text"),),
    }

    loss = trainer.compute_loss(model, batch)
    loss.backward()

    assert provider.calls[0][0] == (17,)
    assert model.embedding.weight.grad is not None


def test_training_runtime_builds_independent_expected_artifact_identity():
    """Training must compare producer witnesses, not adopt them from its manifest."""
    args = type(
        "Args",
        (),
        {
            "reference_dataset_sha256": "d" * 64,
            "reference_checkpoint_id": "stage1-run-42",
            "reference_checkpoint_witness": "approved-step-1200",
            "reference_tokenizer_id": "tokenizer-rev-a",
            "reference_parser_version": "stage2-marker-v1",
            "reference_model_config_id": "model-config-rev-a",
        },
    )()
    config = Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(mode="precomputed", artifact="/artifact"),
        preference=Stage2PreferenceConfig(
            enabled=True,
            lambda_ew=1.0,
            beta=0.4,
            contribution_reduction="sum",
        ),
    )

    identity = resolve_stage2_reference_identity(
        args, config, tokenizer_special_tokens=(0, 90, 91, 92, 99)
    )

    assert identity.dataset_sha256 == "d" * 64
    assert identity.checkpoint_witness == "approved-step-1200"
    assert identity.contribution_reduction == "sum"
    assert identity.intervention == "window_attention_block"


def test_precomputed_trainer_requires_expected_identity_when_it_builds_provider(
    monkeypatch, tmp_path
):
    """A trainer-created provider must never trust manifest identity as expectation."""
    def fake_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_init)
    config = Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(mode="precomputed", artifact=str(tmp_path)),
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=1.0),
    )

    with pytest.raises(ValueError, match="expected.*identity"):
        CustomTrainerStage2(
            model=_TinyModel(),
            evidence_window_config=config,
            latent_token_ids={
                "start": 90,
                "end": 91,
                "pad": 92,
                "padding": 0,
                "special": (0, 90, 91, 92, 99),
            },
        )


def test_distributed_reference_lookup_replays_remote_failure_before_auxiliary_forward(
    monkeypatch, tmp_path
):
    """One rank's missing artifact row must stop every rank before model replay collectives."""
    def fake_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = type("Args", (), {"output_dir": str(tmp_path)})()
        instance.state = type("State", (), {"global_step": 1, "max_steps": 1})()
        instance.accelerator = type("Accelerator", (), {"process_index": 0})()

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_init)
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "get_backend", lambda: "gloo")

    reductions = []

    def inject_remote_failure(value, op):
        del op
        reductions.append(value.item())
        if len(reductions) == 2:
            value.fill_(1)

    def remote_messages(output, local):
        assert local is None
        output[:] = [None, "rank 1: reference artifact has no sample 23"]

    monkeypatch.setattr(trainer_stage2_module.distributed, "all_reduce", inject_remote_failure)
    monkeypatch.setattr(trainer_stage2_module.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        trainer_stage2_module.distributed, "all_gather_object", remote_messages
    )
    targets = ReferenceBatchTargets(
        sample_indices=(17,),
        contributions=torch.tensor([[[0.2]]]),
        pair_mask=torch.tensor([[[True]]]),
        hidden_residuals=None,
    )
    provider = _RecordingProvider(targets)
    config = Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(mode="precomputed", artifact="/artifact"),
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=1.0),
    )
    model = _TinyModel()
    trainer = CustomTrainerStage2(
        model=model,
        evidence_window_config=config,
        latent_token_ids={
            "start": 90,
            "end": 91,
            "pad": 92,
            "padding": 0,
            "special": (0, 90, 91, 92, 99),
        },
        reference_provider=provider,
    )
    batch = {
        "input_ids": torch.tensor([[90, 92, 91, 5, 6]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 5, 6]]),
        "ew_sample_indices": torch.tensor([17]),
        "ew_plan_types": (("latent", "text"),),
    }

    with pytest.raises(RuntimeError, match="rank 1.*no sample 23"):
        trainer.compute_loss(model, batch)

    assert model.calls == 1
    assert len(reductions) == 2


def test_trainer_rejects_reference_batch_returned_in_wrong_sample_order(
    monkeypatch, tmp_path
):
    """A provider must not attach valid tensors to the wrong source-row order."""
    def fake_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = type("Args", (), {"output_dir": str(tmp_path)})()
        instance.state = type("State", (), {"global_step": 1, "max_steps": 1})()
        instance.accelerator = type("Accelerator", (), {"process_index": 0})()

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_init)
    targets = ReferenceBatchTargets(
        sample_indices=(99,),
        contributions=torch.tensor([[[0.2]]]),
        pair_mask=torch.tensor([[[True]]]),
        hidden_residuals=None,
    )
    config = Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(mode="precomputed", artifact="/artifact"),
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=1.0),
    )
    model = _TinyModel()
    trainer = CustomTrainerStage2(
        model=model,
        evidence_window_config=config,
        latent_token_ids={
            "start": 90,
            "end": 91,
            "pad": 92,
            "padding": 0,
            "special": (0, 90, 91, 92, 99),
        },
        reference_provider=_RecordingProvider(targets),
    )
    batch = {
        "input_ids": torch.tensor([[90, 92, 91, 5, 6]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 5, 6]]),
        "ew_sample_indices": torch.tensor([17]),
        "ew_plan_types": (("latent", "text"),),
    }

    with pytest.raises(ValueError, match="sample.*order"):
        trainer.compute_loss(model, batch)

    assert model.calls == 1

from dataclasses import replace
import json

import pytest
import torch
import torch.nn.functional as F

import trainer_stage2 as trainer_stage2_module
import precompute_stage2_reference as precompute_module
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

    def inject_remote_failure(value, op):
        del op
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

import importlib.util
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evidence_window.artifacts import (
    ArtifactIdentity,
    ReferenceBatchTargets,
    ReferenceSampleTargets,
    write_reference_shard,
)
from evidence_window.stage2_config import Stage2EvidenceWindowConfig
from evidence_window.stage2_effects import Stage2ConditionOutput
from evidence_window.stage2_engine import Stage2EvidenceWindowEngine
from evidence_window.stage2_layout import Stage2InterleavedLayout


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPOSITORY_ROOT / "configs" / "evidence_window" / "stage2"
LARGE_SMOKE = (
    REPOSITORY_ROOT
    / ".superpowers"
    / "sdd"
    / "2026-08-22-ilvr-stage2-evidence-window-credit"
    / "stage2_large_model_smoke.py"
)


def _layout() -> Stage2InterleavedLayout:
    return Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0]]],
            dtype=torch.bool,
        ),
        window_token_mask=torch.tensor(
            [[[0, 0, 0, 0, 0, 0, 1]]], dtype=torch.bool
        ),
        evidence_valid=torch.tensor([[True, True]]),
        window_valid=torch.tensor([[True]]),
        causal_pair_mask=torch.tensor([[[True], [True]]]),
    )


def _reference(*, hidden: bool) -> ReferenceBatchTargets:
    return ReferenceBatchTargets(
        sample_indices=(0,),
        contributions=torch.tensor([[[0.2], [-0.1]]], requires_grad=True),
        pair_mask=torch.ones((1, 2, 1), dtype=torch.bool),
        hidden_residuals=(
            torch.tensor(
                [[[[0.4, -0.2]], [[-0.1, 0.3]]]], requires_grad=True
            )
            if hidden
            else None
        ),
    )


class _CountingStage2Policy(torch.nn.Module):
    """One real module that distinguishes ordinary CE and auxiliary calls."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))
        self.calls: list[tuple[str, bool, bool]] = []

    def forward(self, *, labels=None, block_mask=None, need_hidden=False):
        role = "ce" if labels is not None else "aux"
        self.calls.append((role, block_mask is not None, need_hidden))
        if labels is not None:
            return (self.scale - 0.25).square()
        blocked_ordinal = sum(
            blocked for current_role, blocked, _ in self.calls if current_role == "aux"
        ) - 1
        condition_scale = (
            self.scale
            if block_mask is None
            else self.scale * (0.6 if blocked_ordinal == 0 else 0.8)
        )
        token_logps = condition_scale * torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        hidden_state = None
        if need_hidden:
            hidden_state = condition_scale * torch.tensor(
                [[
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 1.0],
                    [0.0, 0.0],
                ]]
            )
        return Stage2ConditionOutput(token_logps, hidden_state)


@pytest.mark.parametrize(
    "name,expected_loss,needs_reference,needs_hidden",
    [
        ("baseline.json", None, False, False),
        ("ew_reference_free.json", "ew_pref", False, False),
        ("ew_precomputed_dpo.json", "ew_pref", True, False),
        ("relation_kd.json", "relation", True, False),
        ("hidden_residual_kd.json", "residual", True, True),
    ],
)
def test_cpu_one_batch_objective_smoke(
    name, expected_loss, needs_reference, needs_hidden
):
    """Each shipped objective must take one finite differentiable tiny optimizer step."""
    config = Stage2EvidenceWindowConfig.from_json(CONFIG_ROOT / name)
    model = _CountingStage2Policy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    ce = model(labels=torch.tensor([[1]]))
    reference_targets = _reference(hidden=needs_hidden) if needs_reference else None
    if config.is_disabled:
        total = ce
        result = None
    else:
        result = Stage2EvidenceWindowEngine(config).compute(
            _layout(),
            lambda block, hidden: model(block_mask=block, need_hidden=hidden),
            progress=1.0,
            reference_targets=reference_targets,
        )
        total = ce + result.auxiliary_loss

        auxiliary_gradient = torch.autograd.grad(
            result.auxiliary_loss,
            model.scale,
            retain_graph=True,
        )[0]
        assert torch.isfinite(auxiliary_gradient)
        assert auxiliary_gradient.abs().item() > 0.0

    before = model.scale.detach().clone()
    optimizer.zero_grad()
    total.backward()
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
    optimizer.step()

    assert torch.isfinite(total)
    assert not torch.equal(model.scale.detach(), before)
    ce_calls = [call for call in model.calls if call[0] == "ce"]
    auxiliary_calls = [call for call in model.calls if call[0] == "aux"]
    assert ce_calls == [("ce", False, False)]
    if expected_loss is None:
        assert result is None
        assert auxiliary_calls == []
        return
    assert result is not None
    assert expected_loss in result.losses
    assert torch.isfinite(result.losses[expected_loss])
    assert result.effects.forward_count == 1 + 2
    assert len(auxiliary_calls) == 1 + 2
    assert len(model.calls) == 1 + (1 + 2)
    assert all(request_hidden is needs_hidden for _, _, request_hidden in auxiliary_calls)
    if reference_targets is not None:
        assert reference_targets.contributions.grad is None
        if reference_targets.hidden_residuals is not None:
            assert reference_targets.hidden_residuals.grad is None


def _load_large_smoke_module():
    spec = importlib.util.spec_from_file_location("stage2_large_model_smoke", LARGE_SMOKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.REPOSITORY_ROOT == REPOSITORY_ROOT
    return module


def test_large_model_harness_exposes_the_documented_cli():
    """A stale placeholder script would make the final B200 validation impossible to invoke."""
    module = _load_large_smoke_module()

    args = module.parse_args(
        [
            "--case",
            "baseline",
            "--model",
            "/models/qwen",
            "--stage1-checkpoint",
            "/checkpoints/stage1",
            "--data",
            "/data/train.jsonl",
            "--image-root",
            "/data",
            "--sample-index",
            "7",
            "--attempt-id",
            "attempt-a",
            "--ew-config",
            "/configs/baseline.json",
            "--deepspeed-config",
            "/configs/zero3.json",
            "--reference-artifact-dir",
            "/artifacts/reference",
            "--reference-dataset-sha256",
            "a" * 64,
            "--reference-checkpoint-id",
            "stage1-approved",
            "--reference-checkpoint-witness",
            "sealed-stage1",
            "--reference-tokenizer-id",
            "qwen-tokenizer",
            "--reference-model-config-id",
            "qwen-stage2",
            "--output-dir",
            "/results",
        ]
    )

    assert args.case == "baseline"
    assert args.sample_index == 7
    assert args.stage1_checkpoint == "/checkpoints/stage1"


def test_named_case_rejects_config_with_a_different_signature():
    """A path override must not relabel reference-free EW as the baseline."""
    module = _load_large_smoke_module()
    args = SimpleNamespace(
        ew_config=str(CONFIG_ROOT / "ew_reference_free.json"),
        reference_artifact_dir=None,
        stage1_checkpoint="/checkpoint",
    )

    with pytest.raises(ValueError, match="case.*signature"):
        module._resolve_stage2_config(args, "baseline")


def test_all_single_gpu_rejects_one_config_override():
    """One override cannot validly describe every named case in the matrix."""
    module = _load_large_smoke_module()

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--case", "all-single-gpu",
                "--model", "/model",
                "--stage1-checkpoint", "/checkpoint",
                "--data", "/data.jsonl",
                "--image-root", "/images",
                "--sample-index", "0",
                "--ew-config", "/config.json",
                "--output-dir", "/results",
            ]
        )


@pytest.mark.parametrize(
    "case,zero_stage",
    [("online-frozen-zero1", 2), ("online-frozen-zero2", 1)],
)
def test_online_named_case_rejects_cross_paired_zero_stage(case, zero_stage):
    """A valid online mode is still the wrong runtime when paired to the other case."""
    module = _load_large_smoke_module()

    with pytest.raises(ValueError, match="requires DeepSpeed ZeRO"):
        module._validate_case_zero_stage(case, zero_stage)


def _valid_witness(case: str) -> dict:
    loss_key = {
        "ew-reference-free": "loss/ew_pref",
        "ew-precomputed-dpo": "loss/ew_pref",
        "online-frozen-zero1": "loss/ew_pref",
        "online-frozen-zero2": "loss/ew_pref",
        "relation-kd": "loss/relation",
        "hidden-residual-kd": "loss/residual",
    }.get(case)
    losses = {"loss": 1.0, "train_loss": 1.0}
    if loss_key is not None:
        losses[loss_key] = 0.5
    online = case.startswith("online-frozen")
    active = loss_key is not None
    return {
        "case": case,
        "zero_stage": 1 if case == "online-frozen-zero1" else (
            2 if case == "online-frozen-zero2" else 3
        ),
        "losses": losses,
        "edge_count": 2 if active else 0,
        "eligible_window_count": 1 if active else 0,
        "global_max_edge_slots": 2 if active else 0,
        "policy_forwards": {
            "total": 4 if active else 1,
            "blocked": 2 if active else 0,
            "hidden": 3 if case == "hidden-residual-kd" else 0,
        },
        "reference_forwards": {
            "total": 3 if online else 0,
            "blocked": 2 if online else 0,
            "hidden": 0,
        },
        "policy_gradient": {"seen": True, "finite": True, "nonzero": True},
        "policy_parameter_update": {"changed": True, "finite": True},
        "reference_has_gradient": False,
        "disabled_auxiliary_forward_count": 0 if case == "baseline" else None,
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["losses"].pop("loss/ew_pref"), "loss/ew_pref"),
        (lambda value: value["losses"].__setitem__("loss", float("nan")), "finite"),
        (lambda value: value.__setitem__("edge_count", 0), "edge"),
        (
            lambda value: value["policy_gradient"].__setitem__("nonzero", False),
            "gradient|parameter update",
        ),
        (lambda value: value["policy_forwards"].__setitem__("total", 3), "policy forward"),
    ],
)
def test_named_witness_mutations_are_rejected(mutation, match):
    """A structurally present but unauthenticated smoke record must never publish."""
    module = _load_large_smoke_module()
    witness = _valid_witness("ew-reference-free")
    mutation(witness)

    with pytest.raises(ValueError, match=match):
        module._validate_case_witness(witness)


def test_relation_requires_a_window_with_two_candidate_edges():
    module = _load_large_smoke_module()
    witness = _valid_witness("relation-kd")
    witness["eligible_window_count"] = 0

    with pytest.raises(ValueError, match="eligible window"):
        module._validate_case_witness(witness)


def test_online_witness_requires_exact_reference_forward_ledger():
    module = _load_large_smoke_module()
    witness = _valid_witness("online-frozen-zero1")
    witness["reference_forwards"]["total"] = 2

    with pytest.raises(ValueError, match="reference forward"):
        module._validate_case_witness(witness)


def test_zero2_parameter_update_is_an_explicit_gradient_fallback():
    """A ZeRO-cleared raw grad is acceptable only with a finite parameter update witness."""
    module = _load_large_smoke_module()
    witness = _valid_witness("online-frozen-zero2")
    witness["policy_gradient"] = {"seen": False, "finite": True, "nonzero": False}

    module._validate_case_witness(witness)


def test_parameter_update_witness_checks_all_trainable_smoke_parameters():
    """LoRA A can stay unchanged while a later LoRA B parameter updates."""
    module = _load_large_smoke_module()
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.tensor(1.0))
    model.second = torch.nn.Parameter(torch.tensor(2.0))
    finish = module._parameter_update_tracker(model)

    with torch.no_grad():
        model.second.add_(0.25)

    assert finish() == {"changed": True, "finite": True, "changed_parameter_count": 1}


def test_stage1_witness_requires_exact_policy_teacher_and_ema_counts():
    module = _load_large_smoke_module()
    witness = {
        "case": "stage1-baseline",
        "losses": {"loss": 1.0},
        "edge_count": 0,
        "eligible_window_count": 0,
        "global_max_edge_slots": 0,
        "latent_evidence_count": 2,
        "policy_forwards": {"total": 1, "blocked": 0, "hidden": 0},
        "teacher_forwards": {"total": 1, "blocked": 0, "hidden": 1},
        "policy_gradient": {"seen": True, "finite": True, "nonzero": True},
        "policy_parameter_update": {"changed": True, "finite": True},
        "teacher_has_gradient": False,
        "ema_update_count": 1,
        "disabled_auxiliary_forward_count": 0,
    }

    with pytest.raises(ValueError, match="teacher forward"):
        module._validate_case_witness(witness)

    witness["teacher_forwards"] = {"total": 3, "blocked": 0, "hidden": 3}
    module._validate_case_witness(witness)


def test_invalid_witness_is_not_published(tmp_path):
    module = _load_large_smoke_module()
    attempt = tmp_path / "attempt"
    module._claim_attempt_destination(
        attempt, "attempt-a", ("ew-reference-free",), world_size=1
    )
    witness = _valid_witness("ew-reference-free")
    witness["policy_gradient"]["seen"] = False
    witness["policy_parameter_update"]["changed"] = False
    witness.update(rank=0, world_size=1, attempt_id="attempt-a")

    with pytest.raises(ValueError, match="gradient|parameter update"):
        module._finalize_attempt(
            attempt,
            [witness],
            ("ew-reference-free",),
            rank=0,
            world_size=1,
            attempt_id="attempt-a",
        )
    assert not (attempt / "completion.json").exists()
    assert not list(attempt.glob("*rank-*.json"))


def _identity(**changes) -> ArtifactIdentity:
    base = ArtifactIdentity(
        format_version=1,
        dataset_sha256="a" * 64,
        checkpoint_id="stage1-approved",
        checkpoint_witness="sealed-stage1",
        tokenizer_id="qwen-tokenizer",
        tokenizer_special_tokens=(0, 90, 91, 92, 99),
        parser_version="stage2-marker-v1",
        contribution_reduction="mean",
        intervention="window_attention_block",
        attention_backend="sdpa",
        max_evidence_edges_per_window=0,
        model_config_id="qwen-stage2",
    )
    return replace(base, **changes)


def test_precomputed_provider_rejects_independently_reconstructed_identity_mismatch(
    tmp_path,
):
    """Using the manifest itself as the expected identity makes mismatches tautological."""
    module = _load_large_smoke_module()
    write_reference_shard(
        tmp_path,
        _identity(),
        {
            0: ReferenceSampleTargets(
                torch.tensor([[0.2]]),
                torch.tensor([[True]]),
            )
        },
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        module._build_precomputed_provider(
            tmp_path,
            _identity(checkpoint_witness="different-sealed-stage1"),
        )


def test_precomputed_expected_identity_comes_from_explicit_consumer_witnesses():
    module = _load_large_smoke_module()
    config = Stage2EvidenceWindowConfig.from_json(
        CONFIG_ROOT / "ew_precomputed_dpo.json"
    )
    args = SimpleNamespace(
        reference_dataset_sha256="a" * 64,
        reference_checkpoint_id="stage1-approved",
        reference_checkpoint_witness="sealed-stage1",
        reference_tokenizer_id="qwen-tokenizer",
        reference_parser_version="stage2-marker-v1",
        reference_model_config_id="qwen-stage2",
    )

    assert module._build_expected_reference_identity(
        args, config, (0, 90, 91, 92, 99)
    ) == _identity()


def _rank_witness(case: str, rank: int, world_size: int) -> dict:
    witness = _valid_witness(case)
    witness.update(rank=rank, world_size=world_size, attempt_id="attempt-a")
    return witness


def test_remote_validation_failure_prevents_rank0_completion(tmp_path):
    """Publishing rank 0 before rank 1 validates would create a false success."""
    module = _load_large_smoke_module()
    attempt = tmp_path / "attempt"
    module._claim_attempt_destination(
        attempt, "attempt-a", ("ew-reference-free",), world_size=2
    )

    def simulated_collective(name, operation):
        result = operation()
        if name == "witness validation":
            raise RuntimeError("rank 1: invalid named loss")
        return result

    with pytest.raises(RuntimeError, match="rank 1"):
        module._finalize_attempt(
            attempt,
            [_rank_witness("ew-reference-free", 0, 2)],
            ("ew-reference-free",),
            rank=0,
            world_size=2,
            attempt_id="attempt-a",
            collective_phase=simulated_collective,
        )

    assert not (attempt / "completion.json").exists()
    assert not list(attempt.glob("*rank-*.json"))


@pytest.mark.parametrize(
    "stale_name",
    ["completion.json", "attempt-a-baseline-rank-00000.json"],
)
def test_stale_attempt_witness_rejects_retry_before_model_work(tmp_path, stale_name):
    module = _load_large_smoke_module()
    (tmp_path / stale_name).write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="attempt|exist|claim"):
        module._claim_attempt_destination(
            tmp_path, "attempt-a", ("baseline",), world_size=1
        )


def test_main_rejects_nonempty_attempt_before_model_load(tmp_path, monkeypatch):
    module = _load_large_smoke_module()
    argv = _preflight_cli(tmp_path, tmp_path / "unused", case="baseline")
    attempt = tmp_path / "attempt"
    module._claim_attempt_destination(
        attempt, "winner", ("baseline",), world_size=1
    )
    loads = 0

    def count_load(*_args, **_kwargs):
        nonlocal loads
        loads += 1
        raise AssertionError("model construction must not begin")

    monkeypatch.setattr(module, "_load_processor_and_model", count_load)
    with pytest.raises(FileExistsError, match="attempt|exist|claim"):
        module.main(argv)
    assert loads == 0
    claim = json.loads((attempt / "attempt.json").read_text(encoding="utf-8"))
    assert claim["attempt_id"] == "winner"


def test_completion_manifest_is_last_after_every_rank_result(tmp_path):
    module = _load_large_smoke_module()
    cases = ("ew-reference-free",)
    attempt = tmp_path / "attempt"
    module._claim_attempt_destination(
        attempt, "attempt-a", cases, world_size=2
    )

    module._write_result(
        attempt, _rank_witness("ew-reference-free", 1, 2)
    )
    assert not (attempt / "completion.json").exists()

    manifest_path = module._finalize_attempt(
        attempt,
        [_rank_witness("ew-reference-free", 0, 2)],
        cases,
        rank=0,
        world_size=2,
        attempt_id="attempt-a",
    )
    assert manifest_path == attempt / "completion.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "format_version": 1,
        "status": "complete",
        "attempt_id": "attempt-a",
        "cases": ["ew-reference-free"],
        "world_size": 2,
        "result_files": [
            {
                "case": "ew-reference-free",
                "rank": 0,
                "attempt_id": "attempt-a",
                "filename": "attempt-a-ew-reference-free-rank-00000.json",
            },
            {
                "case": "ew-reference-free",
                "rank": 1,
                "attempt_id": "attempt-a",
                "filename": "attempt-a-ew-reference-free-rank-00001.json",
            },
        ],
    }


def test_remote_precommit_failure_leaves_no_completion(tmp_path):
    module = _load_large_smoke_module()
    cases = ("ew-reference-free",)
    attempt = tmp_path / "attempt"
    module._claim_attempt_destination(
        attempt, "attempt-a", cases, world_size=1
    )

    def simulated_collective(name, operation):
        result = operation()
        if name == "completion precommit":
            raise RuntimeError("rank 1: precommit failed")
        return result

    with pytest.raises(RuntimeError, match="precommit failed"):
        module._finalize_attempt(
            attempt,
            [_rank_witness("ew-reference-free", 0, 1)],
            cases,
            rank=0,
            world_size=1,
            attempt_id="attempt-a",
            collective_phase=simulated_collective,
        )
    assert not (attempt / "completion.json").exists()


def test_completion_is_terminal_with_no_postpublish_collective(tmp_path):
    module = _load_large_smoke_module()
    cases = ("ew-reference-free",)
    attempt = tmp_path / "attempt"
    module._claim_attempt_destination(
        attempt, "attempt-a", cases, world_size=1
    )
    phases = []

    def simulated_collective(name, operation):
        phases.append(name)
        return operation()

    module._finalize_attempt(
        attempt,
        [_rank_witness("ew-reference-free", 0, 1)],
        cases,
        rank=0,
        world_size=1,
        attempt_id="attempt-a",
        collective_phase=simulated_collective,
    )

    assert phases == [
        "witness validation",
        "rank result publication",
        "completion precommit",
    ]
    assert (attempt / "completion.json").is_file()


def test_rank_and_completion_publication_never_overwrite(tmp_path):
    module = _load_large_smoke_module()
    cases = ("ew-reference-free",)
    attempt = tmp_path / "attempt"
    module._claim_attempt_destination(
        attempt, "attempt-a", cases, world_size=1
    )
    witness = _rank_witness("ew-reference-free", 0, 1)
    rank_path = module._write_result(attempt, witness)
    original_rank = rank_path.read_bytes()

    changed = dict(witness, losses={"loss": 99.0, "loss/ew_pref": 99.0})
    with pytest.raises(FileExistsError):
        module._write_result(attempt, changed)
    assert rank_path.read_bytes() == original_rank

    completion_path = module._write_completion_manifest(
        attempt, cases, 1, "attempt-a"
    )
    original_completion = completion_path.read_bytes()
    with pytest.raises(FileExistsError):
        module._write_completion_manifest(attempt, cases, 1, "attempt-a")
    assert completion_path.read_bytes() == original_completion


def test_precommit_rejects_rank_file_from_another_attempt(tmp_path):
    module = _load_large_smoke_module()
    cases = ("ew-reference-free",)
    attempt = tmp_path / "attempt"
    module._claim_attempt_destination(
        attempt, "attempt-a", cases, world_size=1
    )
    wrong = _rank_witness("ew-reference-free", 0, 1)
    wrong["attempt_id"] = "attempt-b"
    expected = attempt / "attempt-a-ew-reference-free-rank-00000.json"
    expected.write_text(json.dumps(wrong), encoding="utf-8")

    with pytest.raises(ValueError, match="attempt"):
        module._validate_completion_precommit(
            attempt, cases, 1, "attempt-a"
        )
    assert not (attempt / "completion.json").exists()


def _preflight_cli(tmp_path, artifact_dir, *, case="ew-precomputed-dpo"):
    checkpoint = tmp_path / "stage1-checkpoint"
    checkpoint.mkdir(exist_ok=True)
    image_root = tmp_path / "images"
    image_root.mkdir(exist_ok=True)
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    return [
        "--case", case,
        "--model", str(checkpoint),
        "--stage1-checkpoint", str(checkpoint),
        "--data", str(data),
        "--image-root", str(image_root),
        "--sample-index", "0",
        "--attempt-id", "attempt-a",
        "--reference-artifact-dir", str(artifact_dir),
        "--reference-dataset-sha256", "a" * 64,
        "--reference-checkpoint-id", "stage1-approved",
        "--reference-tokenizer-id", "qwen-tokenizer",
        "--reference-model-config-id", "qwen-stage2",
        "--output-dir", str(tmp_path / "attempt"),
    ]


def test_all_case_preflight_rejects_missing_identity_before_any_model_load(
    tmp_path, monkeypatch
):
    """All-case execution must not run Stage 1 before checking later artifacts."""
    module = _load_large_smoke_module()
    artifact = tmp_path / "artifact"
    write_reference_shard(
        artifact,
        _identity(),
        {0: ReferenceSampleTargets(torch.tensor([[0.2]]), torch.tensor([[True]]))},
    )
    loads = 0

    def count_load(*_args, **_kwargs):
        nonlocal loads
        loads += 1
        raise AssertionError("model construction must not begin")

    monkeypatch.setattr(module, "_load_processor_and_model", count_load)
    with pytest.raises(ValueError, match="checkpoint_witness"):
        module.main(_preflight_cli(tmp_path, artifact, case="all-single-gpu"))
    assert loads == 0


def test_preflight_rejects_manifest_identity_mismatch_before_model_load(
    tmp_path, monkeypatch
):
    module = _load_large_smoke_module()
    artifact = tmp_path / "artifact"
    write_reference_shard(
        artifact,
        _identity(),
        {0: ReferenceSampleTargets(torch.tensor([[0.2]]), torch.tensor([[True]]))},
    )
    argv = _preflight_cli(tmp_path, artifact)
    argv.extend(["--reference-checkpoint-witness", "wrong-stage1"])
    loads = 0

    def count_load(*_args, **_kwargs):
        nonlocal loads
        loads += 1
        raise AssertionError("model construction must not begin")

    monkeypatch.setattr(module, "_load_processor_and_model", count_load)
    with pytest.raises(ValueError, match="identity mismatch"):
        module.main(argv)
    assert loads == 0


def test_preflight_rejects_malformed_reference_manifest_before_model_load(
    tmp_path, monkeypatch
):
    module = _load_large_smoke_module()
    artifact = tmp_path / "artifact"
    write_reference_shard(
        artifact,
        _identity(),
        {0: ReferenceSampleTargets(torch.tensor([[0.2]]), torch.tensor([[True]]))},
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "wrong-format"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    argv = _preflight_cli(tmp_path, artifact)
    argv.extend(["--reference-checkpoint-witness", "sealed-stage1"])
    loads = 0

    def count_load(*_args, **_kwargs):
        nonlocal loads
        loads += 1
        raise AssertionError("model construction must not begin")

    monkeypatch.setattr(module, "_load_processor_and_model", count_load)
    with pytest.raises(ValueError, match="manifest format"):
        module.main(argv)
    assert loads == 0

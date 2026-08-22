import importlib.util
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch

from evidence_window.artifacts import (
    ArtifactIdentity,
    ReferenceBatchTargets,
    ReferenceSampleTargets,
    publish_reference_manifest,
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


_DISTRIBUTED_ENVIRONMENT = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "LOCAL_WORLD_SIZE",
)


def _set_distributed_environment(monkeypatch, **overrides):
    values = {
        "RANK": "0",
        "WORLD_SIZE": "2",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29501",
    }
    values.update(overrides)
    for name in _DISTRIBUTED_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        if value is not None:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "missing_name",
    ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"],
)
def test_distributed_environment_requires_every_rendezvous_field(
    monkeypatch, missing_name
):
    """Dropping any launcher field must fail rather than invent rank state."""
    module = _load_large_smoke_module()
    _set_distributed_environment(monkeypatch, **{missing_name: None})
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("rendezvous must not start for an incomplete environment")
        ),
    )

    with pytest.raises(RuntimeError, match="incomplete|requires"):
        module._initialize_distributed_from_environment()


def test_one_distributed_environment_field_cannot_enable_launcher_defaults(monkeypatch):
    """A stray launcher field must not silently become a local rank-zero run."""
    module = _load_large_smoke_module()
    for name in _DISTRIBUTED_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("rendezvous must not start for an incomplete environment")
        ),
    )

    with pytest.raises(RuntimeError, match="incomplete|requires"):
        module._initialize_distributed_from_environment()


@pytest.mark.parametrize(
    "overrides",
    [
        {"RANK": "rank-zero"},
        {"WORLD_SIZE": "0"},
        {"RANK": "2"},
        {"LOCAL_RANK": "-1"},
        {"LOCAL_RANK": "2"},
        {"MASTER_ADDR": ""},
        {"MASTER_PORT": "not-a-port"},
        {"MASTER_PORT": "0"},
        {"MASTER_PORT": "65536"},
        {"LOCAL_RANK": "2", "LOCAL_WORLD_SIZE": "2"},
        {"LOCAL_WORLD_SIZE": "3"},
    ],
)
def test_distributed_environment_rejects_malformed_or_out_of_range_values(
    monkeypatch, overrides
):
    """Invalid rank, local topology, address, and port values fail before rendezvous."""
    module = _load_large_smoke_module()
    _set_distributed_environment(monkeypatch, **overrides)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("rendezvous must not start for invalid launcher values")
        ),
    )

    with pytest.raises(RuntimeError, match="invalid|range|address|port|local"):
        module._initialize_distributed_from_environment()


def test_single_process_without_launcher_environment_remains_supported(monkeypatch):
    """An ordinary local invocation does not need rendezvous variables."""
    module = _load_large_smoke_module()
    for name in _DISTRIBUTED_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    assert module._initialize_distributed_from_environment() == (0, 1)


def test_preinitialized_single_process_without_launcher_env_selects_cuda(monkeypatch):
    """Even a rank-one group reuse must select and validate its CUDA target first."""
    module = _load_large_smoke_module()
    for name in _DISTRIBUTED_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")

    assert module._initialize_distributed_from_environment() == (0, 1)
    assert selected == [0]


def test_local_cuda_rank_is_validated_before_device_selection(monkeypatch):
    """An unavailable local CUDA ordinal must fail before set_device or rendezvous."""
    module = _load_large_smoke_module()
    _set_distributed_environment(monkeypatch, LOCAL_RANK="1")
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)

    with pytest.raises(RuntimeError, match="LOCAL_RANK|device"):
        module._initialize_distributed_from_environment()
    assert selected == []


def test_matching_preinitialized_cuda_group_still_selects_local_device(monkeypatch):
    """Early reuse must not skip the local CUDA selection required by the model."""
    module = _load_large_smoke_module()
    _set_distributed_environment(monkeypatch, RANK="1", LOCAL_RANK="1")
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")

    assert module._initialize_distributed_from_environment() == (1, 2)
    assert selected == [1]


def test_preinitialized_group_rejects_backend_for_wrong_device_target(monkeypatch):
    """A CPU run cannot silently reuse a NCCL process group."""
    module = _load_large_smoke_module()
    _set_distributed_environment(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")

    with pytest.raises(RuntimeError, match="backend"):
        module._initialize_distributed_from_environment()


def test_preinitialized_group_rejects_launcher_rank_mismatch(monkeypatch):
    """A process group from a different launch must fail before model work."""
    module = _load_large_smoke_module()
    _set_distributed_environment(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")

    with pytest.raises(RuntimeError, match="rank/world"):
        module._initialize_distributed_from_environment()


def _reserve_local_ports(count: int) -> list[tuple[socket.socket, int]]:
    reservations = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            reservations.append((listener, int(listener.getsockname()[1])))
    except BaseException:
        for listener, _port in reservations:
            listener.close()
        raise
    assert len({port for _listener, port in reservations}) == count
    return reservations


def test_concurrent_rendezvous_ports_are_distinct_while_reserved():
    """Two launch groups must never be assigned the same released port candidate."""
    reservations = _reserve_local_ports(2)
    try:
        ports = [port for _listener, port in reservations]
        assert len(set(ports)) == 2
        for listener, port in reservations:
            assert listener.getsockname()[1] == port
    finally:
        for listener, _port in reservations:
            listener.close()


def _distributed_attempt_worker(
    rank,
    world_size,
    port,
    launch_name,
    attempt_id,
    output_dir,
    checkpoint,
    data,
    image_root,
    result_path,
    asymmetric_failure,
    start_gate,
):
    """Exercise real env:// Gloo coordination from an uninitialized child."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    module = _load_large_smoke_module()
    phases = []
    loads = 0
    error = None
    initialized_before = torch.distributed.is_initialized()
    try:
        start_gate.wait(timeout=20)
        original_collective = module._run_collective_phase

        def recording_collective(name, operation):
            phases.append(name)
            return original_collective(name, operation)

        def run_counting_stage2(_args, case, _case_dir, _config):
            nonlocal loads
            loads += 1
            witness = _valid_witness(case)
            witness.update(rank=rank, world_size=world_size)
            return witness

        module._run_collective_phase = recording_collective
        module._run_stage2 = run_counting_stage2
        module.main(
            [
                "--case", "baseline",
                "--model", str(checkpoint),
                "--stage1-checkpoint", str(checkpoint),
                "--data",
                str(
                    Path(data).with_name("missing.jsonl")
                    if asymmetric_failure and rank == 1
                    else data
                ),
                "--image-root", str(image_root),
                "--sample-index", "0",
                "--attempt-id", attempt_id,
                "--output-dir", str(output_dir),
            ]
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        initialized_after = torch.distributed.is_initialized()
        if initialized_after:
            torch.distributed.destroy_process_group()
        Path(result_path).write_text(
            json.dumps(
                {
                    "launch": launch_name,
                    "rank": rank,
                    "initialized_before": initialized_before,
                    "initialized_after": initialized_after,
                    "loads": loads,
                    "phases": phases,
                    "error": error,
                }
            ),
            encoding="utf-8",
        )


def _run_spawned_attempts(tmp_path, launches):
    context = torch.multiprocessing.get_context("spawn")
    world_size = 2
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    image_root = tmp_path / "images"
    image_root.mkdir(exist_ok=True)
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    start_gate = context.Barrier(world_size * len(launches))
    processes = []
    result_paths = []
    reservations = _reserve_local_ports(len(launches))
    try:
        for (
            launch_name,
            attempt_id,
            output_dir,
            asymmetric_failure,
        ), (_listener, port) in zip(launches, reservations, strict=True):
            for rank in range(world_size):
                result_path = tmp_path / f"worker-{launch_name}-{rank}.json"
                result_paths.append(result_path)
                process = context.Process(
                    target=_distributed_attempt_worker,
                    args=(
                        rank,
                        world_size,
                        port,
                        launch_name,
                        attempt_id,
                        output_dir,
                        checkpoint,
                        data,
                        image_root,
                        result_path,
                        asymmetric_failure,
                        start_gate,
                    ),
                )
                processes.append(process)
    finally:
        for listener, _port in reservations:
            listener.close()
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    return [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]


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


def test_residual_preflight_ignores_empty_nonowner_shard(tmp_path):
    """Selected-sample distributed artifacts may have empty shards without residual rows."""
    module = _load_large_smoke_module()
    identity = _identity()
    owner = write_reference_shard(
        tmp_path,
        identity,
        {
            7: ReferenceSampleTargets(
                torch.tensor([[0.2]]),
                torch.tensor([[True]]),
                torch.tensor([[[0.1, -0.1]]]),
            )
        },
        rank=1,
        publish_manifest=False,
    )
    empty = write_reference_shard(
        tmp_path, identity, {}, rank=0, publish_manifest=False
    )
    publish_reference_manifest(tmp_path, identity, [empty, owner])
    args = SimpleNamespace(
        reference_artifact_dir=str(tmp_path),
        reference_dataset_sha256="a" * 64,
        reference_checkpoint_id="stage1-approved",
        reference_checkpoint_witness="sealed-stage1",
        reference_tokenizer_id="qwen-tokenizer",
        reference_parser_version="stage2-marker-v1",
        reference_model_config_id="qwen-stage2",
    )
    config = Stage2EvidenceWindowConfig.from_json(
        CONFIG_ROOT / "hidden_residual_kd.json"
    )

    module._validate_precomputed_preflight(args, config)


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


def test_two_rank_gloo_initializes_before_claim_and_preflight(tmp_path):
    """Missing early init makes both ranks run claim/preflight as unrelated locals."""
    output_dir = tmp_path / "attempt"
    outcomes = _run_spawned_attempts(
        tmp_path,
        [("one", "attempt-a", output_dir, False)],
    )

    assert [outcome["rank"] for outcome in outcomes] == [0, 1]
    assert all(outcome["initialized_before"] is False for outcome in outcomes)
    assert all(outcome["initialized_after"] is True for outcome in outcomes)
    assert all(outcome["loads"] == 1 for outcome in outcomes)
    assert all(outcome["error"] is None for outcome in outcomes)
    assert all(
        outcome["phases"]
        == [
            "attempt ownership claim",
            "attempt preflight",
            "witness validation",
            "rank result publication",
            "completion precommit",
        ]
        for outcome in outcomes
    )
    assert json.loads((output_dir / "attempt.json").read_text(encoding="utf-8"))[
        "world_size"
    ] == 2
    assert (output_dir / "completion.json").is_file()


def test_two_rank_asymmetric_preflight_fails_collectively_before_load(tmp_path):
    """One bad rank must stop its peer before either begins model construction."""
    outcomes = _run_spawned_attempts(
        tmp_path,
        [("one", "attempt-a", tmp_path / "attempt", True)],
    )

    assert all(outcome["loads"] == 0 for outcome in outcomes)
    assert all(
        "distributed attempt preflight failed" in outcome["error"]
        for outcome in outcomes
    )
    assert all(
        outcome["phases"] == ["attempt ownership claim", "attempt preflight"]
        for outcome in outcomes
    )


def test_two_concurrent_two_rank_launches_have_one_all_rank_winner(tmp_path):
    """A losing rank zero must notify its own rank one before any model loads."""
    output_dir = tmp_path / "attempt"
    outcomes = _run_spawned_attempts(
        tmp_path,
        [
            ("alpha", "attempt-alpha", output_dir, False),
            ("beta", "attempt-beta", output_dir, False),
        ],
    )
    by_launch = {
        name: sorted(
            (outcome for outcome in outcomes if outcome["launch"] == name),
            key=lambda outcome: outcome["rank"],
        )
        for name in ("alpha", "beta")
    }
    winners = [
        name
        for name, launch in by_launch.items()
        if all(outcome["loads"] == 1 and outcome["error"] is None for outcome in launch)
    ]
    losers = [name for name in by_launch if name not in winners]

    assert len(winners) == 1
    assert len(losers) == 1
    assert all(outcome["loads"] == 0 for outcome in by_launch[losers[0]])
    assert all(
        "distributed attempt ownership claim failed" in outcome["error"]
        for outcome in by_launch[losers[0]]
    )
    assert all(
        outcome["phases"] == ["attempt ownership claim"]
        for outcome in by_launch[losers[0]]
    )
    claim = json.loads((output_dir / "attempt.json").read_text(encoding="utf-8"))
    assert claim["attempt_id"] == f"attempt-{winners[0]}"


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


def test_post_link_directory_fsync_failure_keeps_publication_committed(
    tmp_path, monkeypatch
):
    """A visible completion must not be reported failed by best-effort durability work."""
    module = _load_large_smoke_module()
    destination = tmp_path / "completion.json"

    def fail_fsync(_directory):
        raise OSError("injected post-link directory fsync failure")

    monkeypatch.setattr(module, "_fsync_directory", fail_fsync)
    assert (
        module._atomic_publish_json_no_overwrite(
            destination, {"status": "complete"}
        )
        == destination
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "complete"}


def test_post_link_temp_cleanup_failure_keeps_publication_committed(
    tmp_path, monkeypatch
):
    """Temporary cleanup cannot turn an atomic visible commit into a false failure."""
    module = _load_large_smoke_module()
    destination = tmp_path / "completion.json"
    original_unlink = Path.unlink

    def fail_temp_cleanup(path, *args, **kwargs):
        if path.name.startswith(".completion.json."):
            raise OSError("injected post-link cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temp_cleanup)
    assert (
        module._atomic_publish_json_no_overwrite(
            destination, {"status": "complete"}
        )
        == destination
    )
    assert destination.is_file()


def test_link_failure_leaves_no_publication_destination(tmp_path, monkeypatch):
    """Before the hard-link commit point, failure must leave success invisible."""
    module = _load_large_smoke_module()
    destination = tmp_path / "completion.json"

    def fail_link(_source, _destination):
        raise OSError("injected link failure")

    monkeypatch.setattr(module.os, "link", fail_link)
    with pytest.raises(OSError, match="injected link failure"):
        module._atomic_publish_json_no_overwrite(destination, {"status": "complete"})
    assert not destination.exists()


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

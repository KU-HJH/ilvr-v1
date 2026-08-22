import importlib.util
from pathlib import Path

import pytest
import torch

from evidence_window.artifacts import ReferenceBatchTargets
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
    parameter = torch.nn.Parameter(torch.tensor(0.7))
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    model_calls = ["ce"]
    calls: list[tuple[bool, bool]] = []

    def policy_forward(block_mask, request_hidden):
        calls.append((block_mask is not None, request_hidden))
        edge_ordinal = sum(blocked for blocked, _ in calls) - 1
        if block_mask is None:
            scale = parameter
        else:
            scale = parameter * (0.6 if edge_ordinal == 0 else 0.8)
        token_logps = scale * torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        hidden_state = None
        if request_hidden:
            hidden_state = scale * torch.tensor(
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

    ce = (parameter - 0.25).square()
    reference_targets = _reference(hidden=needs_hidden) if needs_reference else None
    if config.is_disabled:
        total = ce
        result = None
    else:
        result = Stage2EvidenceWindowEngine(config).compute(
            _layout(),
            policy_forward,
            progress=1.0,
            reference_targets=reference_targets,
        )
        total = ce + result.auxiliary_loss

    before = parameter.detach().clone()
    optimizer.zero_grad()
    total.backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad)
    optimizer.step()

    assert torch.isfinite(total)
    assert not torch.equal(parameter.detach(), before)
    if expected_loss is None:
        assert result is None
        assert calls == []
        assert len(model_calls) == 1
        return
    assert result is not None
    assert expected_loss in result.losses
    assert torch.isfinite(result.losses[expected_loss])
    assert result.effects.forward_count == 1 + 2
    assert len(calls) == 1 + 2
    assert len(model_calls) + len(calls) == 1 + (1 + 2)
    assert all(request_hidden is needs_hidden for _, request_hidden in calls)
    if reference_targets is not None:
        assert reference_targets.contributions.grad is None
        if reference_targets.hidden_residuals is not None:
            assert reference_targets.hidden_residuals.grad is None


def test_large_model_harness_exposes_the_documented_cli():
    """A stale placeholder script would make the final B200 validation impossible to invoke."""
    spec = importlib.util.spec_from_file_location("stage2_large_model_smoke", LARGE_SMOKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.REPOSITORY_ROOT == REPOSITORY_ROOT

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
            "--ew-config",
            "/configs/baseline.json",
            "--deepspeed-config",
            "/configs/zero3.json",
            "--reference-artifact-dir",
            "/artifacts/reference",
            "--output-dir",
            "/results",
        ]
    )

    assert args.case == "baseline"
    assert args.sample_index == 7
    assert args.stage1_checkpoint == "/checkpoints/stage1"

"""Selective checkpointing of Stage 2 latent-prefix decoder calls (ported from b92bdb0)."""

from copy import deepcopy
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentRolloutEngine, LatentSpan
from src.multinode_runtime import validated_stage2_environment
from stage2.checkpointing import configure_stage2_metadata
from stage2.rng import capture_rng_state, replay_rng_state
from utils_deepseed import parse_args

from .packed_m_helpers import make_packed_inputs, make_packed_model
from .test_packed_m_runtime import _run_runtime

REPO_ROOT = Path(__file__).resolve().parents[2]


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 6)
        self.dropout = nn.Dropout(0.2)
        self.calls = 0

    def forward(self, *, inputs_embeds, **_kwargs):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.dropout(torch.tanh(self.proj(inputs_embeds))))


def _rollout_case(*, checkpointing):
    decoder = TinyDecoder().train()
    inputs = torch.randn(1, 10, 6, requires_grad=True)
    result = LatentRolloutEngine().rollout(
        language_model=decoder,
        base_inputs_embeds=inputs,
        attention_mask=torch.ones(1, 10, dtype=torch.long),
        full_position_ids=torch.arange(10).view(1, 1, 10).expand(3, -1, -1),
        latent_spans=(LatentSpan(1, (2, 3), 4), LatentSpan(6, (7, 8), 9)),
        retain_latent_grads=True,
        latent_activation_checkpointing=checkpointing,
    )
    loss = result.final_inputs_embeds.square().mean()
    loss.backward()
    return SimpleNamespace(
        result=result, loss=loss.detach(), input_grad=inputs.grad.detach().clone(),
        parameter_grads=tuple(p.grad.detach().clone() for p in decoder.parameters()),
        calls=decoder.calls,
    )


def test_checkpointed_prefix_replay_preserves_outputs_loss_and_gradients():
    """Catches replay changing dropout RNG, latent connectivity, or decoder gradients."""
    torch.manual_seed(42)
    plain = _rollout_case(checkpointing=False)
    torch.manual_seed(42)
    checked = _rollout_case(checkpointing=True)
    torch.testing.assert_close(plain.result.final_inputs_embeds, checked.result.final_inputs_embeds)
    torch.testing.assert_close(plain.loss, checked.loss)
    torch.testing.assert_close(plain.input_grad, checked.input_grad)
    for actual, expected in zip(plain.parameter_grads, checked.parameter_grads, strict=True):
        torch.testing.assert_close(actual, expected)
    assert all(state.grad is not None for state in checked.result.latent_states)
    assert checked.calls > plain.calls == 4


def test_checkpointing_is_opt_in_and_skipped_under_no_grad():
    """Catches no-grad M/reference rollouts paying replay cost or the default changing."""
    decoder = TinyDecoder().eval()
    kwargs = dict(
        language_model=decoder, base_inputs_embeds=torch.randn(1, 5, 6),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        full_position_ids=torch.arange(5).view(1, 1, 5).expand(3, -1, -1),
        latent_spans=(LatentSpan(1, (2, 3), 4),),
    )
    with torch.no_grad():
        LatentRolloutEngine().rollout(**kwargs)
        LatentRolloutEngine().rollout(**kwargs, latent_activation_checkpointing=True)
    assert decoder.calls == 4


def test_cli_metadata_launcher_and_coordinator_carry_the_flag():
    """Catches the approved flag being dropped anywhere between launch and model config."""
    assert parse_args([]).stage2_latent_activation_checkpointing is False
    assert parse_args(["--stage2_latent_activation_checkpointing"]).stage2_latent_activation_checkpointing is True
    config = SimpleNamespace()
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    configure_stage2_metadata(config, LatentTokenIds(90, 91, 92), latent_size=2,
                              supervise_latent_boundaries=True, latent_activation_checkpointing=True)
    assert config.stage2_latent_activation_checkpointing is True

    environment = validated_stage2_environment(
        {"STAGE1_CHECKPOINT": "/m", "DATA_PATH": "/d.jsonl", "IMAGE_ROOT": "/i",
         "STAGE2_LATENT_ACTIVATION_CHECKPOINTING": "true"},
        output_dir="/root/workspace/ILVR/checkpoints/x",
        log_file="/root/workspace/ILVR/logs/zero-stage-compatibility/x/train.log",
        deepspeed_config="/root/workspace/ILVR/configs/deepspeed_zero2.json",
    )
    assert environment["STAGE2_LATENT_ACTIVATION_CHECKPOINTING"] == "true"
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / "run_stage2_training.sh")],
        env={"PATH": os.environ["PATH"], "STAGE2_DRY_RUN": "true", "STAGE1_CHECKPOINT": "/m",
             "DATA_PATH": "/d.jsonl", "IMAGE_ROOT": "/i", "OUTPUT_DIR": "/o",
             "STAGE2_LATENT_ACTIVATION_CHECKPOINTING": "true"},
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--stage2_latent_activation_checkpointing" in shlex.split(completed.stdout)


def test_model_passes_config_flag_to_the_rollout(monkeypatch):
    """Catches the config flag never reaching the F rollout."""
    model = make_packed_model(tuning="full")
    engine = type(model.stage2_rollout_engine)
    seen = []
    original = engine.rollout

    def spy(self, **kwargs):
        seen.append(kwargs.get("latent_activation_checkpointing"))
        return original(self, **kwargs)

    monkeypatch.setattr(engine, "rollout", spy)
    inputs = make_packed_inputs(blocks=1)
    model(**{k: v for k, v in inputs.items() if not k.startswith("stage2_text")})
    model.config.stage2_latent_activation_checkpointing = True
    model(**{k: v for k, v in inputs.items() if not k.startswith("stage2_text")})
    assert seen == [False, True]


@pytest.mark.parametrize(("tuning", "bf16"), [("dora", False), ("dora", True), ("full", False)])
def test_objective_with_packed_m_matches_uncheckpointed_training(tuning, bf16):
    """Catches latent checkpointing changing F CE, Vision loss, M scores, or gradients."""
    torch.manual_seed(311)
    initial = make_packed_model(tuning=tuning, checkpointing=True)
    plain, checked = deepcopy(initial), deepcopy(initial)
    from peft_common import resolve_qwen2_5_vl_model
    resolve_qwen2_5_vl_model(checked).config.stage2_latent_activation_checkpointing = True
    inputs = make_packed_inputs(blocks=2)
    rng = capture_rng_state()
    tolerance = {"atol": 1e-3, "rtol": 1e-2} if bf16 else {"atol": 1e-6, "rtol": 1e-5}
    results = []
    for model in (plain, checked):
        with replay_rng_state(rng), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            results.append(_run_runtime(model, inputs, mode="packed_suffix", retain_latents=True))
    (p_result, p_scores, _, p_grads, p_latents, _), (c_result, c_scores, _, c_grads, c_latents, _) = results
    torch.testing.assert_close(c_result.ce, p_result.ce, **tolerance)
    torch.testing.assert_close(c_result.vis_sum, p_result.vis_sum, **tolerance)
    torch.testing.assert_close(torch.stack(c_scores), torch.stack(p_scores), **tolerance)
    assert c_grads.keys() == p_grads.keys()
    for name, gradient in p_grads.items():
        if gradient is None:
            assert c_grads[name] is None, name
        else:
            torch.testing.assert_close(c_grads[name], gradient, **tolerance)
    assert len(c_latents) == len(p_latents) > 0
    for plain_latent, checked_latent in zip(p_latents, c_latents, strict=True):
        torch.testing.assert_close(checked_latent.grad, plain_latent.grad, **tolerance)

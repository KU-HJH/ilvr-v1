from types import SimpleNamespace
import os
from pathlib import Path
import shlex
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentRolloutEngine,
    LatentSpan,
)
from src.multinode_runtime import validated_stage2_environment
from utils_deepseed import parse_args


REPO_ROOT = Path(__file__).resolve().parents[2]


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 6)
        self.dropout = nn.Dropout(0.2)
        self.calls = 0

    def forward(self, *, inputs_embeds, **_kwargs):
        self.calls += 1
        hidden = self.dropout(torch.tanh(self.proj(inputs_embeds)))
        return SimpleNamespace(last_hidden_state=hidden)


def run_case(*, checkpointing):
    embedding = nn.Embedding(32, 6)
    decoder = TinyDecoder().train()
    output_head = nn.Linear(6, 32)
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    inputs = embedding(token_ids)
    inputs.retain_grad()
    result = LatentRolloutEngine().rollout(
        language_model=decoder,
        base_inputs_embeds=inputs,
        attention_mask=torch.ones(1, 10, dtype=torch.long),
        full_position_ids=torch.arange(10).view(1, 1, 10).expand(3, -1, -1),
        latent_spans=(LatentSpan(1, (2, 3), 4), LatentSpan(6, (7, 8), 9)),
        retain_latent_grads=True,
        latent_activation_checkpointing=checkpointing,
    )
    logits = output_head(result.final_inputs_embeds)
    targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    return SimpleNamespace(
        result=result,
        loss=loss.detach(),
        input_grad=inputs.grad.detach().clone(),
        parameter_grads={
            "embedding": embedding.weight.grad.detach().clone(),
            "decoder": tuple(
                parameter.grad.detach().clone()
                for parameter in decoder.parameters()
            ),
            "output_head": tuple(
                parameter.grad.detach().clone()
                for parameter in output_head.parameters()
            ),
        },
        calls=decoder.calls,
    )


def test_checkpointed_prefix_replay_preserves_outputs_loss_and_gradients():
    """Catches replay changing dropout RNG, latent connectivity, or decoder gradients."""
    torch.manual_seed(42)
    plain = run_case(checkpointing=False)
    torch.manual_seed(42)
    checked = run_case(checkpointing=True)

    torch.testing.assert_close(
        plain.result.final_inputs_embeds, checked.result.final_inputs_embeds
    )
    torch.testing.assert_close(plain.loss, checked.loss)
    torch.testing.assert_close(plain.input_grad, checked.input_grad)
    torch.testing.assert_close(
        plain.parameter_grads["embedding"],
        checked.parameter_grads["embedding"],
    )
    for component in ("decoder", "output_head"):
        for actual, expected in zip(
            plain.parameter_grads[component], checked.parameter_grads[component]
        ):
            torch.testing.assert_close(actual, expected)
    assert all(state.grad is not None for state in checked.result.latent_states)
    assert checked.calls > plain.calls == 4


def test_checkpointing_is_opt_in_and_skipped_under_no_grad():
    """Catches the default or inference path unexpectedly replaying decoder prefixes."""
    decoder = TinyDecoder().eval()
    inputs = torch.randn(1, 5, 6)
    kwargs = dict(
        language_model=decoder,
        base_inputs_embeds=inputs,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        full_position_ids=torch.arange(5).view(1, 1, 5).expand(3, -1, -1),
        latent_spans=(LatentSpan(1, (2, 3), 4),),
    )
    with torch.no_grad():
        LatentRolloutEngine().rollout(**kwargs)
        LatentRolloutEngine().rollout(**kwargs, latent_activation_checkpointing=True)
    assert decoder.calls == 4


def test_cli_defaults_off_and_accepts_both_boolean_forms():
    """Catches selective checkpointing becoming implicit or inseparable from global mode."""
    assert parse_args([]).stage2_latent_activation_checkpointing is False
    assert parse_args(
        ["--stage2_latent_activation_checkpointing"]
    ).stage2_latent_activation_checkpointing is True
    assert parse_args(
        ["--no-stage2_latent_activation_checkpointing"]
    ).stage2_latent_activation_checkpointing is False


def test_launcher_and_coordinator_forward_selective_checkpointing():
    """Catches the approved env value being dropped before the worker CLI."""
    environment = validated_stage2_environment(
        {
            "STAGE1_CHECKPOINT": "/models/stage1",
            "DATA_PATH": "/data/train.jsonl",
            "IMAGE_ROOT": "/data/images",
            "STAGE2_LATENT_ACTIVATION_CHECKPOINTING": "true",
        },
        output_dir="/outputs/stage2",
        log_file="/logs/stage2.log",
        deepspeed_config="/configs/zero.json",
    )
    assert environment["STAGE2_LATENT_ACTIVATION_CHECKPOINTING"] == "true"

    process_environment = {
        "PATH": os.environ["PATH"],
        "STAGE2_DRY_RUN": "true",
        "STAGE1_CHECKPOINT": "/models/stage1",
        "DATA_PATH": "/data/train.jsonl",
        "IMAGE_ROOT": "/data/images",
        "OUTPUT_DIR": "/outputs/stage2",
        "STAGE2_LATENT_ACTIVATION_CHECKPOINTING": "true",
    }
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / "run_stage2_training.sh")],
        env=process_environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    tokens = shlex.split(completed.stdout)
    assert "--stage2_latent_activation_checkpointing" in tokens
    assert "--no-gradient_checkpointing" in tokens

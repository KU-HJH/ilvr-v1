"""Isolated tiny-model smoke for the real ZeRO-2 online-reference runtime."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "true")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29682")

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedType
import torch

from evidence_window.losses import stage2_ew_dpo_loss
from evidence_window.stage2_reference import OnlineFrozenStage2Reference


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(stage="stage1")
        self.projection = torch.nn.Linear(4, 4, bias=False)

    def forward(self, values):
        return self.projection(values)


def _config(directory: str) -> Path:
    path = Path(directory) / "zero2.json"
    path.write_text(
        json.dumps(
            {
                "train_batch_size": 1,
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "bf16": {"enabled": False},
                "fp16": {"enabled": False},
                "zero_optimization": {"stage": 2},
                "steps_per_print": 0,
            }
        ),
        encoding="utf-8",
    )
    return path


def main():
    try:
        with tempfile.TemporaryDirectory(prefix="stage2-zero2-") as directory:
            accelerator = Accelerator(
                mixed_precision="no",
                deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=str(_config(directory))),
            )
            assert accelerator.distributed_type is DistributedType.DEEPSPEED
            assert accelerator.state.deepspeed_plugin.zero_stage == 2

            policy = TinyModel()
            policy_before = {
                name: value.detach().clone() for name, value in policy.named_parameters()
            }
            reference = OnlineFrozenStage2Reference.create(
                TinyModel,
                accelerator,
                zero_stage=2,
                policy_checkpoint="tiny-stage1",
                reference_checkpoint="tiny-stage1",
            )
            before = {
                name: value.detach().clone()
                for name, value in reference.named_parameters()
            }
            optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
            policy, optimizer = accelerator.prepare(policy, optimizer)
            inputs = torch.ones(1, 4, device=accelerator.device)

            optimizer.zero_grad()
            policy_contribution = policy(inputs).mean().reshape(1, 1, 1)
            with torch.no_grad():
                reference_contribution = reference.model(inputs).mean().reshape(1, 1, 1)
            loss = stage2_ew_dpo_loss(
                policy_contribution,
                reference_contribution,
                torch.ones((1, 1, 1), dtype=torch.bool, device=accelerator.device),
                beta=0.4,
            )
            assert torch.isfinite(loss)
            accelerator.backward(loss)
            unwrapped_policy = accelerator.unwrap_model(policy)
            assert any(
                not torch.equal(policy_before[name].to(value.device), value)
                for name, value in unwrapped_policy.named_parameters()
            )
            optimizer.step()

            assert reference.model.config.stage == "stage2"
            assert all(value.grad is None for value in reference.parameters())
            assert all(
                torch.equal(before[name].to(value.device), value)
                for name, value in reference.named_parameters()
            )
            print("STAGE2_ZERO2_ONLINE_REFERENCE_OK", flush=True)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

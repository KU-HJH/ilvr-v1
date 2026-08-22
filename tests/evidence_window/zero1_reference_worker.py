"""Two-rank DeepSpeed ZeRO-1 smoke worker for the frozen reference."""

import json
from pathlib import Path
import tempfile

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedType
import torch

from evidence_window.reference import FrozenCopyReference


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.projection(x)


def _write_zero1_config(directory: str) -> Path:
    path = Path(directory) / "zero1.json"
    path.write_text(
        json.dumps(
            {
                "train_batch_size": 2,
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "bf16": {"enabled": False},
                "fp16": {"enabled": False},
                "zero_optimization": {
                    "stage": 1,
                    "contiguous_gradients": True,
                    "overlap_comm": False,
                },
                "steps_per_print": 0,
            }
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ew-zero1-") as directory:
        plugin = DeepSpeedPlugin(hf_ds_config=str(_write_zero1_config(directory)))
        accelerator = Accelerator(mixed_precision="no", deepspeed_plugin=plugin)

        assert accelerator.distributed_type is DistributedType.DEEPSPEED
        assert accelerator.state.deepspeed_plugin.zero_stage == 1

        torch.manual_seed(17)
        policy = TinyPolicy()
        policy_snapshot = {
            name: parameter.detach().clone()
            for name, parameter in policy.named_parameters()
        }
        reference = FrozenCopyReference.create(policy, accelerator)
        reference_snapshot = {
            name: parameter.detach().clone()
            for name, parameter in reference.named_parameters()
        }
        full_parameter_count = sum(parameter.numel() for parameter in policy.parameters())

        optimizer = torch.optim.SGD(policy.parameters(), lr=0.25)
        policy, optimizer = accelerator.prepare(policy, optimizer)

        reference_parameters = list(reference.parameters())
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        assert sum(parameter.numel() for parameter in reference_parameters) == full_parameter_count
        assert all(not parameter.requires_grad for parameter in reference_parameters)
        assert all(parameter.grad is None for parameter in reference_parameters)
        assert all(
            id(parameter) not in optimizer_parameter_ids
            for parameter in reference_parameters
        )

        local_flat = torch.cat(
            [parameter.detach().reshape(-1) for parameter in reference_parameters]
        ).to(accelerator.device)
        gathered = [torch.empty_like(local_flat) for _ in range(accelerator.num_processes)]
        torch.distributed.all_gather(gathered, local_flat)
        assert all(torch.equal(gathered[0], rank_copy) for rank_copy in gathered[1:])

        optimizer.zero_grad()
        inputs = torch.ones(1, 4, device=accelerator.device)
        loss = policy(inputs).sum()
        accelerator.backward(loss)
        optimizer.step()

        unwrapped_policy = accelerator.unwrap_model(policy)
        assert any(
            not torch.equal(
                policy_snapshot[name].to(parameter.device),
                parameter,
            )
            for name, parameter in unwrapped_policy.named_parameters()
        )
        assert all(parameter.grad is None for parameter in reference_parameters)
        assert all(
            torch.equal(reference_snapshot[name].to(parameter.device), parameter)
            for name, parameter in reference.named_parameters()
        )
        assert accelerator.state.deepspeed_plugin.zero_stage == 1
        print(f"ZERO1_REFERENCE_OK rank={accelerator.process_index}", flush=True)

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

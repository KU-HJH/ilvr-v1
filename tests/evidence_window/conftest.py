from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import trainer as trainer_module
from evidence_window.config import EvidenceWindowConfig
from trainer import CustomTrainerStage1


LATENT_TOKEN_IDS = {
    "start": 90,
    "end": 91,
    "pad": 92,
    "padding": 0,
    "special": (0, 90, 91, 92, 99),
}


@dataclass
class TinyOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    hidden_states: torch.Tensor
    inputs_embeds: torch.Tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError(index)
        return self.loss


class TinyVisual(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = torch.float32
        self.forward_count = 0

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        del grid_thw
        self.forward_count += 1
        rows = pixel_values.reshape(-1, 1).to(dtype=self.dtype)
        return rows.expand(-1, self.hidden_size)


class TinyLatentLM(torch.nn.Module):
    """Small causal LM with observable full/all-null/LOO condition calls."""

    def __init__(self, *, output_bias: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(latent_size=8)
        self.embed_tokens = torch.nn.Embedding(128, 4)
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 128, bias=output_bias)
        self.visual = TinyVisual(4)
        with torch.no_grad():
            values = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)
            self.embed_tokens.weight.copy_((values.remainder(17) - 8) / 16)
            self.projection.weight.copy_(
                torch.tensor(
                    [
                        [0.6, -0.2, 0.1, 0.0],
                        [0.0, 0.5, 0.2, -0.1],
                        [0.1, 0.0, 0.4, 0.3],
                        [-0.2, 0.1, 0.0, 0.7],
                    ]
                )
            )
            head = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)
            self.lm_head.weight.copy_((head.remainder(23) - 11) / 32)
            if self.lm_head.bias is not None:
                self.lm_head.bias.zero_()
        self.reset_counts()

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head

    def reset_counts(self) -> None:
        self.forward_count = 0
        self.full_forward_count = 0
        self.all_null_forward_count = 0
        self.loo_forward_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        latent_hidden_states: torch.Tensor | None = None,
        image_out_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_unused,
    ) -> TinyOutput:
        del attention_mask, output_hidden_states, return_dict
        self.forward_count += 1
        if latent_hidden_states is None:
            self.full_forward_count += 1
        elif torch.count_nonzero(latent_hidden_states).item() == 0:
            self.all_null_forward_count += 1
        elif (latent_hidden_states == 0).all(dim=-1).any():
            self.loo_forward_count += 1
        else:
            self.full_forward_count += 1

        inputs_embeds = self.embed_tokens(input_ids)
        if latent_hidden_states is not None:
            if image_out_mask is None:
                raise ValueError("latent values require image_out_mask")
            flat_values = latent_hidden_states.reshape(-1, latent_hidden_states.shape[-1])
            flat_mask = image_out_mask.to(dtype=torch.bool).reshape(-1)
            if int(flat_mask.sum()) != flat_values.shape[0]:
                raise ValueError("compact latent count does not match mask")
            injected = inputs_embeds.reshape(-1, inputs_embeds.shape[-1]).clone()
            injected[flat_mask] = flat_values.to(injected)
            inputs_embeds = injected.reshape_as(inputs_embeds)

        contextual = inputs_embeds.cumsum(dim=1)
        hidden_states = F.layer_norm(self.projection(contextual), (4,))
        logits = self.lm_head(hidden_states)
        if labels is None or labels.shape[1] <= 1:
            loss = logits.sum() * 0.0
        else:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return TinyOutput(loss, logits, hidden_states, inputs_embeds)


class TinyAccelerator:
    def __init__(self) -> None:
        self.prepared_models: list[torch.nn.Module] = []
        self.state = SimpleNamespace(deepspeed_plugin=None)
        self.num_processes = 1

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def gather_for_metrics(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def prepare_model(
        self, model: torch.nn.Module, *, evaluation_mode: bool = False
    ) -> torch.nn.Module:
        assert evaluation_mode
        self.prepared_models.append(model)
        return model


@pytest.fixture
def snapshot_parameters():
    def snapshot(module):
        return {
            name: tensor.detach().clone()
            for name, tensor in module.named_parameters()
        }

    return snapshot


@pytest.fixture
def tiny_trainer_factory(monkeypatch, tmp_path):
    def fake_sft_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = kwargs.get(
            "args",
            SimpleNamespace(
                past_index=-1,
                average_tokens_across_devices=False,
                use_liger=False,
                output_dir=str(tmp_path),
            ),
        )
        instance.tokenizer = kwargs.get("tokenizer")
        instance.label_smoother = None
        instance.compute_loss_func = None
        instance.model_accepts_loss_kwargs = False
        instance.accelerator = TinyAccelerator()
        instance._metrics = defaultdict(list)
        instance._parent_init_kwargs = dict(kwargs)

    monkeypatch.setattr(trainer_module.SFTTrainer, "__init__", fake_sft_init)

    def factory(
        *,
        evidence_window_config: EvidenceWindowConfig | None = None,
        latent_token_ids: dict | None = None,
        model: TinyLatentLM | None = None,
        output_bias: bool = False,
        sim_weight: float = 0.7,
        ce_weight: float = 1.3,
    ) -> tuple[CustomTrainerStage1, TinyLatentLM]:
        model = TinyLatentLM(output_bias=output_bias) if model is None else model
        kwargs = {}
        if evidence_window_config is not None:
            kwargs["evidence_window_config"] = evidence_window_config
        if latent_token_ids is not None:
            kwargs["latent_token_ids"] = latent_token_ids
        trainer = CustomTrainerStage1(
            model=model,
            args=SimpleNamespace(
                past_index=-1,
                average_tokens_across_devices=False,
                use_liger=False,
                output_dir=str(tmp_path),
            ),
            tokenizer=SimpleNamespace(),
            sim_weight=sim_weight,
            ce_weight=ce_weight,
            **kwargs,
        )
        trainer._ew_engine = getattr(trainer, "_ew_engine", None)
        trainer._reference = getattr(trainer, "_reference", None)

        latent_values = torch.tensor(
            [[[0.25, -0.5, 0.75, 1.0], [-0.75, 0.5, 0.25, -0.25]]]
        )
        active_mask = torch.tensor(
            [[False, False, True, False, False, False, True, False, False, False, False]]
        )

        def fixed_latents(_self, inputs, k, p):
            del inputs, k, p
            return latent_values.clone(), active_mask.clone()

        trainer._teacher_build_latents = MethodType(fixed_latents, trainer)
        return trainer, model

    return factory


def _base_inputs() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[5, 90, 92, 91, 7, 90, 92, 91, 8, 9, 0]])
    labels = torch.tensor([[-100, -100, -100, -100, 7, -100, -100, -100, 8, 9, -100]])
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "labels": labels,
        "image_out_mask": input_ids.eq(92),
        "pixel_values_latent": torch.ones((2, 1)),
        "image_grid_thw_latent": torch.ones((2, 3), dtype=torch.long),
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
    }


@pytest.fixture
def legacy_latent_case():
    return {
        "inputs": _base_inputs(),
        "recorded_legacy_loss": torch.tensor(6.780547618865967),
        "recorded_logits_sum": torch.tensor(-6.302485466003418),
        "recorded_legacy_forward_count": 1,
    }


@pytest.fixture
def legacy_no_latent_case():
    return {
        "inputs": _base_inputs(),
        "recorded_ce_loss": torch.tensor(4.7625346183776855),
        "recorded_logits_sum": torch.tensor(-5.442888259887695),
        "recorded_legacy_forward_count": 1,
    }

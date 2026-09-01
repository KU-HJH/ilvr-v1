from types import SimpleNamespace

import pytest
import torch
from torch import nn

from evaluation.continuous_latent import (
    ContinuousLatentController,
    ContinuousLatentGenerationConfig,
    GenerationLimitError,
)
from stage2.peft import resolve_stage2_qwen_model
from tests.stage2.helpers import ToyCausalLanguageModel, tiny_stage2_qwen
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentRolloutEngine,
    LatentTokenIds,
)


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


class ScriptedHead(nn.Module):
    def __init__(self, token_ids, vocab_size=128):
        super().__init__()
        self.token_ids = list(token_ids)
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, hidden):
        token_id = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        logits = hidden.new_full((*hidden.shape[:-1], self.vocab_size), -1e9)
        logits[..., token_id] = 0
        return logits


class RecordingScriptedHead(ScriptedHead):
    def __init__(self, token_ids, vocab_size=128):
        super().__init__(token_ids, vocab_size)
        self.hidden_inputs = []
        self.outputs = []

    def forward(self, hidden):
        self.hidden_inputs.append(hidden.clone())
        logits = super().forward(hidden)
        self.outputs.append(logits.clone())
        return logits


class ScriptedContinuousLatentModel(nn.Module):
    def __init__(self, *, emitted_ids, latent_size, stage):
        super().__init__()
        self.config = SimpleNamespace(latent_size=latent_size, stage=stage)
        self.model = ToyCausalLanguageModel(4)
        self.lm_head = ScriptedHead(emitted_ids)
        self.stage2_rollout_engine = LatentRolloutEngine()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_stage2_base_inputs_embeds(
        self, *, input_ids, pixel_values, image_grid_thw
    ):
        return self.get_input_embeddings()(input_ids)

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return positions, torch.zeros(input_ids.shape[0], 1, device=input_ids.device)


def scripted_model(*, latent_size, emitted_ids, stage="stage2"):
    return ScriptedContinuousLatentModel(
        emitted_ids=emitted_ids,
        latent_size=latent_size,
        stage=stage,
    )


def test_neutral_controller_records_complete_latent_block_evidence():
    """Catches a controller that omits completed-block evidence or EOS state."""
    model = scripted_model(
        latent_size=2,
        emitted_ids=[TOKENS.start_token_id, 3],
    )
    input_ids = torch.tensor([[1, 2]])
    token_ids = TOKENS

    result = ContinuousLatentController().generate(
        model=model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        token_ids=token_ids,
        generation_config=ContinuousLatentGenerationConfig(
            max_new_tokens=16, temperature=0.0, top_p=1.0, eos_token_id=3
        ),
        return_latent_states=True,
        return_post_end_logits=True,
    )

    assert result.latent_start_count == 1
    assert result.latent_block_count == 1
    assert result.generated_suffix_ids == tuple(
        result.sequences[0, input_ids.shape[1] :].tolist()
    )
    assert result.stop_reason == "eos"
    assert len(result.post_end_logits_by_block) == 1


def test_post_end_logits_use_complete_post_recurrence_scaffold():
    """Catches evidence taken before continuous latent replacements finish."""
    model = scripted_model(
        latent_size=2,
        emitted_ids=[TOKENS.start_token_id, 3],
    )
    model.lm_head = RecordingScriptedHead([TOKENS.start_token_id, 3])

    result = ContinuousLatentController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=ContinuousLatentGenerationConfig(
            max_new_tokens=5, eos_token_id=3
        ),
        return_post_end_logits=True,
    )

    complete_embeds, _, complete_positions = model.model.calls[3]
    expected_hidden = (
        complete_embeds.cumsum(dim=1) * model.model.scale
        + complete_positions[0].unsqueeze(-1).to(complete_embeds.dtype)
    )[:, -1, :]
    torch.testing.assert_close(model.lm_head.hidden_inputs[1], expected_hidden)
    torch.testing.assert_close(
        result.post_end_logits_by_block[0], model.lm_head.outputs[1]
    )


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_neutral_controller_accepts_stage_tag_without_rewriting_it(stage):
    """Catches controller validation that treats a model tag as an API boundary."""
    model = scripted_model(latent_size=2, emitted_ids=[3], stage=stage)

    result = ContinuousLatentController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=ContinuousLatentGenerationConfig(
            max_new_tokens=1, eos_token_id=3
        ),
    )

    assert model.config.stage == stage
    assert result.stop_reason == "eos"


def test_stage2_resolver_remains_strict_for_stage1_contract_model():
    """Catches loosening the Stage 2 resolver while factoring neutral inference."""
    stage1_model = tiny_stage2_qwen()
    stage1_model.config.stage = "stage1"

    with pytest.raises(TypeError, match="Stage 2 Qwen interface"):
        resolve_stage2_qwen_model(stage1_model)


def test_sequence_cap_raises_generation_limit_error_before_partial_block_rollout():
    """Catches returning a partial scaffold when a latent block exceeds the cap."""
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])

    with pytest.raises(GenerationLimitError, match="latent block would exceed"):
        ContinuousLatentController().generate(
            model=model,
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            token_ids=TOKENS,
            generation_config=ContinuousLatentGenerationConfig(
                max_new_tokens=1,
                max_sequence_length=4,
            ),
        )

    assert len(model.model.calls) == 1

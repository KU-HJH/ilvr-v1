from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stage1.targets import Stage1TeacherTargetBuilder
from stage1.types import (
    Stage1BatchMetadata,
    Stage1LatentSpan,
    Stage1SampleMetadata,
    Stage1TeacherTargets,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)


IMAGE_TOKEN_ID = 99
ASSISTANT_TOKEN_ID = 50
LATENT_PAD_TOKEN_ID = 90
LATENT_START_TOKEN_ID = 91
LATENT_END_TOKEN_ID = 92


class LiteralTokenizer:
    _ids = {
        "<|im_start|>assistant": ASSISTANT_TOKEN_ID,
        "<|im_start|>assistant\n": ASSISTANT_TOKEN_ID,
        "<|latent_pad|>": LATENT_PAD_TOKEN_ID,
        "<|latent_start|>": LATENT_START_TOKEN_ID,
        "<|latent_end|>": LATENT_END_TOKEN_ID,
        "<|vision_start|>": 97,
        "<|vision_end|>": 98,
    }

    def __call__(self, text, *, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[self._ids[text]]], dtype=torch.long)}


class MultiTokenAssistantTokenizer(LiteralTokenizer):
    def __call__(self, text, *, return_tensors):
        if text == "<|im_start|>assistant\n":
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor(
                    [[ASSISTANT_TOKEN_ID, 51]], dtype=torch.long
                )
            }
        return super().__call__(text, return_tensors=return_tensors)


class IdentityAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.k_proj = nn.Linear(2, 2, bias=False)
        self.num_heads = 1
        self.num_key_value_heads = 1
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(2))
            self.k_proj.weight.copy_(torch.eye(2))


class RaisingProjection(nn.Module):
    def __init__(self, error: Exception):
        super().__init__()
        self.error = error

    def forward(self, unused):
        raise self.error


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.self_attn = IdentityAttention()
        self.layers = nn.ModuleList([layer])


class CountingVisual(nn.Module):
    spatial_merge_size = 1

    def __init__(self):
        super().__init__()
        self.original_call_count = 0
        self.helper_call_count = 0

    @property
    def dtype(self):
        return torch.float32

    def forward(self, pixel_values, *, grid_thw):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert grid_thw.ndim == 2 and grid_thw.shape[1] == 3
        if pixel_values.numel() == 0:
            return pixel_values.new_empty((0, 2))
        tag = pixel_values[:, 0]
        if torch.all(tag == 0):
            self.original_call_count += 1
        elif torch.all(tag == 1):
            self.helper_call_count += 1
        else:
            raise AssertionError("visual inputs must not mix original and helper views")
        return pixel_values[:, 1:].to(torch.float32)


class CompleteQwenTeacher(nn.Module):
    """Complete Qwen-like output at the heavy model boundary."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN_ID,
            latent_size=2,
            output_hidden_states=False,
        )
        self.model = FakeDecoder()
        self.embedding = nn.Embedding(128, 2)
        self.visual = CountingVisual()
        self.language_forward_count = 0
        with torch.no_grad():
            self.embedding.weight.zero_()
            literal_rows = {
                10: (1.0, 0.0),
                11: (0.0, 1.0),
                12: (1.0, 0.0),
                13: (0.0, 1.0),
                20: (2.0, 0.0),
                21: (0.0, 2.0),
                22: (1.0, -1.0),
                30: (2.0, 0.0),
                31: (0.0, 2.0),
                ASSISTANT_TOKEN_ID: (1.0, 1.0),
            }
            for token_id, value in literal_rows.items():
                self.embedding.weight[token_id] = torch.tensor(value)
        self.requires_grad_(False)

    @property
    def original_visual_call_count(self):
        return self.visual.original_call_count

    @property
    def helper_visual_call_count(self):
        return self.visual.helper_call_count

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        output_hidden_states,
        return_dict,
    ):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert output_hidden_states is True
        assert return_dict is True
        self.language_forward_count += 1

        inputs_embeds = self.embedding(input_ids).detach().clone()
        image_features = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_positions = (input_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=False)
        assert image_positions.shape[0] == image_features.shape[0]
        for feature, (row, column) in zip(image_features, image_positions.tolist()):
            inputs_embeds[row, column] = feature

        batch_size, sequence_length = input_ids.shape
        hidden_pre = inputs_embeds.clone()
        hidden_last = inputs_embeds.clone()
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=None,
            logits=torch.zeros(batch_size, sequence_length, 128),
            past_key_values=(),
            hidden_states=(hidden_pre, hidden_last),
            attentions=(
                torch.zeros(batch_size, 1, sequence_length, sequence_length),
            ),
            rope_deltas=torch.zeros(batch_size, 1, dtype=torch.long),
            inputs_embeds=inputs_embeds,
            stage2_latent_states=None,
            stage2_diagnostics=None,
        )


@dataclass(frozen=True)
class LiteralSample:
    source_index: int
    input_ids: tuple[int, ...]
    spans: tuple[Stage1LatentSpan, ...]
    original_feature: tuple[float, float]
    helper_features: tuple[tuple[tuple[float, float], ...], ...]


ONE_BLOCK = LiteralSample(
    source_index=7,
    input_ids=(12, 99, 13, 50, 30, 91, 90, 90, 92, 31),
    spans=(Stage1LatentSpan(start=5, end=8, latent_positions=(6, 7)),),
    original_feature=(1.0, 1.0),
    helper_features=(((3.0, 0.0), (-3.0, 0.0)),),
)

GOLDEN_MULTIBLOCK = LiteralSample(
    source_index=11,
    input_ids=(10, 99, 11, 50, 20, 91, 90, 90, 92, 21, 91, 90, 90, 92, 22),
    spans=(
        Stage1LatentSpan(start=5, end=8, latent_positions=(6, 7)),
        Stage1LatentSpan(start=10, end=13, latent_positions=(11, 12)),
    ),
    original_feature=(0.0, 2.0),
    helper_features=(
        ((20.0, 0.0), (-20.0, 0.0), (999.0, 999.0)),
        ((0.0, 1.0), (0.0, 3.0), (1.0, 0.0), (3.0, 0.0)),
    ),
)


THREE_BLOCK_HISTORY = LiteralSample(
    source_index=31,
    input_ids=(
        10,
        99,
        11,
        50,
        20,
        91,
        90,
        90,
        92,
        21,
        91,
        90,
        90,
        92,
        22,
        91,
        90,
        90,
        92,
        30,
    ),
    spans=(
        Stage1LatentSpan(start=5, end=8, latent_positions=(6, 7)),
        Stage1LatentSpan(start=10, end=13, latent_positions=(11, 12)),
        Stage1LatentSpan(start=15, end=18, latent_positions=(16, 17)),
    ),
    original_feature=(0.0, 2.0),
    helper_features=(
        ((10.0, 0.0), (20.0, 0.0)),
        ((0.0, 9.0), (0.0, 12.0)),
        ((1.0, 1.0),),
    ),
)


@dataclass
class TargetCase:
    teacher: CompleteQwenTeacher
    inputs: dict[str, object]
    metadata: Stage1BatchMetadata


def make_case(
    samples: tuple[LiteralSample, ...], *, first_helper_scale: float = 1.0
) -> TargetCase:
    width = max(len(sample.input_ids) for sample in samples)
    input_ids = torch.zeros((len(samples), width), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    original_pixels = []
    helper_pixels = []
    original_grids = []
    helper_grids = []
    sample_metadata = []
    original_offset = 0
    helper_offset = 0

    for sample_index, sample in enumerate(samples):
        length = len(sample.input_ids)
        input_ids[sample_index, :length] = torch.tensor(sample.input_ids)
        attention_mask[sample_index, :length] = 1
        labels[sample_index, 4:length] = input_ids[sample_index, 4:length]

        original_pixels.append((0.0, *sample.original_feature))
        original_grids.append((1, 1, 1))
        helper_start = helper_offset
        scale = first_helper_scale if sample_index == 0 else 1.0
        for image in sample.helper_features:
            helper_pixels.extend((1.0, x * scale, y * scale) for x, y in image)
            helper_grids.append((1, len(image), 1))
            helper_offset += 1
        sample_metadata.append(
            Stage1SampleMetadata(
                source_index=sample.source_index,
                source_id=f"sample-{sample.source_index}",
                original_grid_range=(original_offset, original_offset + 1),
                helper_grid_range=(helper_start, helper_offset),
                latent_spans=sample.spans,
                helper_to_block=tuple(range(len(sample.helper_features))),
                unpadded_length=length,
            )
        )
        original_offset += 1

    metadata = Stage1BatchMetadata(
        samples=tuple(sample_metadata),
        batch_size=len(samples),
        sequence_length=width,
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_out_mask": torch.zeros_like(input_ids),
        "pixel_values": torch.tensor(original_pixels, dtype=torch.float32),
        "image_grid_thw": torch.tensor(original_grids, dtype=torch.long),
        "pixel_values_latent": torch.tensor(helper_pixels, dtype=torch.float32),
        "image_grid_thw_latent": torch.tensor(helper_grids, dtype=torch.long),
    }
    return TargetCase(teacher=CompleteQwenTeacher(), inputs=inputs, metadata=metadata)


@pytest.fixture
def target_builder():
    return Stage1TeacherTargetBuilder(
        tokenizer=LiteralTokenizer(),
        coverage_p=0.6,
        image_pool_k=2,
        helper_group_L=2,
    )


def test_batched_targets_equal_row_major_independent_results(target_builder):
    """Catches flattened helper features or packed values crossing row ownership."""
    batched_case = make_case((ONE_BLOCK, GOLDEN_MULTIBLOCK))
    batched = target_builder.build(
        batched_case.teacher, batched_case.inputs, batched_case.metadata
    )
    independent = []
    for sample in (ONE_BLOCK, GOLDEN_MULTIBLOCK):
        case = make_case((sample,))
        independent.append(
            target_builder.build(case.teacher, case.inputs, case.metadata)
        )

    torch.testing.assert_close(
        batched.latent_values,
        torch.cat([result.latent_values for result in independent], dim=0),
    )
    assert batched.selected_counts == ((1,), (1, 1))
    assert batched.latent_mask.tolist() == [
        [False, False, False, False, False, False, True, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, True, False, False, False, False, True, False, False, False],
    ]


def test_second_sample_cannot_observe_first_sample_state(target_builder):
    """Catches previous selections or helper offsets leaking across batch rows."""
    baseline_case = make_case((ONE_BLOCK, GOLDEN_MULTIBLOCK))
    perturbed_case = make_case(
        (ONE_BLOCK, GOLDEN_MULTIBLOCK), first_helper_scale=1000.0
    )
    baseline = target_builder.build(
        baseline_case.teacher, baseline_case.inputs, baseline_case.metadata
    )
    perturbed = target_builder.build(
        perturbed_case.teacher, perturbed_case.inputs, perturbed_case.metadata
    )
    first_count = sum(baseline.selected_counts[0])

    assert not torch.equal(
        baseline.latent_values[:first_count], perturbed.latent_values[:first_count]
    )
    torch.testing.assert_close(
        baseline.latent_values[first_count:],
        perturbed.latent_values[first_count:],
    )


def test_teacher_runs_one_language_and_one_helper_visual_call_for_multiple_blocks(
    target_builder,
):
    """Catches a language or visual teacher call being repeated per latent block."""
    case = make_case((GOLDEN_MULTIBLOCK,))
    result = target_builder.build(case.teacher, case.inputs, case.metadata)

    assert result.latent_values.shape == (2, 2)
    assert case.teacher.language_forward_count == 1
    assert case.teacher.original_visual_call_count == 1
    assert case.teacher.helper_visual_call_count == 1
    assert not result.latent_values.requires_grad
    assert not case.teacher.training


def test_text_history_starts_after_the_longest_assistant_marker():
    """Catches a short marker prefix leaking remaining control tokens."""
    builder = Stage1TeacherTargetBuilder(MultiTokenAssistantTokenizer())
    sample_ids = torch.tensor(
        [10, ASSISTANT_TOKEN_ID, 51, 20, 91, 90, 92], dtype=torch.long
    )
    sample = Stage1SampleMetadata(
        source_index=37,
        source_id="multi-token-assistant-marker",
        original_grid_range=(0, 0),
        helper_grid_range=(0, 1),
        latent_spans=(
            Stage1LatentSpan(start=4, end=6, latent_positions=(5,)),
        ),
        helper_to_block=(0,),
        unpadded_length=7,
    )

    assistant_start, assistant_end = builder._assistant_bounds(sample_ids, sample)

    assert (assistant_start, assistant_end) == (1, 3)
    assert builder._assistant_text_spans(
        sample_ids, sample, assistant_end
    ) == ([3],)


def test_previous_textual_states_matches_repository_queries():
    """Catches the optional all-history path replacing repository behavior."""

    class PreviousQueryBuilder(Stage1TeacherTargetBuilder):
        def __init__(self, textual_states):
            super().__init__(
                tokenizer=LiteralTokenizer(),
                coverage_p=1.0,
                image_pool_k=2,
                helper_group_L=0,
                textual_states=textual_states,
            )
            self.queries = []

        def _select(self, candidates, query, max_count):
            selected_counts = (2, 1, 1)
            block_index = len(self.queries)
            self.queries.append(query.detach().clone())
            return candidates[: min(selected_counts[block_index], max_count)]

    case = make_case((THREE_BLOCK_HISTORY,))
    builder = PreviousQueryBuilder(textual_states="previous")

    result = builder.build(case.teacher, case.inputs, case.metadata)

    torch.testing.assert_close(
        torch.cat(builder.queries, dim=0),
        torch.tensor(
            [
                [0.875, 0.875],
                [61.0 / 12.0, 13.0 / 12.0],
                [5.0 / 12.0, 37.0 / 12.0],
            ]
        ),
    )
    torch.testing.assert_close(
        result.latent_values,
        torch.tensor(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [0.0, 9.0],
                [1.0, 1.0],
            ]
        ),
    )
    assert case.teacher.language_forward_count == 1
    assert case.teacher.original_visual_call_count == 1
    assert case.teacher.helper_visual_call_count == 1


def test_all_textual_states_averages_all_text_and_prior_latent_rows_in_order():
    """Catches chat markers, local-only text, or previous-block-only latents."""

    class QueryRecordingBuilder(Stage1TeacherTargetBuilder):
        def __init__(self, textual_states):
            super().__init__(
                tokenizer=LiteralTokenizer(),
                coverage_p=1.0,
                image_pool_k=2,
                helper_group_L=0,
                textual_states=textual_states,
            )
            self.query_components = []
            self.queries = []

        def _reasoning_query(
            self, global_intent, textual_history, latent_history=None
        ):
            parts = [global_intent, textual_history]
            if latent_history is not None:
                parts.append(latent_history)
            self.query_components.append(torch.cat(parts, dim=0).detach().clone())
            return super()._reasoning_query(
                global_intent, textual_history, latent_history
            )

        def _select(self, candidates, query, max_count):
            selected_counts = (2, 1, 1)
            block_index = len(self.queries)
            self.queries.append(query.detach().clone())
            return candidates[: min(selected_counts[block_index], max_count)]

    case = make_case((THREE_BLOCK_HISTORY,))
    builder = QueryRecordingBuilder(textual_states="all")

    result = builder.build(case.teacher, case.inputs, case.metadata)

    assert [tuple(query.shape) for query in builder.queries] == [(1, 2)] * 3
    torch.testing.assert_close(
        builder.query_components[0],
        torch.tensor(
            [
                [0.25, 1.25],
                [2.0, 0.0],
            ]
        ),
    )
    torch.testing.assert_close(
        builder.queries[2],
        torch.tensor([[3.75, 55.0 / 36.0]]),
    )
    assert [tuple(parts.shape) for parts in builder.query_components] == [
        (2, 2),
        (3, 2),
        (3, 2),
    ]
    torch.testing.assert_close(
        builder.query_components[2],
        torch.tensor(
            [
                [0.25, 1.25],
                [1.0, 1.0 / 3.0],
                [10.0, 3.0],
            ]
        ),
    )
    torch.testing.assert_close(
        result.latent_values,
        torch.tensor(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [0.0, 9.0],
                [1.0, 1.0],
            ]
        ),
    )
    assert case.teacher.language_forward_count == 1
    assert case.teacher.original_visual_call_count == 1
    assert case.teacher.helper_visual_call_count == 1


def test_b1_matches_hand_derived_legacy_target_mask_and_downstream_inputs(
    target_builder,
):
    """Catches drift in coverage, grouping, recurrence, or first-K alignment."""
    case = make_case((GOLDEN_MULTIBLOCK,))
    result = target_builder.build(case.teacher, case.inputs, case.metadata)

    assert isinstance(result, Stage1TeacherTargets)
    torch.testing.assert_close(
        result.latent_values,
        torch.tensor([[20.0, 0.0], [2.0, 0.0]]),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.selected_counts == ((1, 1),)
    assert result.latent_mask.tolist() == [[
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]]

    ce_inputs = dict(case.inputs)
    ce_inputs["latent_hidden_states"] = result.latent_values
    ce_inputs["image_out_mask"] = result.latent_mask
    assert "pixel_values_latent" not in ce_inputs
    assert "image_grid_thw_latent" not in ce_inputs
    torch.testing.assert_close(
        ce_inputs["latent_hidden_states"],
        torch.tensor([[20.0, 0.0], [2.0, 0.0]]),
        rtol=1e-5,
        atol=1e-6,
    )

    injected_inputs = torch.zeros((1, 15, 2))
    injected_inputs[result.latent_mask] = result.latent_values
    alignment_mask = result.latent_mask[:, 1:]
    alignment_targets = injected_inputs[:, 1:, :][alignment_mask]
    assert alignment_mask.nonzero(as_tuple=False).tolist() == [[0, 5], [0, 10]]
    torch.testing.assert_close(
        alignment_targets,
        torch.tensor([[20.0, 0.0], [2.0, 0.0]]),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.diagnostics
    assert all(torch.isfinite(value).all() for value in result.diagnostics.values())


def test_builder_consumes_helper_only_tensors_before_student_boundary(target_builder):
    """Catches helper pixels surviving in the later student forward mapping."""
    case = make_case((ONE_BLOCK,))

    target_builder.build(case.teacher, case.inputs, case.metadata)

    assert "pixel_values_latent" not in case.inputs
    assert "image_grid_thw_latent" not in case.inputs


def test_none_helper_group_uses_legacy_image_pool_k():
    """Catches None disabling the legacy image-pool-sized helper grouping."""
    builder = Stage1TeacherTargetBuilder(
        tokenizer=LiteralTokenizer(),
        coverage_p=0.6,
        image_pool_k=2,
        helper_group_L=None,
    )
    case = make_case((GOLDEN_MULTIBLOCK,))

    result = builder.build(case.teacher, case.inputs, case.metadata)

    assert result.selected_counts == ((1, 1),)
    torch.testing.assert_close(
        result.latent_values,
        torch.tensor([[20.0, 0.0], [2.0, 0.0]]),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("synthetic attention kernel failure"),
        torch.cuda.OutOfMemoryError("synthetic attention OOM"),
        ValueError("synthetic attention shape failure"),
        AttributeError("synthetic projection implementation failure"),
        IndexError("synthetic attention indexing failure"),
    ],
    ids=("runtime", "oom", "shape", "internal-attribute", "index"),
)
def test_attention_query_execution_failures_propagate(target_builder, error):
    """Catches runtime, OOM, or shape failures being hidden by cosine fallback."""
    case = make_case((ONE_BLOCK,))
    case.teacher.model.layers[-1].self_attn.q_proj = RaisingProjection(error)

    with pytest.raises(type(error), match=str(error)) as caught:
        target_builder.build(case.teacher, case.inputs, case.metadata)

    assert caught.value is error


def test_missing_attention_interface_uses_cosine_fallback(target_builder):
    """Catches removal of the intentional fallback for unsupported teachers."""
    case = make_case((ONE_BLOCK,))
    del case.teacher.model

    result = target_builder.build(
        case.teacher, case.inputs, case.metadata
    )

    assert result.selected_counts == ((1,),)
    assert torch.isfinite(result.latent_values).all()


@pytest.mark.parametrize(
    "image_token_id",
    [None, True, -1, "99"],
    ids=("none", "bool", "negative", "string"),
)
def test_invalid_teacher_image_token_id_fails_clearly(
    target_builder, image_token_id
):
    """Catches malformed image-token metadata reaching query construction."""
    case = make_case((ONE_BLOCK,))
    case.teacher.config.image_token_id = image_token_id

    with pytest.raises(
        ValueError,
        match=r"teacher\.config\.image_token_id must be a non-negative integer",
    ):
        target_builder.build(case.teacher, case.inputs, case.metadata)


def test_missing_teacher_image_token_id_fails_clearly(target_builder):
    """Catches the Qwen-specific token ID being replaced by a hard-coded default."""
    case = make_case((ONE_BLOCK,))
    del case.teacher.config.image_token_id

    with pytest.raises(
        ValueError,
        match=r"teacher\.config\.image_token_id must be a non-negative integer",
    ):
        target_builder.build(case.teacher, case.inputs, case.metadata)

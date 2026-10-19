"""CPU runtime witnesses for the bounded historical visual-history substitution.

These literal frozen-teacher probes establish synthetic selector behavior only;
they make no claim about trained-model equivalence or evaluation accuracy.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import pytest
import torch

from stage1.targets import Stage1TeacherTargetBuilder
from stage1.types import Stage1LatentSpan
from tests.stage1.test_targets import (
    BFloat16QwenTeacher,
    LiteralSample,
    LiteralTokenizer,
    ONE_BLOCK,
    TIED_CANDIDATES,
    make_case,
)


HISTORICAL_COMMIT = "c21b4e5c311204df4281317219b3f4a6b95e4d3b"
HISTORICAL_TARGET_SHA256 = (
    "a9b709cf9873ebd2711bc9ceccacbe985cd6bccd64b27c1b6efaacf591d623b4"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def historical_builder_class():
    """Execute the actual pinned historical class, never a rewritten oracle."""
    source = subprocess.check_output(
        ["git", "show", f"{HISTORICAL_COMMIT}:src/stage1/targets.py"],
        cwd=REPO_ROOT,
    )
    assert hashlib.sha256(source).hexdigest() == HISTORICAL_TARGET_SHA256
    module = ModuleType("_historical_stage1_targets_c21b4e5")
    exec(compile(source, f"git:{HISTORICAL_COMMIT}:targets.py", "exec"), module.__dict__)
    return module.Stage1TeacherTargetBuilder


def observed_builder(builder_class, *, coverage_p=0.9):
    """Observe real dispatch while retaining each implementation's real selector."""
    class ObservedBuilder(builder_class):
        def __init__(self):
            super().__init__(
                LiteralTokenizer(),
                coverage_p=coverage_p,
                image_pool_k=2,
                helper_group_L=0,
            )
            self.intents = []
            self.text_spans = []
            self.text_rows = []
            self.text_means = []
            self.selections = []

        def _user_query(self, **kwargs):
            self.current_hidden = kwargs["hidden_last"].detach().clone()
            intent = super()._user_query(**kwargs)
            self.intents.append(intent.detach().clone())
            return intent

        def _assistant_text_spans(self, *args):
            spans = super()._assistant_text_spans(*args)
            self.text_spans.append(spans)
            rows = tuple(self.current_hidden[list(indices)] for indices in spans)
            self.text_rows.append(rows)
            self.text_means.append(
                tuple(row.mean(dim=0, keepdim=True) if len(row) else None for row in rows)
            )
            return spans

        def _select(self, candidates, query, max_count):
            selected, indices, margin = super()._select(candidates, query, max_count)
            self.selections.append(
                SimpleNamespace(
                    candidates=candidates.detach().clone(),
                    query=query.detach().clone(),
                    max_count=max_count,
                    selected=selected.detach().clone(),
                    indices=indices,
                    margin=margin,
                )
            )
            return selected, indices, margin

    return ObservedBuilder()


def sample_with_blocks(
    *,
    texts=((20,), (21,), (22,)),
    helpers=(
        ((12.0, 0.0), (0.0, 2.0), (-1.0, -1.0)),
        ((0.0, 3.0), (3.0, 0.0), (-2.0, -1.0)),
        ((3.0, 0.0), (3.0, 0.0), (0.0, 4.0), (-1.0, 0.0)),
    ),
    capacities=(2, 1, 2),
    source_index=71,
):
    ids = [10, 99, 11, 50]
    spans = []
    for text, capacity in zip(texts, capacities):
        ids.extend(text)
        start = len(ids)
        ids.extend([91, *([90] * capacity), 92])
        spans.append(
            Stage1LatentSpan(
                start=start,
                end=len(ids) - 1,
                latent_positions=tuple(range(start + 1, start + 1 + capacity)),
            )
        )
    ids.append(31)
    return LiteralSample(
        source_index=source_index,
        input_ids=tuple(ids),
        spans=tuple(spans),
        original_feature=(0.0, 2.0),
        helper_features=helpers,
    )


THREE_BLOCK = sample_with_blocks()


def run(builder_class, samples=(THREE_BLOCK,), *, teacher=None, builder=None):
    case = make_case(samples)
    if teacher is not None:
        case.teacher = teacher
    if builder is None:
        builder = observed_builder(builder_class)
    result = builder.build(case.teacher, case.inputs, case.metadata)
    return SimpleNamespace(builder=builder, case=case, result=result)


def independent_summary(text_rows, history_rows):
    """Dense FP32 cosine/softmax oracle independent of the product helper."""
    text = text_rows.float()
    values = history_rows.float()
    text_unit = text / torch.sqrt((text * text).sum(dim=1, keepdim=True))
    values_unit = values / torch.sqrt((values * values).sum(dim=1, keepdim=True))
    probabilities = torch.softmax(text_unit @ values_unit.T, dim=1)
    weights = probabilities.mean(dim=0)
    return weights, (weights[:, None] * values).sum(dim=0, keepdim=True)


def native_query(*parts):
    return torch.stack([part.squeeze(0) for part in parts]).mean(dim=0, keepdim=True)


def assert_content_queries(observed):
    previous = []
    for block, current_text in zip(observed.selections, observed.text_rows[0]):
        parts = [observed.intents[0]]
        if len(current_text):
            parts.append(current_text.mean(dim=0, keepdim=True))
        if len(current_text) and previous:
            history_rows = torch.cat(previous)
            if len(history_rows):
                _, summary = independent_summary(current_text, history_rows)
                parts.append(summary.to(current_text.dtype))
        expected = native_query(*parts)
        assert block.query.dtype == expected.dtype
        # Dense stable softmax and the pinned chunked exp/sum differ by FP32
        # rounding. Native BF16 parts must still match the oracle bit-for-bit.
        torch.testing.assert_close(
            block.query, expected,
            rtol=5e-7 if expected.dtype == torch.float32 else 0,
            atol=2e-6 if expected.dtype == torch.float32 else 0,
        )
        previous.append(block.selected)


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_first_block_preserves_historical_header_global_intent_and_native_mean(
    historical_builder_class, bf16
):
    old_teacher = BFloat16QwenTeacher() if bf16 else None
    new_teacher = BFloat16QwenTeacher() if bf16 else None
    if bf16:
        # Non-integral values expose native BF16 inner/outer rounding and dtype.
        with torch.no_grad():
            for teacher in (old_teacher, new_teacher):
                teacher.embedding.weight[50] = torch.tensor([1.015625, -0.030029296875])
                teacher.embedding.weight[20] = torch.tensor([2.015625, 0.2021484375])
    old = run(historical_builder_class, teacher=old_teacher)
    new = run(Stage1TeacherTargetBuilder, teacher=new_teacher)
    assert old.builder.text_spans == new.builder.text_spans == [([3, 4], [9], [13])]
    assert new.builder.text_rows[0][0].shape == (2, 2)  # Includes assistant header.
    torch.testing.assert_close(old.builder.intents[0], new.builder.intents[0], rtol=0, atol=0)
    for old_mean, new_mean in zip(old.builder.text_means[0], new.builder.text_means[0]):
        torch.testing.assert_close(old_mean, new_mean, rtol=0, atol=0)
    first = new.builder.selections[0]
    torch.testing.assert_close(first.query, old.builder.selections[0].query, rtol=0, atol=0)
    torch.testing.assert_close(
        first.query,
        native_query(new.builder.intents[0], new.builder.text_means[0][0]),
        rtol=0,
        atol=0,
    )
    assert first.query.dtype == (torch.bfloat16 if bf16 else torch.float32)
    assert first.indices == old.builder.selections[0].indices == (0, 1)
    assert_content_queries(new.builder)


def test_three_blocks_substitute_only_all_prior_raw_visual_content(
    historical_builder_class,
):
    old = run(historical_builder_class)
    new = run(Stage1TeacherTargetBuilder)
    torch.testing.assert_close(new.builder.intents[0], torch.tensor([[0.25, 1.25]]))
    assert old.builder.text_spans == new.builder.text_spans
    for old_block, new_block in zip(old.builder.selections, new.builder.selections):
        torch.testing.assert_close(old_block.candidates, new_block.candidates, rtol=0, atol=0)
        assert old_block.max_count == new_block.max_count
    assert old.result.selected_candidate_indices == (((0, 1), (1,), (0, 1)),)
    assert new.result.selected_candidate_indices == (((0, 1), (0,), (0, 1)),)
    assert new.result.selected_counts == old.result.selected_counts == ((2, 1, 2),)

    # The old runtime really retains just the last nonempty selection mean.
    for index in (1, 2):
        torch.testing.assert_close(
            old.builder.selections[index].query,
            native_query(
                old.builder.intents[0],
                old.builder.text_means[0][index],
                old.builder.selections[index - 1].selected.mean(dim=0, keepdim=True),
            ),
            rtol=0,
            atol=0,
        )
    assert_content_queries(new.builder)

    history = torch.cat([block.selected for block in new.builder.selections[:2]])
    assert history.tolist() == [[12.0, 0.0], [0.0, 2.0], [0.0, 3.0]]
    weights, summary = independent_summary(new.builder.text_rows[0][2], history)
    assert weights[0] > weights[1] and weights[1] == weights[2]
    assert not torch.allclose(summary, history.mean(dim=0, keepdim=True))
    assert not torch.allclose(summary, new.builder.selections[1].selected)
    equal_block_mean = torch.stack(
        [block.selected.mean(dim=0) for block in new.builder.selections[:2]]
    ).mean(dim=0, keepdim=True)
    assert not torch.allclose(summary, equal_block_mean)
    assert not torch.equal(old.builder.selections[1].query, new.builder.selections[1].query)
    assert not torch.equal(old.builder.selections[2].query, new.builder.selections[2].query)
    for case in (old.case, new.case):
        assert case.teacher.language_forward_count == 1
        assert case.teacher.original_visual_call_count == 1
        assert case.teacher.helper_visual_call_count == 1
        assert case.teacher.language_batch_sizes == [1]
        assert case.teacher.language_sequence_lengths == [19]
        assert not case.teacher.training
    for key in old.case.inputs:
        torch.testing.assert_close(old.case.inputs[key], new.case.inputs[key], rtol=0, atol=0)


def test_later_selection_depends_on_current_text_content():
    y_text = run(Stage1TeacherTargetBuilder)
    x_sample = replace(
        THREE_BLOCK,
        input_ids=THREE_BLOCK.input_ids[:9] + (20,) + THREE_BLOCK.input_ids[10:],
    )
    x_text = run(Stage1TeacherTargetBuilder, (x_sample,))
    torch.testing.assert_close(
        y_text.builder.selections[0].query, x_text.builder.selections[0].query,
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        y_text.builder.selections[1].candidates, x_text.builder.selections[1].candidates,
        rtol=0, atol=0,
    )
    assert y_text.builder.selections[1].indices == (0,)
    assert x_text.builder.selections[1].indices == (1,)
    assert_content_queries(y_text.builder)
    assert_content_queries(x_text.builder)


def test_history_weights_average_each_current_text_row_before_value_aggregation():
    sample = sample_with_blocks(texts=((20,), (21, 22), (22,)))
    new = run(Stage1TeacherTargetBuilder, (sample,))
    text = new.builder.text_rows[0][1]
    history = new.builder.selections[0].selected
    weights, summary = independent_summary(text, history)
    collapsed_weights, collapsed_summary = independent_summary(
        text.mean(dim=0, keepdim=True), history
    )
    assert text.tolist() == [[0.0, 2.0], [1.0, -1.0]]
    assert not torch.allclose(weights, collapsed_weights)
    assert not torch.allclose(summary, collapsed_summary)
    assert_content_queries(new.builder)


def test_missing_current_text_omits_history_instead_of_reusing_visual_mean(
    historical_builder_class,
):
    sample = sample_with_blocks(texts=((20,), (), (22,)))
    old = run(historical_builder_class, (sample,))
    new = run(Stage1TeacherTargetBuilder, (sample,))
    assert new.builder.text_rows[0][1].shape == (0, 2)
    torch.testing.assert_close(
        new.builder.selections[1].query, new.builder.intents[0], rtol=0, atol=0,
    )
    assert not torch.equal(old.builder.selections[1].query, new.builder.selections[1].query)
    assert_content_queries(new.builder)


def test_no_selected_history_omits_visual_term(historical_builder_class):
    sample = sample_with_blocks(helpers=((), *THREE_BLOCK.helper_features[1:]))
    old = run(historical_builder_class, (sample,))
    new = run(Stage1TeacherTargetBuilder, (sample,))
    assert new.result.selected_counts[0][0] == 0
    torch.testing.assert_close(
        new.builder.selections[1].query,
        native_query(new.builder.intents[0], new.builder.text_means[0][1]),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        new.builder.selections[1].query, old.builder.selections[1].query,
        rtol=0, atol=0,
    )
    assert_content_queries(new.builder)


def test_empty_intermediate_selection_keeps_earlier_raw_history():
    sample = sample_with_blocks(
        helpers=(THREE_BLOCK.helper_features[0], (), THREE_BLOCK.helper_features[2])
    )
    new = run(Stage1TeacherTargetBuilder, (sample,))
    assert new.result.selected_counts == ((2, 0, 2),)
    assert new.builder.selections[1].selected.shape == (0, 2)
    assert_content_queries(new.builder)


def test_current_and_future_helper_values_and_future_text_cannot_change_current_query():
    baseline = run(Stage1TeacherTargetBuilder)
    changed = replace(
        THREE_BLOCK,
        input_ids=THREE_BLOCK.input_ids[:13] + (30,) + THREE_BLOCK.input_ids[14:],
        helper_features=(
            THREE_BLOCK.helper_features[0],
            ((0.0, 300.0), (700.0, 0.0), (-2.0, -1.0)),
            ((-300.0, 17.0), (80.0, 90.0)),
        ),
    )
    perturbed = run(Stage1TeacherTargetBuilder, (changed,))
    for index in (0, 1):
        torch.testing.assert_close(
            baseline.builder.selections[index].query,
            perturbed.builder.selections[index].query,
            rtol=0, atol=0,
        )
    assert not torch.equal(
        baseline.builder.selections[1].candidates, perturbed.builder.selections[1].candidates
    )
    assert_content_queries(baseline.builder)
    assert_content_queries(perturbed.builder)


def test_history_resets_between_samples_and_builder_calls():
    builder = observed_builder(Stage1TeacherTargetBuilder)
    batch = run(Stage1TeacherTargetBuilder, (THREE_BLOCK, ONE_BLOCK), builder=builder)
    first_count = sum(batch.result.selected_counts[0])
    independent = run(Stage1TeacherTargetBuilder, (ONE_BLOCK,))
    torch.testing.assert_close(
        builder.selections[3].query, independent.builder.selections[0].query,
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        batch.result.latent_values[first_count:], independent.result.latent_values,
        rtol=0, atol=0,
    )
    repeated = run(Stage1TeacherTargetBuilder, (ONE_BLOCK,), builder=builder)
    torch.testing.assert_close(repeated.result.latent_values, independent.result.latent_values)
    torch.testing.assert_close(
        builder.selections[-1].query, independent.builder.selections[0].query,
        rtol=0, atol=0,
    )
    assert repeated.result.selected_candidate_indices == independent.result.selected_candidate_indices


def test_packing_keeps_raw_candidates_stable_ties_and_first_k_masks():
    new = run(Stage1TeacherTargetBuilder)
    expected = torch.tensor([[12.0, 0.0], [0.0, 2.0], [0.0, 3.0], [3.0, 0.0], [3.0, 0.0]])
    torch.testing.assert_close(new.result.latent_values, expected, rtol=0, atol=0)
    assert new.result.latent_mask.nonzero(as_tuple=False).tolist() == [
        [0, 6], [0, 7], [0, 11], [0, 15], [0, 16],
    ]
    assert new.builder.selections[2].indices == (0, 1)
    for block in new.builder.selections:
        torch.testing.assert_close(
            block.selected, block.candidates[list(block.indices)], rtol=0, atol=0,
        )
    tie = run(
        Stage1TeacherTargetBuilder, (TIED_CANDIDATES,),
        builder=observed_builder(Stage1TeacherTargetBuilder, coverage_p=0.3),
    )
    assert tie.result.selected_candidate_indices == (((0,),),)
    assert tie.result.selection_margins == ((0.0,),)
    assert tie.result.latent_mask.nonzero(as_tuple=False).tolist() == [[0, 6]]


@pytest.mark.parametrize("magnitude", [0.0, 1e-9], ids=["zero", "near_zero"])
@pytest.mark.parametrize("bad_role", ["text", "values"])
def test_present_history_rejects_zero_or_near_zero_rows_before_next_selection(
    magnitude, bad_role,
):
    sample = THREE_BLOCK
    teacher = make_case((sample,)).teacher
    if bad_role == "text":
        with torch.no_grad():
            teacher.embedding.weight[21] = torch.tensor([magnitude, 0.0])
    else:
        sample = replace(
            sample,
            helper_features=(
                ((12.0, 0.0), (magnitude, 0.0)),
                *sample.helper_features[1:],
            ),
        )
    builder = observed_builder(Stage1TeacherTargetBuilder)
    with pytest.raises(
        FloatingPointError,
        match=rf"{bad_role} row norms must be finite and greater than 1e-8",
    ):
        run(Stage1TeacherTargetBuilder, (sample,), teacher=teacher, builder=builder)
    assert len(builder.selections) == 1

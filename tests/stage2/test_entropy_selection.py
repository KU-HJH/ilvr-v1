"""Step0 high-entropy token selection and the token-level Vision gain objective."""

from __future__ import annotations

import hashlib
import json
import math
import random
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from peft_common import resolve_qwen2_5_vl_model
from stage2.entropy_selection import (
    SELECTION_FORMAT,
    VisionTokenSelectionTable,
    build_vision_token_units,
    candidate_target_positions,
    full_vocabulary_entropy,
    input_ids_key,
    merge_selection_shards,
    select_top_fraction,
    step0_selection_for_batch,
)
from stage2.objective import Stage2ObjectiveConfig
from stage2.objective_runtime import Stage2ObjectiveRuntime
from stage2.packed_m import _packed_layout
from stage2.rng import capture_rng_state, replay_rng_state

from .packed_m_helpers import TEXT_VOCABULARY, TOKENS, make_packed_inputs, make_packed_model

# make_packed_inputs(blocks=3):
#   0..4 = [81 80 80 7 8] | 5..8 block0 [91 90 90 92] | 9..11 = [9 10 11]
#   12..15 block1 | 16..18 = [12 13 14] | 19..22 block2 | 23..25 = [15 16 17]
SPANS = ((5, 8), (12, 15), (19, 22))


def _reference_top_fraction(positions, entropies, fraction):
    count = math.ceil(fraction * len(positions))
    order = sorted(range(len(positions)), key=lambda i: (-entropies[i], positions[i]))
    return tuple(sorted(positions[i] for i in order[:count]))


def test_top_fraction_matches_reference_with_ceil_and_position_ties():
    """Catches floor rounding, unstable tie breaking, or returning unsorted positions."""
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 40)
        positions = sorted(rng.sample(range(1, 500), n))
        entropies = [rng.choice([0.5, 1.0, 1.5, rng.random() * 3]) for _ in range(n)]
        assert select_top_fraction(positions, entropies, 0.2) == _reference_top_fraction(positions, entropies, 0.2)
    assert select_top_fraction([3, 4, 5, 6, 7, 8], [1.0] * 6, 0.2) == (3, 4)   # ceil(1.2)=2, earliest ties
    assert select_top_fraction([], [], 0.2) == ()


def test_candidates_are_supervised_text_targets_only():
    """Catches latent slots, markers, image tokens, or unsupervised rows being ranked."""
    inputs = make_packed_inputs(blocks=3)
    inputs["labels"][0, 3] = -100   # an unsupervised prompt-like text token
    candidates = candidate_target_positions(
        inputs["input_ids"][0], inputs["labels"][0], inputs["attention_mask"][0], TEXT_VOCABULARY.token_ids)
    assert candidates == (4, 9, 10, 11, 16, 17, 18, 23, 24, 25)


def test_step0_selection_uses_full_vocabulary_entropy_at_target_minus_one():
    """Catches entropy read at the target row itself or over a renormalized vocabulary."""
    model = make_packed_model(tuning="full").eval()
    inputs = make_packed_inputs(blocks=3)
    record = step0_selection_for_batch(model, inputs, text_token_ids=TEXT_VOCABULARY.token_ids, fraction=0.2)
    with torch.no_grad():
        logits = model(**{k: v for k, v in inputs.items() if k not in ("labels", "stage2_text_segments")}).logits[0]
    candidates = (3, 4, 9, 10, 11, 16, 17, 18, 23, 24, 25)
    entropy = [float(full_vocabulary_entropy(logits[p - 1])) for p in candidates]
    assert record["candidates"] == 11
    assert tuple(record["selected_targets"]) == _reference_top_fraction(candidates, entropy, 0.2)
    assert len(record["selected_targets"]) == 3


def _units(scope, targets, inputs=None):
    inputs = inputs or make_packed_inputs(blocks=3)
    return build_vision_token_units(
        inputs["input_ids"], inputs["labels"], inputs["attention_mask"], inputs["stage2_text_segments"],
        token_ids=TOKENS, latent_size=2, selected_targets=targets, scope=scope)


def test_units_drop_pre_first_block_targets_and_group_by_scope():
    """Catches pre-block tokens entering the loss or cumulative hiding only the nearest block."""
    targets = (4, 9, 11, 17, 24)
    independent = _units("independent", targets)
    assert independent.excluded == {"pre_first_block_tokens": 1}
    assert [(b.start_position, b.end_position, b.post_positions, b.ablation) for b in independent.blocks] == [
        (5, 8, (8, 10), ((5, 8),)),
        (12, 15, (16,), ((12, 15),)),
        (19, 22, (23,), ((19, 22),)),
    ]
    cumulative = _units("cumulative", targets)
    assert cumulative.excluded == {"pre_first_block_tokens": 1}
    (unit,) = cumulative.blocks
    assert (unit.start_position, unit.end_position) == (5, 8)
    assert unit.post_positions == (8, 10, 16, 23)
    assert unit.target_ids == (9, 11, 13, 16)
    assert unit.ablation == SPANS
    assert _units("independent", (3, 4)).blocks == ()


def _visible(scope, targets):
    inputs = make_packed_inputs(blocks=3)
    capture = SimpleNamespace(root_length=inputs["input_ids"].shape[1], _attention_mask=inputs["attention_mask"])
    blocks = _units(scope, targets, inputs).blocks
    rows, visible, _ = _packed_layout(inputs, blocks, capture)
    return rows.tolist(), visible, capture.root_length


def _logical_visibility(rows, visible, root_length):
    """Map packed visibility back to (query position -> visible key positions)."""
    result = []
    for index, query in enumerate(rows):
        keys = {key for key in range(root_length) if visible[index, key]}
        branch_start = index
        while branch_start > 0 and rows[branch_start - 1] == rows[branch_start] - 1:
            branch_start -= 1
        keys |= {rows[column - root_length] for column in range(root_length, visible.shape[1])
                 if visible[index, column] and branch_start <= column - root_length <= index}
        result.append((query, keys))
    return result


def test_packed_masks_hide_exactly_block_k_or_blocks_one_to_k_with_markers_visible():
    """Catches independent leaking other hides, or cumulative still seeing an earlier block."""
    interiors = {span: set(range(span[0] + 1, span[1])) for span in SPANS}

    rows, visible, root = _visible("independent", (10, 17))
    for query, keys in _logical_visibility(rows, visible, root):
        own = SPANS[0] if query < 12 else SPANS[1]
        assert keys == set(range(query + 1)) - interiors[own], query
    assert rows == [8, 9, 15, 16]

    rows, visible, root = _visible("cumulative", (10, 17, 25))
    assert rows == list(range(8, 25))
    for query, keys in _logical_visibility(rows, visible, root):
        hidden = set().union(*(interiors[span] for span in SPANS if query >= span[1]))
        assert keys == set(range(query + 1)) - hidden, query
        assert {5, 8} <= keys and (query < 15 or {12, 15} <= keys)


def test_decoder_mask_accepts_multiple_hidden_spans():
    """Catches the full-replay oracle hiding only one span or hiding markers."""
    model = make_packed_model(tuning="full").eval()
    decoder = resolve_qwen2_5_vl_model(model).model
    length = 26
    mask = decoder._update_causal_mask(
        torch.ones(1, length, dtype=torch.long), torch.zeros(1, length, 8),
        torch.arange(length), None, False, stage2_ablation=SPANS)
    blocked = mask[0, 0] == torch.finfo(mask.dtype).min
    for query in range(length):
        hidden = {key for start, end in SPANS if query >= end for key in range(start + 1, end)}
        expected = {key for key in range(length) if key > query} | hidden
        assert {key for key in range(length) if blocked[query, key]} == expected, query


def _table_file(tmp_path, inputs, targets):
    payload = {"format": SELECTION_FORMAT, "meta": {"fraction": 0.2},
               "rows": {input_ids_key(inputs["input_ids"][0]): {"selected_targets": list(targets)}}}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _config(path, sha, *, scope, mode, lambda_vis=1.0):
    return Stage2ObjectiveConfig(
        enable_stage2_objective=True, lambda_vis=lambda_vis, m_V=0.5, stage2_m_execution=mode,
        vision_token_reduction="token_loss_mean", vision_token_selection="step0_entropy",
        vision_selection_path=path, vision_selection_sha256=sha, vision_mask_scope=scope)


def _run(model, inputs, config):
    runtime = Stage2ObjectiveRuntime(config, TOKENS, 2, TEXT_VOCABULARY, None)
    selection = runtime.select(inputs)
    lm_head = resolve_qwen2_5_vl_model(model).lm_head
    outputs = []
    handle = lm_head.register_forward_hook(lambda _m, _a, value: outputs.append(value.detach().clone()))
    try:
        result = runtime.forward(model, inputs, selection)
    finally:
        handle.remove()
    return runtime, selection, result, outputs


@pytest.mark.parametrize("scope", ["independent", "cumulative"])
def test_token_loss_is_global_token_mean_of_softplus_with_stopgrad_m(tmp_path, scope):
    """Catches block-level averaging, a missing stopgrad on S_M, or the wrong M oracle."""
    torch.manual_seed(401)
    inputs = make_packed_inputs(blocks=3)
    path, sha = _table_file(tmp_path, inputs, (4, 9, 11, 17, 24, 25))
    model = make_packed_model()
    with torch.no_grad():
        resolve_qwen2_5_vl_model(model).lm_head.weight.mul_(50.0)
    runtime, selection, result, lm_outputs = _run(model, inputs, _config(path, sha, scope=scope, mode="full_replay"))
    assert runtime.units(selection) == result.count == 5
    assert result.metric_sums["excluded_pre_first_block_tokens"] == 1

    expected = torch.zeros(())
    for index, block in enumerate(selection.blocks):
        targets = torch.tensor(block.target_ids).unsqueeze(-1)
        f_logits = result.outputs.logits[0, list(block.post_positions)]
        m_logits = lm_outputs[1 + index][0, list(block.post_positions)]
        s_f = torch.log_softmax(f_logits.float(), -1).gather(-1, targets).squeeze(-1)
        s_m = torch.log_softmax(m_logits.float(), -1).gather(-1, targets).squeeze(-1)
        expected = expected + F.softplus(0.5 - (s_f - s_m)).sum()
    torch.testing.assert_close(result.vis_sum / result.count, expected / 5, atol=1e-6, rtol=1e-5)
    assert len(lm_outputs) == 1 + (3 if scope == "independent" else 1)

    grads = torch.autograd.grad(result.vis_sum, [p for p in model.parameters() if p.requires_grad],
                                allow_unused=True, retain_graph=True)
    manual = torch.autograd.grad(expected, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
    for got, want in zip(grads, manual):
        if want is None:
            assert got is None or not got.abs().any()
        else:
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("scope", ["independent", "cumulative"])
def test_packed_suffix_matches_full_replay_for_both_scopes(tmp_path, scope):
    """Catches the packed multi-span layout diverging from the full-replay M oracle."""
    torch.manual_seed(409)
    inputs = make_packed_inputs(blocks=3)
    path, sha = _table_file(tmp_path, inputs, (9, 11, 17, 18, 24, 25))
    initial = make_packed_model()
    rng = capture_rng_state()
    results = {}
    for mode in ("full_replay", "packed_suffix"):
        model = deepcopy(initial)
        with replay_rng_state(rng):
            _, _, result, _ = _run(model, inputs, _config(path, sha, scope=scope, mode=mode))
        result.vis_sum.backward()
        results[mode] = (result, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    (full, full_grads), (packed, packed_grads) = results["full_replay"], results["packed_suffix"]
    torch.testing.assert_close(packed.vis_sum, full.vis_sum, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(packed.metric_sums["S_M"], full.metric_sums["S_M"], atol=1e-6, rtol=1e-5)
    assert packed.metric_sums["decoder_forwards_M"] == 1
    assert full_grads.keys() == packed_grads.keys() and full_grads
    for name, value in packed_grads.items():
        torch.testing.assert_close(value, full_grads[name], atol=1e-6, rtol=1e-5)


def test_cumulative_and_independent_differ_after_the_second_block(tmp_path):
    """Catches cumulative silently reusing the independent (nearest-block) mask."""
    torch.manual_seed(419)
    inputs = make_packed_inputs(blocks=3)
    path, sha = _table_file(tmp_path, inputs, (17, 24))
    model = make_packed_model(tuning="full").eval()
    with torch.no_grad():
        resolve_qwen2_5_vl_model(model).lm_head.weight.mul_(50.0)
    scores = {}
    for scope in ("independent", "cumulative"):
        with torch.no_grad():
            _, _, result, _ = _run(model, inputs, _config(path, sha, scope=scope, mode="full_replay"))
        scores[scope] = float(result.metric_sums["S_M"])
    assert abs(scores["independent"] - scores["cumulative"]) > 1e-4


def test_selection_lookup_fails_closed(tmp_path):
    """Catches training on an unselected sample or a swapped selection file."""
    inputs = make_packed_inputs(blocks=3)
    path, sha = _table_file(tmp_path, inputs, (9,))
    with pytest.raises(ValueError, match="vision_selection_sha256"):
        Stage2ObjectiveRuntime(_config(path, "0" * 64, scope="independent", mode="full_replay"),
                               TOKENS, 2, TEXT_VOCABULARY, None)
    runtime = Stage2ObjectiveRuntime(_config(path, sha, scope="independent", mode="full_replay"),
                                     TOKENS, 2, TEXT_VOCABULARY, None)
    with pytest.raises(KeyError, match="no step0 selection"):
        runtime.select(make_packed_inputs(blocks=2))
    table = VisionTokenSelectionTable.load(path)
    assert table.selected_targets(inputs["input_ids"][0]) == (9,)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lambda_dyn": 1.0, "reference_model_path": "x"}, "lambda_dyn=0"),
        ({"vision_token_reduction": "score_mean"}, "token_loss_mean"),
        ({"vision_selection_sha256": None}, "vision_selection_sha256"),
        ({"vision_token_selection": "post_window"}, "require step0_entropy"),
        ({"vision_mask_scope": "nearest"}, "vision_mask_scope"),
    ],
)
def test_config_rejects_inconsistent_token_selection(changes, message):
    """Catches token-level selection running with entropy terms, block reduction, or no pinned file."""
    from dataclasses import replace

    config = replace(_config("sel.json", "a" * 64, scope="cumulative", mode="packed_suffix"), **changes)
    with pytest.raises(ValueError, match=message):
        config.validate("stage2", True)


def test_merge_requires_complete_agreeing_shards(tmp_path):
    """Catches a missing shard or conflicting selections being merged silently."""
    def shard(index, rows):
        path = tmp_path / f"shard{index}.json"
        path.write_text(json.dumps({"format": SELECTION_FORMAT,
                                    "meta": {"fraction": 0.2, "num_shards": 2, "shard_index": index},
                                    "rows": rows}))
        return path

    a = shard(0, {"k1": {"selected_targets": [3], "row_ids": [0]}})
    b = shard(1, {"k1": {"selected_targets": [3], "row_ids": [5]}, "k2": {"selected_targets": [4], "row_ids": [1]}})
    merged = merge_selection_shards([a, b], tmp_path / "merged.json")
    assert merged["rows"]["k1"]["row_ids"] == [0, 5] and set(merged["rows"]) == {"k1", "k2"}
    with pytest.raises(ValueError, match="incomplete"):
        merge_selection_shards([a], tmp_path / "partial.json")
    c = shard(1, {"k1": {"selected_targets": [9], "row_ids": [5]}})
    with pytest.raises(ValueError, match="disagree"):
        merge_selection_shards([a, c], tmp_path / "conflict.json")


@pytest.mark.parametrize("scope", ["independent", "cumulative"])
def test_token_probe_equals_eval_mode_full_replay_objective(tmp_path, scope):
    """Catches the probe scoring other tokens, another mask, or averaging per block."""
    from stage2.vision_gain_probe import probe_token_batches

    inputs = make_packed_inputs(blocks=3)
    path, sha = _table_file(tmp_path, inputs, (4, 9, 11, 17, 24, 25))
    model = make_packed_model().eval()
    report = probe_token_batches(model, [inputs], selection_table=VisionTokenSelectionTable.load(path),
                                 scope=scope, token_ids=TOKENS, latent_size=2)
    with torch.no_grad():
        _, _, result, _ = _run(model, inputs, _config(path, sha, scope=scope, mode="full_replay"))
    assert report["tokens"] == result.count == 5
    assert report["excluded_pre_first_block_tokens"] == 1
    assert report["S_F"] == pytest.approx(float(result.metric_sums["S_F"]) / 5, abs=1e-6)
    assert report["S_M"] == pytest.approx(float(result.metric_sums["S_M"]) / 5, abs=1e-6)
    assert report["G_gt_0"] == pytest.approx(float(result.metric_sums["G_gt_0"]) / 5)
    assert [row["target"] for row in report["per_token"]] == [9, 11, 17, 24, 25]

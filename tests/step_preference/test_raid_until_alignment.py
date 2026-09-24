"""Compare explicit until targets with complete latent-delimited text."""

import ast
import dataclasses
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from stage2.data import Stage2DataCollator, prepare_stage2_example
from step_preference.sidecar import build_context_sidecar_plan
from step_preference.spans import (
    ASSISTANT_PLAN_RENDERING_CONCATENATED,
    ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
)
from tests.step_preference.test_spans import STAGE2A_CHECKPOINT
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentSpan, LatentTokenIds, parse_latent_spans,
)


REFERENCE_ROOT = Path("/raid/workspace/temp2/ILVR/.worktrees/ours1/src")


def read_pure_definitions(path, names):
    raw = Path(path).read_bytes()
    tree = ast.parse(raw.decode())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    namespace = {"dataclass": dataclasses.dataclass, "Mapping": Mapping,
                 "Sequence": Sequence, "LatentSpan": LatentSpan}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    print({"reference": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return namespace


def _token_ids(tokenizer):
    return LatentTokenIds(**{
        f"{name}_token_id": tokenizer.convert_tokens_to_ids(f"<|latent_{name}|>")
        for name in ("pad", "start", "end")
    })


def test_until_keeps_join_token_and_final_answer(tmp_path):
    Image.new("RGB", (56, 56)).save(tmp_path / "input.png")
    plan = [
        {"type": "text", "content": "Before."},
        {"type": "latent", "helper_image": "a.png"},
        {"type": "text", "content": "hel"},
        {"type": "text", "content": "lo"},
        {"type": "latent", "helper_image": "b.png"},
        {"type": "text", "content": "Reason.\n"},
        {"type": "text", "content": "The final answer is: 70"},
    ]
    example = prepare_stage2_example(
        {"image_input": "input.png", "text_input": "question",
         "sequence_plan": plan},
        image_root=tmp_path, latent_size=2, require_reasoning_units=True,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_CONCATENATED,
    )
    processor = AutoProcessor.from_pretrained(STAGE2A_CHECKPOINT, local_files_only=True)
    tok = processor.tokenizer
    token_ids = _token_ids(tok)
    offsets = tok("hello", add_special_tokens=False, return_offsets_mapping=True)
    assert any(start < 3 < end for start, end in offsets["offset_mapping"])
    batch = Stage2DataCollator(
        processor, token_ids, 2, True, step_preference_enabled=True,
        preference_target_scope="until_next_latent",
    )([example])
    ids = batch["input_ids"][0]
    spans = batch["reasoning_step_spans"]
    actual = [tok.decode(ids[s.target_text_start:s.target_text_end]) for s in spans]
    assert actual == ["hello", "Reason.\nThe final answer is: 70"]
    assert all(s.target_text_start == s.latent_end + 1 for s in spans)
    assert spans[0].target_text_end == spans[1].latent_start - 1
    assert tok.convert_tokens_to_ids("<|im_end|>") not in ids[
        spans[-1].target_text_start:spans[-1].target_text_end
    ].tolist()
    assert all(ids[s.target_text_start:s.target_text_end].numel() > 0 for s in spans)


@pytest.mark.parametrize("plan,expected", [
    (["Before.", "L", "hel", "lo"], ["hello"]),
    (["Before.", "L", " one \n", "two"], [" one \ntwo"]),
    (["Before.", "L", "The final answer is: 70"], ["The final answer is: 70"]),
    (["Before.", "L", "hel", "lo", "L", "Reason.\n",
      "The final answer is: 70"], ["hello", "Reason.\nThe final answer is: 70"]),
])
def test_until_matches_preserved_raid_token_intervals(tmp_path, plan, expected):
    """Catches target, predictor, text, or input drift from preserved RAID definitions."""
    reference_data = read_pure_definitions(REFERENCE_ROOT / "stage2/data.py", ["_assistant_text"])
    reference_spans = read_pure_definitions(
        REFERENCE_ROOT / "step_preference/spans.py",
        ["ReasoningStepSpan", "_latent_plan_entries", "derive_reasoning_step_spans"],
    )
    sequence_plan = [
        {"type": "latent", "helper_image": f"helper-{i}.png"}
        if part == "L" else {"type": "text", "content": part}
        for i, part in enumerate(plan)
    ]
    Image.new("RGB", (56, 56)).save(tmp_path / "input.png")
    sample = {"image_input": "input.png", "text_input": "question",
              "sequence_plan": sequence_plan}
    example = prepare_stage2_example(
        sample, image_root=tmp_path, latent_size=2, require_reasoning_units=True,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_CONCATENATED,
    )
    ref_assistant, ref_count = reference_data["_assistant_text"](sequence_plan, 2)
    assert ref_count == len(expected)
    assert example.conversation[1]["content"][0]["text"] == ref_assistant
    processor = AutoProcessor.from_pretrained(STAGE2A_CHECKPOINT, local_files_only=True)
    tokenizer = processor.tokenizer
    token_ids = _token_ids(tokenizer)
    local = Stage2DataCollator(
        processor, token_ids, 2, True, step_preference_enabled=True,
        preference_target_scope="until_next_latent",
    )([example])
    ref_conversation = [example.conversation[0],
                        {"role": "assistant", "content": [{"type": "text", "text": ref_assistant}]}]
    ref_rendered = processor.apply_chat_template(
        ref_conversation, tokenize=False, add_generation_prompt=False,
    )
    with Image.open(tmp_path / "input.png") as source:
        ref_batch = processor(
            text=[ref_rendered], images=[[source.convert("RGB")]], padding=True,
            truncation=False, return_tensors="pt",
        )
    assert torch.equal(local["input_ids"], ref_batch["input_ids"])
    ids = ref_batch["input_ids"][0]
    assistant_marker = tokenizer(
        "<|im_start|>assistant", add_special_tokens=False, return_tensors="pt",
    )["input_ids"][0]
    starts = [i + assistant_marker.numel() for i in range(ids.numel() - assistant_marker.numel() + 1)
              if torch.equal(ids[i:i + assistant_marker.numel()], assistant_marker)]
    assert len(starts) == 1
    assistant_begin = starts[0]
    terminator = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = [i for i in range(assistant_begin, ids.numel()) if ids[i].item() == terminator]
    assert len(ends) == 1
    assistant_end = ends[0]
    reference = reference_spans["derive_reasoning_step_spans"](
        sequence_plan=sequence_plan,
        assistant_content_start=assistant_begin,
        assistant_content_end=assistant_end,
        latent_spans=parse_latent_spans(ids, token_ids=token_ids, latent_size=2),
    )
    actual = local["reasoning_step_spans"]
    assert [(s.target_text_start, s.target_text_end) for s in actual] == [
        (s.target_text_start, s.target_text_end) for s in reference
    ]
    local_target_ids = [local["input_ids"][0, s.target_text_start:s.target_text_end].tolist()
                        for s in actual]
    reference_target_ids = [ids[s.target_text_start:s.target_text_end].tolist() for s in reference]
    assert local_target_ids == reference_target_ids
    assert [tokenizer.decode(target) for target in local_target_ids] == expected
    local_predictors = [list(range(s.target_text_start - 1, s.target_text_end - 1))
                        for s in actual]
    reference_predictors = [list(range(s.target_text_start - 1, s.target_text_end - 1))
                            for s in reference]
    assert local_predictors == reference_predictors
    sidecar = build_context_sidecar_plan(
        clean_input_ids=local["input_ids"], attention_mask=local["attention_mask"],
        spans=actual, visual_only=True, suppress_latent_boundaries=True,
    )
    assert list(sidecar.root_selected_positions) == [
        position for span in reference
        for position in range(span.target_text_start - 1, span.target_text_end - 1)
    ]


def test_explicit_immediate_keeps_adjacent_space_lf_targets(tmp_path):
    """Catches until interval logic taking over an explicit immediate scope."""
    Image.new("RGB", (56, 56)).save(tmp_path / "input.png")
    example = prepare_stage2_example(
        {"image_input": "input.png", "text_input": "question", "sequence_plan": [
            {"type": "text", "content": "Before."},
            {"type": "latent", "helper_image": "helper.png"},
            {"type": "text", "content": "Reason."},
            {"type": "text", "content": "Final answer: 70"},
        ]}, image_root=tmp_path, latent_size=2, require_reasoning_units=True,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    )
    processor = AutoProcessor.from_pretrained(STAGE2A_CHECKPOINT, local_files_only=True)
    batch = Stage2DataCollator(
        processor, _token_ids(processor.tokenizer), 2, True,
        step_preference_enabled=True, preference_target_scope="immediate_text",
    )([example])
    span = batch["reasoning_step_spans"][0]
    assert processor.tokenizer.decode(
        batch["input_ids"][0, span.target_text_start:span.target_text_end]
    ) == "Reason."

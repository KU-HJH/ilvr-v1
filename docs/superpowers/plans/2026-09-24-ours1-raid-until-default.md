# Ours1 RAID Until Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans according to the user's selected execution method. Both Subagent-driven and Native are available. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match ours1's training text, preference targets, and activation checkpoint placement to the preserved RAID until-next-latent evidence, and make `until_next_latent` the fresh context-preference default.

**Architecture:** Keep the existing trainer, loss reducer, sidecar planner, and differentiable explicit-KV interface. Use latent-delimited token mapping for until targets, selectively checkpoint cache-free latent recurrence, and resolve mode-dependent defaults before preparing data or Trainer arguments. Reuse only relevant code from existing alignment worktrees.

**Tech Stack:** Repository Python 3.11 venv, PyTorch, patched Transformers Qwen2.5-VL, PEFT/DoRA, TRL/Trainer, Bash, pytest; no dependency changes.

**Spec:** `docs/superpowers/specs/2026-09-24-ours1-raid-until-default-design.md`; behavior approved at `87741bb`, execution-routing restriction removed by subsequent user instruction. Current SHA256 `f35c68216781e80b1ea4b9f1b4289bf79944211aeea531fbd1e5f224cb5c0951`.

## Global Constraints

- 새 context-preference 학습의 target scope 기본값은 `until_next_latent`로 한다.
- text/text 사이에 공백이나 줄바꿈을 추가하지 않는다. 데이터에 원래 포함된 공백과 줄바꿈은 보존한다.
- 각 latent의 target은 반열린 token 구간 `[END + 1, next START)`다.
- 마지막 latent는 `[END + 1, assistant content end)`이며 최종 답변을 포함한다.
- until 구간 내부의 text/text 경계를 걸치는 토큰은 target에 포함한다.
- Native OFF; cache-free latent checkpoint ON; final clean OFF; V-off sidecar OFF.
- final clean에서 생성한 KV는 V-off의 prefix로 전달되고 gradient가 clean 경로까지 연결되어야 한다.
- 명시적 `immediate_text`와 기존 resume의 저장 설정은 우선한다.
- 기존 변경을 임의로 되돌리거나 함께 커밋하지 않는다.
- Python과 테스트는 repository의 `venv/ilvr-venv`를 사용한다.
- 새 preference 실험의 beta 0.5와 weight 1.0, 두 노드 네 rank 등의 운영 정책은 계속 적용된다.
- 이번 설계는 실제 학습이나 평가 실행을 요청하지 않는다.
- No sampler, optimizer, initialization, V-off boundary-mask policy, or whole-algorithm alignment changes. No automatic sidecar checkpoint re-enablement for memory.
- Preserve the historical-source limitation: `uncommitted_source: true`; current RAID code comparison is not proof of exact historical execution equivalence.

## Review Focus

1. A BPE token spanning adjacent text entries, including the entry immediately before the next latent, must stay in the until target and must not make the collator reject the sample. Task 1 pins this.
2. Image-token expansion, original whitespace, and the final chat terminator must not shift or leak into preference targets. Task 1 pins this with the real processor.
3. Trainer's native-checkpoint hook must not re-enable checkpointing after model setup; actual backward must recompute latent prefixes only. Tasks 2 and 3 pin this.
4. Turning off sidecar checkpointing must preserve live clean-KV gradients, including DoRA/dropout paths; tests must isolate V-off loss from CE. Task 2 pins this.
5. Omitted versus explicit options, dormant stages, and saved resume settings must remain distinguishable; an initializer must not be mistaken for a resume. Task 3 pins this.

## Execution boundaries and file map

The user invoked `writing-plans` after being asked to approve the written spec. Treat that as approval to write this plan, not approval to execute an unread plan. Next gate: user reviews this plan and selects Subagent-driven or Native. The later instruction “그 부분은 지워.” removes the AGENTS restriction that previously forced sub-agent implementation; it does not select a method.

The target is `/root/workspace/ILVR/.worktrees/ours1`. Its preexisting dirty tree is meaningful input. The root owns control artifacts and approval/evidence reconciliation. Under Subagent-driven, delegate implementation and fixes according to that skill; under Native, the root implements and fixes through executing-plans. Independent reviewers must be distinct from implementers in either method.

At execution, invoke `using-git-worktrees` before isolation. Capture the original tracked diff and the hashes of relevant untracked source/tests/documents in `results/raid-until-default-20260924/`; exclude caches. Create an owned worktree based on the approved ours1 HEAD, and have the implementer apply that captured input there. If a snapshot commit is needed, make it a separate, explicitly labeled baseline commit in the owned branch; do not combine it with deliverable commits or alter the original ours1 index. Record the baseline commit and compare deliverable commits against it. The final handoff must state that integrating this branch requires preserving the user's original dirty work; never merge its snapshot blindly.

Use the existing `data/`, `checkpoints/`, and repository venv links under the worktree rules. Keep logs under that worktree's `logs/raid-until-default-20260924/` and evidence under its `results/raid-until-default-20260924/`. Tests below run on CPU and do not launch ranks or training. Any later GPU training probe needs the existing two-node coordinator and the applicable launch gate.

| File | Responsibility | Task |
| --- | --- | --- |
| `src/step_preference/spans.py` | Until token-span mapping; immediate mapping preserved | 1 |
| `src/stage2/data.py` | Collator scope normalization and mapping integration | 1, 3 |
| `tests/step_preference/test_raid_until_alignment.py` | Independent RAID text/token comparison | 1 |
| `tests/step_preference/test_target_scope.py` | Real collator, CLI, assembly contracts | 1, 3 |
| Qwen `stage2_latent_rollout.py` | Selective latent checkpoint primitive | 2 |
| Qwen `modeling_qwen2_5_vl.py` | Pass checkpoint choice into rollout | 2 |
| Qwen `modular_qwen2_5_vl.py` | Keep corresponding authored definitions consistent where present | 2 |
| `src/stage2/trainer.py` | Explicit sidecar checkpoint choice | 2, 3 |
| `src/main.py` | Resolved rendering/checkpoint setup and Trainer arguments | 2, 3 |
| `src/step_preference/execution.py` (new) | Small pure resolver for rendering and checkpoint choices | 3 |
| `src/step_preference/config.py` | Omitted scope versus explicit scope | 3 |
| `src/utils_deepseed.py` | CLI defaults and latent checkpoint option | 3 |
| `run_stage2_training.sh` | Effective environment/CLI defaults | 3 |
| `src/multinode_runtime.py` | Forward the existing coordinator's new latent option | 3 |
| `tests/stage2/test_raid_until_checkpointing.py` (new) | Replay, dropout, KV gradient witnesses | 2 |
| `tests/step_preference/test_execution_defaults.py` (new) | Defaults/resume resolver contracts | 3 |
| `tests/stage2/test_routing.py`, `test_launcher.py` | Actual assembly and launcher integration | 3 |
| `tests/multinode/test_raid_until_environment.py` (new) | Coordinator environment forwarding | 3 |
| `documents/ours1-raid-until-default.md` (new), `README.md` | Usage and evidence limits | 3 |

Qwen paths in the table are relative to `transformers/src/transformers/models/qwen2_5_vl/`.

Before each task, read the spec, this plan, applicable AGENTS, and the exact existing code. Apply `superpowers:test-driven-development` during implementation. Subagent-driven uses its per-task implementer and independent review loop; Native uses direct implementation and independent whole-change review. Task review sections below provide review criteria at the selected method's review gates. Tasks are sequential because defaults should only be activated after the token and checkpoint primitives exist. Record red/green commands and review revisions. Do not run the broad suite between tasks.

### Task 1: Concatenated text and latent-delimited targets

**Files:**
- Modify: `src/step_preference/spans.py`, `src/stage2/data.py`.
- Create: `tests/step_preference/test_raid_until_alignment.py`.
- Modify: `tests/step_preference/test_target_scope.py`.

**Interfaces:**
- Consumes: existing `build_assistant_plan`, `map_reasoning_step_spans`, `Stage2DataCollator`, `LatentTokenIds`, and `parse_latent_spans`.
- Produces: the existing `ReasoningStepSpan` tuple and collator batch schema, with correct `until_next_latent` token intervals. No new trainer-facing batch fields.
- Keep immediate-text mapping independent. Defaults are changed in Task 3; this task tests an explicit until scope.

- [ ] **Step 1: Add a real-tokenizer regression test.** Add the following test in the new file. All imported fixtures already exist in ours1.

```python
import torch
from PIL import Image
from transformers import AutoProcessor
from stage2.data import Stage2DataCollator, prepare_stage2_example
from step_preference.spans import ASSISTANT_PLAN_RENDERING_CONCATENATED
from tests.step_preference.test_spans import STAGE2A_CHECKPOINT
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


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
    token_ids = LatentTokenIds(**{
        f"{name}_token_id": tok.convert_tokens_to_ids(f"<|latent_{name}|>")
        for name in ("pad", "start", "end")
    })
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
```

- [ ] **Step 2: Run the causal failure.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_raid_until_alignment.py -q
```

Expected initial failure: a text/text crossing-token boundary or the preceding immediate-text interval cannot be mapped exactly. A missing local processor artifact is a prerequisite failure, not the desired red witness; resolve it without downloading/changing model weights.

- [ ] **Step 3: Add the until-specific token mapping.** In `map_reasoning_step_spans`, keep the existing `token_interval` helper but validate only assistant/latent boundaries in the until branch. The immediate branch keeps its per-entry checks. Before the current token-crossing loop, replace `boundaries` for until with:

```python
if preference_target_scope == "until_next_latent":
    boundaries = {assistant_start, assistant_start + len(assistant_text)}
    for span in plan_char_spans:
        if span.step_type == "latent":
            boundaries.update((
                assistant_start + span.start,
                assistant_start + span.start + len(LATENT_START_TOKEN),
                assistant_start + span.end - len(LATENT_END_TOKEN),
                assistant_start + span.end,
            ))
```

After defining `token_interval` and before the immediate mapper, use this body for until:

```python
assistant_begin, assistant_end = token_interval(
    assistant_start, assistant_start + len(assistant_text))
blocks = []
for step in reasoning_steps:
    start, after_end = token_interval(
        assistant_start + step.latent.start,
        assistant_start + step.latent.end,
    )
    body_start, end = start + 1, after_end - 1
    if any(token != latent_pad_token_id for token in ids[body_start:end]):
        raise ValueError("latent token interval contains a non-latent-pad token")
    blocks.append((start, body_start, end))
mapped = []
previous_end = assistant_begin
for index, (start, body_start, end) in enumerate(blocks):
    target_start = end + 1
    target_end = blocks[index + 1][0] if index + 1 < len(blocks) else assistant_end
    if not (previous_end < start < body_start < end < target_end <= assistant_end):
        raise ValueError("complete reasoning step requires non-empty previous and target text")
    step = reasoning_steps[index]
    mapped.append(ReasoningStepSpan(
        step_id=step.step_id,
        prev_text_start=previous_end, prev_text_end=start,
        latent_start=body_start, latent_end=end,
        target_text_start=target_start, target_text_end=target_end,
        helper_image_idx=step.helper_image_idx,
    ))
    previous_end = target_start
return tuple(mapped)
```

`previous_end` follows the preserved RAID interval convention; it avoids remapping only the last text fragment before a latent, which can split a single token. This changes span metadata for until where adjacent text exists, not the V-off boundary masking policy. Verify optional textual-sidecar consumers still accept the complete interval; do not alter their attention-mask algorithm. Validate block count, PAD identities, nonempty target, assistant bounds, and latent delimiters before returning. Keep the immediate-text branch's previous behavior. Avoid importing RAID modules in production.

- [ ] **Step 4: Add an independent preserved-source comparison.** Use an audit-only helper in the new test file to load only pure definitions from preserved RAID source. This is test evidence, never a runtime dependency. Record source SHA256 values in the result log. The helper is fully defined here:

```python
import ast
import dataclasses
import hashlib
from pathlib import Path
from typing import Mapping, Sequence


def read_pure_definitions(path, names):
    raw = Path(path).read_bytes()
    tree = ast.parse(raw.decode())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    namespace = {"dataclass": dataclasses.dataclass, "Mapping": Mapping,
                 "Sequence": Sequence}
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentSpan
    namespace["LatentSpan"] = LatentSpan
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    print({"reference": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return namespace
```

Load `_assistant_text` from `/raid/workspace/temp2/ILVR/.worktrees/ours1/src/stage2/data.py` and `ReasoningStepSpan`, `_latent_plan_entries`, `derive_reasoning_step_spans` from its `src/step_preference/spans.py`. Use the same prepared plan, processor and image from Step 1 for both rendered strings. Compare resulting `input_ids` tensors, assistant text, target IDs, and predictor indices exactly. Use `parse_latent_spans` on the processor IDs for the reference function; derive assistant bounds from the template/offsets independently of the local target mapper. Compare reference `target_text_start/end` with every local span, and reference `range(start - 1, end - 1)` with `build_context_sidecar_plan(...).root_selected_positions`. Do not manufacture expected intervals by calling the modified local mapper.

Add parameterized plans for `"hello"` split as `"hel"/"lo"`, text with original spaces/newlines, one latent, two latents, and a final answer. Add an immediate-text test using the existing adjacent-space/LF rendering and assert its old target text. Reuse the existing `test_opt_in_real_collator_expands_targets_without_changing_input_or_ce` as an explicit-rendering test; do not retain its assumption that an omitted scope always means immediate after Task 3.

- [ ] **Step 5: Verify and obtain task review.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_raid_until_alignment.py tests/step_preference/test_target_scope.py tests/step_preference/test_spans.py tests/stage2/test_data.py -q
```

Expected: green token/label/mapping tests. Report the local processor source and source-file hashes. If RAID files are unavailable, the comparison is unverified, not a skipped pass. A reviewer checks source independence, assistant terminator exclusion and preserved non-until behavior at the selected method's review gate. Handle findings through the selected implementation method, then rerun only affected tests.

- [ ] **Step 6: Commit only Task 1 changes in the owned execution worktree.**

```bash
git add -- src/step_preference/spans.py src/stage2/data.py tests/step_preference/test_raid_until_alignment.py tests/step_preference/test_target_scope.py
git diff --cached --check
git commit -m 'fix: align until targets with concatenated RAID text'
```

### Task 2: Selective latent checkpointing and live clean KV

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`, `modeling_qwen2_5_vl.py` and matching modular definition where present.
- Modify: `src/main.py`, `src/stage2/trainer.py`.
- Create: `tests/stage2/test_raid_until_checkpointing.py`.

**Interfaces:**
- Extend `LatentRolloutEngine.rollout(..., latent_activation_checkpointing: bool = False)`; return type stays `LatentRolloutResult`.
- Extend `configure_stage2_model_backend(..., latent_activation_checkpointing: bool = False)`; set `model.config.stage2_latent_activation_checkpointing` and reject simultaneous native/selective checkpointing.
- Qwen forward passes `getattr(self.config, "stage2_latent_activation_checkpointing", False)` into rollout.
- `stage2_shared_prefix_inputs["checkpoint"]` is the existing explicit sidecar switch. Read `model.config.stage2_sidecar_activation_checkpointing` with legacy fallback `True` in the trainer, through the existing PEFT model-unwrapping helper. No new KV container or sidecar scorer is needed.
- This task adds capabilities with legacy fallback values. Task 3 activates the new until defaults.

- [ ] **Step 1: Add a causal replay/dropout test.** Use the already reviewed helper implementation from `/root/workspace/ILVR/.worktrees/until-temp2-alignment-20260923/tests/stage2/test_reproduction_checkpointing.py`: copy only `TinyDecoder`, `run_case`, and `test_checkpointed_prefix_replay_preserves_outputs_loss_and_gradients` plus their required imports into the new test file. They exercise two latent blocks and explicit tensor checkpoint inputs with dropout. Add this conflict regression:

```python
import pytest
from main import configure_stage2_model_backend
from tests.stage2.helpers import tiny_stage2_qwen


def test_native_and_latent_checkpointing_conflict():
    with pytest.raises(ValueError, match="native|gradient|checkpoint"):
        configure_stage2_model_backend(
            tiny_stage2_qwen(), gradient_checkpointing=True,
            latent_activation_checkpointing=True,
        )
```

- [ ] **Step 2: Record the initial failures.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_raid_until_checkpointing.py -q
```

Expected: unsupported `latent_activation_checkpointing` argument until implemented. Inspect the donor helpers before copying; do not copy their profile/sampler/launcher assumptions.

- [ ] **Step 3: Implement the recurrence checkpoint.** Reuse the matching donor rollout code, with explicit tensors and a bound `language_model`. The core loop body is:

```python
prefix_inputs = _assemble_latent_inputs(base_inputs_embeds, replacements, stop=position)
prefix_mask = attention_mask[:, :position]
prefix_positions = full_position_ids[..., :position]

def predict_last_hidden(bound_inputs, bound_mask, bound_positions):
    outputs = language_model(
        inputs_embeds=bound_inputs, attention_mask=bound_mask,
        position_ids=bound_positions, use_cache=False, return_dict=True,
    )
    return outputs.last_hidden_state[:, -1, :]

if latent_activation_checkpointing and torch.is_grad_enabled():
    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        latent = torch.utils.checkpoint.checkpoint(
            predict_last_hidden, prefix_inputs, prefix_mask, prefix_positions,
            use_reentrant=False, preserve_rng_state=True,
        )
else:
    latent = predict_last_hidden(prefix_inputs, prefix_mask, prefix_positions)
```

Import `torch.utils.checkpoint` explicitly. Keep append/replacement order and diagnostics unchanged. Extend the backend helper with the conflict check before enabling/disabling native checkpointing. Pass the model-config latent flag into the Qwen rollout call. Pass an explicit sidecar `checkpoint` option from the unwrapped Qwen config; do not globally change `score_sidecars`' legacy fallback or change prefix tensor slicing/detachment.

- [ ] **Step 4: Add a real tiny-model runtime witness.** Extend the new test file with:

```python
import torch
from step_preference.equivalence import _trainer
from tests.step_preference.test_context_preference import context_config
from tests.step_preference.test_context_sidecars import two_step_case


def test_only_latent_decoder_calls_replay(monkeypatch):
    import transformers.models.qwen2_5_vl.stage2_shared_prefix as shared
    model, batch = two_step_case()
    model.train()
    configure_stage2_model_backend(model, gradient_checkpointing=False,
                                   latent_activation_checkpointing=True)
    model.config.stage2_sidecar_activation_checkpointing = False
    records = []
    phase = ["forward"]

    def observed(_module, _args, kwargs):
        kind = ("sidecar" if kwargs.get("stage2_prefix_key_values") is not None
                else "clean" if kwargs.get("stage2_capture_kv", False) else "latent")
        records.append((phase[0], kind, kwargs["inputs_embeds"].shape[1]))

    def forbidden_checkpoint(*args, **kwargs):
        pytest.fail("sidecar checkpoint was invoked")

    monkeypatch.setattr(shared, "checkpoint", forbidden_checkpoint)
    handle = model.model.register_forward_pre_hook(observed, with_kwargs=True)
    try:
        trainer = _trainer(context_config(target_scope="until_next_latent"))
        loss = trainer.compute_loss(model, batch)
        phase[0] = "backward"
        loss.backward()
    finally:
        handle.remove()
    assert not model.is_gradient_checkpointing
    forward_latent = [length for phase_name, kind, length in records
                      if phase_name == "forward" and kind == "latent"]
    backward = [(kind, length) for phase_name, kind, length in records
                if phase_name == "backward"]
    assert forward_latent and backward
    assert all(kind == "latent" for kind, _ in backward)
    assert sorted(length for _, length in backward) == sorted(forward_latent)
```

Hooks must witness the actual decoder calls, not infer replay from the now-finished rollout wrapper. If checkpoint nesting makes a count differ, investigate rather than weakening the latent-only and clean/sidecar-zero assertions.

- [ ] **Step 5: Isolate the V-off-to-clean gradient.** In a separate test, wrap the local `score_sidecars` function to retain the exact `prefix_kv` tensors; return its ordinary results. On `two_step_case`, with native OFF/latent ON/sidecar OFF, call `_trainer(context_config(target_scope="until_next_latent")).compute_loss(..., return_outputs=True)`. Backward only through `-sum(result["mean_logp"] for result in outputs.stage2_sidecar_results if result["kind"] == "visual_off")`. Use the following capture/assertion body:

```python
captured = []
original_score = shared.score_sidecars

def observed_score(model, options, input_ids, clean_embeds, clean_positions, prefix_kv):
    assert options["checkpoint"] is False
    for key, value in prefix_kv:
        assert key.requires_grad and value.requires_grad
        key.retain_grad()
        value.retain_grad()
        captured.extend((key, value))
    return original_score(model, options, input_ids, clean_embeds, clean_positions, prefix_kv)

monkeypatch.setattr(shared, "score_sidecars", observed_score)
_, outputs = _trainer(context_config(target_scope="until_next_latent")).compute_loss(
    model, batch, return_outputs=True,
)
voff = [r["mean_logp"] for r in outputs.stage2_sidecar_results if r["kind"] == "visual_off"]
assert voff
(-torch.stack(voff).sum()).backward()
assert captured
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in captured)
assert any(torch.count_nonzero(t.grad).item() for t in captured)
```

Add the following full/DoRA parity test. These are CPU unit backward probes, not training launches.

```python
import copy
from stage2.peft import Stage2TuningConfig, build_stage2_tuned_model, resolve_stage2_qwen_model
from tests.stage2.helpers import activate_stage2_adapter_deltas


@pytest.mark.parametrize("method", ["full", "dora"])
def test_selective_checkpoint_preserves_tiny_gradients(method):
    base, batch = two_step_case()
    if method == "dora":
        base = build_stage2_tuned_model(base, Stage2TuningConfig(
            method="dora", r=4, alpha=8, dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ))
        activate_stage2_adapter_deltas(base)
    records = []
    for enabled in (False, True):
        model = copy.deepcopy(base).train()
        qwen = resolve_stage2_qwen_model(model)
        configure_stage2_model_backend(qwen, gradient_checkpointing=False,
                                       latent_activation_checkpointing=enabled)
        qwen.config.stage2_sidecar_activation_checkpointing = False
        torch.manual_seed(42)
        loss, out = _trainer(context_config(target_scope="until_next_latent")).compute_loss(
            model, batch, return_outputs=True,
        )
        loss.backward()
        grads = {name: None if p.grad is None else p.grad.detach().clone()
                 for name, p in model.named_parameters() if p.requires_grad}
        records.append((loss.detach(), out.logits.detach(), grads))
    for index in (0, 1):
        torch.testing.assert_close(records[0][index], records[1][index], atol=1e-5, rtol=1e-4)
    assert records[0][2].keys() == records[1][2].keys()
    maxima = []
    for name, left in records[0][2].items():
        right = records[1][2][name]
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4, msg=name)
            maxima.append(float((left - right).abs().max()))
    assert maxima
    print({"method": method, "gradient_tensors": len(maxima), "max_abs_error": max(maxima)})
```

Reuse the donor's `test_checkpointing_is_opt_in_and_skipped_under_no_grad` with its `TinyDecoder` fixture to pin the no-grad case; the low-level primitive still defaults OFF even after Task 3 changes fresh training defaults.

- [ ] **Step 6: Focused verification, review, and commit.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_raid_until_checkpointing.py tests/stage2/test_rollout.py tests/step_preference/test_context_sidecars.py -q
```

Reviewer checks closure inputs, RNG, actual hook paths, cache gradient connectivity, and absence of sidecar checkpoints. Do not claim production memory or historical numerical equivalence. After any fixes, stage only Task 2 files in the owned branch and commit:

```bash
git add -- src/main.py src/stage2/trainer.py transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py transformers/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py tests/stage2/test_raid_until_checkpointing.py
git diff --cached --check
git commit -m 'feat: checkpoint latent recurrence independently of clean and sidecars'
```

### Task 3: Activate defaults consistently and preserve saved execution settings

**Files:**
- Create: `src/step_preference/execution.py`, `tests/step_preference/test_execution_defaults.py`, `tests/multinode/test_raid_until_environment.py`, `documents/ours1-raid-until-default.md`.
- Modify: `src/step_preference/config.py`, `src/utils_deepseed.py`, `src/stage2/data.py`, `src/main.py`, `src/stage2/trainer.py`, `run_stage2_training.sh`, `src/multinode_runtime.py`, `README.md`.
- Modify: `tests/step_preference/test_target_scope.py`, `tests/stage2/test_routing.py`, `tests/stage2/test_launcher.py` and directly affected config expectations.

**Interfaces:**
- `StepPreferenceConfig.target_scope: str | None = None` accepts omission; after `__post_init__`, it is a validated string. Omitted means until for context-preference Stage 2B, immediate for non-applicable modes. Explicit unsupported scope still errors.
- Add `Stage2ExecutionSettings`, a frozen dataclass with `assistant_plan_rendering: str`, `native_checkpointing: bool`, `latent_checkpointing: bool`, `sidecar_checkpointing: bool`.
- Add `resolve_stage2_execution(config: StepPreferenceConfig, *, native_checkpointing: bool, latent_checkpointing: bool | None = None, saved: Mapping | None = None) -> Stage2ExecutionSettings` in `execution.py`.
- Save these four resolved values plus resolved target scope in one model-config dictionary `stage2_preference_execution`. Existing model config saving persists it; do not build a new checkpoint archive/manager.
- Saved settings apply only to an actual resume, never to fresh initialization from Stage 1 or Stage 2A. Existing unsupported Stage 2B resume remains unsupported; this task does not enable a new resume mode.

- [ ] **Step 1: Add omission/explicit/resume unit tests.** Use this base in the new defaults file:

```python
from dataclasses import asdict
import pytest
from step_preference.config import StepPreferenceConfig
from step_preference.execution import resolve_stage2_execution
from utils_deepseed import parse_args

CONTEXT = ["--stage", "stage2", "--step_preference_enabled",
           "--step_preference_mode", "stage2b",
           "--stage2a_checkpoint_path", "checkpoint"]


def test_fresh_context_defaults_to_until():
    args = parse_args(CONTEXT)
    assert args.step_preference_config.target_scope == "until_next_latent"
    resolved = resolve_stage2_execution(args.step_preference_config,
                                       native_checkpointing=False)
    assert asdict(resolved) == dict(
        assistant_plan_rendering="concatenated-v1", native_checkpointing=False,
        latent_checkpointing=True, sidecar_checkpointing=False,
    )


def test_explicit_immediate_and_dormant_defaults():
    explicit = parse_args(CONTEXT + ["--step_preference_target_scope", "immediate_text"])
    assert explicit.step_preference_config.target_scope == "immediate_text"
    assert StepPreferenceConfig().target_scope == "immediate_text"
    assert parse_args(["--stage", "stage2"]).step_preference_config.target_scope == "immediate_text"
    with pytest.raises(ValueError, match="context.preference Stage 2B"):
        StepPreferenceConfig(target_scope="until_next_latent")


def test_saved_until_execution_keeps_historical_rendering_and_checkpointing():
    config = StepPreferenceConfig(enabled=True, mode="stage2b",
                                 stage2a_checkpoint_path="checkpoint")
    saved = dict(target_scope="until_next_latent",
                 assistant_plan_rendering="adjacent-text-space-lf-v1",
                 native_checkpointing=True, latent_checkpointing=False,
                 sidecar_checkpointing=True)
    result = resolve_stage2_execution(config, native_checkpointing=False, saved=saved)
    assert result.assistant_plan_rendering == saved["assistant_plan_rendering"]
    assert result.native_checkpointing is True
    assert result.latent_checkpointing is False
    assert result.sidecar_checkpointing is True
    with pytest.raises(ValueError, match="saved.*settings|missing"):
        resolve_stage2_execution(config, native_checkpointing=False, saved={})
```

Add invalid saved booleans/rendering/scope, incomplete saved fields, native+latent conflict, and new until explicit native-ON regressions. Saved scope and explicit caller scope conflicts must raise before dataset construction. An omitted caller scope must be restored from saved scope before making `StepPreferenceConfig`; do not use the new default as if explicitly requested.

- [ ] **Step 2: Run and record red tests.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_execution_defaults.py -q
```

Expected initial failure: missing execution resolver; after adding that API, omission assertions must still catch the old immediate default until wiring is changed.

- [ ] **Step 3: Implement scope normalization and the execution resolver.** At the start of config `__post_init__`, normalize omission before existing validation:

```python
if self.target_scope is None:
    object.__setattr__(self, "target_scope",
        "until_next_latent" if self.is_context_preference else "immediate_text")
```

Use `None` for CLI scope omission and `from_args` fallback. After constructing config, expose its resolved scope in the public namespace for logging. The collator's omitted scope likewise resolves to until only when preference is enabled without legacy negatives; explicit incompatible until still errors.

Implement the resolver with these imports and concrete dataclass:

```python
from dataclasses import dataclass
from typing import Mapping
from step_preference.config import StepPreferenceConfig
from step_preference.spans import normalize_assistant_plan_rendering, normalize_preference_target_scope


@dataclass(frozen=True)
class Stage2ExecutionSettings:
    assistant_plan_rendering: str
    native_checkpointing: bool
    latent_checkpointing: bool
    sidecar_checkpointing: bool


def resolve_stage2_execution(config: StepPreferenceConfig, *, native_checkpointing: bool,
                            latent_checkpointing: bool | None = None,
                            saved: Mapping | None = None) -> Stage2ExecutionSettings:
    if saved is not None:
        fields = ("assistant_plan_rendering", "native_checkpointing",
                  "latent_checkpointing", "sidecar_checkpointing")
        if any(key not in saved for key in (*fields, "target_scope")):
            raise ValueError("missing saved execution settings")
        scope = normalize_preference_target_scope(saved["target_scope"])
        if scope != config.target_scope:
            raise ValueError("saved target scope conflicts with requested scope")
        normalize_assistant_plan_rendering(saved["assistant_plan_rendering"])
        if any(type(saved[key]) is not bool for key in fields[1:]):
            raise ValueError("saved checkpoint settings must be booleans")
        if saved["native_checkpointing"] and saved["latent_checkpointing"]:
            raise ValueError("saved native and latent checkpointing conflict")
        return Stage2ExecutionSettings(**{key: saved[key] for key in fields})
    until = config.is_context_preference and config.target_scope == "until_next_latent"
    latent = until if latent_checkpointing is None else latent_checkpointing
    if type(native_checkpointing) is not bool or type(latent) is not bool:
        raise ValueError("checkpoint settings must be booleans")
    if native_checkpointing and (until or latent):
        raise ValueError("native checkpointing conflicts with until/latent checkpointing")
    return Stage2ExecutionSettings(
        assistant_plan_rendering=("concatenated-v1" if until or not config.enabled
                                  else "adjacent-text-space-lf-v1"),
        native_checkpointing=native_checkpointing,
        latent_checkpointing=latent,
        sidecar_checkpointing=not until,
    )
```

Do not force fresh until defaults on the saved branch. The caller restores omitted scope before constructing the config and rejects conflicting explicit overrides at the resume boundary; the resolver does not infer whether its arguments were explicit.

- [ ] **Step 4: Wire options, model setup, rendering, and Trainer hooks.** Add the following CLI option with omission retained until settings resolution:

```python
parser.add_argument("--stage2_latent_activation_checkpointing",
                    action=argparse.BooleanOptionalAction, default=None)
```

Use one resolved settings object before manifest validation, example preparation or filtering. Replace main's hardcoded `assistant_plan_rendering` conditional with the resolved value. Apply native/latent settings via Task 2's backend helper; attach the sidecar choice and `stage2_preference_execution` dictionary to the unwrapped Qwen model config. Have `build_stage2_training_arguments` use the same resolved native value rather than a stale raw flag. Keep no-cache rollout and the explicit differentiable KV interface.

At actual resume entry, load the saved model config through the existing source dependency/load-decision path before rendering. Restore saved scope if omitted and validate explicit conflicts; if a preference resume lacks reliable rendering/checkpoint fields, raise an actionable error before loading training data. Do not inspect the initializer's stored Stage 2 execution values as resume settings. Preserve the existing Stage 2B resume prohibition in `StepPreferenceConfig`; test that it still rejects rather than bypassing it.

For supported non-preference resumes, retain their existing rendering/native behavior and keep latent checkpointing OFF unless reliable saved settings or the existing explicit user setting requests it. A missing `stage2_preference_execution` dictionary on ordinary Stage 2/Stage 1 is not an error. Do not load pickled training args just to infer settings.

- [ ] **Step 5: Resolve launcher defaults after effective mode is known.** Preserve unset versus explicitly set `STEP_PREFERENCE_TARGET_SCOPE` before assigning fallbacks. Only fresh context-preference Stage 2B gets default until. Place normalization before constructing `STEP_PREFERENCE_ARGS` and before the current `scope != immediate_text` compatibility guard. The intended shell decision is:

```bash
if [[ ! -v STEP_PREFERENCE_TARGET_SCOPE ]]; then
  if [[ "$STEP_PREFERENCE_ENABLED" == true && "${STEP_PREFERENCE_MODE:-stage2a}" == stage2b && "$STEP_PREFERENCE_OBJECTIVE" == context_preference ]]; then
    STEP_PREFERENCE_TARGET_SCOPE=until_next_latent
  else
    STEP_PREFERENCE_TARGET_SCOPE=immediate_text
  fi
fi
```

Likewise resolve omitted latent checkpointing to true only for the fresh until path, false for other fresh modes; forward the appropriate Boolean CLI flag. Explicit native ON with the until default must fail before launch. On resume, do not replace missing saved settings by fresh defaults; preserve the distinction into Python or use the existing resume rejection. Add `STAGE2_LATENT_ACTIVATION_CHECKPOINTING` to the coordinator's allowed environment keys, then inspect `src/multinode.py` to ensure no separate whitelist drops it. Preserve managed-argument duplicate rejection.

- [ ] **Step 6: Add launcher and coordinator regressions.** Extend the existing launcher test helpers with:

```python
def test_new_context_defaults_and_ce_only_are_launchable():
    context = {"STEP_PREFERENCE_ENABLED": "true", "STEP_PREFERENCE_MODE": "stage2b",
               "STAGE2A_CHECKPOINT_PATH": "/models/stage2a"}
    tokens = dry_run_tokens(run_launcher(env_overrides=context))
    assert option_value(tokens, "--step_preference_target_scope") == "until_next_latent"
    assert "--stage2_latent_activation_checkpointing" in tokens
    assert "--no-gradient_checkpointing" in tokens
    assert run_launcher().returncode == 0
    explicit = dry_run_tokens(run_launcher(env_overrides={
        **context, "STEP_PREFERENCE_TARGET_SCOPE": "immediate_text"}))
    assert option_value(explicit, "--step_preference_target_scope") == "immediate_text"
    conflict = run_launcher(env_overrides={**context, "GRADIENT_CHECKPOINTING": "true"})
    assert conflict.returncode != 0
```

In `tests/multinode/test_raid_until_environment.py`, reuse `validated_stage2_environment` with the same dictionary interface as the existing environment tests:

```python
from src.multinode_runtime import validated_stage2_environment


def test_coordinator_keeps_until_and_latent_checkpoint_options():
    env = validated_stage2_environment(
        {"STAGE1_CHECKPOINT": "/models/stage1", "DATA_PATH": "/data/train.jsonl",
         "IMAGE_ROOT": "/data/images", "STEP_PREFERENCE_TARGET_SCOPE": "until_next_latent",
         "STAGE2_LATENT_ACTIVATION_CHECKPOINTING": "true"},
        output_dir="/outputs/stage2", log_file="/logs/stage2.log",
        deepspeed_config="/configs/zero.json",
    )
    assert env["STEP_PREFERENCE_TARGET_SCOPE"] == "until_next_latent"
    assert env["STAGE2_LATENT_ACTIVATION_CHECKPOINTING"] == "true"
```

- [ ] **Step 7: Add assembly, save/read and non-applicable-stage regressions.** Extend `test_training_assembly_passes_target_scope_to_real_collator` using its existing injected `Stage2SourceDependencies`, `CompositionProcessor` and `stage2_args` fixtures. Capture `prepare_stage2_example`/filter kwargs and assert concatenated rendering, until collator scope, native OFF, latent ON and sidecar OFF for omission. Explicit immediate retains its previous rendering. Use `build_stage2_training_arguments` and actual tiny-model `configure_stage2_model_backend` to assert both model-native and Trainer-native flags are OFF.

For a configuration-only save/read check, use `model.config.to_dict()` and `Qwen2_5_VLConfig.from_dict`, not a model checkpoint writer. Assert all five `stage2_preference_execution` fields survive and the resolver restores them unchanged. Existing Stage 2B resume rejection must remain; supported Stage 2 resumes without preference must retain their old path. Add input cases for Stage 1, CE-only, Stage 2A, legacy negative objective, explicit wrong scope, and a length manifest made with the old renderer (must reject mismatched rendering rather than reuse its retained indices).

Run the Task 2 decoder replay witness through assembled components once, using the existing CPU source-injection approach, to connect CLI/default resolution with actual forward/backward behavior. This is the smallest integration gate; it does not start `Trainer.train` or an optimizer job.

- [ ] **Step 8: Document exactly the new behavior.** Write `documents/ours1-raid-until-default.md` with this usage block and link it from README's Stage 2 section:

```bash
# Fresh context-preference: resolved defaults in ours1.
STEP_PREFERENCE_TARGET_SCOPE=until_next_latent
GRADIENT_CHECKPOINTING=false
STAGE2_LATENT_ACTIVATION_CHECKPOINTING=true
STEP_PREFERENCE_BETA=0.5
STEP_PREFERENCE_WEIGHT=1.0
```

Explain concatenated text and final-answer targets; native/clean/sidecar OFF versus latent ON; explicit immediate option; saved resume precedence and unsupported/ambiguous resume errors. State that V-off boundary masking, sampler and model initialization have not been aligned by this patch. Include the historical uncommitted-source limitation. Retain the mandatory two-node launch policy and reference the existing coordinator rather than adding a new launch recipe. Do not rewrite historical result documents or canonical defaults for other worktrees.

- [ ] **Step 9: Focused verification and independent review.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_execution_defaults.py tests/step_preference/test_target_scope.py tests/stage2/test_routing.py tests/stage2/test_launcher.py tests/stage1/test_launcher.py tests/multinode/test_raid_until_environment.py tests/stage2/test_raid_until_checkpointing.py -q
bash -n run_stage2_training.sh
```

Reviewer checks mode-dependent defaults, explicit overrides, actual saved-setting precedence, unchanged unsupported-resume errors, stale sourced settings, and that no source profile beyond the approved three behaviors was imported. Handle confirmed findings through the selected implementation method. Update tests that intended immediate behavior by specifying immediate explicitly; do not mechanically change assertions to until when the test's purpose is another mode.

- [ ] **Step 10: Commit the reviewed Task 3 changes.**

```bash
git add -- src/step_preference/execution.py src/step_preference/config.py src/utils_deepseed.py src/stage2/data.py src/main.py src/stage2/trainer.py run_stage2_training.sh src/multinode_runtime.py tests/step_preference/test_execution_defaults.py tests/step_preference/test_target_scope.py tests/stage2/test_routing.py tests/stage2/test_launcher.py tests/multinode/test_raid_until_environment.py documents/ours1-raid-until-default.md README.md
git diff --cached --check
git commit -m 'feat: default fresh ours1 preference to RAID until execution'
```

If inspection requires a second coordinator whitelist change or directly affected config tests, list those exact additional files in the task evidence and stage them explicitly. Do not stage all dirty files.

## Final review, verification, and handoff

- [ ] Dispatch the required whole-change review through `superpowers:requesting-code-review`, comparing against the recorded execution baseline. Give the reviewer the spec, plan, actual approval quotes, donor hashes, applicable task review results, and complete diff. Root compares findings to evidence; handle deliverable fixes through the selected implementation method.
- [ ] Reconcile every spec claim with a witness: token/reference comparison (Task 1), actual replay/KV gradients (Task 2), default/override/resume/assembly behavior (Task 3), no unapproved changes (reviewed diff). Unverified production execution remains labeled unverified.
- [ ] Before choosing final tests, invoke `verification-before-completion` and `finishing-a-development-branch` if implementation is ready for branch handoff. Announce both gates. Use one full repository suite invocation for their overlapping test claim:

```bash
mkdir -p logs/raid-until-default-20260924
set -o pipefail
PYTHONPATH="$PWD/transformers/src:$PWD/src" CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest -q 2>&1 | tee logs/raid-until-default-20260924/final-pytest.log
```

Read its complete output. Default environment-gated GPU/distributed skips remain skips; this command does not establish GPU/multi-node training parity. Do not run an identical broad suite again unless a concrete fix changes the reviewed revision. Record any preexisting failing test rather than silently excluding it; finishing cannot claim a green full suite while failures remain.

- [ ] Run `git diff --check` and `bash -n run_stage2_training.sh` against the reviewed final revision for their distinct whitespace/shell claims. Confirm the token oracle and runtime witnesses were included in the green suite. Preserve commands, exits, counts, source hashes, reviewed commit and final commit in `results/raid-until-default-20260924/verification.json`.
- [ ] Present the finishing skill's applicable integration choices and wait for the human decision. The original ours1 worktree is user-owned; do not delete it or overwrite its dirty work. No automatic merge, push, PR, training or evaluation is authorized by this plan.
- [ ] Before the final response, call `/root/.codex/hooks/write_ntfy_summary.sh` with the actual outcome. Report the three synchronized behaviors, default until scope, verified CPU/test scope, and any unresolved limitation concisely.

## Plan self-review record

- Spec sections 1–3 and 7: scope, preserved reference source, existing-work isolation and historical-source limits are carried into execution boundaries and review.
- Spec section 4: Task 1 implements text/target mapping; Task 3 selects concatenated rendering in real assembly.
- Spec section 5: Task 2 implements and witnesses selective replay/live KV; Task 3 makes those values effective defaults.
- Spec section 6: Task 3 handles omitted/explicit mode and saved settings without adding a new resume mode.
- Spec section 8: each of the five Review Focus cases has a named task witness; final full suite and independent review are explicit gates.
- Spec section 9: plan approval is still pending; product implementation has not started.

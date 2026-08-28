# ILVR Stage 2 Reimplementation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a maintainable, differentiable ILVR Stage 2 training and inference path that starts from a Stage 1 checkpoint, rolls out continuous latent states, and applies text-only CE supervision.

**Architecture:** A single Qwen-specific `LatentRolloutEngine` owns the no-cache continuous recurrence. The local Qwen wrapper prepares one visual embedding pass and full-sequence MRoPE positions, while project-level modules own CoMT data, checkpoint transitions, thin Trainer plumbing, and model-specific inference. Stage 1 remains on its existing path, and generic `GenerationMixin` is cleaned only after behavioral coverage exists.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu128, local Transformers 4.50.0.dev0 fork, TRL 0.15.2, DeepSpeed 0.18.0, pytest 9.1.1, NVIDIA B200.

**Spec:** `docs/superpowers/specs/2026-08-24-ilvr-stage2-reimplementation-design.md`

## Global Constraints

- At execution start, invoke `superpowers:using-git-worktrees` and confirm work remains in the existing isolated Stage 2 worktree rather than `main`/`master`.
- Run Python, pytest, training, and evaluation only through `./venv/ilvr-venv`; never use system Python or `./ilvr-venv`.
- Prefix every repository Python command with `PYTHONPATH="$PWD/transformers/src:$PWD/src"` so the worktree-local Transformers fork wins over the shared editable installation.
- Preserve Stage 1 behavior and keep `src/trainer.py`'s EMA/helper-image path out of Stage 2.
- Continuous recurrence is exactly `latent = final_last_hidden_state`; no detach, projection, normalization, quantization, sampling, or embedding lookup is allowed.
- Compute the full multimodal position specification once per Stage 2 scaffold and slice it for recurrent prefixes.
- Stage 2 latent slots always have label `-100`; `supervise_latent_boundaries` controls both boundaries together and defaults to `True`.
- Fresh Stage 2 loads model/config/tokenizer/processor from Stage 1 but creates new optimizer, scheduler, and Trainer state.
- Stage 2 resume is explicit through `resume_from_checkpoint_path` and restores a validated Stage 2 Trainer checkpoint.
- Stage 2 model input contains original user images only; helper images never become tensors.
- Freeze `model.visual`, encode it once per local batch under `torch.no_grad()`, and keep the language-model recurrence differentiable.
- Initial local micro-batch size is exactly one. Reject `B > 1`; use gradient accumulation and rank-local `B=1` for effective batch sizes above one.
- Initial recurrence and inference use `use_cache=False`; no differentiable KV cache or custom latent kernel is part of this plan.
- Do not add Stage 2 state or hard-coded token IDs to generic `GenerationMixin`.
- Stage 2 token IDs come from tokenizer/config and canonical metadata must match legacy Stage 1 aliases when both exist.
- Use TDD for every behavior change. Show the expected red failure before production edits, then run the focused green test.
- Invoke `superpowers:test-driven-development` before starting each production behavior task.
- Invoke `superpowers:systematic-debugging` before changing code in response to any unexpected numerical, gradient, CUDA, FlashAttention, DeepSpeed, checkpoint, or integration failure.
- Invoke `superpowers:verification-before-completion` before claiming any task or milestone is passing or complete.
- Do not push, merge, publish, or create a PR without explicit permission.
- Before a check that reads more than 10 GB or runs longer than five minutes, reuse equivalent existing evidence. Run an expensive immutable checkpoint validation at most once per execution.

## File Responsibility Map

- `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`: token/span value types, structural parser, differentiable recurrence, rollout result, optional diagnostics.
- `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`: model-facing Stage 2 forward, one visual encoding, MRoPE computation, final logits/loss, model-specific generation flag.
- `src/stage2/checkpointing.py`: canonical/legacy token-ID validation, Stage 2 config metadata, fresh-vs-resume source decision.
- `src/stage2/data.py`: CoMT trajectory conversion, label construction, collator, exact length filtering and statistics.
- `src/stage2/trainer.py`: thin standard Trainer subclass that forwards model CE and logs detached diagnostics.
- `src/stage2/inference.py`: model-specific no-cache NORMAL_TEXT/LATENT_MODE controller using the shared engine.
- `src/main.py`: explicit Stage 1/Stage 2 composition root.
- `src/utils_deepseed.py`: backward-compatible CLI definitions.
- `transformers/src/transformers/generation/utils.py`: ordinary generic generation after covered removal of old ILVR state.
- `eval.py` and `src/evaluate_deepseed.py`: evaluation/task scoring around `Stage2InferenceController`.
- `run_stage2_training.sh`: worktree-safe Stage 2 launcher with explicit checkpoint/data settings.
- `tests/stage2/`: tiny deterministic contract, integration, routing, inference, and regression tests.

---

### Task 1: Latent Token Types and Structural Parser

**Files:**
- Create: `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`
- Create: `tests/stage2/__init__.py`
- Create: `tests/stage2/test_structure.py`

**Interfaces:**
- Consumes: `torch.LongTensor` containing one Stage 2 token scaffold.
- Produces: `LatentTokenIds`, `LatentSpan`, `LatentStructureError`, and `parse_latent_spans(input_ids, *, token_ids, latent_size) -> tuple[LatentSpan, ...]`.

- [ ] **Step 1: Write valid-structure and malformed-structure tests**

```python
import pytest
import torch

from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentStructureError,
    LatentTokenIds,
    parse_latent_spans,
)

TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


def test_parse_multiple_latent_spans_in_order():
    ids = torch.tensor([1, 91, 90, 90, 92, 2, 91, 90, 90, 92, 3])
    spans = parse_latent_spans(ids, token_ids=TOKENS, latent_size=2)
    assert [(s.start_position, s.latent_positions, s.end_position) for s in spans] == [
        (1, (2, 3), 4),
        (6, (7, 8), 9),
    ]


@pytest.mark.parametrize(
    "ids, message",
    [
        ([1, 91, 90, 90], "missing latent_end"),
        ([1, 92], "latent_end without latent_start"),
        ([91, 90, 91, 90, 92, 92], "nested latent_start"),
        ([91, 90, 92], "expected 2 latent_pad tokens"),
        ([91, 90, 7, 92], "non-latent_pad token"),
    ],
)
def test_reject_malformed_latent_structure(ids, message):
    with pytest.raises(LatentStructureError, match=message):
        parse_latent_spans(torch.tensor(ids), token_ids=TOKENS, latent_size=2)
```

- [ ] **Step 2: Run the tests and confirm the import failure**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_structure.py -q
```

Expected: collection fails because `stage2_latent_rollout` does not exist.

- [ ] **Step 3: Add immutable value types and a single-pass parser**

```python
from dataclasses import dataclass

import torch


class LatentStructureError(ValueError):
    pass


@dataclass(frozen=True)
class LatentTokenIds:
    pad_token_id: int
    start_token_id: int
    end_token_id: int


@dataclass(frozen=True)
class LatentSpan:
    start_position: int
    latent_positions: tuple[int, ...]
    end_position: int


def parse_latent_spans(
    input_ids: torch.LongTensor,
    *,
    token_ids: LatentTokenIds,
    latent_size: int,
) -> tuple[LatentSpan, ...]:
    if input_ids.ndim != 1:
        raise LatentStructureError("parse_latent_spans expects a 1D token sequence")
    if latent_size <= 0:
        raise LatentStructureError("latent_size must be positive")
    spans: list[LatentSpan] = []
    open_start: int | None = None
    latent_positions: list[int] = []
    for position, raw_token_id in enumerate(input_ids.tolist()):
        token_id = int(raw_token_id)
        if open_start is None:
            if token_id == token_ids.start_token_id:
                open_start = position
                latent_positions = []
            elif token_id == token_ids.end_token_id:
                raise LatentStructureError("latent_end without latent_start")
            elif token_id == token_ids.pad_token_id:
                raise LatentStructureError("latent_pad outside latent block")
            continue
        if token_id == token_ids.start_token_id:
            raise LatentStructureError("nested latent_start")
        if token_id == token_ids.end_token_id:
            if len(latent_positions) != latent_size:
                raise LatentStructureError(f"expected {latent_size} latent_pad tokens")
            spans.append(LatentSpan(open_start, tuple(latent_positions), position))
            open_start = None
            latent_positions = []
            continue
        if token_id != token_ids.pad_token_id:
            raise LatentStructureError("non-latent_pad token inside latent block")
        latent_positions.append(position)
        if len(latent_positions) > latent_size:
            raise LatentStructureError(f"expected {latent_size} latent_pad tokens")
    if open_start is not None:
        raise LatentStructureError("missing latent_end")
    return tuple(spans)
```

Do not pair starts and ends with `zip`, because that silently truncates malformed counts.

- [ ] **Step 4: Run focused tests**

Run the Step 2 command. Expected: all structure tests pass.

- [ ] **Step 5: Commit the structural contract**

```bash
git add transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py tests/stage2
git commit -m "feat: add stage2 latent structure validation"
```

### Task 2: Exact Continuous Latent Rollout

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`
- Create: `tests/stage2/helpers.py`
- Create: `tests/stage2/test_rollout.py`

**Interfaces:**
- Consumes: Task 1 `LatentSpan` values and a callable language model returning `.last_hidden_state`.
- Produces: `LatentRolloutEngine.rollout(...) -> LatentRolloutResult`, where the result exposes `final_inputs_embeds`, `latent_states`, and `prefix_lengths`.

- [ ] **Step 1: Add a deterministic causal language-model fixture**

```python
from types import SimpleNamespace

import torch
from torch import nn


class ToyCausalLanguageModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(128, hidden_size)
        self.calls = []

    def forward(
        self, *, inputs_embeds, attention_mask, position_ids,
        use_cache, return_dict, generate_mode=False,
    ):
        assert use_cache is False
        self.calls.append((inputs_embeds, attention_mask.clone(), position_ids.clone()))
        position_term = position_ids[0].unsqueeze(-1).to(inputs_embeds.dtype)
        hidden = torch.cumsum(inputs_embeds, dim=1) * self.scale + position_term
        return SimpleNamespace(last_hidden_state=hidden)
```

- [ ] **Step 2: Write failing tests for exact indexing, count, multi-block context, and `B > 1`**

```python
def test_rollout_uses_hidden_before_each_latent_slot():
    ids = torch.tensor([1, 91, 90, 90, 92, 2])
    spans = parse_latent_spans(ids, token_ids=TOKENS, latent_size=2)
    embeds = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)
    lm = ToyCausalLanguageModel(hidden_size=2)

    result = LatentRolloutEngine().rollout(
        language_model=lm,
        base_inputs_embeds=embeds,
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        full_position_ids=positions,
        latent_spans=spans,
    )

    assert result.prefix_lengths == (2, 3)
    assert len(result.latent_states) == 2
    assert torch.allclose(result.latent_states[0], lm.calls[0][0][:, :2].cumsum(1)[:, -1] * lm.scale + 1)


def test_rollout_rejects_local_batch_larger_than_one():
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        LatentRolloutEngine().rollout(
            language_model=ToyCausalLanguageModel(2),
            base_inputs_embeds=torch.zeros(2, 4, 2),
            attention_mask=torch.ones(2, 4),
            full_position_ids=torch.zeros(3, 2, 4, dtype=torch.long),
            latent_spans=(LatentSpan(0, (1, 2), 3),),
        )


def test_second_block_keeps_first_block_context():
    ids = torch.tensor([1, 91, 90, 90, 92, 2, 91, 90, 90, 92, 3])
    spans = parse_latent_spans(ids, token_ids=TOKENS, latent_size=2)
    positions = torch.arange(ids.numel()).view(1, 1, -1).expand(3, 1, -1)
    base = torch.arange(ids.numel() * 2, dtype=torch.float32).view(1, -1, 2)
    changed = base.clone()
    changed[:, 0] += 100
    first = LatentRolloutEngine().rollout(
        language_model=ToyCausalLanguageModel(2), base_inputs_embeds=base,
        attention_mask=torch.ones(1, ids.numel()), full_position_ids=positions,
        latent_spans=spans,
    )
    second = LatentRolloutEngine().rollout(
        language_model=ToyCausalLanguageModel(2), base_inputs_embeds=changed,
        attention_mask=torch.ones(1, ids.numel()), full_position_ids=positions,
        latent_spans=spans,
    )
    assert not torch.allclose(first.latent_states[2], second.latent_states[2])
```

- [ ] **Step 3: Run the tests and confirm the missing engine failure**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_rollout.py -q
```

Expected: tests fail because `LatentRolloutEngine` and `LatentRolloutResult` are absent.

- [ ] **Step 4: Implement the minimal differentiable recurrence**

```python
@dataclass(frozen=True)
class LatentRolloutResult:
    final_inputs_embeds: torch.Tensor
    latent_states: tuple[torch.Tensor, ...]
    prefix_lengths: tuple[int, ...]
    diagnostics: "LatentRolloutDiagnostics | None" = None


def _replace_embedding(embeds: torch.Tensor, position: int, value: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (embeds[:, :position], value.unsqueeze(1), embeds[:, position + 1 :]),
        dim=1,
    )


class LatentRolloutEngine:
    def rollout(
        self,
        *,
        language_model,
        base_inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        full_position_ids: torch.Tensor,
        latent_spans: tuple[LatentSpan, ...],
        retain_latent_grads: bool = False,
        collect_diagnostics: bool = False,
    ) -> LatentRolloutResult:
        if base_inputs_embeds.shape[0] != 1:
            raise NotImplementedError("Stage 2 rollout currently supports batch_size=1 only")
        current = base_inputs_embeds
        states = []
        prefix_lengths = []
        for span in latent_spans:
            for position in span.latent_positions:
                outputs = language_model(
                    inputs_embeds=current[:, :position],
                    attention_mask=attention_mask[:, :position],
                    position_ids=full_position_ids[..., :position],
                    use_cache=False,
                    return_dict=True,
                )
                latent = outputs.last_hidden_state[:, -1, :]
                if retain_latent_grads:
                    latent.retain_grad()
                states.append(latent)
                prefix_lengths.append(position)
                current = _replace_embedding(current, position, latent)
        return LatentRolloutResult(current, tuple(states), tuple(prefix_lengths))
```

Validate attention/position/embed sequence dimensions before the loop. Do not add checkpointing or cache handling.

- [ ] **Step 5: Run Task 1 and Task 2 tests**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_structure.py tests/stage2/test_rollout.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the recurrence**

```bash
git add transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py tests/stage2/helpers.py tests/stage2/test_rollout.py
git commit -m "feat: implement exact stage2 latent rollout"
```

### Task 3: Gradient Continuity, Pad Independence, and Diagnostics

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`
- Modify: `tests/stage2/test_rollout.py`

**Interfaces:**
- Consumes: Task 2 rollout result and `retain_latent_grads`/`collect_diagnostics` flags.
- Produces: `LatentRolloutDiagnostics(block_count, latent_l2_norms, adjacent_cosine_similarities)` with detached diagnostic tensors while preserving graph-connected `latent_states`.

- [ ] **Step 1: Write a failing downstream-loss gradient test**

```python
def test_downstream_loss_backpropagates_through_every_latent_state():
    embedding = nn.Embedding(128, 4)
    lm = ToyCausalLanguageModel(4)
    ids = torch.tensor([[1, 91, 90, 90, 92, 2]])
    spans = parse_latent_spans(ids[0], token_ids=TOKENS, latent_size=2)
    positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    result = LatentRolloutEngine().rollout(
        language_model=lm,
        base_inputs_embeds=embedding(ids),
        attention_mask=torch.ones_like(ids),
        full_position_ids=positions,
        latent_spans=spans,
        retain_latent_grads=True,
    )
    final_hidden = lm(
        inputs_embeds=result.final_inputs_embeds,
        attention_mask=torch.ones_like(ids),
        position_ids=positions,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    final_hidden[:, -1].square().mean().backward()
    assert all(state.grad is not None and torch.isfinite(state.grad).all() for state in result.latent_states)
    assert embedding.weight.grad is not None
    assert lm.scale.grad is not None and lm.scale.grad.abs() > 0
```

- [ ] **Step 2: Add pad-embedding independence and detached-diagnostics tests**

```python
def test_latent_pad_embedding_does_not_define_continuous_latents():
    first = run_toy_rollout_with_pad_embedding(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    second = run_toy_rollout_with_pad_embedding(torch.tensor([1e6, -1e6, 5e5, -5e5]))
    assert all(torch.allclose(a, b) for a, b in zip(first.latent_states, second.latent_states))
    assert torch.allclose(first.final_logits, second.final_logits)


def test_diagnostics_are_observability_only():
    result = run_toy_rollout(collect_diagnostics=True)
    assert result.diagnostics.block_count == 1
    assert result.diagnostics.latent_l2_norms.requires_grad is False
    assert result.diagnostics.adjacent_cosine_similarities.requires_grad is False
```

The helper rebuilds token embeddings after each pad-row perturbation so the test proves overwrite independence.

- [ ] **Step 3: Run the new tests and confirm diagnostics are missing**

Run Task 2's focused pytest command. Expected: gradient and pad tests exercise the graph; diagnostics fails because `LatentRolloutDiagnostics` is absent.

- [ ] **Step 4: Add detached metrics without changing rollout values**

```python
@dataclass(frozen=True)
class LatentRolloutDiagnostics:
    block_count: int
    latent_l2_norms: torch.Tensor
    adjacent_cosine_similarities: torch.Tensor


def _build_diagnostics(states, block_count):
    stacked = torch.stack(states, dim=1)
    norms = torch.linalg.vector_norm(stacked.float(), dim=-1).detach()
    adjacent = torch.nn.functional.cosine_similarity(
        stacked[:, :-1].float(), stacked[:, 1:].float(), dim=-1
    ).detach()
    return LatentRolloutDiagnostics(block_count, norms, adjacent)
```

Return `None` when disabled. For zero or one latent state, return a correctly shaped empty adjacent-cosine tensor.

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_rollout.py -q
git add transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py tests/stage2/test_rollout.py
git commit -m "test: verify stage2 latent gradients and diagnostics"
```

### Task 4: Qwen Stage 2 Forward and MRoPE Delegation

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1620-1655,1733-2315`
- Create: `tests/stage2/test_qwen_forward.py`

**Interfaces:**
- Consumes: Task 2 engine, Task 1 parser/types, and Qwen `get_rope_index`.
- Produces: `_forward_stage2`, `prepare_stage2_base_inputs_embeds`, `stage2_latent_states`, and `stage2_diagnostics`.

- [ ] **Step 1: Add a tiny Qwen factory and recording language model**

```python
def tiny_stage2_qwen():
    config = Qwen2_5_VLConfig(
        vocab_size=128, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        vision_config={
            "depth": 1, "hidden_size": 8, "intermediate_size": 16,
            "num_heads": 2, "patch_size": 2, "spatial_merge_size": 1,
            "temporal_patch_size": 1, "window_size": 2,
            "out_hidden_size": 8, "fullatt_block_indexes": [0],
        },
        image_token_id=80, vision_start_token_id=81, video_token_id=82,
        stage="stage2", latent_size=2,
        latent_pad_token_id=90, latent_start_token_id=91, latent_end_token_id=92,
        attn_implementation="eager",
    )
    return Qwen2_5_VLForConditionalGeneration(config)
```

Replace `model.model` with `ToyCausalLanguageModel(8)` in the position test while retaining the real `get_rope_index`.

- [ ] **Step 2: Write failing delegation and MRoPE tests**

```python
def test_stage2_forward_slices_one_full_mrope_tensor(monkeypatch):
    model = tiny_stage2_qwen()
    recorder = ToyCausalLanguageModel(8)
    model.model = recorder
    ids = torch.tensor([[1, 81, 80, 80, 80, 80, 2, 91, 90, 90, 92, 3]])
    mask = torch.ones_like(ids)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    expected_positions, _ = model.get_rope_index(
        ids, image_grid_thw=image_grid_thw, attention_mask=mask
    )
    rope_calls = 0
    original_get_rope_index = model.get_rope_index

    def record_get_rope_index(*args, **kwargs):
        nonlocal rope_calls
        rope_calls += 1
        return original_get_rope_index(*args, **kwargs)

    monkeypatch.setattr(model, "get_rope_index", record_get_rope_index)

    outputs = model(
        input_ids=ids,
        attention_mask=mask,
        image_grid_thw=image_grid_thw,
        labels=ids.masked_fill(ids == 90, -100),
        return_stage2_latent_states=True,
    )

    assert len(outputs.stage2_latent_states) == 2
    assert rope_calls == 1
    assert torch.equal(recorder.calls[0][2], expected_positions[..., :8])
    assert torch.equal(recorder.calls[1][2], expected_positions[..., :9])
```

- [ ] **Step 3: Run and confirm the old forward lacks the new contract**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_qwen_forward.py -q
```

Expected: failure for missing explicit Stage 2 output/delegation.

- [ ] **Step 4: Replace the experimental recurrence with one wrapper branch**

Add an early branch:

```python
if getattr(self.config, "stage", None) == "stage2":
    return self._forward_stage2(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        return_dict=return_dict,
        return_stage2_latent_states=return_stage2_latent_states,
        collect_stage2_diagnostics=collect_stage2_diagnostics,
        retain_stage2_latent_grads=retain_stage2_latent_grads,
    )
```

Inside `_forward_stage2`, reject helper inputs, construct canonical `LatentTokenIds`, parse one sequence, call `get_rope_index` once when positions are absent, invoke the engine with a callable equivalent to `functools.partial(self.model, generate_mode=True)`, run one final full language-model forward, apply `lm_head`, and compute standard shifted CE. Delete the former `if self.config.stage == 'stage2'` recurrence block.

- [ ] **Step 5: Extend output fields without retaining graphs by default**

```python
stage2_latent_states: Optional[Tuple[torch.FloatTensor, ...]] = None
stage2_diagnostics: Optional[Dict[str, torch.Tensor]] = None
```

Populate latent states only when requested and diagnostics only from detached values.

- [ ] **Step 6: Run focused tests and imports**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_structure.py tests/stage2/test_rollout.py tests/stage2/test_qwen_forward.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -c 'from transformers import Qwen2_5_VLForConditionalGeneration; import main'
```

- [ ] **Step 7: Commit the authoritative model path**

```bash
git add transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py tests/stage2/test_qwen_forward.py
git commit -m "feat: route qwen stage2 through rollout engine"
```

### Task 5: Visual-Once, Frozen-Visual, and Final CE Semantics

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Modify: `tests/stage2/test_qwen_forward.py`

**Interfaces:**
- Consumes: Task 4 `_forward_stage2` and `prepare_stage2_base_inputs_embeds`.
- Produces: one-call frozen visual preparation and graph-connected final CE.

- [ ] **Step 1: Add a counting visual stub**

```python
class CountingVisual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.calls = 0

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, pixel_values, grid_thw):
        self.calls += 1
        return self.weight.unsqueeze(0).expand(grid_thw.shape[0], -1)
```

- [ ] **Step 2: Write failing visual and gradient tests**

```python
def test_stage2_encodes_frozen_visual_once_and_backpropagates_through_lm():
    model = tiny_stage2_qwen()
    model.model = ToyCausalLanguageModel(8)
    model.visual = CountingVisual(8)
    model.visual.requires_grad_(False)
    ids = torch.tensor([[81, 80, 1, 91, 90, 90, 92, 2]])
    labels = ids.clone()
    labels[:, :3] = -100
    labels[ids == 90] = -100
    outputs = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
        pixel_values=torch.ones(1, 3),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
    )
    outputs.loss.backward()
    assert model.visual.calls == 1
    assert all(not p.requires_grad and p.grad is None for p in model.visual.parameters())
    assert model.model.scale.grad is not None
```

Also assert recurrent language-model calls never receive pixel tensors.

- [ ] **Step 3: Run and observe the visual invariant failure**

Run `test_qwen_forward.py`. Expected: failure until base-input preparation owns a single `no_grad` visual call.

- [ ] **Step 4: Implement one visual call and strict count matching**

```python
def prepare_stage2_base_inputs_embeds(self, *, input_ids, pixel_values, image_grid_thw):
    inputs_embeds = self.model.embed_tokens(input_ids)
    if pixel_values is None:
        return inputs_embeds
    with torch.no_grad():
        image_embeds = self.visual(pixel_values.to(dtype=self.visual.dtype), grid_thw=image_grid_thw)
    image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    if int(image_mask[..., 0].sum()) != image_embeds.shape[0]:
        raise ValueError("Stage 2 image token/feature count mismatch")
    return inputs_embeds.masked_scatter(
        image_mask,
        image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
    )
```

Reject `pixel_values_latent`, `image_grid_thw_latent`, `latent_hidden_states`, and `image_out_mask` in Stage 2 with a message naming the leaked key.

- [ ] **Step 5: Run and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_rollout.py tests/stage2/test_qwen_forward.py -q
git add transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py tests/stage2/test_qwen_forward.py
git commit -m "feat: isolate stage2 visual and loss path"
```

### Task 6: Checkpoint Metadata and Fresh-vs-Resume Decision

**Files:**
- Create: `src/stage2/__init__.py`
- Create: `src/stage2/checkpointing.py`
- Create: `tests/stage2/test_checkpointing.py`

**Interfaces:**
- Consumes: Task 1 `LatentTokenIds`, a tokenizer with `convert_tokens_to_ids`, and a config.
- Produces: `resolve_latent_token_ids`, `configure_stage2_metadata`, `Stage2LoadDecision`, and `resolve_stage2_load_decision`.

- [ ] **Step 1: Write failing token and checkpoint tests**

```python
def test_resolve_latent_ids_requires_tokenizer_config_parity():
    tokenizer = FakeTokenizer({
        "<|latent_pad|>": 10, "<|latent_start|>": 11, "<|latent_end|>": 12,
    })
    config = SimpleNamespace(latent_token_id=10, latent_start_id=11, latent_end_id=99)
    with pytest.raises(ValueError, match="latent_end_token_id.*tokenizer=12.*config=99"):
        resolve_latent_token_ids(tokenizer, config)


def test_fresh_stage2_resets_trainer_state():
    decision = resolve_stage2_load_decision(
        load_model_path="stage1/model",
        resume_from_checkpoint_path=None,
        load_config=lambda _: SimpleNamespace(stage="stage1"),
    )
    assert decision.model_source == "stage1/model"
    assert decision.trainer_resume_path is None
```

Add a test that a resume source with `stage="stage1"` fails.

- [ ] **Step 2: Run and confirm module import failure**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_checkpointing.py -q
```

- [ ] **Step 3: Implement canonical and legacy validation**

```python
LATENT_TOKEN_STRINGS = {
    "pad_token_id": "<|latent_pad|>",
    "start_token_id": "<|latent_start|>",
    "end_token_id": "<|latent_end|>",
}

LATENT_CONFIG_FIELDS = {
    "pad_token_id": ("latent_pad_token_id", "latent_token_id"),
    "start_token_id": ("latent_start_token_id", "latent_start_id"),
    "end_token_id": ("latent_end_token_id", "latent_end_id"),
}


def resolve_latent_token_ids(tokenizer, config) -> LatentTokenIds:
    resolved: dict[str, int] = {}
    for result_field, token in LATENT_TOKEN_STRINGS.items():
        tokenizer_id = tokenizer.convert_tokens_to_ids(token)
        if tokenizer_id is None or (
            tokenizer.unk_token_id is not None
            and tokenizer_id == tokenizer.unk_token_id
            and token != tokenizer.unk_token
        ):
            raise ValueError(f"tokenizer does not contain {token}")
        for config_field in LATENT_CONFIG_FIELDS[result_field]:
            config_id = getattr(config, config_field, None)
            if config_id is not None and int(config_id) != int(tokenizer_id):
                raise ValueError(
                    f"{config_field} mismatch: tokenizer={tokenizer_id}, config={config_id}"
                )
        resolved[result_field] = int(tokenizer_id)
    return LatentTokenIds(**resolved)
```

Implement metadata assignment explicitly:

```python
def configure_stage2_metadata(
    config,
    token_ids: LatentTokenIds,
    *,
    latent_size: int,
    supervise_latent_boundaries: bool,
) -> None:
    config.stage = "stage2"
    config.latent_size = latent_size
    config.supervise_latent_boundaries = supervise_latent_boundaries
    config.latent_pad_token_id = config.latent_token_id = token_ids.pad_token_id
    config.latent_start_token_id = config.latent_start_id = token_ids.start_token_id
    config.latent_end_token_id = config.latent_end_id = token_ids.end_token_id
```

- [ ] **Step 4: Implement explicit load decisions**

```python
@dataclass(frozen=True)
class Stage2LoadDecision:
    model_source: str
    processor_source: str
    trainer_resume_path: str | None


def resolve_stage2_load_decision(*, load_model_path, resume_from_checkpoint_path, load_config):
    if resume_from_checkpoint_path is None:
        if getattr(load_config(load_model_path), "stage", None) != "stage1":
            raise ValueError("fresh Stage 2 requires a stage1 checkpoint")
        return Stage2LoadDecision(load_model_path, load_model_path, None)
    if getattr(load_config(resume_from_checkpoint_path), "stage", None) != "stage2":
        raise ValueError("expected stage2 checkpoint for resume")
    return Stage2LoadDecision(
        resume_from_checkpoint_path,
        resume_from_checkpoint_path,
        resume_from_checkpoint_path,
    )
```

- [ ] **Step 5: Run and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_checkpointing.py -q
git add src/stage2 tests/stage2/test_checkpointing.py
git commit -m "feat: validate stage2 checkpoint transitions"
```

### Task 7: CoMT Trajectory Conversion and Loss Labels

**Files:**
- Create: `src/stage2/data.py`
- Create: `tests/stage2/test_data.py`

**Interfaces:**
- Consumes: raw CoMT dictionaries and `LatentTokenIds`.
- Produces: `load_jsonl_examples`, `build_stage2_conversation`, `prepare_stage2_example`, `build_stage2_labels`, and `Stage2PreparedExample`.

- [ ] **Step 1: Write failing CoMT conversion tests**

```python
def test_helper_steps_become_latent_blocks_without_helper_images(tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    sample = {
        "image_input": ["input.png"],
        "text_input": "question",
        "sequence_plan": [
            {"type": "text", "content": "A"},
            {"type": "latent", "helper_image": "helper.png"},
            {"type": "text", "content": "B"},
        ],
    }
    conversation = build_stage2_conversation(sample, image_root=tmp_path, latent_size=2)
    assistant = conversation[1]["content"][0]["text"]
    assert assistant == "A<|latent_start|><|latent_pad|><|latent_pad|><|latent_end|>B"
    assert all(item["type"] != "image" for item in conversation[1]["content"])


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda sample: sample.update(sequence_plan="bad"), "sequence_plan must be a list"),
        (lambda sample: sample["sequence_plan"].append({"type": "latent"}),
         "latent step is missing helper_image"),
        (lambda sample: sample["sequence_plan"].append({"type": "unknown"}),
         "unsupported sequence_plan step type"),
    ],
)
def test_rejects_malformed_comt_trajectory(valid_sample, mutation, message, tmp_path):
    mutation(valid_sample)
    with pytest.raises(ValueError, match=message):
        build_stage2_conversation(valid_sample, image_root=tmp_path, latent_size=2)


def test_rejects_missing_original_image(valid_sample, tmp_path):
    valid_sample["image_input"] = "missing.png"
    with pytest.raises(FileNotFoundError, match="missing original image"):
        build_stage2_conversation(valid_sample, image_root=tmp_path, latent_size=2)
```

- [ ] **Step 2: Write atomic boundary-mask tests**

```python
@pytest.mark.parametrize("supervise, expected", [(True, [91, 92]), (False, [-100, -100])])
def test_stage2_labels_mask_slots_and_pair_boundaries(supervise, expected):
    ids = torch.tensor([[5, 6, 91, 90, 90, 92, 7, 8]])
    labels = build_stage2_labels(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        assistant_content_start=2,
        token_ids=TOKENS,
        latent_size=2,
        supervise_latent_boundaries=supervise,
    )
    assert labels[0, 3:5].tolist() == [-100, -100]
    assert [labels[0, 2].item(), labels[0, 5].item()] == expected
    assert torch.all(labels[0, :2] == -100)
```

- [ ] **Step 3: Run and confirm missing data API**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_data.py -q
```

- [ ] **Step 4: Implement exact trajectory construction**

```python
@dataclass(frozen=True)
class Stage2PreparedExample:
    conversation: list[dict]
    original_image_paths: tuple[str, ...]
    latent_block_count: int


def load_jsonl_examples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_stage2_example(sample, *, image_root: Path, latent_size: int) -> Stage2PreparedExample:
    conversation = build_stage2_conversation(sample, image_root=image_root, latent_size=latent_size)
    return Stage2PreparedExample(
        conversation=conversation,
        original_image_paths=tuple(resolve_original_images(sample, image_root)),
        latent_block_count=sum(step.get("type") == "latent" for step in sample["sequence_plan"]),
    )


def resolve_original_images(sample: dict, image_root: Path) -> tuple[str, ...]:
    raw_images = sample.get("image_input")
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("image_input must contain at least one original image")
    paths = tuple(str(image_root / raw) for raw in raw_images)
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing original image: {missing[0]}")
    return paths


def build_stage2_conversation(sample: dict, *, image_root: Path, latent_size: int) -> list[dict]:
    if not isinstance(sample.get("sequence_plan"), list):
        raise ValueError("sequence_plan must be a list")
    original_images = resolve_original_images(sample, image_root)
    latent_block = (
        "<|latent_start|>" + "<|latent_pad|>" * latent_size + "<|latent_end|>"
    )
    assistant_parts: list[str] = []
    for step in sample["sequence_plan"]:
        if step.get("type") == "text":
            content = step.get("content")
            if not isinstance(content, str):
                raise ValueError("text step content must be a string")
            assistant_parts.append(content)
        elif step.get("type") == "latent":
            if not step.get("helper_image"):
                raise ValueError("latent step is missing helper_image")
            assistant_parts.append(latent_block)
        else:
            raise ValueError(f"unsupported sequence_plan step type: {step.get('type')}")
    user_content = [{"type": "image", "image": path} for path in original_images]
    user_content.append({"type": "text", "text": sample["text_input"]})
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": "".join(assistant_parts)}]},
    ]
```

This preserves every text segment exactly, requires but does not open helper-image paths, and resolves/verifies only original images.

- [ ] **Step 5: Implement labels through validated spans**

```python
def build_stage2_labels(
    *, input_ids, attention_mask, assistant_content_start,
    token_ids, latent_size, supervise_latent_boundaries,
):
    if input_ids.shape[0] != 1:
        raise NotImplementedError("Stage 2 labels currently support batch_size=1 only")
    labels = input_ids.clone()
    labels[:, :assistant_content_start] = -100
    labels.masked_fill_(attention_mask == 0, -100)
    spans = parse_latent_spans(
        input_ids[0], token_ids=token_ids, latent_size=latent_size
    )
    for span in spans:
        labels[0, list(span.latent_positions)] = -100
        if not supervise_latent_boundaries:
            labels[0, span.start_position] = -100
            labels[0, span.end_position] = -100
    return labels
```

Do not use `latent_ce_ratio`.

- [ ] **Step 6: Run and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_structure.py tests/stage2/test_data.py -q
git add src/stage2/data.py tests/stage2/test_data.py
git commit -m "feat: add comt stage2 trajectories and labels"
```

### Task 8: Stage 2 Collator and Whole-Sample Length Filter

**Files:**
- Modify: `src/stage2/data.py`
- Modify: `tests/stage2/test_data.py`

**Interfaces:**
- Consumes: Task 7 conversations and a Qwen processor.
- Produces: `Stage2DataCollator`, `Stage2LengthStats`, `find_unique_subsequence_end`, `build_prepared_example_length_measure`, and `filter_stage2_examples`.

- [ ] **Step 1: Write failing collator tests with a recording processor**

```python
def test_stage2_collator_emits_only_original_visual_tensors(prepared_example):
    collator = make_collator()
    batch = collator([prepared_example])
    assert "pixel_values" in batch
    assert "image_grid_thw" in batch
    assert "pixel_values_latent" not in batch
    assert "image_grid_thw_latent" not in batch
    assert batch["input_ids"].shape[0] == 1
    assert collator.processor.calls[-1]["truncation"] is False
    assert collator.processor.calls[-1]["images"] == [
        list(prepared_example.original_image_paths)
    ]


def test_stage2_collator_rejects_two_samples(prepared_example):
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        make_collator()([prepared_example, prepared_example])


def test_stage2_collator_revalidates_tokenized_structure(prepared_example):
    collator = make_collator(overridden_ids=torch.tensor([[1, 91, 90, 92, 2]]))
    with pytest.raises(LatentStructureError, match="expected 2 latent_pad tokens"):
        collator([prepared_example])
```

- [ ] **Step 2: Write failing filter/statistics tests**

```python
def test_length_filter_drops_whole_samples_and_reports_distribution():
    kept, stats = filter_stage2_examples(
        examples=[{"length": 4}, {"length": 8}, {"length": 12}],
        measure_length=lambda item: item["length"],
        max_length=8,
    )
    assert [item["length"] for item in kept] == [4, 8]
    assert (stats.total, stats.filtered) == (3, 1)
    assert stats.filtered_percentage == pytest.approx(100 / 3)
    assert stats.maximum == 12
```

- [ ] **Step 3: Run and confirm missing classes**

Run `test_data.py`. Expected: missing collator/filter APIs.

- [ ] **Step 4: Implement non-truncating collation and assistant boundary discovery**

```python
class Stage2DataCollator:
    def __init__(self, processor, token_ids, latent_size, supervise_latent_boundaries):
        self.processor = processor
        self.token_ids = token_ids
        self.latent_size = latent_size
        self.supervise_latent_boundaries = supervise_latent_boundaries
        self.assistant_pattern = processor.tokenizer(
            "<|im_start|>assistant", add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

    def __call__(self, examples: list[Stage2PreparedExample]) -> dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise NotImplementedError("Stage 2 collator currently supports batch_size=1 only")
        example = examples[0]
        text = self.processor.apply_chat_template(
            example.conversation, tokenize=False, add_generation_prompt=False
        )
        batch = self.processor(
            text=[text], images=[list(example.original_image_paths)],
            padding=True, truncation=False, return_tensors="pt",
        )
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & batch.keys()
        if forbidden:
            raise ValueError(f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}")
        parse_latent_spans(batch["input_ids"][0], token_ids=self.token_ids,
                           latent_size=self.latent_size)
        assistant_start = find_unique_subsequence_end(
            batch["input_ids"][0], self.assistant_pattern
        )
        batch["labels"] = build_stage2_labels(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            assistant_content_start=assistant_start, token_ids=self.token_ids,
            latent_size=self.latent_size,
            supervise_latent_boundaries=self.supervise_latent_boundaries,
        )
        return batch
```

Implement the supported two-turn CoMT assistant-content boundary exactly:

```python
def find_unique_subsequence_end(sequence: torch.Tensor, pattern: torch.Tensor) -> int:
    matches = [
        start + pattern.numel()
        for start in range(sequence.numel() - pattern.numel() + 1)
        if torch.equal(sequence[start:start + pattern.numel()], pattern)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one assistant marker, found {len(matches)}")
    return matches[0]
```

Add the reusable non-truncating length closure used by Task 9:

```python
def build_prepared_example_length_measure(processor):
    def measure(example: Stage2PreparedExample) -> int:
        rendered = processor.apply_chat_template(
            example.conversation, tokenize=False, add_generation_prompt=False
        )
        encoded = processor(
            text=[rendered], images=[list(example.original_image_paths)],
            padding=False, truncation=False, return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[1])
    return measure
```

- [ ] **Step 5: Implement one-pass statistics**

```python
@dataclass(frozen=True)
class Stage2LengthStats:
    total: int
    retained: int
    filtered: int
    filtered_percentage: float
    minimum: int
    median: float
    p90: float
    p95: float
    p99: float
    maximum: int
    latent_block_distribution: dict[int, int]


def filter_stage2_examples(*, examples, measure_length, max_length):
    kept = []
    lengths: list[int] = []
    block_counts: Counter[int] = Counter()
    for example in examples:
        length = int(measure_length(example))
        lengths.append(length)
        block_count = (
            example.latent_block_count
            if isinstance(example, Stage2PreparedExample)
            else int(example.get("latent_block_count", sum(
                step.get("type") == "latent" for step in example.get("sequence_plan", [])
            )))
        )
        block_counts[block_count] += 1
        if length <= max_length:
            kept.append(example)
    if not lengths:
        raise ValueError("cannot compute Stage 2 length statistics for an empty dataset")
    values = torch.tensor(lengths, dtype=torch.float64)
    filtered = len(lengths) - len(kept)
    stats = Stage2LengthStats(
        total=len(lengths), retained=len(kept), filtered=filtered,
        filtered_percentage=100.0 * filtered / len(lengths),
        minimum=min(lengths), median=float(values.quantile(0.50)),
        p90=float(values.quantile(0.90)), p95=float(values.quantile(0.95)),
        p99=float(values.quantile(0.99)), maximum=max(lengths),
        latent_block_distribution=dict(sorted(block_counts.items())),
    )
    return kept, stats
```

Import `Counter` from `collections`; log `asdict(stats)` once after this pass.

- [ ] **Step 6: Run and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_data.py -q
git add src/stage2/data.py tests/stage2/test_data.py
git commit -m "feat: add stage2 collator and length filtering"
```

### Task 9: Thin Stage 2 Trainer and Explicit Main Routing

**Files:**
- Create: `src/stage2/trainer.py`
- Modify: `src/main.py:1-255`
- Modify: `src/utils_deepseed.py:15-63`
- Create: `tests/stage2/test_routing.py`

**Interfaces:**
- Consumes: Tasks 6-8 checkpoint/data APIs and model CE/diagnostics.
- Produces: `Stage2Trainer`, `TrainingPath`, `Stage2TrainingComponents`, `select_training_path`, `parse_args`, and `build_stage2_training_components`.

- [ ] **Step 1: Write failing trainer and routing tests**

```python
def test_stage2_trainer_uses_model_ce_without_teacher():
    trainer = object.__new__(Stage2Trainer)
    trainer.model = FakeStage2Model(loss=torch.tensor(2.0, requires_grad=True))
    loss, outputs = trainer.compute_loss(
        trainer.model, {"input_ids": torch.ones(1, 3)}, return_outputs=True
    )
    assert loss.item() == 2.0
    assert not hasattr(trainer, "_ema")
    assert "pixel_values_latent" not in trainer.model.last_inputs


def test_stage_routing_selects_independent_trainers():
    assert select_training_path("stage1").trainer_class is CustomTrainerStage1
    assert select_training_path("stage1").collator_factory is collate_fn_stage1
    assert select_training_path("stage2").trainer_class is Stage2Trainer
    assert select_training_path("stage2").collator_factory is Stage2DataCollator


def test_boundary_supervision_is_one_default_true_boolean():
    assert parse_args(["--stage", "stage2"]).supervise_latent_boundaries is True
    assert parse_args([
        "--stage", "stage2", "--no-supervise_latent_boundaries"
    ]).supervise_latent_boundaries is False
```

- [ ] **Step 2: Run and confirm Stage 2 remains hard-coded**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_routing.py -q
```

- [ ] **Step 3: Add Stage 2 CLI without changing Stage 1 defaults**

Add these exact parser arguments while leaving every existing Stage 1 default untouched:

```python
parser.add_argument("--supervise_latent_boundaries", action=argparse.BooleanOptionalAction,
                    default=True)
parser.add_argument("--save_final_model", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--image_root", required=False)
parser.add_argument("--max_train_samples", type=int)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                    default=False)
parser.add_argument("--attention_implementation",
                    choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
parser.add_argument("--deepspeed_config")
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
```

Keep the repository's existing `--resume_from_checkpoint_path` definition once. After parsing, reject Stage 2 LoRA and any local batch size other than one with explicit `ValueError`s.

Expose `parse_args(argv: list[str] | None = None) -> argparse.Namespace`; keep `get_args()` as the backward-compatible `parse_args()` wrapper used by Stage 1.

- [ ] **Step 4: Implement explicit path selection**

```python
@dataclass(frozen=True)
class TrainingPath:
    trainer_class: type
    collator_factory: object


def select_training_path(stage: str) -> TrainingPath:
    if stage == "stage1":
        return TrainingPath(CustomTrainerStage1, collate_fn_stage1)
    if stage == "stage2":
        return TrainingPath(Stage2Trainer, Stage2DataCollator)
    raise ValueError(f"Unsupported training stage: {stage}")
```

Define the exact composition result:

```python
@dataclass(frozen=True)
class Stage2TrainingComponents:
    model: object
    processor: object
    train_dataset: object
    data_collator: Stage2DataCollator
    trainer_class: type[Stage2Trainer]
    trainer_resume_path: str | None
```

Implement the composition root with these operations and no optimizer/scheduler load in the fresh branch:

```python
def build_stage2_training_components(args) -> Stage2TrainingComponents:
    decision = resolve_stage2_load_decision(
        load_model_path=args.load_model_path,
        resume_from_checkpoint_path=args.resume_from_checkpoint_path,
        load_config=AutoConfig.from_pretrained,
    )
    processor = AutoProcessor.from_pretrained(decision.processor_source)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        decision.model_source,
        attn_implementation=args.attention_implementation,
    )
    token_ids = resolve_latent_token_ids(processor.tokenizer, model.config)
    configure_stage2_metadata(
        model.config, token_ids, latent_size=args.latent_size,
        supervise_latent_boundaries=args.supervise_latent_boundaries,
    )
    model.config.use_cache = False
    model.visual.requires_grad_(False)
    raw_examples = load_jsonl_examples(Path(args.data_path))
    prepared = [
        prepare_stage2_example(item, image_root=Path(args.image_root),
                               latent_size=args.latent_size)
        for item in raw_examples[:args.max_train_samples]
    ]
    kept, length_stats = filter_stage2_examples(
        examples=prepared,
        measure_length=build_prepared_example_length_measure(processor),
        max_length=args.max_seq_length_train,
    )
    collator = Stage2DataCollator(
        processor, token_ids, args.latent_size, args.supervise_latent_boundaries
    )
    return Stage2TrainingComponents(
        model=model, processor=processor, train_dataset=kept,
        data_collator=collator, trainer_class=Stage2Trainer,
        trainer_resume_path=decision.trainer_resume_path,
    )
```

Task 8 adds `build_prepared_example_length_measure(processor)`, which renders and processes one `Stage2PreparedExample` with `padding=False`, `truncation=False`, and returns `input_ids.shape[1]`. Log `length_stats` once. Treat `max_train_samples=None` as the full list rather than applying the shown slice literally.

In `main`, keep the current Stage 1 body byte-for-byte on the Stage 1 branch. Construct and run Stage 2 explicitly so only a validated Stage 2 resume restores Trainer state:

```python
if args.stage == "stage2":
    components = build_stage2_training_components(args)
    trainer = components.trainer_class(
        model=components.model,
        args=build_stage2_training_arguments(args),
        train_dataset=components.train_dataset,
        data_collator=components.data_collator,
        processing_class=components.processor,
    )
    trainer.train(resume_from_checkpoint=components.trainer_resume_path)
    if args.save_final_model:
        trainer.save_model(args.save_model_path)
        components.processor.save_pretrained(args.save_model_path)
    return
```

`build_stage2_training_arguments(args) -> TrainingArguments` maps existing epoch, learning-rate, logging, gradient accumulation, DeepSpeed, and output arguments; it forces `remove_unused_columns=False`, passes `max_steps`, and does not accept a Stage 1 optimizer or scheduler object.

```python
def build_stage2_training_arguments(args) -> SFTConfig:
    return SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warm_up_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=1,
        optim="adamw_torch", bf16=True, report_to=[],
        remove_unused_columns=False,
        dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length_train,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=False,
    )
```

- [ ] **Step 5: Implement the thin Trainer override**

```python
class Stage2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & inputs.keys()
        if forbidden:
            raise ValueError(f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}")
        outputs = model(**inputs, collect_stage2_diagnostics=True)
        self._latest_stage2_diagnostics = outputs.stage2_diagnostics
        return (outputs.loss, outputs) if return_outputs else outputs.loss
```

The override must reject helper-image keys before calling the model and never alter `outputs.loss`.

- [ ] **Step 6: Add a Stage 1 collator regression**

```python
def test_stage1_collator_contract_is_unchanged(stage1_example, recording_processor):
    batch = collate_fn_stage1([stage1_example], processor=recording_processor,
                              args=stage1_args())
    assert "pixel_values_latent" in batch
    assert "image_grid_thw_latent" in batch
    assert torch.equal(batch["labels"], expected_stage1_labels())
```

- [ ] **Step 7: Run and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_checkpointing.py tests/stage2/test_data.py tests/stage2/test_routing.py -q
git add src/main.py src/utils_deepseed.py src/stage2/trainer.py tests/stage2/test_routing.py
git commit -m "feat: route stage2 training independently"
```

### Task 10: Worktree-Safe Launcher and One-Step Training Smoke

**Files:**
- Create: `run_stage2_training.sh`
- Modify: `README.md`
- Create: `tests/stage2/test_launcher.py`

**Interfaces:**
- Consumes: Task 9 CLI and checkpoint semantics.
- Produces: a worktree-safe launcher and one realistic Stage 1-to-Stage 2 optimizer-step witness.

- [ ] **Step 1: Write a failing launcher contract test**

```python
def test_stage2_launcher_uses_repo_venv_and_local_pythonpath():
    script = Path("run_stage2_training.sh").read_text()
    assert "./venv/ilvr-venv/bin/python" in script
    assert 'PYTHONPATH="$REPO_ROOT/transformers/src:$REPO_ROOT/src"' in script
    assert "--stage stage2" in script
    assert "--load_model_path" in script
    assert "--resume_from_checkpoint_path" in script
```

- [ ] **Step 2: Run and confirm the launcher is absent**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_launcher.py -q
```

- [ ] **Step 3: Add an argument-driven launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONPATH="$REPO_ROOT/transformers/src:$REPO_ROOT/src"
export TOKENIZERS_PARALLELISM=false

: "${STAGE1_CHECKPOINT:?set STAGE1_CHECKPOINT}"
: "${DATA_PATH:?set DATA_PATH}"
: "${IMAGE_ROOT:?set IMAGE_ROOT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${LOG_FILE:?set LOG_FILE}"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint_path "$RESUME_CHECKPOINT")
fi

exec "$REPO_ROOT/venv/ilvr-venv/bin/python" "$REPO_ROOT/src/main.py" \
  --stage stage2 \
  --load_model_path "$STAGE1_CHECKPOINT" \
  --data_path "$DATA_PATH" \
  --image_root "$IMAGE_ROOT" \
  --save_model_path "$OUTPUT_DIR" \
  --log_file "$LOG_FILE" \
  "${RESUME_ARGS[@]}" \
  "$@"
```

Do not reinterpret `STAGE1_CHECKPOINT` when `RESUME_CHECKPOINT` is set.

- [ ] **Step 4: Document fresh/resume and limitations**

Document rank-local `B=1`, effective batch formula, boundary default, helper exclusion, fresh optimizer reset, explicit resume, and no-cache recurrence.

- [ ] **Step 5: Run cheap verification**

```bash
bash -n run_stage2_training.sh
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_launcher.py tests/stage2/test_routing.py -q
```

- [ ] **Step 6: Check for equivalent smoke evidence**

Inspect the current output/log target for a completed one-step Stage 2 optimizer update from the same commit/settings. Reuse it if present. Do not hash or scan the 30 GB checkpoint tree.

- [ ] **Step 7: Run one large-checkpoint optimizer step only when needed**

```bash
STAGE2_SMOKE_DIR=$(mktemp -d)
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/python src/main.py \
  --stage stage2 \
  --load_model_path ../../checkpoints/comt_ckpt \
  --data_path ../../data/TRAIN.jsonl \
  --image_root ../../data \
  --save_model_path "$STAGE2_SMOKE_DIR/output" \
  --log_file "$STAGE2_SMOKE_DIR/stage2.log" \
  --latent_size 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_samples 1 \
  --max_steps 1 \
  --attention_implementation sdpa \
  --no-save_final_model
```

Expected: validated Stage 1 origin, Stage 2 config, one finite CE loss and optimizer/scheduler step, frozen visual parameters, no helper tensor, and exit 0. Stop after one success.

- [ ] **Step 8: Commit launcher and docs**

```bash
git add run_stage2_training.sh README.md tests/stage2/test_launcher.py
git commit -m "docs: add stage2 training launcher"
```

### Task 11: Model-Specific No-Cache Inference and Parity

**Files:**
- Create: `src/stage2/inference.py`
- Create: `tests/stage2/test_inference.py`

**Interfaces:**
- Consumes: Task 2 engine, Task 4 base-input method, and Task 6 token IDs.
- Produces: `Stage2GenerationConfig`, `Stage2GenerationResult`, and `Stage2InferenceController.generate(*, model, input_ids, attention_mask, token_ids, generation_config, pixel_values=None, image_grid_thw=None, return_latent_states=False)`.

- [ ] **Step 1: Create a scripted fake model**

```python
class ScriptedHead(nn.Module):
    def __init__(self, token_ids, vocab_size=128):
        super().__init__()
        self.token_ids = list(token_ids)
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, hidden):
        token_id = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        logits = hidden.new_full((hidden.shape[0], self.vocab_size), -1e9)
        logits[:, token_id] = 0
        return logits


class ScriptedStage2Model(nn.Module):
    def __init__(self, *, emitted_ids, latent_size):
        super().__init__()
        self.config = SimpleNamespace(latent_size=latent_size)
        self.model = ToyCausalLanguageModel(4)
        self.lm_head = ScriptedHead(emitted_ids)
        self.stage2_rollout_engine = LatentRolloutEngine()
        self.visual_calls = 0

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_stage2_base_inputs_embeds(
        self, *, input_ids, pixel_values, image_grid_thw
    ):
        self.visual_calls += int(pixel_values is not None)
        return self.get_input_embeddings()(input_ids)

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return positions.view(1, 1, -1).expand(3, 1, -1), torch.zeros(1, 1)


def scripted_model(*, latent_size, first_token=None):
    emitted = (
        [first_token, 3] if first_token is not None
        else [4, TOKENS.start_token_id, 5, TOKENS.start_token_id, 3]
    )
    return ScriptedStage2Model(emitted_ids=emitted, latent_size=latent_size)
```

Pass one non-`None` fake pixel tensor in the multiple-block test so `visual_calls == 1`; the fixture's normal logits emit `text, latent_start, text, latent_start, eos`.

- [ ] **Step 2: Write failing state-machine and parity tests**

```python
def test_inference_forces_exactly_k_latents_and_supports_multiple_blocks():
    result = Stage2InferenceController().generate(
        model=scripted_model(latent_size=2),
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=16, eos_token_id=3),
        pixel_values=torch.ones(1, 3),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        return_latent_states=True,
    )
    assert count_blocks(result.sequences[0].tolist(), TOKENS) == [2, 2]
    assert len(result.latent_states_by_block) == 2
    assert scripted_model.visual_calls == 1


def test_started_block_finishes_past_nominal_budget():
    result = Stage2InferenceController().generate(
        model=scripted_model(first_token=TOKENS.start_token_id, latent_size=2),
        input_ids=torch.tensor([[1]]), attention_mask=torch.ones(1, 1),
        token_ids=TOKENS, generation_config=Stage2GenerationConfig(max_new_tokens=1),
    )
    assert result.sequences[0, -4:].tolist() == [91, 90, 90, 92]


def test_inference_rejects_batch_greater_than_one():
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        Stage2InferenceController().generate(
            model=scripted_model(latent_size=2), input_ids=torch.ones(2, 2, dtype=torch.long),
            attention_mask=torch.ones(2, 2), token_ids=TOKENS,
            generation_config=Stage2GenerationConfig(),
        )


def test_training_and_inference_share_identical_latent_states():
    training_model = scripted_model(first_token=TOKENS.start_token_id, latent_size=2)
    inference_model = scripted_model(first_token=TOKENS.start_token_id, latent_size=2)
    inference_model.load_state_dict(training_model.state_dict())
    scaffold = torch.tensor([[1, 91, 90, 90, 92]])
    mask = torch.ones_like(scaffold)
    positions, _ = training_model.get_rope_index(scaffold, attention_mask=mask)
    training = training_model.stage2_rollout_engine.rollout(
        language_model=functools.partial(training_model.model, generate_mode=True),
        base_inputs_embeds=training_model.prepare_stage2_base_inputs_embeds(
            input_ids=scaffold, pixel_values=None, image_grid_thw=None
        ),
        attention_mask=mask, full_position_ids=positions,
        latent_spans=(LatentSpan(1, (2, 3), 4),),
    ).latent_states
    inference = Stage2InferenceController().generate(
        model=inference_model, input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long), token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
        return_latent_states=True,
    ).latent_states_by_block[0]
    assert len(training) == len(inference) == 2
    for train_state, inference_state in zip(training, inference):
        assert torch.allclose(train_state, inference_state, atol=1e-6, rtol=1e-5)
```

The scripted fixture records vocabulary-selection calls; assert it is called for `<latent_start>` and later text only, never for any of the K latent slots or the forced end. Configure an EOS-favoring latent-step hidden state and assert the completed suffix still ends with `<latent_end>`.

- [ ] **Step 3: Run and confirm missing controller**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_inference.py -q
```

- [ ] **Step 4: Implement the two-state controller**

```python
@dataclass(frozen=True)
class Stage2GenerationConfig:
    max_new_tokens: int = 64
    max_sequence_length: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    eos_token_id: int | None = None


@dataclass(frozen=True)
class Stage2GenerationResult:
    sequences: torch.LongTensor
    latent_states_by_block: tuple[tuple[torch.Tensor, ...], ...]


class Stage2InferenceController:
    def generate(
        self,
        *,
        model,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_ids: LatentTokenIds,
        generation_config: Stage2GenerationConfig,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        return_latent_states: bool = False,
    ) -> Stage2GenerationResult:
        if input_ids.shape[0] != 1:
            raise NotImplementedError("Stage 2 inference currently supports batch_size=1 only")
        if generation_config.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        sequences = input_ids
        mask = attention_mask
        embeds = model.prepare_stage2_base_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        blocks: list[tuple[torch.Tensor, ...]] = []
        generated = 0
        while generated < generation_config.max_new_tokens:
            positions, _ = model.get_rope_index(
                sequences, image_grid_thw=image_grid_thw, attention_mask=mask
            )
            outputs = model.model(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
                return_dict=True,
                generate_mode=True,
            )
            logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
            next_id = _select_text_token(logits, generation_config)
            sequences = torch.cat((sequences, next_id[:, None]), dim=1)
            mask = torch.cat((mask, torch.ones_like(next_id[:, None])), dim=1)
            embeds = torch.cat((embeds, model.get_input_embeddings()(next_id)[:, None, :]), dim=1)
            generated += 1
            if int(next_id.item()) == token_ids.start_token_id:
                start = sequences.shape[1] - 1
                completed_length = sequences.shape[1] + model.config.latent_size + 1
                if (
                    generation_config.max_sequence_length is not None
                    and completed_length > generation_config.max_sequence_length
                ):
                    raise ValueError("latent block would exceed max_sequence_length")
                suffix = torch.tensor(
                    [[token_ids.pad_token_id] * model.config.latent_size
                     + [token_ids.end_token_id]],
                    device=sequences.device,
                    dtype=sequences.dtype,
                )
                sequences = torch.cat((sequences, suffix), dim=1)
                mask = torch.cat((mask, torch.ones_like(suffix)), dim=1)
                embeds = torch.cat((embeds, model.get_input_embeddings()(suffix)), dim=1)
                positions, _ = model.get_rope_index(
                    sequences, image_grid_thw=image_grid_thw, attention_mask=mask
                )
                span = LatentSpan(
                    start_position=start,
                    latent_positions=tuple(range(start + 1, start + 1 + model.config.latent_size)),
                    end_position=start + model.config.latent_size + 1,
                )
                rollout = model.stage2_rollout_engine.rollout(
                    language_model=functools.partial(model.model, generate_mode=True),
                    base_inputs_embeds=embeds,
                    attention_mask=mask,
                    full_position_ids=positions,
                    latent_spans=(span,),
                )
                embeds = rollout.final_inputs_embeds
                blocks.append(rollout.latent_states)
                generated += model.config.latent_size + 1
                continue
            if (
                generation_config.eos_token_id is not None
                and int(next_id.item()) == generation_config.eos_token_id
            ):
                break
        return Stage2GenerationResult(
            sequences=sequences,
            latent_states_by_block=tuple(blocks) if return_latent_states else (),
        )


def _select_text_token(logits: torch.Tensor, config: Stage2GenerationConfig) -> torch.LongTensor:
    if config.temperature <= 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / config.temperature, dim=-1)
    if config.top_p < 1.0:
        sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
        remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > config.top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1, sorted_indices, sorted_probabilities
        )
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
```

Import `functools`, `LatentSpan`, and the shared engine types. The scaffold append deliberately completes a block even when it takes the result beyond the nominal token budget; EOS is checked only for normally sampled text tokens.

- [ ] **Step 5: Run and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_rollout.py tests/stage2/test_inference.py -q
git add src/stage2/inference.py tests/stage2/test_inference.py
git commit -m "feat: add stage2 latent inference controller"
```

### Task 12: Retire Stage 2 State from Generic Generation

**Files:**
- Modify: `transformers/src/transformers/generation/utils.py:132-166,1971-2325,3197-3664`
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:2317-2358`
- Create: `tests/stage2/test_generation_regression.py`

**Interfaces:**
- Consumes: Task 11 controller and ordinary sampler preserved as `_sample_backup`.
- Produces: one standard `_sample`, standard output type, and Qwen-specific `generate_mode=True`.

- [ ] **Step 1: Write failing cleanup and ordinary-generation tests**

```python
def test_generic_sample_has_standard_signature():
    parameters = inspect.signature(GenerationMixin._sample).parameters
    assert "tokenizer" not in parameters
    assert not hasattr(GenerationMixin, "_sample_backup")


def test_generic_output_has_no_ilvr_latent_state():
    assert "latent_hidden_states" not in GenerateDecoderOnlyOutput.__dataclass_fields__


def test_generic_generation_contains_no_ilvr_ids_or_state_machine():
    source = inspect.getsource(transformers.generation.utils.GenerationMixin)
    for forbidden in ("151665", "151666", "151667", "latent_start_id", "latent_end_id"):
        assert forbidden not in source


def test_tiny_text_model_generates_without_stage2_arguments():
    model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_layer=1, n_head=1, n_embd=8))
    output = model.generate(torch.tensor([[1, 2]]), max_new_tokens=2, do_sample=False)
    assert output.shape == (1, 4)
```

```python
def test_qwen_generation_plumbing_only_sets_model_specific_attention_mode():
    model_inputs = tiny_stage2_qwen().prepare_inputs_for_generation(
        torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2, dtype=torch.long)
    )
    assert model_inputs["generate_mode"] is True
    assert not ({"latent_hidden_states", "latent_start_id", "latent_end_id"} & model_inputs.keys())
```

- [ ] **Step 2: Run and confirm current generic ILVR state**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_generation_regression.py -q
```

- [ ] **Step 3: Restore one ordinary sampler**

Promote the existing `_sample_backup` body/signature to `_sample`, remove the duplicate, stop passing `tokenizer` from `generate` into `_sample`, and remove `latent_hidden_states` from `GenerateDecoderOnlyOutput`. Preserve tokenizer use for stop strings and assisted generation outside `_sample`.

- [ ] **Step 4: Move attention-mode selection into Qwen**

```python
model_inputs["generate_mode"] = True
```

Set this in `prepare_inputs_for_generation`. Do not put latent IDs/state there.

- [ ] **Step 5: Run and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_generation_regression.py tests/stage2/test_qwen_forward.py tests/stage2/test_inference.py -q
git add transformers/src/transformers/generation/utils.py transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py tests/stage2/test_generation_regression.py
git commit -m "refactor: isolate stage2 from generic generation"
```

### Task 13: Evaluation Entry Points Use the Controller

**Files:**
- Modify: `eval.py:292-379`
- Modify: `src/evaluate_deepseed.py:202-252`
- Create: `tests/stage2/test_eval_integration.py`

**Interfaces:**
- Consumes: Task 11 controller and existing answer scoring.
- Produces: both evaluation paths call the controller while preserving result schema.

- [ ] **Step 1: Write failing integration tests**

```python
@pytest.mark.parametrize("module_name", ["eval", "evaluate_deepseed"])
def test_eval_uses_stage2_controller_instead_of_model_generate(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    controller = RecordingStage2Controller(sequence=torch.tensor([[1, 2, 3]]))
    monkeypatch.setattr(module, "Stage2InferenceController", lambda: controller)
    result = module.run_one_example(
        FakeModelWhoseGenerateRaises(), fake_processor(), sample(), generation_kwargs()
    )
    assert controller.calls == 1
    assert result["extracted_final_answer"] == "expected"
```

- [ ] **Step 2: Run and confirm legacy `model.generate` use**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_eval_integration.py -q
```

- [ ] **Step 3: Replace generic generation calls**

Resolve tokens through Task 6, build `Stage2GenerationConfig` from existing CLI values, and call the controller. Remove `LatentTemplateLogitsProcessor` imports and direct ID probing. Preserve input resolution, decode, timing, answer extraction, and output fields.

- [ ] **Step 4: Run and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_inference.py tests/stage2/test_eval_integration.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python eval.py --help
git add eval.py src/evaluate_deepseed.py tests/stage2/test_eval_integration.py
git commit -m "feat: use stage2 controller for evaluation"
```

### Task 14: Gradient Checkpointing and FlashAttention Compatibility

**Files:**
- Modify: `tests/stage2/test_qwen_forward.py`
- Create: `tests/stage2/integration/test_cuda_backends.py`
- Modify: `src/main.py`

**Interfaces:**
- Consumes: Task 4 forward with `generate_mode=True` language-model callable.
- Produces: explicit backend settings and parity evidence without recurrence changes.

- [ ] **Step 1: Write a CPU gradient-checkpointing regression**

Enable checkpointing on one tiny model, keep `use_cache=False`, compare one forward/backward with identical non-checkpointed weights:

```python
assert torch.allclose(plain.loss, checkpointed.loss, atol=1e-6, rtol=1e-5)
assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(plain_states, checkpointed_states))
```

- [ ] **Step 2: Run the checkpointing test**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_qwen_forward.py -k checkpoint -q
```

Invoke systematic debugging before any fix if this fails unexpectedly.

- [ ] **Step 3: Add CUDA eager/FlashAttention parity**

Skip unless CUDA and `flash_attn` exist. Load identical tiny weights into eager and FlashAttention models, run bfloat16, assert finite losses, equal latent count, and latent/logit parity at `atol=2e-2, rtol=2e-2`.

- [ ] **Step 4: Run one B200 comparison**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/integration/test_cuda_backends.py -q
```

Retain this single backend witness.

- [ ] **Step 5: Wire explicit backend/checkpoint settings**

Set attention implementation from the Stage 2 CLI, always set `use_cache=False`, and enable gradient checkpointing only from its Stage 2 flag. Never toggle it inside rollout.

- [ ] **Step 6: Run and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_qwen_forward.py tests/stage2/integration/test_cuda_backends.py -q
git add src/main.py tests/stage2/test_qwen_forward.py tests/stage2/integration/test_cuda_backends.py
git commit -m "test: verify stage2 training backends"
```

### Task 15: DeepSpeed ZeRO-3, Rank-Local B=1, Save, and Resume

**Files:**
- Create: `tests/stage2/integration/tiny_stage2_train.py`
- Create: `tests/stage2/integration/test_distributed_contract.py`
- Modify: `src/main.py`
- Modify: `run_stage2_training.sh`

**Interfaces:**
- Consumes: Task 9 Trainer/load decision and `configs/config_stage3.json`.
- Produces: `validate_stage2_distributed_settings(per_device_train_batch_size)`, rank-local B=1 distributed execution, and Stage 2 save/resume evidence.

- [ ] **Step 1: Write failing distributed-setting tests**

```python
def test_stage2_rejects_deepspeed_micro_batch_above_one():
    with pytest.raises(ValueError, match="train_micro_batch_size_per_gpu must resolve to 1"):
        validate_stage2_distributed_settings(per_device_train_batch_size=2)
```

Also record collator local batch sizes and assert every observed size equals one.

- [ ] **Step 2: Add a tiny two-step save/resume harness**

`tiny_stage2_train.py` builds a tiny Qwen Stage 2 model and synthetic one-image sample, trains one step, saves a Trainer checkpoint, resumes through `resume_from_checkpoint_path`, trains one more step, and writes:

```json
{"first_global_step": 1, "resumed_global_step": 2, "optimizer_state_restored": true, "scheduler_state_restored": true}
```

- [ ] **Step 3: Run without DeepSpeed first**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python tests/stage2/integration/tiny_stage2_train.py --output-dir "$(mktemp -d)"
```

- [ ] **Step 4: Add optional Stage 2 DeepSpeed selection**

```python
def validate_stage2_distributed_settings(per_device_train_batch_size: int) -> None:
    if per_device_train_batch_size != 1:
        raise ValueError("Stage 2 train_micro_batch_size_per_gpu must resolve to 1")
```

When `--deepspeed_config` is set, call this validator and pass the path into TrainingArguments. Keep `configs/config_stage3.json` unchanged unless systematic debugging proves a defect.

- [ ] **Step 5: Run one two-GPU ZeRO-3 smoke**

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/accelerate launch --num_processes 2 \
  tests/stage2/integration/tiny_stage2_train.py \
  --deepspeed configs/config_stage3.json \
  --output-dir "$(mktemp -d)"
```

Expected: rank-local B=1, finite loss, checkpoint save, resume to step 2, exit 0. Stop after one successful witness.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_checkpointing.py tests/stage2/test_routing.py tests/stage2/integration/test_distributed_contract.py -q
git add src/main.py run_stage2_training.sh tests/stage2/integration
git commit -m "feat: verify stage2 zero3 checkpoint resume"
```

### Task 16: CoMT Reproduction Readiness and Documentation

**Files:**
- Modify: `README.md`
- Modify: `run_stage2_training.sh`
- Modify: `src/stage2/data.py`
- Create: `tests/stage2/test_comt_readiness.py`

**Interfaces:**
- Consumes: CoMT JSONL, Task 8 filter, Task 9 configuration, and retained witnesses.
- Produces: one auditable CoMT Stage 2 command and readiness summary.

- [ ] **Step 1: Write a readiness test**

```python
def test_comt_command_exposes_reproduction_critical_settings():
    script = Path("run_stage2_training.sh").read_text()
    for setting in (
        "LATENT_SIZE", "SUPERVISE_LATENT_BOUNDARIES", "LEARNING_RATE",
        "GRADIENT_ACCUMULATION_STEPS", "STAGE1_CHECKPOINT", "DATA_PATH", "IMAGE_ROOT",
    ):
        assert setting in script
```

- [ ] **Step 2: Add explicit launcher defaults**

```bash
LATENT_SIZE=${LATENT_SIZE:-8}
SUPERVISE_LATENT_BOUNDARIES=${SUPERVISE_LATENT_BOUNDARIES:-true}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
PER_DEVICE_TRAIN_BATCH_SIZE=1

case "$SUPERVISE_LATENT_BOUNDARIES" in
  true) BOUNDARY_ARG=--supervise_latent_boundaries ;;
  false) BOUNDARY_ARG=--no-supervise_latent_boundaries ;;
  *) echo "SUPERVISE_LATENT_BOUNDARIES must be true or false" >&2; exit 2 ;;
esac
```

Pass these arguments in the Task 10 `exec` command before `"${RESUME_ARGS[@]}"`:

```bash
  --latent_size "$LATENT_SIZE" \
  "$BOUNDARY_ARG" \
  --learning_rate "$LEARNING_RATE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
```

- [ ] **Step 3: Add the dataset-audit module entry point**

```python
def build_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit CoMT Stage 2 sequence lengths")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--latent-size", type=int, required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    return parser


def build_processor_length_measure(*, processor_path, image_root, latent_size):
    processor = AutoProcessor.from_pretrained(processor_path)

    def measure(sample: dict) -> int:
        prepared = prepare_stage2_example(
            sample, image_root=Path(image_root), latent_size=latent_size
        )
        rendered = processor.apply_chat_template(
            prepared.conversation, tokenize=False, add_generation_prompt=False
        )
        encoded = processor(
            text=[rendered], images=[list(prepared.original_image_paths)],
            padding=False, truncation=False, return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[1])

    return measure


def main(argv: list[str] | None = None) -> int:
    args = build_audit_parser().parse_args(argv)
    examples = load_jsonl_examples(args.data_path)
    _, stats = filter_stage2_examples(
        examples=examples,
        measure_length=build_processor_length_measure(
            processor_path=args.processor_path,
            image_root=args.image_root,
            latent_size=args.latent_size,
        ),
        max_length=args.max_sequence_length,
    )
    print(json.dumps(asdict(stats), sort_keys=True))
    return 0
```

Add `if __name__ == "__main__": raise SystemExit(main())` and a unit test invoking `main([...])` with fake loaders.

- [ ] **Step 4: Run one exact CoMT length audit**

Check for an unchanged existing witness first. Otherwise run once:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m stage2.data \
  --data-path ../../data/TRAIN.jsonl \
  --image-root ../../data \
  --processor-path ../../checkpoints/comt_ckpt \
  --latent-size 8 \
  --max-sequence-length 8192
```

Record total/retained/filtered, percentage, percentiles, and one-to-six block distribution. Do not rerun to reformat records.

If the filtered fraction is unexpectedly large, stop this milestone and invoke `superpowers:systematic-debugging`: reproduce on a small set, inspect rendered lengths and latent spans, test one hypothesis, and add a regression test. Do not introduce partial or block-aware truncation in this implementation.

- [ ] **Step 5: Document evidence levels**

README separately labels implementation, single-step smoke, distributed/DeepSpeed, and paper-level reproduction status. Include checkpoint origin, optimizer reset, visual freeze, boundary setting, effective batch formula, no-cache status, and unsupported local B>1.

- [ ] **Step 6: Run and commit**

```bash
bash -n run_stage2_training.sh
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2/test_comt_readiness.py tests/stage2/test_launcher.py -q
git add README.md run_stage2_training.sh src/stage2/data.py tests/stage2/test_comt_readiness.py
git commit -m "docs: document stage2 comt readiness"
```

### Task 17: Full Verification, Requirement Audit, and Code Review

**Files:**
- Review: `docs/superpowers/specs/2026-08-24-ilvr-stage2-reimplementation-design.md`
- Review: `docs/superpowers/plans/2026-08-24-ilvr-stage2-reimplementation.md`
- Review: all changes after commit `3cd91e9`

**Interfaces:**
- Consumes: prior commits and retained witnesses.
- Produces: a reviewed final branch and evidence-backed report; no new feature behavior.

- [ ] **Step 1: Build a spec-to-diff checklist**

For spec sections 3-11, record the implementing file/test or an explicit unexecuted resource limit. Confirm one recurrence, MRoPE slices, helper exclusion, visual-once, boundary pair, B=1 failure, fresh/resume split, train/inference parity, and generic cleanup.

- [ ] **Step 2: Run the complete feasible test suite once**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -m pytest tests/stage2 -q
```

Reuse unchanged successful Task 10/14/15 expensive witnesses.

- [ ] **Step 3: Run static and import checks**

```bash
bash -n run_training.sh run_stage2_training.sh eval.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python -c 'import main, eval; from stage2.inference import Stage2InferenceController'
git diff --check 3cd91e9..HEAD
git status --short --branch
```

- [ ] **Step 4: Inspect the final diff**

```bash
git diff --stat 3cd91e9..HEAD
git diff 3cd91e9..HEAD -- src transformers/src/transformers/models/qwen2_5_vl transformers/src/transformers/generation/utils.py tests/stage2 README.md run_stage2_training.sh
```

Check for duplicate recurrence, hard-coded IDs, helper tensors in Stage 2, detach on latent states, accidental Stage 1 edits, broad exception handlers, and semantic fallbacks.

- [ ] **Step 5: Run the Superpowers review workflow**

Invoke `superpowers:requesting-code-review`. Resolve confirmed issues with TDD; invoke systematic debugging first for unexpected behavior. Re-run only validators relevant to changed files.

- [ ] **Step 6: Verify and prepare the final report**

Invoke `superpowers:verification-before-completion`. Report files, architecture, exact commands/results, smoke evidence, CUDA/FlashAttention/DeepSpeed/distributed/resume results, unexecuted checks, known limitations, local B>1 status, no-KV-cache status, and whether full paper-level CoMT reproduction ran.

## Plan Self-Review Record

- Spec scope and non-goals are enforced by Global Constraints and Tasks 9, 14, and 15: CoMT/Qwen2.5-VL only, rank-local B=1, no LoRA, no KV-cache optimization, and no unrelated model support.
- The mathematical contract maps to Tasks 1-5: exact off-by-one recurrence, K states per block, full-context multi-block behavior, no detach/projection, downstream gradients, pad independence, multimodal MRoPE slicing, one visual call, and final shifted CE.
- Data and loss requirements map to Tasks 7-8: helper locations become exact K-slot blocks, original images remain, assistant/padding/slot masks are explicit, the boundary pair uses one boolean, malformed scaffolds fail, and overlength examples are filtered whole with one-pass statistics.
- Checkpoint and training requirements map to Tasks 6, 9, 10, and 15: canonical/legacy token parity, Stage 1 fresh initialization, Stage 2-only resume, processor/config saving, new fresh optimizer/scheduler state, visual freeze, thin Trainer, one-step smoke, and ZeRO-3 save/resume evidence.
- Inference and vendor-cleanup requirements map to Tasks 11-13: a model-specific NORMAL_TEXT/LATENT_MODE controller shares the rollout engine, completes started blocks, verifies train/inference parity, removes hard-coded generic generation state only after coverage, and updates both evaluation entry points.
- Compatibility, readiness, and completion requirements map to Tasks 14-17: gradient checkpointing, FlashAttention, DeepSpeed, exact CoMT audit/launcher settings, evidence-level documentation, feasible full tests, retained expensive witnesses, diff inspection, code review, and verification-before-completion.
- Placeholder scan found and removed incomplete parser, token-resolution, trajectory, collator, controller, and audit bodies. All named cross-task contracts now have one defining task; code-fence and whitespace checks are part of the plan-file verification before commit.

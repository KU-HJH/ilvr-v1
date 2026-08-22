# ILVR Stage-2 Evidence–Window Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve ILVR Stage 1 and build an explicit Stage 2 that learns exact, window-local latent-evidence-to-reasoning credit with optional fixed-reference, relation, and hidden-residual objectives.

**Architecture:** Keep the existing Stage-1 trainer and version-1 zero-latent experiments intact. Add a version-2 Stage-2 path whose parser creates all causal `Z_j -> R_k` edges, whose custom Qwen auxiliary forward uses the existing weights through matched SDPA, and whose trainer combines ordinary FlashAttention CE with exact pair-specific auxiliary effects. Use reference-free EW preference by default; use precomputed targets for ZeRO-3 and an independently loaded frozen model for explicitly selected ZeRO-1/2 runs.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu128, CUDA 12.8, FlashAttention 2.8.3, vendored Transformers 4.50.0.dev0, TRL 0.15.2, PEFT 0.17.1, Accelerate/DeepSpeed, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-ilvr-evidence-window-credit-design.md`

## Global Constraints

- Use `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python` for every Python, pytest, training, and evaluation command.
- Execute inside `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/.worktrees/ours-trial1`; invoke `superpowers:using-git-worktrees` before implementation.
- Preserve `CustomTrainerStage1`, its EMA/helper-image supervision, CE/similarity objective, masks, BF16 behavior, and version-1 EW experiments.
- Stage 2 starts from `--load_model_path`, uses self-generated latent states, and uses assistant-text CE without EMA, helper-image teacher forcing, `pixel_values_latent`, `latent_hidden_states`, or similarity loss.
- The intervention blocks only every language layer's `R_k` query to `Z_j` key/value edges. It never deletes/zeros tokens, changes positions, blocks other evidence, or changes `R_<k` computation.
- Features-off Stage-2 CE stays on FlashAttention2. Auxiliary full and LOO forwards both use the same SDPA override.
- Exhaustive all-causal edges are default; a positive cap selects nearest preceding evidence first, then ascending token start.
- Mean log probability is default; sum is explicit. Reference-free EW is default. The fixed reference is the Stage-1 initialization checkpoint executed with Stage-2 semantics, never EMA.
- ZeRO-3 supports only `none` and `precomputed`; reject `online_frozen` before construction. Online frozen is required for ZeRO-1/2.
- All new coefficients default to zero; disabled Stage 2 performs no layout, SDPA auxiliary, artifact, or reference work.
- LOO is a conditional/direct retained-history estimator, not a total causal effect. Do not optimize entropy or claim dense use is better.
- Follow TDD. Use `systematic-debugging` for unexpected failures, `requesting-code-review` at milestone gates, and `verification-before-completion` before completion.
- Inspect existing evidence before validation over 10 GB or five minutes. Run the final 7B/B200 matrix once and reuse its witness.

## File Structure

- Keep existing `src/evidence_window/{config,layout,interventions,engine}.py` and v1 configs as historical Stage-1 EW code.
- Create `stage2_config.py`, `stage2_layout.py`, `attention.py`, `stage2_effects.py`, `stage2_engine.py`, `artifacts.py`, and `stage2_reference.py` under `src/evidence_window/`.
- Create `src/stage2_data.py`, `src/trainer_stage2.py`, and `src/precompute_stage2_reference.py`.
- Modify `src/task_deepseed.py`, `src/main.py`, `src/utils_deepseed.py`, and custom Qwen modeling.
- Extend shared `scores.py`, `losses.py`, and `metrics.py` only for reusable tensor primitives.
- Add `configs/evidence_window/stage2/*.json`, focused `tests/evidence_window/test_stage2_*.py`, README, and continuation guidance.

The subsystems form one dependency chain—data/baseline, layout, mask-capable model, effects, objectives, references, and distillation—so this is one sequential plan. Every task ends in a reviewable commit.

---

### Task 1: Version-2 Configuration and Schedules

**Files:**
- Create: `src/evidence_window/stage2_config.py`
- Create: `tests/evidence_window/test_stage2_config.py`
- Modify: `src/evidence_window/__init__.py`

**Interfaces:**
- Consumes: v2 JSON and DeepSpeed JSON.
- Produces: `Stage2EvidenceWindowConfig.from_json/from_dict`, `.disabled()`, `.is_disabled`, `.required_outputs`, `.validate_runtime(zero_stage)`, `.effective_weight(name, progress)`, `read_deepspeed_zero_stage(path)`.

- [ ] **Step 1: Write failing tests**

```python
def test_reference_free_ew_and_schedule():
    cfg = Stage2EvidenceWindowConfig.from_dict({
        "version": 2,
        "preference": {"enabled": True, "lambda_ew": 2.0, "beta": 0.2},
        "schedule": {"ew_warmup_ratio": 0.1, "ew_ramp_ratio": 0.2},
    })
    assert cfg.reference.mode == "none"
    assert cfg.required_outputs == frozenset({"token_logps"})
    assert cfg.effective_weight("ew", 0.1) == pytest.approx(0.0)
    assert cfg.effective_weight("ew", 0.2) == pytest.approx(1.0)
    assert cfg.effective_weight("ew", 0.3) == pytest.approx(2.0)

def test_relation_requires_reference():
    with pytest.raises(ValueError, match="relation.*reference"):
        Stage2EvidenceWindowConfig.from_dict({
            "version": 2, "relation": {"enabled": True, "lambda_rel": 1.0}
        })

def test_online_frozen_rejected_for_zero3(tmp_path):
    ds = tmp_path / "ds.json"
    ds.write_text('{"zero_optimization":{"stage":3}}')
    cfg = Stage2EvidenceWindowConfig.from_dict({
        "version": 2,
        "reference": {"mode": "online_frozen", "checkpoint": "/fixed/stage1"},
        "preference": {"enabled": True, "lambda_ew": 1.0},
    })
    with pytest.raises(ValueError, match="ZeRO-3"):
        cfg.validate_runtime(read_deepspeed_zero_stage(ds))
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_config.py -q
```

Expected: module import fails.

- [ ] **Step 3: Implement exact schema**

Create frozen nested dataclasses for reference (`none|precomputed|online_frozen`, checkpoint, artifact), preference (enabled, lambda, beta, `mean|sum`), relation, residual, LOO (`window_attention_block`, edge cap), schedules, and diagnostics. Reject unknown keys, version other than 2, invalid finite/range values, missing reference dependencies, and incompatible ZeRO stages. Effective enablement is `enabled and lambda > 0`; zero coefficients perform no work. Schedules are zero before warmup, linear during ramp, full afterward, with an immediate switch for zero ramp.

- [ ] **Step 4: Run GREEN and v1 regression**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_config.py tests/evidence_window/test_config.py tests/evidence_window/test_experiment_configs.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/evidence_window/stage2_config.py src/evidence_window/__init__.py tests/evidence_window/test_stage2_config.py
git commit -m "feat: define stage2 evidence-window config"
```

---

### Task 2: Helper-Free Stage-2 Data

**Files:**
- Create: `src/stage2_data.py`
- Create: `tests/evidence_window/test_stage2_data.py`
- Modify: `src/task_deepseed.py:20-62`
- Modify: `tests/evidence_window/conftest.py`

**Interfaces:**
- Produces: `build_stage2_record(sample, row_index) -> dict` and `collate_fn_stage2(records, processor, args) -> dict` containing model tensors, `ew_sample_indices`, and `ew_plan_types`.

- [ ] **Step 1: Write failing tests**

```python
def test_record_preserves_plan_without_checking_helper(monkeypatch, raw_sample):
    monkeypatch.setattr("task_deepseed._assert_exists", lambda p: (_ for _ in ()).throw(AssertionError(p)))
    record = build_stage2_record(raw_sample, 17, validate_user_images=False)
    assert record["ew_sample_index"] == 17
    assert record["ew_plan_types"] == ("text", "latent", "text")

def test_collator_loads_only_user_images(stage2_processor, stage2_record):
    batch = collate_fn_stage2([stage2_record], stage2_processor, stage2_args())
    for forbidden in ("pixel_values_latent", "image_grid_thw_latent", "latent_hidden_states", "image_out_mask"):
        assert forbidden not in batch
    assert batch["ew_sample_indices"].tolist() == [17]
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_data.py -q
```

Expected: missing `build_stage2_record` or `collate_fn_stage2`.

- [ ] **Step 3: Implement records and collation**

Preserve helper paths only as inert assistant image placeholders for chat-template rendering. Validate user images normally but never validate/decode helper images. Collation reuses current template replacement, `process_batch`, and label generation, and passes the result of `remove_assistant_images(conversations)` to `process_vision_info`. Append row indices and plan-type tuples after processor calls.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_data.py tests/evidence_window/test_backward_compatibility.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/stage2_data.py src/task_deepseed.py tests/evidence_window/test_stage2_data.py tests/evidence_window/conftest.py
git commit -m "feat: add helper-free stage2 data path"
```

---

### Task 3: Explicit CE-Only Stage-2 Baseline

**Files:**
- Create: `src/trainer_stage2.py`
- Create: `tests/evidence_window/test_stage2_baseline.py`
- Modify: `src/main.py:24-246`
- Modify: `src/utils_deepseed.py:15-65`

**Interfaces:**
- Produces: `CustomTrainerStage2.compute_loss`, `select_stage_components(stage)`, `resolve_policy_model_path(args)`.

- [ ] **Step 1: Write failing tests**

```python
def test_baseline_equals_direct_model_ce(tiny_stage2_model, stage2_batch):
    trainer = make_stage2_trainer(tiny_stage2_model, Stage2EvidenceWindowConfig.disabled())
    direct = tiny_stage2_model(**model_only(stage2_batch)).loss
    actual = trainer.compute_loss(tiny_stage2_model, clone_batch(stage2_batch))
    torch.testing.assert_close(actual, direct)
    assert not hasattr(trainer, "ema")

def test_stage2_requires_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Stage 1 checkpoint"):
        resolve_policy_model_path(stage_args("stage2", tmp_path / "missing"))
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_baseline.py -q
```

- [ ] **Step 3: Implement minimal trainer**

Subclass `SFTTrainer`, not `CustomTrainerStage1`. Pop EW metadata, call `model(**inputs)`, and return `outputs.loss`. Do not instantiate EMA, helper teacher, or similarity state.

- [ ] **Step 4: Wire main explicitly**

Stage 1 loads `args.model`, v1 config, current preprocessor/collator/trainer. Stage 2 requires `args.load_model_path`, sets `config.stage="stage2"`, parses v2 config, enumerates raw rows into Stage-2 records, and selects Stage-2 collator/trainer. Reject cross-version configs. Preserve existing Stage-1 constructor arguments only on Stage 1.

- [ ] **Step 5: Run GREEN**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_baseline.py tests/evidence_window/test_backward_compatibility.py tests/evidence_window/test_trainer_integration.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/trainer_stage2.py src/main.py src/utils_deepseed.py tests/evidence_window/test_stage2_baseline.py
git commit -m "feat: separate explicit stage2 baseline"
```

---

### Task 4: Marker-Inclusive Layout and Exact Edges

**Files:**
- Create: `src/evidence_window/stage2_layout.py`
- Create: `tests/evidence_window/test_stage2_layout.py`
- Modify: `src/evidence_window/__init__.py`

**Interfaces:**
- Produces: `EvidenceWindowEdge`, `Stage2InterleavedLayout`, `extract_stage2_layout`, `validate_plan_types`, `select_stage2_edges`.

- [ ] **Step 1: Write failing parser tests**

```python
def test_r1_z1_r2_z2_r3_and_causal_edges():
    ids = torch.tensor([[11,90,92,92,91,21,22,90,92,91,31,0]])
    labels = torch.tensor([[11,-100,-100,-100,-100,21,22,-100,-100,-100,31,-100]])
    layout = extract_stage2_layout(ids, ids.ne(0), labels, TOKENS)
    assert layout.evidence_token_mask[0,0].nonzero().flatten().tolist() == [1,2,3,4]
    assert layout.window_token_mask[0,1].nonzero().flatten().tolist() == [5,6]
    assert layout.causal_pair_mask[0].nonzero().tolist() == [[0,1],[0,2],[1,2]]

def test_cap_selects_nearest_predecessors():
    edges = select_stage2_edges(three_evidence_layout(), 2)
    assert [(e.evidence_index, e.window_index) for e in edges_for_final_window(edges)] == [(1,3),(2,3)]
```

Also test unmatched/nested markers, adjacent spans, no latent, no predecessor, empty window, padding, heterogeneous batch, and plan mismatch with row index.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_layout.py -q
```

- [ ] **Step 3: Implement types and parser**

```python
@dataclass(frozen=True, order=True)
class EvidenceWindowEdge:
    batch_index: int
    evidence_index: int
    window_index: int

@dataclass(frozen=True)
class Stage2InterleavedLayout:
    evidence_token_mask: torch.Tensor  # [B,J,T], marker-inclusive
    window_token_mask: torch.Tensor    # [B,K,T], supervised assistant text
    evidence_valid: torch.Tensor
    window_valid: torch.Tensor
    causal_pair_mask: torch.Tensor
```

Reuse tokenizer-resolved `TokenIds`. Require at least one latent-pad interior token for valid evidence. Select exhaustive edges by default; capped selection is per-window nearest-first and returned in stable `(batch, window, evidence)` order.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_layout.py tests/evidence_window/test_layout.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/evidence_window/stage2_layout.py src/evidence_window/__init__.py tests/evidence_window/test_stage2_layout.py
git commit -m "feat: parse stage2 evidence-window edges"
```

---

### Task 5: Pair-Specific Block Masks

**Files:**
- Create: `src/evidence_window/attention.py`
- Create: `tests/evidence_window/test_stage2_attention_mask.py`

**Interfaces:**
- Produces: `build_window_local_block_mask(layout, edge) -> BoolTensor[B,T,T]` (`True` means blocked) and `slice_block_mask(mask, prefix_length)`.

- [ ] **Step 1: Write failing locality tests**

```python
def test_exact_cartesian_product(layout):
    edge = EvidenceWindowEdge(0, 0, 1)
    mask = build_window_local_block_mask(layout, edge)
    expected = torch.zeros_like(mask)
    expected[0] = layout.window_token_mask[0,1][:,None] & layout.evidence_token_mask[0,0][None,:]
    assert torch.equal(mask, expected)

def test_prefix_slices_query_and_key_axes(layout):
    full = build_window_local_block_mask(layout, EvidenceWindowEdge(0,0,1))
    assert torch.equal(slice_block_mask(full, 6), full[:,:6,:6])
```

Assert other rows/windows/evidence are false, shape and device are unchanged, and noncausal/out-of-range edges raise.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_attention_mask.py -q
```

- [ ] **Step 3: Implement** a contiguous boolean outer product only on the selected batch row with strict validation.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_attention_mask.py -q
git add src/evidence_window/attention.py tests/evidence_window/test_stage2_attention_mask.py
git commit -m "feat: build exact window attention blocks"
```

---

### Task 6: Same-Weight SDPA Override in Custom Qwen

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:701-1614,1961-2315`
- Create: `tests/evidence_window/test_stage2_qwen_attention.py`

**Interfaces:**
- Consumes: 2D padding mask, optional boolean `attention_block_mask [B,T,T]`, and `attention_backend_override: None | "sdpa"`.
- Produces: unchanged default FlashAttention and exact auxiliary SDPA through the same q/k/v/o parameters.

- [ ] **Step 1: Write failing tiny-Qwen tests**

```python
def test_block_leaves_prior_hidden_states_unchanged(tiny_qwen, batch, block, window_start):
    full = tiny_qwen(**batch, attention_backend_override="sdpa").hidden_states
    loo = tiny_qwen(**batch, attention_backend_override="sdpa", attention_block_mask=block).hidden_states
    torch.testing.assert_close(full[:,:window_start], loo[:,:window_start])
    assert not torch.allclose(full[:,window_start], loo[:,window_start])

def test_block_requires_sdpa(tiny_qwen, batch, block):
    with pytest.raises(ValueError, match="SDPA"):
        tiny_qwen(**batch, attention_block_mask=block)
```

Also assert two unblocked SDPA calls match, default calls never enter the override, and the Stage-2 latent prefix loop slices `[B,:p,:p]`.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_qwen_attention.py -q
```

Expected: unknown forward keyword or missing `forward_sdpa`.

- [ ] **Step 3: Add `Qwen2_5_VLAttention.forward_sdpa`**

Move the current `Qwen2_5_VLSdpaAttention.forward` tensor logic into a reusable method on the base attention class. Preserve q/k/v/o weights, multimodal RoPE, KV repetition, dropout, and CUDA contiguity. Delegate the SDPA subclass to it. Decoder layers choose it only when the explicit override equals `sdpa`; all current branches remain byte-for-byte behaviorally unchanged otherwise.

- [ ] **Step 4: Build the auxiliary causal mask**

When overridden, `Qwen2_5_VLModel.forward` calls `_prepare_4d_causal_attention_mask_with_cache_position` even though configured FlashAttention is retained. It fills intervention-true entries with `torch.finfo(dtype).min` without unmasking causal/padding entries. Reject unknown overrides, wrong dtype/shape/device, `use_cache=True`, and a block without the override.

- [ ] **Step 5: Thread through Stage-2 prefixes**

Add both arguments to the conditional-generation forward. Keep the original 2D padding mask for RoPE. Pass a separately sliced block mask to each prefix model call and the full mask to the final call. Preserve latent order, embeddings, positions, and checkpointing.

- [ ] **Step 6: Run GREEN**

```bash
PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_qwen_attention.py tests/evidence_window/test_backward_compatibility.py -q
```

- [ ] **Step 7: Commit**

```bash
git add transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py tests/evidence_window/test_stage2_qwen_attention.py
git commit -m "feat: add exact stage2 sdpa intervention"
```

---

### Task 7: Matched Full/LOO Effects

**Files:**
- Create: `src/evidence_window/stage2_effects.py`
- Create: `tests/evidence_window/test_stage2_effects.py`
- Modify: `src/evidence_window/scores.py`

**Interfaces:**
- Consumes: layout, edges, `ConditionForward(block_mask, need_hidden)`.
- Produces: `Stage2ConditionOutput`, `Stage2EvidenceEffects`, `window_scores`, `compute_stage2_effects`.

- [ ] **Step 1: Write failing tests**

```python
def test_window_scores_exclude_other_tokens():
    logps = torch.tensor([[1.,2.,100.,4.]])
    masks = torch.tensor([[[1,1,0,0],[0,0,0,1]]], dtype=torch.bool)
    assert torch.equal(window_scores(logps, masks, "sum"), torch.tensor([[3.,4.]]))
    assert torch.equal(window_scores(logps, masks, "mean"), torch.tensor([[1.5,4.]]))

def test_one_full_plus_one_forward_per_edge(layout):
    calls = []
    effects = compute_stage2_effects(layout, select_stage2_edges(layout,0), recording_forward(calls), "mean", False)
    assert calls[0] is None
    assert len(calls) == 1 + effects.pair_mask.sum().item()
```

Add exact full-minus-LOO arithmetic, no-edge/heterogeneous/finite cases, both branch gradients, and hidden residual direction tests.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_effects.py -q
```

- [ ] **Step 3: Implement contracts**

```python
@dataclass(frozen=True)
class Stage2ConditionOutput:
    token_logps: torch.Tensor
    final_hidden_state: torch.Tensor | None

@dataclass(frozen=True)
class Stage2EvidenceEffects:
    contributions: torch.Tensor   # [B,J,K]
    pair_mask: torch.Tensor       # [B,J,K]
    hidden_residuals: torch.Tensor | None
    forward_count: int
```

Cache one matched full output. For every exact edge, build one block, run one LOO, score only its batch/window, and write without detaching policy tensors. Mean pooling uses `clamp_min(1)` only numerically; invalid windows remain excluded.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_effects.py tests/evidence_window/test_scores.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/evidence_window/stage2_effects.py src/evidence_window/scores.py tests/evidence_window/test_stage2_effects.py
git commit -m "feat: compute exact stage2 evidence effects"
```

---

### Task 8: Reference-Free EW and Trainer Integration

**Files:**
- Create: `src/evidence_window/stage2_engine.py`
- Create: `tests/evidence_window/test_stage2_engine.py`
- Modify: `src/evidence_window/losses.py`
- Modify: `src/evidence_window/metrics.py`
- Modify: `src/trainer_stage2.py`

**Interfaces:**
- Produces: `Stage2EvidenceWindowResult` and `Stage2EvidenceWindowEngine.compute(layout, policy_forward, progress, reference_targets=None)`.

- [ ] **Step 1: Write saturation and EW tests**

```python
def test_local_gradient_survives_global_saturation():
    g = torch.tensor([20.,0.], requires_grad=True)
    gg = torch.autograd.grad(-F.logsigmoid(g.sum()), g)[0]
    w = torch.tensor([20.,0.], requires_grad=True)
    wg = torch.autograd.grad(-F.logsigmoid(w).sum(), w)[0]
    assert abs(gg[1].item()) < 1e-7
    assert abs(wg[1].item()) == pytest.approx(0.5)

def test_reference_free_formula():
    c = torch.tensor([[[.4,-.2]]], requires_grad=True)
    mask = torch.tensor([[[True,True]]])
    torch.testing.assert_close(stage2_ew_preference_loss(c,mask,.3), -F.logsigmoid(.3*c[mask]).mean())
```

Add no-edge zero, schedule boundaries, outer coefficient, softmax top-1/entropy, raw threshold count, and finite metric tests.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_engine.py -q
```

- [ ] **Step 3: Implement objective and metrics**

```python
@dataclass(frozen=True)
class Stage2EvidenceWindowResult:
    auxiliary_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]]
    effects: Stage2EvidenceEffects
```

Compute selected effects and reference-free EW for `reference.mode=none`. Apply scheduled weight. Log CE/EW, global margin and `beta*sigmoid(-beta*M)`, mean contribution, positive ratio, edge count, per-window summary, temperature-softmax top-1/normalized entropy, and raw threshold count. At `matrix_dump_interval`, reuse `MatrixJsonlWriter` to write detached contribution/pair matrices keyed by opaque rank/step/sample counters; never dump every step. Never add global diagnostics to loss.

- [ ] **Step 4: Integrate trainer**

Run ordinary CE first and return immediately if disabled. Otherwise parse/validate layout, define a policy callback using the same model with `attention_backend_override="sdpa"`, convert logits through `chunked_causal_target_logps`, request hidden only as configured, compute progress from `global_step/max_steps`, and return CE plus auxiliary. Reuse sufficient-statistic distributed logging.

- [ ] **Step 5: Run GREEN**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_engine.py tests/evidence_window/test_stage2_baseline.py tests/evidence_window/test_metrics.py tests/evidence_window/test_preference_losses.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/evidence_window/stage2_engine.py src/evidence_window/losses.py src/evidence_window/metrics.py src/trainer_stage2.py tests/evidence_window/test_stage2_engine.py
git commit -m "feat: train stage2 with local evidence preference"
```

---

### Task 9: Precomputed Reference Artifacts and DPO

**Files:**
- Create: `src/evidence_window/artifacts.py`
- Create: `src/evidence_window/stage2_reference.py`
- Create: `src/precompute_stage2_reference.py`
- Create: `tests/evidence_window/test_stage2_artifacts.py`
- Modify: `src/evidence_window/stage2_engine.py`
- Modify: `src/trainer_stage2.py`
- Modify: `src/utils_deepseed.py`

**Interfaces:**
- Produces: `ArtifactIdentity`, `ReferenceBatchTargets`, shard IO, `PrecomputedReferenceProvider.get_batch`, and EW-DPO.

- [ ] **Step 1: Write failing tests**

```python
def test_identity_mismatch_rejected(tmp_path):
    write_reference_shard(tmp_path, identity(checkpoint_id="a"), {7: target()})
    with pytest.raises(ValueError, match="checkpoint_id"):
        PrecomputedReferenceProvider(tmp_path, identity(checkpoint_id="b"))

def test_dpo_stops_reference_gradient():
    policy = torch.tensor([.7], requires_grad=True)
    reference = torch.tensor([.2], requires_grad=True)
    stage2_ew_dpo_loss(policy, reference, torch.tensor([True]), .4).backward()
    assert policy.grad is not None and reference.grad is None
```

Also cover sample/edge absence, duplicates, malformed tensors, layout order, and optional residual absence.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_artifacts.py -q
```

- [ ] **Step 3: Implement artifact contract**

`ArtifactIdentity` contains format version, one-time dataset SHA256, explicit checkpoint witness/id, tokenizer id, parser version, reduction, intervention, and edge cap. Store `manifest.json` and primitive/tensor `shard-rank-XXXXX.pt` payloads readable with `weights_only=True`. Never hash checkpoint weights. Validate all fields and pair masks.

- [ ] **Step 4: Implement precompute CLI**

Add reference checkpoint/id/artifact directory and hidden-residual flags. Load once with `config.stage="stage2"`, eval/no-grad, run matched effects, write rank shards, and let rank zero publish the small manifest from shard metadata.

- [ ] **Step 5: Consume precomputed targets**

Trainer retrieves targets by `ew_sample_indices`; engine uses `policy_C-reference_C.detach()`. Preserve relation/residual fields for later tasks.

- [ ] **Step 6: Run GREEN**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_artifacts.py tests/evidence_window/test_stage2_engine.py -q
```

- [ ] **Step 7: Commit**

```bash
git add src/evidence_window/artifacts.py src/evidence_window/stage2_reference.py src/evidence_window/stage2_engine.py src/trainer_stage2.py src/precompute_stage2_reference.py src/utils_deepseed.py tests/evidence_window/test_stage2_artifacts.py
git commit -m "feat: add precomputed stage2 reference targets"
```

---

### Task 10: Online Frozen Reference for ZeRO-1/2

**Files:**
- Modify: `src/evidence_window/stage2_reference.py`
- Modify: `src/trainer_stage2.py`
- Modify: `src/main.py`
- Create: `tests/evidence_window/test_stage2_online_reference.py`
- Create: `tests/evidence_window/stage2_zero1_reference_worker.py`
- Create: `tests/evidence_window/stage2_zero2_reference_worker.py`

**Interfaces:**
- Produces: `OnlineFrozenStage2Reference.create(model_factory, accelerator, zero_stage)` and no-grad matched reference effects.

- [ ] **Step 1: Write failing ownership tests**

```python
def test_zero3_rejects_before_factory():
    calls = []
    def factory(): calls.append(1)
    with pytest.raises(ValueError, match="ZeRO-3"):
        OnlineFrozenStage2Reference.create(factory, accelerator_stub(), 3)
    assert calls == []

def test_independent_reference_is_frozen(model_factory):
    ref = OnlineFrozenStage2Reference.create(model_factory, accelerator_stub(), 2)
    assert ref.model is not model_factory.policy
    assert ref.model.config.stage == "stage2"
    assert all(not p.requires_grad for p in ref.model.parameters())
```

Test one policy optimizer step leaves reference values unchanged and gradients `None`.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_online_reference.py -q
```

- [ ] **Step 3: Implement independent loading**

Factory calls `Qwen2_5_VLForConditionalGeneration.from_pretrained` with the configured reference checkpoint, Stage-2 model config, BF16 dtype, and FlashAttention2 base backend; never `deepcopy(policy)`. Freeze before calling `accelerator.prepare_model(reference_model, evaluation_mode=True)`, retain a cheap parameter witness, and compute the same edges/reduction/SDPA effects inside `torch.no_grad()`.

- [ ] **Step 4: Run CPU and isolated workers**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_online_reference.py -q
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python tests/evidence_window/stage2_zero1_reference_worker.py
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python tests/evidence_window/stage2_zero2_reference_worker.py
```

Expected: finite DPO, policy gradients, zero reference gradients, unchanged reference after the step.

- [ ] **Step 5: Commit**

```bash
git add src/evidence_window/stage2_reference.py src/trainer_stage2.py src/main.py tests/evidence_window/test_stage2_online_reference.py tests/evidence_window/stage2_zero1_reference_worker.py tests/evidence_window/stage2_zero2_reference_worker.py
git commit -m "feat: support stage2 frozen reference on zero1 and zero2"
```

---

### Task 11: Relation Distillation

**Files:**
- Modify: `src/evidence_window/stage2_engine.py`
- Modify: `src/evidence_window/losses.py`
- Create: `tests/evidence_window/test_stage2_relation.py`

**Interfaces:**
- Produces: `stage2_relation_distribution` and `stage2_relation_jsd`.

- [ ] **Step 1: Write failing tests**

```python
def test_softmax_uses_only_valid_evidence():
    c = torch.tensor([[[1.],[2.],[100.]]])
    mask = torch.tensor([[[True],[True],[False]]])
    q, valid = stage2_relation_distribution(c, mask, 1.0)
    assert q[0,2,0] == 0
    torch.testing.assert_close(q[:,:,0].sum(1), torch.ones(1))
    assert valid[0,0]

def test_single_evidence_jsd_is_zero():
    s = torch.tensor([[[.2]]], requires_grad=True)
    assert stage2_relation_jsd(s, torch.tensor([[[3.]]]), torch.ones_like(s,dtype=torch.bool), .7) == 0
```

Also verify only policy receives gradients and variable candidate sets remain finite.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_relation.py -q
```

- [ ] **Step 3: Implement** FP32 masked softmax and symmetric JSD. Exclude no-evidence windows, return exact zero for one-evidence windows, detach reference at engine boundary, add scheduled `lambda_rel` and `loss/relation`, and add no entropy regularizer.

- [ ] **Step 4: Run GREEN**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_relation.py tests/evidence_window/test_relation_kd.py tests/evidence_window/test_stage2_engine.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/evidence_window/stage2_engine.py src/evidence_window/losses.py tests/evidence_window/test_stage2_relation.py
git commit -m "feat: distill stage2 evidence relations"
```

---

### Task 12: Hidden Residual-Effect Distillation

**Files:**
- Modify: `src/evidence_window/stage2_effects.py`
- Modify: `src/evidence_window/stage2_engine.py`
- Modify: `src/evidence_window/losses.py`
- Create: `tests/evidence_window/test_stage2_hidden_residual.py`

**Interfaces:**
- Produces: `pooled_hidden_effect` and `stage2_hidden_residual_loss` over `[B,J,K,H]`.

- [ ] **Step 1: Write failing tests**

```python
def test_residual_is_mean_full_minus_mean_loo():
    full = torch.tensor([[[1.,3.],[3.,5.],[20.,20.]]])
    loo = torch.tensor([[[0.,1.],[1.,3.],[99.,99.]]])
    mask = torch.tensor([[True,True,False]])
    torch.testing.assert_close(pooled_hidden_effect(full,loo,mask), torch.tensor([[1.,2.]]))

def test_huber_ignores_invalid_edges():
    s = torch.tensor([[[[1.,2.]],[[100.,100.]]]], requires_grad=True)
    t = torch.tensor([[[[0.,2.]],[[0.,0.]]]])
    mask = torch.tensor([[[True],[False]]])
    assert stage2_hidden_residual_loss(s,t,mask,1.0).item() == pytest.approx(.25)
```

Verify gradients reach policy full and LOO branches but not reference targets.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_hidden_residual.py -q
```

- [ ] **Step 3: Implement** mean pooling on exact `R_k`, `full-loo`, equal-width validation, FP32 elementwise Huber averaged over hidden width and valid edges, and stopped reference targets. Do not add projections or residual subtraction.

- [ ] **Step 4: Wire** scheduled `lambda_res`, `loss/residual`, valid edge count, and residual norm. Request final normalized hidden states only when needed.

- [ ] **Step 5: Run GREEN**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_hidden_residual.py tests/evidence_window/test_stage2_effects.py tests/evidence_window/test_stage2_engine.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/evidence_window/stage2_effects.py src/evidence_window/stage2_engine.py src/evidence_window/losses.py tests/evidence_window/test_stage2_hidden_residual.py
git commit -m "feat: distill stage2 hidden evidence effects"
```

---

### Task 13: Config Matrix, Docs, and One-Batch Harness

**Files:**
- Create: `configs/evidence_window/stage2/{baseline,ew_reference_free,ew_precomputed_dpo,relation_kd,hidden_residual_kd,online_frozen_zero1,online_frozen_zero2}.json`
- Create: `tests/evidence_window/test_stage2_experiment_configs.py`
- Create: `tests/evidence_window/test_stage2_smoke.py`
- Create: `.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/stage2_large_model_smoke.py`
- Modify: `README.md`
- Modify: `documents/ours-trial1/guidance1.md`

**Interfaces:**
- Produces: source-edit-free variants and reusable CPU/7B smoke cases.

- [ ] **Step 1: Write failing config matrix**

```python
@pytest.mark.parametrize("name,mode,outputs", [
    ("baseline.json","none",frozenset()),
    ("ew_reference_free.json","none",frozenset({"token_logps"})),
    ("ew_precomputed_dpo.json","precomputed",frozenset({"token_logps"})),
    ("relation_kd.json","precomputed",frozenset({"token_logps"})),
    ("hidden_residual_kd.json","precomputed",frozenset({"token_logps","hidden_states"})),
    ("online_frozen_zero1.json","online_frozen",frozenset({"token_logps"})),
    ("online_frozen_zero2.json","online_frozen",frozenset({"token_logps"})),
])
def test_variant(name, mode, outputs):
    cfg = Stage2EvidenceWindowConfig.from_json(CONFIG_ROOT / name)
    assert cfg.reference.mode == mode and cfg.required_outputs == outputs
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_experiment_configs.py -q
```

- [ ] **Step 3: Add exact v2 JSON files**. Baseline is fully disabled; each other file enables only named losses. Pair online frozen with DeepSpeed `config_stage1.json` or `config_stage2.json`, never ZeRO-3.

- [ ] **Step 4: Add CPU and large-model smoke harnesses**. CPU tests parameterize baseline, EW, DPO, relation, and residual and assert finite losses, policy gradients, stopped reference, baseline one call, and EW `1 + E` auxiliary calls in addition to CE. The large-model harness accepts `--case`, `--model`, `--stage1-checkpoint`, `--data`, `--image-root`, `--sample-index`, `--ew-config`, `--deepspeed-config`, `--reference-artifact-dir`, and `--output-dir`; it executes one optimizer step and writes a small JSON result containing losses, edge/forward counts, and gradient witnesses.

- [ ] **Step 5: Document commands and semantics**

```bash
./venv/ilvr-venv/bin/python src/main.py \
  --stage stage2 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --load_model_path /absolute/path/to/stage1-checkpoint \
  --data_path /absolute/path/to/TRAIN.jsonl \
  --ew_config configs/evidence_window/stage2/ew_reference_free.json \
  --deepspeed_config configs/config_stage3.json
```

Document exact `Z_j/R_k`, marker-inclusive block, mean scoring, `O(1+E)`, artifact workflow, schedules, v1 legacy configs, and scientific limitations.

- [ ] **Step 6: Run GREEN**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_stage2_experiment_configs.py tests/evidence_window/test_stage2_smoke.py -q
```

- [ ] **Step 7: Commit**

```bash
git add configs/evidence_window/stage2 tests/evidence_window/test_stage2_experiment_configs.py tests/evidence_window/test_stage2_smoke.py README.md .superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/stage2_large_model_smoke.py
git commit -m "docs: add stage2 evidence-window variants"
```

Update the canonical guidance file through its absolute target, but do not stage the local `documents` symlink. If that target is intentionally outside this Git worktree, record it as a local continuation artifact rather than forcing it into the commit.

---

### Task 14: Review and Final Verification

**Files:**
- Modify only substantiated review findings.
- Record: `.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/verification.md`

**Interfaces:**
- Produces: reviewed final diff and one consolidated verification witness.

- [ ] **Step 1: Request code review** using `superpowers:requesting-code-review` with spec, plan, base `c415a55`, and current HEAD. Ask explicitly about Stage-1 preservation, mask locality, prefix slicing, reference ownership, and disabled computation. Use systematic debugging for failures.

- [ ] **Step 2: Run complete CPU suite once**

```bash
PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window -q
```

- [ ] **Step 3: Run static checks**

```bash
git diff --check c415a55..HEAD
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m compileall -q src transformers/src/transformers/models/qwen2_5_vl
```

- [ ] **Step 4: Inspect prior expensive evidence**. Existing Stage-1 EW results can cover only unchanged Stage 1. If no equivalent final-HEAD Stage-2 evidence exists, launch exactly one B200 matrix.

- [ ] **Step 5: Run the consolidated 7B/B200 matrix once**

Use the already validated local immutable model and data paths and sample 920:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python .superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/stage2_large_model_smoke.py --case all-single-gpu --model /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/models/Qwen2.5-VL-7B-Instruct --stage1-checkpoint /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/model_stage1 --data /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data --sample-index 920 --output-dir .superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/large-results
```

Before executing, verify the Stage-1 checkpoint directory exists; if it does not, use the actual approved Stage-1 checkpoint and record that exact command in the verification report rather than falling back to the base model. The harness covers Stage 1 baseline, Stage 2 baseline, reference-free EW, precomputed DPO, relation, and hidden residual. Run online frozen ZeRO-1 and ZeRO-2 as two-rank invocations using the same harness and `configs/config_stage1.json` / `configs/config_stage2.json`. For every case record dtype/device, losses, edges/forwards, policy gradient, reference/teacher no-gradient, and disabled auxiliary count. If online frozen is genuinely memory-infeasible, report it after its guard and isolated worker pass; do not substitute semantics.

- [ ] **Step 6: Inspect compatibility and scope**

```bash
git status --short
git diff --stat c415a55..HEAD
git diff --name-only c415a55..HEAD
```

Confirm Stage-1 loss/masks and Stage-2 direct CE tolerances. Confirm the local `documents` symlink is untracked.

- [ ] **Step 7: Write verification record from captured outputs**, including changed files, implemented objective, exact spans/intervention, complexity, configs, tests, gradients, limitations, and deviations. Reuse existing artifact/checkpoint witnesses; do not rerun or hash weights.

- [ ] **Step 8: Invoke `superpowers:verification-before-completion`**, inspect fresh evidence, then stage only exact reviewed paths and the verification record. Never use `git add -A`. Commit with `test: verify stage2 evidence-window training`.

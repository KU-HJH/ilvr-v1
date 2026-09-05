# Shared-prefix Sidecars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in clean/V/T grouped Stage 2B replay without changing losses or reference conditioning.

**Architecture:** Pure branch plans describe suffix rows and rectangular visibility. Explicit differentiable final-clean K/V feed checkpointed grouped replay inside one outer policy call; Trainer reuses its existing objective reducer.

**Tech Stack:** Repository Python venv, vendored Qwen2.5-VL Transformers, PyTorch SDPA/Flex, PEFT DoRA, DeepSpeed ZeRO-2.

**Spec:** `docs/superpowers/specs/2026-09-06-shared-prefix-sidecars-design.md` (approved commitment v2).

## Global Constraints

- Work only in `/root/workspace/ILVR/.worktrees/ours1`; retain user-owned worktree and legacy routes. Narrow final documentation corrections in the primary checkout are separately authorized by the spec.
- Run Python/tests using `PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python` or `./venv/ilvr-venv/bin/pytest`. No dependency changes.
- Checkpoints only under `checkpoints/`, results under `results/`, logs under `logs/`; never replace the checkpoint symlink.
- Clean limit 8192 is distinct from legacy combined packed cap 4608. New route never consumes old packed capacity certificates.
- Preserve frozen-reference own-latent scores, existing microbatch loss denominators, one Trainer backward, DoRA rank32/alpha64/dropout0.05 in production.
- Strict equivalence: >=2 layers, FP32, dropout off, atol=1e-5/rtol=1e-4 for every trainable gradient and objective component. Active dropout and BF16 are separate witnesses, not strict naive equivalence claims.
- Immutable explicit tensor K/V inputs/outputs, no detach/no_grad of policy state, no cache hooks, no per-branch backward. Whole suffix decoder + head + score reduction checkpointed nonreentrantly with RNG preserved.
- Dense is correctness backend; rectangular Flex and its CUDA measurement are required. No full training campaign, merge, or worktree cleanup.
- Use apply_patch for edits. Each task records RED/GREEN commands and exact output in its SDD report, self-reviews, and commits only its own files.
- Coordinates are zero-based, half-open; supported Stage2B microbatch is one right-padded sample. `spans`/`steps` are the existing collator's `ReasoningStepSpan` list and private step mappings. `a,b` mean chosen target bounds; `p` means encoded rejected completion token positions. Exactly two groups V then T, each chunk budget counts suffix input rows (not root). Embedding sources index the completed clean rollout embeddings; logical positions index the relevant clean or rejected full input's three-axis mRoPE. Invalid negative means the matching `StepNegative.valid_*_negative` is false. Mean logp and entropy use existing float32 selected-logit reduction definitions.

---

### Task 1: Pure sidecar planning and visibility

**Files:** Create `src/step_preference/sidecar.py`; create `tests/step_preference/test_sidecar.py`.

**Interfaces:**
- Consume `ReasoningStepSpan`, `StepNegative`, and collator `steps` dictionaries used by `build_packed_tree_plan`.
- Produce frozen `SidecarBranch` with `name`, `kind`, `step_id`, `token_ids: tuple[int,...]`, `clean_embedding_sources: tuple[int,...]` (-1 means direct embedding), `logical_positions: tuple[int,...]`, `clean_ancestor_end: int`, `blocked_start/end: int`, `selected_positions: tuple[int,...]` (branch-local prediction rows), `target_token_ids: tuple[int,...]`, `rejected_model_inputs: Mapping|None` (full own input for mRoPE).
- Produce frozen `SidecarPlan(root_length: int, root_selected_positions: tuple[int,...], groups: tuple[tuple[SidecarBranch,...],tuple[SidecarBranch,...]])`; groups are V then T.
- `build_sidecar_plan(*, clean_input_ids, attention_mask, spans, steps) -> SidecarPlan`.
- `chunk_sidecar_branches(branches, query_budget: int) -> tuple[tuple[SidecarBranch,...],...]`; retain order, no splitting, oversized branch alone.
- `sidecar_dense_visibility(root_length: int, branches, device=None) -> bool Tensor[Q,root_length+Q]`; no padding required for dense, each query sees its root ancestor except its own blocked range and causal rows of only its own branch.

- [ ] **Step 1: RED independent masks and boundaries.** Manually enumerate expected edges without planner predicates:

```python
expected = torch.zeros((2, 7), dtype=torch.bool)
expected[0, [0, 1, 5]] = True
expected[1, [0, 1, 5, 6]] = True
branch = SidecarBranch(name="s0_visual_off", kind="visual_off", step_id=0,
    token_ids=(92, 12), clean_embedding_sources=(3, 4), logical_positions=(3, 4),
    clean_ancestor_end=3, blocked_start=2, blocked_end=3,
    selected_positions=(0, 1), target_token_ids=(12, 13), rejected_model_inputs=None)
torch.testing.assert_close(sidecar_dense_visibility(5, (branch,)), expected)
```

Add actual fixture tests for first-token-different rejects, LCP boundary divergence, longer-than-root rejects, latent-end prediction input, invalid negative skips, no-step empty plan, ragged padding invisibility, empty target and unfinished-latent errors, chunk retention and oversized behavior.
Run `PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_sidecar.py`; expected initial missing module/API failure.

- [ ] **Step 2: Implement planner.** Reuse existing pure validation where valid, but do not inherit legacy nonempty-step or rejected-within-root limitations. Core replay boundaries:

```python
chosen_start = span.target_text_start - 1
chosen_stop = span.target_text_end - 1
rejected_start = min(common_prefix_length, completion_positions[0] - 1)
rejected_stop = completion_positions[-1]
selected = tuple(p - 1 - rejected_start for p in completion_positions)
```

Use full rejected own encoding for later position computation; validate selected IDs and strictly increasing in-range completion positions. Replay may start at latent-end token but not before completion of required latent pads. Root ancestor excludes replay start. Never share chosen target hidden rows. For each query, root visibility is `k < ancestor and not(blocked_start <= k < blocked_end)`; local visibility uses same-branch and local key<=query.

- [ ] **Step 3: GREEN and commit.** Run the focused file once green, record evidence, commit `feat: plan grouped shared-prefix sidecars`.

### Task 2: Differentiable decoder capture and dense grouped scoring

**Files:** Modify `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`; create adjacent `stage2_shared_prefix.py`; create `tests/step_preference/test_shared_prefix_model.py`.

**Interfaces:**
- Consume Task1 `SidecarPlan`, `SidecarBranch`, chunk and visibility functions.
- Top model accepts `stage2_shared_prefix_inputs: Mapping|None` with `plan`, `backend` (`dense` initially), `query_budget`, `checkpoint` (default True). Mutually exclusive with legacy packed inputs; only Stage2 supported.
- Model output adds `stage2_sidecar_results: tuple[Mapping,...]|None` and `stage2_sidecar_stats: Mapping|None`. Each reduced result contains `name`, `kind`, `step_id`, `mean_logp: scalar Tensor`, `entropies: Tensor[target_length]`; no full branch logits escape the checkpoint. Clean CE/selected logits keep old API.
- Decoder explicit kwargs `stage2_capture_kv: bool=False`, `stage2_prefix_key_values: tuple[tuple[Tensor,Tensor],...]|None`; capture output `stage2_key_values` immutable tuple of post-RoPE K/unexpanded V. Layer and attention kwargs/tuple return extend only opt-in paths; normal module hooks remain active.
- Provenance example: clean IDs `[11,91,90,90,92,12,13]` have latent slots 2,3. Existing rollout calls prefixes ending before each slot, differentiably replaces those two embeddings, then final clean decode captures all layer K/V from that single final call, never from intermediate rollout calls. Target `[5,7)` replays completed embeddings at `[4,6)` and shares final-clean keys `<4`; token92 at row4 is replayed as boundary. Rejected `[11,91,90,90,92,14,15]` with scored positions5,6 also replays `[4,6)`, uses own input IDs and mRoPE, and shares keys `<4`. Prefix and latent graph edges remain differentiable. Full rejected `model_inputs` retains collator keys `input_ids`, `attention_mask` and applicable `pixel_values`/`image_grid_thw`; no new image input is introduced.
- Dense replay positions combine chosen clean mRoPE gathers and rejected own `get_rope_index`; assert shared ancestor tokens/positions match. Shared K/V differentiable; no mutable cache side effects.

- [ ] **Step 1: RED two-layer dense oracle.** Build tiny two-layer FP32/dropout0 Qwen from `step_preference.equivalence` fixtures. Construct per-branch independently truncated inputs and literal shifted rectangular ablation mask; use existing full prefix clean model recomputation, never new planner visibility as expected output.

```python
mask = torch.ones(length, length, dtype=torch.bool).tril()
mask[target_start-1:target_stop-1, blocked_start:blocked_stop] = False
mask4d = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, torch.finfo(torch.float32).min)[None,None]
```

Assert branch mean logp/entropy matches, no-op suffix matches clean, latent-end boundary responds, rejected first-token-different clean scoring matches, longer rejects work. Compare accumulated gradients from selected clean+branch scalar scores with independent oracle all trainable parameters using exact stated tolerances, noting near-zero gradients. Run focused file to capture expected unsupported kwarg/output failure.

- [ ] **Step 2: Implement explicit capture/replay.** Reuse existing attention projection/rotary/GQA conventions. Every suffix layer projects only Q suffix rows, concatenates immutable root K/V once with group K/V, uses rectangular dense mask with `is_causal=False`, runs normal layer norms/MLP residual. Extend normal module forward dispatch rather than bypassing projection hooks. Keep existing clean rollout once; capture final clean K/V only. Return a focused ModelOutput extension rather than changing legacy tuple outputs.

```python
def score_chunk(embeds, positions, *flat_prefix_kv):
    hidden = decoder(inputs_embeds=embeds, position_ids=positions,
        stage2_prefix_key_values=tuple(zip(flat_prefix_kv[::2], flat_prefix_kv[1::2])),
        attention_mask=rectangular_mask, use_cache=False).last_hidden_state
    # Select prediction rows, apply lm_head, float32 log_softmax, return only
    # scalar logps and per-target entropies in deterministic branch order.
```

Nonreentrant checkpoint covers this entire computation; tensors including prefix K/V passed explicitly, RNG preserved. Avoid nested layer replay within whole-chunk checkpoint without changing clean layer policy. No-step path remains clean-only. Stats include logical groups, chunks, oversized branches, query rows, entropy/rejected scored rows; runtime-only counts measured separately in Task4.

- [ ] **Step 3: GREEN plus checkpoint witnesses.** Compare checkpoint on/off gradients; witness nonzero gradients in prefix-only inputs/latent generation parameters, immutable K/V and branch isolation; dropout active + checkpoint RNG equivalence to same shared path (not naive). Run focused new tests and existing `tests/stage2/test_stage2_modeling.py`, commit `feat: decode checkpointed sidecars from clean differentiable kv`.

### Task 3: Rectangular Flex and Trainer/config integration

**Files:** Modify `src/step_preference/sidecar.py`, adjacent Qwen helper/model from Task2, `src/stage2/trainer.py`, `src/step_preference/config.py`, `src/utils_deepseed.py`, `src/main.py`, `run_stage2_training.sh`; create `tests/step_preference/test_shared_prefix_trainer.py`, `tests/step_preference/test_shared_prefix_flex.py`; extend existing config/CLI tests as needed.

**Interfaces:**
- Consume Task2 top-model input and reduced result APIs.
- Config adds `shared_prefix_enabled: bool=False`, `sidecar_attention_backend: str="flex"`, `sidecar_query_budget: int=1024`.
- CLI adds `--step_shared_prefix_enabled`, `--step_sidecar_attention_backend {dense,flex}`, `--step_sidecar_query_budget`; shell env equivalents `STEP_SHARED_PREFIX_ENABLED`, `STEP_SIDECAR_ATTENTION_BACKEND`, `STEP_SIDECAR_QUERY_BUDGET`. Follow existing boolean parser style.
- Sidecar module adds `build_sidecar_flex_block_mask(root_length, branches, device)` returning rectangular BlockMask with dense-equivalent valid edges. Optional kernel padding must be invisible and excluded from scores/stats.
- Trainer dispatches shared path before legacy only when enabled. Calls existing `_reduce_stage2b_objective` with clean selected logps/entropies and reduced sidecar tensors, stored reference scores unchanged.

- [ ] **Step 1: RED objective/config/Flex tests.** Verify config rejects shared+legacy, shared outside Stage2B, nonpositive/bool budget, unknown backend. Ensure default disabled preserves old route. Add two-layer multistep test comparing naive/dense-shared components,total and all gradients; chunk budget 1 versus large preserves reduction.

```python
shared = dataclasses.replace(base_config, shared_prefix_enabled=True,
    sidecar_attention_backend="dense", sidecar_query_budget=1)
torch.testing.assert_close(shared_total, naive_total, atol=1e-5, rtol=1e-4)
```

CUDA tests compare dense/Flex scores and backward under same input and dtype, assert actual Flex invocation; compare BlockMask materialized valid edges against independent hand-written dense oracle, not the new predicate. Run CPU/config tests first RED, then CUDA test RED separately.

- [ ] **Step 2: Implement backend and routing.** SDPA clean remains configured; Flex handles rectangular suffix attention. Reuse installed integration/compile facilities. Preserve final clean CE and all current objective coefficients; invalid negatives skip DPO only, zero-step clean legal. Keep admission validation and sampler exclusively legacy; reject contradictory packed certificates in shared mode rather than silently consuming them. Production dropout stays 0.05.

```python
if config.shared_prefix_enabled:
    return self._compute_stage2b_shared_prefix_loss(...)
if config.packing_enabled:
    return self._compute_stage2b_packed_loss(...)
```

Run required reference/store preflight unchanged; no online scoring fallback. Add stats actual cached score entries consumed/missing (missing consumption error); never equate intended calls to runtime calls.

- [ ] **Step 3: GREEN/commit.** Focused shared/config/CLI and CUDA tests; legacy packed smoke plus ordinary Stage2 feature-off; commit `feat: integrate flex shared-prefix stage2b training`.

### Task 4: Runtime witnesses, benchmarks, reference audit, and handoff

**Files:** Extend `src/step_preference/equivalence.py`, `src/step_preference/benchmark.py`, `scripts/verify_packed_step_preference.py`, `scripts/benchmark_packed_step_preference.py`; extend `tests/stage2/integration/tiny_stage2_train.py` and focused distributed tests to exercise shared DoRA ZeRO2; create `tests/step_preference/test_shared_prefix_measurement.py`; update approved spec status and three source specs narrowly as named in design.

**Interfaces:** Preserve legacy defaults; verification/benchmark CLI accepts `--shared-prefix`, `--sidecar-attention-backend {dense,flex}`, `--sidecar-query-budget 1024`. Existing context loaders/checkpoint validation remain authority. Reuse `benchmark_training_path` timing rather than new runner. JSON measurement records explicit mode/backend/dtype/dropout/input lengths/valid scored tokens, warmup/measurement counts, elapsed forward+backward, throughput, peak allocated bytes and runtime counters.

- [ ] **Step 1: RED counter and smoke coverage.** Hook actual outer policy, decoder, rollout invocation, first-layer q_proj/MLP and lm_head (ordinary module hooks) to distinguish initial forward versus backward checkpoint replay; no hooks capture model state.

```python
assert counters["rollout_calls_forward"] == 1
assert counters["sidecar_projected_rows_forward"] == sum(len(b.token_ids) for g in plan.groups for b in g)
assert counters["entropy_scored_tokens"] == 2 * sum(s.target_length for s in spans)
```

Use actual span interval subtraction if no target_length property. Counter tests must compare observed calls/rows, not constants copied from configured group count. Distributed test launches current two-rank DoRA32/64/.05 ZeRO2 with gradient accumulation2, shared Flex, and asserts completed optimizer step/finite loss/gradients from emitted witnesses. Check no unrelated GPU process before launch.

- [ ] **Step 2: Implement measurements and run evidence.** Same deterministic input/loss/precision/adapter settings for naive/new/legacy wherever each fits; compile/warmup excluded, full backward included. Tiny tests: FP32 dense strict equivalence and BF16 Flex report errors separately. Benchmarks vary K at fixed total target length and growing target length, plus a small available real sample/long sample where existing context loader artifacts permit. Do not regenerate expensive missing negatives or restart cancelled full training. Record unavailable/OOM cases explicitly as incomplete.

Audit reference actual input call path; manifests omit external image bytes. Recompute a small available frozen-reference chosen/rejected score against stored scalars only with validated unchanged inputs; record exact compared values/tolerance/settings. If required original artifacts unavailable, report cache validity unresolved, do not invent hits or claim production validation. No generalized provenance system.

- [ ] **Step 3: Document/commit.** Correct three primary source docs' forward/cache statements: three logical groups with internal latent calls, frozen reference generates own latents, cache inputs must be fixed. Keep untracked user docs untracked. Write concise measured result under distinct `results/shared-prefix-<run-id>/` and logs under matching `logs/`; model files only checkpoint tree. Commit worktree source/test/docs files as `test: verify and measure shared-prefix sidecars`.

- [ ] **Step 4: Controller completion gate.** Request one whole-change review relative to `1912677`, fix concrete findings, choose fresh claim-specific tests once. Full branch finishing does not apply: no merge/worktree cleanup authorized. Report only witnessed completion; unavailable required distributed/Flex checks prohibit full completion or production-enable claims. Write ntfy summary before final response.

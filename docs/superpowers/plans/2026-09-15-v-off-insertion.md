# V-off Insertion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add genuinely image-free OFF execution with independent/cumulative visual insertion ON branches without changing existing methods.

**Architecture:** A pure token planner removes visual blocks and latent slots for OFF and maps original target tokens into new sequences. A separate scorer captures OFF KV, forks ON prefixes, reuses Stage2 latent rollout, and returns ON/OFF scores to the existing reducer. The CLI and distributed forwarding expose the optional route.

**Tech Stack:** Repository Python 3.11 venv, patched Transformers Qwen2.5-VL, PyTorch, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-15-v-off-insertion-design.md`, final design commitment v1, user-approved by “진행해라”.

## Global Constraints

- OFF: 원본 이미지 블록과 visual latent 슬롯을 제거한 토큰 배열을 실제 실행한다. `latent_start/end`와 기존 텍스트는 유지한다. 텍스트-only latent를 생성하거나 대체 벡터를 넣지 않는다. clean latent·hidden·KV를 재사용하지 않는다.
- ON: OFF의 삽입 직전 KV에서 분기하여 현재 `latent_start` 직전에 원본 `image_input` 블록을 넣는다. 기존 Stage2 rollout으로 그 분기의 visual latent를 새로 생성하고 같은 target 텍스트를 채점한다.
- `independent` starts each branch from original OFF; `cumulative` retains earlier ON image/latent/KV and repeats original-image insertion at each subsequent active latent block. OFF stays image-free.
- Same teacher-forced target IDs, Stage2 recurrence, context preference loss, clean CE. No helper-image features, text-only latent, extra image masking, reference model, or OFF stop-gradient.
- Preserve `clean_prefix`, existing `v_off_prefix`, all unrelated defaults and dirty user changes. Only new mode defaults to branch mode `independent`.
- Use `./venv/ilvr-venv/bin/python`, `PYTHONPATH=transformers/src:src`, apply_patch for edits. No training launch, no dependency installs, no remote mutation.
- All test logs under `logs/v-off-insertion-20260915/`; no model-bearing artifacts outside checkpoints. CPU tiny-model tests do not constitute training runs.
- Do not commit pre-existing dirty user changes. New files may be committed; changed tracked files must stage only new hunks or remain uncommitted and be reviewed as a working-tree diff.

---

### Task 1: Implement optional insertion preference end to end

**Files:**
- Create `src/step_preference/insertion.py` — immutable token plans and remapping.
- Create `transformers/src/transformers/models/qwen2_5_vl/stage2_v_off_insertion.py` — differentiable OFF/ON execution.
- Modify `src/step_preference/config.py`, `src/utils_deepseed.py`, `run_stage2_training.sh`, `src/multinode_runtime.py` — validation and forwarding.
- Modify `src/stage2/trainer.py`, `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`, `transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py` — optional dispatch only; no legacy semantic change.
- Create `tests/step_preference/test_v_off_insertion.py`, `tests/multinode/test_insertion_forwarding.py`; focused helpers can live in `tests/step_preference/insertion_helpers.py`.
- Update `documents/preference-prefix-options.md` with new mode separately from historical old mode.

**Interfaces:**
- Consumes existing `SidecarPlan` visual groups (their `blocked_start` is first latent pad; `blocked_end` is END index), original `input_ids`, config image/vision/latent token IDs, raw encoder embeddings and `image_grid_thw`. Raw encoder embeddings MUST NOT be `rollout.final_inputs_embeds`.
- Produces `build_insertion_plan(*, input_ids, sidecar_plan, image_token_id, vision_start_token_id, vision_end_token_id, latent_start_token_id, latent_pad_token_id, latent_end_token_id, branch_mode)`. Output immutable plan fields identify OFF token IDs, source indices, ON sequences, active latent spans, prefix length, target IDs, and predictor positions; exact dataclass field names may be chosen cohesively in this single task.
- Produces `score_v_off_insertion(model, options, input_ids, base_inputs_embeds, image_grid_thw)` returning existing `(sidecar_results, stats)` format with `visual_on` and `visual_off` per step. Scoring never accepts clean KV or clean-generated latent.
- Options add `prefix_mode='v_off_insertion'`, `branch_mode='independent'|'cumulative'`; config/CLI `step_preference_branch_mode`, environment `STEP_PREFERENCE_BRANCH_MODE`. Expose evidence ON/OFF metrics without substituting clean CE scores.

- [ ] **Step 1: Write and run the failing literal sequence planner tests.** Use synthetic IDs `Q=11, vision_start=78, image=80, vision_end=79, S=91, pad=90, E=92`. Two-block original is `[11,78,80,79,12,91,90,90,92,13,14,91,90,90,92,15,16]`. OFF must be `[11,12,91,92,13,14,91,92,15,16]`. Second independent ON must be `[11,12,91,92,13,14,78,80,79,91,90,90,92,15,16]`; cumulative must be `[11,12,78,80,79,91,90,90,92,13,14,78,80,79,91,90,90,92,15,16]`. Assert target IDs `[13,14]` and `[15,16]` and mapped predictor positions by hand, first prefix length2 and second independent prefix length6. Reject malformed image/latent blocks and unsupported options.

```python
def test_off_removes_all_visual_slots_but_keeps_boundaries():
    plan = build_insertion_plan(**two_step_inputs(), branch_mode="independent")
    assert plan.off_token_ids == (11,12,91,92,13,14,91,92,15,16)
```

Run `PYTHONPATH=transformers/src:src CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_v_off_insertion.py -q`; record expected missing-function/feature failure before code.

- [ ] **Step 2: Implement pure remapping and rerun planner tests.** Strip each validated original image block and every canonical latent pad interval; retain all other original IDs. Track `old_index -> new_index` for retained tokens. Insert the whole original image block set before active S and reinstate only its pad slots. Independent builds one active block per sequence; cumulative builds one sequence with all active blocks. If some latent blocks lack preference targets, strip them from OFF and leave them empty unless designated active; do not infer new targets. Valid target score positions are mapped target positions minus1. Reject missing source images for nonempty visual insertion requests. Return no branches for empty plans without inventing an objective.

```python
off_ids = tuple(ids[j] for j in retained_sources)
selected = tuple(old_to_new[j] - 1 for j in original_target_positions)
assert tuple(off_ids[p + 1] for p in selected) == target_ids
```

- [ ] **Step 3: Write failing native scorer/runtime tests before adding the scorer.** Use an existing tiny Qwen model with dropout0, sdpa backend and small valid vision grids. Invoke actual native forward/Trainer path, not only fake decoder. Observe OFF-first calls and exact pre-S KV reuse with test-side hooks. Assert original image value/resolution perturbation leaves OFF IDs/positions/KV/scores unchanged; image perturbation changes ON. Test raw image derivative is zero for OFF and finite/nonzero for ON; markers contribute in both. A two-step witness distinguishes independent reset from cumulative carry and confirms first-step agreement. Assert no clean-generated embedding enters OFF. Keep hook/tracing code test-only.

```python
torch.testing.assert_close(off_a, off_b, atol=0, rtol=0)
assert not torch.allclose(on_a, on_b, atol=1e-7, rtol=0)
assert len(observed_off_latent_calls) == 0
```

- [ ] **Step 4: Implement the scorer reusing actual OFF KV and the Stage2 engine.** OFF lookup calls `model.model.embed_tokens`, recomputes position IDs without visual grid, performs `model.model(..., stage2_capture_kv=True, use_cache=False, generate_mode=True)`. Slice each layer KV at the recorded prefix length without detach. For ON build token embeddings with inserted raw image features; generate only active spans via `model.stage2_rollout_engine.rollout` using a decoder callback bound to actual OFF KV and rectangular causal visibility. Compute full ON positions with repeated image grids, and assert pre-fork positions equal OFF. The callback accepts engine-local suffix tensors; latent spans/positions are suffix-local. Final scoring uses same branch context and no new image mask. Reuse existing checkpoint/scoring helpers where compatible; avoid copying whole scoring logic. Honor selected dense/flex backend or fail explicitly for an unsupported combination; do not silently change backend/dropout settings.

```python
prefix_kv = tuple((k[:, :, :prefix_length], v[:, :, :prefix_length])
                  for k, v in off_output.stage2_key_values)
# Engine recurrence remains unchanged; only its prefix-bound decoder differs.
rollout = model.stage2_rollout_engine.rollout(
    language_model=prefix_bound_decoder,
    base_inputs_embeds=suffix_embeds, attention_mask=suffix_mask,
    full_position_ids=suffix_positions, latent_spans=active_suffix_spans)
```

- [ ] **Step 5: Add failing native option/forwarding tests, then connect flags and reducer.** Add new mode to config and shared-input validation; require context preference Stage2B, V-only, boundaries kept, shared-prefix path, no legacy packing, `block_original_image=False`. Reject cumulative on old modes. Forward branch mode in CLI, launcher and both worker commands. In `_forward_stage2`, route insertion to raw `base_inputs_embeds`/grid, preserving clean CE execution. Trainer maps new `visual_on` like existing evidence-on result, computes existing loss and records genuine evidence metrics. Validate option behavior via actual parsing/config consumption and coordinator command construction, not grep-source tests.

```python
assert worker_environment["STEP_PREFERENCE_BRANCH_MODE"] == "cumulative"
torch.testing.assert_close(total, output.loss + 10 * (-torch.nn.functional.logsigmoid(
    .1 * (on_score - off_score))).mean())
```

- [ ] **Step 6: Verify native numerical and backward behavior.** Build a test-only full-sequence oracle using literal sequences/targets and Stage2 engine but NO new planner or prefix scorer. Compare first/second-step latent states, logits, mean logp, and parameter gradients in FP32/dropout0 against new KV route; start tolerances logits/latent `atol=1e-5,rtol=1e-5`, gradients `atol=2e-5,rtol=2e-3`, justify any change. Use finite and nonzero checks on preference-only ON/OFF gradients, not just CE+pref total. Run ordinary and nonreentrant checkpoint cases. Future target/image perturbations must not alter earlier scores. Test multiple images/order, variable image grid sizes, adjacent text entries (`until_next_latent`), empty target plan, and malformed combinations. GPU flex smoke is a forward/backward correctness probe only with available free resources; no optimizer, dataset training, remote mutation, or checkpoint writes.

- [ ] **Step 7: Focused regression, documentation and self-review.** Run new tests plus `test_evidence_gain.py`, `test_config.py`, `test_context_preference.py`, `test_context_sidecars.py`, `test_target_scope.py`, and multinode forwarding tests using repository venv. Describe commands and exact test output in report; distinguish skipped CUDA tests. Document new example flags and no convergence/accuracy claim. Stage only new files/new hunks; preserve pre-existing edits. Commit task-owned changes where safely isolatable and explicitly report any retained uncommitted task hunks for reviewer. Do not start training.

## Completion checks

- [ ] Task-scoped spec + quality review against the full task diff, including uncommitted task-only hunks if needed.
- [ ] Whole-change review, focused fix if needed, then claim-specific fresh verification.
- [ ] Report implemented modes, verified scope and remaining actual-GPU/training uncertainties. No branch merge/push/worktree cleanup without user choice.

## Execution tracking

Approval: user “진행해라” after final spec. Existing ours1 linked worktree retained; dependency reinstall unnecessary because venv is already functional. Standing AGENTS selects subagent-driven execution; no execution-choice question is needed.

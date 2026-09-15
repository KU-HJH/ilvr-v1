# Stage 2 Packed M Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group independent M suffix branches into one decoder call without changing normal Stage 2 latent generation or the fixed objective.

**Architecture:** Observe final F embeddings, original mRoPE positions, per-layer detached K/V and adapter dropout RNG. Execute independent contiguous M suffixes through the existing decoder with a fresh cache and explicit block-isolated additive visibility mask. Retain full replay as the default compatibility oracle; integrate packed scores into the existing runtime and coordinator.

**Tech Stack:** Repository Python 3.11 venv, PyTorch, patched Transformers Qwen2.5-VL, PEFT LoRA/DoRA, pytest, native CPU DDP.

**Spec:** `docs/superpowers/specs/2026-09-15-stage2-packed-m-design.md`, approved commitment v2 (sections 1–6 and 8). The fixed objective remains `documents/ILVR_Codex_Superpowers_Implementation_Prompt.md`.

Section 7 is the historical pre-approval boundary, not an additional design
requirement; the user has now approved commitment v2. Repository API signatures
must be traced locally; this plan is a roadmap plus binding interfaces, not a
replacement copy of framework code.

## Global Constraints

- Work only in `/root/workspace/ILVR/.worktrees/stage2-entropy-vision`; preserve pre-existing untracked files and other plans' scratch. Feature baseline is `516cf49`.
- Do not modify normal latent generation, F/reference call schedules, Stage 1, feature-off behavior, full-replay behavior, loss formula, text-vocabulary rules, windows or distributed reduction.
- `stage2_m_execution = full_replay | packed_suffix`, default `full_replay`; examples explicitly use `packed_suffix`. Missing saved mode means `full_replay`; explicit resume mismatch fails.
- Support the current eager/SDPA, FP32/BF16, B=1, native non-reentrant checkpoint and native DDP envelope. No expansion to ZeRO/FlashAttention/FP16/FSDP/compile.
- Preserve DoRA dropout **0.05**, exact module attributes, state-dict keys, F graph and outer RNG. M/cache/dropout reconstruction are graph-free.
- One active unmerged standard Linear LoRA/DoRA adapter at allowed language projection suffixes; ordinary non-inplace dropout `0 <= p < 1`, exactly one final-F invocation per stochastic site, contiguous rank-3 `[1,T,features]`. Fail explicitly on unsupported layouts/site counts; full tuning with zero native attention dropout is supported.
- Root keys `[0,T)`, packed query count `Q=sum(max(post_positions)-end_position+1)`, additive mask `[1,1,Q,T+Q]`, physical `cache_position=arange(T,T+Q)`, original three-axis RoPE coordinates gathered separately. Only selected predictor rows are projected.
- For each WindowBlock, `s_k=block.start_position`, `e_k=block.end_position`; logical suffix rows are every integer from e_k through max(block.post_positions), inclusive. `post_positions` are original predictor-logit indices; targets are `inputs["labels"][0, p+1]`, with existing selector validity unchanged.
- Branch k permits active root keys `j < e_k` excluding `s_k < j < e_k`, and its own active causal suffix keys only. No clean F END/post or other M branch access. START, images, prefix, logical positions and teacher-forced tokens remain intact.
- Packed counters: `model_forwards_M=0`, `decoder_forwards_M=1`, `m_branches=K`, `m_query_rows=Q`; all zero when M skipped. Counts must reflect runtime observations.
- FP32 comparison `atol=1e-6, rtol=1e-5`; BF16 autocast `atol=1e-3, rtol=1e-2`. Exact F outputs/RNG/serialized parameters; identical gradient None sets, finite and independently nonzero post-only latent gradients. FP32 leakage: safe M delta `<=1e-7`, F delta `>1e-4`, exposed END control `>1e-6`.
- No training, GPU training smoke, remote synchronization, merge or push. CPU unit/integration tests only; no GPU readiness, speedup or task-performance claim. No automatic chunking/OOM fallback.
- Use `./venv/ilvr-venv/bin/` exclusively, no dependency installation. All edits via apply_patch. Logs in `logs/packed-m-implementation-20260915/`, results in `results/packed-m-implementation-20260915/`, model-bearing CPU test artifacts in `checkpoints/` only.
- All test commands below assume `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/transformers/src:$PWD/src"` exported for the command. Capture RED before implementation and GREEN after; do not promote the throwaway prototype without production TDD.
- Task suites are scoped to the task. Root runs the project full CPU suite once at final verification, reusing that evidence for the finishing gate.

---

### Task 1: Observational capture and one-call packed suffix scorer

**Files:**
- Create: `src/stage2/packed_m.py` (capture, layout, replay/scoring; keep narrow responsibilities in clearly named private helpers).
- Create: `tests/stage2/packed_m_helpers.py` (shared real tiny-Qwen fixtures and full-replay oracle).
- Create: `tests/stage2/test_packed_m.py` (core execution/causality/stochastic contracts).

**Interfaces:**
- Consumes: `resolve_qwen2_5_vl_model(model)`, `WindowBlock` from `stage2.objective_windows`, `target_log_likelihood`, current decoder `DynamicCache` interface and actual `apply_multimodal_rotary_pos_emb` implementation.
- Produces: `PackedMPrefixCapture(model, root_length: int)` context manager. After the F context exits, its detached captured state remains usable by the scorer. Capture must be removed before scoring/backward. Public `score_packed_m(model, inputs: dict, blocks: tuple[WindowBlock, ...], capture: PackedMPrefixCapture) -> PackedMScores`.
- Produces: frozen `PackedMScores` with `scores: tuple[torch.Tensor, ...]` in block order, `decoder_calls: int`, `query_rows: int`; each score is graph-free full-vocabulary token-mean log-likelihood. No full-sequence placeholder logits.
- Produces test helper `make_packed_model(*, backend="eager", tuning="dora", checkpointing=False)` using genuinely indexed two-layer Qwen config, nonzero adapter deltas and dropout .05, and `make_packed_inputs(*, blocks=2, padding=None)` returning runtime-compatible inputs. Later tasks may extend fixture parameters without changing these defaults.

- [x] **Step 1: Write failing capture/scoring API tests with actual Qwen forwards.**

```python
from stage2.packed_m import PackedMPrefixCapture, score_packed_m

def test_packed_m_one_decoder_call_matches_independent_full_replays():
    model = make_packed_model()
    inputs = make_packed_inputs()
    # Select WindowBlocks through the existing objective selector, then run
    # normal F with captured pre-F RNG and the independent full-M oracle.
    with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as captured:
        model(**model_inputs(inputs), collect_stage2_diagnostics=True)
    scored = score_packed_m(model, inputs, selected.blocks, captured)
    assert scored.decoder_calls == observed_decoder_calls == 1
    assert len(scored.scores) == len(selected.blocks)
    assert all(not score.requires_grad for score in scored.scores)
    torch.testing.assert_close(torch.stack(scored.scores), oracle_scores,
                               atol=1e-6, rtol=1e-5)
```

Here `selected`, `oracle_scores`, and `observed_decoder_calls` are test-local values built with the existing selector, a real full-replay call under captured pre-F RNG, and a decoder hook respectively; they must not be mocked expectations. Inspect `tests/stage2/test_objective_trainer.py` and `.objective_ablation` for input/oracle patterns, but construct two actual indexed layers, not deepcopy of layer 0.

Chronology: save pre-F RNG; run unchanged F inside the capture context; exit
capture; run packed M from that capture while preserving post-F RNG; run each
independent full-M oracle under the saved pre-F RNG with the existing RNG replay
context; exit that context so outer RNG remains post-F. Observe the two paths
separately, without counting oracle calls as packed execution.

- [x] **Step 2: Run the new focused test and retain the expected missing-module/API failure.**

Run: `./venv/ilvr-venv/bin/pytest tests/stage2/test_packed_m.py -xq`.
Expected: missing packed API before production code exists, not an environment failure.

- [x] **Step 3: Implement capture, exact dropout replay, layout and selected-row scoring.**

```python
# Visibility for query original e+u in branch offset o:
allowed_root = active_root & (root_positions < e)
allowed_root &= ~((root_positions > s) & (root_positions < e))
allowed_suffix = own_branch & causal_local & active_suffix
# Build a fresh detached cache, never reuse/mutate the F cache object.
# Original position_ids.index_select(-1, logical_rows) is distinct from:
cache_position = torch.arange(T, T + Q, device=logical_rows.device)
# Project selected predictor rows only, then score their unchanged labels.
scores = target_log_likelihood(selected_logits, selected_targets)
```

Observe final F only (the unique full-original-length decoder call; latent calls are strict prefixes), capture actual post-PEFT K/V and actual rotary embeddings; detach/reconstruct K in no_grad. Do not require grad mode for identification: inherited Trainer evaluation can run F under no_grad. Save each supported dropout's pre-call CPU/site-device RNG and full input shape/dtype/device/layout. During M, regenerate full-shape masks with saved RNG under fork_rng and gather logical rows; restore RNG and exact previous instance forward attribute presence/value in finally. Do not initialize unused CUDA devices. Remove every hook after F including exceptions; prohibit a second capture invocation from overwriting state. Validate stochastic sites before/while capture and validate exact replay visitation. Score under no_grad using fresh DynamicCache and one direct decoder call; observe actual decoder calls. Empty blocks return empty scores and zero counters without execution. Add a focused eval/no_grad capture and scoring witness.

Use `peft_common.ALLOWED_QWEN_DECODER_SUFFIXES` as the allowed projection
inventory; the active Linear adapter's `lora_dropout[active_adapter]` is the
supported stochastic site. Track identity by module and one final-F invocation,
reject repeats and missing/unexpected replay sites. Read the actual PEFT forward
and Qwen attention dispatch locally before installing hooks; do not infer them
from class names or comments.

- [x] **Step 4: Add and run causal, numerical, cleanup and geometry regressions.**

```python
assert safe_m_delta <= 1e-7
assert normal_f_delta > 1e-4
assert deliberately_exposed_end_delta > 1e-6
assert torch.equal(rng_before_m, rng_after_m)
assert instance_forward_attributes_after == instance_forward_attributes_before
assert hooks_after == hooks_before
```

Use actual selected-logit observations, not counters alone. Cover eager/SDPA × FP32/BF16 × checkpoint on/off with active nonzero DoRA; LoRA and full tuning; no-intervention packed equals F; post windows 1/2, skipped targets, left/right padding, zero blocks, larger K/long suffixes; actual tiny image input and nontrivial mRoPE. Perturb selected latent values and test selected branch independence, cross-branch and END-cache negative controls plus unmatched-dropout sensitivity. Inject exceptions both in capture and M and verify exact restoration and subsequent normal F/backward. Reject repeated stochastic sites, multiple/merged adapters, nonstandard/inplace/layout mismatches; test CPU use does not initialize CUDA. Keep all comparison tolerances from Global Constraints.

Run: `./venv/ilvr-venv/bin/pytest tests/stage2/test_packed_m.py -q`.
Expected: all focused tests pass. Report actual matrix and gaps, not planned coverage.

- [x] **Step 5: Self-review and commit only Task 1 files.**

Run: `git add src/stage2/packed_m.py tests/stage2/packed_m_helpers.py tests/stage2/test_packed_m.py` then `git commit -m "feat: add graph-free packed Stage 2 M suffix scorer"`.

### Task 2: Runtime routing, configuration and resume compatibility

**Files:**
- Modify: `src/stage2/objective.py`, `src/stage2/objective_runtime.py`, `src/stage2/trainer.py`, `src/stage2/reference.py`, `src/utils_deepseed.py`.
- Test: `tests/stage2/test_objective_trainer.py`, `tests/stage2/test_objective_config.py`, `tests/stage2/test_objective_reference.py`; create `tests/stage2/test_packed_m_runtime.py` for full/packed comparisons.

**Interfaces:**
- Consumes Task 1: `PackedMPrefixCapture(model, root_length: int)` context manager, then `score_packed_m(model, inputs: dict, blocks: tuple[WindowBlock, ...], capture: PackedMPrefixCapture) -> PackedMScores(scores, decoder_calls, query_rows)`.
- Consumes fixtures: `make_packed_model(*, backend="eager", tuning="dora", checkpointing=False)` and `make_packed_inputs(*, blocks=2, padding=None)`.
- Produces appended config field `stage2_m_execution: str = "full_replay"`, matching CLI `--stage2_m_execution` choices and serialized key. Trainer fixed metric schema gains `m_branches`, `m_query_rows`; existing model/decoder counters retain literal meaning.

- [x] **Step 1: Add failing config, old-resume and actual-runtime tests.**

```python
assert Stage2ObjectiveConfig().stage2_m_execution == "full_replay"
packed = replace(config, stage2_m_execution="packed_suffix")
assert packed_result.metric_sums["model_forwards_M"] == 0
assert packed_result.metric_sums["decoder_forwards_M"] == 1
assert packed_result.metric_sums["m_branches"] == len(selection.blocks)
torch.testing.assert_close(packed_result.vis_sum, full_result.vis_sum,
                           atol=1e-6, rtol=1e-5)
```

Instantiate actual runtime/Trainer using existing helpers and use real selected blocks; capture logits/per-block scores through scoped observation hooks. Remove mode from saved objective identity to test legacy full-replay resume succeeds and packed mismatch fails, while unrelated identity corruption remains rejected.

- [x] **Step 2: Run focused new tests and retain RED.**

Run: `./venv/ilvr-venv/bin/pytest tests/stage2/test_packed_m_runtime.py -xq`.
Expected: new config/routing absent, before implementation.

- [x] **Step 3: Integrate narrowly with no changed F/reference or reduction logic.**

```python
use_packed = (config.stage2_m_execution == "packed_suffix"
              and config.lambda_vis > 0 and bool(selection.blocks))
capture_context = PackedMPrefixCapture(student, T) if use_packed else nullcontext()
with capture_context as captured:
    outputs, f_calls = counted_forward(model, unchanged_inputs, student=student,
                                      collect_stage2_diagnostics=True)
# Reference runs exactly as before. Packed M is called once, outside block loop.
packed_scores = score_packed_m(student, inputs, selection.blocks, captured) if use_packed else None
# In the unchanged block loop use packed_scores.scores[index], or full replay.
```

Use the existing unwrapped student for capture/direct graph-free M and the wrapped model for F. Preserve pre-F RNG capture for full replay; no capture for zero blocks, feature off, CE-only or entropy-only. Append dataclass field to preserve positional callers; use getattr default in from_args. Normalize only absent `stage2_m_execution` to `full_replay` in copied saved/expected config dictionaries during identity comparison; retain all schema/reference/vocabulary strictness and source-byte hashing. Do not rewrite old metadata to pass validation.

- [x] **Step 4: Complete runtime gradient and baseline regression witnesses.**

```python
assert set(full_grads) == set(packed_grads)
assert full_none_names == packed_none_names
assert all(torch.isfinite(g).all() for g in packed_grads.values())
assert all(g.abs().sum() > 0 for g in post_only_latent_grads)
assert not any(p.grad is not None for p in reference.parameters())
```

Compare actual F output, selected M logits/scores, total loss, parameter and retained latent gradients across full/packed, active DoRA/LoRA/full tuning, checkpointing and BF16. Compare F-only direct outputs, serialized parameters and outer RNG exactly. Record no graph on M/reference. Assert normal F/reference decoder counts unchanged as K grows while packed M remains one; zero/skipped paths make no capture call. Run existing Stage 1, legacy, CE-only, entropy-only and identity regression subsets. New metrics for full replay should consistently report actual branch count and conceptual suffix query rows without changing its calls.

Run: `./venv/ilvr-venv/bin/pytest tests/stage2/test_packed_m_runtime.py tests/stage2/test_objective_trainer.py tests/stage2/test_objective_config.py -q` plus the concrete reference/Stage 1 regression files identified in Step 1; record the exact complete command in the report.

- [x] **Step 5: Self-review and commit Task 2's explicit changed paths.**

Run: `git diff --check` then explicit `git add` for the Files list and `git commit -m "feat: route Stage 2 Vision gain through opt-in packed M"`.

### Task 3: Coordinator examples, native DDP witness and documentation

**Files:**
- Modify: `src/multinode_runtime.py`, `run_stage2_objective_examples.sh`, existing Stage 2 launcher only if its generic forwarding needs a change.
- Test: `tests/multinode/test_native_ddp_launch.py`, `tests/stage2/test_objective_distributed.py`, `tests/stage2/test_objective_examples.py`.
- Modify: `docs/stage2-objective.md`; create `results/packed-m-implementation-20260915/verification.md` with executed evidence and explicit remaining gaps.

**Interfaces:**
- Consumes Task 2 CLI/config `stage2_m_execution`, modes `full_replay`/`packed_suffix`, default full replay, and fixed metrics `m_branches`, `m_query_rows`, literal `model_forwards_M`/`decoder_forwards_M`.
- Produces existing coordinator allowlist/forwarding of `--stage2_m_execution`; all three existing examples explicitly select packed_suffix with no new launcher. No changes to mandatory topology/baseline or remote coordination.

- [x] **Step 1: Add failing coordinator/example and packed native CPU DDP tests.**

```python
assert resolved_config["stage2_m_execution"] == "full_replay"
assert "--stage2_m_execution packed_suffix" in normalized_dry_run_command
# Existing DDP fixture has rank block counts [2,1,0,0,0] and [0,0,0,0,1].
assert packed_logged_model_m == [0, 0, 0]
assert packed_logged_decoder_m == [2, 0, 1]
assert packed_logged_branches == [3, 0, 1]
```

Parameterize the existing real two-CPU-process native Trainer integration by execution mode while retaining its independent full-replay optimizer oracle, unequal/zero-block ranks and accumulation/remainder behavior. Avoid a mock DDP substitute. Check actual model/decoder hooks and unchanged reference/gradient results.

The three expected entries are global sums logged per optimizer update over
accumulation windows of 2, 2 and 1 microbatches, respectively. The first window
has two nonempty microbatches (both rank 0), hence two packed M decoder calls
for three branches; the middle has zero; the final has one branch on rank 1.

- [x] **Step 2: Run focused added checks and retain RED.**

Run: `./venv/ilvr-venv/bin/pytest tests/multinode/test_native_ddp_launch.py -xq`.
Expected: mode not forwarded or examples not selecting packed mode, before delivery edits. DDP tests may already pass due Task 2; do not invent a failure.

- [x] **Step 3: Extend existing forwarding and example command assembly.**

```bash
# Add to the existing common objective example arguments (do not add a launcher):
--stage2_m_execution packed_suffix
```

Add the value-bearing CLI flag to the existing allowlist and preserve generic forwarding. Update exact config expectations to include default mode. Test all three existing example variants through the actual coordinator's dry-run path with temporary readable inputs where existing tests require them; no training or remote writes. Preserve existing checkpoint/data/seed matching and canonical defaults.

- [x] **Step 4: Document measured counts and limits, then run covering tests once.**

Document opting in/out, old-resume behavior, dropout/backend limitations, mask/cache semantics, memory still unmeasured, and the following prediction counts excluding unchanged latent recurrence: CE=1, CE+entropy=2, CE+Vision=2, combined=3. Clarify one M decoder call is not one top-level model call and not an assurance of constant memory/work as K grows. Add three executable dry-run examples using existing wrapper syntax and same Stage 1/data/seed. Report actual CPU tests separately from unrun GPU/ZeRO/two-host training.

Run: `./venv/ilvr-venv/bin/pytest tests/multinode/test_native_ddp_launch.py tests/stage2/test_objective_distributed.py -q` plus actual objective-example test file(s); `bash -n run_stage2_objective_examples.sh run_stage2_training.sh run_multinode_stage2_training.sh`; `git diff --check`.

- [x] **Step 5: Self-review and commit only Task 3 files.**

Run explicit `git add` for changed Files paths and `git commit -m "test: verify packed M DDP and coordinator examples"`.

## Root verification and handoff

- [ ] Task-level spec and code-quality reviews must pass before the next task; one final feature review covers `516cf49..HEAD`, without re-reviewing already completed objective/DDP work below this feature baseline.
- [ ] Read finishing and verification skills before the final command. Run the repository project CPU suite once: `./venv/ilvr-venv/bin/pytest -q tests -m 'not gpu'` with the CPU environment above, full output saved under the designated logs directory; use that single run for both overlapping gates. This excludes the vendored upstream Transformers suite and explicitly gated GPU tests, matching the established project command.
- [ ] Separately inspect `git diff --check`, shell syntax and actual forward-count/causal witnesses from their specific test logs. Record full-suite failures honestly; do not count unrun GPU/large-checkpoint/two-host verification as passed.
- [ ] Keep the implementation on this feature branch/worktree pending the user's integration choice; do not merge, push, synchronize, train or delete user artifacts. Send required completion summary via the local ntfy summary hook before the final response.

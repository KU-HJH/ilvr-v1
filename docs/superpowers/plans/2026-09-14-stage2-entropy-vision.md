# Stage 2 Entropy and Vision Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved fixed three-term Stage 2 objective with faithful local attention ablation and regression evidence.

**Architecture:** Extend current main's no-cache Stage 2 rollout with an explicit per-forward intervention. Keep math/window construction, frozen reference ownership, and Trainer orchestration separate. Do not reuse ours1 V-off.

**Tech Stack:** Repository Python 3.11 venv, PyTorch 2.7.1, local Transformers 4.50.0.dev0 Qwen2.5-VL fork, PEFT 0.17.1, native Trainer/Accelerate.

**Spec:** `docs/superpowers/specs/2026-09-14-stage2-entropy-vision-design.md`, commitment v1 approved 2026-09-14; binding detailed requirements in `documents/ILVR_Codex_Superpowers_Implementation_Prompt.md`.

## Global Constraints

- Stage 1 is unchanged; feature off retains legacy behavior; enabled weights zero is the same self-generated-latent CE-only baseline.
- Fixed formulas: `dyn=-sigmoid((Hpre+eps)/(Hrefpre+eps)-(Hpost+eps)/(Hrefpost+eps)-gamma_H)`; `vis=softplus(m_V-(S_F-stop_gradient(S_M)))`; total is CE plus weighted block means.
- Entropy renormalizes actual text vocabulary in FP32. CE/S_F/S_M use full output vocabulary. Default boundary positions are s-1 and e. Marker CE remains supervised.
- Each M blocks edges `q>=e and s<j<e` in all decoder layers/heads, preserves image/tokens/positions, independently regenerates downstream latents, and never reuses F suffix state.
- F gradients reach latent generation. M/reference no_grad; reference additionally frozen/eval and separately loaded from original Stage 1 source. Explicit forward mask; no global mode flag.
- Defaults: enable=false, lambda_dyn=0, lambda_vis=0, reference_model_path=unset, gamma_H=0, m_V=0, entropy_eps=1e-6, pre_window=1, post_window=1.
- Active auxiliary support targets eager then SDPA, B1, FP32/BF16, native non-reentrant checkpointing, zero native attention dropout, and controlled baseline DoRA dropout 0.05. Reject FlashAttention2, effective sliding window, ZeRO-3. ZeRO-1/2 remain rejected without actual execution evidence.
- Auxiliary normalization is one valid-block mean across ranks and the complete optimizer accumulation window. CE retains existing reduction. Every rank participates with zero valid blocks too.
- Execute in `/root/workspace/ILVR/.worktrees/stage2-entropy-vision`; do not modify main or unrelated worktrees. Source baseline is `31f745e`. User-modified launchers on main are not copied or overwritten.
- Python/tests use `PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python` or the corresponding `pytest`. Reuse the validated venv; no dependency reinstall/download.
- CPU tests may run locally. All training including GPU smoke requires the existing two-node/four-rank coordinator and operator preflight. No full training/push/download without separate approval.
- Broad CPU regression commands set `CUDA_VISIBLE_DEVICES=""` and select `-m "not gpu"` so existing CUDA-availability guards cannot accidentally launch GPU integration tests. New actual-Qwen CPU BF16-autocast tests belong in Task4; scalar BF16 tests alone do not prove model/autocast compatibility.
- All model-bearing artifacts live under `./checkpoints/`, datasets under `./data/`, results under `./results/`, logs under `./logs/`. Preserve symlinks; test checkpoint fixtures also use a distinct temporary directory under checkpoints.
- Reproduction examples use same Stage 1 checkpoint/data/seed; four ranks, microbatch 1, accumulation 2, global batch 8, LR 5e-5, seed 42, warmup 10. Derive update budget, do not copy 431.
- TDD and task-scoped spec/quality review for each task. Children never spawn agents or invoke paperthin hooks. Report observed red/green commands and evidence, not assumed passes.

## File responsibilities

`stage2_latent_rollout.py` and Qwen modeling own explicit attention intervention propagation; `src/stage2/rng.py` owns scoped replay. `objective.py` owns config and fixed math; `objective_windows.py` owns actual token vocabulary and window metadata. `reference.py` owns separate reference loading and identity. `objective_runtime.py` owns per-sample branch orchestration; existing `trainer.py` owns window normalization/save/log hooks. `main.py` and `utils_deepseed.py` own entry wiring. Existing Stage 2 data owns assistant bounds. One runnable example wrapper and one operator document finish delivery.

### Task 1: Causal intervention and first-nail witness

**Files:** Modify `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py` and `modeling_qwen2_5_vl.py`; create `src/stage2/rng.py`; create `tests/stage2/test_objective_ablation.py`.

**Interfaces:**
- Produce model forward keyword `stage2_ablation: tuple[int,int] | None = None` (s,e) passed explicitly through `_forward_stage2` and `LatentRolloutEngine.rollout`; valid only in Stage 2.
- Produce `stage2_finite_ce: bool=False`, opt-in finite graph-zero for all-ignore labels; default unchanged.
- Produce `capture_rng_state() -> RNGState` and contextmanager `replay_rng_state(state: RNGState)` in `stage2.rng`, preserving outer CPU/initialized-CUDA RNG state without initializing unused devices.
- Keep existing logits/output fields and default forward behavior. No selected-logit output API is needed: later tasks index returned logits before any additional FP32 conversion.
- F means normal student forward; M means independently replayed masked forward. s/e are the actual start/end marker positions, e strictly inside sequence; example slots [3,4] mean (s,e)=(2,5). Require the pair to match a parsed nonempty latent span, reject bool/noninteger indices, padding markers, B>1 runtime, and intervention/finite-CE use outside Stage 2. Active intervention supports eager/SDPA only, not FlashAttention or sliding-window/native-nonzero-attention-dropout configurations. Errors are explicit ValueError/NotImplementedError with reason, not an exact-text contract.
- Finite CE applies when every shifted supervised label in the B1 forward is ignored, returning graph-connected FP32 zero; normal labels use untouched legacy CE. The gradient witness means loss ONLY on post-text predictions reaches the generated latent states. Compare all non-None trainable gradients between matched-RNG F-forward then M then F-backward versus F-only backward, and checkpoint on/off, with tight dtype-appropriate numerical tolerances and reported maximum errors. RNGState is an internal snapshot, not a serialization API; clone states, support nested replay, and reject incompatible initialized-device sets rather than silently perturb new streams.

- [x] Write failing tests using existing `tests/stage2/helpers.py:tiny_stage2_qwen`, two layers/two blocks, eager first. The intended new kwarg must cause the missing-feature failure; do not stop at an import/fixture error. Include deterministic generated-latent perturbation, not overwritten pad-embedding perturbation.

```python
# Hand-selected slots [3,4], marker e=5. Capture/inject at the existing
# embedding-replacement boundary through test monkeypatching, not a production
# test-only option. Compare fixed processed inputs with changed generated slots.
assert not torch.allclose(f_original.logits[:, 5:8], f_changed.logits[:, 5:8])
torch.testing.assert_close(m_original.logits[:, 5:8], m_changed.logits[:, 5:8])
torch.testing.assert_close(m_original.logits[:, :3], f_original.logits[:, :3])
```

- [x] Run RED: `PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/stage2/test_objective_ablation.py`. Save failure reason in task report/log.
- [x] Implement immutable per-forward (s,e), strict span/backend validation, original 2-D padding mask for RoPE, additive causal/padding mask for decoder calls, sliced to current prefix. Preserve defaults, positions and sequence. Core restriction:

```python
q = torch.arange(length, device=device)[:, None]
j = torch.arange(length, device=device)[None, :]
blocked = (q >= end) & (j > start) & (j < end)
# Combine with the actual causal and original key-padding restrictions.
mask = mask.masked_fill(blocked[None, None], torch.finfo(dtype).min)
```

- [x] Add first-nail tests for DoRA 0.05 paired no-intervention replay in train/no_grad mode, state restoration, downstream generated-latent invariance, F post-only latent gradients, and M-then-F-checkpoint-backward equality. Force informative nonzero adapter deltas. Test image/position/input preservation and SDPA separately; reject any unsupported mode explicitly.
- [x] Run GREEN plus existing `tests/stage2/test_qwen_forward.py` and `tests/stage2/test_rollout.py`. If first nail fails causally, use systematic-debugging and resolve before Task 2; do not change ablation meaning.
- [x] Self-review, record complete red/green evidence, and commit only Task 1 files.

### Task 2: Fixed math, vocabulary, windows, and opt-in settings

**Files:** Create `src/stage2/objective.py`, `src/stage2/objective_windows.py`; modify `src/utils_deepseed.py:parse_args`, `src/stage2/data.py:Stage2DataCollator`; create `tests/stage2/test_objective_math.py`, `test_objective_windows.py`, `test_objective_config.py`.

**Interfaces:**
- `Stage2ObjectiveConfig` frozen dataclass with the nine options in Global Constraints, `from_args(args)`, `validate(stage, supervise_latent_boundaries)`, `active` property (enabled and either weight positive), `to_dict()` for JSON metadata.
- `entropy_transition(h_pre,h_post,ref_pre,ref_post,*,eps,gamma_H) -> dict[str,Tensor]` returning r_pre,r_post,D,loss_dyn per block. Detach reference at math boundary.
- `vision_gain(s_f,s_m,*,m_V) -> dict[str,Tensor]` returning S_F,S_M,G,loss_vis per block; detach M.
- `text_entropy(logits,text_token_ids) -> Tensor` and `target_log_likelihood(logits,target_ids) -> Tensor`, FP32 selected-row operations, no mean across blocks inside these helpers.
- `TextVocabulary(token_ids: tuple[int,...], excluded: dict[int,str], output_size:int)` with deterministic serializable identity; `build_text_vocabulary(tokenizer,model_config,output_size) -> TextVocabulary`.
- `WindowBlock(batch_index,block_index,start_position,end_position,pre_positions,post_positions,target_ids)` frozen dataclass, position/ID collections immutable tuples. `WindowSelection(blocks: tuple[WindowBlock,...], excluded: dict[str,int])`.
- `select_objective_windows(input_ids,labels,attention_mask,text_segments,*,token_ids,latent_size,text_vocabulary,pre_window,post_window) -> WindowSelection`. `text_segments` is a per-batch list of half-open assistant text-token ranges, split at latent blocks; helpers support B>1.
- Collator optional `enable_stage2_objective=False`; only when on emit `stage2_text_segments` metadata. Derive verified token ranges from the actual unique assistant content bounds and strict latent markers, excluding chat termination. Do not retokenize prefixes with different context, alter rendered bytes, or use next-label validity as pre criterion.
- The existing assistant pattern excludes the template newline; preserve its legacy CE labels but exclude that formatting newline from objective P metadata. An empty pre-text plan must not acquire a valid pre window from the chat header. Tokenizer offsets on the complete rendered text, with verified final assistant-suffix token equality after processor image expansion, are an acceptable way to identify content boundaries; detect BPE-boundary ambiguity explicitly. Include a leading-newline/empty-P regression.

- [x] Write scalar/logit RED tests with independent literals, including:

```python
a = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
r = entropy_transition(a,b,a.detach(),b.detach(),eps=1e-6,gamma_H=0.0)
assert r['loss_dyn'].item() == pytest.approx(-0.5)
r['loss_dyn'].sum().backward()
assert a.grad.item() < 0 < b.grad.item()
scores = vision_gain(torch.tensor([0.0], requires_grad=True),
                     torch.tensor([0.0], requires_grad=True),m_V=0.0)
assert scores['loss_vis'].item() == pytest.approx(math.log(2))
```

- [x] Run RED on the three new test files before adding production helpers; use callable existence assertions inside tests if necessary so a missing implementation is an assertion failure, not collection error.
- [x] Implement fixed math exactly; natural-log text-only softmax on selected rows, full-vocab score gathers never using -100, nonfinite validation with location context, no clipping. Validate finite config, weights, integer windows, Stage2-only enable and marker supervision.
- [x] Implement tokenizer-resolved partition including actual non-special structural tool/FIM IDs and unmapped output rows, retaining ordinary added tokens. Enumerate exclusions/reasons and reject unresolved IDs/mismatches. Keep control-logit changes from affecting text entropy.
- [x] Implement eight-token literal alignment pre=(1,), post=(5,), target=(y0,); pre candidate positions are text-state positions in P. Remove any pre/post overlap from both sides simultaneously without refill; exclude empty windows with reasons. Reference later consumes identical final blocks. Test padding, assistant boundaries, zero blocks, malformed markers, short segments, B>1 helpers and B1 runtime error. Test actual collator output unchanged apart from feature metadata.
- [x] Run GREEN on new files and `tests/stage2/test_data.py`; verify BF16/extreme finite gradients, per-block nonlinearity versus nonlinear global mean, CE vocabulary unchanged. Commit Task 2 files with evidence.

### Task 3: Independent fixed reference and checkpoint identity

**Files:** Create `src/stage2/reference.py`; create `tests/stage2/test_objective_reference.py`; use existing `src/stage2/checkpointing.py:resolve_stage2_load_decision`, `src/stage2/peft.py:load_stage1_dora_for_fresh_stage2`, and `src/stage1/checkpointing.py` contracts without changing Stage 1.

**Interfaces:**
- `FrozenReference(model, identity: dict)` owns only a frozen model and JSON-safe identity; no optimizer/EMA ownership.
- `load_frozen_reference(args, *, student, tokenizer, source_dependencies) -> FrozenReference | None`; dyn zero returns None before any load. Dependencies use the existing Stage2SourceDependencies model/config/processor entry points. Avoid importing main into reference module.
- `validate_objective_resume(path, config_dict, reference_identity, vocabulary_identity)` and `save_objective_identity(path, config_dict, reference_identity, vocabulary_identity)` use a single `stage2_objective.json` sidecar. Reference path resolves to original Stage 1 source, never resume student. Missing/mismatched identity is a clear error.

- [x] Write RED tests with tiny locally serialized Stage1 full and schema-2 DoRA fixtures stored under a unique checkpoints test directory. Compare storage pointers, optimizer parameter IDs, token rows, eval/no_grad state, own latent outputs, and actual parameter values after student optimizer update/reload.

```python
before = {n:p.detach().clone() for n,p in ref.model.named_parameters()}
student_loss.backward(); optimizer.step()
for n,p in ref.model.named_parameters():
    assert not p.requires_grad and p.grad is None
    torch.testing.assert_close(p, before[n], rtol=0, atol=0)
assert not ({p.data_ptr() for p in student.parameters()} &
            {p.data_ptr() for p in ref.model.parameters()})
```

- [x] Run RED `PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/stage2/test_objective_reference.py`.
- [x] Implement separate source resolution/load using existing Stage1 full/adapter loaders including sparse marker rows. Freeze/eval, align tokenizer/output layout, preserve student RNG around construction, honor backend/dtype policy. No student deepcopy. Compute checkpoint/tokenizer identity once with hashes of load-relevant weight/config/tokenizer artifacts; do not hash optimizer shards or per microbatch.
- [x] Implement sidecar exact-config/reference/vocabulary checks and original-source reconstruction. Tests mutate source file/sidecar/config and require rejection; matched deterministic student/reference initialization yields D near zero. Config feature-off save behavior remains untouched.
- [x] Run GREEN and existing Stage2 source/checkpoint loader tests relevant to changed calls, record actual commands, commit Task 3.

### Task 4: Actual Trainer integration, global mean, and regressions

**Files:** Create `src/stage2/objective_runtime.py`; modify `src/stage2/trainer.py`, `src/main.py`; create `tests/stage2/test_objective_trainer.py`, `tests/stage2/test_objective_distributed.py`.

**Interfaces:**
- `Stage2ObjectiveRuntime(config,token_ids,latent_size,text_vocabulary,reference)` exposes `select(inputs)->WindowSelection`, `forward(model,inputs,selection,*,counterfactual_model=None)->ObjectiveBatchResult`. Trainer supplies its unwrapped current student as counterfactual_model: F uses the training wrapper, while no_grad M bypasses DDP forward buffer broadcasts. This is the same student, not a copy. Test with registered buffers and unequal M call counts; do not assume no_grad disables forward collectives.
- `ObjectiveBatchResult(outputs,ce,dyn_sum,vis_sum,count,metric_sums,metric_counts)` carries graph-bearing local sums and detached metric sums; no global mean inside runtime.
- Trainer optional kwargs `stage2_objective_config`, `stage2_objective_runtime`; legacy when absent/off. `get_batch_samples` attaches private per-window global count to each batch after calling parent and stripping nothing the parent needs. `compute_loss` strips objective metadata before model calls.
- Main config validation occurs before expensive model/reference loading; wire feature metadata collator, lazy reference and runtime after student initialization, and pass trainer kwargs only in enabled path.
- For enabled weights-zero, bypass auxiliary vocabulary/window construction and denominator collectives entirely: pass the collator metadata flag only when config.active, keep runtime None, and use F with opt-in finite CE. Save enabled config identity with absent reference/vocabulary fields. This keeps unused window parsing from rejecting a CE-only example accepted by legacy. Test zero-weight equivalence on empty-P/leading-newline trajectories too.

- [x] Write RED tests on actual tiny Qwen + Stage2Trainer (not just a toy compute_loss replacement). Include four modes, all-ignore labels, malformed/empty windows, F latent gradients, ref/M no graph, optimizer ownership, off/CE-only equality, resume sidecar mismatch. Tests witness exact forward counts and no helper tensors.
- [x] Run RED `PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/stage2/test_objective_trainer.py`.
- [x] Implement runtime F once with graph and diagnostics, reference once only when needed, each M no_grad within replayed F RNG state from original inputs and own rollout. M/reference omit unnecessary CE; select rows before auxiliary FP32 ops; release M logits sequentially. Sum blockwise nonlinear losses; detach only logging/M/reference.

```python
loss = result.ce
if global_count:
    scale = world_size * trainer_loss_divisor / global_count
    loss = loss + scale * (config.lambda_dyn * result.dyn_sum
                          + config.lambda_vis * result.vis_sum)
else:
    loss = loss + result.ce * 0.0
```

- [x] Implement count collection over actual upcoming accumulation batches, all ranks including empty, respecting last partial window and actual parent/Accelerate scaling. Force or verify native Trainer no-loss-kwargs scaling contract; do not use double division. Keep CE reduction untouched. Direct compute_loss outside training gets an explicitly documented single-batch denominator, including global count when distributed.
- [x] Add CPU Gloo subprocess regression using actual Trainer path at unequal counts, zero rank, remainder, and gradient accumulation; compare gradients/updates against independent literal concatenated-block oracle. CPU unit-test distributed processes are not GPU training jobs. Fail fast active-aux DeepSpeed, FlashAttention, sliding window, attention dropout, or unvalidated GC settings; test error occurs before model load.
- [x] Integrate existing save/resume metadata; global count-weighted logs for all required components/ratios/sign fractions/exclusions/low entropy/actual forward counts. Label absent metrics, separate backward-scaled scalar from logged objective. Preserve existing sampler/runtime hooks and feature-off behavior.
- [x] Run GREEN new integration files and the repository Stage 1/Stage 2 CPU regression suite once; record native hook/scaling witnesses and any pre-existing failures without weakening assertions. Commit Task 4.

Task4 evidence boundary: commit `abf7462`, task-scoped spec/quality review approved.
Post-remediation focused tests: 32 passed; separate original Stage1 mask symptom
plus eager/SDPA controls: 3 passed. PRE-REMEDY broad run: 1943 passed, 5 failed,
7 skipped, 3 deselected, 10 warnings. Three platform failures independently
reproduced on canonical baseline; the two other failures received focused fixes.
This does not establish a current-head passing full suite. Initial normalization
and multi-batch logging implementation preceded behavioral tests, violating TDD;
those portions were removed, causal RED observed, and reimplemented from tests.
That remedy does not retroactively erase the original process violation.
FP16 boundary validation and broad-run PEFT warning attribution are deferred
minor review items, not GPU support evidence. Task5/final verification remains open.

### Task 5: Reproducible examples and evidence-based handoff

**Current status, 2026-09-15:** Shared native-DDP resumed test repaircc9ba9f
passed scoped review and finalCPU2632passed/0failed/7skipped/3deselected/
10warnings421.86s. Prior four failures described below are historical and
resolved by test-only portability/fixture repairs, with production unchanged.
CPU gate passed; no GPU/real-host/deployment/merge claim. See the current
evidence section of docs/stage2-objective.md for exact commands and limits.

**Delivery update:** The subsequently approved native-DDP plan Task4 supersedes
this original wrapper interface and its sketch commands: it requires an explicit
update budget and port, and invokes the real coordinator with `--dry-run` rather
than skipping it. Use `docs/stage2-objective.md` for current executable examples.
Delivery67a7676 and remainder-doc fix41a6285 passed their task review. Shared
combined review and single fix092a95e/scoped re-review addressed all7findings.
Final CPU execution2626passed/4failed/7skipped/3deselected/10warnings429.00s:
three known platform failures and one new Stage1 test-double/backend mismatch.
Finishing was blocked at that revision; the resumed repair above resolved the
CPU failures without establishing GPU readiness.
Exact evidence, acceptance mapping and remaining limits: docs/stage2-objective.md.

**Files:** Create `run_stage2_objective_examples.sh`, `docs/stage2-objective.md`, `tests/stage2/test_objective_examples.py`; update plan checklist/evidence only, not frozen proposal text.

**Interfaces:** Wrapper accepts mode `ce|entropy|combined`, original Stage1 source, data/image roots, run ID and lambda values, validates inputs, invokes existing `run_multinode_stage2_training.sh` with explicit approved baseline and objective flags/env forwarding. Never launches all three implicitly; dry-run prints exact launch without starting ranks. If coordinator lacks passthrough, smallest feature-only CLI forwarding change plus causal test is allowed; do not port unrelated launcher changes from main.

- [x] Write RED executable-shell tests with temporary fake coordinator/environment boundary: modes preserve same source/data/seed and only intended lambda flags differ; missing paths, unsupported mode, destructive output collision fail; dry-run never executes coordinator. Assert observable argv/exit behavior rather than source grep.
- [x] Run RED `PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/stage2/test_objective_examples.py`.
- [x] Implement direct wrapper and operator doc with formulas/windows/vocabulary/ref identity/local ablation/gradients/forward cost/backend limits. Show three invocations with user-supplied paths/weights and distinct run IDs. Retained-sample-derived Stage2 budget must be supplied or validated by existing coordinator; no hardcoded 431. No new orchestration/recovery system.

```bash
./run_stage2_objective_examples.sh ce --stage1 "$STAGE1" --data "$DATA" --image-root "$IMAGES" --run-id "$CE_RUN" --dry-run
./run_stage2_objective_examples.sh entropy --stage1 "$STAGE1" --data "$DATA" --image-root "$IMAGES" --run-id "$DYN_RUN" --lambda-dyn "$LAMBDA_DYN" --dry-run
./run_stage2_objective_examples.sh combined --stage1 "$STAGE1" --data "$DATA" --image-root "$IMAGES" --run-id "$BOTH_RUN" --lambda-dyn "$LAMBDA_DYN" --lambda-vis "$LAMBDA_VIS" --dry-run
```

- [x] Run GREEN shell behavior tests and `bash -n run_stage2_objective_examples.sh`. Document actual CPU tests separately from NOT RUN four-rank GPU/ZeRO/real-data/full-training checks; include reproduction commands for those. Check read-only resource/checkpoint availability before deciding GPU smoke is runnable, not by assuming it.
- [x] Record all A–E criteria with exact test ownership/evidence; perform requesting-code-review over complete branch. Address findings through the designated fixer and focused rereview, then select final full project CPU suite with verification-before-completion/finishing-a-development-branch. Do not run upstream Transformers' unrelated enormous suite as ILVR regression evidence. The original four-failure gate was subsequently resolved by the resumed repair above.
- [x] Commit only delivery files; preserve implementation worktree pending human integration choice. Final response lists changes, tests/run limits, examples, and no claim of task performance gain.

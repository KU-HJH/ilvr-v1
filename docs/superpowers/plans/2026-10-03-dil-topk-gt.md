# GT-conditioned DIL Shared Union Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dil_bce_rf_topk_gt`, whose rejected probability and clean/OFF mass use exactly the same OFF Top-k union GT token IDs, preserving existing objectives.

**Architecture:** One token primitive constructs the deduplicated support and returns explicitly named conditional GT statistics. The existing reducer applies mean-logp-before-softplus; the shared-prefix sidecar carries scalar statistics through checkpoint recomputation. Selector routing and saved formula identity distinguish the new loss end to end.

**Tech Stack:** Repository Python 3.11 venv, PyTorch autograd, patched Qwen2.5-VL Transformers, existing PEFT/Trainer, pytest and Bash coordinator.

**Spec:** [2026-10-03-dil-topk-gt-design.md](../specs/2026-10-03-dil-topk-gt-design.md), version presented at `1c3eae3`; the subsequent user invocation of `$superpowers:writing-plans` authorizes this planning phase.

## Global Constraints

- New selector `dil_bce_rf_topk_gt`; formula identity `dil_bce_rf_topk_gt_v1`; preserve existing selectors, default selector and saved resume meanings.
- `S_t = unique(TopK(stopgrad(z_OFF,t), k).indices union {GT_t})`; rejected and both masses share S_t; never select clean top-k.
- `negative_j = softplus(beta * mean_selected_t(log q_GT,t))`; `positive_j = softplus(-beta * mean_selected_t(clean full-vocabulary GT logp))`.
- `mass_t = KL(Bernoulli(stopgrad(M_clean,t)) || Bernoulli(M_OFF,t))`; masses sum full-vocabulary probabilities over S_t.
- `2 <= k < |V|`; BF16 inputs produce FP32 statistics; S_t=V gives mass KL=0 and zero mass gradient.
- Fresh DIL defaults: beta=1, margin=0, preference weight=1, CE weight=0.1. Mass coefficient stays independently configurable with existing fallback 0; enabled-mass usage explicitly sets 1.
- Preserve full-vocabulary entropy selection, selected-token then original-step averaging, empty-step zeros, original-step denominator and distributed sample weighting.
- Use `./venv/ilvr-venv/bin/python`; preserve data/checkpoint symlinks. Test model artifacts belong under `./checkpoints/`, logs under `./logs/`, result metadata under `./results/`.
- No actual training launch, dependency installation, Reference imports, performance claims, merge/push or unrelated refactoring.
- Follow TDD and selected-method independent review. Record actual commands, reviews, agents and revisions in `docs/superpowers/2026-10-03-dil-topk-gt-task.md`.

## Review Focus

1. Equal OFF logits at the k boundary: reuse one support, including GT deduplication. Task 1 `test_gt_ties_share_one_support`.
2. Mixed full/partial union rows at k=V-1: zero KL/gradient for the full row, finite nonzero mass gradient for the partial row. Task 1 `test_gt_mixed_full_support`.
3. Entropy selection discards a whole step: zero contribution without removing the step denominator. Task 2 `test_gt_reduction_empty_selected_step`.
4. Lambda_P=0 and lambda_M>0: actual checkpoint replay retains OFF/prefix mass gradients; CE is not accidentally left unweighted. Task 3 replay test parameterization.
5. Omitted resume coefficients: restore saved settings, not fresh defaults; reject a different selector before parent Trainer restoration. Tasks 2/3 identity and entry witnesses.

---

## File Map and Execution Context

| Responsibility | Files |
| --- | --- |
| Shared support and token statistics | Modify `src/step_preference/dil_topk_loss.py`; create `tests/step_preference/test_dil_topk_gt.py` |
| Config, reduction and formula identity | Modify `src/step_preference/config.py`, `src/step_preference/dil_config.py`, `src/step_preference/dil_identity.py`, `src/step_preference/dil_reduction.py`, `src/utils_deepseed.py`; create `tests/step_preference/test_dil_gt_contract.py` |
| Real runtime and worker routing | Modify `src/stage2/trainer.py`, `src/stage2/planned_stop.py`, `transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py`, `run_stage2_training.sh` |
| Integration witnesses and usage | Modify `tests/step_preference/test_dil_trainer.py`, `tests/step_preference/test_dil_sidecar.py`, `tests/step_preference/test_loss_defaults.py`, `tests/stage2/test_dil_resume_entry.py`, `tests/stage2/test_dil_peft_roundtrip.py`, `tests/stage2/test_planned_checkpoint_stop.py`, `tests/multinode/test_dil_forwarding.py`, `docs/dil-bce.md` |

Paths are relative to `/root/workspace/ILVR/.worktrees/dil-loss-defaults-20261001`. Code base is `2f48c51985c395c16bea80a89847d7586728f9cd`; spec/control records were committed as `1c3eae3`.
At execution, apply using-git-worktrees detection to the existing worktree and preserve untracked caches. Do not create another worktree or reinstall the working environment automatically.

Set up execution-only environment:

```bash
export PYTHONPATH="$PWD/transformers/src:$PWD/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
DIL_GT_EXEC_ID="dil-topk-gt-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "logs/$DIL_GT_EXEC_ID" "checkpoints/$DIL_GT_EXEC_ID" "results/$DIL_GT_EXEC_ID"
```

Validate resolved checkpoint paths first. Before **each** pytest invocation, set `PYTEST_ADDOPTS="--basetemp=$PWD/checkpoints/$DIL_GT_EXEC_ID/<unique-phase>"` using a fresh phase directory. Never use an existing run as basetemp. Capture full stdout/stderr and exit status under `logs/$DIL_GT_EXEC_ID/` (enable pipefail if using tee). Baseline: `./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_dil_topk_loss.py tests/step_preference/test_dil_reverse.py -q`. Report baseline failures before implementation.

### Task 1: Shared-union token statistics

**Files:** Modify `src/step_preference/dil_topk_loss.py`; create `tests/step_preference/test_dil_topk_gt.py`. Keep `tests/step_preference/test_dil_topk_loss.py` as the original primitive regression witness.

**Interfaces:**
- Consumes existing `_group_gap` and `_weighted_off_cross_entropy` stable arithmetic.
- Produces `TopkGtTokenResult(NamedTuple)` with seven Tensor fields in order: `gt_logq, mass_loss, entropy, clean_mass, off_mass, mass_abs_gap, gt_coverage`. Every field is FP32 `[T]`; entropy uses S; coverage uses original K.
- Produces `topk_gt_token_terms(clean_logits: Tensor, off_logits: Tensor, target_ids: Tensor, *, k: int, compute_mass: bool = True, chunk_size: int = 128, stage2_debugging: bool = True) -> TopkGtTokenResult`. No beta argument: beta belongs in reduction.
- Internal extraction `_group_mass_terms(clean_chunk: Tensor, off_chunk: Tensor, support: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]` returns `(mass_loss, clean_mass, off_mass, mass_abs_gap)` for both original and GT primitives; old public interfaces stay unchanged.
- Test-only `gt_oracle(clean, off, targets, k) -> tuple[Tensor, ...]` returns the seven fields using independent FP64 probability arithmetic and per-row Python set union; never call production support or mass helpers.

- [ ] **Step 1: Write failing value/gradient tests.** Import the production module as `module`; test missing feature with a runtime callable assertion so collection itself succeeds. The core test is:

```python
def test_gt_union_values_and_live_gradients():
    clean = torch.tensor([[0.,1.,3.,2.], [0.,1.,3.,2.]], requires_grad=True)
    off = torch.tensor([[3.,2.,0.,1.], [3.,2.,0.,1.]], requires_grad=True)
    targets = torch.tensor([0, 2])  # K={0,1}; S={0,1}, then {0,1,2}
    assert callable(getattr(module, "topk_gt_token_terms", None))
    got = module.topk_gt_token_terms(clean, off, targets, k=2, chunk_size=1)
    want = gt_oracle(clean.double(), off.double(), targets, 2)
    for actual, expected in zip(got, want, strict=True):
        torch.testing.assert_close(actual, expected.float(), atol=1e-6, rtol=1e-5)
    assert got.gt_coverage.tolist() == [1., 0.]
    grad = torch.autograd.grad((got.gt_logq+got.mass_loss).sum(), (clean, off), allow_unused=True)
    oracle_grad = torch.autograd.grad((want[0]+want[1]).sum(), off)[0]
    assert grad[0] is None
    torch.testing.assert_close(grad[1], oracle_grad, atol=1e-5, rtol=1e-4)
```

Add the following named tests with these inputs/assertions:

| Test | Input and assertions |
| --- | --- |
| `test_gt_ties_share_one_support` | OFF `[2,1,1,0]`, clean=zeros, k=2. Choose GT as the ID among 1,2 absent from one observed original K. Assert coverage=0, clean_mass=3/4, and `exp(gt_logq)==exp(1)/(exp(2)+2*exp(1))`. The observed K is used only for this tie-specific oracle. |
| `test_gt_mixed_full_support` | OFF rows `[3,2,1,0]`, clean rows `[0,1,2,3]`, k=3, GTs `[3,0]`. Assert row0 masses=1, KL=0, exactly zero mass gradient; row1 finite nonzero mass gradient. |
| `test_gt_extreme_and_bf16` | OFF `[1000,0,-1000,-900]`, clean `[0,-900,1000,-1000]`, k=2, GT=2. Assert gt_logq≈-2000, finite fields and gradients. For ordinary BF16 inputs, compare to FP64 oracle using the actual BF16-quantized values. |
| `test_gt_chunking_and_mass_disabled` | Core input with chunk_size=1 and 128: assert values and gradients close. `compute_mass=False`: same gt_logq/entropy, all four mass fields zero. |
| `test_gt_invalid_inputs` | For V=4, k in `[1,4,True]`, target=-1/V, empty T, NaN/Inf logits, wrong target shape and chunk_size=0: assert `ValueError`. |

- [ ] **Step 2: Observe red.** Run `./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_dil_topk_gt.py -q`. Expected missing-primitive assertion failures; separate fixture/import mistakes from the intended red and preserve its log.
- [ ] **Step 3: Implement Task 1 interfaces.** Select K once per chunk; append GT with a validity mask to exclude duplicates from normalization and entropy. Derive the mass support from the same IDs. Use selected live logits with FP32 log-softmax and safely masked entropy. Extract the existing stable group-mass arithmetic; subset partial rows before calling `_group_gap`. Full rows have masses=1 and graph-connected zero KL; avoid summing huge logits to construct zero. Retain input checks and old primitive/wrapper behavior.
- [ ] **Step 4: Observe green and regression.** Run `./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_dil_topk_gt.py tests/step_preference/test_dil_topk_loss.py -q`. Expected all passed, including independent old-loss oracle.
- [ ] **Step 5: Commit the unit.** Stage only the two Task 1 files; `git commit -m "feat: add shared-union GT token statistics"`. Record red/green logs and commit. Apply the selected method's task review gate before proceeding.

### Task 2: Objective contract, reduction and identity

**Files:** Modify `src/step_preference/config.py`, `src/step_preference/dil_config.py`, `src/step_preference/dil_identity.py`, `src/step_preference/dil_reduction.py`, `src/utils_deepseed.py`; create `tests/step_preference/test_dil_gt_contract.py`.

**Interfaces:**
- Consumes Task 1 primitive/result.
- Produces `StepPreferenceConfig(preference_loss_type="dil_bce_rf_topk_gt", ...)`, accepted by existing `parse_args`, `validate_dil_config` and `validate_dil_runtime`; no new independent flags.
- Extends `reduce_dil_context_loss(clean_ce: Tensor, clean_token_logps: Sequence[Tensor], off_token_logps: Sequence[Tensor], *, config: StepPreferenceConfig, topk_terms: Sequence[TopkTokenResult | TopkGtTokenResult] | None = None, selection_entropies: Sequence[Tensor] | None = None) -> tuple[Tensor, dict[str, Tensor]]`; enforce the result type appropriate to the selector.
- Extends existing `build_dil_identity`, `read_dil_identity`, `validate_dil_resume` for `dil_bce_rf_topk_gt_v1`; old schema values stay unchanged.
- Test-only `gt_config(**overrides)` merges `_bce_config` updates with selector=GT, k=2, beta=1, CE=.1, mass=1; GT is the string `dil_bce_rf_topk_gt`. Reuse existing CLI builder from `test_loss_defaults` only for argv.

- [ ] **Step 1: Write reduction/contract tests.** Define a fixture with leaf clean logps `[-.1,-4.,-2.]`, `[-.7]`; OFF full-vocabulary GT logps `[-9.,-8.,-7.]`, `[-6.]`; gt_logq `[-.2,-3.,-.5]`, `[-.6]`; mass `[.2,.4,.8]`, `[.9]`; ce=2. Other diagnostic fields are finite same-shape zeros. Entropies `[8,7,1]`, `[0]` with top50 select only the first two tokens of step0. Call the real reducer and assert:

```python
# test_gt_reduction_empty_selected_step
want_pos = F.softplus(torch.tensor(2.05)) / 2
want_neg = F.softplus(torch.tensor(-1.6)) / 2
want_mass = torch.tensor(.15)
torch.testing.assert_close(metrics["loss/dil_positive"], want_pos)
torch.testing.assert_close(metrics["loss/dil_negative"], want_neg)
torch.testing.assert_close(metrics["loss/topk_mass"], want_mass)
torch.testing.assert_close(metrics["stats/topk_gt_logq"], torch.tensor(-.8))
torch.testing.assert_close(metrics["stats/logp_visual_off"], torch.tensor(-4.25))
torch.testing.assert_close(total, .1*ce + want_pos + want_neg + want_mass)
assert metrics["stats/selected_steps"].item() == 1
```

Also derive the expected loss from the fixture leaves and compare actual/expected gradients; assert zero for discarded token entries and the empty selected step (not softplus(0)). Add:

| Test | Assertions |
| --- | --- |
| `test_gt_independent_coefficients_and_empty_sample` | `(lambda_P,lambda_M)` in `(1,0),(0,1),(1,1)`; `total==.1*ce+lambda_P*(positive+negative)+lambda_M*mass`; CE gradient=.1; empty sample gives `.1*ce`. |
| `test_gt_config_defaults_and_invalid_contract` | CLI/direct/from_args: beta=1, margin=0, CE=.1, preference=1, omitted mass=0, explicit mass=1. Reject nonzero margin/entropy, bad k, unsupported prefix/missing shared-prefix. Accept both target scopes. |
| `test_gt_identity_resume_roundtrip` | Formula version exactly `dil_bce_rf_topk_gt_v1`; save/read equality. Reject old→GT and GT→old for both old top-k variants and corrupted formula version. |
| `test_gt_resume_omission_retains_saved_coefficients` | Save GT identity with beta=.5, CE=1, margin=0, mass=1. Existing resume CLI resolution restores omitted CE/beta; explicit mismatches fail. |
| `test_gt_statistics_type_and_shape_validation` | Old result type passed to GT, GT type passed to old selector, or wrong-length gt_logq: assert `ValueError`, not reinterpretation. |

- [ ] **Step 2: Observe red.** Run `./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_dil_gt_contract.py -q`. Expected selector/config/parser/identity/reducer rejection; record the actual failures.
- [ ] **Step 3: Implement Task 2 interfaces.** Extend selector allowlists, fresh-default resolver and resume property dispatch in the listed files. Add the formula without changing old identity `_FIXED` values. Reducer explicitly uses GT `gt_logq[mask].mean()` then one softplus; add `stats/topk_gt_logq` only for GT, selected-token then original-step mean with empty step=0. Preserve full-vocabulary off-score metrics, positive and independent coefficients. Keep empty-step graph participation for all new statistics.
- [ ] **Step 4: Observe green and regression.** Run `./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_dil_gt_contract.py tests/step_preference/test_dil_config.py tests/step_preference/test_dil_identity.py tests/step_preference/test_dil_reverse.py -q`. Expected all passed. This proves the local contract, not actual model dispatch yet.
- [ ] **Step 5: Commit the contract.** Stage Task 2 files; `git commit -m "feat: define GT DIL reduction and resume identity"`. Record evidence and complete the selected method's task review gate.

### Task 3: Actual runtime, worker commands and artifact compatibility

**Files:** Modify all runtime/integration/usage files in the File Map. `src/multinode_runtime.py` already forwards selector/mass settings: modify only if a forwarding witness exposes a missing route, without changing coordinator protocol.

**Interfaces:**
- Consumes Tasks 1/2 primitive, typed result, config, reducer and identity.
- Existing shared-prefix `dil_objective={"loss_type":GT,"beta":float,"k":int}` is accepted by `validate_shared_inputs`; `_score_chunk(..., dil_objective=..., dil_clean_logits=...)` calls the GT primitive without beta.
- `score_sidecars(...)` returns `dil_tokens: TopkGtTokenResult` for GT, old `TopkTokenResult` for old variants. Both have seven fields: preserve the 10-value-per-branch layout (mean logp, entropy, full-vocabulary target logps, seven scalar token fields). Reconstruct by selector.
- Existing `Stage2Trainer.compute_loss`, `_compute_context_preference_loss`, `_reduce_context_objective`, `train` signatures stay unchanged; new selector reaches correct reducer, validation and same-run resume.
- Existing env/CLI selects GT; both generated worker commands retain the values. Documentation uses the existing coordinator `--dry-run`, explicitly setting beta=1, margin=0, CE=.1, preference=1, mass=1.

- [ ] **Step 1: Add actual-forward tests before dispatch code.** Extend `dil_config`, `full_logit_oracle` and GT parameterizations in `test_dil_trainer.py`. Independent oracle uses per-row Python union of OFF argsort IDs and GT, full-vocabulary probabilities for both masses, `log(p_GT/p_S)` for the rejected score, mean before softplus. Initialize oracle total (including no-span return) with `config.lambda_ce * output.loss`, not unweighted CE. Never call the production primitive/helper. GT cases explicitly use beta=1, CE=.1 and mass=1; include normal, lambda_P=0/mass=1, top20-clean, top20-voff, empty sample and invalid-boundary/vocabulary pre-forward cases. Keep historical non-GT coefficients and assertions unchanged. Retain existing assertions:

```python
# test_actual_unequal_step_loss_and_parameter_gradients_match_full_logits
torch.testing.assert_close(actual, oracle, atol=1e-6, rtol=1e-5)
compare_grads(actual, oracle, model)
# test_sidecar_returns_only_per_token_statistics (new selector case)
assert isinstance(result["dil_tokens"], TopkGtTokenResult)
assert all(t.ndim == 1 for t in result["dil_tokens"])
```

Extend `test_dil_sidecar.py` corruption, checkpoint replay (normal and mass-only), and call-count cases. Preserve nonzero prefix-KV/latent gradients and checkpoint-on/off equality assertions; subtract `config.lambda_ce*out.loss` when isolating preference. Include GT in `test_loss_defaults.SELECTORS` for its real-forward CE-once witness. Keep all old cases.
- [ ] **Step 2: Observe runtime red.** Run `./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_dil_trainer.py tests/step_preference/test_dil_sidecar.py tests/step_preference/test_loss_defaults.py -q`. Expected new-selector dispatch/type failures; preserve actual evidence.
- [ ] **Step 3: Connect runtime.** Update shared-input validation, `_score_chunk` primitive choice and `score_sidecars` typed unpacking. Extend every top-k/DIL path in `src/stage2/trainer.py`: pre-forward checks, sidecar options, step/target alignment, result collection and resume initialization. Pass GT terms to Task 2. Preserve clean GT scoring, full-vocabulary selection entropies, checkpoint placement and clean mass detach.
- [ ] **Step 4: Verify runtime green.** Re-run Step 2 with fresh basetemp. Expected all passed, including real values/gradients and replay witnesses. Record this as CPU tiny-model evidence only.
- [ ] **Step 5: Add worker/lifecycle tests before routing changes.** Extend `test_dil_forwarding.py` fresh/resume cases with GT, k=1516, mass=1, percent=20, source=clean and retain existing per-worker argv assertions. Add omitted-beta fresh case asserting both workers resolve beta=1, margin=0, CE=.1 and mass=1. Extend GT helper recognition and parameterizations in `test_dil_resume_entry.py` and `test_dil_peft_roundtrip.py`; retain assertions on saved identity, loaded logits/latents and parent Trainer dispatch. Assert old↔GT rejection occurs before parent restoration. Extend parser-only planned-stop case with GT/k=8; assert max_steps=3 and stop=2.
- [ ] **Step 6: Observe routing red.** Run `./venv/ilvr-venv/bin/python -m pytest tests/multinode/test_dil_forwarding.py tests/stage2/test_dil_resume_entry.py tests/stage2/test_dil_peft_roundtrip.py tests/stage2/test_planned_checkpoint_stop.py -q`. Expect GT failure in shell selector/resume or planned-stop gate. A route already passing needs no forced production change.
- [ ] **Step 7: Connect lifecycle and document usage.** Extend fresh/default and resume branches in `run_stage2_training.sh`, and selector gate in `validate_planned_checkpoint_stop`. Resolve demonstrated coordinator forwarding gaps only. Add `docs/dil-bce.md` formulas, shared support, pre-union GT coverage, conditional metric, full-union zero mass, old options and identity/default semantics. Add the existing-coordinator dry-run example; do not run training or contact remote workers for documentation.
- [ ] **Step 8: Verify lifecycle green.** Re-run Step 6 with fresh basetemp; expected all passed. Run `bash -n run_stage2_training.sh`; expected exit 0. Record worker argv and save/reload witnesses, not a claim of four-rank participation.
- [ ] **Step 9: Commit integration.** Stage changed Task 3 files; `git commit -m "feat: connect GT DIL sidecars and worker lifecycle"`. Record revision and complete applicable task/final review gates.

## Final Review, Verification and Handoff

- [ ] Obtain independent specification-and-quality review using requesting-code-review and the selected execution method. Provide spec, plan, base `2f48c51985c395c16bea80a89847d7586728f9cd`, actual implementation HEAD and task evidence. Require call/data/gradient tracing; record unresolved findings. Self-review is not a substitute.
- [ ] Route fixes through the selected method, add causal tests and re-review as required. Do not add speculative hardening or repeat broad reviews after approval.
- [ ] Invoke verification-before-completion and finishing-a-development-branch before choosing final commands. Run the project's full suite once on the reviewed revision: `./venv/ilvr-venv/bin/python -m pytest -q`, with fresh checkpoint basetemp and complete log. Read all output and report failures/skips. Reuse this result for both gates' overlapping test-pass claim; do not run the identical suite twice.
- [ ] Run `git diff --check`; reconcile each spec requirement to an actual test/observation. Record commands, exit codes, totals, reviews and reviewed commit/diff in the workflow record. Extra verification only for distinct claims or changed/failing code. No real GPU/multi-node training is authorized here.
- [ ] After all witnesses pass, present the finishing skill's integration choices. User owns merge/PR/keep decisions. Preserve the worktree/caches and write the required ntfy summary before the final report.

## Plan Self-Review and Approval Status

Coverage: spec §§1–3 → Task 1 and Task 2 selector; §4 → Task 2 plus Task 3 runtime oracle; §5 → Task 1; §6 → all interfaces; §7 → Tasks 2/3; §8 → tests and final gates; §9 → handoff. All five Review Focus items have owning tests.
Type/order check: seven-field GT result → 10-value sidecar tuple → selector-specific reconstruction → `gt_logq` reduction. Old `negative_loss` is never repurposed. Unchecked boxes remain unexecuted; this plan is not evidence that tests or implementation ran.

User must review this written plan and choose Native or Subagent-driven before implementation. No method has been selected. Recommendation: **Native**, because these three tasks share one small statistics interface and one continuous runtime path that one independent whole-change reviewer can assess together.

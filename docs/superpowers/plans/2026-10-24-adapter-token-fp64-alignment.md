# Adapter Token FP64 Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. The user already selected this method. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test an explicit Stage1 latent-OFF arithmetic profile that removes the demonstrated genuine KV-OFF/ON numerical disagreement, adopting no candidate without complete finite exact-tensor evidence.

**Architecture:** Keep native generation and global precision controls. A focused context in `adapter_numerics.py` observes actual standard DoRA call roles and changes only token A/B linear arithmetic to temporary FP64, returning FP32; one shared execution scope connects evaluator, controls, probe and replay. A short original-operand precheck precedes the original32-token counterexample, with a strictly conditional MATH-SDPA candidate and source0/3 confirmation.

**Tech Stack:** Existing repository Python3.11 venv, PyTorch2.7.1+cu128, PEFT0.17.1, native patched Qwen2.5-VL, pytest; no dependency changes.

**Spec:** [Exact approved design commitment v4](../../../results/kv-adapter-precision-20261024T091107Z/design-commitment-v4.json); approval/hash in adjacent `approval-v4.json`. Baseline code `f10461735fc107e5cd769ed7490838a6f974a767`; actual diagnosis and replay evidence in `results/kv-mlp-isolation-20261024T081537Z/outcome-v3.json`.

## Global Constraints

- Existing chosen subagent-driven TDD plus independent task/final review and one final fullCPU. This multi-part arithmetic/evaluator/probe/identity integration triggers writing-plans after design approval; its separate later human plan review remains required.
- Temporary FP64 does not guarantee shape invariance or a complete numerical fix. Other base/attention/later-step differences may remain. New-mode outputs may differ from historical OFF. No fullTEST400, universal equivalence or speed claim.
- Only explicit continuous Stage1 latent-OFF profiles are new: `adapter_token_fp64` and `adapter_token_fp64_math_sdpa`. Existing `historical`, `tf32_off`, `full_accumulation`, `math_sdpa`, their payloads/defaults and historical artifacts retain their meaning. No training, dependency, Reference, host/container, coordinator or native generation changes.
- Only actual token LoRA A/B linear operands/results use temporary `torch.float64`, with returned `torch.float32`. Preserve Parameter dtype/storage/values, base projections and the actual2D identity reconstruction/normalization path. This explicitly permits transient arithmetic Tensor conversion; it never permits Parameter/model/checkpoint casting, merge or save.
- Actual supported DoRA roles/bindings must be validated; unknown paths fail explicitly. Same shared arithmetic must reach real evaluator/control/observed/replay calls after loading. Restore original instance-forward presence/value and global precision states on normal/exception exits and supported nesting; reject incompatible nesting before mutation.
- Real OFF stays full prefix, `use_cache=False`, no past/returned cache; ON stays native cached prefill/incremental. No recomputation fallback, token override, EOS/argmax change or threshold relaxation. First source12 acceptance requires32 predictions; additional0/3 cap32 preserve natural EOS and report actual lengths.
- A local replay pass only permits source12 full32 actual-profile test. Require exact finite selected A/B + first Q + every raw/post logit + full suffix + native cache geometry. Confirm0/3 at the first complete passing profile.
- First remaining difference must be an SDPA output with equal actual Q, complete K/V and visibility and no preceding unequal projection; otherwise do not test MATH.
- Root alone runs all newly approved GPU experiments, including prechecks/candidates/confirmations, in aggregate<=200process-seconds; previous91.73137581901392seconds plus new work must remain<=300seconds. Budget exhaustion, incomplete evidence or a failed candidate never becomes a pass. Implementers/reviewers run CPU only.
- Use only `venv/ilvr-venv`; datasets/cache under current `data/`, outputs under current `results/kv-adapter-precision-20261024T091107Z/`, logs under matching `logs/`. All model-bearing pytest fixtures use a unique explicit basetemp under canonical `checkpoints/kv-adapter-precision-20261024T091107Z/`; mkdir parent first and preserve symlinks. No tensor/model dumps from the probe.
- TDD each added behavior before implementation. Preserve actual RED/GREEN commands/output and immediately report a procedure deviation before affected progression. Prior diagnostic/helper chronology violations and replay limitations remain historical; no retroactive compliance assertion.
- One final full project CPU invocation on final independently reviewed bytes after required fixes, then commit identical reviewed/tested source/test bytes. Do not commit task source or start GPU before those gates. Content-hash/diff task reviews precede the final commit. No repeated broad suite for identical content.

## Review Focus

1. A norm-reconstruction call or unexpected token shape/binding must never silently receive token FP64 arithmetic; real DoRA call-role/2D preservation and invalid-role tests own this.
2. Nested probe/evaluator scopes or an exception must not double-wrap, leave hooks/forward overrides, lose incoming `medium`, or leak into later historical generation; ownership/subset/nesting/restoration tests own this.
3. Pre-generation identity metadata has no observed arithmetic yet; row/control evidence must come from actual calls in its own interval, never stale counters or parameter dtype inference; runtime observation tests own this.
4. A short/local/token-only/missing/nonfinite witness cannot be a complete correction, and MATH cannot cover an earlier projection/mask/cache mismatch; precheck/verdict/eligibility tests own this.
5. Unsupported models/static-cache combinations and runtime role errors must not publish a spurious successful zero-accuracy run or mix shard profiles; global validation/fatal-error/real-merger tests own this.

## File Map

- Create `src/evaluation/adapter_numerics.py`: supported DoRA discovery/role-aware temporary A/B arithmetic, scoped binding restoration, interval evidence. Lazy PEFT imports keep old policy import behavior small.
- Modify `src/evaluation/numerics.py`: add the two IDs, map global controls, shared execution context, model-validation seam and explicit arithmetic observations.
- Modify `eval.py`: validate new execution compatibility before sample outputs, use shared execution scope and observe actual arithmetic after generation, keep role errors fatal and historical error boundaries unchanged.
- Modify `scripts/probe_stage1_off_kv.py`: same shared scope for all actual/replay paths, opt-in operand retention/direct comparison, cheap precheck, extended profile/component validation and MATH eligibility. No new campaign/retry framework.
- Modify `src/multinode_evaluation.py`: pass static-cache state into central numerical validation; retain existing normalization/mismatch logic for new IDs.
- Create `tests/evaluation/test_adapter_numerics.py`; modify `tests/evaluation/test_numerics.py`, `test_off_kv_cache.py`, `test_kv_cache_probe.py`, and `tests/multinode/test_evaluation.py` for their owned contracts.
- `src/evaluation/protocol.py`, native Transformers, static-inference implementation and launchers need no source edit. Existing execution identity/CLI-choice plumbing is reused rather than requiring artificial hunks.

---

### Task 1: Integrate the conditional token-adapter profile and its falsification witness

**Files:** The ten source/test files in File Map, as one coupled arithmetic/identity/evidence feature. The helper alone is not a user-operable or adoptable correction. One implementer owns the integration; independent task and final reviews cover its frozen complete diff before the single finalCPU/commit.

**Interfaces:**
- `adapter_numerics.ADAPTER_TOKEN_FP64_POLICIES`: exact two-string tuple `('adapter_token_fp64', 'adapter_token_fp64_math_sdpa')`.
- `AdapterNumericsError(ValueError)`: explicit unsupported execution/role/nesting failure; main must re-raise it ahead of its legacy generic per-sample error handler.
- `validate_adapter_token_fp64(owner: torch.nn.Module, *, require_adapters: bool = True) -> int`: count eligible DoRA owners. For resolved Qwen traverse decoder `.model.layers`, not vision; for partial replay traverse that supplied module. Require standard eval/unmerged/enabled single-active DoRA, standard Linear A/B with FP32 Parameters and supported installed forward bindings. Reject custom/quantized/training/disabled/merged/multi-adapter cases; recognize only this scope's own exact wrappers during supported nesting. Strict entry validation requires>=1; a generic partial replay may explicitly allow0.
- `temporary_fp64_token_linear(linear: torch.nn.Linear, operand: torch.Tensor) -> tuple[torch.Tensor, dict]`: real3D FP32 token input, temporary float64 input/weight/optional bias and linear result, FP32 return. Actual evidence keys: `input`, `weight`, `bias`, `linear_output`, `returned`, holding dtype strings (`bias=None` when absent). No Parameter assignment/cast or side effect. Used by live scope and shadow replay, not duplicated arithmetic.
- `AdapterTokenEvidence.snapshot() -> dict`: plain JSON-safe interval evidence with `requested_dtype='torch.float64'`, `dora_modules`, `wrapped_linear_modules`, `token_calls`, `normalization_calls`, `actual_operand_dtypes` (the five keys above, or None if no protected call). Snapshot after calls reflects this context interval; a closed snapshot must not change when an outer scope continues. No Tensor/module/private handle is serialized.
- `adapter_token_fp64_scope(owner: torch.nn.Module, *, policy: str, require_adapters: bool = True) -> Iterator[AdapterTokenEvidence]` is the contextmanager interface. Same-policy nested scopes over the same/subset eligible modules reuse wrappers; conflicting policy/overlap/new unsupported owner fails before mutation. Actual protected calls require eval, no-grad/inference and autocast off; metadata-only preflight may enter without running any forward.
- Existing `validate_numerical_policy(policy: str, *, protocol: str, stage: str, disable_latent_generation: bool, static_inference_cache: bool = False) -> str` adds an optional keyword; reject static cache with the new IDs because it replaces DoRA forward, preserving all old combinations/meaning.
- `numerical_execution_scope(policy: str, model, *, require_adapters: bool = True) -> Iterator[AdapterTokenEvidence | None]` yields evidence for new IDs and None for old IDs, wrapping existing global controls plus the shared model scope. `observe_numerical_policy(policy: str, model=None, *, execution=None) -> dict` keeps old payloads unchanged; only new IDs add `adapter_token_arithmetic` from an actual interval snapshot or a clearly not-observed/zero-call request record.
- `MLPComponents(..., retain_adapter_operands: bool = False)` / corresponding `PassiveTrace` opt-in retain actual token A/B operands, module linkage, input capture hashes/layout and native outputs only when required by the new precheck/component witness. Legacy defaults still release them. Direct input comparisons are additional evidence, not a claim that v3 serialized those values.
- `fp64_adapter_precheck(model, processor, sample, *, baseline_report: dict) -> dict` returns `status` (pass/fail/inconclusive), `baseline_fidelity` ([bool,bool]), `input_difference`, `native_replays_match` ({A:[bool,bool],B:[bool,bool]}), `protected_A_difference`, `A64_B32_difference`, `A64_B64_difference`, and `temporary_operand_dtypes`. Use existing tensor-difference shape/equal/finite/max_abs fields. Pass requires both baseline fidelities, finite equal actual A inputs, four original replay matches and finite exact protected A plus A64/B64 outputs. A64/B32 is diagnostic only and may differ. Missing/changed baseline or mutated/unreproduced operands is inconclusive; a finite witnessed protected difference is fail; no live Tensors in JSON.
- `math_sdpa_eligible(report: dict) -> bool` derives the first numeric difference from ordered actual comparisons, rejects missing evidence, and requires matched Q/full K/V/visibility at that first SDPA output plus valid cache geometry. Do not trust a declared method name or first-difference label alone.

**Role algorithm and restoration decisions:** Validate all target owners before patching. At actual DoRA entry record the actual token operand and enforce the fixed supported native3D/FP32/no-grad/eval route. Observe A(identity), B(actual preceding A output), A(the actual DoRA token operand), B(actual preceding token A output), then the original DoRA exit. Preserve2D identity calls through their original forward; only the actual token roles call the shared FP64 primitive. Validate actual identity values/linkage/order, not dimensions alone. Unknown/incomplete/reentrant roles raise `AdapterNumericsError`. Use the existing MethodType/ExitStack restoration pattern; preserve whether each original forward was an instance override. Manage only the current single-process scope and supported nesting, without general ownership/recovery machinery. Actual norm/correction/add/cast and CUDA kernels remain unobserved unless directly witnessed.

- [ ] **Step1 — Write scoped-arithmetic RED.** In the new test file, use real installed tiny PEFT DoRA and real delegated PyTorch linear calls, not mocked results. Tests `test_token_fp64_uses_real_operands_and_preserves_norm_path`, `test_scope_restores_exact_forward_presence_on_success_and_exception`, `test_scope_nested_subset_reuses_bindings_and_interval_counts`, `test_scope_rejects_unknown_roles_bindings_and_unsupported_models`. Write them before code; retain the expected missing-interface/behavior failures. Hand-derived primitive assertion (construct actual Linear weight explicitly):

```python
out, actual = temporary_fp64_token_linear(linear_with_weight([[1.5, -2.0], [0.5, 1.0]]),
                                        torch.tensor([[[2.0, 3.0]]]))
assert torch.equal(out, torch.tensor([[[-3.0, 4.0]]]))
assert out.dtype == torch.float32
assert actual['input'] == actual['weight'] == actual['linear_output'] == 'torch.float64'
assert actual['returned'] == 'torch.float32'
```

`linear_with_weight` is a test fixture creating real `nn.Linear(2, 2, bias=False)` and copying the stated weight under no_grad; it is not a production helper. Independently observe actual F.linear operand/result dtypes by delegating the real operation. Compare all tiny Parameters/storage/dtypes to their pre-scope snapshots, observe original FP32 identity outputs, and verify exact original binding attributes afterward. Invalid-role/model and incompatible-nesting cases must raise before altered arithmetic; actual forward failure must restore everything.

For one real eval PEFT Linear+DoRA fixture (not the magnitude module alone), pin the role/count contract:

```python
with torch.inference_mode(), adapter_token_fp64_scope(projection, policy='adapter_token_fp64') as evidence:
    projection(x)
    observed = evidence.snapshot()
assert observed['token_calls'] == 2
assert observed['normalization_calls'] == 2
assert observed['actual_operand_dtypes']['input'] == 'torch.float64'
```

The nesting test performs one outer call, one inner call and another outer call; assert inner token_calls2 stays2 after close, outer token_calls6, and exact original binding state after exit.

- [ ] **Step2 — Implement the scoped primitive and role state.** Create only the stated helper file. Run Step1 tests GREEN with the repository CPU environment and unique checkpoint basetemp. Add no untested guards/statistics/behaviors; new behavior gets its own RED first. Observation is per real context interval, so nested previous controls cannot fake evidence of a later call.

- [ ] **Step3 — Write shared-control/evaluator/identity RED.** Extend real existing fixtures for both new IDs and genuine OFF/ON. Tests `test_new_adapter_policy_controls_restore_medium_and_math`, `test_evaluator_observes_actual_fp64_after_generation_and_no_historical_leak`, `test_new_policy_rejects_static_or_unsupported_model_before_outputs`, `test_role_error_is_fatal_before_success_summary`, and real four-shard roundtrip/mixed-ID/row-manifest cases. Keep requested/runtime-before-generation evidence distinct from real post-generation call evidence:

```python
assert runtime['numerical_observation']['adapter_token_arithmetic']['token_calls'] == 0
assert row['numerical_observation']['adapter_token_arithmetic']['token_calls'] > 0
assert row['numerical_observation']['adapter_token_arithmetic']['actual_operand_dtypes']['input'] == 'torch.float64'
assert row['numerical_policy'] == 'adapter_token_fp64'
```

Use actual tiny native/PEFT generation for dtype/cache evidence and a real unsupported role to exercise the fatal exception; stub only external loading/processor/Git seams. Test actual merger rejects new-ID mixing with existing/new policies and rejects new+static even when old static configurations remain valid. Historical absent/explicit labels remain compatible, and no new fields leak into old row/runtime payloads.

- [ ] **Step4 — Integrate real execution and metadata.** Extend the central IDs/global-control mapping: both new policies use highest/TF32-off/BF16-reduction-off; the math variant enables only MATH SDPA. Reuse exact incoming precision/SDPA restoration, never touch cuDNN convolution TF32/environment variables. Apply `numerical_execution_scope` after loading in actual generation, model preflight/runtime publication and every probe/replay path. Observe row/control arithmetic after real generation while the intended controls remain active. Reject new+static/model incompatibility globally before sample outputs; preserve historical main error boundaries and strict historical runtime dataclass. Rethrow only the new explicit role exception as fatal. Pass static state into central merger validation; do not add execution counters to generation-config identity.

- [ ] **Step5 — Write precheck/probe/eligibility RED.** Tests `test_precheck_replays_actual_original_A_B_before_fp64`, `test_precheck_requires_equal_actual_A_inputs_and_finite_reproduced_baseline`, `test_precheck_failure_skips_full_candidate_without_faking_alignment`, `test_candidate_requires_full32_selected_A_B_Q_and_logits`, `test_math_gate_requires_first_sdpa_output_and_equal_operands`, and legacy CLI/report fidelity. Use real original PEFT operand references and literal damaged evidence for mutation/missing/NaN cases; never synthesize a successful GPU outcome. Check direct actual token A input equality is reported, and distinguish protected A with ordinary B from protected A/B. Cases with earlier projection, unequal K/V/mask, missing rows or nonfinite values must reject MATH. Keep natural EOS for0/3; short source12 cannot pass.

Pin the new precheck/gate result semantics with independently generated real-tensor fixture reports:

```python
assert passing_precheck['status'] == 'pass'
assert passing_precheck['native_replays_match'] == {'A': [True, True], 'B': [True, True]}
assert passing_precheck['protected_A_difference']['equal'] is True
assert passing_precheck['A64_B64_difference']['equal'] is True
assert damaged_operand_precheck['status'] == 'inconclusive'
assert unequal_protected_precheck['status'] == 'fail'
assert math_sdpa_eligible(sdpa_output_only_report) is True
assert math_sdpa_eligible(earlier_mlp_difference_report) is False
assert math_sdpa_eligible(unequal_kv_or_visibility_report) is False
```

These names are test fixtures built from actual tiny execution/literal tensor events, never stubbed returned verdicts. A precheck rejection must leave candidate controls/results empty and alignment unexecuted/inconclusive.

- [ ] **Step6 — Implement the cheap local stage and full witness.** For source12/new IDs require explicit component observation. Run the precheck with `full_accumulation` global controls and `torch.inference_mode()` while no FP64 model scope is installed, before entering the candidate execution scope; finish/release that baseline scope before full candidate generation, using the already loaded unchanged model. Read the existing full_accumulation source12 baseline at `results/kv-mlp-isolation-20261024T081537Z/probe-r1/report.json`; check model/base/source/profile identity. Run an explicitly reported2-token baseline profile (max_new_tokens2, all other frozen settings unchanged), unobserved controls before observed arms, with PassiveTrace(max_steps2). Require both2-token prefixes/prompt/latent semantics and original selected A/B observations to reproduce the saved original32-token witness; this is not a32-token success claim or EOS forcing. After hooks close, verify actual A/B input capture hashes/layout, direct finite newest A input equality, and both original FP32 A/B replay matches before shared FP64 shadow arithmetic. Preserve original full rows and report A-FP64/B-FP32 versus A-FP64/B-FP64. A failed/incomplete precheck writes its result, leaves full alignment unexecuted/inconclusive, and stops that candidate without new retry orchestration. A passing precheck permits the existing32-token fresh control/observed candidate pair under the selected shared execution scope. Release local operand references; save summaries only. New profiles source0/3 use the same32 cap/natural EOS, after root's passing-primary gate. Preserve existing old profile CLI/report behavior.

- [ ] **Step7 — Focused GREEN, self-review and independent reviews.** Run these five files as one owned CPU set: `tests/evaluation/test_adapter_numerics.py tests/evaluation/test_numerics.py tests/evaluation/test_off_kv_cache.py tests/evaluation/test_kv_cache_probe.py tests/multinode/test_evaluation.py`, with `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src` and a unique canonical checkpoint basetemp. Retain complete RED/GREEN outputs and every warning/failure, frozen diff/all changed hashes and self-review. Freeze for root's independent spec+quality task review and whole-branch final review; route any fixes through the selected implementer and required scoped re-review. No source commit, broad suite or GPU before root authorizes its gate.

- [ ] **Step8 — One fresh final CPU and identical-content commit.** Determine branch-finishing applicability/claim commands before selecting verification: immediate next gate is committed identity and GPU correction evidence, so finishing is not immediate yet. After reviews/fixes and root authorization, mkdir canonical basetemp parent and run once:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q -m 'not gpu and not distributed' --basetemp=checkpoints/kv-adapter-precision-20261024T091107Z/final-cpu
```

Retain/read complete output/start/end/exit. Baseline5163pass is not changed-code verification. After green, commit only the ten authorized source/test files; root independently checks reviewed/tested/current/committed hashes match. Do not repeat the identical broad suite.

- [ ] **Step9 — Root bounded falsification and confirmation.** Use existing probe, explicit source/profile, fresh controls, `--mlp-components` for primary and distinct output IDs. New-profile primary source12 command is `./venv/ilvr-venv/bin/python scripts/probe_stage1_off_kv.py --fresh-controls --source-index 12 --numerical-policy adapter_token_fp64 --mlp-components --output-dir results/kv-adapter-precision-20261024T091107Z/trials/source12-adapter_token_fp64`. Root verifies actual controls/dtypes/cache/first nail before interpreting full32 results. Only after an actual eligible first remaining attention difference may root run the math variant with a separate ID. Stop at the first fully passing candidate; confirm0/3 under that exact policy (no component flag, max32/natural EOS). All process wall time, including model loading/prechecks/failed candidates/confirmation/report, counts toward new200 and original300 limits. Root owns minimal preflight, process deadline/log/exit and one output validation; no new generic coordinator or automatic retry. Missing coverage/budget means no fix/adoption claim.

- [ ] **Step10 — Exact-claim verification and handoff.** Persist approvals, actual agents/reviews, content/commit identity, full test and empirical evidence, unwaived procedure history and every ruling before deleting this plan's scratch only. If all required correction witnesses are green, invoke verification-before-completion and finishing-a-development-branch, announce finishing, reuse the same unchanged-content fullCPU witness for the overlapping test-pass claim and present human-owned integration choices. Otherwise report the precise failed/incomplete correction, preserve branch/results and return to diagnosis; never call code availability or local replay a fix, widen arithmetic or reset the budget.

## Self-Review and Handoff

Coverage: exact v4 commitment maps to Task1 Steps1–10: role-preserving FP64/Parameters/base/norm (1–2), shared nested execution/runtime/one-four-shard identity (3–4), cheaper original-operand positive control/direct input/dtype evidence/full finite acceptance/conditional MATH (5–6,9), selected method/review/freshCPU/budget/finish (7–10). No spec gap is knowingly deferred. Five Review Focus classes each have explicit owned tests. API names/IDs/metadata fields agree across interfaces and consumer steps; helper arithmetic is shared with shadow replay. No artificial protocol edit, kernel, dependency, training, coefficient or default change is planned.

This is a decision plan, not an implementation transcript: exact interfaces, two profile IDs, role order, state/observation boundaries and causal assertions are fixed; implementation bodies remain with the selected fresh implementer. One coupled task prevents committing a helper/profile without its actual inference/evidence integration before the required final suite. Root may author this control artifact; requested source/tests remain agent-owned.

**Human handoff:** v4 design and separate new GPU<=200s experiment are already approved in `approval-v4.json`. This newly written implementation plan still requires human review under writing-plans; do not infer that review from the earlier design approval. Preserve the user's existing subagent-driven method after confirmation.

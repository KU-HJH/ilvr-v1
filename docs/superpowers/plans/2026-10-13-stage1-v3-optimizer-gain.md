# Stage1-v3 Optimizer Gain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user already selected native root execution with independent final review. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit Stage1-v3 mode that submits Gain through native AdamW and constrains its current-step incremental FP32 displacement against raw RV.

**Architecture:** Retain ordinary model/VJP execution and legacy post-Adam Gain. Add a pure native-AdamW counterfactual/projection module, a pinned ZeRO-1 boundary adapter, and mode-aware controller dispatch. The real joint update owns moment state; the temporary base candidate supplies only the current-step displacement reference.

**Tech Stack:** Repository Python 3.11 venv, PyTorch 2.7.1 native AdamW, DeepSpeed 0.18.7 ZeRO-1, existing Transformers/Accelerate Trainer hooks, pytest CPU oracles.

**Spec:** [Approved commitment v1](../specs/2026-10-13-stage1-v3-optimizer-gain-commitment.md), incorporating [frozen proposal v1](../specs/2026-10-13-stage1-v3-optimizer-gain-design.md). User `승인` binds commit `a4d5fd5feced91e216d0ac0703f9323b43e48de0`; commitment SHA256 `ffa511a90a9eda7a2f6faf96a89f68cbf9e953b812d2ec30a001c5bb78ebce24`. The commitment's zero-LR exclusion overrides the proposal's unrestricted projection support.

## Global Constraints

- Work in `/root/.codex/worktrees/541e/ILVR`, branch `stage1-v3`; preserve existing untracked references, models and bytecode. Product baseline is `154561dd84b1ecae4933c6e849d913eb87a07c3d`.
- No training, model GPU execution, or remote launch is authorized. CPU unit tests may use tiny tensors/native optimizer oracles and controlled ZeRO storage; they do not establish real distributed/GPU equivalence.
- Use `venv/ilvr-venv/bin/python`; set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=transformers/src:src:.`. Save execution logs under `logs/stage1-v3-optimizer-gain-20261013/`; test artifacts containing parameters/optimizer/RNG state stay under the existing `checkpoints/` symlink.
- New mode is `gain_update_mode=adamw_projected`; legacy/default is `post_adam`. `gain_optimizer_weight` is an explicitly supplied finite nonnegative gradient coefficient for a fresh new-mode run, never inferred from `gain_alpha=0.001`.
- Preserve actual global-block averaging of Gain/h and the ordinary loss's configured accumulation/remainder denominator. CE remains 0.1 and all other standing profile values remain unchanged.
- Production adapter supports DeepSpeed 0.18.7 ZeRO-1 without offload, BF16 model/FP32 masters, loss scale 1, and torch 2.7.1 `AdamW(fused=True, foreach=None, amsgrad=False, maximize=False, capturable=False, differentiable=False)`.
- Preserve per-group betas/epsilon/LR/weight decay. Capture finite nonnegative group LR before scheduler advancement; zero-LR groups receive no correction.
- Real moments/counters follow the clipped joint gradient once. Protection concerns only the current incremental FP32-master displacement; historical Gain momentum, BF16-visible alignment and nonlinear score monotonicity are outside the approved guarantee.
- Projection uses global FP64 statistics, `min_h_norm_sq=1e-30`, direct native joint values when unconstrained, and whole-increment fallback to the base candidate when a conflicting projected candidate has a negative rounded FP64 RV dot.
- Native/group and final candidate guards check FP32/moments and BF16 casts. Failure is collective and checkpoint-ineligible; no native-state rollback, retry framework, dependency change, or third-party source patch.
- Preserve existing saved metadata bytes and legacy formula semantics. Resume inherits omitted settings and rejects explicit mode/weight changes. New formula requires a fresh run.
- Follow native TDD, independent spec-and-quality review and fresh verification. Do not reuse prior feature tests as proof of this implementation.

## Review Focus

1. A parameter split across rank partitions plus a padded final shard must contribute exactly once; cover in Task 3's `test_owned_fragments_cross_parameter_and_padding`.
2. Mixed zero/positive group LR, including a scheduler changing LR inside engine.step, must preserve the step's zero-LR coordinates; cover in Tasks 2/4.
3. A first step with empty Adam state and a resumed nonzero state must produce the same base candidate as native AdamW without mutating live state; cover in Tasks 2/4.
4. One rank discovering an invalid native moment/BF16 cast must fail before that group's low-precision copy, and remove scoped hooks; cover in Tasks 3/4.
5. An explicitly supplied optimizer weight of zero must not become a missing value or a legacy alpha, and stale environment values must not reinterpret resume; cover in Tasks 1/5.

## File responsibilities and test commands

`v3_config.py`/`v3_resume.py` own validated identity. New `v3_adamw_update.py` owns tensor/state math without engine mutation; new `v3_zero1_update.py` owns ZeRO mappings and native-step guards. `v3_update.py` and the existing Trainer adapter own lifecycle dispatch. Existing parser, coordinator and launcher own argument transport. Document the user-visible contract in `docs/stage1-v3.md` only after implementation.

Before the first test, create a distinct parent under `checkpoints/stage1-v3-optimizer-gain-cpu-<UTC run id>/` and verify its resolved path is within the canonical checkpoints target. Each RED/GREEN invocation gets a new `--basetemp` child. Define a session-local `v3check` shell function that runs the exact repository Python/pytest/environment above, saves complete stdout/stderr and exit status under this task's log directory, and uses that checkpoint-contained basetemp. Do not write a reusable runner or launch a training script. Commands below use `v3check <test paths...>` for this exact invocation.

After validating and creating `$v3_test_root` and setting `$v3_test_log` to this task's log directory, use this session-only wrapper. The root includes the unique UTC run ID; preserve its value in the workflow record so no resumed session reuses a previous basetemp/log prefix.

```bash
v3_test_index=0
v3check() {
    v3_test_index=$((v3_test_index + 1))
    v3_test_label="$(basename "$v3_test_root")-$v3_test_index"
    if OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=transformers/src:src:. \
        venv/ilvr-venv/bin/python -m pytest -q -m 'not gpu and not distributed' \
        --basetemp "$v3_test_root/$v3_test_index" "$@" >"$v3_test_log/$v3_test_label.log" 2>&1; then
        v3_test_status=0
    else
        v3_test_status=$?
    fi
    cat "$v3_test_log/$v3_test_label.log"
    printf '%s\n' "$v3_test_status" >"$v3_test_log/$v3_test_label.exit"
    return "$v3_test_status"
}
```

---

### Task 1: Config and saved formula identity

**Files:** Modify `src/stage1/v3_config.py`, `src/stage1/v3_resume.py`, `tests/stage1/test_v3_metadata.py`, `tests/stage1/test_v3_resume.py`; create `tests/stage1/test_v3_optimizer_config.py`.

**Interfaces:** `Stage1V3Config` gains `gain_update_mode: str='post_adam'` and `gain_optimizer_weight: float|None=None`. Existing `to_dict()`, `build_v3_checkpoint_contract(...)`, `validate_v3_resume(...)` and `read_v3_contract(...)` keep their signatures. Add `_resolve_gain_update(saved_config: Stage1V3Config|None, mode: str|None, weight: float|None) -> tuple[str,float|None]` for explicit values; environment/CLI transport belongs to Task 5.

- [ ] **Step 1 — Write failing config/identity tests.** Parameterize both weighting modes. Pin these assertions and rejection cases:

```python
assert Stage1V3Config().gain_update_mode == 'post_adam'
assert Stage1V3Config().gain_optimizer_weight is None
assert Stage1V3Config(gain_update_mode='adamw_projected', gain_optimizer_weight=0.).gain_optimizer_weight == 0.
assert _resolve_gain_update(saved_new_config, None, None) == ('adamw_projected', saved_new_config.gain_optimizer_weight)
assert checkpoint_file.read_bytes() == original_bytes  # after old-contract read/validation
```

Reject a new-mode missing coefficient, a legacy-mode supplied coefficient, bool/NaN/Inf/negative weights, unknown modes and explicit saved mode/weight changes. Historical fixtures omit both new fields; only those omissions normalize to `post_adam`/None. Preserve exact old formula IDs and every existing default coefficient.
- [ ] **Step 2 — RED:** `v3check tests/stage1/test_v3_optimizer_config.py`. Expect failures on missing fields/API or absent rejection, not fixture/import setup.
- [ ] **Step 3 — Implement validation and identity.** Add the two fields, validate optional numeric handling without weakening existing checks, and normalize only legacy omissions in `_contract_identity`. For new mode set `formula_ids['gain']` to `global_block_mean_same_state_adamw_joint_moments_active_lr_project_fp64_fp32_master_v1`, or prefix `F_T1_detached_H_logV_token_gap_` for token entropy. Add new-mode-only `formula_ids['gain_alpha_role']='inactive_legacy_displacement_v1'`. Keep the legacy dictionary of formula IDs unchanged. Use `is None`, not truthiness, for weights. Do not rewrite a loaded saved contract.
- [ ] **Step 4 — GREEN:** `v3check tests/stage1/test_v3_optimizer_config.py tests/stage1/test_v3_metadata.py tests/stage1/test_v3_resume.py`. Expect all selected tests passing; update exact config dictionary expectations only for the approved fields.
- [ ] **Step 5 — Commit** these five files as `feat(stage1-v3): define optimizer gain identity` after reading RED/GREEN evidence.

### Task 2: Native AdamW counterfactual and displacement projection

**Files:** Create `src/stage1/v3_adamw_update.py` and `tests/stage1/test_v3_adamw_update.py`.

**Interfaces:** Export `clip_base_gradient(gradient: Tensor, *, global_norm: Tensor, max_norm: float) -> Tensor`; `adamw_base_candidate(master: Tensor, gradient: Tensor, state: Mapping[str,Tensor], group: Mapping[str,object]) -> Tensor`; and `project_adamw_increment(base: tuple[Tensor,...], joint: tuple[Tensor,...], protection: tuple[Tensor,...], *, valid_masks: tuple[Tensor,...], lrs: tuple[float,...], sum_fp64: Callable[[Tensor],Tensor], min_h_norm_sq: float=1e-30) -> ProjectionResult`. `ProjectionResult` contains `parameters`, `dot_before`, `dot_after`, `gain_norm_before`, `gain_norm_after`, `projected`, `tiny_h`, and `rounding_fallback`; scalar metrics are Python finite floats/bools. Inputs are owned FP32 master shards; masks are same-shaped bool tensors.

- [ ] **Step 1 — Write failing math/oracle tests.** Use independent `torch.optim.AdamW` instances as the oracle, with `lr=.03`, betas `(.9,.999)`, eps `1e-8`, decay `.01` and `0`, fresh state and three warmed-up steps. CPU fixtures set `fused=False`; no GPU equivalence claim.

```python
torch.testing.assert_close(candidate, native_base_parameter, rtol=0, atol=0)
assert torch.equal(live_master, master_before)
assert all(torch.equal(live_state[k], state_before[k]) for k in live_state)
assert torch.equal(result.parameters[0], joint[0])  # LR=0 group's candidate is unchanged
assert result.dot_after >= 0
assert torch.equal(unconstrained.parameters[0], joint[0])
```

Name the mixed-group regression `test_projection_does_not_move_zero_lr_group`: base `(0,0)`, joint `(0,-1)`, h `(1,1)`, separate groups with LRs `(0,.03)`; result must be `(0,0)`. Reject a nonzero joint-minus-base increment in a zero-LR valid coordinate. Cover h=0, norm below `1e-30`, multiple groups, ignored padding, global reduction and a deterministic negative-dot-after-FP32-rounding fixture. Also pin `test_fp32_protection_does_not_imply_bf16_protection` using base `(2,1)`, increment `(1/256,-1/256)`, h `(1,1)`; FP32 dot equals zero and BF16-visible dot is negative.
Pin `test_base_clipping_matches_zero1_divisor`: gradient `(3,4)`, global norm `5`, max norm `1` returns the FP32 gradient divided by `5+1e-6`; max norm `0` returns an unchanged independent clone.
- [ ] **Step 2 — RED:** `v3check tests/stage1/test_v3_adamw_update.py`. Expect missing module/functions or wrong mathematical results.
- [ ] **Step 3 — Implement native functional calculation and exact projection.** Use the installed `torch.optim.adamw.adamw` functional operation on cloned masters/moments/step tensors, copying the actual group flags. For lazy state, reproduce installed Adam initialization (fused step tensor device/dtype included) without inserting into live optimizer.state. Base clipping must use DeepSpeed's `max(1,(global_norm+1e-6)/max_norm)` divisor when max_norm>0; otherwise clone unchanged. Use FP64 local dot/norm statistics and supplied global SUM; active mask is valid AND group LR>0. On inactive valid coordinates require identical base/joint values. Preserve native values directly when unconstrained and padding always; after a conflicting rounded candidate, a negative recomputed global dot selects the base candidate on all valid coordinates. Validate shapes/dtypes/devices/finiteness and nonnegative finite LR.
- [ ] **Step 4 — GREEN:** `v3check tests/stage1/test_v3_adamw_update.py`. Expect CPU oracle, zero-LR and precision-limit tests passing. Do not introduce a tolerance to make the mixed-LR or rounded-dot invariant pass.
- [ ] **Step 5 — Commit** the two files as `feat(stage1-v3): compute projected AdamW gain displacement`.

### Task 3: Owned ZeRO-1 fragments and native-step guards

**Files:** Create `src/stage1/v3_zero1_update.py`, `tests/stage1/test_v3_zero1_update.py`, and `tests/stage1/v3_zero1_helpers.py`.

**Interfaces:** `Zero1AdamWGainAdapter(engine: object, parameters: tuple[nn.Parameter,...], config: Stage1V3Config)` exposes `prepare(gain: tuple[Tensor,...], protection: tuple[Tensor,...], *, block_count: int) -> None`, `finish(*, successful: bool) -> dict[str,float|int|bool|str|None]`, and `abort() -> None`. Internal `OwnedFragment` identifies parameter index, group index, LP start, HP start, size and gradient-list index; `OwnedGroup` holds the FP32 master, valid mask and fragments. `_require_supported_gpu_optimizer(optimizer: torch.optim.Optimizer) -> None` isolates the CUDA/fused profile check for explicitly simulated CPU fixtures; common optimizer identity/options, ownership and ZeRO validation still run. It consumes Task 2 functions and the raw globally averaged tensors from Task 4.

- [ ] **Step 1 — Write failing fragment and lifecycle tests.** Build a controlled four-rank layout with a parameter crossing a partition boundary, two groups, and final padding. Use actual DeepSpeed fragment address/data structures where practical; only storage/collectives/engine dispatch are simulated. Assert:

```python
assert mapped_global_gain == expected_gain_once
assert joint_gradient.dtype == torch.float32
torch.testing.assert_close(joint_gradient[valid], ordinary[valid] - .25 * gain[valid], rtol=0, atol=0)
assert joint_gradient[padding].count_nonzero() == 0
assert zero.averaged_gradients is original_gradient_dictionary
assert counters['safe_get_full_fp32_param'] == 0
assert counters['safe_get_full_grad'] == 0
assert events.index('native_guard') < events.index('group_bf16_copy')
```

Include `test_owned_fragments_cross_parameter_and_padding` and `test_remote_native_cast_failure_prevents_group_copy`. All four simulated ranks must use the same ordered scalar collectives even with no local Gain blocks. Reject unsupported optimizer flags/version/offload/loss scale before mutation. Check finite FP32 values that cast to BF16 Inf, NaN moments, negative second moments, missing gradient fragments and overlap/gap mappings. CPU harness explicitly substitutes only the production GPU-profile guard; record that this is not a real ZeRO acceptance test.
- [ ] **Step 2 — RED:** `v3check tests/stage1/test_v3_zero1_update.py`. Expect missing adapter or invariant failures.
- [ ] **Step 3 — Implement the pinned bridge.** Map `_hp_mapping.lp_fragment_address`, `.hp_fragment_address`, `.param_group_index` and `parameter._index_in_param_group`; absent local mappings are normal for non-owned parameters. Validate unique complete ownership and zero padding. Bind the existing common data-parallel process group; reject incompatible group/partition metadata before mutation. Agree validation errors collectively before entering later norm/optimizer collectives; local exceptions must not leave peers in a different phase. Snapshot LR and base norm before injection; construct base candidates one group at a time with Task 2 and release copied moments. Replace owned averaged-gradient list entries with shape-preserving FP32 joint values while keeping the dictionary object referenced by mappings. Never reduce full gradients again. Validate the native joint norm after injection.

Register scoped native optimizer pre/post hooks: the pre-hook records only the current group's master snapshot for displacement diagnostics; the post-hook checks masters/moments/casts collectively before DeepSpeed can copy that group to BF16, records the joint norm contribution and releases its snapshot. Preserve the native optimizer's per-group step dispatch; require each group once on success and none on a genuine skip. At finish use Task 2 across all unique shards, validate every corrected candidate before writing any correction, then call existing `update_lp_params()`. Abort removes both hooks and temporaries without rollback. Zero weight/zero global blocks skip counterfactual/projection but retain finite native guards; use explicit bypass metrics rather than fabricated norm values.

Boundary metrics use `mode`, `optimizer_weight`, `applied`, `bypass_reason`, `base_gradient_norm`, `joint_gradient_norm`, `base_displacement_norm`, `joint_displacement_norm`, `gain_displacement_norm_before`, `gain_displacement_norm_after`, `rv_dot_before`, `rv_dot_after`, `projection_count`, `rounding_fallback_count`, `tiny_h_count`, and `lr_group_<index>`. Norms/dots unavailable on bypass or skip are None; emit numeric values only when measured. Event counts are per-boundary 0/1 values, sufficient to compute observed rates without new checkpoint state. Test the measured metrics against the same native oracle and check that the logger omits unavailable scalar values while preserving explicit bypass/applied status.
- [ ] **Step 4 — GREEN:** `v3check tests/stage1/test_v3_zero1_update.py tests/stage1/test_v3_adamw_update.py`. Expect all cases green; inspect the actual collective/copy event trace and ensure the fixture calls native CPU AdamW.
- [ ] **Step 5 — Commit** the bridge and test fixtures as `feat(stage1-v3): bridge optimizer gain into ZeRO-1`.

### Task 4: Controller and actual Trainer boundary integration

**Files:** Modify `src/stage1/v3_update.py`, `src/stage1/trainer.py`, `tests/stage1/test_v3_update.py`; create `tests/stage1/test_v3_optimizer_integration.py`.

**Interfaces:** Preserve `V3GainController.prepare(process_group=None)` and `finish(engine, *, successful)` signatures. Add `raw_gain: tuple[Tensor,...]`, `raw_rv: tuple[Tensor,...]`, `last_metrics: dict`, `prepare_optimizer_step(engine: object) -> None`, and `abort_optimizer_step() -> None`. Add `CustomTrainerStage1._log_v3_optimizer_metrics(metrics: dict) -> None`. Task 3 adapter exists only for the new mode. Legacy `direction` arithmetic and successful-step behavior remain intact.

- [ ] **Step 1 — Write failing integration/state tests.** Use the actual `_DeepSpeedStepAdapter` with Task 3's controlled engine/storage and real CPU AdamW; do not prove this using Trainer.optimizer_step alone. Assert:

```python
assert torch.equal(actual_state['exp_avg'], native_joint_state['exp_avg'])
assert torch.equal(actual_state['exp_avg_sq'], native_joint_state['exp_avg_sq'])
assert actual_state['step'].item() == previous_step + 1
torch.testing.assert_close(actual_master, expected_base_plus_projected_increment, rtol=0, atol=0)
assert events == ['gain_prepare', 'joint_injection', 'joint_norm_check', 'native_step', 'native_guard', 'group_bf16_copy', 'projection_sync', 'finish']
assert controller.state == 'IDLE' and adapter_hooks_remaining == 0
```

Parameterize none/token_entropy, success/skip/failure, zero global blocks, zero weight, mixed group LR and scheduler advancement. Add `test_rejected_gain_retains_joint_momentum_on_next_zero_gain_step`: b=0, g=-1, h=1 on first step, then g=0; first parameter increment is suppressed, moments equal the joint oracle, later motion can be negative. This is the literal first-nail witness of the approved limitation. Add save/load versus uninterrupted two-step CPU state equality under checkpoint-contained tmp_path. Require no real moment/parameter change on skipped step and failed-state checkpoint rejection on exceptions. Extend the existing tiny-Qwen parent training-step call witness for new mode to confirm unchanged F/gain_clean/packed_replay calls and reductions before any optimizer binding.
Add `test_base_and_joint_clip_with_their_own_norms` using b=(3,4), g=(4,0), coefficient 1 and max norm 1: compare base clipping at norm 5 and joint clipping at norm sqrt(17) to separate native AdamW oracles, then compare live moments to the joint oracle and final parameters to the projected candidate. Cover all-group LR=0 with unchanged parameters and normally advanced moments/counters.
- [ ] **Step 2 — RED:** `v3check tests/stage1/test_v3_optimizer_integration.py tests/stage1/test_v3_update.py`. Expect missing dispatch/state behavior; preserve historical failures separately.
- [ ] **Step 3 — Integrate at the actual engine boundary.** After `gain.prepare()` call `gain.prepare_optimizer_step(engine)` before `_validate_deepspeed_gradient_norm(engine)` and `original_step`. In new mode prepare keeps unprojected global Gain and raw RV; do not subtract RV twice in weighted mode. Successful finish delegates to Task 3 before EMA/counters; false success cleans up without projection/state advancement. Exception cleanup calls `abort_optimizer_step()` and sets FAILED. Legacy dispatch remains byte-for-byte arithmetically equivalent and does not instantiate the new adapter. The metrics helper emits `stage1_v3/`-prefixed values through `super().log`, with explicit applied/skipped status, successful optimizer-update counter, and one scalar LR per group. Call it after completed new-mode boundaries, including skip cleanup. Bypass the existing override's ordinary-diagnostics consumption and leave ordinary loss reporting unchanged.
- [ ] **Step 4 — GREEN:** `v3check tests/stage1/test_v3_optimizer_integration.py tests/stage1/test_v3_update.py tests/stage1/test_v3_gain_device.py tests/stage1/test_v3_gradients.py tests/stage1/test_v3_resume.py`. Expect native-moment/parameter-oracle equality, cleanup and forward-count witnesses passing. Keep the CPU simulation limitation in logs.
- [ ] **Step 5 — Commit** the four files as `feat(stage1-v3): integrate gain with native optimizer boundaries`.

### Task 5: CLI, node-command parity and user documentation

**Files:** Modify `src/utils_deepseed.py`, `src/stage1/v3_resume.py`, `src/stage1/composition.py`, `src/multinode_runtime.py`, `run_stage1_training.sh`, `tests/stage1/test_v3_profile.py`, `tests/multinode/test_v3_node_paths.py`, `docs/stage1-v3.md`; create `tests/stage1/test_v3_optimizer_cli.py`.

**Interfaces:** Add Python `--stage1_v3_gain_update_mode` and `--stage1_v3_gain_optimizer_weight`, launcher/coordinator hyphenated equivalents, and fresh-run environment aliases `STAGE1_V3_GAIN_UPDATE_MODE`/`STAGE1_V3_GAIN_OPTIMIZER_WEIGHT`. Preserve existing parser/launcher function signatures; Task 1 resolver owns mode/weight decisions.

- [ ] **Step 1 — Write failing parsing/transport tests.** CLI wins over fresh environment; on resume omitted CLI values inherit saved values and ignore stale fresh-run environment defaults. Fresh fallback is post_adam/None. Assert:

```python
assert parsed.stage1_v3_gain_update_mode == 'adamw_projected'
assert parsed.stage1_v3_gain_optimizer_weight == 0.
assert saved_file.read_bytes() == before_bytes
assert value(remote_worker, '--stage1-v3-gain-update-mode') == value(local_worker, '--stage1-v3-gain-update-mode') == 'adamw_projected'
assert value(remote_worker, '--stage1-v3-gain-optimizer-weight') == value(local_worker, '--stage1-v3-gain-optimizer-weight') == '0.25'
assert '--stage1-v3-gain-optimizer-weight' not in legacy_worker
```

Reject explicit new-mode options outside Stage1-v3, new mode without a coefficient, and resume changes. Exercise split/equals flag forms, shell dry-run, native/remote paths and both selector/weighting combinations without SSH or worker launch. Ensure composition passes exact new config into the controller and existing commit/environment preflight contracts are preserved.
- [ ] **Step 2 — RED:** `v3check tests/stage1/test_v3_optimizer_cli.py tests/multinode/test_v3_node_paths.py`. Expect rejected unknown options/missing transport rather than dependency failures.
- [ ] **Step 3 — Wire existing transport and document the contract.** Extend parser, `LAUNCHER_ALIASES`, coordinator allowlist, launcher profile trigger/export, and composition. Resolve once and send explicit effective mode/weight to both workers; never serialize None as a numeric argument. No new orchestration or separate preflight system. Document separate legacy alpha/new gradient weight, saved resume identity, local moment/BF16 limits, no extra model calls by design, and example **dry-run only** arguments with an explicitly illustrative coefficient. Do not recommend an untested training weight. `src/main.py` and the multinode wrapper need no speculative edits because actual option ownership is in the listed modules.
- [ ] **Step 4 — GREEN:** `v3check tests/stage1/test_v3_optimizer_cli.py tests/stage1/test_v3_profile.py tests/stage1/test_v3_token_entropy.py tests/multinode/test_v3_node_paths.py tests/multinode/test_v3_roots.py`; then `bash -n run_stage1_training.sh`. Expect passing tests and shell exit 0; no launched ranks or GPU jobs.
- [ ] **Step 5 — Commit** these nine files as `feat(stage1-v3): expose optimizer gain mode and resume contract`.

### Task 6: Independent review and final CPU verification

**Files:** Review the combined product diff from `a4d5fd5` and exact approved spec/plan; maintain evidence in `logs/stage1-v3-optimizer-gain-20261013/`. Requested-deliverable fixes remain native root work under the active execution skill.

**Interfaces:** Review package supplies baseline/HEAD/diff hashes, task RED/GREEN outputs, approved commitment identity, source call path and explicit CPU/GPU evidence limits. Independent reviewer returns specification-and-quality findings against that exact revision, not a general review of all historical Stage1 code.

- [ ] **Step 1 — Assemble the whole-change review package.** Confirm every requirement maps to Tasks 1-5. Include actual moment equality, parameter displacement, raw RV denominator, zero-LR support, checkpoint compatibility, exception/hook cleanup, BF16 finite boundary, and unchanged model call witnesses. Read the applicable requesting-code-review/executing-plans requirements; use the required independent reviewer/model, not a prior design adviser as a substitute for implementation review.
- [ ] **Step 2 — Obtain independent review and resolve findings.** Root performs the allowed native fix pass with a causal failing test before each behavioral fix. Preserve unresolved findings and do not claim a clean approval if any remain. Material scope changes return to design approval; ordinary implementation fixes do not request repeat approval.
- [ ] **Step 3 — Determine finish applicability, then run fresh verification once against the reviewed revision.** This task is local implementation/CPU checking; no merge/PR/worktree deletion is requested, so do not invent a branch handoff. If finishing becomes applicable, read both finish and verification skills first and reuse the one project suite for the overlapping claim. Use the established project command, with a fresh existing checkpoint-contained basetemp parent:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=transformers/src:src:. venv/ilvr-venv/bin/python -m pytest -q -m 'not gpu and not distributed' --basetemp "$v3_final_test_root/pytest"
```

Read complete output and exit status. Passing means zero failures/errors; skipped/excluded GPU/distributed cases remain unverified. Rerun only if an actual failure or later fix invalidates the witness. No second identical full suite merely for another gate.
- [ ] **Step 4 — Run distinct final witnesses.** `git diff --check`, shell syntax if changed since its prior check, exact reviewed-file hashes/diff identity, and absence of tracked unrelated modifications. Save command outputs, review verdict, fixes, commit identity and limits in the durable workflow/verification record. Do not claim fused GPU/ZeRO runtime, OOM or learning-quality results from CPU tests.
- [ ] **Step 5 — Report the implemented behavior and CPU results.** Preserve the worktree and existing artifacts. State that real four-rank/GPU checks and training were not run under this scope. Run the required ntfy summary hook before final delivery; do not send external notifications directly.

## Plan handoff

The written spec is approved. This plan is prepared for the required human plan review; execution method is already native and must not be asked again. No implementation, optimizer probe, GPU work or training starts during plan preparation. After plan approval, invoke `superpowers:executing-plans`, reconcile this exact artifact identity, and execute the tasks with their required TDD/review/verification evidence.

# Stage1-v5 Cross-helper Reverse-top-k V-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional same-sample reverse-query-ranked visual-latent replacement with independent and cumulative rejected history while preserving existing attention-masked V-off.

**Architecture:** Reuse detached teacher helper/original-image features and each existing clean teacher query. Replay recomputes from changed latent embeddings, using attached clean prefix KV only before the change; cumulative uses one causal rejected suffix, while independent packs per-block rows. Preserve independent clean Text-off and all existing loss, optimizer and normalization behavior.

**Tech Stack:** Existing Python 3.11 repository venv, PyTorch 2.7.1, patched Qwen2.5-VL Transformers, PEFT DoRA, native Trainer/DeepSpeed ZeRO-1, pytest and current shell/coordinator scripts. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-24-stage1-v5-cross-helper-reverse-topk-design.md`, commitment v1, approved by user `진행해` after the explicit final-approval question. Approved SHA256: `a188c6d03536b4389c5b27ccbaf4031586a8fc511f753cacfe344f77e2c8024e`. Approval and remaining-evidence acceptance: `logs/stage1-v5-cross-helper-reverse-topk-design-20261024-050727/design-approval-v1.json`.

## Global Constraints

- Use only `./venv/ilvr-venv/bin/python`; run tests with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:transformers/src`.
- Existing workspace `/root/.codex/worktrees/f187/ILVR` is already an isolated externally managed detached worktree. Do not create/delete a worktree or switch unrelated branches; preserve unrelated untracked files and all symlinks. Reconcile baseline at execution time.
- Default mode `attention_mask`; new values `cross_helper_reverse_topk_independent` and `cross_helper_reverse_topk_cumulative`. Worker flag `--stage1_v5_voff_mode`; coordinator/shell flag `--stage1-v5-voff-mode`.
- Donors: same sample's other-step helper images, exclude every occurrence of current-helper identity; deduplicate eligible identities and pool globally in source order. No other samples. Original-image projected teacher features only when no other helper remains.
- Use the already computed clean teacher query in both modes and both configured legacy/adaptive query paths; never alter positive selection/history or add teacher/vision forwards. FP32 ascending stable cosine; detached donor values.
- K equals the entire current latent-body slot count (currently 8), including positive placeholders. Preserve START/END and original positions. Fewer than K eligible candidates or missing/invalid fallback fails; never repeat, skip, retain positive slots or use the current helper silently.
- Independent replaces current block only with attached clean KV strictly before its first body token. Cumulative begins at the first original body's token even if that block has no targets, and retains all preceding replacements. Replacement latents remain visible to causal attention.
- Text-off stays independent clean-prefix. Independent V-off plus Text-off uses one packed decoder invocation; cumulative uses one rejected-suffix invocation plus a separate Text-off packed invocation within one packed graph scope. One native backward; preserve source-coordinate dropout/checkpoint closures.
- Preserve chosen/reference DIL, clean-top20 selection, vocabulary support, entropy/mass formula and coefficients (mass default half entropy), Text-off KL, target ordering, reductions and sampler/schedule. Strict historical/source/reference resume checks remain; a changed objective requires a fresh run.
- The approved scope includes development and CPU test witnesses; execution waits for this plan's review and execution-method selection. No training/GPU campaign, evaluation campaign, checkpoint mutation, merge, PR, push or destructive cleanup is included. Real ILVR training requires separate two-node/four-rank preflight and coordinator execution.

## Review Focus

1. Duplicate/aliased current-helper paths or shared in-memory images must not re-enter donors through another step; test identity exclusion and stable deduplication in Task 2.
2. A singleton fallback must extract actual detached projected original-image features, with zero extra vision calls; a generic image-special-token embedding fallback must fail (Task 2).
3. A clean block with fewer positive-selected features than its body width still receives all K rejected replacements; underfilled donor pools fail rather than leaving positive values (Tasks 2/3).
4. Empty-target and adjacent/non-text original blocks before a later target must still influence cumulative history; independent later logits must ignore prior donor changes (Task 3).
5. Sourced stale flags and historical contracts lacking the new field must resolve predictably on both node worker paths; no unvalidated default injection, sidecar mutation or objective-changing resume (Tasks 1/4).

## File Structure and Interfaces

- `src/stage1/v5_config.py`, `v5_resume.py`, `src/utils_deepseed.py`: mode validation, explicit argument tracking, schema-aware saved identity and pre-model conflict rejection.
- New `src/stage1/v5_donors.py`: pure, sample-local reverse ranking and provenance; no model calls or trainer dependency.
- `src/stage1/types.py`, `data.py`, `targets.py`: source-image identities, detached rejected-block carrier, existing teacher query/feature reuse in both query modes.
- `src/stage1/trainer.py`, `v5_trainer.py`: optional target request, replay wiring and source-contract bindings while retaining the native parent lifecycle.
- `transformers/src/transformers/models/qwen2_5_vl/stage1_v5_replay.py`: replacement-aware row packing and causal cumulative suffix. Only narrow changes to `stage1_v5_trace.py` or `v5_execution.py` if the comparative witness proves necessary.
- `run_stage1_training.sh`, `src/multinode_runtime.py`, `scripts/stage1_v5_train_eval.sh`: actual worker/coordinator/convenience option propagation; preserve protocol and automatic post-training evaluation behavior.
- New focused tests named below; reuse existing tiny actual-Qwen fixtures, independent full-sequence oracle and native Trainer witnesses. Do not create a parallel runtime framework.

## Execution Prerequisites

After the user reviews this plan and selects an execution method, use that method's required skill. `using-git-worktrees` Step 0 already established the existing linked worktree (git-dir `/root/workspace/ILVR/.git/worktrees/ILVR7`, common-dir `/root/workspace/ILVR/.git`, no superproject). Reconfirm unchanged baseline/ownership; use the existing validated venv and symlinks without reinstalling dependencies. Before changing product code run the focused existing baseline command with the global test environment: `./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_config.py tests/stage1/v5/test_resume.py tests/stage1/v5/test_replay.py`. Record full output and exit status; an existing failure requires reporting and the user's choice before implementation. Do not run the full project suite as this baseline and again at completion.

## Task 1: Mode identity, strict historical contracts and Python CLI

**Files:** Modify `src/stage1/v5_config.py`, `src/stage1/v5_resume.py`, `src/utils_deepseed.py`; extend `tests/stage1/v5/test_config.py`, `test_resume.py`, `test_clean_training.py` as needed.

**Interfaces:**
- Produces `Stage1V5Config.voff_mode: str = 'attention_mask'`, validated against exactly the three global values, plus alias `stage1_v5_voff_mode -> voff_mode`.
- Existing `resolve_v5_config`, `build_v5_contract`, `load_v5_contract`, `validate_v5_resume`, `resolve_v5_query_resume` keep signatures; resolved historical contracts return `voff_mode='attention_mask'` only after exact historical payload authentication.
- Python parser exports `args.stage1_v5_voff_mode` and `args.stage1_v5_config.voff_mode`; track explicit presence so omitted resume values inherit saved mode.

- [ ] **Step 1: Add failing mode and contract tests.** Assertions include `resolve_v5_config({}).voff_mode == 'attention_mask'`; every exact mode roundtrips; an unknown mode fails; replacement is rejected outside Stage1-v5. Freeze representative pre-change schema 1/2/3 payloads independently of the new dataclass and assert read-only loading does not add a key or alter source/reference/formula/query content. Schema 1 remains non-resumable; conflicting saved mode and missing/extra/malformed historical config keys fail before model loading. New mode retains clean/all denominators and saved entropy/mass values.
The minimum pinned test is:

```python
def test_default_voff_mode_stays_attention_mask():
    assert resolve_v5_config({}).voff_mode == 'attention_mask'
    assert resolve_v5_config({}).lambda_topk_mass == 0.0
```

- [ ] **Step 2: Run RED.** `./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_config.py tests/stage1/v5/test_resume.py tests/stage1/v5/test_clean_training.py`; new failures must identify the missing mode/contract behavior rather than unrelated import errors.
- [ ] **Step 3: Implement identity.** Add the field and alias; preserve historical key sets and reconstruction without the new key for schemas 1/2/3. Continue writing the exact historical shape for fresh mask mode (schema 2 all / schema 3 clean); write schema 4 for replacement modes. Schema 4 records both the selected mode and fixed donor/query/fallback/rank/K/prefix/Text-off semantics, with conditional existing all/clean DIL denominator validation. Preserve `v5_formula_id` because the loss formula is unchanged; authenticate the intervention separately. Source-bound historical training may still fail unchanged strict source guards; do not bypass them or relabel artifacts. Update parser special-name exclusions and explicit-mode inheritance.
- [ ] **Step 4: Run GREEN.** Repeat the focused command from Step 2; require exit 0, including historical fixtures and pre-model rejection.
- [ ] **Step 5: Commit.** Stage only this task's reviewed paths; `git commit -m "feat: define Stage1-v5 replacement V-off contracts"`.

## Task 2: Sample-owned reverse-top-k teacher donors

**Files:** Create `src/stage1/v5_donors.py`, `tests/stage1/v5/test_donors.py`, `tests/stage1/v5/test_donor_targets.py`; modify `src/stage1/types.py`, `data.py`, `targets.py`; extend existing data/history tests only where necessary.

**Interfaces:**
- Add defaulted `helper_image_ids: tuple[str, ...] = ()` and `original_image_ids: tuple[str, ...] = ()` to `Stage1SampleMetadata`. Generate identities from prepared source paths and row-local shared-object identities for in-memory images; never persist memory addresses. Preserve defaults and default-mode behavior for older callers.
- Add frozen `Stage1RejectedLatentBlock(batch_index: int, block_index: int, positions: tuple[int, ...], values: Tensor, donor_kind: str, donor_image_indices: tuple[int, ...], donor_candidate_indices: tuple[int, ...], query_provenance: str)` in `types.py`; add `Stage1TeacherTargets.rejected_blocks: tuple[Stage1RejectedLatentBlock, ...] = ()`.
- In new module define frozen `V5DonorSelection(values: Tensor, donor_kind: str, image_indices: tuple[int, ...], candidate_indices: tuple[int, ...])` and `select_reverse_topk_donors(helper_candidates: tuple[Tensor, ...], helper_image_ids: tuple[str, ...], helper_to_block: tuple[int, ...], block_index: int, original_candidates: tuple[Tensor, ...], query: Tensor, *, count: int, source_index: int) -> V5DonorSelection`. Inputs are already per-image grouped, sample-owned features; returned image indices refer to the chosen helper or original source tuple.
- Extend `Stage1TeacherTargetBuilder.build(teacher, batch, metadata, *, voff_mode: str = 'attention_mask') -> Stage1TeacherTargets`. Same donor output for both replacement modes. Do not change default callers or positive target tensors.

- [ ] **Step 1: Add failing pure-selector tests.** Use hand-computed unit vectors with known ascending cosine order; assert exact global ordering across two donor images, stable source/index ties, current-image exclusion including duplicate identity across steps, eligible deduplication, original fallback only when the eligible helper set is empty, no cross-row candidates, K=8 shape, M<K error, mismatched ownership, nonfinite features/query/score, zero/invalid count and malformed provenance errors. If helper pool exists but M<K, do not supplement from originals.
A pinned hand-computed selector example (import torch and the new selector in the test file):

```python
def test_global_reverse_cosine_excludes_current_image():
    result = select_reverse_topk_donors(
        (torch.tensor([[1., 0.]]), torch.tensor([[-1., 0.], [0., 1.], [1., 0.]])),
        ('current', 'other'), (0, 1), 0, (), torch.tensor([[1., 0.]]),
        count=2, source_index=0)
    assert result.donor_kind == 'helper'
    assert result.image_indices == (1, 1)
    assert result.candidate_indices == (0, 1)
    torch.testing.assert_close(result.values, torch.tensor([[-1., 0.], [0., 1.]]))
```

- [ ] **Step 2: Add failing target-path tests.** Instrument existing legacy and adaptive teacher fixtures to expose projected original-image values distinct from token embeddings. Assert same query and donor tensors in both modes, unchanged positive selections and adaptive ledger, all body slots filled even when positive selection count is smaller, detached helper/fallback outputs, and identical teacher/vision call counts to mask baseline. Invalid or absent actual original-image embedding/grid ownership must fail without silently using ordinary token embeddings. Assert no rejected output/ranking in default mode; debug-only helper/fallback block counters are accurate. Cover path aliases and shared in-memory image objects in collator metadata.
- [ ] **Step 3: Run RED.** `./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_donors.py tests/stage1/v5/test_donor_targets.py`; require causal missing-feature failures.
- [ ] **Step 4: Implement the interfaces.** Group helper and original features with the existing helper grouping; rank in FP32 using stable ascending cosine with the existing epsilon, then gather original-dtype detached values. Exclude by identity before ranking. Extract fallback from authentic legacy prepared teacher embeddings or adaptive prepared prompt embeddings already available. Hook donor selection immediately beside each existing positive selection using that exact query; never feed rejected values into clean query construction. Build full-body positions and block provenance in source order, including blocks without immediate text targets. Failure messages identify source and block.
- [ ] **Step 5: Run GREEN and compatibility witnesses.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_donors.py tests/stage1/v5/test_donor_targets.py tests/stage1/test_data.py tests/stage1/test_targets.py tests/stage1/test_history_targets.py tests/stage1/v5/test_history_integration.py`. Require all selected tests exit 0 and positive/default tensors unchanged.
- [ ] **Step 6: Commit.** `git commit -m "feat: select same-sample reverse-topk rejected latents"` after staging only this task's paths.

## Task 3: Replacement-aware independent and cumulative replay

**Files:** Modify `transformers/src/transformers/models/qwen2_5_vl/stage1_v5_replay.py`; create `tests/stage1/v5/test_replacement_replay.py`; extend `tests/stage1/v5/test_execution.py` only for any proven execution-context change.

**Interfaces:**
- Extend `V5ReplayBlock` with final defaulted `replacement_values: Tensor | None = None`; ordered shape is `[end-start-1, hidden_size]`, representing every body position. Existing six-argument construction remains valid.
- Extend `run_v5_packed_replay(..., chunk_positions: int = 128, voff_mode: str = 'attention_mask') -> V5ReplayOutput`; existing return/logit ordering remains `metadata.query_keys`. Replacement-mode callers supply all original blocks including empty-target blocks, with detached replacement tensors. No training-package imports inside Transformers.
- Existing `V5Execution.graph('packed')` owns all OFF closures, while each decoder invocation gets its own captured source-coordinate context; no duplicate graph entry or second backward.

- [ ] **Step 1: Add failing independent full-sequence comparisons.** Reuse the existing tiny Qwen/DoRA fixture and literal full-forward oracle from `test_replay.py` for execution bindings only. Independently build clean and rejected embeddings/masks without calling production packing or selector helpers. For both modes compare ordered selected logits, entropy/mass/Text-off terms, total loss and every trainable parameter gradient; use existing FP32 tolerances `ATOL=1e-5, RTOL=1e-4`. Cover uneven prefix/suffix sizes, real three-axis position IDs, full-body replacement, original padding, one block, two blocks and empty first-block targets. Require direct evidence that prefix gradients remain attached and clean trace tensors are unmodified.
Name the primary comparison `test_replacement_logits_and_gradients_match_full_sequence`; after constructing native results and the independent full-sequence reference, pin:

```python
assert native_query_keys == oracle_query_keys
torch.testing.assert_close(native_voff, oracle_voff, atol=1e-5, rtol=1e-4)
torch.testing.assert_close(native_loss, oracle_loss, atol=1e-5, rtol=1e-4)
assert native_gradients.keys() == oracle_gradients.keys()
for name in native_gradients:
    torch.testing.assert_close(native_gradients[name], oracle_gradients[name], atol=1e-5, rtol=1e-4)
```

- [ ] **Step 2: Add failing history and execution witnesses.** Perturb only first-block donors: independent later-block logits stay identical, cumulative later-block logits change; future donors cannot change earlier outputs. Assert replacement bodies are attended rather than V-off-masked. Parametrize dropout 0 and existing DoRA 0.05, checkpoint off/early-stop/full recompute, V-only/Text-only/both, and no eligible queries. Count original OFF decoder calls: independent both=1, cumulative both=2; count one native backward and verify each call's retained checkpoint coordinates. Do not count checkpoint recomputation as an original invocation.
- [ ] **Step 3: Run RED.** `./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_replacement_replay.py`; require missing replay/intervention behavior failures. This is the separately unexecuted first-nail test now within approved development, not evidence already obtained.
- [ ] **Step 4: Implement replay.** Retain default mask implementation behavior. Independent rows start at `start+1` with clean attached prefix `[:start+1]`; Text-off rows retain their existing `end` cutoff and masks. Use row-specific starts/cutoffs, full replacement bodies and unmodified original position coordinates. Cumulative runs one causal suffix beginning at the first original body's token, overwrites all encountered bodies (including empty-target blocks), scores only required queries through the last target, then invokes the existing Text-off-only path if needed. Both calls remain within the caller's single packed graph. Keep replacement embeddings detached and prefix KV attached; functional scatter must block gradients into overwritten clean body embeddings. Keep logits in original query order and existing projection chunks.
- [ ] **Step 5: Run GREEN.** `./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_replacement_replay.py tests/stage1/v5/test_replay.py tests/stage1/v5/test_execution.py`; exit 0 establishes tiny CPU comparative fidelity only. If the first-nail witness contradicts the chosen cache premise, stop and report the contradiction before widening scope or weakening tolerances.
- [ ] **Step 6: Commit.** `git commit -m "feat: replay independent and cumulative latent replacement"` after staging scoped changes.

## Task 4: Native trainer wiring, actual launch propagation and usage

**Files:** Modify `src/stage1/trainer.py`, `src/stage1/v5_trainer.py`, `run_stage1_training.sh`, `src/multinode_runtime.py`, `scripts/stage1_v5_train_eval.sh`, `docs/stage1-v5.md`; create `tests/stage1/v5/test_replacement_trainer.py`; extend `tests/stage1/v5/test_launcher.py`, `test_resume.py`, `test_workflow.py` only where these actual paths require it.

**Interfaces:**
- Extend `_prepare_stage1_loss_inputs(self, inputs, *, voff_mode: str = 'attention_mask')` without changing its return tuple. Call the target builder with the new keyword only for enabled replacement requests, preserving default custom-builder callers.
- V5 compute_loss requests donors only when V-off is actually required, attaches each `Stage1RejectedLatentBlock.values` to the matching `V5ReplayBlock` by original row/block/position identity (not by active-target order), and passes `self.v5_config.voff_mode` to replay. No objective math changes.
- Preserve the existing coordinator entrypoint. Allowlist and translate `--stage1-v5-voff-mode`; convenience recipe captures explicit mode before sourcing settings and enforces it after sourcing, defaults fresh mask mode, and passes it to workers. Resume paths inherit authenticated saved mode when omitted and reject conflicts.

- [ ] **Step 1: Add failing real trainer tests.** Extend the actual Qwen/DoRA + parent Trainer integration fixture to execute both replacement modes with clean-top20, entropy 0.1 and mass 0.05, Text-off 0.01, CE/alignment 0.1. Compare losses and gradients against an independently assembled reference; verify one target build, unchanged clean/reference/Text-off outputs and positive targets, original block/selected-token reductions including empty blocks and partial accumulation, one native backward, immutable frozen reference and skipped donor/replay work when auxiliary branches are off. Assert wrong/missing replacement ownership fails before replay.
Name the actual trainer test `test_replacement_uses_one_native_backward_and_saved_loss_profile`; assert against its captured real runtime results:

```python
assert target_build_calls == 1 and native_backward_calls == 1
assert off_original_calls == (2 if mode == 'cross_helper_reverse_topk_cumulative' else 1)
assert trainer.v5_config.lambda_topk_entropy == 0.1
assert trainer.v5_config.lambda_topk_mass == 0.05
assert trainer.v5_config.text_kl_weight == 0.01
assert digest_state(trainer.ema_teacher.model) == reference_digest_before
```

- [ ] **Step 2: Add failing real command tests.** Exercise convenience recipe -> coordinator dry-run -> both generated node worker dry-runs -> actual Python parser, with every mode and stale sourced/env values. Assert rank0/rank1 two-node/four-rank topology, resolved mode, all original coefficients, sampler/schedule and query controls, with no SSH, model load or training. Add saved-mode omission/conflict and standard/Stage2 rejection witnesses; document sample/helper fallback source counters without claiming effectiveness.
- [ ] **Step 3: Run RED.** `./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_replacement_trainer.py tests/stage1/v5/test_launcher.py`; new failures must trace missing wiring or propagation.
- [ ] **Step 4: Implement wiring and usage.** Forward the opt-in mode through the actual target/replay paths; source-bind the new donor and replay implementation in current runtime contracts without weakening old strict guards. Keep sample/block indexing explicit and include empty blocks for cumulative history. Add shell/coordinator propagation and saved-identity handling. Update `docs/stage1-v5.md` with the exact three values, teacher query/donor/fallback/K semantics, independent/cumulative difference, OFF decoder counts, fresh-run requirement and the limits of CPU evidence. Do not change convenience recipe's existing evaluation protocol or introduce another launcher.
- [ ] **Step 5: Run focused GREEN.** `./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_replacement_trainer.py tests/stage1/v5/test_launcher.py tests/stage1/v5/test_trainer.py tests/stage1/v5/test_resume.py tests/stage1/v5/test_clean_training.py`; require exit 0. Use the existing smallest native-dispatch fixture rather than launching a training smoke.
- [ ] **Step 6: Commit.** `git commit -m "feat: wire optional replacement V-off through Stage1-v5"` after staging only reviewed task paths.

## Review, Final Verification and Handoff

- [ ] Complete the selected execution method's required task/fixer/review workflow. An independent reviewer distinct from the implementer must assess both approved-spec compliance and quality; preserve findings, exact commits, test evidence and any unresolved limitations. Do not substitute tests or self-review for that review.
- [ ] Before final tests, read `verification-before-completion` and determine applicability of `finishing-a-development-branch`. If finishing applies next, use its full project command once for the overlapping test-pass claim; do not run it twice. Project command here is `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q`. Record full output and exit status against the final reviewed revision. Run focused suites during TDD; one final full suite after all review fixes is sufficient unless a new failure/change warrants more.
- [ ] Run `git diff --check` and inspect the actual changed paths, source/default-mode compatibility, exact approved requirement coverage and preserved symlinks. Distinguish CPU comparative tests, dry-run topology and real distributed/GPU evidence. Do not claim GPU efficiency, training effectiveness or end-to-end multi-node execution from CPU tests.
- [ ] Persist exact approval, execution choice, agent IDs if used, changed commit identities, reviewer verdicts/fixes and claim-specific verification before any scratch cleanup. Honor externally managed worktree ownership; the human owns branch/push/PR/keep decisions.
- [ ] Report the available option and tested behavior concisely; state that training has not been launched. Run the repository ntfy completion-summary hook before the final response.

## Plan Self-review and Execution Gate

Spec coverage: public mode/compatibility -> Task 1; source identities/query/donor/fallback/slots -> Task 2; prefixes/positions/gradients/dropout/checkpoints/Text-off calls -> Task 3; native trainer/entropy-mass/reductions/worker propagation/docs -> Task 4; review/verification/ownership -> final gate. All five Review Focus conditions have owning tests above. Interfaces use one training carrier, one pure selection result, and one defaulted replay-block field, avoiding training imports in Transformers. Internal factoring and fixture placement remain builder choices as permitted by the spec.

Implementation method remains unselected. This plan must be presented for user review and selection between Subagent-driven and Native execution before implementation, as required by writing-plans and the standing user policy. Recommendation: Native, because the four tasks share target/replay/contract interfaces and the existing full-sequence oracle supplies a strong final independent review boundary with less per-task handoff overhead. This is a recommendation, not a method choice on the user's behalf.

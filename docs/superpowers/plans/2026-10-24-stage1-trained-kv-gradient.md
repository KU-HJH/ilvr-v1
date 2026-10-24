# Trained Stage1-v5 KV Loss/Gradient Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user has already selected this method; do not switch to native execution.

**Goal:** Measure the unchanged trained checkpoint's local BF16 loss and trainable-parameter gradient difference between historical attached-prefix replay and independent full decoder-prefix recomputation.

**Architecture:** Add one standalone diagnostic and one CPU test file in an isolated descendant of historical revision `07eb774281925866502e72bdfcaa4df54073103a`. Keep historical loss, helper, selection, dropout and reduction code intact; replace only the replay call in the oracle arm. Validate the carrier and independent oracle on CPU before four bounded GPU passes.

**Tech Stack:** Repository Python 3.11 venv, PyTorch 2.7.1+cu128, repository Transformers, PEFT DoRA, native Stage1-v5 APIs, pytest.

**Spec:** `docs/superpowers/specs/2026-10-24-stage1-trained-kv-gradient-design.md` — approved v2, SHA256 `137d1c5be09a596effacd3fedc34dea3c1f02584c684228a0725bfeedf7f6f34`. Approval is recorded in the control worktree's `results/kv-stage1-training-parity-20261024T123825Z/design-approval-v2.json`; the spec's original proposal-status line is superseded by that approval record, without changing approved bytes.

## Global Constraints

- “Use an isolated diagnostic descendant of `07eb774`.” The full revision is `07eb774281925866502e72bdfcaa4df54073103a`; the existing KV worktree supplies control documents only.
- “Add only a focused CLI `scripts/probe_stage1_training_kv.py` and its CPU regression file `tests/stage1/v5/test_training_kv_probe.py`.”
- “Do not edit native model, Trainer, loss, coordinator, existing tests or production launchers.” Superpowers control documents and evidence are separate permitted controller artifacts.
- “This is a standalone forward/autograd diagnostic. It performs no optimizer step, Trainer.train, checkpoint save, scheduler update or distributed training.”
- “All imported native modules must resolve inside the isolated historical descendant; no mixing current and historical runtime modules.”
- “Use only the repository venv. Preserve `data/`, `checkpoints/` and venv links.” Model-bearing CPU fixtures require a unique explicit `checkpoints/` basetemp; results/logs stay in this execution worktree.
- “Before any GPU metric, select the first retained TRAIN row by ascending source index with one or two original latent blocks, at least one scored target, native collated length at most 512 and complete original/helper images.” No truncation, bounds relaxation, fallback or metric-based reselection.
- “The student BF16 representation must match the saved artifact”; “Preserve the authenticated reference's own dtypes.” No persistent artifact changes.
- “Keep saved CE0.1/sim0.1 and every saved DIL/entropy/TextKL coefficient and formula.” Preserve source-coordinate dropout0.05, saved checkpointing, and BF16 storage without autocast.
- “With B original blocks, the historical V5Window is `(global_original_steps=16*B, actual_microbatches=4, configured_gas=4, world_size=4)`, yielding the native block scale `1/B`.” Report the raw objective and gradients actually computed from `loss/4`; these nominal coefficients do not represent a distributed run.
- “The oracle receives only this restricted input view and frozen block/target coordinates.” No native-trace embeddings/positions, K/V, attention rows, visibility/planner output or prefix states; no cache/prefix inputs.
- “Preserve and passively observe each actual result; never substitute A's support into B or force a common support.” Entropy supports remain native per-arm `independent` selections.
- “Run fresh unobserved native U, observed native A, observed native A-repeat and observed oracle B” under identical represented state and stochastic context.
- “BF16 results are measurements, never declared equivalent using FP32 tolerances.” Repeat fidelity is a prerequisite, not a cross-path error tolerance.
- “The separately approved additional GPU budget is 300 aggregate process-seconds; historical use is 289.42876996000996 seconds.” Imports, authentication, loading, failures, backward and reporting all count; new approved-budget use is currently zero.
- Missing prerequisites, inactive branches, unsupported runtime, OOM, nonfinite values and deadline exits retain their actual incomplete/inconclusive/unexercised meaning. No training-repair, distributed-engine or branch-Jacobian equivalence claim follows from this experiment.

## Review Focus

1. A second block overlaps an earlier prefix: its full-prefix graph must keep clean prefix visibility and live token-row gradients, while helper targets stay detached. Owned by Task 1 cycle B.
2. V-off ranking crosses the entropy top-k boundary between arms: native independent supports must differ without being replaced by a shared support. Owned by Task 1 cycle C.
3. A forward, observer or backward exception occurs after temporary bindings/hooks are installed: restore exact attribute presence/value and release execution contexts. Owned by Task 1 cycles B/C.
4. Gradient sets contain matched unused parameters, one-sided missing gradients, zero vectors or nonfinite entries: preserve the distinctions and report defined metrics without manufacturing a pass. Owned by Task 1 cycle D.
5. No real row meets all eligibility bounds, a foreign native module resolves, or four passes cannot finish: return explicit incomplete evidence without silently changing data, runtime or precision. Owned by Task 1 cycles A/D.

---

## Controller setup after human plan review

Do not implement or spend the additional GPU budget before the human reviews this plan. The separate design approval and execution method are already settled.

- [ ] Read `superpowers:using-git-worktrees` at execution setup. Start from canonical `/root/workspace/ILVR`, whose Git directory equals its common directory, to create the approved historical isolation. Do not reset/rebase the existing control worktree or modify canonical dirty source.
- [ ] Recheck no native worktree tool exists, `.worktrees` is ignored, and the explicit target and branch are unused; create `/root/workspace/ILVR/.worktrees/stage1-trained-kv-20261024` on `diagnostic/stage1-trained-kv-20261024` from the exact historical revision. Preserve any preexisting path instead of overwriting it.
- [ ] Link canonical `venv`, `data`, and `checkpoints` using the established repository layout; verify all three resolved targets. Do not reinstall dependencies. Create this worktree's own results/log directories with run ID `kv-stage1-training-parity-20261024T123825Z`.
- [ ] Copy the exact approved spec and reviewed plan as control files into that worktree and commit only those control files. Copy bound approval/hash/source-map evidence into its results directory. Record historical fork, control commit, path ownership and hashes.
- [ ] Establish a focused historical baseline with the command below, substituting a unique `baseline-cpu-<UTC>` basetemp. Inspect complete output; report failures before proceeding. This is a baseline, not the final whole-project verification.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_replay.py tests/stage1/v5/test_trainer.py -m 'not gpu and not distributed' --basetemp=checkpoints/kv-stage1-training-parity-20261024T123825Z/baseline-cpu-<UTC>
```

- [ ] Start this plan's SDD workspace with the skill's `scripts/sdd-workspace`, reconcile any existing ledger using `references/resuming.md`, then run its preflight conflict scan. One coupled implementation task has no cross-task interface pairs; still record its internal consistency row. Generate its task brief through `scripts/task-brief`, record BASE and dispatch one fresh implementer. Workers must not dispatch children or invoke Paperthin hooks. Only root may run the budgeted GPU experiment.

## File ownership and historical API map

| File | Responsibility |
| --- | --- |
| `scripts/probe_stage1_training_kv.py` (new) | CPU selection/preparation, training-shaped loading, original-input channel, independent replay, native carrier, four-arm measurements and JSON CLI report. |
| `tests/stage1/v5/test_training_kv_probe.py` (new) | CPU tests with actual tiny Qwen/DoRA/native carrier and parent dispatch, causal mutations, observer fidelity and bounded failure/report contracts. |

The two files form one testable diagnostic. Do not split setup or helper functions into extra deliverables. Test-only mutation variants belong in the test file, not production CLI options.

Read these historical implementations before writing tests:

- `src/stage1/composition.py`: `_add_fresh_latent_tokens`, `_configure_latent_model`, `_load_stage1_model` and the resume branch of `build_stage1_training_components`. The later `scripts/analyze_stage1_v5_causal.py` does **not** exist at this revision and is not a runtime dependency.
- `src/stage1/checkpointing.py`: `load_stage1_peft_metadata`, `preflight_stage1_peft_artifact`, `load_stage1_peft_artifact(..., requested_base_model_reference=..., is_trainable=True, v5_resume=True)`; sparse rows use the native wrappers. Historical loading permits `v5_resume=True`.
- `src/stage1/data.py`: `prepare_stage1_comt_examples(data_path: Path, image_root: Path)`, `Stage1DataCollator(processor, token_ids, latent_size, stage1_variant='v5')`. Preserve its exact retention/preparation behavior; do not invent a row-skipping filter for preparation errors.
- `src/stage1/v5_resume.py`: `load_v5_contract(checkpoint)` reads **`stage1_v5.json`**, not `stage1_v5_contract.json`. Keep its validated config, coefficients and formula ID.
- `src/stage1/v5_reference.py`: `restore_v5_reference(teacher, student, processor, *, base_path, directory, expected)` must run before helper/reference computation.
- `src/stage1/v5_trainer.py`: actual `compute_loss` imports replay inside the method, so patch the replay module's actual `run_v5_packed_replay` binding. It calls the historical objective unchanged and enters `execution.backward()` through `_v5_step_stack`.
- `src/stage1/v5_objective.py`: wrap that module's actual `build_entropy_support` binding, returning the exact original support object; do not patch only the original definition in `topk_entropy_loss.py`.
- `tests/stage1/v5/test_trainer.py::integrated_fixture` and `tests/stage1/v5/test_replay.py::fixture`: reuse actual tiny models and native dispatch. The parent fixture substitutes engine/profile/helper pieces and constructs an optimizer; do not call its optimizer, and report those substitutions honestly.
- `stage1_v5_trace.py`: `bind_v5_execution`, `_PACKED` source-coordinate mapping and checkpoint closures. Oracle may use execution-only bindings, but must independently construct masks and source mappings, never reuse trace/planner outputs.

## Task 1: Implement the complete standalone diagnostic with CPU evidence

**Files:**
- Create: `scripts/probe_stage1_training_kv.py`
- Create/Test: `tests/stage1/v5/test_training_kv_probe.py`

**Interfaces:**
- Consumes: historical APIs above; saved checkpoint/base/reference; original TRAIN and images. No output from another implementation task.
- Produces these script interfaces (all tensor-bearing objects remain in memory):
  - `select_example(examples: Sequence[PreparedStage1Example], collator: Stage1DataCollator) -> tuple[PreparedStage1Example, dict[str, object], dict[str, object]]`: returns the first eligible native batch plus scalar/hash input evidence; raises a clear prerequisite error if none qualifies.
  - `diagnostic_window(original_blocks: int) -> V5Window`: exact repeated-example normalization from the spec; reject zero/invalid counts.
  - Frozen `OriginalDecoderInputs(inputs_embeds: Tensor, position_ids: Tensor, padding: Tensor)` and `capture_original_inputs(decoder: nn.Module) -> Iterator[list[OriginalDecoderInputs]]` as a context manager: one original F input view, no detached embedding copy.
  - `full_prefix_replay(decoder: nn.Module, lm_head: nn.Module, original: OriginalDecoderInputs, blocks: tuple[V5ReplayBlock, ...], *, execution: V5Execution, need_voff: bool, need_textoff: bool, chunk_positions: int = 128) -> V5ReplayOutput`: independent full-prefix implementation with no trace argument. A local adapter accepts the native replay signature but never reads its trace argument.
  - `run_local_backward(trainer: CustomTrainerStage1V5, batch: dict[str, object], *, stochastic: Stage1StochasticContext, oracle: bool, observe: bool) -> ArmResult`: native carrier with one ordinary backward of local `loss/4`, fresh input/helper graph and `.grad` clearing, exact state restoration.
  - `ArmResult` contains `raw_loss: float`, `scaled_loss: float`, `components: dict[str, float]`, `gradients: dict[str, Tensor | None]` keyed by **every** named trainable parameter, and `evidence: dict[str, object]`. Gradient tensors are temporary CPU comparison state, never serialized.
  - `compare_gradients(left: Mapping[str, Tensor | None], right: Mapping[str, Tensor | None]) -> dict[str, object]`: parameter-wise and global metrics with explicit missing/unused/zero/nonfinite handling.
  - `run_probe(args: argparse.Namespace) -> dict[str, object]` and `main(argv: Sequence[str] | None = None) -> int`: compose validated historical loading and the four passes, persist scalar/hash report, return explicit status.
- CLI: required `--checkpoint`, `--base`, `--data-path`, `--image-root`, `--run-id`; optional `--device` (default `cuda:0`) and `--prepare-only` (CPU selection/authentication metadata without model loading or CUDA). Block count, 512 bound, dtype, dropout, coefficients, arm order and seed are fixed by the approved contract, not tunable CLI knobs.
- CLI output: `results/<run-id>/prepared-input.json` and `results/<run-id>/report.json`; logs are redirected by the controller to `logs/<run-id>/`. Refuse existing result files rather than overwrite previous experiment evidence. Status `measured`/exit0 means valid local measurements, **not equivalence**; `incomplete` or `inconclusive` exits nonzero. Include `schema_version=1`, `scope`, `runtime`, `input`, `objective`, `arms`, `controls`, `comparisons`, `limitations` and a reason on failure. Never emit NaN/Inf JSON numbers; retain nonfinite flags with null metrics.

### TDD cycle A: exact inputs, objective and historical loading boundaries

- [ ] **Write failing tests** in the new test file, using assertions below as the contract; adapt only test-local fixture construction, not pinned values.

```python
def test_first_eligible_row_uses_original_bounds():
    assert selected.source_index == first_eligible_source_index
    assert 1 <= block_count <= 2 and scored_targets >= 1 and length <= 512
    assert unchanged_original_tokens_images_and_source_order

def test_no_eligible_row_does_not_fallback():
    with pytest.raises(ValueError, match="eligible"):
        select_example(no_eligible_examples, native_collator)

@pytest.mark.parametrize("blocks", [1, 2])
def test_window_preserves_historical_block_normalization(blocks):
    assert diagnostic_window(blocks) == V5Window(16 * blocks, 4, 4, 4)
    assert v5_auxiliary_scale(diagnostic_window(blocks)) == 1 / blocks
```

Also add `test_runtime_rejects_foreign_native_import`, `test_saved_contract_and_original_added_tokens_are_preserved`, `test_trainable_artifact_values_and_reference_dtype_are_preserved`, and `test_prepare_only_never_loads_model_or_initializes_cuda`. Use native tiny checkpoint/reference factories under pytest's checkpoint basetemp for restoration tests; assert restored represented values and exact processor/reference identities, not just a loader mock call. The foreign-root fixture must actually alter one imported module path and observe rejection.

- [ ] **Run cycle A RED.** Run the command below, adding `-k` with the cycle A test names; substitute an actual unique UTC suffix for each basetemp. Use test-local assertions on required missing interfaces before importing them so initial RED is an assertion failure, not a collection/import crash. Record complete output and failing test names before any corresponding implementation. Reuse this command with the relevant test names for subsequent cycle RED/GREEN checks.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_training_kv_probe.py --basetemp=checkpoints/kv-stage1-training-parity-20261024T123825Z/cycle-<UTC>
```

- [ ] **Implement cycle A minimal code.** Use historical native processor/model/PEFT/reference and collator paths. Reconstruct original AddedToken types, validate `stage1_v5.json` and all saved coefficients. Preserve exact student artifact values in BF16 and reference dtypes; report effective trainable inventory, checkpoint step, formula and backend settings. Reapply the saved backend configuration before execution and read back actual settings; do not add the rejected evaluation FP64/MATH/reduction profile. Authenticate module roots/hashes, the three recorded Trainer hashes and package versions. Report partial historical authentication honestly.
- [ ] **Run cycle A GREEN** with the same selected tests and fresh basetemp. Confirm real restore/selection behavior; fix implementation rather than weakening exact identities or eligibility.

### TDD cycle B: carrier, original-input channel and independent oracle

- [ ] **Write the carrier prerequisite test** `test_carrier_matches_actual_parent_loss_and_all_gradients`, parameterized with native checkpointing on/off and source dropout `.05`. Use two independently initialized but equal tiny trained DoRA/row models with real native parent `training_step` and diagnostic carrier, the same `V5Window(32,4,4,4)` for two blocks, the same source-coordinate context and actual `loss/4` backward. Include the fixture's `real_helper=True` native target-builder case; do not claim the default synthetic helper authenticates native helper preparation. Assert matching raw/scaled losses and every trainable gradient using the existing tiny FP32 oracle's `atol=1e-5, rtol=1e-4`; record substituted engine pieces. These CPU tolerances certify only this FP32 prerequisite.
- [ ] **Write oracle boundary tests:**

```python
def test_oracle_never_reads_native_trace_or_supplies_prefix_state():
    assert forbidden_trace_reads == []  # raising trace sentinel supplied to adapter
    assert all(call.get("past_key_values") is None for call in oracle_calls)
    assert all(call.get("stage2_prefix_key_values") is None for call in oracle_calls)
    assert oracle_full_lengths_match_original_coordinates

def test_two_overlapping_prefixes_preserve_live_rows_and_detached_helpers():
    assert token_row_gradient_is_nonzero and helper_gradient is None
    assert full_prefix_result_matches_independent_literal_mask_fixture
    assert detached_original_input_control_changes_token_row_gradient
```

Add `test_all_row_latent_mask_changes_oracle_result`, `test_detached_native_prefix_changes_gradient`, `test_textoff_edges_do_not_contaminate_voff`, `test_dropout_and_recompute_use_original_source_coordinates`, and `test_failed_replay_restores_execution_and_binding_presence`. Use actual model outputs/gradients, two original blocks, nonzero trained adapters and sparse rows; run wrong-mask/detachment variants only through test-local mutations. Assert clean START/END visibility and suffix-only latent-body removal, target-query-specific Text-off edges and checkpoint recomputation site ownership.

- [ ] **Run cycle B RED** for those new tests before implementing their corresponding carrier/oracle behavior. Preserve expected assertion failures and no-GPU evidence.
- [ ] **Implement cycle B minimal code.** Invoke actual historical `compute_loss` through a minimally populated carrier and real `V5Execution`/ExitStack; allow native loss to open its backward scope. Capture original live decoder inputs before native trace changes kwargs. Construct one packed batch of independent full causal sequences, keeping earlier prefix rows clean; gather only native target predictors. Bind `_PACKED` source indices/positions/validity independently so native dropout and checkpoint closures see the same original coordinates. Temporary replay adapter ignores its trace argument. Restore module bindings and hooks exactly, including instance-attribute absence, on every exit. Enumerate constructor/profile/engine bypasses.
- [ ] **Run cycle B GREEN** and verify both positive matches and deliberately wrong variants produce the expected divergent gradients/visibility. Failure of carrier fidelity blocks GPU use; do not reclassify it as a tolerance problem.

### TDD cycle C: passive observation, native independent supports and four-arm fidelity

- [ ] **Write tests** `test_entropy_observer_returns_original_support_object`, `test_crossing_rank_keeps_each_arms_native_independent_support`, `test_observer_preserves_native_loss_and_all_gradients`, `test_observer_exception_restores_original_binding`, and `test_four_passes_use_fresh_graphs_and_unchanged_state`. Force clean versus V-off rank crossing in an actual native loss fixture and assert observed indices equal each arm's native unwrapped result, differ across arms, and never come from a forced shared support. Compare the complete unobserved/observed native scalar and gradient sets, helper/reference/clean identities, selected Text-off coordinates and parameter values.
- [ ] **Run cycle C RED**, then implement passive support observation at the historical objective's actual binding, preserving the exact returned object and native arithmetic. Keep saved `stage1_v5_debug=False` consistent across all four production arms; do not turn on a native alternate execution mode merely to collect components. If native diagnostics omit component sums, an observed-arm wrapper may capture the actual returned `V5LossSums` without changing/recomputing it, with the same fidelity and restoration tests. U has no new observer/replay wrapper: use native `compute_loss(..., return_outputs=True)` to compare its returned loss/clean logits, gradients and boundary state with A; compare observed helper/reference/support details between A and A-repeat. Report the scope of each directly observed control. Keep the replay intervention exclusive to B; use fresh U/A/A-repeat/B loss graphs with the same parameter values and stochastic key `(saved seed42, recorded checkpoint global_step, canonical slot/source index, unpadded length, full site inventory)`. Record raw source ID separately from the actual canonical dropout key.
- [ ] **Run cycle C GREEN.** U/A and A/A-repeat controls must compare exact represented scalar/gradient/state observations, without borrowing a cross-path FP32 tolerance. A failed control yields inconclusive attribution. Differing A/B entropy supports are preserved and labeled `selection_mediated`, while clean Text-off keys/block coordinates remain common.

### TDD cycle D: honest numerical reports and failures

- [ ] **Write the metric/failure tests** below and `test_inactive_branch_stays_unexercised`, `test_cli_failure_report_is_json_and_never_overwrites_evidence`, and `test_call_witness_distinguishes_native_suffix_and_oracle_full_prefix`.

```python
def test_gradient_report_distinguishes_unused_missing_and_zero():
    assert report["matched_unused"] == ["unused"]
    assert report["presence_mismatches"] == ["one_sided"]
    assert report["parameters"]["zero"]["both_zero"] is True
    assert report["parameters"]["zero"]["relative_l2"] == 0.0
    assert report["parameters"]["zero"]["cosine"] is None

def test_nonfinite_report_cannot_be_measured():
    assert report["status"] in {"incomplete", "inconclusive"}
    assert report["finite"] is False
    json.dumps(report, allow_nan=False)
```

Also test nonzero error against a zero left vector: relative L2 is null with an explicit zero-denominator flag; one-sided zero cosine is null. Define ordinary relative L2 as `||right-left||₂ / ||left||₂` and global metrics by summed squared norms/dot products over matched present tensors, accumulating in FP64 **only for detached report statistics**, never model forward/backward arithmetic. Presence mismatches invalidate whole-gradient equality; matched unused remains valid. Compare all named trainables, including token/head rows and DoRA magnitude; do not fill None with zeros. `torch.equal` exactness and max-absolute error are separate from finite flags.

- [ ] **Run cycle D RED**, implement reports and CLI composition minimally, then run cycle D GREEN. Include loss components, raw/scaled objective, local coefficients, all trainable dtypes/gradient metrics, nonzero adapters/gradients, actual branch edges/query activity, actual decoder call lengths/prefix arguments, parameter before/after hashes, helper/reference/input identities, support indices/hashes/overlap, runtime hashes/versions and evidence limitations. Do not dump tensors, create models in results, claim total-gradient equality proves OFF Jacobians, or call a measured discrepancy a cache defect or remedy.

### Task verification and commit

- [ ] **Run the focused CPU integration set once after all four cycles**, with a unique `task-green-<UTC>` checkpoint basetemp. Read complete output, including warnings, and record results. This is the implementer's focused gate; the root runs the final project suite after reviews on final bytes.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_training_kv_probe.py tests/stage1/v5/test_replay.py tests/stage1/v5/test_trainer.py -m 'not gpu and not distributed' --basetemp=checkpoints/kv-stage1-training-parity-20261024T123825Z/task-green-<UTC>
```

- [ ] **Self-review and commit only the two deliverable files**, e.g. `git add scripts/probe_stage1_training_kv.py tests/stage1/v5/test_training_kv_probe.py` followed by `git commit -m "test: diagnose trained Stage1-v5 KV loss and gradients"`. Do not claim whole-project completion yet. Write the task report beside the generated brief with RED/GREEN commands/output paths, exact tests, source hashes/commit, native-versus-substituted boundaries, unresolved concerns and GPU use zero.

## Controller review and final verification

- [ ] Generate the task diff package from recorded BASE through actual HEAD; obtain an independent task reviewer with both specification and quality verdicts. Use SDD's mandatory report handling and fix loop, rereading its relevant reference before each round. Implementer/fixer agents own code fixes; root never substitutes self-review or implements fixes.
- [ ] Obtain the required broad final whole-branch review on the most capable available model, using the historical fork and full diff, plus approval/spec/plan and evidence. Preserve unresolved findings exactly. Use the skill's single final fix-wave/scoped-review rules if needed. No reviewer GPU use or redundant broad test reruns.
- [ ] Before completion verification, invoke `verification-before-completion` and `finishing-a-development-branch`; the latter applies to the implemented diagnostic branch. Select its project full **CPU** suite below once on final reviewed source. The actual GPU diagnostic is the distinct original-question witness; the full CPU suite cannot replace it. Read the full log and exit status; report every failure, skip, deselection and warning. Do not present finishing options or a completion claim on a red required witness.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q -m 'not gpu and not distributed' --basetemp=checkpoints/kv-stage1-training-parity-20261024T123825Z/final-cpu-<UTC>
```

- [ ] Confirm final deliverable files are committed and identical to reviewed/tested hashes; `git diff 07eb774281925866502e72bdfcaa4df54073103a --` must show no native-source or existing-test changes. Record exact commit, module hashes and commands before GPU. Persist minimal approvals, agent IDs, review verdicts/fixes, ledger rulings and verification evidence to this run's results before removing only this plan's disposable SDD workspace. Keep the worktree and experiment evidence.

## Root-only trained-model experiment

- [ ] Run the reviewed CLI with `--prepare-only` on CPU first. Inspect the chosen real row and complete image/processor identities; verify all input and output path policies. Do not substitute a row if none qualifies. Recheck actual GPU resources and no foreign workload conflict before launch.
- [ ] Record the additional budget ledger at zero or its actual prior consumption; never reset it. Run one owned process, with a root-managed deadline whose total external wall time is at most the remaining additional300s. Include interpreter startup/imports/loading/authentication/report writing. A shell timeout may terminate **only that owned process/group**. Record partial logs and external termination if no JSON report is completed; no general retry/recovery layer.

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python scripts/probe_stage1_training_kv.py --checkpoint checkpoints/stage1-v5-mn-g16-e3-s648-20261014-150317/checkpoint-648 --base checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct --data-path data/TRAIN.jsonl --image-root data/images_comt --run-id kv-stage1-training-parity-20261024T123825Z --device cuda:0
```

The controller wraps this exact command with its process deadline and log redirection. CPU preparation and GPU execution share the same frozen prepared-input identity; GPU mode validates that prepared record without overwriting it, then creates the previously absent report. Refuse any mismatch or existing report.

- [ ] Inspect actual U/A, A/A-repeat and A/B reports, decoder call witnesses and support changes before interpreting error magnitudes. Report raw local loss differences and per-parameter/global gradient errors with dtype/scope. Timing is standalone diagnostic wall time, not a training throughput comparison. State if any intervention remains unexercised or any prerequisite/GPU step is incomplete.
- [ ] Report the measured result promptly. Preserve all failures and actual budget use. No precision fix, full training run, merge, push, PR or branch integration is authorized by this plan. If implementation and all required witnesses are green, present the finishing skill's applicable integration menu for the human to decide; otherwise retain the branch and report the unmet witness. Run the required ntfy summary hook before the final response.

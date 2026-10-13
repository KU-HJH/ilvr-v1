# DIL-BCE-ORIG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the approved `dil_bce_orig` Stage 2 objective with a separately loaded frozen reference, exact student-history replay, optional OFF entropy/mass, and reproducible save/resume behavior.

**Architecture:** Capture a detached clean-history packet from the student's existing latent rollout and replay it through an independently loaded reference. New tensor helpers and an orig-only Trainer reducer implement the objective while preserving legacy branches. A run-level reference snapshot binds both workers and saved objective identity to the same effective policy.

**Tech Stack:** Repository Python 3.11 environment, patched Transformers/Qwen2.5-VL, PyTorch, PEFT/DoRA, pytest, CPU Gloo, existing ZeRO-2 multi-node coordinator.

**Spec:** [Design commitment v1](../specs/2026-10-13-dil-bce-orig-design.md), approved by the user's `계속` after the written-spec approval request. Approval binds commit `112b09c230250e3e3adcaec77ea6171701b2032d`, spec SHA-256 `065e24b0d953eceddfa173ff842d3e8259f57861d233e2cf3fa72297ed936f52`. Its historical pending-status text is superseded by the approval event in [the workflow record](../2026-10-13-dil-bce-orig-workflow.md), not by editing the approved bytes.

## Global Constraints

- Worktree: `/root/workspace/ILVR/.worktrees/dil-bce-orig`; branch: `dil-bce-orig`; implementation base: `00c212cb598d88e5cfe22a40d11d320f04c609d5`.
- Preserve `documents/dil-bce-orig-v1.md` byte-for-byte; its SHA-256 is `9b26bd40ec40dbf882e7254f0b650595538d2432e7ace2fe363dbe8ecaf1eed1`.
- Canonical aliases: only `dil-bce-orig` and `dil_bce_orig` map to `dil_bce_orig`; preserve the default selector and every legacy selector.
- Orig: beta=1, margin=0, margin_mode=logit, chosen_mode=gt, chosen_reduction=token_mean, rejected_mode=null, legacy lambda_entropy=0, lambda_topk_chosen_mass=0.
- Fresh defaults: CE=0.1, preference=1, topk_entropy=0, mass=0, entropy selection=None, topk_include_gt=false, topk_mode=independent, loss_chunk_positions=128.
- Reference defaults: frozen_clean / shared_clean_history / full_vocab; fresh orig requires an explicit reference checkpoint.
- Entropy requires explicit `2 <= k < V`; positive mass requires positive entropy; a valid k with entropy off is inactive.
- All BCE reference vocabulary entries, including GT, participate; no probability floor, truncation, extra 1/V, extra 1/2 or old rejected BCE.
- Keep detached reference/selectors and live student latent gradients; entropy keeps q and logq live; mass keeps OFF support and complement logits live.
- Keep V-only Stage2B context_preference, clean_prefix, independent branches, shared prefix, visible supervised boundaries and original-image access.
- Keep immediate_text for fresh runs and the saved/explicit target scope for resumes; reject unsupported image_embedding_replacement, image_gap, packed, FSDP and ZeRO-3 orig profiles.
- Keep SDPA language attention, FlashAttention2 vision, Flex sidecars/query budget1024, native checkpointing off, latent and sidecar checkpointing on, ZeRO-2.
- Preserve DIL identity schema1 bytes and PEFT layout4; orig alone uses objective identity schema2 and formula_domain=teacher_forced_token.
- Reference snapshot: `checkpoints/<run-id>/reference_snapshot/`; content fingerprint is independent of its location; no mutable source hardlinks.
- Generated model-bearing fixtures belong under `checkpoints/`; output under `results/`; execution logs under `logs/`; preserve existing storage symlinks.
- Use `./venv/ilvr-venv/bin/python`; do not install dependencies, import Reference/ repositories or change the patched dependency baseline.
- FP32 helper comparisons start at atol=1e-5/rtol=1e-4; independent FP64 checks at atol=1e-10/rtol=1e-8; justify model/BF16 tolerances separately.
- The implementation plan does not authorize training, evaluation or a production smoke launch. Any later training requires explicit authorization and existing two-node/four-rank coordination.
- A later launch must re-read canonical AGENTS.md/training-defaults.md, enforce micro1/GAS4/global16/LR5e-5/seed42/warmup10/BF16 and the complete applicable profile after sourcing settings.
- Do not claim improvement in accuracy, production capacity, full runtime equivalence or completed GPU acceptance from CPU tests.

## Review Focus

1. Equal vocabulary sizes with swapped token IDs must fail before replay: Task 4 `test_same_size_swapped_token_mapping_rejected`.
2. Reused paths with changed weights must fail, while byte-identical relocation works: Task 4 `test_snapshot_mutation_and_content_identical_relocation`.
3. Shell defaults must not turn omitted resume options into explicit overrides or hide conflicting aliases: Task 6 `test_worker_preserves_orig_omission_and_alias_conflicts`.
4. Padding must contribute neither count nor loss while participating in the same metric collectives: Task 5 `test_compute_loss_padding_participates_without_counting`.
5. Chunk checkpointing and ragged entropy must preserve the mass complement gradient: Task 1 `test_checkpointed_ragged_mass_keeps_complement_gradient`.

---

## Execution and evidence conventions

This is one coupled feature, with six independently reviewable deliverables. A math helper, replay API or snapshot loader can be rejected separately; none is a separate user-facing project. Tasks run in order. Read both the spec and the applicable execution skill before implementation. Read the TDD skill and its `writing-good-tests.md` before writing tests. The authorized throwaway probe is premise evidence, not production code to copy ahead of RED.

The user must review this plan and select Subagent-driven or Native. Preserve that selection in the workflow record. Each task below has focused RED/GREEN witnesses; the selected skill supplies its actual review/commit ordering and any additional mandatory checks. For Subagent-driven execution, dispatch the required fresh implementer and independent review for every task, including fixes. For Native, root implements and a distinct final reviewer assesses specification and quality. Never replace independent review with self-review or test assertions.

For CPU tests, establish this shell context once per implementation session:

```bash
export PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
ORIG_RUN_ID="dil-bce-orig-$(date -u +%Y%m%dt%H%M%Sz)"
export ORIG_RUN_ID
mkdir -p "logs/$ORIG_RUN_ID" "results/$ORIG_RUN_ID" "checkpoints/$ORIG_RUN_ID"
set -o pipefail
```

Record command, exit status, complete output, reviewed diff/commit and actual reviewer IDs/verdicts in the existing workflow record. RED means an assertion exposing missing/wrong behavior, not a syntax error or failed import. For new modules, use an assertion on `importlib.util.find_spec` inside the test before importing. RED/GREEN commands below set unique basetemp locations; do not reuse a directory containing useful saved artifacts. Tests may create tiny local model files there, never production checkpoints. A CPU Gloo unit test or one tiny optimizer-step fixture is not a training launch.

## File map

| Task | Files and responsibility |
| --- | --- |
| 1 | New `src/step_preference/dil_bce_orig_loss.py`, `topk_entropy_loss.py`: numerical kernels, detached candidate supports, chunk/checkpoint evaluation |
| 2 | Existing `src/step_preference/config.py`, `dil_config.py`, `dil_identity.py`, `src/utils_deepseed.py`: orig-only settings, validation, explicit-option provenance and schema2 |
| 3 | New `transformers/src/transformers/models/qwen2_5_vl/stage2_reference_replay.py`; existing `modeling_qwen2_5_vl.py`: capture and reference-owned replay |
| 4 | New `src/stage2/reference_artifact.py`, `frozen_reference.py`; narrow existing `checkpointing.py`/`peft.py` changes only where necessary: immutable effective policy and separate owner |
| 5 | New `src/step_preference/dil_bce_orig_reduction.py`; existing `entropy_selection.py`, `src/stage2/trainer.py`, Qwen `stage2_shared_prefix.py`: scheduling, token statistics, global step reduction and metrics |
| 6 | Existing `src/main.py`, `src/multinode.py`, `src/multinode_runtime.py`, `run_stage2_training.sh`, `scripts/stage2/dil_bce_profile.sh`: composition, snapshot preparation and worker forwarding |
| 6 | New `src/stage2/dil_bce_orig_probe.py`, `docs/training/dil-bce-orig.md`: bounded opt-in production witness and usage; reuse existing observation/transport helpers |

Test paths and any necessary additional modifications are listed in each task. Do not refactor unrelated legacy modules. New-file locations abbreviated in this table are expanded in the task that owns them.

### Task 1: Exact token BCE and optional entropy/mass kernels

**Files:** Create `src/step_preference/dil_bce_orig_loss.py`, `src/step_preference/topk_entropy_loss.py`; create `tests/step_preference/test_dil_bce_orig_loss.py`, `tests/step_preference/test_topk_entropy_loss.py`. Read `dil_topk_loss.py::_group_mass_terms` and reuse its stable computation without changing legacy semantics.

**Interfaces:** All logits are Tensor[P,V], GT is integral Tensor[P]. Dataclasses below contain Tensor[P] fields unless stated otherwise.

- Produces `OrigTokenComponents(chosen, reference, policy_gt_logp, reference_gt_logp, gt_logratio, reference_prob_sum_error)`.
- Produces `dil_bce_orig_components(clean_logits: Tensor, reference_logits: Tensor, target_ids: Tensor) -> OrigTokenComponents`.
- Produces `orig_components_chunked(clean_logits: Tensor, reference_logits: Tensor, target_ids: Tensor, *, chunk_positions: int = 128, checkpoint: bool = True) -> OrigTokenComponents`.
- Produces `CandidateSupport(ids: Tensor[P,Kmax], valid: Tensor[P,Kmax], gt_coverage: Tensor[P])`, with long IDs and boolean valid mask.
- Produces `build_entropy_support(clean_logits: Tensor, off_logits: Tensor, target_ids: Tensor, *, k: int, mode: str, include_gt: bool) -> CandidateSupport`.
- Produces `TopkEntropyComponents(kl, entropy, normalized_entropy, support_size)` and `topk_entropy_components(off_logits: Tensor, support: CandidateSupport) -> TopkEntropyComponents`.
- Produces `MassTokenComponents(loss, clean_mass, off_mass, mass_abs_gap)` and `entropy_support_mass_components(clean_logits: Tensor, off_logits: Tensor, support: CandidateSupport) -> MassTokenComponents`.
- Consumes only PyTorch and the existing stable group-mass helper; no config, Trainer, model or artifact imports.

- [ ] **Step 1: Write the BCE oracle tests.** Use literal FP64 expressions independent of production helpers. Include the following named assertions, population gradient at p=a, detached reference and input validation for shape/device/finite values/GT bounds/empty P.

```python
# test_full_vocab_literal_oracle: p=(.5,.3,.2), mu=(.2,.5,.3), y=0
assert_close(chosen, 0.336472236621213)
assert_close(reference, 0.638802095451739)
assert_close(chosen + reference, 0.975274332072952)
# test_equal_policy_reference_has_nonzero_single_gt_gradient
assert_close(chosen, math.log(2)); assert_close(reference, math.log(2))
torch.testing.assert_close(grad, (mu - onehot) / 2, atol=1e-5, rtol=1e-4)
# test_wrong_expectations_and_mean_score_are_distinguishable
assert not torch.allclose(expected, gt_only)
assert not torch.allclose(expected, truncated_reference)
assert not torch.allclose(expected_gradient, live_policy_weighted_gradient)
assert not torch.allclose(token_mean, softplus_of_mean_score)
```

- [ ] **Step 2: Observe BCE RED.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_loss.py --basetemp="checkpoints/$ORIG_RUN_ID/t1-bce-red"`. Expect missing-kernel assertions; inspect every failure.
- [ ] **Step 3: Implement the BCE interfaces.** Use FP32 log_softmax/softplus, detach reference before normalization, sum the complete reference vocabulary. Concatenate token results across non-reentrant checkpointed position chunks; no chunk-mean averaging. Return student-connected empty values for P=0.
- [ ] **Step 4: Write entropy/mass and chunk tests.** Implement `test_uniform_and_saturated_entropy`, `test_live_q_analytic_gradient`, `test_candidate_modes_and_ragged_gt_union`, `test_entropy_only_support_gradient`, `test_checkpointed_ragged_mass_keeps_complement_gradient`, `test_chunk_1_128_all_loss_and_gradient`, and malformed/duplicate candidate tests. Finite differences hold reference logits and selected IDs fixed.

```python
# uniform valid support, and separately a nearly one-hot valid support
assert_close(uniform_kl, 0); assert_close(uniform_grad, 0)
assert_close(saturated_kl, math.log(n)); assert torch.isfinite(saturated_grad).all()
# nonuniform: analytic dU/dz_j = q_j * (log(q_j) + H)
assert_close(actual_grad[support], q * (logq + entropy))
assert actual_grad.abs().sum() > 0
assert_close(detached_q_variant_grad, 0)
# test_checkpointed_ragged_mass_keeps_complement_gradient
assert_close(entropy_grad[complement], 0)
assert mass_grad[complement].abs().sum() > 0
assert_close(full_support_mass, 0); assert_close(full_support_mass_grad, 0)
assert torch.isfinite(ragged_loss) and torch.isfinite(ragged_grad).all()
```

- [ ] **Step 5: Observe entropy/chunk RED.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_loss.py tests/step_preference/test_topk_entropy_loss.py --basetemp="checkpoints/$ORIG_RUN_ID/t1-entropy-red"`; expect assertions for missing entropy APIs and any unimplemented checkpoint requirements, not infrastructure errors.
- [ ] **Step 6: Implement support, entropy and mass interfaces.** Detach candidate construction, deduplicate optional GT, mask invalid entries before entropy multiplication, and preserve q/logq derivatives. Adapt valid support to the existing stable full-normalizer mass calculation. Validate k against V, support uniqueness/range and all alignments; avoid probability floors and loss clamping.
- [ ] **Step 7: Verify GREEN and legacy numerical regressions.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_loss.py tests/step_preference/test_topk_entropy_loss.py tests/step_preference/test_dil_bce_loss.py tests/step_preference/test_dil_topk_loss.py --basetemp="checkpoints/$ORIG_RUN_ID/t1-green"`. Require zero failures; chunk 1/128/all outputs and gradients agree, including a non-divisible P=259 fixture. Inspect saved tensors to ensure loss checkpointing does not retain each full FP32 intermediate; do not claim total O(PV) storage disappeared.
- [ ] **Step 8: Record the task revision.** Commit the four Task 1 files with message `feat: add exact orig BCE and entropy kernels`; perform the selected execution method's task review/fix gate against that diff and record its real result before proceeding.

### Task 2: Orig configuration, CLI provenance and objective identity

**Files:** Modify `src/step_preference/config.py`, `src/step_preference/dil_config.py`, `src/step_preference/dil_identity.py`, `src/utils_deepseed.py`; create `tests/step_preference/test_dil_bce_orig_config.py`, `tests/step_preference/test_dil_bce_orig_identity.py`. Extend `tests/step_preference/test_resume_loss_resolution.py` only for shared omitted-option behavior.

**Interfaces:** Extend `StepPreferenceConfig` with fields `reference_mode: str | None`, `reference_checkpoint: str | None`, `reference_context_mode: str | None`, `reference_expectation: str | None`, `lambda_topk_entropy: float | None`, `topk_mode: str | None`, `topk_include_gt: bool | None`, `loss_chunk_positions: int | None`, `margin_mode: str | None`, `chosen_mode: str | None`, `chosen_reduction: str | None`, `rejected_mode: str | None`, `lambda_topk_chosen_mass: float | None`, `reference_manifest: str | None`, `reference_identity: dict[str, object] | None`. Resolve omitted fresh values only after saved settings are considered. Keep `explicit_fields: frozenset[str]` separately from resolved values, with canonical field names captured by the real CLI resolver; direct config construction treats supplied non-None values as explicit and uses this set to distinguish an explicit null.

- Produces `normalize_preference_loss_type(value: str) -> str` in `dil_config.py`, with the two explicit aliases only.
- Produces `orig_requires_off(config: StepPreferenceConfig) -> bool` in `dil_config.py`; selection=None/100 is inactive, so it needs no OFF-only ranking.
- Produces `bind_orig_reference_identity(config: StepPreferenceConfig, *, fingerprint: str, compatibility: dict[str, object], manifest_path: str) -> StepPreferenceConfig` in `dil_identity.py`; returns a dataclass replacement, validating the manifest-independent identity `{fingerprint, compatibility}`.
- Extends existing `build_dil_identity`, `read_dil_identity`, `validate_dil_resume` and `attach_dil_identity`; schema2 construction requires a bound reference identity. Config validation alone may precede artifact binding.
- CLI new fields use `--step_preference_<field>`; `--step_preference_reference_manifest` is a validated coordinator input. Preserve existing topk/mass storage, adding only public aliases `--preference_topk_k` and `--lambda_topk_mass`.
- `reference_identity` is derived from verified artifacts, never accepted as an arbitrary CLI JSON value. Reject unequal explicit aliases; equal duplicate values are allowed. A literal `null` rejected_mode maps to None while retaining explicit provenance.

- [ ] **Step 1: Write config/identity tests.** Parameterize every constraint in spec sections 3/7/8, aliases and legacy noninterference. Capture existing schema1 representative serialized values before adding schema2. Test config validation before binding and refusal to serialize an unbound orig identity.

```python
# test_orig_aliases_and_defaults
assert hyphen.preference_loss_type == underscore.preference_loss_type == 'dil_bce_orig'
assert (fresh.beta, fresh.lambda_ce, fresh.lambda_preference) == (1., .1, 1.)
assert (fresh.lambda_topk_entropy, fresh.lambda_topk_mass) == (0., 0.)
assert fresh.loss_chunk_positions == 128 and fresh.topk_include_gt is False
# test_resume_omission_precedes_defaults_and_explicit_conflicts
assert resumed.lambda_topk_entropy == saved['lambda_topk_entropy']
with pytest.raises(ValueError): resolve_with_conflicting_explicit_aliases()
# test_orig_identity_v2_preserves_legacy_v1
assert identity['schema_version'] == 2
assert identity['formula_domain'] == 'teacher_forced_token'
assert legacy_serialized_after == legacy_serialized_before
```

Fixture helpers in these snippets are test-local, not new production APIs. Tests must exercise the existing `parse_args` and config resolver rather than a parallel parser. Assert the three exact formula versions in spec section 8 and identity rejection for altered reduction/reference/coefficients.

- [ ] **Step 2: Observe RED.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_config.py tests/step_preference/test_dil_bce_orig_identity.py --basetemp="checkpoints/$ORIG_RUN_ID/t2-red"`; expect missing orig selector/settings/identity behavior.
- [ ] **Step 3: Implement canonical settings and validation.** Route orig before legacy DIL validation. Preserve omitted CLI values through config resolution, validate finite weights and integer chunk size, and add gap/intersection only for orig. Reject unavailable intervention/backend and unsupported formula options explicitly. Keep legacy `lambda_entropy` defaults and explicit beta behavior intact.
- [ ] **Step 4: Implement identity binding and schema2 save/read/resume checks.** Keep schema1 tables/serialization unchanged. Encode all formula/selection/execution settings plus compatibility and content fingerprint; record `microbatch_global_original_step_mean_occupancy_fixed_gas_v1` as orig reduction and `token_mean` diagnostic aggregation. Manifest location is a separate checkpoint reference, not part of the path-independent content fingerprint. Same-objective validation runs before any parent Trainer restore.
- [ ] **Step 5: Verify GREEN and defaults regressions.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_config.py tests/step_preference/test_dil_bce_orig_identity.py tests/step_preference/test_dil_config.py tests/step_preference/test_dil_identity.py tests/step_preference/test_loss_defaults.py tests/step_preference/test_resume_loss_resolution.py --basetemp="checkpoints/$ORIG_RUN_ID/t2-green"`. Require zero failures and unchanged serialized legacy identities.
- [ ] **Step 6: Record the task revision.** Commit only Task 2 changes with `feat: define orig objective configuration and identity`; complete the selected method's review gate and record verdict/fixes.

### Task 3: Capture actual student history and replay reference-owned inputs

**Files:** Create `transformers/src/transformers/models/qwen2_5_vl/stage2_reference_replay.py`; modify `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`; create `tests/step_preference/test_dil_bce_orig_replay.py`. Reuse `tests/step_preference/insertion_helpers.py` fixtures and the existing PEFT wrapper resolver.

**Interfaces:** New module defines immutable dataclasses:

- `CleanHistoryRequest(sample_id: str, predictor_positions: tuple[int, ...], target_ids: tuple[int, ...], step_ids: tuple[int, ...], original_step_ids: tuple[int, ...])`; the first three position-associated tuples have equal length. Step IDs refer to original steps, including original steps with no selected tokens.
- `CleanHistoryReplay(request: CleanHistoryRequest, input_ids: Tensor, attention_mask: Tensor, position_ids: Tensor, pixel_values: Tensor | None, image_grid_thw: Tensor | None, latent_positions: Tensor, latent_values: Tensor)`; all tensors are owned detached copies. `latent_positions` holds batch/sequence index pairs; positions are the actual full model positions, not reconstructed text offsets. Unsupported raw modalities fail explicitly.
- `capture_clean_history(request: CleanHistoryRequest, *, input_ids: Tensor, attention_mask: Tensor, position_ids: Tensor, pixel_values: Tensor | None, image_grid_thw: Tensor | None, final_inputs_embeds: Tensor, latent_positions: Tensor) -> CleanHistoryReplay`.
- `replay_clean_history(model: torch.nn.Module, packet: CleanHistoryReplay) -> Tensor` returns detached Tensor[P,V]. It resolves native/PEFT Qwen, uses that model's embedding/vision preparation and full decoder, and does not call latent rollout.
- Patched model forward accepts `stage2_clean_history_request: CleanHistoryRequest | None = None`; `Qwen2_5_VLCausalLMOutputWithPast.stage2_clean_history` is the packet or None. Legacy callers incur no packet allocation.

- [ ] **Step 1: Write replay runtime tests.** Hook actual tiny model calls, including sparse-row PEFT. Build a literal reference embedding/decoder evaluation separately from the new replay helper. Test two latent blocks, image inputs, actual multimodal positions, predictor/GT alignment errors, empty P and live student backward.

```python
# test_reference_owned_replay_matches_literal_decoder
torch.testing.assert_close(replay_logits, literal_reference_logits)
assert reference_decoder_calls == 1 and reference_rollout_calls == 0
assert torch.equal(observed_latent_values, captured_student_latent_values)
assert not torch.allclose(observed_ordinary_embeddings, student_ordinary_embeddings)
# test_equal_weights_eval_identity_and_wrong_replay_negative_controls
torch.testing.assert_close(equal_weight_eval_logits, student_eval_logits)
assert not torch.allclose(all_student_embeddings_logits, literal_reference_logits)
assert not torch.allclose(reference_rerolled_latents, captured_student_latent_values)
# test_replay_detaches_input_but_preserves_student_graph
assert not replay_logits.requires_grad and reference_input_leaf.grad is None
assert student_latent_grad.abs().sum() > 0
assert packet.latent_values.data_ptr() != live_student_latent_values.data_ptr()
```

- [ ] **Step 2: Observe RED.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_replay.py --basetemp="checkpoints/$ORIG_RUN_ID/t3-red"`; expect absent packet/capture/replay assertions.
- [ ] **Step 3: Implement capture at the real clean-forward boundary.** Capture after live latent rollout with actual full position IDs, before sidecar scoring. Do not replace/detach live student inputs or change clean CE/head ordering. Reject malformed position/target/step/latent alignments before reference decoding.
- [ ] **Step 4: Implement reference-owned replay.** Under no_grad, rebuild ordinary tokens, START/END and images with reference weights; overwrite only latent-body positions with packet values; run its full decoder once with its own attention/position state. Never pass student KV or run reference latent rollout; select full-vocabulary logits at packet predictors. Validate empty-P behavior without inventing a different collective schedule.
- [ ] **Step 5: Verify GREEN and model regressions.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_replay.py tests/step_preference/test_context_embedding_alignment.py tests/step_preference/test_clean_first_order.py --basetemp="checkpoints/$ORIG_RUN_ID/t3-green"`. Require native and PEFT literal-oracle agreement and preserved legacy clean-first ordering. Record dropout/eval and dtype assumptions; changed-reference equality with student is not an expected assertion.
- [ ] **Step 6: Record the task revision.** Commit Task 3 files with `feat: replay frozen reference on captured clean history`; complete the selected method's review gate and record its evidence.

### Task 4: Effective reference snapshot and independently frozen owner

**Files:** Create `src/stage2/reference_artifact.py`, `src/stage2/frozen_reference.py`, `tests/stage2/test_orig_reference_artifact.py`, `tests/stage2/test_orig_frozen_reference.py`; narrowly modify `src/stage2/checkpointing.py`/`peft.py` only if the tested existing loaders need explicit source-provenance versus local-copy path separation.

**Interfaces:** `reference_artifact.py` defines `ReferenceSnapshot(manifest_path: Path, fingerprint: str, compatibility: dict[str, object], artifact_kind: str, component_paths: dict[str, Path])`.

- Produces `prepare_reference_snapshot(reference_checkpoint: Path, *, destination: Path) -> ReferenceSnapshot` for one coordinator writer. Resolve full effective base/adapter/processor from source metadata; fingerprint and copy required files once; publish the complete manifest last. Existing incompatible destination is an error, never overwrite.
- Produces `validate_reference_snapshot(manifest_path: Path, *, expected_fingerprint: str | None = None) -> ReferenceSnapshot`; validate relative regular-file entries, full content hashes and manifest identity before loading. No worker write or file-wait loop.
- Produces `validate_reference_compatibility(student_model: torch.nn.Module, student_processor: object, snapshot: ReferenceSnapshot, reference_model: torch.nn.Module, reference_processor: object) -> None`.
- `FrozenCleanReference` is a plain owner, not an nn.Module child of the student. `FrozenCleanReference.load(snapshot: ReferenceSnapshot, *, device: torch.device, dtype: torch.dtype) -> FrozenCleanReference`; attributes `model`, `processor`, `snapshot`; methods `assert_frozen() -> None`, `score(packet: CleanHistoryReplay) -> Tensor` using Task 3's replay API.
- Consumes existing artifact inspectors, full/native loader, Stage1 DoRA loader, Stage2 `load_stage2_model_for_inference`, sparse token-row restoration, canonical artifact guards and streaming SHA-256 pattern. Adapter settings come only from its artifact. Do not load optimizer state or clone/disable the student adapter.

- [ ] **Step 1: Write snapshot and independent-loader tests.** Reuse tiny native/original-base/Stage1-DoRA/Stage2-sparse fixture construction from `tests/stage2/test_stage1_checkpoint_handoff.py`. Test missing shards/indices/base/sparse rows, altered adapter config, incomplete generic adapters, escaping manifest entries, and a modified byte at the same source path. Assert copied-file inode differs from source, source mutation after preparation cannot change the snapshot, and no files are emitted under results.

```python
# test_snapshot_mutation_and_content_identical_relocation
assert relocated.fingerprint == original.fingerprint
with pytest.raises(ValueError): validate_mutated_snapshot()
assert source_weight.stat().st_ino != snapshot_weight.stat().st_ino
# test_same_size_swapped_token_mapping_rejected
assert len(student_tokenizer) == len(reference_tokenizer)
with pytest.raises(ValueError, match='token'): validate_swapped_mapping()
# test_reference_owner_excludes_optimizer_and_freezes_all_effective_weights
assert reference_parameter_ids.isdisjoint(student_parameter_ids | optimizer_parameter_ids)
assert not any(p.requires_grad for p in reference.model.parameters())
assert not any(m.training for m in reference.model.modules())
assert all(torch.equal(before[k], after[k]) for k in before)  # parameters AND buffers
```

- [ ] **Step 2: Observe RED.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/stage2/test_orig_reference_artifact.py tests/stage2/test_orig_frozen_reference.py --basetemp="checkpoints/$ORIG_RUN_ID/t4-red"`; expect missing snapshot/owner behavior.
- [ ] **Step 3: Implement manifest preparation and verification.** Include every effective weight/shard/index/config/tokenizer/processor/sparse-row component and replay compatibility contract. Preserve original provenance metadata bytes, resolving copied load locations through manifest component paths. Do not silently rewrite PEFT metadata to bypass compatibility. Fingerprint canonical relative content records, excluding source/destination location strings. Destination must be inside the current canonical checkpoints tree.
- [ ] **Step 4: Implement independent loading and compatibility checks.** Load each supported artifact on the requested local device, freeze/eval the entire owner and validate actual token mapping, special IDs, latent width/layout, vocabulary, architecture, preprocessing and position contract against the student. Check storage aliasing, not only Python parameter identity. Reference replay reasserts eval/frozen invariants after Trainer mode transitions.
- [ ] **Step 5: Verify GREEN and real artifact regressions.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/stage2/test_orig_reference_artifact.py tests/stage2/test_orig_frozen_reference.py tests/stage2/test_stage1_checkpoint_handoff.py tests/stage2/test_dil_resume_entry.py --basetemp="checkpoints/$ORIG_RUN_ID/t4-green"`. Require real saved tiny artifacts to reload and replay. Include one tiny student optimizer update and changed-latent replay: reference weights/buffers stay exact while conditional reference logits may change. No production artifact is touched.
- [ ] **Step 6: Record the task revision.** Commit Task 4 changes with `feat: preserve and load immutable orig reference policies`; complete the selected method's review gate and retain any unresolved loader limitations explicitly.

### Task 5: Orig scheduling, sidecar statistics and distributed Trainer reduction

**Files:** Create `src/step_preference/dil_bce_orig_reduction.py`, `tests/step_preference/test_dil_bce_orig_trainer.py`, `tests/step_preference/test_dil_bce_orig_distributed.py`; modify `src/step_preference/entropy_selection.py`, `src/step_preference/topk_entropy_loss.py`, `src/stage2/trainer.py`, `transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py`, and `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`. Extend the Task 3 output schema with orig statistics, never persistent full OFF logits.

**Interfaces:** Consumes Tasks 1-4. `Stage2Trainer` accepts `frozen_clean_reference: FrozenCleanReference | None = None`; it is required for orig and remains outside student wrappers/optimizer/state_dict.

- Produces `select_orig_entropy_masks(clean_entropies: tuple[Tensor, ...], off_entropies: tuple[Tensor, ...] | None, *, source: str, percent: float | None) -> tuple[Tensor, ...]` in `entropy_selection.py`. Reuse stable ceil selection; gap=abs(Hoff-Hclean); intersection=independent masks AND, no refill.
- Produces `OrigSidecarStats(off_entropy: Tensor, entropy: TopkEntropyComponents | None, mass: MassTokenComponents | None, gt_coverage: Tensor | None)` in `topk_entropy_loss.py`; fields align exactly with requested predictors. The existing sidecar checkpoint returns tuples of tensor statistics, assembled into this type outside its checkpoint closure.
- Produces `OrigReduction(loss: Tensor, metrics: dict[str, Tensor])` and `reduce_orig_objective(components: OrigTokenComponents, sidecar: OrigSidecarStats | None, *, step_ids: Tensor, original_step_ids: tuple[int, ...], selected: Tensor, padding: bool, graph_anchor: Tensor, config: StepPreferenceConfig) -> OrigReduction` in `dil_bce_orig_reduction.py`. It excludes CE; Trainer adds existing clean CE exactly once.
- The reducer obtains W/counts from initialized torch.distributed, or W=1 for unit tests. It reduces real-slot/original-step counts and detached metric sums in the same order on every rank, including P=0 and padding. `graph_anchor` supplies connected zeros.

- [ ] **Step 1: Write scheduling, weighting and diagnostic tests.** Traverse actual `Stage2Trainer.compute_loss`, actual patched clean model and sidecar hooks where claims concern calls. Parameterize entropy on/off and clean/voff/gap/intersection selection, stable ties, intersection empty steps, CE0/0.1 and legacy selection. Assert clean reference is always used, OFF is absent when unnecessary, no extra entropy-only decoder call, and clean projection/CE precedes sidecars.

```python
# test_orig_total_has_no_legacy_rejected_term
assert_close(total, .1 * ce + pref_weight * (chosen + reference)
             + entropy_weight * entropy + mass_weight * mass)
assert 'loss/dil_bce_orig_reference' in metrics
assert 'loss/dil_bce_negative_gt' not in metrics
# test_selection_keeps_original_empty_step_denominator
assert_close(two_original_steps_one_selected_loss, selected_step_loss / 2)
assert empty_selection_loss.requires_grad
# test_compute_loss_padding_participates_without_counting
assert padded_local_sum == 0 and padded_step_count == 0 and padded_real_slots == 0
assert rank_collective_sequences.count(rank_collective_sequences[0]) == 4
```

- [ ] **Step 2: Observe scheduling RED.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_trainer.py --basetemp="checkpoints/$ORIG_RUN_ID/t5-trainer-red"`; expect absent orig dispatch/owner/statistics behavior.
- [ ] **Step 3: Implement selection and the checkpointed sidecar statistics.** Call Task 1 helpers inside the existing sidecar checkpoint, returning only aligned scalar token statistics. Compute OFF full-vocabulary entropy only when required for selection/diagnostics. Sync support reads detached clean logits; mass clean targets are detached. Do not modify legacy RF/topk/reverse/GT outputs.
- [ ] **Step 4: Implement orig Trainer dispatch and raw reduction.** Branch before generic non-DPO RF handling, request capture in the actual clean call, replay the separate owner, evaluate BCE by chunks, and combine selected per-token components into original-step sums. Read `_stage2_padding_active` because compute_loss removes the input sample weight. Count empty-selected original steps; exclude padding from both sums and counts; preserve live zeros.
- [ ] **Step 5: Write CPU distributed and actual-parent tail tests.** Use four CPU Gloo processes and the real compute_loss wrapper with test-local tiny tensors/models; compare gradients to a separately assembled global expression. Reuse the actual `training_step` fixture pattern from `tests/stage2/test_reference_random_sampler.py` for the parent's fixed GAS path, without launching a training job. Tests cover equal/unequal step counts, all-empty, one empty real sample, padding, complete GAS4, partial GAS4 and regrouped microbatches.

```python
# test_unequal_steps_are_global_step_mean_not_sample_mean
assert_close(raw_mean, 5 / 8)  # steps=(1,1,1,5), losses=(0,0,0,1)
assert not math.isclose(raw_mean, 1 / 4)
# test_actual_parent_gas4_tail_uses_configured_denominator
assert_close(actual_tail_gradient, literal_three_real_global_step_gradient * .75 / 4)
assert_close(actual_full_gradient, sum(literal_microbatch_gradients) / 4)
# test_regrouping_is_microbatch_then_accumulation_not_window_step_mean
assert not torch.allclose(microbatch_grouped_oracle, optimizer_window_step_oracle)
torch.testing.assert_close(actual_gradient, microbatch_grouped_oracle)
```

- [ ] **Step 6: Observe distributed RED.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_distributed.py --basetemp="checkpoints/$ORIG_RUN_ID/t5-distributed-red"`; expect incorrect/missing global count and occupancy behavior. Configure test process timeouts so a missing collective yields an actionable failure.
- [ ] **Step 7: Implement distributed scaling and diagnostics.** Per synchronized microbatch compute N=sum original real steps, R=sum real slots, rho=R/W. Return rho*W/N*local_step_sum with N=0 connected zero, leaving the existing parent GAS division untouched. Execute orig metric collectives before/outside the legacy padding metric early-return. Implement every source section10.1 key: loss fields are raw global step means plus separately named occupancy-adjusted training components; token diagnostics use selected-token denominators. Disabled entropy metrics are explicit zeros, never legacy rejected keys. Persist the two aggregation conventions in schema2.
- [ ] **Step 8: Verify GREEN and Trainer regressions.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_dil_bce_orig_trainer.py tests/step_preference/test_dil_bce_orig_distributed.py tests/step_preference/test_dil_trainer.py tests/step_preference/test_dil_sidecar.py tests/step_preference/test_dil_entropy_integration.py tests/stage2/test_dil_reference_tail.py tests/stage2/test_reference_random_sampler.py --basetemp="checkpoints/$ORIG_RUN_ID/t5-green"`. Require no hangs/failures; compare actual CPU gradients and observed call counts, not only reducer algebra. Existing RF tail numerical expectations remain unchanged.
- [ ] **Step 9: Record the task revision.** Commit Task 5 files with `feat: integrate orig replay loss and distributed reduction`; complete the selected method's independent review/fix gate, emphasizing actual parent dispatch and collective ordering.

### Task 6: Entry points, snapshot lifecycle, resume and runtime witness handoff

**Files:** Modify `src/main.py`, `src/stage2/trainer.py` save/resume hooks, `src/multinode.py`, `src/multinode_runtime.py`, `run_stage2_training.sh`, `scripts/stage2/dil_bce_profile.sh`; create `tests/stage2/test_orig_entry_resume.py`, `tests/multinode/test_orig_forwarding.py`, `tests/multinode/test_orig_reference_transfer.py`, `src/stage2/dil_bce_orig_probe.py`, `tests/stage2/test_orig_probe.py`, `docs/training/dil-bce-orig.md`. Modify `src/multinode_artifacts.py` only for a narrow reusable path/transport primitive if its current public boundary requires it.

**Interfaces:** `Stage2TrainingComponents` adds `frozen_clean_reference: FrozenCleanReference | None`; `build_stage2_training_components` validates/binds the manifest and loads the owner before constructing the Trainer. Worker CLI/env fields mirror Task 2 with `STEP_PREFERENCE_<FIELD>` uppercase names. The shell forwards optional settings only when supplied; explicit false/zero/null remains explicit. The resolved snapshot path uses `--step_preference_reference_manifest`.

- Add coordinator-only `prepare-orig-reference --reference-checkpoint PATH --destination PATH` and `verify-orig-reference --manifest PATH --fingerprint SHA256` subcommands to the existing module dispatcher, returning JSON `{manifest_path, fingerprint}` from Task 4 APIs. They do not launch ranks.
- Add `SubprocessCoordinatorRunner.prepare_orig_reference(*, source: Path, destination: Path, expected_fingerprint: str | None = None) -> dict[int, str]` in `multinode_runtime.py`: prepare once, reuse existing guarded file transport to the second node, verify both and return fingerprints keyed by machine rank0/1. Do not pretend reference_snapshot is a numbered checkpoint-N or fabricate a global step for existing transfer validation.
- Add opt-in CLI `--step_preference_orig_probe_dir`/env `STEP_PREFERENCE_ORIG_PROBE_DIR` for the bounded witness. `OrigProbeObserver(trainer: Stage2Trainer, engine: torch.nn.Module, output_dir: Path)` and `check_orig_probe_records(records: list[dict[str, object]]) -> dict[str, object]` live in `dil_bce_orig_probe.py`; reuse DeepSpeed observation/hook isolation from `reference_sampler_probe.py`/`reproduction_probe.py`, not their historical GAS2/denominator8 oracle.
- The probe requires fresh orig, world4, two nodes, micro1/GAS4/ZeRO2, a 19-real-sample fixture with unequal original step counts (16 real first update; three real plus one padding slot in the single-microbatch second update). It rejects ordinary/resumed runs, extra updates and incompatible geometry. It records same-forward component gradients against the literal approved reduction, reference invariance, finite gradients, actual ranks/hosts, synchronized reference latency and peak allocated/reserved memory.

- [ ] **Step 1: Write entry and save/resume tests.** Use real saved tiny student/reference artifacts; mock only external process transport or attention backends where CPU requires it. Assert snapshot identity is bound before Trainer creation, owner stays outside optimizer/state_dict, checkpoint references one run-level manifest, omission restores saved values, same-content relocated sources work, and mismatch/missing identity fails before parent train/optimizer restore. Distinguish intercepted parent rejection tests from actual artifact round trips.
- [ ] **Step 2: Write both-worker/coordinator tests.** Reuse actual shell dry-run subprocesses from `tests/multinode/test_dil_forwarding.py`. Exercise every new option and aliases, paths containing spaces, sourced stale settings, resume omission, invalid image replacement and orig coefficients. Spy on coordinator call order and verify dry-run is read-only.

```python
# test_worker_preserves_orig_omission_and_alias_conflicts
assert local_resolved == remote_resolved
assert omitted_resume_fields.isdisjoint(explicit_worker_fields)
assert unequal_alias_worker_exit != 0
# test_snapshot_preparation_precedes_workers_and_preserves_command_digest
assert events.index('verify_local') < events.index('start_workers')
assert events.index('verify_remote') < events.index('start_workers')
assert fingerprints[0] == fingerprints[1]
assert dry_run_written_files == [] and dry_run_spawned_workers == []
assert published_command_digest == digest_of_actual_worker_command
# test_reference_transfer_rejects_incomplete_or_mismatched_payload
assert start_workers_calls == 0
```

- [ ] **Step 3: Observe entry/forwarding RED.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/stage2/test_orig_entry_resume.py tests/multinode/test_orig_forwarding.py tests/multinode/test_orig_reference_transfer.py --basetemp="checkpoints/$ORIG_RUN_ID/t6-entry-red"`; expect missing composition/snapshot/forwarding behavior.
- [ ] **Step 4: Implement main and checkpoint binding.** Resolve omitted resume settings from saved schema2 before fresh defaults, validate snapshot, load reference on local rank device, check compatibility, bind identity and pass owner to Trainer. Save only the run-relative manifest reference/content identity with student checkpoints. Explicit alternative reference source must match saved fingerprint. RF↔orig remains a fresh weight-only start; never restore a foreign objective's optimizer.
- [ ] **Step 5: Implement worker and pre-rank coordinator preparation.** Extend existing allowlists/value forwarding and the shell without changing old effective arguments. In `_training_main`, validate inputs/resources first, prepare/transport/verify snapshots before GPU worker launch, and build final worker commands/digests from resolved manifest paths. A dry-run reports planned preparation and effective arguments without copying model data. Keep node-local writer ownership in the coordinator so GPU ranks only read/verify. Adapt fresh collision preflight to the intentionally prepared run directory without weakening unrelated overwrite guards. Resume preserves the existing snapshot and fails if unavailable or different.
- [ ] **Step 6: Write the production-witness contract tests.** `test_orig_probe_rejects_wrong_geometry_and_incomplete_rank_records` supplies missing/duplicate hosts/ranks, historical GAS2, wrong tail counts, reference mutations and fabricated pass flags. `test_orig_probe_gradient_oracle_uses_step_counts_and_fixed_gas4` independently constructs nonuniform step-count component gradients and requires numeric agreement with actual recorded gradients; a legacy sample-mean oracle must fail. Assert collection uses real Trainer/DeepSpeed callbacks and records memory/timing only for the measured scope.
- [ ] **Step 7: Observe witness RED.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/stage2/test_orig_probe.py --basetemp="checkpoints/$ORIG_RUN_ID/t6-probe-red"`; expect missing observer/record validation behavior.
- [ ] **Step 8: Implement the bounded observer and documented invocation.** Keep the observer opt-in and off ordinary runs. Form an independent same-forward oracle from unscaled per-step components with realized reference/selectors detached; isolate oracle autograd from DeepSpeed reduction hooks using the existing proven mechanism. Compare actual accumulated/sharded gradients before optimizer mutation, including the 3/4 occupancy and 1/4 GAS tail factors. Record numeric errors and finite status, not just booleans; BF16 global-L2 <=2% follows the existing witness criterion and is reported separately from CPU tolerances. Do not add another launcher, retry framework or recovery service.
- [ ] **Step 9: Document exact usage and limits.** `docs/training/dil-bce-orig.md` covers required reference paths, CLI aliases/defaults, pure/entropy/GT-union/mass examples, gap/intersection definitions, immutable snapshot/resume behavior, conditional semi-gradient, raw versus occupancy metrics and unsupported profiles. Include the existing coordinator dry-run command and a separately authorized 19-sample production witness recipe. The recipe enforces the whole canonical profile after sourcing, uses the original base/reproduced Stage1 student initializer, records an explicit reference choice, and keeps model/log/output artifacts in their required trees. Do not launch it or imply a reference checkpoint has been chosen by the user.
- [ ] **Step 10: Verify GREEN and entry regressions.** Run `./venv/ilvr-venv/bin/python -m pytest -q tests/stage2/test_orig_entry_resume.py tests/multinode/test_orig_forwarding.py tests/multinode/test_orig_reference_transfer.py tests/stage2/test_orig_probe.py tests/multinode/test_dil_forwarding.py tests/multinode/test_stage2b_forwarding.py tests/stage2/test_dil_resume_entry.py tests/stage2/test_orig_reference_artifact.py --basetemp="checkpoints/$ORIG_RUN_ID/t6-green"`. Run `bash -n run_stage2_training.sh scripts/stage2/dil_bce_profile.sh`. Require both generated worker argv paths to resolve identical orig settings, with remote rank0/local rank1/world4 and no actual launch.
- [ ] **Step 11: Record the task revision.** Commit Task 6 changes with `feat: wire orig reference lifecycle through stage2 entrypoints`; complete the selected method's task review gate and record remaining production uncertainty without marking GPU acceptance complete.

## Requirements coverage and final gates

| Approved requirement | Owning task and concrete witness |
| --- | --- |
| Spec1/2 compatibility and source reconciliation | Tasks2/5/6 alias, legacy, gap/intersection and unsupported-backend tests |
| Spec3 and source D1-D8 | Task1 FP64/population/single-GT/detach/negative-control tests; Task5 token-before-step reduction |
| Source E1-E7 and N1-N3 | Task1 entropy/live-q/ragged/mass/full-support/chunk/dtype tests |
| Source N4 and I1/I10 | Tasks2/4/6 config, actual tokenizer/artifact, real CLI and legacy regressions |
| Spec4 and I2-I4 | Tasks3/4 native+PEFT replay, different embeddings, optimizer/mode/freeze/storage witnesses |
| Spec5 and I5/I6/I11 | Tasks1/3/5 saved tensors, real call ordering and OFF scheduling; Task6 production memory/cost observer |
| Spec6 and I7/I8 | Task5 Gloo actual compute_loss, padding metrics, parent GAS4/tail/regrouping; Task6 actual ZeRO observer |
| Spec7/8 and I9 | Tasks2/4/6 provenance, schema, snapshot, both-worker forwarding and parent-restore rejection |
| Spec9 and I12 | Task5 complete diagnostic values/denominators; Task6 numeric witness validation |
| Spec10/11 uncertainties | Preserve premise limits; independent review and fresh verification below; production execution separate |

- [ ] **Request final independent review.** Use `superpowers:requesting-code-review` against the complete branch diff from `00c212cb598d88e5cfe22a40d11d320f04c609d5`. The reviewer reads the approved spec/plan and actual runtime entry paths, assesses specification and quality, and distinguishes tested behavior from inferred equivalence. Task reviews do not replace applicable final review. Route fixes through the selected method, with causal RED/GREEN and required rereview.
- [ ] **Determine completion scope before verification.** Read `superpowers:verification-before-completion` and determine whether `superpowers:finishing-a-development-branch` applies immediately next. If a production witness is still required for the claimed completion, do not claim it passed or advance to finishing Step2. Present the concrete reviewed runtime recipe for separate authorization; absence of authorization is not a waiver. A report of implemented CPU-verified changes must explicitly say four-rank runtime acceptance remains pending.
- [ ] **Run fresh claim-specific checks on the reviewed revision.** For a branch-finish claim, use the project full-suite command once for both gates: `./venv/ilvr-venv/bin/python -m pytest -q --basetemp="checkpoints/$ORIG_RUN_ID/final-suite"`. Some existing unit tests require CUDA; inspect available resources and use only a verified idle GPU for those tests, restoring its approved visibility instead of the CPU context's empty CUDA_VISIBLE_DEVICES. Do not claim full-suite success from a CPU-only skip/failure or silently launch training. Preserve complete output and exit status; failures require causal fixes and new appropriate verification.
- [ ] **Verify distinct shell/diff claims once.** Run `bash -n run_stage2_training.sh scripts/stage2/dil_bce_profile.sh` after final changes, and `git diff --check` over authored changes. The copied source input intentionally contains original Markdown hard-break whitespace; verify it by SHA-256/byte comparison instead of modifying it. Source SHA and spec SHA must still match this plan's header/constraints. Record actual commits and review verdicts in the workflow record.
- [ ] **Report and hand off within evidence.** Link the implementation, tests and outstanding runtime gate. If all required acceptance witnesses are green and branch finishing applies, announce/use its required handoff options and preserve user ownership of merge/PR/keep/delete decisions. No automatic cleanup of this explicitly requested worktree. Run the required ntfy summary hook before the final response with the outcome that the actual evidence supports.

## Plan self-review record

Performed by root inline during writing-plans; this is a control-document review, not independent code review. All eleven design sections and source D1-D8/E1-E7/N1-N4/I1-I12 map to tasks above. Interfaces share the same packet/support/component/snapshot types; artifact binding precedes schema2 serialization; sidecar statistics are introduced with their consuming task. Each Review Focus item has a named owning test. No implementation, test pass or GPU launch is implied by unchecked steps.

The only scheduling refinement is implementing the approved one-writer-per-node snapshot contract before rank launch using the existing coordinator. This avoids new rank-side wait/retry machinery and preserves the approved artifact semantics. The production observer is bounded to the spec's required actual Trainer/DeepSpeed witness; it does not authorize its execution. Plan approval and implementation-method selection remain pending.

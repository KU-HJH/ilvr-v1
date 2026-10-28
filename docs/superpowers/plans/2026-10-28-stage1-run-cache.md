# Stage 1 Run Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the already selected Native method. The alternative superpowers:subagent-driven-development method has not been selected. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Avoid repeated static preprocessing and eligible frozen-teacher work across Stage 1 epochs while preserving the approved numerical and training semantics.

**Architecture:** Add an optional run-owned, lossless SSD cache at the existing collator, helper-target and V5 reference boundaries. The first epoch computes normally and publishes records; an epoch callback and the existing two-node coordinator establish complete local copies before workers enter read-only replay. Separate validation owns expensive diagnostics and deterministic backward; production retains the existing fast backward.

**Tech Stack:** Existing Python 3.11 repository venv, PyTorch 2.7.1, patched Transformers, Accelerate 1.10.1, existing DeepSpeed profiles; no new dependency or service.

**Spec:** [Frozen design v2](../specs/2026-10-28-stage1-run-cache-design-v2.md), SHA-256 `1d11731ad490e476efeb54f7a2610257bcbb3b9060736d040a5a9d14ce8fd8b7`; [approved commitment v2](../specs/2026-10-28-stage1-run-cache-commitment-v2.md), SHA-256 `f9a5bfe4542ceb1be78a32b4aa3d5652b98aac6477bdf72bb0666f148c65a087`.

## Global Constraints

- User's subsequent “진행해” approves commitment v2, its stated empirical uncertainty, and Native implementation. Preserve that approval; written-plan handoff is the remaining writing-plans prerequisite.
- Root implements; one fresh independent whole-change specification-and-quality review follows implementation. No implementer agents or per-task review cycles.
- Reuse `/root/.codex/worktrees/f187/ILVR` at implementation base `381606dfd9183abe3f2ec1e15cf4e3a9a7cf8033`. Preserve unrelated files and canonical checkout changes.
- Cached payload bits and semantic selection results, sample order, RNG behavior, objectives and optimizer/EMA semantics must remain unchanged. Whole-production-gradient bitwise reproducibility is not promised.
- Deterministic mode is validation-only. Do not change production attention, dtype, projection chunking, dropout, alignment noise, optimizer schedule or EMA ordering.
- Keep fixed/EMA standard Stage 1 and fixed-teacher V5 distinct. No moving-EMA V5, dynamic student cache, full logits, original-image vision-feature cache, lossy conversion or cross-run reuse.
- Cache is opt-in (`off` by default, including single-epoch runs). Required correctness witnesses must pass before real experiments use it. No ten-epoch experiment is authorized by this plan.
- All training/smoke uses the existing coordinator, remote rank 0 at `korea@172.16.11.41`, local rank 1, two GPUs each, `bond0.11`, microbatch 1, accumulation 4, global batch 16, workers 4 per rank. Read canonical training defaults/operator guidance before launch; preserve saved resume settings.
- Reuse DS0.18.7 for V5/matched standard and the preserved DS0.18.0 repository venv for native standard EMA on both nodes. No reinstall or native backend substitution.
- Cache lives under logical `data/cache/<run-id>/`, backed by a dedicated SSD subpath; retain the existing `data` symlink. All model/teacher/resume state stays under `checkpoints/`; output under `results/`, logs under `logs/`.
- First-epoch population, once-only cross-node completion, read-only later epochs, preparation-only missing-cache rebuild on resume, and own-run deletion after normal scheduled train/eval completion are binding.
- No per-step hashes, equality checks, diagnostic finite/norm scans, profiler hooks or target statistics in production. Required arithmetic, structural lookup checks and framework safeguards remain.
- Old checkpoint identities and historical artifacts remain authoritative. An old resume must not silently acquire new cache/check semantics.

## Review Focus

1. Prefetch/epoch-tail behavior: real Trainer/Accelerate workers must see read mode before any epoch-two batch, including interrupted first epochs (Tasks 2, 5).
2. Sample ownership: shuffled rows, duplicate sampler slots, empty helper/query sets and reverse-donor fallback must replay exact values with current batch offsets (Tasks 2–4).
3. Skipped-call effects: frozen eval forwards can change RoPE/buffers or consume RNG; exact serialization alone does not prove safe replay (Tasks 3, 4, 7).
4. Resume identity: a missing cache must never rebuild a fixed reference from the resumed student; moving EMA must still update targets online (Tasks 3, 6, 7).
5. Completion ownership: evaluation failure, planned checkpoint stop, open workers, or a foreign/symlinked run path must never cause premature or cross-run deletion (Tasks 5, 6).

## Runtime boundaries checked during planning

- `Stage1DataCollator.__call__` already preprocesses singleton rows, then `_collate_processed_rows` pads/concatenates and `_build_metadata` adds live occurrence/offset ownership. Cache this singleton static result, not a padded batch.
- `Stage1TeacherTargetBuilder.build` owns row packing. Both legacy/adaptive paths call the helper visual encoder; the EMA cache boundary is that output before grouping/selection. Fixed target replay may omit the whole helper builder, subject to forward-state evidence.
- V5 reference receives the helper-injected `model_inputs`. Qwen selects hidden rows and calls `lm_head` on `split(loss_chunk_positions)`; preserve that exact projection, including a final short chunk and empty queries.
- Patched Trainer calls `on_epoch_begin` before `iter(epoch_dataloader)` and `on_epoch_end` after consuming the epoch. Accelerate's `DataLoaderShard.__iter__` prefetches one batch. Use a worker-shared phase value and this callback boundary; do not rely on a parent-only Python flag.
- `monitor_owned_jobs` already supervises both nodes. Attach one cache-boundary exchange to this existing loop instead of adding a new job supervisor.
- `_stage1_checks_enabled()` defaults to true in native standard; V5/matched standard already gate several scans. Extend the common policy without removing semantic reductions.

## Files and responsibilities

- New `src/stage1/run_cache.py`: immutable launch policy, identity, lossless per-source records, coverage and worker-visible phase. No model or cluster orchestration.
- New `src/stage1/cache_payloads.py`: static-row/target/reference record codecs and forward-state replay for supported owners. No filesystem lifecycle.
- New `src/stage1/cache_lifecycle.py`: Trainer callback, saved cache contract and preparation/close behavior.
- New `src/stage1/cache_transport.py`: run-scoped transfer/cleanup operations using the existing coordinator's remote-container boundary.
- New `src/stage1/cache_validation.py`: opt-in acceptance observations/comparisons; never imported by the production hot path.
- Modify `src/stage1/{data,targets,trainer,v5_trainer,composition}.py` only at the existing boundaries; `v5_resume.py`/`matched_resume.py` only where needed for additive cache identity.
- Modify `src/utils_deepseed.py`, `run_stage1_training.sh`, `src/multinode.py`, `src/multinode_runtime.py` for explicit CLI forwarding, supported runtime selection and lifecycle wiring.
- Modify `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` only to expose the already selected reference hidden rows and share the unchanged projection function.
- New focused tests named below, operator documentation `docs/stage1-run-cache.md`, and verification driver `scripts/verify_stage1_run_cache.py`.

### Task 1: Bind launch/check policy and lossless record storage

**Files:** Create `src/stage1/run_cache.py`, `tests/stage1/test_run_cache.py`, `tests/stage1/test_check_policy.py`; modify `src/stage1/trainer.py`, `src/utils_deepseed.py`, `run_stage1_training.sh`.

**Interfaces:**
- `Stage1CachePlan` frozen dataclass: `run_id: str`, `root: Path`, `helper: Literal['off','targets','vision']`, `reference_hidden: bool`, `identity: dict[str, object]`.
- `resolve_stage1_check_policy(*, requested: str | None, saved: dict | None, validation: bool) -> str`, returning `production`, `validation` or preserved `legacy` once at construction.
- `Stage1RunCache(plan: Stage1CachePlan, phase)` where `phase` is a multiprocessing shared integer (`0=populate`, `1=replay`). `publish(layer: str, source_index: int, record: dict) -> bool`, `read(layer: str, source_index: int) -> dict`, `validate_coverage(expected: dict[str, tuple[int, ...]]) -> dict`, `close() -> None`.
- CLI `--stage1_cache {off,run}` / launcher `--stage1-cache`, default `off`; explicit run identity comes from the coordinator. No stale environment value can override resolved launch arguments.

- [ ] Write `test_lossless_roundtrip_and_no_backing_mutation`, `test_duplicate_publication_never_overwrites`, `test_incomplete_or_wrong_identity_is_rejected`, `test_cache_off_has_no_cache_io`, `test_production_skips_optional_scans_but_keeps_applied_step_and_alignment_noise`. Assert exact BF16/FP32/integer tensors, preserved dtype, duplicate computation allowed, immutable published contents and zero diagnostic call counts.
- [ ] Run `PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_run_cache.py tests/stage1/test_check_policy.py`. Expected RED for missing behavior, not unrelated import/environment failures.
- [ ] Implement per-layer/per-source `.pt` files using existing `torch.save` and `torch.load(weights_only=True, mmap=True, map_location='cpu')`; store tensors and primitive metadata only. Publish bounded CPU copies via a temporary complete file and atomic no-overwrite link; remove only that temporary file. Reads expose mapping-backed data to codecs which materialize current-batch mutable tensors. Never keep a whole-cache tensor dictionary.
- [ ] Bind checks before target-builder construction. Gate diagnostic-only full scans/norm recomputation with the resolved policy; preserve real clipping, nonfinite/overflow handling, successful-step/EMA ordering and result-producing `.item()`/`.tolist()` operations. Resume without the new policy retains its legacy behavior.
- [ ] Run the task command again; expected GREEN. Commit the task's files with `feat(stage1): add run-owned lossless cache and execution check policy`.

### Task 2: Replay static preprocessing with live ownership

**Files:** Create `src/stage1/cache_payloads.py`, `tests/stage1/test_cache_data.py`; modify `src/stage1/data.py`, `src/stage1/v5_metadata.py` only if needed for reconstruction.

**Interfaces:**
- `encode_static_row(row: dict, *, sample: Stage1SampleMetadata, v5_blocks: tuple) -> dict` and `decode_static_row(record: dict) -> dict` in `cache_payloads.py`; serialized metadata contains no occurrence, rank or padded batch offsets.
- `Stage1DataCollator(..., run_cache: Stage1RunCache | None = None)`; existing signature remains compatible.
- First-epoch helper pixels remain transient. A replay row has cached static helper geometry but no helper pixels; the target path receives an explicit cache source descriptor instead of pretending missing pixels are live data.

- [ ] Write `test_replay_skips_all_decode_processor_and_tokenizer_calls`, `test_padding_and_occurrence_rebuilt_after_shuffle`, `test_duplicates_compute_then_publish_once`, `test_empty_helper_and_empty_v5_query_rows`, `test_four_worker_phase_transition`. Use existing test processors/images and actual CPU DataLoader workers; monkeypatch forbidden replay calls to fail.
- [ ] Run `PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_cache_data.py`. Expected RED.
- [ ] Extract the existing singleton preprocessing body without changing processor calls/order. Persist tokens/labels/masks, original pixels/grid, spans, helper partitions and static V5 block mapping; omit helper pixels. Rebuild batch metadata and V5 batch indices from the current examples. Bind populate/replay per worker phase; never hash inputs per batch.
- [ ] Run the task command plus `tests/stage1/test_data.py`; expected GREEN and no live preprocessing calls in replay. Commit with `feat(stage1): replay static rows with current sample ownership`.

### Task 3: Fixed target replay and EMA frozen-vision replay

**Files:** Modify `src/stage1/cache_payloads.py`, `src/stage1/targets.py`, `src/stage1/trainer.py`; create `tests/stage1/test_cache_targets.py`.

**Interfaces:**
- `encode_target_sample(targets: Stage1TeacherTargets, metadata: Stage1BatchMetadata, row: int, *, forward_state: dict) -> dict`.
- `decode_target_batch(records: tuple[dict, ...], metadata: Stage1BatchMetadata, *, device: torch.device) -> Stage1TeacherTargets`.
- `Stage1TeacherTargetBuilder(..., run_cache: Stage1RunCache | None = None)`; factor the duplicated helper visual call into `_helper_visual(teacher, sample, pixels, grid) -> torch.Tensor` without changing its shape/dtype or call order.
- `capture_teacher_forward_state(teacher) -> dict` / `restore_teacher_forward_state(teacher, state: dict) -> None` cover the explicitly witnessed mutable Qwen forward fields (including RoPE); do not capture/restore global RNG as a substitute for proving that a skipped producer consumes none.

- [ ] Write `test_fixed_targets_and_reverse_donors_roundtrip`, `test_debug_off_metadata_stays_absent`, `test_ema_reuses_vision_but_queries_updated_teacher`, `test_original_image_vision_stays_live_for_ema`, `test_target_replay_preserves_teacher_forward_state`. Cover zero helpers, partial selected counts, same-sample other-image donors and original-image fallback; include duplicate source rows with different batch indices.
- [ ] Run `PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_cache_targets.py`. Expected RED.
- [ ] For fixed teachers, persist final chosen values/counts and enabled reverse values/provenance; reconstruct masks and batch indices, preserving absent optional diagnostics. For EMA, persist only detached visual/merger output and partitions; execute current teacher language/query/selection each call. Resolve frozen-visual eligibility once, preserving full/LoRA/DoRA distinctions; reject an unsupported mutable producer before enabling it.
- [ ] Run task tests plus `tests/stage1/test_targets.py`; expected GREEN with live/replay call counters and actual EMA-target change. Commit with `feat(stage1): cache fixed targets and frozen EMA vision outputs`.

### Task 4: Frozen V5 reference hidden-state replay

**Files:** Modify `src/stage1/cache_payloads.py`, `src/stage1/v5_trainer.py`, the Qwen modeling file; create `tests/stage1/v5/test_cache_reference.py`.

**Interfaces:**
- Qwen output adds optional `stage1_selected_hidden_states`; expose it only for requested cache capture, without retaining an extra full hidden-state copy.
- Qwen method `project_stage1_selected_hidden(hidden: torch.Tensor, chunk_positions: int) -> torch.Tensor` contains the existing `torch.cat([self.lm_head(chunk) for chunk in hidden.split(chunk_positions)], dim=0)` operation unchanged. Both live and replay call this same owner under the same no-grad/autocast scope.
- `reference_record` includes all ordered query positions, detached selected hidden rows and necessary forward-state metadata; source-row replay reassembles the original batch projection order before chunking.

- [ ] Write `test_reference_replay_uses_original_head_and_chunk_geometry`, `test_all_query_positions_survive_dynamic_entropy_selection`, `test_reference_eligibility_includes_injected_latents`, `test_empty_query_skips_reference_as_before`, `test_reference_rope_and_rng_state_match`. Assert exact hidden bits and logits; inspect short-final-chunk shape, owner, dtype and autocast.
- [ ] Run `PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_cache_reference.py`. Expected RED.
- [ ] Capture immediately before native reference projection. Replay only for frozen V5 with invariant full inputs; do not use clean student states, the changing entropy mask or a different helper source. Restore witnessed forward effects before returning reference logits. Leave clean/V-off/Text-off execution unchanged.
- [ ] Run task tests plus `tests/stage1/v5/test_trainer.py`; expected GREEN. Commit with `feat(stage1): replay frozen reference hidden states through native projection`.

### Task 5: Complete both-node caches before epoch-two prefetch

**Files:** Create `src/stage1/cache_lifecycle.py`, `src/stage1/cache_transport.py`, `tests/stage1/test_cache_lifecycle.py`, `tests/multinode/test_stage1_cache_transport.py`; modify composition and the existing multinode coordinator/runtime.

**Interfaces:**
- `Stage1CacheEpochCallback(cache: Stage1RunCache, *, expected: dict[str, tuple[int, ...]], control_dir: Path)` owns `on_epoch_begin`/`on_epoch_end`, write completion and worker phase publication.
- `exchange_stage1_cache(plan, runner, *, expected: dict[str, tuple[int, ...]]) -> dict` operates only after both nodes publish completed first-epoch manifests; no tensor payload collective.
- Existing `monitor_owned_jobs(..., boundary_action: Callable[[], None] | None = None)` gets a narrowly scoped optional hook. The cache hook is retired after one exchange; cache-off launches do not install it.

- [ ] Write `test_epoch_two_waits_for_both_node_manifests`, `test_shuffled_remote_record_is_available_to_workers`, `test_incomplete_epoch_never_enters_replay`, `test_transfer_is_bounded_and_rejects_foreign_paths`, `test_replay_missing_record_fails_without_recompute`. Test callbacks through the actual Trainer/Accelerate iterator sequence, not callback invocation alone.
- [ ] Run `PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_cache_lifecycle.py tests/multinode/test_stage1_cache_transport.py`. Expected RED.
- [ ] Finish local publications, exchange missing records once through existing host/container transport with bounded streaming, and validate full coverage/integrity on each node before release. Use an attempt-bound ready marker so an old resume attempt cannot release the new one. Preserve sampler order and existing failure/owned-process supervision; no retry service.
- [ ] Set the shared phase before the next iterator starts. Support both ordinary worker recreation and persistent-worker visibility in CPU tests; do not change the production persistent-worker/prefetch settings. A native epoch tail which omits required rows is an eligibility failure, not permission to change sampling silently.
- [ ] Run task tests plus `tests/multinode/test_coordinator.py`; expected GREEN. Commit with `feat(stage1): synchronize run caches at the first epoch boundary`.

### Task 6: Same-run resume, SSD placement and completion cleanup

**Files:** Modify run-cache/lifecycle/transport, composition, checkpoint hooks, `src/multinode.py` and `src/multinode_runtime.py`; create `tests/stage1/test_cache_resume.py`, `tests/multinode/test_stage1_cache_completion.py`, `docs/stage1-run-cache.md`.

**Interfaces:**
- `write_stage1_cache_contract(checkpoint: Path, plan: Stage1CachePlan, *, phase: int) -> None` writes additive `stage1_run_cache.json`; existing loss/runtime contracts are unchanged.
- `prepare_stage1_cache_resume(*, cache: Stage1RunCache, checkpoint: Path, initial_teacher, examples, collator) -> None` executes before `Trainer.train`; epoch-one resumes populate, later resumes validate/rebuild only missing payloads without changing restored training RNG.
- `finish_stage1_run_cache(*, run_id: str, completion: str, runner) -> None` closes readers and removes only the matching owned namespace after validated terminal success. Coordinator CLI `--stage1-cache-completion {train,train-eval}` persists the scheduled boundary; evaluation receives explicit `--stage1-cache-training-run-id` when applicable.
- Preflight keyword `stage1_runtime_profile` admits only explicit supported native-standard DS0.18.0 or the unchanged default DS0.18.7. The selected profile travels through both actual worker environments; Stage2 behavior is unchanged.

- [ ] Write `test_first_epoch_resume_continues_population`, `test_later_resume_rebuilds_from_initial_teacher_not_student`, `test_resume_rebuild_preserves_rng_and_saved_objective`, `test_old_resume_does_not_enable_cache`, `test_interrupted_or_failed_evaluation_retains_cache`, `test_completion_deletes_only_closed_owned_run`, `test_native_profile_requires_matching_supported_versions_on_both_nodes`.
- [ ] Run `PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_cache_resume.py tests/multinode/test_stage1_cache_completion.py`. Expected RED.
- [ ] Bind source/image content, processor, effective frozen weights, query/donor config and backend/dtype/schema once at launch/resume, reusing existing identities. Persist any additional initial-teacher state only under the run's `checkpoints/`, reusing existing EMA serialization; V5 reuses its saved initial reference. An old run without reconstructible initial identity must not be upgraded silently.
- [ ] Rebuild derived payloads only in the coordinator's four-rank preparation phase, using immutable initial state for fixed teachers or unchanged vision for EMA. Never treat a missing epoch-two record as a live fallback. No automatic training on resumed-student reference weights.
- [ ] Document and validate dedicated SSD cache placement on each node without replacing dataset links. Cleanup runs after final checkpoint/result validation, and after scheduled evaluation when declared; planned checkpoint stops and abnormal exits retain caches. Preserve other caches/checkpoints/results/logs. Add only the narrow DS0.18.0 preflight exception required by native EMA verification.
- [ ] Run task tests plus existing resume/preflight tests affected by the diff; expected GREEN. Commit with `feat(stage1): preserve cache identity across resume and scheduled completion`.

### Task 7: Independent review and bounded numerical/runtime acceptance

**Files:** Create `src/stage1/cache_validation.py`, `scripts/verify_stage1_run_cache.py`, `tests/stage1/test_cache_validation.py`; modify composition/CLI only for explicit isolated verifier entry. Record observations under `results/stage1-run-cache-validation-<attempt>/` and logs under the matching `logs/` directory.

**Interfaces:**
- `Stage1CacheValidationObserver(manifest: dict, output_dir: Path)` is constructed only by the explicit verification entry and observes payloads, projection geometry, RNG/forward state, actual gradients and optimizer/EMA transitions.
- Driver CLI `--phase {dry-run,correctness,performance} --profile {v5,native-ema} --run-id ID`; every GPU child uses `run_multinode_stage1_training.sh`. There is no single-node training route.
- Comparison returns independent `payload`, `projection`, `rng_state`, `semantic_selection`, `loss`, `gradient`, `nonzero_update`, `ema`, `lifecycle`, `production_checks` results; absent evidence is not a pass.

- [ ] Write `test_verifier_rejects_unequal_aa_before_ab`, `test_lr_zero_cannot_establish_update_equivalence`, `test_production_does_not_import_or_invoke_observer`, `test_verifier_cannot_enable_full_experiment_or_change_coefficients`. Include a deliberately perturbed cache and a stateful/RNG-consuming fake producer that the verifier rejects.
- [ ] Run `PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_cache_validation.py`. Expected RED; implement the isolated observer/driver, then repeat for GREEN. Commit with `test(stage1): add isolated cache acceptance witnesses`.
- [ ] Finish the Native task ledger, create the whole-change review package, and dispatch one fresh independent reviewer under requesting-code-review/executing-plans. Review both spec compliance and quality with the five Review Focus items above. Fix causal Critical/Important findings once with RED→GREEN tests; preserve deferred findings and rulings.
- [ ] Before GPU acceptance, verify both nodes' reviewed commit, inputs, venv/profile, free devices, resolved coefficients/schedule, actual four-rank participation and workers 4. Reuse existing detached coordinator and ownership/evidence machinery. A dry-run is not proof of runtime participation.
- [ ] Run the smallest native fast-path payload/reload/projection/RNG/forward-state witness first. Stop the affected layer on any mismatch. Then deterministic uncached A/A for the same actual geometry; only after A/A repeats, compare cache/live A/B. Do not invent tolerance, substitute attention kernels/dtypes, or remove dropout to obtain a pass.
- [ ] Bound correctness to two successful updates per arm (A/A/B for V5 and native EMA: at most 12 optimizer updates total), preserving warmup 10, LR 5e-5 and the recorded full schedule so update two has nonzero LR. Use the same explicitly recorded short verification dataset and sample order within each profile; do not mislabel this as a full-data epoch/convergence result. Exercise first-write→epoch boundary→replay using a small complete verification dataset, plus a planned same-run resume with one removed derived record. The resume continues the arm's budget, not a fresh budget.
- [ ] Use at most one additional fast-backward update per cache/live arm per profile (4 total) for bounded throughput observations, with detailed timing enabled only in these verifier runs. Measure cache bytes, resident-memory observation, write/exchange overhead and replay data/transfer waits. Report noisy short measurements and projected amortization separately; no whole-run speedup claim without adequate evidence. Total initial GPU ceiling: 16 updates; stop early on a failed prerequisite.
- [ ] Invoke verification-before-completion and finishing-a-development-branch together if branch completion applies. Run `PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/python -m pytest -q` once for the final overlapping suite claim, retain/read the complete log, and run distinct missing claim-specific witnesses once. Required gated GPU tests and actual acceptance results are reported separately. No duplicate back-to-back full suites.
- [ ] Persist approvals, plan/spec hashes, actual skill outcomes, commits, review identity/verdict, rulings, test commands/results and runtime witness paths in `logs/stage1-cache-design-20261028/implementation.json` before any skill scratch cleanup. Report exact verified support and failures; never enable a failed/unverified cache layer for a real experiment. Preserve this externally managed worktree and leave merge/branch handoff to the user.

## Planning self-review and handoff

Spec coverage: eligibility/check policy → Task 1; static records/dynamic ownership → Task 2; fixed/EMA payloads → Task 3; frozen reference projection → Task 4; first-epoch/worker/two-node transition → Task 5; resume/storage/cleanup/runtime compatibility → Task 6; numerical and performance witnesses/review/finish → Task 7. Each Review Focus item has named tests in its owning task.

Interfaces share `Stage1CachePlan`, `Stage1RunCache`, primitive record dictionaries and existing Stage1 dataclasses; no later task relies on an unnamed cache service. Validation uses the same live/replay implementation. The plan intentionally does not prescribe function bodies beyond the unchanged projection and minimal storage format.

Implementation method remains Native. Per writing-plans' Execution Handoff, present this saved plan for review before editing product code. This is not another approval of the numerical policy, cache lifetime or implementation method. Frozen proposal/commitment contents remain unchanged; approval and phase transitions are recorded separately.

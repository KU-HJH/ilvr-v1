# Same-image Evidence Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Repository policy requires delegated implementation; executing-plans/inline implementation is not an alternative. Steps use checkbox syntax for tracking.

**Goal:** Add the approved opt-in same-image evidence-replacement comparison to ours1 CE+Preference training, with deterministic selection and detached replacement copies.

**Architecture:** A pure planner selects earlier latent donors or image-feature fallback. A Qwen scorer replays original/replacement prefixes with native visibility and paired RNG. Trainer, configuration and launch paths expose the comparison and source-weighted diagnostics without changing old modes.

**Tech Stack:** Repository Python venv, patched Transformers/Qwen2.5-VL, PyTorch, pytest, shell worker launcher and existing multi-node coordinator.

**Spec:** `docs/superpowers/specs/2026-09-16-evidence-replacement-design.md`, design commitment v1. Approved by user reply "진헹하자" after presentation; approved bytes SHA256 `8fa8e36fb433ebe2e7563852ee309ce1ab2e97fda6f6a30bc6841ab1983e1f4e` at source commit `5557bdd1ef41ea8e5354a2ac1990b636f0d5cf19`.

## Global Constraints

- Work only in `/root/workspace/ILVR/.worktrees/ours1-evidence-replacement-20260916`, branch `feat/ours1-evidence-replacement-20260916`. Source baseline is `ae5a1cd`, a faithful snapshot of current dirty ours1 plus relevant untracked files. Base branch for later human integration choice is `ours1`.
- Use `./venv/ilvr-venv/bin/python` with `PYTHONPATH=src:transformers/src`; reuse the installed environment. No dependency reinstall, pretrained-model loading, training launch or external reference code.
- Opt-in mode string is `evidence_replacement`; original image access remains, boundaries remain, per-device microbatch one, one image, independent visual-only context preference, entropy weight zero. Do not change old defaults.
- Donors are earlier validated scored steps with exactly configured latent size. First/single steps use same-image encoder features; M >= L samples without replacement, 0 < M < L with replacement, sorted source indices. Both replacement copies are detached; original donor occurrence remains differentiable.
- Sampling version is `ilvr-evidence-replacement-v1`. Seed uses SHA256 of UTF-8 compact JSON `[version, training_seed, canonical_sample_id, step_id, source_kind]`, first eight bytes big-endian masked to 63 bits. A private CPU generator chooses donor via `randint(N,(1,))`, image via `randperm(M)[:L]` or `randint(M,(L,))`; no global RNG consumption.
- Scores gather `input_ids[:, target_start:target_end]` from predictors `target_start-1 .. target_end-2`. Full-prefix original/replacement replays share positions/native visibility/dropout draws; no clean/donor KV reuse. Dense uses native 2D mask, Flex matches actual native causal/sliding rule. No new public Stage2B resume support.
- Preserve code/data/checkpoint symlinks; outputs under this worktree's results/, logs under logs/. No generated model weights. CPU tests and tiny CUDA forward/backward verification are allowed; optimizer/training jobs are not.
- Implementers must read TDD plus writing-good-tests, show RED before implementation and GREEN afterward, self-review and commit. No subagents or paperthin hooks from implementers/reviewers. Root alone coordinates review/fix gates and writes workflow control records.

## Handoff clarifications

`ae5a1cd` snapshots existing ours1 changes on top of the approved-spec commit `5557bdd`; it does not change the bound spec. Tiny models are randomly initialized local fixtures, never pretrained downloads. Invalid planner metadata raises `ValueError` with a useful field-specific message. Existing `SidecarPlan` records define scored/active branches; empty branches mean CE-only. Consult their definitions for shapes and the existing target-scope implementation, not a new interpretation. Paired RNG covers CPU plus the active embedding CUDA device; one process uses one GPU. The durable workflow record is the sibling `2026-09-16-evidence-replacement-evidence.md`. Exact new diagnostics may follow existing naming style but must name original/replacement and keep donor/image sum/count weighting explicit. The TDD skill and `writing-good-tests.md` are under `/root/.codex/plugins/cache/openai-curated-remote/superpowers/6.3.0/skills/test-driven-development/`.

## File Structure and Interfaces

Task 1 creates `src/step_preference/replacement.py` and planner tests. Task 2 creates `transformers/src/transformers/models/qwen2_5_vl/stage2_evidence_replacement.py`, connects Qwen/shared-input validation, and adds runtime tests. Task 3 exposes config/CLI/launcher/Trainer/metadata/diagnostics plus documentation and end-to-end tests. Sequential task execution prevents shared-file conflicts.

## Task 1: Deterministic replacement planner

**Files:** Create `src/step_preference/replacement.py`, `tests/step_preference/test_replacement_plan.py`, `tests/step_preference/replacement_helpers.py` (test-only fixtures).

**Interfaces:** Consume `SidecarPlan`/`SidecarBranch` from `step_preference.sidecar`, already built with `visual_only=True, suppress_latent_boundaries=False`. Produce the following frozen records/function (all sequence fields tuples):

```python
@dataclass(frozen=True)
class ReplacementBranch:
    step_id: int
    latent_start: int
    latent_end: int
    prefix_length: int
    predictor_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    source_kind: str  # donor or image
    source_positions: tuple[int, ...]  # absolute positions in source input
    donor_step_id: int | None

@dataclass(frozen=True)
class ReplacementPlan:
    root_length: int
    branches: tuple[ReplacementBranch, ...]

```

The public function signature is `build_replacement_plan(*, input_ids, sidecar_plan, image_token_id, image_count, latent_size, sample_id, seed) -> ReplacementPlan`.

- [ ] **Write causal planner tests first.** Test fixture `planner_case()` returns `(ids, sidecar_plan)` with IDs `[11,81,80,80,80,83,14,91,90,90,92,21,22,91,90,90,92,23,24]`, all-one attention, and spans `ReasoningStepSpan(0,0,7,8,10,11,13,None)` and `ReasoningStepSpan(1,11,13,14,16,17,19,None)`. Use existing `build_context_sidecar_plan`; do not duplicate its implementation.

```python
def test_first_step_samples_image_and_next_step_uses_only_earlier_donor():
    # Catches self/future donors, wrong absolute source positions, or shifted targets.
    ids, sidecar = planner_case()
    p = build_replacement_plan(input_ids=ids, sidecar_plan=sidecar,
        image_token_id=80, image_count=1, latent_size=2,
        sample_id="example", seed=42)
    a, b = p.branches
    assert (a.source_kind, a.source_positions, a.prefix_length) == ("image", (3,4), 12)
    assert (a.predictor_positions, a.target_token_ids) == ((10,11), (21,22))
    assert (b.source_kind, b.donor_step_id, b.source_positions) == ("donor", 0, (8,9))
    assert (b.predictor_positions, b.target_token_ids) == ((16,17), (23,24))
```

Also test one-step-only, zero-step CE-plan, multiple earlier donor candidates, deterministic local selection despite changed global RNG, global RNG unchanged, M < L duplicate sampling, absent/multiple images, malformed donor length, suppressed boundaries, textual/rejected branches, empty sample ID and invalid seed types. Expectations must be literal/invariant, not recomputed using the planner.

- [ ] **Run RED:** `PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_replacement_plan.py`; log the expected missing-module/function failure under `logs/evidence-replacement-20260916/`.
- [ ] **Implement the planner minimally.** Validate B1/root bounds and active visual-only context plan; empty branches return an empty replacement plan. Validate every block length/target/predictor range and image cardinality before selection; reject malformed donors rather than silently filtering them. Source positions are absolute clean positions for donor and absolute base-input image positions for image. A branch's `prefix_length` is its last predictor plus one. Preserve the already-validated target IDs and logical predictor positions. Generate deterministic indices exactly as the spec and populate frozen metadata.

```python
raw = json.dumps(["ilvr-evidence-replacement-v1", seed, sample_id, step_id, kind],
                 ensure_ascii=False, separators=(",", ":")).encode("utf-8")
local_seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)
generator = torch.Generator(device="cpu").manual_seed(local_seed)
# For seed=42, sample_id="example", step_id=0, kind="image":
# local_seed is 4361521485388870428 and sorted randperm(3)[:2] is [1,2].
```

- [ ] **Run GREEN** on the new planner suite and existing `test_sidecar.py`/`test_context_sidecars.py` excluding gpu/distributed. Self-review diff and preserve RED/GREEN command/output evidence.
- [ ] **Commit** the planner and tests only; write the task report including commit IDs, focused results, self-review and any concerns. Root then obtains independent spec-and-quality review before Task 2.

## Task 2: Native paired replay and Qwen dispatch

**Files:** Create `transformers/src/transformers/models/qwen2_5_vl/stage2_evidence_replacement.py`, `tests/step_preference/test_replacement_runtime.py`; extend test-only `replacement_helpers.py`; modify `modeling_qwen2_5_vl.py` and `stage2_shared_prefix.py` narrowly.

**Consumes:** Task 1 `ReplacementBranch`, `ReplacementPlan`, `build_replacement_plan` signatures above. Existing tiny Qwen `_tiny_model()` from `step_preference.equivalence` is available. Existing `SidecarPlan` remains the top-level shared input plan.

**Produces:** `score_evidence_replacement(model, options, input_ids, base_inputs_embeds, clean_inputs_embeds, clean_positions, image_grid_thw)` returns `(tuple_of_results, stats)`. New-mode options additionally contain `sample_id` (canonical str), `replacement_seed` (training int) and `replacement_version` (exact sampling version). Each result contains `step_id`, `kind` (`visual_original` or `visual_replacement`), `mean_logp` (scalar tensor), `entropies` (target-length tensor), `source_kind`, `donor_step_id`, `source_positions`. Stats use the existing scoring-stat keys and source counts as appropriate.

- [ ] **Write runtime tests before production edits.** Use fixture input/spans from Task 1 and an actual tiny model with manually supplied 3-axis positions. Direct scorer tests may use separate leaf tensors for base image vectors, clean latents, and other embeddings. Build a literal independent reference by cloning prefix inputs, replacing exactly one block with detached known source vectors, and calling `model.model` through native attention.

```python
target = ids[:, 11:13]
rows = torch.tensor([10, 11])
hidden = model.model(inputs_embeds=literal_prefix, position_ids=positions[:, :, :12],
    attention_mask=torch.ones((1,12), dtype=torch.long), use_cache=False,
    return_dict=True).last_hidden_state
expected = model.lm_head(hidden.index_select(1, rows)).float().log_softmax(-1)
expected_score = expected.gather(-1, target.unsqueeze(-1)).mean()
torch.testing.assert_close(actual_result["mean_logp"], expected_score)
```

Cover first-target effect, self-substitution score identity, original image access in both branches, no current-latent influence left in replacement, copied donor detach versus live original donor position, nonzero decoder/original gradients, source-specific input perturbation, variable target lengths, zero steps, no text generation or stale KV. Test checkpoint on/off and paired dropout against native literal replay, including RNG state after the pair. Compare Flex visibility with the actual native mask helper at lengths below/above sliding window, including `use_sliding_window=False`; include a GPU-marked dense/Flex output-and-gradient test gated by `RUN_REPLACEMENT_CUDA=1`.

- [ ] **Run RED** for `test_replacement_runtime.py` before implementing; retain expected missing-scorer/mode failure.
- [ ] **Implement scorer and strict input validation.** Build immutable plan once before checkpointing. Select donor values from `clean_inputs_embeds`, image values from `base_inputs_embeds`, `.detach()` either replacement copy, and replace only current latent pad slots. Bind branch closure metadata. Run native dense 2-D masking, or bounded causal/sliding Flex mask; never call the old `_evidence_mask`. Capture CPU/active-CUDA RNG before original; fork/restore for replacement; after the fork retain post-original RNG. Use non-reentrant checkpointing with `preserve_rng_state=True`.

```python
original = clean_inputs_embeds[:, :branch.prefix_length]
source = clean_inputs_embeds if branch.source_kind == "donor" else base_inputs_embeds
replacement_values = source.index_select(1, source_indices).detach()
replacement = torch.cat((original[:, :branch.latent_start], replacement_values,
                         original[:, branch.latent_end:]), dim=1)
# For each of original/replacement, gather target logps only at predictor rows.
```

Extend `validate_shared_inputs` to accept this mode and only its extra keys, require V-only context/boundaries kept/independent/image access. `_forward_stage2` checks active samples have pixel values and one image grid, then routes base+clean embeddings and positions to the scorer after the once-only clean rollout. No new CE/KV capture behavior in old modes. Enforce existing backend restrictions and malformed source metadata errors.

- [ ] **Run GREEN:** runtime and planner suites plus `test_shared_prefix_model.py`, `test_context_sidecars.py`, `test_evidence_gain.py` with non-GPU selection. Run the tiny CUDA comparison separately only if a GPU is free, using the existing venv; this is a forward/backward test without optimizer/model saving. Capture real Flex dispatch and complete output, not merely a configured backend. No training launcher.
- [ ] **Commit** scorer/model/tests; report RED/GREEN, GPU evidence or exact limitation, self-review and changed interfaces. Independent task review is required before Task 3.

## Task 3: End-to-end preference integration, metadata and user-facing usage

**Files:** Modify `src/step_preference/config.py`, `src/utils_deepseed.py`, `src/stage2/trainer.py`, `run_stage2_training.sh`, and where necessary `src/main.py`, `src/multinode.py`/`src/multinode_runtime.py`. Create `tests/step_preference/test_replacement_trainer.py`, extend `test_config.py`, `tests/stage2/test_launcher.py`, `tests/multinode/test_evidence_gain_forwarding.py`. Update `documents/preference-prefix-options.md` with the new mode's concrete invocation and limits.

**Consumes:** Task 2 new-mode shared-input keys and original/replacement result records; Task 1 sampling version and planner contracts. **Produces:** Public mode routed from coordinator/worker/CLI to actual Trainer/model, exact loss, source-weighted metrics, and checkpoint metadata identifying mode/seed/version. Preserve existing modes and public resume restrictions.

- [ ] **Write tests first.** Config accepts the new mode only for CE+Preference with the approved constraints; omitted boundary suppression resolves false for it and retains old behavior otherwise. Reject entropy >0, explicit suppression, image blocking, T-off, cumulative, inactive Stage2B or legacy DPO combinations. Seed derives from public training `--seed` via `StepPreferenceConfig.from_args` (a `replacement_seed` field is acceptable). Sampling version is the spec's constant, not a new independently mutable CLI knob.

```python
cfg = StepPreferenceConfig(enabled=True, mode="stage2b", prefix_mode="evidence_replacement",
    stage2a_checkpoint_path="tiny-stage2a", lambda_entropy=0.0,
    sidecar_attention_backend="dense")
assert cfg.visual_off_suppress_latent_boundaries is False
loss, output = trainer.compute_loss(model, batch, return_outputs=True)
by_step = {r["step_id"]: {} for r in output.stage2_sidecar_results}
for r in output.stage2_sidecar_results:
    by_step[r["step_id"]][r["kind"]] = r["mean_logp"]
margins = torch.stack([v["visual_original"]-v["visual_replacement"] for v in by_step.values()])
expected = output.loss - cfg.lambda_preference * torch.nn.functional.logsigmoid(
    cfg.beta*(margins-cfg.preference_margin)).mean()
torch.testing.assert_close(loss, expected)
```

Use the real tiny vision encoder with patch/grid fixtures and actual Trainer->model calls: count one encoder/clean rollout, observe both source kinds, check no generation, backpropagate, exercise both target scopes and zero steps. Test source diagnostics with uneven counts (e.g. margins sum 2/count1 and sum12/count3 must report 3.5, not mean of per-batch means); use an existing accelerator-reduction pattern to sum counts/margins over ranks, without new distributed training. Launcher tests must execute dry runs and parse argv, not search source text; worker forwarding must preserve mode, entropy 0, seed and boundary choice for both nodes.

- [ ] **Run RED** on new/extended focused tests and preserve expected rejection/missing-mode failures before production edits.
- [ ] **Wire the config and runtime.** Extend allowed mode choices and fail-fast validation; `from_args` reads the unchanged public training seed. Existing context metadata must remain span-only. In replacement mode, forward canonical sample ID/seed/version, consume both score kinds, preserve clean CE, calculate the specified step-mean preference, and never label new metrics as `visual_off`. Add a mode-specific metrics-required set and sum/count accumulators; distributed logging aggregates raw sums/counts before division. Zero counts report zero. Ordinary loss reduction and old-mode logging remain unchanged.

```python
preference = context_preference_loss(original_scores, replacement_scores,
                                     config.beta, config.preference_margin)
total = clean_ce + config.lambda_preference * preference
# Logging is separate from differentiable loss:
# source_mean = globally_summed_margin / globally_summed_count if count else 0.
```

Persist a small replacement objective identity (mode, training seed, sampling version and fixed selection/gradient semantics) through the existing model config save path for this mode only; both full and DoRA `_save` paths already save model config. Do not alter old-mode identity schemas or relax public resume. Add a save/reload metadata contract test under tmp fixtures; no real checkpoint weights. If a smaller existing objective metadata path is found, use it and report it.

Worker launcher accepts `STEP_PREFERENCE_PREFIX_MODE=evidence_replacement`, requires explicit `STEP_ENTROPY_WEIGHT=0`, resolves unspecified suppression to false, and retains coordinator topology. Document this mode and selection/gradient/entropy constraints alongside current prefix modes, including a launch example with existing coordinator, explicit experiment ID, paths under checkpoints/logs/results, and no actual launch. Do not edit historical run outcomes.

- [ ] **Run GREEN:** all new tests plus config/context/launcher/forwarding regressions. Report exact runtime witness for source metadata, logged weighted metrics and saved identity. Do not re-run unaffected broad suites repeatedly; the root runs the full project CPU suite once on the reviewed final revision.
- [ ] **Self-review and commit** all integration/test/document changes; provide report with RED/GREEN commands/output and limitations. Root obtains task review, then one whole-feature review against `ae5a1cd`, required fix review and fresh completion verification.

## Completion and Handoff

Root selects the finishing gate now: it will apply after implementation/reviews. The project full CPU suite is `PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q -m 'not gpu and not distributed'`. Run once on the final reviewed revision and reuse for verification-before-completion plus finishing Step 1. Run the distinct `RUN_REPLACEMENT_CUDA=1` GPU witness on that revision for the Flex claim. Actual multi-node training/full benchmark are outside approval and must remain separately unverified.

Preserve task agent IDs, base/head commits, reports, review verdicts/fixes, commands/output and any rulings in the existing durable workflow record before deleting this plan's SDD scratch. Human owns merge/PR/keep decisions; do not integrate into dirty ours1 automatically.

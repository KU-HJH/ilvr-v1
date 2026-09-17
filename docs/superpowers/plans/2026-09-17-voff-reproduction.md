# V-off Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the supplied external V-off protocol with the explicitly approved local Stage1 initialization, preserving and disclosing the mandatory four-rank topology difference.

**Architecture:** Add opt-in behavior to existing data, latent rollout, and evaluation paths. Retain existing defaults and the existing multi-node coordinator. A small experiment-specific verification program supplies comparative and actual distributed update evidence before the full run.

**Tech Stack:** repository Python 3.11 venv, patched Transformers, PyTorch, PEFT DoRA, DeepSpeed ZeRO2, SDPA/FA2/Flex.

**Spec:** `/root/workspace/ILVR/results/v-off-result-search-20260917/reproduction-design-commitment-v2.json`, approved by user message `진행해` following its presentation. Initializer separately approved by `현재 재현한 Stage 1을 초기값으로 사용해라`.

## Global Constraints

- Worktree `/root/workspace/ILVR/.worktrees/voff-reproduction-20260917`, base commit `44b2967cd767fb2fd4c3465167b2ba511695d2c2`; preserve other worktrees.
- Use `./venv/ilvr-venv/bin/python`; results in `results/`, logs in `logs/`, every model-bearing artifact in `checkpoints/`, prepared models only in `checkpoints/evaluation-staging/`.
- Every real training/probe uses existing two-node/four-rank coordinator; remote `korea@10.1.102.41`, container `llm_kd`, rank0; local rank1; master10.1.102.41, interfacebond1.102.
- Reproduction only: microbatch1, accumulation4, global16, max_steps212, LR5e-5, warmup10, seed42, DoRA r32/alpha64/dropout0.05, beta0.1, preference weight1, entropy0, until_next_latent, clean_prefix, latent-body masking, START/END kept, direct Stage1 initialization.
- Initializer: canonical `checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431`. Fresh optimizer, no resume of cancelled run.
- Global gradient checkpointing OFF; selective latent-prefix activation checkpointing ON; no-cache recurrence, languageSDPA/visionFA2/sidecarFlex.
- Normal evaluator defaults and all existing training defaults remain unchanged. New options reject invalid values and are recorded wherever effective runtime/provenance is saved.
- External source at `/raid/workspace/ILVR/ILVR/.worktrees/ours1/source` is read-only evidence, never installed/imported as a production dependency. Differential probes may load it only as isolated test evidence. No reference repo changes.
- External image bytes, historical sealed source, single-rank minibatch order and exact initial weights are not authenticated; no bitwise/54%-guarantee claim.
- Implementers/fixers never invoke paperthin hooks or spawn agents. Each task follows TDD and receives independent specification-and-quality review. Root authors control artifacts only.
- No generalized retry/provenance infrastructure. One focused suite per task, final relevant regression suite, one critical distributed verification campaign; broaden only for failures or unresolved claims.

## Task 1: Selectable assistant rendering with consistent spans and manifests

**Files:** Modify `src/utils_deepseed.py`, `src/main.py`, `run_stage2_training.sh`, `src/multinode_runtime.py`; modify data/config validation only where the explicit option reaches existing rendering checks. Test `tests/stage2/test_length_manifest.py`, `tests/stage2/test_launcher.py`, `tests/step_preference/test_target_scope.py` and a focused `tests/stage2/test_reproduction_rendering.py` if useful.

**Interfaces:** Produce CLI `--stage2_assistant_plan_rendering {auto,concatenated-v1,adjacent-text-space-lf-v1}` and env `STAGE2_ASSISTANT_PLAN_RENDERING`, default `auto`. Resolve `auto` exactly as before. The resolved rendering must be used by launcher length manifest, main dataset creation, existing manifest validation and token/character span mapping. This task does not change target-scope defaults.

- [ ] Write failing behavioral tests using the existing processor/data fixtures. For plan `[text, latent, text("reason."), text("The final answer is: 7")]`, assert concatenated rendering contains `reason.The final answer is: 7`, adjacent rendering retains its separator, and until_next_latent target contains the answer under both. Assert launcher/main resolve the same enum and malformed mode fails.

```python
assert rendered_concat.endswith("reason.The final answer is: 7")
assert "The final answer is: 7" in decoded_until_next_target
assert resolved_manifest_rendering == resolved_training_rendering
```

- [ ] Run the focused new test to capture RED before implementation.
- [ ] Implement smallest resolver and thread it through existing calls. Use a named resolver in an existing rendering module if launcher/main need the same policy; do not copy the resolution rule into divergent branches. Preserve resume identity through existing manifest rendering checks.

```python
if requested == "auto":
    return existing_preference_rendering if preference_enabled else "concatenated-v1"
return requested
```

- [ ] Run the rendering/target/launcher focused tests, record GREEN and source locations proving the runtime path. Commit task changes and full report.

## Task 2: Selective latent-prefix checkpointing, opt-in and gradient-safe

**Files:** Modify `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`, `modeling_qwen2_5_vl.py`, `src/main.py`, `src/utils_deepseed.py`, `run_stage2_training.sh`, `src/multinode_runtime.py`; add focused `tests/stage2/test_reproduction_checkpointing.py`; use existing config/metadata mechanisms to record the option.

**Interfaces:** Produce CLI boolean `--stage2_latent_activation_checkpointing` / `--no-stage2_latent_activation_checkpointing`, env `STAGE2_LATENT_ACTIVATION_CHECKPOINTING`, default false. Wire model configuration to `LatentRolloutEngine.rollout(..., latent_activation_checkpointing=False)`. Task1 rendering remains independent. Preserve configured flags on reload/resume or reject mismatches through existing identity mechanisms.

- [ ] Write RED tests using tiny decoder with trainable embedding/linear layers and dropout. Compare final embeddings, CE-derived loss and input/parameter gradients checkpoint on/off with restored RNG. Test multiple latent positions, two blocks, recompute and retained grad connectivity. Confirm false preserves current behavior and inference/no-grad does not checkpoint.

```python
torch.manual_seed(42)
plain = run_case(checkpointing=False)
torch.manual_seed(42)
checked = run_case(checkpointing=True)
torch.testing.assert_close(plain.loss, checked.loss)
for a, b in zip(plain.parameter_grads, checked.parameter_grads):
    torch.testing.assert_close(a, b)
```

- [ ] Implement the supplied external no-cache latent-prefix replay semantics around each decoder prefix call, preserving this branch's allocation-efficient assembled inputs and returned interfaces. Bind each closure's tensors explicitly.

```python
with torch.utils.checkpoint.set_checkpoint_early_stop(False):
    latent = torch.utils.checkpoint.checkpoint(
        predict_last_hidden, prefix_inputs, prefix_mask, prefix_positions,
        use_reentrant=False, preserve_rng_state=True,
    )
```

- [ ] Validate opt-in forwarding through model/launcher/coordinator. Do not enable global checkpointing, introduce cache checkpointing, detach latent state, or change loss/dropout semantics.
- [ ] Run focused new and existing rollout/launcher tests, record RED/GREEN and self-review; commit.

## Task 3: External generation-limit semantics and evaluator forwarding

**Files:** Modify `src/evaluation/continuous_latent.py`, `eval.py`, `src/multinode_evaluation.py`, `src/multinode_runtime.py`, `src/multinode.py`, `run_multinode_evaluation.sh` only if its parser requires changes; test `tests/evaluation/test_continuous_latent.py` and relevant existing multi-node evaluation tests. Add a small test file if cleaner.

**Interfaces:** Produce explicit option `--generation-limit-policy external_full_latent_block` at coordinator and evaluator; default `bounded` preserves current behavior. Generation-config/manifest records policy; preserve reading existing schema/normal results rather than invalidating old records. Existing `--repetition-penalty 1.0` disables penalty; existing max-sequence128000 matches supplied model config.

- [ ] RED tiny-controller tests: at `max_new_tokens=1`, model emits START, external policy completes all8PAD+END (10 generated IDs including START); current bounded policy remains unchanged. Complete block exceeding max_sequence_length raises the external error; EOS exits normally; target/START choices with repetition1 follow raw logits.

```python
assert generated_ids_external == [START] + [PAD] * 8 + [END]
assert len(generated_ids_external) == 10
```

- [ ] Implement policy as a small branch around capacity/admission and block completion, sharing existing model calls and latent recurrence. Unknown policy fails. No source/controller duplication.
- [ ] Add coordinator argument propagation and result-manifest compatibility tests. Differential tiny traces against supplied external controller must include normal text, two latent blocks, EOS and generation/sequence-boundary cases.
- [ ] Run focused tests, record evidence and commit.

## Task 4: Comparative and distributed reproduction witnesses

**Files:** Create experiment-specific `scripts/verify_voff_reproduction.py` (controller/read-only comparisons and launch preparation) and, only if necessary, `src/stage2/reproduction_probe.py` (opt-in temporary real training observation); targeted tests under `tests/stage2/test_reproduction_probe.py`. Thread explicit disabled-by-default probe argument through existing coordinator/launcher only if no existing sanctioned observation hook supports it. No generic probe framework.

**Interfaces:** Consume Task1 rendering enum, Task2 latent checkpoint option, Task3 generation policy. Produce JSON evidence under a fresh `results/<probe-id>/`, logs under matchinglogs, optional model/checkpoint state only undercheckpoints. CLI `python scripts/verify_voff_reproduction.py --output-dir results/<id> --external-source /raid/workspace/ILVR/ILVR/.worktrees/ours1/source` runs CPU comparisons; real-GPU modes must generate/use existing coordinator commands and never manually start ranks.

- [ ] Specify arithmetic reference before observing Trainer results: each example's existing mean-token CE plus mean-over-valid-latent-steps preference; global objective averages examples over the actual group, with beta0.1/weight1. Identify actual external parent Trainer scaling and report any disparity against that reference instead of silently altering production loss. Clipping once per optimizer update at1.0, then existing AdamW. Controlled dropout for arithmetic comparison; separate production dropout/replay determinism test. Handle partial final group explicitly and record denominator.
- [ ] RED test independently hand-calculated two-example scalar/gradient and a partial group; test a deliberately doubled or omitted accumulation divisor fails. The reference must not call Trainer's scaling helper to compute its expected value.

```python
expected = sum(example_losses) / actual_group_size
expected_grad = torch.autograd.grad(expected, parameters)
# Compare captured effective gradient before clipping and actual AdamW delta,
# not merely Trainer's reported scalar loss or configured world size.
```

- [ ] CPU probe supplied external data/sidecar functions on representative actual TRAIN cases (one/multiple latent blocks and final adjacent text): compare rendered input IDs, CElabels, preference target IDs, visibility and replay positions. Record source hashes, tested sampleIDs and mismatches. Compare common-input loss/gradients to external snapshot. Any mismatch must be explained/fixed before fullrun.
- [ ] Implement opt-in observation only as needed to witness the actual4-rank trainer/DeepSpeed path. Fixed16 examples must be heterogeneous in lengths/latent counts. Include a final partial group (e.g. retained20 produces16+4, with actual balanced padding recorded); infer actual groups from observed sample IDs, not assumed order. Capture effective gradients and parameter deltas without changing normal production computation. Bound any copied reference state; do not duplicate full-model optimizer state unnecessarily. If direct runtime reference cannot be made trustworthy, escalate with exact missing evidence; do not replace it with a configured-world-size assertion.
- [ ] Before fullrun, use existing coordinator for at most one focused verification campaign covering update arithmetic/partial behavior and longest retained sequence/block-count inputs under selected production flags. Verify allfour ranks, actual sampler retainedcoverage/padding, finitegradients and no OOM. Keep full212 launch blocked if evidence fails. No automatic global-checkpoint/batch/precision fallback.
- [ ] Record test RED/GREEN and commit implementation; actual root-controlled coordinator execution follows independent review.

## Task 5: Fresh verification and authorized experiment execution

**Files:** Runtime control/evidence only in logs/results; no new generalized orchestration. Root may reuse the proven detached campaign command pattern.

**Interfaces:** Consume approved/reviewed commits and Task4 evidence. Produce Stage1 and Stage2 TEST400 summaries plus run configuration. The human retains merge/PR/worktree disposition decisions.

- [ ] Root runs final whole-branch review and resolves required fixes via one fixer, then fresh claim-specific verification. Read finishing and verification skills before selecting final test command; use one full-suite run for overlapping test-pass claim. Baseline failures must be reported and classified, never silently counted as pass.
- [ ] Synchronize the isolated committed checkout to peer using existing safe workflow; preserve canonical dirty trees. Read canonical defaults/multinode guide. Record this reproduction's explicit beta0.1 override after sourcing environments on bothnodes. Dry-run, matching code/environment/inputs andresourcepreflight required.
- [ ] Execute reviewed Task4 distributed witnesses. Failure stops fullrun for targeted diagnosis; do not fabricate success.
- [ ] Launch fresh run with212updates/save212 and approved flags, then preparecheckpoint212 undercheckpoints/evaluation-staging. Verify actualfour-rank participation before reporting trainingstarted.
- [ ] Evaluate approved Stage1 initialization and Stage2checkpoint212 under identical external_full_latent_block policy, repetition1.0, cap128000, greedy4096, TEST400. Existing four-shard evaluator and scorer; no gold/sequence_plan in generation.
- [ ] Verify unique400coverage, source gold, generatedsuffix rescoring, model identity, finitefourrank training/exit evidence. Report within-local-initialization delta and external216/400 comparison with explicit initialization/topology/provenance limits; never infer a cause from one accuracy difference.

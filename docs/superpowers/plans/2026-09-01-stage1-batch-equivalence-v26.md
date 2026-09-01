# Stage 1 B1/B2 Equivalence v26 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the partial true-batch branch so heterogeneous B=2 execution is semantically equivalent to independently reset B=1 execution for the frozen-EMA DoRA configuration, without launching an epoch.

**Architecture:** The production collator and EMA target builder keep exact sample ownership. The trainer builds original-image and teacher-latent replacements samplewise, executes one packed decoder forward, reduces CE/alignment per sample, and keys all randomness by immutable sample occurrence. Tiny production-path tests precede one bounded actual-3B common-state comparison.

**Tech Stack:** Python 3.11, PyTorch 2.7.1, local Transformers/Qwen2.5-VL, PEFT/DoRA, TRL Trainer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-stage1-batch-equivalence-design.md` Sections 4-11 as amended by Section 15

**Execution order:** Start only after every task in `docs/superpowers/plans/2026-09-01-stage1-evaluation-repair.md` is complete and committed. This plan stops after the actual-3B probe and report; it never launches an epoch.

## Global Constraints

- Continue from branch commit `9ec608b`; the canonical schedule, manifest, sample-stable dropout, samplewise preprocessing, and samplewise teacher-target commits are existing dependencies, not work to redo.
- Use only `./venv/ilvr-venv` and `PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src"`.
- Preserve one true packed student decoder forward for B=2; do not simulate B=2 with two decoder forwards.
- Frozen original-image visual execution and EMA-teacher execution are samplewise and match independent B=1 ownership/order.
- Both B=1 and B=2 use one per-sample FP32 CE implementation and one per-sample alignment reduction.
- DoRA dropout and alignment noise are keyed by `(base_seed, optimizer_global_step, canonical_slot_id, stable_site_name)`.
- Frozen EMA `tau=1` is the normative two-step common-state gate. Moving EMA may be compared only after a common reset or labeled a trajectory witness.
- Actual-3B heterogeneous B=2 comparisons use a common state and a B=1 accumulated-gradient reference: run each B=1 row forward/backward without stepping, average its named gradients, then perform exactly one optimizer/scheduler step. Never compare B=2's one step to two separate B=1 optimizer steps.
- Run focused tests and bounded probes only. Do not launch an epoch or save a trained checkpoint.

---

### Task 1: Pack Samplewise Student Vision into One Decoder Forward

**Files:**
- Modify: `src/stage1/trainer.py`
- Modify: `tests/stage1/test_trainer.py`
- Modify: `tests/stage1/test_qwen_latent_injection.py`

**Interfaces:**
- Produces `Stage1PackedStudentInputs(inputs_embeds, position_ids, attention_mask, labels, replacement_mask)`.
- Produces `_build_stage1_student_inputs(model, working_inputs, metadata, targets) -> Stage1PackedStudentInputs`.
- Calls frozen `qwen.visual` once per metadata row under `torch.no_grad()` and the composed PEFT decoder exactly once for the packed batch.

- [ ] **Step 1: Write causal RED tests**

Assert event order and forwarding:

```python
assert events == [
    "student_visual_row_0",
    "student_visual_row_1",
    "student_decoder_batch_2",
]
assert forwarded["pixel_values"] is None
assert forwarded["latent_hidden_states"] is None
assert forwarded["inputs_embeds"] is packed.inputs_embeds
```

Add a B=1 legacy witness that the packed inputs/logits equal the old B=1 forward on a real tiny Qwen fixture. Add gradients proving overwritten original-image and teacher-latent positions do not update their token embedding rows, while text and supervised latent boundary rows retain finite gradients.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py -k 'student_visual or packed or one_student_forward' \
  tests/stage1/test_qwen_latent_injection.py -q
```

- [ ] **Step 3: Build detached row-owned replacements**

Use the metadata's `original_pixel_range`, `original_grid_range`, and `unpadded_length`. For each row, slice its pixels/grids, call `qwen.visual` once, validate feature count against that row's image-token positions, and write detached values to replacement storage. Write clean detached EMA target values at `targets.latent_mask` positions in row-major order.

Compose without in-place mutation of trainable embeddings:

```python
token_embeds = qwen.model.embed_tokens(input_ids)
replacement_embeds = torch.zeros_like(token_embeds).detach()
inputs_embeds = torch.where(
    replacement_mask.unsqueeze(-1), replacement_embeds, token_embeds
)
```

Compute position IDs from original `input_ids`, concatenated `image_grid_thw`, and attention mask before calling the wrapper with `input_ids=None`, `inputs_embeds=inputs_embeds`, explicit `position_ids`, `pixel_values=None`, `image_grid_thw=None`, and `latent_hidden_states=None`.

- [ ] **Step 4: Run focused tests and confirm GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py -k 'student_visual or packed or one_student_forward or helper_images_never' \
  tests/stage1/test_qwen_latent_injection.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/stage1/trainer.py tests/stage1/test_trainer.py \
  tests/stage1/test_qwen_latent_injection.py
git commit -m "fix: pack samplewise Stage 1 visual inputs"
```

### Task 2: Unify Per-Sample CE and Alignment Noise

**Files:**
- Modify: `src/stage1/equivalence.py`
- Modify: `src/stage1/trainer.py`
- Modify: `tests/stage1/test_equivalence.py`
- Modify: `tests/stage1/test_trainer.py`

**Interfaces:**
- Produces `stage1_per_sample_ce(logits, labels) -> tuple[Tensor, Tensor]` as `(batch_mean, row_losses[B])`.
- Produces `alignment_noise_rows(metadata, selected_counts, hidden_size, *, base_seed, optimizer_global_step, device) -> tuple[Tensor, ...]`.
- Produces `stage1_per_sample_alignment(predicted, clean_targets, latent_mask, noise_rows) -> tuple[Tensor, Tensor]` as `(batch_mean, row_losses[B])`.

- [ ] **Step 1: Write unequal-row RED tests**

Use one row with one valid CE token/one latent and another with three valid CE tokens/three latents. Compare B=2 to the arithmetic mean of independently reset B=1 calls with identical explicit sample keys:

```python
torch.testing.assert_close(loss_b2, (loss_b1_a + loss_b1_b) / 2, rtol=1e-6, atol=1e-7)
torch.testing.assert_close(grad_b2, (grad_b1_a + grad_b1_b) / 2, rtol=1e-6, atol=1e-7)
```

Assert both B=1 and B=2 remove `labels` from the model forward. Preserve a B=1 golden test using the same sample-keyed noise values rather than global `randn_like` order.

- [ ] **Step 2: Run loss tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py -k 'equal_sample or unequal_latent or legacy_golden' \
  tests/stage1/test_equivalence.py -k 'noise' -q
```

- [ ] **Step 3: Use one CE path for every batch size**

Always omit labels from the student forward. For each row, shift logits/labels, calculate FP32 cross-entropy sums in ascending 1,024-token chunks, divide by that row's non-ignored token count, then mean the row scalars. Reject a zero-valid-token row. Delete the B=1 native-loss branch and `model_ce_loss` argument.

- [ ] **Step 4: Draw and reduce alignment noise per sample**

For each metadata row, derive the noise seed with stable site name `stage1.alignment_noise`. Draw FP32 normal values for every latent destination position in block/position order before selected-count slicing, then gather selected positions. Add `0.01 * noise` only to the detached alignment target. Compute cosine mean within each sample, then mean sample losses. Clean targets used as decoder replacements remain unperturbed.

- [ ] **Step 5: Run all trainer/equivalence tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence.py \
  tests/stage1/test_trainer.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/stage1/equivalence.py src/stage1/trainer.py \
  tests/stage1/test_equivalence.py tests/stage1/test_trainer.py
git commit -m "fix: equalize Stage 1 sample loss weighting"
```

### Task 3: Bind Sample-Stable Dropout Through the Real Training Step

**Files:**
- Modify: `src/stage1/trainer.py`
- Modify: `tests/stage1/test_trainer.py`

**Interfaces:**
- Overrides `CustomTrainerStage1.training_step(model, inputs, num_items_in_batch=None)` only to activate one `Stage1StochasticContext` around `super().training_step(...)`.
- The context rows are ordered `(canonical_slot_id, unpadded_length)` from `stage1_batch_metadata`; `optimizer_global_step == self.state.global_step`.
- Cleanup occurs in `finally`, including exceptions and gradient-checkpoint recomputation.

- [ ] **Step 1: Write real-training-step RED tests**

Call Trainer `training_step`, assert every registered DoRA dropout site receives exactly one original invocation and at most one recompute invocation, and assert no context remains afterward. Add an injected forward exception and backward/recompute exception; both must clean the context. Assert stale contexts cannot be reused.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_trainer.py -k 'stochastic_context or checkpoint' -q
```

- [ ] **Step 3: Implement the narrow override**

Validate every metadata row has non-null `canonical_slot_id`. Build `Stage1StochasticContext(base_seed=self.args.seed, optimizer_global_step=self.state.global_step, rows=..., expected_sites=tuple(model.stage1_stochastic_sites))`, activate it, and return `super().training_step(...)`. Do not reproduce Trainer loss scaling, gradient accumulation, backward, clipping, or optimizer logic.

- [ ] **Step 4: Run trainer and checkpoint-context tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py \
  tests/stage1/test_equivalence.py -k 'stochastic or checkpoint' -q
```

- [ ] **Step 5: Commit**

```bash
git add src/stage1/trainer.py tests/stage1/test_trainer.py
git commit -m "fix: bind Stage 1 stochastic context to training steps"
```

### Task 4: Prove Tiny Two-Step B2/B1 Common-State Equivalence

**Files:**
- Create: `tests/stage1/test_equivalence_integration.py`
- Modify: `tests/stage1/helpers.py`

**Interfaces:**
- Executes the production collator, target builder, trainer `training_step`, optimizer step, scheduler step, and frozen EMA path.
- Compares one heterogeneous B=2 microbatch to two independently reset B=1 microbatches at two optimizer steps with `tau=1`.

- [ ] **Step 1: Write the two-step RED witness**

Use different sequence lengths and latent-block counts. At common `S0`, run each B=1 row forward/backward without stepping, divide/sum named row gradients into their arithmetic mean, then perform exactly one optimizer and scheduler step; this is the B=1 reference `S1`. B=2 also performs exactly one step from an independent copy of `S0`. Repeat step 2 from independent copies of one common accepted `S1`, not from separately drifted arm states. Capture exact input IDs, attention, labels, masks, target ownership, occurrence IDs, and dropout/noise keys. Compare row logits, CE, alignment, total loss, named gradients, trainable parameters, optimizer moments/step counters, scheduler epoch/LR, and frozen teacher state.

Floating assertions are:

```python
torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
```

Integer, boolean, name, ordering, counter, and state-key assertions are exact. Reset model, optimizer, scheduler, EMA, and RNG/context state independently for every B=1 row/arm.

- [ ] **Step 2: Run the witness and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_equivalence_integration.py -q
```

- [ ] **Step 3: Add fixture support only**

Expose deterministic row-owned visual values, heterogeneous metadata, named parameter/state capture, and independently constructed optimizer/scheduler state. Do not add a second production path or relax tolerance.

- [ ] **Step 4: Run the focused Stage 1 suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_equivalence_integration.py \
  tests/stage1/test_equivalence.py \
  tests/stage1/test_data.py \
  tests/stage1/test_targets.py \
  tests/stage1/test_peft.py \
  tests/stage1/test_trainer.py \
  tests/stage1/test_qwen_latent_injection.py -q
```

- [ ] **Step 5: Commit**

```bash
git add tests/stage1/test_equivalence_integration.py tests/stage1/helpers.py
git commit -m "test: witness Stage 1 B2 and B1 update equivalence"
```

### Task 5: Run One Actual-3B Heterogeneous Common-State Probe

**Files:**
- Create: `src/stage1/batch_equivalence_diagnostic.py`
- Create: `tests/stage1/test_batch_equivalence_diagnostic.py`
- Create at runtime: `results/stage1-batch-equivalence-v26-<timestamp>/...`
- Create at runtime: `logs/stage1-batch-equivalence-v26-<timestamp>/...`

**Interfaces:**
- CLI requires `--model-dir`, `--base-model-dir`, `--train-jsonl`, `--image-root`, `--source-indices`, `--output-json`, and `--cache-dir`; it loads one 3B Stage 1 DoRA state and executes bounded common-state B=2/B=1 comparisons without saving model state.
- Produces a finite JSON report with exact discrete comparisons and per-tensor numerical metrics.

- [ ] **Step 1: Write reducer/schema RED tests**

Require stable sorted tensor names, exact state-key comparison, symmetric relative L2, cosine, max absolute difference, and an explicit verdict per category. Reject a report that omits logits, CE, alignment, total loss, gradients, parameters, optimizer state, scheduler state, or EMA state.

- [ ] **Step 2: Run reducer tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_batch_equivalence_diagnostic.py -q
```

- [ ] **Step 3: Implement the bounded actual-3B probe**

Use TRAIN indices `0,926` (one versus six latent blocks) as the fixed heterogeneous pair. Assign canonical slot IDs equal to those source indices, base seed `42`, and optimizer steps `0` then `1`. Load the prepared adapter at `checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga4-e1-egb16-20260901-090240/prepared-checkpoint-215` over base `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct` as common `S0`.

For each update, B=1 runs both rows from the same common state without stepping, averages named gradients, and performs one AdamW/scheduler step. B=2 performs one packed step from an independent copy. After step 1, use the B=1 accumulated reference's complete model/optimizer/scheduler state as the common reset for both step-2 arms. Use frozen EMA `tau=1`, so teacher state remains exact and its update count does not advance.

Both arms use AdamW with learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, global-norm clipping `1.0`, and the linear schedule with `10` warmup steps and `216` total steps. Capture accumulated gradients before clipping; call optimizer then scheduler exactly once per compared update.

Exact categories: input IDs, labels, attention, latent masks, selected counts/indices, occurrence/stochastic keys, parameter names/shapes, optimizer state keys/step counters, scheduler step/LR, and EMA state. Samplewise visual/teacher tensors require `rtol=1e-6, atol=1e-6`. Scalar CE/alignment/total losses require `rtol=1e-6, atol=1e-7`. For each nonzero vector category—row logits, final decoder hidden rows, accumulated pre-clip gradients, student deltas, Adam first moments, and Adam second moments—require symmetric relative L2 `<= 2.0e-2` and cosine `>= 0.9998`; both-zero vectors match exactly and a one-sided zero fails. Do not save a checkpoint and do not evaluate quality accuracy.

Optionally record moving EMA only by resetting both arms to one independently constructed common `S1/E1` before step 2; otherwise label it `trajectory_witness` and exclude it from the common-state verdict.

- [ ] **Step 4: Run focused tests once, then the actual-3B probe once**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_batch_equivalence_diagnostic.py \
  tests/stage1/test_equivalence_integration.py \
  tests/stage1/test_trainer.py -q
```

Then run exactly:

```bash
stage1_batch_suffix=$(date -u +%Y%m%d-%H%M%S)
BATCH_RUN_ID="stage1-batch-equivalence-v26-$stage1_batch_suffix"
mkdir -p "$PWD/results/$BATCH_RUN_ID" "$PWD/logs/$BATCH_RUN_ID"
PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/python -m stage1.batch_equivalence_diagnostic \
  --model-dir "$PWD/checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga4-e1-egb16-20260901-090240/prepared-checkpoint-215" \
  --base-model-dir "$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct" \
  --train-jsonl "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data/images_comt" \
  --source-indices 0,926 \
  --cache-dir "$PWD/cache" \
  --output-json "$PWD/results/$BATCH_RUN_ID/report.json" \
  >"$PWD/logs/$BATCH_RUN_ID/probe.log" 2>&1
```

Record peak GPU memory and wall time. Stop after reporting; do not launch an epoch.

- [ ] **Step 5: Commit code and tests, not runtime artifacts**

```bash
git add src/stage1/batch_equivalence_diagnostic.py \
  tests/stage1/test_batch_equivalence_diagnostic.py
git commit -m "test: compare Stage 1 B2 and B1 on actual 3B state"
```

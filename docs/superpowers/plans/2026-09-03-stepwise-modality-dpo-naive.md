# Step-wise Modality DPO Naive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan task by task.

**Goal:** Implement and verify milestone steps 1–8 of the approved step-wise modality DPO design, ending with one finite real-data CUDA optimizer step.

**Architecture:** Preserve the existing Stage 2 clean path. Add a `step_preference` package for span metadata, negative artifacts, losses, and attention-edge ablation. Extend the Qwen Stage 2 final decoder pass with opt-in additive masks and selective logits, then integrate a fresh Stage 2B objective initialized from Stage 2A weights. Packed/shared-prefix work is explicitly deferred until this milestone is green.

**Tech Stack:** Python 3.11, PyTorch 2.7, vendored Transformers/Qwen2.5-VL, pytest, Hugging Face Trainer, CUDA.

**Design:** `docs/superpowers/specs/2026-09-03-stepwise-modality-dpo-design.md`

**Execution constraints:** Run Python through `./venv/ilvr-venv`; set `PYTHONPATH="$PWD/transformers/src:$PWD/src"`; store weights in `checkpoints/`, results in `results/`, and logs in `logs/`. Do not begin packed milestone steps 9–11 until Task 10 passes.

## Normative contracts used by every task

The input row has `image_input: list[str]`, `text_input: str`,
`sequence_plan: list[dict]`, and `original_final_answer: str`. Plan entries are
`{"type":"text","content":str}` or
`{"type":"latent","helper_image":str}`. With step preference enabled, every
latent entry must have an immediately preceding non-empty text entry and an
immediately following non-empty text entry. That triple is one reasoning unit;
`step_id` is its zero-based latent ordinal and `helper_image_idx` is the same
sample-local helper ordinal.

Use a non-empty explicit `sample_id` when present. Otherwise hash the UTF-8
bytes of `json.dumps(row, sort_keys=True, separators=(",", ":"),
ensure_ascii=False)` with SHA-256. Assistant character spans and token spans
are half-open. Text spans include only the corresponding assistant content;
latent spans include only latent pad tokens, excluding latent start/end tokens.
Any token whose offset crosses a plan boundary is an error.

For target token span `[a,b)`, the causal decoder query rows are `[a-1,b-1)`.
V-off blocks those query rows from the current latent pad span; T-off blocks
them from the immediately previous text span. The additive mask is
`[1,1,L,L]`, uses the model compute dtype, contains zero at visible edges and
`torch.finfo(dtype).min` at blocked/otherwise-masked edges, and preserves the
existing causal and padding restrictions. Ablation is applied only to the final
decoder call after the clean latent rollout. Each step and modality gets its
own naive forward.

For completion token set `S`, the block score is
`sum(log_softmax(logits.float())[S, token_ids[S]]) / len(S)`. A valid pair loss
is
`-logsigmoid(beta * ((policy_chosen - policy_rejected) -
(reference_chosen - reference_rejected)))`. Overall DPO is the sum over all
valid V/T pairs divided by the total valid pair count; an empty set returns
`clean_ce * 0.0`. The per-token entropy term is
`relu(entropy_margin - (entropy_ablated - entropy_clean))`. Aggregate entropy
by target-token count across all steps and both modalities. Reference artifact
scores are detached.

Each negative JSONL row contains `sample_id`, `step_id`, `chosen_text`,
`rejected_visual_text`, `rejected_textual_text`, two validity booleans,
`reference_chosen_score`, and the corresponding visual/text rejected reference
scores. Invalid rejected scores may be null. Whitespace normalization is
`" ".join(text.split())`. Token minimums use scoreable completion tokens from
the combined clean-prefix-plus-completion tokenization. EOS and chat/control
terminators are not completion tokens. The manifest records the source file's
raw-byte SHA-256, selected zero-based row indices, resolved checkpoint path,
ordered `(relative path, size, SHA-256)` tuples for consumed checkpoint files,
tokenizer/config identity, latent size, generation parameters, and negative
JSONL raw-byte SHA-256. Duplicate, missing, stale, structurally inconsistent,
or non-finite data fails closed.

The new public training flags are `--step_preference_enabled`,
`--step_preference_mode {stage2a,stage2b}`, `--step_negative_path`,
`--step_negative_manifest`, `--step_dpo_beta`, `--step_dpo_weight`,
`--step_entropy_weight`, `--step_entropy_margin`, `--stage2a_ratio`,
`--stage2_total_steps`, and `--stage2a_checkpoint_path`. Defaults are disabled,
mode `stage2a`, beta `0.1`, DPO weight `1.0`, entropy weight `0.1`, entropy
margin `0.1`, and ratio `0.5`; paths and total steps default to null. Stage 2B
requires its checkpoint and both negative paths, rejects
`flash_attention_2`, and rejects simultaneous `resume_from_checkpoint_path`.
`stage2a_checkpoint_path` initializes model/processor weights through the
existing checkpoint resolver but always yields `trainer_resume_path=None`.

The generator CLI writes exactly the JSONL named by `--output_path` and the
sidecar named by `--manifest_path`. Its required arguments are `--data_path`,
`--image_root`, `--stage2a_checkpoint_path`, `--output_path`, and
`--manifest_path`; it also accepts `--max_samples`, `--latent_size` (default
8), `--min_tokens` (default 2), `--max_new_tokens` (default 32),
`--attention_implementation` (`eager` or `sdpa`, default `sdpa`), and `--device`
(default `cuda:0`). The real-data gate fails if the initial two-row run and the
single permitted eight-row retry both yield zero valid pairs.

## Task 1: Baseline and shifted-mask first nail

**Files:**

- Create: `tests/step_preference/test_decoder_mask_probe.py`
- Read: `tests/stage2/helpers.py`
- Read: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

- [ ] Run the existing Stage 2 suite once and retain its log.

```bash
mkdir -p logs/step-preference-phase-a
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/stage2 | tee logs/step-preference-phase-a/baseline.log
```

- [ ] Add a decoder-only test using `tiny_stage2_qwen(attn_implementation="eager")`, fixed embeddings, fixed position IDs, and a 4-D additive causal mask. Block only physical query rows corresponding to a target span from attending to a chosen earlier span.
- [ ] Assert that the target distribution changes, rows earlier than the target do not change, and the same position IDs are used in both calls.
- [ ] Run the probe and stop implementation if the vendored decoder ignores the mask.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_decoder_mask_probe.py
```

- [ ] Commit the runtime witness.

```bash
git add tests/step_preference/test_decoder_mask_probe.py logs/step-preference-phase-a/baseline.log
git commit -m "test: prove shifted decoder edge masking"
```

## Task 2: Stable identity and reasoning-plan spans

**Files:**

- Create: `src/step_preference/__init__.py`
- Create: `src/step_preference/spans.py`
- Create: `tests/step_preference/test_spans.py`
- Modify: `src/stage2/data.py`

- [ ] Write failing tests for explicit and SHA-256 sample IDs, duplicate detection, exact immediate `text -> latent -> text` validation, empty text rejection, zero-based latent ordinals, helper ordinals, and half-open assistant character spans.
- [ ] Implement immutable `PlanCharSpan`, `ReasoningStepPlan`, and `ReasoningStepSpan` records plus `canonical_sample_id`, `reject_duplicate_sample_ids`, and `build_assistant_plan`.
- [ ] Make the existing Stage 2 assistant-text construction delegate to `build_assistant_plan` without changing disabled-path output.
- [ ] Run focused tests and existing Stage 2 data tests.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_spans.py tests/stage2/test_data.py
```

- [ ] Commit.

```bash
git add src/step_preference src/stage2/data.py tests/step_preference/test_spans.py
git commit -m "feat: preserve reasoning step identity"
```

## Task 3: Offset-mapped token spans and collator compatibility

**Files:**

- Modify: `src/step_preference/spans.py`
- Modify: `src/stage2/data.py`
- Create: `tests/step_preference/test_collator_spans.py`

- [ ] Write failing tests covering exact rendered-text offsets, image-placeholder expansion, exclusion of latent start/end controls, rejection of boundary-crossing tokens, and tensor equality between the disabled collator and the pre-change processor result.
- [ ] Implement `expand_qwen_image_placeholders` and `map_reasoning_step_spans`. Tokenize the exact expanded full chat string with offsets and assert that all normal tensor fields equal the established processor output.
- [ ] Add `step_preference_enabled=False` to `Stage2DataCollator`. Only the enabled path adds `sample_ids`, `reasoning_step_spans`, and private `step_preference` metadata.
- [ ] Run the focused collator tests and all existing Stage 2 data tests.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_collator_spans.py tests/stage2/test_data.py tests/stage2/test_data_mapping.py
```

- [ ] Commit.

```bash
git add src/step_preference/spans.py src/stage2/data.py tests/step_preference/test_collator_spans.py
git commit -m "feat: map reasoning spans to model tokens"
```

## Task 4: Negative artifact and frozen-reference contract

**Files:**

- Create: `src/step_preference/negatives.py`
- Create: `tests/step_preference/test_negatives.py`

- [ ] Write failing tests for exact, whitespace-normalized, empty, and minimum-token filters; duplicate `(sample_id, step_id)` entries; non-finite scores; source/checkpoint/artifact hashes; and manifest mismatch rejection.
- [ ] Implement immutable `StepNegative` and `NegativeManifest` records plus `filter_negative`, `write_negative_artifact`, and `load_negative_store`.
- [ ] Store chosen/rejected mean reference log-probabilities, validity flags, generation settings, source hash, checkpoint-load-file hashes, and JSONL hash. Reject stale or malformed artifacts before training.
- [ ] Run focused tests.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_negatives.py
```

- [ ] Commit.

```bash
git add src/step_preference/negatives.py tests/step_preference/test_negatives.py
git commit -m "feat: define step negative artifacts"
```

## Task 5: Step-DPO and entropy-margin losses

**Files:**

- Create: `src/step_preference/dpo_loss.py`
- Create: `src/step_preference/entropy_loss.py`
- Create: `tests/step_preference/test_dpo_loss.py`
- Create: `tests/step_preference/test_entropy_loss.py`

- [ ] Write failing DPO tests against a direct scalar formula, mean-token normalization, unequal visual/text pair counts, invalid-pair omission, and differentiable zero loss.
- [ ] Implement `mean_completion_logps`, `dpo_pair_losses`, and `aggregate_step_dpo`. Aggregate all valid modality pairs with one global denominator while reporting modality-specific diagnostics.
- [ ] Write failing entropy tests using manual categorical distributions, token-weighted cross-step aggregation, BF16 inputs promoted to FP32, an inactive margin, and finite gradients.
- [ ] Implement `token_entropies` and `entropy_margin_loss` for `relu(H_clean + margin - H_ablate)`.
- [ ] Run focused tests.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_dpo_loss.py tests/step_preference/test_entropy_loss.py
```

- [ ] Commit.

```bash
git add src/step_preference/dpo_loss.py src/step_preference/entropy_loss.py tests/step_preference/test_dpo_loss.py tests/step_preference/test_entropy_loss.py
git commit -m "feat: add step preference objectives"
```

## Task 6: Naive modality ablation and selective Qwen logits

**Files:**

- Create: `src/step_preference/ablation.py`
- Create: `tests/step_preference/test_ablation.py`
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Modify: `tests/stage2/test_stage2_modeling.py`

- [ ] Write failing tests for the logical-target to physical-query shift `[a,b) -> [a-1,b-1)`, V-off and T-off blocked rectangles, causal/padding preservation, invalid spans, dtype minimum values, and rejection of `flash_attention_2`.
- [ ] Implement `StepAblationSpec` and `build_additive_step_attention_mask` for eager/SDPA 4-D masks.
- [ ] Add opt-in model arguments `stage2_final_attention_mask`, `stage2_logit_positions`, and `stage2_selective_logits_only`, plus output field `stage2_selected_logits`.
- [ ] Apply the custom mask only to the post-rollout final decoder pass. Preserve the normal 2-D mask for latent rollout and preserve byte-for-byte default control flow when new arguments are absent.
- [ ] Run ablation and Stage 2 modeling tests.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_ablation.py tests/stage2/test_stage2_modeling.py
```

- [ ] Commit.

```bash
git add src/step_preference/ablation.py tests/step_preference/test_ablation.py transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py tests/stage2/test_stage2_modeling.py
git commit -m "feat: add naive modality ablation forwards"
```

## Task 7: Offline greedy negatives and reference scoring

**Files:**

- Modify: `src/step_preference/negatives.py`
- Create: `scripts/generate_step_negatives.py`
- Create: `tests/step_preference/test_negative_generation.py`

- [ ] Write failing tiny-model tests for deterministic no-cache greedy decoding, visual/text masks, EOS termination, maximum length, full-string tokenization, exclusion of EOS/control tokens from completion scores, and clean-context reference scoring.
- [ ] Implement `greedy_generate_step_negative` and `score_step_completion`. Generate from the Stage 2A snapshot under each ablated context and score chosen/rejected completions under the clean context.
- [ ] Add the direct CLI with dataset, image root, Stage 2A checkpoint, output JSONL, manifest, sample limit, latent size, minimum/maximum completion length, attention implementation, and device arguments.
- [ ] Load the frozen checkpoint through the existing Stage 2 inference loader, write one JSONL plus one manifest, validate them once, and exit nonzero if no valid pair survives.
- [ ] Run focused tests and CLI help.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_negative_generation.py
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python scripts/generate_step_negatives.py --help
```

- [ ] Commit.

```bash
git add src/step_preference/negatives.py scripts/generate_step_negatives.py tests/step_preference/test_negative_generation.py
git commit -m "feat: generate offline step negatives"
```

## Task 8: Stage 2B configuration, lifecycle, and naive objective

**Files:**

- Create: `src/step_preference/config.py`
- Modify: `src/utils_deepseed.py`
- Modify: `src/stage2/checkpointing.py`
- Modify: `src/stage2/data.py`
- Modify: `src/stage2/trainer.py`
- Modify: `src/main.py`
- Modify: `run_stage2_training.sh`
- Create: `tests/step_preference/test_config.py`
- Create: `tests/step_preference/test_stage2b_trainer.py`
- Modify: `tests/stage2/test_checkpointing.py`
- Modify: `tests/stage2/test_launcher.py`

- [ ] Write failing tests for positive phase budgets, invalid ratios, new CLI defaults, fresh Stage 2B initialization without optimizer/trainer resume, strict negative lookup, disabled-path equivalence, pair-count aggregation, and finite diagnostic counters.
- [ ] Implement immutable `StepPreferenceConfig` and `split_stage2_budget`.
- [ ] Add the exact opt-in Stage 2B arguments and defaults from the normative contracts. Propagate their environment equivalents through `run_stage2_training.sh` without changing disabled launcher output.
- [ ] Resolve `stage2a_checkpoint_path` with the existing model loader, then clear only `trainer_resume_path`; keep actual Stage 2B resume as a distinct mutually exclusive path.
- [ ] Tokenize matched rejected branches in the collator under private `step_preference` metadata.
- [ ] In `Stage2Trainer.compute_loss`, run one clean trajectory, one V-off and one T-off branch per reasoning step, and one clean-context rejected forward per valid pair. Combine clean CE, globally averaged DPO, and token-weighted entropy margin. Log loss components, valid pair counts, branch forwards, and scored tokens.
- [ ] Keep the existing trainer call unchanged when the feature is disabled.
- [ ] Run focused lifecycle/trainer tests and the existing Stage 2 checkpoint tests.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_config.py tests/step_preference/test_stage2b_trainer.py tests/stage2/test_checkpointing.py
```

- [ ] Commit.

```bash
git add src/step_preference/config.py src/utils_deepseed.py src/stage2/checkpointing.py src/stage2/data.py src/stage2/trainer.py src/main.py run_stage2_training.sh tests/step_preference/test_config.py tests/step_preference/test_stage2b_trainer.py tests/stage2/test_checkpointing.py tests/stage2/test_launcher.py
git commit -m "feat: integrate naive stage 2b training"
```

## Task 9: Focused integration verification

**Files:**

- Create: `scripts/verify_step_preference_phase_a.py`
- Create: `tests/step_preference/test_phase_a_verifier.py`

- [ ] Write a failing verifier test covering finite total/component losses, finite gradient norm, at least one surviving pair, at least one optimizer step, and a loadable checkpoint.
- [ ] Implement a single-summary verifier that reads the negative manifest, Trainer metrics, and saved checkpoint, then writes one JSON report and exits nonzero on any failed gate.
- [ ] Run one focused combined suite; do not repeat it if green.

```bash
mkdir -p logs/step-preference-phase-a
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference tests/stage2 | tee logs/step-preference-phase-a/focused-suite.log
```

- [ ] Commit.

```bash
git add scripts/verify_step_preference_phase_a.py tests/step_preference/test_phase_a_verifier.py logs/step-preference-phase-a/focused-suite.log
git commit -m "test: verify naive step preference integration"
```

## Task 10: Real-data CUDA one-step gate

**Artifacts:**

- Read checkpoint: `checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827`
- Read data: `data/TRAIN.jsonl`
- Read images: `data/images_comt`
- Write negatives: `results/step-preference-phase-a-<run-id>/`
- Write model: `checkpoints/step-preference-phase-a-<run-id>/`
- Write logs: `logs/step-preference-phase-a-<run-id>/`

- [ ] Use a fresh explicit run ID. Generate offline negatives on CUDA from two real samples with greedy maximum 32 new tokens and minimum two completion tokens. If zero valid pairs survive, make exactly one fresh run on eight samples.

```bash
run_id=step-preference-phase-a-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "results/$run_id" "logs/$run_id"
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python scripts/generate_step_negatives.py \
  --data_path data/TRAIN.jsonl \
  --image_root data/images_comt \
  --stage2a_checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --output_path "results/$run_id/negatives.jsonl" \
  --manifest_path "results/$run_id/negative-manifest.json" \
  --max_samples 2 --latent_size 8 --min_tokens 2 --max_new_tokens 32 \
  --attention_implementation sdpa --device cuda:0 \
  2>&1 | tee "logs/$run_id/generate-negatives.log"
```

- [ ] If the generator exits specifically because no pair survived, choose a new run ID and repeat the preceding command once with `--max_samples 8`; any second zero-pair result fails Phase A.
- [ ] Start a fresh Stage 2B policy from the checkpoint weights with `max_steps=1`, at most two training samples, SDPA, and no restored optimizer/Trainer state.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python src/main.py \
  --stage stage2 \
  --load_model_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --stage2a_checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --data_path data/TRAIN.jsonl --image_root data/images_comt \
  --save_model_path "checkpoints/$run_id" --log_file "logs/$run_id/stage2b.log" \
  --step_preference_enabled --step_preference_mode stage2b \
  --step_negative_path "results/$run_id/negatives.jsonl" \
  --step_negative_manifest "results/$run_id/negative-manifest.json" \
  --max_steps 1 --max_train_samples 2 --save_strategy no \
  --gradient_accumulation_steps 1 --per_device_train_batch_size 1 \
  --attention_implementation sdpa --no-gradient_checkpointing \
  --latent_size 8 --max_seq_length_train 8192
```
- [ ] Run the Phase A verifier and require one optimizer step, finite total/CE/DPO/entropy losses, finite gradient norm, at least one valid pair, and a loadable saved checkpoint.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python scripts/verify_step_preference_phase_a.py \
  --negative_manifest "results/$run_id/negative-manifest.json" \
  --negative_path "results/$run_id/negatives.jsonl" \
  --training_log "logs/$run_id/stage2b.log" \
  --checkpoint_path "checkpoints/$run_id" \
  --output_path "results/$run_id/phase-a-verification.json"
```
- [ ] If a runtime failure occurs, retain its log, identify the root cause, add one causal regression test, make the smallest fix, and retry with a new run ID.
- [ ] Record the exact commands and verifier JSON in the run results. Do not claim training effectiveness from this smoke test.
- [ ] Commit only source, tests, and small metadata; do not commit checkpoints or generated runtime payloads.

## Phase A exit gate

Milestone steps 1–8 are green only when Tasks 1–10 pass. Then write the separate packed milestone steps 9–11 plan from the approved design. Until then, no tree-packed/shared-prefix implementation is authorized.

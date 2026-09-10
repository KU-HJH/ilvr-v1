# ILVR visual causal analysis implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add verified, opt-in CoMT visual-content interventions and evaluate the approved three checkpoints without changing training.

**Architecture:** Reuse the existing no-cache continuous controller and scorer. A generation-local policy wraps every decoder call and records free-token decisions; a separate runner owns conditions and JSONL, and CPU metrics own paired comparisons.

**Tech Stack:** repository Python 3.11 environment, PyTorch 2.7.1, vendored Transformers/Qwen2.5-VL, eager and SDPA decoder, FA2 vision, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-ilvr-visual-causal-analysis-design.md` (approved v1).

## Global Constraints

- Use `./venv/ilvr-venv` exclusively, `PYTHONPATH=transformers/src:src:.`, and the existing environment.
- Keep model artifacts under the worktree's `checkpoints/` symlink, datasets under `data/`, outputs under `results/`, execution logs under `logs/`.
- No training changes, no global monkeypatches, no dependency installation, no hardcoded special-token IDs. Preserve existing evaluators and no-policy generation behavior.
- New behavior is opt-in. Full masks preserve causal/padding/sliding-window restrictions and explicit 2D-derived mRoPE; E3 includes START and every PAD query, E4 includes every non-image query from the first decoder call.
- No cross-condition cache or hidden-state reuse. All decoder calls, including rollout and post-END, use the same generation-local policy with `use_cache=False`.
- Approved main panel is existing seed42 100 rows; E5 uses separate matched groups; approved checkpoints are Stage1 step215, Stage2A step212, V-off default step212, precisely listed in spec.
- Decoding: temperature=0, top_p=1, repetition_penalty=1.05, max_new_tokens=4096, max_sequence_length=32768, batch=1, runtime bfloat16.
- Treat native-backend reproduction, same-backend no-op equivalence, tiny numerical masks, real-checkpoint witness, and completed ablation panels as distinct evidence gates.

### Task 1: Generation-local interventions and numerical mask contract

**Files:** Create `src/evaluation/visual_interventions.py`, `tests/evaluation/test_visual_interventions.py`; modify `src/evaluation/continuous_latent.py` only.

**Interfaces:** Produce `VisualIntervention(mode, image_token_id, sliding_window=None)`, generation-local `.events` and `.forward_records`; `.begin(input_ids, token_ids)`, `.register_latent_span(span)`, `.wrap_decoder(language_model)` returning a keyword-callable, and `.process_logits(sequences, raw_logits, decision_logits, token_ids)` returning intervention-adjusted logits. Add optional `intervention=None` to controller `generate`. `begin` resets all per-generation state; wrapper derives current prefix length from received embeddings and mask. Image ID is supplied by runner after tokenizer/config validation. Expose `build_image_attention_mask(validity_mask, image_positions, latent_query_positions, mode, dtype, sliding_window=None)` and `identify_image_positions(input_ids, image_token_id)`. Positions are physical zero-based sequence indices, batch=1.

- [ ] Write tests first for exact allowed-edge matrices with independently enumerated expected edges, including START producer and previous blocks. Example:

```python
image_positions = (1, 2)
latent_query_positions = (4, 5, 6)
# E3: rows 4..6 keys 1,2 blocked; row 3 key1 allowed; causal future remains blocked.
# E4: all non-image rows keys1,2 blocked; image rows retain causal image edges.
```

Test no-policy and baseline mode logits/masks/positions unchanged, only START changes under E2 relative to ordinary decision logits, forced template still inserts K pads/end, and begin prevents state carryover. Use actual controller scripted model helpers from `tests/evaluation/test_continuous_latent.py` where appropriate. Assert all normal/rollout/post-END calls have use_cache=False; injected prior cache is rejected in the wrapper.

- [ ] Run RED and retain output in `logs/visual-causal-implementation-20260910/task1-red.log`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B -m pytest -q tests/evaluation/test_visual_interventions.py
```

- [ ] Implement the minimal hook. Maintain a shared callable in controller; leave direct behavior unchanged when intervention is None. Register new span before rollout. Existing rollout accepts that callable and continues slicing 2D masks normally. Keep explicit position_ids untouched. Compose full masks only for E3/E4. Do not modify vendored modules or pass 4D masks into generic HF generate.

```python
decoder = partial(model.model, generate_mode=True)
if intervention is not None:
    intervention.begin(input_ids, token_ids)
    decoder = intervention.wrap_decoder(decoder)
# Use decoder for the existing text, rollout language_model, and post-END calls.
# Preserve a raw_logits reference before cloning/repetition processing.
# After ordinary PAD/END exclusion:
if intervention is not None:
    logits = intervention.process_logits(sequences, raw_logits, logits, token_ids)
```

Events record suffix-relative free decision position, original START logit, repetition-adjusted START logit, diagnostic untempered softmax probability, rank, best non-control token/logit, pre/post argmax and changed flag. Nonfinite floats serialize as null at the runner boundary. Probabilities are vocabulary diagnostic probabilities, not a probability assigned to continuous V. Forward records store shape/role/position metadata without retaining full attention tensors.

- [ ] Add tiny actual-Qwen eager masked-image-attention probability==0 witness and native-SDPA fixed-grid image-embedding perturbation E4 non-image-hidden/logit invariance, plus unmasked positive control that changes. Exercise recurrence to catch masked-final-only implementations. Preserve position IDs and include an identity/no-op control. E3 is not required to be image-invariant because indirect text paths remain. Tiny eager output_attentions evidence and SDPA evidence must be labeled separately.

- [ ] Run GREEN and existing affected tests once, record commands/output; inspect diff and commit:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B -m pytest -q tests/evaluation/test_visual_interventions.py tests/evaluation/test_continuous_latent.py tests/stage2/test_generation_regression.py tests/stage2/test_eval_integration.py
git add src/evaluation/visual_interventions.py src/evaluation/continuous_latent.py tests/evaluation/test_visual_interventions.py
git commit -m 'Add opt-in visual causal interventions to continuous inference'
```

### Task 2: Paired metrics and counterfactual grouping

**Files:** Create `src/evaluation/visual_metrics.py`, `tests/evaluation/test_visual_metrics.py`.

**Interfaces:** Produce `paired_summary(baseline_rows, condition_rows, *, seed=42, bootstrap_samples=10000)` returning JSON-safe dict, `aggregate_rows(rows)` for one condition, `matched_image_groups(samples)` returning original-index groups of normalized identical questions with differing valid golds, and `matched_group_summary(rows, groups)` for independent group-level image discrimination. Row keys are `index`, `match`, `invalid_gold_reason`, `prediction_normalized`, `generated_suffix_token_ids`, `latent_block_count`, `inference_time_sec`, `category`, optional `error`; runner provides those exact keys.

- [ ] Write meaningful RED tests using the explicit toy transition table:

```python
baseline = [True, True, False, False]
ablated = [True, False, True, False]
# cc=1, cw=1, wc=1, ww=1, delta=0, exact McNemar p=1.
# All identical paired correctness => bootstrap CI [0,0].
# Eight losses and zero gains => exact two-sided p=2/256.
```

Test unpaired/duplicate indices rejected, gold mismatch rejected, errors not silently included as valid pairs, invalid gold reported separately, prediction changes independent of correctness, deterministic bootstrap and reversed-condition delta/CI behavior. Test one identical-question three-row group without counting every directed pair as independent; near-question different choices must not group together.

- [ ] Run RED:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B -m pytest -q tests/evaluation/test_visual_metrics.py
```

- [ ] Implement CPU-only code with no torch/CUDA/model imports. Paired bootstrap resamples paired indices, never each condition independently. Use the exact binomial McNemar probability with b+c discordant pairs, including p=1 if b+c=0. Coverage denominator and valid-gold denominator are separate. Match validity requires identical gold and successful inference. Report counts and exclusions explicitly. Use local seeded RNG; categorize by runner-supplied category. Numeric deltas and CI units are explicitly percentage points.

```python
delta_pp = 100.0 * sum(int(a) - int(b) for b, a in paired_correctness) / len(paired_correctness)
# McNemar: min(1, 2 * sum(comb(n,k) for k in range(min(b,c)+1)) / 2**n).
```

- [ ] Run GREEN, self-review and commit:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B -m pytest -q tests/evaluation/test_visual_metrics.py
git add src/evaluation/visual_metrics.py tests/evaluation/test_visual_metrics.py
git commit -m 'Add paired visual intervention metrics and matched-image analysis'
```

### Task 3: Analysis runner, native-baseline bridge, and operator documentation

**Files:** Create `scripts/analyze_comt_visual.py`, `tests/evaluation/test_visual_analysis_runner.py`, `docs/evaluation/visual-causal-analysis.md`. Use Tasks 1/2 interfaces. No changes to existing evaluator entrypoints.

**Interfaces:** CLI `run` accepts `--model-dir`, optional `--base-model-dir`, `--test-data-path`, `--source-data-path`, optional `--indices` comma-separated original indices, `--output-dir`, `--visual-ablation-mode`, `--attention-backend {native,sdpa}`, optional `--replacement-map` JSON original-index to donor-index, `--max-new-tokens`, `--max-sequence-length`, `--repetition-penalty`, `--temperature`, `--top-p`, optional `--limit`. CLI `compare` accepts baseline/condition JSONL paths and output directory. CLI `groups` accepts source test JSONL and output JSON path under data/. Each run writes `predictions.jsonl` and `metadata.json`; compare writes `summary.json`, `summary.csv`, `analysis.md`. Loading model state never writes under results.

- [ ] Write RED tests with injected lightweight loader/generator seams, but real prompt/scoring and result serialization logic. Assert gold/helper fields do not reach prompt; E1 has no images/grid/placeholders and allows latent generation; image ID derives from tokenizer; raw IDs/START/END positions are saved before text cleanup; actual backend/dtype/config metadata comes from instantiated objects; missing/error rows are explicit; replacement keeps recipient text and uses donor gold only for valid matched question pairs. Baseline equivalence compares raw IDs, prompt identity and scoring against existing evaluator output. Make tests fail for missing functionality before implementation.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B -m pytest -q tests/evaluation/test_visual_analysis_runner.py
```

- [ ] Implement runner using existing artifact loaders and `prepare_evaluation_input`, scorer, token resolver and controller. Native Stage1 loader is eval.py; native Stage2 loader is src/evaluate_deepseed.py; analysis SDPA uses existing mixed loader through eval.py artifact loading. Authenticate module classes for every decoder/vision layer; masked modes reject incompatible decoder modules instead of fallback. Resolve input indices against source rows, keeping stable original IDs for all paired comparisons. Use image-path resolution relative to dataset location. Tensor operations occur only in run, so groups/compare do not load models.

```python
prepared = prepare_evaluation_input(processor, sample, images)
policy = VisualIntervention(mode, resolved_image_id, sliding_window=model.config.sliding_window)
# Move prepared tensors to model device; call ContinuousLatentController.generate.
# Record suffix IDs first; derive positions by tokenizer-resolved IDs; then use existing cleanup/scoring.
```

Preserve native E0 decoder behavior by using no policy when not collecting events if necessary; baseline with observer must also be tested as no-op. A generation-local policy is reset for every sample/condition. Timing surrounds synchronized generation, excluding load/processor time, matching eval.py; record wall time separately. Native Stage2 existing evaluator lacks per-row timing, so do not infer zero as measured runtime. Nonfinite diagnostics are null, not nonstandard JSON numbers. Errors fail run status and remain explicit rows.

- [ ] Write documentation with actual CLI command examples for the spec's Stage1 and Stage2 prepared checkpoints, E0 native/SDPA bridge, E1-E5, group creation, pairing and output schema. Explain E3 START inclusion, residual indirect paths, E4 geometry, E2 budget, invalid golds, and deferred patching. Include no-op and numeric verification commands.

- [ ] Run GREEN and focused integrated suite, `--help`, groups/compare toy smoke, self-review and commit:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B -m pytest -q tests/evaluation/test_visual_interventions.py tests/evaluation/test_visual_metrics.py tests/evaluation/test_visual_analysis_runner.py tests/evaluation/test_continuous_latent.py tests/stage2/test_eval_integration.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B scripts/analyze_comt_visual.py --help
git add scripts/analyze_comt_visual.py tests/evaluation/test_visual_analysis_runner.py docs/evaluation/visual-causal-analysis.md
git commit -m 'Add CoMT visual causal analysis CLI and runbook'
```

### Task 4: Runtime gates, detached campaign, and evidence report

**Files:** Execution scripts/logs in `logs/<run-id>/`; per-example/summary/report outputs in `results/<run-id>/`; selection and E5 group manifests under `data/evaluation/<run-id>/`. Documentation may append observed verification commands to `docs/evaluation/visual-causal-analysis.md`. Do not add an orchestration subsystem.

**Interfaces:** Consume the Task3 CLI and Tasks1/2 tests. Produce native/same-backend bridge evidence, completed condition JSONLs, paired aggregate CSV/JSON and markdown interpretation. Root coordinates resource checks and this runtime task; no expensive inference is authorized to bypass validation gates.

- [ ] Check local/remote GPU/process availability, existing artifact validity and dataset identity. Reuse prepared artifacts and environment, no installation/model copy outside checkpoints. Use original model configs' latent size and exact generation defaults.
- [ ] Run tiny numerical tests from Task1, then one real-checkpoint short sample comparing native evaluator baseline with analysis no-op, and SDPA policy baseline. Compare raw token IDs, prompt identity and extracted score; capture actual modules/dtypes and forward use_cache flags. Exercise E1, E2, E3, E4 on a real sample; masked probabilities are tiny-eager evidence and SDPA effect evidence, not a fabricated attention-map claim.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B scripts/analyze_comt_visual.py run --model-dir checkpoints/evaluation-staging/ours1-stage2a-s1matched-egb16-lr5e5-dora-z2-20260908t022000z-r3/prepared-checkpoint-212 --test-data-path data/TEST.jsonl --source-data-path data/TEST.jsonl --indices 3 --output-dir results/visual-causal-real-smoke-20260910 --visual-ablation-mode baseline --attention-backend sdpa --max-new-tokens 64
```

- [ ] After task review and passing runtime gates, execute native E0 and analysis SDPA E0, then E1/E2/E5, then E3/E4 for three models. Use unchanged main100 selection; E5 supplementary19 can reuse identical donor prompt/image runs after exact input equality verification. Persist explicit commands before detached launch. Do not create artificial new golds or change sample selection using predictions. Native versus SDPA differences are reported, not pooled into intervention deltas.
- [ ] Validate each completed JSONL once for original IDs, expected count, gold, condition, no errors, raw tokens, fresh-state/attention evidence; recompute scores with existing scorer. Generate paired summaries and category results. Do not characterize non-significance as equivalence.
- [ ] Write analysis interpreting E0-E5 through Cases A-E from the request, conditioned on measured results; distinguish computation, direct attention, useful content, preserved paths, backend effects and checkpoint baseline competence. Include commands, observed runtime and remaining uncertainty. Activation patching remains deferred pending these results.
- [ ] Run claim-specific final checks and code review, record `git diff --stat ff53706..HEAD`, report each modified file, output artifacts and execution state. Preserve worktree and experiments pending human branch-integration decision; do not merge or remove user work.

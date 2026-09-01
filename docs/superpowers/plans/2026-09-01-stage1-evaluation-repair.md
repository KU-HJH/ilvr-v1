# Stage 1 Paper-Aligned Evaluation Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Stage 1 answer scoring and replace static latent-pad generation with the paper's continuous hidden-state recurrence, then re-evaluate the existing failed 3B checkpoint once without training.

**Architecture:** Shared `evaluation` modules own answer parsing/scoring, prompt lineage, result rows, and the stage-neutral continuous-latent controller. `eval.py` and `src/evaluate_deepseed.py` remain thin entry points; Stage 1 keeps its validated loader and stage tag, while Stage 2 keeps its strict resolver. Small deterministic diagnostics validate recurrence and the latent-boundary path before the existing four-shard evaluator is launched.

**Tech Stack:** Python 3.11, PyTorch 2.7.1, local Transformers/Qwen2.5-VL, PEFT, pytest, JSONL.

**Spec:** `docs/superpowers/specs/2026-09-01-stage1-batch-equivalence-design.md` Section 15 (design commitment v26)

**Execution order:** Complete and commit Tasks 1-5 in this plan first, including the single authorized full-400 evaluation. Then execute `docs/superpowers/plans/2026-09-01-stage1-batch-equivalence-v26.md`. The two plans are sequential, not alternatives.

## Global Constraints

- Use only `./venv/ilvr-venv` and `PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src"`.
- Preserve the Stage 1 artifact loader and `config.stage == "stage1"`; never mutate the stage tag.
- Do not weaken `resolve_stage2_qwen_model`; Stage 2 artifact validation remains separate.
- Preserve the paper recurrence `e_(t+1) = h_t` for exactly `latent_size` states after `<|latent_start|>`.
- Missing or blank gold is invalid and can never match any prediction.
- Report both coverage-penalized `correct / 400` and valid-label `correct / 398`, including invalid indices and reasons.
- Persist prompt identity/length, generated suffix IDs, raw decode, extracted answer, latent counts, stop reason, and parse/invalid status per row.
- The actual-3B recurrence gate requires exact discrete scaffold/mask/position/token equality, symmetric relative L2 `<= 1e-6`, and cosine `>= 0.999999` for every latent state and post-end logit.
- Run exactly one deterministic paper-aligned full-400 evaluation of `checkpoints/stage1-dora-z1-ema-frozen-b1-ga4-e1-egb16-20260901-090240/checkpoint-215` with four shards, `max_new_tokens=4096`, explicit total `max_sequence_length=32768`, temperature `0`, and top-p `1`.
- Store new model-bearing staging only under `checkpoints/evaluation-staging/<run-id>/`, results under `results/<run-id>/`, and logs under `logs/<run-id>/` or the existing multinode log namespace.
- Do not train an epoch, redesign the latent head, or begin Stage 2 work.

---

### Task 1: Centralize Balanced Answer Parsing and Dual-Denominator Scoring

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/scoring.py`
- Create: `tests/evaluation/test_scoring.py`
- Modify: `eval.py`
- Modify: `src/evaluate_deepseed.py`

**Interfaces:**
- Produces `AnswerParseResult(extracted: str, normalized: str, status: str, detail: str | None)`.
- Produces `AnswerScore(prediction: AnswerParseResult, gold: AnswerParseResult, match: bool, invalid_gold_reason: str | None)`.
- Produces `extract_final_answer(text: str) -> AnswerParseResult`, `parse_gold_answer(value: object) -> AnswerParseResult`, and `score_answer(prediction_text: str, gold_value: object) -> AnswerScore`.
- Produces `EvaluationScoreSummary.from_rows(rows, source_count)`, with `correct`, `source_count`, `valid_gold_count`, `invalid_gold_count`, `invalid_gold`, `coverage_accuracy`, and `valid_gold_accuracy`.

- [ ] **Step 1: Write the golden scorer RED tests**

Add literal cases that assert balanced nested braces are retained and malformed input is explicit:

```python
@pytest.mark.parametrize(
    ("text", "extracted", "status"),
    [
        ("Final answer: \\boxed{\\frac{1}{2}}<|im_end|>", r"\frac{1}{2}", "parsed"),
        (r"Final answer: \\boxed{\frac{3}{4}}", r"\frac{3}{4}", "parsed"),
        (r"Answer: **{A}**.", "{A}", "parsed"),
        (r"Final answer: \\boxed{\\frac{1}{2}", "", "malformed"),
        ("<|latent_start|><|latent_pad|><|latent_end|>", "", "empty_prediction"),
    ],
)
def test_answer_parser_golden(text, extracted, status):
    result = extract_final_answer(text)
    assert (result.extracted, result.status) == (extracted, status)

def test_blank_or_missing_gold_never_matches_blank_prediction():
    for gold in (None, "", "   "):
        score = score_answer("", gold)
        assert score.match is False
        assert score.invalid_gold_reason in {"missing", "blank"}
```

Add a 400-row summary fixture with invalid gold at indices 57 and 93 and assert `correct/400` and `correct/398` are distinct named fields. Assert no general SymPy/math equivalence: `"1/2"` and `"0.5"` do not become equal unless the existing numeric normalization already applies directly to both complete spans.

The fixed parser/normalizer table is:

| Input form | Rule |
|---|---|
| one or two consecutive backslashes before `boxed{` | recognize the marker and unwrap its balanced outer braces; preserve the answer span's backslashes |
| multiple balanced boxes | use the last balanced box |
| an unclosed final box marker | return `malformed`; do not fall back to an earlier answer |
| no box, one or more `final answer` markers | use the last marker's first line |
| no box/final marker, one or more `answer:` markers | use the last marker's first line |
| outer `**`, backticks, quotes, or terminal `. , ! ? ; :` | remove only the balanced outer wrapper/punctuation |
| `yes/true/a`, `no/false/b` | normalize respectively to `Yes` and `No` |
| a complete decimal numeric span, with optional commas | parse with `float`; integral values normalize without `.0` |
| all other nonblank spans | Unicode-preserving lowercase plus surrounding-whitespace trim; no symbolic equivalence |
| no extracted content | `empty_prediction` for predictions and `invalid_gold/blank` for gold |

- [ ] **Step 2: Run the scorer tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest tests/evaluation/test_scoring.py -q
```

Expected: import failure for `evaluation.scoring`.

- [ ] **Step 3: Implement the smallest balanced scanner and scorer**

Scan from the final `\boxed{` or `\\boxed{` marker, incrementing depth on `{` and decrementing on `}` until the matching close; return `malformed` if no close exists. Strip only surrounding whitespace, markdown emphasis, quotes, and terminal sentence punctuation; never strip a balanced `{` or `}` from inside the span. Keep yes/no aliases and exact numeric normalization, but add no symbolic algebra.

Use frozen dataclasses and reject non-finite summary values. Gold classification is exact:

```python
if value is None:
    return AnswerParseResult("", "", "invalid_gold", "missing")
if not isinstance(value, str) or not value.strip():
    return AnswerParseResult("", "", "invalid_gold", "blank")
```

- [ ] **Step 4: Replace duplicate entry-point parsers with imports**

Delete `_clean_answer_span`, `extract_final_answer`, and `normalize_for_match` from both entry points. Keep `extract_assistant_content` only if it remains entry-point-specific. Convert the default task branch to call `score_answer(pred["assistant_only"], sample.get("original_final_answer"))` and populate explicit parse/invalid fields.

- [ ] **Step 5: Run focused and integration tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/evaluation/test_scoring.py \
  tests/stage2/test_eval_integration.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/evaluation eval.py src/evaluate_deepseed.py \
  tests/evaluation/test_scoring.py tests/stage2/test_eval_integration.py
git commit -m "fix: make Stage 1 answer scoring coverage-safe"
```

### Task 2: Factor a Stage-Neutral Continuous-Latent Controller

**Files:**
- Create: `src/evaluation/continuous_latent.py`
- Modify: `src/stage2/inference.py`
- Create: `tests/evaluation/test_continuous_latent.py`
- Modify: `tests/stage2/test_inference.py`

**Interfaces:**
- Produces `ContinuousLatentGenerationConfig(max_new_tokens=4096, max_sequence_length=None, temperature=0.0, top_p=1.0, eos_token_id=None)`.
- Produces `ContinuousLatentGenerationResult(sequences, latent_states_by_block, generated_suffix_ids, latent_start_count, latent_block_count, stop_reason, post_end_logits_by_block)`.
- Produces `build_continuous_generation_config(...) -> ContinuousLatentGenerationConfig`.
- Produces `ContinuousLatentController.generate(...) -> ContinuousLatentGenerationResult`.
- Retains `Stage2GenerationConfig`, `Stage2GenerationResult`, `build_stage2_generation_config`, and `Stage2InferenceController` as compatibility aliases/wrappers without changing Stage 2 validation.
- Consumes a model satisfying `ContinuousLatentModelProtocol`: `config.latent_size`, `prepare_stage2_base_inputs_embeds`, `get_rope_index`, `get_input_embeddings`, `model`, `lm_head`, and `stage2_rollout_engine`. The attribute names remain compatible with the local Qwen class; protocol validation does not inspect or rewrite `config.stage`.

- [ ] **Step 1: Write controller RED tests**

Move the generic recurrence expectations from `tests/stage2/test_inference.py` into a stage-neutral test and add evidence assertions:

```python
result = ContinuousLatentController().generate(
    model=model,
    input_ids=input_ids,
    attention_mask=torch.ones_like(input_ids),
    token_ids=token_ids,
    generation_config=ContinuousLatentGenerationConfig(
        max_new_tokens=16, temperature=0.0, top_p=1.0, eos_token_id=3
    ),
    return_latent_states=True,
    return_post_end_logits=True,
)
assert result.latent_start_count == 1
assert result.latent_block_count == 1
assert result.generated_suffix_ids == tuple(result.sequences[0, input_ids.shape[1]:].tolist())
assert result.stop_reason == "eos"
assert len(result.post_end_logits_by_block) == 1
```

Also assert Stage 1 and Stage 2 model tags are accepted by the neutral controller when they satisfy `ContinuousLatentModelProtocol`, while `resolve_stage2_qwen_model(stage1_model)` still raises.

- [ ] **Step 2: Run controller tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/evaluation/test_continuous_latent.py \
  tests/stage2/test_inference.py -q
```

- [ ] **Step 3: Move only the generic recurrence implementation**

Copy the existing validated no-cache loop into `evaluation.continuous_latent` and rename stage-specific identifiers. Continue to call the existing `model.stage2_rollout_engine.rollout(...)`; every local Stage 1 and Stage 2 Qwen instance installs that kernel in its constructor. Do not edit `LatentRolloutEngine` arithmetic. Count a block only after its complete start/pad/end scaffold and recurrence finish. Successful results use stop reasons `eos` or `max_new_tokens`. A sequence cap violation raises `GenerationLimitError` and returns no result.

The controller and independent probe implement this exact sequence:

```text
1. Build token+vision embeddings for the prompt; sequences=input_ids; mask=attention_mask.
2. Run the no-cache language model over the complete current embeddings and MRoPE positions.
3. Apply lm_head to the last hidden row, suppress latent_pad/latent_end outside a block, and select one text token.
4. Append the selected token ID, attention=1, and its ordinary token embedding.
5. If it is latent_start, append exactly K latent_pad IDs then one latent_end ID, all with attention=1 and ordinary placeholder embeddings.
6. Recompute positions for that complete scaffold. For latent position p in ascending order, run the language model on embeddings[:p], mask[:p], positions[:p]; replace embedding[p] with the last hidden state. This is e_(t+1)=h_t.
7. After all K replacements, run once through the complete scaffold; lm_head at latent_end is the post-end next-token logit captured for evidence.
8. Resume at step 2. EOS ends with stop_reason=eos; exhausting max_new_tokens ends with stop_reason=max_new_tokens.
```

- [ ] **Step 4: Preserve Stage 2 API compatibility**

Make `src/stage2/inference.py` import and alias the neutral types. `Stage2InferenceController.generate` delegates unchanged after its existing Stage 2 caller has passed the strict resolver. No Stage 1 tag mutation and no resolver changes are permitted.

- [ ] **Step 5: Run controller, rollout, and Stage 2 integration tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/evaluation/test_continuous_latent.py \
  tests/stage2/test_inference.py \
  tests/stage2/test_rollout.py \
  tests/stage2/test_eval_integration.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/continuous_latent.py src/stage2/inference.py \
  tests/evaluation/test_continuous_latent.py tests/stage2/test_inference.py
git commit -m "refactor: expose continuous latent inference by contract"
```

### Task 3: Route Stage 1 Through Continuous Inference and Persist Auditable Rows

**Files:**
- Create: `src/evaluation/runtime.py`
- Modify: `eval.py`
- Modify: `src/evaluate_deepseed.py`
- Modify: `src/multinode_evaluation.py`
- Modify: `src/multinode_results.py`
- Modify: `tests/stage2/test_eval_integration.py`
- Modify: `tests/multinode/test_evaluation.py`
- Modify: `tests/multinode/test_result_validation.py`

**Interfaces:**
- Produces `PreparedEvaluationInput(prompt, tensors, prompt_token_length, prompt_identity)`.
- Produces `prepare_evaluation_input(processor, sample, images) -> PreparedEvaluationInput`.
- Every default-task result row includes `index` (the source index), `prompt_identity`, `prompt_token_length`, `generated_suffix_token_ids`, `raw_output`, `assistant_only`, `prediction`, `extracted_final_answer`, `prediction_parse_status`, `gold_parse_status`, `invalid_gold_reason`, `latent_start_count`, `latent_block_count`, `stop_reason`, `match`, and timing when available.
- Shard merge summary contains both denominator conventions and invalid-gold records without recomputing from hidden fields.

- [ ] **Step 1: Write prompt-lineage and Stage 1 route RED tests**

Replace `test_stage1_eval_uses_stage1_generation_instead_of_stage2_controller` with a test that makes `model.generate` raise and records one `ContinuousLatentController.generate` call on the resolved Stage 1 Qwen model.

Add prompt invariance:

```python
base = {"text_input": "question", "image_input": [str(image)], "original_final_answer": "42"}
mutated = dict(base, original_final_answer="SECRET-GOLD")
removed = {key: value for key, value in base.items() if key != "original_final_answer"}
prepared = [prepare_evaluation_input(processor, sample, images) for sample in (base, mutated, removed)]
assert all(torch.equal(item.tensors["input_ids"], prepared[0].tensors["input_ids"]) for item in prepared)
assert all(torch.equal(item.tensors["pixel_values"], prepared[0].tensors["pixel_values"]) for item in prepared)
assert {item.prompt_identity for item in prepared} == {prepared[0].prompt_identity}
```

Make the fake generation result include suffix IDs, block counts, stop reason, and assert the exact JSONL fields above.

- [ ] **Step 2: Run integration tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_eval_integration.py \
  tests/multinode/test_evaluation.py \
  tests/multinode/test_result_validation.py -q
```

- [ ] **Step 3: Implement prompt identity and Stage 1 routing**

For prompt identity, use the fixed field order `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`. For each present tensor compute the existing `canonical_tensor_sha256` digest over dtype, shape, and canonical little-endian contiguous CPU bytes; encode the ordered `(field_name, tensor_digest)` list with `canonical_json_bytes`; the final identity is SHA-256 of those JSON bytes. Do not hash the gold field. In `eval.py`, resolve Stage 1 with `resolve_qwen2_5_vl_model`, validate `stage == "stage1"`, validate `ContinuousLatentModelProtocol`, build the neutral generation config, and call `ContinuousLatentController`. Stage 2 continues through `resolve_stage2_qwen_model` and its compatibility controller.

Decode both the full sequence and suffix. `raw_output` is the full-sequence audit decode. `assistant_only` and `prediction` are the assistant text decoded/extracted from the generated suffix only, and `score_answer` always consumes `assistant_only`; prompt text can never enter scoring. Compute `generated_suffix_token_ids` by slicing at `prompt_token_length` and retain the controller's stop/block evidence.

- [ ] **Step 4: Upgrade summaries and shard validation**

For a merged 400-row default-task evaluation, require source indices `0..399`, count invalid gold independently, and write:

```json
{
  "correct": 65,
  "source_count": 400,
  "valid_gold_count": 398,
  "invalid_gold_count": 2,
  "invalid_gold": [{"index": 57, "reason": "blank"}, {"index": 93, "reason": "blank"}],
  "coverage_accuracy": 0.1625,
  "valid_gold_accuracy": 0.16331658291457288
}
```

The numbers in the example are fixture data, not hard-coded production expectations. Preserve the VSP simulation branch and Stage 2 behavior.

- [ ] **Step 5: Run focused evaluation tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/evaluation \
  tests/stage2/test_eval_integration.py \
  tests/multinode/test_evaluation.py \
  tests/multinode/test_result_validation.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/runtime.py eval.py src/evaluate_deepseed.py \
  src/multinode_evaluation.py src/multinode_results.py \
  tests/stage2/test_eval_integration.py tests/multinode/test_evaluation.py \
  tests/multinode/test_result_validation.py
git commit -m "fix: route Stage 1 evaluation through latent recurrence"
```

### Task 4: Add Historical Rescore, Forced-Prefix Recurrence, and Boundary Diagnostics

**Files:**
- Create: `src/stage1/evaluation_diagnostics.py`
- Create: `tests/stage1/test_evaluation_diagnostics.py`
- Create: `documents/vcsd-ilvr-stage2/guideline-stage1.md`

**Interfaces:**
- CLI modes: `rescore`, `recurrence-probe`, and `boundary-probe`.
- `rescore` requires `--input-jsonl`, `--dataset-jsonl`, and `--output-json`.
- `recurrence-probe` and `boundary-probe` require `--model-dir`, `--base-model-dir`, `--dataset-jsonl`, `--output-json`, and `--cache-dir`; `boundary-probe` additionally requires `--source-indices` as a comma-separated fixed list.
- Writes JSON evidence only beneath a caller-supplied `results/<run-id>/` path and execution logs beneath `logs/<run-id>/`.
- `recurrence-probe` compares the production controller to an independently coded no-cache recurrence, never by calling the production controller from the reference implementation.

- [ ] **Step 1: Write diagnostic reducer RED tests**

Test pure helpers for symmetric relative L2, cosine, rank calculation, historical JSONL rescore, forced-prefix comparison, and parameter-delta summaries. Require exact integer fields and reject NaN/Inf.

```python
assert symmetric_relative_l2(x, x.clone()) == 0.0
assert cosine_similarity(x, x.clone()) == pytest.approx(1.0)
assert token_rank(torch.tensor([3.0, 1.0, 2.0]), 2) == 2
```

- [ ] **Step 2: Run diagnostic tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_evaluation_diagnostics.py -q
```

- [ ] **Step 3: Implement three bounded modes**

`rescore` joins old rows to the dataset by required unique integer `index`; missing, duplicate, or out-of-range indices are fatal. Because the old artifact has no raw IDs/raw full decode, use its saved `prediction` field as legacy assistant-only text, mark every row `evidence_status="legacy_assistant_only"`, and leave generated-token/stop/latent evidence explicitly unavailable. Record old/new match changes and the missing-raw-ID limitation.

`recurrence-probe` loads the existing checkpoint, runs two forced prefixes (latent immediately, and one text token then latent), and compares sequences, masks, MRoPE positions, latent states, and post-end logits. It exits nonzero if any discrete value differs, symmetric relative L2 exceeds `1e-6`, or cosine is below `0.999999`. Add a static/clamped control that substitutes the original latent-pad embedding only at the feedback input and reports its distances without a pass threshold.

`boundary-probe` uses TEST indices `0,1,2,7` and two forced known-boundary prefixes per row: the tokenized assistant cue immediately before `<|latent_start|>`, and the same prefix with one ordinary text token before `<|latent_start|>`. Record latent-start/EOS logits and ranks, per-position CE, trainable latent-row and DoRA gradients, and parameter deltas before/after one fresh optimizer step. The in-memory diagnostic optimizer is AdamW with learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, global-norm clipping `1.0`, and no scheduler; capture gradients before clipping and deltas after clipping+step. Run the four cue-colon samples with budgets `4096` and `8192` and record EOS versus token-limit stop. Never save an optimizer/model checkpoint from this diagnostic.

- [ ] **Step 4: Correct the stale guide**

Create the tracked replacement for the previously untracked stale guide. State that Stage 1 has a paper-defined continuous-latent evaluation path, Stage 1 full/PEFT loaders remain distinct from Stage 2, and static `<|latent_pad|>` generation is a diagnostic control only. Include the scorer's two denominators and the rule that invalid gold never matches. Do not retain the claim that Stage 1 cannot be evaluated independently.

- [ ] **Step 5: Run diagnostic tests and focused evaluation suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_evaluation_diagnostics.py \
  tests/evaluation \
  tests/stage2/test_inference.py \
  tests/stage2/test_eval_integration.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/stage1/evaluation_diagnostics.py \
  tests/stage1/test_evaluation_diagnostics.py \
  documents/vcsd-ilvr-stage2/guideline-stage1.md
git commit -m "test: add Stage 1 recurrence and boundary diagnostics"
```

### Task 5: Run Gates, Rescore, and the One Authorized Full-400 Evaluation

**Files:**
- Create at runtime: `results/stage1-eval-v26-<timestamp>/...`
- Create at runtime: `logs/stage1-eval-v26-<timestamp>/...`

**Interfaces:**
- Consumes the existing checkpoint and old evaluation JSONL.
- Produces `historical-rescore.json`, `recurrence-probe.json`, `boundary-probe.json`, four shard JSONLs/manifests, `merged.jsonl`, and `summary.json`.

- [ ] **Step 1: Run the focused CPU suite once**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/evaluation \
  tests/stage1/test_evaluation_diagnostics.py \
  tests/stage2/test_inference.py \
  tests/stage2/test_rollout.py \
  tests/stage2/test_eval_integration.py \
  tests/multinode/test_evaluation.py \
  tests/multinode/test_result_validation.py -q
```

- [ ] **Step 2: Create a fresh run ID and rescore the historical static output**

Use one shell-generated timestamp and explicit paths:

```bash
stage1_eval_suffix=$(date -u +%Y%m%d-%H%M%S)
DIAG_RUN_ID="stage1-eval-v26-diagnostics-$stage1_eval_suffix"
EVAL_RUN_ID="stage1-eval-v26-full400-$stage1_eval_suffix"
mkdir -p "$PWD/results/$DIAG_RUN_ID" "$PWD/logs/$DIAG_RUN_ID"
PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/python -m stage1.evaluation_diagnostics rescore \
  --input-jsonl /root/workspace/ILVR/results/eval-stage1-dora-z1-ema-frozen-b1-ga4-e1-egb16-20260901-100338/merged.jsonl \
  --dataset-jsonl /root/workspace/ILVR/data/TEST.jsonl \
  --output-json "$PWD/results/$DIAG_RUN_ID/historical-rescore.json" \
  >"$PWD/logs/$DIAG_RUN_ID/historical-rescore.log" 2>&1
```

- [ ] **Step 3: Run the two actual-3B recurrence cases**

Invoke the recurrence probe on the validated prepared adapter and its exact base. Stop before dataset evaluation if either forced-prefix gate fails:

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/python -m stage1.evaluation_diagnostics recurrence-probe \
  --model-dir "$PWD/checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga4-e1-egb16-20260901-090240/prepared-checkpoint-215" \
  --base-model-dir "$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct" \
  --dataset-jsonl "$PWD/data/TEST.jsonl" \
  --cache-dir "$PWD/cache" \
  --output-json "$PWD/results/$DIAG_RUN_ID/recurrence-probe.json" \
  >"$PWD/logs/$DIAG_RUN_ID/recurrence-probe.log" 2>&1
```

- [ ] **Step 4: Run the bounded boundary diagnostic**

Invoke `boundary-probe` on the same artifact and fixed TEST source indices `0,1,2,7`. This is one fresh optimizer step in memory only; do not write model state:

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/python -m stage1.evaluation_diagnostics boundary-probe \
  --model-dir "$PWD/checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga4-e1-egb16-20260901-090240/prepared-checkpoint-215" \
  --base-model-dir "$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct" \
  --dataset-jsonl "$PWD/data/TEST.jsonl" \
  --source-indices 0,1,2,7 \
  --cache-dir "$PWD/cache" \
  --output-json "$PWD/results/$DIAG_RUN_ID/boundary-probe.json" \
  >"$PWD/logs/$DIAG_RUN_ID/boundary-probe.log" 2>&1
```

- [ ] **Step 5: Launch exactly one deterministic four-shard evaluation**

Reuse the already validated model-bearing artifact at `checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga4-e1-egb16-20260901-090240/prepared-checkpoint-215`; do not duplicate it. Then invoke:

```bash
./venv/ilvr-venv/bin/python -m src.multinode evaluate \
  --run-id "$EVAL_RUN_ID" \
  --model-dir "$PWD/checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga4-e1-egb16-20260901-090240/prepared-checkpoint-215" \
  --base-model-dir "$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct" \
  --test-data-path "$PWD/data/TEST.jsonl" \
  --task-name zebra-cot \
  --output-dir "$PWD/results/$EVAL_RUN_ID" \
  --cache-dir "$PWD/cache" \
  --max-new-tokens 4096 \
  --max-sequence-length 32768 \
  --temperature 0 \
  --top-p 1
```

Do not relaunch on a score outcome. A runtime failure gets one causal diagnosis and fresh run ID only after the failure is understood.

`python -m src.multinode evaluate` is the existing fixed four-worker launcher: shard 0/1 run on remote GPU 0/1 and shard 2/3 on local GPU 0/1, each receives `--num-shards 4 --shard-index N`, and all four successful manifests are required before merge.

- [ ] **Step 6: Validate and report the result**

Require 400 ordered rows, successful four manifests, `invalid_gold_count == 2`, invalid indices 57 and 93 with reasons, both accuracies, latent-entry/block totals, stop-reason counts, and no model-bearing files under `results/`. Compare against historical corrected static-path results only within the fields the old artifact retained.

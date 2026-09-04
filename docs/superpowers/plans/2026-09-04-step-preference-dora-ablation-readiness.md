# Step-preference DoRA Ablation Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make packed Stage 2B trainable with gradient checkpointing and run a controlled DoRA comparison of CE-only, DPO-only, and entropy-only recipes.

**Architecture:** Version the step-preference assistant rendering and every artifact that depends on it, then expose a real CE coefficient in the shared Stage 2B loss reducer. Reuse Qwen's existing non-reentrant layer checkpointing for the packed FlexAttention call and add one bounded capacity probe before any full run. The three training arms share all data/runtime artifacts and differ only in their objective coefficients.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu128, Transformers/Qwen2.5-VL, FlexAttention, PEFT DoRA, Hugging Face Trainer, Accelerate, DeepSpeed ZeRO-2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-step-preference-dora-ablation-readiness-design.md`

## Global Constraints

- Run Python, pytest, training, and evaluation through `./venv/ilvr-venv` with `PYTHONPATH="$PWD/transformers/src:$PWD/src"`.
- Keep model-bearing artifacts under `checkpoints/`, results under `results/`, and logs under `logs/` in this worktree.
- Preserve ordinary Stage 2 byte-for-byte behavior when step preference is disabled.
- Use `assistant_plan_rendering="adjacent-text-space-lf-v1"` only when step preference is enabled; legacy rendering is `concatenated-v1`.
- Use negative manifest schema 3 and Stage 2 length-manifest schema 2 after the rendering change; reject older artifacts instead of upgrading them implicitly.
- Keep packed attention as FlexAttention with the existing `BlockMask`; public Stage 2B attention remains `sdpa`.
- Use non-reentrant gradient checkpointing with RNG-state preservation and `use_cache=False`.
- Use DoRA rank 32, alpha 64, dropout 0.05, the seven approved decoder projection suffixes, BF16, two local H100 processes, ZeRO-2, seed 42, local batch 1, accumulation 2, two epochs, and learning rate `5e-5`.
- Do not launch a full training arm until the focused tests and all capacity gates are green.

---

### Task 1: Versioned assistant rendering and length manifests

**Files:**
- Modify: `src/step_preference/spans.py`
- Modify: `src/stage2/data.py`
- Modify: `src/stage2/length_manifest.py`
- Modify: `src/main.py`
- Modify: `run_stage2_training.sh`
- Modify: `tests/step_preference/test_spans.py`
- Modify: `tests/stage2/test_data.py`
- Modify: `tests/stage2/test_length_manifest.py`
- Modify: `tests/stage2/test_launcher.py`

**Interfaces:**
- Produces: `ASSISTANT_PLAN_RENDERING_CONCATENATED`, `ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF`, and `normalize_assistant_plan_rendering(value: str) -> str`.
- Produces: `build_assistant_plan(..., assistant_plan_rendering: str = ASSISTANT_PLAN_RENDERING_CONCATENATED)` and the same optional keyword on `build_stage2_conversation` and `prepare_stage2_example`.
- Produces: `Stage2LengthManifest.assistant_plan_rendering: str`, schema version 2, with the rendering passed through build, read, validate, and CLI paths.
- Consumes: the existing strict half-open `PlanCharSpan` and no-truncation length measurement.

- [ ] **Step 1: Add failing rendering tests**

Add tests which construct `text("A.") -> text("The final answer")` and assert:

```python
assistant, spans, _ = build_assistant_plan(
    plan,
    1,
    assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
)
assert assistant == "A. \nThe final answer"
assert (spans[0].start, spans[0].end) == (0, 2)
assert (spans[1].start, spans[1].end) == (4, 20)
assert assistant[spans[0].end:spans[1].start] == " \n"
```

Also assert the default returns `"A.The final answer"`, existing leading or trailing whitespace is not normalized, and an unknown rendering value raises `ValueError`.

Add the causal tokenizer regression using the real fast tokenizer from the
fixed Stage 2A checkpoint. First assert legacy rendering produces one offset
covering characters from both sides of the `.The` boundary, and direct
tokenization of `"A.\nThe final answer"` produces one offset covering `".\n"`.
Then render with space-LF and
call `map_reasoning_step_spans`; assert no token crosses the two text endpoints,
the mapped target interval starts/ends on exact offsets, and the delimiter token
position belongs to neither the reasoning target nor final-answer span.

- [ ] **Step 2: Run the rendering tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_spans.py -k 'rendering or adjacent_text'
```

Expected: FAIL because the constants/keyword do not exist.

- [ ] **Step 3: Implement the minimal rendering policy**

In `spans.py`, validate exactly two identities and insert an ASCII space followed by LF before the current entry only when both the previous and current plan entries are text:

```python
ASSISTANT_PLAN_RENDERING_CONCATENATED = "concatenated-v1"
ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF = "adjacent-text-space-lf-v1"

def normalize_assistant_plan_rendering(value: str) -> str:
    if value not in {
        ASSISTANT_PLAN_RENDERING_CONCATENATED,
        ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    }:
        raise ValueError(f"unsupported assistant plan rendering: {value}")
    return value
```

Before appending a rendered entry, append `" \n"` and increment the character cursor by two when the space-LF identity is active and the adjacent raw plan types are both `text`. Do not widen either `PlanCharSpan` over the delimiter.

- [ ] **Step 4: Add failing prepared-example and length-manifest tests**

Assert that `prepare_stage2_example(..., assistant_plan_rendering=...)` uses the same rendered assistant text and span coordinates. Extend length-manifest construction/round-trip tests to require schema 2 and the exact rendering identity. Assert validation fails when a legacy-rendered manifest is supplied for space-LF-rendered Stage 2B.

Add an end-to-end ordinary Stage 2 golden test: prepare and collate the same
sample through the default path and assert conversation, `input_ids`,
`attention_mask`, labels, pixel tensors, image grids, and latent spans equal the
existing expected fixture values.

- [ ] **Step 5: Run the data/manifest tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_data.py tests/stage2/test_length_manifest.py
```

Expected: FAIL on the missing keyword and manifest field.

- [ ] **Step 6: Thread the rendering through data and manifests**

Add the optional keyword to `_assistant_text`, `build_stage2_conversation`, and `prepare_stage2_example`, and pass it to both assistant-plan calls so conversation and spans cannot diverge. Set `STAGE2_LENGTH_MANIFEST_SCHEMA_VERSION = 2`; add the field to the dataclass, writer, strict reader, validator, snapshot preparation, and CLI parser as `--assistant-plan-rendering` with the legacy default.

In `src/main.py`, derive the expected rendering from `StepPreferenceConfig.enabled` and pass it both when validating the length manifest and preparing examples. In the launcher, pass `adjacent-text-space-lf-v1` to length-manifest generation whenever `STEP_PREFERENCE_ENABLED=true`, otherwise pass `concatenated-v1`.

- [ ] **Step 7: Add and run launcher regression tests**

Assert a Stage 2B dry-run includes the space-LF rendering in its length-manifest command/identity, while an ordinary Stage 2 dry-run retains the legacy identity. Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_spans.py \
  tests/stage2/test_data.py \
  tests/stage2/test_length_manifest.py \
  tests/stage2/test_launcher.py
```

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/step_preference/spans.py src/stage2/data.py \
  src/stage2/length_manifest.py src/main.py run_stage2_training.sh \
  tests/step_preference/test_spans.py tests/stage2/test_data.py \
  tests/stage2/test_length_manifest.py tests/stage2/test_launcher.py
git commit -m "fix: version step preference rendering"
```

---

### Task 2: Bind negative artifacts to the rendering

**Files:**
- Modify: `src/step_preference/negatives.py`
- Modify: `scripts/generate_step_negatives.py`
- Modify: `src/main.py`
- Modify: `tests/step_preference/test_negatives.py`
- Modify: `tests/step_preference/test_negative_generation.py`
- Modify: `tests/stage2/test_routing.py`

**Interfaces:**
- Consumes: `ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF` and the rendering-aware `prepare_stage2_example` from Task 1.
- Produces: `NEGATIVE_MANIFEST_SCHEMA_VERSION = 3` and `NegativeManifest.assistant_plan_rendering: str`.
- Produces: `assistant_plan_rendering` on `write_negative_artifact` and
  `expected_assistant_plan_rendering` on `load_negative_store`.

- [ ] **Step 1: Add failing strict-manifest tests**

Extend manifest fixtures with:

```python
"schema_version": 3,
"assistant_plan_rendering": "adjacent-text-space-lf-v1",
```

Assert missing, unknown, or `concatenated-v1` rendering fails when the consumer expects `adjacent-text-space-lf-v1`. Assert canonical write/read retains the field.

- [ ] **Step 2: Run negative tests and verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_negatives.py \
  tests/stage2/test_routing.py
```

Expected: FAIL because schema 2 has no rendering field.

- [ ] **Step 3: Implement schema 3 and strict consumer matching**

Set schema version 3, add the string field to `NegativeManifest`, require its
value to be exactly `adjacent-text-space-lf-v1`, include it in
`_manifest_from_bytes`, and require equality in `load_negative_store`. Do not
accept schema 2, a schema-3 legacy rendering, or synthesize a default.

- [ ] **Step 4: Make the generator use space-LF rendering**

In `_selected_examples`, validate and prepare every source row with:

```python
    assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF
```

Pass the same identity to `write_negative_artifact` and its post-publication `load_negative_store` check. Include the identity directly in the manifest, not only inside `generation_settings`.

- [ ] **Step 5: Make Stage 2B preflight demand schema 3 space-LF artifacts**

In `src/main.py`, supply `expected_assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF` when loading the negative store. Add a test proving preflight rejects an otherwise valid legacy artifact before model construction.

- [ ] **Step 6: Run focused tests**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_negatives.py \
  tests/step_preference/test_negative_generation.py \
  tests/stage2/test_routing.py \
  tests/step_preference/test_collator_spans.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/step_preference/negatives.py scripts/generate_step_negatives.py \
  src/main.py tests/step_preference/test_negatives.py \
  tests/step_preference/test_negative_generation.py \
  tests/stage2/test_routing.py
git commit -m "fix: authenticate preference rendering"
```

---

### Task 3: Express isolated CE, DPO, and entropy objectives

**Files:**
- Modify: `src/step_preference/config.py`
- Modify: `src/stage2/trainer.py`
- Modify: `run_stage2_training.sh`
- Modify: `tests/step_preference/test_config.py`
- Modify: `tests/step_preference/test_stage2b_trainer.py`
- Modify: `tests/stage2/test_launcher.py`

**Interfaces:**
- Produces: `StepPreferenceConfig.lambda_ce: float`, mapped from existing `--ce_weight` and defaulting to `1.0`.
- Produces: `total_loss = lambda_ce * clean_ce + lambda_dpo * dpo.loss + lambda_entropy * entropy_loss` in the one shared reducer used by naive and packed paths.
- Produces: launcher environment `CE_WEIGHT`, default `1.0`.

- [ ] **Step 1: Add failing configuration tests**

Assert default `lambda_ce == 1.0`, `--ce_weight 0` maps to zero, and each of the three weights rejects negative, NaN, and infinity. Keep beta strictly positive and entropy margin nonnegative.

- [ ] **Step 2: Run config tests and verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_config.py
```

Expected: FAIL because `lambda_ce` is absent and negative DPO/entropy weights are accepted.

- [ ] **Step 3: Implement the configuration field**

Add `lambda_ce: float = 1.0` before the DPO/entropy coefficients, use `_finite`, and reject values below zero for all three objective coefficients. Map `lambda_ce=getattr(args, "ce_weight", 1.0)` in `from_args`.

- [ ] **Step 4: Add failing loss and gradient-isolation tests**

Extend the deterministic fake-policy test to capture raw CE, DPO, and entropy scalars. For configurations `(1,0,0)`, `(0,1,0)`, and `(0,0,1)`, assert the returned scalar equals the selected component. On a tiny trainable fake policy, compare the concatenated gradient (using zero tensors for missing grads) with a separate backward of only the selected raw component using `rtol=1e-5, atol=1e-7`.

- [ ] **Step 5: Run trainer tests and verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_stage2b_trainer.py -k 'objective or coefficient or gradient'
```

Expected: DPO-only and entropy-only assertions fail because CE is still added.

- [ ] **Step 6: Change the single shared reducer**

Replace the reducer expression with:

```python
total_loss = (
    config.lambda_ce * clean_ce
    + config.lambda_dpo * dpo.loss
    + config.lambda_entropy * entropy_loss
)
```

Do not prune packed branches when a coefficient is zero. Add raw unweighted
`loss/dpo` and `loss/entropy` to `_STEP_PREFERENCE_LOG_NAMES` and the reducer's
metrics beside the existing modality splits; keep every raw metric unweighted.
This gives the capacity probe an exact finite scalar for each component.

- [ ] **Step 7: Forward `CE_WEIGHT` through the launcher**

Define `CE_WEIGHT=${CE_WEIGHT:-1.0}`, validate it as finite/nonnegative alongside the preference weights, add `--ce_weight "$CE_WEIGHT"` to the launcher-owned Stage 2 arguments, and add `--ce_weight` to `MANAGED_OPTIONS`. Test the three dry-run coefficient triplets exactly.

- [ ] **Step 8: Run focused tests**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_config.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_launcher.py
```

Expected: PASS.

- [ ] **Step 9: Commit Task 3**

```bash
git add src/step_preference/config.py src/stage2/trainer.py \
  run_stage2_training.sh tests/step_preference/test_config.py \
  tests/step_preference/test_stage2b_trainer.py tests/stage2/test_launcher.py
git commit -m "feat: isolate stage2b objective weights"
```

---

### Task 4: Enable packed non-reentrant gradient checkpointing

**Files:**
- Modify: `src/utils_deepseed.py`
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Modify: `tests/step_preference/test_config.py`
- Modify: `tests/step_preference/test_packed_stage2_model.py`

**Interfaces:**
- Consumes: existing `configure_stage2_model_backend(..., gradient_checkpointing=True)` which installs `use_reentrant=False`.
- Produces: packed Stage 2B acceptance with checkpointing while retaining the current decoder-option guards.

- [ ] **Step 1: Replace rejection assertions with failing acceptance tests**

Change the CLI test to parse packed Stage 2B plus `--gradient_checkpointing` successfully. Replace `test_packed_stage2_rejects_active_gradient_checkpointing` with a CUDA test that clones one dropout-disabled FP32 tiny packed model, enables checkpointing on one copy, runs the same packed loss/backward, and compares loss, selected logits, and concatenated decoder projection gradients at the spec tolerances.

Save both TF32 flags, set `torch.backends.cuda.matmul.allow_tf32=False` and
`torch.backends.cudnn.allow_tf32=False`, assert both are false during the two
runs, and restore both values in `finally`.

Register a forward hook on each decoder layer's `self_attn.q_proj`; count only calls whose input sequence length equals `plan.padded_length`. Assert the plain copy records one such call per layer and the checkpointed copy records two, proving backward recomputation.

- [ ] **Step 2: Run tests and verify RED**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_config.py::test_packed_stage2b_configuration \
  tests/step_preference/test_packed_stage2_model.py::test_packed_stage2_gradient_checkpointing_matches_plain
```

Expected: FAIL at the current CLI and model prohibitions.

- [ ] **Step 3: Remove only the two prohibitions**

Delete the packing/checkpointing `ValueError` in `parse_args` and the `self.model.gradient_checkpointing` rejection in `_forward_stage2`. Do not add a new wrapper: allow the final `language_model(...)` call to enter the existing `Qwen2_5_VLModel.forward` layer checkpoint loop with its `BlockMask` argument.

- [ ] **Step 4: Run the focused CUDA and Stage 2 checkpoint tests**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_config.py \
  tests/step_preference/test_packed_stage2_model.py \
  tests/stage2/test_qwen_forward.py -k gradient_checkpointing
```

Expected: PASS with no CUDA skip.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/utils_deepseed.py \
  transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  tests/step_preference/test_config.py \
  tests/step_preference/test_packed_stage2_model.py
git commit -m "fix: checkpoint packed stage2b decoder"
```

---

### Task 5: Add the bounded packed-capacity probe

**Files:**
- Modify: `src/step_preference/equivalence.py`
- Create: `scripts/probe_packed_stage2b_capacity.py`
- Create: `scripts/audit_packed_stage2b_lengths.py`
- Create: `scripts/select_jsonl_rows.py`
- Create: `tests/step_preference/test_packed_capacity_probe.py`
- Create: `tests/step_preference/test_packed_capacity_audit.py`
- Create: `tests/step_preference/test_select_jsonl_rows.py`

**Interfaces:**
- Extends: `load_production_context(..., tuning_method: str = "full", gradient_checkpointing: bool = False)`.
- Produces: a one-shot CLI that writes one exclusive JSON result containing status, identity, clean/packed lengths, component finiteness, elapsed time, and CUDA baseline/peak allocated/reserved bytes.
- Produces: a read-only packed-length audit CLI which writes one exclusive JSON report and an exact one-row source subset for the maximum tuple `(padded_length, unpadded_length, clean_length, source_index)`.
- Produces: `select_jsonl_rows(input_path: Path, output_path: Path, indices: tuple[int, ...])` which copies exact selected source lines to one exclusive JSONL in the requested order.

`probe_packed_stage2b_capacity.py` accepts required `--checkpoint_path`,
`--data_path`, `--image_root`, `--negative_path`, `--negative_manifest`, and
`--output_path`; optional `--source_row_index` (default 0), `--device` (default
`cuda:0`), `--tuning_method` (`full|lora|dora`, default `full`), and Boolean
`--gradient_checkpointing`. Its JSON has exact top-level keys
`schema_version,status,identity,sample,losses,gradient_norm,memory,elapsed_seconds,errors`.
`sample` contains source index/sample ID/clean length/step count/branch count and
unpadded/padded lengths; `losses` contains raw CE/DPO/entropy and total;
`memory` contains device total, baseline/peak allocated, and baseline/peak
reserved bytes. Exit is zero only for `status="pass"`, one for `oom`, and two
for every other error.

`audit_packed_stage2b_lengths.py` accepts the same five authenticated input
paths plus required `--report_path` and `--subset_output_path`. Its JSON has
`schema_version,identity,records,winner`; each record and winner contain
`source_index,sample_id,clean_length,step_count,branch_count,unpadded_length,padded_length`.
It exits zero only after strict negative/preflight validation and exclusive
publication of both outputs.

- [ ] **Step 1: Add failing context-routing and report tests**

Mock only model loading and assert the production context parser includes `--tuning_method dora` and `--gradient_checkpointing` when requested, does not disable checkpointing afterward, and records both fields in identity. Test report construction with a fake runtime for success and `torch.OutOfMemoryError`, requiring statuses `pass` and `oom` respectively.

- [ ] **Step 2: Run the new tests and verify RED**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_capacity_probe.py
```

Expected: FAIL because the CLI/module do not exist and context is hardcoded to full/no-checkpointing.

- [ ] **Step 3: Extend the context loader minimally**

Append `--tuning_method` and the appropriate checkpoint Boolean flag to its `parse_args` list. Remove the unconditional `model.gradient_checkpointing_disable()` and instead assert the resulting model state matches the requested Boolean. Do not change existing callers' defaults.

- [ ] **Step 4: Implement one-shot capacity execution**

The script must:

1. reject an existing output path;
2. load one authenticated contiguous source subset through `load_production_context` with DoRA/checkpointing;
3. build the packed plan once to record `clean_length`, `packed_unpadded_length`, `packed_padded_length`, step count, and branch count;
4. reset CUDA peak statistics, execute exactly one `Stage2BTrainingStep` packed forward/backward, synchronize, and capture allocated/reserved peaks;
5. require finite raw CE, DPO, entropy, and total loss metrics plus a finite
   nonzero trainable-adapter gradient norm; any non-finite raw component makes
   the report fail even when its coefficient is zero;
6. write status `pass`, or status `oom` only for a CUDA OOM; preserve other exceptions as status `error` and exit nonzero.

Use `write_evidence_exclusive`; never retry inside the script.

- [ ] **Step 5: Run probe unit tests**

Implement `scripts/select_jsonl_rows.py` with required `--input-path`,
`--output-path`, and one-or-more integer `--index` arguments. Reject duplicate,
negative, or out-of-range indices and an existing output. Preserve each selected
input line's bytes and terminate every emitted row with exactly one LF. Test
indices `(2, 0)` against a three-row fixture and verify output order and bytes.

Implement `scripts/audit_packed_stage2b_lengths.py` against the authenticated
Stage 2B preflight/data-collator path. For every retained example, build one
`PackedTreePlan` without model execution, collect source index, sample ID, clean
length, unpadded length, and padded length, and select Python tuple maximum in
the exact order declared above. Write all records plus the winner to an
exclusive JSON report and call `select_jsonl_rows` to publish the winner's
one-row source subset. Unit-test the tuple ordering and exclusive outputs with
three synthetic records.

Then run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/step_preference/test_packed_capacity_audit.py \
  tests/step_preference/test_select_jsonl_rows.py \
  tests/step_preference/test_packed_benchmark.py
```

Expected: PASS without loading the production model.

- [ ] **Step 6: Commit Task 5**

```bash
git add src/step_preference/equivalence.py \
  scripts/probe_packed_stage2b_capacity.py \
  scripts/audit_packed_stage2b_lengths.py scripts/select_jsonl_rows.py \
  tests/step_preference/test_packed_capacity_probe.py \
  tests/step_preference/test_packed_capacity_audit.py \
  tests/step_preference/test_select_jsonl_rows.py
git commit -m "test: probe packed dora capacity"
```

---

### Task 6: Verify and run the capacity gates

**Files:**
- Read: `data/TRAIN.jsonl`
- Write generated subsets/results: `results/step-pref-capacity-<UTC>/`
- Write logs: `logs/step-pref-capacity-<UTC>/`
- Write one-step checkpoint: `checkpoints/step-pref-capacity-<UTC>/`

**Interfaces:**
- Consumes: Tasks 1–5 and the fixed Stage 2A checkpoint.
- Produces: focused test log, row-2102 capacity JSON, row-1078 capacity JSON, maximum-packed-row capacity JSON, and a two-rank ZeRO-2 optimizer-step witness.

- [ ] **Step 1: Run one focused implementation suite once**

First verify the frozen identities exactly:

```bash
capacity_run_id="step-pref-capacity-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "results/$capacity_run_id"
test ! -e "logs/$capacity_run_id"
test ! -e "checkpoints/$capacity_run_id-zero2"
mkdir -p "results/$capacity_run_id" "logs/$capacity_run_id"
test "$(sha256sum data/TRAIN.jsonl | cut -d' ' -f1)" = 5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8
test "$(sha256sum data/TEST.jsonl | cut -d' ' -f1)" = 6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
test "$(sha256sum configs/deepspeed_zero2.json | cut -d' ' -f1)" = 20615765b665d38647ee0447ca984ea28d16a1b8738363b505b054871bfc3f85
test "$(sha256sum checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827/config.json | cut -d' ' -f1)" = cbec86a55c91603e7f91d506d94c975113de3af4ef6e9cce6e124e6825bc7b44
test "$(sha256sum checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827/model.safetensors.index.json | cut -d' ' -f1)" = d28f59cc1060d05418f59a2320b4f53f83ba50374d7283d59ee559aefe35864
```

Then run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference tests/stage2
```

Expected: PASS; packed CUDA tests execute rather than skip.

- [ ] **Step 2: Create fresh explicit capacity artifacts**

Allocate one collision-free UTC run ID. Use `scripts/select_jsonl_rows.py` to
create one-row JSONL files from source indices 2102 and 1078 under the capacity
result directory. Run `generate_step_negatives.py` separately for each with
schema 3, the fixed Stage 2A checkpoint, SDPA, BF16, `min_tokens=2`, and
`max_new_tokens=32`.

Use the checked-in selector exactly:

```bash
./venv/ilvr-venv/bin/python scripts/select_jsonl_rows.py \
  --input-path data/TRAIN.jsonl \
  --output-path "results/$capacity_run_id/row-2102.jsonl" --index 2102
./venv/ilvr-venv/bin/python scripts/select_jsonl_rows.py \
  --input-path data/TRAIN.jsonl \
  --output-path "results/$capacity_run_id/row-1078.jsonl" --index 1078
```

For each subset invoke:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/generate_step_negatives.py \
  --data_path "results/$capacity_run_id/row-$row_index.jsonl" \
  --image_root data \
  --stage2a_checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --output_path "results/$capacity_run_id/row-$row_index-negatives.jsonl" \
  --manifest_path "results/$capacity_run_id/row-$row_index-manifest.json" \
  --latent_size 8 --min_tokens 2 --max_new_tokens 32 \
  --attention_implementation sdpa --device cuda:0
```

- [ ] **Step 3: Run the first nail on row 2102**

Run `scripts/probe_packed_stage2b_capacity.py` on the row-2102 JSONL/negative pair with `--tuning_method dora --gradient_checkpointing --device cuda:0`. Tee the log. Require status `pass`, finite losses/gradient norm, and recorded peak memory. Stop the execution plan if status is `oom` or `error`.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python scripts/probe_packed_stage2b_capacity.py \
  --checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --data_path "results/$capacity_run_id/row-2102.jsonl" --image_root data \
  --negative_path "results/$capacity_run_id/row-2102-negatives.jsonl" \
  --negative_manifest "results/$capacity_run_id/row-2102-manifest.json" \
  --source_row_index 0 --device cuda:0 --tuning_method dora \
  --gradient_checkpointing \
  --output_path "results/$capacity_run_id/row-2102-capacity.json" \
  2>&1 | tee "logs/$capacity_run_id/row-2102-capacity.log"
```

- [ ] **Step 4: Run the near-8192 probe on row 1078**

Repeat Step 3 with the row-1078 pair. Require its space-LF-rendered clean length to remain `<=8192`, no truncation, status `pass`, and finite witnesses. Stop on failure.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python scripts/probe_packed_stage2b_capacity.py \
  --checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --data_path "results/$capacity_run_id/row-1078.jsonl" --image_root data \
  --negative_path "results/$capacity_run_id/row-1078-negatives.jsonl" \
  --negative_manifest "results/$capacity_run_id/row-1078-manifest.json" \
  --source_row_index 0 --device cuda:0 --tuning_method dora \
  --gradient_checkpointing \
  --output_path "results/$capacity_run_id/row-1078-capacity.json" \
  2>&1 | tee "logs/$capacity_run_id/row-1078-capacity.log"
```

- [ ] **Step 5: Generate the shared full negative artifact once**

Allocate `step-pref-negatives-adjacent-text-space-lf-v1-<UTC>` and run the generator on all 3,450 rows. Tee one log and validate the emitted schema-3 manifest/store once. Record its negative and manifest SHA-256; do not regenerate per training arm.

```bash
negative_run_id="step-pref-negatives-adjacent-text-space-lf-v1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "results/$negative_run_id" "logs/$negative_run_id"
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python scripts/generate_step_negatives.py \
  --data_path data/TRAIN.jsonl --image_root data \
  --stage2a_checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --output_path "results/$negative_run_id/negatives.jsonl" \
  --manifest_path "results/$negative_run_id/negative-manifest.json" \
  --latent_size 8 --min_tokens 2 --max_new_tokens 32 \
  --attention_implementation sdpa --device cuda:0 \
  2>&1 | tee "logs/$negative_run_id/generate.log"
sha256sum "results/$negative_run_id/negatives.jsonl" \
  "results/$negative_run_id/negative-manifest.json"
```

- [ ] **Step 6: Identify and probe the maximum packed row**

Run `scripts/audit_packed_stage2b_lengths.py` with the fixed full source,
Stage 2A checkpoint, shared negative JSONL/manifest, and output paths
`results/$capacity_run_id/packed-length-audit.json` and
`results/$capacity_run_id/max-packed-row.jsonl`. It collates every retained row
without gradients and selects tuple-max
`(padded_length, unpadded_length, clean_length, source_index)`. Generate one
fresh schema-3 negative/manifest pair against `max-packed-row.jsonl`. If the
winning source index differs from 1078, run the one-shot capacity probe against
that subset and require status `pass`.

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  scripts/audit_packed_stage2b_lengths.py \
  --checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --data_path data/TRAIN.jsonl --image_root data \
  --negative_path "results/$negative_run_id/negatives.jsonl" \
  --negative_manifest "results/$negative_run_id/negative-manifest.json" \
  --report_path "results/$capacity_run_id/packed-length-audit.json" \
  --subset_output_path "results/$capacity_run_id/max-packed-row.jsonl"
```

Generate the subset artifact exactly:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python scripts/generate_step_negatives.py \
  --data_path "results/$capacity_run_id/max-packed-row.jsonl" --image_root data \
  --stage2a_checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --output_path "results/$capacity_run_id/max-packed-negatives.jsonl" \
  --manifest_path "results/$capacity_run_id/max-packed-manifest.json" \
  --latent_size 8 --min_tokens 2 --max_new_tokens 32 \
  --attention_implementation sdpa --device cuda:0
```

If the winner is not row 1078, invoke the probe with those three
`max-packed-*` paths and
`--output_path "results/$capacity_run_id/max-packed-capacity.json"`; otherwise
record the specified alias entry without another model execution.

- [ ] **Step 7: Complete a two-process ZeRO-2 optimizer step**

Launch `run_stage2_training.sh` on the maximum-packed one-row subset with DoRA, BF16, ZeRO-2, gradient checkpointing, packed Stage 2B, both preference weights enabled, local batch 1, accumulation 1, and `--max_steps 1`. Require exit zero, one completed optimizer step, a valid DoRA checkpoint under the capacity checkpoint directory, finite logged component losses, and no OOM. The single-device probes own exact peak-memory reporting; this distributed gate owns optimizer-allocation readiness.

Using the schema-3 `max-packed-negatives.jsonl` and manifest already generated
in Step 6, run:

```bash
capacity_train_id="$capacity_run_id-zero2"
STAGE1_CHECKPOINT="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
STAGE2A_CHECKPOINT_PATH="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
DATA_PATH="$PWD/results/$capacity_run_id/max-packed-row.jsonl" IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$PWD/checkpoints/$capacity_train_id" \
LOG_FILE="$PWD/logs/$capacity_run_id/zero2-step.log" \
STEP_PREFERENCE_ENABLED=true STEP_PREFERENCE_MODE=stage2b \
STEP_PREFERENCE_PACKING=true \
STEP_NEGATIVE_PATH="$PWD/results/$capacity_run_id/max-packed-negatives.jsonl" \
STEP_NEGATIVE_MANIFEST="$PWD/results/$capacity_run_id/max-packed-manifest.json" \
CE_WEIGHT=1 STEP_DPO_WEIGHT=1 STEP_ENTROPY_WEIGHT=0.1 \
STEP_DPO_BETA=0.1 STEP_ENTROPY_MARGIN=0.1 \
TUNING_METHOD=dora LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" LEARNING_RATE=5e-5 \
GRADIENT_ACCUMULATION_STEPS=1 GRADIENT_CHECKPOINTING=true \
ATTENTION_IMPLEMENTATION=sdpa MAX_SEQUENCE_LENGTH=8192 \
SUPERVISE_LATENT_BOUNDARIES=true DATALOADER_NUM_WORKERS=0 \
STAGE2_NUM_PROCESSES=2 ./run_stage2_training.sh --max_steps 1 --epochs 1
```

- [ ] **Step 8: Record the capacity decision**

Write `results/<capacity-run-id>/capacity-summary.json` with named probe records
`row_2102`, `row_1078`, and `max_packed`. When row 1078 is also the maximum,
`max_packed` has `status="alias"` and points to the row-1078 evidence path;
otherwise it has `status="pass"` and its distinct evidence path. Include the
maximum-row selection, distributed checkpoint path, exact commands, exit
statuses, GPU model, and git commit. Overall status is pass only if every
applicable gate passed.

---

### Task 7: Launch matched DoRA training and evaluation

**Files:**
- Write model/trainer state: `checkpoints/step-pref-dora-{ce,dpo,entropy}-<UTC>/`
- Write normalized configurations/results: `results/step-pref-dora-{ce,dpo,entropy}-<UTC>/`
- Write logs: `logs/step-pref-dora-{ce,dpo,entropy}-<UTC>/`

**Interfaces:**
- Consumes: Task 6 pass summary and its one shared negative artifact.
- Produces: three complete final DoRA artifacts, three 400-row evaluations, and one comparison summary ranked by coverage accuracy.

- [ ] **Step 1: Freeze and compare normalized launch configurations**

Materialize three JSON configurations containing every launcher environment value and resolved path. Normalize only `run_id`, checkpoint/result/log destinations, and the coefficient triplet. Assert all remaining fields and the length-manifest payloads are equal. The exact triplets are CE `(1,0,0)`, DPO `(0,1,0)`, and entropy `(0,0,1)`.

Use one shared timestamp and this direct launch function as the single command
template; invoke it sequentially, never concurrently:

```bash
ablation_stamp=$(date -u +%Y%m%dT%H%M%SZ)
negative_dir="$PWD/results/$negative_run_id"
launch_arm() {
  arm=$1
  ce=$2
  dpo=$3
  entropy=$4
  run_id="step-pref-dora-$arm-$ablation_stamp"
  STAGE1_CHECKPOINT="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
  STAGE2A_CHECKPOINT_PATH="$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
  DATA_PATH="$PWD/data/TRAIN.jsonl" IMAGE_ROOT="$PWD/data" \
  OUTPUT_DIR="$PWD/checkpoints/$run_id" LOG_FILE="$PWD/logs/$run_id/train.log" \
  STEP_PREFERENCE_ENABLED=true STEP_PREFERENCE_MODE=stage2b \
  STEP_PREFERENCE_PACKING=true \
  STEP_NEGATIVE_PATH="$negative_dir/negatives.jsonl" \
  STEP_NEGATIVE_MANIFEST="$negative_dir/negative-manifest.json" \
  CE_WEIGHT="$ce" STEP_DPO_WEIGHT="$dpo" STEP_ENTROPY_WEIGHT="$entropy" \
  STEP_DPO_BETA=0.1 STEP_ENTROPY_MARGIN=0.1 \
  TUNING_METHOD=dora LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.05 \
  LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" LEARNING_RATE=5e-5 \
  GRADIENT_ACCUMULATION_STEPS=2 GRADIENT_CHECKPOINTING=true \
  ATTENTION_IMPLEMENTATION=sdpa MAX_SEQUENCE_LENGTH=8192 \
  SUPERVISE_LATENT_BOUNDARIES=true DATALOADER_NUM_WORKERS=0 \
  STAGE2_NUM_PROCESSES=2 ./run_stage2_training.sh \
    --epochs 2 --warm_up_steps 10 --save_steps 612
}
```

Before real launch, invoke the same environment with `STAGE2_DRY_RUN=true` for
each triplet and save stdout under one separate preflight result directory.
Parse the three shell-escaped argument vectors with `shlex.split`, replace only
the three output paths and coefficient values with named sentinels, and assert
the normalized vectors are identical. Do not create any arm's checkpoint, log,
or result directory before its real launcher reserves them.

Create only the preflight directory, capture dry runs, then execute this exact
normalizer:

```bash
preflight_dir="results/step-pref-dora-preflight-$ablation_stamp"
test ! -e "$preflight_dir"
mkdir -p "$preflight_dir"
STAGE2_DRY_RUN=true launch_arm ce 1 0 0 \
  >"$preflight_dir/ce-launch-command.txt"
STAGE2_DRY_RUN=true launch_arm dpo 0 1 0 \
  >"$preflight_dir/dpo-launch-command.txt"
STAGE2_DRY_RUN=true launch_arm entropy 0 0 1 \
  >"$preflight_dir/entropy-launch-command.txt"
negative_sha256=$(sha256sum "$negative_dir/negatives.jsonl" | cut -d' ' -f1)
./venv/ilvr-venv/bin/python - "$ablation_stamp" "$negative_sha256" "$preflight_dir" <<'PY'
import json, shlex, sys
from pathlib import Path

ablation_stamp = sys.argv[1]
negative_sha256 = sys.argv[2]
preflight_dir = Path(sys.argv[3])
arms = ("ce", "dpo", "entropy")
triplets = {"ce": [1, 0, 0], "dpo": [0, 1, 0], "entropy": [0, 0, 1]}
varying = {
    "--save_model_path": "<CHECKPOINT>",
    "--log_file": "<LOG>",
    "--stage2_length_manifest": "<LENGTH_MANIFEST>",
    "--ce_weight": "<CE>",
    "--step_dpo_weight": "<DPO>",
    "--step_entropy_weight": "<ENTROPY>",
}
normalized = {}
for arm in arms:
    path = preflight_dir / f"{arm}-launch-command.txt"
    tokens = shlex.split(path.read_text())
    for index, token in enumerate(tokens[:-1]):
        if token in varying:
            tokens[index + 1] = varying[token]
    normalized[arm] = tokens
assert normalized["ce"] == normalized["dpo"] == normalized["entropy"]
for arm in arms:
    run_id = f"step-pref-dora-{arm}-{ablation_stamp}"
    payload = {
        "schema_version": 1,
        "arm": arm,
        "run_id": run_id,
        "coefficient_triplet": triplets[arm],
        "normalized_command": normalized[arm],
        "train_sha256": "5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8",
        "stage2a_config_sha256": "cbec86a55c91603e7f91d506d94c975113de3af4ef6e9cce6e124e6825bc7b44",
        "negative_sha256": negative_sha256,
    }
    (preflight_dir / f"{arm}-launch-config.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
PY
```

- [ ] **Step 2: Launch CE-only training**

Use the canonical `run_stage2_training.sh` contract from the spec with `CE_WEIGHT=1`, `STEP_DPO_WEIGHT=0`, and `STEP_ENTROPY_WEIGHT=0`. Require two epochs, no non-finite or skipped optimizer step, and a valid final DoRA artifact.

```bash
launch_arm ce 1 0 0
```

- [ ] **Step 3: Launch DPO-only training**

Repeat with `CE_WEIGHT=0`, `STEP_DPO_WEIGHT=1`, and `STEP_ENTROPY_WEIGHT=0`. Use the identical shared negative paths and all common runtime values.

```bash
launch_arm dpo 0 1 0
```

- [ ] **Step 4: Launch entropy-only training**

Repeat with `CE_WEIGHT=0`, `STEP_DPO_WEIGHT=0`, and `STEP_ENTROPY_WEIGHT=1`. Use the identical shared negative paths and all common runtime values.

```bash
launch_arm entropy 0 0 1
```

- [ ] **Step 4a: Validate post-launch manifests and publish launch records**

```bash
ce_result="results/step-pref-dora-ce-$ablation_stamp"
dpo_result="results/step-pref-dora-dpo-$ablation_stamp"
entropy_result="results/step-pref-dora-entropy-$ablation_stamp"
cmp "$ce_result/stage2-length-manifest.json" "$dpo_result/stage2-length-manifest.json"
cmp "$ce_result/stage2-length-manifest.json" "$entropy_result/stage2-length-manifest.json"
for arm in ce dpo entropy; do
  run_id="step-pref-dora-$arm-$ablation_stamp"
  cp "$preflight_dir/$arm-launch-command.txt" "results/$run_id/launch-command.txt"
  cp "$preflight_dir/$arm-launch-config.json" "results/$run_id/launch-config.json"
done
```

The byte comparisons prove identical measured lengths, retained indices, and
rendering identity before any evaluation result is compared.

- [ ] **Step 5: Evaluate all three final adapters**

For each arm, invoke `eval.py` with the fixed Stage 2A base, its final DoRA adapter, `data/TEST.jsonl`, task `comt`, max sequence length 32768, max new tokens 4096, repetition penalty 1.05, temperature 0, and top-p 1. Require exactly 400 sample records and one summary line.

```bash
evaluate_arm() {
  arm=$1
  run_id="step-pref-dora-$arm-$ablation_stamp"
  mkdir -p "$PWD/results/$run_id/evaluation" "$PWD/logs/$run_id/evaluation"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
    ./venv/ilvr-venv/bin/python eval.py \
    --model_dir "$PWD/checkpoints/$run_id" \
    --base_model_dir "$PWD/checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827" \
    --test_data_path "$PWD/data/TEST.jsonl" --task_name comt \
    --output_json_path "$PWD/results/$run_id/evaluation/predictions.jsonl" \
    --max_sequence_length 32768 --max_new_tokens 4096 \
    --repetition_penalty 1.05 --temperature 0 --top_p 1 \
    >"$PWD/logs/$run_id/evaluation/eval.log" 2>&1
}
evaluate_arm ce
evaluate_arm dpo
evaluate_arm entropy
```

- [ ] **Step 6: Write the comparison result**

Write one JSON and one compact Markdown table under `results/step-pref-dora-comparison-<UTC>/` containing run IDs, final checkpoint identities, coefficient triplets, correct/400, coverage accuracy, valid-gold accuracy, invalid-gold count, and inference time. Rank descending by coverage accuracy; report exact equality as a tie and do not use valid-gold accuracy to break it.

Run this exact validator/writer through the repository venv with
`$ablation_stamp` as its only argument:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/python \
  - "$ablation_stamp" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
from stage2.checkpointing import Stage2ArtifactKind, inspect_stage2_artifact

stamp = sys.argv[1]
arms = ("ce", "dpo", "entropy")
expected_triplets = {"ce": [1, 0, 0], "dpo": [0, 1, 0], "entropy": [0, 0, 1]}
rows = []
normalized = None
for arm in arms:
    run_id = f"step-pref-dora-{arm}-{stamp}"
    checkpoint = Path("checkpoints") / run_id
    inspection = inspect_stage2_artifact(checkpoint)
    assert inspection.artifact_kind is Stage2ArtifactKind.PEFT
    config = json.loads((Path("results") / run_id / "launch-config.json").read_text())
    assert config["arm"] == arm and config["run_id"] == run_id
    assert config["coefficient_triplet"] == expected_triplets[arm]
    normalized = config["normalized_command"] if normalized is None else normalized
    assert config["normalized_command"] == normalized
    prediction_path = Path("results") / run_id / "evaluation" / "predictions.jsonl"
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            assert key not in value
            value[key] = item
        return value
    records = [
        json.loads(line, object_pairs_hook=reject_duplicates)
        for line in prediction_path.read_text().splitlines()
    ]
    assert len(records) == 401
    assert all(record.get("summary") is not True for record in records[:400])
    summary = records[-1]
    assert summary.get("summary") is True and summary.get("num_samples") == 400
    for key in ("coverage_accuracy", "valid_gold_accuracy", "average_inference_time_sec"):
        assert math.isfinite(float(summary[key]))
    identities = {}
    for name in ("stage2_peft_metadata.json", "adapter_model.safetensors"):
        payload = (checkpoint / name).read_bytes()
        identities[name] = hashlib.sha256(payload).hexdigest()
    rows.append({
        "arm": arm,
        "run_id": run_id,
        "checkpoint_sha256": identities,
        "coefficient_triplet": expected_triplets[arm],
        "correct": int(summary["correct"]),
        "source_count": 400,
        "coverage_accuracy": float(summary["coverage_accuracy"]),
        "valid_gold_accuracy": float(summary["valid_gold_accuracy"]),
        "invalid_gold_count": int(summary["invalid_gold_count"]),
        "average_inference_time_sec": float(summary["average_inference_time_sec"]),
    })
ranked = sorted(rows, key=lambda row: (-row["coverage_accuracy"], row["arm"]))
for index, row in enumerate(ranked):
    row["rank"] = 1 + sum(
        prior["coverage_accuracy"] > row["coverage_accuracy"] for prior in ranked
    )
comparison_id = f"step-pref-dora-comparison-{stamp}"
destination = Path("results") / comparison_id
destination.mkdir()
(destination / "comparison.json").write_text(
    json.dumps({"schema_version": 1, "primary_metric": "coverage_accuracy", "runs": ranked},
               sort_keys=True, separators=(",", ":")) + "\n"
)
lines = [
    "| Rank | Arm | Correct/400 | Coverage | Valid-gold | Invalid gold | Avg seconds |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
for row in ranked:
    lines.append(
        f"| {row['rank']} | {row['arm']} | {row['correct']}/400 | "
        f"{row['coverage_accuracy']:.6f} | {row['valid_gold_accuracy']:.6f} | "
        f"{row['invalid_gold_count']} | {row['average_inference_time_sec']:.6f} |"
    )
(destination / "comparison.md").write_text("\n".join(lines) + "\n")
PY
```

- [ ] **Step 7: Verify completion evidence**

Run the verification commands selected by `superpowers:verification-before-completion`: the focused project suite once, strict artifact inspectors for all three final DoRA outputs, JSONL cardinality/summary validation for all evaluations, and comparison-config equality validation. Do not claim intrinsic objective superiority or comparison with historical Stage 2 runs.

Use `inspect_stage2_artifact(Path(...))` and require
`Stage2ArtifactKind.PEFT` for each final output. For every predictions file,
parse every line with duplicate-key rejection, require 401 records, require the
first 400 have no `summary` flag, and require the last record has
`summary is True`, `num_samples == 400`, and finite accuracy/time fields. The
comparison writer consumes only these validated last records and the three
validated `launch-config.json` files.

The claim-specific commands are the Task 1–5 focused suite, the two `cmp`
commands in Step 4a, and the Step 6 validator/writer (which includes artifact,
configuration, and evaluation cardinality checks). Run each once with fresh
output; do not repeat the full suite after a green identical run.

---

## Plan self-review record

- Spec coverage: Task 1 owns rendering and length identity; Task 2 owns negative identity; Task 3 owns pure objectives; Task 4 owns packed checkpoint semantics; Task 5 owns a reusable one-shot measurement; Task 6 owns the three capacity gates; Task 7 owns matched runs and frozen evaluation.
- Placeholder scan: all implementation interfaces, constants, schema versions, commands, pass/fail criteria, coefficient triplets, and artifact roots are explicit. Runtime-generated UTC IDs and post-generation digests are intentionally created once and then frozen.
- Type consistency: both manifests use the same rendering string; data preparation, length validation, negative generation, and Stage 2B preflight consume the same identity; `lambda_ce` is sourced only from `--ce_weight`; packed checkpointing reuses the existing model flag and non-reentrant function.

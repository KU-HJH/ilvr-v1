# Task 5 Report: Training Composition and Adapter-Only Trainer Saving

## Status

Complete. Stage 2 now composes full, fresh LoRA/DoRA, and resumed PEFT models
from the Task 4 load decision. `Stage2Trainer` adds thin model-native diagnostic
collection/logging plus PEFT metadata sidecars to ordinary Hugging Face flows.

Implementation commit: `a058158d0867680eab447110efb1f30336b126bd`

## TDD Evidence

### RED: training composition and thin Trainer contract

Command:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2/test_routing.py -q
```

Result after correcting one test-fixture metadata omission:

```text
7 failed, 17 passed in 12.92s
```

The exact expected RED reasons were:

- `Stage2Trainer.compute_loss` was still an override rather than
  `Trainer.compute_loss`.
- `build_stage2_training_components` never called
  `build_stage2_tuned_model` for `full`, `lora`, or `dora`.
- PEFT resume did not invoke compatibility validation before Stage 2 config
  mutation.
- `Stage2TrainingComponents` did not accept or return `peft_metadata`.

### RED: final and numbered adapter saves

Command:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft_checkpointing.py -q
```

Result after correcting the same tiny-model fixture field:

```text
2 failed, 43 passed in 9.12s
```

Both failures were the intended missing API:

```text
TypeError: Trainer.__init__() got an unexpected keyword argument
'stage2_peft_metadata'
```

No production file was edited before these RED runs.

### GREEN: focused Task 5 regressions

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_routing.py tests/stage2/test_peft_checkpointing.py -q
```

Result:

```text
69 passed in 13.13s
```

### Final Stage 2 regression suite

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2 -q
```

Result:

```text
291 passed, 1 skipped in 24.72s
```

The repository root was added to `PYTHONPATH` for this broader run because the
existing evaluation integration tests import the top-level `eval.py`. The one
skip is an existing opt-in environment/backend integration test, not a Task 5
failure.

## Implemented Behavior

- Routes every Stage 2 tuning argument into `resolve_stage2_load_decision`.
- Loads the processor only from `decision.processor_source`, the base model
  only from `decision.base_model_source`, and an adapter only from
  `decision.adapter_source`.
- Validates resumed adapter metadata/config/tensors against the unmodified
  Stage 1 base before calling `configure_stage2_metadata`.
- Applies Stage 2 metadata and backend settings, fully freezes vision, and then
  constructs or loads the selected full/LoRA/DoRA topology.
- Validates and logs one Stage 2 parameter report.
- Builds fresh LoRA/DoRA metadata with the explicit Stage 1 reference and
  canonical full decoder paths. PEFT internally stores targets in a set, so
  canonical paths are resolved before wrapping to keep sidecars deterministic.
- Preserves `decision.peft_metadata` by object identity on resume; relocated
  base provenance is not rewritten.
- Passes the already wrapped model and sidecar metadata to `Stage2Trainer`
  without a `peft_config` argument.
- Keeps `Stage2Trainer` limited to thin model-native diagnostic collection/log
  enrichment, a metadata initializer, and the `save_model` sidecar hook.
  `_save_checkpoint` remains inherited.
- Confirms final and numbered PEFT saves contain adapter config/weights and the
  sidecar but no full model weights. Numbered checkpoints retain Trainer state,
  optimizer, scheduler, and RNG artifacts.
- Confirms full tuning writes no PEFT sidecar and the Stage 2 processor/tokenizer
  latent IDs are unchanged during composition.

## Files Changed

- `src/main.py`
- `src/stage2/trainer.py`
- `tests/stage2/test_routing.py`
- `tests/stage2/test_peft_checkpointing.py`

## Systematic Debugging Notes

Three unexpected integration issues were investigated before changes:

1. The real Qwen model stores checkpoint kwargs on each module's
   `_gradient_checkpointing_func` partial, not in the old fake fixture's
   top-level attribute. The test now observes the real model state.
2. PEFT 0.17.1 converts list-valued target modules to a set. Fresh metadata now
   uses the canonical pre-wrap resolver order instead of the unordered PEFT
   config/report order.
3. An earlier DeepSpeed `TrainingArguments` test leaves both global state and
   `ACCELERATE_USE_DEEPSPEED` active in-process. The new CPU Trainer tests use a
   narrow before/after fixture that resets Transformers/Accelerate test state
   and temporarily removes that flag. A reproducing sequence passed 4/4 before
   the final full suite.

## Self-Review

- No Stage 1 path, generic `GenerationMixin`, Qwen recurrence, or Task 4
  checkpoint schema was modified.
- Fresh Stage 1-to-Stage 2 runs still return no Trainer resume path, while PEFT
  Stage 2 resume still passes the numbered checkpoint to `trainer.train`.
- Resume compatibility runs before any mutation capable of hiding a relocated
  base mismatch.
- Full mode remains a non-PEFT model and does not create a sidecar.
- The save hook delegates to `Trainer.save_model` first and does not replace
  numbered checkpoint logic.
- `git diff --check` passed before the implementation commit.

## Concerns / Follow-Up

- The broader suite's one opt-in backend/distributed test remained skipped by
  its existing environment gate. No GPU/ZeRO-3 training job was run for this
  focused unit/integration task.
- The PEFT fixture emits a known base-config lookup warning during adapter save;
  only that exact external warning is filtered in the two affected unit tests.

## Review Fix Round 1

Review-fix commit: `86fa15741fd2e03628b592df339465d9c4994bde`

### Controller Resolution

The Task 5 brief's original “override only model saving” wording conflicted with
the binding design's requirement to retain latent diagnostics. The controller
resolved this in favor of a narrow observability boundary:

- request `collect_stage2_diagnostics=True` while calling the existing
  model-native loss;
- cache detached diagnostic tensors and enrich the next ordinary Trainer log;
- preserve helper-image fail-fast validation;
- do not move recurrence or CE ownership into Trainer, add losses, or
  reimplement checkpointing.

### Systematic Debugging Result

Qwen's Stage 2 forward defaults `collect_stage2_diagnostics=False`. Generic
`Trainer.compute_loss` calls `model(**inputs)` without changing that default,
so `stage2_diagnostics` was never produced. Qwen already computes the required
metrics inside its authoritative rollout path and returns detached L2/cosine
tensors when collection is enabled. The fix therefore belongs only at the
Trainer/model call boundary and ordinary logging boundary.

The parameter-report review finding was also confirmed: `main.py` obtained the
complete `Stage2ParameterReport` but logged only counts, omitting its resolved
decoder paths.

### RED

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_routing.py \
  tests/stage2/test_peft_checkpointing.py::test_stage2_trainer_numbered_peft_checkpoint_keeps_resume_state \
  -q
```

Result:

```text
6 failed, 22 passed in 17.25s
```

The six intended failures were:

- generic `Trainer.compute_loss` did not request or cache diagnostics;
- generic `Trainer.log` did not add latent block-count, norm, or cosine metrics;
- helper-image tensors were not rejected at the Trainer boundary;
- all three full/LoRA/DoRA composition cases omitted resolved targets from the
  one-time parameter report log.

The expanded numbered-checkpoint test passed during this RED run. That confirms
the reviewer's classification: processor saving and actual resume were a
regression-coverage gap, not a demonstrated production failure.

### Minimal GREEN Implementation

- Restored only the thin `compute_loss` call boundary needed to request
  model-native diagnostics, detach/cache them, and return the model's existing
  loss unchanged.
- Restored one-use log enrichment and helper-image fail-fast checks.
- Kept `_save_checkpoint` inherited and left save/resume machinery unchanged.
- Logged `resolved_target_names` with total/trainable parameter counts.
- Extended the real one-step LoRA checkpoint test with a tiny processing class,
  asserted Trainer-saved processor metadata, loaded the adapter, resumed the
  numbered checkpoint, and reached global step 2/checkpoint-2.

Focused GREEN command (same selection as RED):

```text
28 passed in 15.20s
```

### Final Focused Verification

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_routing.py tests/stage2/test_peft_checkpointing.py -q
```

Result:

```text
72 passed in 16.14s
```

Per controller direction, no broad suite rerun was performed in this review
round.

### Review-Fix Self-Review

- Model-native CE and recurrence remain authoritative and unchanged.
- Diagnostic tensors are detached before Trainer caching/logging.
- Diagnostics are consumed once, then cleared to prevent replay.
- No auxiliary loss, generic generation change, or checkpoint override was
  introduced.
- Standard `Trainer.save_model` still owns processing-class persistence.
- Standard Trainer checkpoint resume restored state and advanced the real tiny
  LoRA run from global step 1 to 2.

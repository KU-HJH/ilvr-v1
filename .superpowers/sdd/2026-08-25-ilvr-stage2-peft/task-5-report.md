# Task 5 Report: Training Composition and Adapter-Only Trainer Saving

## Status

Complete. Stage 2 now composes full, fresh LoRA/DoRA, and resumed PEFT models
from the Task 4 load decision, and `Stage2Trainer` adds only PEFT metadata
sidecars to the ordinary Hugging Face save/checkpoint flow.

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
- Keeps `Stage2Trainer` limited to a metadata initializer and `save_model`
  sidecar hook. `compute_loss`, `log`, and `_save_checkpoint` remain inherited.
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

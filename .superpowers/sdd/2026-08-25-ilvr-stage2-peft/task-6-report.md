# Task 6 Report: Base-Plus-Adapter Evaluation and Inference Parity

## Status

Complete. Stage 2 inference now validates and loads an unmodified Stage 1 base
plus a frozen LoRA/DoRA adapter, while both evaluation entry points select full
versus PEFT artifacts explicitly and resolve `DeepSpeedEngine -> PeftModel ->
Qwen` before invoking the existing inference controller.

Implementation commit: `4f32f1e44ced757f3e083ae18dc6ba0b8bc6fce1`

## TDD Evidence

### Baseline

The plan's literal combined baseline command omitted the repository root from
`PYTHONPATH`. Because pytest was invoked through the venv executable, the root
`eval.py` module was not importable:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_inference.py tests/stage2/test_eval_integration.py -q
```

Result:

```text
9 failed, 41 passed in 6.31s
ModuleNotFoundError: No module named 'eval'
```

Adding `$PWD` proved the starting implementation baseline was otherwise clean:

```bash
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_inference.py tests/stage2/test_eval_integration.py -q
```

```text
50 passed in 5.76s
```

### RED: inference-only adapter loader

Command:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2/test_inference.py -q
```

Result:

```text
collection error: cannot import name 'load_stage2_model_for_inference'
1 error in 4.74s
```

This was the intended missing-symbol RED. The new parameterized test constructs
and saves real tiny LoRA and DoRA adapters, makes their adapter deltas nonzero,
and requires reload parity for text logits and every continuous latent state.

The first GREEN attempt exposed an incomplete test fixture before the loader:
the tiny config's `supervise_latent_boundaries` was `None`, while the existing
strict sidecar builder requires a boolean. Setting that test-only metadata to
`True` fixed the fixture without weakening production validation.

Focused GREEN:

```text
33 passed, 2 warnings in 6.99s
```

### RED: evaluation artifact routing and wrapper resolution

Command:

```bash
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2/test_eval_integration.py -q
```

Result:

```text
11 failed, 18 passed in 4.94s
```

The intended failures showed:

- both CLIs rejected `--base_model_dir`;
- neither entry point exposed full-versus-PEFT artifact selection;
- both model loaders retried unrelated validation errors as eager attention;
- the DeepSpeed-shaped path stopped at the outer `PeftModel` instead of the
  adapter-injected Qwen model.

Focused GREEN after the minimal implementation and mutation-coverage additions:

```text
35 passed, 6 warnings in 4.74s
```

### RED/GREEN: real Flash Attention error spelling

Self-review found that Transformers may spell the backend as `Flash Attention`
with a space. The positive fallback test was changed to that spelling first:

```bash
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2/test_eval_integration.py \
  -q -k flash_backend_initialization_error
```

RED:

```text
2 failed, 33 deselected in 4.46s
```

GREEN after adding that spelling to the backend-only classifier:

```text
2 passed, 33 deselected in 4.40s
```

### Final verification

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_inference.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/test_generation_regression.py -q
```

Result:

```text
77 passed, 8 warnings in 7.68s
```

`git diff --check` also passed before the implementation commit.

## Implemented Behavior

- Added `load_stage2_model_for_inference`, which reads the adapter config and
  validates sidecar/config/tokens/target tensors against the truly unmodified
  Stage 1 base before configuring any Stage 2 values.
- Configures Stage 2 latent IDs, latent size, and boundary supervision only from
  the validated sidecar, then calls `PeftModel.from_pretrained(...,
  is_trainable=False)`, switches the wrapper to eval mode, and validates that
  every parameter is frozen. No merge/unload path was added.
- Preserves the outer `PeftModel` as the loaded runtime artifact. Both entry
  points explicitly resolve the adapter-injected Qwen only when accessing model
  metadata or calling `Stage2InferenceController`.
- Adds `--base_model_dir` to both evaluation CLIs.
- Treats a missing sidecar as a full Stage 2 artifact and rejects a stray base
  path. Treats a present sidecar as PEFT, requires a Stage 1 base, loads the
  processor from the adapter artifact, loads Qwen from the base path, and
  attaches the validated adapter before reading evaluation samples.
- Restricts FlashAttention-to-eager retry to recognizable backend
  initialization errors. Metadata/config/adapter/token errors propagate.
- Derives the runtime device from the outer wrapper first and from resolved
  Qwen second, without merging or disabling the active adapter modules.
- Keeps original image inputs and ignores helper-image fields in evaluation.
  Existing multi-block recurrence and explicit batch-size-one tests remain
  green.
- Adds non-vacuous LoRA and DoRA save/reload parity for logits and `z1..zK`.

## Files Changed

- `src/stage2/peft.py`
- `eval.py`
- `src/evaluate_deepseed.py`
- `tests/stage2/test_inference.py`
- `tests/stage2/test_eval_integration.py`

`src/stage2/inference.py` did not require a change; the existing controller
already owns the correct continuous-latent recurrence.

## Systematic Debugging Notes

1. The initial top-level `eval` import failures came from the plan command's
   import path, not repository behavior. Adding the repository root produced a
   clean 50-test baseline.
2. The first inference GREEN failure came from omitted boolean metadata in the
   tiny test config; the strict sidecar API behaved correctly.
3. The first commit attempt encountered a transient linked-worktree
   `index.lock`. The lock disappeared before inspection, and no live Git
   process owned it; the retry succeeded without deleting a lock file.
4. Test imports created bytecode caches. Cleanup revealed four pre-existing
   tracked cache files, which were restored exactly from HEAD so the commit
   remained limited to the five intended Task 6 files.

## Self-Review

- Compatibility validation occurs before `configure_stage2_metadata`, so an
  incompatible relocated Stage 1 base cannot be made to look valid by mutation.
- LoRA and DoRA tests use nonzero saved adapter deltas and compare both ordinary
  logits and all recurrent continuous latent states after reload.
- Full and PEFT artifact selection is symmetric across both entry points.
- Missing bases, stray bases, and token mismatches fail during artifact loading,
  before dataset iteration.
- `DeepSpeedEngine.module -> PeftModel -> Qwen` and direct `PeftModel -> Qwen`
  paths are both covered; the Qwen passed to the controller still contains
  active injected LoRA modules.
- Original image paths remain the only images opened; helper-image data is not
  reintroduced.
- No generic `GenerationMixin`, latent rollout engine, Stage 2 recurrence,
  latent token IDs, merge/export path, or `B > 1` behavior was changed.
- No unrelated file is present in the implementation commit.

## Concerns / Follow-Up

- No GPU or DeepSpeed training job was needed or run for this focused inference
  and CPU/tiny-model task.
- PEFT emits its existing warning that the synthetic tiny base has no external
  config path when saving adapters. The warning is expected for these local
  fixtures and does not affect the 77 passing assertions.
- The repository tracks several `.pyc` files under `src/__pycache__`; they were
  preserved unchanged and are unrelated to Task 6.

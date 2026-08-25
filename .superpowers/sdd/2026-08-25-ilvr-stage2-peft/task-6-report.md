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

## Review Fix Round 1: Narrow FlashAttention Fallback Classification

### Review Finding and Root Cause

The reviewer reported that both evaluation classifiers treated any exception
chain containing a FlashAttention spelling as a backend-initialization error.
A direct probe reproduced the false positives in both modules:

```text
True  No such checkpoint: /checkpoints/flashattention-experiment/missing
True  Malformed config: attn_implementation=flash_attention_2
True  Flash Attention 2 is unavailable
```

The classifier had only one condition: a FlashAttention identifier. It had no
independent signal that the exception described availability, import/install,
support, device, dtype, or backend initialization. Consequently unrelated
errors triggered a second potentially expensive eager model load.

### RED

The existing behavioral test was parameterized across both entry points and
six false-positive families: missing checkpoint path, unknown model ID,
metadata mismatch, latent-token mismatch, adapter mismatch, and malformed
configuration. Every message deliberately includes a FlashAttention spelling.
The test requires the original exception object to propagate and the external
model loader to be called exactly once.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2/test_eval_integration.py -q \
  -k 'non_backend_model_error or flash_backend_initialization_error'
```

Result before the production change:

```text
12 failed, 2 passed, 31 deselected in 4.24s
```

All 12 failures showed the unwanted call sequence:

```text
['flash_attention_2', 'eager']
```

The two real positive cases, using `Flash Attention 2 is unavailable`, already
passed and remained the preservation boundary.

### Minimal GREEN

Both classifiers now require two independently detected conditions while
walking the exception chain:

1. a recognized FlashAttention identifier, including the spaced spelling;
2. recognized initialization-failure language covering availability,
   import/install, support/compatibility, CUDA/GPU placement, dtype, or backend
   initialization.

The surrounding broad catch, re-raise behavior, eager retry, model load
arguments, and all adapter-loading semantics were left unchanged.

Focused RED-selection result after the change:

```text
14 passed, 31 deselected in 4.37s
```

### Focused Final Verification

Per controller direction, only the focused evaluation integration file was run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2/test_eval_integration.py -q
```

Result:

```text
45 passed, 6 warnings in 4.65s
```

`git diff --check` also passed.

### Review-Fix Self-Review

- A FlashAttention spelling alone can no longer authorize eager fallback.
- Missing paths/model IDs and metadata/token/adapter/config mismatches propagate
  by object identity after exactly one load attempt.
- The real positive `Flash Attention 2 is unavailable` spelling still retries
  eager in both entry points.
- Local Transformers initialization messages for unsupported models, missing
  installs/imports, unavailable CUDA, unsupported dtype, and CPU/disk dispatch
  are covered by the recognized second-signal vocabulary.
- The change is identical in both entry points and does not modify their catch
  boundaries, Qwen loading arguments, PEFT validation order, adapter attachment,
  controller resolution, or inference recurrence.
- No broad suite was rerun, in accordance with the focused-validation request
  and the repository's stop-redundant-verification policy.

## Review Fix Round 2: Per-Exception Concrete Backend Patterns

### Correction to Round 1

The round-1 report overstated the classifier boundary. Although it added a
second vocabulary gate, generic standalone markers such as `dtype` and `not
compatible` still authorized fallback. It also accumulated the identifier and
failure flags across different exceptions in a cause/context chain, and it did
not recognize the real package spelling `flash_attn`.

A read-only probe reproduced all three defects in both entry points:

```text
True   FlashAttention adapter dtype metadata mismatch
True   Flash Attention adapter is not compatible with the base model
True   outer: FlashAttention adapter load failed
       cause: adapter dtype config mismatch
False  ModuleNotFoundError: No module named 'flash_attn'
```

### RED Cycle 1: Direct/Nested Negatives and Package Positives

Actual `load_model` tests were added for both entry points. The negative cases
require the original exception object to propagate and exactly one
`flash_attention_2` load. The positives require an eager retry and successful
model return.

The new coverage includes:

- direct adapter dtype-metadata and base-compatibility errors containing a
  FlashAttention spelling;
- an outer FlashAttention adapter error whose inner config error contains the
  generic `dtype` term;
- direct and nested `ModuleNotFoundError("No module named 'flash_attn'")`;
- the existing spaced `Flash Attention 2 is unavailable` positive.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2/test_eval_integration.py -q \
  -k 'non_backend_model_error or nested_non_backend_error or \
flash_attn_module_import_error or flash_backend_initialization_error'
```

RED result:

```text
10 failed, 14 passed, 31 deselected in 4.91s
```

The six negative failures showed an unwanted eager retry. The four package
positive failures propagated the missing-package exception without retry.

After replacing accumulated generic flags with per-exception concrete pattern
matching:

```text
24 passed, 31 deselected in 4.64s
```

### RED Cycle 2: General ImportError Contract

Requirement review found that the first minimal implementation recognized the
exact `ModuleNotFoundError` form but not every `ImportError` naming
`flash_attn`. An additional actual-loader test used:

```text
ImportError: cannot import flash_attn CUDA extension
```

RED:

```text
2 failed, 55 deselected in 4.53s
```

The package-import predicate was then changed to require exactly the conjunction
of `isinstance(error, ImportError)` and `flash_attn` in that same exception's
message.

Combined classifier GREEN:

```text
26 passed, 31 deselected in 4.30s
```

### Implementation Boundary

Each exception in the cause/context chain is now classified independently. No
state is carried from an outer exception to an inner exception. An individual
exception qualifies only through one of these concrete local-Transformers
families:

- an `ImportError`/`ModuleNotFoundError` naming `flash_attn`;
- `package flash_attn` install, support, or version failure text;
- `FlashAttention2 has been toggled on`;
- a model that `does not support Flash Attention 2`;
- `Flash Attention 2 is not available` or `is unavailable`;
- `Flash Attention 2 and BetterTransformer` incompatibility;
- `attempting to use Flash Attention 2` with the local dtype/GPU/CPU/disk
  initialization details;
- `Flash Attention ... only supports torch.float16`;
- an explicit Flash Attention 2 backend initialization failure.

Standalone `dtype`, `not compatible`, `support`, device, and initialization
terms are no longer classifiers.

### Focused Final Verification

Per controller direction, only the focused evaluation integration file was run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest tests/stage2/test_eval_integration.py -q
```

Result:

```text
57 passed, 6 warnings in 4.69s
```

`git diff --check` passed before commit.

### Round-2 Self-Review

- Direct and nested negative errors preserve object identity and perform one
  model load in both entry points.
- Direct and nested missing-`flash_attn` errors retry eager in both entry
  points, as does the retained spaced availability error.
- Generic adapter/config metadata, dtype, and compatibility language cannot be
  combined with a FlashAttention identifier in the same or another exception.
- Concrete patterns are grounded in local
  `transformers/modeling_utils.py:2226-2305`; no broad standalone vocabulary
  markers remain.
- The broad catch location, original re-raise, eager call arguments, adapter
  compatibility/loading, controller resolution, and recurrence remain
  unchanged.
- No broader suite was run, per the focused-validation instruction and
  repository verification policy.

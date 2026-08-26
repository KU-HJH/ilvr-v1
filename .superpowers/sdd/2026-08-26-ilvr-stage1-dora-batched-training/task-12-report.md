# Task 12 implementation report

## Status

DONE on base `6f86169`.

## Implemented

- Replaced the in-memory-only inference entry point with
  `load_stage2_model_for_inference(base_model_path, stage2_artifact_path,
  processor)`. The helper classifies the Stage 2 artifact and strictly checks
  metadata, processor/token IDs, immutable source selectors, source config,
  adapter config/topology, adapter tensors, and sparse-row payload before the
  large base-model load.
- Reused Task 11's exact sparse-source contract. Schema-v3 inference requires a
  tag-free, adapter-free original base; runs adapter discovery and config/model
  loading with the same immutable `cache_dir`, `revision`, `token`, and
  `local_files_only` mapping; rejects model-only or duplicated source options;
  resizes exactly once; initializes deterministic frozen output rows; and
  applies Stage 2 configuration.
- Loads exactly one frozen adapter without merging. Sparse output rows are
  attached after PEFT, then the already-validated artifact-owned row bytes are
  decoded/applied in memory. The recorded Stage 1 provenance path is never
  read; the v3 test deletes that directory and reloads a relocated Stage 2
  artifact successfully.
- Hardened direct and single-DeepSpeed-wrapper resolution. Sparse inference
  requires one active adapter and one active sparse output-row wrapper, with
  exactly three distinct in-range IDs. Zero/multiple adapters and malformed or
  duplicate row wrappers fail closed.
- Routed both `eval.py` and `src/evaluate_deepseed.py` through the shared
  inference helper, including processor cache selection and immutable source
  selectors. Full-model loading and schema-v2/base-source PEFT behavior remain
  on their established branches; no load flow was duplicated.
- Kept continuous-latent generation in the existing model-specific controller.
  No generic `GenerationMixin` behavior or Stage 1/training/checkpoint-save
  path was changed.

## TDD evidence

All commands used the repository virtual environment and CPU tiny models.

### RED

The initial Task 12 focused test failed because the public path-based loader
did not exist:

```text
1 failed in 16.64s
TypeError: load_stage2_model_for_inference() missing 1 required positional argument: 'tokenizer'
```

After the minimal loader was introduced, added preflight/topology tests exposed
four missing fail-closed boundaries:

```text
4 failed, 2 passed in 16.37s
```

- corrupt adapter safetensors reached the base-model load;
- an unsupported model-only source selector reached adapter discovery;
- zero/multiple active adapters were not rejected;
- malformed/duplicate sparse row wrappers were not rejected.

Updating the eval call sites began with the expected old-signature failure:

```text
2 failed, 101 deselected
TypeError: load_stage2_model_for_inference() got an unexpected keyword argument 'base_model'
```

### GREEN

Focused witnesses during implementation were:

```text
v3 loader/preflight/topology:        6 passed in 18.19s
eval entry-point routing:            4 passed, 99 deselected in 7.51s
v3 B=1/two-block reload parity:      1 passed, 6 deselected in 15.25s
source selectors/one resize:         1 passed, 7 deselected in 14.12s
complete eval integration module:  103 passed, 8 warnings in 7.49s
```

The final exact Task 12 inference and generation command was:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_peft_inference.py \
  tests/stage2/test_inference.py \
  tests/stage2/test_rollout.py \
  tests/stage2/test_generation_regression.py -q
```

```text
60 passed, 2 warnings in 19.71s
```

The two warnings are the pre-existing PEFT missing-config warnings in the
schema-v2/base-source reload parametrization.

## Systematic-debugging record

- The first strict original-base fixture failed because Transformers'
  `save_pretrained()` injected its model class into `architectures`, while the
  source model's captured provenance had an empty list. Inspection of the
  serialized config isolated fixture serialization drift. The controlled tiny
  fixture now restores the exact captured value; production provenance checks
  were not relaxed.
- The first v2/base eval routing fixtures pointed at a nonexistent base path,
  which had been harmless while model loading was mocked before preflight.
  Publishing a real tiny config then exposed the actual integration regression:
  the base branch configured `stage2` before the compatibility validator still
  required the legacy `stage1` boundary. The minimal fix preserves the v2/base
  order (`resolve tokens -> validate Stage 1 -> configure Stage 2`) while v3
  retains its distinct prepare/configure/validate order. A second fixture-only
  mismatch came from `save_pretrained()` mutating the in-memory `architectures`;
  the helper now restores it after serialization.
- The broader eval integration run initially reported 18 failures. The first
  traceback showed that the new sparse scan unconditionally called `.modules()`
  on existing duck-typed Qwen controller doubles. Sparse topology discovery now
  scans only actual `nn.Module` instances. Real loader outputs still take the
  strict one-row/one-adapter path; legacy controller doubles and base-source
  behavior are unchanged.

## Files changed

- `src/stage2/peft.py`
- `eval.py`
- `src/evaluate_deepseed.py`
- `tests/stage2/test_peft_inference.py`
- `tests/stage2/test_inference.py`
- `tests/stage2/test_eval_integration.py`
- `.superpowers/sdd/2026-08-26-ilvr-stage1-dora-batched-training/task-12-report.md`

`tests/stage2/test_rollout.py` and
`tests/stage2/test_generation_regression.py` were exercised unchanged.

## Self-review and risks

- Reviewed pre-large-load ordering, immutable source-selector reuse, clean-base
  adapter discovery, exact tag-free provenance, one resize, deterministic base
  rows, adapter tensor topology, no-merge/frozen inference, post-PEFT sparse-row
  restoration, deleted Stage 1 provenance, v2/base ordering, direct/DeepSpeed
  resolution, eval routing, and B=1 multiple-block latent/logit parity.
- No GPU, DeepSpeed engine, large checkpoint, network download, training flow,
  or checkpoint-save flow was run or changed. Real sharded/remote model loading
  remains an environment integration risk, not a known correctness defect.
- No open implementation concern found.

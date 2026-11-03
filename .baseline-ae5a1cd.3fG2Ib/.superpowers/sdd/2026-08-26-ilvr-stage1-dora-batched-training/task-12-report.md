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

## Fix round 1/5 — immutable inference witness and exact topology

### Corrections

- Source provenance selectors (`cache_dir`, `revision`, `token`, and
  `local_files_only`) are now forbidden in `model_load_kwargs` unconditionally,
  including when the immutable source mapping is empty. The model-only mapping
  can no longer become an alternate source selector channel.
- Stage 2 classification now has an internal immutable inspection result while
  retaining the public `classify_stage2_artifact(...) -> Stage2ArtifactKind`
  API. The inspection binds a canonical non-symlink artifact directory to the
  exact one-read metadata snapshot and optional sparse-row bytes.
- Added a frozen `Stage2InferencePreflightWitness`. Preflight reads metadata,
  adapter config, adapter safetensors, and sparse safetensors once and retains
  immutable bytes plus canonical artifact/source identity, immutable selector
  tuples, source architecture JSON, and the processor token/vocabulary
  contract. The actual loaded model, source selectors, architecture, and
  processor are rebound to that witness before adapter application.
- Removed all post-large-load artifact reopening. Inference now constructs the
  validated adapter with `get_peft_model`, decodes a new state dictionary from
  witnessed safetensor bytes, restores it with
  `set_peft_model_state_dict`, attaches repository sparse output rows after
  PEFT, and applies witnessed sparse bytes in memory. It never calls
  `PeftModel.from_pretrained`, never merges, and freezes every parameter.
- Adapter preflight now validates an exact tensor schema. Both legacy
  base-source and sparse-v3 branches require exactly one LoRA A/B tensor and,
  for DoRA, one magnitude tensor for every resolved target, with exact logical
  shapes and no unexpected keys. Sparse v3 additionally requires exactly one
  selective input-row delta with shape `[3, hidden_size]` and adapter-config
  IDs equal to metadata; legacy base-source explicitly forbids that tensor.
- PEFT resolution now validates exactly one configured and one matching active
  adapter for every PEFT model, even if no sparse row wrapper is present. The
  active adapter's selective-token contract determines whether exactly one
  sparse wrapper must be at the active Qwen `lm_head`, or whether legacy
  base-source inference must have none. Direct and one-level DeepSpeed wrappers
  use the same checks.

### TDD evidence

The four-finding RED selection was:

```text
8 failed, 6 passed, 8 deselected in 18.10s
```

The failures independently demonstrated:

- a source selector escaped through model options when the source mapping was
  empty;
- missing/malformed selective rows and an extra adapter tensor reached the
  large-model boundary;
- metadata/config/adapter/sparse payloads were not read once into a retained
  witness, and delete/replace after preflight could not restore in memory;
- a selective-token PEFT adapter with no sparse `lm_head` wrapper passed direct
  and DeepSpeed-wrapped resolution.

Existing LoRA A/B and DoRA-magnitude missing-key checks and wrong-location row
checks were the six controls that already passed. After the first minimal
implementation the same selection passed, but exposed a new PEFT warning:

```text
14 passed, 8 deselected, 4 warnings in 17.14s
```

After the config-copy-only warning fix, the identical selection was pristine:

```text
14 passed, 8 deselected in 18.22s
```

The complete Task 12 loader module then passed after the clean-base ordering
correction:

```text
22 passed in 21.27s
```

The first combined Task 12 plus checkpoint/PEFT/generic run caught the legacy
processor regression documented below:

```text
206 passed, 2 failed, 2 warnings in 29.97s
```

The two original legacy reload cases passed after the v3-only vocabulary fix:

```text
2 passed, 31 deselected, 2 warnings in 9.58s
```

Final combined verification used the repository virtual environment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_peft_inference.py \
  tests/stage2/test_inference.py \
  tests/stage2/test_rollout.py \
  tests/stage2/test_peft_checkpointing.py \
  tests/stage2/test_peft.py \
  tests/stage2/test_generation_regression.py -q
```

```text
208 passed, 2 warnings in 31.79s
```

The two warnings are the pre-existing PEFT missing-config warnings emitted
while the legacy v2 test fixture saves its temporary adapters. This round's
new relocated-source warning is absent. Eval entrypoints did not change in this
round, so the previously recorded focused eval witness was not rerun.

### Systematic-debugging record

- The four review findings traced to four distinct guards: model selectors
  checked only set overlap; adapter validation matched only expected LoRA/DoRA
  suffixes without a complete matched-key set or selective-row branch;
  classification returned only an enum and discarded already-read payloads
  before path-based PEFT restoration; and resolver checks lived under
  `if sparse_rows`, so zero rows skipped the active adapter contract. Each was
  fixed at its originating boundary rather than with post-load checks.
- The first in-memory GREEN produced four new warnings. Inspection of PEFT
  `mapping_func.get_peft_model` showed it overwrites
  `base_model_name_or_path` and warns when a valid relocated runtime model
  differs from the artifact's recorded source. Metadata, adapter config, and
  runtime architecture had already passed strict witness validation. The
  minimal fix changes only the freshly decoded adapter-config copy immediately
  before PEFT construction; immutable artifact bytes/provenance remain intact.
  Repeating the identical focused selection removed all four warnings.
- The first complete loader run failed one source-order assertion. Tracing call
  order showed extraction of common config logic had moved `load_config`
  before Task 11's clean-base adapter discovery. Sparse v3 now restores the
  required `processor -> adapter discovery -> source config` sequence; legacy
  base-source performs only its established config path.
- The first combined run reproducibly failed both legacy v2 reload variants at
  `len(TinyLatentTokenizer)`. Contract comparison showed sparse v3 requires an
  exact appended vocabulary and therefore a length witness, while legacy v2
  historically requires only resolved token IDs and permits tokenizer doubles
  without `__len__`. The witness now records vocabulary size only for sparse
  v3 and records `None` explicitly for legacy base-source. The original two
  failures then passed, followed by the successful 208-test combined run.

### Files additionally changed

- `src/stage2/checkpointing.py`
- `src/stage2/peft.py`
- `tests/stage2/test_peft_inference.py`
- `.superpowers/sdd/2026-08-26-ilvr-stage1-dora-batched-training/task-12-report.md`

No evaluation entrypoint, Stage 1, training, checkpoint-save, latent controller,
or generic generation implementation changed in this round. No generated
`__pycache__` remains from the implementation checks, and no open concern was
found.

## Fix round 2/5 — canonical PEFT keys and one-read eval inspection

### Corrections

- Adapter preflight no longer recognizes expected tensors by suffix. On the
  already-created meta-only Qwen skeleton it constructs the explicit
  `default` adapter with installed PEFT 0.17.1 and calls
  `get_peft_model_state_dict(..., adapter_name="default",
  save_embedding_layers=False)`. The resulting immutable exact key-to-logical-
  shape map is retained in the inference witness and used both before the
  large base load and when binding the actual loaded model.
- The v3 contract contains every LoRA A/B and DoRA magnitude tensor plus the
  one canonical selective `trainable_tokens_delta` key with logical shape
  `[3, hidden_size]`. Its selective-token IDs remain strictly bound across
  metadata, adapter config, processor, and runtime model. The explicit legacy
  v2/base contract contains the same per-target adapter tensors and no
  selective-token tensor. Missing, extra, prefix-renamed, duplicated-alias,
  and adapter-name-corrupted keys now fail as an exact set mismatch before a
  large model load.
- `Stage2ArtifactInspection` is now a public frozen inspection capability and
  retains the canonical artifact directory plus exact metadata, adapter-config,
  adapter-safetensors, and optional sparse-safetensors bytes. The compatibility
  classifier still returns only `Stage2ArtifactKind`, while direct inference
  may inspect internally once.
- Both evaluation entrypoints now inspect once for routing and pass that exact
  inspection to `load_stage2_model_for_inference`. The loader binds it to the
  requested canonical artifact path and consumes only retained bytes; it does
  not classify or reopen any captured payload. Delete/replace after route
  inspection therefore cannot alter the state, effective sparse logits, or
  inference result.

### TDD and verification evidence

The focused RED selection was:

```text
9 failed, 6 passed, 119 deselected in 19.32s
```

Prefix-renamed and adapter-name-corrupted tensors passed suffix validation and
reached the large-load double; a duplicate alias was noticed only as an
unexpected suffix match. The eval tests also observed metadata/sparse reads at
routing time but zero config/adapter reads, then a second classification in the
loader. After replacing those originating boundaries, the same selection was
pristine:

```text
15 passed, 119 deselected in 21.76s
```

Canonical artifact-path rebinding passed independently:

```text
1 passed in 15.73s
```

The legacy v2/base loader controls passed without a v3-only schema leak:

```text
4 passed, 2 pre-existing warnings in 10.82s
```

The existing path-based Stage 2 checkpoint compatibility controls also passed:

```text
3 passed in 7.96s
```

Final combined verification used the repository virtual environment exactly
once for the requested Task 12, checkpoint/PEFT, eval, rollout, and generic
generation scope:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_peft_inference.py \
  tests/stage2/test_inference.py \
  tests/stage2/test_rollout.py \
  tests/stage2/test_peft_checkpointing.py \
  tests/stage2/test_peft.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/test_generation_regression.py -q
```

```text
321 passed, 10 warnings in 35.86s
```

All ten warnings are the existing PEFT missing-config warning emitted by
temporary legacy fixture saves. The new PEFT meta-contract and immutable
in-memory restore selection emitted no warning.

### Systematic-debugging record

- The adapter-key defect was reproduced before the large-load boundary, then
  compared with PEFT 0.17.1's supported saved-state contract. On the existing
  meta-only Qwen skeleton, explicit-default
  `get_peft_model_state_dict` produced exactly the artifact's 22 v3 keys and
  every logical shape, including the one `[3, 8]` selective delta. This showed
  that manually broadening suffix rules would preserve the defect. The minimal
  fix uses PEFT's exact normalized saved-key map; the legacy public checkpoint
  validator uses the corresponding explicit default-adapter schema without
  changing save/training behavior.
- The eval double-read defect traced to route classification discarding all
  information except its enum. The loader consequently had no capability to
  consume the already-read payloads and inspected the path again. Retaining
  all four payload bytes in the frozen public inspection, then threading that
  same object through both entrypoints, removed the second read without
  duplicating loader logic.
- No new PEFT normalization, in-memory application, warning, or legacy v2
  failure appeared after the fixes. Exact v2 reload and path-based checkpoint
  controls were run before the single final combined verification.

### Files additionally changed

- `src/stage2/checkpointing.py`
- `src/stage2/peft.py`
- `src/stage2/__init__.py`
- `eval.py`
- `src/evaluate_deepseed.py`
- `tests/stage2/test_peft_inference.py`
- `tests/stage2/test_eval_integration.py`
- `.superpowers/sdd/2026-08-26-ilvr-stage1-dora-batched-training/task-12-report.md`

No Stage 1, training, checkpoint-save, latent-controller, rollout, or generic
generation implementation changed in this round.

# Task 4 Report: Versioned Adapter Sidecar and Load Decisions

## Status

Implemented the versioned Stage 2 PEFT sidecar, fresh/full/PEFT load
decisions, PEFT-only Stage 1 base rejection, and relocation-safe structural
compatibility validation. Full Stage 2 checkpoint loading remains
single-source and retains the existing `model_source` accessor.

## RED / GREEN evidence

All commands used the repository virtual environment at `./venv/ilvr-venv`.

1. Metadata RED:
   `PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft_checkpointing.py -q`
   Result: collection error, `ImportError: cannot import name
   'STAGE2_PEFT_METADATA_FILENAME'`.
2. Metadata GREEN: same command.
   Result: `4 passed in 4.86s`.
3. Load-decision RED: same command after decision tests were added.
   Result: `8 failed, 4 passed`; every failure was the missing
   `load_peft_metadata` interface.
4. Load-decision GREEN: same command.
   Result: `12 passed in 5.06s`.
5. Relocation RED:
   `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft_checkpointing.py -q`
   Result: collection error, missing
   `validate_stage2_peft_compatibility`.
6. Relocation GREEN: same command.
   Result: `22 passed in 5.27s`.
7. DoRA safetensors regression RED:
   `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft_checkpointing.py::test_relocated_stage1_base_accepts_dora_magnitude_tensor_naming -q`
   Result: `1 failed`; PEFT saves `lora_magnitude_vector` without a
   `.weight` suffix.
8. DoRA safetensors regression GREEN: same command.
   Result: `1 passed in 4.95s`.
9. Real Stage 1 adapter-directory RED:
   `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft_checkpointing.py::test_fresh_training_rejects_a_real_stage1_peft_config_directory -q`
   Result: `1 failed` because no rejection was raised.
10. Real Stage 1 adapter-directory GREEN: same command.
    Result: `1 passed in 4.85s`.
11. Sidecar-first PEFT classification RED:
    `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft_checkpointing.py::test_peft_sidecar_classifies_resume_without_loading_adapter_as_full_config -q`
    Result: `1 failed`; the adapter checkpoint was incorrectly sent through
    full-config loading before sidecar classification.
12. Sidecar-first PEFT classification GREEN: same command.
    Result: `1 passed in 4.80s`.
13. Final combined GREEN:
    `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_checkpointing.py tests/stage2/test_peft_checkpointing.py -q`
    Result: `33 passed in 5.25s`.

## Schema coverage

- Frozen `Stage2PeftMetadata` with exactly the required 15 fields and schema
  version 1.
- Strict JSON-object field set, schema, stage, method, primitive types,
  non-empty strings/lists, unique resolved targets, latent values, and
  `Stage2TuningConfig` validation.
- Tuple/list conversion round-trip, sorted-key JSON, final newline, regular-file
  requirement, sibling temporary file, and atomic `Path.replace()` publication.
- Fresh metadata construction copies configured latent semantics, exact target
  paths, tuning values, base reference, and installed PEFT version.

## Load-decision coverage

- Fresh full/LoRA/DoRA uses the explicit Stage 1 model and processor with no
  trainer resume or adapter source.
- Full Stage 2 resume remains backward-compatible: model, processor, and trainer
  all use the full checkpoint.
- Sidecar presence classifies PEFT resume before any attempt to treat its
  directory as a full model config.
- PEFT resume keeps the explicit (possibly relocated) Stage 1 base, loads
  processor/adapter/trainer state from the Stage 2 checkpoint, inherits omitted
  tuning method from the preserved sidecar, and returns the loaded metadata.
- Missing sidecar for requested PEFT, LoRA/DoRA switching, explicit full against
  PEFT, changed latent semantics, malformed metadata, and local/fake Stage 1
  PEFT-only bases fail closed. `PeftConfig.from_pretrained` validates a real
  local Stage 1 adapter config before rejection; no adapter stacking or merge is
  attempted.

## Relocation coverage

- A different base path with identical tiny Qwen structure and latent IDs is
  accepted and logged as relocation.
- Failures cover latent token IDs, latent size/boundary semantics, hidden size,
  missing/nonlinear resolved targets, base/adapter tensor shapes, method,
  `use_dora`, rank, alpha, dropout, requested targets, and resolved targets.
- Both LoRA and DoRA safetensors layouts are supported. Validation opens only
  `adapter_model.safetensors` and uses its header slices for shapes; it does not
  traverse, hash, or reload any base-model weight tree.

## Files

- `src/stage2/checkpointing.py`
- `src/stage2/__init__.py`
- `tests/stage2/test_checkpointing.py`
- `tests/stage2/test_peft_checkpointing.py`
- `.superpowers/sdd/2026-08-25-ilvr-stage2-peft/task-4-report.md`

## Self-review

- Reviewed the full diff against every brief step and the preflight ruling.
- Confirmed sidecar-first classification, provenance preservation, full-resume
  compatibility, and no base-weight reads/hashes.
- Confirmed `git diff --check` is clean.
- External reviewer dispatch was intentionally skipped because the task
  explicitly prohibited subagents; review was performed locally.

## Concerns

- PEFT-only Stage 1 detection through `adapter_config.json` is deterministic for
  explicit local checkpoint directories. Non-local model identifiers rely on
  the supplied config loader exposing `peft_type`; no remote probing is added to
  the load-decision path.

## Review fix round 1

### Verification and root causes

Applied `superpowers:receiving-code-review` and
`superpowers:systematic-debugging` before production changes. Focused probes
confirmed all four Important findings:

- The general Stage 2 resolver rejected the required unmodified `stage1` base
  before structural checks.
- The blanket `in_features == hidden_size` rule rejected `down_proj`, whose
  expected input is `intermediate_size`.
- Explicit PEFT resumes ignored rank, alpha, dropout, and target differences.
- `use_rslora=True` and other unsupported adapter variants passed validation.

The probes also confirmed that direct invalid metadata could be serialized,
and code inspection confirmed that current-base target topology was not
recomputed. The four negative-coverage claims identified by review described
existing rejection branches but lacked tests.

### RED evidence

Command:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft_checkpointing.py -q`

Result: `28 failed, 14 passed in 5.81s`. Failures covered invalid direct
serialization, four explicit tuning conflicts, genuine Stage 1 relocation,
the full default projection set, five unsupported LoRA variants, added decoder
topology, and the previously masked compatibility matrix.

Focused RED confirmation for the default projection set produced
`ValueError: hidden size mismatch ... down_proj: config=8, target=16`.

### GREEN evidence

- Genuine unmodified Stage 1 relocation: `1 passed in 4.93s`.
- All default projection shapes including `down_proj`: `1 passed in 5.07s`.
- Explicit rank/alpha/dropout/target conflicts: `4 passed in 4.59s`.
- Unsupported rsLoRA/pattern/task/bias variants: `5 passed in 4.71s`.
- Added decoder topology rejection: `1 passed in 4.76s`.
- Invalid direct `to_dict`/save rejection: `2 passed in 4.81s`.
- Complete PEFT checkpointing file: `42 passed in 5.33s`.
- Final combined checkpoint suites:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_checkpointing.py tests/stage2/test_peft_checkpointing.py -q`
  produced `50 passed in 5.37s`.

### Changes

- `src/stage2/checkpointing.py`
  - Requires an unmodified, non-PEFT Stage 1 Qwen base at the pre-configuration
    compatibility boundary.
  - Uses Qwen projection-specific architectural shapes for q/k/v/o and MLP
    gate/up/down projections, then checks adapter A/B header shapes.
  - Rejects explicit tuning-setting conflicts and unsupported rsLoRA,
    rank/alpha patterns, non-causal task types, and non-`none` bias.
  - Recomputes exact resolved target topology on the relocated base.
  - Validates direct metadata instances before `to_dict` or atomic save.
- `tests/stage2/test_peft_checkpointing.py`
  - Uses a real `stage1` relocation fixture and covers all findings plus latent
    boundary, nonlinear target, requested-target, and resolved-target failures.

### Commit

- `f0288cc fix: harden stage2 adapter compatibility`

### Self-review

- Rechecked every reviewer finding against the fixed data flow and regression
  tests.
- Confirmed Stage 1 validation occurs before `configure_stage2_metadata`; the
  Stage 2 runtime resolver remains unchanged.
- Confirmed full Stage 2 checkpoint behavior is unchanged.
- Confirmed compatibility reads only the small adapter safetensors header and
  performs no base-weight traversal, hashing, stacking, or merge.
- Confirmed exact requested/resolved topology and public serialization now fail
  closed.
- `git diff --check` was clean before the fix commit.

### Review-round concerns

No additional concerns beyond the non-local Stage 1 adapter identifier note
above.

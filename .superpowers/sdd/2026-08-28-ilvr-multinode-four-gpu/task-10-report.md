# Task 10 report: 18-case tiny lifecycle matrix

## Result

Implemented the deterministic four-GPU lifecycle matrix, parameterized both tiny
harnesses, added the strict final evidence validator and paired JSON/Markdown
publisher, and added the operator wrapper. No SSH, GPU, or live training command
was executed; verification used unit fakes and dry-run output only.

The matrix order is exactly stage (`stage1`, `stage2`), then method (`full`,
`lora`, `dora`), then ZeRO (`1`, `2`, `3`). Every plan binds a safe child ID,
one port, and one approved Task 5-owned tiny harness job. That single
distributed job performs initial `0 -> 2` save and `2 -> 3` resume internally.
After both node supervisors exit zero, live orchestration validates the strict
raw harness witness and then uses Task 6 manifest/shard collection for
`checkpoint-2` before publishing the direct case summary. It does not retry.
Case witnesses are published exclusively and the parent completion marker is
published only after all 18 cases validate.

The result validator reads only the documented direct matrix, evaluation, and
capacity paths. Its decoders require exact fields and exact scalar/container
types, including rejecting Python boolean/integer equivalence. JSON and Markdown
outputs are prechecked together, prepared only after validation, created with
exclusive mode, and the first is removed if the second publication fails.

## TDD evidence

### RED: missing Task 10 interfaces

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_lifecycle_matrix.py \
  tests/multinode/test_result_validation.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/test_distributed_contract.py -q
```

Observed output (exit 2):

```text
ERROR tests/multinode/test_lifecycle_matrix.py
ImportError: cannot import name 'build_lifecycle_matrix' from 'src.multinode'
ERROR tests/multinode/test_result_validation.py
ModuleNotFoundError: No module named 'src.multinode_results'
ERROR tests/stage1/test_distributed_smoke.py
packaging.version.InvalidVersion: Invalid version: '3.11.15+'
3 errors in 2.53s
```

The first two failures were the intended missing-feature RED. The third exposed
an environment-only collection defect before any Task 10 assertion ran. The
same repository-local import-boundary normalization already used by the adjacent
Stage 2 contract test was added to the listed Stage 1 contract test: strip this
interpreter build's trailing `+` only while importing dependencies, then restore
`platform.python_version` immediately.

### RED: wrapper absent

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_lifecycle_matrix.py::test_wrapper_dry_run_prints_exactly_eighteen_lifecycle_plan_objects -q
```

Observed output (exit 1):

```text
FileNotFoundError: [Errno 2] No such file or directory:
'.../run_multinode_lifecycle_matrix.sh'
1 failed in 0.08s
```

### GREEN: focused four-suite command

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_lifecycle_matrix.py \
  tests/multinode/test_result_validation.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/test_distributed_contract.py -q
```

Observed output (exit 0):

```text
.................................sss.................................... [ 84%]
..........sss                                                            [100%]
79 passed, 6 skipped in 9.76s
```

The six skips are the existing opt-in GPU/distributed harness executions.

## Wrapper verification

Commands:

```bash
bash -n run_multinode_lifecycle_matrix.sh
./run_multinode_lifecycle_matrix.sh --help
./run_multinode_lifecycle_matrix.sh \
  --run-id four-gpu-matrix-dry \
  --first-port 29700 \
  --result-root /tmp/ilvr-task10-wrapper-dry-5337593 \
  --dry-run
```

The dry-run JSON inspection reported:

```text
{'plans': 18, 'first_port': 29700, 'last_port': 29717,
 'single_lifecycle_arrays': True, 'steps': [0, 2, 3]}
```

The chosen result root was absent before and after the command. Help and dry-run
therefore performed no filesystem publication, SSH execution, or GPU work. The
remote and local command arrays are intentionally visible in each plan for
operator inspection but were not executed.

## Fake validator and output diff verification

The original temporary fake evidence tree was removed after verification; it is
not stored in the repository or result tree. The real CLI was run against a
bounded generated fixture before that exact fixture directory was removed:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/python -m src.multinode validate-results \
  --matrix-root "$TASK10_FIX_TMP/fixture/matrix" \
  --evaluation-root "$TASK10_FIX_TMP/fixture/evaluation" \
  --capacity-root "$TASK10_FIX_TMP/fixture/capacity-root" \
  --output-manifest "$TASK10_FIX_TMP/report.json" \
  --output-summary "$TASK10_FIX_TMP/report.md"
```

Observed report:

```text
matrix_cases=18, evaluation_samples=4,
9 Stage 1 recommendations all 2,
9 Stage 2 recommendations all 1
```

Independent literal `diff -u` comparisons of the complete minified JSON and the
complete Markdown returned exit 0:

```text
validator JSON and Markdown literal diffs: clean
```

The focused tests additionally prove missing capacity cells cannot be supplied
by recursive or unrelated JSON, NaN diagnostics and boolean ZeRO identities are
rejected, existing outputs block publication, and a simulated second-file
failure rolls back the first file.

## Review fix round 1/5

### RED evidence

The production-interface matrix tests were added before the replacement
planner. The exact command was:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_lifecycle_matrix.py::test_matrix_plans_one_exact_approved_harness_job_per_cell \
  tests/multinode/test_lifecycle_matrix.py::test_owned_harness_builder_rejects_arbitrary_path_or_arguments -q
```

Observed RED (exit 2):

```text
ImportError: cannot import name 'build_owned_lifecycle_harness_commands'
```

Stage 2 producer tests were then added before their implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/integration/test_distributed_contract.py::test_stage2_producer_evidence_requires_observed_finite_values_on_all_ranks \
  tests/stage2/integration/test_distributed_contract.py::test_stage2_trainer_records_a_finite_representative_trainable_gradient -q
```

The targeted commands observed the missing module and gradient source:

```text
ModuleNotFoundError: No module named 'stage2.evidence'
AttributeError: 'Stage2Trainer' object has no attribute
'_record_representative_gradient_norm'
```

The normal Stage 2 optimizer/global-step disagreement regression was run with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/integration/test_distributed_contract.py::test_normal_stage2_evidence_rejects_optimizer_global_step_disagreement -q
```

It failed because no `ValueError` was raised before the producer call. The
nested validator fixture then failed against the old flat validator with:

```text
ValueError: matrix case witness fields are invalid
```

These failures were causal: the approved single-job interface, observed Stage 2
producer, and current nested producer layout did not yet exist.

### GREEN evidence

Focused four-suite command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_lifecycle_matrix.py \
  tests/multinode/test_result_validation.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/test_distributed_contract.py -q
```

Observed output (exit 0):

```text
84 passed, 6 skipped in 9.70s
```

Affected Task 5/capacity command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_coordinator.py \
  tests/multinode/test_capacity.py -q
```

Observed output (exit 0):

```text
91 passed in 3.74s
```

The affected Stage 2 checkpoint/trainer suite was run through the repository
venv with the already documented trailing-`+` Python-version import
normalization:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/python - <<'PY'
import platform
import pytest
original = platform.python_version
platform.python_version = lambda: original().rstrip('+')
raise SystemExit(pytest.main(['tests/stage2/test_peft_checkpointing.py', '-q']))
PY
```

Observed output (exit 0):

```text
139 passed in 4.60s
```

Wrapper syntax/help/dry-run commands were:

```bash
bash -n run_multinode_lifecycle_matrix.sh
bash -n run_stage2_training.sh
./run_multinode_lifecycle_matrix.sh --help
./run_multinode_lifecycle_matrix.sh \
  --run-id four-gpu-matrix-dry --first-port 29700 \
  --result-root "$TASK10_DRY_ROOT" --dry-run
STAGE2_DRY_RUN=true \
BENCHMARK_EVIDENCE_PATH="$TASK10_STAGE2_EVIDENCE" \
STAGE1_CHECKPOINT=/root/workspace/ILVR/checkpoints/stage1-fixture \
DATA_PATH=/root/workspace/ILVR/data/fixture.json \
IMAGE_ROOT=/root/workspace/ILVR/data/images \
OUTPUT_DIR="$TASK10_STAGE2_CHECKPOINT" LOG_FILE="$TASK10_STAGE2_LOG" \
./run_stage2_training.sh
```

The lifecycle inspection produced exactly 18 JSON plan lines with ports
`29700..29717`, one `lifecycle` object per line, and steps `[0,2,3]`. The
Stage 2 command contained the exact supplied owned producer path. All selected
checkpoint/result/log paths were absent before and after. No SSH or GPU command
ran.

Two independent validator publications compared byte-for-byte with `diff -u`
(exit 0 for JSON and Markdown). After changing shard 2's observed `match`, the
real validator CLI exited 2 and created neither requested output. The exact
generated `/tmp/ilvr-task10-fix.*` fixture was removed afterward.

### Fix-round self-review

- Each matrix cell has one fixed Accelerate job and only the Stage 1 or Stage 2
  approved harness path with exact canonical arguments.
- The lifecycle job uses Task 5 immutable ownership, ordered readiness/start,
  bounded monitoring/cleanup and preflight, then Task 6 checkpoint collection.
- Raw case claims are copied from four-rank harness records and bound by direct
  path, byte size, SHA-256 digest, exact steps, artifact, loss, gradient and
  worker-error fields.
- Normal Stage 2 records observed loss, representative trainable gradient, and
  a callback-observed optimizer step on every rank. The collective validates
  rank `0..3` agreement before exclusive publication and propagates writer
  failure.
- Task 5 supplies a fresh Stage 2 producer path and strictly decodes its bytes
  before writing a successful training witness.
- Final validation reads descriptor-bound no-follow snapshots, validates the
  four JSONLs against sidecars and summary, validates nested Stage 1 child
  witnesses/recommendations, and revalidates Stage 2 producer bytes and the
  run-owned `checkpoint-1`.
- Output JSON and Markdown are fully prepared after validation, prechecked
  absent together, exclusively published, and exact-byte rollback is attempted
  without deleting a swapped foreign file.

## Self-review

- Exact 18-case order, unique child identities, and ports `first_port..+17` are
  validated before side effects.
- Each single harness command retains one child ID and port while performing
  both lifecycle phases around `checkpoint-2`; no retry path exists.
- Dry-run returns before all filesystem and subprocess boundaries.
- Parent namespace freshness is checked before the first attempt; completion is
  the last exclusive publication.
- Matrix completion and case schemas enforce world size 4, machine ranks 0/1,
  process ranks 0-3, steps, method artifact family, finite diagnostics, zero
  exits, and no worker errors.
- Evaluation requires the exact summary, JSONLs and sidecars 0-3, complete
  source coverage, common identities, zero errors, and recomputed accuracy.
- Capacity reads nine nested `capacity/<run-id>/capacity.json` reports and nine
  `stage2-batch1/<run-id>/initial-train-witness.json` files; Stage 1 begins at
  batch 2 and recommendations are successful witnessed attempts, while Stage 2
  is fixed to batch/GPU 1 with strict producer and checkpoint evidence.
- Markdown uses methods as rows and ZeRO 1/2/3 as columns for both stages.
- `git diff --check` passed. Existing untracked `__pycache__` trees were
  preserved and excluded from the commit.

No live hardware result is claimed; that execution was explicitly outside this
Task 10 fake/dry-run scope.

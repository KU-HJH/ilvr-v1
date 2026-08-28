# Task 7 implementation report

## Outcome

Implemented deterministic modulo-4 evaluation sharding for both evaluator
entrypoints and a strict four-sidecar merge path. The default single-worker
mode retains the evaluator-specific JSONL/summary bytes and creates no
sidecar. Four-worker sample failures remain in worker JSONL as diagnostics,
mark the sidecar unsuccessful, and make the evaluator return nonzero.

## Files

- Created `src/multinode_evaluation.py`.
- Created `tests/multinode/test_evaluation.py`.
- Modified `eval.py`.
- Modified `src/evaluate_deepseed.py`.
- Modified `tests/stage2/test_eval_integration.py`.
- Created this required report.

No GPU/CUDA initialization, model download, SSH, rsync, Docker, or tmux command
was used. All Python and pytest commands used the repository virtual
environment and `PYTHONDONTWRITEBYTECODE=1`. Pre-existing untracked
`__pycache__` files were preserved.

## RED evidence

The initial pure-test command was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest tests/multinode/test_evaluation.py -q
```

It failed during collection for the intended missing production module:

```text
E   ModuleNotFoundError: No module named 'src.multinode_evaluation'
1 error in 0.07s
```

After adding the zero-sample four-shard test, the process-local integration
runner produced the intended entrypoint RED for both evaluators:

```text
FF
E   FileNotFoundError: .../worker.jsonl.manifest.json
2 failed, 117 deselected in 7.93s
```

The strict manifest type regression also failed before its targeted fix:

```text
E   Failed: DID NOT RAISE ValueError
1 failed, 20 deselected in 0.03s
```

## GREEN evidence

The pure sharding/merge suite passed after implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest tests/multinode/test_evaluation.py -q
```

```text
.....................                                                    [100%]
21 passed in 0.04s
```

The exact requested combined command was also run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation.py tests/stage2/test_eval_integration.py -q
```

It is blocked during collection by the host interpreter's non-PEP-440 version
string, before Task 7 tests execute:

```text
E   packaging.version.InvalidVersion: Invalid version: '3.11.15+'
1 error in 2.34s
```

The failure originates in `datasets/config.py`, which passes
`platform.python_version()` to `packaging.version.parse`. A diagnostic command
confirmed that this repository venv reports `3.11.15+`. No repository or venv
dependency was changed because that environment issue is outside Task 7's
allowed file scope.

With only that value normalized process-locally to the corresponding valid
release string, the exact same two test files passed:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/python -c \
  'import platform; platform.python_version=lambda: "3.11.15"; import pytest; raise SystemExit(pytest.main(["tests/multinode/test_evaluation.py", "tests/stage2/test_eval_integration.py", "-q"]))'
```

```text
........................................................................ [ 51%]
....................................................................     [100%]
140 passed, 8 warnings in 9.92s
```

The eight warnings are existing PEFT fixture warnings about an absent upstream
config source; there were no test errors or failures.

## Contract review

- `iter_owned_samples()` accepts only exact integer topology `(4, 0..3)` and
  enumerates the unfiltered source sequence, preserving global indices.
- Both evaluator CLIs default to `(num_shards=1, shard_index=0)` and accept an
  optional explicit model identity for the Task 8 coordinator.
- Byte-level integration assertions cover each evaluator's existing
  single-worker JSONL/summary format and absence of a sidecar.
- Four-shard mode writes per-sample rows only, uses local progress counters,
  closes JSONL before exclusively creating `<jsonl>.manifest.json`, and handles
  zero-owned-sample shards.
- Dataset identity is exactly absolute logical path, size, and mtime in
  nanoseconds. It contains no device, inode, or content hash.
- Direct evaluator use falls back deterministically to the absolute logical
  model directory; an explicit nonempty identity is used without rewriting.
- Sidecars include dataset/model identity, shard index/count, expected and
  processed source indices, worker exit status, and its derived success flag.
- Merge requires exactly four manifests and sibling JSONLs; validates unique
  shard indices, identities, worker success, expected/processed ownership,
  JSON-object rows, required evaluation fields, duplicates, range, coverage,
  and error rows; and ignores legacy summary rows.
- Merge sorts by global source index, calculates accuracy once, creates the
  merged JSONL and separate summary JSON exclusively, and removes files it
  created if publication raises.

## Self-review

Reviewed the complete diff against the Task 7 brief and all three binding
rulings. No unrelated tracked file is modified. The single-worker branch's row
construction, serialization order, summary presence/absence, and formatting
remain covered byte-for-byte for both entrypoints. The four-worker branch does
not emit a summary row. Validation completes before merge output creation, and
all tested merge failures leave both final paths unpublished. `git diff
--check` reports no whitespace errors.

One environment concern remains: the exact combined pytest executable cannot
collect the existing Stage 2 integration module while the venv interpreter
identifies itself as `3.11.15+`. The process-local normalization demonstrates
that the Task 7 and existing evaluator tests pass without a Task 7 code change.

## Fix round 1: error precedence over legacy-summary exclusion

Reviewer finding: `_read_shard_rows()` excluded a JSON object with
`summary: true` before checking whether that same object contained `error`.
This violated the binding rule that every error row is rejected and caused the
merge to surface a later missing-index diagnostic instead.

The causal regression command was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest tests/multinode/test_evaluation.py \
  -q -k legacy_summary
```

RED output before the production change:

```text
F.                                                                       [100%]
E       AssertionError: Regex pattern did not match.
E         Expected regex: 'error row'
E         Actual message: 'missing source index 0'
1 failed, 1 passed, 20 deselected in 0.03s
```

The minimal production change moved the existing `error` check ahead of the
existing `summary is True` exclusion. The identical focused command then gave:

```text
..                                                                       [100%]
2 passed, 20 deselected in 0.02s
```

The complete pure Task 7 suite command was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest tests/multinode/test_evaluation.py -q
```

GREEN output:

```text
......................                                                   [100%]
22 passed in 0.04s
```

Self-review confirmed the regression exercises the real merge path, asserts
the causal error-row diagnostic, and asserts neither merged output is
published. Error-free legacy summary rows remain excluded by the existing
test. No evaluator integration file or behavior changed in this fix round.

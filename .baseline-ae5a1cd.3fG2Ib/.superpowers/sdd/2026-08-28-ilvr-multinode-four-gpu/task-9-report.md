# Task 9 report: Stage 1 multi-node capacity controller

## Outcome

Implemented the strict Stage 1 capacity probe through the existing multi-node
training coordinator. The public wrapper starts at batch/GPU 2, uses the
caller-supplied four distinct ports for 2, 4, 8, and the conditional batch-1
fallback, and exclusively publishes the bounded parent `capacity.json`.

No SSH, Docker, tmux, CUDA, GPU, or live training operation was run while
implementing or validating this task. All execution-boundary coverage used
fakes; public CLI checks used only `--help` and `--dry-run`.

## TDD evidence

### Initial RED and GREEN

The brief's initial test command failed because the module did not exist:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest tests/multinode/test_capacity.py -q

E   ModuleNotFoundError: No module named 'src.multinode_capacity'
1 error in 0.07s
```

After implementing only candidate generation, fallback authorization, and
failure classification, the same command reported:

```text
7 passed in 0.01s
```

The controller tests were then added before the controller API. Collection
failed on the missing public types:

```text
E   ImportError: cannot import name 'CapacityEvidence' from
E   'src.multinode_capacity'
1 error in 0.07s
```

After the bounded controller was added, the batch-2 fallback regression caught
an early loop exit:

```text
FAILED test_candidate_two_oom_runs_batch_one_exactly_once_on_fourth_port
E assert [2] == [2, 1]
1 failed, 15 passed in 0.08s
```

Changing the candidate-2 OOM branch to execute exactly one batch-1 attempt on
the fourth port produced:

```text
16 passed in 0.06s
```

### Causal boundary regressions

A zero-exit child with OOM text but no valid witness initially unlocked the
fallback incorrectly:

```text
FAILED test_zero_exit_without_witness_cannot_use_oom_text_to_unlock_fallback
E assert [2, 1] == [2]
1 failed, 20 passed in 0.23s
```

The controller now classifies every zero exit lacking the exact valid child
witness as `integration_failure`, regardless of log text.

An exclusive-report regression initially showed that a duplicate parent ran
children before discovering the existing report:

```text
FAILED test_capacity_report_is_exclusive
E AssertionError: assert [<two child commands>] == []
1 failed, 20 deselected in 0.06s
```

The parent report is now checked before any launch and remains protected by the
existing exclusive no-replace publisher against races.

Self-review exposed an existing Stage 1 coordinator integration blocker. The
coordinator passed `checkpoint-N` to `write_training_witness()`, but that API
requires the run-owned checkpoint root named after the child run ID. The new
fake-only CLI regression failed with:

```text
training attempt incomplete: training witness checkpoint must be owned by the run ID
1 failed, 21 deselected in 0.11s
```

The one-line caller fix now passes the run-owned checkpoint root; the same
focused regression reports:

```text
1 passed, 21 deselected in 0.04s
```

Finally, a simulated missing child executable escaped without an attempt
record:

```text
FAILED test_subprocess_runner_records_child_launch_error_as_integration
E OSError: executable unavailable
1 failed, 22 deselected in 0.06s
```

The launch boundary now records that as bounded coordinator evidence and
returns an integration failure without querying nonexistent node logs:

```text
1 passed, 22 deselected in 0.04s
```

Staged-diff review then caught an overly broad bare `checkpoint` match that
would hide a real OOM when a normal `checkpoint-2` path appeared in the log:

```text
FAILED test_only_cuda_capacity_text_without_integration_markers_is_oom
E AssertionError: assert 'integration_failure' == 'capacity_oom'
1 failed, 5 passed, 18 deselected in 0.05s
```

Checkpoint, SSH, Docker, and rendezvous text now overrides OOM only when paired
with failure language. NCCL, collective mismatch, and nonfinite diagnostics
remain unconditional integration markers. The targeted set then reported:

```text
6 passed, 18 deselected in 0.04s
```

## GREEN verification

The focused Task 9 and existing coordinator regression set reported:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_capacity.py tests/multinode/test_coordinator.py -q

67 passed in 0.56s
```

The wrapper syntax, public help, and a fully specified dry run all returned
zero. An independent JSON assertion confirmed primary batches `[2, 4, 8]`,
ports `[29801, 29802, 29803]`, and fallback batch/port `1/29804`; it printed:

```text
dry-run contract validated
```

The dry-run parent path remained absent. `git diff --check` returned zero.

## Self-review

- Primary execution is strictly 2→4→8. The first OOM at 4 or 8 stops and
  recommends the last valid witness; candidate-2 OOM alone replaces the queue
  with one batch-1 attempt on the fourth port.
- Any integration failure clears the recommendation and stops immediately.
  NCCL, collective mismatch, nonfinite loss/gradient, and checkpoint, SSH,
  Docker, or rendezvous failures override CUDA OOM markers.
- Every candidate uses the exact safe child ID `<parent>-batch<N>`, the existing
  `train-stage1` coordinator, and one distinct validated port.
- A success requires the exact child witness identity plus the still-matching
  bounded finite-loss evidence size and SHA-256. Zero exit alone is insufficient.
- Failure classification consumes only capped tails of exact child coordinator,
  local-node, and remote-node paths. Remote evidence paths are allowlisted to
  the exact child. The parent JSON stores paths, never log contents.
- `capacity.json` is an ordered, exclusive report with batch, child run ID,
  port, exact status, evidence paths, and an integer recommendation or null.
- Help and dry run construct no runner, create no directory, publish nothing,
  and perform no external or GPU work.
- Pre-existing untracked `__pycache__` files/directories were left untouched.
- An independent reviewer subagent was not used because the Task 9 assignment
  explicitly prohibited subagents; the checks above are the required
  line-by-line self-review.

## Deferred hardware concern

Live SSH/Docker/tmux behavior, remote bounded evidence retrieval, CUDA OOM
classification on real node logs, and four-GPU training remain intentionally
deferred to the planned hardware validation task.

## Fix round 1/5

Four Important review findings were reproduced and fixed without external or
hardware execution.

### RED: line-local conditional integration classification

The causal table included the two binding examples, an allocator OOM beside a
normal checkpoint path, cross-line subject/token separation, and unconditional
NCCL/nonfinite overrides:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest tests/multinode/test_capacity.py -q \
  -k conditional_integration_subjects

4 failed, 3 passed, 37 deselected in 0.08s
```

The old global predicate combined conditional subjects with failure substrings
from unrelated lines and from `OutOfMemoryError`/allocator OOM phrases. The new
classifier preserves global NCCL, collective-mismatch, and nonfinite overrides,
then processes each remaining line independently after removing recognized CUDA
OOM exception phrases. Conditional subjects require a standalone failure token
on that sanitized line. The classifier-focused set then reported:

```text
13 passed, 31 deselected in 0.04s
```

### RED: canonical remote evidence environment

The fake control boundary decoded the complete fixed SSH/Docker/bash command
and failed because the container argv began directly with Python:

```text
FAILED test_subprocess_runner_rejects_remote_evidence_outside_exact_child
E At index 0 diff: '/root/workspace/ILVR/venv/ilvr-venv/bin/python' != 'env'
1 failed, 43 deselected in 0.07s
```

The command now matches the Task 4 pattern exactly inside the fixed container:
`env PYTHONPATH=/root/workspace/ILVR` followed by the canonical absolute venv
Python and `-m src.multinode capacity-evidence-show`. The same test reported:

```text
1 passed, 43 deselected in 0.05s
```

No SSH, Docker, or shell command executed; the injected control runner inspected
argv and returned fake JSON.

### RED: exact child witness decoding

Parameterized real-file/controller tests mutated top-level and nested witness
fields, JSON scalar types, ranks, exit codes, and finite-loss metadata. Ten of
fourteen malformed variants incorrectly authorized the next candidate:

```text
10 failed, 4 passed, 32 deselected in 0.23s
```

The replacement decoder rejects duplicate, extra, and missing fields and
requires exact JSON container/scalar types. In particular, floats and booleans
cannot compare equal to integer schema, batch, world-size, rank, exit-code, or
finite-evidence fields. It also validates the exact nested finite-loss field set
before rechecking producer bytes, SHA-256, size, and finite status. The same
table then reported:

```text
14 passed, 32 deselected in 0.07s
```

### Task 12 plan reconciliation and dry-run regression

The real wrapper accepted the corrected Task 12 shape before the prose edit:

```text
1 passed, 43 deselected in 0.10s
```

The authorized plan update now uses `--tuning-method` and four repeated,
distinct `--main-process-port` values for every capacity cell. Task 10 and Task
12 consumers refer to `attempts[0].batch_per_gpu` and
`recommended_batch_per_gpu`, not the nonexistent `micro_batch`/implicit
selection fields. A separate manual Task 12-shaped dry run printed:

```text
Task12-shaped dry run validated without publication
```

It used four ports, method DoRA, canonical real-model/input paths, and
`--max-steps 1`; the parent result path remained absent.

### Fix-round GREEN verification and self-review

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_capacity.py tests/multinode/test_coordinator.py -q

89 passed in 0.68s
```

`bash -n run_multinode_stage1_batch_probe.sh`, public wrapper help, internal
evidence help, the exact Task 12-shaped dry run, its no-publication assertion,
and `git diff --check` all returned zero.

- Conditional integration evidence is now line-local and OOM phrases cannot
  donate their `Error`/`failed` tokens to normal checkpoint paths.
- The remote evidence subprocess is cwd-independent and retains the fixed
  noninteractive SSH/Docker boundary.
- Recommendations are impossible until the exact child witness schema, nested
  types, ranks/exits, finite producer reference, and digest all validate.
- Task 12 assigns ports 30000–30035 across nine capacity cells without overlap;
  Stage 2 remains in its separate 30100+ range.
- The checkpoint-root caller fix remains unchanged and covered.
- Pre-existing untracked `__pycache__` files/directories remain untouched.
- Live remote/container/CUDA/four-GPU validation remains deferred to Task 12.

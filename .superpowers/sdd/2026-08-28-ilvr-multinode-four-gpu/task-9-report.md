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

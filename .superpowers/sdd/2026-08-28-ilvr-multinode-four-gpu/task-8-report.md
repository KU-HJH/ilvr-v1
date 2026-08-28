# Task 8 report — four-worker evaluation coordinator

## Result

Implemented the fixed four-worker evaluation coordinator and shell wrapper.
Remote GPU 0/1 own shards 0/1 and local GPU 0/1 own shards 2/3. A Task 6
prepared inference manifest is validated before worker command creation, its
observed stage selects `eval.py` or `src/evaluate_deepseed.py`, and its digest
is passed explicitly as the same model identity to all four evaluators.

Each shard uses a distinct attempt-scoped tmux session, ownership record, exit
record, and exact `CUDA_VISIBLE_DEVICES`. Only exact live owners can reach the
existing node-side pidfd signal boundary. Any nonzero or invalid/incomplete
owned exit, any missing/invalid Task 7 sidecar, or any shard error prevents the
merge. Remote result transfer is one explicit NUL-delimited shard 0/1 JSONL and
sidecar list with `rsync --archive --ignore-existing`; it never copies a cache
or output tree recursively.

Task 6 inference copies now publish a bounded no-replace sibling manifest. Task
8 validates the remote prepared tree against that manifest, creates one fresh
verified local inference copy through Task 6's public copy protocol, and uses
the remote and local model directories on their assigned nodes respectively.

## RED evidence

Initial Task 8 test command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation_coordinator.py -q

ImportError: cannot import name 'build_evaluation_pull_command' from
'src.multinode_artifacts'
1 error in 0.10s
```

The focused causal regression for a malformed owned exit record was also
observed failing before its control-flow fix:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation_coordinator.py -q \
  -k invalid_owned_exit_record

ValueError: invalid owned exit record
1 failed, 9 deselected in 0.09s
```

The failure showed that a malformed completion record escaped without asking
the exact ownership boundary to clean up the still-pending peers.

A second causal sidecar test supplied an exit-zero sidecar whose processed
indices were incomplete. Before the strict pre-merge gate, the focused command
failed with `TypeError: validate_evaluation_sidecars() got an unexpected
keyword argument 'expected_sample_count'` (`1 failed, 10 deselected in 0.07s`).

## GREEN evidence

After the minimal implementation and the causal completion-record fix:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation_coordinator.py -q \
  -k invalid_owned_exit_record

1 passed, 9 deselected in 0.05s
```

The incomplete-sidecar causal command then reported `1 passed, 10 deselected
in 0.06s` after the pre-merge gate required exact expected and processed index
sets.

The complete Task 7 + Task 8 focused set then reported:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation.py \
  tests/multinode/test_evaluation_coordinator.py -q

33 passed in 0.10s
```

The Task 5 ownership/coordinator regression set reported `43 passed in 0.48s`.
The Task 6 inference-preparation compatibility selection reported `6 passed,
113 deselected in 0.19s`.

## Shell and dry-run checks

`bash -n run_multinode_evaluation.sh` returned zero.
`PYTHONDONTWRITEBYTECODE=1 ./run_multinode_evaluation.sh --help` returned zero
and listed the required model, dataset, task, output, and run flags. A dry run
returned two observed-stage command maps, each with four immutable worker
arrays, plus exactly four remote result names; it performed no Task 6 copy,
SSH, rsync, tmux, evaluator import, filesystem creation, CUDA import, or GPU
operation.

## Self-review

- Worker commands use the fixed host/container/repository/venv topology and
  preserve argv boundaries through the existing remote command builder.
- Task 6 identity validation and local copy finish before any worker command is
  built or tmux session is started in an actual run.
- The Stage is read only from the validated manifest; no path or caller flag
  selects an evaluator.
- Result and control publication is fresh/no-replace. The two remote JSONLs and
  their two sidecars are the complete rsync allowlist.
- Cleanup passes the exact run, attempt, node rank, PID, start ticks, commit,
  command digest, argv, and environment to the existing pidfd-backed
  `signal-owned` command. No PID/name/session/GPU-wide signaling was added.
- Tests used temporary files, fakes, and dry-run/help only. No real SSH, rsync,
  Docker, tmux, evaluator, CUDA, or GPU action was performed.
- Pre-existing untracked `__pycache__` files were preserved and excluded from
  the commit.

The remaining external concern is intentional: Task 12 must validate the real
remote prepared-artifact provisioning, Docker namespace paths, rsync host-path
mapping, tmux ownership lifecycle, and evaluator/CUDA behavior on hardware.

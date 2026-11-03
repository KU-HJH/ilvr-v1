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

## Fix round 1

Addressed all three review findings. Every artifact rsync command now retains
the fixed SSH transport and `--` boundary while adding `--secluded-args`.
Coordinator-owned remote artifact and result subpaths accept only canonical
alphanumeric, dot, underscore, and hyphen components beginning with an
alphanumeric character. Backticks, dollar signs, quotes, colons, controls,
absolute paths, and traversal are rejected while the plan or command argv is
still being built.

The production evaluation runner now reserves the local Task 5 attempt
directory and then the remote Task 5 attempt directory through the exclusive
`prepare-attempt` interfaces before output reservation, artifact copy, or any
worker launch. Both are called once; an existing local or remote control
namespace blocks launch.

The coordinator records each worker as possibly live before invoking its tmux
launch subprocess. Launch and poll failures clean every worker without a
proven exit independently. Cleanup failures are collected as notes on the
primary exception, so no peer is skipped and the original failure is retained.

### Fix-round RED evidence

The path/rsync causal set first exposed eight failures:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation_coordinator.py -q \
  -k 'secluded or unsafe_result_component or backtick_source or remote_result_pull'

8 failed, 10 deselected
```

The failures were the absent `--secluded-args` option and unsafe path cases
that did not raise before command construction. After the security change, the
same command reported:

```text
8 passed, 10 deselected in 0.06s
```

The ambiguous worker lifecycle causal set initially reported three failures:
the launch-timeout shard was omitted from cleanup, a poll timeout performed no
cleanup, and the first cleanup exception masked the primary launch timeout.
After moving insertion before launch and isolating cleanup calls, that set
reported `3 passed, 20 deselected in 0.06s`.

The final cleanup-error collection assertion was deliberately added after the
best-effort loop and failed because the swallowed error was not observable:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation_coordinator.py -q -k cleanup_exception

1 failed, 27 deselected in 0.08s
```

After preserving cleanup failures as notes on the primary exception, the exact
command reported:

```text
1 passed, 27 deselected in 0.07s
```

A final production-runner causal test distinguished a safely missing ownership
record from failure after an exact owner had been validated:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation_coordinator.py -q \
  -k exact_owned_signal_failure

1 failed, 29 deselected in 0.08s
```

The runner had returned `False` and lost the signal failure. It now raises that
failure for the coordinator's isolated cleanup collector, while a missing or
invalid ownership record remains an ignored `False`. The same command then
reported `1 passed, 29 deselected in 0.06s`.

Direct production-runner tests using only fake command execution cover fresh
local/remote control preparation, stale local and remote rejection, inference
preparation ordering, poll timeout propagation, exact NUL-list result
transfer, and exact Task 5 owned cleanup. No test invoked SSH, rsync, Docker,
tmux, an evaluator, CUDA, or a GPU.

### Fix-round GREEN verification

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_evaluation.py \
  tests/multinode/test_evaluation_coordinator.py -q

52 passed in 0.13s
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_artifacts.py tests/multinode/test_coordinator.py -q

162 passed in 3.84s
```

`bash -n run_multinode_evaluation.sh`, wrapper `--help`, a fully specified
evaluation `--dry-run`, and `git diff --check` all returned zero. Help and dry
run remained local-only and mutation-free.

### Fix-round self-review

- The two exact rsync argv assertions include `--archive`,
  `--ignore-existing`, `--secluded-args`, the NUL allowlist, fixed SSH
  transport, `--`, and canonical source/destination operands.
- Canonical path checks run before runner construction or transfer invocation;
  the backtick cases therefore cannot reach an external command.
- Only workers lacking a proven owned exit remain cleanup candidates. The
  production cleanup still delegates identity and liveness proof to Task 5 and
  safely ignores a missing ownership record.
- Local preparation precedes remote preparation, both precede launch, and
  failure in either namespace leaves `_controls_prepared` false.
- The intentional Task 12 hardware concern remains unchanged.

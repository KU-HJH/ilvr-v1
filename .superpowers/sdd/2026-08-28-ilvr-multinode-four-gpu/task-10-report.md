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

## Review fix round 2/5

This section supersedes the round-1 description of one harness process doing
both phases. Each cell now uses two separately owned fixed harness jobs with the
same run ID and port: `initial` produces and validates `checkpoint-2`, Task 6
collects and redistributes that same logical checkpoint to the local node, and
only then does `resume` start and advance step 2 to 3. The legacy in-process
`lifecycle` harness default remains available for its existing tests, but the
matrix command builder accepts only the exact `initial` or `resume` phase.

### RED evidence

The first phase/evidence selection failed during collection because the normal
Stage 2 producer did not expose the required observed-artifact interface:

```text
ImportError: cannot import name 'observe_stage2_checkpoint_artifact'
```

The first validator run reported 14 errors before product validation because
the new canonical checkpoint root was patched in the result validator fixture
but not in the shared lifecycle evidence decoder. After fixing that test-only
root binding, the validator exposed no product failures and passed 14/14.

An additional exact-type regression was deliberately added and run with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_result_validation.py::test_validate_results_rejects_bool_or_float_identity_types
```

Observed RED (exit 1):

```text
..FF
2 failed, 2 passed in 3.05s
```

The failures proved that `False == 0` could satisfy a matrix attempt step and
that `[False] == [0]` could satisfy an evaluation sidecar source-index list.
Explicit container and element type checks closed both gaps.

The documented-direct-path regression was also run before its checks existed:

```text
FFFF
4 failed in 2.93s
```

Unexpected direct JSON under the evaluation root, capacity root, a Stage 1
capacity run, and a Stage 2 batch-1 run was still ignored. Exact direct-child
sets were then added for those producer namespaces and canonical matrix raw
phase directories.

One final causal check replaced the fixture checkpoint root with the repository's
documented fixed logical symlink. It failed because the generic no-follow tree
opener rejected the root symlink before reaching its protected descendants.
The checkpoint snapshot now resolves only that fixed logical root, opens the
physical root, and retains no-follow descriptors for every run/checkpoint child.
Symlinked descendants remain rejected. The resulting standalone validator suite
passed 21/21.

### GREEN evidence

Focused Task 10 suites:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_lifecycle_matrix.py \
  tests/multinode/test_result_validation.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/test_distributed_contract.py
```

```text
112 passed, 6 skipped in 10.47s
```

The six skips remain the opt-in real distributed/GPU harness executions.
Standalone validator verification was `21 passed in 3.24s`.

Affected Task 5, Task 6 and capacity suites:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_coordinator.py \
  tests/multinode/test_artifacts.py \
  tests/multinode/test_capacity.py
```

```text
211 passed in 4.28s
```

Affected Stage 2 checkpoint suite, using the repository's existing
trailing-`+` interpreter-version import normalization:

```text
139 passed in 4.65s
```

Syntax and dry-run inspection passed:

```text
wrapper: help ok; 18 plans; ports 29700..29717;
initial/redistribute/resume; no result root
AST OK: 8 files
```

The wrapper emitted exactly 18 JSON objects. Every object had ordered
`initial`, `redistribute`, and `resume` lifecycle members, initial steps `[0,2]`,
resume steps `[2,3]`, exact phase arguments in both node command arrays, and no
filesystem publication, SSH execution, or GPU work.

Two independent fake validator publications were compared with literal
`diff -u` for both complete JSON and Markdown outputs. Both diffs returned zero;
the report covered 18 matrix cases, four evaluation samples, nine Stage 1
capacity cells, and nine Stage 2 batch-1 cells. The exact generated directory
`/tmp/ilvr-task10-round2-validator-symlink-ogdrva1m` used the documented fixed
checkpoint-root symlink and was removed when verification finished; it is not
stored in the repository or result tree.

### Round-2 self-review

- Both approved harnesses retain `phase=lifecycle` as their default and expose
  exact `initial` and `resume` phases. Each phase publishes its own four-rank
  loss, gradient, optimizer/global step, artifact and error observations.
- Global rank 0 classifies the real saved artifact and broadcasts the observed
  family. Machine-local leaders then publish identical agreed bytes on both
  node-local filesystems; either leader's writer failure propagates.
- The coordinator compares the remote and local initial evidence bytes,
  validates the phase schema, performs Task 6 collection and same-path
  redistribution, and does not create the resume process set on transfer
  failure. It repeats the node-byte comparison and strict validation for the
  separately observed resume evidence.
- The production command test inspects both real Accelerate arrays and rejects
  arbitrary harness paths/arguments. Remote evidence reads use only the fixed
  bounded `lifecycle-evidence-show --run-id ... --phase ...` node CLI; local
  reads use the descriptor-bound bounded reader.
- A direct nonshared-filesystem regression proves the Stage 2 resume phase
  fails before the model loader when the local checkpoint is absent.
- Launch/readiness/poll/control exceptions retain the primary error, attach
  cleanup notes, and independently attempt exact-owned cleanup for every
  possibly-live node.
- Validation snapshots all allowed evidence bytes and checkpoint metadata from
  pinned no-follow directory descriptors, including child-directory inode/dev
  revalidation. It then validates only those immutable observations and exact
  canonical raw/checkpoint paths.
- Matrix attempts and evaluation sidecars now reject boolean/float substitutions
  for integer identities. Stage 2 producer bytes, digest, all-rank diagnostics,
  observed artifact family and canonical run-owned `checkpoint-1` remain
  cross-validated.
- Output rollback records the created device/inode and removes the destination
  only while it still matches; the same-byte foreign-inode swap regression
  leaves the foreign file intact.
- `git diff --check` passed. Pre-existing untracked `__pycache__` paths were
  preserved. No real SSH, GPU, or training command was run.

## Fix round 3: strict types, pinned checkpoint root, atomic rollback capture

The round-2 review found five remaining contract gaps. This round used focused
causal tests for each one and made only the corresponding evidence decoder,
validator publication/snapshot, and fixed remote-read command changes.

### RED evidence

The two raw-evidence decoders and result validator were first exercised with
boolean/float aliases in top-level identities, rank lists, finite flags, and a
nested matrix case exit code:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_lifecycle_matrix.py::test_lifecycle_phase_decoder_rejects_top_level_bool_float_aliases \
  tests/stage2/integration/test_distributed_contract.py::test_stage2_producer_decoder_rejects_top_level_bool_float_aliases \
  tests/multinode/test_result_validation.py::test_validate_results_rejects_bool_or_float_identity_types
```

```text
11 failed, 4 passed in 2.50s
```

All 11 failures were equality-compatible JSON aliases accepted by the old
decoders. Exact `type(...) is ...` checks were added before reconstruction,
including both completion-case and raw-case exit-code containers and values.

The checkpoint-root regression retargeted the documented logical symlink after
its first resolution. Before the fix, later run snapshots followed the new
target:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_result_validation.py::test_validate_results_pins_checkpoint_root_once_before_symlink_retarget
```

```text
1 failed in 3.05s
```

Two publication causals then reproduced the rollback bugs. One swapped a
same-byte foreign inode immediately after the old identity `stat`; the old
rollback unlinked that foreign file. The other failed the first parent-directory
`fsync` after the manifest link; the linked manifest remained because ownership
had not yet returned to the caller.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_result_validation.py::test_rollback_does_not_unlink_foreign_swap_after_identity_check \
  tests/multinode/test_result_validation.py::test_parent_fsync_failure_after_link_rolls_back_created_output
```

```text
2 failed in 2.92s
```

Finally, the fake runner inspected the complete production remote argv. It
showed that the bounded lifecycle evidence read lacked a fixed container
working directory and explicit repository `PYTHONPATH`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_lifecycle_matrix.py::test_remote_phase_evidence_uses_only_the_fixed_bounded_node_cli
```

```text
1 failed in 0.07s
```

### GREEN evidence

The strict-type selection passed 16/16 after exact top-level and nested type
checks. The checkpoint retarget test passed 1/1 after resolving the fixed
logical root once, retaining that physical root descriptor across the whole
validation transaction, and opening all 27 run directories fd-relative.

Publication now returns ownership immediately after the exclusive link. Temp
cleanup and parent sync happen only after the caller stores that ownership.
Rollback retains the already trusted parent fd and uses Linux
`renameat2(RENAME_NOREPLACE)` to atomically capture the current destination
before inspecting its inode. An owned capture is removed; a foreign capture is
restored without replacing any concurrently created destination. This is a
rename-capture protocol, not a claim that Linux provides conditional pathname
unlink. The five focused existing/new publication tests passed:

```text
5 passed in 2.99s
```

The remote lifecycle evidence command now has exactly this nested boundary:

```text
/usr/bin/docker exec --workdir /root/workspace/ILVR llm_kd /usr/bin/bash -lc \
  "env PYTHONPATH=/root/workspace/ILVR \
   /root/workspace/ILVR/venv/ilvr-venv/bin/python -m src.multinode \
   lifecycle-evidence-show --run-id <fixed-child-id> --phase <initial|resume>"
```

The exact argv causal and the unchanged generic remote quoting test passed 2/2.

Final focused Task 10 suites:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_lifecycle_matrix.py \
  tests/multinode/test_result_validation.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/integration/test_distributed_contract.py
```

```text
127 passed, 6 skipped in 10.55s
```

The skips remain only the opt-in real distributed/GPU executions. Affected
Task 5/6/capacity/preflight suites passed:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_coordinator.py \
  tests/multinode/test_artifacts.py \
  tests/multinode/test_capacity.py \
  tests/multinode/test_preflight.py
```

```text
237 passed in 4.27s
```

The affected Stage 2 checkpoint suite, with the repository's documented
trailing-`+` interpreter normalization, passed `139 passed in 4.73s`.

Wrapper syntax/help and in-memory dry-run inspection also passed:

```text
wrapper: syntax/help ok; 18 plans; ports 29700..29717;
initial/redistribute/resume; no result root
```

### Round-3 self-review

- Raw lifecycle and Stage 2 producer decoders reject boolean/float aliases
  before equality-based reconstruction; rank record builders retain their exact
  nested integer and finite-number checks.
- Completion-case and raw case exit maps require exact containers, exact keys,
  exact lengths, and exact integer exit values.
- The checkpoint physical root is resolved and opened once before any evidence
  snapshot and remains pinned until validation finishes; every run/checkpoint
  descendant is traversed only relative to retained no-follow descriptors.
- Publication ownership includes the created device/inode and retained parent
  descriptor before any parent sync can fail. Rollback never uses the old
  stat-then-unlink decision and preserves foreign replacements.
- The only changed remote command is the fixed lifecycle evidence read. Other
  existing remote commands retain their established argv, as confirmed by the
  affected preflight/coordinator suites.
- `git diff --check` passed. Pre-existing untracked `__pycache__` paths were
  preserved. No SSH, Docker, GPU, tmux, rsync, or real training command ran.

## Fix round 4: one atomic immutable report bundle

This section supersedes every earlier description of independently publishing
and rolling back JSON and Markdown output names. The validator now accepts one
`--output-root` naming an absent bundle directory. It prepares the fixed
`validation-manifest.json` and `validation-summary.md` files inside one
exclusively created mode-private sibling staging directory, fsyncs each regular
file and the staging directory, freezes the files/directory read-only, and
performs one Linux `renameat2(RENAME_NOREPLACE)` into the final name. It then
fsyncs the parent directory.

No destructive action occurs after the final rename. If parent fsync fails,
the caller receives `ReportBundleDurabilityError` explaining that durability
is ambiguous, while the complete two-file bundle remains visible. Failures
before rename remove only the fixed files inside the retained private staging
directory and that private directory itself. There are no rollback, capture,
or independent-output publication names in the production protocol.

### RED evidence

The new bundle publication and CLI causals were run before changing production:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_result_validation.py \
  -k 'publish_validation_outputs_writes_matching_tables_exclusively or second_bundle_file_failure or final_bundle_rename_collision or parent_fsync_failure_leaves_complete_bundle or foreign_independent_output_names or validate_results_cli_publishes_only_output_root_bundle'
```

Observed output (exit 1):

```text
7 failed, 21 deselected in 3.29s
```

Six publisher tests failed because the old function still required separate
manifest and summary paths. The CLI test failed because it still required
`--output-manifest` and `--output-summary`. These were the intended missing
bundle-interface failures.

A mutation check then covered failure immediately after exclusive staging
directory creation but before its descriptor opens:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_result_validation.py::test_staging_open_failure_removes_only_private_empty_directory
```

Observed RED (exit 1): the private `.validation-report.<token>.staging`
directory remained. Recording staging creation separately from descriptor
acquisition closed that pre-rename cleanup gap.

The final mutation check injected an exception immediately after the real
no-replace rename succeeded:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q \
  tests/multinode/test_result_validation.py::test_exception_after_successful_rename_never_cleans_final_bundle
```

Observed RED (exit 1): the old flag-based `finally` unlinked both files through
the retained directory descriptor and then masked the injected
`KeyboardInterrupt` with `FileNotFoundError`. Cleanup is now confined to
preparation exceptions and `renameat2` results that definitively report
failure. Ambiguous unwind after the syscall only closes retained descriptors.

### GREEN evidence

The complete result-validator test file now passes:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q tests/multinode/test_result_validation.py
```

```text
30 passed in 3.53s
```

The causals prove that a second-file write or fsync failure leaves the final
bundle absent; a final rename collision preserves the foreign directory and
its bytes; parent-fsync failure leaves both complete readable files and never
rolls back; foreign sibling `validation-manifest.json` and
`validation-summary.md` files are not consulted, moved, or removed; and no
hidden staging/rollback/capture name remains after the tested success or
failure paths. A staging-open failure likewise cleans its one private empty
directory, and an exception immediately after successful rename retains both
complete final files.

CLI help now exposes only the bundle path:

```text
--matrix-root MATRIX_ROOT --evaluation-root EVALUATION_ROOT
--capacity-root CAPACITY_ROOT --output-root OUTPUT_ROOT
```

Task 10 and Task 12 plan references use the same interface. The Task 12
operator path is `$hardware_result_root/validation-report/`, containing the
two fixed report files. `git diff --check` passed. Pre-existing untracked
`__pycache__` paths were preserved. No SSH, Docker, GPU, tmux, rsync, or real
training command ran.

# Task 10 report: 18-case tiny lifecycle matrix

## Result

Implemented the deterministic four-GPU lifecycle matrix, parameterized both tiny
harnesses, added the strict final evidence validator and paired JSON/Markdown
publisher, and added the operator wrapper. No SSH, GPU, or live training command
was executed; verification used unit fakes and dry-run output only.

The matrix order is exactly stage (`stage1`, `stage2`), then method (`full`,
`lora`, `dora`), then ZeRO (`1`, `2`, `3`). Every plan binds a safe child ID,
one port, an initial `0 -> 2` command, and a resume `2 -> 3` command. Live
orchestration validates each existing coordinator witness, numbered checkpoint
output, and digest-bound finite-loss evidence before it invokes the next
attempt. It does not retry. Case witnesses are published exclusively and the
parent completion marker is published only after all 18 pairs validate.

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
 'initial_resume_arrays': True, 'ssh_tokens': 0}
```

The chosen result root was absent before and after the command. Help and dry-run
therefore performed no filesystem publication, SSH, or GPU work.

## Fake validator and output diff verification

The complete fake evidence tree is stored at
`results/task10-validator-fake-5337593/` (44 bounded files, 200 KiB). The real
CLI was run as follows:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src" \
./venv/ilvr-venv/bin/python -m src.multinode validate-results \
  --matrix-root results/task10-validator-fake-5337593/matrix \
  --evaluation-root results/task10-validator-fake-5337593/evaluation \
  --capacity-root results/task10-validator-fake-5337593/capacity-root \
  --output-manifest results/task10-validator-fake-5337593/validation.json \
  --output-summary results/task10-validator-fake-5337593/validation.md
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

## Self-review

- Exact 18-case order, unique child identities, and ports `first_port..+17` are
  validated before side effects.
- Initial and resume commands retain the same child ID and port; resume names
  `checkpoint-2`, and no retry path exists.
- Dry-run returns before all filesystem and subprocess boundaries.
- Parent namespace freshness is checked before the first attempt; completion is
  the last exclusive publication.
- Matrix completion and case schemas enforce world size 4, machine ranks 0/1,
  process ranks 0-3, steps, method artifact family, finite diagnostics, zero
  exits, and no worker errors.
- Evaluation requires the exact summary plus sidecars 0-3, one unique source
  index per sidecar, common dataset/model identities, and successful workers.
- Capacity reads only the nine direct `capacity/` reports and nine direct
  `stage2-batch1/` witnesses; Stage 1 begins at batch 2 and reads the public
  recommendation field, while Stage 2 is fixed to batch/GPU 1.
- Markdown uses methods as rows and ZeRO 1/2/3 as columns for both stages.
- `git diff --check` passed. Existing untracked `__pycache__` trees were
  preserved and excluded from the commit.

No live hardware result is claimed; that execution was explicitly outside this
Task 10 fake/dry-run scope.

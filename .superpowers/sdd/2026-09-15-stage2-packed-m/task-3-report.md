# Task 3 delivery report — packed M coordinator/examples/DDP witness

## Delivery

The existing Stage 2 coordinator allowlist now forwards the value-bearing
`--stage2_m_execution` flag. The existing CE, entropy, and combined wrapper
examples all explicitly select `packed_suffix`; no launcher or coordinator was
added. Missing mode remains parser/config default `full_replay`.

The native two-process CPU Trainer witness now runs both `full_replay` and
`packed_suffix`, preserving its independent full-replay optimizer oracle,
unequal/zero-block rank layout, accumulation windows `[2,2,1]`, and remainder
step. Packed runtime metrics are literal: model M `[0,0,0]`, decoder M
`[2,0,1]`, branches `[3,0,1]`, and query rows `[3,0,1]`. A direct cached-decoder
pre-hook independently observed packed decoder calls `[2,0,1]`; the student
model forward hook independently observed packed top-level M calls `[0,0,0]`.

## TDD evidence

RED commands (before Task 3 production edits):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/multinode/test_native_ddp_launch.py -xq
# 1 failed, 4 passed — --stage2_m_execution is not an approved Stage 2 argument

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_objective_examples.py::test_each_example_forwards_packed_m_through_the_actual_coordinator_dry_run -q
# 3 failed — each actual coordinator dry-run resolved the missing wrapper flag as full_replay
```

GREEN commands:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/multinode/test_native_ddp_launch.py -xq
# 31 passed in 17.93s

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage2/test_objective_examples.py -q
# 29 passed in 10.01s

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest tests/stage2/test_objective_distributed.py -q
# 2 passed in 26.20s

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest \
  tests/multinode/test_native_ddp_launch.py \
  tests/stage2/test_objective_distributed.py \
  tests/stage2/test_objective_examples.py -q
# 63 passed in 52.49s

bash -n run_stage2_objective_examples.sh run_stage2_training.sh run_multinode_stage2_training.sh
git diff --check
# both exited 0
```

Raw output is in `logs/packed-m-implementation-20260915/`.

## Files

- `src/multinode_runtime.py`
- `run_stage2_objective_examples.sh`
- `tests/multinode/test_native_ddp_launch.py`
- `tests/stage2/test_objective_distributed.py`
- `tests/stage2/test_objective_examples.py`
- `docs/stage2-objective.md`
- `results/packed-m-implementation-20260915/verification.md`

## Limits

Only CPU checks ran. No GPU or training launch, remote synchronization, ZeRO
test, two-host run, real-image comparison, or memory/throughput measurement was
performed. Normal latent recurrence is unchanged. In full replay,
`m_query_rows` is the conceptual selected contiguous suffix-row total; in
packed mode it is actual Q sent to the single M decoder. Neither is a memory or
speed claim.

## Self-review

The diff is limited to Task 3 source, wrapper, tests, documentation, and the
intentional packed-M verification artifact. The coordinator keeps generic
forwarding and parser-owned mode validation; all actual wrapper dry-runs remain
dry and preserve topology/checkpoint/data/seed defaults. No root-owned plan or
untracked policy file was staged or changed.

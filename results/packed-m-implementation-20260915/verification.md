# Packed M Task 3 verification

## Executed CPU evidence

All commands used the repository virtual environment with
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`PYTHONPATH="$PWD/transformers/src:$PWD/src"`.

- RED coordinator allowlist: `pytest tests/multinode/test_native_ddp_launch.py
  -xq` exited nonzero with **1 failed, 4 passed**. The expected cause was
  `--stage2_m_execution is not an approved Stage 2 argument`.
- RED wrapper/coordinator integration: `pytest
  tests/stage2/test_objective_examples.py::test_each_example_forwards_packed_m_through_the_actual_coordinator_dry_run
  -q` exited nonzero with **3 failed**. Each real coordinator dry-run resolved
  the missing wrapper flag as `full_replay`.
- GREEN coordinator suite: `pytest tests/multinode/test_native_ddp_launch.py
  -xq` passed **31 tests**. This includes parser-default/full-replay coverage,
  value-bearing packed forwarding, and a real coordinator dry-run.
- GREEN wrapper suite: `pytest tests/stage2/test_objective_examples.py -q`
  passed **29 tests**. CE, entropy, and combined each reach the actual
  coordinator dry-run with `packed_suffix`.
- GREEN native CPU DDP witness: `pytest
  tests/stage2/test_objective_distributed.py -q` passed **2 tests**. Its real
  two-process Trainer keeps unequal/zero-block ranks, accumulation windows
  `[2,2,1]`, global branches `[3,0,1]`, and the independent full-replay
  optimizer oracle. Full replay logs model/decoder M counts `[3,0,1]` and
  `[15,0,5]`; packed logs `[0,0,0]` and `[2,0,1]`. The packed direct-decoder
  pre-hook independently observed `[2,0,1]` cached M calls.
- GREEN covering task suite: `pytest tests/multinode/test_native_ddp_launch.py
  tests/stage2/test_objective_distributed.py tests/stage2/test_objective_examples.py
  -q` passed **63 tests in 52.49 seconds**. `bash -n
  run_stage2_objective_examples.sh run_stage2_training.sh
  run_multinode_stage2_training.sh` and `git diff --check` both exited 0.

Complete command output is retained in
`logs/packed-m-implementation-20260915/`.

## Scope and remaining gaps

This is CPU test evidence only. No GPU run, GPU memory/time measurement,
ZeRO test, two-host execution, remote synchronization, or training launch was
performed. Normal latent recurrence was not changed. `m_query_rows` is the
conceptual selected contiguous suffix total in full replay and the actual packed
decoder query total in packed mode; neither counter establishes a memory or
throughput result.

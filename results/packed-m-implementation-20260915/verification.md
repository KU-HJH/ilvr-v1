# Packed M implementation verification

## Final project CPU gate — 2026-09-15

Verified code commit: `d1d2cc1`. The root ran the project CPU suite once after
all implementation and review fixes:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
./venv/ilvr-venv/bin/pytest -q tests -m 'not gpu'
```

Exit **0**: **2698 passed, 7 skipped, 3 deselected, 10 warnings in 456.20s**.
Complete output: `logs/packed-m-implementation-20260915/final-cpu.log`.
The ten warnings are the existing PEFT missing-config warning from
`test_eval_integration.py` (8) and `test_inference.py` (2). Skipped/deselected
tests are not counted as passes. The vendored upstream Transformers suite was
not run. This one run supplies the overlapping verification/branch-finishing
test-pass evidence; it was not repeated.

`git diff --check 516cf49 HEAD`, shell syntax checks of the three existing
Stage 2 scripts, and `git diff --exit-code 516cf49 HEAD -- transformers`
produced no errors. The Transformers fork/normal latent engine is unchanged.

## Implementation and review evidence

- Approved commitment v2 was implemented through writing-plans, isolated-worktree
  SDD/TDD, task reviews, one final feature review and one scoped final fix review.
  The plan cold read clarified token mapping, RNG chronology and DDP aggregation.
- Core: real two-layer Qwen and tiny image/nontrivial mRoPE; eager/SDPA,
  FP32/BF16 autocast and native checkpointing; independent full/packed selected
  M vocabulary logits and scores, causal leakage/negative controls, dropout/RNG
  preservation and exception cleanup.
- Runtime: full/packed parameter and retained-latent gradient equality/None
  structure, finite gradients, independently nonzero post-only latent gradients,
  unchanged F outputs/parameters/RNG, skip paths, literal call counters and old
  missing-mode resume compatibility. Existing fixed-reference regression tests
  are included in the project gate.
- Final review found and resolved unsupported stochastic-module guard gaps,
  missing durable selected-M-logit evidence, late-bound dropout probability
  validation, and an overly weak negative-control predicate. Final core/runtime
  covering command passed **58 tests in 4.03s** before the project gate.
- No unresolved or parked review findings remain. GPU numeric/dropout fidelity,
  large-checkpoint/image-data training, memory/time and two-host execution remain
  unverified. These results establish no task-performance improvement.

## Controller decisions

1. Use the established project CPU command `pytest -q tests -m 'not gpu'`, not
   bare pytest: the latter also collects vendored upstream tests. Cost if this
   scope is insufficient: upstream/GPU validation remains missing, not passed.
2. Identify final F by its unique full input length rather than gradient mode,
   because inherited Trainer evaluation can call F under `no_grad`. Retain
   repeated-call rejection and train/eval tests. Cost if the premise changes:
   final F could be misidentified or evaluation rejected; no normal F execution
   or M intervention semantics were changed.

## Task 3 delivery evidence

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

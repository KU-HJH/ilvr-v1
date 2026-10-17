# Stage1-v5 throughput bounded task brief

Authority: design commitment v1 in `docs/superpowers/2026-10-15-stage1-v5-throughput-workflow.md`, approved by user's `진행해`. User selected subagent implementation. Intake revision: `148b0e8ead14456d02cc595f0cc32a0752ee22e8`. This is a concrete bounded task brief, not an architectural plan.

## Global Constraints

Preserve selected Text-off keys, prepared tensors, objective, gradients, helper ownership, RNG/dropout/checkpoint behavior, sampler/GAS/resume settings and workers=0. No full-model training or benchmark launch. Use `/root/.codex/worktrees/6b9d/ILVR` and its `./venv/ilvr-venv/bin/python`, with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=transformers/src:src:.`. Logs under `logs/stage1-v5-throughput-20261015/`, results under `results/stage1-v5-throughput-20261015/`, datasets under `data/`. No model state writes. Keep unrelated untracked files untouched. No downloads or environment changes.

Implementers follow TDD and read the TDD skill plus writing-good-tests.md. No child agents, no Paperthin hooks. Commit only task deliverables; root owns workflow/control artifacts. Use focused causal RED/GREEN and integration tests; controller runs the full project suite once after both task gates and final review, per user's no-repeated-broad-suite policy. This final green suite is required for completion. Benchmarks must use frozen unchanged original code as an independent oracle and identical inputs/conditions; report medians, fixture size, repetitions, exact equality, memory and limitations. No flaky wall-clock assertion in pytest. Prefer small reproducible local measurement scripts over infrastructure. All reports include exact commands, output/log paths, RED failure cause, commits, files and self-review. Measurements establish local savings only.

## Task 1: Attention observation and Text-off selection

Read Global Constraints above (also included by controller in extracted brief). Implement only the attention/selection part of design v1.

Scope: `transformers/src/transformers/models/qwen2_5_vl/stage1_v5_trace.py`, `src/stage1/v5_selection.py`, and compact candidate handoff in `src/stage1/v5_trainer.py` if needed. Focused tests in `tests/stage1/v5/`; a small reproducible benchmark may live in `scripts/` with results/logs in designated directories.

Hoist repeated per-sample FP32 K conversion and invariant coordinates out of per-query observation. Keep K lifetime bounded to one sample/layer. Preserve existing per-query GEMM shapes, scaling/softmax/head-mean/layer-addition order, mask semantics and detach behavior. Do not batch/change GEMMs or FP32 reductions. Keep hooked KV graph behavior. Bulk transfer candidate weights, GPU-computed masses and rho values; preserve Python sorting, ties, cumulative arithmetic, thresholds, caps and low-mass branches. Avoid expanding compact candidates to full sequence rows in trainer where safely possible. Keep public behavior and existing direct selector callers compatible.

Before changing code, add causal tests for redundant operation/synchronization removal and show them fail on original. Use independent frozen-original oracle for exact selected key equality across empty candidates, multiple samples/queries, ties, zero cap, low mass, near-threshold values and disjoint candidates. Verify observed attention against original/native causal attention. Run existing tiny actual Qwen trace/replay/debug tests that cover logits/loss/gradients, dropout and checkpoint recomputation. Preserve exact unchanged original oracle rather than deriving expected output from optimized helpers.

Measure attention/selection local time before/after with identical bounded CPU or idle-CUDA operator inputs, peak memory and operation/host-transfer counts. These are operator tests, not a training run. Preserve raw measurements and reproducer. Report limits: no production BF16/four-rank equivalence or utilization fix established. Reference useful tests: `test_selection.py`, `test_execution.py`, `test_replay.py`, `test_debug.py`. Report commit(s) and selected test evidence; full project suite is controller's final gate.

## Task 2: Single-pass v5 input preparation

Read Global Constraints above (also included by controller in extracted brief). Implement only the CPU collation part of design v1 after Task 1 is reviewed.

Scope: `src/stage1/data.py`, relevant tests under `tests/stage1/`, and a small reproducible local benchmark if needed. Keep standard Stage1 behavior and processor calls unchanged.

For v5 singleton sample processing, call the existing full-conversation processor once, materializing each image once. Reuse exact full tokenization and split image grids / raw pixel patch rows into original/helper ownership, with raw rows equal to grid_thw.prod (no merge-size divisor). Preserve order and all tensor outputs, labels/masks, source/occurrence metadata, helper_to_block, validation errors. No-helper output must preserve old `(0,0)` pixel tensor shape and dtype rather than slicing an empty `(0,patch_width)` tensor. Preserve validation coverage for invalid grid/pixel cardinality and sample ownership. No workers/prefetch/cache/profile changes.

TDD: causal processor/materialization count test fails against original first. Compare complete collator outputs with unchanged original three-call oracle, including real locally available Qwen processor (offline only), multiple originals/helpers, variable image sizes, no-helper, multi-example batching and metadata. Standard path must still use original behavior. Exercise malformed inputs and error validation. Use actual pixel and grid outputs, not only mocked shapes. Keep frozen original comparison independent from new splitting code.

Measure before/after complete CPU collation wall time and memory under identical inputs and repetitions; record exact tensors/metadata equality and processor/image-open counts. Run `tests/stage1/test_data.py`, `tests/stage1/v5/test_metadata.py`, new equivalence tests and smallest relevant integration. Do not rerun a broad full suite; controller owns final full project verification after review. Report commits, RED/GREEN evidence, measurements, files and concerns.

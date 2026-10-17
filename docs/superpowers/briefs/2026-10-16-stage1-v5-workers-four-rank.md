# Stage1 worker forwarding bounded task brief

Authority: design commitment v1 in
`docs/superpowers/2026-10-16-stage1-v5-workers-four-rank-workflow.md`, approved
by the user's `ㅇㅋ`. This is a bounded task brief, not an architectural plan.

## Global Constraints

Use `/root/.codex/worktrees/6b9d/ILVR` and its `./venv/ilvr-venv/bin/python`.
Preserve default workers0 and all existing objective, sampler, prefetch,
persistence, pin-memory, batch, RNG/dropout and resume semantics. Honor only
explicit nonnegative worker overrides. No profiler subsystem, alternative
launcher, automatic worker default adoption, training, remote mutation or package
installation in the implementation task. Training remains controller-owned via
the existing two-node/four-rank coordinator. Use real code behavior in tests;
source-text assertions and mock-only forwarding tests are insufficient.

## Task 1: Honor the existing Stage1 dataloader worker argument

Implement the smallest correction so `build_stage1_training_arguments` forwards
the existing validated `dataloader_num_workers` argument into SFTConfig instead
of hardcoding0. Retain default0 for omitted args/legacy callers, support
0,2,4,8,16,32,64, and reject negatives before GPU/model loading. Add only narrowly
necessary Stage1 argument validation; preserve existing Stage2 behavior.

Files: `src/stage1/composition.py`, narrowly necessary `src/utils_deepseed.py`
validation, and focused tests in `tests/stage1/` or `tests/stage1/v5/`. Existing
launcher/coordinator forwarding is already present and is not to be redesigned.
Confirm effective prefetch/persistence/pin-memory defaults remain unchanged.
Existing v5 execution/resume contract already records worker count; retain it.

Follow TDD: first run a causal regression showing a nonzero requested count is
ignored; then implement. Keep baseline default0 coverage and negative-input
coverage. Add the smallest meaningful CPU integration witness that uses native
Trainer/Accelerate prepared-loader behavior to establish actual worker execution
and identical consumed sample order/tensors versus0 for a positive count. Check
RNG behavior relevant to the training process; do not claim full GPU or four-rank
equivalence from CPU tests. Reuse existing test fixtures and actual call paths.
If a concrete correctness concern expands scope, report before changing design.

Run one focused covering set (composition/argument tests plus the loader witness,
and existing launcher tests only if necessary). Root will run the full project
suite once after independent reviews as the final verification gate; do not run a
duplicate broad suite. Logs go under `logs/stage1-v5-workers4r-20261016/`; generated
results under `results/stage1-v5-workers4r-20261016/`. Do not modify unrelated
untracked caches or the cancelled retrieval workflow.

Commit only task code/tests. Self-review and write a detailed report with RED/GREEN
commands, exit codes, output, exact runtime evidence, files, limitations and
concerns to the controller-provided report path. Do not spawn agents or invoke
Paperthin skills. Return the short SDD status/commit/test-summary/report contract.

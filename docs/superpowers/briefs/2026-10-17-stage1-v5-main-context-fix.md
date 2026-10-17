# Merge compatibility fix: one stochastic owner per Stage1 step

## Existing authorization and bound scope

User selected finishing Option1: merge Stage1-v5 into main. The approved feature
commitment keeps the training lifecycle, dropout rules, gradients/loss reductions
and ordinary parent dispatch unchanged. The merged main-only fix must continue
to own ordinary Stage1 dropout through backward. This task restores those two
already-specified paths after their automatic integration; it introduces no
algorithm, public option, loss or ownership architecture. No material design
revision or renewed formula approval is needed. Retain selected SDD method,
TDD and independent spec/quality/final integration review.

Base merge:2f6b5b7ae00c607b59797009c0956505e3ff6ece, on main in existing isolated
host worktree6b9d. Canonical checkout/user files remain untouched. No new
worktree, dependencies, training, GPU or real-data probe.

## Observed root cause and single fix hypothesis

Independent `/root/stage1_v5_main_merge_review` found blocking C1. Focused
actual CPU tests:7failed/5passed in11.30s. Main ordinary5 regressions pass;
all7 V5 parent/history cases fail. V5Execution enters first, new parent
training_step opens the legacy owner second; SampleStableStage1Dropout chooses
V5 dispatch and leaves legacy counters empty. Actual backward completes, then
legacy exit demands one original invocation and raises. A real14-site/p=.05
checkpoint probe observed44 finite nonzero gradients, complete F/packed
original/recompute events, empty legacy counters, and cleaned FAILED V5 state.
Full evidence: results/stage1-v5-main-merge-20261017/integration-review.md.

Single hypothesis: let the existing active V5 execution remain the sole owner;
only steps outside V5 enter the ordinary Stage1 wrapper. Mirror the existing
SampleStableStage1Dropout dispatch priority, keep native parent training_step
and all existing validations. Do not suppress the legacy validation failure or
invent fake invocation counts.

## Concrete task F1

Owned files only:

- src/stage1/trainer.py: add the narrow active-V5 dispatch guard to the new
  ordinary training_step. Use the existing current_v5_execution getter from
  stage1.v5_execution; a local import avoids module-cycle changes. With an
  already active V5 owner, call the same super().training_step directly. Without
  it, preserve the entire new main implementation byte-for-byte where practical.
- tests/stage1/v5/test_trainer.py: add one causal real-parent regression making
  exclusive ownership explicit at actual backward, with nonzero DoRA dropout
  and checkpoint replay. Reuse existing tiny fixture/native backward and check
  finite student gradients plus context cleanup. Do not mock away real dispatch.

TDD steps:

1. Read the actual failed review/log/trace; add the focused regression FIRST.
   Run it RED before changing production. It must distinguish an active legacy
   owner inside a V5-owned backward, not merely assert a private guard exists.
2. Make the single active-owner guard fix. Do not bypass CustomTrainerStage1V5's
   existing parent dispatch, change V5Execution/equivalence, relax validation,
   alter dropout RNG/math or touch Teacher/query/loss/reference/sampler/workers.
3. Run the new regression GREEN and the focused main5 plus existing7 V5 cases
   from the failed review. Their pass must include actual backward/checkpoint
   replay, both query modes, shared targets and unchanged denominators. Keep
   prior failure logs. No repeated broad suite or new execution framework.
4. Self-review, diff check, commit only these two files. Report actual commit,
   RED/GREEN commands, complete outputs/exit codes and remaining uncertainty.
   Independent spec, quality, then final integration re-review remain gates.

Use repository venv, PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src. All logs under
logs/stage1-v5-main-merge-20261017; any scalar checkpoint fixtures beneath the
canonical checkpoint tree. Root owns this brief/workflow; implementer must not
edit controls or invoke advisory hooks/spawn agents. If preserving both paths
requires a material architectural change, stop and report rather than expand.

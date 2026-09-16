# Baseline diagnosis: full-suite suspected failures

## Scope and method

This was a read-only reproduction.  I did not alter the active worktree's
source, index, or HEAD (`3112e9677771269ddd38a1a1e2337840e66d3458`).

I archived the prefeature source commit
`ae5a1cdd030882544b37b022835e343302db1ddf` into the disposable directory
`/root/workspace/ILVR/.baseline-ae5a1cd.3fG2Ib`, then linked only the existing
repository `venv`, `data`, and `checkpoints` directories.  The first `/tmp`
archive could not collect `test_capacity.py` because the historical code
requires its root below `/root/workspace/ILVR`; that exact environmental guard
failure is retained in the log.  The second archive meets the guard and ran the
requested command unchanged.

```bash
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q \
  tests/multinode/test_capacity.py::test_stage1_coordinator_uses_training_evidence_step_without_max_steps \
  tests/stage2/test_comt_readiness.py::test_stage2_launcher_dry_run_exposes_reproduction_defaults \
  tests/step_preference/test_dora_ablation_reporting.py::test_real_launcher_phase_budget_is_rejected_at_freeze_and_reopen \
  tests/step_preference/test_evidence_gain.py::test_cuda_flex_checkpoint_matches_dense_loss_and_gradients
```

Full setup and unabridged output: `logs/evidence-replacement-20260916/baseline-failures.log`.

## Runtime result

The archived `ae5a1cd` run finished `3 failed, 2 passed in 42.29s`.

| Suspected current failure | Archived baseline result | Direct failure witness |
| --- | --- | --- |
| `test_stage1_coordinator_uses_training_evidence_step_without_max_steps` | Failed: pre-existing | `ValueError: Stage 1 training evidence configuration is inconsistent` |
| `test_stage2_launcher_dry_run_exposes_reproduction_defaults` | Failed: pre-existing | Expected Stage 2 launcher argv differs at index 2 (`--epochs` appears before `--load_model_path`) and has two extra arguments. |
| `test_real_launcher_phase_budget_is_rejected_at_freeze_and_reopen` | Failed: pre-existing | Expected a `step|phase` `ValueError`; the archived launcher instead says `launcher requires exactly one --epochs`. |
| `test_cuda_flex_checkpoint_matches_dense_loss_and_gradients[False]` | Passed on archived baseline | No baseline failure observed. |
| `test_cuda_flex_checkpoint_matches_dense_loss_and_gradients[True]` | Passed on archived baseline | No baseline failure observed. |

This proves the first three failures predate the feature snapshot.  It does
not prove a cause for the two CUDA failures in the current full suite: the
same exact parametrized tests passed at `ae5a1cd`, so they are not baseline
failures under this environment and require separate current-revision tracing.

## Current CUDA state probe

At current revision `4b27265907a595278855b5382a78d263171ecc84`, the same two
CUDA parametrizations passed when launched normally in isolation: `2 passed in
11.42s`.  A minimal fresh-process probe then set the CUDA matmul and cuDNN TF32
flags before invoking only that test:

| Forced process state | Result |
| --- | --- |
| `torch.backends.cuda.matmul.allow_tf32=False`; `torch.backends.cudnn.allow_tf32=False` | `2 passed in 10.50s` |
| both flags `True` | `2 failed in 10.19s`, each at the existing `model.embed_tokens.weight` gradient assertion |

The matching archived-baseline probe with both flags `True` also failed both
parametrizations (`2 failed in 10.54s`) at the same embed-token gradient
assertion.  Thus the forced-state failure mode predates the feature snapshot.
It does not directly observe the original full-suite state and does not rule
out a separate current-revision regression under other states.

Static tracing does identify a concrete earlier suite setter consistent with
the full-suite order: `tests/stage2/test_eval_integration.py` runs
`test_real_transformers_flash_attention_error_retries_eager` for
`evaluate_deepseed`, whose successful `module.load_model("model", "cache")`
reaches `src/evaluate_deepseed.py:112`.  That function sets
`torch.backends.cuda.matmul.allow_tf32 = True` and has no restoration.  The
test helper only monkeypatches `torch.cuda.is_available`; this setter itself is
unconditional inside a `try`.  The Stage 2 evaluation test file occurs before
the later `tests/step_preference/test_evidence_gain.py` in the full-suite
collection.  This is a static order-based explanation; the final-suite log did
not record backend state immediately before the failing test, so that original
state remains inferred rather than directly observed.  No source change is
proposed.

The isolated-current and controlled-probe commands with full output are in
`logs/evidence-replacement-20260916/current-suspected-failures.log`.

## Pattern comparison

The archived failures enter separate launcher/capacity assertion paths before
any evidence-replacement scorer is involved.  The historical test files named
in the command have no diff from `ae5a1cd` to the current revision.  The later
repository changes include `run_stage2_training.sh`, `src/utils_deepseed.py`,
and `src/stage2/trainer.py`; this comparison is contextual only, not a causal
claim.  No fix is proposed or applied.

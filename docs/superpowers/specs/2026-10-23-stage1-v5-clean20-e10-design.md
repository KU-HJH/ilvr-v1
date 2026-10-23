# Stage1-v5 clean-entropy top20 training: frozen proposal v1

Status: bounded design proposal v1; not approved for product implementation.
Intake: user requests DIL-BCE-ORIG plus entropy and mass KL for ten epochs;
user explicitly answered `20%` for clean-entropy token selection.
Source: existing `stage1_v5` worktree, HEAD
`68b46d906243b5338350990f42d429d8063a7deb`.

## Problem and proposed scope

The existing selector can rank tokens by clean full-vocabulary entropy, but
fresh token-mean training rejects every selector except `all`. Extend only the
`clean` path needed by this request, retaining the existing global token-mean
DIL objective rather than merely removing the guard and dividing by all tokens.
Do not add another orchestration system, model pass or future selector support.

## Exact behavior

- Per source sample, rank the existing eligible immediate-text targets across
  its original blocks by detached clean entropy at T=1. Keep the highest20%,
  with the existing stable predictor-position tie order. A nonempty sample with
  N eligible tokens contributes ceil(N*20/100); an empty sample contributes0.
- DIL chosen/reference, V-off top-k entropy and mass KL use that same selected
  mask, as in the existing objective. CE/alignment retain their full existing
  targets; Text-off KL retains its own existing selection and reduction.
- Prefetch the exact selected-token count from metadata using the same rational
  ceiling rule as the existing selector. Sum counts across actual accumulation
  microbatches and all four ranks. Normalize DIL by that global selected count,
  including the actual partial tail. No forward is needed to know the count.
- Preserve entropy/mass's original-block denominator, including empty blocks;
  preserve native CE/alignment/Text-KL reductions, backward and optimizer paths.
  Preserve `all` behavior, strict saved source identity and old-checkpoint
  semantics. Give clean-selected normalization an unambiguous saved identity;
  never relabel historical block-mean or all-token contracts as this objective.
- Keep unsupported `voff`, `gap` and `intersection` training selectors rejected.

## Fresh training execution

Use original Qwen2.5-VL-3B initialization, frozen EMA1 reference, DoRA
r32/alpha64/dropout0.05/latent8, decoder SDPA, vision FA2, existing non-reentrant
checkpointing and ZeRO1. Keep legacy query mode, max sequence32768, workers0,
CE0.1, alignment0.1, DIL1, beta1/margin0, entropy0.1, mass0.05 and Text-KL0.01.
Top-k is independent k50 with no GT union. Requested ten epochs explicitly
override the standing one-epoch exposure. Keep global batch16, micro1/GAS4,
LR5e-5, seed42, warmup10 and existing AdamW/linear schedule/decay/clip.

Reuse the existing Stage1 coordinator, remote rank0 at korea@172.16.11.41,
local rank1, two GPUs/node, llm_kd and bond0.11. Preserve the current original
worktree's unrelated modifications. Provision clean reviewed execution copies
as needed; the minimal existing current-network address/interface edits must
be reviewed together with the execution profile before committing an execution
revision. Do not overwrite historical checkouts, checkpoints or active sessions.
Do not claim Stage1 IB based on Stage2-only IB evidence.

Derive the update budget from retained input and the actual loader/Trainer tail
policy, then verify the resolved ten-epoch schedule. Existing records suggest
3450 samples,216 updates/epoch,2160 total; these are not used as constants in
place of fresh audit. Store weights below each run's checkpoints symlink,
results under results and logs under logs. Save complete epoch checkpoints and
the final artifact through the existing path. No evaluation campaign is added.

## Gates and acceptance

This is one bounded support extension followed by execution of the existing
training protocol. Retain the user-selected Subagent-driven method, TDD,
independent spec/quality and final review, and fresh required verification.
Prior task's full-suite exception was specific to that task and is not reused.
Before launch, require a causal count/mask agreement witness, exact selected
global-token gradient normalization across unequal/empty/tail cases, unchanged
all behavior, saved-contract roundtrip/conflict checks, and real clean20 CLI
resolution. Reuse existing focused tests and coordinator checks.
Verify compatible code/environment, readable inputs, exact profile, available
resources and safe output paths on both nodes, dry-run, then launch detached
with a fresh run ID. Confirm all four ranks, finite loss and actual optimizer
progress before reporting started; do not claim ten epochs are completed at
launch. No additional launch-permission question is required after applicable
design approval: the user already requested this training.

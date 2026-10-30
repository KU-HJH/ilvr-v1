# Stage 1 EMA baseline — design commitment v1

Status: final design v1 approved by user `ㅇㅋ` after its linked presentation and implementation-through-analysis question. Native implementation remains the user-selected method. Approval evidence: results/stage1-ema-baseline-20261030/design-evidence-v1.json.

## Intent and approaches

Measure a fresh EMA-teacher standard Stage 1 baseline against the fixed-teacher baseline (206/400, 51.5%) using the same original/native KV-ON TEST400 evaluator. Selected approach: extend the existing matched-standard execution profile to support native EMA. Rejected alternative: use unmatched legacy standard execution, because it changes execution semantics beyond the teacher.

This is a bounded compatibility change, not a new teacher algorithm. Retain native parameter-space EMA for DoRA parameters and trained token rows; leave the frozen vision/base parameters and all V5 objectives untouched. Evaluate the final student checkpoint.

## Implementation boundary

Allow valid native EMA decay values consistently in matched fresh-run and resume validation. Keep tau=1 frozen behavior and historical checkpoints unchanged. Reject invalid decay values; resume must restore the exact saved tau and teacher tensors, never convert a frozen run into an EMA run. Reuse the existing successful-optimizer-step update hook and EMA save/restore implementation. Do not relax objective, topology, or other execution constraints.

## Experiment

- Fresh original Qwen2.5-VL-3B base, native DoRA r32/alpha64/dropout0.05, latent8.
- EMA tau=0.999: teacher = 0.999 * teacher + 0.001 * student after each successful optimizer update; no update on accumulation-only or skipped steps.
- Standard CE1 + alignment1 over the existing full supervised label set. No DIL, entropy, mass KL or Text-off KL objectives.
- 3,450 retained samples, 863 rank-local microbatches per epoch, 216 updates per epoch, 10 epochs / 2,160 updates; preserve final partial accumulation denominator.
- Two nodes/four GPU ranks, microbatch1/accumulation4/global16, LR5e-5/warmup10/seed42/BF16, existing AdamW/linear schedule, weight decay0.01, max norm1, ZeRO1, SDPA decoder/FlashAttention2 vision and existing checkpoint/dropout contexts.
- DataLoader workers4 per rank; training run-cache OFF. Historical fixed baseline used workers0: record this execution difference explicitly.
- Use the reviewed Stage1 IB coordinator path on llm_kd_rdma with both-node resolved preflight and actual four-rank/NET-IB evidence. Preserve existing runs and artifacts.
- Save resumable state every216 updates including EMA teacher, optimizer, scheduler and RNG. Keep the saved profile binding.
- After training, automatically prepare final student PEFT with original base and evaluate TEST400 using reviewed original/native generation, KV ON, greedy, max-new4096/max-sequence128000/rep1/no vision ablation/static cache OFF.
- Compare accuracy and paired sample outcomes against fixed baseline51.5%, previous V5 50.5%, and latest all-token/loss10x54.0%. Report single-seed limitations and do not attribute all-token and coefficient effects separately.

## Critique, evidence and validation

Frozen proposal: results/stage1-ema-baseline-20261030/design-proposal-v1.json.
Hate root: admitting tau<1 does not establish that matched DeepSpeed updates and checkpoint continuation actually support EMA.
First nail: unchanged existing tests for DoRA EMA parameter updates, matched DeepSpeed lifecycle and EMA checkpoint round-trip. Result: 5 passed in10.24s; logs/stage1-ema-baseline-premise-20261030.log. This supports reuse of existing components; it does not prove the new combined tau0.999 matched execution on four GPUs.
No design scope change arose from this critique. Macrothink is not triggered: a bounded compatibility change with user-selected approach and known call paths does not justify fresh multi-agent direction discovery.

After approval: native implementation, causal focused tests for fresh/resume validation and actual matched EMA updates, independent specification/quality review, fresh required completion verification once, and the smallest critical integration witness before the long run. Verify real tau0.999 teacher mutation, correct update count and preserved saved tau/teacher state; fail before full launch if these witnesses disagree. No optional per-step debug checks in production training. No new retry/recovery framework.

## Workflow evidence

Applicable skills executed so far: using-superpowers; readchk intake; brainstorming approach selection and bounded design; hate design critique; verification-before-completion for the five-test evidence. Implementation, review and completion gates remain pending. Source inspected: d0f872864f0a6761621b5cffc054fc34fc20aa5f. Evidence hashes are recorded in results/stage1-ema-baseline-20261030/intake.json and design-evidence-v1.json.

# Stage 2 Vision-margin experiment: pre-registration

Status: frozen before any arm is trained. Written 2026-09-26.
Seed: `seed_c0c2cf91e4a2` v1.0.2 (interview `interview_20260925_105946`).
Implementation commit: `d8fe9c1`; training deploys `af2a644` (this document, the launch wrapper, and the full-replay rollout fix) on `feat/stage2-entropy-vision`.

This experiment asks whether the Vision gain term engages its intended
mechanism. It makes no task-improvement claim. All verdicts are provisional
(single seed 42 per arm).

## Arms

All arms start from Stage 1 `checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431`
on base `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`.

| Arm | lambda_dyn | lambda_vis | m_V | Frozen reference |
|---|---|---|---|---|
| `visiononly-mv05` | 0 | 1.0 | 0.5 | none (not loaded) |
| `combined-mv0` | 1.0 | 1.0 | 0.0 | Stage 1 checkpoint-431 |
| `combined-mv05` | 1.0 | 1.0 | 0.5 | Stage 1 checkpoint-431 |

Common to every arm: `vision_token_reduction=score_mean`, `post_window=1`,
`pre_window=1`, `gamma_H=0`, `entropy_eps=1e-6`, `stage2_m_execution=packed_suffix` (Amendment 1; originally `full_replay`),
seed 42, LR 5e-5, warmup 10 updates, linear schedule, AdamW, weight decay 0.01,
max grad norm 1.0, decoder SDPA + vision FlashAttention2 and latent activation checkpointing (Amendment 2), 4 ranks × microbatch 1 × accumulation 4 = global batch 16,
212 updates (`--max_steps 212 --save_steps 212`), DoRA r32/alpha64/dropout 0.05,
latent size 8, supervised latent boundaries, BF16, max sequence 8192, TRAIN.jsonl
(3,379 retained samples), native DDP, non-reentrant gradient checkpointing,
decoder SDPA. Run ids: `stage2-<arm>-gb16-s212-<UTC timestamp>`.

Launch (dry-run by default; `--launch` only after live preflight):

```bash
scripts/stage2/launch_vision_margin_arm.sh <arm> --main-process-port <free port> [--launch]
```

Dry-run plans for all three arms were produced on `d8fe9c1`:
`logs/stage2-vision-gain-probe/dry-run/`.

## M execution mode (decided before training)

CUDA BF16 likelihood budget v1 (`tests/stage2/integration/check_packed_m_bf16.py`,
real TRAIN.jsonl data, local H100 NVL, train-mode dropout replay):
10 samples, 25 blocks, 50 tokens. Block max abs error **0.0514 nat** (limit 0.05;
one failing block, sample 3207 block 1), token max abs error 0.0841 nat (limit 0.10).
Result: **FAIL → `full_replay` for all three arms.** Record:
`results/stage2-vision-gain-probe/m_mode_budget_v1.json`.

## Amendment 1 (2026-09-26, user decision, before any arm completed an update)

The first launch `stage2-visiononly-mv05-gb16-s212-20260926-050608` (visiononly-mv05, `full_replay`) failed with
CUDA OOM on local rank 2 before completing any optimizer update: the
full-replay M forward re-runs the vision encoder, whose SDPA attention tried to
allocate 24.85 GiB while about 89 GiB was already in use (H100 NVL, 93 GiB).
Under the stop rule this run is recorded as `FAILED`; it is not retried with the
same settings.

Because `full_replay` re-encodes images for every M branch, it cannot fit this
execution path. The user chose to train all three arms with
`stage2_m_execution=packed_suffix`, which reuses the F vision embeddings and
passed the earlier two-node smoke. This deviates from the M-mode rule above:
budget v1 failed by 0.0014 nat on one of 25 blocks (block error 0.0514 vs 0.05;
token error 0.084 within 0.10). That numerical gap between training-time S_M and
full-replay S_M is a declared limitation. The probe is unchanged: S_M is still
computed with `full_replay` in eval mode for every checkpoint, so verdicts do not
depend on the training M mode. All other settings and thresholds are unchanged.

## Amendment 2 (2026-09-26, user decisions, before any arm completed training)

The packed-suffix relaunch `stage2-visiononly-mv05-gb16-s212-20260926-052315`
reached update 33/212 (~57 s/update) and then OOMed on remote rank 0 in the
**F** branch's vision encoder (SDPA attention allocating 43.67 GiB for a large
image). It is recorded `FAILED`. This path OOMs independently of the Vision
objective.

Two memory changes were ported, each with tests, per the user's decisions:

1. Split attention backend: decoder SDPA, vision encoder FlashAttention2
   (from `c70a620`), for the student and the frozen reference. The CE control
   also used vision FA2.
2. Selective latent activation checkpointing (from `b92bdb0`): every
   gradient-carrying latent-prefix decoder call is wrapped in non-reentrant
   checkpointing with preserved RNG. CPU tests show identical CE, Vision loss,
   M scores, parameter and latent gradients with and without it (DoRA, full,
   BF16). The CE control also used latent activation checkpointing.

Before relaunch, at the user's request, the heaviest arm configuration
(`combined-mv05`: reference + packed M) was run for one optimizer step on each
of the 7 worst retained samples on one H100 NVL (top 6 by token length and top
6 by largest image patch count; max 8,095 tokens, 30,888 patches). All 7
completed; peak GPU memory 64.7 GiB of 93 GiB (without latent checkpointing the
same probe OOMed). Report: `results/stage2-vision-gain-probe/worst_memory_samples.json`;
logs: `logs/stage2-vision-margin-oom-probe/`.

All arms therefore also set `VISION_ATTENTION_IMPLEMENTATION=flash_attention_2`
and `STAGE2_LATENT_ACTIVATION_CHECKPOINTING=true`. Objective settings, data,
thresholds and stop rules are unchanged.

## Probe (diagnostic only)

- Source: `data/TEST.jsonl` gold `sequence_plan` (sha256 `6a1dccf3…9618`).
- row_id = 0-based index of non-empty lines. Rows ordered by
  `sha256("42:{row_id}")`; rows without latent steps skipped; taken until the
  cumulative latent block count reaches 256 → 80 rows, 256 blocks.
  Builder: `scripts/stage2/build_probe_manifest.py`; manifest:
  `results/stage2-vision-gain-probe/probe_manifest.json`.
- Checkpoints probed: step0 (Stage 1 checkpoint-431), CE control
  (adapter: `checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212`; the canonical checkpoint-212 holds weights only on the remote node),
  and each arm's checkpoint-212.
- S_M is always computed with `full_replay` in eval mode (no dropout), for every
  checkpoint, regardless of the training M mode. post_window=1.
- Reported per checkpoint: mean S_F, mean S_M, G_mean, G_gt_0 over valid blocks.
- Probe results are used only for the thresholds below. Never for checkpoint
  selection, hyperparameter tuning, or early stopping.

## Thresholds (Δ = checkpoint probe − step0 probe)

| Criterion | Pass condition |
|---|---|
| (a) | ΔS_F ≥ +0.05 nats **and** ΔS_F(arm) ≥ ΔS_F(CE) − 0.02 |
| (b) | ΔS_F ≥ 0.4·ΔG **and** ΔS_M ≥ -0.10 nats |
| (c) | probe G_gt_0 ≥ 0.60 **and** ΔG_mean ≥ +0.05 nats |
| (d) | TEST400 accuracy ≥ 0.51 **and** no significant decrease vs CE-only by McNemar (p ≥ 0.05) |

(a), (b), (c) all pass → mechanism PASS. (d) pass → no regression. Any of
(a)–(c) failing → mechanism FAIL, recorded without adjusting thresholds.

McNemar: exact test, one-sided (decrease), α = 0.05, paired on the same 400
items against the CE-only per-sample output
`.worktrees/voff-reproduction-20260917/results/eval-stage2-ce-matched-test400-20260919-155759/merged.jsonl`
(accuracy 0.53, merged sha256 `76380b6e…`).

TEST400 settings copied from the CE baseline: greedy (temperature 0, top-p 1),
max new tokens 4096, max sequence 128000, `external_full_latent_block`, no vision
ablation. If TEST400 does not complete for an arm, its (d) is `undecided`.

## Stop rules

- OOM, non-finite loss, or wall-clock above the arm's time cap → arm `FAILED`.
- Time cap per arm = mean s/update over its first 10 updates × 212 × 3, recorded
  as `time_cap_seconds` in `results/<run-id>/run_config.json`.
- No automatic retry with changed batch, accumulation, sequence length, or M mode.

## Known confounds and limits (declared in advance)

- The CE-only control is reused, not retrained. It was trained and evaluated on
  voff-reproduction commit `f8f80be` (`1a63667` plus a sampler guard) with ZeRO-2, gradient checkpointing off,
  `reference_random` sampler and `concatenated-v1` assistant rendering. The arms
  use native DDP, gradient checkpointing on, the default sampler and this
  branch's rendering (vision FA2 and latent activation checkpointing now match,
  Amendment 2). CE comparisons therefore mix objective effects with
  execution-path, data-order and rendering differences.
- Without `reference_random` padding, 212 × 16 = 3,392 presentations exceed the
  3,379 retained samples, so the Trainer revisits 13 samples at the start of a
  second pass.
- Single seed per arm; TEST400 binomial SE ≈ 2.5 pp.
- No vision-only m_V=0 arm: the margin effect is measured only under the
  combined objective; entropy × margin interaction is not testable.
- Training batch-mean S_F/S_M/G logs are not used for verdicts (batch noise
  ~0.4 nats between steps in the smoke run).
- The full-replay rollout change (ablation passed only to prefix calls past the
  selected END) is included in the trained code as commit `af2a644`, per the
  user decision of 2026-09-26. The budget v1 check above ran with this change present.

## Deliverables

- `checkpoints/<run-id>/checkpoint-212`, `logs/<run-id>/`, `results/<run-id>/run_config.json` per arm
- `results/stage2-vision-gain-probe/probe_{step0,ce,visiononly-mv05,combined-mv0,combined-mv05}.json`
- `results/eval-stage2-<arm>-test400-*/merged.jsonl`, `results/stage2-vision-gain-probe/test400_mcnemar.json`
- `results/stage2-vision-gain-probe/decision_table.json`, `documents/stage2-vision-margin-results.md`

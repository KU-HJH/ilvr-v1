# Stage 2 Vision-margin experiment: pre-registration

Status: frozen before any arm is trained. Written 2026-09-26.
Seed: `seed_c0c2cf91e4a2` v1.0.2 (interview `interview_20260925_105946`).
Implementation commit: `d8fe9c1`; training deploys this pre-registration commit (which adds only this document and the launch wrapper) on `feat/stage2-entropy-vision`.

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
`pre_window=1`, `gamma_H=0`, `entropy_eps=1e-6`, `stage2_m_execution=full_replay`,
seed 42, LR 5e-5, warmup 10 updates, linear schedule, AdamW, weight decay 0.01,
max grad norm 1.0, 4 ranks × microbatch 1 × accumulation 4 = global batch 16,
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

## Probe (diagnostic only)

- Source: `data/TEST.jsonl` gold `sequence_plan` (sha256 `6a1dccf3…9618`).
- row_id = 0-based index of non-empty lines. Rows ordered by
  `sha256("42:{row_id}")`; rows without latent steps skipped; taken until the
  cumulative latent block count reaches 256 → 80 rows, 256 blocks.
  Builder: `scripts/stage2/build_probe_manifest.py`; manifest:
  `results/stage2-vision-gain-probe/probe_manifest.json`.
- Checkpoints probed: step0 (Stage 1 checkpoint-431), CE control
  (`checkpoints/stage2-ce-matched-gb16-s212-20260919-155759/checkpoint-212`),
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

- The CE-only control is reused, not retrained. It ran on commit `1a63667`
  (voff-reproduction worktree) with ZeRO-2, gradient checkpointing off, vision
  FlashAttention2, latent activation checkpointing, `reference_random` sampler,
  and `concatenated-v1` assistant rendering. This branch has none of those
  sampler/rendering/activation-checkpointing features, so CE comparisons mix
  objective effects with execution-path, data-order and rendering differences.
- Without `reference_random` padding, 212 × 16 = 3,392 presentations exceed the
  3,379 retained samples, so the Trainer revisits 13 samples at the start of a
  second pass.
- Single seed per arm; TEST400 binomial SE ≈ 2.5 pp.
- No vision-only m_V=0 arm: the margin effect is measured only under the
  combined objective; entropy × margin interaction is not testable.
- Training batch-mean S_F/S_M/G logs are not used for verdicts (batch noise
  ~0.4 nats between steps in the smoke run).
- Uncommitted worktree edits (including a `stage2_latent_rollout.py` change to
  full-replay prefix calls) are **not** part of `d8fe9c1` and therefore not part
  of the trained code unless separately committed before launch.

## Deliverables

- `checkpoints/<run-id>/checkpoint-212`, `logs/<run-id>/`, `results/<run-id>/run_config.json` per arm
- `results/stage2-vision-gain-probe/probe_{step0,ce,visiononly-mv05,combined-mv0,combined-mv05}.json`
- `results/eval-stage2-<arm>-test400-*/merged.jsonl`, `results/stage2-vision-gain-probe/test400_mcnemar.json`
- `results/stage2-vision-gain-probe/decision_table.json`, `documents/stage2-vision-margin-results.md`

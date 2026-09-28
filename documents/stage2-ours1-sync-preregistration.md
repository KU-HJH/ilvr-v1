# Stage 2 Vision gain synchronized with ours1 (temp2): pre-registration

Status: frozen before any arm trains. Written 2026-09-28.
User request: "타 repository 동기화해서 다시 학습해봐" — reproduce the ours1 (temp2) Vision-gain
training on this code path, with a CE-only arm under identical settings.
User decisions: synchronize settings and the Stage 1 starting point; arms = Vision + CE-only.
Implementation: `2470819` on `feat/stage2-entropy-vision`.

## Why

Both earlier Vision-gain experiments (Vision-margin, high-entropy tokens) fell below the reused
CE control on TEST400, and that control ran on a different execution path. The ours1
implementation (`/raid/workspace/temp2/ILVR/.worktrees/ours1`, `src/stage2_objective/`) uses the
same objective math but different coefficients, dropout, Stage 1 source, M execution and batch.

## Arms

| Arm | λ_vis | Everything else |
|---|---|---|
| `sync-vision` | 0.3 | identical |
| `sync-ce` | 0 (CE only) | identical |

## Settings synchronized with ours1

| Setting | ours1 | This experiment |
|---|---|---|
| λ_dyn / λ_vis / m_V | 0 / 0.3 / 0 (vision variant) | same |
| Vision gain | G_b = mean log p_F − sg(mean log p_M) over the first post-block token; softplus(m_V − G_b); global block mean | same (`score_mean`, post_window 1) |
| M | independent per block, body keys hidden from queries ≥ END, markers visible; one full forward per block, cut at the last scored predictor | same (`full_replay` + `--stage2_m_prefix_cut`) |
| Dropout | effective 0; adapter dropout → Identity | same (`--stage2_zero_dropout`) |
| Stage 1 source | `stage1-dora-frozen-b1-lr5e5-zero1-1epoch-exact32768-20260902/checkpoint-431` | same checkpoint, copied to both nodes (see below) |
| Global batch / updates | 8 / 423 (one epoch) | 8 (4 ranks × 1 × GA 2) / 423 |
| LR, schedule, warmup, WD, clip | 5e-5, linear, 10, 0.01, 1.0 | same |
| DoRA / precision / max length / seed | r32 α64 / BF16 / 8192 / 42 | same |
| Attention | decoder SDPA + vision FA2 | same |
| Latent activation checkpointing | on | on |

## Remaining differences (declared)

- **Distributed:** ours1 used ZeRO-2 on 2 GPUs; this experiment uses native DDP on 2 nodes × 2 GPUs
  (the mandatory coordinator; the objective path rejects DeepSpeed). With DoRA, ZeRO-2 only shards
  optimizer state; the optimizer math is the same AdamW.
- **Native gradient checkpointing is on** (ours1: off). Without it one step on the worst samples
  OOMed in the F decoder forward on a 93 GiB H100 NVL
  (`logs/stage2-ours1sync-oom-probe/attempt1-no-native-ckpt/`). Checkpointing recomputes the
  forward; with dropout zero it does not change the loss or gradients.
- **Data order:** same balanced sampler class, but 4 replicas instead of 2, so the sample order differs.
- **Stage 1 copy:** byte-identical copy except that `stage1_peft_metadata.json` and
  `adapter_config.json` point to this host's base-model path; the two base models are
  byte-identical (`logs/stage2-ours1sync-stage1-copy/`). Its schema-3 metadata is read as schema 2.
- Image root `data/images_comt` (ours1: `data`), same files.

## Memory check (before training)

`sync-vision` settings, one optimizer step per sample on the previous worst 7 TRAIN samples
(`data/experiments/stage2-vision-margin-oom-probe/worst.jsonl`), one H100 NVL: all 7 completed,
peak 49,187 MiB (48.0 GiB), 17.7 min (≈150 s/sample, M = 33 full forwards). The zero-dropout
policy replaced 288 sites (adapter p 0.05). Batch-mean G at step0 +0.0065 nats, G_gt_0 0.52.
Logs: `logs/stage2-ours1sync-oom-probe/vision/`. `sync-ce` runs no M and needs less memory.

## Evaluation

- TEST400 with the CE-matched evaluation code `f8f80be` and settings (greedy, max new tokens 4096,
  max sequence 128000, `external_full_latent_block`, no vision ablation) for both arms.
- Probe (diagnostic only): the existing 80-row TEST manifest, eval mode, full-replay M,
  first post-block token; checkpoints step0 (ours1 Stage 1), `sync-ce`, `sync-vision`.

## Thresholds (unchanged; the control is now `sync-ce`)

| Criterion | Pass condition |
|---|---|
| (a) | ΔS_F ≥ +0.05 nats **and** ΔS_F(vision) ≥ ΔS_F(sync-ce) − 0.02 |
| (b) | ΔS_F ≥ 0.4·ΔG **and** ΔS_M ≥ −0.10 nats |
| (c) | G_gt_0 ≥ 0.60 **and** ΔG_mean ≥ +0.05 nats |
| (d) | TEST400 accuracy ≥ 0.51 **and** exact one-sided McNemar p ≥ 0.05 vs `sync-ce` |

Δ = checkpoint − step0. The old CE control (0.53) is reported as a secondary comparison only.

## Stop rules

OOM, non-finite loss, or wall-clock above 3× the first-10-update pace → arm FAILED; no automatic
retry with changed settings. Single seed; verdicts provisional.

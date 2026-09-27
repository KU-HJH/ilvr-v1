# Stage 2 high-entropy token Vision gain: pre-registration

Status: frozen before any arm trains. Written 2026-09-27.
Seed: `seed_f29f8eb974c9` (direct interview; the Ouroboros MCP interview could not run as root).
Implementation commit: `bd10564` on `feat/stage2-entropy-vision`.

Request: "Vision gain을 전체 response 기준 20%의 high entropy token에서만 계산하도록해서
학습해보자 entropy는 제외하고". This experiment asks whether computing the Vision
gain only on each response's high-entropy tokens engages the mechanism. It makes
no task-improvement claim. All verdicts are provisional (single seed 42 per arm).

## Arms

Both arms start from Stage 1 `checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431`
on base `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`, and differ only in M masking.

| Arm | M hides, for a selected target after latent block k |
|---|---|
| `hient-independent` | block k only (blocks 1..k-1 stay visible) |
| `hient-cumulative` | blocks 1..k; every position after the first block is recomputed without any latent slots, so no earlier text keeps a view of them either |

Latent START/END markers stay visible in both.

## Objective

- λ_dyn = 0 (entropy-transition term excluded; no frozen reference loaded), λ_vis = 1.0, m_V = 0.5.
- For each selected target t: G_t = log p_F(t) − stopgrad(log p_M(t)), full-vocabulary log-probabilities.
- L_vis = Σ_t softplus(0.5 − G_t) / N_tokens, N_tokens = selected targets in the global batch
  (each token weighs equally, so longer responses weigh more).
- Loss = CE_F + λ_vis · L_vis.

## Token selection (fixed, step0)

- Model: Stage 1 checkpoint-431, eval mode, BF16, decoder SDPA, vision FA2, teacher-forced on the gold response.
- Candidates: supervised assistant text targets (labels ≠ −100, text vocabulary; latent slots and markers excluded).
- Score: full-vocabulary softmax entropy (nats, FP32) of the logits at t−1.
- Per sample, top ceil(0.2 · n_candidates) targets; ties go to the earlier position.
- Targets before the first latent block may be selected but have no latent to hide; they are dropped
  from the loss and counted (`excluded_pre_first_block_tokens`).
- Selections are looked up by sha256 of the sample's input_ids; a missing sample or a file whose
  sha256 differs from the configured value stops training.
- TRAIN selection: `results/stage2-highentropy-vision/train_selection.json`
  (sha256 `10618bbe2ed9e6b627f92df7ca6783cc20ca0946b25916e0cc12768c951d6699`): 3,379 samples,
  candidates per sample median 151 (max 457), selected median 31 (max 92), 100,915 selected targets.
  The per-sample 20% entropy boundary is low: p10 0.012, median 0.071, p90 0.73 nats.
- Probe selection: `results/stage2-highentropy-vision/probe_selection.json`
  (sha256 `7229a39c56e6e329f4d52265efb49404480c3672ba2c95f13d39ce280b25a908`): 80 rows, 2,402 selected targets.
- Built by `python -m stage2.entropy_selection` at `bd10564` (2 shards, merged); logs in
  `logs/stage2-highentropy-vision-selection/`.

## Training settings (identical to visiononly-mv05)

`vision_token_selection=step0_entropy`, `vision_token_reduction=token_loss_mean`,
`stage2_m_execution=packed_suffix`, seed 42, LR 5e-5, warmup 10 updates, linear schedule,
AdamW, weight decay 0.01, max grad norm 1.0, 4 ranks × microbatch 1 × accumulation 4 =
global batch 16, 212 updates, DoRA r32/alpha64/dropout 0.05, latent size 8, supervised latent
boundaries, BF16, max sequence 8192, TRAIN.jsonl (3,379 retained samples), native DDP,
non-reentrant gradient checkpointing, decoder SDPA + vision FA2, latent activation
checkpointing. Run ids: `stage2-<arm>-gb16-s212-<UTC timestamp>`.

Launch (dry-run by default):

```bash
scripts/stage2/launch_highentropy_vision_arm.sh <arm> --main-process-port <free port> [--launch]
```

## Memory check (before training)

One optimizer step per sample, single H100 NVL, microbatch 1, on 13 TRAIN rows: the previous
worst 7 (longest token length, largest image) plus the 6 with the most selection candidates
(`data/experiments/stage2-highentropy-vision-oom-probe/rows.json`). Both arms completed all 13
with peak GPU memory 48,547 MiB (47.4 GiB of 93 GiB). 733 selected targets entered the loss and
40 (5%) were dropped as pre-first-block. At step0 the batch-mean G was −0.001 (independent) and
−0.002 nats (cumulative), G_gt_0 0.47 / 0.46. Report:
`results/stage2-highentropy-vision/worst_memory_samples.json`; logs:
`logs/stage2-highentropy-vision-oom-probe/`. Dry-run plans for both arms:
`logs/stage2-highentropy-vision-dry-run/`.

## Probe (diagnostic only)

- Rows: the existing 80-row TEST manifest `results/stage2-vision-gain-probe/probe_manifest.json`.
- Tokens: the step0 probe selection above (same tokens for every checkpoint).
- Eval mode, BF16, M by full replay with the arm's hiding rule.
- Per checkpoint and scope: S_F = mean over tokens of log p_F, S_M likewise, G_mean = S_F − S_M,
  G_gt_0 = fraction of tokens with G_t > 0.
- Probed: step0 and the CE control under both scopes; each arm under its own scope.
- Never used for checkpoint selection, tuning, or early stopping.

## Thresholds (unchanged from the Vision-margin experiment; Δ = checkpoint − step0, same scope)

| Criterion | Pass condition |
|---|---|
| (a) | ΔS_F ≥ +0.05 nats **and** ΔS_F(arm) ≥ ΔS_F(CE) − 0.02 |
| (b) | ΔS_F ≥ 0.4·ΔG **and** ΔS_M ≥ −0.10 nats |
| (c) | G_gt_0 ≥ 0.60 **and** ΔG_mean ≥ +0.05 nats |
| (d) | TEST400 accuracy ≥ 0.51 **and** exact one-sided McNemar p ≥ 0.05 vs CE-only |

(a)–(c) all pass → mechanism PASS; (d) pass → no regression. Units are now tokens, not blocks.

TEST400: CE-matched evaluation code `f8f80be` and settings (greedy, max new tokens 4096,
max sequence 128000, `external_full_latent_block`, no vision ablation), paired against
`.worktrees/voff-reproduction-20260917/results/eval-stage2-ce-matched-test400-20260919-155759/merged.jsonl`
(accuracy 0.53).

## Stop rules

- OOM, non-finite loss, or wall-clock above the arm's time cap → arm `FAILED`.
- Time cap = mean s/update over the first 10 updates × 212 × 3.
- No automatic retry with changed batch, accumulation, sequence length, or M mode.

## Known confounds and limits

- The CE-only control is reused (user decision): trained and evaluated on `f8f80be` with ZeRO-2,
  gradient checkpointing off, `reference_random` sampler and `concatenated-v1` rendering. CE
  comparisons mix the objective with execution-path, data-order and rendering differences.
- Single seed; TEST400 binomial SE ≈ 2.5 pp.
- Training M uses packed_suffix; the probe uses full replay.
- Per-sample 20% means low-entropy responses still contribute near-deterministic tokens.
- 212 × 16 = 3,392 presentations > 3,379 retained samples (13 samples seen twice).

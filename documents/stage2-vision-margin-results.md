# Stage 2 Vision-margin experiment: results

Status: complete, provisional (single seed 42 per arm). Written 2026-09-27.
Pre-registration: `documents/stage2-vision-margin-preregistration.md` (with
Amendments 1 and 2). Seed `seed_c0c2cf91e4a2`. Trained code: `41b674e`.

**Verdict: the Vision-gain mechanism was not demonstrated in any arm.** No
task-performance claim is made.

## Runs

| Arm | λ_dyn | λ_vis | m_V | Run id | Updates | Wall time | Time cap | Outcome |
|---|---|---|---|---|---|---|---|---|
| visiononly-mv05 | 0 | 1.0 | 0.5 | `stage2-visiononly-mv05-gb16-s212-20260926-102932` | 212 | 4.5 h | 46,809 s | COMPLETED |
| combined-mv0 | 1.0 | 1.0 | 0.0 | `stage2-combined-mv0-gb16-s212-20260926-150310` | 212 | 5.4 h | 57,240 s | COMPLETED |
| combined-mv05 | 1.0 | 1.0 | 0.5 | `stage2-combined-mv05-gb16-s212-20260926-203536` | 212 | 5.6 h | 57,558 s | COMPLETED |

All: 2 nodes × 2 H100 NVL, native DDP, global batch 16, packed_suffix M,
decoder SDPA + vision FA2, latent activation checkpointing. Earlier FAILED
launches (stop rule, not retried with the same settings): full_replay OOM
(`…-20260926-050608`) and SDPA-vision OOM at update 33 (`…-20260926-052315`).

## Probe (eval mode, full-replay S_M, 80 TEST.jsonl rows, 256 blocks)

| Checkpoint | S_F | S_M | G_mean | G_gt_0 | ΔS_F | ΔS_M | ΔG |
|---|---|---|---|---|---|---|---|
| step0 (Stage 1 ckpt-431) | −1.1749 | −1.1710 | −0.0039 | 0.535 | – | – | – |
| CE control | −1.1454 | −1.1439 | −0.0015 | 0.598 | +0.0294 | +0.0270 | +0.0024 |
| visiononly-mv05 | −1.1527 | −1.1565 | +0.0039 | 0.613 | +0.0222 | +0.0144 | +0.0078 |
| combined-mv0 | −1.1795 | −1.1743 | −0.0052 | 0.375 | −0.0046 | −0.0033 | −0.0013 |
| combined-mv05 | −1.1716 | −1.1670 | −0.0046 | 0.551 | +0.0033 | +0.0040 | −0.0007 |

Hiding one latent block changes the first post-block token log-probability by
only ~0.004 nat at step0, and no arm moved G by more than 0.008 nat. For
comparison, CE-only training raised S_F by more (+0.029) than any Vision arm.

## TEST400 (CE-matched evaluation code f8f80be and settings)

| Model | Accuracy | CE-only correct / arm-only correct | Exact one-sided McNemar p |
|---|---|---|---|
| CE control | 0.530 | – | – |
| visiononly-mv05 | 0.4875 | 46 / 29 | 0.032 |
| combined-mv0 | 0.485 | 43 / 25 | 0.019 |
| combined-mv05 | 0.510 | 36 / 28 | 0.191 |

## Pre-registered verdicts

| Arm | (a) | (b) | (c) | (d) | Mechanism | No regression |
|---|---|---|---|---|---|---|
| visiononly-mv05 | FAIL (ΔS_F +0.022 < 0.05) | PASS | FAIL (ΔG +0.008 < 0.05) | FAIL (p 0.032) | **FAIL** | **FAIL** |
| combined-mv0 | FAIL | FAIL (ΔS_F < 0.4·ΔG) | FAIL (G_gt_0 0.375) | FAIL (acc 0.485, p 0.019) | **FAIL** | **FAIL** |
| combined-mv05 | FAIL | PASS | FAIL | PASS (acc 0.51, p 0.191) | **FAIL** | **PASS** |

`results/stage2-vision-gain-probe/decision_table.json`,
`results/stage2-vision-gain-probe/test400_mcnemar.json`.

## Limits (read before interpreting)

- **Confounded control.** The CE-only control was trained and evaluated on
  voff-reproduction `f8f80be` with ZeRO-2, gradient checkpointing off, the
  `reference_random` sampler and `concatenated-v1` rendering. The arms differ in
  all of these as well as in the objective. The two significant regressions
  (visiononly-mv05, combined-mv0) therefore cannot be attributed to the Vision
  objective; they may come from the execution path, data order, or rendering. A
  CE-only arm on this code path would be needed to separate them.
- **Single seed**, TEST400 binomial SE ≈ 2.5 pp. All verdicts are provisional.
- Training used packed_suffix M, which failed budget v1 by 0.0014 nat on one of 25
  blocks (Amendment 1). The probe used full_replay for every checkpoint.
- No vision-only m_V=0 arm: the margin effect is observable only under the
  combined objective (m_V 0 → 0.5 raised G_gt_0 0.375 → 0.551, but ΔG stayed
  below 0.001 nat in magnitude).
- The probe is diagnostic only and was not used for any selection or tuning.
- 212 × 16 = 3,392 presentations > 3,379 retained samples (13 samples seen twice).
- The venv's editable Transformers points at the canonical checkout, which was
  at `41b674e` during the arm evaluations (CE evaluation ran with `31f745e`'s copy
  on the local shards); the generation paths are equivalent per code comparison.

## Artifacts

- Adapters: `checkpoints/evaluation-staging/<run-id>/adapter-checkpoint-212` (sha256 files alongside);
  full Trainer checkpoints on the remote rank-0 node.
- Probe: `results/stage2-vision-gain-probe/probe_{step0,ce,visiononly-mv05,combined-mv0,combined-mv05}.json`,
  manifest `probe_manifest.json`.
- TEST400: `results/eval-stage2-{visiononly-mv05-test400-20260927-031644,combined-mv0-test400-20260927-041447,combined-mv05-test400-20260927-054704}/merged.jsonl`.
- Logs: `logs/stage2-vision-margin-arms-20260926/`, `logs/stage2-vision-gain-probe/`,
  `logs/stage2-vision-margin-oom-probe/`, `logs/stage2-vision-margin-deploy-20260926/`.

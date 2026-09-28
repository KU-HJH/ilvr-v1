# Stage 2 high-entropy token Vision gain: results

Status: complete, provisional (single seed 42 per arm). Written 2026-09-28.
Pre-registration: `documents/stage2-highentropy-vision-preregistration.md`. Seed `seed_f29f8eb974c9`.
Trained code: `dd8d81a` (implementation `bd10564`).

**Verdict: the Vision-gain mechanism was not demonstrated by either arm, and neither arm
passed the no-regression criterion.** No task-performance claim is made.

## Runs

| Arm | M hides | Run id | Updates | Wall time | Time cap | Outcome |
|---|---|---|---|---|---|---|
| hient-independent | the block right before each target | `stage2-hient-independent-gb16-s212-20260927-120446` | 212 | 4.6 h | 47,954 s | COMPLETED |
| hient-cumulative | every latent block | `stage2-hient-cumulative-gb16-s212-20260927-163924` | 212 | 4.6 h | 47,254 s | COMPLETED |

Both: λ_dyn 0, λ_vis 1.0, m_V 0.5, step0 top-20% entropy targets per response (TRAIN selection
sha256 `10618bbe…6699`), token-level mean of softplus(0.5 − G_t), 2 nodes × 2 H100 NVL, native DDP,
global batch 16, packed_suffix M, decoder SDPA + vision FA2, latent activation checkpointing.
Memory check before training: 13 worst samples, peak 47.4 GiB, no OOM.

## Probe (eval mode, full-replay M, 80 TEST rows, 2,152 step0-selected tokens)

250 of the 2,402 selected probe tokens (10%) fall before the first latent block and are excluded.

| Mask | Checkpoint | S_F | S_M | G_mean | G_gt_0 | ΔS_F | ΔS_M | ΔG |
|---|---|---|---|---|---|---|---|---|
| independent | step0 | −1.1015 | −1.1018 | +0.0003 | 0.506 | – | – | – |
| | CE control | −1.0856 | −1.0856 | +0.0001 | 0.518 | +0.0159 | – | – |
| | hient-independent | −1.0944 | −1.0937 | −0.0007 | 0.495 | +0.0071 | +0.0081 | −0.0009 |
| cumulative | step0 | −1.1015 | −1.1023 | +0.0008 | 0.500 | – | – | – |
| | CE control | −1.0856 | −1.0854 | −0.0002 | 0.511 | +0.0159 | – | – |
| | hient-cumulative | −1.0934 | −1.0923 | −0.0011 | 0.508 | +0.0081 | +0.0100 | −0.0019 |

Even on each response's highest-entropy tokens, hiding the latent blocks changes the target
log-probability by less than 0.001 nat on average at step0, and G > 0 for about half of the
tokens. Training on these tokens did not raise G (ΔG −0.001 / −0.002 nat) and raised S_F less
than CE-only training did.

## TEST400 (CE-matched evaluation code f8f80be and settings)

| Model | Accuracy | CE-only correct / arm-only correct | Exact one-sided McNemar p |
|---|---|---|---|
| CE control | 0.530 | – | – |
| hient-independent | 0.4975 | 46 / 33 | 0.088 |
| hient-cumulative | 0.495 | 49 / 35 | 0.078 |

Neither decrease is significant at α = 0.05, but both accuracies are below the 0.51 floor.

## Pre-registered verdicts

| Arm | (a) | (b) | (c) | (d) | Mechanism | No regression |
|---|---|---|---|---|---|---|
| hient-independent | FAIL (ΔS_F +0.007 < 0.05) | PASS | FAIL (ΔG −0.001, G_gt_0 0.495) | FAIL (acc 0.4975 < 0.51; p 0.088) | **FAIL** | **FAIL** |
| hient-cumulative | FAIL (ΔS_F +0.008 < 0.05) | PASS | FAIL (ΔG −0.002, G_gt_0 0.508) | FAIL (acc 0.495 < 0.51; p 0.078) | **FAIL** | **FAIL** |

`results/stage2-highentropy-vision/probe-{independent,cumulative}/decision_table.json`,
`…/test400_mcnemar.json`.

## Limits (read before interpreting)

- **Confounded control.** The CE-only control was trained and evaluated on `f8f80be` with ZeRO-2,
  gradient checkpointing off, the `reference_random` sampler and `concatenated-v1` rendering. The
  accuracy gap to CE therefore mixes the objective with execution-path, data-order and rendering
  differences. A CE-only arm on this code path is still missing.
- Single seed; TEST400 binomial SE ≈ 2.5 pp.
- Low-entropy selection: the per-sample 20% boundary has median 0.071 nats (p10 0.012), so many
  "high-entropy" targets are nearly deterministic.
- Training used packed_suffix M; the probe used full replay.
- The probe is diagnostic only and was not used for any selection or tuning.
- 212 × 16 = 3,392 presentations > 3,379 retained samples (13 samples seen twice).
- The venv's editable Transformers points at the canonical checkout, which was at `dd8d81a` during
  both arm evaluations (as in the Vision-margin experiment).

## Artifacts

- Selections: `results/stage2-highentropy-vision/{train,probe}_selection.json`; memory check
  `results/stage2-highentropy-vision/worst_memory_samples.json`.
- Adapters: `checkpoints/evaluation-staging/<run-id>/adapter-checkpoint-212` (sha256 files alongside);
  full Trainer checkpoints on the remote rank-0 node.
- Probes: `results/stage2-highentropy-vision/probe-{independent,cumulative}/probe_{step0,ce,<arm>}.json`.
- TEST400: `results/eval-stage2-hient-independent-test400-20260927-221337/merged.jsonl`,
  `results/eval-stage2-hient-cumulative-test400-20260927-230941/merged.jsonl`.
- Logs: `logs/stage2-highentropy-vision-{selection,oom-probe,dry-run,deploy-20260927,arms-20260927,post}/`.
- Deployment: both canonical repositories were restored byte-identically after evaluation; the named
  stashes `stage2-hient-20260927-local` and `stage2-hient-20260927-remote` are kept.

# Stage 2 Vision gain synchronized with ours1: results

Status: complete, provisional (single seed 42 per arm). Written 2026-09-29.
Pre-registration: `documents/stage2-ours1-sync-preregistration.md`. Trained code: `57d7c55`.

**Verdict: mechanism FAIL and no-regression FAIL for `sync-vision`** (thresholds unchanged, control
= `sync-ce` on the same code path). The same-path CE control reached 0.52 on TEST400, close to the
old CE control (0.53).

## Runs

| Arm | λ_vis | Run id | Updates | Wall time | Time cap | Outcome |
|---|---|---|---|---|---|---|
| sync-vision | 0.3 | `stage2-sync-vision-gb8-s423-20260928-052303` | 423 | 6.4 h | 59,769 s | COMPLETED |
| sync-ce | 0 | `stage2-sync-ce-gb8-s423-20260928-114847` | 423 | 4.0 h | 37,816 s | COMPLETED |

Both: ours1 Stage 1 checkpoint-431, zero dropout (Identity adapter dropout), λ_dyn 0, m_V 0,
first post-block token, `score_mean`, full-replay M cut at the last scored predictor, global
batch 8 (4 ranks × 1 × GA 2), LR 5e-5, native DDP, native + latent activation checkpointing,
decoder SDPA + vision FA2.

## Probe (eval mode, full-replay M, 80 TEST rows, 256 blocks)

| Checkpoint | S_F | S_M | G_mean | G_gt_0 | ΔS_F | ΔS_M | ΔG |
|---|---|---|---|---|---|---|---|
| step0 (ours1 Stage 1) | −1.1852 | −1.1758 | −0.0094 | 0.484 | – | – | – |
| sync-ce | −1.1502 | −1.1457 | −0.0045 | 0.609 | +0.0350 | +0.0301 | +0.0049 |
| sync-vision | −1.1516 | −1.1578 | +0.0062 | 0.668 | +0.0336 | +0.0180 | +0.0155 |

This arm moved G more than any earlier Vision arm (ΔG +0.016, G_gt_0 0.67), mostly by
lowering S_M relative to CE rather than raising S_F. ΔG is still a third of the +0.05 threshold,
and S_F did not rise more than under CE-only training.

## TEST400 (CE-matched evaluation code f8f80be and settings)

| Model | Accuracy | Paired vs sync-ce (CE-only / arm-only correct) | Exact one-sided McNemar p |
|---|---|---|---|
| sync-ce | 0.520 (208) | – | – |
| sync-vision | 0.5025 (201) | 30 / 23 | 0.205 |

Secondary, vs the old CE control (0.53, different execution path): sync-ce 40/36, p 0.366;
sync-vision 45/34, p 0.130.

## Pre-registered verdicts

| Arm | (a) | (b) | (c) | (d) | Mechanism | No regression |
|---|---|---|---|---|---|---|
| sync-vision | FAIL (ΔS_F +0.034 < 0.05; the CE-relative part holds, 0.034 ≥ 0.015) | PASS | FAIL (ΔG +0.016 < 0.05; G_gt_0 0.67 ≥ 0.60) | FAIL (acc 0.5025 < 0.51; p 0.205) | **FAIL** | **FAIL** |

`results/stage2-ours1sync/{decision_table,test400_mcnemar}.json`.

## What this says about the earlier gap

- The same-path CE control (0.52) is within noise of the old CE control (0.53, p 0.37), so the
  execution-path difference does not explain the earlier Vision arms' deficit by itself.
- With ours1 settings the Vision arm is 1.75 pp below its matched CE control, not significant
  (p 0.21), versus 4.25 pp (p 0.03) for the first Vision-margin arm. Coefficient, margin, dropout,
  Stage 1 source and M execution all changed together, so the improvement cannot be attributed to
  one of them.

## Limits

- Single seed; TEST400 SE ≈ 2.5 pp.
- Remaining differences from ours1: native DDP on 4 GPUs instead of ZeRO-2 on 2 (data order
  differs), native gradient checkpointing on (off OOMed on H100 NVL 93 GiB), rewritten base path in
  the copied Stage 1 metadata (byte-identical base model).
- The probe is diagnostic only. ours1 has no TEST400 record for its Vision models, so no direct
  comparison with ours1 accuracy is possible.
- The venv's editable Transformers pointed at the canonical checkout (`57d7c55`) during evaluation.

## Artifacts

- Adapters: `checkpoints/evaluation-staging/<run-id>/adapter-checkpoint-423` (sha256 alongside).
- Probes: `results/stage2-ours1sync/probe_{step0,ce,sync-vision}.json`.
- TEST400: `results/eval-stage2-sync-ce-test400-20260928-162742/merged.jsonl`,
  `results/eval-stage2-sync-vision-test400-20260928-171946/merged.jsonl`.
- Logs: `logs/stage2-ours1sync-{oom-probe,stage1-copy,dry-run,deploy-20260928,arms-20260928,post}/`.
- Deployment: both canonical repositories restored byte-identically; named stashes
  `stage2-ours1sync-20260928-local` and `stage2-ours1sync-20260928-remote` are kept.

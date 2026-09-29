# Vision gain coefficient/margin follow-up: pre-registration

Status: frozen before training. Written 2026-09-29.
User request: "coefficient를 0.3, margin을 0.0으로 해서 visiononly, heint-independent,
heint-cumulative를 측정해봐". User decisions: keep every other setting of the original arms;
evaluate TEST400 only with the paper-aligned profile (upstream ILVR `eval.py` semantics).

## Arms (only λ_vis and m_V change)

| New arm | Original arm | λ_vis | m_V | Everything else |
|---|---|---|---|---|
| `visiononly-l03m0` | `visiononly-mv05` (0.4875) | 0.3 | 0 | as original: first post-block token, `score_mean` |
| `hient-independent-l03m0` | `hient-independent` (0.4975) | 0.3 | 0 | as original: step0 top-20% selection sha `10618bbe…`, `token_loss_mean`, independent |
| `hient-cumulative-l03m0` | `hient-cumulative` (0.495) | 0.3 | 0 | as original, cumulative |

Common (unchanged): Stage 1 `stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431`,
λ_dyn 0, DoRA dropout 0.05 active, packed_suffix M, global batch 16 (4 × 1 × GA 4), 212 updates,
LR 5e-5, native DDP, native + latent activation checkpointing, decoder SDPA + vision FA2, seed 42.
Launch: `scripts/stage2/run_coef_followup_arms.sh` (dry-runs in `logs/stage2-coef-dry-run/`).
No new memory check: the objective path and samples are unchanged, and λ/m_V do not change memory.

## Evaluation

- TEST400 with the paper-aligned profile only (`EVAL_PROFILE=paper`: bounded, repetition penalty
  1.05, max sequence 32768, `f8f80be`), for the three arms and the old CE control
  (`stage2-ce-matched-gb16-s212-20260919-155759`, same Stage 1 source) as the matched reference.
- Probes as in the original experiments: block-level probe for `visiononly-l03m0`; token-level probe
  with the arm's scope for the hient arms; same step0/CE probe files as before.

## Comparisons (all reported; single seed, provisional)

1. Each new arm vs the CE control under the paper profile (thresholds (a)–(d); (d) uses the
   paper-profile CE result).
2. Each new arm vs its original arm: probe deltas only (the originals have no paper-profile TEST400).
Stop rules as before (OOM, non-finite loss, 3× first-10-update pace → FAILED; no retry with changes).

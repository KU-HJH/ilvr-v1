# Stage1-v3 Gain through AdamW — design proposal v1

Status: conversational direction approved; written specification awaiting final human approval. Product baseline is `154561dd84b1ecae4933c6e849d913eb87a07c3d`. This document proposes implementation and CPU checks; it authorizes no training or GPU experiment.

## Intent and scope

The user wants Gain to affect the same AdamW gradient, first/second moments, clipping, learning rate, and schedule as the ordinary objective. Matching the norm of a separate raw-gradient displacement does not satisfy that goal. Preserve the existing raw RV protection intent as a local first-order condition on this step's additional update.

The accepted direction is: combine gradients, compare AdamW parameter updates with and without this step's Gain from the same pre-step optimizer state, then project their difference against raw RV. Do not execute two real optimizer updates. Reuse existing F/R/RV execution and VJPs, including optional detached F token entropy weighting. No added model forward/backward is required by the update algorithm.

This is a new optimizer variant, not a numerically equivalent rewrite of the current post-Adam displacement. Existing checkpoints retain their exact formula and coefficients. The default legacy path remains available and unchanged; use an explicit new mode for the requested variant.

## Alternatives already considered

1. Cap raw Gain by the AdamW displacement norm: useful as a size guard, but Gain still bypasses moments and coordinatewise scaling. Does not meet the clarified goal.
2. Add a raw-projected Gain gradient to the optimizer: routes Gain through AdamW, but subsequent adaptive scaling/momentum need not preserve the raw RV dot condition.
3. Selected direction: combine gradients in AdamW, then constrain the resulting incremental parameter displacement. This adds a local-shard counterfactual calculation and makes optimizer-state semantics explicit.

## Objective and mathematical contract

Let `b` be the ordinary accumulated/reduced optimizer gradient, including configured CE, alignment, and scheduled Text-KL, after unscaling and before clipping. Its existing accumulation/remainder denominator remains unchanged.

Let `g` be the current Gain gradient: raw global block-mean R-RV, or its optional detached F token-entropy-weighted version. Let `h` be the global block-mean gradient of the **unweighted** raw RV score. Use the existing actual global block count. Do not apply the legacy raw-gradient projection to `g` in the new mode.

Define `C(x)` as the pinned DeepSpeed global-norm clipping operation, including its epsilon and configured max norm. The base and joint branches each use their own global clipping norm:

```
b_base  = C(b)
b_joint = C(b - lambda_gain * g)
```

For the same pre-step FP32 master parameters `theta`, Adam state `s=(m,v,t)`, parameter-group settings, and current pre-scheduler LR, define the native AdamW transition `A`:

```
(theta_base, s_base)   = A(theta, s, b_base)
(theta_joint, s_joint) = A(theta, s, b_joint)
u = theta_joint - theta_base
```

`theta_base` is a counterfactual calculation on temporary copies. Only the joint branch changes the real AdamW state. The effect of Gain on clipping and the second moment is part of `u`; it is not approximated by applying a fixed diagonal matrix to `g`. Both candidates include the same weight decay; the live update applies decay once.

Project over all owned trainable coordinates and all parameter groups, with global FP64 dot/norm reduction over unique ZeRO-owned fragments:

```
n = sum(h_i ** 2)
d = sum(h_i * u_i)
u_safe = 0                    if 0 < n < min_h_norm_sq
u_safe = u - (d / n) * h      if n >= min_h_norm_sq and d < 0
u_safe = u                    otherwise, including h == 0
theta_next = theta_base + u_safe
state_next = s_joint
```

Use the existing threshold `min_h_norm_sq=1e-30`. In the nonconflicting case (including `h==0`), retain `theta_joint` directly instead of reconstructing it through subtraction/addition. After a conflicting projection and FP32 rounding, recompute the global FP64 dot of `h` with `theta_next-theta_base`. If that computed dot is negative, conservatively suppress this step's entire additional displacement by using `theta_base` and log a rounding-fallback event. Do not repeatedly project or silently loosen a tolerance. Retain the specified joint moment state. This conservative fallback may discard useful projected updates; its activation rate is an explicit diagnostic, not an assumed negligible cost.

### Moment and protection semantics

The stored moments and step counter are exactly those of the clipped **joint** gradient; do not project, reset, double-advance, or substitute the base moments. This is projected AdamW: its parameter update can differ from the unconstrained AdamW proposal while its moments follow the submitted gradient.

Protection is only the nonnegative global FP64 dot of `h(theta)` with `theta_next-theta_base` for this step's FP32 masters. It is not a guarantee about total AdamW movement, the realized nonlinear RV score, BF16 rounding, training quality, or future steps. The shared pre-step state already includes historical Gain. Even when the current correction is fully suppressed, the current Gain remains in `s_joint` and may affect later updates. This limitation must be stated in user documentation and covered by a two-step state test, not described as trajectory-wide protection.

At LR=0, current parameter displacement is zero but native Adam moments/step counters still advance on a successful step. At `lambda_gain=0`, or an accumulation window with zero valid Gain blocks, use the normal AdamW path without a counterfactual/projection. A zero-current-Gain window does not erase earlier moment history. An actual skipped optimizer step changes neither real moments nor parameters and receives no projection or EMA advance.

## Configuration and resume identity

Add `gain_update_mode` with values `post_adam` and `adamw_projected`; default and legacy-missing value are `post_adam`.

Add a distinct `gain_optimizer_weight` coefficient for `adamw_projected`. A fresh new-mode run must provide a finite nonnegative value explicitly; do not infer it from legacy `gain_alpha=0.001`. Zero is supported for an exact-disabled equivalence check. Resume inherits the saved value and rejects explicit changes. Supplying an optimizer weight in legacy mode is an error. New-mode saved metadata must identify `gain_alpha` as legacy/inactive rather than apply both coefficients.

Expose matching underscore CLI and hyphenated coordinator flags, environment forwarding, dry-run records, both-node compatibility checks, and checkpoint formula identity. New mode and coefficient are independent of `text_selector` and `gain_weighting`.

Keep old JSON/formula identity valid without rewriting historical files; missing new fields normalize only to the legacy interpretation. Mode changes on resume are rejected before model/training execution. A fresh run is required for a different optimizer formula. All standing CE/profile coefficients remain unchanged.

## Runtime integration

Reuse the existing `_DeepSpeedStepAdapter` at the real accumulation boundary. Keep all four ranks in the same collective order, including ranks with no local Gain blocks.

1. Accumulate and globally reduce the auxiliary FP32 gradient sums and actual block count as today. Expose raw `g` and `h` to the new update adapter instead of legacy `project_gain(g,h)`.
2. Bind and validate DeepSpeed 0.18.7 ZeRO-1, no offload, BF16 model/FP32 masters, native `torch.optim.AdamW` with the repository's fused GPU setting, and loss scale 1. Bind PyTorch 2.7.1, `fused=True`, `foreach=None`, and `amsgrad=maximize=capturable=differentiable=False` for the supported GPU path; preserve configured per-group betas/epsilon/LR/decay. Reject other optimizer types/options and inconsistent mappings before mutation. CPU oracle tests use torch AdamW without fused CUDA; they do not establish fused GPU equivalence.
3. Before `original_step`, read the already reduced ordinary gradient fragments. Build FP32 owned fragments of `g`/`h` using `_hp_mapping` bounds, never full parameter all-gathers. Padding is zero and excluded from projection statistics. Promote the new mode's `averaged_gradients` entries to FP32 for the combined input, retaining shape/order and gradient-dictionary identity. Do not round the injected Gain into BF16 or reduce it a second time.
4. Compute `theta_base` on temporary local master/state copies using the installed PyTorch AdamW functional operation with the same flags, epsilon, bias correction, LR, decay, counters, and DeepSpeed base clipping. Reuse one temporary group at a time; discard its copied moments after computing the base candidate. Retain only owned base candidate values needed for the final projection. No persistent second optimizer or counterfactual checkpoint is introduced.
5. Pass the joint fragments into the unchanged DeepSpeed step. Its global norm, clipping, actual native AdamW moment update, overflow/skip logic, and scheduler stay authoritative. Validate the norm after injection, not only the base norm.
6. Install a scoped native-optimizer post-step guard that checks all updated owned FP32 masters and moment tensors, and checks candidate BF16 casts **before** DeepSpeed's per-group BF16 copy. All ranks agree on failure in identical group order. This guard observes native group updates; it must not call optimizer.step again.
7. After a successful engine step, compute `u` from real joint masters and stored base candidates, globally project it against owned `h`, check all final FP32 values and BF16 casts, then write corrected masters and use the existing ZeRO low-precision synchronization before EMA/checkpoint visibility. This is a parameter-space correction to an AdamW-generated update, not a second raw-gradient Gain step.
8. Clear temporary candidates and auxiliary buffers at the successful/skip boundary. Existing failed-state checkpoint rejection remains effective.

The post-step projection occurs after native AdamW has changed its state. A detected error aborts all ranks and forbids checkpointing; it does not promise rollback of the native update. Do not build transactional recovery or generalized retry machinery. Resume uses the last complete saved checkpoint.

## Numerical checks and observability

Reject nonfinite reduced auxiliary gradients, ordinary/joint gradients or norms, native Adam moments, base/joint/corrected FP32 masters, and BF16 candidate casts. Global dot/norm calculations use FP64 and finite checks. Validate parameters/settings before mutation where possible; runtime failures are fatal under the existing lifecycle.

Record at update boundaries: base and joint gradient norms, base/joint/Gain displacement norms, RV dot before/after projection, projection/rounding-fallback counts, current LR, mode, and gradient coefficient. Do not label an objective diagnostic as the optimizer's ordinary loss. Scalar calculations need not add model calls. The original trainable aliases and current auxiliary buffers remain; new temporary memory is limited to local ZeRO shards, not a second full model/state replica.

## Files and component boundaries

- `src/stage1/v3_update.py`: preserve legacy controller; expose raw new-mode gradients and delegate boundary operations.
- New `src/stage1/v3_adamw_update.py`: native AdamW counterfactual and update-space math with explicit input/state contracts.
- New `src/stage1/v3_zero1_update.py`: pinned ZeRO fragment mapping, joint-gradient handoff, native-step guards, and low-precision synchronization.
- `src/stage1/trainer.py`: select the adapter and preserve real step/skip/EMA ordering.
- `src/stage1/v3_config.py`, `v3_resume.py`, `composition.py`, `src/main.py`, existing multinode parser/coordinator and shell launcher: option propagation, saved identity and compatibility validation.
- Focused `tests/stage1` optimizer/state/numerical tests plus existing multinode CLI/dry-run tests. No unrelated cleanup or dependency change.

## Verification requirements and limits

First use tiny CPU tensors and independent native torch AdamW oracles. Test fresh/nonzero moments, multiple groups/decay values, changing LR including zero, joint clipping, zero weight, h=0/tiny h, conflicting/nonconflicting directions, and the exact saved-state choice. The strongest basic oracle is: real moments equal AdamW on the joint gradient, while FP32 parameters equal base candidate plus projected **actual** joint-minus-base displacement. Verify both claims separately.

Test two-step behavior explicitly: projecting away current displacement must not secretly remove its moment contribution. Report that behavior as the documented local protection limit. Test save/resume versus uninterrupted CPU trajectories, legacy metadata byte preservation, explicit mismatch rejection, and one real native step/counter increment.

Test local-fragment partitioning, group/parameter crossing, padding, empty-local-block ranks, global denominators and collective order with controlled CPU mappings. Include FP32-finite/BF16-infinite candidates and nonfinite moment rejection. Same-size/global-norm tests alone are insufficient to prove optimizer semantics.

Independent review must assess specification compliance and quality after implementation. Fresh focused tests and the applicable project CPU suite are required before claiming CPU completion. Real GPU fused AdamW/ZeRO runtime equivalence, four-rank guard/copy ordering, memory overhead, long-sample OOM behavior and learning quality remain unverified until separately authorized GPU checks. The user's current no-training instruction prevents launching such jobs now.

## Approval boundary

The user's `그렇게 진행해봐` approves the conversational direction. Final approval of this concrete written specification is still pending under the architectural workflow. After frozen-proposal critique and final spec approval, write and review the implementation plan using the already selected native execution preference. Do not interpret document preparation as product implementation or training authorization.

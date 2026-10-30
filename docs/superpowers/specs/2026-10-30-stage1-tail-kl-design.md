# Stage1 V5 conditional tail KL — design commitment v1

Status: distribution direction approved by user `ok`; complete design v1 below awaits final implementation approval.
Base: 03aeb2d3dc0b39dbcab01bebae2c79ecd10b396b (standard loss logging already integrated and verified).

## Intent and chosen approach

Add a separately weighted KL on the non-topk vocabulary to Stage1 V5 dil-bce-orig. The user confirmed `KL(clean.detach tail || V-off/rejected tail)`, with the same excluded candidates and separate normalization on each side. Native implementation remains the selected method. This request adds the objective/configuration; it does not launch training or alter the completed EMA standard baseline.

Reuse the existing clean and V-off selected logits, candidate support and auxiliary reduction. A separate model forward is unnecessary when the existing entropy/mass path already produces V-off. Computing a masked KL from full-vocabulary probabilities without renormalizing is unsuitable: it couples the result to tail mass and can underflow. Mask logits and normalize directly over the complement instead.

Reference: [official DKD implementation](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py), `dkd_loss` NCKD. DKD excludes GT and compares temperature-scaled conditional teacher/student distributions with T squared scaling. This design replaces excluded GT with the existing candidate support and uses detached clean as the target, as the user requested. It does not copy DKD's dataset coefficients or warmup schedule.

## Frozen proposal v1

1. For each selected prediction position, let S be exactly the existing entropy/mass candidate support. Honor `sync_topk` (clean source), `independent` (V-off source), and the existing optional GT union. Tail is vocabulary minus valid S. Both distributions share this detached membership; no independent second topk selection.
2. For i outside S, p_i = softmax(z_clean.detach / T over tail)_i and q_i = softmax(z_voff / T over tail)_i. The token loss is T squared times sum p_i (log p_i - log q_i). No multiplier by clean/rejected tail mass. No clean-logit gradient from this term; direct topk-logit gradients are zero. Rejected tail logits remain live; existing shared-prefix gradient semantics stay intact.
3. Use FP32 masked log-softmax and the existing position chunks/checkpoint strategy. Do not form `1 - topk_mass`, add probability floors, or use an approximate large negative mask. Empty or singleton tails contribute zero, including zero gradients; they keep their existing position/block denominator. Handle padded/duplicate GT union slots by the existing valid-support semantics. Avoid all-masked softmax NaNs without new host synchronization.
4. Add independent `lambda_tail_kl` (finite, nonnegative; default 0/off) and `tail_kl_temperature` (finite, positive; default 1), exposed through worker and existing multi-node coordinator/launcher CLI. Existing coefficients stay unchanged. Positive tail weight requires explicit valid k; tail can be used with entropy/mass off. In that case enable the existing V-off replay as required; it is not an additional pass when V-off already exists.
5. Apply tail to the same prediction-position selection as entropy/mass. Reduce by token mean within each original block, sum blocks, and use the current auxiliary block/window/rank denominator, including empty original blocks. Add `lambda_tail_kl * normalized_tail_kl` to total without changing existing loss terms.
6. When tail is enabled, emit raw and weighted tail KL plus total in production at ordinary logging intervals, independent of optional diagnostics. Extend the recently added detached interval accumulator to also expose CE, alignment, DIL, entropy, mass and Text-off KL for that V5 run, so weighted terms explain total. Raw means the component after its existing objective reduction and before its coefficient. No per-microbatch host reads, diagnostic scans or extra forward/backward. Preserve the standard baseline logger and tail-disabled V5 execution/objective behavior.
7. Save tail coefficient, temperature and formula semantics in a new authenticated V5 contract version when tail-specific settings are used. Historical schemas 1–4 keep their exact field sets and interpretation (tail off); retain the existing source/execution identity guard. Do not promise that different source revisions can resume old checkpoints. Resumes inherit saved tail settings and reject explicit drift; adding/changing this objective to an existing run requires a fresh run.

## Critique and commitment

The seven-point proposal above is frozen as v1. Root controller applied `hate` to that exact version:

{ "root": "The implementation can appear to be tail KL while still retaining a full-vocabulary denominator or tail-mass factor, so it would train a different objective from the user's decoupled conditional KL.", "first_nail": "A small hand-derived two-tail-class fixture with fixed unequal conditional distributions must retain exactly the same KL and rejected-tail gradient when arbitrary topk logits change their full-vocabulary mass; clean and excluded-topk gradients must be zero. Use extreme head logits to also expose full-softmax underflow." }

First nail status: proposed causal CPU test, not yet executed; it is the initial TDD gate after approval. No costly/state-changing premise experiment or GPU training is proposed. The official implementation supports the conditional-KL structure; it does not verify this new implementation. No changes from proposal v1 arose from critique. Macrothink is not triggered: the key target/prediction ambiguity was resolved explicitly, the existing objective/launcher paths are known, and the disabled-by-default additive feature can be falsified cheaply before deployment. No material unresolved approach disagreement remains.

## Scope, evidence and remaining gates

This is a bounded extension of existing loss/config/resume/logging paths, not a new teacher or execution architecture. Anticipated product files: `src/step_preference/topk_entropy_loss.py`, `src/stage1/v5_objective.py`, `v5_config.py`, `v5_resume.py`, `v5_trainer.py`, `trainer.py`, `src/utils_deepseed.py`, `src/multinode_runtime.py`, and `run_stage1_training.sh`; tests follow these runtime boundaries. No Stage2 rollout is included.

Using-superpowers/readchk and brainstorming intake/source tracing were applied; target direction was confirmed in chat. Hate commitment critique above is complete. Full design approval, native implementation, independent specification/quality review, verification and branch handoff remain pending. After approval, use one concrete bounded native task brief, causal CPU tests (formula/gradients/extremes/empty/singleton, actual V5 objective/logging, CLI and strict resume), independent review, and the required project completion suite once on the final reviewed changes. Preserve existing experiments and artifacts. No claim of training-quality improvement or measured GPU overhead is made.

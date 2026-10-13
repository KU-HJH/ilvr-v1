# Stage1-v3 optimizer Gain — design commitment v1

Status: ready for final written-spec approval; not implementation or training authorization.

This commitment incorporates [frozen design proposal v1](2026-10-13-stage1-v3-optimizer-gain-design.md), SHA256 `eabc4c572368d92aa63c5f471b041bcc625a385198ba5275e28e888a42d0b534`, with the explicit correction below. The proposal is preserved unchanged as critique evidence. This commitment takes precedence on projection support; all other proposal contracts remain part of the final design.

## Agreed problem and scope

The user approved moving Gain into AdamW's gradient/moment path and constraining its actual added parameter displacement. Implement that new mode in the existing Stage1-v3 worktree. Reuse model execution, optional F token entropy weighting, raw unweighted RV gradient, global block averaging, and the ordinary loss's existing scaling.

This work covers code and CPU checks after the required design/plan approvals. No training, model GPU execution, or remote launch is authorized. Native root implementation with independent review remains the existing selected method.

## Final algorithm

1. Combine the accumulated ordinary gradient and Gain as `b-lambda_gain*g`, before global clipping and native AdamW. The combined gradient affects the real Adam first/second moments, LR, schedule and weight decay behavior.
2. Compute a base candidate using only `b` from the same pre-step FP32 parameters and Adam state, with its own global clipping norm. This uses temporary owned ZeRO shards, not a persistent reference optimizer or another model forward.
3. Run one real native AdamW update on the joint gradient. Let `u=theta_joint-theta_base`.
4. Project the incremental displacement against raw RV **only on coordinates belonging to groups whose LR for this update is greater than zero**. Capture group LRs before native step/scheduler advancement. Validate finite nonnegative LR. Let `h_active` equal `h` on those coordinates and zero elsewhere. Require `u` to be zero on zero-LR coordinates; otherwise fail rather than silently write there.
5. Apply the proposal's global FP64 projection, tiny-norm rule and post-rounding dot check with `h_active` replacing `h` in the correction/dot/norm calculations. The full RV dot equals the active RV dot because every inactive displacement is zero. Nonconflicting updates retain the native joint value directly. A conflicting update with a negative rounded dot falls back to the base candidate and records the event.
6. Store `theta_base+u_safe` and retain **joint** moments/step counters. Weight decay and real counters advance once under the existing successful-step lifecycle. A skipped step changes neither parameters nor moment state. At globally zero LR there is no parameter displacement; normal successful Adam moment updates still occur.

The protection condition is a nonnegative global FP64 dot between raw RV gradient and this step's additional FP32-master displacement. This is a projected AdamW variant. The constraint can change the native proposed parameter direction; it does not assert that the final displacement equals an unconstrained AdamW step or preserves every relative coordinatewise scale after projection.

## Required safety and compatibility

- Keep native ZeRO-1 stepping and real engine dispatch. Add Gain to FP32 owned accumulated-gradient fragments before clipping; do not inject a BF16-rounded approximation or average it twice.
- Validate the pinned native torch AdamW/DeepSpeed configuration and mappings. Guard native FP32 masters/moments and BF16 casts before each DeepSpeed group copy. Guard projected candidates before final synchronization. Rank failures are collective; failed boundaries cannot checkpoint. This does not promise rollback after native state mutation.
- Separate `gain_optimizer_weight` from old displacement `gain_alpha`; a fresh new-mode run must provide its gradient weight explicitly. Preserve legacy mode, saved coefficients/formulas, and omitted-field interpretation. Reject mode/weight changes on resume.
- Add diagnostics for norms, RV dots, group LR support, and projection/fallback rates. Keep the model forward/backward budget unchanged by design; measure runtime/memory later rather than claim equivalence from the design.

## Material limitations for explicit acceptance

**Historical momentum:** even if projection removes this step's entire Gain parameter displacement, Gain remains in the saved joint moments. It may affect later movement. A future base/joint comparison starts from that shared Gain-influenced state. This design protects the current submission's incremental movement, not all movement caused by Gain history. Broader history protection would require a different state/reference design and is not silently included.

**BF16 and nonlinear scores:** the constrained object is the FP32 master increment. Finite BF16 casts can change the RV dot's sign through rounding, and neither dot establishes monotonic nonlinear RV score or learning quality. The BF16 guard prevents Inf/NaN propagation; it does not claim BF16-visible directional protection.

**Numerical fallback:** exact-boundary projection followed by FP32 rounding can trigger the conservative whole-increment fallback. Its frequency and learning impact are unknown and must be recorded. No negligible-impact assumption is made.

## Critique synthesis and changes from proposal

The root ran `hate` on the frozen proposal and captured exactly `{root, first_nail}` in `logs/stage1-v3-optimizer-gain-20261013/hate-v1.json`. Its load-bearing objection is that joint-moment retention is weaker than a broader interpretation of RV protection. The proposed first nail is a two-step tiny native-AdamW comparison with a rejected first Gain displacement and zero second Gain.

Three read-only same-model fresh sessions (`/root/v3_optimizer_fresh_a`, `_b`, `_c`) independently framed the problem before reading the identical frozen proposal. Their classified evidence is in `logs/stage1-v3-optimizer-gain-20261013/macrothink-v1.json`.

Report incompatible divergence first: global Euclidean projection could move an LR=0 group. For `u=(0,-a)` and `h=(1,1)`, the original formula yields `(a/2,-a/2)`. The active-coordinate correction above resolves this algebraic counterexample and adds a mixed-LR regression requirement. This is the only algorithmic change from proposal v1.

Compatible divergences sharpen the temporal and precision limits. A rejected one-dimensional Gain update can leave moment state that moves against `h=1` next step. Also, base `(2,1)` plus FP32 increment `(1/256,-1/256)` has zero dot with `h=(1,1)` but its BF16-visible increment has a negative dot. Those are independent analytical counterexamples to broader claims, not observed training failures. They remain explicit limitations rather than clean review approvals.

All three reads considered the same-state AdamW comparison plausible. This convergence is reassurance only, not proof, a majority decision, independent implementation review, or fused/ZeRO runtime validation.

First-nail status: no new optimizer/model probe was executed. The independent analytical argument substantiates the retained-momentum limitation; the literal two-step CPU witness remains a required implementation test after approval. No learning/overflow-rate benefit has been demonstrated. Accepting this commitment explicitly accepts the stated local FP32 protection scope; it does not waive later implementation review or runtime verification.

## Verification and handoff

Carry the proposal's CPU native-AdamW oracle, state, clipping, resume, dtype, partition and collective-order tests into the implementation plan. Add a mixed group-LR case proving that projection never moves a zero-LR group, and a global-zero-LR case showing zero parameter movement with native moment behavior. Test the retained-moment and BF16 counterexamples as documented limits, not as successes of broader protection.

Before completion, an independent reviewer must assess both specification compliance and quality. Run fresh applicable CPU verification against the reviewed revision. Fused GPU/real ZeRO execution, four-rank behavior, long-sample memory and learning quality require separately authorized checks and remain outside a CPU-only completion claim.

Final written-spec approval binds this commitment plus the frozen proposal. Approval permits the required implementation-plan phase; the plan then receives the architectural workflow's review before native implementation. Product code has not been changed in this design phase.

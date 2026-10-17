# Stage1-v5 external content metric — design commitment v1

Status: self-reviewed final written commitment awaiting human approval.
The implementation request confirms direction; this exact commitment is the
final approval target. No code or training is authorized by this document alone.

## Problem, scope and accepted direction

The user wants the target-selection Teacher to use ILVR language input and
forward semantics, with history attention computed outside the Teacher.
Teacher means target selection; reference means DIL scoring. Keep current
DoRA/frozen initial weights (EMA1), reference rules, student target forcing,
candidate pooling/top-p/top-k, all loss formulas/coefficients/denominators,
sampler, worker settings and training lifecycle. No training/evaluation launch.
Matching ILVR input is not a claim of matching its original weights/EMA/runtime.

Replace current `query_mode=adaptive`; keep default `legacy` and its current
behavior. No third public mode or new learnable parameters. Preserve existing
checkpoint files and historical contract inspection; changed semantics require
a fresh run rather than silent native-adaptive continuation.

## Teacher observation and ownership

Use the exact current legacy Teacher wrapper input: full sample input IDs,
native attention mask, original pixels/grids, hidden-state output enabled.
Do not pass helper pixels or selected visual targets to this language call.
Latent pads retain their actual token embeddings. Teacher remains eval/no-grad.
Reuse this single language observation for global intent and text hidden states.
Extract helper candidates separately with the same frozen visual encoder.

Factor the existing legacy observation into a narrow shared helper if needed,
without changing its arguments, hidden-state extraction or processor ownership.
Tests compare to an independent literal pre-change legacy wrapper invocation,
not only to two consumers of the new helper. Preserve legacy outputs/RNG.

New adaptive uses `history_text_positions` to select each current assistant
text segment, excluding headers, special/padding and latent bodies. It consumes
only text positions before the current START and only previously selected
visual targets. A causal full-sequence pass does not permit future hidden states
or future helper content in the selection query. Native mask/window semantics
remain the legacy semantics; no claim of full-context native attention mass.

Old HistoryPrefixAdapter/ReducedHistoryAttentionObserver have no role in new
adaptive execution. They may remain as historical utilities/tests to avoid
unrelated deletions; they must not be invoked by the new path. No prefix replay,
teacher attention hooks or all-layer attention tensors are needed.

## External metric and query equation

For block i let H_i contain the Teacher's final-normalized hidden rows at the
current text positions, and Z_<i the raw visual target rows selected by earlier
blocks of this sample, in their existing selection order. Neither is a student
self-generated latent or a placeholder-position contextual hidden state.

Compute in FP32, no gradients, temperature tau=1.0 for this first implementation:

`a[t,p] = softmax_over_all_p(cos(H_i[t], Z_<i[p]) / tau)`
`w[p] = mean_over_current_text_tokens(a[t,p])`
`b_i = sum_p(w[p] * Z_<i[p])`.

L2-normalize only for cosine scoring; aggregate original raw target values.
All past selected rows have the same prior; no added block/age/count correction.
This is relative content relevance, not native attention or causal attribution.
With both text and history present, the history weight sum is one. Do not call
it native latent attention mass or use the previous native mass as a gate.

Keep discounted segment text summary d_i with gamma=0.9 using existing recurrence.
Keep global intent u and combine available parts with equal component weights:

`q_i = (u + I_text*d_i + I_history*b_i) / (1 + I_text + I_history)`.

I_text means a discounted text summary exists. I_history means both current
text and past selected targets exist. With no history or no current text, omit
the history term explicitly; no uniform fallback. A previously accumulated text
summary may still exist for an empty current segment. Current candidates are
selected using existing cosine/top-p/first-K rules and then added to the ledger.

Reject malformed/nonfinite tensors and near-zero scoring-vector norms explicitly;
do not hide invalid metric input by uniform weights. Query norm validation stays.
Bound score workspace by current query/key chunks, with one global softmax
denominator over all history rows, never independent per-chunk softmax. Do not
retain graphs or create another Teacher pass. Temperature is recorded, fixed1.0
initially; it is not an unrequested public tuning surface.

## Configuration and historical identity

Keep `--query_mode legacy|adaptive` and `--history_text_gamma`. New adaptive
uses final-layer hidden states; `--history_attention_layers=last` remains valid
as the final-layer setting, while explicit last4/all fails before model loading.
Historical last4/all contracts remain inspectable under their exact old semantics.
Do not accept unused head/layer knobs silently in current execution metadata.

Give new adaptive a new strict query schema/formula identity recording ILVR
placeholder context, external cosine relative weights, final hidden state,
tau1, gamma and relevant query/key chunk sizes. Keep the legacy query identity
unchanged. Preserve strict old native-adaptive JSON reconstruction for read-only
artifact/reference inspection; reject its training resume before model work
with original-code/fresh-run guidance. New roundtrip/resume requires exact new
semantics and existing execution/source/reference/schedule agreement.

Do not weaken the outer DIL schema/reduction identity. Include the external
metric's actual source in runtime source bindings. Existing old utility hashes
may remain for historical validation, but cannot substitute for the new metric
binding. Record semantics honestly in existing diagnostics/docs; avoid introducing
a logging framework or claiming native mass. Update launcher/verification
consumers where their old layer/call/formula assumptions are affected.

Reference input rules/weights and loss computations stay the same. New selections
naturally supply different targets to both student and reference through existing
interfaces. Reference logits need not match when targets differ; verify invariance
using fixed identical target inputs. Logical role separation does not require
duplicating the shared frozen weights or changing their update policy.

## Verification and uncertainty

TDD witnesses: literal cosine softmax and gradients absent; unequal history rows,
chunk invariance, gamma edges, empty text/history, invalid/zero/nonfinite vectors;
known directional vectors distinguish metric from uniform and affect query.
Actual tiny frozen DoRA Teacher: same literal legacy embeddings/hidden states,
no selected-target injection, no helper features in LM context, one observation;
changing external history leaves Teacher hidden states fixed while changing metric
on a discriminating fixture. Verify native mask/MRoPE and future text/helper
exclusion, row ownership, target/mask order, no input mutation/RNG/hook leakage.
Run actual parent integration: shared detached targets, fixed-input reference
invariance, student gradient, legacy parity and unchanged global-token DIL scales.
Test strict old/new contracts and early incompatible resume/CLI rejection.

Use retained SDD method, independent specification/quality/final reviews and one
fresh full-project completion suite after reviewed changes. Existing reviewed
baseline is reused, not reimplemented. No real training or GPU campaign authorized.

The main unverified premise is whether cosine between final text hidden states
and raw selected visual features carries useful history relevance rather than
near-uniform/anisotropic noise. ILVR's existing query-to-candidate cosine motivates
the space choice but does not prove this per-text metric. Tau1 can yield weak
contrast. Unit tests establish arithmetic/contracts, not relevance or accuracy.
No real-data metric distribution, BF16 cost or training improvement is established.
Before expensive training compare uniform and metric history under identical
Teacher context, gamma, history scope and losses. Any costly/state-changing
premise probe needs separate authorization; absent such a probe final approval
must explicitly accept the remaining uncertainty for implementation-only scope.

Self-review: scope, input and role distinction, fixed weights, exact metric,
unit-weight semantics, empty behavior, finite checks, CLI migration, historical
inspection/training split and independent tests are explicit. No placeholders.


## Critique synthesis and final clarifications

Frozen proposal v1 SHA256 is
`7748b23f4973d235fb827b8419abc35ff24115c21dab37fa0ef882633f53c6d8`.
The adjacent critique record preserves root hate and three fresh read-only
same-model readings, their identities, classifications and static limits.
No divergent-incompatible defect was established; convergence is reassurance,
not proof. The shared compatible root is informativeness versus arithmetic
activity. Cosine at tau1 may be weakly differentiated or semantically noisy;
a unit-sum history term still enters with its fixed coefficient. Equal
component coefficients do not guarantee equal vector influence. The row-wise
prior gives blocks with more retained rows more aggregate prior weight.

Changes from proposal, binding on implementation:

- Add a known directional, tie-free selector-boundary regression. With fixed
  identical Teacher observation, global/text components, history and candidates,
  metric versus uniform history must change actual selected candidate indices
  or counts. Weights/query changes alone are insufficient. Do not impose this
  difference or improvement on arbitrary real samples.
- One Teacher observation means one language-wrapper forward; helper visual
  extraction and existing global-intent projections are separate operations.
- Call the metric relative content relevance and the mixture equal coefficients;
  never present its sum as native attention mass, confidence or balanced influence.
- Distinguish new native-mask placeholder semantics from historical full-causal
  injected-prefix identity, with separate historical inspection and early
  training-resume entrypoint witnesses.

First-nail status: NOT RUN. The proposed fixed real-sample metric/uniform/control
comparison and any paired-loss probe need separate explicit authorization.
The control uses mismatched current text or shuffles weights relative to value
rows; jointly permuting history rows and recomputing their weights is merely
permutation invariance and cannot test semantic relevance.
No runtime evidence proves semantic usefulness or accuracy benefit. Tau1 and
cross-representation relevance remain unvalidated design choices. Approval
accepts this uncertainty for reversible implementation and discriminating tests
only; it does not authorize training, a costly premise probe or performance claims.

The user-selected method remains Subagent-driven implementation after final
approval and the required implementation plan. Existing isolation is reused;
canonical checkout, historical checkpoints and unrelated files stay preserved.
The fixed weights, reference rules and all loss denominators remain as specified.

Self-review of this exact commitment: complete architecture/data flow/formula,
empty/error cases, semantic migration, tests, critique synthesis, first-nail
status and uncertainty are explicit. No placeholders or contradictory mass/input
claims remain. Final human written-spec approval is pending.

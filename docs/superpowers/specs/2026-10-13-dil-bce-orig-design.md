# DIL-BCE-ORIG implementation design

Date: 2026-10-13. Status: **design proposal v1; not finally approved**.
Workspace: `/root/workspace/ILVR/.worktrees/dil-bce-orig`.
Base: `00c212cb598d88e5cfe22a40d11d320f04c609d5` (`feat/dil-bce-port`).

## 1. Intent, authority and completion boundary

Implement the user's `documents/dil-bce-orig-v1.md` as a separately selectable
Stage 2 objective. Its token-level chosen/frozen-reference BCE uses the full
reference vocabulary; optional V-off conditional entropy and mass penalties
are separate terms. Preserve existing selectors, saved objectives, clean CE,
latent gradients and the reproduced Stage 2 execution profile.

The user approved the separate frozen-model architecture and then the proposed
loss, resume and verification contracts. Those are design-section approvals.
This written specification still needs final human approval before planning;
implementation additionally needs the written plan reviewed and an execution
method selected. No training or evaluation launch is included in this design.

The source document is preserved unchanged (SHA-256
`9b26bd40ec40dbf882e7254f0b650595538d2432e7ace2fe363dbe8ecaf1eed1`).
It describes some features absent from the selected base. Section 2 makes
these differences explicit instead of claiming source equivalence.

## 2. Actual base and compatibility decisions

| Area | Observed base | Design for the new selector |
| --- | --- | --- |
| Entry and worker | `src/utils_deepseed.py`, `src/main.py`, `run_stage2_training.sh`, existing coordinator | Extend these routes, including dry-run and both-worker argument tests |
| Preference configuration | `StepPreferenceConfig`, `dil_config.py` | Explicit alias map and selector-specific defaults/validation |
| Existing DIL | RF, topk, reverse, GT-union selectors | Preserve their formulas, defaults, validation and identity bytes |
| Clean model | Patched Qwen `_forward_stage2`, latent rollout, selected logits, shared-prefix scorer | Add an opt-in replay packet; leave legacy calls unchanged |
| Frozen reference | No same-history reference owner/API | Add independent loader/owner and detached replay API |
| Selection | `clean`/`voff`, stable sample-local ranking | Add `gap`/`intersection` for orig, with definitions below |
| Candidate selection | OFF top-k; no `sync_topk` option | Add independent/sync modes for orig entropy |
| Visual intervention | DIL allows clean-prefix, independent, latent-body attention masking | Reuse that supported branch; reject unsupported combinations |
| Image embedding replacement | No matching `image_embedding_replacement` interface in this base; existing `evidence_replacement` has a different contract | Do not silently equate them or invent image replacement semantics; orig v1 rejects this unavailable backend |
| Serialization | PEFT schema 4, DIL identity schema 1 | Keep PEFT weight layout v4; introduce DIL objective identity v2 for orig, retain v1 for legacy |

The source document's assumed schema 21 and linked current/reference design
were not found in this base. Moving to schema 22 or importing another worktree
wholesale is not part of the design.

For the absent selection modes this spec proposes explicit definitions:
`gap = abs(H_off - H_clean)`, using full-vocabulary FP32 entropies; the absolute
gap is consistent with the documented `entropy_gap` in the sibling
`dil-both-softplus-20261001`, not evidence about the unavailable source document.
`intersection` is the intersection of independently selected clean and OFF
top-percent position sets, with no refill. Each set keeps `ceil(P * percent/100)`
eligible positions across the sample; ties use original position order.
These definitions and rejection of the unavailable replacement backend are
explicit reconciliation decisions for final human review.

## 3. Mathematical contract

At each eligible predictor t, over the full vocabulary V:

```text
logp = log_softmax(clean_logits.float())
logmu = stopgrad(log_softmax(reference_logits.float()))
f = logp - logmu
chosen[t] = softplus(-f[y[t]])
reference[t] = sum_v exp(logmu[v]) * softplus(f[v])
```

Reference expectation includes GT; no extra `1/V`, `1/2`, probability floor,
reference truncation, independent reference weight or mean-score softplus.
Orig requires beta=1, margin=0, margin_mode=logit, chosen_mode=gt,
chosen_reduction=token_mean, rejected_mode=null and legacy lambda_entropy=0.
For p=mu, each component is log(2), and the single-GT clean-logit gradient is
`(mu-onehot(y))/2`; equal values do not imply a zero gradient.

When entropy is active, select k detached candidate IDs from OFF logits
(`independent`) or clean logits (`sync_topk`). Reuse the base's `topk` tie behavior
without modifying legacy execution. Require `2 <= k < V`. Optional GT union
adds GT once; ragged support uses a valid mask and actual support size n.
Only selected OFF values remain live:

```text
logq = selected_off_logits - logsumexp(selected_off_logits over valid support)
H = -sum(q * logq)           # both q and logq live
U = log(n) - H              # KL(q_off || uniform), in nats
M = KL(Bern(stopgrad(clean support mass)) || Bern(OFF support mass))
```

Mass uses full-vocabulary probabilities over exactly the entropy support.
Use the existing stable group-gap computation where applicable; full support
has exactly zero mass loss and gradient. `lambda_topk_mass > 0` requires entropy
active; chosen-mass weight must be zero. No old rejected BCE is added.

Apply the same selected-position mask to chosen/reference/U/M. Within each
original step average selected token losses, then average original steps,
including steps whose selection is empty. Empty terms are student-connected
zeros; no reference graph is retained. CE uses its existing labels/reduction.

```text
total = lambda_ce * clean_CE
      + lambda_preference * mean_steps(chosen + reference)
      + lambda_topk_entropy * mean_steps(U)
      + lambda_topk_mass * mean_steps(M)
```

Fresh defaults are CE=0.1, preference=1, topk_entropy=0, mass=0, no entropy
selection, topk_include_gt=false, independent candidate selection and chunk
positions=128. k is required explicitly only when entropy is active. Valid
explicit CE=0 is supported as a configuration value; launching an ablation
still follows the user's standing launch policy.

## 4. Frozen model lifecycle and exact replay boundary

Add a trainer-owned `FrozenCleanReference` in `src/stage2/frozen_reference.py`.
It is not a submodule of the student, is not passed into student DDP/DeepSpeed,
and contributes no parameters to the student optimizer. Load a separate model
on each rank's local device using the existing full/Stage1-DoRA/Stage2-PEFT
artifact inspectors and loaders. Its adapter/base settings come from its own
artifact, not the student's current tuning flags. No student deepcopy or
adapter-disabled fallback defines the effective reference policy.

Fresh runs require an explicit reference checkpoint. For PEFT sources resolve
the recorded original base, adapter weights/configuration, sparse token rows
and processor together. An incomplete, inaccessible or ambiguous effective
policy fails before trainer startup. Generic adapters without the necessary
latent/sparse-token identity are unsupported. Do not load optimizer state into
the reference. Freeze every parameter and keep every module in eval mode;
verify these invariants before replay and after Trainer mode transitions.
Reference buffers/parameters must not alias mutable student storage.

Validate actual tokenizer token-to-ID mapping, special IDs, output vocabulary,
latent layout/hidden width, architecture, preprocessing and position contract.
Size equality alone is insufficient. Reference and student may have different
weights but must accept the same external conditioning history.

The opt-in clean forward captures a `CleanHistoryReplay` packet after student
latent rollout and before sidecar work. It records raw token IDs, raw image
inputs/grid information, attention mask, the actual full position IDs, latent
slot indices and detached copies of the values inserted into those slots,
predictor/GT/step alignment and sample identity. It does not expose student KV
as reference input. Capturing it must not detach or replace the student's own
live final embeddings or change its clean/sidecar computation order.

Reference replay runs entirely under no_grad. Its own
`prepare_stage2_base_inputs_embeds` handles ordinary tokens, START/END and images;
only latent-body slots are overwritten by packet values. Validate alignment,
then run its decoder once over the complete clean history using its own weights,
positions and attention. It does not run its own latent rollout. Its KV, if
materialized, is computed internally from this reference call and is never
shared with the student. Select the packet's predictors and return detached
full-vocabulary logits. No permanent reference-logit cache is allowed.

The corresponding model helper belongs beside the patched Qwen Stage2 code,
for example `stage2_reference_replay.py`; `modeling_qwen2_5_vl.py` exposes the
packet and replay entry point. PEFT resolution follows the existing wrapper
resolver. Unsupported replay/backend combinations fail explicitly.

## 5. Forward scheduling and memory

Orig gets a separate trainer dispatch before the generic non-DPO RF reducer.
Keep the actual clean CE and clean-head-before-sidecar order. Reference is
required for orig. OFF is needed only for nonzero entropy/mass or active
OFF-dependent selection (`voff`, `gap`, `intersection`). With no OFF need,
build a clean scoring plan with no sidecar groups and do not compute unused
OFF scores. Empty selections do not change collective ordering across ranks.

Extend the existing checkpointed sidecar scorer with orig-only token statistics:
OFF full-vocabulary entropy for selection, conditional entropy KL, support and
coverage diagnostics, and optional mass. For sync candidate IDs it reads aligned
detached clean logits; these selectors and clean mass targets have no gradient.
Do not materialize persistent full OFF logits or add an entropy-only decoder call.

Add `dil_bce_orig_loss.py` and `topk_entropy_loss.py` with validated pure tensor
helpers. Validate finite logits, shape/device/alignment, integral in-range GT
and candidate IDs, nonduplicated valid candidates and empty P. Mask invalid
ragged values before entropy multiplication. Use FP32 stable primitives and an
independent FP64 oracle, without loss clamping to hide errors.

The trainer evaluates chosen/reference loss in position chunks and accumulates
token sums/counts, not means of chunk means. Non-reentrant checkpointing of
loss chunks bounds retained FP32 intermediates; reference tensors remain
detached. Existing clean selected logits and the detached reference selected
logits can still occupy O(PV) storage. This design does not claim all peak memory
is bounded by chunk size. Verify saved tensors, actual peak GPU memory and cost;
if the supported production shape does not fit, stop for a reviewed revision
instead of silently truncating V or adding offload infrastructure.

## 6. Distributed reduction and padding

Reduction is orig-specific; legacy sample-local step means and CE are unchanged.
For a synchronized microbatch, let N_r be the number of original steps in real
samples on rank r, N=sum_r N_r, W the distributed world size. Each rank's raw
preference contribution is `W/N * sum_local_step_losses`; DDP's rank-average
gradient then equals the global original-step mean. Empty-selected steps count;
zero-loss sampler padding does not count. N=0 returns graph-connected zero.

Preserve the configured accumulation denominator G, including the last partial
update. The standing zero-loss-sample exposure policy additionally requires
occupancy correction: R real sample slots among W ranks gives rho=R/W. Feed
`rho * W/N * local_sum` to the existing accumulation path, separately from CE.
Thus a four-rank tail with three real samples is downweighted by 3/4, and a
one-microbatch final update still divides by G=4. Do not undo padding by
renormalizing a three-sample tail to a full configured batch.

This distinguishes the raw global-step mean from the sample-occupancy-adjusted
training contribution. It is a reconciliation with canonical training-defaults,
not a claim that the source document explicitly defined padding. The averaging
unit is the synchronized microbatch, followed by the existing fixed accumulation
mean; it is not a single step-weighted mean over an entire optimizer window.
Persist that identity. Test equal/unequal step counts, real empty-step samples,
zero-loss slots, all-empty counts, complete accumulation and partial tails.
Each rank executes the same count collectives; logging reduces detached sums
with the same denominators, rather than averaging already unequal rank means.

## 7. Configuration, CLI and errors

Only map `dil-bce-orig` and `dil_bce_orig` to `dil_bce_orig`. No global hyphen
normalization or new `dil_bce` alias. Preserve the default selector.
Use one canonical value per setting, retain explicit/omitted provenance through
CLI and worker resolution, and reject conflicting aliases instead of last-wins.

Extend the existing `--step_preference_*` namespace for reference mode/checkpoint,
context/expectation, topk_entropy weight, topk_mode, include_gt and chunk size.
Expose the source document's `--preference_topk_k` as an alias of existing
`--step_preference_topk`, and `--lambda_topk_mass` as an alias of existing
`--step_preference_lambda_topk_mass`. The formula-only fields (margin_mode,
chosen_mode/reduction, rejected_mode and chosen_mass weight) get explicit
validated orig options and metadata; do not misrepresent them as existing knobs.
Mirror supported inputs in the worker environment and coordinator forwarding.
Examples use CLI flags; this does not introduce a YAML loader.

Orig resolves omitted reference_mode/context/expectation to frozen_clean /
shared_clean_history / full_vocab. Reject none, reference top-k, chosen top-k,
mean-logp reductions, beta!=1, nonzero margin, any non-null rejected mode,
nonzero legacy entropy, image_gap and incompatible execution branches.
Require positive finite preference weight and chunk size; other allowed weights
are finite/nonnegative. If entropy is off, a valid configured k is inactive and
does not trigger candidate operations. Wrong tokenizer, replay layout, manifest,
reference identity or unsupported loader is a startup error, never a fallback.

Fresh orig remains V-only Stage2B context_preference, shared-prefix,
clean_prefix/independent, supervised visible boundaries, original image access,
SDPA decoder, FlashAttention2 vision and existing Flex sidecars. Keep current
target scopes; new runs resolve immediate_text while explicit/resumed scope
retains its represented semantics. No new packed, FSDP or ZeRO-3 support is
claimed; unsupported orig runtime profiles must fail before training.

## 8. Artifact and resume contract

Create one immutable reference snapshot under
`checkpoints/<run-id>/reference_snapshot/` using the existing artifact path
validator. Copy the required effective base/adapter/processor/config components
once per run, preserving their content identity. Do not hardlink mutable source
weights or store model files in results. No general snapshot/recovery service.
Use the existing streaming SHA-256 pattern; manifest entries cover weight shards
and indices, adapter configuration, special token rows, tokenizer mapping,
processor, architecture and replay contract. Store relative component paths and
a canonical content fingerprint independent of the source path string.

Each node prepares or accesses that snapshot using the existing coordinator's
artifact transport and verifies the same fingerprint before launch. Avoid a
four-rank write race: one node-local writer prepares a shared node-local
snapshot; other local ranks validate after publication. Distinct inaccessible
node filesystems must not be mistaken for one shared directory.

Student checkpoints store a reference to the run-level manifest and the orig
objective identity. Use DIL identity schema 2 only for orig; schema 1 legacy
serialization/read behavior and PEFT layout v4 remain supported unchanged.
The new identity includes teacher_forced_token domain, all source-document
formula fields, reference fingerprint and compatibility identity, selection,
target/execution profile, reduction/padding convention and formula version:

- `dil_bce_orig_token_ref_full_vocab_v1`
- `dil_bce_orig_token_ref_full_vocab_topk_entropy_v1`
- `dil_bce_orig_token_ref_full_vocab_topk_gt_union_entropy_v1`

Resume resolves omitted values from saved identity before applying fresh
defaults. Explicit conflicting values fail before parent Trainer restores state.
Restore the saved snapshot; a supplied replacement path is accepted only when
its content fingerprint matches. Missing artifacts, altered bytes or missing
orig identity are errors. RF/orig switches require a fresh weight-only start.
Do not rewrite historical checkpoints or reinterpret their formulas.

## 9. Diagnostics and acceptance evidence

Implement every source-document section-10 orig/entropy diagnostic with its
stated raw/weighted meaning. Separate reference loss from rejected loss; never
reuse legacy negative keys for the reference term. Record raw global step means,
occupancy-adjusted training components, eligible/selected token and step counts,
reference probability-sum error, support size, entropy/normalized entropy and
mass separately. Token diagnostics are token means, loss components are step
means; metadata records both conventions. Disabled entropy fields are explicit.

Acceptance matrix:

| Requirement | Required evidence |
| --- | --- |
| Source D1-D8 | Independent FP64 formulas, population and single-GT gradients, detached reference, token-before-step reduction, wrong-approximation fixtures |
| Source E1-E7 | Uniform/nonuniform/saturated entropy, live-q derivative, candidate modes/GT union, support-only gradients, full-support mass zero |
| Source N1-N4 | Chunk 1/128/all loss+gradient comparison, dtype/extremes, ragged padding and bad-input/config failures |
| Source I1/I7/I10/I11/I12 | Real alias/CLI resolution, empty masks/steps, CE range and coefficient, all legacy regressions, no duplicate objective, diagnostic values |
| Source I2-I4 | Tiny patched-model and PEFT runtime: same-history logits, differing embeddings, frozen weights/buffers after update, no reference-to-student input gradient, Trainer mode transitions |
| Source I5-I6 | Actual decoder/sidecar call witnesses for entropy off/on and each selection source; no unnecessary OFF forward |
| Source I8 | CPU Gloo multi-process gradient comparison against a literal global oracle, including padding/accumulation conventions above |
| Source I9 | Fresh snapshot, content mismatch/missing file, same-content relocated source, omitted/explicit resume options and parent-dispatch rejection tests |
| Production path | Coordinator dry-run and both-worker argv; independently reviewed four-rank GPU smoke for finite gradients, reference invariance, peak memory and added cost |

For helper comparisons begin with FP32 atol=1e-5/rtol=1e-4; FP64 oracle internal
checks use atol=1e-10/rtol=1e-8. Justify distinct model/BF16 tolerances.
Run TDD, independent specification/quality review and fresh verification against
the reviewed revision. A final full project suite is required at branch finish;
do not repeat broad suites without a causal reason. Training smoke requires
separate authorization, the existing two-node/four-rank coordinator and full
canonical launch preflight. Unit tests are not a training or performance result.

## 10. Files and design-stage uncertainty

New focused modules: `src/step_preference/dil_bce_orig_loss.py`,
`topk_entropy_loss.py`, `dil_bce_orig_reduction.py`,
`src/stage2/frozen_reference.py`, `reference_artifact.py`, and the patched-Qwen
`stage2_reference_replay.py`. Existing integration points: preference
config/identity, Stage2 trainer and model composition, Qwen clean output and
shared-prefix scorer, CLI/worker/coordinator artifact forwarding and tests.
Reuse existing helpers where contracts match; no unrelated trainer rewrite.

Fresh baseline evidence is 116 existing DIL tests passing at the base revision.
No new implementation, same-history reference replay, new gradient formula,
distributed orig path, reference snapshot round trip or production-memory claim
has yet been tested. The final design commitment will record critique findings,
first-nail status, any revisions and the remaining uncertainty before approval.

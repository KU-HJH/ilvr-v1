# Step-wise Modality DPO Design

**Status:** design commitment v1, approved 2026-09-03
**Date:** 2026-09-03  
**Target branch/worktree:** `ours1` / `.worktrees/ours1`  
**Source brief:** `documents/TODO-ours1.md` in the primary ILVR checkout

## 1. Purpose

Extend the completed ILVR Stage 1 and Stage 2 implementation with genuine
step-wise DPO and modality-ablation entropy regularization. Correctness and
methodological fidelity come before optimization.

The work has two strictly ordered milestones:

1. Implement and verify TODO steps 1–8 with independent, naive forwards.
2. Only after that gate is green, implement TODO steps 9–11 with
   shared-prefix/tree-packed computation and measure its benefit.

The numbered execution order is normative: (1) baseline Stage 2 tests,
(2) reasoning-span metadata, (3) offline negatives, (4) Step-DPO loss,
(5) entropy-margin loss, (6) naive V/T-off forwards, (7) Stage 2B objective
integration, (8) small-subset sanity training, (9) tree-packed/shared-prefix
implementation, (10) naive/packed loss and gradient equivalence, and
(11) throughput/peak-memory benchmarking.

The first milestone is not complete merely because unit tests pass. It also
requires a short GPU sanity run on real CoMT samples. For that inexpensive
sanity run, an existing valid Stage 2 checkpoint may stand in for a newly
trained Stage 2A snapshot. This shortcut is a smoke-test convention, not a
claim that a full Stage 2A/2B experiment has been run.

### 1.1 Existing runtime boundary

The current implementation is Qwen2.5-VL-specific. Its Stage 2 entry point is
`src/main.py::run_stage2`; `src/stage2/data.py::Stage2DataCollator` renders one
two-turn user/assistant conversation; `src/stage2/trainer.py::Stage2Trainer`
delegates the clean CE loss to the vendored Qwen implementation; and
`transformers/.../stage2_latent_rollout.py::LatentRolloutEngine` replaces latent
pad embeddings before one final full-sequence decoder pass.

The source row shape relevant to this design is:

```json
{
  "image_input": ["relative/original.png"],
  "text_input": "question",
  "sequence_plan": [
    {"type": "text", "content": "previous reasoning"},
    {"type": "latent", "helper_image": "relative/helper.png"},
    {"type": "text", "content": "target reasoning"}
  ],
  "original_final_answer": "answer"
}
```

Stage 2 constructs exactly one user turn containing the original images and
`text_input`, followed by exactly one assistant turn containing the concatenated
plan entries. There are no system/developer turns or later user turns in this
contract. `clean history` means every rendered token in those two turns before
the target block, with only the two named modality-edge exceptions.

## 2. Fixed method

For each valid reasoning unit,

```text
T_(k-1) -> Z_k -> T_k
```

the clean context is the original question, original image inputs, all clean
history before `T_(k-1)`, `T_(k-1)`, and `Z_k`. Visual-off blocks only the
attention edge from predictions in `T_k` to `Z_k`. Text-off blocks only the
edge from predictions in `T_k` to `T_(k-1)`. No token or latent is deleted,
replaced, or reordered, and multimodal RoPE positions remain those of the clean
trajectory.

Ablation is used to generate the rejected completion and to compute the
entropy branch. Chosen and rejected DPO scores are both evaluated under the
same clean context.

Stage 2B optimizes

```text
clean CE + lambda_dpo * step DPO + lambda_entropy * relative entropy margin
```

Invalid preference pairs are omitted only from DPO. Their structurally valid
reasoning steps still contribute to the entropy objective.

## 3. Lifecycle and optimizer-step budget

The implementation uses three explicit commands rather than an in-process
phase transition:

1. Run ordinary Stage 2 CE as Stage 2A, or select an existing valid Stage 2
   checkpoint for smoke verification.
2. Run `scripts/generate_step_negatives.py` against the frozen Stage 2A
   snapshot.
3. Start a fresh Stage 2B trainer from the Stage 2A model weights, without
   restoring its optimizer, scheduler, RNG, or Trainer state.

For a real split lifecycle, one `total_steps` and `stage2a_ratio` determine the
two positive integer budgets:

```text
N_2A = floor(total_steps * stage2a_ratio)
N_2B = total_steps - N_2A
N_2A + N_2B = total_steps
```

Configuration rejects a ratio outside `(0, 1)`, a total budget below two
steps, or a split that gives either phase zero steps. The existing-checkpoint
smoke path uses an explicitly labelled, small Stage 2B budget and does not
present that run as a full budget-split experiment.

This contract preserves optimizer-step count, not training FLOPs, tokens, wall
time, or energy. The naive Stage 2B objective intentionally performs more
forwards per optimizer step. Verification records branch-forward and
scored-token counts so the fixed step budget is not misreported as a fixed
compute budget.

Stage 2B receives a distinct `stage2a_checkpoint_path`. This is intentionally
different from `resume_from_checkpoint_path`: the former initializes model
weights for a new objective, while the latter resumes a Stage 2B run and all of
its training state.

All model-bearing artifacts remain under `checkpoints/`. Negative datasets,
metrics, verification records, and benchmarks remain under `results/`. All
execution logs remain under `logs/`.

## 4. Data and reasoning-span contract

### 4.1 Prepared examples

`Stage2PreparedExample` is extended to retain:

- `sample_id`;
- the validated `sequence_plan` needed to reconstruct reasoning identity;
- assistant-content character spans for the plan's text and latent steps;
- the existing conversation, original image paths, and latent-block count.

If the source row contains a non-empty string `sample_id`, that value is used.
Otherwise, the ID is the SHA-256 of the canonical JSON source row. Duplicate
IDs are rejected before training or negative generation.

The current Stage 2 behavior remains unchanged when step preference is
disabled. In particular, examples that are legal for ordinary Stage 2 do not
acquire new structural restrictions merely because the feature exists.

### 4.2 Valid reasoning units

When step preference is enabled, each latent plan entry must be immediately
preceded and immediately followed by a non-empty text plan entry. Those three
entries define `T_(k-1)`, `Z_k`, and `T_k`. A latent entry without that exact
shape is a data error rather than a silently skipped preference step.

A read-only scan of the current `data/TRAIN.jsonl` found 10,701 latent entries
across 3,450 rows, all with this immediate `text -> latent -> text` shape. This
observation supports the contract but does not replace runtime validation.

`step_id` is the zero-based ordinal among latent entries in the sample, not the
raw `sequence_plan` list index. `helper_image_idx` is the corresponding
zero-based ordinal among the sample's latent/helper entries; it is `None` only
for a separately constructed synthetic span without helper provenance.

### 4.3 Token spans

The collator starts from the existing exact calls:

```python
rendered = processor.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=False
)
batch = processor(
    text=[rendered], images=[original_images], padding=True,
    truncation=False, return_tensors="pt"
)
```

The offset-aware path performs the same image preprocessing and the same
Qwen2.5-VL image-placeholder expansion once, then calls the same fast tokenizer
with `return_offsets_mapping=True`. It uses the existing tokenizer padding side
and adds no truncation or maximum length. Character spans from the expanded,
rendered assistant content are mapped to the exact full-sequence token spans.
After dropping only `offset_mapping`, every field of the normal collated batch
must be tensor-equal to the existing processor path for the same example.

```python
@dataclass(frozen=True)
class ReasoningStepSpan:
    step_id: int
    prev_text_start: int
    prev_text_end: int
    latent_start: int
    latent_end: int
    target_text_start: int
    target_text_end: int
    helper_image_idx: int | None
```

All intervals are half-open. Text spans contain assistant text tokens only.
The latent span contains latent pad positions only; latent start/end control
tokens are excluded. Padding, chat-template control tokens, unrelated text,
and latent control tokens never enter DPO or entropy targets.

The collator returns the existing ILVR tensors plus `sample_ids` and
`reasoning_step_spans`. When a Stage 2B negative store is configured, it also
returns the matched, tokenized rejected branches and their validity/reference
metadata. Batch size remains one.

All prefix/completion variants are rendered and tokenized as complete strings.
The implementation never concatenates independently tokenized prefix and
completion IDs. Completion offsets are taken from that combined tokenization,
so a tokenizer merge at the boundary cannot change the counted/scored tokens.

## 5. Offline negatives and frozen-reference scores

### 5.1 Generation

The generator loads exactly one frozen Stage 2A snapshot and the selected
source rows. For each reasoning step and modality it builds the clean prefix
through `Z_k`, applies only the requested attention-edge ablation, and performs
deterministic greedy decoding until EOS or `max_new_tokens`. The first version
does not add sampling controls or an external judge.

Generated completion text excludes EOS and assistant/chat control tokens. For
reference and policy scoring, the input is the clean prefix followed by only
the chosen or rejected `T_k` completion and the normal assistant terminator;
no original future reasoning blocks are appended. Only completion text tokens
are scored. `min_tokens` is counted from the same combined tokenization used for
scoring, not from isolated negative text.

The generator applies these filters independently to visual and textual
negatives:

1. reject exact chosen/negative equality;
2. reject equality after whitespace normalization;
3. reject empty or shorter-than-`min_tokens` output;
4. use an already-existing step-level correctness checker if one is available
   at the integration point; do not introduce a new checker dependency.

The current repository exposes final-answer exact-match scoring, not a checker
for partial reasoning blocks. It is therefore not used as a negative filter in
this first implementation; only filters 1–3 apply.

### 5.2 Reference scoring

After generation and filtering, the same frozen snapshot scores the chosen and
each rejected completion under the clean context. Scores are mean token log
probabilities over only the relevant completion text. A rejected score may be
omitted when its pair is invalid. The chosen score is shared by both modality
pairs for the same step.

Each JSONL row has this minimum schema:

```json
{
  "sample_id": "...",
  "step_id": 0,
  "chosen_text": "...",
  "rejected_visual_text": "...",
  "rejected_textual_text": "...",
  "valid_visual_negative": true,
  "valid_textual_negative": true,
  "reference_chosen_score": -0.42,
  "reference_visual_rejected_score": -0.81,
  "reference_textual_rejected_score": -0.73
}
```

A small sidecar manifest records the SHA-256 of the source JSONL's raw bytes,
selected zero-based row indices, resolved Stage 2A checkpoint path, an ordered
list of SHA-256/size pairs for files consumed by the existing Stage 2 artifact
loader, tokenizer/config identity, latent size, generation parameters, and the
negative JSONL's raw-byte SHA-256. Fallback `sample_id` uses UTF-8 bytes from
`json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
Stage 2B validates this manifest before joining rows by `(sample_id, step_id)`.
Missing, duplicate, stale, non-finite, or structurally inconsistent rows fail
closed.

## 6. Naive model computation

### 6.1 Selective Stage 2 outputs

The vendored Qwen Stage 2 path gains a narrow internal API for:

- an optional additive 4-D final-pass attention mask;
- selected next-token logit positions;
- a selective-logits-only result that avoids materializing logits for every
  sequence position.

The additive mask has shape `[1, 1, sequence_length, sequence_length]`, the
model's floating compute dtype, zero at visible positions, and
`torch.finfo(dtype).min` at blocked positions. It starts from the existing
causal/padding mask and overwrites only the requested query-row/key-column
rectangle. It is broadcast across decoder heads and is passed unchanged through
every decoder layer.

The existing clean CE fast path remains the default. The visual encoder remains
frozen and the continuous latent rollout remains clean. Here, the `final`
language-model pass means the full-sequence decoder call made after
`LatentRolloutEngine` has produced `final_inputs_embeds`. Ablation applies only
to that call, because its query rows are `T_k` prediction rows and do not alter
the earlier computation of `Z_k`.

### 6.2 Causal next-token alignment

`ReasoningStepSpan.target_text_start/end` names target token positions. In a
causal LM, token at position `p` is scored by the hidden/query row at `p - 1`.
Therefore the attention-mask implementation translates a logical target span
`[a, b)` to physical prediction rows `[a - 1, b - 1)`. This includes the first
target token, whose prediction row is the clean prefix's last token. The public
ablation spec remains expressed in logical target-token coordinates so it
cannot be confused with model-internal shift alignment.

Tests must explicitly catch an unshifted implementation, because it would
leave the first target-token distribution clean and ablate the wrong final
row.

### 6.3 Per-sample forward graph

The naive Stage 2B policy computation is:

```text
1 clean full-trajectory forward
  -> clean CE
  -> all chosen step scores
  -> clean target entropies

1 visual-off trajectory forward per reasoning step
  -> that step's visual-off target entropy

1 text-off trajectory forward per reasoning step
  -> that step's text-off target entropy

1 clean-context forward per valid visual rejected completion
1 clean-context forward per valid textual rejected completion
  -> rejected policy scores
```

Each ablated entropy branch carries exactly one step mask. Combining masks for
multiple steps in one ordinary transformer forward is forbidden in the naive
implementation: an earlier ablated text representation could otherwise enter
a later step's history and violate the clean-history invariant. Rejected
branches are likewise deliberately independent in the correctness
implementation.

The naive milestone supports `eager` and `sdpa`, whose Qwen decoder paths consume additive
4-D masks. Enabling step preference with `flash_attention_2` fails during
configuration rather than silently losing the requested edge mask.

## 7. Losses and logging

`dpo_loss.py` computes length-normalized chosen and rejected block scores and
the standard numerically stable sigmoid DPO loss:

```text
-logsigmoid(beta * ((policy_chosen - policy_rejected)
                    - (reference_chosen - reference_rejected)))
```

Visual and textual losses are averaged over their own valid pairs, then the
overall Step-DPO term is the sum of all pair losses divided by the total valid
pair count across both modalities. `lambda_dpo` multiplies only that overall
term. Per-modality means are for logging and remain zero when that modality has
no valid pair. A globally empty pair set contributes `clean_ce * 0.0`, which
keeps the correct device/dtype and a valid autograd connection, and emits zero
pair counts.

`entropy_loss.py` selects only physical next-token logits corresponding to
`T_k`, performs `log_softmax` in float32, immediately reduces Shannon entropy,
and applies

```text
relu(entropy_margin - (ablated_entropy - clean_entropy))
```

per target token before averaging. It never retains a full-sequence
probability tensor. Each modality entropy loss is the token-weighted mean over
all structurally valid reasoning steps for that modality. The overall entropy
term is the token-weighted mean across both modalities, and `lambda_entropy`
multiplies only that overall term.

The Trainer combines the clean CE, Step-DPO, and entropy terms and accumulates
the following interval means:

```text
loss/total
loss/ce
loss/dpo_visual
loss/dpo_text
loss/entropy_visual
loss/entropy_text
stats/dpo_margin_visual
stats/dpo_margin_text
stats/entropy_clean
stats/entropy_visual_off
stats/entropy_text_off
stats/valid_visual_pairs
stats/valid_textual_pairs
```

Reference values are detached artifact inputs. Policy scores and all entropy
branches preserve gradients through the policy and its clean latent rollout.

## 8. Configuration and integration boundaries

One validated `StepPreferenceConfig` owns:

- `enabled`;
- `mode` (`stage2a` or `stage2b`);
- `beta`, `lambda_dpo`, `lambda_entropy`, and `entropy_margin`;
- `negative_path`;
- `min_tokens` and `max_new_tokens`;
- `stage2a_ratio`, `total_steps`, and `stage2a_checkpoint_path`.

Initial values are `beta=0.1`, `lambda_dpo=1.0`,
`lambda_entropy=0.1`, and `entropy_margin=0.1`. These are experiment defaults,
not method constants.

Responsibilities remain narrow:

```text
src/step_preference/spans.py        plan and token-span contracts
src/step_preference/negatives.py    row schema, filtering, loading, join checks
src/step_preference/ablation.py     logical specs and additive masks
src/step_preference/dpo_loss.py     block scores and Step-DPO
src/step_preference/entropy_loss.py relative entropy-margin loss
src/step_preference/packing.py      milestone 2 only
scripts/generate_step_negatives.py  offline generation/reference scoring
src/stage2/data.py                  preserve and collate metadata
src/stage2/trainer.py               orchestrate model branches and metrics
src/main.py                         construct phase-specific runtime
src/utils_deepseed.py               parse and validate CLI configuration
```

Launcher changes expose the configuration without creating an automatic
multi-phase recovery/orchestration system. Existing launch commands remain
valid when the feature is disabled.

## 9. Error handling

The feature fails before a training write when any correctness-critical
contract is broken:

- invalid step structure or token-span mapping;
- duplicate sample/step identity;
- negative artifact/source/checkpoint mismatch;
- non-finite reference scores or configured weights;
- invalid total budget split;
- missing Stage 2A initialization for Stage 2B;
- unsupported attention backend;
- a requested target span with no scoreable causal row.

An invalid generated pair is ordinary data, not a runtime error. It is marked
invalid, excluded from DPO, and retained for entropy training.

## 10. Milestone 1 verification: TODO steps 1–8

Implementation is test-driven. The focused verification set covers:

1. current Stage 2 baseline tests before changes, using the repository venv and
   `PYTHONPATH="$PWD/transformers/src:$PWD/src" pytest -q tests/stage2`;
2. synthetic character-to-token span correctness, including multiple steps,
   control-token exclusion, and tokenizer boundary cases;
3. V-off and T-off mask isolation, causal shift alignment, unchanged remaining
   visibility, and unchanged logical positions;
4. custom DPO equality with TRL's standard sigmoid equation on toy log-probs;
5. hand-computed entropy values and a zero loss once the margin is met;
6. negative filtering, manifest validation, and sample/step joining;
7. `enabled=false` equality for collated tensors, model-call arguments, and
   Stage 2 loss;
8. tiny-model finite combined loss and finite representative gradients;
9. a short CUDA run using an existing Stage 2 checkpoint, a few real
   `data/TRAIN.jsonl` rows, generated offline negatives, and a small Stage 2B
   optimizer-step count.

The current host exposed H100 GPUs during design, while the repository's
validated deployment baseline documents B200 GPUs. Either is acceptable for
this smoke witness if it uses the repository's pinned Python/PyTorch runtime.
Pass means at least one completed optimizer step, finite total and component
losses, a finite representative gradient norm, at least one surviving DPO pair,
and a valid Stage 2B checkpoint under `checkpoints/`.

The smoke run checks that at least one visual or textual DPO pair survives. If
none survives greedy filtering in the initially selected rows, the subset is
expanded once rather than weakening the filters.

Survival establishes only that the code produced a non-identical, non-empty
pair. It does not establish that the generated continuation is semantically
worse than the source target.

The final milestone record is
`results/<run-id>/phase-a-verification.json`; logs are written to
`logs/<run-id>/`. It records exact commands, exit status, test counts, finite
loss/gradient witnesses, surviving-pair counts, and the output checkpoint path,
but contains no model state.

A runtime failure receives the smallest causal fix, one regression test, and a
fresh run ID. The failed log is preserved. No broad repeated review loop or
recovery framework is added.

## 11. Milestone 2: TODO steps 9–11

Milestone 2 is forbidden until every milestone 1 witness is green. The naive
implementation and a fixed tiny input become the oracle.

`packing.py` then represents the clean trajectory and its chosen/rejected and
V/T-off sidecars in a tree layout. Each sidecar can attend only to its clean
ancestors and its own earlier branch tokens. Sidecars cannot attend to each
other. The specified modality edge remains blocked, clean-prefix gradients are
preserved, and logical multimodal positions are copied rather than recomputed
from packed physical offsets.

A rejected branch may begin on the clean `<|latent_end|>` query row when its
first completion token immediately follows the latent block. That direct slot
is allowed only when its token ID is `latent_end_token_id` and the token at its
copied logical source in the clean root is identical. Other direct image,
video, vision-start, latent-start, latent-pad, or mismatched latent-end control
tokens fail closed. This narrow exception is design commitment v2, prompted by
the first real-data packed run; it preserves the causal row needed to score the
first target token without permitting multimodal embedding substitution.

Reducing Python forward-call count alone is insufficient. The optimized path
must use block-sparse/FlexAttention or an equivalently measured mechanism that
skips masked computation.

Completion requires, on the same inputs:

- naive and packed total/component loss agreement within declared
  `epsilon_loss`;
- relative L2 gradient error below `epsilon_grad` on a named representative
  trainable parameter subset;
- a synchronized warmup/measurement benchmark showing lower wall-clock time or
  lower peak allocated GPU memory;
- unchanged behavior when packing is disabled.

The deterministic FP32 tiny-model equivalence thresholds are
`epsilon_loss=1e-5` and `epsilon_grad=1e-4`. A BF16 production-shaped witness
uses `epsilon_loss=1e-3` and `epsilon_grad=1e-2`. Gradient comparison covers all
trainable parameters in the tiny model; the production-shaped record names and
checks the first trainable decoder `q_proj`, `o_proj`, and `down_proj` parameters
that exist after tuning wrappers are applied.

The benchmark uses the same process, GPU, model state, precision, and fixed
collated samples for both paths. Each path performs three unmeasured warmups and
ten measured iterations, with `torch.cuda.synchronize()` immediately before
and after every timed iteration and peak-memory reset after warmup. It records
the median iteration time and maximum peak allocated bytes. Milestone 2 requires
at least a 5% median-time reduction or a 1% peak-memory reduction; otherwise the
optimization is not complete.

Benchmark metrics and equivalence witnesses go under `results/<run-id>/`, and
benchmark logs go under `logs/<run-id>/`.

## 12. Explicit non-goals

The first implementation does not add null-latent replacement, Gaussian latent
noise, helper-image perturbation, global modality ablation, calibrated DPO,
SimPO, adaptive margins, per-step modality weights, online negative
regeneration, a new judge, or generalized failure recovery.

## 13. Critique resolution and interpretation limit

The final critique identified one non-blocking methodological hypothesis:
attention-ablated continuations may be coherent, equally valid, or better than
the source `T_k`, while the configured DPO dataset labels the source as chosen
and the generated continuation as rejected. Filters 1–3 do not independently
establish a preference ordering, and the repository has no step-level
correctness checker. Adding a judge or human-labelled preference dataset is
outside the approved first-version scope. Whether the weak-label method improves
the model is decided later by full training and an independent held-out
evaluation, not by the training objective's own loss.

Accordingly, this implementation uses the source target as chosen **by
construction** and must not claim that milestone 1 proves negative inferiority,
causal attribution, or improved reasoning quality. It proves that the requested
objective and interventions are implemented and executable. Any scientific
claim about those labels requires a later independent step-level audit or
evaluation.

The design's load-bearing implementation assumption is that the current Qwen
final decoder path honors the shifted additive 4-D edge mask. The cheapest
falsification is a deterministic tiny-Qwen runtime probe showing that the first
target-token distribution and subsequent target rows respond to the requested
mask, while unrelated earlier logits and logical position IDs remain equal.
This first-nail probe has not been run because it requires the new mask boundary
and is therefore implementation-like work behind final design approval. It is
the first implementation test; failure returns the work to design rather than
being patched around.

## 14. Known premises pending implementation evidence

- The current Qwen eager/SDPA decoder must preserve arbitrary additive 4-D
  masks through every decoder layer.
- The processor-equivalent offset path must reproduce existing collated tensors
  exactly while exposing stable assistant offsets.
- The available existing Stage 2 checkpoint must be loadable as a fresh
  Stage 2B policy and as the offline frozen reference.
- Greedy modality ablation must yield at least one valid preference pair on a
  reasonably small real-data subset.
- A tree-packed FlexAttention path compatible with the vendored Qwen multimodal
  RoPE can produce an actual time or memory improvement on this hardware.
- Source-target preference over each ablated continuation is an explicit weak
  label assumption, not an independently established fact.

These are not treated as established merely because the design is internally
consistent. Milestone-specific runtime tests must supply the evidence.

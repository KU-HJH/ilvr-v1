# Step-preference DoRA ablation readiness design

## Status

Design commitment v2, approved by the user on 2026-09-04.

## Problem and scope

The requested experiment compares three Stage 2B training recipes:

| Recipe | CE coefficient | DPO coefficient | entropy coefficient |
|---|---:|---:|---:|
| CE-only | 1 | 0 | 0 |
| DPO-only | 0 | 1 | 0 |
| Entropy-only | 0 | 0 | 1 |

All three recipes use the same Stage 2A checkpoint, DoRA adapter configuration,
two local H100 NVL GPUs, ZeRO-2 runtime, source rows and ordering, authenticated
negative artifact, random seed, optimizer-step budget, and evaluation protocol.
No conclusion will compare these runs directly with older runs that used the
original assistant rendering. The result ranks these three fixed recipes at one
operating point; it does not establish an intrinsic ordering among objectives
whose gradient scales have not been calibrated.

The fixed local identities and common training settings are:

- Stage 2A initialization:
  `checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827`;
  its `config.json` SHA-256 is
  `cbec86a55c91603e7f91d506d94c975113de3af4ef6e9cce6e124e6825bc7b44`
  and its `model.safetensors.index.json` SHA-256 is
  `d28f59cc1060d05418f59a2320b4f53f83ba50374d7283d59ee559aefe35864b`.
  The negative manifest records and authenticates every loaded checkpoint file.
- Training source: `data/TRAIN.jsonl`, 3,450 rows, SHA-256
  `5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8`;
  image root `data`; complete samples longer than 8,192 tokens are filtered by
  the existing immutable Stage 2 length-manifest contract.
- Evaluation source: `data/TEST.jsonl`, 400 rows, SHA-256
  `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`.
- DoRA: rank 32, alpha 64, dropout 0.05, target suffixes
  `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`.
- Runtime: seed 42, BF16, rank-local batch 1, gradient accumulation 2,
  learning rate `5e-5`, two epochs, two local processes, ZeRO-2 config
  `configs/deepspeed_zero2.json` (SHA-256
  `20615765b665d38647ee0447ca984ea28d16a1b8738363b505b054871bfc3f85`),
  gradient checkpointing enabled, public attention implementation `sdpa`, and
  packed FlexAttention enabled. The public Stage 2B setting remains `sdpa`
  because FlexAttention is selected internally only for the packed `BlockMask`.
- Trainer defaults retained explicitly: `adamw_torch`, weight decay `0.01`,
  linear scheduler, 10 warmup steps, gradient clipping `1.0`, logging every 20
  steps, save every 612 steps, one retained numbered checkpoint, latent-boundary
  supervision enabled, and dataloader workers zero.

This work addresses three observed blockers before training:

1. Packed BF16 full tuning exhausted a 93.09 GiB H100 at clean length 759 and
   packed padded length 1280.
2. Of 3,450 source rows, 2,818 have one tokenizer boundary crossing. Every
   observed crossing is between the final reasoning text entry and the terminal
   final-answer text entry, commonly through a single `.The` token. There were
   no observed crossings at intermediate reasoning boundaries.
3. The current Stage 2B loss always adds CE with coefficient one, so zeroing the
   DPO or entropy weights cannot express DPO-only or entropy-only training.

## Data boundary contract

When and only when `--step_preference_enabled` is active,
`build_assistant_plan` inserts one literal LF character between every pair of
adjacent text plan entries. It does not strip or normalize either entry, even if
one already starts or ends with whitespace. The current dataset contains such
adjacency only at the final reasoning/final-answer boundary. Plan character
spans are half-open and are computed after insertion. The LF belongs to neither
entry: the left span ends before it and the right span starts after it. Token
intervals remain the minimal contiguous intervals whose offset endpoints equal
the corresponding text span endpoints; the LF is excluded from reasoning
targets. Each source text entry otherwise remains unchanged. Ordinary Stage 2
keeps its existing byte-identical rendering.

This explicit delimiter is preferred to assigning a boundary-crossing token to
one side: `.The` contains characters from both entries, so either assignment
would make the scored token interval disagree with the source character span.
It is also preferred to filtering invalid rows, which would remove most of the
training set and bias the retained distribution.

Negative generation, reference scoring, Stage 2B collation, and all three
training recipes consume the new step-preference rendering. Negative manifest
schema version 3 adds the exact field
`assistant_plan_rendering="adjacent-text-lf-v1"`. The strict dataclass parser
requires that field and value, and canonical manifest serialization covers it.
An absent or different value fails preflight before model loading. Artifacts
made before the separator contract are therefore rejected even though the raw
source JSON digest is unchanged.

The Stage 2 length-manifest schema gains the same rendering field. The launcher
passes it to `python -m stage2.length_manifest` for every Stage 2B run; ordinary
Stage 2 writes the existing rendering identity. Each run-local manifest remains
at `results/<run-id>/stage2-length-manifest.json`, contains the ordered retained
source indices after complete no-truncation measurement, and is authenticated
by the SHA-256 already passed to training. The three payloads must have identical
retained indices and measured lengths; path and run ID are not part of that
comparison.

## Packed gradient checkpointing

The packed final decoder continues to use the FlexAttention `BlockMask`, logical
positions, shared clean embeddings, and selective logits already implemented.
The change removes the two explicit prohibitions against combining packing and
gradient checkpointing:

- the CLI validation in `src/utils_deepseed.py`;
- the packed Stage 2 model validation in
  `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py::Qwen2_5_VLForConditionalGeneration._forward_stage2`.

No new checkpoint wrapper is introduced. The packed decoder uses the model's
existing non-reentrant per-decoder-layer checkpoint function. `use_cache=False`,
`return_dict=True`, and `output_attentions=False` remain required. The latent
rollout uses the same existing checkpoint configuration. The final packed call
enters `Qwen2_5_VLModel.forward`; its `_gradient_checkpointing_func` receives
the hidden states and the same `BlockMask`, position IDs, cache position, shared
RoPE embeddings, and `generate_mode` values used by the unchecked call.
Checkpointing retains PyTorch's default RNG-state preservation and explicitly
uses `use_reentrant=False`.

The implementation is accepted only if a dropout-disabled, TF32-disabled,
seed-42 FP32 CUDA tiny model shows that every packed decoder layer executes once
during forward and once again during backward. Compared with an identical
non-checkpointed copy, loss must satisfy `rtol=1e-5, atol=1e-6`, selected logits
`rtol=2e-4, atol=2e-5`, and the full concatenated gradient vector of all
trainable decoder projection parameters `relative_l2 <= 1e-4`; all compared
values must be finite and the gradient vector must be nonzero.

## Isolated objective coefficients

`StepPreferenceConfig` gains `lambda_ce`, sourced from the existing
`--ce_weight` argument and defaulting to `1.0`. Stage 2B computes

```text
total = lambda_ce * clean_ce
      + lambda_dpo * dpo_loss
      + lambda_entropy * entropy_loss
```

Ordinary Stage 2 behavior is unchanged. Stage 2B continues to construct the
same packed tree and compute all three components even when a component's
coefficient is zero. This keeps the data, masks, forward path, and random-number
consumption matched among recipes. Tests must show that each recipe's gradient
matches the gradient of its intended isolated component and that disabled
components contribute no gradient.

`lambda_ce`, `lambda_dpo` (`--step_dpo_weight`), and `lambda_entropy`
(`--step_entropy_weight`) must each be finite and nonnegative. `beta`
(`--step_dpo_beta`) is fixed at `0.1`, finite, and strictly positive; the entropy
margin (`--step_entropy_margin`) is fixed at `0.1`, finite, and nonnegative. A
tiny deterministic model computes each raw component separately, concatenates
every trainable-parameter gradient with missing gradients represented as zeros,
and requires the recipe gradient to match the selected raw-component gradient
at `rtol=1e-5, atol=1e-7`.

## Verification and launch gates

Implementation follows test-driven development and stops on the first failed
gate:

1. CPU tests cover the newline, exact reasoning token spans, unchanged ordinary
   Stage 2 rendering, the versioned negative-manifest invalidation,
   `lambda_ce` parsing, and the exact total-loss equation.
2. A tiny CUDA packed case compares checkpointed and non-checkpointed forward,
   loss, selected logits, and representative gradients, and records an actual
   backward recomputation witness.
3. Source row 2102 reproduces the previous clean-759/packed-1280 case after a
   fresh schema-3 negative artifact is generated. It is retried first with
   single-device DoRA and the intended BF16/checkpointed packed model path.
4. Source row 1078, previously measured as the longest retained clean sample at
   8,095 tokens, is regenerated under the LF rendering and accepted only if it
   remains at or below 8,192 without truncation. It is collated and packed, then
   run through one BF16 DoRA forward/backward. After full negative generation,
   the retained row with maximum packed padded length is also run if it is not
   row 1078.
5. A two-process ZeRO-2 preflight on the maximum-packed-length row initializes
   model, DoRA adapter, optimizer, and scheduler and completes one optimizer
   step. Each memory gate requires no OOM, no truncation, finite total and raw
   component losses, and a finite nonzero adapter-gradient norm. The
   single-device gates record clean, unpadded/padded packed lengths plus peak
   allocated/reserved bytes; the two-process gate establishes optimizer-state
   allocation and one completed distributed update. Avoiding OOM is the
   capacity threshold; no unmeasured headroom claim is made.
6. Only after the preceding gates pass are the three matched training runs
   launched. Their configuration snapshots must differ only in the three loss
   coefficients and run/output identifiers.
7. Evaluation uses one frozen protocol and records per-run metrics and samples
   below `results/`; model-bearing artifacts remain below `checkpoints/`, and
   execution logs remain below `logs/`.

Fresh run IDs use
`step-pref-dora-{ce,dpo,entropy}-<UTC timestamp>`. Each run writes its adapter
and Trainer state only below `checkpoints/<run-id>/`, its normalized launch
configuration and metrics below `results/<run-id>/`, and its logs below
`logs/<run-id>/`. The existing sampler and length manifest freeze the retained
source-index list and order. A normalized cross-run comparison ignores only the
run ID, output/log paths, and the three declared coefficients; every other
training field must be equal.

One shared negative-generation run ID uses
`step-pref-negatives-adjacent-text-lf-v1-<UTC timestamp>`, with
`results/<negative-run-id>/negatives.jsonl`,
`results/<negative-run-id>/negative-manifest.json`, and
`logs/<negative-run-id>/generate.log`. It runs
`scripts/generate_step_negatives.py` on all 3,450 source rows with the fixed
Stage 2A checkpoint, `latent_size=8`, `min_tokens=2`, `max_new_tokens=32`,
`attention_implementation=sdpa`, BF16, and `cuda:0`. Its post-write consumer
validation must pass; the resulting manifest and negative SHA-256 are then
copied into each normalized launch record, and every arm receives the exact
same two paths.

Negative decoding is deterministic greedy `argmax`: it does not sample and has
no temperature, top-p, or beam-search branch. Seed 42 is still set before model
loading. Generation and all chosen/rejected reference scores use the same
frozen, unadapted Stage 2A model identified above in evaluation mode; no DoRA
adapter is attached during negative production.

The canonical launch is `run_stage2_training.sh` with
`STEP_PREFERENCE_ENABLED=true`, `STEP_PREFERENCE_MODE=stage2b`,
`STEP_PREFERENCE_PACKING=true`, the shared Stage 2A/negative paths,
`TUNING_METHOD=dora`, the DoRA and runtime values fixed above,
`DEEPSPEED_CONFIG=configs/deepspeed_zero2.json`, and
`STAGE2_NUM_PROCESSES=2`. The launcher adds and owns `CE_WEIGHT`, default `1.0`,
and forwards it as `--ce_weight`. The arm-specific values are exactly:

```text
CE-only:      CE_WEIGHT=1 STEP_DPO_WEIGHT=0 STEP_ENTROPY_WEIGHT=0
DPO-only:     CE_WEIGHT=0 STEP_DPO_WEIGHT=1 STEP_ENTROPY_WEIGHT=0
Entropy-only: CE_WEIGHT=0 STEP_DPO_WEIGHT=0 STEP_ENTROPY_WEIGHT=1
```

The experiment also records component losses and gradient norms. Those values
are diagnostic evidence for scale differences, not a post-hoc basis for
retuning one recipe. No generalized retry, recovery, or orchestration layer is
added.

Each successful run must finish two epochs, publish its final DoRA artifact,
and have no non-finite loss or skipped optimizer step. Evaluation first uses the
final DoRA artifact directly; if preparation requires a copy, only the existing
`python -m src.multinode prepare-inference-destination` plus the validated
preparation workflow may place it under
`checkpoints/evaluation-staging/<run-id>/`. `eval.py` consumes the same Stage 2A
base and each final DoRA adapter on all 400 TEST rows with
`max_sequence_length=32768`, `max_new_tokens=4096`,
`repetition_penalty=1.05`, `temperature=0`, and `top_p=1`. Completion requires
400 per-sample records plus the summary record with `correct`,
`coverage_accuracy`, `valid_gold_accuracy`, invalid-gold counts, and inference
time; predictions and the summary remain under `results/<run-id>/`.

The primary ranking metric is `coverage_accuracy` (higher is better), which
is `correct / 400`: invalid-gold rows remain in the denominator and cannot count
as correct. Exact equality is reported as a tie. `valid_gold_accuracy` is
`correct / (400 - invalid_gold_count)` when at least one gold is valid; it is
secondary and is reported without breaking a primary-metric tie. All raw
predictions remain available for error analysis.

## First nail and residual uncertainty

The load-bearing premise is that non-reentrant layer checkpointing reduces the
packed FlexAttention activation peak enough for the longest retained sample on
an H100. The cheapest decisive experiment is one real DoRA forward/backward on
the existing clean-759/packed-1280 reproduction, followed by the near-8192
sample only if the first succeeds.

This first nail has not been run because the current implementation explicitly
rejects packed gradient checkpointing. It is the first runtime action after the
approved code change and is not treated as already-proven by the tiny semantic
test. If it still OOMs, training does not launch; the result returns to design
rather than silently shortening the dataset or changing the comparison.

## Known divergence and accepted interpretation

Independent review preferred, as an alternative, preserving the original text
and assigning or masking the crossing token. That avoids a rendering shift but
cannot give `.The` an exact source-span ownership. The selected newline design
accepts a small, explicit Stage 2B rendering shift and controls it by applying
the same rendering to the CE baseline and both preference recipes. This controls
the within-experiment comparison but does not make it comparable to historical
Stage 2 runs without an additional control, which is outside the requested
simple three-run experiment.

The remaining unvalidated premises are FlexAttention checkpoint compatibility
under backward, sufficient real-sample memory reduction, and stable two-GPU
DoRA+ZeRO-2 execution. The verification gates above are the required evidence;
they are not waived.

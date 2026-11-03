# Text-masked consistency KL

The optional Text-only loss teaches the masked reader R to reproduce the
current student's clean prediction F on the same scored GT query positions:

`L_text = beta(step) * tau^2 * KL(stopgrad(softmax(F/tau)) || softmax(R/tau))`.

This compares the full vocabulary, averages tokens within each valid sample,
then averages valid samples. The clean target is detached; the masked reader
still differentiates through its shared clean-prefix KV states. Existing clean
CE, alignment, and the R/RV Gain projection remain unchanged. This option is
independent of `--clean-kl-beta`, which compares the clean student to EMA clean
reference logits. Both default to zero and may be explicitly combined.

## Configuration

For an existing Text-only launch, add `--text-kl-beta 0.01` (illustrative weight)
and optionally `--text-kl-warmup-ratio 0.05`. Environment equivalents are
`STAGE1_TEXT_KL_BETA` and `STAGE1_TEXT_KL_WARMUP_RATIO`; Python CLI equivalents are
`--stage1_text_kl_beta` and `--stage1_text_kl_warmup_ratio`.

Enabling requires packed Gain with nonzero alpha, prefix conditioning enabled,
`image_max_mask_ratio=0`, and `text_max_mask_ratio>0`. These conditions reuse the
existing Text-only path. No new text-masking trajectory is created. A query
with an empty selected mask is still included in the loss. Temperature comes
from the prefix configuration (default 2). Beta ramps linearly from zero over
`ceil(total_optimizer_steps * warmup_ratio)` completed steps; a zero warmup
ratio applies full beta immediately.

Each checkpoint saves `stage1_text_kl.json`; resume rejects changes to this
contract. Older checkpoints without it remain compatible when text KL is off.
Use the existing initial-weight loading workflow, rather than silently changing
the objective of a resumed run, to start a new experiment from old weights.

Logs include `text_kl_raw`, `text_kl_weighted`, `text_kl_beta`,
`text_kl_p95`, and `text_kl_query_count`. Raw KL includes the `tau^2` factor.

## Computation and validation limits

The target comes from existing base clean logits. R logits are consumed in the
existing packed score loop; neither another decoder nor another LM-head call
is introduced. KL is not added to the Gain score, since that would change the
projection. Instead, two cotangent lanes extract the R and KL derivatives in
one `torch.vmap(..., randomness='same')` traversal. The existing RV derivative
uses the second traversal. Do not replace this with a third serial
`autograd.grad` or set vmap `chunk_size=1`: that would add checkpoint replays.

A first-order autograd scalar transports detached KL derivatives from the
Gain aliases to optimizer-owned parameters. Its backward honors upstream loss
scaling, accumulation and the ordinary optimizer's clipping/reduction hooks;
it does not revisit the alias graph. During zero-beta warmup the existing
ordinary Gain VJPs suffice. There is no retry or serial fallback on unsupported
batching operations.

The extra vocabulary arithmetic, backward cotangent lane and saved parameter
gradients increase memory and backward work. Unchanged model-forward counts
do not mean unchanged runtime. CPU tensor tests cover KL direction, detached
targets, sample weighting, optimizer gradient transport, accumulation,
checkpoint/dropout replay counts and RNG preservation. Trainer tests supply
precomputed logits at model boundaries.

CUDA adapter dropout and checkpoint recomputation must preserve saved-mask
format and RNG. The implementation narrowly excludes FuncTorchVmapMode around
unbatched dropout operations while retaining batched backward cotangents.
The included CUDA tensor test documents this compatibility contract. Validation
must be performed in the integrated environment; prior host-specific test counts
are not a validation claim for a new workspace.

The v1 configuration keeps direct original-image access and uses the KL-based
text budget. Additional image-mask or entropy-budget variants are outside this
specification. See the main specification for the prior reference accuracy and
new-workspace integration requirements.

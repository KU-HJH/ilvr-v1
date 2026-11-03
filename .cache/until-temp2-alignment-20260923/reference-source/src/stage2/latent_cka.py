"""Within-model CKA of initial image outputs and generated visual latents.

Estimator: Reference/PyTorch-Model-Compare/torch_cka/cka.py (unbiased HSIC).
Rows are aligned examples; token averaging defines the compared representation.
"""
from __future__ import annotations

import torch


def _hsic(k, l):
    n = k.shape[0]
    return ((k * l).sum()
            + k.sum() * l.sum() / ((n - 1) * (n - 2))
            - 2 * (k.sum(1) * l.sum(1)).sum() / (n - 2)) / (n * (n - 3))


def minibatch_cka(x, y, batch_size=32):
    """Sum unbiased HSIC terms, then normalize once; do not average CKAs.

Use FP64 accumulation and merge a final 1--3-example tail into the prior batch.
Negative cross-HSIC estimates are retained, not clipped.
"""
    x, y = torch.as_tensor(x).double().cpu(), torch.as_tensor(y).double().cpu()
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("features must be two aligned sample-by-feature matrices")
    if len(x) < 4 or batch_size < 4:
        raise ValueError("unbiased HSIC requires at least four samples per batch")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("features must be finite")
    if (x == x[:1]).all() or (y == y[:1]).all():
        raise ValueError("degenerate constant representation")
    sizes = [min(batch_size, len(x) - start) for start in range(0, len(x), batch_size)]
    if sizes[-1] < 4:
        tail = sizes.pop()
        sizes[-1] += tail
    totals = torch.zeros(3, dtype=torch.float64)
    for xb, yb in zip(x.split(sizes), y.split(sizes), strict=True):
        k, l = xb @ xb.T, yb @ yb.T
        k.fill_diagonal_(0)
        l.fill_diagonal_(0)
        totals += torch.stack((_hsic(k, k), _hsic(k, l), _hsic(l, l)))
    if not torch.isfinite(totals).all() or totals[0] <= 0 or totals[2] <= 0:
        raise ValueError("degenerate HSIC self-normalizer")
    return {"cka": float(totals[1] / torch.sqrt(totals[0] * totals[2])),
            "n_samples": len(x), "batch_sizes": sizes,
            "hsic_sums": totals.tolist()}


@torch.inference_mode()
def extract_representations(*, language_model, rollout_engine, base_inputs_embeds,
                            input_ids, image_token_id, attention_mask,
                            position_ids, latent_spans):
    """Capture H0 at image positions and Hi from the real no-cache recurrence.

H0 is the decoder's final normalized output, never the input embedding.
Hi is the generated state BEFORE reinsertion at a latent-pad position, not the
output obtained by processing that inserted state (which is one token later).
The first latent-prefix call already contains all original image tokens; causal
attention makes their outputs independent of later reasoning. No replay needed.
"""
    image_positions = torch.where(input_ids[0] == image_token_id)[0]
    if not len(image_positions) or not latent_spans:
        raise ValueError("sample must contain an initial image and a latent block")
    if int(image_positions[-1]) >= latent_spans[0].latent_positions[0]:
        raise ValueError("initial image must precede the first generated latent")
    captured = {}

    def observe(**kwargs):
        output = language_model(**kwargs)
        if not captured:
            image_hidden = output.last_hidden_state[0, image_positions].float()
            image_input = base_inputs_embeds[0, image_positions].float()
            captured["h0"] = image_hidden.mean(0).cpu()
            captured["h0_vs_input_relative_l2"] = float(
                (image_hidden - image_input).norm() / image_input.norm().clamp_min(1e-12)
            )
        return output

    rollout = rollout_engine.rollout(
        language_model=observe, base_inputs_embeds=base_inputs_embeds,
        attention_mask=attention_mask, full_position_ids=position_ids,
        latent_spans=latent_spans, retain_latent_grads=False,
        collect_diagnostics=False, latent_activation_checkpointing=False,
    )
    counts = [len(span.latent_positions) for span in latent_spans]
    states = torch.stack(tuple(rollout.latent_states), dim=1)[0].float()
    if sum(counts) != len(states):
        raise ValueError("generated latent count disagrees with parsed blocks")
    captured["hi"] = torch.stack([block.mean(0) for block in states.split(counts)]).cpu()
    if not torch.isfinite(captured["h0"]).all() or not torch.isfinite(captured["hi"]).all():
        raise ValueError("nonfinite extracted hidden states")
    captured["image_token_count"] = len(image_positions)
    captured["latent_token_counts"] = counts
    captured["decoder_calls"] = rollout.decoder_call_stats.clean_forward_calls
    return captured

"""Token DIL sums, block auxiliary sums and sample-local Text-off KL."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from step_preference.dil_bce_orig_loss import orig_components_chunked
from step_preference.topk_entropy_loss import (
    build_entropy_support, topk_entropy_components, entropy_support_mass_components,
    entropy_support_tail_kl,
)
from .v5_config import Stage1V5Config
from .v5_metadata import V5BatchMetadata
from .v5_selection import QueryKey, _validate_selected_logits, select_v5_orig_positions


@dataclass(frozen=True)
class V5LossSums:
    dil_sum: Tensor
    entropy_sum: Tensor
    mass_sum: Tensor
    text_kl: Tensor
    original_steps: int
    diagnostics: dict[str, Tensor]
    tail_kl_sum: Tensor


def _zero(logits: Tensor) -> Tensor:
    # Do not convert an entire [N,V] tensor merely to produce a connected zero.
    return logits[:0].float().sum()


def _text_kl(clean: Tensor, off: Tensor, keys: tuple[QueryKey, ...],
             config: Stage1V5Config, *, validate: bool = True) -> Tensor:
    if not keys:
        return _zero(off)
    parts = []
    def token_kl(target: Tensor, restricted: Tensor) -> Tensor:
        f = (target.detach().float()/config.temperature).log_softmax(-1)
        r = (restricted.float()/config.temperature).log_softmax(-1)
        if validate and (not bool(torch.isfinite(f.detach()).all()) or not bool(torch.isfinite(r.detach()).all())):
            raise ValueError('nonfinite Text-off normalized logits')
        return config.temperature**2 * (f.exp() * (f-r)).sum(-1)
    for start in range(0, len(keys), config.loss_chunk_positions):
        stop = start + config.loss_chunk_positions
        args = (clean[start:stop].detach(), off[start:stop])
        if torch.is_grad_enabled() and off.requires_grad:
            parts.append(checkpoint(token_kl, *args, use_reentrant=False))
        else:
            parts.append(token_kl(*args))
    tokens = torch.cat(parts)
    return torch.stack([tokens[[i for i, key in enumerate(keys) if key[0] == sample]].mean()
                        for sample in sorted({key[0] for key in keys})]).mean()


def compute_v5_loss_sums(
    clean_logits: Tensor, reference_logits: Tensor, voff_logits: Tensor | None,
    textoff_logits: Tensor | None, metadata: V5BatchMetadata, config: Stage1V5Config, *,
    validate: bool = True, collect_diagnostics: bool = True,
) -> V5LossSums:
    """Consume selected [N,V] rows in metadata.query_keys order.

    D is a sum of token losses. U/mass remain sums of original-block token
    means, including empty blocks in their trainer-owned denominator. K instead
    averages queries per valid sample, then samples.
    """
    if validate:
        _validate_selected_logits(clean_logits, metadata, 'clean')
        _validate_selected_logits(reference_logits, metadata, 'reference', like=clean_logits)
    active_u = config.lambda_topk_entropy > 0
    active_mass = config.lambda_topk_mass > 0
    active_tail = config.lambda_tail_kl > 0
    keys = metadata.query_keys
    if active_u and voff_logits is None and keys:
        raise ValueError('V-off logits required for active conditional entropy')
    if active_tail and voff_logits is None and keys:
        raise ValueError('V-off logits required for active tail KL')
    if validate and voff_logits is not None:
        _validate_selected_logits(voff_logits, metadata, 'V-off', like=clean_logits)
    if config.text_kl_weight > 0 and textoff_logits is None and keys:
        raise ValueError('Text-off logits required for active Text-KL')
    if validate and textoff_logits is not None:
        _validate_selected_logits(textoff_logits, metadata, 'Text-off', like=clean_logits)
    selected = select_v5_orig_positions(metadata, clean_logits, voff_logits, config, validate=validate)
    clean_zero = _zero(clean_logits)
    off_zero = _zero(voff_logits) if voff_logits is not None else clean_zero
    chosen_sum, reference_sum = clean_zero, clean_zero
    entropy_sum, mass_sum, conditional_entropy_sum = off_zero, off_zero, off_zero
    tail_sum = off_zero
    support_size_sum = clean_logits.new_zeros((), dtype=torch.float32) if collect_diagnostics else None
    support_tokens = 0
    gt_coverage_sum = clean_logits.new_zeros((), dtype=torch.float32) if collect_diagnostics else None
    cursor, selected_steps = 0, 0
    for block in metadata.blocks:
        count = len(block.target_ids)
        mask = selected[cursor:cursor+count]
        clean = clean_logits[cursor:cursor+count][mask]
        reference = reference_logits[cursor:cursor+count][mask]
        targets = torch.tensor(block.target_ids, device=clean_logits.device, dtype=torch.long)[mask]
        if clean.shape[0]:
            selected_steps += 1
            d = orig_components_chunked(clean, reference, targets,
                                        chunk_positions=config.loss_chunk_positions,
                                        validate=validate, collect_diagnostics=collect_diagnostics)
            chosen_sum = chosen_sum + d.chosen.sum()
            reference_sum = reference_sum + d.reference.sum()
            if active_u or active_tail:
                off = voff_logits[cursor:cursor+count][mask]
                u_parts, mass_parts, h_parts = [], [], []
                tail_parts = []
                for start in range(0, len(targets), config.loss_chunk_positions):
                    stop = start + config.loss_chunk_positions
                    c, o, y = clean[start:stop], off[start:stop], targets[start:stop]
                    support = build_entropy_support(c, o, y, k=config.preference_topk_k,
                        mode=config.preference_topk_mode, include_gt=config.topk_include_gt,
                        validate=validate, collect_diagnostics=collect_diagnostics)
                    if active_u:
                        u = topk_entropy_components(o, support, validate=validate,
                                                    collect_diagnostics=collect_diagnostics)
                        u_parts.append(u.kl)
                    if collect_diagnostics:
                        if active_u:
                            h_parts.append(u.entropy.detach())
                        support_size_sum = support_size_sum + support.valid.sum(-1).float().sum()
                        support_tokens += len(y)
                        gt_coverage_sum = gt_coverage_sum + support.gt_coverage.float().sum()
                    if active_mass:
                        # Bind each chunk's detached support for backward recomputation.
                        def mass_terms(c: Tensor, o: Tensor, support=support) -> Tensor:
                            return entropy_support_mass_components(c, o, support, validate=validate,
                                collect_diagnostics=collect_diagnostics).loss
                        if torch.is_grad_enabled() and o.requires_grad:
                            mass_parts.append(checkpoint(mass_terms, c.detach(), o, use_reentrant=False))
                        else:
                            mass_parts.append(mass_terms(c.detach(), o))
                    if active_tail:
                        def tail_terms(c: Tensor, o: Tensor, support=support) -> Tensor:
                            return entropy_support_tail_kl(c, o, support,
                                temperature=config.tail_kl_temperature, validate=validate)
                        if torch.is_grad_enabled() and o.requires_grad:
                            tail_parts.append(checkpoint(tail_terms, c.detach(), o, use_reentrant=False))
                        else:
                            tail_parts.append(tail_terms(c.detach(), o))
                if active_u:
                    entropy_sum = entropy_sum + torch.cat(u_parts).mean()
                if collect_diagnostics and active_u:
                    conditional_entropy_sum = conditional_entropy_sum + torch.cat(h_parts).mean()
                if active_mass:
                    mass_sum = mass_sum + torch.cat(mass_parts).mean()
                if active_tail:
                    tail_sum = tail_sum + torch.cat(tail_parts).mean()
        cursor += count
    dil_sum = chosen_sum + reference_sum
    text_kl = (_text_kl(clean_logits, textoff_logits, keys, config, validate=validate)
               if config.text_kl_weight > 0 and textoff_logits is not None
               else _zero(textoff_logits) if textoff_logits is not None else clean_zero)
    def scalar(value: int) -> Tensor:
        return clean_logits.new_tensor(value, dtype=torch.long)
    diagnostics = {
        'dil_chosen_sum': chosen_sum.detach(), 'dil_reference_sum': reference_sum.detach(),
        'dil_raw_sum': dil_sum.detach(), 'dil_weighted_sum': config.lambda_preference*dil_sum.detach(),
        'topk_entropy_raw_sum': entropy_sum.detach(),
        'topk_entropy_weighted_sum': config.lambda_topk_entropy*entropy_sum.detach(),
        'topk_mass_raw_sum': mass_sum.detach(),
        'topk_mass_weighted_sum': config.lambda_topk_mass*mass_sum.detach(),
        'tail_kl_raw_sum': tail_sum.detach(),
        'tail_kl_weighted_sum': config.lambda_tail_kl*tail_sum.detach(),
        'text_kl_raw': text_kl.detach(), 'text_kl_weighted': config.text_kl_weight*text_kl.detach(),
        'conditional_entropy_sum': conditional_entropy_sum.detach(),
        'support_size_sum': support_size_sum.detach(), 'support_tokens': scalar(support_tokens),
        'topk_gt_coverage_sum': gt_coverage_sum.detach(),
        'original_steps': scalar(metadata.original_step_count), 'eligible_tokens': scalar(len(keys)),
        'selected_tokens': selected.sum().detach(), 'selected_steps': scalar(selected_steps),
        'text_kl_queries': scalar(len(keys) if config.text_kl_weight > 0 else 0),
        'text_kl_samples': scalar(len({k[0] for k in keys}) if config.text_kl_weight > 0 else 0),
    } if collect_diagnostics else {}
    return V5LossSums(dil_sum, entropy_sum, mass_sum, text_kl,
                      metadata.original_step_count, diagnostics, tail_sum)

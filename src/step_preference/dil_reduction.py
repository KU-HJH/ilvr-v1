"""Sample-local DIL reductions over checkpoint-local scalar token statistics."""
from collections.abc import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from step_preference.config import StepPreferenceConfig
from step_preference.dil_config import uses_entropy_selection
from step_preference.dil_topk_loss import TopkTokenResult, TopkGtTokenResult
from step_preference import entropy_selection


def reduce_dil_context_loss(
    clean_ce: Tensor,
    clean_token_logps: Sequence[Tensor],
    off_token_logps: Sequence[Tensor],
    *,
    config: StepPreferenceConfig,
    topk_terms: Sequence[TopkTokenResult | TopkGtTokenResult] | None = None,
    selection_entropies: Sequence[Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Average selected tokens within each step, then all original sample steps."""
    if config.preference_loss_type not in ("dil_bce_rf", "dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_gt"):
        raise ValueError("DIL reduction requires a DIL loss selector")
    count = len(clean_token_logps)
    topk = config.preference_loss_type in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_gt")
    reverse = config.preference_loss_type == "dil_bce_rf_topk_reverse"
    gt = config.preference_loss_type == "dil_bce_rf_topk_gt"
    if len(off_token_logps) != count or (topk and (topk_terms is None or len(topk_terms) != count)):
        raise ValueError("DIL step cardinality mismatch")
    for clean, off in zip(clean_token_logps, off_token_logps):
        if clean.ndim != 1 or off.ndim != 1 or not clean.numel() or not off.numel():
            raise ValueError("DIL requires nonempty token vectors")
        if topk and clean.shape != off.shape:
            raise ValueError("DIL top-k token alignment mismatch")
    if topk:
        result_type = TopkGtTokenResult if gt else TopkTokenResult
        for clean, terms in zip(clean_token_logps, topk_terms):
            if not isinstance(terms, result_type) or any(t.shape != clean.shape for t in terms):
                raise ValueError("DIL top-k statistics alignment mismatch")
    if uses_entropy_selection(config):
        if selection_entropies is None or len(selection_entropies) != count or any(
            e.shape != c.shape for e, c in zip(selection_entropies, clean_token_logps)
        ):
            raise ValueError("DIL selection entropy alignment mismatch")
        masks = entropy_selection.select_entropy_masks(selection_entropies, config.preference_entropy_top_percent)
    else:
        masks = tuple(torch.ones_like(c, dtype=torch.bool) for c in clean_token_logps)
    zero = clean_ce * 0.
    names = ("positive", "negative", "mass", "clean", "off", "q_entropy", "gt_coverage", "clean_mass", "off_mass", "mass_gap")
    if gt:
        names += ("gt_logq",)
    sums = dict.fromkeys(names, zero)
    selected_tokens = selected_steps = 0
    for index, (clean, off, mask) in enumerate(zip(clean_token_logps, off_token_logps, masks)):
        selected_tokens += int(mask.sum())
        if not bool(mask.any()):
            # Keep the discarded preference graph participating with zero gradient.
            sums["positive"] = sums["positive"] + clean.sum()*0.
            sums["negative"] = sums["negative"] + off.sum()*0.
            if topk:
                sums["mass"] = sums["mass"] + sum(t.sum()*0. for t in topk_terms[index])
                if gt:
                    sums["gt_logq"] = sums["gt_logq"] + topk_terms[index].gt_logq.sum()*0.
            continue
        selected_steps += 1
        clean_score = clean[mask].float().mean()
        off_score = (off[mask] if topk else off).float().mean()
        positive = (F.softplus(-config.beta*clean[mask].float()).mean() if reverse
                    else F.softplus(-config.beta*clean_score))
        sums["positive"] = sums["positive"] + positive
        sums["clean"] = sums["clean"] + clean_score
        sums["off"] = sums["off"] + off_score
        if topk:
            terms = topk_terms[index]
            if gt:
                gt_score = terms.gt_logq[mask].float().mean()
                sums["gt_logq"] = sums["gt_logq"] + gt_score
                sums["negative"] = sums["negative"] + F.softplus(config.beta*gt_score)
            term_names = ("mass", "q_entropy", "clean_mass", "off_mass", "mass_gap", "gt_coverage")
            values = terms[1:] if gt else terms
            if not gt:
                term_names = ("negative", *term_names)
            for name, value in zip(term_names, values):
                if reverse and name == "negative":
                    # E_q[log q] = -H(q); q and its expectation stay live.
                    # Average candidates and selected tokens before one softplus.
                    sums[name] = sums[name] + F.softplus(-config.beta*terms.entropy[mask].mean())
                    continue
                sums[name] = sums[name] + value[mask].mean()
        else:
            sums["negative"] = sums["negative"] + F.softplus(config.beta*off_score)
    means = {name: value/count if count else value for name, value in sums.items()}
    preference = means["positive"] + means["negative"]
    total = config.lambda_ce*clean_ce + config.lambda_preference*preference + config.lambda_topk_mass*means["mass"]
    if not bool(torch.isfinite(total)):
        raise ValueError("DIL total loss must be finite")
    metrics = {
        "loss/total": total, "loss/ce": clean_ce,
        "loss/context_preference": preference, "loss/context_preference_visual": preference,
        "loss/dil_positive": means["positive"], "loss/dil_negative": means["negative"],
        "loss/topk_mass": means["mass"], "stats/logp_clean": means["clean"],
        "stats/logp_visual_off": means["off"],
        "stats/context_margin_visual": means["clean"]-means["off"],
        "stats/topk_q_entropy": means["q_entropy"], "stats/topk_gt_coverage": means["gt_coverage"],
        "stats/topk_clean_mass": means["clean_mass"], "stats/topk_off_mass": means["off_mass"],
        "stats/topk_mass_abs_gap": means["mass_gap"],
        "stats/eligible_steps": clean_ce.new_tensor(count),
        "stats/selected_steps": clean_ce.new_tensor(selected_steps),
        "stats/eligible_tokens": clean_ce.new_tensor(sum(c.numel() for c in clean_token_logps)),
        "stats/selected_tokens": clean_ce.new_tensor(selected_tokens),
        "stats/entropy_top_percent": clean_ce.new_tensor(config.preference_entropy_top_percent or 100.),
    }
    if gt:
        metrics["stats/topk_gt_logq"] = means["gt_logq"]
    for name in ("loss/entropy", "loss/entropy_visual", "loss/entropy_text", "loss/context_preference_text",
                 "stats/context_margin_text", "stats/logp_text_off", "stats/entropy_text_off"):
        metrics[name] = zero
    return total, metrics

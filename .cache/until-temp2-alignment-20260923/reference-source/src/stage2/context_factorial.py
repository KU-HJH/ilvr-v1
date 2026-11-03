"""Factorial body, boundary, and full-block context interventions."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
import math
from pathlib import Path

import torch

from .context_ablation_analysis import _atomic_write_text, _condition_metrics
from step_preference.ablation import build_inverted_causal_mask


CONDITIONS = (
    "clean",
    "body_mask",
    "body_delete",
    "boundary_mask",
    "boundary_delete",
    "full_mask",
    "full_delete",
)
INTERVENTIONS = CONDITIONS[1:]
SCOPES = ("body", "boundary", "full")
DIRECT_CONTROL_CONDITIONS = {
    "clean_replay_vs_native": ("clean", "clean"),
    "body_mask_vs_direct": ("body_mask", "attention_mask"),
    "full_delete_vs_direct": ("full_delete", "remove_full_visual_block"),
}


@dataclass(frozen=True)
class FactorialBranchBatch:
    input_ids: torch.LongTensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    predictor_positions: tuple[torch.LongTensor, ...]
    target_ids: tuple[torch.LongTensor, ...]


def _validate_inputs(input_ids, attention_mask, clean_final_inputs_embeds, items):
    if (
        input_ids.ndim != 2 or input_ids.shape[0] != 1
        or attention_mask.shape != input_ids.shape
        or clean_final_inputs_embeds.ndim != 3
        or clean_final_inputs_embeds.shape[:2] != input_ids.shape
        or not items
    ):
        raise ValueError("factorial interventions require one aligned nonempty clean sample")


def _scope_positions(item, scope):
    if scope not in SCOPES:
        raise ValueError("scope must be body, boundary, or full")
    span = item.span
    full_start, full_end = span.prev_text_end, span.target_text_start
    if span.latent_start != full_start + 1 or span.latent_end != full_end - 1:
        raise ValueError("factorial scope requires complete START/body/END boundaries")
    if scope == "body":
        return tuple(range(span.latent_start, span.latent_end))
    if scope == "boundary":
        return (full_start, full_end - 1)
    return tuple(range(full_start, full_end))


def build_length_preserving_branches(*, input_ids, attention_mask,
                                     clean_final_inputs_embeds, items, scope,
                                     neutral_embedding):
    """Create one site-local branch per item while retaining clean length/positions."""
    _validate_inputs(input_ids, attention_mask, clean_final_inputs_embeds, items)
    if neutral_embedding.ndim == 2 and neutral_embedding.shape[0] == 1:
        neutral_embedding = neutral_embedding[0]
    if neutral_embedding.shape != clean_final_inputs_embeds.shape[-1:]:
        raise ValueError("neutral embedding must match the hidden width")
    branch_count, length = len(items), input_ids.shape[1]
    branch_ids = input_ids.expand(branch_count, -1).clone()
    branch_embeds = clean_final_inputs_embeds.expand(branch_count, -1, -1).clone()
    base = build_inverted_causal_mask(attention_mask, clean_final_inputs_embeds.dtype)
    branch_mask = base.expand(branch_count, -1, -1, -1).clone()
    predictors, targets = [], []
    minimum = torch.finfo(branch_mask.dtype).min
    for branch_index, item in enumerate(items):
        positions = item.chosen_predictor_positions.to(input_ids.device)
        blocked = _scope_positions(item, scope)
        branch_mask[branch_index, :, positions[:, None], torch.tensor(
            blocked, dtype=torch.long, device=input_ids.device
        )[None, :]] = minimum
        if scope in {"boundary", "full"}:
            neutralized = blocked if scope == "full" else _scope_positions(item, "boundary")
            branch_embeds[branch_index, list(neutralized)] = neutral_embedding
        predictors.append(positions)
        targets.append(item.chosen_target_ids.to(input_ids.device))
    return FactorialBranchBatch(
        branch_ids, branch_embeds, branch_mask, tuple(predictors), tuple(targets)
    )


def build_deleted_branches(*, input_ids, attention_mask,
                           clean_final_inputs_embeds, items, scope):
    """Physically delete one requested current-block scope per branch."""
    _validate_inputs(input_ids, attention_mask, clean_final_inputs_embeds, items)
    branch_ids, branch_embeds, branch_masks, predictors, targets = [], [], [], [], []
    removed_count = None
    for item in items:
        removed = _scope_positions(item, scope)
        if removed_count is None:
            removed_count = len(removed)
        elif len(removed) != removed_count:
            raise ValueError("all branches must remove the same number of positions")
        keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        keep[torch.tensor(removed, dtype=torch.long, device=input_ids.device)] = False
        branch_ids.append(input_ids[0, keep])
        branch_embeds.append(clean_final_inputs_embeds[0, keep])
        branch_masks.append(attention_mask[0, keep])
        original = item.chosen_predictor_positions.to(input_ids.device)
        shift = torch.tensor(
            [sum(position < int(value) + 1 for position in removed) for value in original],
            dtype=torch.long,
            device=input_ids.device,
        )
        shifted = original - shift
        if int(shifted.min()) < 0:
            raise ValueError("deleted predictor positions must remain in bounds")
        predictors.append(shifted)
        targets.append(item.chosen_target_ids.to(input_ids.device))
    return FactorialBranchBatch(
        torch.stack(branch_ids),
        torch.stack(branch_embeds),
        torch.stack(branch_masks),
        tuple(predictors),
        tuple(targets),
    )


def compare_factorial_logits(*, logits_by_condition, target_ids,
                             excluded_output_ids, vision_start_id):
    if tuple(logits_by_condition) != CONDITIONS:
        raise ValueError("factorial logits must contain every condition in canonical order")
    if target_ids.ndim != 1 or not target_ids.numel():
        raise ValueError("target_ids must be a nonempty vector")
    expected = (target_ids.numel(), logits_by_condition["clean"].shape[-1])
    if any(value.ndim != 2 or value.shape != expected for value in logits_by_condition.values()):
        raise ValueError("all factorial logits must align with one target vector")
    device = logits_by_condition["clean"].device
    if any(value.device != device for value in logits_by_condition.values()):
        raise ValueError("factorial logits must share one device")
    targets = target_ids.to(device)
    conditions = {
        name: _condition_metrics(value, targets, excluded_output_ids, vision_start_id)
        for name, value in logits_by_condition.items()
    }
    clean_argmax = torch.tensor(conditions["clean"]["argmax_ids"])
    comparisons = {}
    for name in INTERVENTIONS:
        changed = torch.tensor(conditions[name]["argmax_ids"]) != clean_argmax
        comparisons[name] = {
            "first_argmax_changed": bool(changed[0]),
            "any_argmax_changed": bool(changed.any()),
            "token_argmax_changed_count": int(changed.sum()),
            "token_argmax_changed_rate": float(changed.float().mean()),
            "gt_logp_sum_delta": conditions[name]["gt_logp_sum"] - conditions["clean"]["gt_logp_sum"],
            "first_gt_logp_delta": conditions[name]["first_gt_logp"] - conditions["clean"]["first_gt_logp"],
        }
    return {
        "target_token_count": target_ids.numel(),
        "conditions": conditions,
        "comparisons": comparisons,
    }


def summarize_factorial_records(records):
    sites = [site for record in records for site in record.get("sites", ())]
    if not sites or any(site.get("status") != "ok" or "factorial_comparison" not in site for site in sites):
        raise ValueError("summary requires complete successful factorial comparisons")
    analyses = [site["factorial_comparison"] for site in sites]
    token_count = sum(item["target_token_count"] for item in analyses)
    summary = {
        "samples": len(records),
        "sites": len(sites),
        "target_tokens": token_count,
        "conditions": {},
        "comparisons": {},
    }
    for name in CONDITIONS:
        rows = [item["conditions"][name] for item in analyses]
        total_logp = sum(row["gt_logp_sum"] for row in rows)
        summary["conditions"][name] = {
            "mean_gt_logp": total_logp / token_count,
            "mean_nll": -total_logp / token_count,
            "mean_first_gt_logp": sum(row["first_gt_logp"] for row in rows) / len(rows),
            "mean_first_full_entropy_nats": sum(row["first_full_entropy_nats"] for row in rows) / len(rows),
            "mean_first_text_entropy_nats": sum(row["first_text_entropy_nats"] for row in rows) / len(rows),
            "mean_first_text_mass": sum(row["first_text_mass"] for row in rows) / len(rows),
            "mean_first_vision_start_probability": sum(
                row["first_vision_start_probability"] for row in rows
            ) / len(rows),
            "first_argmax_vision_count": sum(row["first_argmax_is_vision"] for row in rows),
            "first_argmax_vision_rate": sum(row["first_argmax_is_vision"] for row in rows) / len(rows),
            "gt_argmax_match_count": sum(row["gt_argmax_match_count"] for row in rows),
            "gt_argmax_match_rate": sum(row["gt_argmax_match_count"] for row in rows) / token_count,
        }
    clean_mean = summary["conditions"]["clean"]["mean_gt_logp"]
    for name in INTERVENTIONS:
        rows = [item["comparisons"][name] for item in analyses]
        changed = sum(row["token_argmax_changed_count"] for row in rows)
        summary["comparisons"][name] = {
            "first_argmax_changed_count": sum(row["first_argmax_changed"] for row in rows),
            "first_argmax_changed_rate": sum(row["first_argmax_changed"] for row in rows) / len(rows),
            "any_argmax_changed_count": sum(row["any_argmax_changed"] for row in rows),
            "any_argmax_changed_rate": sum(row["any_argmax_changed"] for row in rows) / len(rows),
            "token_argmax_changed_count": changed,
            "token_argmax_changed_rate": changed / token_count,
            "mean_gt_logp_delta": summary["conditions"][name]["mean_gt_logp"] - clean_mean,
            "mean_first_gt_logp_delta": sum(row["first_gt_logp_delta"] for row in rows) / len(rows),
        }
    if not all(math.isfinite(value) for row in summary["conditions"].values()
               for key, value in row.items() if key.startswith("mean_")):
        raise ValueError("summary contains non-finite metrics")
    return summary


def summarize_direct_reference_controls(records, direct_records):
    """Quantify cross-run numerical drift against the completed direct analysis."""
    direct_sites = {}
    for record in direct_records:
        if record.get("status") != "ok":
            raise ValueError("direct reference records must be successful")
        for site in record.get("sites", ()):
            key = (record["data_index"], site.get("step_id"))
            if key in direct_sites:
                raise ValueError("direct reference contains duplicate sites")
            direct_sites[key] = site

    aligned = []
    seen = set()
    for record in records:
        for site in record.get("sites", ()):
            key = (record["data_index"], site.get("step_id"))
            if key in seen or key not in direct_sites:
                raise ValueError("factorial sites do not uniquely align with direct reference")
            seen.add(key)
            reference = direct_sites[key]
            if (
                site.get("status") != "ok"
                or reference.get("status") != "ok"
                or site.get("target_token_ids") != reference.get("target_token_ids")
            ):
                raise ValueError("factorial targets differ from direct reference")
            aligned.append((site, reference))
    if not aligned:
        raise ValueError("direct-reference controls require nonempty aligned sites")

    token_count = sum(len(site["target_token_ids"]) for site, _ in aligned)
    comparisons = {}
    for label, (current_name, reference_name) in DIRECT_CONTROL_CONDITIONS.items():
        first_changed = any_changed = changed_tokens = 0
        gt_logp_delta = first_gt_logp_delta = 0.0
        for site, reference in aligned:
            current = site["factorial_comparison"]["conditions"][current_name]
            previous = reference["direct_comparison"]["conditions"][reference_name]
            changed = [left != right for left, right in zip(
                current["argmax_ids"], previous["argmax_ids"], strict=True
            )]
            first_changed += changed[0]
            any_changed += any(changed)
            changed_tokens += sum(changed)
            gt_logp_delta += current["gt_logp_sum"] - previous["gt_logp_sum"]
            first_gt_logp_delta += current["first_gt_logp"] - previous["first_gt_logp"]
        comparisons[label] = {
            "first_argmax_changed_count": first_changed,
            "first_argmax_changed_rate": first_changed / len(aligned),
            "any_argmax_changed_count": any_changed,
            "any_argmax_changed_rate": any_changed / len(aligned),
            "token_argmax_changed_count": changed_tokens,
            "token_argmax_changed_rate": changed_tokens / token_count,
            "mean_gt_logp_delta": gt_logp_delta / token_count,
            "mean_first_gt_logp_delta": first_gt_logp_delta / len(aligned),
        }
    return {
        "sites": len(aligned),
        "target_tokens": token_count,
        "comparisons": comparisons,
        "interpretation": (
            "Cross-run batch/execution numerical control, not an intervention effect. "
            "Body-mask changes comparable to this floor must be treated as non-robust."
        ),
    }


def native_replay_control(native_logits, replay_logits, target_ids):
    """Same-model, same-sample native/replay control; never use another model."""
    if native_logits.shape != replay_logits.shape or native_logits.shape[0] != target_ids.numel():
        raise ValueError("native and replay logits must align with the same targets")
    targets = target_ids.to(native_logits.device).unsqueeze(-1)
    native_gt = native_logits.float().log_softmax(-1).gather(-1, targets).squeeze(-1)
    replay_gt = replay_logits.float().log_softmax(-1).gather(-1, targets).squeeze(-1)
    changed = native_logits.argmax(-1) != replay_logits.argmax(-1)
    return {
        "first_argmax_changed": bool(changed[0]),
        "any_argmax_changed": bool(changed.any()),
        "token_argmax_changed_count": int(changed.sum()),
        "native_gt_logp_sum": float(native_gt.sum()),
        "replay_gt_logp_sum": float(replay_gt.sum()),
        "gt_logp_sum_delta": float(replay_gt.sum() - native_gt.sum()),
        "first_gt_logp_delta": float(replay_gt[0] - native_gt[0]),
    }


def summarize_native_controls(records):
    sites = [site for record in records for site in record["sites"]]
    if not sites:
        raise ValueError("native control requires nonempty records")
    rows = [site["native_clean_control"] for site in sites]
    tokens = sum(site["target_token_count"] for site in sites)
    return {
        "sites": len(sites), "target_tokens": tokens,
        "definition": "same-model batched clean replay minus native clean forward",
        "first_argmax_changed_count": sum(row["first_argmax_changed"] for row in rows),
        "any_argmax_changed_count": sum(row["any_argmax_changed"] for row in rows),
        "token_argmax_changed_count": sum(row["token_argmax_changed_count"] for row in rows),
        "mean_gt_logp_delta": sum(row["gt_logp_sum_delta"] for row in rows) / tokens,
        "mean_first_gt_logp_delta": sum(row["first_gt_logp_delta"] for row in rows) / len(sites),
    }


def write_direct_reference_controls(output, records, direct_records):
    output = Path(output)
    controls = summarize_direct_reference_controls(records, direct_records)
    _atomic_write_text(
        output / "numerical_control.json",
        json.dumps(controls, indent=2, ensure_ascii=False) + "\n",
    )
    lines = [
        "# Numerical replay control", "",
        controls["interpretation"], "",
        "| comparison | first changed | any changed | token changed | mean GT logp delta | mean first GT logp delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in controls["comparisons"].items():
        lines.append(
            f"| {name} | {row['first_argmax_changed_count']}/{controls['sites']} "
            f"({row['first_argmax_changed_rate']:.4%}) | "
            f"{row['any_argmax_changed_count']}/{controls['sites']} "
            f"({row['any_argmax_changed_rate']:.4%}) | "
            f"{row['token_argmax_changed_count']}/{controls['target_tokens']} "
            f"({row['token_argmax_changed_rate']:.4%}) | "
            f"{row['mean_gt_logp_delta']:.8f} | {row['mean_first_gt_logp_delta']:.8f} |"
        )
    lines += ["", (
        "The main factorial report uses internally paired batched replay conditions. "
        "The clean-replay row measures its numerical distance from the native batch-1 clean forward."
    ), ""]
    _atomic_write_text(output / "numerical_control.md", "\n".join(lines))
    return controls


def write_factorial_outputs(output, records, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_factorial_records(records)
    _atomic_write_text(
        output / "records.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )
    _atomic_write_text(
        output / "summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    condition_rows = [dict(condition=name, **summary["conditions"][name]) for name in CONDITIONS]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(condition_rows[0]))
    writer.writeheader()
    writer.writerows(condition_rows)
    _atomic_write_text(output / "condition_summary.csv", buffer.getvalue())
    comparison_rows = [dict(condition=name, **summary["comparisons"][name]) for name in INTERVENTIONS]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(comparison_rows[0]))
    writer.writeheader()
    writer.writerows(comparison_rows)
    _atomic_write_text(output / "comparison_summary.csv", buffer.getvalue())

    lines = [
        "# Factorial Vision-context interventions at T(i+1)", "",
        f"Status: {config.get('status', 'unknown')}; samples: {summary['samples']}; "
        f"sites: {summary['sites']}; target tokens: {summary['target_tokens']}.", "",
        "## Operator × scope", "",
        "Condition IDs: body_mask/body_delete, boundary_mask/boundary_delete, full_mask/full_delete.", "",
        "| scope | mask/neutral first changed | mask/neutral any changed | mask/neutral token changed | delete first changed | delete any changed | delete token changed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in SCOPES:
        masked = summary["comparisons"][f"{scope}_mask"]
        deleted = summary["comparisons"][f"{scope}_delete"]
        lines.append(
            f"| {scope} | {masked['first_argmax_changed_rate']:.4%} | "
            f"{masked['any_argmax_changed_rate']:.4%} | {masked['token_argmax_changed_rate']:.4%} | "
            f"{deleted['first_argmax_changed_rate']:.4%} | "
            f"{deleted['any_argmax_changed_rate']:.4%} | {deleted['token_argmax_changed_rate']:.4%} |"
        )
    lines += ["", "## Length-preserving four-condition", "",
              "| condition | mean NLL | first GT logp | full H | text-cond. H | text mass | latent START p | first argmax latent START |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("clean", "body_mask", "boundary_mask", "full_mask"):
        row = summary["conditions"][name]
        lines.append(
            f"| {name} | {row['mean_nll']:.8f} | {row['mean_first_gt_logp']:.8f} | "
            f"{row['mean_first_full_entropy_nats']:.8f} | {row['mean_first_text_entropy_nats']:.8f} | "
            f"{row['mean_first_text_mass']:.8g} | {row['mean_first_vision_start_probability']:.8g} | "
            f"{row['first_argmax_vision_rate']:.4%} |"
        )
    lines += ["", "Mask/neutral branches preserve every sequence position. Delete branches remove "
              "body (8), boundary (2), or the full START/body/END block (10) and recompute positions.", ""]
    control = config.get("native_clean_control_summary")
    if control:
        lines += [
            "## Same-model numerical control", "",
            f"Native clean vs batched replay: first-token changes "
            f"{control['first_argmax_changed_count']}/{control['sites']}; "
            f"any-token changes {control['any_argmax_changed_count']}/{control['sites']}; "
            f"token changes {control['token_argmax_changed_count']}/{control['target_tokens']}.", "",
            f"Replay minus native mean GT logp: {control['mean_gt_logp_delta']:.8f}; "
            f"mean first GT logp: {control['mean_first_gt_logp_delta']:.8f}.", "",
            "Small intervention effects comparable to this control need numerical caution. "
            "Different-checkpoint reference predictions are not numerical controls.", "",
        ]
    _atomic_write_text(output / "report.md", "\n".join(lines))
    _atomic_write_text(
        output / "run_config.json",
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
    )
    return summary

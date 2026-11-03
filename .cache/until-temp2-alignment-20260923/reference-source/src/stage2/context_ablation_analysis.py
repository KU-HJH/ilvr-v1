"""Metrics for paired clean, visual-mask, and full-removal teacher forcing."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path

import torch


CONDITIONS = ("clean", "attention_mask", "remove_full_visual_block")
ABLATIONS = CONDITIONS[1:]


def _condition_metrics(logits, targets, excluded_output_ids, vision_start_id):
    values = logits.float()
    log_probs = values.log_softmax(dim=-1)
    target_logps = log_probs.gather(-1, targets[:, None]).squeeze(-1)
    argmax_ids = values.argmax(dim=-1)
    first_probs = log_probs[0].exp()
    full_entropy = -(first_probs * log_probs[0]).sum()
    allowed = torch.ones(values.shape[-1], dtype=torch.bool, device=values.device)
    excluded = [index for index in excluded_output_ids if 0 <= index < values.shape[-1]]
    if excluded:
        allowed[torch.tensor(excluded, dtype=torch.long, device=values.device)] = False
    if not bool(allowed.any()):
        raise ValueError("text-conditional vocabulary must not be empty")
    text_log_probs = values[0, allowed].log_softmax(dim=-1)
    text_probs = text_log_probs.exp()
    return {
        "gt_logp_sum": float(target_logps.sum()),
        "mean_gt_logp": float(target_logps.mean()),
        "mean_nll": float(-target_logps.mean()),
        "first_gt_logp": float(target_logps[0]),
        "first_argmax_id": int(argmax_ids[0]),
        "first_argmax_is_gt": bool(argmax_ids[0] == targets[0]),
        "first_argmax_is_vision": int(argmax_ids[0]) == vision_start_id,
        "first_full_entropy_nats": float(full_entropy),
        "first_text_entropy_nats": float(-(text_probs * text_log_probs).sum()),
        "first_text_mass": float(first_probs[allowed].sum()),
        "first_vision_start_probability": float(first_probs[vision_start_id]),
        "argmax_ids": argmax_ids.tolist(),
        "gt_argmax_match_count": int((argmax_ids == targets).sum()),
        "gt_argmax_match_rate": float((argmax_ids == targets).float().mean()),
    }


def compare_condition_logits(*, clean_logits, masked_logits, removed_logits, target_ids,
                             excluded_output_ids, vision_start_id):
    """Compare identical GT targets under three context interventions."""
    logits_by_condition = {
        "clean": clean_logits,
        "attention_mask": masked_logits,
        "remove_full_visual_block": removed_logits,
    }
    if not isinstance(target_ids, torch.Tensor) or target_ids.ndim != 1 or target_ids.numel() == 0:
        raise ValueError("target_ids must be a non-empty vector")
    expected = (target_ids.numel(), clean_logits.shape[-1])
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape != expected
           for value in logits_by_condition.values()):
        raise ValueError("all condition logits must align with the same target vector and vocabulary")
    targets = target_ids.to(device=clean_logits.device, dtype=torch.long)
    if any(value.device != clean_logits.device for value in logits_by_condition.values()):
        raise ValueError("condition logits must share one device")
    conditions = {
        name: _condition_metrics(value, targets, excluded_output_ids, vision_start_id)
        for name, value in logits_by_condition.items()
    }
    clean_argmax = torch.tensor(conditions["clean"]["argmax_ids"])
    comparisons = {}
    for name in ABLATIONS:
        changed = torch.tensor(conditions[name]["argmax_ids"]) != clean_argmax
        comparisons[name] = {
            "first_argmax_changed": bool(changed[0]),
            "any_argmax_changed": bool(changed.any()),
            "token_argmax_changed_count": int(changed.sum()),
            "token_argmax_changed_rate": float(changed.float().mean()),
            "gt_logp_sum_delta": conditions[name]["gt_logp_sum"] - conditions["clean"]["gt_logp_sum"],
            "mean_gt_logp_delta": conditions[name]["mean_gt_logp"] - conditions["clean"]["mean_gt_logp"],
            "first_gt_logp_delta": conditions[name]["first_gt_logp"] - conditions["clean"]["first_gt_logp"],
        }
    return {
        "target_token_count": target_ids.numel(),
        "conditions": conditions,
        "comparisons": comparisons,
    }


def build_site_comparisons(*, items, native_clean_logits, clean_logits_by_key, masked_logits_by_key,
                           removed_logits_by_key, excluded_output_ids, vision_start_id):
    """Split flat clean logits and align both sidecar mappings by stable step key."""
    if (not isinstance(native_clean_logits, torch.Tensor) or native_clean_logits.ndim != 3
            or native_clean_logits.shape[0] != 1):
        raise ValueError("native clean logits must have shape [1, target_tokens, vocabulary]")
    ordered = sorted(items, key=lambda item: item.span.target_text_start)
    cursor, sites = 0, []
    for item in ordered:
        count = item.chosen_target_ids.numel()
        key = f"{item.sample_id}:{item.span.step_id}:visual:chosen_entropy"
        if (key not in clean_logits_by_key or key not in masked_logits_by_key
                or key not in removed_logits_by_key):
            raise ValueError(f"missing sidecar logits for {key}")
        stop = cursor + count
        if stop > native_clean_logits.shape[1]:
            raise ValueError("clean logits end before the requested target span")
        direct = compare_condition_logits(
            clean_logits=clean_logits_by_key[key],
            masked_logits=masked_logits_by_key[key],
            removed_logits=removed_logits_by_key[key],
            target_ids=item.chosen_target_ids,
            excluded_output_ids=excluded_output_ids,
            vision_start_id=vision_start_id,
        )
        native = _condition_metrics(
            native_clean_logits[0, cursor:stop],
            item.chosen_target_ids.to(native_clean_logits.device),
            excluded_output_ids,
            vision_start_id,
        )
        replay = direct["conditions"]["clean"]
        native_argmax = torch.tensor(native["argmax_ids"])
        replay_argmax = torch.tensor(replay["argmax_ids"])
        control_changed = native_argmax != replay_argmax
        control = {
            "native_mean_gt_logp": native["mean_gt_logp"],
            "replay_mean_gt_logp": replay["mean_gt_logp"],
            "gt_logp_sum_delta": replay["gt_logp_sum"] - native["gt_logp_sum"],
            "mean_gt_logp_delta": replay["mean_gt_logp"] - native["mean_gt_logp"],
            "first_argmax_changed": bool(control_changed[0]),
            "any_argmax_changed": bool(control_changed.any()),
            "token_argmax_changed_count": int(control_changed.sum()),
            "token_argmax_changed_rate": float(control_changed.float().mean()),
        }
        span = item.span
        sites.append({
            "status": "ok",
            "step_id": span.step_id,
            "target_text_interval": [span.target_text_start, span.target_text_end],
            "target_token_count": count,
            "target_token_ids": item.chosen_target_ids.tolist(),
            "latent_body_interval": [span.latent_start, span.latent_end],
            "full_visual_interval": [span.prev_text_end, span.target_text_start],
            "native_clean_control": control,
            "direct_comparison": direct,
        })
        cursor = stop
    if cursor != native_clean_logits.shape[1]:
        raise ValueError("clean logits contain unassigned target tokens")
    return sites


def summarize_records(records):
    """Aggregate sites without mixing per-site and per-token denominators."""
    sites = [site for record in records for site in record.get("sites", ())]
    if not sites or any(site.get("status") != "ok" or "direct_comparison" not in site for site in sites):
        raise ValueError("summary requires complete successful direct comparisons")
    analyses = [site["direct_comparison"] for site in sites]
    token_count = sum(item["target_token_count"] for item in analyses)
    if token_count <= 0:
        raise ValueError("summary requires at least one target token")
    summary = {
        "samples": len(records),
        "sites": len(sites),
        "target_tokens": token_count,
        "conditions": {},
        "comparisons": {},
    }
    controls = [site["native_clean_control"] for site in sites]
    summary["native_clean_control"] = {
        "first_argmax_changed_count": sum(row["first_argmax_changed"] for row in controls),
        "first_argmax_changed_rate": sum(row["first_argmax_changed"] for row in controls) / len(controls),
        "any_argmax_changed_count": sum(row["any_argmax_changed"] for row in controls),
        "any_argmax_changed_rate": sum(row["any_argmax_changed"] for row in controls) / len(controls),
        "token_argmax_changed_count": sum(row["token_argmax_changed_count"] for row in controls),
        "token_argmax_changed_rate": sum(row["token_argmax_changed_count"] for row in controls) / token_count,
        "mean_gt_logp_delta": sum(row["gt_logp_sum_delta"] for row in controls) / token_count,
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
    for name in ABLATIONS:
        rows = [item["comparisons"][name] for item in analyses]
        first = sum(row["first_argmax_changed"] for row in rows)
        any_changed = sum(row["any_argmax_changed"] for row in rows)
        changed_tokens = sum(row["token_argmax_changed_count"] for row in rows)
        summary["comparisons"][name] = {
            "first_argmax_changed_count": first,
            "first_argmax_changed_rate": first / len(rows),
            "any_argmax_changed_count": any_changed,
            "any_argmax_changed_rate": any_changed / len(rows),
            "token_argmax_changed_count": changed_tokens,
            "token_argmax_changed_rate": changed_tokens / token_count,
            "mean_gt_logp_delta": summary["conditions"][name]["mean_gt_logp"] - clean_mean,
            "mean_first_gt_logp_delta": sum(row["first_gt_logp_delta"] for row in rows) / len(rows),
        }
    if not all(math.isfinite(value)
               for row in summary["conditions"].values()
               for key, value in row.items() if key.startswith("mean_")):
        raise ValueError("summary contains non-finite condition metrics")
    return summary


def _atomic_write_text(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload)
    os.replace(temporary, path)


def write_outputs(output, records, config):
    """Write resumable raw records and compact, denominator-explicit summaries."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_records(records)
    _atomic_write_text(output / "records.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    _atomic_write_text(output / "summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    condition_rows = [dict(condition=name, **summary["conditions"][name]) for name in CONDITIONS]
    handle = io.StringIO(newline="")
    try:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)
        _atomic_write_text(output / "condition_summary.csv", handle.getvalue())
    finally:
        handle.close()
    comparison_rows = [dict(ablation=name, **summary["comparisons"][name]) for name in ABLATIONS]
    handle = io.StringIO(newline="")
    try:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
        _atomic_write_text(output / "comparison_summary.csv", handle.getvalue())
    finally:
        handle.close()
    lines = [
        "# Direct Vision-context ablation at T(i+1)", "",
        f"Status: {config.get('status', 'unknown')}; samples: {summary['samples']}; "
        f"sites: {summary['sites']}; target tokens: {summary['target_tokens']}.", "",
        "The primary clean condition is the native frozen Stage 2 full forward. attention_mask uses "
        "the same full-forward path with only the requested query/key rectangles changed.", "",
        "`first text entropy` is conditional on the documented ordinary-text vocabulary; text mass and latent-START "
        "probability show how much unconditional probability remains on that event.", "",
        "| condition | mean NLL | first GT logp | full H | text-cond. H | text mass | latent START p | first argmax latent START | GT argmax match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONDITIONS:
        row = summary["conditions"][name]
        lines.append(
            f"| {name} | {row['mean_nll']:.8f} | {row['mean_first_gt_logp']:.8f} | "
            f"{row['mean_first_full_entropy_nats']:.8f} | {row['mean_first_text_entropy_nats']:.8f} | "
            f"{row['mean_first_text_mass']:.8g} | {row['mean_first_vision_start_probability']:.8g} | "
            f"{row['first_argmax_vision_rate']:.4%} | "
            f"{row['gt_argmax_match_rate']:.4%} |"
        )
    lines += ["", "| ablation vs clean | first argmax changed | any T(i+1) argmax changed | token argmax changed | mean GT logp delta |",
              "|---|---:|---:|---:|---:|"]
    for name in ABLATIONS:
        row = summary["comparisons"][name]
        lines.append(
            f"| {name} | {row['first_argmax_changed_rate']:.4%} | "
            f"{row['any_argmax_changed_rate']:.4%} | {row['token_argmax_changed_rate']:.4%} | "
            f"{row['mean_gt_logp_delta']:.8f} |"
        )
    control = summary["native_clean_control"]
    lines += ["", "## Clean-logit alignment control", "",
              f"Flat native clean logits versus their per-site slices: first argmax changed "
              f"{control['first_argmax_changed_count']}/{summary['sites']} "
              f"({control['first_argmax_changed_rate']:.4%}); token argmax changed "
              f"{control['token_argmax_changed_count']}/{summary['target_tokens']} "
              f"({control['token_argmax_changed_rate']:.4%}); mean GT logp delta "
              f"{control['mean_gt_logp_delta']:.8f}.", "",
              "attention_mask retains latent START/END and all sequence positions while blocking only latent-body keys. "
              "remove_full_visual_block removes START, body, and END from the effective context and compresses positions.", ""]
    _atomic_write_text(output / "report.md", "\n".join(lines))
    # Completion is published last so readers never observe a complete config
    # with missing or stale derived artifacts.
    _atomic_write_text(
        output / "run_config.json",
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
    )
    return summary

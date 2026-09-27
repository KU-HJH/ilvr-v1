"""Diagnostic Vision-gain probe: eval-mode S_F / S_M / G on fixed teacher-forced blocks.

S_F uses the normal forward; S_M uses an independent full-replay forward with the
block's latent slots hidden from queries at/after its END (``stage2_ablation``).
Both are mean full-vocabulary target log-probabilities over the block's post
window. The probe is diagnostic only (pre-registration: never used for
checkpoint selection, tuning, or early stopping).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .objective import target_log_likelihood
from .objective_runtime import model_inputs
from .objective_windows import select_objective_windows


def probe_batches(model, batches, *, token_ids, latent_size, text_vocabulary,
                  post_window=1, pre_window=1, autocast_dtype=None):
    if model.training:
        raise ValueError("the probe requires an eval mode model")
    rows = []
    device = next(model.parameters()).device
    for batch_index, cpu_inputs in enumerate(batches):
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in cpu_inputs.items()}
        selection = select_objective_windows(
            inputs["input_ids"].cpu(), inputs["labels"].cpu(), inputs["attention_mask"].cpu(),
            inputs["stage2_text_segments"], token_ids=token_ids, latent_size=latent_size,
            text_vocabulary=text_vocabulary, pre_window=pre_window, post_window=post_window,
        )
        if not selection.blocks:
            continue
        clean = {key: value for key, value in model_inputs(inputs).items() if key != "labels"}
        context = (torch.autocast(device.type, dtype=autocast_dtype)
                   if autocast_dtype is not None else torch.autocast(device.type, enabled=False))
        with torch.no_grad(), context:
            clean_logits = model(**clean).logits
            for block in selection.blocks:
                targets = torch.tensor(block.target_ids, device=device)
                s_f = target_log_likelihood(
                    clean_logits[block.batch_index, list(block.post_positions)], targets).mean()
                masked = model(**clean, stage2_ablation=(block.start_position, block.end_position))
                s_m = target_log_likelihood(
                    masked.logits[block.batch_index, list(block.post_positions)], targets).mean()
                del masked
                rows.append({"batch": batch_index, "block": block.block_index,
                             "S_F": float(s_f), "S_M": float(s_m), "G": float(s_f - s_m)})
        del clean_logits
    if not rows:
        raise ValueError("no valid probe blocks")
    count = len(rows)
    s_f = sum(row["S_F"] for row in rows) / count
    s_m = sum(row["S_M"] for row in rows) / count
    return {
        "blocks": count, "S_F": s_f, "S_M": s_m, "G_mean": s_f - s_m,
        "G_gt_0": sum(row["G"] > 0 for row in rows) / count,
        "s_m_mode": "full_replay", "model_mode": "eval",
        "post_window": post_window, "pre_window": pre_window, "per_block": rows,
    }


def probe_token_batches(model, batches, *, selection_table, scope, token_ids, latent_size,
                        autocast_dtype=None):
    """Token-level probe: step0-selected targets, M hiding per ``scope`` (full replay)."""
    from .entropy_selection import build_vision_token_units

    if model.training:
        raise ValueError("the probe requires an eval mode model")
    rows = []
    excluded = 0
    device = next(model.parameters()).device
    for batch_index, cpu_inputs in enumerate(batches):
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in cpu_inputs.items()}
        selection = build_vision_token_units(
            inputs["input_ids"].cpu(), inputs["labels"].cpu(), inputs["attention_mask"].cpu(),
            inputs["stage2_text_segments"], token_ids=token_ids, latent_size=latent_size,
            selected_targets=selection_table.selected_targets(inputs["input_ids"][0]), scope=scope,
        )
        excluded += selection.excluded.get("pre_first_block_tokens", 0)
        if not selection.blocks:
            continue
        clean = {key: value for key, value in model_inputs(inputs).items() if key != "labels"}
        context = (torch.autocast(device.type, dtype=autocast_dtype)
                   if autocast_dtype is not None else torch.autocast(device.type, enabled=False))
        with torch.no_grad(), context:
            clean_logits = model(**clean).logits
            for block in selection.blocks:
                targets = torch.tensor(block.target_ids, device=device)
                s_f = target_log_likelihood(clean_logits[0, list(block.post_positions)], targets)
                masked = model(**clean, stage2_ablation=block.ablation)
                s_m = target_log_likelihood(masked.logits[0, list(block.post_positions)], targets)
                del masked
                for position, f_value, m_value in zip(block.post_positions, s_f.tolist(), s_m.tolist()):
                    rows.append({"batch": batch_index, "block": block.block_index, "target": position + 1,
                                 "S_F": f_value, "S_M": m_value, "G": f_value - m_value})
        del clean_logits
    if not rows:
        raise ValueError("no valid probe tokens")
    count = len(rows)
    s_f = sum(row["S_F"] for row in rows) / count
    s_m = sum(row["S_M"] for row in rows) / count
    return {
        "unit": "token", "tokens": count, "excluded_pre_first_block_tokens": excluded,
        "S_F": s_f, "S_M": s_m, "G_mean": s_f - s_m,
        "G_gt_0": sum(row["G"] > 0 for row in rows) / count,
        "mask_scope": scope, "s_m_mode": "full_replay", "model_mode": "eval", "per_token": rows,
    }


def _load(kind, checkpoint, base_model, device, dtype):
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from peft_common import resolve_qwen2_5_vl_model
    from .checkpointing import configure_stage2_metadata, resolve_latent_token_ids
    from .peft import (ALLOWED_STAGE2_TARGET_SUFFIXES, Stage2TuningConfig,
                       load_stage1_dora_for_fresh_stage2, load_stage2_model_for_inference)

    processor = AutoProcessor.from_pretrained(checkpoint)
    if kind == "stage1":
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, attn_implementation="sdpa", torch_dtype=dtype)
        model = load_stage1_dora_for_fresh_stage2(
            base, processor, checkpoint,
            Stage2TuningConfig("dora", 32, 64, 0.05, ALLOWED_STAGE2_TARGET_SUFFIXES),
            requested_base_model_reference=str(base_model),
        )
    else:
        model = load_stage2_model_for_inference(
            base_model_path=str(base_model), stage2_artifact_path=str(checkpoint),
            processor=processor,
            model_load_kwargs={"attn_implementation": "sdpa", "torch_dtype": dtype},
        )
    qwen = resolve_qwen2_5_vl_model(model)
    token_ids = resolve_latent_token_ids(processor.tokenizer, qwen.config)
    configure_stage2_metadata(qwen.config, token_ids, latent_size=8, supervise_latent_boundaries=True)
    qwen.config.use_cache = False
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model, processor, token_ids, AutoConfig


def main() -> int:
    from .data import Stage2DataCollator, load_jsonl_examples, prepare_stage2_example
    from .objective_windows import build_text_vocabulary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("data/TEST.jsonl"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="step0 selection file: token-level probe")
    parser.add_argument("--mask-scope", choices=("independent", "cumulative"))
    args = parser.parse_args()
    if (args.selection is None) != (args.mask_scope is None):
        parser.error("--selection and --mask-scope go together")

    manifest = json.loads(args.manifest.read_text())
    raw = args.test.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["test_jsonl_sha256"]:
        raise ValueError("TEST.jsonl does not match the probe manifest")
    samples = load_jsonl_examples(args.test)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model, processor, token_ids, _ = _load(args.kind, args.checkpoint, args.base_model, device, dtype)
    from peft_common import resolve_qwen2_5_vl_model
    qwen = resolve_qwen2_5_vl_model(model)
    vocabulary = build_text_vocabulary(processor.tokenizer, qwen.config, qwen.config.vocab_size)
    collator = Stage2DataCollator(processor, token_ids, 8, True, enable_stage2_objective=True)
    batches = [collator([prepare_stage2_example(samples[row], image_root=args.image_root, latent_size=8)])
               for row in manifest["selected_row_ids"]]
    if args.selection is not None:
        from .entropy_selection import VisionTokenSelectionTable

        table = VisionTokenSelectionTable.load(args.selection)
        report = probe_token_batches(model, batches, selection_table=table, scope=args.mask_scope,
                                     token_ids=token_ids, latent_size=8, autocast_dtype=dtype)
        report.update({"selection": str(args.selection), "selection_sha256": table.sha256})
    else:
        report = probe_batches(model, batches, token_ids=token_ids, latent_size=8,
                               text_vocabulary=vocabulary, autocast_dtype=dtype)
    report.update({"label": args.label, "kind": args.kind, "checkpoint": str(args.checkpoint),
                   "base_model": str(args.base_model), "manifest": str(args.manifest),
                   "test_jsonl_sha256": manifest["test_jsonl_sha256"],
                   "rows": len(manifest["selected_row_ids"]), "dtype": "bfloat16",
                   "language_attention": "sdpa", "vision_attention": "sdpa"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("label", "blocks", "tokens", "S_F", "S_M", "G_mean", "G_gt_0")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

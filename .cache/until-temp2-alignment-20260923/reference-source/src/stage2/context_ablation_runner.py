"""Run paired clean, latent-body-mask, and full-visual-removal analysis."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from .context_ablation_analysis import build_site_comparisons, write_outputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--reference-run", required=True,
                        help="Completed historical removal run fixing TEST sample IDs and boundaries")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_witness(root, relative_paths):
    root = Path(root)
    result = {}
    for relative in sorted(set(relative_paths)):
        path = root / relative
        stat = path.stat()
        result[str(relative)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def _build_run_identity(*, checkpoint, base_model, data_path, image_root,
                        reference_run, indices, sample_ids, seed,
                        provenance_kind="pre_run_immutable"):
    repo = Path(__file__).resolve().parents[2]
    source_files = (
        "scripts/analyze_context_ablation.py",
        "src/stage2/context_ablation_analysis.py",
        "src/stage2/context_ablation_runner.py",
        "src/stage2/checkpointing.py",
        "src/stage2/data.py",
        "src/stage2/dora.py",
        "src/stage2/peft.py",
        "src/stage2/trainer.py",
        "src/step_preference/ablation.py",
        "src/step_preference/packing.py",
        "src/step_preference/spans.py",
        "transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
        "transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py",
        "transformers/src/transformers/models/qwen2_5_vl/stage2_prefix_cache.py",
    )
    checkpoint_witnesses = (
        "stage2_peft_metadata.json", "adapter_config.json",
        "adapter_model.safetensors", "ilvr_token_rows.safetensors",
    )
    base_manifests = (
        "config.json", "model.safetensors.index.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json",
    )
    data_rows = _read_jsonl(data_path)
    image_paths = []
    for index in indices:
        row = data_rows[index]
        image_paths.extend(row.get("image_input", ()))
        image_paths.extend(
            step["helper_image"] for step in row.get("sequence_plan", ())
            if step.get("helper_image")
        )
    base_weight_paths = [path.name for path in Path(base_model).glob("*.safetensors")]
    return {
        "schema_version": 2,
        "provenance_kind": provenance_kind,
        "checkpoint": str(checkpoint),
        "base_model_dir": str(base_model),
        "data": str(data_path),
        "data_sha256": _sha256_file(data_path),
        "image_root": str(image_root),
        "reference_run": str(reference_run),
        "reference_config_sha256": _sha256_file(reference_run / "run_config.json"),
        "reference_records_sha256": _sha256_file(reference_run / "records.jsonl"),
        "checkpoint_witness_sha256": {
            name: _sha256_file(checkpoint / name) for name in checkpoint_witnesses
        },
        "base_model_manifest_sha256": {
            name: _sha256_file(base_model / name) for name in base_manifests
        },
        "base_model_weight_stat_witness": _stat_witness(base_model, base_weight_paths),
        "selected_image_stat_witness": _stat_witness(image_root, image_paths),
        "sample_indices": list(indices),
        "sample_ids": list(sample_ids),
        "seed": seed,
        "conditions": ["clean", "attention_mask", "remove_full_visual_block"],
        "target_scope": "entire immediate GT text item after each Vision block",
        "source_sha256": {name: _sha256_file(repo / name) for name in source_files},
    }


def _validated_resume_records(output, identity):
    output = Path(output)
    previous = json.loads((output / "run_config.json").read_text())
    if previous.get("status") == "complete":
        raise ValueError("completed output must not be resumed")
    stored_identity = json.loads((output / "run_identity.json").read_text())
    if stored_identity != identity:
        raise ValueError("resume run identity differs from the immutable original")
    records = _read_jsonl(output / "records.jsonl")
    identities = [(record.get("data_index"), record.get("sample_id")) for record in records]
    expected = list(zip(identity["sample_indices"], identity["sample_ids"]))[:len(records)]
    if (not records or len(records) >= len(identity["sample_indices"]) or identities != expected
            or any(record.get("status") != "ok" or not record.get("sites") for record in records)):
        raise ValueError("resume records must be an exact sample prefix of successful outputs")
    return records, previous


def _validate_sites_against_reference(sites, reference_sites):
    if len(sites) != len(reference_sites):
        raise ValueError("Vision boundary count differs from reference")
    for site, reference in zip(sites, reference_sites, strict=True):
        removal = reference.get("vision_removal_entropy", {})
        full_start, full_end = site.get("full_visual_interval", (None, None))
        target_start, _ = site.get("target_text_interval", (None, None))
        if (
            site.get("status") != "ok"
            or site.get("step_id") != reference.get("gt_step")
            or full_start != reference.get("target_position")
            or full_end != removal.get("normal_target_position")
            or target_start != removal.get("normal_target_position")
            or full_end - full_start != removal.get("removed_token_count")
            or not site.get("target_token_ids")
            or site["target_token_ids"][0] != removal.get("gt_next_text_token_id")
        ):
            raise ValueError("Vision boundary structure differs from reference")
        payload = json.dumps(site, allow_nan=False)
        if not payload:
            raise ValueError("site metrics must be finite and serializable")


def _command():
    env = " ".join(
        f"{name}={shlex.quote(os.environ[name])}"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        if name in os.environ
    )
    return (env + " " if env else "") + 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/transformers/src:$PWD/src" ' + shlex.join([sys.executable, *sys.argv])


def _build_combined_visual_mask(attention_mask, items, dtype):
    """Apply each site's V-body mask only to that site's GT predictor rows."""
    from step_preference.ablation import (
        apply_step_ablation,
        build_inverted_causal_mask,
        build_step_ablation_spec,
    )

    result = build_inverted_causal_mask(attention_mask, dtype)
    for item in items:
        positions = item.chosen_predictor_positions
        if positions.ndim != 1 or not positions.numel():
            raise ValueError("chosen predictor positions must be a nonempty vector")
        expected = torch.arange(
            int(positions[0]), int(positions[-1]) + 1, device=positions.device
        )
        if not torch.equal(positions, expected):
            raise ValueError("chosen predictor positions must be contiguous")
        spec = replace(
            build_step_ablation_spec(item.span, "visual"),
            query_start=int(positions[0]),
            query_end=int(positions[-1]) + 1,
        )
        result = apply_step_ablation(result, spec)
    return result


def _remove_visual_block(model_inputs, item):
    """Delete one START/body/END interval and remap its immediate-text scores."""
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError("visual removal requires one aligned input and attention row")
    span = item.span
    start, end = span.prev_text_end, span.target_text_start
    if start < 1 or span.latent_start != start + 1 or span.latent_end != end - 1:
        raise ValueError("full visual removal requires complete START/body/END boundaries")
    result = dict(model_inputs)
    for key in ("input_ids", "attention_mask"):
        value = model_inputs[key]
        result[key] = torch.cat((value[:, :start], value[:, end:]), dim=1)
    prepared = model_inputs.get("prepared_base_inputs_embeds")
    if prepared is not None:
        if prepared.ndim != 3 or prepared.shape[:2] != input_ids.shape:
            raise ValueError("prepared embeddings must align with the clean token sequence")
        result["prepared_base_inputs_embeds"] = torch.cat(
            (prepared[:, :start], prepared[:, end:]), dim=1
        )
    shift = end - start
    positions = item.chosen_predictor_positions.to(input_ids.device) - shift
    if positions.ndim != 1 or not positions.numel() or int(positions[0]) < 0:
        raise ValueError("removed predictor positions must be a nonempty in-bounds vector")
    score_mask = torch.zeros_like(result["input_ids"], dtype=torch.bool)
    score_mask[0, positions] = True
    if int(score_mask.sum()) != item.chosen_target_ids.numel():
        raise ValueError("removed score positions must align with every target")
    return result, score_mask


def _forward_prepared(qwen, model_inputs, score_mask, final_attention_mask=None):
    """Run frozen Stage 2 text/latent inference with one reused image embedding pass."""
    prepared = model_inputs.get("prepared_base_inputs_embeds")
    if prepared is None:
        raise ValueError("prepared embeddings are required")
    return qwen._forward_stage2(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        position_ids=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=model_inputs.get("image_grid_thw"),
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        return_stage2_latent_states=False,
        collect_stage2_diagnostics=False,
        retain_stage2_latent_grads=False,
        stage2_loss_only=True,
        stage2_final_attention_mask=final_attention_mask,
        stage2_score_token_mask=score_mask,
        prepared_base_inputs_embeds=prepared,
    )


def _split_selected_logits(items, logits):
    """Map a flat selected-logit row to stable per-step visual branch keys."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("selected logits must have shape [1, target_tokens, vocabulary]")
    cursor, result = 0, {}
    for item in sorted(items, key=lambda value: value.span.target_text_start):
        count = item.chosen_target_ids.numel()
        key = f"{item.sample_id}:{item.span.step_id}:visual:chosen_entropy"
        result[key] = logits[0, cursor:cursor + count]
        cursor += count
    if cursor != logits.shape[1]:
        raise ValueError("selected logits do not align with preference items")
    return result


def run(args):
    from transformers import AutoProcessor
    from stage2.checkpointing import resolve_latent_token_ids
    from stage2.data import Stage2DataCollator, load_jsonl_examples, prepare_stage2_example
    from stage2.peft import load_stage2_model_for_inference, resolve_stage2_qwen_model
    from stage2.trainer import Stage2Trainer
    from step_preference.spans import stable_sample_id
    import transformers

    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    output = Path(args.output_dir).resolve()
    output_nonempty = output.exists() and any(output.iterdir())
    if output_nonempty and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if args.resume and not output_nonempty:
        raise FileNotFoundError(f"resume output directory is empty or missing: {output}")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint).resolve()
    base_model = Path(args.base_model_dir).resolve()
    data_path = Path(args.data).resolve()
    image_root = Path(args.image_root).resolve()
    reference_run = Path(args.reference_run).resolve()
    reference_config = json.loads((reference_run / "run_config.json").read_text())
    reference_records = _read_jsonl(reference_run / "records.jsonl")
    if reference_config.get("status") != "complete":
        raise ValueError("reference run must be complete")
    if Path(reference_config["checkpoint"]).resolve() != checkpoint:
        raise ValueError("reference run checkpoint differs")
    if Path(reference_config["data"]).resolve() != data_path:
        raise ValueError("reference run data differs")
    if reference_config.get("seed") != args.seed:
        raise ValueError("reference run seed differs")
    available_indices = reference_config["sample_indices"]
    if len(available_indices) < args.max_samples:
        raise ValueError("reference sample selection is incomplete")
    indices = available_indices[:args.max_samples]
    if len(set(indices)) != len(indices):
        raise ValueError("reference sample selection contains duplicates")
    if any(record.get("status") != "ok" for record in reference_records):
        raise ValueError("reference records must all be successful")
    reference_by_index = {record["data_index"]: record for record in reference_records}
    if len(reference_by_index) != len(reference_records):
        raise ValueError("reference records contain duplicate data indices")
    if any(index not in reference_by_index for index in indices):
        raise ValueError("reference records are missing selected samples")

    examples = load_jsonl_examples(data_path)
    expected_ids = [reference_by_index[index]["sample_id"] for index in indices]
    if [stable_sample_id(examples[index]) for index in indices] != expected_ids:
        raise ValueError("reference sample identities differ from data")
    identity = _build_run_identity(
        checkpoint=checkpoint, base_model=base_model, data_path=data_path,
        image_root=image_root, reference_run=reference_run, indices=indices,
        sample_ids=expected_ids, seed=args.seed,
    )
    records, previous_config = ([], None)
    if args.resume:
        records, previous_config = _validated_resume_records(output, identity)
        for record in records:
            _validate_sites_against_reference(
                record["sites"], reference_by_index[record["data_index"]]["sites"]
            )
    else:
        (output / "run_identity.json").write_text(
            json.dumps(identity, indent=2, ensure_ascii=False) + "\n"
        )

    config = {
        "status": "loading",
        "checkpoint": str(checkpoint),
        "base_model_dir": str(base_model),
        "data": str(data_path),
        "image_root": str(image_root),
        "reference_run": str(reference_run),
        "sample_indices": indices,
        "sample_ids": expected_ids,
        "seed": args.seed,
        "conditions": ["clean", "attention_mask", "remove_full_visual_block"],
        "clean_definition": "native frozen Stage 2 full forward on the unmodified sequence",
        "native_clean_role": "primary clean condition",
        "attention_mask_definition": "block current latent-body keys; retain latent START/END and all sequence positions",
        "removal_definition": "remove current latent START/body/END from effective context and compress positions",
        "target_scope": "entire immediate GT text item after each Vision block",
        "model_frozen": True,
        "optimizer_updates": 0,
        "command": _command(),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "transformers_import": transformers.__file__,
        "smoke_checks": [],
    }
    if previous_config is not None:
        config.update(
            smoke_checks=previous_config.get("smoke_checks", []),
            resumed_record_count=len(records),
            resumed_previous_status=previous_config.get("status"),
            resumed_previous_error_type=previous_config.get("error_type"),
        )
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.perf_counter()
    try:
        torch.manual_seed(args.seed)
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        model = load_stage2_model_for_inference(
            base_model,
            checkpoint,
            processor,
            source_provenance_kwargs={"local_files_only": True},
            model_load_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "cuda:0",
                "attn_implementation": "sdpa",
                "stage2_split_attention": True,
            },
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        qwen = resolve_stage2_qwen_model(model)
        qwen.config.stage2_latent_activation_checkpointing = False
        device = next(model.parameters()).device
        token_ids = resolve_latent_token_ids(processor.tokenizer, qwen.config)
        collator = Stage2DataCollator(
            processor,
            token_ids,
            qwen.config.latent_size,
            bool(getattr(qwen.config, "supervise_latent_boundaries", False)),
            context_preference_enabled=True,
            target_scope="immediate_text",
        )
        control_ids = set(processor.tokenizer.all_special_ids) | set(
            processor.tokenizer.get_added_vocab().values()
        )
        config.update(
            status="running",
            dtype=str(next(model.parameters()).dtype),
            device=str(device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_name=torch.cuda.get_device_name(device),
            attention_implementation=qwen.config._attn_implementation,
            vision_attention_implementation="flash_attention_2",
            intervention_execution="inference full-forward; one combined mask pass and one compressed pass per site",
            image_embedding_reuse="one frozen vision-encoder pass per sample",
            latent_size=qwen.config.latent_size,
            latent_start_token_id=token_ids.start_token_id,
            load_seconds=time.perf_counter() - started,
        )
        (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

        for ordinal, index in enumerate(indices[len(records):], start=len(records)):
            sample_started = time.perf_counter()
            example = prepare_stage2_example(examples[index], image_root=image_root,
                                              latent_size=qwen.config.latent_size)
            batch = collator([example])
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            model_inputs = {key: value for key, value in batch.items()
                            if isinstance(value, torch.Tensor) and key != "labels"}
            items = tuple(replace(
                item,
                chosen_predictor_positions=item.chosen_predictor_positions.to(device),
                chosen_target_ids=item.chosen_target_ids.to(device),
            ) for item in batch["step_preference_items"])
            if not items:
                raise ValueError(f"selected sample has no immediate post-Vision text targets: {index}")
            score_mask = Stage2Trainer._chosen_score_mask(model_inputs, items, stage2_debugging=True)
            mask_dtype = next(model.parameters()).dtype
            combined_mask = _build_combined_visual_mask(
                model_inputs["attention_mask"], items, mask_dtype
            )
            with torch.inference_mode():
                base_inputs_embeds = qwen.prepare_stage2_base_inputs_embeds(
                    input_ids=model_inputs["input_ids"],
                    pixel_values=model_inputs.get("pixel_values"),
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                )
                prepared_inputs = dict(
                    model_inputs, prepared_base_inputs_embeds=base_inputs_embeds
                )
                outputs = _forward_prepared(qwen, prepared_inputs, score_mask)
                native_clean_logits = outputs.stage2_scored_logits
                clean_logits = _split_selected_logits(items, native_clean_logits)
                del outputs
                masked = _forward_prepared(
                    qwen, prepared_inputs, score_mask,
                    final_attention_mask=combined_mask,
                )
                masked_logits = _split_selected_logits(items, masked.stage2_scored_logits)
                del masked
                removed_logits = {}
                for item in items:
                    removed_inputs, removed_score_mask = _remove_visual_block(prepared_inputs, item)
                    removed = _forward_prepared(qwen, removed_inputs, removed_score_mask)
                    key = f"{item.sample_id}:{item.span.step_id}:visual:chosen_entropy"
                    removed_logits[key] = removed.stage2_scored_logits[0]
                    del removed, removed_inputs, removed_score_mask
            excluded = control_ids | set(range(len(processor.tokenizer), native_clean_logits.shape[-1]))
            sites = build_site_comparisons(
                items=items,
                native_clean_logits=native_clean_logits,
                clean_logits_by_key=clean_logits,
                masked_logits_by_key=masked_logits,
                removed_logits_by_key=removed_logits,
                excluded_output_ids=excluded,
                vision_start_id=token_ids.start_token_id,
            )
            reference = reference_by_index[index]
            _validate_sites_against_reference(sites, reference["sites"])
            if args.smoke_check and not config["smoke_checks"]:
                stored = reference["sites"][0]["vision_removal_entropy"]["normal"]["argmax_id"]
                native = int(native_clean_logits[0, 0].argmax())
                if native != stored:
                    raise AssertionError("native clean first-token argmax differs from completed reference")
                config["smoke_checks"].append({
                    "data_index": index,
                    "stored_normal_argmax_id": stored,
                    "native_clean_argmax_id": native,
                    "passed": True,
                })
            records.append({
                "data_index": index,
                "sample_id": example.sample_id,
                "status": "ok",
                "reference_site_count": len(reference["sites"]),
                "sites": sites,
                "elapsed_seconds": time.perf_counter() - sample_started,
            })
            config["elapsed_seconds"] = time.perf_counter() - started
            write_outputs(output, records, config)
            print(json.dumps({
                "progress": f"{ordinal + 1}/{len(indices)}",
                "data_index": index,
                "sites": len(sites),
                "target_tokens": sum(site["target_token_count"] for site in sites),
                "seconds": records[-1]["elapsed_seconds"],
            }), flush=True)
            del (
                native_clean_logits, clean_logits, masked_logits, removed_logits,
                batch, model_inputs, prepared_inputs, base_inputs_embeds, items,
                score_mask, combined_mask,
            )
            gc.collect()
            torch.cuda.empty_cache()
        expected_sites = sum(len(reference_by_index[index]["sites"]) for index in indices)
        actual_sites = sum(len(record["sites"]) for record in records)
        if actual_sites != expected_sites:
            raise ValueError(f"direct comparison site count {actual_sites} != reference {expected_sites}")
        config.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                      total_sites=actual_sites,
                      total_target_tokens=sum(site["target_token_count"] for record in records for site in record["sites"]))
        return write_outputs(output, records, config)
    except BaseException as error:
        config.update(status="error", error_type=type(error).__name__, error_message=str(error),
                      elapsed_seconds=time.perf_counter() - started)
        (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        if records:
            write_outputs(output, records, config)
        raise


def main():
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()

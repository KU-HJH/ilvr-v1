"""Run the seven-condition factorial Vision-context baseline analysis."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from .context_ablation_analysis import _atomic_write_text
from .context_factorial import (
    CONDITIONS,
    FactorialBranchBatch,
    build_deleted_branches,
    build_length_preserving_branches,
    compare_factorial_logits,
    native_replay_control,
    summarize_native_controls,
    write_direct_reference_controls,
    write_factorial_outputs,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--boundary-reference-run", required=True)
    parser.add_argument("--direct-reference-run", required=True)
    parser.add_argument(
        "--reference-role", choices=("same_model", "structure_only"), default="same_model",
        help="structure_only reuses reference samples/targets, never their prediction values",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-check", action="store_true")
    return parser


def _command():
    environment = " ".join(
        f"{name}={shlex.quote(os.environ[name])}"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        if name in os.environ
    )
    return (environment + " " if environment else "") + shlex.join([sys.executable, *sys.argv])


def _project_branch_logits(hidden_states, lm_head, predictor_positions):
    if hidden_states.ndim != 3 or hidden_states.shape[0] != len(predictor_positions):
        raise ValueError("hidden states must align with branch predictor positions")
    return tuple(
        lm_head(hidden_states[index, positions.to(hidden_states.device)])
        for index, positions in enumerate(predictor_positions)
    )


def _decode_branches(qwen, branches, position_ids):
    outputs = qwen.model(
        inputs_embeds=branches.inputs_embeds,
        attention_mask=branches.attention_mask,
        position_ids=position_ids,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        generate_mode=True,
    )
    return _project_branch_logits(
        outputs.last_hidden_state, qwen.lm_head, branches.predictor_positions
    )


def _clean_branches(input_ids, attention_mask, final_embeds, items):
    from step_preference.ablation import build_inverted_causal_mask

    count = len(items)
    causal = build_inverted_causal_mask(attention_mask, final_embeds.dtype)
    return FactorialBranchBatch(
        input_ids.expand(count, -1).clone(),
        final_embeds.expand(count, -1, -1).clone(),
        causal.expand(count, -1, -1, -1).clone(),
        tuple(item.chosen_predictor_positions.to(input_ids.device) for item in items),
        tuple(item.chosen_target_ids.to(input_ids.device) for item in items),
    )


def _deleted_position_ids(qwen, branches, image_grid_thw):
    if branches.attention_mask.ndim != 2:
        raise ValueError("deleted RoPE positions require a 2D binary attention mask")
    positions = []
    for index in range(branches.input_ids.shape[0]):
        current, _ = qwen.get_rope_index(
            branches.input_ids[index:index + 1],
            image_grid_thw=image_grid_thw,
            attention_mask=branches.attention_mask[index:index + 1],
        )
        positions.append(current)
    return torch.cat(positions, dim=1)


def _prepare_deleted_decoder_batch(qwen, branches, image_grid_thw):
    from step_preference.ablation import build_inverted_causal_mask

    position_ids = _deleted_position_ids(qwen, branches, image_grid_thw)
    causal = build_inverted_causal_mask(
        branches.attention_mask, branches.inputs_embeds.dtype
    )
    prepared = FactorialBranchBatch(
        branches.input_ids,
        branches.inputs_embeds,
        causal,
        branches.predictor_positions,
        branches.target_ids,
    )
    return prepared, position_ids


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _validate_direct_site_alignment(sites, direct_sites):
    if len(sites) != len(direct_sites):
        raise ValueError("direct reference site alignment differs")
    for site, reference in zip(sites, direct_sites, strict=True):
        if (
            site.get("status") != "ok"
            or reference.get("status") != "ok"
            or site.get("step_id") != reference.get("step_id")
            or site.get("target_text_interval") != reference.get("target_text_interval")
            or site.get("target_token_count") != reference.get("target_token_count")
            or site.get("target_token_ids") != reference.get("target_token_ids")
        ):
            raise ValueError("direct reference site alignment differs")


def _validate_reference_config(*, label, config, checkpoint, data_path, seed, role):
    if role not in {"same_model", "structure_only"}:
        raise ValueError("unknown reference role")
    if config.get("status") != "complete":
        raise ValueError(f"{label} reference run must be complete")
    if role == "same_model" and Path(config["checkpoint"]).resolve() != checkpoint:
        raise ValueError(f"{label} reference checkpoint differs")
    if Path(config["data"]).resolve() != data_path or config.get("seed") != seed:
        raise ValueError(f"{label} reference data or seed differs")


def _smoke_result(role, index, sites, direct_sites):
    # Called only after every site has passed structural and target-token validation.
    if role == "structure_only":
        return {
            "data_index": index, "reference_role": role,
            "all_site_targets_aligned": True,
            "native_replay_first_argmax_changed_count": sum(
                site["native_clean_control"]["first_argmax_changed"] for site in sites
            ),
        }
    previous = direct_sites[0]["direct_comparison"]["conditions"]
    actual = sites[0]["factorial_comparison"]["conditions"]
    smoke = {"data_index": index, "reference_role": role}
    for current, old in (("clean", "clean"), ("body_mask", "attention_mask"),
                         ("full_delete", "remove_full_visual_block")):
        smoke[f"{current}_first_argmax_matches"] = (
            actual[current]["first_argmax_id"] == previous[old]["first_argmax_id"]
        )
    if not all(value for key, value in smoke.items() if key.endswith("matches")):
        raise AssertionError("factorial smoke disagrees with completed direct reference")
    return smoke


def run(args):
    from transformers import AutoProcessor
    import transformers

    from stage2.checkpointing import resolve_latent_token_ids
    from stage2.context_ablation_runner import (
        _build_run_identity,
        _forward_prepared,
        _sha256_file,
        _validate_sites_against_reference,
    )
    from stage2.data import Stage2DataCollator, load_jsonl_examples, prepare_stage2_example
    from stage2.peft import load_stage2_model_for_inference, resolve_stage2_qwen_model
    from stage2.trainer import Stage2Trainer
    from step_preference.spans import stable_sample_id

    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    base_model = Path(args.base_model_dir).resolve()
    data_path = Path(args.data).resolve()
    image_root = Path(args.image_root).resolve()
    boundary_reference = Path(args.boundary_reference_run).resolve()
    direct_reference = Path(args.direct_reference_run).resolve()

    boundary_config = json.loads((boundary_reference / "run_config.json").read_text())
    boundary_records = _read_jsonl(boundary_reference / "records.jsonl")
    direct_config = json.loads((direct_reference / "run_config.json").read_text())
    direct_records = _read_jsonl(direct_reference / "records.jsonl")
    for label, config in (("boundary", boundary_config), ("direct", direct_config)):
        _validate_reference_config(
            label=label, config=config, checkpoint=checkpoint, data_path=data_path,
            seed=args.seed, role=args.reference_role,
        )
    available = boundary_config["sample_indices"]
    if len(available) < args.max_samples:
        raise ValueError("boundary reference has fewer samples than requested")
    indices = available[:args.max_samples]
    if direct_config["sample_indices"][:args.max_samples] != indices or len(set(indices)) != len(indices):
        raise ValueError("reference sample selections differ or contain duplicates")
    boundary_by_index = {record["data_index"]: record for record in boundary_records}
    direct_by_index = {record["data_index"]: record for record in direct_records}
    if (
        len(boundary_by_index) != len(boundary_records)
        or len(direct_by_index) != len(direct_records)
        or any(record.get("status") != "ok" for record in boundary_records)
        or any(record.get("status") != "ok" for record in direct_records)
    ):
        raise ValueError("reference records must be successful and uniquely indexed")
    if any(index not in boundary_by_index or index not in direct_by_index for index in indices):
        raise ValueError("reference records are missing selected samples")

    examples = load_jsonl_examples(data_path)
    sample_ids = [boundary_by_index[index]["sample_id"] for index in indices]
    if [stable_sample_id(examples[index]) for index in indices] != sample_ids:
        raise ValueError("reference sample identities differ from TEST data")
    if [direct_by_index[index]["sample_id"] for index in indices] != sample_ids:
        raise ValueError("direct reference sample identities differ")

    identity = _build_run_identity(
        checkpoint=checkpoint,
        base_model=base_model,
        data_path=data_path,
        image_root=image_root,
        reference_run=boundary_reference,
        indices=indices,
        sample_ids=sample_ids,
        seed=args.seed,
    )
    repo = Path(__file__).resolve().parents[2]
    extra_sources = (
        "scripts/analyze_context_factorial.py",
        "src/stage2/context_factorial.py",
        "src/stage2/context_factorial_runner.py",
    )
    identity.update(
        analysis="seven_condition_body_boundary_full_factorial",
        conditions=list(CONDITIONS),
        reference_role=args.reference_role,
        direct_reference_run=str(direct_reference),
        direct_reference_config_sha256=_sha256_file(direct_reference / "run_config.json"),
        direct_reference_records_sha256=_sha256_file(direct_reference / "records.jsonl"),
    )
    identity["source_sha256"].update(
        {name: _sha256_file(repo / name) for name in extra_sources}
    )
    training_metadata_path = checkpoint / "step_preference_metadata.json"
    training_metadata = None
    if training_metadata_path.is_file():
        training_metadata = json.loads(training_metadata_path.read_text())
        identity["checkpoint_witness_sha256"]["step_preference_metadata.json"] = (
            _sha256_file(training_metadata_path)
        )
    _atomic_write_text(output / "run_identity.json", json.dumps(identity, indent=2) + "\n")

    config = {
        "status": "loading",
        "checkpoint": str(checkpoint),
        "base_model_dir": str(base_model),
        "data": str(data_path),
        "image_root": str(image_root),
        "boundary_reference_run": str(boundary_reference),
        "direct_reference_run": str(direct_reference),
        "reference_role": args.reference_role,
        "reference_checkpoint": direct_config["checkpoint"],
        "training_preference_metadata": training_metadata,
        "training_target_scope": (
            training_metadata.get("target_scope", "until_next_latent")
            if training_metadata else None
        ),
        "sample_indices": indices,
        "sample_ids": sample_ids,
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "condition_definitions": {
            "clean": "START/body/END unchanged",
            "body_mask": "block eight body K/V positions; retain length and boundaries",
            "body_delete": "delete eight body positions; retain adjacent START/END; compress positions by 8",
            "boundary_mask": "replace START/END inputs with latent-pad embedding and block their K/V; retain body and length",
            "boundary_delete": "delete START/END; retain clean body embeddings; compress positions by 2",
            "full_mask": "replace all ten block inputs with latent-pad embedding and block all block K/V; retain length",
            "full_delete": "delete START/body/END; compress positions by 10",
        },
        "target_scope": "entire immediate GT text item after each Vision block",
        "primary_clean": "unmodified clean final-decoder replay batched per site",
        "native_clean_role": "numerical control and completed-reference alignment",
        "model_frozen": True,
        "optimizer_updates": 0,
        "negative_generation": False,
        "command": _command(),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "transformers_import": transformers.__file__,
        "smoke_checks": [],
    }
    _atomic_write_text(output / "run_config.json", json.dumps(config, indent=2) + "\n")
    records = []
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
            gpu_name=torch.cuda.get_device_name(device),
            text_attention_implementation=qwen.config._attn_implementation,
            vision_attention_implementation="flash_attention_2",
            latent_size=qwen.config.latent_size,
            latent_start_token_id=token_ids.start_token_id,
            latent_pad_token_id=token_ids.pad_token_id,
            load_seconds=time.perf_counter() - started,
        )
        _atomic_write_text(output / "run_config.json", json.dumps(config, indent=2) + "\n")

        for ordinal, index in enumerate(indices):
            sample_started = time.perf_counter()
            example = prepare_stage2_example(
                examples[index], image_root=image_root, latent_size=qwen.config.latent_size
            )
            batch = collator([example])
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            model_inputs = {
                key: value for key, value in batch.items()
                if isinstance(value, torch.Tensor) and key != "labels"
            }
            items = tuple(sorted(
                batch["step_preference_items"], key=lambda item: item.span.target_text_start
            ))
            score_mask = Stage2Trainer._chosen_score_mask(
                model_inputs, items, stage2_debugging=True
            )
            with torch.inference_mode():
                base_embeds = qwen.prepare_stage2_base_inputs_embeds(
                    input_ids=model_inputs["input_ids"],
                    pixel_values=model_inputs.get("pixel_values"),
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                )
                native = _forward_prepared(
                    qwen,
                    dict(model_inputs, prepared_base_inputs_embeds=base_embeds),
                    score_mask,
                )
                final_embeds = native.inputs_embeds
                native_logits = native.stage2_scored_logits
                clean_positions, _ = qwen.get_rope_index(
                    model_inputs["input_ids"],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    attention_mask=model_inputs["attention_mask"],
                )
                neutral = qwen.model.embed_tokens(torch.tensor(
                    [[token_ids.pad_token_id]], dtype=torch.long, device=device
                ))[0, 0]
                clean_branches = _clean_branches(
                    model_inputs["input_ids"], model_inputs["attention_mask"],
                    final_embeds, items,
                )
                clean_logits = _decode_branches(
                    qwen, clean_branches, clean_positions.expand(-1, len(items), -1)
                )
                logits_by_site = [dict(clean=value) for value in clean_logits]
                for scope in ("body", "boundary", "full"):
                    branches = build_length_preserving_branches(
                        input_ids=model_inputs["input_ids"],
                        attention_mask=model_inputs["attention_mask"],
                        clean_final_inputs_embeds=final_embeds,
                        items=items,
                        scope=scope,
                        neutral_embedding=neutral,
                    )
                    branch_logits = _decode_branches(
                        qwen, branches, clean_positions.expand(-1, len(items), -1)
                    )
                    for site_index, value in enumerate(branch_logits):
                        logits_by_site[site_index][f"{scope}_mask"] = value
                for scope in ("body", "boundary", "full"):
                    branches = build_deleted_branches(
                        input_ids=model_inputs["input_ids"],
                        attention_mask=model_inputs["attention_mask"],
                        clean_final_inputs_embeds=final_embeds,
                        items=items,
                        scope=scope,
                    )
                    branches, deleted_positions = _prepare_deleted_decoder_batch(
                        qwen, branches, model_inputs.get("image_grid_thw")
                    )
                    branch_logits = _decode_branches(
                        qwen, branches, deleted_positions,
                    )
                    for site_index, value in enumerate(branch_logits):
                        logits_by_site[site_index][f"{scope}_delete"] = value

            excluded = control_ids | set(
                range(len(processor.tokenizer), native_logits.shape[-1])
            )
            sites, cursor = [], 0
            direct_sites = direct_by_index[index]["sites"]
            boundary_sites = boundary_by_index[index]["sites"]
            for site_index, item in enumerate(items):
                count = item.chosen_target_ids.numel()
                ordered_logits = {
                    name: logits_by_site[site_index][name] for name in CONDITIONS
                }
                comparison = compare_factorial_logits(
                    logits_by_condition=ordered_logits,
                    target_ids=item.chosen_target_ids,
                    excluded_output_ids=excluded,
                    vision_start_id=token_ids.start_token_id,
                )
                native_site = native_logits[0, cursor:cursor + count]
                native_control = native_replay_control(
                    native_site, ordered_logits["clean"], item.chosen_target_ids
                )
                span = item.span
                site = {
                    "status": "ok",
                    "step_id": span.step_id,
                    "target_text_interval": [span.target_text_start, span.target_text_end],
                    "target_token_count": count,
                    "target_token_ids": item.chosen_target_ids.tolist(),
                    "latent_body_interval": [span.latent_start, span.latent_end],
                    "full_visual_interval": [span.prev_text_end, span.target_text_start],
                    "deleted_token_counts": {"body": 8, "boundary": 2, "full": 10},
                    "native_clean_control": native_control,
                    "factorial_comparison": comparison,
                }
                sites.append(site)
                cursor += count
            if cursor != native_logits.shape[1]:
                raise ValueError("native clean logits contain unassigned targets")
            _validate_sites_against_reference(sites, boundary_sites)
            _validate_direct_site_alignment(sites, direct_sites)
            if args.smoke_check and not config["smoke_checks"]:
                config["smoke_checks"].append(_smoke_result(
                    args.reference_role, index, sites, direct_sites
                ))
            records.append({
                "data_index": index,
                "sample_id": example.sample_id,
                "status": "ok",
                "sites": sites,
                "elapsed_seconds": time.perf_counter() - sample_started,
            })
            config["elapsed_seconds"] = time.perf_counter() - started
            write_factorial_outputs(output, records, config)
            print(json.dumps({
                "progress": f"{ordinal + 1}/{len(indices)}",
                "data_index": index,
                "sites": len(sites),
                "target_tokens": sum(site["target_token_count"] for site in sites),
                "seconds": records[-1]["elapsed_seconds"],
            }), flush=True)
            del (
                batch, model_inputs, base_embeds, native, final_embeds, native_logits,
                clean_branches, clean_logits, logits_by_site, sites,
            )
            gc.collect()
            torch.cuda.empty_cache()

        expected_sites = sum(len(boundary_by_index[index]["sites"]) for index in indices)
        actual_sites = sum(len(record["sites"]) for record in records)
        if actual_sites != expected_sites:
            raise ValueError(f"factorial site count {actual_sites} != reference {expected_sites}")
        controls = summarize_native_controls(records)
        _atomic_write_text(output / "native_control.json", json.dumps(controls, indent=2) + "\n")
        if args.reference_role == "same_model":
            write_direct_reference_controls(output, records, direct_records)
        config.update(
            status="complete",
            native_clean_control_summary=controls,
            elapsed_seconds=time.perf_counter() - started,
            total_sites=actual_sites,
            total_target_tokens=sum(
                site["target_token_count"] for record in records for site in record["sites"]
            ),
            publication="atomic derived outputs; complete config published last",
        )
        return write_factorial_outputs(output, records, config)
    except BaseException as error:
        config.update(
            status="error",
            error_type=type(error).__name__,
            error_message=str(error),
            elapsed_seconds=time.perf_counter() - started,
        )
        if records:
            write_factorial_outputs(output, records, config)
        else:
            _atomic_write_text(output / "run_config.json", json.dumps(config, indent=2) + "\n")
        raise


def main():
    print(json.dumps(run(build_parser().parse_args()), indent=2))

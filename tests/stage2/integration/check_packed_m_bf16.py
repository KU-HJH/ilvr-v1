"""Forward-only real-model CUDA comparison using the approved BF16 budget v1.

Run from the worktree with PYTHONPATH=transformers/src:src:. and its venv.
This is an explicit diagnostic CLI, not a training or default pytest entrypoint.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from stage2.checkpointing import configure_stage2_metadata, resolve_latent_token_ids
from stage2.data import load_jsonl_examples, prepare_stage2_example, Stage2DataCollator
from stage2.objective import Stage2ObjectiveConfig, target_log_likelihood
from stage2.objective_runtime import Stage2ObjectiveRuntime, model_inputs
from stage2.objective_windows import build_text_vocabulary
from stage2.packed_m import PackedMPrefixCapture, score_packed_m
from stage2.peft import (
    ALLOWED_STAGE2_TARGET_SUFFIXES, Stage2TuningConfig,
    load_stage1_dora_for_fresh_stage2,
)
from stage2.rng import capture_rng_state, replay_rng_state
from tests.stage2.packed_m_helpers import compare_cuda_bf16_likelihoods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="freeze sample manifest without loading a model or using CUDA")
    args = parser.parse_args()
    args.output.resolve().relative_to((Path.cwd() / "results").resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.adapter)
    config = AutoConfig.from_pretrained(args.base_model)
    tokens = resolve_latent_token_ids(processor.tokenizer, config)
    configure_stage2_metadata(config, tokens, latent_size=8, supervise_latent_boundaries=True)
    collator = Stage2DataCollator(processor, tokens, 8, True, enable_stage2_objective=True)
    runtime = Stage2ObjectiveRuntime(
        Stage2ObjectiveConfig(True, 0, 1, post_window=2, stage2_m_execution="packed_suffix"),
        tokens, 8, build_text_vocabulary(processor.tokenizer, config, config.vocab_size), None,
    )
    samples = load_jsonl_examples(args.data)

    def prepare(index):
        inputs = collator([prepare_stage2_example(
            samples[index], image_root=args.images, latent_size=8,
        )])
        return inputs, runtime.select(inputs).blocks

    # Freeze selection before model loading or any numerical comparison.
    selected = [(index, *prepare(index)) for index in (920, 1838)]
    counts = dict.fromkeys((1, 2, 3, 4), 0)
    candidates = [index for index in range(len(samples)) if index not in (920, 1838)]
    random.Random(42).shuffle(candidates)
    for index in candidates:
        inputs, blocks = prepare(index)
        count = len(blocks)
        if count in counts and counts[count] < 2:
            selected.append((index, inputs, blocks))
            counts[count] += 1
        if all(value == 2 for value in counts.values()):
            break
    if len(selected) != 10:
        raise RuntimeError(f"could not select two samples per block count: {counts}")
    manifest = {
        "policy": "cuda-bf16-likelihood-v1", "block_atol_nat": 0.05,
        "token_atol_nat": 0.10, "rtol": 0, "raw_logits_gate": False,
        "post_window": 2, "latent_size": 8, "seed_per_sample": 42,
        "selection": "known 920/1838 plus first two per K=1..4 in Random(42) shuffled indices",
        "base_model": str(args.base_model), "adapter": str(args.adapter),
        "data": str(args.data), "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "images": str(args.images),
        "samples": [
            {"index": index, "sequence_length": inputs["input_ids"].shape[1],
             "blocks": [asdict(block) for block in blocks]}
            for index, inputs, blocks in selected
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "selection_frozen", **manifest}), flush=True)
    if args.prepare_only:
        return 0

    torch.manual_seed(42)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, attn_implementation="sdpa",
    )
    model = load_stage1_dora_for_fresh_stage2(
        base, processor, args.adapter,
        Stage2TuningConfig("dora", 32, 64, .05, ALLOWED_STAGE2_TARGET_SUFFIXES),
        requested_base_model_reference=str(args.base_model),
    )
    configure_stage2_metadata(base.config, tokens, latent_size=8, supervise_latent_boundaries=True)
    base.config.use_cache = False
    model.to("cuda:0").train()
    print(json.dumps({"event": "environment", "torch": torch.__version__,
                      "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                      "parameter_dtype": str(next(base.parameters()).dtype),
                      "autocast": "bfloat16", "backend": "sdpa", "training_mode": True,
                      "backward": False, "optimizer": False}), flush=True)
    reports = []
    with (args.output / "comparisons.jsonl").open("x") as stream:
        for index, cpu_inputs, blocks in selected:
            inputs = {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
                      for key, value in cpu_inputs.items()}
            clean = {key: value for key, value in model_inputs(inputs).items() if key != "labels"}
            torch.manual_seed(42)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rng = capture_rng_state()
                with PackedMPrefixCapture(model, inputs["input_ids"].shape[1]) as capture:
                    output = model(**clean)
                    del output
                observed = []
                handle = base.lm_head.register_forward_hook(
                    lambda _module, _args, value: observed.append(value[0].detach())
                )
                try:
                    packed = score_packed_m(model, inputs, blocks, capture)
                finally:
                    handle.remove()
                assert packed.decoder_calls == 1 and len(observed) == 1
                packed_logits = observed[0]
                assert packed_logits.shape[0] == sum(len(block.target_ids) for block in blocks)
                offset = 0
                for block_index, block in enumerate(blocks):
                    with replay_rng_state(rng):
                        output = model(**clean, stage2_ablation=(block.start_position, block.end_position))
                    full_logits = output.logits[block.batch_index, list(block.post_positions)]
                    targets = torch.tensor(block.target_ids, device=full_logits.device)
                    selected_logits = packed_logits[offset:offset + len(block.target_ids)]
                    offset += len(block.target_ids)
                    full_ll = target_log_likelihood(full_logits, targets)
                    packed_ll = target_log_likelihood(selected_logits, targets)
                    # Verify the observed head rows really supplied the production score.
                    assert torch.equal(packed_ll.mean(), packed.scores[block_index])
                    report = {
                        "sample": index, "block": block_index,
                        "post_positions": list(block.post_positions),
                        "target_ids": list(block.target_ids),
                        "full_token_ll": full_ll.tolist(), "packed_token_ll": packed_ll.tolist(),
                        "raw_logit_max_abs_error": (full_logits.float() - selected_logits.float()).abs().max().item(),
                        "packed_decoder_calls": packed.decoder_calls, "packed_query_rows": packed.query_rows,
                        **compare_cuda_bf16_likelihoods(full_ll.tolist(), packed_ll.tolist()),
                    }
                    reports.append(report)
                    line = json.dumps(report)
                    stream.write(line + "\n")
                    stream.flush()
                    print(line, flush=True)
                    del output, full_logits, selected_logits
                del capture, packed, packed_logits, observed
            del inputs, clean
    summary = {
        "passed": all(report["passed"] for report in reports),
        "samples": len(selected), "blocks": len(reports),
        "tokens": sum(len(report["target_ids"]) for report in reports),
        "failed_blocks": [{"sample": report["sample"], "block": report["block"]}
                          for report in reports if not report["passed"]],
        "block_max_abs_error": max(report["block_abs_error"] for report in reports),
        "token_max_abs_error": max(report["token_max_abs_error"] for report in reports),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", **summary}), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

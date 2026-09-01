# evaluate_deepseed.py
import os
import re
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Union

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from stage2.checkpointing import (
    Stage2ArtifactKind,
    inspect_stage2_artifact,
    resolve_latent_token_ids,
)
from stage2.backend_fallback import is_flash_attention_initialization_error
from stage2.inference import (
    Stage2InferenceController,
    build_stage2_generation_config,
)
from stage2.peft import (
    load_stage2_model_for_inference,
    resolve_stage2_qwen_model,
)
from multinode_evaluation import (
    dataset_identity,
    iter_owned_samples,
    resolve_evaluation_image_paths,
    resolve_model_identity,
    shard_manifest_path,
    write_shard_manifest,
)
from evaluation.scoring import (
    EvaluationScoreSummary,
    extract_final_answer,
    parse_gold_answer,
    score_answer,
)
from evaluation.runtime import prepare_evaluation_input

try:
    from mathruler.grader import extract_boxed_content
except Exception:
    extract_boxed_content = None
import re

ACTION_MAP = {
    "LEFT": (0,-1), "DOWN": (1,0), "RIGHT": (0,1), "UP": (-1,0),
    "L": (0,-1), "D": (1,0), "R": (0,1), "U": (-1,0),
}


BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  
def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--base_model_dir", type=str, help="Stage 1 base directory for a Stage 2 PEFT artifact.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data jsonl file.")
    parser.add_argument("--task_name", type=str, required=True, help="Task name for bookkeeping.")
    parser.add_argument("--output_json_path", type=str, default="evaluation_results.jsonl", help="Where to save per-sample results.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="HF cache dir.")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_sequence_length", type=int, default=32768)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--model-identity", type=str)
    return parser.parse_args()


def load_processor(model_dir: str, cache_dir: str):

    try:
        return AutoProcessor.from_pretrained(model_dir, cache_dir=cache_dir)
    except Exception:
        logging.warning("Failed to load processor from model_dir, falling back to base model id.")
        return AutoProcessor.from_pretrained(BASE_MODEL_ID, cache_dir=cache_dir)

def load_model(model_dir: str, cache_dir: str):

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2"
        )
    except Exception as e:
        if not is_flash_attention_initialization_error(e):
            raise
        logging.warning(f"flash_attention_2 initialization failed ({e}), falling back to eager.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="eager"
        )
    model.eval()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    return model


def load_evaluation_artifacts(
    *, model_dir: str, base_model_dir: str | None, cache_dir: str
):
    """Load either one full Stage 2 artifact or a Stage 1 base plus adapter."""
    artifact_inspection = inspect_stage2_artifact(model_dir)
    if artifact_inspection.artifact_kind is Stage2ArtifactKind.FULL:
        if base_model_dir is not None:
            raise ValueError("base_model_dir is invalid for a full Stage 2 artifact")
        return load_model(model_dir, cache_dir), load_processor(model_dir, cache_dir)

    if base_model_dir is None:
        raise ValueError("base_model_dir is required for a Stage 2 PEFT artifact")
    processor = AutoProcessor.from_pretrained(model_dir, cache_dir=cache_dir)
    model = load_stage2_model_for_inference(
        base_model_dir,
        model_dir,
        processor,
        source_provenance_kwargs={"cache_dir": cache_dir},
        load_model=lambda source, **selectors: load_model(
            source, selectors["cache_dir"]
        ),
        artifact_inspection=artifact_inspection,
    )
    return model, processor

def _resolve_image_paths(image_input: Union[str, List[str]]) -> List[str]:

    if image_input is None:
        return []
    if isinstance(image_input, str):
        return [image_input]
    if isinstance(image_input, list):
        return list(image_input)
    return []

def open_images(paths: List[str]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        # Raise error immediately for directories, be explicit
        if os.path.isdir(p):
            raise IsADirectoryError(f"Expected image file, but received directory path: {p}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def _strip_special_tokens(s: str) -> str:
    # Common end/special tokens, extend if necessary
    end_markers = [
        "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>",
        "</s>", "<s>", "[/INST]", "[INST]",
        "<|latent_start|>", "<|latent_end|>", "<|latent_pad|>",
    ]
    for m in end_markers:
        s = s.replace(m, "")
    return s.strip()

def extract_assistant_content(text: str) -> str:

    s = text or ""


    m = re.search(r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)", s, flags=re.S|re.I)
    if m:
        return _strip_special_tokens(m.group(1))


    m = re.search(
        r"<\|assistant\|>\s*(.*?)(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|<\|end\|>|$)",
        s, flags=re.S|re.I
    )
    if m:
        return _strip_special_tokens(m.group(1))


    m = re.search(r"(?:^|\n|\r)assistant\s*[:：]?\s*(.*)$", s, flags=re.S|re.I)
    if m:
        return _strip_special_tokens(m.group(1))


    return _strip_special_tokens(s)


def run_one_example(model, processor, sample: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    img_list = _resolve_image_paths(sample.get("image_input", []))
    images = open_images(img_list) if img_list else None
    prepared = prepare_evaluation_input(processor, sample, images)

    stage2_model = resolve_stage2_qwen_model(model)
    effective_device = getattr(model, "device", None)
    if effective_device is None:
        effective_device = getattr(stage2_model, "device", None)
    if effective_device is None:
        raise ValueError("Unable to determine the Stage 2 evaluation device")
    inputs = {
        k: v.to(effective_device)
        for k, v in prepared.tensors.items()
    }

    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("Stage 2 evaluation currently supports batch_size=1 only")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    latent_size = getattr(stage2_model.config, "latent_size", None)
    if not isinstance(latent_size, int) or isinstance(latent_size, bool) or latent_size <= 0:
        raise ValueError("Stage 2 checkpoint config must contain a positive latent_size")
    token_ids = resolve_latent_token_ids(processor.tokenizer, stage2_model.config)
    generation_config = build_stage2_generation_config(
        model_config=stage2_model.config,
        model_generation_config=getattr(stage2_model, "generation_config", None),
        tokenizer=processor.tokenizer,
        generation_kwargs=gen_kwargs,
    )

    with torch.inference_mode():
        generation_result = Stage2InferenceController().generate(
            model=stage2_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_ids=token_ids,
            generation_config=generation_config,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            return_latent_states=False,
        )
    out_ids = generation_result.sequences
    out_text = processor.batch_decode(out_ids, skip_special_tokens=False)[0].strip()
    suffix_ids = out_ids[:, prepared.prompt_token_length:]
    assistant_text = processor.batch_decode(
        suffix_ids, skip_special_tokens=False
    )[0].strip()

    # Extract the answer from the assistant's segment only — running the
    # regex over the full decoded text (prompt included) can match text in
    # the user prompt and keeps trailing special tokens like <|im_end|>.
    assistant_only = extract_assistant_content(assistant_text)
    extracted = extract_final_answer(assistant_only).extracted
    return {
        "prompt_identity": prepared.prompt_identity,
        "prompt_token_length": prepared.prompt_token_length,
        "generated_suffix_token_ids": [
            int(token_id) for token_id in suffix_ids[0].tolist()
        ],
        "raw_output": out_text,
        "assistant_only": assistant_only,
        "extracted_final_answer": extracted,
        "latent_start_count": int(generation_result.latent_start_count),
        "latent_block_count": int(generation_result.latent_block_count),
        "stop_reason": generation_result.stop_reason,
    }

def main():
    args = get_eval_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    sharded = args.num_shards == 4
    if not (
        (args.num_shards == 1 and args.shard_index == 0)
        or (sharded and args.shard_index in range(4))
    ):
        raise ValueError(
            "evaluation requires num_shards=1 with shard_index=0 or four "
            "shards indexed 0-3"
        )

    dataset_path = Path(args.test_data_path).resolve(strict=True)
    if not dataset_path.is_file():
        raise ValueError("evaluation test data must be a file")
    dataset_root = dataset_path.parent

    logging.info(f"Loading model: {args.model_dir}")
    model, processor = load_evaluation_artifacts(
        model_dir=args.model_dir,
        base_model_dir=args.base_model_dir,
        cache_dir=args.cache_dir,
    )


    data = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not data:
        logging.error("Test set is empty, exiting.")
        if not sharded:
            return

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        max_sequence_length=args.max_sequence_length,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(args.temperature > 0)
    )

    total = len(data)
    owned_samples = (
        list(iter_owned_samples(data, args.num_shards, args.shard_index))
        if sharded
        else list(enumerate(data))
    )
    worker_total = len(owned_samples)
    success = 0
    results = []
    processed_indices = []
    worker_failed = False

    logging.info(f"Starting evaluation: num_samples={worker_total if sharded else total}")
    for local_index, (i, sample) in enumerate(owned_samples):
        processed_indices.append(i)
        try:
            evaluation_sample = dict(sample)
            evaluation_sample["image_input"] = resolve_evaluation_image_paths(
                sample.get("image_input", []), dataset_root
            )
            pred = run_one_example(
                model, processor, evaluation_sample, gen_kwargs
            )
            out_text = pred["raw_output"]
            assistant_only = pred["assistant_only"]

            if args.task_name == "vsp-spatial-planning-cot":

                map_desc = sample.get("map_desc", [])
                path_str = extract_path_from_text(assistant_only)
                sim = simulate_vsp(map_desc, path_str)
                ok = bool(sim["success"])
                if ok: success += 1

                results.append({
                    "index": i,
                    "task_name": args.task_name,
                    "text_input": sample.get("text_input", ""),
                    "image_input": sample.get("image_input", []),
                    "predicted_full_output": out_text,
                    "extracted_path_string": path_str,
                    "vsp_simulation_result": sim,
                    "match": ok,
                })

            else:  # Standard tasks
                gold = sample.get("original_final_answer")
                answer_score = score_answer(assistant_only, gold)
                ok = answer_score.match
                if ok: success += 1
                
                results.append({
                    "index": i,
                    "task_name": args.task_name,
                    "prompt_identity": pred.get("prompt_identity"),
                    "prompt_token_length": pred.get("prompt_token_length"),
                    "generated_suffix_token_ids": pred.get(
                        "generated_suffix_token_ids", []
                    ),
                    "raw_output": pred.get("raw_output"),
                    "assistant_only": assistant_only,
                    "prediction": assistant_only,
                    "extracted_final_answer": answer_score.prediction.extracted,
                    "prediction_parse_status": answer_score.prediction.status,
                    "prediction_parse_detail": answer_score.prediction.detail,
                    "prediction_normalized": answer_score.prediction.normalized,
                    "gold_final_answer": gold,
                    "gold_parse_status": answer_score.gold.status,
                    "gold_parse_detail": answer_score.gold.detail,
                    "gold_normalized": answer_score.gold.normalized,
                    "invalid_gold_reason": answer_score.invalid_gold_reason,
                    "latent_start_count": pred.get("latent_start_count"),
                    "latent_block_count": pred.get("latent_block_count"),
                    "stop_reason": pred.get("stop_reason"),
                    "match": bool(ok)
                })

            progress = local_index + 1 if sharded else i + 1
            progress_total = worker_total if sharded else total
            if progress % 10 == 0 or progress == progress_total:
                logging.info(f"[{progress}/{progress_total}] Current accuracy={success/progress:.4f}")

        except Exception as e:
            logging.error(f"[{i}] Evaluation failed: {e}")
            if sharded:
                worker_failed = True
            error_result = {
                "index": i,
                "task_name": args.task_name,
                "error": str(e),
            }
            if args.task_name != "vsp-spatial-planning-cot":
                gold = sample.get("original_final_answer")
                gold_parse = parse_gold_answer(gold)
                invalid_gold_reason = (
                    gold_parse.detail if gold_parse.status == "invalid_gold" else
                    gold_parse.status if gold_parse.status != "parsed" else None
                )
                error_result.update({
                    "prompt_identity": None,
                    "prompt_token_length": None,
                    "generated_suffix_token_ids": [],
                    "raw_output": None,
                    "assistant_only": None,
                    "prediction": None,
                    "extracted_final_answer": None,
                    "prediction_parse_status": "inference_error",
                    "gold_final_answer": gold,
                    "gold_parse_status": gold_parse.status,
                    "gold_parse_detail": gold_parse.detail,
                    "gold_normalized": gold_parse.normalized,
                    "invalid_gold_reason": invalid_gold_reason,
                    "latent_start_count": None,
                    "latent_block_count": None,
                    "stop_reason": None,
                    "match": False,
                })
            results.append(error_result)

    metric_total = worker_total if sharded else total
    score_summary = (
        EvaluationScoreSummary.from_rows(results, metric_total)
        if args.task_name != "vsp-spatial-planning-cot"
        else None
    )
    acc = (
        score_summary.coverage_accuracy if score_summary is not None
        else success / metric_total if metric_total > 0 else 0.0
    )
    with open(args.output_json_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    if sharded:
        write_shard_manifest(
            shard_manifest_path(args.output_json_path),
            shard_jsonl=args.output_json_path,
            dataset_identity=dataset_identity(args.test_data_path),
            model_identity=resolve_model_identity(
                args.model_dir, args.model_identity
            ),
            generation_config={
                "max_new_tokens": args.max_new_tokens,
                "max_sequence_length": args.max_sequence_length,
                "repetition_penalty": args.repetition_penalty,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            expected_source_indices=[index for index, _sample in owned_samples],
            processed_source_indices=processed_indices,
            worker_exit_status=1 if worker_failed else 0,
        )

    report_total = worker_total if sharded else total
    logging.info(f"Evaluation finished: success {success}/{report_total}; accuracy={acc:.4f}")
    print(f"Accuracy = {acc:.4f}  ({success}/{report_total})")
    if score_summary is not None:
        print(
            "Valid-gold accuracy = "
            f"{score_summary.valid_gold_accuracy:.4f} "
            f"({score_summary.correct}/{score_summary.valid_gold_count})"
        )
    if sharded:
        return 1 if worker_failed else 0
    



def parse_action_sequence(path_str: str):
    s = (path_str or "").upper()
    return [ch for ch in s if ch in ['U','D','R','L']]

def simulate_vsp(map_desc, path_str):
    actions = parse_action_sequence(path_str)

    # Find start point 1
    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c); break
        if start is not None: break
    if start is None:
        raise ValueError("The map description does not contain a start position (cell value 1).")

    cur = start
    for a in actions:
        if a not in ACTION_MAP:
            return {"success": False, "status": "Invalid action", "invalid": True}
        dr, dc = ACTION_MAP[a]
        nr, nc = cur[0] + dr, cur[1] + dc
        if nr < 0 or nr >= len(map_desc) or nc < 0 or nc >= len(map_desc[0]):
            continue
        cur = (nr, nc)
        if map_desc[nr][nc] == -1:
            return {"success": False, "status": "Fell in hole", "invalid": False}

    return {"success": map_desc[cur[0]][cur[1]] == 2,
            "status": "Reached goal" if map_desc[cur[0]][cur[1]] == 2 else "Did not reach goal",
            "invalid": False}

def extract_path_from_text(generated_text: str) -> str:
    if extract_boxed_content is not None:
        s = extract_boxed_content(generated_text)
        if s: return s
    m = re.search(r"\\boxed\{([UDLRudlr]+)\}", generated_text)
    if m: return m.group(1)
    m = re.search(r"final\s*answer\s*(?:is)?\s*[:：]\s*([UDLRudlr]+)", generated_text, re.I)
    return m.group(1) if m else ""


if __name__ == "__main__":
    sys.exit(main())

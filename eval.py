# evaluate_deepseed.py
import os
import re
import json
import argparse
import logging
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Union

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    LogitsProcessorList,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from peft_common import resolve_qwen2_5_vl_model
from stage1.checkpointing import (
    STAGE1_PEFT_METADATA_FILENAME,
    Stage1ArtifactKind,
    classify_stage1_artifact,
    load_stage1_peft_artifact,
)
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
from evaluation.scoring import EvaluationScoreSummary, extract_final_answer, score_answer

try:
    from mathruler.grader import extract_boxed_content
except Exception:
    extract_boxed_content = None


ACTION_MAP = {
    "LEFT": (0, -1), "DOWN": (1, 0), "RIGHT": (0, 1), "UP": (-1, 0),
    "L": (0, -1), "D": (1, 0), "R": (0, 1), "U": (-1, 0),
}

BASE_MODEL_ID = ""  


class Stage1LatentTemplateLogitsProcessor:
    """Force exactly ``latent_size`` pad tokens inside a Stage 1 latent block."""

    def __init__(self, start_id: int, end_id: int, pad_id: int, latent_size: int):
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.pad_id = int(pad_id)
        self.latent_size = int(latent_size)

    def __call__(self, input_ids, scores):
        for batch_index in range(input_ids.size(0)):
            sequence = input_ids[batch_index]
            starts = (sequence == self.start_id).nonzero().squeeze(-1)
            ends = (sequence == self.end_id).nonzero().squeeze(-1)
            last_start = int(starts[-1].item()) if starts.numel() else -1
            last_end = int(ends[-1].item()) if ends.numel() else -1
            if last_start == -1 or last_end >= last_start:
                scores[batch_index, self.pad_id] = -float("inf")
                scores[batch_index, self.end_id] = -float("inf")
                continue
            position = last_start + 1
            pad_count = 0
            while (
                position < sequence.size(0)
                and int(sequence[position].item()) == self.pad_id
            ):
                pad_count += 1
                position += 1
            scores[batch_index, :] = -float("inf")
            forced_token = (
                self.pad_id if pad_count < self.latent_size else self.end_id
            )
            scores[batch_index, forced_token] = 0.0
        return scores

def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the trained model directory.")
    parser.add_argument("--base_model_dir", type=str,
                        help="Stage 1 base directory for a Stage 2 PEFT artifact.")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to the test data JSONL file.")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name (used for bookkeeping and branching logic).")
    parser.add_argument("--output_json_path", type=str, default="evaluation_results.jsonl",
                        help="Path to save per-sample evaluation results.")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Hugging Face cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--model-identity", type=str)
    return parser.parse_args()


def load_processor(model_dir: str, cache_dir: str):
    """
    Prefer loading the processor from the finetuned model directory
    (including any special tokens). If this fails, fall back to the
    base model processor.

    During evaluation, no new special tokens should be introduced,
    as this may cause mismatches with the model weights.
    """
    try:
        return AutoProcessor.from_pretrained(model_dir, cache_dir=cache_dir)
    except Exception:
        logging.warning(
            "Failed to load processor from model_dir; falling back to the base model."
        )
        return AutoProcessor.from_pretrained(BASE_MODEL_ID, cache_dir=cache_dir)


def load_model(model_dir: str, cache_dir: str):
    """
    Prefer flash_attention_2 for efficiency; automatically fall back
    to eager attention if flash attention is unavailable.
    """
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        if not is_flash_attention_initialization_error(e):
            raise
        logging.warning(
            f"flash_attention_2 initialization failed ({e}); "
            f"falling back to eager attention."
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="eager",
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
    """Load one full artifact or a validated Stage 1/2 PEFT artifact."""
    artifact_path = Path(model_dir)
    if (
        artifact_path.is_dir()
        and (artifact_path / STAGE1_PEFT_METADATA_FILENAME).is_file()
    ):
        artifact_kind = classify_stage1_artifact(artifact_path)
        if artifact_kind is not Stage1ArtifactKind.PEFT:
            raise ValueError("Stage 1 metadata requires a PEFT artifact")
        if base_model_dir is None:
            raise ValueError(
                "base_model_dir is required for a Stage 1 PEFT artifact"
            )
        processor = AutoProcessor.from_pretrained(model_dir, cache_dir=cache_dir)
        model = load_stage1_peft_artifact(
            load_model(base_model_dir, cache_dir),
            processor,
            artifact_path,
            requested_base_model_reference=base_model_dir,
            is_trainable=False,
        )
        model.eval()
        return model, processor

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
    """
    Resolve image paths from either a single string or a list of strings.

    - Relative paths are resolved using BASE_DATASET_DIR.
    - Absolute paths are returned unchanged.
    """
    if image_input is None:
        return []

    if isinstance(image_input, str):
        return [image_input]

    if isinstance(image_input, list):
        return list(image_input)

    return []


def open_images(paths: List[str]) -> List[Image.Image]:
    """
    Load images from disk and convert them to RGB.

    Raises explicit errors if a directory path or a non-existent file is encountered.
    """
    imgs = []
    for p in paths:
        if os.path.isdir(p):
            raise IsADirectoryError(
                f"Expected an image file, but received a directory path: {p}"
            )
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def _strip_special_tokens(s: str) -> str:
    """
    Remove common special or end-of-sequence markers from decoded text.
    """
    end_markers = [
        "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>",
        "</s>", "<s>", "[/INST]", "[INST]",
        "<|latent_start|>", "<|latent_end|>", "<|latent_pad|>",
    ]
    for m in end_markers:
        s = s.replace(m, "")
    return s.strip()


def extract_assistant_content(text: str) -> str:
    """
    Extract only the assistant's content from the raw model output.

    Designed to be compatible with ChatML / Qwen / LLaMA-style templates.

    Priority:
      1) <|im_start|>assistant ... <|im_end|>
      2) <|assistant|> ... (until an end marker or end of text)
      3) Fallback: content after the last occurrence of 'assistant'
    """
    s = text or ""

    # 1) ChatML-style pattern
    m = re.search(
        r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)",
        s,
        flags=re.S | re.I,
    )
    if m:
        return _strip_special_tokens(m.group(1))

    # 2) Qwen / other styles
    m = re.search(
        r"<\|assistant\|>\s*(.*?)(?:<\|im_end\|>|<\|endoftext\|>|"
        r"<\|eot_id\|>|<\|end\|>|$)",
        s,
        flags=re.S | re.I,
    )
    if m:
        return _strip_special_tokens(m.group(1))

    # 3) Fallback: last assistant segment
    m = re.search(
        r"(?:^|\n|\r)assistant\s*[:：]?\s*(.*)$",
        s,
        flags=re.S | re.I,
    )
    if m:
        return _strip_special_tokens(m.group(1))

    # If nothing matches, return cleaned full text
    return _strip_special_tokens(s)


def run_one_example(
    model,
    processor,
    sample: Dict[str, Any],
    gen_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run inference on a single example and return raw output,
    extracted final answer, and inference time.
    """
    text_input = sample.get("text_input", "")
    img_list = _resolve_image_paths(sample.get("image_input", []))
    images = open_images(img_list) if img_list else None

    # Align with training: put image entries into user.content, then append text
    content = []
    if images:
        for _ in images:
            content.append({"type": "image"})
    content.append({"type": "text", "text": text_input})

    conversations = [
        {"role": "user", "content": content}
    ]

    prompt = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    )

    qwen_model = resolve_qwen2_5_vl_model(model)
    effective_device = getattr(model, "device", None)
    if effective_device is None:
        effective_device = getattr(qwen_model, "device", None)
    if effective_device is None:
        raise ValueError("Unable to determine the evaluation device")

    # Move tensor inputs to the outer runtime wrapper's device when available.
    inputs = {
        k: v.to(effective_device)
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }

    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("evaluation currently supports batch_size=1 only")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    latent_size = getattr(qwen_model.config, "latent_size", None)
    if not isinstance(latent_size, int) or isinstance(latent_size, bool) or latent_size <= 0:
        raise ValueError("checkpoint config must contain a positive latent_size")
    token_ids = resolve_latent_token_ids(processor.tokenizer, qwen_model.config)
    stage = getattr(qwen_model.config, "stage", None)
    if stage == "stage1":
        logits_processor = LogitsProcessorList(
            [
                Stage1LatentTemplateLogitsProcessor(
                    token_ids.start_token_id,
                    token_ids.end_token_id,
                    token_ids.pad_token_id,
                    latent_size,
                )
            ]
        )
    elif stage == "stage2":
        stage2_model = resolve_stage2_qwen_model(model)
        generation_config = build_stage2_generation_config(
            model_config=stage2_model.config,
            model_generation_config=getattr(
                stage2_model, "generation_config", None
            ),
            tokenizer=processor.tokenizer,
            generation_kwargs=gen_kwargs,
        )
    else:
        raise ValueError(f"unsupported evaluation stage: {stage!r}")

    # ---- Start inference timing (generation only) ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        if stage == "stage1":
            out_ids = model.generate(
                **inputs,
                **gen_kwargs,
                logits_processor=logits_processor,
                tokenizer=processor.tokenizer,
            )
        else:
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

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    # ---- End inference timing ----

    out_text = processor.batch_decode(
        out_ids, skip_special_tokens=False
    )[0].strip()

    # Extract the answer from the assistant's segment only — running the
    # regex over the full decoded text (prompt included) can match text in
    # the user prompt and keeps trailing special tokens like <|im_end|>.
    assistant_only = extract_assistant_content(out_text)
    extracted = extract_final_answer(assistant_only).extracted
    return {
        "raw_output": out_text,
        "assistant_only": assistant_only,
        "extracted_final_answer": extracted,
        "inference_time": inference_time,
    }


def main():
    args = get_eval_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

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

    logging.info(f"Loading model from: {args.model_dir}")
    model, processor = load_evaluation_artifacts(
        model_dir=args.model_dir,
        base_model_dir=args.base_model_dir,
        cache_dir=args.cache_dir,
    )

    # Load test data
    data = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not data:
        logging.error("Test dataset is empty. Exiting.")
        if not sharded:
            return

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(args.temperature > 0),
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
    total_inference_time = 0.0
    processed_indices = []
    worker_failed = False

    logging.info(
        f"Starting evaluation: num_samples={worker_total if sharded else total}"
    )
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
            sample_infer_time = pred.get("inference_time", None)
            if sample_infer_time is not None:
                total_inference_time += sample_infer_time

            out_text = pred["raw_output"]
            assistant_only = pred["assistant_only"]

            if args.task_name == "vsp-spatial-planning-cot":
                # Extract path and run simulation
                map_desc = sample.get("map_desc", [])
                path_str = extract_path_from_text(assistant_only)
                sim = simulate_vsp(map_desc, path_str)
                ok = bool(sim["success"])
                if ok:
                    success += 1

                results.append({
                    "index": i,
                    "task_name": args.task_name,
                    "text_input": sample.get("text_input", ""),
                    "image_input": sample.get("image_input", []),
                    "predicted_full_output": out_text,
                    "extracted_path_string": path_str,
                    "vsp_simulation_result": sim,
                    "match": ok,
                    "inference_time_sec": sample_infer_time,
                })

            else:
                # Default zebra-cot branch
                gold = sample.get("original_final_answer")
                answer_score = score_answer(assistant_only, gold)
                ok = answer_score.match
                if ok:
                    success += 1

                results.append({
                    "index": i,
                    "task_name": args.task_name,
                    "image_input": sample.get("image_input", []),
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
                    "match": bool(ok),
                    "inference_time_sec": sample_infer_time,
                })

            progress = local_index + 1 if sharded else i + 1
            progress_total = worker_total if sharded else total
            if progress % 10 == 0 or progress == progress_total:
                logging.info(
                    f"[{progress}/{progress_total}] Current accuracy = "
                    f"{success/progress:.4f}"
                )

        except Exception as e:
            logging.error(f"[{i}] Evaluation failed: {e}")
            if sharded:
                worker_failed = True
            results.append({
                "index": i,
                "task_name": args.task_name,
                "error": str(e),
                "inference_time_sec": None,
            })

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
    avg_inference_time = (
        total_inference_time / metric_total if metric_total > 0 else 0.0
    )

    with open(args.output_json_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        if not sharded:
            # Write a summary line as the last entry
            summary = {
                "summary": True,
                "num_samples": total,
                "average_inference_time_sec": avg_inference_time,
                "total_inference_time_sec": total_inference_time,
                "accuracy": acc,
                "success": success,
            }
            if score_summary is not None:
                summary.update({
                    "correct": score_summary.correct,
                    "source_count": score_summary.source_count,
                    "valid_gold_count": score_summary.valid_gold_count,
                    "invalid_gold_count": score_summary.invalid_gold_count,
                    "invalid_gold": score_summary.invalid_gold,
                    "coverage_accuracy": score_summary.coverage_accuracy,
                    "valid_gold_accuracy": score_summary.valid_gold_accuracy,
                })
            fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

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
    logging.info(
        f"Evaluation finished: {success}/{report_total} correct; accuracy={acc:.4f}"
    )
    print(f"Accuracy = {acc:.4f} ({success}/{report_total})")
    if sharded:
        return 1 if worker_failed else 0


def parse_action_sequence(path_str: str):
    """
    Parse a path string into a sequence of movement actions.
    """
    s = (path_str or "").upper()
    return [ch for ch in s if ch in ['U', 'D', 'R', 'L']]


def simulate_vsp(map_desc, path_str):
    """
    Simulate the spatial planning environment given a map and an action sequence.
    """
    actions = parse_action_sequence(path_str)

    # Locate the start position (cell value = 1)
    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c)
                break
        if start is not None:
            break

    if start is None:
        raise ValueError(
            "The map description does not contain a start position (cell value 1)."
        )

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
            return {"success": False, "status": "Fell into a hole", "invalid": False}

    return {
        "success": map_desc[cur[0]][cur[1]] == 2,
        "status": (
            "Reached goal" if map_desc[cur[0]][cur[1]] == 2
            else "Did not reach goal"
        ),
        "invalid": False,
    }


def extract_path_from_text(generated_text: str) -> str:
    """
    Extract a movement path string from model output.

    Priority:
      1) Use mathruler's boxed-content extractor if available
      2) Match '\\boxed{UDLR...}'
      3) Match 'final answer is: UDLR...'
    """
    if extract_boxed_content is not None:
        s = extract_boxed_content(generated_text)
        if s:
            return s

    m = re.search(r"\\boxed\{([UDLRudlr]+)\}", generated_text)
    if m:
        return m.group(1)

    m = re.search(
        r"final\s*answer\s*(?:is)?\s*[:：]\s*([UDLRudlr]+)",
        generated_text,
        re.I,
    )
    return m.group(1) if m else ""


if __name__ == "__main__":
    sys.exit(main())

"""Greedy trajectory (re)generation under the analysis numerics (ruling R24).

`trajectories` regenerates the model's own greedy TEST trajectories with the same
numerics the teacher-forced stages use (`apply_precision` + `numerical_policy()`),
so the fp32 replay reproduces them. It runs before `prepare`, writes one line per
question to `<out-dir>/shard-{i}.jsonl` (eval-row compatible), prompts over the
R12 length budget to `<out-dir>/skipped-shard-{i}.jsonl`, and the shard's
provenance to `<out-dir>/shard-{i}.meta.json`. `merge-trajectories` combines the
shards into `merged.jsonl` + `skipped.jsonl` and writes `trajectories-manifest.json`,
which `prepare` verifies before accepting the regenerated file as `--eval-rows`.

`generate_question` is shared with the `freegen` stage (which adds an image mask).
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from evaluation import latent_image_cli as cli
from evaluation.latent_image_cli import LENGTH_BUDGET

TASK_NAME = "zebra-cot"          # eval.py `--task_name` for TEST.jsonl (see merged.jsonl rows)
NUMERICS = "numerical_policy (SDPA MATH, TF32 off)"
GENERATION_KWARGS = {"max_new_tokens": 4096, "temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0,
                     "max_sequence_length": 128000, "generation_limit_policy": "external_full_latent_block"}
_ROW_FIELDS = ("prompt_identity", "prompt_token_length", "generated_suffix_token_ids", "assistant_only",
               "extracted_final_answer", "match", "latent_block_count", "stop_reason")


# ---------------------------------------------------------------------------
# Shared generation helper (trajectories and freegen)
# ---------------------------------------------------------------------------

def prepare_question(processor, test_row, data_root):
    from evaluation.latent_image_trajectory import _open_images
    from evaluation.runtime import prepare_evaluation_input
    images, _ = _open_images(test_row, data_root)
    return prepare_evaluation_input(processor, test_row, images)


def _generate(qwen, processor, prepared, mode):
    from evaluation.visual_interventions import VisualIntervention
    from stage2.checkpointing import resolve_latent_token_ids  # same helper eval.py uses
    from stage2.inference import Stage2InferenceController, build_stage2_generation_config
    device = qwen.model.embed_tokens.weight.device
    inputs = {k: v.to(device) for k, v in prepared.tensors.items()}
    config = build_stage2_generation_config(
        model_config=qwen.config, model_generation_config=getattr(qwen, "generation_config", None),
        tokenizer=processor.tokenizer, generation_kwargs=dict(GENERATION_KWARGS))
    intervention = None if mode is None else VisualIntervention(mode, qwen.config.image_token_id)
    with torch.inference_mode():
        return Stage2InferenceController().generate(
            model=qwen, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            token_ids=resolve_latent_token_ids(processor.tokenizer, qwen.config), generation_config=config,
            pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
            intervention=intervention)


def row_from_result(processor, prepared, test_row, result):
    """Decode and score one generation exactly as eval.py's zebra-cot branch does."""
    from eval import extract_assistant_content
    from evaluation.scoring import score_answer
    suffix = result.sequences[:, prepared.prompt_token_length:]
    assistant = extract_assistant_content(processor.batch_decode(suffix, skip_special_tokens=False)[0].strip())
    score = score_answer(assistant, test_row.get("original_final_answer"))
    return {"prompt_identity": prepared.prompt_identity, "prompt_token_length": int(prepared.prompt_token_length),
            "generated_suffix_token_ids": [int(t) for t in suffix[0].tolist()], "assistant_only": assistant,
            "extracted_final_answer": score.prediction.extracted, "match": bool(score.match),
            "latent_block_count": int(result.latent_block_count), "stop_reason": result.stop_reason}


def generate_question(model, processor, test_row, data_root, *, mode=None, expected_prompt_identity=None):
    """Greedy generation for one question; `mode` None means no image intervention.

    The caller applies precision and `numerical_policy()`. Returns a skip record
    instead of generating when the prompt's dual length (prompt tokens + one extra
    image span) already exceeds `LENGTH_BUDGET` (R12)."""
    from stage2.peft import resolve_stage2_qwen_model
    qwen = resolve_stage2_qwen_model(model)
    prepared = prepare_question(processor, test_row, data_root)
    if expected_prompt_identity is not None and prepared.prompt_identity != expected_prompt_identity:
        raise ValueError("prompt identity differs from the eval row; the prompt is reconstructed differently")
    image_tokens = int((prepared.tensors["input_ids"] == qwen.config.image_token_id).sum())
    dual = int(prepared.prompt_token_length) + image_tokens
    if dual > LENGTH_BUDGET:
        return {"skipped": "length_budget", "prompt_token_length": int(prepared.prompt_token_length),
                "image_tokens": image_tokens, "prompt_dual_length": dual}
    return row_from_result(processor, prepared, test_row, _generate(qwen, processor, prepared, mode))


# ---------------------------------------------------------------------------
# Crash-safe JSONL
# ---------------------------------------------------------------------------

def append_jsonl(path, row):
    """Append one complete line and fsync it before returning."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path, *, trim_partial=False):
    """Rows of a JSONL file. An unterminated last line is a crash leftover: it is
    truncated away when `trim_partial` (resume), otherwise it is ignored."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    complete = data[:data.rfind(b"\n") + 1]
    if trim_partial and len(complete) != len(data):
        with open(path, "r+b") as handle:
            handle.truncate(len(complete))
            handle.flush()
            os.fsync(handle.fileno())
    return [json.loads(line) for line in complete.decode("utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _shard_paths(out_dir, shard):
    out_dir = Path(out_dir)
    return (out_dir / f"shard-{shard}.jsonl", out_dir / f"skipped-shard-{shard}.jsonl",
            out_dir / f"shard-{shard}.meta.json")


def _provenance(args):
    """Everything a shard's rows depend on; must stay identical across resumes and shards."""
    return {**cli.model_identity(args), "selection_sha256": cli.sha256(args.selection),
            "data_sha256": cli.sha256(args.data), "generation_config": dict(GENERATION_KWARGS),
            "numerics": NUMERICS, "commit": cli.git_commit(), "git_status": cli.git_status_porcelain()}


def stage_trajectories(args):
    from evaluation.latent_image_trajectory import load_eval_rows, load_selection
    from evaluation.vision_block_runtime import numerical_policy
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path, skipped_path, meta_path = _shard_paths(args.out_dir, args.shard)
    meta = {**_provenance(args), "shard": args.shard, "num_shards": args.num_shards}
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        changed = sorted(k for k in set(meta) | set(previous) if k != "created" and meta.get(k) != previous.get(k))
        if changed:
            raise ValueError(f"shard meta differs from the current execution on {changed}; use a fresh --out-dir")
    else:
        cli.write_json_atomic(meta_path, {**meta, "created": datetime.now(timezone.utc).isoformat()})

    done = {int(r["index"]) for r in read_jsonl(rows_path, trim_partial=True)}
    done |= {int(r["index"]) for r in read_jsonl(skipped_path, trim_partial=True)}
    owned = cli.shard_items(load_selection(args.selection), args.shard, args.num_shards)
    pending = [idx for idx in owned if idx not in done]
    if not pending:
        return 0
    test_rows = cli.load_test_rows(args.data)
    eval_rows = load_eval_rows(args.eval_rows) if args.eval_rows else {}
    model, processor, _ = cli._teacher_forced_setup(args)
    dtypes = cli.apply_precision(model, args.precision)
    with numerical_policy():
        for idx in pending:
            out = generate_question(model, processor, test_rows[idx], args.data_root,
                                    expected_prompt_identity=eval_rows.get(idx, {}).get("prompt_identity"))
            if "skipped" in out:
                append_jsonl(skipped_path, {"index": idx, **out})
            else:
                append_jsonl(rows_path, {"index": idx, "task_name": TASK_NAME, **out,
                                         "precision": args.precision, "precision_dtypes": dtypes})
    return 0


def stage_merge_trajectories(args):
    from evaluation.latent_image_trajectory import load_selection
    out_dir = Path(args.out_dir)
    current = _provenance(args)
    metas = {int(p.name.split("-")[1].split(".")[0]): json.loads(p.read_text())
             for p in out_dir.glob("shard-*.meta.json")}
    num_shards = {m["num_shards"] for m in metas.values()}
    if len(num_shards) != 1 or set(metas) != set(range(next(iter(num_shards)))):
        raise ValueError(f"shard meta files are incomplete or inconsistent (found shards {sorted(metas)})")
    for shard, meta in sorted(metas.items()):
        changed = sorted(k for k in current if meta.get(k) != current[k] and k not in ("commit", "git_status"))
        if changed or (meta["commit"], meta["git_status"]) != (metas[0]["commit"], metas[0]["git_status"]):
            raise ValueError(f"shard meta {shard} differs from the current execution or shard 0 on {changed or 'commit'}")

    rows, skipped = [], []
    for shard in sorted(metas):
        rows_path, skipped_path, _ = _shard_paths(out_dir, shard)
        rows += read_jsonl(rows_path)
        skipped += read_jsonl(skipped_path)
    selection = load_selection(args.selection)
    seen = [int(r["index"]) for r in rows + skipped]
    duplicates = sorted({i for i in seen if seen.count(i) > 1})
    missing = sorted(set(selection) - set(seen))
    extra = sorted(set(seen) - set(selection))
    problems = [f"{name} indices {items}" for name, items in
                (("duplicate", duplicates), ("missing", missing), ("unselected", extra)) if items]
    if problems:
        raise ValueError("refusing to merge trajectories: " + "; ".join(problems))

    rows.sort(key=lambda r: int(r["index"]))
    skipped.sort(key=lambda r: int(r["index"]))
    merged_path = out_dir / "merged.jsonl"
    for path, items in ((merged_path, rows), (out_dir / "skipped.jsonl", skipped)):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in items))
        os.replace(tmp, path)
    cli.write_json_atomic(out_dir / "trajectories-manifest.json", {
        **{k: metas[0][k] for k in current}, "num_shards": len(metas), "merged_sha256": cli.sha256(merged_path),
        "skipped": skipped, "count": len(rows), "correct": sum(bool(r["match"]) for r in rows),
        "merge_commit": current["commit"], "created": datetime.now(timezone.utc).isoformat()})
    return 0

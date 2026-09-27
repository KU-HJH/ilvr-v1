"""Fixed step0 high-entropy token selection for the token-level Vision gain.

A frozen step0 model (Stage 1 checkpoint) is run once, teacher-forced and in
eval mode, over every sample. For each supervised text target t its
full-vocabulary predictive entropy at logit row t-1 is computed, and the top
``fraction`` of each response's candidates (ceil, ties by position) is stored
under a digest of the sample's input_ids. Training and the probe look the
selection up by that digest and turn it into M branches:

- ``independent``: one branch per latent block k scoring the selected targets in
  the text segment right after block k, hiding block k only.
- ``cumulative``: one branch scoring every selected target after the first
  block, hiding every latent block (so a target after block k sees none of
  blocks 1..k, and no earlier text keeps a view of them either).

Targets before the first block have no latent to hide and are dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

from transformers.models.qwen2_5_vl.stage2_latent_rollout import parse_latent_spans

from .objective_windows import WindowBlock, WindowSelection

SELECTION_FORMAT = "stage2-step0-entropy-selection-v1"
MASK_SCOPES = ("independent", "cumulative")


def input_ids_key(input_ids: torch.Tensor) -> str:
    """Digest one 1D token sequence independently of device and dtype."""
    if input_ids.ndim != 1:
        raise ValueError("input_ids_key expects a 1D token sequence")
    values = input_ids.detach().to("cpu", torch.int64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def candidate_target_positions(input_ids, labels, attention_mask, text_token_ids) -> tuple[int, ...]:
    """Supervised text targets t >= 1 of one row (latent slots and markers excluded)."""
    text_ids = frozenset(text_token_ids)
    positions = []
    for position in range(1, input_ids.shape[0]):
        label = int(labels[position])
        if not bool(attention_mask[position]) or label == -100:
            continue
        if label != int(input_ids[position]):
            raise ValueError(f"labels disagree with input_ids at position {position}")
        if label in text_ids:
            positions.append(position)
    return tuple(positions)


def select_top_fraction(positions, entropies, fraction: float) -> tuple[int, ...]:
    """Top ceil(fraction * n) positions by entropy, ties broken by earlier position."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if len(positions) != len(entropies):
        raise ValueError("positions and entropies must have equal length")
    if not positions:
        return ()
    count = math.ceil(fraction * len(positions))
    ranked = sorted(zip(positions, entropies), key=lambda item: (-item[1], item[0]))
    return tuple(sorted(position for position, _ in ranked[:count]))


def full_vocabulary_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Natural-log entropy of softmax(logits) over the last dimension, in FP32."""
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
    if not bool(torch.isfinite(entropy).all()):
        raise ValueError("nonfinite step0 entropy")
    return entropy


class VisionTokenSelectionTable:
    """Read-only precomputed selection, keyed by input_ids digest."""

    def __init__(self, payload: dict, *, source: str = "<memory>"):
        if payload.get("format") != SELECTION_FORMAT:
            raise ValueError(f"{source}: not a {SELECTION_FORMAT} file")
        self.meta = payload["meta"]
        self.rows = payload["rows"]
        self.source = source

    @classmethod
    def load(cls, path) -> "VisionTokenSelectionTable":
        path = Path(path)
        raw = path.read_bytes()
        table = cls(json.loads(raw), source=str(path))
        table.sha256 = hashlib.sha256(raw).hexdigest()
        return table

    def selected_targets(self, input_ids: torch.Tensor) -> tuple[int, ...]:
        key = input_ids_key(input_ids)
        row = self.rows.get(key)
        if row is None:
            raise KeyError(f"{self.source}: no step0 selection for input_ids sha256={key}")
        return tuple(row["selected_targets"])


def build_vision_token_units(
    input_ids, labels, attention_mask, text_segments, *, token_ids, latent_size,
    selected_targets, scope: str,
) -> WindowSelection:
    """Turn one row's selected targets into M branches for ``scope``."""
    if scope not in MASK_SCOPES:
        raise ValueError(f"scope must be one of {MASK_SCOPES}")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("token-level Vision gain supports batch_size=1 only")
    ids, row_labels, mask = input_ids[0], labels[0], attention_mask[0]
    spans = parse_latent_spans(ids, token_ids=token_ids, latent_size=latent_size)
    segments = tuple(tuple(segment) for segment in text_segments[0])
    if len(segments) != len(spans) + 1:
        raise ValueError("text_segments count must equal latent blocks plus one")
    for index, span in enumerate(spans):
        if segments[index][1] != span.start_position or segments[index + 1][0] != span.end_position + 1:
            raise ValueError(f"block_index={index}: text_segments do not align with latent markers")

    excluded: Counter[str] = Counter()
    per_block: list[list[int]] = [[] for _ in spans]
    for target in selected_targets:
        if not 1 <= target < ids.shape[0] or not bool(mask[target]):
            raise ValueError(f"selected target {target} is not an active position")
        label = int(row_labels[target])
        if label == -100 or label != int(ids[target]):
            raise ValueError(f"selected target {target} is not a supervised token")
        segment = next((index for index, (start, end) in enumerate(segments) if start <= target < end), None)
        if segment is None:
            raise ValueError(f"selected target {target} is not a text token")
        if segment == 0:
            excluded["pre_first_block_tokens"] += 1
            continue
        per_block[segment - 1].append(target)

    def unit(span, targets, hidden_spans=()):
        targets = sorted(targets)
        return WindowBlock(
            batch_index=0, block_index=spans.index(span),
            start_position=span.start_position, end_position=span.end_position,
            pre_positions=(), post_positions=tuple(target - 1 for target in targets),
            target_ids=tuple(int(row_labels[target]) for target in targets),
            hidden_spans=hidden_spans,
        )

    if scope == "independent":
        blocks = tuple(unit(span, targets) for span, targets in zip(spans, per_block) if targets)
    else:
        targets = [target for group in per_block for target in group]
        hidden = tuple((span.start_position, span.end_position) for span in spans)
        blocks = (unit(spans[0], targets, hidden),) if targets else ()
    return WindowSelection(blocks=blocks, excluded=dict(sorted(excluded.items())))


def token_count(selection: WindowSelection) -> int:
    return sum(len(block.post_positions) for block in selection.blocks)


def step0_selection_for_batch(model, inputs, *, text_token_ids, fraction):
    """Run the frozen step0 F forward and select one row's high-entropy targets."""
    if model.training:
        raise ValueError("step0 selection requires an eval mode model")
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("step0 selection supports batch_size=1 only")
    candidates = candidate_target_positions(
        ids[0].cpu(), inputs["labels"][0].cpu(), inputs["attention_mask"][0].cpu(), text_token_ids)
    clean = {key: value for key, value in inputs.items()
             if key not in ("labels", "stage2_text_segments") and not key.startswith("_stage2_")}
    with torch.no_grad():
        logits = model(**clean).logits[0]
        rows = torch.tensor([position - 1 for position in candidates], dtype=torch.long, device=logits.device)
        entropies = torch.cat([full_vocabulary_entropy(logits.index_select(0, chunk))
                               for chunk in rows.split(1024)]) if candidates else torch.empty(0)
    values = entropies.cpu().tolist()
    selected = select_top_fraction(candidates, values, fraction)
    by_position = dict(zip(candidates, values))
    threshold = min((by_position[position] for position in selected), default=None)
    return {
        "candidates": len(candidates),
        "selected_targets": list(selected),
        "selected_entropy": [by_position[position] for position in selected],
        "threshold": threshold,
        "mean_candidate_entropy": sum(values) / len(values) if values else None,
    }


def _load_step0(checkpoint, base_model, device, dtype):
    from transformers import AutoProcessor

    from peft_common import resolve_qwen2_5_vl_model
    from .attention_backend import load_qwen2_5_vl_with_attention
    from .checkpointing import configure_stage2_metadata, resolve_latent_token_ids
    from .peft import (ALLOWED_STAGE2_TARGET_SUFFIXES, Stage2TuningConfig,
                       load_stage1_dora_for_fresh_stage2)

    processor = AutoProcessor.from_pretrained(checkpoint)
    base = load_qwen2_5_vl_with_attention(
        str(base_model), attention_implementation="sdpa",
        vision_attention_implementation="flash_attention_2",
        model_load_kwargs={"torch_dtype": dtype})
    model = load_stage1_dora_for_fresh_stage2(
        base, processor, checkpoint,
        Stage2TuningConfig("dora", 32, 64, 0.05, ALLOWED_STAGE2_TARGET_SUFFIXES),
        requested_base_model_reference=str(base_model),
    )
    qwen = resolve_qwen2_5_vl_model(model)
    token_ids = resolve_latent_token_ids(processor.tokenizer, qwen.config)
    configure_stage2_metadata(qwen.config, token_ids, latent_size=8, supervise_latent_boundaries=True)
    qwen.config.use_cache = False
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model, processor, token_ids


def main(argv=None) -> int:
    from .data import (Stage2DataCollator, build_prepared_example_length_measure,
                       filter_stage2_examples, load_jsonl_examples, prepare_stage2_example)
    from .objective_windows import build_text_vocabulary
    from peft_common import resolve_qwen2_5_vl_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="step0 Stage 1 checkpoint")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--row-ids", type=Path, help="probe manifest; select only its selected_row_ids")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="drop rows longer than this, as training does (ignored with --row-ids)")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    samples = load_jsonl_examples(args.data)
    device, dtype = torch.device("cuda:0"), torch.bfloat16
    model, processor, token_ids = _load_step0(args.checkpoint, args.base_model, device, dtype)
    qwen = resolve_qwen2_5_vl_model(model)
    vocabulary = build_text_vocabulary(processor.tokenizer, qwen.config, qwen.get_output_embeddings().weight.shape[0])
    if args.row_ids is not None:
        row_ids = json.loads(args.row_ids.read_text())["selected_row_ids"]
        prepared = [(row, prepare_stage2_example(samples[row], image_root=args.image_root, latent_size=8))
                    for row in row_ids]
    else:
        examples = [prepare_stage2_example(sample, image_root=args.image_root, latent_size=8) for sample in samples]
        kept, _ = filter_stage2_examples(examples=examples,
                                         measure_length=build_prepared_example_length_measure(processor),
                                         max_length=args.max_length)
        kept_ids = {id(example) for example in kept}
        prepared = [(row, example) for row, example in enumerate(examples) if id(example) in kept_ids]
    prepared = prepared[args.shard_index::args.num_shards]
    collator = Stage2DataCollator(processor, token_ids, 8, True, enable_stage2_objective=True)
    rows = {}
    for count, (row, example) in enumerate(prepared, 1):
        batch = collator([example])
        key = input_ids_key(batch["input_ids"][0])
        inputs = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
        with torch.autocast(device.type, dtype=dtype):
            record = step0_selection_for_batch(model, inputs, text_token_ids=vocabulary.token_ids,
                                               fraction=args.fraction)
        record["row_ids"] = [row]
        previous = rows.get(key)
        if previous is not None:
            if previous["selected_targets"] != record["selected_targets"]:
                raise ValueError(f"rows {previous['row_ids']} and {row} share input_ids but not a selection")
            previous["row_ids"].append(row)
        else:
            rows[key] = record
        print(json.dumps({"done": count, "of": len(prepared), "row": row, "candidates": record["candidates"],
                          "selected": len(record["selected_targets"])}), flush=True)
    payload = {
        "format": SELECTION_FORMAT,
        "meta": {"checkpoint": str(args.checkpoint), "base_model": str(args.base_model),
                 "data": str(args.data), "data_sha256": data_sha,
                 "row_ids_manifest": None if args.row_ids is None else str(args.row_ids),
                 "max_length": None if args.row_ids is not None else args.max_length,
                 "fraction": args.fraction, "rounding": "ceil", "ties": "earlier position",
                 "entropy": "full-vocabulary softmax entropy (nats, FP32) of logits at target-1",
                 "model_mode": "eval", "dtype": "bfloat16", "language_attention": "sdpa",
                 "vision_attention": "flash_attention_2",
                 "num_shards": args.num_shards, "shard_index": args.shard_index},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream)
    return 0


def merge_selection_shards(paths, output) -> dict:
    """Merge shard files; shards must agree on every meta field but the shard index."""
    payloads = [json.loads(Path(path).read_text()) for path in paths]
    metas = [{k: v for k, v in payload["meta"].items() if k != "shard_index"} for payload in payloads]
    if any(meta != metas[0] for meta in metas) or any(p["format"] != SELECTION_FORMAT for p in payloads):
        raise ValueError("selection shards disagree")
    if sorted(p["meta"]["shard_index"] for p in payloads) != list(range(metas[0]["num_shards"])):
        raise ValueError("selection shards are incomplete")
    rows = {}
    for payload in payloads:
        for key, record in payload["rows"].items():
            if key in rows:
                if rows[key]["selected_targets"] != record["selected_targets"]:
                    raise ValueError(f"shards disagree on input_ids sha256={key}")
                rows[key]["row_ids"] = sorted(rows[key]["row_ids"] + record["row_ids"])
            else:
                rows[key] = record
    merged = {"format": SELECTION_FORMAT, "meta": {**metas[0], "num_shards": 1, "shard_index": 0,
                                                   "merged_from": [str(path) for path in paths]},
              "rows": rows}
    with Path(output).open("x") as stream:
        json.dump(merged, stream)
    return merged


if __name__ == "__main__":
    raise SystemExit(main())

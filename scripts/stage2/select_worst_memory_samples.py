"""Select the retained Stage 2 samples most likely to exhaust GPU memory.

Ranks retained samples (token length <= max length, measured exactly as the
training length filter does) by (1) full token length and (2) the largest
single-image vision patch count from the processor's image_grid_thw. Writes the
union of the top-k of each ranking as a JSONL subset plus a JSON report.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor

from stage2.data import (
    _load_original_images,
    build_prepared_example_length_measure,
    load_jsonl_examples,
    prepare_stage2_example,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.processor)
    measure = build_prepared_example_length_measure(processor)
    samples = load_jsonl_examples(args.data)
    rows = []
    for index, sample in enumerate(samples):
        prepared = prepare_stage2_example(sample, image_root=args.image_root, latent_size=args.latent_size)
        length = measure(prepared)
        grids = processor.image_processor(
            images=_load_original_images(prepared.original_image_paths), return_tensors="pt"
        )["image_grid_thw"]
        patches = [int(t * h * w) for t, h, w in grids.tolist()]
        rows.append({"index": index, "length": length, "max_image_patches": max(patches, default=0),
                     "total_image_patches": sum(patches), "latent_blocks": prepared.latent_block_count})
    retained = [row for row in rows if row["length"] <= args.max_length]
    by_length = sorted(retained, key=lambda row: -row["length"])[: args.top_k]
    by_patches = sorted(retained, key=lambda row: -row["max_image_patches"])[: args.top_k]
    chosen = sorted({row["index"] for row in by_length + by_patches})
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("x") as stream:
        for index in chosen:
            stream.write(json.dumps(samples[index], ensure_ascii=False) + "\n")
    report = {
        "data": str(args.data), "total": len(rows), "retained": len(retained),
        "max_length": args.max_length, "top_k": args.top_k,
        "top_by_length": by_length, "top_by_max_image_patches": by_patches,
        "chosen_indices": chosen,
        "retained_max_length": max(row["length"] for row in retained),
        "retained_max_image_patches": max(row["max_image_patches"] for row in retained),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("total", "retained", "chosen_indices",
                                              "retained_max_length", "retained_max_image_patches")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

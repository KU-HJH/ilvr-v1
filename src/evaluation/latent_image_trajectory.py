"""Rebuild the model's own greedy TEST trajectory as a verified teacher-forcing input."""
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from evaluation.latent_image_layout import Layout, LayoutTokenIds, build_layout
from evaluation.runtime import prepare_evaluation_input


@dataclass(frozen=True)
class Trajectory:
    source_index: int
    category: str
    image_paths: tuple
    images: tuple
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    prompt_length: int
    layout: Layout
    eval_match: bool
    prompt_identity: str


def layout_token_ids(processor, config):
    tok = processor.tokenizer
    return LayoutTokenIds(
        image=config.image_token_id, vision_start=config.vision_start_token_id,
        vision_end=config.vision_end_token_id, im_end=tok.convert_tokens_to_ids("<|im_end|>"),
        latent_start=config.latent_start_token_id, latent_pad=config.latent_pad_token_id,
        latent_end=config.latent_end_token_id)


def load_eval_rows(path):
    rows = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["index"])] = row
    return rows


def load_selection(path):
    return tuple(int(i) for i in json.loads(Path(path).read_text())["selection"]["requested_indices"])


def _open_images(test_row, data_root):
    from eval import open_images
    from multinode_evaluation import resolve_evaluation_image_paths
    paths = resolve_evaluation_image_paths(test_row["image_input"], Path(data_root).resolve())
    relative = tuple(str(Path(p).resolve().relative_to(Path(data_root).resolve())) for p in paths)
    return open_images(paths), relative


def build_trajectory(processor, test_row, eval_row, source_index, data_root, token_ids, *, latent_size=8):
    if int(eval_row["index"]) != source_index:
        raise ValueError("eval row index differs from source index")
    images, relative = _open_images(test_row, data_root)
    prepared = prepare_evaluation_input(processor, test_row, images)
    if (prepared.prompt_identity != eval_row["prompt_identity"]
            or prepared.prompt_token_length != eval_row["prompt_token_length"]):
        raise ValueError(f"prompt identity mismatch for source {source_index}")
    prompt = prepared.tensors["input_ids"]
    suffix = torch.tensor([eval_row["generated_suffix_token_ids"]], dtype=prompt.dtype)
    ids = torch.cat((prompt, suffix), dim=1)
    layout = build_layout(ids[0].tolist(), prompt.shape[1], token_ids, latent_size=latent_size)
    category = Path(relative[0]).parts[1]
    return Trajectory(source_index, category, relative, tuple(images), ids, torch.ones_like(ids),
                      prepared.tensors["pixel_values"], prepared.tensors["image_grid_thw"],
                      prompt.shape[1], layout, bool(eval_row["match"]), prepared.prompt_identity)

"""Deterministic same-grid image corruptions for latent image patching."""
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def processed_size(image, image_processor):
    w, h = image.size
    factor = image_processor.patch_size * image_processor.merge_size
    return smart_resize(h, w, factor=factor, min_pixels=image_processor.min_pixels,
                        max_pixels=image_processor.max_pixels)


def blur_image(image, sigma_fraction=0.08):
    return image.filter(ImageFilter.GaussianBlur(radius=sigma_fraction * min(image.size)))


def mean_image(image):
    mean = tuple(int(round(v)) for v in np.asarray(image, dtype=np.float64).reshape(-1, 3).mean(0))
    return Image.new("RGB", image.size, mean)


def resized_swap(source, target, image_processor):
    h, w = processed_size(target, image_processor)
    return source.resize((w, h), Image.BICUBIC)


def select_swaps(source_index, category, pool, count=4, seed=42):
    candidates = [i for i in pool.get(category, ()) if i != source_index]
    if len(candidates) < count:
        raise ValueError(f"not enough swap candidates for source {source_index} in {category}")
    rng = np.random.default_rng([seed, source_index])
    return tuple(int(i) for i in rng.choice(sorted(candidates), size=count, replace=False))


def pixel_inputs_from(image_processor, image):
    out = image_processor(images=[image], return_tensors="pt")
    return out["pixel_values"], out["image_grid_thw"]


def pixel_inputs(processor, image):
    return pixel_inputs_from(processor.image_processor, image)


def require_same_grid(expected, actual, label):
    if not torch.equal(torch.as_tensor(expected).cpu(), torch.as_tensor(actual).cpu()):
        raise ValueError(f"{label} grid differs: {expected.tolist()} vs {actual.tolist()}")

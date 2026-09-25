import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from evaluation import latent_image_corruptions as lc


@pytest.fixture
def proc():
    return Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=28 * 28 * 1280)


def img(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_blur_and_mean_preserve_size_and_are_deterministic():
    a = img(301, 187)
    assert lc.blur_image(a).size == a.size
    assert np.array_equal(np.asarray(lc.blur_image(a)), np.asarray(lc.blur_image(a)))
    m = np.asarray(lc.mean_image(a))
    assert m.shape == (187, 301, 3) and (m == m[0, 0]).all()


def test_resized_swap_matches_target_grid(proc):
    target, source = img(301, 187), img(640, 480, 1)
    swapped = lc.resized_swap(source, target, proc)
    _, g_target = lc.pixel_inputs_from(proc, target)
    _, g_swap = lc.pixel_inputs_from(proc, swapped)
    lc.require_same_grid(g_target, g_swap, "swap")


def test_grid_mismatch_raises():
    with pytest.raises(ValueError, match="swap grid differs"):
        lc.require_same_grid(torch.tensor([[1, 4, 4]]), torch.tensor([[1, 4, 6]]), "swap")


def test_select_swaps_same_category_deterministic():
    pool = {"deletion": (1, 2, 3, 4, 5, 6), "update": (7, 8)}
    a = lc.select_swaps(3, "deletion", pool)
    assert a == lc.select_swaps(3, "deletion", pool)
    assert len(a) == 4 and 3 not in a and set(a) <= set(pool["deletion"])
    with pytest.raises(ValueError, match="swap candidates"):
        lc.select_swaps(7, "update", pool)

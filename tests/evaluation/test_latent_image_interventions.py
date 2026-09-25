import pytest
import torch
from step_preference.equivalence import _tiny_model
from evaluation.latent_image_layout import LayoutTokenIds, build_layout
from evaluation.latent_image_trajectory import Trajectory
from evaluation import latent_image_interventions as li

T = LayoutTokenIds(80, 81, 83, 84, 91, 90, 92, (70, 71))
PROMPT = [1, 2, 81, 80, 80, 80, 80, 83, 5, 6, 84, 7, 8]
SUFFIX = [10, 11, 91, 90, 90, 92, 13, 14, 91, 90, 90, 92, 15, 70, 71, 16, 84]


def layout():
    return build_layout(PROMPT + SUFFIX, len(PROMPT), T, latent_size=2, control_k=2)


@pytest.fixture(scope="module")
def tiny():
    # Same construction as tests/evaluation/test_vision_block_runtime.py::image_case.
    torch.manual_seed(0)
    config = _tiny_model().config
    config.num_hidden_layers = 2
    config.vision_end_token_id = 83
    config._attn_implementation = "sdpa"
    return type(_tiny_model())(config).eval()


def trajectory(pixel_seed=0):
    ids = torch.tensor([PROMPT + SUFFIX])
    g = torch.Generator().manual_seed(pixel_seed)
    return Trajectory(0, "deletion", ("x",), (), ids, torch.ones_like(ids),
                      torch.randn(4, 12, generator=g), torch.tensor([[1, 2, 2]]), len(PROMPT),
                      layout(), True, "id")


def test_physical_layout_inserts_alt_after_image():
    phys = li.build_physical_layout(layout(), with_alt=True)
    assert phys.clean_image == (3, 4, 5, 6) and phys.alt_image == (7, 8, 9, 10)
    assert phys.length == len(PROMPT + SUFFIX) + 4
    assert phys.physical_of[11] == 15 and phys.logical_of[8] == 4


def test_visibility_routes_sources():
    phys = li.build_physical_layout(layout(), with_alt=True)
    direct = li.build_visibility(phys, li.Condition("direct", "clean", "alt"))
    latent_row, text_row = phys.latent_rows[0], phys.physical_of[layout().pre_text[0][0]]
    assert not direct[latent_row, list(phys.clean_image)].any() and direct[latent_row, list(phys.alt_image)].all()
    assert direct[text_row, list(phys.clean_image)].all() and not direct[text_row, list(phys.alt_image)].any()
    assert not direct[phys.alt_image[-1], list(phys.clean_image)].any()
    with pytest.raises(ValueError, match="alt"):
        li.build_visibility(li.build_physical_layout(layout(), False), li.Condition("x", "alt", "alt"))


def test_random_keys_before_block_and_deterministic():
    sets, matched = li.random_key_sets(layout(), seed=42, source_index=0)
    assert set(sets) == {0, 1}
    assert all(k < layout().blocks[b].start for b, keys in sets.items() for k in keys)
    assert matched == {0: 1.0, 1: 1.0}
    assert sets == li.random_key_sets(layout(), seed=42, source_index=0)[0]


def test_random_keys_capped_when_eligible_is_short():
    # Block right after the prompt: only the two question tokens are eligible for 4 image tokens.
    short = build_layout(PROMPT + [91, 90, 90, 92, 70, 71, 84], len(PROMPT), T, latent_size=2)
    sets, matched = li.random_key_sets(short, seed=42, source_index=0)
    assert sets[0] == (8, 9) and matched[0] == pytest.approx(0.5)


def test_noop_matches_native_and_dual_clean_matches_single(tiny):
    tr = trajectory()
    native = li.run_condition(tiny, tr, li.Condition("reference", native=True), mid_layer=0)
    noop = li.run_condition(tiny, tr, li.Condition("noop"), mid_layer=0)
    alt_pixels = torch.randn(4, 12, generator=torch.Generator().manual_seed(5))
    dual = li.run_condition(tiny, tr, li.Condition("dual_clean"), alt_pixel_values=alt_pixels,
                            alt_grid=tr.image_grid_thw, mid_layer=0)
    for other in (noop, dual):
        torch.testing.assert_close(other.latents, native.latents, rtol=1e-4, atol=1e-5)
        assert {k: v[0] for k, v in other.scores.items()} == {k: v[0] for k, v in native.scores.items()}


def test_total_alt_equals_override_and_knockout_changes_latents(tiny):
    tr = trajectory()
    alt_pixels = torch.randn(4, 12, generator=torch.Generator().manual_seed(5))
    total = li.run_condition(tiny, tr, li.Condition("total", "alt", "alt"), alt_pixel_values=alt_pixels,
                             alt_grid=tr.image_grid_thw, mid_layer=0)
    override = li.run_condition(tiny, tr, li.Condition("swap"), pixel_override=(alt_pixels, tr.image_grid_thw),
                                mid_layer=0)
    torch.testing.assert_close(total.latents, override.latents, rtol=1e-4, atol=1e-5)
    noop = li.run_condition(tiny, tr, li.Condition("noop"), mid_layer=0)
    k_all = li.run_condition(tiny, tr, li.Condition("k_all", latent_source="none"), mid_layer=0)
    assert not torch.allclose(k_all.latents, noop.latents)


def test_head_subset_all_heads_equals_k_all(tiny):
    tr = trajectory()
    heads = tuple((layer, h) for layer in range(tiny.config.num_hidden_layers)
                  for h in range(tiny.config.num_attention_heads))
    subset = li.run_condition(tiny, tr, li.Condition("k_heads", head_subset=heads), mid_layer=0)
    k_all = li.run_condition(tiny, tr, li.Condition("k_all", latent_source="none"), mid_layer=0)
    torch.testing.assert_close(subset.latents, k_all.latents, rtol=1e-4, atol=1e-5)


def test_head_subset_single_head_is_between(tiny):
    tr = trajectory()
    noop = li.run_condition(tiny, tr, li.Condition("noop"), mid_layer=0)
    one = li.run_condition(tiny, tr, li.Condition("k_one", head_subset=((0, 1),)), mid_layer=0)
    assert not torch.allclose(one.latents, noop.latents)

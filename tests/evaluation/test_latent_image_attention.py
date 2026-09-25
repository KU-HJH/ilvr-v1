import numpy as np
import pytest
import torch
from evaluation import latent_image_attention as la
from evaluation.latent_image_layout import (
    IMAGE, SINK, TEMPLATE, CURRENT_BLOCK, CB_SELF, CB_CUR_START, CB_CUR_PREV, CB_OUTSIDE,
)
from step_preference.equivalence import _tiny_model
from tests.evaluation.test_latent_image_interventions import tiny, trajectory  # noqa: F401  (fixture reuse)


def test_query_specs_cover_blocks_and_controls():
    tr = trajectory()
    specs = la.query_specs(tr.layout)
    latent = [s for s in specs if s.group == "latent"]
    assert len(latent) == len(tr.layout.blocks) * (2 + 2)       # start, 2 pads, end (tiny latent_size=2)
    assert {s.slot for s in specs if s.group == "pre_text"} == {"pre"}
    assert all(s.position in tr.layout.post_text[s.block] for s in specs if s.group == "post_text")


def test_capture_reproduces_sdpa_and_partitions_mass(tiny):
    tr = trajectory()
    cap = la.capture_attention(tiny, tr, keep_image_probs=True)
    assert cap.g1_max_rel < 1e-3
    assert cap.g1b_linear_rel < 1e-3
    assert cap.g2_max_abs < 1e-3
    layers = tiny.config.num_hidden_layers
    assert cap.mass.shape[1:] == (layers, tiny.config.num_attention_heads, 7)
    np.testing.assert_allclose(cap.mass.astype(np.float32).sum(-1), 1, atol=2e-3)
    np.testing.assert_allclose(cap.vw_share.astype(np.float32).sum(-1), 1, atol=2e-3)
    assert cap.image_probs.shape == (len(cap.specs), layers, 4)
    rows = [i for i, s in enumerate(cap.specs) if s.group == "latent"]
    assert (cap.mass[rows, :, :, IMAGE] >= 0).all()
    # Independent cross-check of the region partition: image_probs is computed directly
    # from layout.image_positions (not via the codes/index_add_ path used for mass), so
    # agreement here would catch a mislabeled or miscounted IMAGE partition that the
    # index_add_-based G2 check alone cannot (it is tautological about total mass).
    mass_image = cap.mass.astype(np.float32)[..., IMAGE].mean(axis=2)      # mean over heads -> [rows, layers]
    probs_image = cap.image_probs.astype(np.float32).sum(axis=-1)          # [rows, layers]
    np.testing.assert_allclose(mass_image, probs_image, atol=2e-3)


def test_capture_handles_dora_adapter_linearly():
    # G1 (SDPA reproduction) and G1b (o_proj linearity) must still hold when o_proj is
    # replaced by a PEFT DoRA-wrapped linear layer, since DoRA's effective transform at a
    # fixed set of parameters is still a bias-free matrix multiply (the magnitude/direction
    # decomposition only rescales the weight, it does not depend on the input).
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    config = _tiny_model().config
    config.num_hidden_layers = 2
    config.vision_end_token_id = 83
    config._attn_implementation = "sdpa"
    model = type(_tiny_model())(config)
    model = get_peft_model(model, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["model.layers.0.self_attn.o_proj"], use_dora=True))
    model.eval()
    tr = trajectory()
    cap = la.capture_attention(model, tr)
    assert cap.g1_max_rel < 1e-3
    assert cap.g1b_linear_rel < 1e-3


def test_capture_g1_holds_for_bf16_model(tiny):
    # R20: the G1 reproduction multiplies fp32 probabilities by fp32 values (no bf16 cast
    # of probs), so it stays within the 1e-2 gate for a bf16 model.
    import copy
    model = copy.deepcopy(tiny).to(torch.bfloat16)
    cap = la.capture_attention(model, trajectory())
    assert cap.g1_max_rel <= 1e-2


# ---------------------------------------------------------------------------
# Top-1 / top-5 region labels (R25 addendum)
# ---------------------------------------------------------------------------

def _code(*regions):
    return torch.tensor(regions, dtype=torch.int64)


def test_row_topk_picks_argmax_region_including_sink():
    # key 0 (sink) has the largest raw prob -> top1_region is SINK; the second-largest is
    # the image key, so excluding key 0 (nosink) must select it instead.
    p = torch.tensor([[0.7, 0.1, 0.2]])                       # [heads=1, n_visible=3]
    w = p.clone()
    code = _code(SINK, TEMPLATE, IMAGE)
    t1, t1n, t1v, t5, t5n, t5c, t5v = la._row_topk(p, w, code)
    assert t1.tolist() == [SINK]
    assert t1n.tolist() == [IMAGE]
    assert t1v.tolist() == [SINK]


def test_row_topk_nosink_sentinel_when_only_sink_visible():
    # A single visible key (the sink itself, query == 0): the nosink pool is empty, so the
    # nosink outputs are the documented sentinels (255 region, no image in top-5, count 0).
    p = torch.tensor([[1.0], [1.0]])                          # heads=2, n_visible=1
    w = p.clone()
    code = _code(SINK)
    t1, t1n, t1v, t5, t5n, t5c, t5v = la._row_topk(p, w, code)
    assert t1.tolist() == [SINK, SINK]
    assert t1n.tolist() == [255, 255]
    assert t5n.tolist() == [False, False]
    assert t5c.tolist() == [0, 0]


def test_row_topk_top5_counts_image_keys_and_vw_uses_weighted():
    # 6 visible keys: sink, 3 image keys, 2 template keys. Raw probs rank sink highest, then
    # the 3 image keys, then the 2 template keys -> nosink top-5 (k=5) contains all 3 image
    # keys plus the 2 template keys, so top5_image_count == 3. The weighted (vw) row reorders
    # so the argmax is a template key instead of the sink, proving vw uses `w_row`, not `p_row`.
    p = torch.tensor([[0.30, 0.25, 0.20, 0.15, 0.06, 0.04]])   # sink, img, img, img, tmpl, tmpl
    w = torch.tensor([[0.10, 0.20, 0.15, 0.10, 0.40, 0.05]])   # template key 4 now dominates
    code = _code(SINK, IMAGE, IMAGE, IMAGE, TEMPLATE, TEMPLATE)
    t1, t1n, t1v, t5, t5n, t5c, t5v = la._row_topk(p, w, code)
    assert t1.tolist() == [SINK]
    assert t1n.tolist() == [IMAGE]
    assert t1v.tolist() == [TEMPLATE]
    assert t5.tolist() == [True]
    assert t5n.tolist() == [True]
    assert t5c.tolist() == [3]
    assert t5v.tolist() == [True]                              # image keys still in the 5-of-6 vw top-5


def test_row_topk_fewer_than_five_visible_keys():
    # n_visible == 3 (sink + 2 keys): k = min(5, n_visible) == 3 so top-5 degenerates to "any
    # visible key"; nosink pool has 2 keys so k5n == 2 as well.
    p = torch.tensor([[0.5, 0.3, 0.2]])
    w = p.clone()
    code = _code(SINK, TEMPLATE, IMAGE)
    t1, t1n, t1v, t5, t5n, t5c, t5v = la._row_topk(p, w, code)
    assert t5.tolist() == [True]        # image key visible among the (only 3) candidates
    assert t5n.tolist() == [True]
    assert t5c.tolist() == [1]


# ---------------------------------------------------------------------------
# R26: current-block sub-classification (SELF / CUR_START / CUR_PREV / OUTSIDE)
# ---------------------------------------------------------------------------

def _sub(*codes):
    return torch.tensor(codes, dtype=torch.int64)


def test_row_current_block_picks_argmax_subcode_and_nosink_excludes_key0():
    # key 0 (highest raw prob) is OUTSIDE; excluding it (nosink) leaves SELF as the argmax.
    p = torch.tensor([[0.7, 0.2, 0.1]])                       # [heads=1, n_visible=3]
    w = p.clone()
    sub = _sub(CB_OUTSIDE, CB_SELF, CB_CUR_START)
    top1, top1n, top1v, top5, top5n, top5v, mass, vw = la._row_current_block(p, w, sub)
    assert top1.tolist() == [CB_OUTSIDE]
    assert top1n.tolist() == [CB_SELF]
    assert top1v.tolist() == [CB_OUTSIDE]


def test_row_current_block_nosink_sentinel_when_only_key0_visible():
    p = torch.tensor([[1.0], [1.0]])                          # heads=2, n_visible=1
    w = p.clone()
    sub = _sub(CB_SELF)
    top1, top1n, top1v, top5, top5n, top5v, mass, vw = la._row_current_block(p, w, sub)
    assert top1.tolist() == [CB_SELF, CB_SELF]
    assert top1n.tolist() == [la.NOSINK_SENTINEL, la.NOSINK_SENTINEL]
    assert top5n.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]


def test_row_current_block_top5_counts_and_vw_uses_weighted():
    # 6 visible keys: OUTSIDE, 3x SELF/CUR_START/CUR_PREV (one each), 2x OUTSIDE. Raw probs
    # rank key0 (OUTSIDE) highest, then the 3 in-block keys, then the 2 remaining OUTSIDE keys
    # -> nosink top-5 (k=5) contains all 3 in-block keys plus 2 OUTSIDE keys. The weighted (vw)
    # row reorders so the argmax is CUR_PREV instead of the sink, proving vw uses `w_row`.
    p = torch.tensor([[0.30, 0.25, 0.20, 0.15, 0.06, 0.04]])
    w = torch.tensor([[0.10, 0.10, 0.15, 0.40, 0.20, 0.05]])   # index 3 (CUR_PREV) dominates
    sub = _sub(CB_OUTSIDE, CB_SELF, CB_CUR_START, CB_CUR_PREV, CB_OUTSIDE, CB_OUTSIDE)
    top1, top1n, top1v, top5, top5n, top5v, mass, vw = la._row_current_block(p, w, sub)
    assert top1.tolist() == [CB_OUTSIDE]
    assert top1n.tolist() == [CB_SELF]
    assert top1v.tolist() == [CB_CUR_PREV]
    assert top5.tolist() == [[1, 1, 1, 2]]           # SELF, CUR_START, CUR_PREV, 2x OUTSIDE
    assert top5n.tolist() == [[1, 1, 1, 2]]          # same 5 keys once key 0 is excluded (n=5)
    assert top5v.tolist() == [[1, 1, 1, 2]]


def test_row_current_block_fewer_than_five_visible_keys():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    w = p.clone()
    sub = _sub(CB_OUTSIDE, CB_SELF, CB_CUR_START)
    top1, top1n, top1v, top5, top5n, top5v, mass, vw = la._row_current_block(p, w, sub)
    assert top5.tolist() == [[1, 1, 0, 1]]
    assert top5n.tolist() == [[1, 1, 0, 0]]          # only 2 nosink candidates


def test_row_current_block_mass_partitions_raw_probs():
    p = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    w = p.clone()
    sub = _sub(CB_SELF, CB_CUR_START, CB_CUR_PREV, CB_OUTSIDE)
    *_, mass, vw = la._row_current_block(p, w, sub)
    np.testing.assert_allclose(mass.numpy(), [[0.4, 0.3, 0.2, 0.1]], atol=1e-6)
    assert mass[:, :3].sum().item() == pytest.approx(0.9)


def test_capture_topk_fields_shapes_and_nosink_excludes_sink(tiny):
    tr = trajectory()
    cap = la.capture_attention(tiny, tr)
    layers, heads = tiny.config.num_hidden_layers, tiny.config.num_attention_heads
    shape = (len(cap.specs), layers, heads)
    for field in ("top1_region", "top1_region_nosink", "top1_vw_region"):
        arr = getattr(cap, field)
        assert arr.shape == shape and arr.dtype == np.uint8
    for field in ("top5_image", "top5_image_nosink", "top5_image_count", "top5_vw_image"):
        arr = getattr(cap, field)
        assert arr.shape == shape and arr.dtype == np.uint8
    # R25 section 1: nosink must never resolve to the sink region (key 0 is always excluded
    # from the nosink candidate pool once at least one other key is visible; every query here
    # sees more than just the sink since row 0 already includes prompt tokens after position 0).
    assert (cap.top1_region_nosink != SINK).all()
    assert set(np.unique(cap.top1_region_nosink)) <= set(range(7)) | {255}     # valid codes or sentinel only
    # top1_region_nosink == IMAGE must imply top5_image_nosink is also true for that
    # (row, layer, head): the argmax key is trivially inside its own top-5.
    imaged = cap.top1_region_nosink == IMAGE
    assert (cap.top5_image_nosink[imaged] == 1).all()
    assert (cap.top5_image_count[imaged] >= 1).all()


def test_capture_current_block_fields_shapes_sentinels_and_mass_consistency(tiny):
    tr = trajectory()
    cap = la.capture_attention(tiny, tr)
    layers, heads = tiny.config.num_hidden_layers, tiny.config.num_attention_heads
    shape = (len(cap.specs), layers, heads)
    shape5 = shape + (4,)
    for field in ("cb_top1_raw", "cb_top1_nosink", "cb_top1_vw"):
        arr = getattr(cap, field)
        assert arr.shape == shape and arr.dtype == np.uint8
    for field in ("cb_top5_raw", "cb_top5_nosink", "cb_top5_vw"):
        arr = getattr(cap, field)
        assert arr.shape == shape5 and arr.dtype == np.uint8
    for field in ("cb_mass", "cb_vw_share"):
        arr = getattr(cap, field)
        assert arr.shape == shape5 and arr.dtype == np.float16

    latent_rows = [i for i, s in enumerate(cap.specs) if s.group == "latent"]
    other_rows = [i for i, s in enumerate(cap.specs) if s.group != "latent"]
    assert other_rows                       # the trajectory fixture has pre/post-text rows too
    # non-LATENT rows carry only the documented sentinel / zero defaults (R26 brief §1).
    assert (cap.cb_top1_raw[other_rows] == la.NOSINK_SENTINEL).all()
    assert (cap.cb_top1_nosink[other_rows] == la.NOSINK_SENTINEL).all()
    assert (cap.cb_top1_vw[other_rows] == la.NOSINK_SENTINEL).all()
    assert (cap.cb_top5_raw[other_rows] == 0).all()
    assert (cap.cb_mass[other_rows] == 0).all()
    assert (cap.cb_vw_share[other_rows] == 0).all()

    # LATENT rows: the 4 sub-code top-5 counts sum to 5 (every latent query here sees >= 5
    # keys), and cb_mass[...,:3] sums to the existing CURRENT_BLOCK region mass.
    assert (cap.cb_top5_raw[latent_rows].sum(-1) == 5).all()
    cb_inblock_mass = cap.cb_mass[latent_rows].astype(np.float32)[..., :3].sum(-1)
    current_block_mass = cap.mass[latent_rows].astype(np.float32)[..., CURRENT_BLOCK]
    np.testing.assert_allclose(cb_inblock_mass, current_block_mass, atol=2e-3)
    np.testing.assert_allclose(cap.cb_vw_share.astype(np.float32).sum(-1)[latent_rows], 1, atol=2e-3)

    # START rows: only SELF is possible in-block (R26 spec), so nosink top-1 is SELF or
    # OUTSIDE only, and the top-5 counts for CUR_START/CUR_PREV are always 0.
    start_rows = [i for i, s in enumerate(cap.specs) if s.group == "latent" and s.slot == "start"]
    assert start_rows
    assert np.isin(cap.cb_top1_raw[start_rows], [CB_SELF, CB_OUTSIDE]).all()
    assert (cap.cb_top5_raw[start_rows][..., CB_CUR_START] == 0).all()
    assert (cap.cb_top5_raw[start_rows][..., CB_CUR_PREV] == 0).all()

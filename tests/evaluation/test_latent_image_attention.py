import numpy as np
import torch
from evaluation import latent_image_attention as la
from evaluation.latent_image_layout import IMAGE
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

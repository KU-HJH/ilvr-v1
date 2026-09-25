import numpy as np
import torch
from evaluation import latent_image_attention as la
from evaluation.latent_image_layout import IMAGE
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
    assert cap.g2_max_abs < 1e-3
    layers = tiny.config.num_hidden_layers
    assert cap.mass.shape[1:] == (layers, tiny.config.num_attention_heads, 7)
    np.testing.assert_allclose(cap.mass.astype(np.float32).sum(-1), 1, atol=2e-3)
    np.testing.assert_allclose(cap.vw_share.astype(np.float32).sum(-1), 1, atol=2e-3)
    assert cap.image_probs.shape == (len(cap.specs), layers, 4)
    rows = [i for i, s in enumerate(cap.specs) if s.group == "latent"]
    assert (cap.mass[rows, :, :, IMAGE] >= 0).all()

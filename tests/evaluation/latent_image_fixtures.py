"""Synthetic latent-image run directories with the Task 8 on-disk schema."""
import json
from pathlib import Path

import numpy as np

LAYERS, HEADS, HIDDEN = 4, 2, 6
KNOCKOUT = ("reference", "noop_repeat", "k_all", "k_heads", "k_random")
PATCH = tuple(f"{k}_{n}" for k in ("blur", "mean", "swap") for n in ("dual_clean", "total", "direct", "indirect"))
CANDS = ("cand1", "cand2", "cand3", "cand4")


def write_synthetic_run(run_dir, questions=3, blocks=2, with_freegen=False, image_boost=0.3):
    run_dir, rng = Path(run_dir), np.random.default_rng(0)
    (run_dir / "manifest.json").write_text(json.dumps({"inputs": {}, "commit": "c"}))
    (run_dir / "heads.json").write_text(json.dumps({"heads": [[0, 1]], "share": np.zeros((LAYERS, HEADS)).tolist()}))
    for sub in ("attention", "interventions"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    for q in range(questions):
        specs = []
        for b in range(blocks):
            specs += [{"position": 0, "group": "latent", "block": b, "slot": s}
                      for s in ("start", "pad1", "pad2", "end")]
            specs += [{"position": 0, "group": g, "block": b, "slot": s}
                      for g, s in (("pre_text", "pre"), ("post_text", "post"))]
        rows = len(specs)
        share = rng.dirichlet(np.ones(7), size=(rows, LAYERS, HEADS)).astype(np.float16)
        for r, spec in enumerate(specs):          # latent pads look more at the image (region 2)
            if spec["slot"].startswith("pad"):
                share[r, :, :, 2] += np.float16(image_boost)
        np.savez(run_dir / "attention" / f"q{q}.npz", mass=share, vw_share=share,
                 enrichment=rng.random((rows, LAYERS, HEADS)).astype(np.float32),
                 image_grid_thw=np.array([[1, 2, 2]]))
        (run_dir / "attention" / f"q{q}.json").write_text(json.dumps(
            {"specs": specs, "g1": 1e-3, "g1b": 5e-4, "g2": 1e-4, "g5": {"excluded": False, "rate": 1.0},
             "category": "deletion", "n_blocks": blocks, "eval_match": bool(q % 2)}))
        names = KNOCKOUT + PATCH + CANDS
        out = {key: {} for key in ("latent_cos", "latent_cos_pads", "post_dnll", "post_flip",
                                   "answer_dnll", "answer_flip", "control_cos")}
        for name in names:
            out["latent_cos"][name] = rng.random(blocks).tolist()
            out["latent_cos_pads"][name] = rng.random((blocks, 2)).tolist()
            out["post_dnll"][name] = rng.normal(size=blocks).tolist()
            out["post_flip"][name] = rng.random(blocks).tolist()
            out["answer_dnll"][name] = float(rng.normal())
            out["answer_flip"][name] = float(rng.random())
            out["control_cos"][name] = float(rng.random() * 0.1)
        out.update(random_matched={str(b): 1.0 for b in range(blocks)}, answer_status="ok",
                   gates={"G3": {"pass": True}, "G4": {"pass": True}, "G5": {"excluded": False, "rate": 1.0}})
        (run_dir / "interventions" / f"q{q}.json").write_text(json.dumps(out))
        image = rng.normal(size=(5, HIDDEN))
        np.savez(run_dir / "interventions" / f"q{q}.npz",
                 mid_body=(image[:, None] + 0.01 * rng.normal(size=(5, blocks, HIDDEN))).astype(np.float32),
                 final_body=rng.normal(size=(5, blocks, HIDDEN)).astype(np.float32),
                 image_mid=image.astype(np.float32), image_final=rng.normal(size=(5, HIDDEN)).astype(np.float32))
    if with_freegen:
        for mode in ("causal_mask", "latent_mask"):
            (run_dir / "freegen" / mode).mkdir(parents=True, exist_ok=True)
            for q in range(questions):
                (run_dir / "freegen" / mode / f"q{q}.json").write_text(
                    json.dumps({"match": mode == "causal_mask" or q == 0}))

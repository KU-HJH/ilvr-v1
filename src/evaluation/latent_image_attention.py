"""Descriptive attention of latent and control queries over input regions (analysis ③)."""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import numpy as np
import torch

from evaluation.latent_image_interventions import Condition, run_condition
from evaluation.latent_image_layout import IMAGE, REGION_NAMES, query_regions
from stage2.peft import resolve_stage2_qwen_model
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv


@dataclass(frozen=True)
class QuerySpec:
    position: int
    group: str
    block: int
    slot: str


@dataclass
class AttentionCapture:
    specs: tuple
    mass: np.ndarray
    vw_share: np.ndarray
    enrichment: np.ndarray
    image_probs: object
    g1_max_rel: float
    g1b_linear_rel: float
    g2_max_abs: float


def query_specs(layout):
    specs = []
    for b in layout.blocks:
        specs.append(QuerySpec(b.start, "latent", b.index, "start"))
        specs.extend(QuerySpec(p, "latent", b.index, f"pad{i + 1}") for i, p in enumerate(b.pads))
        specs.append(QuerySpec(b.end, "latent", b.index, "end"))
        specs.extend(QuerySpec(p, "pre_text", b.index, "pre") for p in layout.pre_text[b.index])
        specs.extend(QuerySpec(p, "post_text", b.index, "post") for p in layout.post_text[b.index])
    return tuple(specs)


def capture_attention(model, trajectory, *, keep_image_probs=False):
    qwen = resolve_stage2_qwen_model(model)
    layout = trajectory.layout
    specs = query_specs(layout)
    rows = [s.position for s in specs]              # single-image layout: physical == logical
    codes = [torch.tensor(query_regions(layout, p)) for p in rows]
    n_layers, heads = len(qwen.model.layers), qwen.config.num_attention_heads
    mass = np.zeros((len(rows), n_layers, heads, len(REGION_NAMES)), np.float32)
    vw = np.zeros_like(mass)
    enrich = np.zeros((len(rows), n_layers, heads), np.float32)
    image_probs = np.zeros((len(rows), n_layers, len(layout.image_positions)), np.float32) if keep_image_probs else None
    worst = {"g1": 0.0, "g1b": 0.0, "g2": 0.0}

    def layer_hooks(index, attn):
        seen = {}

        def pre(module, args, kwargs):
            seen["x"] = kwargs["hidden_states"].detach()
            seen["pe"] = kwargs["position_embeddings"]
            seen["mask"] = kwargs["attention_mask"]

        def post(module, args, kwargs, output):
            x, (cos, sin), mask = seen["x"], seen["pe"], seen["mask"]
            _, length, _ = x.shape
            hd = attn.head_dim
            q = attn.q_proj(x).view(1, length, -1, hd).transpose(1, 2)
            k = attn.k_proj(x).view(1, length, -1, hd).transpose(1, 2)
            v = attn.v_proj(x).view(1, length, -1, hd).transpose(1, 2)
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling["mrope_section"])
            kk, vv = repeat_kv(k, attn.num_key_value_groups), repeat_kv(v, attn.num_key_value_groups)
            r = torch.tensor(rows, device=x.device)
            scores = (q[:, :, r].float() @ kk.float().transpose(-1, -2)) / math.sqrt(hd)
            scores = scores + mask[:, :, r, :length].float()
            probs = scores.softmax(-1)[0]                                      # [H, R, L]
            # fp32 probs @ fp32 values; only the head outputs are cast back for o_proj.
            out = (probs @ vv[0].float()).transpose(0, 1).reshape(len(rows), -1)
            mine = attn.o_proj(out.to(vv.dtype)).float()
            actual = output[0][0, r].float()
            worst["g1"] = max(worst["g1"], float(((mine - actual).norm(dim=-1) / actual.norm(dim=-1).clamp_min(1e-6)).max()))
            # One head at a time bounds memory at [L, heads*hd] for the 3B model.
            contrib = torch.empty(heads, length, device=x.device)
            total = torch.zeros(length, heads * hd, device=x.device, dtype=v.dtype)
            summed = None
            for h in range(heads):
                block = torch.zeros(length, heads * hd, device=x.device, dtype=v.dtype)
                block[:, h * hd:(h + 1) * hd] = vv[0, h]
                projected = attn.o_proj(block).float()
                contrib[h] = projected.norm(dim=-1)
                summed = projected if summed is None else summed + projected
                total += block
            linear_err = float((attn.o_proj(total).float() - summed).abs().max() / summed.abs().max().clamp_min(1e-6))
            worst["g1b"] = max(worst["g1b"], linear_err)
            weighted = probs * contrib[:, None, :]
            for i, code in enumerate(codes):
                code = code.to(x.device)
                # Independent partition check: the visible-key count from the actual hooked
                # mask (head 0; Condition("noop") has no per-head mask) must equal the code
                # count, and no key beyond the code range may be visible. Without this,
                # index_add_ trivially makes m.sum() == p_row.sum() regardless of whether the
                # codes are the right labels or cover the right keys.
                visible = torch.isfinite(mask[0, 0, r[i], :length])
                n_visible = int(visible.sum())
                if n_visible != code.numel() or bool(visible[code.numel():].any()):
                    worst["g2"] = max(worst["g2"], float(abs(n_visible - code.numel())) + 1.0)
                p_row, w_row = probs[:, i, :code.numel()], weighted[:, i, :code.numel()]
                m = torch.zeros(heads, len(REGION_NAMES), device=x.device).index_add_(1, code, p_row)
                w = torch.zeros_like(m).index_add_(1, code, w_row)
                w = w / w.sum(-1, keepdim=True)
                worst["g2"] = max(worst["g2"], float((m.sum(-1) - 1).abs().max()),
                                  float((p_row.sum(-1) - probs[:, i].sum(-1)).abs().max()))
                mass[i, index], vw[i, index] = m.cpu().numpy(), w.cpu().numpy()
                n_img = int((code == IMAGE).sum())
                enrich[i, index] = (m[:, IMAGE] / (n_img / max(n_visible, 1))).cpu().numpy()
                if image_probs is not None:
                    image_probs[i, index] = probs[:, i, list(layout.image_positions)].mean(0).cpu().numpy()
            seen.clear()
        return pre, post

    @contextmanager
    def observer():
        handles = []
        try:
            for index, layer in enumerate(qwen.model.layers):
                pre, post = layer_hooks(index, layer.self_attn)
                handles.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
                handles.append(layer.self_attn.register_forward_hook(post, with_kwargs=True))
            yield
        finally:
            for handle in handles:
                handle.remove()

    run_condition(model, trajectory, Condition("noop"), final_observer=observer, mid_layer=0)
    return AttentionCapture(specs, mass.astype(np.float16), vw.astype(np.float16), enrich, image_probs,
                            worst["g1"], worst["g1b"], worst["g2"])

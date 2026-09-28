"""Fixed scalar math and validated settings for the Stage 2 objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


def _require_finite_tensor(value: torch.Tensor, location: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{location} must be a tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"nonfinite value at {location}")


def _require_finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    return result


@dataclass(frozen=True)
class Stage2ObjectiveConfig:
    """Serializable opt-in configuration for the auxiliary Stage 2 losses."""

    enable_stage2_objective: bool = False
    lambda_dyn: float = 0.0
    lambda_vis: float = 0.0
    reference_model_path: str | None = None
    gamma_H: float = 0.0
    m_V: float = 0.0
    entropy_eps: float = 1e-6
    pre_window: int = 1
    post_window: int = 1
    stage2_m_execution: str = "full_replay"
    vision_token_reduction: str = "score_mean"
    # "post_window": first post_window targets after each block (historical).
    # "step0_entropy": precomputed per-response high-entropy targets, one global token mean.
    vision_token_selection: str = "post_window"
    vision_selection_path: str | None = None
    vision_selection_sha256: str | None = None
    vision_mask_scope: str = "independent"
    # Cut full-replay M forwards at the last scored predictor when score-preserving (ours1).
    stage2_m_prefix_cut: bool = False

    @classmethod
    def from_args(cls, args) -> "Stage2ObjectiveConfig":
        return cls(
            enable_stage2_objective=args.enable_stage2_objective,
            lambda_dyn=args.lambda_dyn,
            lambda_vis=args.lambda_vis,
            reference_model_path=args.reference_model_path,
            gamma_H=args.gamma_H,
            m_V=args.m_V,
            entropy_eps=args.entropy_eps,
            pre_window=args.pre_window,
            post_window=args.post_window,
            stage2_m_execution=getattr(args, "stage2_m_execution", "full_replay"),
            vision_token_reduction=getattr(args, "vision_token_reduction", "score_mean"),
            vision_token_selection=getattr(args, "vision_token_selection", "post_window"),
            vision_selection_path=getattr(args, "vision_selection_path", None),
            vision_selection_sha256=getattr(args, "vision_selection_sha256", None),
            vision_mask_scope=getattr(args, "vision_mask_scope", "independent"),
            stage2_m_prefix_cut=getattr(args, "stage2_m_prefix_cut", False),
        )

    @property
    def active(self) -> bool:
        return bool(
            self.enable_stage2_objective
            and (self.lambda_dyn > 0.0 or self.lambda_vis > 0.0)
        )

    def to_dict(self) -> dict[str, bool | float | int | str | None]:
        return asdict(self)

    def validate(self, stage: str, supervise_latent_boundaries: bool) -> None:
        if not isinstance(self.enable_stage2_objective, bool):
            raise ValueError("enable_stage2_objective must be boolean")
        for name in ("lambda_dyn", "lambda_vis", "gamma_H", "m_V", "entropy_eps"):
            value = _require_finite_number(getattr(self, name), name)
            if name.startswith("lambda_") and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.entropy_eps <= 0.0:
            raise ValueError("entropy_eps must be positive")
        for name in ("pre_window", "post_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.reference_model_path is not None and (
            not isinstance(self.reference_model_path, str)
            or not self.reference_model_path.strip()
        ):
            raise ValueError("reference_model_path must be an unset or non-empty string")
        if self.stage2_m_execution not in ("full_replay", "packed_suffix"):
            raise ValueError(
                "stage2_m_execution must be 'full_replay' or 'packed_suffix'"
            )
        if self.vision_token_reduction not in ("score_mean", "token_loss_mean"):
            raise ValueError(
                "vision_token_reduction must be 'score_mean' or 'token_loss_mean'"
            )
        if not isinstance(self.stage2_m_prefix_cut, bool):
            raise ValueError("stage2_m_prefix_cut must be boolean")
        if self.vision_token_selection not in ("post_window", "step0_entropy"):
            raise ValueError("vision_token_selection must be 'post_window' or 'step0_entropy'")
        if self.vision_mask_scope not in ("independent", "cumulative"):
            raise ValueError("vision_mask_scope must be 'independent' or 'cumulative'")
        if self.vision_token_selection == "step0_entropy":
            if not self.vision_selection_path or not self.vision_selection_sha256:
                raise ValueError("step0_entropy selection requires vision_selection_path and vision_selection_sha256")
            if self.lambda_dyn != 0.0:
                raise ValueError("step0_entropy selection requires lambda_dyn=0")
            if self.vision_token_reduction != "token_loss_mean":
                raise ValueError("step0_entropy selection requires vision_token_reduction=token_loss_mean")
        elif self.vision_mask_scope != "independent" or self.vision_selection_path or self.vision_selection_sha256:
            raise ValueError("vision_mask_scope and vision_selection_* require step0_entropy selection")
        if not self.enable_stage2_objective:
            return
        if stage != "stage2":
            raise ValueError("enable_stage2_objective is valid only for Stage 2")
        if not supervise_latent_boundaries:
            raise ValueError("Stage 2 objective requires supervised latent boundaries")
        if self.lambda_dyn > 0.0 and self.reference_model_path is None:
            raise ValueError("reference_model_path is required when lambda_dyn is positive")


def entropy_transition(
    h_pre: torch.Tensor,
    h_post: torch.Tensor,
    ref_pre: torch.Tensor,
    ref_post: torch.Tensor,
    *,
    eps: float,
    gamma_H: float,
) -> dict[str, torch.Tensor]:
    """Apply the fixed entropy-ratio transition independently per block."""
    eps = _require_finite_number(eps, "eps")
    gamma_H = _require_finite_number(gamma_H, "gamma_H")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    values = {
        "h_pre": h_pre,
        "h_post": h_post,
        "ref_pre": ref_pre,
        "ref_post": ref_post,
    }
    for location, value in values.items():
        _require_finite_tensor(value, location)
    if not all(value.shape == h_pre.shape for value in values.values()):
        raise ValueError("entropy_transition tensors must have identical shapes")

    reference_pre = ref_pre.detach()
    reference_post = ref_post.detach()
    r_pre = (h_pre + eps) / (reference_pre + eps)
    r_post = (h_post + eps) / (reference_post + eps)
    difference = r_pre - r_post
    loss_dyn = -torch.sigmoid(difference - gamma_H)
    for location, value in (
        ("r_pre", r_pre),
        ("r_post", r_post),
        ("D", difference),
        ("loss_dyn", loss_dyn),
    ):
        _require_finite_tensor(value, location)
    return {
        "r_pre": r_pre,
        "r_post": r_post,
        "D": difference,
        "loss_dyn": loss_dyn,
    }


def vision_gain(
    s_f: torch.Tensor,
    s_m: torch.Tensor,
    *,
    m_V: float,
) -> dict[str, torch.Tensor]:
    """Apply the fixed counterfactual margin with an explicitly detached M score."""
    margin = _require_finite_number(m_V, "m_V")
    _require_finite_tensor(s_f, "s_f")
    _require_finite_tensor(s_m, "s_m")
    if s_f.shape != s_m.shape:
        raise ValueError("s_f and s_m must have identical shapes")
    gain = s_f - s_m.detach()
    loss_vis = F.softplus(margin - gain)
    _require_finite_tensor(gain, "G")
    _require_finite_tensor(loss_vis, "loss_vis")
    return {"S_F": s_f, "S_M": s_m.detach(), "G": gain, "loss_vis": loss_vis}


def vision_gain_tokens(
    s_f: torch.Tensor,
    s_m: torch.Tensor,
    *,
    m_V: float,
) -> dict[str, torch.Tensor]:
    """Apply the counterfactual margin per target token, then average the losses."""
    margin = _require_finite_number(m_V, "m_V")
    _require_finite_tensor(s_f, "s_f")
    _require_finite_tensor(s_m, "s_m")
    if s_f.shape != s_m.shape:
        raise ValueError("s_f and s_m must have identical shapes")
    if s_f.numel() == 0:
        raise ValueError("vision_gain_tokens requires at least one token")
    token_gain = s_f - s_m.detach()
    loss_vis = F.softplus(margin - token_gain).mean()
    gain = token_gain.mean()
    _require_finite_tensor(gain, "G")
    _require_finite_tensor(loss_vis, "loss_vis")
    return {
        "S_F": s_f.mean(),
        "S_M": s_m.detach().mean(),
        "G": gain,
        "loss_vis": loss_vis,
    }


def vision_gain_token_terms(
    s_f: torch.Tensor,
    s_m: torch.Tensor,
    *,
    m_V: float,
) -> dict[str, torch.Tensor]:
    """Per-token G_t = S_F,t - sg(S_M,t) and softplus(m_V - G_t), unreduced."""
    margin = _require_finite_number(m_V, "m_V")
    _require_finite_tensor(s_f, "s_f")
    _require_finite_tensor(s_m, "s_m")
    if s_f.shape != s_m.shape or s_f.ndim != 1 or s_f.numel() == 0:
        raise ValueError("s_f and s_m must be equal nonempty vectors")
    gain = s_f - s_m.detach()
    loss_vis = F.softplus(margin - gain)
    _require_finite_tensor(gain, "G")
    _require_finite_tensor(loss_vis, "loss_vis")
    return {"S_F": s_f, "S_M": s_m.detach(), "G": gain, "loss_vis": loss_vis}


def text_entropy(logits: torch.Tensor, text_token_ids: tuple[int, ...]) -> torch.Tensor:
    """Return natural-log entropy after text-only vocabulary renormalization."""
    if not isinstance(logits, torch.Tensor) or logits.ndim < 1:
        raise ValueError("logits must be a tensor with an output-vocabulary dimension")
    if not text_token_ids:
        raise ValueError("text_token_ids must not be empty")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in text_token_ids
    ):
        raise ValueError("text_token_ids must contain integers")
    if len(set(text_token_ids)) != len(text_token_ids):
        raise ValueError("text_token_ids must be unique")
    if min(text_token_ids) < 0 or max(text_token_ids) >= logits.shape[-1]:
        raise ValueError("text_token_ids must resolve inside the output vocabulary")
    indices = torch.tensor(text_token_ids, dtype=torch.long, device=logits.device)
    selected = logits.index_select(-1, indices)
    _require_finite_tensor(selected, "logits selected for text entropy")
    log_probabilities = F.log_softmax(selected.float(), dim=-1)
    entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
    _require_finite_tensor(entropy, "text entropy")
    return entropy


def target_log_likelihood(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Return per-row teacher-forced log likelihood over the full output vocabulary."""
    if not isinstance(logits, torch.Tensor) or logits.ndim < 1:
        raise ValueError("logits must be a tensor with an output-vocabulary dimension")
    if not isinstance(target_ids, torch.Tensor):
        raise TypeError("target_ids must be a tensor")
    if target_ids.shape != logits.shape[:-1]:
        raise ValueError("target_ids shape must match the non-vocabulary logits shape")
    if target_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("target_ids must use an integer dtype")
    if bool((target_ids == -100).any()):
        raise ValueError("target_ids must never contain -100")
    if target_ids.numel() and (
        int(target_ids.min()) < 0 or int(target_ids.max()) >= logits.shape[-1]
    ):
        raise ValueError("target_ids must resolve inside the output vocabulary")
    _require_finite_tensor(logits, "logits for target log likelihood")
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    scores = log_probabilities.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)
    _require_finite_tensor(scores, "target log likelihood")
    return scores

"""OFF/ON scorer for optional visual insertion preference comparisons."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, fields
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from .stage2_latent_rollout import Stage2DecoderCallStats


@dataclass(frozen=True)
class EvidenceScore:
    mean_logp: torch.Tensor
    entropies: torch.Tensor | None


@dataclass(frozen=True)
class InsertionStepScores:
    on: EvidenceScore
    off: EvidenceScore


@dataclass
class InsertionComputeStats:
    clean_prediction_calls: int = 0
    clean_latent_calls: int = 0
    off_prediction_calls: int = 0
    on_prediction_calls: int = 0
    on_latent_calls: int = 0
    checkpoint_decoder_recompute_calls: int = 0
    checkpoint_layer_recompute_calls: int = 0
    reference_calls: int = 0
    reference_score_hits: int = 0
    rejected_dpo_tokens: int = 0
    _clean_decoder_stats: Stage2DecoderCallStats | None = field(default=None, repr=False)

    @property
    def prediction_forward_calls(self):
        return self.clean_prediction_calls + self.off_prediction_calls + self.on_prediction_calls

    @property
    def latent_recurrence_calls(self):
        return self.clean_latent_calls + self.on_latent_calls

    @property
    def checkpoint_recompute_calls(self):
        clean = (0 if self._clean_decoder_stats is None
                 else self._clean_decoder_stats.checkpoint_recompute_calls)
        return clean + self.checkpoint_decoder_recompute_calls + self.checkpoint_layer_recompute_calls

    def counters(self):
        values = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        values["clean_checkpoint_decoder_recompute_calls"] = (
            0 if self._clean_decoder_stats is None else self._clean_decoder_stats.checkpoint_recompute_calls)
        return values


@dataclass(frozen=True)
class Stage2InsertionOutput:
    by_step: dict[tuple[str, int], InsertionStepScores]
    stats: InsertionComputeStats


def score_selected_hidden(hidden, lm_head, target, offset=0, *, compute_entropy=True):
    """Score exactly the target's teacher-forced predictor states."""
    if target.target_ids.numel() == 0 or target.predictor_positions.numel() == 0:
        raise ValueError("insertion targets must be non-empty")
    if target.target_ids.numel() != target.predictor_positions.numel():
        raise ValueError("insertion target IDs and predictor positions must match")
    positions = target.predictor_positions - offset
    if torch.any(positions < 0) or torch.any(positions >= hidden.shape[1]):
        raise ValueError("insertion predictor is outside scored suffix")
    log_probs = lm_head(hidden[0, positions]).float().log_softmax(-1)
    logps = log_probs.gather(-1, target.target_ids[:, None]).squeeze(-1)
    entropies = -(log_probs.exp() * log_probs).sum(-1) if compute_entropy else None
    return EvidenceScore(logps.mean(), entropies)


def _score_key(target):
    return target.sample_id, target.step_id


def score_v_off_insertion(
    *, qwen, plan, base_inputs_embeds, image_grid_thw, request, attention_backend,
    checkpoint_branches=True, compute_stats=None,
):
    """Score one image-free trajectory and ON branches with its immutable prefix KV."""
    if plan.branch_mode != request.branch_mode:
        raise ValueError("insertion plan and request branch modes must match")
    stats = compute_stats if compute_stats is not None else InsertionComputeStats()

    recompute_depth = 0

    @contextmanager
    def recomputing():
        nonlocal recompute_depth
        recompute_depth += 1
        try:
            yield
        finally:
            recompute_depth -= 1

    def checkpoint_contexts():
        return nullcontext(), recomputing()

    def observe_checkpoint(event):
        if event == "layer_recompute":
            stats.checkpoint_layer_recompute_calls += 1

    def decode(kind, **kwargs):
        if kind not in {"off_prediction_calls", "on_prediction_calls", "on_latent_calls"}:
            raise ValueError(f"unknown insertion decoder call kind: {kind}")
        if recompute_depth:
            stats.checkpoint_decoder_recompute_calls += 1
        else:
            setattr(stats, kind, getattr(stats, kind) + 1)
        return qwen.model(
            **kwargs, attention_mask=None, generate_mode=True, return_dict=True,
            stage2_checkpoint_observer=observe_checkpoint,
        )

    off_positions, _ = qwen.get_rope_index(
        input_ids=plan.off.input_ids, attention_mask=torch.ones_like(plan.off.input_ids))
    off_output = decode(
        "off_prediction_calls", inputs_embeds=qwen.model.embed_tokens(plan.off.input_ids),
        position_ids=off_positions, use_cache=False, stage2_capture_kv=True,
        stage2_attention_backend=attention_backend,
    )
    def score_off(hidden):
        flat_scores = []
        for target in plan.off.targets:
            score = score_selected_hidden(hidden, qwen.lm_head, target)
            flat_scores.extend((score.mean_logp, score.entropies))
        return tuple(flat_scores)

    off_flat_scores = (checkpoint(
        score_off, off_output.last_hidden_state, use_reentrant=False,
        preserve_rng_state=True, context_fn=checkpoint_contexts,
    ) if checkpoint_branches else score_off(off_output.last_hidden_state))
    off_scores = {}
    for index, target in enumerate(plan.off.targets):
        key = _score_key(target)
        if key in off_scores:
            raise ValueError("insertion OFF targets must be unique")
        off_scores[key] = EvidenceScore(off_flat_scores[index * 2], off_flat_scores[index * 2 + 1])

    def run_on(seq, embeds, *flat_prefix):
        prefix = tuple(zip(flat_prefix[0::2], flat_prefix[1::2], strict=True))
        grids = image_grid_thw[seq.image_grid_indices]
        positions, _ = qwen.get_rope_index(
            input_ids=seq.input_ids, image_grid_thw=grids,
            attention_mask=torch.ones_like(seq.input_ids))
        if request.stage2_debugging:
            torch.testing.assert_close(positions[..., :seq.prefix_length],
                                       off_positions[..., :seq.prefix_length])
        current = embeds[:, seq.prefix_length:]
        for absolute_slot in seq.latent_positions:
            slot = absolute_slot - seq.prefix_length
            before = decode(
                "on_latent_calls", inputs_embeds=current[:, :slot],
                position_ids=positions[..., seq.prefix_length:absolute_slot],
                stage2_prefix_key_values=prefix,
                stage2_checkpoint_prefix=True,
                stage2_layer_checkpoint_stride=request.on_layer_checkpoint_stride,
                stage2_attention_backend=attention_backend, use_cache=False,
            )
            state = before.last_hidden_state[:, -1:]
            current = torch.cat((current[:, :slot], state, current[:, slot + 1:]), dim=1)
        final = decode(
            "on_prediction_calls", inputs_embeds=current,
            position_ids=positions[..., seq.prefix_length:], stage2_prefix_key_values=prefix,
            stage2_checkpoint_prefix=True,
            stage2_layer_checkpoint_stride=request.on_layer_checkpoint_stride,
            stage2_attention_backend=attention_backend, use_cache=False,
        )
        flat_scores = []
        for target in seq.targets:
            score = score_selected_hidden(final.last_hidden_state, qwen.lm_head,
                                          target, offset=seq.prefix_length)
            flat_scores.extend((score.mean_logp, score.entropies))
        return tuple(flat_scores)

    by_step = {}
    for seq in plan.on:
        embeds = base_inputs_embeds[:, seq.source_positions]
        flat_prefix = tuple(
            tensor[..., :seq.prefix_length, :]
            for pair in off_output.stage2_layer_key_values for tensor in pair
        )
        branch = partial(run_on, seq)
        flat_scores = (checkpoint(
            branch, embeds, *flat_prefix, use_reentrant=False,
            preserve_rng_state=True, context_fn=checkpoint_contexts,
        ) if checkpoint_branches else branch(embeds, *flat_prefix))
        if len(flat_scores) != len(seq.targets) * 2:
            raise RuntimeError("insertion ON scorer returned an invalid score layout")
        for index, target in enumerate(seq.targets):
            key = _score_key(target)
            if key not in off_scores or key in by_step:
                raise ValueError("insertion ON targets must match OFF targets exactly once")
            by_step[key] = InsertionStepScores(
                on=EvidenceScore(flat_scores[index * 2], flat_scores[index * 2 + 1]),
                off=off_scores[key],
            )
    if set(by_step) != set(off_scores):
        raise ValueError("insertion ON targets must cover OFF targets exactly once")
    return Stage2InsertionOutput(by_step=by_step, stats=stats)

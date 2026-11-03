"""Execute gated ON trajectories and score same-GT ON/OFF comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

if TYPE_CHECKING:
    from step_preference.gated import GatedPlan, GatedSequence

from .stage2_insertion import EvidenceScore, InsertionStepScores, score_selected_hidden
from .stage2_latent_rollout import Stage2DecoderCallStats, Stage2LatentStates


@dataclass
class GatedComputeStats:
    """Graph-free live counters; replay witnesses remain valid after backward."""

    clean_backend: str
    on_backend: str
    clean_forward_calls: int = 0
    off_forward_calls: int = 0
    off_checkpoint_recompute_calls: int = 0
    on_forward_calls: int = 0
    clean_forward_query_tokens: int = 0
    on_forward_query_tokens: int = 0
    _forward_stats: list[Stage2DecoderCallStats] = field(default_factory=list, repr=False)

    @property
    def checkpoint_recompute_calls(self) -> int:
        return self.off_checkpoint_recompute_calls + sum(
            s.checkpoint_recompute_calls for s in self._forward_stats
        )

    def counters(self):
        # Backend identities are strings, not distributed numeric counters.
        return {name: getattr(self, name) for name in (
            "clean_forward_calls", "off_forward_calls", "on_forward_calls",
            "clean_forward_query_tokens", "on_forward_query_tokens",
            "checkpoint_recompute_calls",
        )}


@dataclass(frozen=True)
class GatedScores:
    by_step: dict[tuple[str, int], InsertionStepScores]
    stats: GatedComputeStats


@dataclass(frozen=True)
class GatedRun:
    hidden_states: torch.Tensor
    latent_states: Stage2LatentStates
    decoder_stats: Stage2DecoderCallStats


def run_gated_sequence(
    *,
    qwen,
    sequence: GatedSequence,
    base_inputs_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    backend: str,
    attention_backend: str,
    retain_latent_grads: bool = False,
    trim_to_targets: bool = False,
) -> GatedRun:
    """Run an ON coordinate plan without rebuilding its image embeddings."""
    source_positions = sequence.sequence.source_positions.to(base_inputs_embeds.device)
    on_input_ids = sequence.sequence.input_ids.to(base_inputs_embeds.device)
    visibility = sequence.visibility
    if trim_to_targets and backend == "differentiable_cache" and sequence.sequence.targets:
        # Only predictor hidden states contribute to this branch's GT scores.
        # Causal suffixes cannot affect them; independent ON otherwise decodes
        # every later step in the sample again for each selected block.
        stop = 1 + max(int(target.predictor_positions.max())
                       for target in sequence.sequence.targets)
        source_positions = source_positions[:stop]
        on_input_ids = on_input_ids[:, :stop]
        visibility = replace(
            visibility, sequence_length=stop,
            image_positions=tuple(p for p in visibility.image_positions if p < stop),
            producer_positions=tuple(p for p in visibility.producer_positions if p < stop),
            active_start_positions=tuple(p for p in visibility.active_start_positions if p < stop),
        )
    embeds = base_inputs_embeds[:, source_positions]
    mask = torch.ones_like(on_input_ids, device=base_inputs_embeds.device)
    image_grid_indices = sequence.sequence.image_grid_indices.to(image_grid_thw.device)
    on_image_grid_thw = image_grid_thw[image_grid_indices].to(on_input_ids.device)
    positions, _ = qwen.get_rope_index(
        on_input_ids,
        image_grid_thw=on_image_grid_thw,
        attention_mask=mask,
    )
    positions = positions.to(base_inputs_embeds.device)
    decoder = partial(
        qwen.model,
        generate_mode=True,
        stage2_visibility=visibility,
        stage2_attention_backend=attention_backend,
    )

    common = dict(
        language_model=decoder,
        base_inputs_embeds=embeds,
        attention_mask=mask,
        full_position_ids=positions,
        latent_spans=sequence.latent_spans,
        retain_latent_grads=retain_latent_grads,
    )
    if backend == "differentiable_cache":
        result = qwen.stage2_cache_rollout_engine.rollout(
            **common,
            stage2_cache_activation_checkpointing=qwen.config.stage2_cache_activation_checkpointing,
        )
        hidden = result.sequence_hidden_states
    elif backend == "no_cache":
        result = qwen.stage2_rollout_engine.rollout(
            **common,
            latent_activation_checkpointing=qwen.config.stage2_latent_activation_checkpointing,
        )
        invoked = False

        def final_decode(final_embeds):
            nonlocal invoked
            stats = result.decoder_call_stats
            if invoked:
                stats.checkpoint_recompute_calls += 1
                stats.recompute_query_tokens += final_embeds.shape[1]
            else:
                stats.clean_forward_calls += 1
                stats.forward_query_tokens += final_embeds.shape[1]
                invoked = True
            return decoder(
                inputs_embeds=final_embeds, attention_mask=mask,
                position_ids=positions, use_cache=False, return_dict=True,
            ).last_hidden_state

        # Checkpoint the final decoder only. Latent prefixes already have their
        # own checkpoints; replaying the whole trajectory would duplicate them.
        with set_checkpoint_early_stop(False):
            hidden = checkpoint(final_decode, result.final_inputs_embeds,
                                use_reentrant=False, preserve_rng_state=True)
    else:
        raise ValueError("unsupported gated rollout backend")
    return GatedRun(hidden, result.latent_states, result.decoder_call_stats)


def score_gated_preference(
    *, qwen, plan: GatedPlan, base_inputs_embeds, image_grid_thw,
    backend: str, attention_backend: str, compute_entropy: bool = True,
) -> GatedScores:
    """Score one OFF pass and isolated ON branches with RNG-preserving replay."""
    stats = GatedComputeStats(clean_backend=backend, on_backend=backend)
    if not plan.off.targets:
        return GatedScores({}, stats)
    off_ids = plan.off.input_ids.to(base_inputs_embeds.device)
    off_sources = plan.off.source_positions.to(base_inputs_embeds.device)
    off_embeds = base_inputs_embeds[:, off_sources]
    off_mask = torch.ones_like(off_ids)
    off_grid_indices = plan.off.image_grid_indices.to(image_grid_thw.device)
    off_image_grid_thw = image_grid_thw[off_grid_indices].to(off_ids.device)
    off_positions, _ = qwen.get_rope_index(
        off_ids,
        image_grid_thw=off_image_grid_thw,
        attention_mask=off_mask,
    )
    off_invoked = False

    def off_decode(embeds):
        nonlocal off_invoked
        if off_invoked:
            stats.off_checkpoint_recompute_calls += 1
        else:
            stats.off_forward_calls += 1
            off_invoked = True
        return qwen.model(
            inputs_embeds=embeds, attention_mask=off_mask,
            position_ids=off_positions, use_cache=False, generate_mode=True,
            return_dict=True,
        ).last_hidden_state

    if qwen.config.stage2_cache_activation_checkpointing:
        # OFF can retain the full image while ON builds several differentiable
        # cache branches. Do not keep all decoder-layer activations alive beside
        # those branches; replay this one decoder pass during backward instead.
        with set_checkpoint_early_stop(False):
            off_hidden = checkpoint(
                off_decode, off_embeds, use_reentrant=False,
                preserve_rng_state=True,
            )
    else:
        off_hidden = off_decode(off_embeds)

    def selected_scores(hidden, targets):
        flat = []
        for target in targets:
            score = score_selected_hidden(hidden, qwen.lm_head, target,
                                          compute_entropy=compute_entropy)
            flat.extend((score.mean_logp, score.entropies))
        return tuple(flat)

    off_flat = checkpoint(
        partial(selected_scores, targets=plan.off.targets), off_hidden,
        use_reentrant=False, preserve_rng_state=True,
    )
    off_scores = {
        (target.sample_id, target.step_id): EvidenceScore(off_flat[2 * i], off_flat[2 * i + 1])
        for i, target in enumerate(plan.off.targets)
    }
    if len(off_scores) != len(plan.off.targets):
        raise ValueError("gated OFF targets must be unique")

    def run_branch(sequence, embeds):
        # Each branch owns its cache. Recompute only individual decoder calls,
        # never an additional outer replay of the complete ON trajectory.
        run = run_gated_sequence(
            qwen=qwen, sequence=sequence, base_inputs_embeds=embeds,
            image_grid_thw=image_grid_thw, backend=backend,
            attention_backend=attention_backend,
            trim_to_targets=True,
        )
        witness = run.decoder_stats
        stats._forward_stats.append(witness)
        stats.on_forward_calls += witness.clean_forward_calls
        stats.on_forward_query_tokens += witness.forward_query_tokens
        return checkpoint(partial(selected_scores, targets=sequence.sequence.targets),
                          run.hidden_states, use_reentrant=False, preserve_rng_state=True)

    by_step = {}
    for sequence in plan.on:
        flat = run_branch(sequence, base_inputs_embeds)
        for i, target in enumerate(sequence.sequence.targets):
            key = (target.sample_id, target.step_id)
            if key not in off_scores or key in by_step:
                raise ValueError("gated ON targets must match OFF targets exactly once")
            by_step[key] = InsertionStepScores(
                on=EvidenceScore(flat[2 * i], flat[2 * i + 1]), off=off_scores[key],
            )
    if set(by_step) != set(off_scores):
        raise ValueError("gated ON targets must cover OFF targets exactly once")
    return GatedScores(by_step, stats)

"""Structural types and parser for Stage 2 latent token blocks."""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.utils.checkpoint

from .stage2_objective import ObjectiveForwardRequest, build_objective_attention_mask

from .stage2_prefix_cache import (
    DifferentiablePrefixState,
    OrderedCacheTensors as _OrderedCacheTensors,
    _flatten_dynamic_cache,
    _rebuild_dynamic_cache,
    _tensor_is_differentiable,
    rebuild_stage2_dynamic_cache,
)


STAGE2_ROLLOUT_BACKENDS = frozenset({"no_cache", "differentiable_cache"})
_FULL_CACHE_CHECKPOINT_SEQUENCE_THRESHOLD = 4096


def normalize_stage2_rollout_backend(
    value: object,
) -> Literal["no_cache", "differentiable_cache"]:
    """Require one rollout implementation with stable checkpoint meaning."""
    if not isinstance(value, str) or value not in STAGE2_ROLLOUT_BACKENDS:
        raise ValueError(
            "Stage 2 rollout backend must be no_cache or differentiable_cache"
        )
    return value


def stage2_prefix_dropout_semantics(
    backend: str,
) -> Literal["resample_per_prefix", "reuse_cached_prefix"]:
    """Return the dropout behavior implied by a rollout implementation."""
    backend = normalize_stage2_rollout_backend(backend)
    return (
        "reuse_cached_prefix"
        if backend == "differentiable_cache"
        else "resample_per_prefix"
    )


class LatentStructureError(ValueError):
    """Raised when a token scaffold does not contain valid latent blocks."""


@dataclass(frozen=True)
class LatentTokenIds:
    pad_token_id: int
    start_token_id: int
    end_token_id: int


_LATENT_CONFIG_FIELDS = {
    "pad_token_id": ("latent_pad_token_id", "latent_token_id"),
    "start_token_id": ("latent_start_token_id", "latent_start_id"),
    "end_token_id": ("latent_end_token_id", "latent_end_id"),
}


def validate_stage2_config(
    config,
    *,
    fallback_token_ids: LatentTokenIds | None = None,
    require_canonical_ids: bool = True,
    require_latent_size: bool = True,
) -> tuple[LatentTokenIds, int | None]:
    """Validate Stage 2 structural metadata without requiring a tokenizer."""
    resolved = {}
    for result_field, (canonical_field, legacy_field) in _LATENT_CONFIG_FIELDS.items():
        canonical_value = getattr(config, canonical_field, None)
        legacy_value = getattr(config, legacy_field, None)
        if require_canonical_ids and (
            not isinstance(canonical_value, int) or isinstance(canonical_value, bool)
        ):
            raise ValueError(f"{canonical_field} must be present as an integer")
        for field, value in (
            (canonical_field, canonical_value),
            (legacy_field, legacy_value),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"{field} must be an integer")
        if canonical_value is not None and legacy_value is not None:
            if canonical_value != legacy_value:
                raise ValueError(
                    f"{canonical_field} and {legacy_field} disagree: "
                    f"{canonical_value} != {legacy_value}"
                )
        value = canonical_value if canonical_value is not None else legacy_value
        if value is None and fallback_token_ids is not None:
            value = getattr(fallback_token_ids, result_field)
        if value is None:
            raise ValueError(f"{canonical_field} must be present as an integer")
        resolved[result_field] = value

    token_ids = LatentTokenIds(**resolved)
    if len(set(resolved.values())) != 3:
        raise ValueError("latent pad/start/end token IDs must be distinct")

    latent_size = getattr(config, "latent_size", None)
    if require_latent_size and (
        not isinstance(latent_size, int)
        or isinstance(latent_size, bool)
        or latent_size <= 0
    ):
        raise ValueError("latent_size must be a positive integer")
    return token_ids, latent_size


@dataclass(frozen=True)
class LatentSpan:
    start_position: int
    latent_positions: tuple[int, ...]
    end_position: int


@dataclass(frozen=True)
class LatentRolloutDiagnostics:
    block_count: int
    latent_l2_norms: torch.Tensor
    adjacent_cosine_similarities: torch.Tensor


def parse_latent_spans(
    input_ids: torch.LongTensor,
    *,
    token_ids: LatentTokenIds,
    latent_size: int,
) -> tuple[LatentSpan, ...]:
    """Parse all ordered latent blocks from one-dimensional token IDs."""
    if input_ids.ndim != 1:
        raise LatentStructureError("parse_latent_spans expects a 1D token sequence")
    if latent_size <= 0:
        raise LatentStructureError("latent_size must be positive")

    spans: list[LatentSpan] = []
    open_start: int | None = None
    latent_positions: list[int] = []
    for position, raw_token_id in enumerate(input_ids.tolist()):
        token_id = int(raw_token_id)
        if open_start is None:
            if token_id == token_ids.start_token_id:
                open_start = position
                latent_positions = []
            elif token_id == token_ids.end_token_id:
                raise LatentStructureError("latent_end without latent_start")
            elif token_id == token_ids.pad_token_id:
                raise LatentStructureError("latent_pad outside latent block")
            continue

        if token_id == token_ids.start_token_id:
            raise LatentStructureError("nested latent_start")
        if token_id == token_ids.end_token_id:
            if len(latent_positions) != latent_size:
                raise LatentStructureError(f"expected {latent_size} latent_pad tokens")
            spans.append(LatentSpan(open_start, tuple(latent_positions), position))
            open_start = None
            latent_positions = []
            continue
        if token_id != token_ids.pad_token_id:
            raise LatentStructureError("non-latent_pad token inside latent block")
        latent_positions.append(position)
        if len(latent_positions) > latent_size:
            raise LatentStructureError(f"expected {latent_size} latent_pad tokens")

    if open_start is not None:
        raise LatentStructureError("missing latent_end")
    return tuple(spans)


class Stage2LatentStates(tuple):
    """Tuple whose loss-bearing identities must survive ZeRO output hooks."""

    __ilvr_preserve_zero3_output_identity__ = True

    def __new__(cls, values=()):
        return super().__new__(cls, values)


@dataclass(frozen=True)
class LatentRolloutResult:
    final_inputs_embeds: torch.Tensor
    latent_states: Stage2LatentStates
    prefix_lengths: tuple[int, ...]
    diagnostics: LatentRolloutDiagnostics | None = None
    decoder_call_stats: "Stage2DecoderCallStats | None" = None


@dataclass
class Stage2DecoderCallStats:
    """Tensor-free live witness updated at decoder invocations, including backward."""

    clean_forward_calls: int
    sidecar_forward_calls: int = 0
    checkpoint_recompute_calls: int = 0
    forward_query_tokens: int = 0
    recompute_query_tokens: int = 0


@dataclass(frozen=True)
class DifferentiableCacheRolloutResult:
    """Hidden states and final prefix emitted by a differentiable cache rollout."""

    sequence_hidden_states: torch.Tensor
    latent_states: Stage2LatentStates
    chunk_lengths: tuple[int, ...]
    prefix_state: DifferentiablePrefixState
    decoder_call_stats: Stage2DecoderCallStats


def _replace_embedding(embeds: torch.Tensor, position: int, value: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (embeds[:, :position], value.unsqueeze(1), embeds[:, position + 1 :]),
        dim=1,
    )


def _build_diagnostics(
    states: list[torch.Tensor],
    block_count: int,
    *,
    batch_size: int,
    device: torch.device,
) -> LatentRolloutDiagnostics:
    if not states:
        empty = torch.empty((batch_size, 0), dtype=torch.float32, device=device)
        return LatentRolloutDiagnostics(block_count, empty, empty.clone())

    detached_states = [state.detach() for state in states]
    with torch.no_grad():
        stacked = torch.stack(detached_states, dim=1)
        norms = torch.linalg.vector_norm(stacked.float(), dim=-1)
        adjacent = torch.nn.functional.cosine_similarity(
            stacked[:, :-1].float(), stacked[:, 1:].float(), dim=-1
        )
    return LatentRolloutDiagnostics(block_count, norms, adjacent)


class LatentRolloutEngine:
    """Run the differentiable, no-cache continuous latent recurrence."""

    def rollout(
        self,
        *,
        language_model,
        base_inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        full_position_ids: torch.Tensor,
        latent_spans: tuple[LatentSpan, ...],
        retain_latent_grads: bool = False,
        collect_diagnostics: bool = False,
        latent_activation_checkpointing: bool = False,
        objective_request: ObjectiveForwardRequest | None = None,
    ) -> LatentRolloutResult:
        if type(latent_activation_checkpointing) is not bool:
            raise ValueError("latent_activation_checkpointing must be boolean")
        if base_inputs_embeds.ndim != 3:
            raise ValueError("base_inputs_embeds must have shape [batch, sequence, hidden]")
        if base_inputs_embeds.shape[0] != 1:
            raise NotImplementedError("Stage 2 rollout currently supports batch_size=1 only")
        sequence_shape = base_inputs_embeds.shape[:2]
        if attention_mask.shape != sequence_shape:
            raise ValueError("attention_mask must match the embedding batch and sequence dimensions")
        if full_position_ids.ndim != 3 or full_position_ids.shape[-2:] != sequence_shape:
            raise ValueError("full_position_ids must match the embedding batch and sequence dimensions")

        current = base_inputs_embeds
        states = []
        prefix_lengths = []
        decoder_call_stats = Stage2DecoderCallStats(clean_forward_calls=0)
        for span in latent_spans:
            for position in span.latent_positions:
                prefix_mask = attention_mask[:, :position]
                if objective_request is not None:
                    prefix_mask = build_objective_attention_mask(
                        attention_mask, length=position,
                        intervention=objective_request.intervention, dtype=current.dtype,
                    )
                latent = _run_latent_prefix(
                    language_model,
                    current[:, :position],
                    prefix_mask,
                    full_position_ids[..., :position],
                    decoder_call_stats,
                    latent_activation_checkpointing,
                )
                if retain_latent_grads:
                    latent.retain_grad()
                states.append(latent)
                prefix_lengths.append(position)
                current = _replace_embedding(current, position, latent)
        diagnostics = None
        if collect_diagnostics:
            diagnostics = _build_diagnostics(
                states,
                len(latent_spans),
                batch_size=base_inputs_embeds.shape[0],
                device=base_inputs_embeds.device,
            )
        return LatentRolloutResult(
            current,
            Stage2LatentStates(states),
            tuple(prefix_lengths),
            diagnostics,
            decoder_call_stats,
        )


def _run_latent_prefix(language_model, inputs, mask, positions, stats, checkpointing):
    """Checkpoint a cache-free prefix; bind each replay's tensors explicitly."""
    invoked = False

    def predict(prefix_inputs, prefix_mask, prefix_positions):
        nonlocal invoked
        if invoked:
            stats.checkpoint_recompute_calls += 1
            stats.recompute_query_tokens += prefix_inputs.shape[1]
        else:
            stats.clean_forward_calls += 1
            stats.forward_query_tokens += prefix_inputs.shape[1]
            invoked = True
        outputs = language_model(
            inputs_embeds=prefix_inputs,
            attention_mask=prefix_mask,
            position_ids=prefix_positions,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, -1, :]

    if checkpointing and torch.is_grad_enabled():
        # Complete replay keeps decoder entry/exit hooks and recompute counters
        # balanced; no clean cache or sidecar is captured by this function.
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            return torch.utils.checkpoint.checkpoint(
                predict, inputs, mask, positions,
                use_reentrant=False, preserve_rng_state=True,
            )
    return predict(inputs, mask, positions)


def _trainable_parameter_ids(language_model) -> frozenset[int]:
    """Identify decoder leaves that a differentiable cache may depend on."""
    while isinstance(language_model, partial):
        language_model = language_model.func
    parameters = getattr(language_model, "parameters", None)
    if not callable(parameters):
        return frozenset()
    return frozenset(
        id(parameter) for parameter in parameters() if parameter.requires_grad
    )


def _autograd_leaf_ids(tensor: torch.Tensor) -> set[int]:
    """Collect the leaf tensors reachable from one trusted input graph."""
    leaves = {id(tensor)} if tensor.is_leaf else set()
    pending = [tensor.grad_fn]
    visited = set()
    while pending:
        function = pending.pop()
        if function is None or function in visited:
            continue
        visited.add(function)
        variable = getattr(function, "variable", None)
        if variable is not None:
            leaves.add(id(variable))
        pending.extend(next_function for next_function, _ in function.next_functions)
    return leaves


def _is_trusted_cache_graph_function(
    function,
    *,
    trusted_input_ids: set[int],
    trusted_input_grad_fns: set,
    trainable_parameter_ids: frozenset[int],
) -> bool:
    """Recognize one trusted streamed-input or decoder-parameter graph node."""
    if function in trusted_input_grad_fns:
        return True
    variable = getattr(function, "variable", None)
    return variable is not None and (
        id(variable) in trusted_input_ids
        or id(variable) in trainable_parameter_ids
    )


def _are_graph_connected_cache_tensors(
    cache_tensors: tuple[torch.Tensor, ...],
    *,
    trusted_input_ids: set[int],
    trusted_input_grad_fns: set,
    trainable_parameter_ids: frozenset[int],
    require_all: bool = True,
) -> bool:
    """Prove all or any cache roots with shared reachability results."""
    connected_by_function = {}

    def reaches_trusted_root(root) -> bool:
        if root is None:
            return False
        if root in connected_by_function:
            return connected_by_function[root]
        if _is_trusted_cache_graph_function(
            root,
            trusted_input_ids=trusted_input_ids,
            trusted_input_grad_fns=trusted_input_grad_fns,
            trainable_parameter_ids=trainable_parameter_ids,
        ):
            connected_by_function[root] = True
            return True

        active = {root}
        stack = [[root, iter(root.next_functions), None]]
        while stack:
            function, next_functions, pending_child = stack[-1]
            if pending_child is not None:
                stack[-1][2] = None
                if connected_by_function[pending_child]:
                    connected_by_function[function] = True
                    active.remove(function)
                    stack.pop()
                continue

            try:
                next_function, _ = next(next_functions)
            except StopIteration:
                connected_by_function[function] = False
                active.remove(function)
                stack.pop()
                continue

            if next_function is None or next_function in active:
                continue
            if next_function in connected_by_function:
                if connected_by_function[next_function]:
                    connected_by_function[function] = True
                    active.remove(function)
                    stack.pop()
                continue

            stack[-1][2] = next_function
            if _is_trusted_cache_graph_function(
                next_function,
                trusted_input_ids=trusted_input_ids,
                trusted_input_grad_fns=trusted_input_grad_fns,
                trainable_parameter_ids=trainable_parameter_ids,
            ):
                connected_by_function[next_function] = True
            else:
                active.add(next_function)
                stack.append(
                    [
                        next_function,
                        iter(next_function.next_functions),
                        None,
                    ]
                )
        return connected_by_function[root]

    if not cache_tensors:
        return False
    any_connected = False
    for cache_tensor in cache_tensors:
        connected = cache_tensor.requires_grad and (
            id(cache_tensor) in trusted_input_ids
            or reaches_trusted_root(cache_tensor.grad_fn)
        )
        if require_all and not connected:
            return False
        if not require_all and connected:
            return True
        any_connected = any_connected or connected
    return any_connected


def _is_graph_connected_cache_tensor(
    cache_tensor: torch.Tensor,
    *,
    trusted_input_ids: set[int],
    trusted_input_grad_fns: set,
    trainable_parameter_ids: frozenset[int],
) -> bool:
    """Compatibility wrapper for one cache tensor."""
    return _are_graph_connected_cache_tensors(
        (cache_tensor,),
        trusted_input_ids=trusted_input_ids,
        trusted_input_grad_fns=trusted_input_grad_fns,
        trainable_parameter_ids=trainable_parameter_ids,
    )


class DifferentiableCacheRolloutEngine:
    """Stream latent recurrence through one sample-local, autograd-connected cache."""

    def rollout(
        self,
        *,
        language_model,
        base_inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        full_position_ids: torch.Tensor,
        latent_spans: tuple[LatentSpan, ...],
        stage2_cache_activation_checkpointing: bool = False,
        retain_latent_grads: bool = False,
    ) -> DifferentiableCacheRolloutResult:
        if type(stage2_cache_activation_checkpointing) is not bool:
            raise TypeError("stage2_cache_activation_checkpointing must be a bool")
        if not torch.is_grad_enabled():
            raise RuntimeError("Differentiable cache rollout requires gradients enabled")
        if base_inputs_embeds.ndim != 3:
            raise ValueError("base_inputs_embeds must have shape [batch, sequence, hidden]")
        if base_inputs_embeds.shape[0] != 1:
            raise NotImplementedError("Stage 2 rollout currently supports batch_size=1 only")
        sequence_shape = base_inputs_embeds.shape[:2]
        if attention_mask.shape != sequence_shape:
            raise ValueError("attention_mask must match the embedding batch and sequence dimensions")
        if full_position_ids.ndim != 3 or full_position_ids.shape[-2:] != sequence_shape:
            raise ValueError("full_position_ids must match the embedding batch and sequence dimensions")

        sequence_length = sequence_shape[1]
        ordered_latent_positions = tuple(
            position for span in latent_spans for position in span.latent_positions
        )
        if (
            any(position <= 0 or position >= sequence_length for position in ordered_latent_positions)
            or tuple(sorted(ordered_latent_positions)) != ordered_latent_positions
            or len(set(ordered_latent_positions)) != len(ordered_latent_positions)
        ):
            raise ValueError("latent positions must be unique, ordered, and inside the sequence")

        trainable_parameter_ids = _trainable_parameter_ids(language_model)
        trusted_input_ids = _autograd_leaf_ids(base_inputs_embeds)
        trusted_input_grad_fns = set()
        expected_layer_count = None
        decoder_call_stats = Stage2DecoderCallStats(clean_forward_calls=0)

        def consume(
            inputs_embeds: torch.Tensor,
            start: int,
            prior_cache_tensors: _OrderedCacheTensors,
        ):
            nonlocal expected_layer_count
            end = start + inputs_embeds.shape[1]
            trusted_input_ids.add(id(inputs_embeds))
            if inputs_embeds.grad_fn is not None:
                trusted_input_grad_fns.add(inputs_embeds.grad_fn)
            chunk_attention_mask = attention_mask[:, :end]
            chunk_position_ids = full_position_ids[..., start:end]
            chunk_cache_position = torch.arange(
                start, end, device=base_inputs_embeds.device
            )
            prior_layouts = prior_cache_tensors.layouts
            prior_expected_layer_count = (
                expected_layer_count if prior_cache_tensors else None
            )
            expected_return_layouts = None
            invoked = False

            def checkpoint_function(
                current_inputs_embeds: torch.Tensor,
                *checkpoint_prior_cache_tensors: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                nonlocal expected_layer_count, expected_return_layouts, invoked
                witnessed_prior = _OrderedCacheTensors(
                    checkpoint_prior_cache_tensors, prior_layouts
                )
                replay_cache = _rebuild_dynamic_cache(
                    witnessed_prior,
                    expected_layer_count=prior_expected_layer_count,
                    expected_sequence_length=start,
                )
                if invoked:
                    decoder_call_stats.checkpoint_recompute_calls += 1
                    decoder_call_stats.recompute_query_tokens += current_inputs_embeds.shape[1]
                else:
                    decoder_call_stats.clean_forward_calls += 1
                    decoder_call_stats.forward_query_tokens += current_inputs_embeds.shape[1]
                    invoked = True
                output = language_model(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=chunk_attention_mask,
                    position_ids=chunk_position_ids,
                    past_key_values=replay_cache,
                    cache_position=chunk_cache_position,
                    use_cache=True,
                    return_dict=True,
                )
                returned_cache = getattr(output, "past_key_values", None)
                if returned_cache is None:
                    raise ValueError("cached rollout requires returned past_key_values")
                returned_tensors = _flatten_dynamic_cache(
                    returned_cache, expected_sequence_length=end
                )
                returned_layer_count = len(returned_tensors) // 2
                if expected_layer_count is None:
                    expected_layer_count = returned_layer_count
                elif returned_layer_count != expected_layer_count:
                    raise ValueError("returned cache does not match expected layer count")
                if start and len(returned_tensors) != len(witnessed_prior):
                    raise ValueError("returned cache does not match expected layer count")
                if expected_return_layouts is None:
                    expected_return_layouts = returned_tensors.layouts
                elif returned_tensors.layouts != expected_return_layouts:
                    raise ValueError(
                        "cache tensor layout changed between original call and replay"
                    )
                if not _are_graph_connected_cache_tensors(
                    returned_tensors,
                    trusted_input_ids=trusted_input_ids,
                    trusted_input_grad_fns=trusted_input_grad_fns,
                    trainable_parameter_ids=trainable_parameter_ids,
                    require_all=False,
                ):
                    raise ValueError(
                        "returned cache tensors must preserve the autograd graph"
                    )
                hidden_states = getattr(output, "last_hidden_state", None)
                if (
                    not isinstance(hidden_states, torch.Tensor)
                    or hidden_states.ndim != 3
                    or hidden_states.shape != current_inputs_embeds.shape
                ):
                    raise ValueError(
                        "cached rollout must return current-chunk hidden states"
                    )
                return (hidden_states, *returned_tensors)

            # Short samples keep tiny single-latent activation graphs live for
            # speed. On long samples those graphs retain a large prefix cache,
            # so checkpoint every decoder call to bound peak memory.
            checkpoint_chunk = stage2_cache_activation_checkpointing and (
                inputs_embeds.shape[1] > 1
                or sequence_length > _FULL_CACHE_CHECKPOINT_SEQUENCE_THRESHOLD
            )
            if checkpoint_chunk:
                with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                    tensor_output = torch.utils.checkpoint.checkpoint(
                        checkpoint_function,
                        inputs_embeds,
                        *prior_cache_tensors,
                        use_reentrant=False,
                        preserve_rng_state=True,
                        determinism_check="default",
                    )
            else:
                tensor_output = checkpoint_function(inputs_embeds, *prior_cache_tensors)
            if isinstance(tensor_output, torch.Tensor):
                tensor_output = (tensor_output,)
            if not isinstance(tensor_output, tuple) or not tensor_output:
                raise ValueError("cached rollout must return tensor outputs")
            if expected_return_layouts is None:
                raise ValueError("cached rollout did not return cache tensor layout")
            current_hidden = tensor_output[0]
            returned_tensors = _OrderedCacheTensors(
                tensor_output[1:], expected_return_layouts
            )
            caller_cache = _rebuild_dynamic_cache(
                returned_tensors,
                expected_layer_count=expected_layer_count,
                expected_sequence_length=end,
            )
            return current_hidden, _flatten_dynamic_cache(
                caller_cache, expected_sequence_length=end
            )

        cursor = 0
        cache_tensors = _flatten_dynamic_cache(None, expected_sequence_length=0)
        hidden_chunks = []
        latent_states = []
        for latent_index, position in enumerate(ordered_latent_positions):
            if cursor < position:
                current_hidden, cache_tensors = consume(
                    base_inputs_embeds[:, cursor:position], cursor, cache_tensors
                )
                hidden_chunks.append(current_hidden)
            latent = hidden_chunks[-1][:, -1, :]
            if retain_latent_grads:
                latent.retain_grad()
            latent_states.append(latent)
            # A produced latent and the following known text can share a
            # decoder call. Stop immediately before the next latent, whose
            # input depends on the last hidden state of this chunk.
            next_position = (ordered_latent_positions[latent_index + 1]
                             if latent_index + 1 < len(ordered_latent_positions)
                             else sequence_length)
            chunk = torch.cat((latent.unsqueeze(1),
                               base_inputs_embeds[:, position + 1:next_position]), dim=1)
            current_hidden, cache_tensors = consume(chunk, position, cache_tensors)
            hidden_chunks.append(current_hidden)
            cursor = next_position

        if cursor < sequence_length:
            current_hidden, cache_tensors = consume(
                base_inputs_embeds[:, cursor:], cursor, cache_tensors
            )
            hidden_chunks.append(current_hidden)

        if not hidden_chunks:
            return DifferentiableCacheRolloutResult(
                base_inputs_embeds[:, :0],
                Stage2LatentStates(()),
                tuple(),
                DifferentiablePrefixState((), (), 0, ()),
                decoder_call_stats,
            )
        prefix_tensors = tuple(cache_tensors)
        return DifferentiableCacheRolloutResult(
            torch.cat(hidden_chunks, dim=1),
            Stage2LatentStates(latent_states),
            tuple(chunk.shape[1] for chunk in hidden_chunks),
            DifferentiablePrefixState(
                prefix_tensors,
                cache_tensors.layouts,
                sequence_length,
                tuple(id(tensor) for tensor in prefix_tensors),
                tuple(
                    _tensor_is_differentiable(tensor)
                    for tensor in prefix_tensors
                ),
            ),
            decoder_call_stats,
        )

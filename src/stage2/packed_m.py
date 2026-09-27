"""Graph-free one-call packed suffix execution for Stage 2 M scores."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from peft import PeftModel
from peft.tuners.lora.layer import Linear as PeftLinear
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import AuxiliaryTrainingWrapper, TrainableTokensWrapper
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import normalize_stage2_ablation

from peft_common import ALLOWED_QWEN_DECODER_SUFFIXES, resolve_qwen2_5_vl_model
from stage1.token_rows import SparseOutputTokenRows
from .objective import target_log_likelihood
from .objective_windows import WindowBlock


@dataclass(frozen=True)
class PackedMScores:
    """Per-block graph-free scores and observed packed decoder geometry."""

    scores: tuple[torch.Tensor, ...]
    token_scores: tuple[torch.Tensor, ...]
    decoder_calls: int
    query_rows: int


@dataclass(frozen=True)
class _DropoutRecord:
    module: nn.Dropout
    rng_state: torch.Tensor
    shape: tuple[int, int, int]
    dtype: torch.dtype
    device: torch.device
    stride: tuple[int, int, int]
    probability: float


def _sparse_embedding_exception(wrapped: nn.Module, base: nn.Module):
    """Allow only the production Stage 1 DoRA sparse-token pair, by identity.

    M consumes captured F embeddings, not a second embedding lookup; its head
    still calls SparseOutputTokenRows.forward, including the output deltas.
    """
    embedding = base.model.embed_tokens
    if not isinstance(embedding, TrainableTokensWrapper):
        return None, None, None
    if type(embedding) is not TrainableTokensWrapper or not isinstance(wrapped, PeftModel):
        raise ValueError("packed M requires the canonical sparse embedding wrapper")
    active = tuple(wrapped.active_adapters)
    if len(active) != 1 or set(wrapped.peft_config) != set(active):
        raise ValueError("packed M sparse rows require one active DoRA adapter")
    adapter = active[0]
    config = wrapped.peft_config[adapter]
    layer = embedding.token_adapter
    ids = tuple(getattr(base.config, name, None) for name in (
        "latent_pad_token_id", "latent_start_token_id", "latent_end_token_id"
    ))
    if (
        not config.use_dora
        or any(type(value) is not int or not 0 <= value < base.config.vocab_size for value in ids)
        or len(set(ids)) != 3
        or config.trainable_token_indices != {"model.embed_tokens": list(ids)}
        or type(layer) is not TrainableTokensLayer
        or type(layer.base_layer) is not nn.Embedding
        or layer.base_layer.max_norm is not None
        or layer.tied_adapter is not None
        or layer.merged
        or embedding.disable_adapters
        or layer.disable_adapters
        or tuple(embedding.active_adapters) != active
        or tuple(layer.active_adapters) != active
        or layer.token_indices != {adapter: list(ids)}
        or set(layer.trainable_tokens_delta) != {adapter}
    ):
        raise ValueError("packed M requires canonical unmerged active Stage 1 sparse input rows")
    head = base.lm_head
    if (
        type(head) is not SparseOutputTokenRows
        or type(head.base_head) is not nn.Linear
        or tuple(head.token_ids.detach().cpu().tolist()) != ids
        or tuple(module for module in base.modules() if isinstance(module, SparseOutputTokenRows)) != (head,)
    ):
        raise ValueError("packed M requires the matching canonical sparse output rows")
    return embedding, layer, adapter


def _active_adapter_sites(model: nn.Module, decoder: nn.Module) -> tuple[nn.Dropout, ...]:
    """Validate and return every supported active stochastic adapter site."""
    wrapped = getattr(model, "module", model)
    if isinstance(wrapped, PeftModel) and len(wrapped.peft_config) != 1:
        raise ValueError("packed M requires exactly one configured adapter")

    allowed_projection_ids = set()
    for layer in decoder.layers:
        for owner in (layer.self_attn, layer.mlp):
            allowed_projection_ids.update(
                id(getattr(owner, suffix))
                for suffix in ALLOWED_QWEN_DECODER_SUFFIXES
                if hasattr(owner, suffix)
            )

    base = resolve_qwen2_5_vl_model(model)
    sparse_wrapper, sparse_layer, sparse_adapter = _sparse_embedding_exception(wrapped, base)
    auxiliary_modules = tuple(
        name
        for name, module in base.named_modules()
        if isinstance(module, AuxiliaryTrainingWrapper) and module is not sparse_wrapper
    )
    if auxiliary_modules:
        raise ValueError(
            f"packed M does not support auxiliary PEFT modules: {auxiliary_modules}"
        )
    tuner_modules = tuple(
        (name, module)
        for name, module in base.named_modules()
        if isinstance(module, BaseTunerLayer) and module is not sparse_layer
    )
    for name, module in tuner_modules:
        if type(module) is not PeftLinear:
            raise ValueError(
                f"packed M supports only standard Linear LoRA/DoRA tuner modules: {name}"
            )
        if id(module) not in allowed_projection_ids:
            raise ValueError(
                f"packed M active PEFT modules must be allowed decoder projections: {name}"
            )

    sites: list[nn.Dropout] = []
    adapters: set[str] = set()
    adapter_dropout_ids: set[int] = set()
    for name, module in tuner_modules:
        if type(module.get_base_layer()) is not nn.Linear:
            raise ValueError("packed M supports only standard Linear LoRA/DoRA adapters")
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in ALLOWED_QWEN_DECODER_SUFFIXES:
            raise ValueError(f"packed M found an unsupported adapter target: {name}")
        active = tuple(module.active_adapters)
        if len(active) != 1 or active[0] not in module.lora_A:
            raise ValueError("packed M requires exactly one active adapter")
        if module.merged:
            raise ValueError("packed M requires an unmerged adapter")
        adapter = active[0]
        if sparse_adapter is not None and (
            adapter != sparse_adapter or module.disable_adapters or not module.use_dora.get(adapter, False)
        ):
            raise ValueError("packed M sparse rows require the same active DoRA adapter at every projection")
        adapters.add(adapter)
        dropout = module.lora_dropout[adapter]
        if isinstance(dropout, nn.Dropout):
            if type(dropout) is not nn.Dropout or dropout.inplace:
                raise ValueError("packed M requires ordinary non-inplace adapter dropout")
            if not 0.0 <= dropout.p < 1.0:
                raise ValueError("packed M adapter dropout probability must be in [0, 1)")
            adapter_dropout_ids.add(id(dropout))
            if module.training and dropout.training and dropout.p > 0.0:
                sites.append(dropout)
        elif not isinstance(dropout, nn.Identity):
            raise ValueError("packed M found a nonstandard adapter dropout module")
    if len(adapters) > 1:
        raise ValueError("packed M requires one shared active adapter")

    unexpected = [
        name
        for name, module in decoder.named_modules()
        if isinstance(module, nn.modules.dropout._DropoutNd)
        and module.training
        and module.p > 0.0
        and (type(module) is not nn.Dropout or id(module) not in adapter_dropout_ids)
    ]
    if unexpected:
        raise ValueError(
            f"packed M found unsupported decoder dropout sites: {unexpected}"
        )
    return tuple(sites)


def _rng_state_for(device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return torch.get_rng_state().clone()
    if device.type == "cuda":
        if not torch.cuda.is_initialized():
            raise RuntimeError("a CUDA dropout tensor exists without an initialized CUDA generator")
        index = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.default_generators[index].get_state().clone()
    raise NotImplementedError(f"packed M dropout replay does not support {device.type}")


class PackedMPrefixCapture:
    """Observe one unchanged final F decoder call and retain detached replay state."""

    def __init__(self, model: nn.Module, root_length: int):
        if isinstance(root_length, bool) or not isinstance(root_length, int) or root_length <= 0:
            raise ValueError("root_length must be a positive integer")
        self._model = model
        self._base = resolve_qwen2_5_vl_model(model)
        self.root_length = root_length
        self._entered = False
        self._active = False
        self._ready = False
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._raw: dict[tuple[int, str], torch.Tensor] = {}
        self._kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._dropout_records: dict[int, _DropoutRecord] = {}
        self._dropout_sites: tuple[nn.Dropout, ...] = ()
        self._inputs_embeds: torch.Tensor | None = None
        self._position_ids: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        return self._ready and not self._active and not self._handles

    def __enter__(self) -> "PackedMPrefixCapture":
        if self._entered:
            raise RuntimeError("PackedMPrefixCapture cannot be reused")
        self._entered = True
        config = self._base.config
        if config._attn_implementation not in ("eager", "sdpa"):
            raise NotImplementedError("packed M supports eager and SDPA attention only")
        if bool(getattr(config, "use_sliding_window", False)):
            raise NotImplementedError("packed M does not support sliding-window attention")
        if float(getattr(config, "attention_dropout", 0.0)) != 0.0:
            raise ValueError("packed M requires zero native attention dropout")

        decoder = self._base.model
        self._dropout_sites = _active_adapter_sites(self._model, decoder)

        def decoder_start(_module, _args, kwargs):
            embeds = kwargs.get("inputs_embeds")
            if not isinstance(embeds, torch.Tensor):
                return
            # Latent rollout calls are strict prefixes; the unchanged final F call
            # is the unique full-length decoder invocation in both train and eval.
            is_final_f = embeds.ndim == 3 and embeds.shape[1] == self.root_length
            if not is_final_f:
                return
            if self._inputs_embeds is not None:
                raise RuntimeError("packed M capture observed more than one final F invocation")
            position_ids = kwargs.get("position_ids")
            attention_mask = kwargs.get("attention_mask")
            if not isinstance(position_ids, torch.Tensor) or position_ids.shape != (3, 1, self.root_length):
                raise ValueError("packed M requires final F three-axis position_ids shaped [3,1,T]")
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (1, self.root_length):
                raise ValueError("packed M requires final F attention_mask shaped [1,T]")
            if embeds.shape[0] != 1 or not embeds.is_contiguous():
                raise ValueError("packed M requires contiguous final F embeddings shaped [1,T,features]")
            self._active = True
            self._inputs_embeds = embeds.detach().clone()
            self._position_ids = position_ids.detach().clone()
            self._attention_mask = attention_mask.detach().clone()

        def decoder_end(_module, _args, _output):
            self._active = False

        self._handles.append(
            decoder.register_forward_pre_hook(decoder_start, with_kwargs=True)
        )
        self._handles.append(decoder.register_forward_hook(decoder_end, always_call=True))

        for layer_index, layer in enumerate(decoder.layers):
            for label, projection in (
                ("key", layer.self_attn.k_proj),
                ("value", layer.self_attn.v_proj),
            ):

                def projected(_module, _args, output, layer_index=layer_index, label=label):
                    if not self._active:
                        return
                    key = (layer_index, label)
                    if key in self._raw:
                        raise RuntimeError("packed M capture observed a repeated final F projection")
                    if not isinstance(output, torch.Tensor) or output.ndim != 3:
                        raise ValueError("packed M projection capture requires rank-3 tensors")
                    self._raw[key] = output.detach()

                self._handles.append(projection.register_forward_hook(projected))

            def attention_done(
                module, _args, kwargs, _output, layer_index=layer_index
            ):
                if not self._active:
                    return
                # An always-call hook also runs while an earlier projection/site
                # is propagating its own validation error. Do not mask that cause.
                if _output is None:
                    return
                raw_key = self._raw.get((layer_index, "key"))
                raw_value = self._raw.get((layer_index, "value"))
                if raw_key is None or raw_value is None:
                    raise RuntimeError("packed M capture missed a key/value projection")
                position_embeddings = kwargs.get("position_embeddings")
                if (
                    not isinstance(position_embeddings, tuple)
                    or len(position_embeddings) != 2
                    or not all(isinstance(value, torch.Tensor) for value in position_embeddings)
                ):
                    raise ValueError("packed M capture requires actual rotary embeddings")
                with torch.no_grad():
                    key = raw_key.view(
                        1, self.root_length, -1, module.head_dim
                    ).transpose(1, 2)
                    value = raw_value.view(
                        1, self.root_length, -1, module.head_dim
                    ).transpose(1, 2)
                    cos, sin = position_embeddings
                    _, key = apply_multimodal_rotary_pos_emb(
                        key,
                        key,
                        cos,
                        sin,
                        module.rope_scaling["mrope_section"],
                    )
                    self._kv[layer_index] = (key.detach().clone(), value.detach().clone())

            self._handles.append(
                layer.self_attn.register_forward_hook(
                    attention_done, with_kwargs=True, always_call=True
                )
            )

        for site in self._dropout_sites:

            def before_dropout(module, args, site=site):
                if not self._active:
                    return
                if id(site) in self._dropout_records:
                    raise RuntimeError(
                        "packed M capture requires one final F invocation per stochastic site"
                    )
                if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                    raise ValueError("packed M adapter dropout must receive one tensor")
                value = args[0]
                if (
                    value.ndim != 3
                    or value.shape[0] != 1
                    or value.shape[1] != self.root_length
                    or not value.is_contiguous()
                ):
                    raise ValueError(
                        "packed M adapter dropout requires contiguous [1,T,features] input"
                    )
                self._dropout_records[id(site)] = _DropoutRecord(
                    module=module,
                    rng_state=_rng_state_for(value.device),
                    shape=tuple(value.shape),
                    dtype=value.dtype,
                    device=value.device,
                    stride=tuple(value.stride()),
                    probability=float(module.p),
                )

            self._handles.append(site.register_forward_pre_hook(before_dropout))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._active = False
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if exc_type is not None:
            return None
        layer_count = len(self._base.model.layers)
        if self._inputs_embeds is None or len(self._kv) != layer_count:
            raise RuntimeError("packed M capture did not observe one complete final F decoder call")
        if set(self._dropout_records) != {id(site) for site in self._dropout_sites}:
            raise RuntimeError("packed M capture missed an expected stochastic adapter site")
        self._ready = True
        return None


def _packed_layout(
    inputs: dict, blocks: tuple[WindowBlock, ...], capture: PackedMPrefixCapture
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...]]:
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    labels = inputs.get("labels")
    if not all(isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, labels)):
        raise TypeError("packed M inputs require input_ids, attention_mask, and labels tensors")
    if input_ids.shape != attention_mask.shape or input_ids.shape != labels.shape:
        raise ValueError("packed M token tensors must have identical shapes")
    if input_ids.shape != (1, capture.root_length):
        raise ValueError("packed M supports inputs shaped [1,T] matching the capture")
    if not torch.equal(
        attention_mask.to(capture._attention_mask.device), capture._attention_mask
    ):
        raise ValueError("packed M attention_mask does not match its prefix capture")

    branch_rows: list[tuple[int, ...]] = []
    for block in blocks:
        if not isinstance(block, WindowBlock) or block.batch_index != 0:
            raise ValueError("packed M supports WindowBlocks for batch_index=0 only")
        if not block.post_positions:
            raise ValueError("packed M WindowBlocks must contain post_positions")
        if not (0 <= block.start_position < block.end_position < capture.root_length):
            raise ValueError("packed M WindowBlock marker positions are invalid")
        if normalize_stage2_ablation(block.ablation)[0] != (block.start_position, block.end_position):
            # Root keys before END come from F; an earlier hidden block would leak through them.
            raise ValueError("packed M suffix must start at the first hidden latent END")
        maximum = max(block.post_positions)
        if maximum < block.end_position or maximum >= capture.root_length:
            raise ValueError("packed M WindowBlock post positions are invalid")
        rows = tuple(range(block.end_position, maximum + 1))
        branch_rows.append(rows)
        if any(not bool(attention_mask[0, row]) for row in rows):
            raise ValueError("packed M suffix rows must be active tokens")
        selected_targets = tuple(int(labels[0, position + 1]) for position in block.post_positions)
        if selected_targets != block.target_ids or any(target == -100 for target in selected_targets):
            raise ValueError("packed M WindowBlock targets disagree with labels")

    logical_rows = torch.tensor(
        [row for rows in branch_rows for row in rows],
        dtype=torch.long,
        device=input_ids.device,
    )
    query_rows = int(logical_rows.numel())
    visible = torch.zeros(
        (query_rows, capture.root_length + query_rows),
        dtype=torch.bool,
        device=input_ids.device,
    )
    root_positions = torch.arange(capture.root_length, device=input_ids.device)
    active_root = attention_mask[0].bool()
    offset = 0
    for block, rows in zip(blocks, branch_rows, strict=True):
        length = len(rows)
        allowed_root = (active_root & (root_positions < block.end_position)).expand(length, -1).clone()
        suffix_active = attention_mask[0, list(rows)].bool()
        allowed_suffix = torch.ones(
            (length, length), dtype=torch.bool, device=input_ids.device
        ).tril()
        allowed_suffix &= suffix_active.unsqueeze(0)
        query_positions = torch.tensor(rows, device=input_ids.device)
        for start, end in block.ablation:
            queries = (query_positions >= end).unsqueeze(1)
            allowed_root &= ~(queries & ((root_positions > start) & (root_positions < end)))
            allowed_suffix &= ~(queries & ((query_positions > start) & (query_positions < end)))
        visible[offset : offset + length, : capture.root_length] = allowed_root
        visible[
            offset : offset + length,
            capture.root_length + offset : capture.root_length + offset + length,
        ] = allowed_suffix
        offset += length
    return logical_rows, visible, tuple(branch_rows)


@contextmanager
def _replay_adapter_dropout(
    capture: PackedMPrefixCapture, logical_rows: torch.Tensor
) -> Iterator[None]:
    restored: list[tuple[nn.Dropout, bool, object | None]] = []
    visits: dict[int, int] = {site_id: 0 for site_id in capture._dropout_records}
    try:
        for site_id, record in capture._dropout_records.items():
            module = record.module
            had_forward = "forward" in module.__dict__
            previous_forward = module.__dict__.get("forward")
            restored.append((module, had_forward, previous_forward))

            def replay(value, record=record, site_id=site_id, module=module):
                visits[site_id] += 1
                if visits[site_id] > 1:
                    raise RuntimeError("packed M replay visited a stochastic site more than once")
                if torch.is_grad_enabled():
                    raise RuntimeError("packed M dropout replay must be graph-free")
                if (
                    value.ndim != 3
                    or value.shape[0] != 1
                    or value.shape[1] != logical_rows.numel()
                    or value.shape[2] != record.shape[2]
                    or value.dtype != record.dtype
                    or value.device != record.device
                    or not value.is_contiguous()
                    or tuple(value.stride()[1:]) != record.stride[1:]
                    or float(module.p) != record.probability
                ):
                    raise ValueError("packed M dropout replay layout does not match final F")
                device_rows = logical_rows.to(record.device)
                cuda_devices = []
                if record.device.type == "cuda":
                    cuda_devices = [
                        record.device.index
                        if record.device.index is not None
                        else torch.cuda.current_device()
                    ]
                with torch.random.fork_rng(devices=cuda_devices):
                    if record.device.type == "cpu":
                        torch.set_rng_state(record.rng_state.clone())
                    else:
                        torch.cuda.default_generators[cuda_devices[0]].set_state(
                            record.rng_state.clone()
                        )
                    mask = F.dropout(
                        torch.ones(
                            record.shape,
                            dtype=record.dtype,
                            device=record.device,
                        ),
                        p=record.probability,
                        training=True,
                    )
                return value * mask.index_select(1, device_rows)

            module.forward = replay
        yield
        if any(count != 1 for count in visits.values()):
            raise RuntimeError("packed M replay missed an expected stochastic adapter site")
    finally:
        for module, had_forward, previous_forward in reversed(restored):
            if had_forward:
                module.forward = previous_forward
            else:
                del module.forward


def score_packed_m(
    model: nn.Module,
    inputs: dict,
    blocks: tuple[WindowBlock, ...],
    capture: PackedMPrefixCapture,
) -> PackedMScores:
    """Score all independent M suffixes with one direct, graph-free decoder call."""
    blocks = tuple(blocks)
    if not blocks:
        return PackedMScores(scores=(), token_scores=(), decoder_calls=0, query_rows=0)
    if not isinstance(capture, PackedMPrefixCapture) or not capture.ready:
        raise ValueError("packed M requires a completed, inactive prefix capture")
    base = resolve_qwen2_5_vl_model(model)
    if base is not capture._base:
        raise ValueError("packed M model does not match its prefix capture")
    current_sites = _active_adapter_sites(model, base.model)
    if tuple(map(id, current_sites)) != tuple(map(id, capture._dropout_sites)):
        raise ValueError("packed M stochastic site identities changed after final F")
    if any(
        float(site.p) != capture._dropout_records[id(site)].probability
        for site in current_sites
    ):
        raise ValueError("packed M stochastic site probabilities changed after final F")

    logical_rows, visible, branch_rows = _packed_layout(inputs, blocks, capture)
    query_rows = int(logical_rows.numel())
    embeds = capture._inputs_embeds.index_select(1, logical_rows)
    position_ids = capture._position_ids.index_select(2, logical_rows)
    mask = torch.zeros(
        visible.shape, dtype=embeds.dtype, device=embeds.device
    ).masked_fill(~visible.to(embeds.device), torch.finfo(embeds.dtype).min)
    mask = mask.unsqueeze(0).unsqueeze(0)
    cache_position = torch.arange(
        capture.root_length,
        capture.root_length + query_rows,
        dtype=torch.long,
        device=embeds.device,
    )
    cache = DynamicCache()
    for layer_index in range(len(base.model.layers)):
        key, value = capture._kv[layer_index]
        if key.requires_grad or value.requires_grad:
            raise RuntimeError("packed M root cache must be detached")
        cache.update(key, value, layer_index)

    decoder_calls = 0

    def observed(_module, _args):
        nonlocal decoder_calls
        decoder_calls += 1

    handle = base.model.register_forward_pre_hook(observed)
    try:
        with torch.no_grad(), _replay_adapter_dropout(capture, logical_rows):
            hidden = base.model(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=False,
                return_dict=True,
                cache_position=cache_position,
                generate_mode=True,
            ).last_hidden_state
            selected_query_rows = []
            offset = 0
            for block, rows in zip(blocks, branch_rows, strict=True):
                selected_query_rows.extend(
                    offset + position - block.end_position
                    for position in block.post_positions
                )
                offset += len(rows)
            selected_hidden = hidden.index_select(
                1,
                torch.tensor(
                    selected_query_rows, dtype=torch.long, device=hidden.device
                ),
            )
            selected_logits = base.lm_head(selected_hidden)[0]
    finally:
        handle.remove()
    if decoder_calls != 1:
        raise RuntimeError(
            f"packed M expected one decoder call, observed {decoder_calls}"
        )

    scores = []
    token_scores = []
    selected_offset = 0
    labels = inputs["labels"]
    for block in blocks:
        count = len(block.post_positions)
        logits = selected_logits[selected_offset : selected_offset + count]
        targets = labels[0, [position + 1 for position in block.post_positions]].to(
            logits.device
        )
        block_token_scores = target_log_likelihood(logits, targets).detach()
        token_scores.append(block_token_scores)
        scores.append(block_token_scores.mean())
        selected_offset += count
    return PackedMScores(
        scores=tuple(scores),
        token_scores=tuple(token_scores),
        decoder_calls=decoder_calls,
        query_rows=query_rows,
    )

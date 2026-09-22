"""Observe ordinary CE Stage 2, then evaluate full-length local interventions."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from evaluation.vision_block_inputs import PreparedSample, Site, original_image_positions
from evaluation.vision_block_metrics import DEFAULT_PROFILE, IMAGE_PROFILE, conditions_for_profile
from stage2.peft import resolve_stage2_qwen_model


def numerical_policy_identity():
    """Report effective process flags, including when called outside the policy."""
    return {"sdpa_math": torch.backends.cuda.math_sdp_enabled(),
            "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdpa_cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32}


@contextmanager
def numerical_policy():
    """Apply the analysis math policy and restore the caller's nested state."""
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@dataclass(frozen=True)
class CleanState:
    embeds: torch.Tensor
    positions: torch.Tensor
    prefix_kv: tuple
    normal_scores: dict
    captured_scores: dict
    dispatch: dict = field(default_factory=dict)
    image_positions: tuple[int, ...] = ()
    image_token_contract: dict = field(default_factory=dict)


def _finite(tensor, name):
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"nonfinite {name}")


def _validate_kv(qwen, bank, embeds):
    """Validate full-sequence, unrepeated K/V against the actual decoder layers."""
    layers = qwen.model.layers
    if not layers or len(layers) != qwen.config.num_hidden_layers:
        raise ValueError("decoder layer count disagrees with config")
    if not isinstance(bank, (tuple, list)) or len(bank) != len(layers):
        raise ValueError("K/V layer count disagrees with decoder")
    for layer, pair in zip(layers, bank):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("K/V requires one key/value pair per layer")
        shape = (embeds.shape[0], layer.self_attn.num_key_value_heads,
                 embeds.shape[1], layer.self_attn.head_dim)
        for tensor in pair:
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or not tensor.is_floating_point() or tensor.device != embeds.device):
                raise ValueError("K/V requires [batch, kv_heads, sequence, head_dim] on the input device")
            _finite(tensor, "K/V")


def _scores(qwen, hidden, predictors, target_ids):
    """Bound vocabulary projection to 128 selected rows; accumulate in FP32."""
    result = []
    for offset in range(0, len(predictors), 128):
        indices = torch.tensor(predictors[offset:offset+128], device=hidden.device)
        targets = torch.tensor(target_ids[offset:offset+128], device=hidden.device)
        logits = qwen.lm_head(hidden[0].index_select(0, indices)).float()
        _finite(logits, "logits")
        log_probs = logits.log_softmax(-1)
        nll = -log_probs.gather(-1, targets[:, None]).squeeze(-1)
        _finite(nll, "NLL")
        result.extend(zip(logits.argmax(-1).tolist(), nll.tolist()))
    return result


@torch.inference_mode()
@numerical_policy()
def capture_clean(model, sample: PreparedSample, *, analysis_profile=DEFAULT_PROFILE) -> CleanState:
    """Observe, never modify, the one full ordinary final decoder invocation.

    Processor tensors may be on CPU. Transfer a new input mapping to the input
    embedding device without altering the caller's PreparedSample or identity.
    """
    conditions_for_profile(analysis_profile)
    qwen = resolve_stage2_qwen_model(model)
    image_positions, image_contract = (), {}
    if analysis_profile == IMAGE_PROFILE:
        image_contract = {
            "image_token_id": qwen.config.image_token_id,
            "vision_start_token_id": qwen.config.vision_start_token_id,
            "vision_end_token_id": qwen.config.vision_end_token_id,
            "merge_size": qwen.config.vision_config.spatial_merge_size,
        }
        image_positions = original_image_positions(sample, **image_contract)
    model.eval()
    device = qwen.model.embed_tokens.weight.device
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
              for key, value in sample.model_inputs.items()}
    if set(inputs) - {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "position_ids"}:
        raise ValueError("capture requires original-image inference inputs only")
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not sample.sites:
        raise ValueError("capture requires one sample with evaluated sites")
    if not bool(inputs["attention_mask"].bool().all()):
        raise ValueError("capture requires an unpadded logical sequence")
    if qwen.config._attn_implementation != "sdpa":
        raise ValueError("analysis requires SDPA language attention")
    length = ids.shape[1]
    selected = [(site.plan_index, position, token_id)
                for site in sample.sites
                for position, token_id in zip(site.target_positions, site.target_ids)]
    predictors = [position - 1 for _, position, _ in selected]
    targets = [token_id for _, _, token_id in selected]
    observed_inputs, observed_hidden = [], []

    def before(module, args, kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None and embeds.shape[1] == length:
            if kwargs.get("stage2_capture_kv") or kwargs.get("stage2_prefix_key_values") is not None:
                raise ValueError("ordinary clean reference unexpectedly enabled capture/replay")
            observed_inputs.append((embeds.detach().clone(), kwargs["position_ids"].detach().clone()))

    def after(module, args, kwargs, output):
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None and embeds.shape[1] == length:
            observed_hidden.append(output.last_hidden_state.detach())

    pre = qwen.model.register_forward_pre_hook(before, with_kwargs=True)
    post = qwen.model.register_forward_hook(after, with_kwargs=True)
    try:
        model(**inputs, labels=None, stage2_shared_prefix_inputs=None,
              stage2_selective_logits_only=True,
              stage2_logit_positions=torch.tensor([predictors[0]], device=device),
              use_cache=False, return_dict=True)
    finally:
        pre.remove()
        post.remove()
    if len(observed_inputs) != 1 or len(observed_hidden) != 1:
        raise ValueError("expected exactly one full-length ordinary final decoder call")
    embeds, positions = observed_inputs[0]
    if positions.shape != (3, 1, length):
        raise ValueError("clean capture requires original three-axis positions")
    _finite(embeds, "clean embeddings")
    _finite(observed_hidden[0], "normal hidden states")
    normal = _scores(qwen, observed_hidden.pop(), predictors, targets)
    captured = qwen.model(inputs_embeds=embeds, position_ids=positions,
        attention_mask=inputs["attention_mask"], generate_mode=True,
        use_cache=False, return_dict=True, stage2_capture_kv=True)
    _finite(captured.last_hidden_state, "captured hidden states")
    _validate_kv(qwen, captured.stage2_key_values, embeds)
    bank = tuple(tuple(tensor.detach() for tensor in pair) for pair in captured.stage2_key_values)
    captured_scores = _scores(qwen, captured.last_hidden_state, predictors, targets)
    keys = [(plan, position) for plan, position, _ in selected]
    return CleanState(embeds, positions, bank, dict(zip(keys, normal)),
        dict(zip(keys, captured_scores)), {
            "ordinary": {"stage2_capture_kv": False, "stage2_shared_prefix_inputs": None,
                         "matching_full_length_calls": len(observed_inputs)},
            "capture": {"stage2_capture_kv": True, "generate_mode": True,
                        "layer_dispatch": "shared-prefix capture branch"},
            "replay": {"layer_dispatch": "full-length capture branch", "mask": "dense additive square"},
            "language_backend": qwen.config._attn_implementation,
            "vision_backend": getattr(qwen.visual.config, "_attn_implementation", None),
            "attention_classes": [type(layer.self_attn).__name__ for layer in qwen.model.layers],
        }, image_positions, image_contract)


def build_replay_inputs(state: CleanState, site: Site, condition: str, *, analysis_profile=DEFAULT_PROFILE):
    """Keep every original row; intervene only on queries at/after this END."""
    if condition not in conditions_for_profile(analysis_profile):
        raise ValueError(f"unknown condition: {condition}")
    length = state.embeds.shape[1]
    if (not site.target_positions or len(site.target_positions) != len(site.target_ids)
            or tuple(sorted(set(site.target_positions))) != site.target_positions
            or not 0 <= site.start < site.body_start < site.body_end == site.end < site.target_positions[0]
            or site.target_positions[-1] >= length):
        raise ValueError("invalid local site geometry or targets")
    if analysis_profile == IMAGE_PROFILE:
        payload = state.image_positions
        if (not isinstance(payload, tuple) or not payload
                or any(type(position) is not int for position in payload)
                or tuple(sorted(set(payload))) != payload
                or not all(0 <= position < site.start for position in payload)):
            raise ValueError("invalid original image positions")
    device = state.embeds.device
    embeds = state.embeds.clone()
    positions = state.positions
    visibility = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    if analysis_profile == IMAGE_PROFILE:
        if condition in ("image_mask", "image_body_mask"):
            visibility[site.end:, list(state.image_positions)] = False
        if condition == "image_body_mask":
            visibility[site.end:, site.body_start:site.body_end] = False
    else:
        if condition in ("body_mask", "full_mask"):
            visibility[site.end:, site.body_start:site.body_end] = False
        if condition in ("boundary_mask", "full_mask"):
            visibility[site.end:, site.start] = False
            embeds[:, site.end] = 0
    mask = torch.zeros((1, 1, length, length), dtype=embeds.dtype, device=device)
    mask.masked_fill_(~visibility[None, None], float("-inf"))
    return embeds, positions, mask


@contextmanager
def _image_visibility_observer(qwen, state, site, condition):
    """Check decoder inputs and effective per-layer masks on the live call path."""
    length = state.embeds.shape[1]
    blocked = set() if condition == "clean" else set(state.image_positions)
    if condition == "image_body_mask":
        blocked.update(range(site.body_start, site.body_end))
    query = torch.arange(site.end, site.target_positions[-1], device=state.embeds.device)
    key = torch.arange(length, device=state.embeds.device)
    expected = key[None, :] <= query[:, None]
    if blocked:
        expected[:, sorted(blocked)] = False
    observed, decoder_calls, handles = set(), [], []

    def before_decoder(module, args, kwargs):
        decoder_calls.append(1)
        if len(decoder_calls) != 1:
            raise ValueError("visibility requires exactly one decoder call")
        for name, original in (("inputs_embeds", state.embeds), ("position_ids", state.positions)):
            actual = kwargs.get(name)
            if (not isinstance(actual, torch.Tensor) or actual.dtype != original.dtype
                    or actual.device != original.device or not torch.equal(actual, original)):
                label = "embeddings" if name == "inputs_embeds" else "positions"
                raise ValueError(f"image replay {label} changed")

    def before_attention(index):
        def observe(module, args, kwargs):
            if index in observed:
                raise ValueError("duplicate visibility layer call")
            observed.add(index)
            positions = kwargs.get("position_ids")
            if (not isinstance(positions, torch.Tensor)
                    or positions.dtype != state.positions.dtype
                    or not torch.equal(positions, state.positions)):
                raise ValueError("image replay layer positions changed")
            mask = kwargs.get("attention_mask")
            if (not isinstance(mask, torch.Tensor) or mask.shape != (1, 1, length, length)
                    or not mask.is_floating_point() or mask.device != state.embeds.device):
                raise ValueError("image replay visibility mask dimensions or type changed")
            effective = mask[0, 0, site.end:site.target_positions[-1]]
            if (not torch.equal(torch.isfinite(effective), expected)
                    or not bool((effective[expected] == 0).all())
                    or not bool(torch.isneginf(effective[~expected]).all())):
                raise ValueError("image replay effective visibility changed")
        return observe

    try:
        handles.append(qwen.model.register_forward_pre_hook(before_decoder, with_kwargs=True))
        for index, layer in enumerate(qwen.model.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(
                before_attention(index), with_kwargs=True))
        yield
        if len(decoder_calls) != 1 or len(observed) != len(qwen.model.layers):
            raise ValueError("missing visibility layer call")
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
@numerical_policy()
def replay_site(model, state: CleanState, site: Site, condition: str, *, prefix_witness=None,
                visibility_witness=None, analysis_profile=DEFAULT_PROFILE) -> list[dict]:
    qwen = resolve_stage2_qwen_model(model)
    embeds, positions, mask = build_replay_inputs(state, site, condition, analysis_profile=analysis_profile)
    _validate_kv(qwen, state.prefix_kv, embeds)
    observer = (_image_visibility_observer(qwen, state, site, condition)
                if analysis_profile == IMAGE_PROFILE else nullcontext())
    with observer:
        output = qwen.model(inputs_embeds=embeds, position_ids=positions,
            attention_mask=mask, stage2_capture_kv=True,
            generate_mode=True, use_cache=False, return_dict=True)
    _validate_kv(qwen, output.stage2_key_values, embeds)
    for actual_pair, clean_pair in zip(output.stage2_key_values, state.prefix_kv):
        for actual, clean in zip(actual_pair, clean_pair):
            if actual.dtype != clean.dtype:
                raise ValueError("full-length prefix K/V dtype changed")
            if not torch.equal(actual[..., :site.end, :], clean[..., :site.end, :]):
                raise ValueError("full-length prefix K/V changed")
    hidden = output.last_hidden_state
    _finite(hidden, "replayed hidden states")
    scores = _scores(qwen, hidden, [p - 1 for p in site.target_positions], site.target_ids)
    if prefix_witness is not None:
        layers = len(qwen.model.layers)
        prefix_witness.append({"plan_index": site.plan_index, "condition": condition,
            "prefix_length": site.end, "layers": layers, "kv_comparisons": 2*layers, "exact": True})
    if analysis_profile == IMAGE_PROFILE and visibility_witness is not None:
        blocked = set() if condition == "clean" else set(state.image_positions)
        if condition == "image_body_mask":
            blocked.update(range(site.body_start, site.body_end))
        visibility_witness.append({"plan_index": site.plan_index, "condition": condition,
            "query_start": site.end, "query_stop": site.target_positions[-1],
            "blocked_positions": sorted(blocked), "layers_observed": len(qwen.model.layers),
            "exact": True, "embeddings_unchanged": True})
    return [{"source_index": site.source_index, "plan_index": site.plan_index,
             "target_position": position, "target_id": token_id,
             "condition": condition, "argmax_id": argmax, "nll": nll}
            for position, token_id, (argmax, nll) in zip(site.target_positions, site.target_ids, scores)]


def _comparison(left, right):
    if not left or left.keys() != right.keys():
        raise ValueError("fidelity comparison requires identical nonempty target sets")
    differences = [abs(left[key][1] - right[key][1]) for key in left]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("nonfinite fidelity scores")
    result = {"tokens": len(left),
        "argmax_mismatches": sum(left[key][0] != right[key][0] for key in left),
        "max_nll_abs_diff": max(differences),
        "mean_nll_abs_diff": math.fsum(differences)/len(differences)}
    if (result["argmax_mismatches"] or result["max_nll_abs_diff"] > .01
            or result["mean_nll_abs_diff"] > .001):
        raise ValueError(f"clean fidelity threshold failed: {result}")
    return result


@torch.inference_mode()
@numerical_policy()
def analyze_sample(model, sample: PreparedSample, repeat_clean=False, *, analysis_profile=DEFAULT_PROFILE):
    conditions = conditions_for_profile(analysis_profile)
    state = capture_clean(model, sample, analysis_profile=analysis_profile)
    witness = {"normal_vs_captured": _comparison(state.normal_scores, state.captured_scores)}
    prefix_identity, repeat_prefix_identity, visibility_identity = [], [], []
    rows = [row for condition in conditions for site in sample.sites
            for row in replay_site(model, state, site, condition, prefix_witness=prefix_identity,
                visibility_witness=visibility_identity, analysis_profile=analysis_profile)]
    clean = {(row["plan_index"], row["target_position"]): (row["argmax_id"], row["nll"])
             for row in rows if row["condition"] == "clean"}
    witness["normal_vs_replay"] = _comparison(state.normal_scores, clean)
    witness["captured_vs_replay"] = _comparison(state.captured_scores, clean)
    witness["repeat_clean"] = None
    if repeat_clean:
        repeated = capture_clean(model, sample, analysis_profile=analysis_profile)
        repeated_rows = [row for site in sample.sites for row in replay_site(
            model, repeated, site, "clean", prefix_witness=repeat_prefix_identity,
            analysis_profile=analysis_profile)]
        repeated_scores = {(row["plan_index"], row["target_position"]): (row["argmax_id"], row["nll"])
                           for row in repeated_rows}
        witness["repeat_clean"] = _comparison(clean, repeated_scores)
        witness["repeat_normal"] = _comparison(state.normal_scores, repeated.normal_scores)
        witness["repeat_captured"] = _comparison(repeated.normal_scores, repeated.captured_scores)
        if analysis_profile == IMAGE_PROFILE:
            witness["repeat_normal_vs_captured"] = witness["repeat_captured"]
            witness["repeat_captured"] = _comparison(state.captured_scores, repeated.captured_scores)
    if analysis_profile == IMAGE_PROFILE:
        witness.update({"analysis_profile": analysis_profile, "condition_order": list(conditions),
            "image_token_contract": state.image_token_contract,
            "original_image_positions": list(state.image_positions),
            "visibility_identity": visibility_identity})
    witness.update({"source_index": sample.source_index, "finite": True,
        "dtype": str(state.embeds.dtype), "device": str(state.embeds.device),
        "embeds_shape": list(state.embeds.shape), "positions_shape": list(state.positions.shape),
        "position_columns": state.positions.cpu().tolist(),
        "prefix_kv_shapes": [[list(t.shape) for t in pair] for pair in state.prefix_kv],
        "dispatch": state.dispatch, "execution_strategy": "full_length_v2",
        "numerical_policy": numerical_policy_identity(),
        "prefix_identity": prefix_identity, "repeat_prefix_identity": repeat_prefix_identity})
    return rows, witness

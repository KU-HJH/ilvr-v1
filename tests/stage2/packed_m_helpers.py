"""Real tiny-Qwen fixtures and the independent packed-M replay oracle."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

from peft_common import resolve_qwen2_5_vl_model
from stage2.objective import target_log_likelihood
from stage2.objective_runtime import model_inputs
from stage2.objective_windows import TextVocabulary, select_objective_windows
from stage2.peft import (
    ALLOWED_STAGE2_TARGET_SUFFIXES,
    Stage2TuningConfig,
    build_stage2_tuned_model,
)
from stage2.rng import replay_rng_state
from .helpers import activate_stage2_adapter_deltas


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)
TEXT_VOCABULARY = TextVocabulary(
    tuple(token_id for token_id in range(128) if token_id not in (0, 80, 81, 82, 90, 91, 92)),
    {},
    128,
)


def make_packed_model(*, backend="eager", tuning="dora", checkpointing=False):
    """Return a two-layer indexed Qwen with the production Stage 2 adapter shape."""
    torch.manual_seed(101)
    config = Qwen2_5_VLConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 2,
            "out_hidden_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=80,
        vision_start_token_id=81,
        video_token_id=82,
        stage="stage2",
        latent_size=2,
        latent_pad_token_id=90,
        latent_start_token_id=91,
        latent_end_token_id=92,
        supervise_latent_boundaries=True,
        use_cache=False,
        attention_dropout=0.0,
        attn_implementation=backend,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    if tuning in ("lora", "dora"):
        model = build_stage2_tuned_model(
            model,
            Stage2TuningConfig(
                method=tuning,
                r=4,
                alpha=8,
                dropout=0.05,
                target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
            ),
        )
        activate_stage2_adapter_deltas(model, value=0.125)
    elif tuning != "full":
        raise ValueError(f"unsupported fixture tuning: {tuning}")
    model.train()
    model.config.use_cache = False
    if checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def make_packed_inputs(*, blocks=2, padding=None):
    """Return one image-bearing Stage 2 row with selectable block count/padding."""
    if blocks not in (1, 2, 3):
        raise ValueError("fixture blocks must be 1, 2, or 3")
    # Two real image patches make the three mRoPE axes observably nontrivial.
    ids = [81, 80, 80, 7, 8]
    segments = []
    segment_start = 0
    for block_index in range(blocks):
        start = len(ids)
        segments.append((segment_start, start))
        ids.extend((91, 90, 90, 92))
        ids.extend((9 + block_index * 3, 10 + block_index * 3, 11 + block_index * 3))
        segment_start = start + 4
    segments.append((segment_start, len(ids)))

    if padding == "left":
        ids.insert(0, 0)
        segments = [(start + 1, end + 1) for start, end in segments]
        segments[0] = (0, segments[0][1])
    elif padding == "right":
        ids.append(0)
        segments[-1] = (segments[-1][0], len(ids))
    elif padding is not None:
        raise ValueError("padding must be None, left, or right")

    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = input_ids.ne(0).long()
    labels = input_ids.clone()
    labels[(input_ids == 0) | (input_ids == 90)] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": torch.arange(24, dtype=torch.float32).reshape(2, 12) / 24.0,
        "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
        "stage2_text_segments": [segments],
    }


def select_packed_blocks(inputs, *, post_window=2):
    return select_objective_windows(
        inputs["input_ids"],
        inputs["labels"],
        inputs["attention_mask"],
        inputs["stage2_text_segments"],
        token_ids=TOKENS,
        latent_size=2,
        text_vocabulary=TEXT_VOCABULARY,
        pre_window=2,
        post_window=post_window,
    ).blocks


def independent_full_replay_scores(model, inputs, blocks, pre_f_rng):
    """Run the unchanged blockwise full-sequence M oracle from the paired RNG."""
    clean = {key: value for key, value in model_inputs(inputs).items() if key != "labels"}
    scores = []
    with torch.no_grad():
        for block in blocks:
            with replay_rng_state(pre_f_rng):
                output = model(
                    **clean,
                    stage2_ablation=(block.start_position, block.end_position),
                )
            logits = output.logits[block.batch_index, list(block.post_positions)]
            targets = torch.tensor(block.target_ids, device=logits.device)
            scores.append(target_log_likelihood(logits, targets).mean())
    return torch.stack(scores) if scores else torch.empty(0)


def independent_full_replay_logits(model, inputs, blocks, pre_f_rng):
    """Observe the oracle M LM-head outputs at the selected predictor rows."""
    clean = {key: value for key, value in model_inputs(inputs).items() if key != "labels"}
    selected_logits = []
    lm_head = resolve_qwen2_5_vl_model(model).lm_head
    with torch.no_grad():
        for block in blocks:
            observed = []
            handle = lm_head.register_forward_hook(
                lambda _module, _args, value: observed.append(value.detach().clone())
            )
            try:
                with replay_rng_state(pre_f_rng):
                    model(
                        **clean,
                        stage2_ablation=(block.start_position, block.end_position),
                    )
            finally:
                handle.remove()
            assert len(observed) == 1
            selected_logits.append(observed[0][block.batch_index, list(block.post_positions)])
    return torch.cat(selected_logits, dim=0) if selected_logits else torch.empty(0)


def autocast_for(dtype):
    return (
        torch.autocast("cpu", dtype=torch.bfloat16)
        if dtype == torch.bfloat16
        else nullcontext()
    )


def decoder(model):
    return resolve_qwen2_5_vl_model(model).model

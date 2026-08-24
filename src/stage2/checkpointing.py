"""Validate Stage 2 token metadata and checkpoint transitions."""

from dataclasses import dataclass

from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentTokenIds,
    validate_stage2_config,
)


LATENT_TOKEN_STRINGS = {
    "pad_token_id": "<|latent_pad|>",
    "start_token_id": "<|latent_start|>",
    "end_token_id": "<|latent_end|>",
}

LATENT_CONFIG_FIELDS = {
    "pad_token_id": ("latent_pad_token_id", "latent_token_id"),
    "start_token_id": ("latent_start_token_id", "latent_start_id"),
    "end_token_id": ("latent_end_token_id", "latent_end_id"),
}


def resolve_latent_token_ids(tokenizer, config) -> LatentTokenIds:
    """Resolve latent IDs without modifying the tokenizer or its vocabulary.

    Every latent token must be present in the tokenizer. Any canonical or legacy
    config alias that is present must agree with the tokenizer's ID.
    """
    resolved: dict[str, int] = {}
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    unknown_token = getattr(tokenizer, "unk_token", None)

    for result_field, token in LATENT_TOKEN_STRINGS.items():
        tokenizer_id = tokenizer.convert_tokens_to_ids(token)
        if tokenizer_id is None or (
            unknown_id is not None
            and tokenizer_id == unknown_id
            and token != unknown_token
        ):
            raise ValueError(f"tokenizer does not contain {token}")

        tokenizer_id = int(tokenizer_id)
        resolved[result_field] = tokenizer_id

    tokenizer_ids = LatentTokenIds(**resolved)
    validate_stage2_config(
        config,
        fallback_token_ids=tokenizer_ids,
        require_canonical_ids=False,
        require_latent_size=False,
    )
    for result_field, tokenizer_id in resolved.items():
        canonical_field = LATENT_CONFIG_FIELDS[result_field][0]
        for config_field in LATENT_CONFIG_FIELDS[result_field]:
            config_id = getattr(config, config_field, None)
            if config_id is not None and int(config_id) != tokenizer_id:
                alias = "" if config_field == canonical_field else f" via {config_field}"
                raise ValueError(
                    f"{canonical_field} mismatch{alias}: "
                    f"tokenizer={tokenizer_id}, config={config_id}"
                )

    return tokenizer_ids


def configure_stage2_metadata(
    config,
    token_ids: LatentTokenIds,
    *,
    latent_size: int,
    supervise_latent_boundaries: bool,
) -> None:
    """Write the complete, backwards-compatible Stage 2 config metadata."""
    config.stage = "stage2"
    config.latent_size = latent_size
    config.supervise_latent_boundaries = supervise_latent_boundaries
    config.latent_pad_token_id = config.latent_token_id = token_ids.pad_token_id
    config.latent_start_token_id = config.latent_start_id = token_ids.start_token_id
    config.latent_end_token_id = config.latent_end_id = token_ids.end_token_id


@dataclass(frozen=True)
class Stage2LoadDecision:
    """The model, processor, and trainer sources for one Stage 2 run."""

    model_source: str
    processor_source: str
    trainer_resume_path: str | None
    latent_size: int
    supervise_latent_boundaries: bool


def resolve_stage2_load_decision(
    *,
    load_model_path,
    resume_from_checkpoint_path,
    load_config,
    latent_size: int,
    supervise_latent_boundaries: bool,
) -> Stage2LoadDecision:
    """Choose fresh Stage 2 initialization or a complete Stage 2 resume.

    A fresh run reads model and processor artifacts from a Stage 1 source but
    intentionally returns no trainer resume path, so optimizer/scheduler/global
    state starts new. A resume uses only the Stage 2 checkpoint source for all
    three consumers.
    """
    if resume_from_checkpoint_path is None:
        if getattr(load_config(load_model_path), "stage", None) != "stage1":
            raise ValueError("fresh Stage 2 requires a stage1 checkpoint")
        return Stage2LoadDecision(
            load_model_path,
            load_model_path,
            None,
            latent_size,
            supervise_latent_boundaries,
        )

    resume_config = load_config(resume_from_checkpoint_path)
    if getattr(resume_config, "stage", None) != "stage2":
        raise ValueError("expected stage2 checkpoint for resume")
    for field, requested in (
        ("latent_size", latent_size),
        ("supervise_latent_boundaries", supervise_latent_boundaries),
    ):
        checkpoint_value = getattr(resume_config, field, None)
        if checkpoint_value != requested:
            raise ValueError(
                f"cannot resume Stage 2 with changed {field}: "
                f"checkpoint={checkpoint_value}, requested={requested}"
            )
    return Stage2LoadDecision(
        resume_from_checkpoint_path,
        resume_from_checkpoint_path,
        resume_from_checkpoint_path,
        resume_config.latent_size,
        resume_config.supervise_latent_boundaries,
    )

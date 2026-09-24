"""Resolve rendering and checkpoint placement together, including saved execution."""
from dataclasses import dataclass
from typing import Mapping

from step_preference.config import StepPreferenceConfig
from step_preference.spans import normalize_assistant_plan_rendering, normalize_preference_target_scope


@dataclass(frozen=True)
class Stage2ExecutionSettings:
    assistant_plan_rendering: str
    native_checkpointing: bool
    latent_checkpointing: bool
    sidecar_checkpointing: bool


def resolve_stage2_execution(config: StepPreferenceConfig, *, native_checkpointing: bool,
                            latent_checkpointing: bool | None = None,
                            saved: Mapping | None = None) -> Stage2ExecutionSettings:
    if saved is not None:
        fields = ("assistant_plan_rendering", "native_checkpointing",
                  "latent_checkpointing", "sidecar_checkpointing")
        if not isinstance(saved, Mapping) or any(key not in saved for key in (*fields, "target_scope")):
            raise ValueError("missing saved execution settings")
        scope = normalize_preference_target_scope(saved["target_scope"])
        if scope != config.target_scope:
            raise ValueError("saved target scope conflicts with requested scope")
        normalize_assistant_plan_rendering(saved["assistant_plan_rendering"])
        if any(type(saved[key]) is not bool for key in fields[1:]):
            raise ValueError("saved checkpoint settings must be booleans")
        if saved["native_checkpointing"] and saved["latent_checkpointing"]:
            raise ValueError("saved native and latent checkpointing conflict")
        return Stage2ExecutionSettings(**{key: saved[key] for key in fields})
    until = config.is_context_preference and config.target_scope == "until_next_latent"
    latent = until if latent_checkpointing is None else latent_checkpointing
    if type(native_checkpointing) is not bool or type(latent) is not bool:
        raise ValueError("checkpoint settings must be booleans")
    if native_checkpointing and (until or latent):
        raise ValueError("native checkpointing conflicts with until/latent checkpointing")
    return Stage2ExecutionSettings(
        assistant_plan_rendering=("concatenated-v1" if until or not config.enabled
                                  else "adjacent-text-space-lf-v1"),
        native_checkpointing=native_checkpointing,
        latent_checkpointing=latent,
        sidecar_checkpointing=not until,
    )

"""Version-2 configuration for exact Stage-2 evidence-window credit."""

from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class Stage2ReferenceConfig:
    mode: Literal["none", "precomputed", "online_frozen"] = "none"
    checkpoint: str | None = None
    artifact: str | None = None


@dataclass(frozen=True)
class Stage2PreferenceConfig:
    enabled: bool = False
    lambda_ew: float = 0.0
    beta: float = 0.1
    contribution_reduction: Literal["mean", "sum"] = "mean"


@dataclass(frozen=True)
class Stage2RelationConfig:
    enabled: bool = False
    lambda_rel: float = 0.0
    temperature: float = 1.0


@dataclass(frozen=True)
class Stage2ResidualConfig:
    enabled: bool = False
    lambda_res: float = 0.0
    huber_delta: float = 1.0


@dataclass(frozen=True)
class Stage2LOOConfig:
    intervention: Literal["window_attention_block"] = "window_attention_block"
    max_evidence_edges_per_window: int = 0


@dataclass(frozen=True)
class Stage2ScheduleConfig:
    ew_warmup_ratio: float = 0.0
    ew_ramp_ratio: float = 0.0
    relation_warmup_ratio: float = 0.0
    relation_ramp_ratio: float = 0.0
    residual_warmup_ratio: float = 0.0
    residual_ramp_ratio: float = 0.0


@dataclass(frozen=True)
class Stage2DiagnosticsConfig:
    contribution_threshold: float = 0.0
    matrix_dump_interval: int = 0
    matrix_dump_dir: str = "stage2_evidence_window_matrices"


@dataclass(frozen=True)
class Stage2EvidenceWindowConfig:
    """Validated Stage-2 objective configuration.

    A positive coefficient is active only when its corresponding ``enabled``
    flag is true.  This makes zero-coefficient experiments inert without
    requiring callers to special-case enabled flags.
    """

    version: int = 2
    reference: Stage2ReferenceConfig = field(default_factory=Stage2ReferenceConfig)
    preference: Stage2PreferenceConfig = field(default_factory=Stage2PreferenceConfig)
    relation: Stage2RelationConfig = field(default_factory=Stage2RelationConfig)
    residual: Stage2ResidualConfig = field(default_factory=Stage2ResidualConfig)
    loo: Stage2LOOConfig = field(default_factory=Stage2LOOConfig)
    schedule: Stage2ScheduleConfig = field(default_factory=Stage2ScheduleConfig)
    diagnostics: Stage2DiagnosticsConfig = field(default_factory=Stage2DiagnosticsConfig)

    @classmethod
    def disabled(cls) -> "Stage2EvidenceWindowConfig":
        return cls()

    @classmethod
    def from_json(cls, path: str | Path) -> "Stage2EvidenceWindowConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Stage2EvidenceWindowConfig":
        if not isinstance(raw, dict):
            raise ValueError("stage-2 evidence-window configuration must be a JSON object")

        _reject_unknown_keys(raw, cls, "top-level")
        sections: dict[str, type[Any]] = {
            "reference": Stage2ReferenceConfig,
            "preference": Stage2PreferenceConfig,
            "relation": Stage2RelationConfig,
            "residual": Stage2ResidualConfig,
            "loo": Stage2LOOConfig,
            "schedule": Stage2ScheduleConfig,
            "diagnostics": Stage2DiagnosticsConfig,
        }
        parsed: dict[str, Any] = {"version": raw.get("version", 2)}
        for name, config_type in sections.items():
            section = raw.get(name, {})
            if not isinstance(section, dict):
                raise ValueError(f"{name} must be a JSON object")
            _reject_unknown_keys(section, config_type, name)
            parsed[name] = config_type(**section)

        config = cls(**parsed)
        config.validate()
        return config

    def validate(self) -> None:
        if not _is_integer(self.version) or self.version != 2:
            raise ValueError("version must be 2")

        if self.reference.mode not in {"none", "precomputed", "online_frozen"}:
            raise ValueError("unsupported reference mode")
        _require_optional_nonempty_string(self.reference.checkpoint, "reference.checkpoint")
        _require_optional_nonempty_string(self.reference.artifact, "reference.artifact")
        if self.reference.mode == "precomputed" and self.reference.artifact is None:
            raise ValueError("precomputed reference requires an artifact")
        if self.reference.mode == "online_frozen" and self.reference.checkpoint is None:
            raise ValueError("online_frozen reference requires a checkpoint")

        _require_bool(self.preference.enabled, "preference.enabled")
        _require_nonnegative(self.preference.lambda_ew, "preference.lambda_ew")
        _require_positive(self.preference.beta, "preference.beta")
        if self.preference.contribution_reduction not in {"mean", "sum"}:
            raise ValueError("unsupported preference.contribution_reduction")

        _require_bool(self.relation.enabled, "relation.enabled")
        _require_nonnegative(self.relation.lambda_rel, "relation.lambda_rel")
        _require_positive(self.relation.temperature, "relation.temperature")

        _require_bool(self.residual.enabled, "residual.enabled")
        _require_nonnegative(self.residual.lambda_res, "residual.lambda_res")
        _require_positive(self.residual.huber_delta, "residual.huber_delta")

        if self.loo.intervention != "window_attention_block":
            raise ValueError("unsupported LOO intervention")
        _require_nonnegative_integer(
            self.loo.max_evidence_edges_per_window,
            "loo.max_evidence_edges_per_window",
        )

        for name, value in self._schedule_values().items():
            _require_ratio(value, f"schedule.{name}")
        for objective in ("ew", "relation", "residual"):
            warmup, ramp = self._schedule_for(objective)
            if warmup + ramp > 1.0:
                raise ValueError(f"schedule.{objective} warmup and ramp ratios must total at most 1")

        _require_nonnegative(
            self.diagnostics.contribution_threshold,
            "diagnostics.contribution_threshold",
        )
        _require_nonnegative_integer(
            self.diagnostics.matrix_dump_interval,
            "diagnostics.matrix_dump_interval",
        )
        _require_nonempty_string(self.diagnostics.matrix_dump_dir, "diagnostics.matrix_dump_dir")

        if self._relation_enabled() and self.reference.mode == "none":
            raise ValueError("relation objective requires a reference")
        if self._residual_enabled() and self.reference.mode == "none":
            raise ValueError("residual objective requires a reference")

    @property
    def is_disabled(self) -> bool:
        return not (
            self._preference_enabled()
            or self._relation_enabled()
            or self._residual_enabled()
        )

    @property
    def required_outputs(self) -> frozenset[str]:
        if self.is_disabled:
            return frozenset()
        outputs = {"token_logps"}
        if self._residual_enabled():
            outputs.add("hidden_states")
        return frozenset(outputs)

    def validate_runtime(self, zero_stage: int) -> None:
        if self.is_disabled:
            return
        if not _is_integer(zero_stage) or zero_stage not in {0, 1, 2, 3}:
            raise ValueError("ZeRO stage must be an integer from 0 through 3")
        if self.reference.mode != "online_frozen":
            return
        if zero_stage == 3:
            raise ValueError("online_frozen reference is incompatible with ZeRO-3")
        if zero_stage not in {1, 2}:
            raise ValueError("online_frozen reference requires ZeRO-1 or ZeRO-2")

    def effective_weight(self, name: Literal["ew", "relation", "residual"], progress: float) -> float:
        _require_ratio(progress, "progress")
        coefficients = {
            "ew": self.preference.lambda_ew if self._preference_enabled() else 0.0,
            "relation": self.relation.lambda_rel if self._relation_enabled() else 0.0,
            "residual": self.residual.lambda_res if self._residual_enabled() else 0.0,
        }
        if name not in coefficients:
            raise ValueError(f"unsupported stage-2 objective: {name}")

        coefficient = coefficients[name]
        if coefficient == 0.0:
            return 0.0
        warmup, ramp = self._schedule_for(name)
        if ramp == 0.0:
            return float(coefficient) if progress >= warmup else 0.0
        if progress <= warmup:
            return 0.0
        if progress >= warmup + ramp:
            return float(coefficient)
        return float(coefficient) * ((progress - warmup) / ramp)

    def _preference_enabled(self) -> bool:
        return self.preference.enabled and self.preference.lambda_ew > 0.0

    def _relation_enabled(self) -> bool:
        return self.relation.enabled and self.relation.lambda_rel > 0.0

    def _residual_enabled(self) -> bool:
        return self.residual.enabled and self.residual.lambda_res > 0.0

    def _schedule_values(self) -> dict[str, float]:
        return {
            "ew_warmup_ratio": self.schedule.ew_warmup_ratio,
            "ew_ramp_ratio": self.schedule.ew_ramp_ratio,
            "relation_warmup_ratio": self.schedule.relation_warmup_ratio,
            "relation_ramp_ratio": self.schedule.relation_ramp_ratio,
            "residual_warmup_ratio": self.schedule.residual_warmup_ratio,
            "residual_ramp_ratio": self.schedule.residual_ramp_ratio,
        }

    def _schedule_for(self, name: str) -> tuple[float, float]:
        schedules = {
            "ew": (self.schedule.ew_warmup_ratio, self.schedule.ew_ramp_ratio),
            "relation": (
                self.schedule.relation_warmup_ratio,
                self.schedule.relation_ramp_ratio,
            ),
            "residual": (
                self.schedule.residual_warmup_ratio,
                self.schedule.residual_ramp_ratio,
            ),
        }
        return schedules[name]


def read_deepspeed_zero_stage(path: str | Path) -> int:
    """Read a DeepSpeed JSON file and return its configured ZeRO stage."""

    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("DeepSpeed configuration must be a JSON object")

    zero_optimization = raw.get("zero_optimization", {})
    if not isinstance(zero_optimization, dict):
        raise ValueError("DeepSpeed zero_optimization must be a JSON object")
    stage = zero_optimization.get("stage", 0)
    if not _is_integer(stage) or stage not in {0, 1, 2, 3}:
        raise ValueError("DeepSpeed zero_optimization.stage must be an integer from 0 through 3")
    return stage


def _reject_unknown_keys(data: dict[str, Any], config_type: type[Any], location: str) -> None:
    allowed = {item.name for item in fields(config_type)}
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {location} configuration key(s): {names}")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _require_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _require_nonnegative(value: Any, name: str) -> None:
    _require_number(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_positive(value: Any, name: str) -> None:
    _require_number(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_ratio(value: Any, name: str) -> None:
    _require_number(value, name)
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_nonnegative_integer(value: Any, name: str) -> None:
    if not _is_integer(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_optional_nonempty_string(value: Any, name: str) -> None:
    if value is not None:
        _require_nonempty_string(value, name)


__all__ = [
    "Stage2DiagnosticsConfig",
    "Stage2EvidenceWindowConfig",
    "Stage2LOOConfig",
    "Stage2PreferenceConfig",
    "Stage2ReferenceConfig",
    "Stage2RelationConfig",
    "Stage2ResidualConfig",
    "Stage2ScheduleConfig",
    "read_deepspeed_zero_stage",
]

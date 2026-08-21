"""Strict, versioned configuration for evidence-window objectives."""

from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ReferenceConfig:
    mode: Literal["none", "adapter_disabled", "frozen_copy"] = "none"


@dataclass(frozen=True)
class PreferenceConfig:
    beta: float = 0.1
    lambda_global: float = 0.0
    lambda_window: float = 0.0
    lambda_ew: float = 0.0


@dataclass(frozen=True)
class RelationConfig:
    lambda_jsd: float = 0.0
    lambda_magnitude: float = 0.0
    temperature: float = 1.0
    huber_delta: float = 1.0


@dataclass(frozen=True)
class ResidualConfig:
    lambda_logit: float = 0.0
    metric: Literal["huber", "cosine"] = "huber"
    huber_delta: float = 1.0
    vocab_chunk_size: int = 4096
    max_tokens_per_window: int = 0


@dataclass(frozen=True)
class LOOConfig:
    intervention: Literal["zero_latent"] = "zero_latent"
    max_spans_per_example: int = 0


@dataclass(frozen=True)
class DiagnosticsConfig:
    global_saturation: bool = False
    contribution_threshold: float = 0.0
    matrix_dump_interval: int = 0
    matrix_dump_dir: str = "evidence_window_matrices"


@dataclass(frozen=True)
class EvidenceWindowConfig:
    version: int = 1
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    relation: RelationConfig = field(default_factory=RelationConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    loo: LOOConfig = field(default_factory=LOOConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    @classmethod
    def disabled(cls) -> "EvidenceWindowConfig":
        return cls()

    @classmethod
    def from_json(cls, path: str | Path) -> "EvidenceWindowConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("evidence-window configuration must be a JSON object")

        _reject_unknown_keys(raw, cls, "top-level")
        sections = {
            "reference": ReferenceConfig,
            "preference": PreferenceConfig,
            "relation": RelationConfig,
            "residual": ResidualConfig,
            "loo": LOOConfig,
            "diagnostics": DiagnosticsConfig,
        }
        parsed: dict[str, Any] = {"version": raw.get("version", 1)}
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
        if not _is_integer(self.version) or self.version != 1:
            raise ValueError("version must be 1")
        if (
            not isinstance(self.reference.mode, str)
            or self.reference.mode not in {"none", "adapter_disabled", "frozen_copy"}
        ):
            raise ValueError("unsupported reference mode")
        if (
            not isinstance(self.residual.metric, str)
            or self.residual.metric not in {"huber", "cosine"}
        ):
            raise ValueError("unsupported residual metric")
        if self.loo.intervention != "zero_latent":
            raise ValueError("unsupported LOO intervention")

        _require_positive(self.preference.beta, "preference.beta")
        _require_positive(self.relation.temperature, "relation.temperature")
        _require_positive(self.relation.huber_delta, "relation.huber_delta")
        _require_positive(self.residual.huber_delta, "residual.huber_delta")
        _require_positive_integer(self.residual.vocab_chunk_size, "residual.vocab_chunk_size")
        _require_nonnegative_integer(
            self.residual.max_tokens_per_window, "residual.max_tokens_per_window"
        )
        _require_nonnegative_integer(
            self.loo.max_spans_per_example, "loo.max_spans_per_example"
        )
        _require_nonnegative_integer(
            self.diagnostics.matrix_dump_interval,
            "diagnostics.matrix_dump_interval",
        )
        _require_number(
            self.diagnostics.contribution_threshold,
            "diagnostics.contribution_threshold",
        )
        if not isinstance(self.diagnostics.global_saturation, bool):
            raise ValueError("diagnostics.global_saturation must be a boolean")
        if not isinstance(self.diagnostics.matrix_dump_dir, str):
            raise ValueError("diagnostics.matrix_dump_dir must be a string")

        for name, value in self._coefficients().items():
            _require_nonnegative(value, name)

        if self._needs_reference() and self.reference.mode == "none":
            raise ValueError(
                "preference and reference-adjusted diagnostics require a fixed reference"
            )

    @property
    def is_disabled(self) -> bool:
        return (
            all(value == 0.0 for value in self._coefficients().values())
            and not self.diagnostics.global_saturation
            and self.diagnostics.matrix_dump_interval == 0
        )

    def required_conditions(self, role: str) -> frozenset[str]:
        if role not in {"policy", "reference", "ema"}:
            raise ValueError(f"unsupported evidence-window role: {role}")
        if self.is_disabled:
            return frozenset()

        needs_global = (
            self.preference.lambda_global > 0
            or self.preference.lambda_window > 0
            or self.diagnostics.global_saturation
        )
        needs_loo_preference = self.preference.lambda_ew > 0
        needs_relation = (
            self.relation.lambda_jsd > 0
            or self.relation.lambda_magnitude > 0
            or self.residual.lambda_logit > 0
        )

        if role == "policy":
            conditions: set[str] = set()
            if needs_global:
                conditions.update({"full", "all_null"})
            if needs_loo_preference or needs_relation:
                conditions.update({"full", "loo"})
            return frozenset(conditions)
        if role == "reference":
            if needs_global:
                return frozenset({"full", "all_null"})
            if needs_loo_preference:
                return frozenset({"full", "loo"})
            return frozenset()
        if needs_relation:
            return frozenset({"full", "loo"})
        return frozenset()

    def _coefficients(self) -> dict[str, float]:
        return {
            "preference.lambda_global": self.preference.lambda_global,
            "preference.lambda_window": self.preference.lambda_window,
            "preference.lambda_ew": self.preference.lambda_ew,
            "relation.lambda_jsd": self.relation.lambda_jsd,
            "relation.lambda_magnitude": self.relation.lambda_magnitude,
            "residual.lambda_logit": self.residual.lambda_logit,
        }

    def _needs_reference(self) -> bool:
        return (
            self.preference.lambda_global > 0
            or self.preference.lambda_window > 0
            or self.preference.lambda_ew > 0
            or self.diagnostics.global_saturation
        )


def _reject_unknown_keys(data: dict[str, Any], config_type: type[Any], location: str) -> None:
    allowed = {item.name for item in fields(config_type)}
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {location} configuration key(s): {names}")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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


def _require_positive_integer(value: Any, name: str) -> None:
    if not _is_integer(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(value: Any, name: str) -> None:
    if not _is_integer(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")

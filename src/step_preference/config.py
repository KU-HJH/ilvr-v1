"""Immutable Stage 2A/2B preference-training configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import Literal

from step_preference.spans import normalize_preference_target_scope
from step_preference.replacement import EVIDENCE_REPLACEMENT_SAMPLING_VERSION


StepPreferenceMode = Literal["stage2a", "stage2b"]
StepPreferenceObjective = Literal["context_preference", "legacy_dpo"]


def split_stage2_budget(total_steps: int, stage2a_ratio: float) -> tuple[int, int]:
    """Split one optimizer-step budget into two positive exact-sum phases."""
    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise ValueError("stage2_total_steps must be an integer")
    if total_steps < 2:
        raise ValueError("stage2_total_steps must be at least two")
    if isinstance(stage2a_ratio, bool) or not isinstance(stage2a_ratio, Real):
        raise ValueError("stage2a_ratio must be a finite number in (0, 1)")
    ratio = float(stage2a_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("stage2a_ratio must be a finite number in (0, 1)")
    stage2a_steps = math.floor(total_steps * ratio)
    stage2b_steps = total_steps - stage2a_steps
    if stage2a_steps <= 0 or stage2b_steps <= 0:
        raise ValueError("stage2 budget split must give both phases a positive step count")
    return stage2a_steps, stage2b_steps


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_path(name: str, value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty path when provided")


def _optional_sha256(name: str, value: str | None) -> None:
    if value is not None and (
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 when provided")


def _optional_path_hash_pair(path_name: str, path: str | None, hash_name: str, sha256: str | None) -> None:
    _optional_path(path_name, path)
    _optional_sha256(hash_name, sha256)
    if (path is None) != (sha256 is None):
        raise ValueError(f"{path_name} and {hash_name} must be provided together")


@dataclass(frozen=True)
class StepPreferenceConfig:
    """One validated configuration for ordinary Stage 2 or fresh Stage 2B."""

    enabled: bool = False
    mode: StepPreferenceMode = "stage2a"
    objective: StepPreferenceObjective = "context_preference"
    target_scope: str = "immediate_text"
    beta: float = 0.1
    lambda_ce: float = 1.0
    lambda_dpo: float = 1.0
    lambda_preference: float = 1.0
    preference_margin: float = 0.0
    prefix_mode: str = "clean_prefix"
    branch_mode: str = "independent"
    block_original_image: bool = False
    visual_only_context_preference: bool | None = None
    visual_off_suppress_latent_boundaries: bool | None = None
    lambda_entropy: float = 0.1
    entropy_margin: float = 0.1
    negative_path: str | None = None
    negative_manifest: str | None = None
    min_tokens: int = 2
    max_new_tokens: int = 32
    stage2a_ratio: float = 0.5
    total_steps: int | None = None
    stage2a_checkpoint_path: str | None = None
    direct_from_stage1: bool = False
    attention_implementation: str = "sdpa"
    resume_from_checkpoint_path: str | None = None
    packing_enabled: bool = False
    shared_prefix_enabled: bool = False
    sidecar_attention_backend: str = "flex"
    sidecar_query_budget: int = 1024
    replacement_seed: int = 42
    max_packed_sequence_length: int | None = None
    packed_admission_manifest: str | None = None
    packed_admission_manifest_sha256: str | None = None
    packed_admission_verification: str | None = None
    packed_admission_verification_sha256: str | None = None

    def __post_init__(self) -> None:
        normalize_preference_target_scope(self.target_scope)
        if self.visual_only_context_preference is None:
            object.__setattr__(
                self,
                "visual_only_context_preference",
                self.is_context_preference,
            )
        if self.visual_off_suppress_latent_boundaries is None:
            object.__setattr__(
                self,
                "visual_off_suppress_latent_boundaries",
                self.is_context_preference
                and self.visual_only_context_preference
                and self.prefix_mode != "evidence_replacement",
            )
        if self.prefix_mode not in (
            "clean_prefix", "v_off_prefix", "v_off_insertion", "evidence_replacement"
        ):
            raise ValueError("unknown preference prefix_mode")
        if self.branch_mode not in ("independent", "cumulative"):
            raise ValueError("branch_mode must be independent or cumulative")
        if self.prefix_mode == "evidence_replacement" and self.branch_mode != "independent":
            raise ValueError("evidence_replacement requires independent branch_mode")
        if self.branch_mode == "cumulative" and self.prefix_mode != "v_off_insertion":
            raise ValueError("cumulative branch_mode requires v_off_insertion")
        if self.prefix_mode == "v_off_insertion" and not self.shared_prefix_enabled:
            raise ValueError("v_off_insertion requires shared-prefix execution")
        if type(self.block_original_image) is not bool:
            raise ValueError("block_original_image must be a boolean")
        if (
            self.block_original_image
            and self.prefix_mode not in ("v_off_prefix", "evidence_replacement")
        ):
            raise ValueError("block_original_image requires v_off_prefix")
        if self.prefix_mode == "evidence_replacement":
            if not self.is_context_preference:
                raise ValueError("evidence_replacement requires context-preference Stage 2B")
            if not self.visual_only_context_preference:
                raise ValueError("evidence_replacement requires visual-only context preference")
            if self.visual_off_suppress_latent_boundaries:
                raise ValueError("evidence_replacement requires latent boundaries kept")
            if self.block_original_image:
                raise ValueError("evidence_replacement preserves original image access")
        if self.prefix_mode in ("v_off_prefix", "v_off_insertion") and (
            not self.is_context_preference
            or not self.visual_only_context_preference
            or self.visual_off_suppress_latent_boundaries
        ):
            raise ValueError(f"{self.prefix_mode} requires V-only context preference with START/END kept")
        if self.target_scope != "immediate_text" and not self.is_context_preference:
            raise ValueError("full preference target scope requires context-preference Stage 2B")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(self.packing_enabled, bool):
            raise ValueError("packing_enabled must be a boolean")
        if type(self.shared_prefix_enabled) is not bool:
            raise ValueError("shared_prefix_enabled must be a boolean")
        if type(self.visual_only_context_preference) is not bool:
            raise ValueError("visual_only_context_preference must be a boolean")
        if type(self.visual_off_suppress_latent_boundaries) is not bool:
            raise ValueError(
                "visual_off_suppress_latent_boundaries must be a boolean"
            )
        if type(self.direct_from_stage1) is not bool:
            raise ValueError("direct_from_stage1 must be a boolean")
        if self.sidecar_attention_backend not in ("dense", "flex"):
            raise ValueError("sidecar_attention_backend must be dense or flex")
        if type(self.sidecar_query_budget) is not int or self.sidecar_query_budget <= 0:
            raise ValueError("sidecar_query_budget must be a positive integer")
        if type(self.replacement_seed) is not int:
            raise ValueError("replacement_seed must be an integer")
        if self.shared_prefix_enabled:
            if not self.is_stage2b:
                raise ValueError("shared prefix is only valid for Stage 2B")
            if self.packing_enabled or any((self.packed_admission_manifest,
                    self.packed_admission_manifest_sha256, self.packed_admission_verification,
                    self.packed_admission_verification_sha256)):
                raise ValueError("shared prefix cannot consume legacy packing or packed admission")
        if self.mode not in ("stage2a", "stage2b"):
            raise ValueError("step preference mode must be stage2a or stage2b")
        if self.objective not in ("context_preference", "legacy_dpo"):
            raise ValueError("unknown step preference objective")
        beta = _finite("step_dpo_beta", self.beta)
        if beta <= 0:
            raise ValueError("step_dpo_beta must be positive")
        for name, value in (
            ("ce_weight", self.lambda_ce),
            ("step_dpo_weight", self.lambda_dpo),
            ("step_entropy_weight", self.lambda_entropy),
        ):
            if _finite(name, value) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if _finite("step_preference_weight", self.lambda_preference) < 0:
            raise ValueError("step_preference_weight must be nonnegative")
        if self.prefix_mode == "evidence_replacement" and self.lambda_entropy != 0.0:
            raise ValueError("evidence_replacement requires step_entropy_weight=0")
        _finite("step_preference_margin", self.preference_margin)
        entropy_margin = _finite("step_entropy_margin", self.entropy_margin)
        if entropy_margin < 0:
            raise ValueError("step_entropy_margin must be nonnegative")
        if (
            isinstance(self.min_tokens, bool)
            or not isinstance(self.min_tokens, int)
            or self.min_tokens <= 0
        ):
            raise ValueError("min_tokens must be a positive integer")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        for name in (
            "negative_path",
            "negative_manifest",
            "stage2a_checkpoint_path",
            "resume_from_checkpoint_path",
        ):
            _optional_path(name, getattr(self, name))
        _optional_path_hash_pair(
            "packed_admission_manifest",
            self.packed_admission_manifest,
            "packed_admission_manifest_sha256",
            self.packed_admission_manifest_sha256,
        )
        _optional_path_hash_pair(
            "packed_admission_verification",
            self.packed_admission_verification,
            "packed_admission_verification_sha256",
            self.packed_admission_verification_sha256,
        )
        ratio = _finite("stage2a_ratio", self.stage2a_ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError("stage2a_ratio must be a finite number in (0, 1)")
        if self.total_steps is not None:
            if self.direct_from_stage1:
                if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int) or self.total_steps <= 0:
                    raise ValueError("stage2_total_steps must be a positive integer")
            else:
                split_stage2_budget(self.total_steps, self.stage2a_ratio)
        if self.is_context_preference:
            if self.lambda_ce != 1.0:
                raise ValueError("context preference requires ce_weight=1")
            if self.negative_path is not None or self.negative_manifest is not None:
                raise ValueError(
                    "context preference does not consume negative artifacts; "
                    "select legacy_dpo explicitly"
                )
            if self.packing_enabled or any(
                (
                    self.packed_admission_manifest,
                    self.packed_admission_manifest_sha256,
                    self.packed_admission_verification,
                    self.packed_admission_verification_sha256,
                )
            ):
                raise ValueError(
                    "context preference does not consume legacy packed admission"
                )
        elif self.visual_only_context_preference:
            raise ValueError(
                "visual-only context preference requires context-preference Stage 2B"
            )
        if self.visual_off_suppress_latent_boundaries and not (
            self.is_context_preference and self.visual_only_context_preference
        ):
            raise ValueError(
                "visual-off boundary suppression requires visual-only "
                "context-preference Stage 2B"
            )
        if self.direct_from_stage1 and not (
            self.is_context_preference and self.visual_only_context_preference
        ):
            raise ValueError(
                "direct Stage 1 initialization requires visual-only "
                "context-preference Stage 2B"
            )
        if self.is_stage2b:
            if self.direct_from_stage1:
                if self.stage2a_checkpoint_path is not None:
                    raise ValueError(
                        "--stage2b_direct_from_stage1 cannot use "
                        "--stage2a_checkpoint_path"
                    )
            required = [] if self.direct_from_stage1 else [
                ("stage2a_checkpoint_path", "--stage2a_checkpoint_path")
            ]
            if not self.is_context_preference:
                required += [
                    ("negative_path", "--step_negative_path"),
                    ("negative_manifest", "--step_negative_manifest"),
                ]
            for field_name, option in required:
                if getattr(self, field_name) is None:
                    raise ValueError(f"Stage 2B requires {option}")
            if self.attention_implementation not in ("eager", "sdpa"):
                raise ValueError("Stage 2B attention implementation must be eager or sdpa")
            if self.resume_from_checkpoint_path is not None:
                raise ValueError(
                    "--stage2a_checkpoint_path and --resume_from_checkpoint_path "
                    "are mutually exclusive for Stage 2B"
                )
            if self.packing_enabled and (
                isinstance(self.max_packed_sequence_length, bool)
                or not isinstance(self.max_packed_sequence_length, int)
                or self.max_packed_sequence_length <= 0
            ):
                raise ValueError(
                    "max_packed_sequence_length must be a positive integer when packing is enabled"
                )
        elif self.packing_enabled:
            raise ValueError("step preference packing is only valid for Stage 2B")

    @property
    def is_stage2b(self) -> bool:
        return self.enabled and self.mode == "stage2b"

    @property
    def is_context_preference(self) -> bool:
        return self.is_stage2b and self.objective == "context_preference"

    @property
    def phase_steps(self) -> int | None:
        if not self.enabled or self.total_steps is None:
            return None
        if self.direct_from_stage1:
            return self.total_steps
        stage2a, stage2b = split_stage2_budget(
            self.total_steps, self.stage2a_ratio
        )
        return stage2b if self.mode == "stage2b" else stage2a

    @property
    def replacement_sampling_version(self) -> str:
        """Fixed planner version recorded with an evidence-replacement run."""
        return EVIDENCE_REPLACEMENT_SAMPLING_VERSION

    @classmethod
    def from_args(cls, args) -> "StepPreferenceConfig":
        """Build from the public training namespace without mutating it."""
        return cls(
            enabled=getattr(args, "step_preference_enabled", False),
            mode=getattr(args, "step_preference_mode", "stage2a"),
            objective=getattr(args, "step_preference_objective", "context_preference"),
            target_scope=getattr(args, "step_preference_target_scope", "immediate_text"),
            beta=getattr(args, "step_dpo_beta", 0.1),
            lambda_ce=getattr(args, "ce_weight", 1.0),
            lambda_dpo=getattr(args, "step_dpo_weight", 1.0),
            lambda_preference=getattr(args, "step_preference_weight", 1.0),
            preference_margin=getattr(args, "step_preference_margin", 0.0),
            prefix_mode=getattr(args, "step_preference_prefix_mode", "clean_prefix"),
            branch_mode=getattr(args, "step_preference_branch_mode", "independent"),
            block_original_image=getattr(args, "step_preference_block_original_image", False),
            visual_only_context_preference=getattr(
                args, "step_visual_only_context_preference", None
            ),
            visual_off_suppress_latent_boundaries=getattr(
                args, "step_visual_off_suppress_latent_boundaries", None
            ),
            lambda_entropy=getattr(args, "step_entropy_weight", 0.1),
            entropy_margin=getattr(args, "step_entropy_margin", 0.1),
            negative_path=getattr(args, "step_negative_path", None),
            negative_manifest=getattr(args, "step_negative_manifest", None),
            stage2a_ratio=getattr(args, "stage2a_ratio", 0.5),
            total_steps=getattr(args, "stage2_total_steps", None),
            stage2a_checkpoint_path=getattr(args, "stage2a_checkpoint_path", None),
            direct_from_stage1=getattr(args, "stage2b_direct_from_stage1", False),
            attention_implementation=getattr(args, "attention_implementation", "sdpa"),
            resume_from_checkpoint_path=getattr(
                args, "resume_from_checkpoint_path", None
            ),
            packing_enabled=getattr(args, "step_preference_packing", False),
            shared_prefix_enabled=getattr(args, "step_shared_prefix_enabled", False),
            sidecar_attention_backend=getattr(args, "step_sidecar_attention_backend", "flex"),
            sidecar_query_budget=getattr(args, "step_sidecar_query_budget", 1024),
            replacement_seed=getattr(args, "seed", 42),
            max_packed_sequence_length=getattr(
                args, "step_max_packed_sequence_length", None
            ),
            packed_admission_manifest=getattr(
                args, "step_packed_admission_manifest", None
            ),
            packed_admission_manifest_sha256=getattr(
                args, "step_packed_admission_manifest_sha256", None
            ),
            packed_admission_verification=getattr(
                args, "step_packed_admission_verification", None
            ),
            packed_admission_verification_sha256=getattr(
                args, "step_packed_admission_verification_sha256", None
            ),
        )

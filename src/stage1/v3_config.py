"""Serializable, validated identity of the opt-in Stage1-v3 objective."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Stage1V3Config:
    schema_version: int = 1
    objective: str = "text_only_gain_v1"
    text_selector: str = "kl"
    gain_alpha: float = .001
    min_h_norm_sq: float = 1e-30
    text_kl_beta: float = .01
    text_kl_warmup_ratio: float = .05
    temperature: float = 2.
    rho_min: float = .3
    rho_max: float = .5
    text_cap: float = .3
    low_mass_threshold: float = .1
    low_mass_ratio: float = .1
    image_cap: float = 0.
    gain_weighting: str = "none"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported Stage1-v3 schema")
        if self.objective != "text_only_gain_v1":
            raise ValueError("unsupported Stage1-v3 objective")
        if type(self.text_selector) is not str or self.text_selector not in ("kl", "entropy"):
            raise ValueError("Stage1-v3 text selector must be kl or entropy")
        if type(self.gain_weighting) is not str or self.gain_weighting not in ("none", "token_entropy"):
            raise ValueError("Stage1-v3 Gain weighting must be none or token_entropy")
        for name, value in asdict(self).items():
            if name in ("schema_version", "objective", "text_selector", "gain_weighting"):
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Stage1-v3 {name} must be finite numeric")
        if self.gain_alpha < 0 or self.text_kl_beta < 0:
            raise ValueError("Stage1-v3 loss/update weights must be nonnegative")
        if self.temperature <= 0 or self.min_h_norm_sq <= 0:
            raise ValueError("Stage1-v3 temperature and norm threshold must be positive")
        if not 0 <= self.rho_min <= self.rho_max <= 1:
            raise ValueError("Stage1-v3 requires 0 <= rho_min <= rho_max <= 1")
        for name in ("text_kl_warmup_ratio", "text_cap", "low_mass_threshold", "low_mass_ratio"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"Stage1-v3 {name} must be in [0,1]")
        if self.image_cap != 0:
            raise ValueError("Stage1-v3 supports text-only masking (image_cap=0)")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

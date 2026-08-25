from dataclasses import dataclass
from typing import Literal, Sequence


TuningMethod = Literal["full", "lora", "dora"]
ALLOWED_STAGE2_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def normalize_stage2_target_modules(
    value: str | Sequence[str],
) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    targets = tuple(item.strip() for item in raw if item.strip())
    if not targets:
        raise ValueError("Stage 2 PEFT target modules must not be empty")
    if len(set(targets)) != len(targets):
        raise ValueError("Stage 2 PEFT target modules must be unique")
    invalid = sorted(set(targets) - set(ALLOWED_STAGE2_TARGET_SUFFIXES))
    if invalid:
        raise ValueError(f"unsupported Stage 2 PEFT target modules: {invalid}")
    return targets


@dataclass(frozen=True)
class Stage2TuningConfig:
    method: TuningMethod
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method not in ("full", "lora", "dora"):
            raise ValueError(f"unsupported Stage 2 tuning method: {self.method}")
        if self.r <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        normalize_stage2_target_modules(self.target_modules)

    @property
    def is_peft(self) -> bool:
        return self.method in ("lora", "dora")

    @property
    def use_dora(self) -> bool:
        return self.method == "dora"

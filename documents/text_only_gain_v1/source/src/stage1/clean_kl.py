"""Optional clean KL loss from existing base logits and streamed EMA logits.

No model/head calls or checkpoint replay: this graph joins the base CE backward.
Selection continues to use the separate, detached gain-clean KL.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class CleanKLConfig:
    beta: float = 0.0
    warmup_ratio: float = 0.05

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'clean KL {name} must be finite')
        if self.beta < 0 or not 0 <= self.warmup_ratio <= 1:
            raise ValueError('clean KL requires beta >= 0 and warmup_ratio in [0, 1]')

    @property
    def enabled(self):
        return self.beta > 0

    @classmethod
    def from_args(cls, args):
        return cls(getattr(args, 'stage1_clean_kl_beta', 0.),
                   getattr(args, 'stage1_clean_kl_warmup_ratio', .05))

    def validate_runtime(self, prefix, gain):
        if self.enabled and (not prefix.enabled or not gain.alpha or gain.execution != 'packed'):
            raise ValueError('clean KL requires enabled prefix conditioning and nonzero packed gain')

    def weight(self, completed_steps, total_steps):
        if not self.enabled:
            return 0.
        if completed_steps < 0 or total_steps <= 0:
            raise ValueError('clean KL warmup requires valid Trainer step counts')
        warmup_steps = math.ceil(total_steps * self.warmup_ratio)
        return self.beta * min(1., completed_steps / warmup_steps) if warmup_steps else self.beta

    def to_dict(self):
        return dict(schema_version=1, **asdict(self),
            student='existing_base_clean_logits', reference='existing_ema_clean_logits_detached',
            queries='existing_gain_valid_gt_queries', direction='student_to_reference',
            temperature='prefix_conditioning.temperature', scale='temperature_squared',
            reduction='mean_tokens_per_valid_sample_then_mean_valid_samples',
            warmup='completed_trainer_steps_over_ceil_planned_steps_times_ratio',
            additional_model_forwards=0)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'stage1_clean_kl.json').write_text(
            json.dumps(self.to_dict(), sort_keys=True, indent=2) + '\n', encoding='utf-8')

    def validate_resume(self, directory):
        path = Path(directory) / 'stage1_clean_kl.json'
        if not path.exists():
            if not self.enabled:
                return
            raise ValueError('missing clean KL checkpoint contract')
        from .prefix_mask_config import read_small_json
        if read_small_json(path) != self.to_dict():
            raise ValueError('clean KL checkpoint contract mismatch')


class CleanKLAccumulator:
    """Consume each reference query once, retaining only the student loss graph."""

    def __init__(self, student_logits, query_keys, *, temperature):
        if (not isinstance(student_logits, torch.Tensor) or student_logits.ndim != 3
                or student_logits.shape[-1] == 0 or not student_logits.requires_grad):
            raise ValueError('clean KL requires differentiable base logits [B,T,V]')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('clean KL temperature must be positive and finite')
        self.logits, self.keys, self.temperature = student_logits, tuple(query_keys), temperature
        if len(set(self.keys)) != len(self.keys) or any(
                not (0 <= b < student_logits.shape[0] and 0 <= q < student_logits.shape[1])
                for b, q in self.keys):
            raise ValueError('invalid clean KL query coordinates')
        self.counts = Counter(b for b, _ in self.keys)
        self.consumed = 0
        self.losses, self.values = [], []

    def consume(self, keys, reference_logits):
        count = len(keys)
        if not count or tuple(keys) != self.keys[self.consumed:self.consumed + count]:
            raise ValueError('clean KL reference query order mismatch')
        if reference_logits.shape != (count, self.logits.shape[-1]):
            raise ValueError('clean KL reference vocabulary/rows mismatch')
        # The EMA callback is under no_grad. Re-enable only tensor arithmetic on
        # the existing base outputs, never the teacher or an additional LM head.
        with torch.enable_grad():
            b, q = zip(*keys)
            student = self.logits[list(b), list(q)].float() / self.temperature
            reference = reference_logits.detach().to(device=student.device, dtype=torch.float32)
            reference = reference / self.temperature
            logp, logr = student.log_softmax(-1), reference.log_softmax(-1)
            values = self.temperature ** 2 * (logp.exp() * (logp - logr)).sum(-1)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('nonfinite clean KL')
            weights = values.new_tensor([1 / (len(self.counts) * self.counts[sample]) for sample in b])
            self.losses.append((values * weights).sum())
        self.values.append(values.detach())
        self.consumed += count

    def finish(self):
        if self.consumed != len(self.keys):
            raise ValueError('incomplete clean KL reference coverage')
        loss = torch.stack(self.losses).sum() if self.losses else self.logits.reshape(-1)[:0].float().sum()
        values = torch.cat(self.values) if self.values else loss.detach().new_empty(0)
        return loss, dict(clean_kl_raw=loss.detach(),
            clean_kl_p95=torch.quantile(values, .95) if values.numel() else loss.detach(),
            clean_kl_query_count=loss.detach().new_tensor(len(self.keys)))

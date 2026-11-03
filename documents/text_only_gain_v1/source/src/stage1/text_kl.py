"""KL(stopgrad(clean) || text-masked), consuming existing vocabulary logits.

The masked graph uses gain aliases. Its derivative is extracted alongside the
positive gain derivative and carried into the ordinary optimizer backward.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class TextKLConfig:
    beta: float = 0.0
    warmup_ratio: float = 0.05

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'text KL {name} must be finite numeric')
        if self.beta < 0 or not 0 <= self.warmup_ratio <= 1:
            raise ValueError('text KL requires beta >= 0 and warmup_ratio in [0, 1]')

    @property
    def enabled(self):
        return self.beta > 0

    @classmethod
    def from_args(cls, args):
        return cls(getattr(args, 'stage1_text_kl_beta', 0.),
                   getattr(args, 'stage1_text_kl_warmup_ratio', .05))

    def validate_runtime(self, prefix, gain):
        if not self.enabled:
            return
        if not prefix.enabled or not gain.alpha or gain.execution != 'packed':
            raise ValueError('text KL requires enabled prefix conditioning and nonzero packed gain')
        if prefix.image_max_mask_ratio != 0 or prefix.text_max_mask_ratio <= 0:
            raise ValueError('text KL requires text-only conditioning (image cap 0, text cap > 0)')

    def weight(self, completed_steps, total_steps):
        if not self.enabled:
            return 0.
        if completed_steps < 0 or total_steps <= 0:
            raise ValueError('text KL warmup requires valid Trainer step counts')
        steps = math.ceil(total_steps * self.warmup_ratio)
        return self.beta * min(1., completed_steps / steps) if steps else self.beta

    def to_dict(self):
        return dict(schema_version=1, **asdict(self),
            direction='clean_to_text_masked', target='existing_base_clean_logits_detached',
            student='existing_packed_positive_R_logits', queries='existing_gain_valid_gt_queries',
            temperature='prefix_conditioning.temperature', scale='temperature_squared',
            reduction='mean_tokens_per_valid_sample_then_mean_valid_samples',
            gradient='joint_positive_and_kl_vjp_then_optimizer_backward',
            warmup='completed_trainer_steps_over_ceil_planned_steps_times_ratio',
            additional_model_forwards=0, additional_checkpoint_traversals=0)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'stage1_text_kl.json').write_text(
            json.dumps(self.to_dict(), sort_keys=True, indent=2) + '\n', encoding='utf-8')

    def validate_resume(self, directory):
        path = Path(directory) / 'stage1_text_kl.json'
        if not path.exists():
            if not self.enabled:
                return
            raise ValueError('missing text KL checkpoint contract')
        from .prefix_mask_config import read_small_json
        if read_small_json(path) != self.to_dict():
            raise ValueError('text KL checkpoint contract mismatch')


class TextKLAccumulator:
    def __init__(self, clean_logits, query_keys, *, temperature):
        if (not isinstance(clean_logits, torch.Tensor) or clean_logits.ndim != 3
                or clean_logits.shape[-1] == 0):
            raise ValueError('text KL requires existing clean logits [B,T,V]')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('text KL temperature must be positive and finite')
        self.clean = clean_logits.detach()
        self.keys = set(query_keys)
        if len(self.keys) != len(query_keys) or any(
                not (0 <= b < clean_logits.shape[0] and 0 <= q < clean_logits.shape[1])
                for b, q in self.keys):
            raise ValueError('invalid text KL query coordinates')
        self.counts = Counter(b for b, _ in self.keys)
        self.temperature = temperature
        self.seen = set()
        self.losses, self.values = [], []
        self.optimizer_loss = None
        self.metrics = None

    def consume(self, keys, masked_logits):
        keys = tuple(keys)
        if not keys or len(set(keys)) != len(keys) or any(
                key not in self.keys or key in self.seen for key in keys):
            raise ValueError('text KL query is duplicate or unowned')
        if masked_logits.shape != (len(keys), self.clean.shape[-1]):
            raise ValueError('text KL vocabulary/rows mismatch')
        b, q = zip(*keys)
        clean = self.clean[list(b), list(q)].to(device=masked_logits.device, dtype=torch.float32)
        log_clean = (clean / self.temperature).log_softmax(-1)
        log_masked = (masked_logits.float() / self.temperature).log_softmax(-1)
        values = self.temperature ** 2 * (log_clean.exp() * (log_clean - log_masked)).sum(-1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('nonfinite text KL')
        weights = values.new_tensor([1 / (len(self.counts) * self.counts[sample]) for sample in b])
        self.losses.append((values * weights).sum())
        self.values.append(values.detach())
        self.seen.update(keys)

    def finish(self):
        if self.seen != self.keys:
            raise ValueError('incomplete text KL query coverage')
        loss = torch.stack(self.losses).sum() if self.losses else self.clean.new_zeros((), dtype=torch.float32)
        values = torch.cat(self.values) if self.values else loss.detach().new_empty(0)
        self.metrics = dict(text_kl_raw=loss.detach(),
            text_kl_p95=torch.quantile(values, .95) if values.numel() else loss.detach(),
            text_kl_query_count=loss.detach().new_tensor(len(self.keys)))
        return loss, self.metrics


def joint_positive_kl_gradients(positive, kl_loss, parameters):
    """Two cotangents in one engine traversal, including checkpoint recompute.

Modern vmap's 'same' randomness permits existing checkpointed adapter dropout.
Legacy autograd.grad(is_grads_batched=True) forbids that random recomputation.
There is deliberately no serial fallback, which would add a third traversal.
"""
    def vjp(seed):
        gradients = torch.autograd.grad((positive, kl_loss), parameters,
            grad_outputs=tuple(seed.unbind()), allow_unused=True, retain_graph=True)
        return {i: gradient for i, gradient in enumerate(gradients) if gradient is not None}

    basis = torch.eye(2, device=positive.device, dtype=positive.dtype)
    gradients = torch.vmap(vjp, randomness='same')(basis)
    return tuple(gradients[i][0] if i in gradients else None for i in range(len(parameters))), tuple(
        gradients[i][1].detach().clone() if i in gradients else None for i in range(len(parameters)))


class _OptimizerGradientLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gradients, *parameters):
        ctx.gradients = gradients
        return value.detach().clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, scale):
        return (None, None, *(None if g is None else g * scale.to(g) for g in ctx.gradients))


def optimizer_gradient_loss(value, gradients, parameters):
    """Apply ordinary loss scaling, accumulation, clipping and optimizer hooks."""
    if len(gradients) != len(parameters) or any(
            g is not None and (g.shape != p.shape or not bool(torch.isfinite(g).all()))
            for g, p in zip(gradients, parameters)):
        raise ValueError('invalid text KL optimizer gradients')
    # An edge to a parameter whose backward returns None would be counted by
    # DeepSpeed's expected-hook query but never fire a post-accumulate hook.
    active = [(g, p) for g, p in zip(gradients, parameters) if g is not None]
    if not active:
        return value.detach()
    gradients, parameters = zip(*active)
    return _OptimizerGradientLoss.apply(value.detach(), gradients, *parameters)

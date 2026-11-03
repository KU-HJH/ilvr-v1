"""Same-target clean, independent reference, and local masked execution."""
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn

from transformers.models.qwen2_5_vl.stage2_objective import (
    LocalLatentIntervention, ObjectiveForwardRequest, ObjectiveReplayStats,
)
from .config import ObjectiveConfig
from .losses import entropy_transition, target_logps, text_entropies, vision_gain
from .reference import FrozenReference
from .reduction import MetricTotal
from .vocabulary import TextVocabulary
from .windows import WindowPlan, score_mask


@dataclass(frozen=True)
class DropoutPolicy:
    version: str
    original_values: tuple[tuple[str, float], ...]
    effective_value: float = 0.0


@dataclass(frozen=True)
class ObjectiveRuntime:
    config: ObjectiveConfig
    vocabulary: TextVocabulary
    reference: FrozenReference | None
    dropout_policy: DropoutPolicy
    visual_ablation: str = 'latent_body'

    def __post_init__(self):
        if self.visual_ablation not in {'latent_body', 'full_block_attention_mask'}:
            raise ValueError('visual_ablation must be latent_body or full_block_attention_mask')
        if self.visual_ablation != 'latent_body' and self.config.lambda_vis <= 0:
            raise ValueError('full-block visual ablation requires positive lambda_vis')


@dataclass(frozen=True)
class BranchReplayStats:
    branch: str
    stats: ObjectiveReplayStats


@dataclass(frozen=True)
class ObjectiveResult:
    ce_loss: Tensor
    weighted_block_sum: Tensor
    block_count: int
    metrics: dict[str, MetricTotal]
    outputs: Any
    replay_stats: tuple[BranchReplayStats, ...] = ()


DROPOUT_POLICY_VERSION = 'zero_dropout_adapter_identity_v2'


def _adapter_dropouts(model):
    from peft.tuners.lora.layer import LoraLayer
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for adapter, dropout in module.lora_dropout.items():
                if not isinstance(dropout, (nn.Dropout, nn.Identity)):
                    raise ValueError(f'unsupported adapter dropout {name}.{adapter}')
                path = f'{name}.lora_dropout.{adapter}'.lstrip('.')
                yield path, module.lora_dropout, adapter, dropout


def validate_objective_dropout_policy(model, policy):
    if policy.version != DROPOUT_POLICY_VERSION or policy.effective_value != 0:
        raise ValueError('objective effective dropout policy mismatch')
    current = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            current[name + '.p'] = module.p
        if hasattr(module, 'attention_dropout'):
            current[name + '.attention_dropout'] = module.attention_dropout
    for name, _, _, dropout in _adapter_dropouts(model):
        if not isinstance(dropout, nn.Identity):
            raise ValueError('objective adapter dropout must use Identity arithmetic')
        current[name + '.p'] = 0.0
    if set(current) != {name for name, _ in policy.original_values} or any(current.values()):
        raise ValueError('objective effective dropout policy mismatch')


def apply_objective_dropout_policy(student: nn.Module) -> DropoutPolicy:
    """Apply once after faithful restoration, retaining artifact configuration.

    Validate all settings before mutating any module. Functional dropout and
    stochastic-depth controls other than decoder attention are unsupported.
    """
    adapters = list(_adapter_dropouts(student))
    changes = [(name + '.p', dropout, None, 0.0)
               for name, _, _, dropout in adapters if isinstance(dropout, nn.Identity)]
    for name, module in student.named_modules():
        if isinstance(module, nn.Dropout):
            changes.append((name + '.p', module, 'p', module.p))
        if hasattr(module, 'attention_dropout'):
            changes.append((name + '.attention_dropout', module, 'attention_dropout', module.attention_dropout))
        if isinstance(module, (nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)):
            if module.p != 0:
                raise ValueError(f'unsupported stochastic dropout module {name}')
        for attr in ('dropout', 'dropout_p', 'dropout_prob', 'hidden_dropout', 'hidden_dropout_prob', 'activation_dropout',
                     'resid_pdrop', 'embd_pdrop', 'attn_pdrop', 'drop_path',
                     'drop_path_rate', 'layerdrop'):
            value = vars(module).get(attr)
            if isinstance(value, Real) and value != 0:
                raise ValueError(f'unsupported stochastic setting {name}.{attr}')
    for name, _, _, value in changes:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f'invalid dropout probability {name}: {value}')
    originals = tuple(sorted((name, float(value)) for name, _, _, value in changes))
    for _, module, attr, _ in changes:
        if attr is not None:
            setattr(module, attr, 0.0)
    # PEFT DoRA branches on Identity versus Dropout even when p=0. Keep
    # train/eval arithmetic equal without changing the student's train mode.
    for _, container, adapter, dropout in adapters:
        container[adapter] = nn.Identity().train(dropout.training)
    return DropoutPolicy(DROPOUT_POLICY_VERSION, originals)


def selected_logit_rows(outputs: Any, mask: Tensor) -> dict[tuple[int, int], Tensor]:
    """Restore original predictor coordinates while retaining the score graph."""
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError('score mask must be boolean [batch, sequence]')
    logits = outputs.stage2_scored_logits
    if logits is None or logits.ndim != 3 or logits.shape[0] != mask.shape[0]:
        raise ValueError('selected logits must have shape [batch, selected, vocabulary]')
    rows = {}
    for batch_index in range(mask.shape[0]):
        positions = mask[batch_index].nonzero(as_tuple=False).flatten().tolist()
        if len(positions) != logits.shape[1]:
            raise ValueError('selected logits count does not match score mask')
        for index, position in enumerate(positions):
            rows[batch_index, position] = logits[batch_index, index]
    return rows


def replay_metric_totals(result: ObjectiveResult) -> dict[str, MetricTotal]:
    """Sample live counters AFTER backward; replace these keys, never add twice.

    These keys are deliberately absent from ``result.metrics``. Task 7 can use
    ``{**result.metrics, **replay_metric_totals(result)}`` at its reporting boundary.
    """
    totals = {}
    for branch in ('f', 'reference', 'm'):
        records = [record.stats for record in result.replay_stats if record.branch == branch]
        if not records:
            continue
        for attr in ('decoder_forward_calls', 'prefix_checkpoint_replay_calls', 'native_checkpoint_recompute_calls'):
            totals[f'{attr}_{branch}'] = MetricTotal(
                result.ce_loss.detach().new_tensor(sum(getattr(record, attr) for record in records), dtype=torch.float32), 1)
    return totals


_METADATA_KEYS = frozenset({'sample_ids', 'reasoning_step_spans', 'step_preference_items',
                            'stage2_objective_plan'})


def masked_prefix_inputs(model_inputs: dict, mask: Tensor, *, end_marker: int) -> dict:
    """Keep the causal prefix through the last scored M predictor, inclusive.

    Only M uses this helper. Targets remain in the immutable WindowPlan; their
    input tokens, later text, and later latent rollouts cannot affect an earlier
    predictor. Slice views without shifting positions or sharing branch states.
    """
    ids = model_inputs['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape or mask.dtype != torch.bool:
        raise ValueError('masked prefix requires input_ids and boolean mask of shape [1, sequence]')
    if model_inputs.get('labels') is not None:
        raise ValueError('masked prefix is scoring-only; clean CE must keep the full trajectory')
    positions = mask[0].nonzero(as_tuple=False).flatten()
    if not positions.numel() or int(positions[0]) < end_marker:
        raise ValueError('masked prefix must score at or after its latent end marker')
    length = int(positions[-1]) + 1
    result = dict(model_inputs, input_ids=ids[:, :length], stage2_score_token_mask=mask[:, :length])
    padding = model_inputs.get('attention_mask')
    if padding is not None:
        if padding.shape != ids.shape:
            raise ValueError('masked prefix requires the original two-dimensional padding mask')
        result['attention_mask'] = padding[:, :length]
    positions = model_inputs.get('position_ids')
    if positions is not None:
        if positions.shape not in (ids.shape, (3, *ids.shape)):
            raise ValueError('masked prefix requires original 2D or 3D position IDs')
        result['position_ids'] = positions[..., :length]
    return result


def execute_objective(*, forward_model: Any, student_core: nn.Module,
                      model_inputs: dict, plan: WindowPlan,
                      runtime: ObjectiveRuntime) -> ObjectiveResult:
    config = runtime.config
    ids = model_inputs['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('objective execution requires microbatch size 1')
    if config.lambda_dyn and plan.blocks and runtime.reference is None:
        raise ValueError('entropy transition requires an independent frozen reference')
    base = {key: value for key, value in model_inputs.items() if key not in _METADATA_KEYS}
    # Explicitly choose this path on every call, including the CE-only mode.
    base.update(stage2_loss_only=True, return_dict=True)
    base.pop('stage2_score_token_mask', None)
    mask = score_mask(plan, ids, include_pre=bool(config.lambda_dyn),
                      include_post=bool(config.lambda_dyn or config.lambda_vis))
    f_kwargs = dict(base, stage2_objective_request=ObjectiveForwardRequest())
    active = bool(plan.blocks and (config.lambda_dyn or config.lambda_vis))
    if active:
        f_kwargs['stage2_score_token_mask'] = mask
    context = ', '.join(f'{b.segment.sample_id!r}/step {b.segment.step_id}' for b in plan.blocks)
    if not context:
        context = repr(model_inputs.get('sample_ids', '<empty plan>'))
    outputs = forward_model(**f_kwargs)
    ce = outputs.loss
    if ce is None or ce.ndim != 0 or not torch.isfinite(ce):
        raise ValueError(f'sample {context} branch F: nonfinite or missing scalar CE')
    weighted = ce * 0.0
    metrics: dict[str, MetricTotal] = {}
    records = []

    def metric(name, value, count=None):
        value = torch.as_tensor(value, device=ce.device).detach().float()
        metrics[name] = MetricTotal(value.sum(), value.numel() if count is None else count)

    def record(branch, out):
        stats = getattr(out, 'stage2_objective_stats', None)
        if not isinstance(stats, ObjectiveReplayStats):
            raise ValueError(f'sample {context} branch {branch}: missing live objective replay witness')
        records.append(BranchReplayStats(branch, stats))

    record('f', outputs)
    metric('loss_ce', ce)
    metric('valid_blocks', len(plan.blocks))
    metric('invocations_f', 1)
    metric('invocations_reference', int(bool(config.lambda_dyn and plan.blocks)))
    metric('invocations_m', len(plan.blocks) if config.lambda_vis else 0)
    for reason in ('empty_pre', 'empty_post', 'overlap_empty_pre', 'overlap_empty_post'):
        metric('excluded_' + reason, sum(item.reason == reason for item in plan.excluded))
    if not active:
        return ObjectiveResult(ce, weighted, len(plan.blocks), metrics, outputs, tuple(records))
    rows = selected_logit_rows(outputs, mask)

    def stack(rows, block, positions):
        return torch.stack([rows[block.segment.batch_index, position] for position in positions])

    scoring_base = {key: value for key, value in base.items() if key != 'labels'}
    scoring_base['retain_stage2_latent_grads'] = False
    if config.lambda_dyn:
        ref_kwargs = dict(scoring_base, stage2_objective_request=ObjectiveForwardRequest(),
                          stage2_score_token_mask=mask)
        with torch.no_grad():
            ref_outputs = runtime.reference.model(**ref_kwargs)
        record('reference', ref_outputs)
        ref_rows = selected_logit_rows(ref_outputs, mask)
        text_ids = torch.tensor(runtime.vocabulary.text_ids, device=ids.device)
        entropies = []
        for source in (rows, ref_rows):
            for attr in ('pre_positions', 'post_positions'):
                means = []
                for block in plan.blocks:
                    try:
                        means.append(text_entropies(stack(source, block, getattr(block, attr)), text_ids).mean())
                    except (ValueError, RuntimeError) as error:
                        branch = 'F' if source is rows else 'reference'
                        raise ValueError(f'sample {block.segment.sample_id!r} step {block.segment.step_id} branch {branch}: {error}') from error
                entropies.append(torch.stack(means))
        pre, post, ref_pre, ref_post = entropies
        try:
            dyn = entropy_transition(pre, post, ref_pre, ref_post,
                                     eps=config.entropy_eps, gamma_h=config.gamma_h)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f'sample {context} branch dyn: {error}') from error
        weighted = weighted + config.lambda_dyn * dyn.losses.sum()
        for name, value in zip(('entropy_pre', 'entropy_post', 'entropy_ref_pre', 'entropy_ref_post'), entropies):
            metric(name, value)
        for name, value in (('loss_dyn', dyn.losses), ('ratio_pre', dyn.r_pre),
                            ('ratio_post', dyn.r_post), ('entropy_delta', dyn.delta),
                            ('ratio_pre_gt_one', dyn.r_pre > 1), ('ratio_post_lt_one', dyn.r_post < 1)):
            metric(name, value)
        metric('low_reference_entropy_pre', (ref_pre < config.entropy_eps).sum())
        metric('low_reference_entropy_post', (ref_post < config.entropy_eps).sum())
    if config.lambda_vis:
        clean_scores, masked_scores = [], []
        # Dynamic RoPE can depend on the maximum sequence position and mutable
        # frequency state. Preserve the original path unless rotary is fixed.
        fixed_rope = all(getattr(module, 'rope_type', 'default') in {'default', 'mrope'}
                         for module in student_core.modules())
        image_id = getattr(student_core.config, 'image_token_id', None)
        for block in plan.blocks:
            m_mask = score_mask(WindowPlan((block,), ()), ids, include_pre=False, include_post=True)
            intervention = LocalLatentIntervention(
                block.segment.start_marker,
                block.segment.end_marker,
                block_markers=runtime.visual_ablation == 'full_block_attention_mask')
            cutoff = max(block.post_positions) + 1
            later_image = image_id is not None and bool((ids[:, cutoff:] == image_id).any())
            if fixed_rope and not later_image:
                m_kwargs = masked_prefix_inputs(scoring_base, m_mask, end_marker=block.segment.end_marker)
            else:
                m_kwargs = dict(scoring_base, stage2_score_token_mask=m_mask)
            m_kwargs['stage2_objective_request'] = ObjectiveForwardRequest(intervention)
            with torch.no_grad():
                m_outputs = student_core(**m_kwargs)
            record('m', m_outputs)
            m_rows = selected_logit_rows(m_outputs, m_mask)
            targets = torch.tensor(block.post_ids, device=ids.device)
            for source, destination, branch in ((rows, clean_scores, 'F'), (m_rows, masked_scores, 'M')):
                try:
                    destination.append(target_logps(stack(source, block, block.post_positions), targets).mean())
                except (ValueError, RuntimeError) as error:
                    raise ValueError(f'sample {block.segment.sample_id!r} step {block.segment.step_id} branch {branch}: {error}') from error
        clean, masked = torch.stack(clean_scores), torch.stack(masked_scores).detach()
        try:
            vis = vision_gain(clean, masked, margin=config.vision_margin)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f'sample {context} branch vis: {error}') from error
        weighted = weighted + config.lambda_vis * vis.losses.sum()
        for name, value in (('loss_vis', vis.losses), ('score_clean', clean),
                            ('score_masked', masked), ('vision_gain', vis.gain),
                            ('vision_gain_gt_zero', vis.gain > 0)):
            metric(name, value)
    if not torch.isfinite(weighted):
        raise ValueError(f'sample {context} branch auxiliary: nonfinite weighted loss')
    return ObjectiveResult(ce, weighted, len(plan.blocks), metrics, outputs, tuple(records))

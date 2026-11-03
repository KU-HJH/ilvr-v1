"""Optional packed score executor; the independent executor stays unchanged."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
import torch
from torch import nn
from transformers.models.qwen2_5_vl.stage1_vision_gain_packed import (
    Stage1PackedGainBlock, Stage1PackedGainRequest, bind_dropout_forwards,
)
from .prefix_mask_config import PrefixConditioningConfig
from .clean_kl import CleanKLAccumulator
from .text_kl import TextKLAccumulator
from .prefix_mask_metadata import HelperSnapshot, build_trace_request
from .prefix_mask_accounting import DecoderCallLedger
from transformers.models.qwen2_5_vl.stage1_prefix_conditioning import ConditioningRequest
from .vision_gain import score_stage1_gain_block
from .vision_gain_execution import Stage1GainScores, _restore_rng, _score_model_state


@dataclass
class PrefixConditioningContext:
    config: PrefixConditioningConfig
    snapshot: HelperSnapshot
    teacher: nn.Module
    ledger: DecoderCallLedger
    report: dict[str, object]
    clean_kl: CleanKLAccumulator | None = None
    text_kl: TextKLAccumulator | None = None
    text_kl_weight: float = 0.

    def validate(self, model_inputs, plan):
        if not self.config.enabled:
            raise ValueError('conditioning context requires enabled config')
        snapshot = self.snapshot
        if (snapshot.parameter_version != self.report.get('parameter_version')
                or snapshot.teacher_version != self.report.get('teacher_version')):
            raise ValueError('stale conditioning snapshot version')
        if (tuple((b.batch_index, b.block_index) for b in plan.blocks)
                != tuple(source.key for source in snapshot.sources)):
            raise ValueError('snapshot block ownership differs from gain plan')
        if any(tuple((block.batch_index, p - 1) for p in block.target_positions) != source.gt_queries
               for block, source in zip(plan.blocks, snapshot.sources)):
            raise ValueError('snapshot GT ownership differs from gain plan')
        if (not torch.equal(snapshot.latent_values, model_inputs['latent_hidden_states'])
                or not torch.equal(snapshot.latent_mask, model_inputs['image_out_mask'])):
            raise ValueError('conditioning snapshot differs from injected helper targets')

    def request(self, model_inputs):
        # Bind once to this loss call. No targets or decoder outputs are rebuilt.
        from .prefix_mask_reference import score_clean_reference_once
        from .prefix_mask_plan import build_prefix_mask_plan
        snapshot = self.snapshot
        self.ledger.report = self.report
        self.report.update(actual_selected_positions=[source.selected_latent_positions for source in snapshot.sources],
            latent_prediction_query_ids=[(source.key[0], p - 1) for source in snapshot.sources for p in source.selected_latent_positions],
            source_groups=[asdict(source) for source in snapshot.sources],
            reference_query_rows=len(snapshot.gt_queries))
        called = False
        def build(hidden, head, positions, rows):
            nonlocal called
            if called:
                raise RuntimeError('conditioning selector called more than once')
            called = True
            inputs = dict(model_inputs, position_ids=positions)
            with self.ledger.role('reference'):
                kl = score_clean_reference_once(self.teacher, inputs, snapshot, hidden, head,
                    config=self.config,
                    **({'consume_reference': self.clean_kl.consume} if self.clean_kl is not None else {}))
            with self.ledger.time_region('selector', device=kl.device):
                plan, diagnostics = build_prefix_mask_plan(snapshot, rows, kl, config=self.config)
            if (plan.snapshot_id, plan.parameter_version, plan.teacher_version) != (
                    snapshot.snapshot_id, snapshot.parameter_version, snapshot.teacher_version):
                raise ValueError('selector returned stale mask plan')
            if tuple(row.query for row in plan.rows) != snapshot.gt_queries:
                raise ValueError('selector returned incomplete mask plan')
            from .prefix_mask_accounting import canonical_digest
            self.report.update(diagnostics)
            selected = [asdict(row) for row in plan.rows]
            self.report.update(plan_hash=canonical_digest(asdict(plan)), selected_source_ids=selected,
                trace_buffer_bytes=sum(row.numel() * row.element_size() for row in rows.values))
            per_sample = {}
            for row in diagnostics['queries']:
                per_sample.setdefault(row['query'][0], []).append(row['kl'])
            self.report['flat_KL_fraction'] = (sum(len(values) for values in per_sample.values()
                if max(values) - min(values) <= self.config.epsilon) / len(snapshot.gt_queries)
                if snapshot.gt_queries else 0.)
            return plan
        return ConditioningRequest(build_trace_request(snapshot), build,
            self.config.max_packed_tokens, self.config.max_mask_bytes, self.ledger.role,
            self.config.trace_chunk_rows, self.config.logit_chunk_rows,
            consume_positive_logits=self.text_kl.consume if self.text_kl is not None else None)


def _checkpoint_packed_layer(function, *args, ledger=None, checkpoint_kind='decoder',
                             layer_indices=None, **kwargs):
    """Bind aliases AND the token-correlated dropout phase during recomputation."""
    from torch.utils.checkpoint import checkpoint
    layer = getattr(function, '__self__', None)
    if not isinstance(layer, nn.Module):
        raise ValueError('packed gain checkpoint requires a bound layer')
    parameters, buffers = dict(layer.named_parameters()), dict(layer.named_buffers())
    dropouts = [(m, m.forward) for m in layer.modules() if isinstance(m, nn.Dropout)]
    role = ledger.current_role if ledger is not None else None
    calls = 0
    def replay(*inputs):
        nonlocal calls
        recompute = calls > 0 and ledger is not None
        if recompute:
            ledger.record_recompute(role, layer_indices[id(layer)], kind=checkpoint_kind)
        calls += 1
        device = next((v.device for v in inputs if isinstance(v, torch.Tensor)), None)
        timing = ledger.time_region(f'recompute/{checkpoint_kind}/{role}', device=device) if recompute else nullcontext()
        with timing, bind_dropout_forwards(dropouts):
            return torch.func.functional_call(layer, (parameters, buffers), inputs, kwargs, tie_weights=True)
    return checkpoint(replay, *args, use_reentrant=False, preserve_rng_state=True)


def execute_packed_stage1_gain(model, model_inputs, plan, clean_outputs,
                               rng_before_clean, rng_after_clean, *, accumulator=None,
                               branch_mode='independent',
                               conditioning_context: PrefixConditioningContext | None = None):
    if conditioning_context is not None:
        conditioning_context.validate(model_inputs, plan)
    if not plan.blocks:
        zero = model_inputs['input_ids'].new_zeros((), dtype=torch.float32)
        return Stage1GainScores(zero, zero, 0, ())
    while isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        model = model.module
    if type(model).__name__ == 'DeepSpeedEngine':
        model = model.module
    aliases = {name: p.detach().requires_grad_(True) for name, p in model.named_parameters() if p.requires_grad}
    if not aliases:
        raise ValueError('vision gain requires trainable student parameters')
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    inputs = dict(model_inputs)
    inputs.pop('labels', None)
    inputs.pop('past_key_values', None)
    inputs.update(use_cache=False, output_hidden_states=False, return_dict=True)
    inputs['latent_hidden_states'] = inputs['latent_hidden_states'].detach()
    inputs['stage1_vision_gain_packed_request'] = Stage1PackedGainRequest(tuple(
        Stage1PackedGainBlock(b.batch_index, b.start_marker, b.end_marker, b.target_positions, b.target_ids)
        for b in plan.blocks), branch_mode=branch_mode,
        conditioning=conditioning_context.request(model_inputs) if conditioning_context is not None else None)
    try:
        _restore_rng(rng_before_clean)
        with _score_model_state(model, checkpoint_scores=accumulator is not None):
            if accumulator is not None:
                for module in model.modules():
                    if getattr(module, 'gradient_checkpointing', False):
                        layers = getattr(module, 'layers', getattr(module, 'blocks', ()))
                        module._gradient_checkpointing_func = partial(_checkpoint_packed_layer,
                            ledger=conditioning_context.ledger if conditioning_context is not None else None,
                            checkpoint_kind='decoder' if hasattr(module, 'layers') else 'vision',
                            layer_indices={id(layer): i for i, layer in enumerate(layers)})
            scope = conditioning_context.ledger.role('gain_clean') if conditioning_context is not None else nullcontext()
            with scope:
                outputs = torch.func.functional_call(model, (aliases, buffers), (), inputs, tie_weights=True)
            if clean_outputs is not None and conditioning_context is None:
                for b in plan.blocks:
                    positions = [i - 1 for i in b.target_positions]
                    expected = clean_outputs.logits[b.batch_index, positions].detach().float()
                    observed = outputs.logits[b.batch_index, positions].detach().float()
                    if not torch.allclose(expected, observed, atol=2e-5, rtol=2e-4):
                        raise RuntimeError('vision gain clean replay differs from Stage 1 forward')
            if conditioning_context is None:
                clean = torch.stack([score_stage1_gain_block(outputs.logits, b) for b in plan.blocks]).sum()
            else:
                if outputs.logits is not None or outputs.stage1_gain_positive_scores is None:
                    raise RuntimeError('conditioned replay must return selected scores without full clean logits')
                clean = outputs.stage1_gain_positive_scores.sum()
            masked = outputs.stage1_gain_masked_scores.sum()
            if conditioning_context is not None:
                conditioning_context.report.update(
                    S_R=outputs.stage1_gain_positive_scores.detach().float().tolist(),
                    S_RV=outputs.stage1_gain_masked_scores.detach().float().tolist(),
                    local_K=len(plan.blocks), branch_mode=branch_mode,
                    U_i=[list(b.target_positions) for b in plan.blocks],
                    e_i=[b.end_marker for b in plan.blocks],
                    m_i=[max(b.target_positions) for b in plan.blocks])
            scores = Stage1GainScores(clean, masked, len(plan.blocks), tuple(aliases.values()))
            if accumulator is not None:
                # Shared prefix requires retaining the clean graph until the
                # masked derivative has flowed through its prefix K/V tensors.
                text_kl = conditioning_context.text_kl if conditioning_context is not None else None
                if text_kl is not None:
                    kl_loss, _ = text_kl.finish()
                    if conditioning_context.text_kl_weight > 0:
                        text_kl.optimizer_loss = accumulator.accumulate(scores, optimizer_loss=kl_loss)
                    else:
                        accumulator.accumulate(scores)
                        text_kl.optimizer_loss = kl_loss.detach()
                    # The alias graph was consumed. Keep only its value and
                    # transported gradients for ordinary optimizer backward.
                    text_kl.losses.clear()
                else:
                    accumulator.accumulate(scores)
                scores = Stage1GainScores(clean.detach(), masked.detach(), len(plan.blocks), tuple(aliases.values()))
            return scores
    finally:
        _restore_rng(rng_after_clean)

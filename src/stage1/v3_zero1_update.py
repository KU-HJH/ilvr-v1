"""Pinned ZeRO-1 boundary adapter for optimizer-integrated Stage1-v3 Gain."""
from dataclasses import dataclass
from importlib.metadata import version
import hashlib
import json
import math

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .v3_config import Stage1V3Config
from .v3_adamw_update import adamw_base_candidate, clip_base_gradient, project_adamw_increment


@dataclass(frozen=True)
class OwnedFragment:
    parameter_index: int
    group_index: int
    lp_start: int
    hp_start: int
    size: int
    gradient_index: int


@dataclass(frozen=True)
class OwnedGroup:
    master: Tensor
    valid_mask: Tensor
    fragments: tuple[OwnedFragment, ...]


def _require_supported_gpu_optimizer(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        if group['fused'] is not True or any(p.device.type != 'cuda' for p in group['params']):
            raise ValueError('optimizer Gain requires CUDA fused torch AdamW')


def _finite_master_and_state(master, state, *, require_state=False):
    if master.dtype != torch.float32 or not torch.isfinite(master).all():
        raise ValueError('invalid FP32 master')
    if not torch.isfinite(master.to(torch.bfloat16)).all():
        raise ValueError('master cast is nonfinite in BF16')
    if not state and not require_state:
        return
    if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
        raise ValueError('unsupported native AdamW state')
    for name, value in state.items():
        if not isinstance(value, Tensor) or not torch.isfinite(value).all():
            raise ValueError('nonfinite native AdamW state')
        if name != 'step' and (value.dtype != torch.float32 or value.shape != master.shape or value.device != master.device):
            raise ValueError('native moment shape/dtype/device mismatch')
    if state['step'].numel() != 1 or state['step'].item() < 0 or (state['exp_avg_sq'] < 0).any():
        raise ValueError('invalid native AdamW counter/second moment')


class Zero1AdamWGainAdapter:
    def __init__(self, engine: object, parameters: tuple[nn.Parameter, ...], config: Stage1V3Config):
        self.engine, self.parameters, self.config = engine, parameters, config
        self.zero = engine.optimizer
        self.process_group = getattr(self.zero, 'dp_process_group', None)
        self.device = parameters[0].device
        self._hooks = []
        self._active = False
        self._base = ()
        self._protection = ()
        self._snapshot = None
        self._original_groups = None

    def _sum(self, tensor):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group)
        return tensor

    def _agree(self, label, operation):
        error, result = None, None
        try:
            result = operation()
        except (ValueError, RuntimeError, TypeError, KeyError, AttributeError, IndexError) as exc:
            error = exc
        invalid = self._sum(torch.tensor(int(error is not None), dtype=torch.int32, device=self.device))
        if invalid.item():
            raise RuntimeError(f'optimizer Gain {label} failed on a rank' + (f': {error}' if error else '')) from error
        return result

    def _validate(self):
        z = self.zero
        if (version('deepspeed') != '0.18.7' or version('torch').split('+')[0] != '2.7.1'
                or self.engine.zero_optimization_stage() != 1 or z.partition_gradients is not False
                or z.cpu_offload is not False or z.loss_scale != 1.
                or getattr(z, 'has_moe_layers', False) or getattr(z, 'round_robin_gradients', False)
                or getattr(z, 'model_parallel_world_size', 1) != 1
                or getattr(z, 'torch_autocast_gradscaler', None) is not None or getattr(z, 'zenflow', False)
                or not callable(getattr(z, 'get_grad_norm_direct', None))):
            raise ValueError('unsupported torch/ZeRO-1 profile')
        self.native = z.optimizer
        if type(self.native) is not torch.optim.AdamW:
            raise ValueError('optimizer Gain requires native torch AdamW')
        if self.config.gain_update_mode != 'adamw_projected':
            raise ValueError('optimizer Gain adapter requires adamw_projected')
        _require_supported_gpu_optimizer(self.native)
        native_groups = self.native.param_groups
        masters = z.single_partition_of_fp32_groups
        if not len(native_groups) == len(masters) == len(z.bit16_groups) == len(z.real_dp_process_group):
            raise ValueError('ZeRO group counts disagree')
        if any(g is not self.process_group for g in z.real_dp_process_group):
            raise ValueError('optimizer Gain requires one common DP group')
        if len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError('duplicate trainable parameter')
        actual = [p for group in z.bit16_groups for p in group]
        if len(actual) != len(self.parameters) or {id(p) for p in actual} != {id(p) for p in self.parameters}:
            raise ValueError('optimizer parameter ownership mismatch')
        rank, world = dist.get_rank(self.process_group), dist.get_world_size(self.process_group)
        indices = {id(p): i for i, p in enumerate(self.parameters)}
        owned, lrs, signature = [], [], []
        for gi, (group, master, lp_params) in enumerate(zip(native_groups, masters, z.bit16_groups)):
            if len(group['params']) != 1 or group['params'][0] is not master or master.ndim != 1:
                raise ValueError('expected one flat owned master per group')
            if any(group[k] is not False for k in ('amsgrad', 'maximize', 'capturable', 'differentiable')) or group['foreach'] is not None:
                raise ValueError('unsupported native AdamW flags')
            if group.get('decoupled_weight_decay') is not True:
                raise ValueError('optimizer Gain requires decoupled AdamW decay')
            for key in ('lr', 'eps', 'weight_decay'):
                if type(group[key]) not in (float, int) or not math.isfinite(group[key]) or group[key] < 0:
                    raise ValueError('invalid optimizer group setting: ' + key)
            if any(type(b) not in (float, int) or not math.isfinite(b) or not 0 <= b < 1 for b in group['betas']):
                raise ValueError('invalid AdamW betas')
            _finite_master_and_state(master, self.native.state.get(master, {}))
            size = master.numel()
            if size != z.partition_size[gi] or z.bit16_groups_flat[gi].numel() != size * world:
                raise ValueError('partition size mismatch')
            total = sum(p.numel() for p in lp_params)
            valid = max(0, min(size, total-rank*size))
            mask = torch.arange(size, device=master.device) < valid
            fragments, offset = [], 0
            grads = z.averaged_gradients[gi]
            if not isinstance(grads, list):
                raise ValueError('owned accumulated gradients unavailable')
            for p in lp_params:
                if not p.requires_grad or p.dtype != torch.bfloat16 or p.device != master.device:
                    raise ValueError('invalid BF16 trainable parameter')
                start, end = max(offset, rank*size), min(offset+p.numel(), (rank+1)*size)
                mapping = getattr(p, '_hp_mapping', None)
                if start < end:
                    fi = len(fragments)
                    lp, hp = mapping.lp_fragment_address, mapping.hp_fragment_address
                    if (mapping.use_offload or mapping.param_group_index != gi
                            or mapping.gradient_dict is not z.averaged_gradients
                            or p._index_in_param_group != fi or lp.start != start-offset
                            or hp.start != start-rank*size or lp.numel != end-start or hp.numel != end-start
                            or mapping.hp_fragment.data_ptr() != master[hp.start:].data_ptr()
                            or grads[fi].numel() != end-start):
                        raise ValueError('fragment mapping gap/overlap/gradient identity mismatch')
                    fragments.append(OwnedFragment(indices[id(p)], gi, lp.start, hp.start, lp.numel, fi))
                elif mapping is not None:
                    raise ValueError('non-owned parameter has a local mapping')
                offset += p.numel()
            if len(grads) != len(fragments) + int(valid < size) or sum(g.numel() for g in grads) != size:
                raise ValueError('gradient fragments do not cover the partition')
            for gradient in grads:
                if gradient.device != master.device or gradient.dtype not in (torch.float32, torch.bfloat16) or not torch.isfinite(gradient).all():
                    raise ValueError('invalid owned gradient')
            if valid < size and grads[-1].count_nonzero():
                raise ValueError('nonzero gradient padding')
            owned.append(OwnedGroup(master, mask, tuple(fragments)))
            lrs.append(float(group['lr']))
            signature.append((tuple(p.numel() for p in lp_params), size,
                              tuple(group[k] for k in ('lr', 'betas', 'eps', 'weight_decay'))))
        if not math.isfinite(z.clip_grad) or z.clip_grad < 0:
            raise ValueError('invalid native clipping setting')
        self.groups, self.lrs = tuple(owned), tuple(lrs)
        self._signature = signature
        self._original_groups = native_groups

    def _owned_inputs(self, gain, protection):
        if len(gain) != len(self.parameters) or len(protection) != len(self.parameters):
            raise ValueError('auxiliary parameter count mismatch')
        for p, g, h in zip(self.parameters, gain, protection):
            for t in (g, h):
                if t.shape != p.shape or t.device != p.device or t.dtype != torch.float32 or not torch.isfinite(t).all():
                    raise ValueError('invalid raw global auxiliary gradient')
        joint, h_owned = [], []
        for gi, owned in enumerate(self.groups):
            grads = [g.float().clone() for g in self.zero.averaged_gradients[gi]]
            h_flat = torch.zeros_like(owned.master)
            for f in owned.fragments:
                g = gain[f.parameter_index].flatten()[f.lp_start:f.lp_start+f.size]
                h = protection[f.parameter_index].flatten()[f.lp_start:f.lp_start+f.size]
                grads[f.gradient_index].sub_(g.reshape_as(grads[f.gradient_index]), alpha=self.config.gain_optimizer_weight)
                h_flat[f.hp_start:f.hp_start+f.size].copy_(h)
            if any(not torch.isfinite(g).all() for g in grads):
                raise ValueError('nonfinite joint gradient')
            joint.append(grads)
            h_owned.append(h_flat)
        return joint, tuple(h_owned)

    @torch.no_grad()
    def prepare(self, gain: tuple[Tensor, ...], protection: tuple[Tensor, ...], *, block_count: int) -> None:
        if self._active:
            raise RuntimeError('duplicate optimizer Gain preparation')
        self._active = True
        try:
            self._agree('profile/mapping validation', self._validate)
            self._agree('block count', lambda: self._check_block_count(block_count))
            # Agree shared metadata before entering any per-group collective loops.
            metadata = (self._signature, block_count, self.config.gain_optimizer_weight, self.zero.clip_grad)
            digest = hashlib.sha256(json.dumps(metadata).encode()).digest()
            signature = torch.tensor([int.from_bytes(digest[i:i+4], 'big') for i in range(0, 32, 4)],
                                     device=self.device, dtype=torch.int64)
            total = self._sum(signature.clone())
            self._agree('shared group identity', lambda: self._check_equal_signature(total, signature))
            self.metrics = dict(mode='adamw_projected', optimizer_weight=self.config.gain_optimizer_weight,
                applied=False, bypass_reason=None, base_gradient_norm=None, joint_gradient_norm=None,
                base_displacement_norm=None, joint_displacement_norm=None, gain_displacement_norm_before=None,
                gain_displacement_norm_after=None, rv_dot_before=None, rv_dot_after=None,
                projection_count=0, rounding_fallback_count=0, tiny_h_count=0)
            self.metrics.update({f'lr_group_{i}': lr for i, lr in enumerate(self.lrs)})
            self._seen = []
            self._joint_delta_sq = torch.zeros((), dtype=torch.float64, device=self.device)
            bypass = 'zero_weight' if self.config.gain_optimizer_weight == 0 else ('zero_blocks' if block_count == 0 else None)
            self.metrics['bypass_reason'] = bypass
            norm = self._checked_global_norm('base')
            self.metrics['base_gradient_norm'] = float(norm)
            self.metrics['joint_gradient_norm'] = float(norm)
            if not bypass:
                joint, self._protection = self._agree('owned gradients', lambda: self._owned_inputs(gain, protection))
                base, delta_sq = [], torch.zeros((), dtype=torch.float64, device=self.device)
                def calculate_base():
                    nonlocal delta_sq
                    for gi, owned in enumerate(self.groups):
                        ordinary = torch.cat([g.flatten() for g in self.zero.averaged_gradients[gi]]).float()
                        clipped = clip_base_gradient(ordinary, global_norm=norm, max_norm=self.zero.clip_grad)
                        candidate = adamw_base_candidate(owned.master, clipped, self.native.state.get(owned.master, {}),
                                                         self._original_groups[gi])
                        _finite_master_and_state(candidate, {})
                        delta_sq += (candidate[owned.valid_mask].double()-owned.master[owned.valid_mask].double()).square().sum()
                        base.append(candidate)
                self._agree('base candidate', calculate_base)
                self._base = tuple(base)
                self.metrics['base_displacement_norm'] = float(self._sum(delta_sq).sqrt())
                for gi, gradients in enumerate(joint):
                    self.zero.averaged_gradients[gi] = gradients
                norm = self._checked_global_norm('joint')
                self.metrics['joint_gradient_norm'] = float(norm)
            self._hooks = [self.native.register_step_pre_hook(self._before_native),
                           self.native.register_step_post_hook(self._after_native)]
        except BaseException:
            self.abort()
            raise

    @staticmethod
    def _check_block_count(count):
        if type(count) is not int or count < 0:
            raise ValueError('invalid global block count')

    def _check_equal_signature(self, total, local):
        if not torch.equal(total, local * dist.get_world_size(self.process_group)):
            raise ValueError('different optimizer group metadata on ranks')

    @staticmethod
    def _check_norm(norm):
        if norm.numel() != 1 or not torch.isfinite(norm).all() or norm < 0:
            raise ValueError('invalid native gradient norm')

    def _checked_global_norm(self, label):
        # DS masks invalid per-group norms to -1. Its subsequent vector norm
        # loses that sign, so validate each native value before aggregation.
        norms = []
        for i in range(len(self.groups)):
            norm = self.zero.get_grad_norm_direct(self.zero.averaged_gradients[i],
                                                 self.zero.params_in_partition[i])
            self._agree(f'{label} group norm', lambda: self._check_norm(norm))
            norms.append(norm)
        result = torch.linalg.vector_norm(torch.stack(norms), ord=2)
        self._agree(f'{label} norm', lambda: self._check_norm(result))
        return result

    def _before_native(self, optimizer, args, kwargs):
        def check():
            index = len(self._seen)
            if index >= len(self.groups) or len(optimizer.param_groups) != 1 or optimizer.param_groups[0] is not self._original_groups[index]:
                raise ValueError('unexpected native group dispatch')
            if float(optimizer.param_groups[0]['lr']) != self.lrs[index]:
                raise ValueError('LR changed before native group step')
            self._snapshot = self.groups[index].master.detach().clone()
        self._agree('native pre-step', check)

    def _after_native(self, optimizer, args, kwargs):
        index = len(self._seen)
        def check():
            master = self.groups[index].master
            _finite_master_and_state(master, optimizer.state.get(master, {}), require_state=True)
            mask = self.groups[index].valid_mask
            self._joint_delta_sq.add_((master[mask].double()-self._snapshot[mask].double()).square().sum())
        self._agree('native post-step', check)
        self._snapshot = None
        self._seen.append(index)

    @torch.no_grad()
    def finish(self, *, successful: bool) -> dict[str, float | int | bool | str | None]:
        if not self._active or type(successful) is not bool:
            raise RuntimeError('unprepared optimizer Gain finish')
        try:
            def check_boundary():
                if len(self._seen) != (len(self.groups) if successful else 0):
                    raise ValueError('native group count/skip mismatch')
                if successful:
                    for i, owned in enumerate(self.groups):
                        _finite_master_and_state(owned.master, self.native.state.get(owned.master, {}), require_state=True)
                        if self._base and self.lrs[i] == 0 and not torch.equal(owned.master[owned.valid_mask], self._base[i][owned.valid_mask]):
                            raise ValueError('zero-LR group moved')
            self._agree('finish boundary', check_boundary)
            self.metrics['applied'] = successful
            if successful:
                self.metrics['joint_displacement_norm'] = float(self._sum(self._joint_delta_sq).sqrt())
                if self._base:
                    result = project_adamw_increment(self._base, tuple(g.master for g in self.groups), self._protection,
                        valid_masks=tuple(g.valid_mask for g in self.groups), lrs=self.lrs, sum_fp64=self._sum,
                        min_h_norm_sq=self.config.min_h_norm_sq)
                    def validate_final():
                        for candidate in result.parameters:
                            _finite_master_and_state(candidate, {})
                    self._agree('final candidate', validate_final)
                    for owned, candidate in zip(self.groups, result.parameters):
                        owned.master.copy_(candidate)
                    self.zero.update_lp_params()
                    self.metrics.update(gain_displacement_norm_before=result.gain_norm_before,
                        gain_displacement_norm_after=result.gain_norm_after, rv_dot_before=result.dot_before,
                        rv_dot_after=result.dot_after, projection_count=int(result.projected),
                        rounding_fallback_count=int(result.rounding_fallback), tiny_h_count=int(result.tiny_h))
            else:
                self.metrics['bypass_reason'] = 'skipped'
                for key in self.metrics:
                    if key.endswith('_norm') or key.startswith('gain_displacement_norm') or key.startswith('rv_dot'):
                        self.metrics[key] = None
            return dict(self.metrics)
        finally:
            self.abort()

    def abort(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if self._original_groups is not None:
            self.native.param_groups = self._original_groups
        self._base, self._protection, self._snapshot = (), (), None
        self._active = False

"""Controlled CPU ZeRO storage/collectives; actual AdamW and DS fragment linkage.

This fixture does not establish CUDA fused or real multi-process ZeRO behavior.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Condition
from types import SimpleNamespace
import math

import torch
from deepspeed.utils.mixed_precision_linkage import link_hp_params
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer


class Collective:
    def __init__(self, size):
        self.size = size
        self.condition = Condition()
        self.slots = {}
        self.indices = [0] * size
        self.traces = [[] for _ in range(size)]

    def reduce(self, rank, tensor):
        with self.condition:
            index = self.indices[rank]
            self.indices[rank] += 1
            self.traces[rank].append((str(tensor.dtype), tuple(tensor.shape)))
            slot = self.slots.setdefault(index, {})
            slot[rank] = tensor.clone()
            self.condition.notify_all()
            if not self.condition.wait_for(lambda: len(slot) == self.size, timeout=15):
                raise RuntimeError('controlled collective order diverged')
            return torch.stack([slot[i] for i in range(self.size)]).sum(0)


def install_cpu_collectives(monkeypatch, api):
    def all_reduce(tensor, op=None, group=None):
        tensor.copy_(group.collective.reduce(group.rank, tensor))
    monkeypatch.setattr(api, 'dist', SimpleNamespace(is_available=lambda: True,
        is_initialized=lambda: True, get_rank=lambda group: group.rank,
        get_world_size=lambda group: group.size, all_reduce=all_reduce,
        ReduceOp=SimpleNamespace(SUM='sum')))
    # This is the only supported-profile relaxation: CPU/non-fused instead of CUDA/fused.
    monkeypatch.setattr(api, '_require_supported_gpu_optimizer', lambda optimizer: None)


class ControlledZero:
    _optimizer_step = DeepSpeedZeroOptimizer._optimizer_step
    unscale_and_clip_grads = DeepSpeedZeroOptimizer.unscale_and_clip_grads

    def __init__(self, group_values, *, collective, rank=0, lrs=None, clip=1.):
        self.dp_process_group = SimpleNamespace(rank=rank, size=collective.size, collective=collective)
        self.real_dp_process_group = [self.dp_process_group] * len(group_values)
        self.partition_gradients = self.cpu_offload = self.has_moe_layers = self.round_robin_gradients = False
        self.model_parallel_world_size = 1
        self.torch_autocast_gradscaler = self.zenflow = None
        self.loss_scale = 1.
        self.clip_grad = clip
        self.events = []
        self.bit16_groups, self.params_in_partition = [], []
        self.single_partition_of_fp32_groups, self.partition_size = [], []
        self.bit16_groups_flat = []
        self.averaged_gradients = {}
        groups = []
        for i, values in enumerate(group_values):
            params = [torch.nn.Parameter(torch.tensor(v, dtype=torch.bfloat16)) for v in values]
            self.bit16_groups.append(params)
            flat = torch.cat([p.detach().flatten() for p in params])
            size = math.ceil(flat.numel() / (2 * collective.size)) * 2
            padded = torch.nn.functional.pad(flat, (0, size * collective.size - flat.numel()))
            self.bit16_groups_flat.append(padded)
            master = torch.nn.Parameter(padded[rank*size:(rank+1)*size].float().clone())
            self.single_partition_of_fp32_groups.append(master)
            self.partition_size.append(size)
            groups.append(dict(params=[master], lr=.03 if lrs is None else lrs[i]))
            link_hp_params(params, master, self.averaged_gradients, {}, False, i,
                           rank*size, size, self.dp_process_group)
            self.params_in_partition.append([p for p in params if p._hp_mapping is not None])
        self.optimizer = torch.optim.AdamW(groups, fused=False)
        self.reset_gradients()

    def _lazy_init_hp_params_optimizer_state(self):
        pass

    def reset_gradients(self, values=None):
        # Ordinary gradients start in BF16, matching the standing accumulation profile.
        for i, params in enumerate(self.params_in_partition):
            fragments = []
            for p in params:
                m = p._hp_mapping
                full = torch.ones_like(p) if values is None else values[self.all_parameters.index(p)]
                addr = m.lp_fragment_address
                fragments.append(full.flatten()[addr.start:addr.start+addr.numel].clone())
            used = sum(t.numel() for t in fragments)
            if used < self.partition_size[i]:
                fragments.append(torch.zeros(self.partition_size[i]-used, dtype=torch.bfloat16))
            self.averaged_gradients[i] = fragments

    @property
    def all_parameters(self):
        return tuple(p for group in self.bit16_groups for p in group)

    def scaled_global_norm(self):
        norms = []
        group = self.dp_process_group
        for i in range(len(self.bit16_groups)):
            grads = self.averaged_gradients[i][:len(self.params_in_partition[i])]
            local = sum((g.double().norm().square() for g in grads), torch.tensor(0., dtype=torch.float64)).float()
            norms.append(group.collective.reduce(group.rank, local).sqrt())
        return torch.stack(norms).norm()

    def step(self):
        norm = self.scaled_global_norm()
        for i, master in enumerate(self.single_partition_of_fp32_groups):
            master.grad = torch.cat([g.flatten() for g in self.averaged_gradients[i]]).float()
            self.averaged_gradients[i] = None
            self.unscale_and_clip_grads([master.grad], norm)
            self.events.append('native_step')
            self._optimizer_step(i)
            master.grad = None
            self.events.append('group_bf16_copy')
            self.bit16_groups_flat[i][self.dp_process_group.rank*self.partition_size[i]:
                (self.dp_process_group.rank+1)*self.partition_size[i]].copy_(master.detach())

    def update_lp_params(self):
        self.events.append('projection_sync')


def engine_for(zero):
    return SimpleNamespace(optimizer=zero, zero_optimization_stage=lambda: 1)


def run_ranks(functions):
    with ThreadPoolExecutor(max_workers=len(functions)) as pool:
        futures = [pool.submit(f) for f in functions]
        return [f.result(timeout=30) for f in futures]

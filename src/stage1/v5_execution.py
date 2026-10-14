"""Source-coordinate stochastic ownership for F and OFF in one native backward."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
from typing import Literal

import torch
from torch import Tensor
from .equivalence import Stage1StochasticContext, stage1_sample_seed

_ACTIVE = ContextVar('stage1_v5_execution', default=None)
_FRAME = ContextVar('stage1_v5_frame', default=None)
GraphRole = Literal['F', 'packed']


def current_v5_execution():
    return _ACTIVE.get()


class _CheckpointPhase:
    def __init__(self, owner, role, unit, phase):
        self.owner, self.role, self.unit, self.phase = owner, role, unit, phase
        self.token = None

    def __enter__(self):
        owner = self.owner
        owner._require_active()
        frame = _FRAME.get()
        if self.token is not None or (self.phase == 'original' and frame != (self.role,None,'original')):
            raise RuntimeError('wrong or nested v5 checkpoint original context')
        if self.phase == 'recompute' and frame != (None,None,'backward'):
            raise RuntimeError('v5 recompute requires its ordinary backward')
        self.token = _FRAME.set((self.role,self.unit,self.phase))
        owner.events.append(dict(role=self.role,unit=self.unit,phase=self.phase,site=None))
        return self

    def __exit__(self,*exc):
        token,self.token = self.token,None
        _FRAME.reset(token)
        # Non-reentrant early-stop exceptions must restore the parent frame.
        return False


class V5Execution:
    """Keep optimizer-owned F/packed closures alive through one backward scope.

    Diagnostic autograd traversals may run inside that same scope with
    retain_graph=True; they never create another original decoder invocation.
    """
    def __init__(self, spec: Stage1StochasticContext):
        if not isinstance(spec,Stage1StochasticContext):
            raise TypeError('v5 execution requires Stage1StochasticContext')
        self.spec = spec
        self.state = 'NEW'
        self.events = []
        self._graphs = []
        self._backward_started = False
        self._backward_done = False
        self._original_sites = set()

    def _require_active(self):
        if _ACTIVE.get() is not self or self.state not in ('ACTIVE','BACKWARD'):
            raise RuntimeError('inactive or stale v5 execution')

    def __enter__(self):
        from .equivalence import _STAGE1_STOCHASTIC_CONTEXT
        if self.state != 'NEW' or _ACTIVE.get() is not None or _STAGE1_STOCHASTIC_CONTEXT.get() is not None:
            raise RuntimeError('nested or stale v5 execution')
        self.state = 'ACTIVE'
        self._token = _ACTIVE.set(self)
        self._frame_token = _FRAME.set(None)
        self._autograd = torch.autograd.set_multithreading_enabled(False)
        self._autograd.__enter__()
        return self

    def __exit__(self, exc_type,exc,traceback):
        try:
            if exc_type is None:
                if not self._backward_done or not self._graphs or self._graphs[0]!='F':
                    raise RuntimeError('incomplete v5 ordinary backward')
                if self._original_sites != set(self.spec.expected_sites):
                    raise RuntimeError('missing v5 original F stochastic sites')
                self.state = 'CLOSED'
            else:
                self.state = 'FAILED'
        finally:
            if self.state != 'CLOSED': self.state = 'FAILED'
            self._autograd.__exit__(exc_type,exc,traceback)
            _FRAME.reset(self._frame_token)
            _ACTIVE.reset(self._token)
        return False

    @contextmanager
    def graph(self, role: GraphRole):
        self._require_active()
        expected = 'F' if not self._graphs else 'packed'
        if (role not in ('F','packed') or role != expected or role in self._graphs
                or self._backward_started or _FRAME.get() is not None):
            raise RuntimeError('wrong or duplicate v5 original graph')
        token = _FRAME.set((role,None,'original'))
        try:
            yield
            self._graphs.append(role)
        finally:
            _FRAME.reset(token)

    @contextmanager
    def backward(self):
        self._require_active()
        if self._backward_started or not self._graphs or _FRAME.get() is not None:
            raise RuntimeError('wrong or duplicate v5 ordinary backward')
        self._backward_started = True
        self.state = 'BACKWARD'
        token = _FRAME.set((None,None,'backward'))
        try:
            yield
            self._backward_done = True
        finally:
            self.state = 'ACTIVE'
            _FRAME.reset(token)

    def checkpoint_contexts(self, role: GraphRole, unit: tuple[str,int]):
        self._require_active()
        if (role not in ('F','packed') or not isinstance(unit,tuple) or len(unit)!=2
                or unit[0] not in ('decoder','vision') or type(unit[1]) is not int or unit[1]<0
                or (role=='packed' and unit[0]=='vision')):
            raise ValueError('invalid v5 checkpoint role/unit')
        return (_CheckpointPhase(self,role,unit,'original'),
                _CheckpointPhase(self,role,unit,'recompute'))

    def dropout_mask(self, site, p, *, sample_indices, token_positions, hidden_size, device) -> Tensor:
        self._require_active()
        frame = _FRAME.get()
        if frame is None or frame[0] not in ('F','packed') or frame[1] is None:
            raise RuntimeError('v5 dropout requires native checkpoint unit ownership')
        if site not in self.spec.expected_sites:
            raise RuntimeError(f'unregistered v5 stochastic site: {site}')
        if type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<1:
            raise ValueError('v5 dropout probability must be finite in [0,1)')
        if (not isinstance(sample_indices,Tensor) or not isinstance(token_positions,Tensor)
                or sample_indices.ndim!=1 or token_positions.shape!=sample_indices.shape
                or sample_indices.dtype!=torch.long or token_positions.dtype!=torch.long
                or type(hidden_size) is not int or hidden_size<=0):
            raise ValueError('v5 dropout requires aligned int64 source coordinates')
        samples,positions = sample_indices.tolist(),token_positions.tolist()
        if any(not 0<=r<len(self.spec.rows) or not 0<=t<self.spec.rows[r][1] for r,t in zip(samples,positions)):
            raise ValueError('v5 dropout coordinates outside source valid length')
        role,unit,phase = frame
        if role=='F' and phase=='original':
            if site in self._original_sites: raise RuntimeError('duplicate v5 original F stochastic site')
            self._original_sites.add(site)
        self.events.append(dict(role=role,unit=unit,phase=phase,site=site))
        masks = {}
        for row in set(samples):
            slot,length = self.spec.rows[row]
            generator = torch.Generator(device=device)
            generator.manual_seed(stage1_sample_seed(self.spec.base_seed,self.spec.optimizer_global_step,slot,site))
            masks[row] = torch.rand((length,hidden_size),device=device,dtype=torch.float32,generator=generator).ge(p).float()
        return (torch.stack([masks[r][t] for r,t in zip(samples,positions)]) if samples
                else torch.empty((0,hidden_size),device=device,dtype=torch.float32))

    def native_dropout(self, site, p, x):
        if (not isinstance(x,Tensor) or x.ndim!=3 or not x.is_floating_point()
                or x.layout!=torch.strided or x.shape[0]!=len(self.spec.rows)
                or min(x.shape[1:])<=0 or any(n>x.shape[1] for _,n in self.spec.rows)):
            raise ValueError('v5 native dropout requires source floating [B,T,H]')
        lengths = torch.tensor([n for _,n in self.spec.rows],device=x.device)[:,None]
        samples = torch.arange(x.shape[0],device=x.device)[:,None].expand(x.shape[:2])
        positions = torch.arange(x.shape[1],device=x.device)[None].expand(x.shape[:2])
        valid = positions<lengths
        raw = self.dropout_mask(site,p,sample_indices=samples[valid],token_positions=positions[valid],
                                hidden_size=x.shape[-1],device=x.device)
        mask = torch.ones_like(x)
        mask[valid] = raw.to(x.dtype)/(1-p)
        return x*mask

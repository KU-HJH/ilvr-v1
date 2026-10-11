"""Per-microbatch graph/VJP identity with reusable checkpoint phase managers."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Literal, Mapping
import math

import torch
from torch import Tensor

from .equivalence import Stage1StochasticContext, stage1_sample_seed

GraphRole = Literal["F", "gain_clean", "packed_replay"]
BackwardRole = Literal["ordinary", "joint_r_kl", "rv"]
Unit = tuple[Literal["decoder", "vision"], int]
_ACTIVE: ContextVar[V3Lifecycle | None] = ContextVar("stage1_v3_lifecycle", default=None)
_FRAME = ContextVar("stage1_v3_frame", default=(None, None, None, None))


def current_v3_lifecycle() -> V3Lifecycle | None:
    return _ACTIVE.get()


class _CheckpointPhase:
    """PyTorch retains and sequentially re-enters this object for each VJP."""
    def __init__(self, owner, graph, unit, phase):
        self.owner, self.graph, self.unit, self.phase = owner, graph, unit, phase
        self.token = None

    def __enter__(self):
        owner = self.owner
        owner._require_active()
        if self.token is not None:
            owner._error("nested checkpoint phase manager")
        graph, unit, phase, backward = _FRAME.get()
        if self.phase == "original":
            if graph != self.graph or unit is not None or backward is not None:
                owner._error("wrong graph/unit for original checkpoint")
        else:
            allowed = ("F",) if backward == "ordinary" else ("gain_clean", "packed_replay")
            if backward is None or self.graph not in allowed or unit is not None:
                owner._error("wrong graph/VJP for checkpoint recompute")
        key = (self.graph, self.unit, self.phase, backward)
        if key in owner._unit_visits:
            owner._error(f"duplicate {self.phase} checkpoint unit in {backward}")
        owner._unit_visits.add(key)
        self.token = _FRAME.set((self.graph, self.unit, self.phase, backward))
        return self

    def __exit__(self, exc_type, exc, traceback):
        # Early-stop checkpointing exits via an internal exception. Only the
        # enclosing backward/activation decides whether a failure escaped.
        token, self.token = self.token, None
        _FRAME.reset(token)
        return False


class V3Lifecycle:
    def __init__(self, spec: Stage1StochasticContext,
                 inventory: Mapping[tuple[GraphRole, Unit], tuple[str, ...]], *, has_auxiliary: bool):
        if not isinstance(spec, Stage1StochasticContext) or type(has_auxiliary) is not bool:
            raise ValueError("v3 lifecycle requires immutable stochastic identity and boolean auxiliary state")
        self.spec, self.has_auxiliary = spec, has_auxiliary
        self._graphs = ("F", "gain_clean", "packed_replay") if has_auxiliary else ("F",)
        self._backwards = ("joint_r_kl", "rv", "ordinary") if has_auxiliary else ("ordinary",)
        frozen = dict(inventory)
        for (graph, unit), sites in frozen.items():
            if (graph not in self._graphs or not isinstance(unit, tuple) or len(unit) != 2
                    or unit[0] not in ("decoder", "vision") or type(unit[1]) is not int or unit[1] < 0
                    or not isinstance(sites, tuple) or len(set(sites)) != len(sites)
                    or any(type(s) is not str or not s for s in sites)):
                raise ValueError("invalid v3 graph/unit/site inventory")
        if {g for g, _ in frozen} != set(self._graphs):
            raise ValueError("missing v3 graph inventory")
        for graph in self._graphs:
            sites = [s for (g, _), values in frozen.items() if g == graph for s in values]
            if len(set(sites)) != len(sites):
                raise ValueError("v3 site must belong to one unit per graph")
            expected = (set(spec.expected_sites) if graph != "packed_replay" else
                        {s for (g, u), values in frozen.items() if g == "F" and u[0] == "decoder" for s in values})
            if set(sites) != expected:
                raise ValueError(f"v3 {graph} inventory differs from predeclared native sites")
        self.inventory = MappingProxyType(frozen)
        self.state = "NEW"
        self.events = []
        self._completed_graphs = []
        self._started_backwards = set()
        self._completed_backwards = []
        self._unit_visits = set()
        self._site_visits = set()

    @property
    def frame(self):
        return _FRAME.get() if _ACTIVE.get() is self else (None, None, None, None)

    def _error(self, message):
        self.state = "FAILED"
        raise RuntimeError(message)

    def _require_active(self):
        if _ACTIVE.get() is not self or self.state in ("NEW", "CLOSED", "FAILED"):
            raise RuntimeError("stale or inactive v3 lifecycle")

    @contextmanager
    def activate(self):
        from .equivalence import _STAGE1_STOCHASTIC_CONTEXT
        if self.state != "NEW":
            raise RuntimeError("stale v3 lifecycle cannot be reactivated")
        if _ACTIVE.get() is not None or _STAGE1_STOCHASTIC_CONTEXT.get() is not None:
            raise RuntimeError("nested v3/ordinary stochastic activation")
        self.state = "ACTIVE"
        token = _ACTIVE.set(self)
        frame_token = _FRAME.set((None, None, None, None))
        try:
            # CUDA engine worker threads do not inherit the ContextVars used
            # by checkpoint recompute. Keep backward on the activating thread;
            # the manager restores the caller's setting on every exit path.
            with torch.autograd.set_multithreading_enabled(False):
                yield self
                if self.state == "FAILED" or tuple(self._completed_backwards) != self._backwards:
                    self._error("incomplete v3 backward lifecycle")
                expected = {(g, u, s, "original", None)
                            for (g, u), sites in self.inventory.items() for s in sites}
                if expected - self._site_visits:
                    self._error("missing v3 original stochastic site observations")
                self.state = "CLOSED"
        except BaseException:
            self.state = "FAILED"
            raise
        finally:
            _FRAME.reset(frame_token)
            _ACTIVE.reset(token)

    @contextmanager
    def graph(self, role: GraphRole):
        self._require_active()
        if _FRAME.get() != (None, None, None, None):
            self._error("nested v3 graph")
        if role in self._completed_graphs:
            self._error(f"duplicate original graph {role}")
        if (len(self._completed_graphs) >= len(self._graphs)
                or role != self._graphs[len(self._completed_graphs)] or self._started_backwards):
            self._error(f"wrong v3 graph order: {role}")
        self.state = role
        token = _FRAME.set((role, None, "original", None))
        try:
            yield
            self._completed_graphs.append(role)
        except BaseException:
            self.state = "FAILED"
            raise
        finally:
            _FRAME.reset(token)

    @contextmanager
    def backward(self, role: BackwardRole):
        self._require_active()
        if role in self._started_backwards:
            self._error(f"duplicate v3 backward {role}")
        if (_FRAME.get() != (None, None, None, None)
                or tuple(self._completed_graphs) != self._graphs
                or len(self._completed_backwards) >= len(self._backwards)
                or role != self._backwards[len(self._completed_backwards)]):
            self._error(f"wrong v3 backward order: {role}")
        self._started_backwards.add(role)
        self.state = role
        token = _FRAME.set((None, None, None, role))
        try:
            yield
            self._completed_backwards.append(role)
        except BaseException:
            self.state = "FAILED"
            raise
        finally:
            _FRAME.reset(token)

    def checkpoint_contexts(self, graph: GraphRole, unit: Unit):
        self._require_active()
        if (graph, unit) not in self.inventory:
            self._error(f"unregistered v3 graph/unit: {graph}/{unit}")
        return (_CheckpointPhase(self, graph, unit, "original"),
                _CheckpointPhase(self, graph, unit, "recompute"))

    def record(self, site: str, *, origin: Literal["native", "packed"]) -> None:
        self._require_active()
        graph, unit, phase, backward = _FRAME.get()
        if site not in self.inventory.get((graph, unit), ()):
            self._error(f"unregistered v3 stochastic site: {graph}/{unit}/{site}")
        if origin != ("packed" if graph == "packed_replay" else "native"):
            self._error(f"wrong v3 stochastic origin {origin} for {graph}")
        key = (graph, unit, site, phase, backward)
        if key in self._site_visits:
            self._error(f"duplicate {phase} invocation in {backward}: {site}")
        self._site_visits.add(key)
        self.events.append(dict(graph=graph, unit=unit, site=site,
                                phase=phase, backward=backward, origin=origin))

    def native_dropout(self, site: str, p: float, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor) or x.ndim != 3 or not x.is_floating_point() or x.layout != torch.strided:
            raise ValueError("v3 native dropout requires floating strided [B,T,H]")
        if (x.shape[0] != len(self.spec.rows) or min(x.shape[1:]) <= 0
                or any(length > x.shape[1] for _, length in self.spec.rows)):
            raise ValueError("v3 native dropout shape does not match sample identity")
        self.record(site, origin="native")
        if p == 0:
            return x
        mask = torch.ones_like(x)
        for row, (slot, length) in enumerate(self.spec.rows):
            raw = _raw_mask(self.spec, site, p, slot, length, x.shape[-1], x.device)
            mask[row, :length] = raw.to(x.dtype) * (1 / (1 - p))
        return x * mask


def _raw_mask(spec, site, p, slot, length, hidden_size, device):
    if p == 0:
        return torch.ones((length, hidden_size), device=device, dtype=torch.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(stage1_sample_seed(spec.base_seed, spec.optimizer_global_step, slot, site))
    return torch.rand((length, hidden_size), device=device, dtype=torch.float32,
                      generator=generator).ge(p).to(torch.float32)


def packed_dropout_mask(site: str, p: float, *, sample_indices: Tensor,
                        token_positions: Tensor, hidden_size: int, device: torch.device) -> Tensor:
    """FP32 Bernoulli mask; cast then scale by1/(1-p) in the activation dtype."""
    owner = current_v3_lifecycle()
    if owner is None:
        raise RuntimeError("missing v3 stochastic lifecycle")
    if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p < 1:
        raise ValueError("packed dropout probability must be finite in [0,1)")
    if (type(hidden_size) is not int or hidden_size <= 0
            or not isinstance(sample_indices, Tensor) or not isinstance(token_positions, Tensor)
            or sample_indices.ndim != 1 or token_positions.shape != sample_indices.shape
            or sample_indices.dtype != torch.long or token_positions.dtype != torch.long):
        raise ValueError("packed dropout requires aligned int64 source coordinates")
    samples, positions = sample_indices.tolist(), token_positions.tolist()
    for row, position in zip(samples, positions):
        if not 0 <= row < len(owner.spec.rows) or not 0 <= position < owner.spec.rows[row][1]:
            raise ValueError("packed dropout coordinate outside source valid length")
    owner.record(site, origin="packed")
    masks = {row: _raw_mask(owner.spec, site, p, *owner.spec.rows[row], hidden_size, device)
             for row in set(samples)}
    return (torch.stack([masks[row][position] for row, position in zip(samples, positions)])
            if samples else torch.empty((0, hidden_size), device=device, dtype=torch.float32))

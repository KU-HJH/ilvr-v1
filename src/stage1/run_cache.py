"""Run-owned, lossless Stage1 records; no model work or cluster orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Literal

import torch


def resolve_stage1_check_policy(*, requested: str | None, saved: dict | None,
                                validation: bool) -> str:
    policies = ('production', 'validation', 'legacy')
    inherited = saved.get('check_policy', 'legacy') if saved is not None else None
    result = requested or inherited or ('validation' if validation else 'production')
    if result not in policies:
        raise ValueError('unsupported Stage1 check policy')
    if inherited is not None and result != inherited:
        raise ValueError('resume must preserve its saved check policy')
    if result == 'validation' and not validation and inherited != 'validation':
        raise ValueError('validation checks require the isolated validation path')
    return result


@dataclass(frozen=True)
class Stage1CachePlan:
    run_id: str
    root: Path
    helper: Literal['off', 'targets', 'vision']
    reference_hidden: bool
    identity: dict[str, object]

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', self.run_id):
            raise ValueError('invalid cache run identity')
        object.__setattr__(self, 'root', Path(self.root))
        if self.root.name != self.run_id or '..' in self.root.parts:
            raise ValueError('cache root must identify exactly one run')
        if self.helper not in ('off', 'targets', 'vision'):
            raise ValueError('unsupported helper cache layer')
        if type(self.reference_hidden) is not bool or type(self.identity) is not dict:
            raise ValueError('invalid cache identity')
        json.dumps(self.identity, sort_keys=True, allow_nan=False)


def _cpu_record(value):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError('cache requires dense tensors')
        return value.detach().to(device='cpu').contiguous().clone()
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError('cache metadata keys must be strings')
        return {key: _cpu_record(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_cpu_record(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f'unsupported cache record value: {type(value).__name__}')


def _tensor_layout(value, path=()):
    if isinstance(value, torch.Tensor):
        return [(path, tuple(value.shape), str(value.dtype))]
    if type(value) is dict:
        return [item for key, child in value.items()
                for item in _tensor_layout(child, (*path, key))]
    if type(value) in (list, tuple):
        return [item for index, child in enumerate(value)
                for item in _tensor_layout(child, (*path, index))]
    return []


def file_sha256(path: Path) -> str:
    """Boundary/verification use only; never called by training reads."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Stage1RunCache:
    def __init__(self, plan: Stage1CachePlan, phase):
        self.plan, self.phase = plan, phase
        self._closed = False
        self.layers = ('preprocess',) + (() if plan.helper == 'off' else (plan.helper,))
        if plan.reference_hidden:
            self.layers += ('reference',)
        if phase.value not in (0, 1):
            raise ValueError('unsupported cache phase')
        if plan.root.is_symlink():
            raise ValueError('cache run must not be a symlink')
        identity = {'schema': 1, 'run_id': plan.run_id, 'helper': plan.helper,
                    'reference_hidden': plan.reference_hidden, 'identity': plan.identity}
        self.identity_json = json.dumps(identity, sort_keys=True, allow_nan=False)
        self.identity_sha256 = hashlib.sha256(self.identity_json.encode()).hexdigest()
        plan.root.parent.mkdir(parents=True, exist_ok=True)
        # Both local ranks may initialize together. Lock the existing parent
        # directory briefly; no lock file or unowned run directory is adopted.
        parent_fd = os.open(plan.root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            marker = plan.root / 'identity.json'
            if plan.root.is_symlink() or marker.is_symlink():
                raise ValueError('cache run/identity must not be a symlink')
            if plan.root.exists():
                if not marker.is_file():
                    raise ValueError('existing cache directory is unowned')
                if marker.read_text() != self.identity_json:
                    raise ValueError('cache identity conflicts with existing run')
            else:
                plan.root.mkdir()
                marker.write_text(self.identity_json)
        finally:
            os.close(parent_fd)
        for layer in self.layers:
            directory = plan.root / layer
            if directory.is_symlink():
                raise ValueError('cache layer must not be a symlink')
            directory.mkdir(exist_ok=True)

    def _path(self, layer: str, source_index: int) -> Path:
        if self._closed:
            raise RuntimeError('cache is closed')
        if layer not in self.layers or type(source_index) is not int or source_index < 0:
            raise ValueError('invalid or ineligible cache record')
        return self.plan.root / layer / f'{source_index}.pt'

    def publish(self, layer: str, source_index: int, record: dict) -> bool:
        target = self._path(layer, source_index)
        if self.phase.value != 0:
            raise RuntimeError('cannot publish during cache replay')
        if type(record) is not dict:
            raise TypeError('cache record must be a dictionary')
        # Duplicate occurrences have already computed normally. Avoid copying
        # their tensors to CPU when a complete record has been published.
        if target.exists():
            if target.is_symlink():
                raise ValueError('cache record must not be a symlink')
            return False
        payload = _cpu_record(record)
        envelope = {'schema': 1, 'identity': self.identity_sha256,
                    'layer': layer, 'source_index': source_index,
                    'layout': _tensor_layout(payload), 'record': payload}
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='record-', suffix='.pt', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            # A file handle gives the ZIP archive a stable internal root. A
            # random temporary filename would change identical records' bytes.
            with temporary.open('wb') as stream:
                torch.save(envelope, stream)
            try:
                os.link(temporary, target)
            except FileExistsError:
                return False
        finally:
            temporary.unlink()
        return True

    def _load(self, layer: str, source_index: int, *, validate: bool = False) -> dict:
        path = self._path(layer, source_index)
        if path.is_symlink():
            raise ValueError('cache record must not be a symlink')
        # PRIVATE mappings share OS-backed pages but cannot write back to the
        # file. Consumers copy only tensors they need to mutate for this batch.
        with torch.serialization.set_default_mmap_options(torch.serialization.MAP_PRIVATE):
            payload = torch.load(path, mmap=True, weights_only=True, map_location='cpu')
        if (type(payload) is not dict or payload.get('schema') != 1
                or payload.get('identity') != self.identity_sha256
                or payload.get('layer') != layer or payload.get('source_index') != source_index
                or type(payload.get('record')) is not dict):
            raise ValueError('malformed cache record identity')
        if validate and _tensor_layout(payload['record']) != payload.get('layout'):
            raise ValueError('cache record shape/dtype mismatch')
        return payload['record']

    def read(self, layer: str, source_index: int) -> dict:
        if self._closed:
            raise RuntimeError('cache is closed')
        if self.phase.value != 1:
            raise RuntimeError('cache reads require replay phase')
        return self._load(layer, source_index)

    def validate_coverage(self, expected: dict[str, tuple[int, ...]]) -> dict:
        records = {}
        for layer, indices in expected.items():
            for index in indices:
                path = self._path(layer, index)
                if not path.is_file():
                    raise FileNotFoundError(f'missing cache coverage: {layer}/{index}')
                self._load(layer, index, validate=True)
                records[f'{layer}/{index}.pt'] = {'sha256': file_sha256(path),
                                                 'bytes': path.stat().st_size}
        return {'schema': 1, 'identity': self.identity_sha256, 'records': records}

    def close(self) -> None:
        # The store holds no tensor mappings. Consumers/worker processes own
        # their current batch and must be drained before lifecycle cleanup.
        self._closed = True

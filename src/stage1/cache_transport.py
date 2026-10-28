"""One boundary exchange of run-owned records; payload IO is always streamed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

from .run_cache import Stage1RunCache


BLOCK = 1024 * 1024


def record_path(cache, name):
    match = re.fullmatch(r'(preprocess|targets|vision|reference)/(0|[1-9][0-9]*)\.pt', name)
    if match is None:
        raise ValueError('invalid cache record path')
    path = cache._path(match[1], int(match[2]))
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('cache record path must not be a symlink')
    return path


def local_manifest(cache, expected):
    present = {layer: tuple(index for index in indices if cache._path(layer, index).is_file())
               for layer, indices in expected.items()}
    return cache.validate_coverage(present)


def validate_manifest(cache, manifest):
    if (manifest.get('schema') != 1 or manifest.get('identity') != cache.identity_sha256
            or type(manifest.get('records')) is not dict):
        raise ValueError('cache manifest identity mismatch')
    for name, info in manifest['records'].items():
        record_path(cache, name)
        if (set(info) != {'bytes', 'sha256'} or type(info['bytes']) is not int
                or info['bytes'] <= 0 or re.fullmatch(r'[0-9a-f]{64}', info['sha256']) is None):
            raise ValueError('malformed cache record manifest')


def export_records(cache, manifest, names, stream):
    validate_manifest(cache, manifest)
    with tarfile.open(fileobj=stream, mode='w|') as archive:
        for name in names:
            path = record_path(cache, name)
            info = tarfile.TarInfo(name)
            info.size = manifest['records'][name]['bytes']
            if path.stat().st_size != info.size:
                raise ValueError('cache record size changed before transfer')
            with path.open('rb') as source:
                archive.addfile(info, source)


def import_records(cache, manifest, names, stream):
    validate_manifest(cache, manifest)
    remaining = set(names)
    if len(remaining) != len(names) or not remaining <= manifest['records'].keys():
        raise ValueError('invalid requested cache records')
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            path = record_path(cache, member.name)
            if member.name not in remaining or not member.isfile():
                raise ValueError('unexpected cache record path or type')
            expected = manifest['records'][member.name]
            if member.size != expected['bytes']:
                raise ValueError('cache record transfer size mismatch')
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='transfer-', delete=False) as target:
                temporary = Path(target.name)
                try:
                    source = archive.extractfile(member)
                    count = 0
                    for block in iter(lambda: source.read(BLOCK), b''):
                        target.write(block)
                        digest.update(block)
                        count += len(block)
                    target.flush()
                    if count != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                        raise ValueError('cache record transfer integrity mismatch')
                    # Never overwrite a concurrent/foreign publication.
                    os.link(temporary, path)
                finally:
                    temporary.unlink()
            remaining.remove(member.name)
    if remaining:
        raise ValueError('cache transfer missing records')


def exchange_stage1_cache(plan, runner, *, expected):
    """Runner supplies the existing two-node transport, not a tensor collective."""
    manifests = {rank: runner.manifest(rank, expected) for rank in (0, 1)}
    if (manifests[0]['schema'], manifests[0]['identity']) != (manifests[1]['schema'], manifests[1]['identity']):
        raise ValueError('cache node identity conflict')
    union = dict(manifests[0]['records'])
    for name, info in manifests[1]['records'].items():
        if name in union and union[name] != info:
            raise ValueError(f'cache duplicate record conflict: {name}')
        union[name] = info
    required = {f'{layer}/{index}.pt' for layer, indices in expected.items() for index in indices}
    if set(union) != required:
        raise FileNotFoundError(f'missing cache coverage: {sorted(required - union.keys())[:8]}')
    combined = {**manifests[0], 'records': union}
    for destination in (0, 1):
        names = tuple(sorted(required - manifests[destination]['records'].keys()))
        if names:
            runner.transfer(1 - destination, destination, combined, names)
    for rank in (0, 1):
        if runner.validate(rank, expected) != combined:
            raise ValueError('cache node coverage differs after transfer')
    # Both complete before either iterator can start its next epoch.
    for rank in (0, 1):
        runner.release(rank, combined)
    return {'records': len(union), 'bytes_per_node': sum(item['bytes'] for item in union.values())}


class LocalCacheExchange:
    """Same filesystem adapter for the isolated CPU/integration verifier."""
    def __init__(self, caches):
        self.caches = caches

    def manifest(self, rank, expected):
        return local_manifest(self.caches[rank], expected)

    def transfer(self, source, destination, manifest, names):
        with tempfile.TemporaryFile() as stream:
            export_records(self.caches[source], manifest, names, stream)
            stream.seek(0)
            import_records(self.caches[destination], manifest, names, stream)

    def validate(self, rank, expected):
        return self.caches[rank].validate_coverage(expected)

    def release(self, rank, manifest):
        pass


class SubprocessCacheExchange:
    """Use the existing coordinator's fixed host/container and failure handling."""
    def __init__(self, plan, runner, *, cache_root, control_dir):
        self.plan, self.runner = plan, runner
        self.cache_root, self.control_dir = Path(cache_root), Path(control_dir)
        self.completed = False
        self.ready = {}

    def command(self, rank, operation):
        from src.multinode_runtime import node_repository, node_path, _build_remote_stdin_exec
        repo = node_repository(self.plan, machine_rank=rank)
        control = node_path(self.plan, self.control_dir, machine_rank=rank)
        command = ['/usr/bin/env', f'PYTHONPATH={repo / "transformers/src"}:{repo / "src"}:{repo}',
                   str(repo / 'venv/ilvr-venv/bin/python'), '-m', 'src.stage1.cache_transport',
                   operation, str(self.cache_root), str(control)]
        if rank == 0:
            return _build_remote_stdin_exec(command, working_directory=repo, cluster=self.plan.cluster)
        return command

    def request(self, rank, operation, payload=None):
        result = self.runner._run(self.command(rank, operation),
                                 input_bytes=(json.dumps(payload or {}) + '\n').encode())
        return json.loads(result.stdout)

    def boundary(self):
        if self.completed or not (self.control_dir / 'ready.json').exists():
            return
        local = json.loads((self.control_dir / 'ready.json').read_text())
        remote = self.request(0, 'ready')
        if remote is None:
            return
        if (local['run_id'] != self.plan.run_id or remote['run_id'] != self.plan.run_id
                or local['expected'] != remote['expected']):
            raise ValueError('node cache boundary contracts differ')
        self.ready = {0: remote, 1: local}
        exchange_stage1_cache(self.plan, self, expected=local['expected'])
        self.completed = True

    def manifest(self, rank, expected):
        return self.ready[rank]['manifest']

    def transfer(self, source, destination, manifest, names):
        import subprocess
        header = (json.dumps({'manifest': manifest, 'names': names}) + '\n').encode()
        self.control_dir.mkdir(parents=True, exist_ok=True)
        # stderr stays in the run log rather than a pipe that could fill while
        # the parent streams payloads. Only children created here are signaled.
        children = []
        with (self.control_dir / f'transfer-{source}-{destination}.log').open('ab') as log:
            try:
                exporter = subprocess.Popen(self.command(source, 'export'), stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=log)
                children.append(exporter)
                importer = subprocess.Popen(self.command(destination, 'import'), stdin=subprocess.PIPE,
                                            stdout=log, stderr=log)
                children.append(importer)
                exporter.stdin.write(header)
                exporter.stdin.close()
                importer.stdin.write(header)
                for block in iter(lambda: exporter.stdout.read(BLOCK), b''):
                    importer.stdin.write(block)
                importer.stdin.close()
                exporter.stdout.close()
                codes = [child.wait() for child in children]
                if any(codes):
                    raise RuntimeError(f'cache boundary transfer failed: {codes}')
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                        child.wait()

    def validate(self, rank, expected):
        # Hashing a full SSD cache can exceed ordinary control RPC's 30s bound.
        # The existing artifact-transfer timeout is used for this one boundary.
        from src.multinode_runtime import _ARTIFACT_TRANSFER_TIMEOUT_SECONDS
        result = self.runner.command_runner(self.command(rank, 'validate'),
            input_bytes=(json.dumps({'expected': expected}) + '\n').encode(),
            timeout_seconds=_ARTIFACT_TRANSFER_TIMEOUT_SECONDS)
        if result.returncode:
            raise RuntimeError(f'cache coverage validation failed: {result.stderr}')
        return json.loads(result.stdout)

    def release(self, rank, manifest):
        self.request(rank, 'release', {'manifest': manifest})


def main():
    import argparse
    import multiprocessing
    import sys
    from .run_cache import Stage1CachePlan
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('ready', 'export', 'import', 'validate', 'release'))
    parser.add_argument('root', type=Path)
    parser.add_argument('control', type=Path)
    args = parser.parse_args()
    request = json.loads(sys.stdin.buffer.readline())
    if args.operation == 'ready':
        ready = args.control / 'ready.json'
        print(ready.read_text() if ready.exists() else 'null')
        return
    identity = json.loads((args.root / 'identity.json').read_text())
    cache = Stage1RunCache(Stage1CachePlan(identity['run_id'], args.root, identity['helper'],
        identity['reference_hidden'], identity['identity']), multiprocessing.Value('i', 0))
    if args.operation == 'export':
        export_records(cache, request['manifest'], request['names'], sys.stdout.buffer)
    elif args.operation == 'import':
        import_records(cache, request['manifest'], request['names'], sys.stdin.buffer)
    elif args.operation == 'validate':
        print(json.dumps(cache.validate_coverage(request['expected'])))
    elif args.operation == 'release':
        from .cache_lifecycle import publish_release
        validate_manifest(cache, request['manifest'])
        publish_release(args.control, request['manifest'])
        print('null')


if __name__ == '__main__':
    main()

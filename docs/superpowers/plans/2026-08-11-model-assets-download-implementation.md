# ILVR Model and Dataset Asset Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one tested, resumable command that downloads the Qwen2.5-VL 3B base model, the complete CoMT dataset, and the published ILVR evaluation checkpoint into repository-local paths.

**Architecture:** A focused `scripts.model_assets` module defines model asset metadata, snapshot verification, and resumable Hugging Face snapshot acquisition. A thin `scripts.download_assets` module removes online-download blockers, selects assets, delegates CoMT construction to the existing preparation API, and reports actionable retry commands. Launchers consume the resulting local paths while behavioral tests execute copied launchers against fake virtual-environment commands.

**Tech Stack:** Python 3.11, `huggingface_hub` 0.35.3, standard-library `argparse`/`dataclasses`/`json`/`pathlib`/`unittest`, existing CoMT preparation APIs, Bash.

## Global Constraints

- Run every Python command through `./venv/ilvr-venv/bin/python`; never use system Python or `./ilvr-venv`.
- Implement against the completed CoMT tooling on branch `feat/comt-dataset` in `.worktrees/comt-dataset`.
- Bring design commit `0b7aa4f` and this plan commit into that branch before implementation so the branch contains its governing documents.
- Preserve unrelated dirty files in the main worktree and stage only files named by the active task.
- Use the official `Qwen/Qwen2.5-VL-3B-Instruct`, `shuai22/comt`, and `shuai22/comt_ckpt` repositories.
- Store final model artifacts in `models/Qwen2.5-VL-3B-Instruct/` and `models/comt_ckpt/`; store prepared data under `data/`.
- Process `model`, `dataset`, then `checkpoint` by default and support `--only model|dataset|checkpoint|all`.
- Preserve caller-provided `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `HF_TOKEN`, and `HF_ENDPOINT`; remove only `HF_HUB_OFFLINE` before lazily importing Hugging Face download code.
- Never select an unofficial mirror, overwrite a completed asset, delete a partial transfer, or start training/evaluation automatically.
- Treat `shuai22/comt_ckpt` as a separate published 7B-derived comparison artifact; never combine it with the 3B base model.
- Follow strict RED-GREEN-REFACTOR: add one behavior test, observe the expected failure, implement the minimum behavior, and rerun the focused and full suites before each commit.

---

## File Map

- `scripts/model_assets.py`: immutable model asset definitions, index/path/file verification, and resumable snapshot acquisition.
- `scripts/download_assets.py`: unified command-line parsing, online environment setup, ordered selection, CoMT delegation, summaries, and retry errors.
- `tests/test_model_assets.py`: real temporary snapshot fixtures and command orchestration tests with only the network boundary replaced.
- `tests/test_launcher_defaults.py`: executes copied launchers against fake repository-local virtual-environment commands and asserts captured arguments.
- `run_training.sh`: defaults to the downloaded local 3B model while retaining `MODEL_NAME` override and offline runtime behavior.
- `eval.sh`: defaults to the downloaded published checkpoint while retaining `MODEL_DIR` override and offline runtime behavior.
- `.gitignore`: excludes `models/` and downloader metadata/artifacts.
- `README.md`: documents the unified command, partial commands, artifact distinction, paths, storage, environment variables, resume behavior, and training/evaluation commands.

---

### Task 1: Indexed Model Snapshot Verifier

**Files:**
- Create: `scripts/model_assets.py`
- Create: `tests/test_model_assets.py`

**Interfaces:**
- Produces: `SnapshotValidationError(ValueError)`.
- Produces: `ModelAssetSpec(key: str, repo_id: str, directory_name: str, index_filename: str, weight_suffix: str, expected_shards: int)`.
- Produces: `MODEL_SPEC` and `CHECKPOINT_SPEC` constants.
- Produces: `verify_model_snapshot(directory: Path, spec: ModelAssetSpec) -> Path`.
- Consumes: Hugging Face model snapshots with a JSON object at `spec.index_filename` containing a dictionary-valued `weight_map`.

- [ ] **Step 1: Create complete temporary snapshot fixtures and the first failing success test**

Create `tests/test_model_assets.py` with imports and this fixture helper. The helper writes literal metadata and a hand-derived weight map; it does not call production code to calculate the expectation.

```python
import json
import tempfile
import unittest
from pathlib import Path

from scripts.model_assets import MODEL_SPEC, SnapshotValidationError, verify_model_snapshot


REQUIRED_METADATA = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def write_snapshot(root: Path, index_name: str, shard_names: tuple[str, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_METADATA:
        (root / name).write_text("{}\n", encoding="utf-8")
    weight_map = {
        f"layer.{number}.weight": shard
        for number, shard in enumerate(shard_names)
    }
    (root / index_name).write_text(
        json.dumps({"metadata": {"total_size": len(shard_names)}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    for shard in shard_names:
        (root / shard).write_bytes(b"weight")


class ModelSnapshotVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "model"
        self.shards = (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_accepts_complete_indexed_snapshot(self):
        write_snapshot(self.root, "model.safetensors.index.json", self.shards)
        self.assertEqual(verify_model_snapshot(self.root, MODEL_SPEC), self.root)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest \
  tests.test_model_assets.ModelSnapshotVerificationTests.test_accepts_complete_indexed_snapshot -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.model_assets'`.

- [ ] **Step 3: Add minimal public types, constants, and success-path verification**

Create `scripts/model_assets.py` with these definitions and an initial verifier:

```python
"""Download and validate repository-local model snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class SnapshotValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ModelAssetSpec:
    key: str
    repo_id: str
    directory_name: str
    index_filename: str
    weight_suffix: str
    expected_shards: int


MODEL_SPEC = ModelAssetSpec(
    key="model",
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    directory_name="Qwen2.5-VL-3B-Instruct",
    index_filename="model.safetensors.index.json",
    weight_suffix=".safetensors",
    expected_shards=2,
)
CHECKPOINT_SPEC = ModelAssetSpec(
    key="checkpoint",
    repo_id="shuai22/comt_ckpt",
    directory_name="comt_ckpt",
    index_filename="pytorch_model.bin.index.json",
    weight_suffix=".bin",
    expected_shards=7,
)
REQUIRED_METADATA = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def verify_model_snapshot(directory: Path, spec: ModelAssetSpec) -> Path:
    directory = Path(directory)
    for name in (*REQUIRED_METADATA, spec.index_filename):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SnapshotValidationError(f"{spec.key}: missing or empty file: {name}")
    try:
        index = json.loads((directory / spec.index_filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(f"{spec.key}: invalid weight index") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SnapshotValidationError(f"{spec.key}: weight_map must be a non-empty object")
    shards = set(weight_map.values())
    if len(shards) != spec.expected_shards:
        raise SnapshotValidationError(
            f"{spec.key}: expected {spec.expected_shards} indexed shards, found {len(shards)}"
        )
    for shard in shards:
        path = directory / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise SnapshotValidationError(f"{spec.key}: missing or empty shard: {shard}")
    return directory
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest \
  tests.test_model_assets.ModelSnapshotVerificationTests.test_accepts_complete_indexed_snapshot -v
```

Expected: PASS.

- [ ] **Step 5: Add failing containment, emptiness, and exact-index tests**

Add these methods to `ModelSnapshotVerificationTests`:

```python
def test_rejects_missing_indexed_shard(self):
    write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
    (self.root / self.shards[1]).unlink()
    with self.assertRaisesRegex(SnapshotValidationError, "missing or empty shard"):
        verify_model_snapshot(self.root, MODEL_SPEC)

def test_rejects_empty_indexed_shard(self):
    write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
    (self.root / self.shards[0]).write_bytes(b"")
    with self.assertRaisesRegex(SnapshotValidationError, "missing or empty shard"):
        verify_model_snapshot(self.root, MODEL_SPEC)

def test_rejects_missing_required_metadata(self):
    write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
    (self.root / "preprocessor_config.json").unlink()
    with self.assertRaisesRegex(SnapshotValidationError, "missing or empty file"):
        verify_model_snapshot(self.root, MODEL_SPEC)

def test_rejects_parent_traversing_index_path(self):
    write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
    index_path = self.root / MODEL_SPEC.index_filename
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["layer.0.weight"] = "../outside.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with self.assertRaisesRegex(SnapshotValidationError, "unsafe shard path"):
        verify_model_snapshot(self.root, MODEL_SPEC)

def test_rejects_absolute_index_path(self):
    write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
    index_path = self.root / MODEL_SPEC.index_filename
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["layer.0.weight"] = "/tmp/outside.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with self.assertRaisesRegex(SnapshotValidationError, "unsafe shard path"):
        verify_model_snapshot(self.root, MODEL_SPEC)

def test_rejects_unindexed_weight_file(self):
    write_snapshot(self.root, MODEL_SPEC.index_filename, self.shards)
    (self.root / "model-00003-of-00003.safetensors").write_bytes(b"extra")
    with self.assertRaisesRegex(SnapshotValidationError, "unindexed weight files"):
        verify_model_snapshot(self.root, MODEL_SPEC)
```

- [ ] **Step 6: Run the six new tests and verify RED for the intended branches**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest tests.test_model_assets -v
```

Expected: the missing/empty cases PASS with the initial implementation; traversal, absolute-path, and unindexed-file cases FAIL because those protections are absent.

- [ ] **Step 7: Implement safe indexed-path parsing and exact shard-set verification**

Replace the shard loop in `verify_model_snapshot` with a helper and exact local set check:

```python
def _resolve_indexed_shards(
    directory: Path, weight_map: dict[str, object], spec: ModelAssetSpec
) -> set[Path]:
    root = directory.resolve()
    shards: set[Path] = set()
    for value in weight_map.values():
        if not isinstance(value, str):
            raise SnapshotValidationError(f"{spec.key}: shard names must be strings")
        relative = Path(value)
        candidate = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not candidate.is_relative_to(root)
        ):
            raise SnapshotValidationError(f"{spec.key}: unsafe shard path: {value}")
        if candidate.suffix != spec.weight_suffix:
            raise SnapshotValidationError(f"{spec.key}: unexpected shard type: {value}")
        shards.add(candidate)
    return shards
```

After checking `expected_shards`, verify each resolved shard, then compare it with all regular files matching `*{spec.weight_suffix}` below the snapshot root:

```python
indexed_shards = _resolve_indexed_shards(directory, weight_map, spec)
if len(indexed_shards) != spec.expected_shards:
    raise SnapshotValidationError(
        f"{spec.key}: expected {spec.expected_shards} indexed shards, "
        f"found {len(indexed_shards)}"
    )
for shard in indexed_shards:
    if not shard.is_file() or shard.stat().st_size == 0:
        raise SnapshotValidationError(f"{spec.key}: missing or empty shard: {shard.name}")
local_weights = {
    path.resolve()
    for path in directory.rglob(f"*{spec.weight_suffix}")
    if path.is_file()
}
extras = sorted(path.name for path in local_weights - indexed_shards)
if extras:
    raise SnapshotValidationError(
        f"{spec.key}: unindexed weight files: {', '.join(extras)}"
    )
```

- [ ] **Step 8: Run the verifier tests and full existing suite**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest tests.test_model_assets -v
./venv/ilvr-venv/bin/python -m unittest discover -s tests -v
```

Expected: all new verifier tests PASS and the existing 18 CoMT/launcher tests remain PASS.

- [ ] **Step 9: Commit the verifier**

```bash
git add scripts/model_assets.py tests/test_model_assets.py
git commit -m "feat: verify indexed model snapshots"
```

---

### Task 2: Resumable Model Snapshot Acquisition

**Files:**
- Modify: `scripts/model_assets.py`
- Modify: `tests/test_model_assets.py`

**Interfaces:**
- Consumes: `ModelAssetSpec` and `verify_model_snapshot()` from Task 1.
- Produces: `SnapshotFetch = Callable[..., str]` type alias.
- Produces: `ensure_model_snapshot(spec: ModelAssetSpec, models_dir: Path, snapshot_fetch: SnapshotFetch | None = None) -> Path`.
- Calls: `huggingface_hub.snapshot_download(repo_id=spec.repo_id, local_dir=destination)` only when local verification fails.

- [ ] **Step 1: Add failing acquisition and reuse tests**

Add this test class to `tests/test_model_assets.py`. The fake replaces only the external network transfer and writes a real snapshot that the real verifier consumes.

```python
from scripts.model_assets import CHECKPOINT_SPEC, ensure_model_snapshot


class ModelSnapshotAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temporary.name) / "models"

    def tearDown(self):
        self.temporary.cleanup()

    def test_fetches_expected_repository_into_expected_directory(self):
        calls = []

        def fake_fetch(**kwargs):
            calls.append(kwargs)
            destination = Path(kwargs["local_dir"])
            write_snapshot(
                destination,
                MODEL_SPEC.index_filename,
                ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
            )
            return str(destination)

        result = ensure_model_snapshot(MODEL_SPEC, self.models_dir, fake_fetch)

        self.assertEqual(result, self.models_dir / "Qwen2.5-VL-3B-Instruct")
        self.assertEqual(
            calls,
            [{
                "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                "local_dir": self.models_dir / "Qwen2.5-VL-3B-Instruct",
            }],
        )

    def test_reuses_complete_snapshot_without_fetching(self):
        destination = self.models_dir / MODEL_SPEC.directory_name
        write_snapshot(
            destination,
            MODEL_SPEC.index_filename,
            ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
        )

        def forbidden_fetch(**kwargs):
            self.fail(f"unexpected network fetch: {kwargs}")

        self.assertEqual(
            ensure_model_snapshot(MODEL_SPEC, self.models_dir, forbidden_fetch),
            destination,
        )
```

- [ ] **Step 2: Run acquisition tests and verify RED**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest \
  tests.test_model_assets.ModelSnapshotAcquisitionTests -v
```

Expected: FAIL with `ImportError` because `ensure_model_snapshot` does not exist.

- [ ] **Step 3: Implement lazy download and post-download verification**

Add this implementation to `scripts/model_assets.py`:

```python
from collections.abc import Callable
from typing import Any

SnapshotFetch = Callable[..., str]


def _default_snapshot_fetch(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def ensure_model_snapshot(
    spec: ModelAssetSpec,
    models_dir: Path,
    snapshot_fetch: SnapshotFetch | None = None,
) -> Path:
    destination = Path(models_dir) / spec.directory_name
    try:
        return verify_model_snapshot(destination, spec)
    except SnapshotValidationError:
        pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetch = snapshot_fetch or _default_snapshot_fetch
    fetch(repo_id=spec.repo_id, local_dir=destination)
    return verify_model_snapshot(destination, spec)
```

Do not catch errors from `snapshot_download` or the final verifier. The CLI in Task 3 needs the original exception to identify the failed asset, while partial Hub metadata must remain available for resume.

- [ ] **Step 4: Run acquisition and full model-asset tests**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest \
  tests.test_model_assets.ModelSnapshotAcquisitionTests -v
./venv/ilvr-venv/bin/python -m unittest tests.test_model_assets -v
```

Expected: PASS.

- [ ] **Step 5: Commit resumable acquisition**

```bash
git add scripts/model_assets.py tests/test_model_assets.py
git commit -m "feat: resume model snapshot downloads"
```

---

### Task 3: Unified Asset Download Command

**Files:**
- Create: `scripts/download_assets.py`
- Modify: `tests/test_model_assets.py`

**Interfaces:**
- Consumes: `MODEL_SPEC`, `CHECKPOINT_SPEC`, and `ensure_model_snapshot()` from Tasks 1–2.
- Consumes lazily: `scripts.comt_dataset.prepare_dataset(data_dir: Path) -> Path` from the completed CoMT branch.
- Produces: `ASSET_ORDER = ("model", "dataset", "checkpoint")`.
- Produces: `configure_online_environment(environ: MutableMapping[str, str]) -> None`.
- Produces: `download_assets(selection: str, project_root: Path, snapshot_fetch: SnapshotFetch | None = None, dataset_prepare: DatasetPrepare | None = None, output: TextIO = sys.stdout) -> dict[str, Path]`.
- Produces: `build_parser() -> argparse.ArgumentParser` and `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Add failing default-order, subset, and environment tests**

Append these imports and test class to `tests/test_model_assets.py`:

```python
import os
from unittest.mock import patch

from scripts.download_assets import configure_online_environment, download_assets


class UnifiedAssetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_fetch(self, events):
        def fake_fetch(**kwargs):
            repo_id = kwargs["repo_id"]
            destination = Path(kwargs["local_dir"])
            events.append(("snapshot", repo_id))
            if repo_id == MODEL_SPEC.repo_id:
                shards = (
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                )
                write_snapshot(destination, MODEL_SPEC.index_filename, shards)
            else:
                shards = tuple(
                    f"pytorch_model-{number:05d}-of-00007.bin" for number in range(1, 8)
                )
                write_snapshot(destination, CHECKPOINT_SPEC.index_filename, shards)
            return str(destination)

        return fake_fetch

    def test_downloads_all_assets_in_documented_order(self):
        events = []

        def fake_dataset_prepare(data_dir):
            events.append(("dataset", "shuai22/comt"))
            data_dir.mkdir(parents=True, exist_ok=True)
            manifest = data_dir / "comt_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            return manifest

        results = download_assets(
            "all",
            self.project_root,
            snapshot_fetch=self.make_fetch(events),
            dataset_prepare=fake_dataset_prepare,
        )

        self.assertEqual(
            events,
            [
                ("snapshot", "Qwen/Qwen2.5-VL-3B-Instruct"),
                ("dataset", "shuai22/comt"),
                ("snapshot", "shuai22/comt_ckpt"),
            ],
        )
        self.assertEqual(set(results), {"model", "dataset", "checkpoint"})

    def test_only_model_never_prepares_dataset_or_checkpoint(self):
        events = []

        def forbidden_dataset_prepare(data_dir):
            self.fail(f"unexpected dataset preparation: {data_dir}")

        results = download_assets(
            "model",
            self.project_root,
            snapshot_fetch=self.make_fetch(events),
            dataset_prepare=forbidden_dataset_prepare,
        )

        self.assertEqual(events, [("snapshot", "Qwen/Qwen2.5-VL-3B-Instruct")])
        self.assertEqual(set(results), {"model"})

    def test_online_setup_removes_only_offline_flag(self):
        environment = {
            "HF_HUB_OFFLINE": "1",
            "HTTPS_PROXY": "http://proxy.example:8080",
            "HF_TOKEN": "secret-token",
            "HF_ENDPOINT": "https://configured.example",
        }
        configure_online_environment(environment)
        self.assertEqual(
            environment,
            {
                "HTTPS_PROXY": "http://proxy.example:8080",
                "HF_TOKEN": "secret-token",
                "HF_ENDPOINT": "https://configured.example",
            },
        )
```

- [ ] **Step 2: Run the unified-command tests and verify RED**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest \
  tests.test_model_assets.UnifiedAssetDownloadTests -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.download_assets'`.

- [ ] **Step 3: Implement environment setup, selection, and orchestration**

Create `scripts/download_assets.py`:

```python
"""Download all repository-local assets required by ILVR."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import TextIO

from scripts.model_assets import (
    CHECKPOINT_SPEC,
    MODEL_SPEC,
    SnapshotFetch,
    ensure_model_snapshot,
)


ASSET_ORDER = ("model", "dataset", "checkpoint")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DatasetPrepare = Callable[[Path], Path]


def configure_online_environment(environ: MutableMapping[str, str]) -> None:
    environ.pop("HF_HUB_OFFLINE", None)


def _default_dataset_prepare(data_dir: Path) -> Path:
    from scripts.comt_dataset import prepare_dataset

    return prepare_dataset(data_dir)


def download_assets(
    selection: str,
    project_root: Path,
    snapshot_fetch: SnapshotFetch | None = None,
    dataset_prepare: DatasetPrepare | None = None,
    output: TextIO = sys.stdout,
) -> dict[str, Path]:
    selected = ASSET_ORDER if selection == "all" else (selection,)
    models_dir = Path(project_root) / "models"
    prepare = dataset_prepare or _default_dataset_prepare
    results: dict[str, Path] = {}
    for key in selected:
        print(f"[{key}] starting", file=output)
        if key == "model":
            result = ensure_model_snapshot(MODEL_SPEC, models_dir, snapshot_fetch)
        elif key == "dataset":
            result = prepare(Path(project_root) / "data")
        elif key == "checkpoint":
            result = ensure_model_snapshot(CHECKPOINT_SPEC, models_dir, snapshot_fetch)
        else:
            raise ValueError(f"unknown asset selection: {key}")
        results[key] = result
        print(f"[{key}] ready: {result}", file=output)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=(*ASSET_ORDER, "all"), default="all")
    return parser
```

- [ ] **Step 4: Run selection/environment tests and verify GREEN**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest \
  tests.test_model_assets.UnifiedAssetDownloadTests -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Add a failing active-asset error test**

Add this method to `UnifiedAssetDownloadTests`:

```python
def test_tags_the_asset_that_failed_during_an_all_run(self):
    from scripts.download_assets import AssetDownloadError

    events = []

    def failing_dataset_prepare(data_dir):
        events.append(("dataset", "shuai22/comt"))
        raise OSError("dataset endpoint timed out")

    with self.assertRaises(AssetDownloadError) as raised:
        download_assets(
            "all",
            self.project_root,
            snapshot_fetch=self.make_fetch(events),
            dataset_prepare=failing_dataset_prepare,
        )

    self.assertEqual(raised.exception.key, "dataset")
    self.assertEqual(str(raised.exception), "dataset endpoint timed out")
```

- [ ] **Step 6: Run the active-asset test and verify RED**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest \
  tests.test_model_assets.UnifiedAssetDownloadTests.test_tags_the_asset_that_failed_during_an_all_run -v
```

Expected: FAIL with `ImportError` because `AssetDownloadError` does not exist.

- [ ] **Step 7: Add active-key error context around each selected asset**

Define this exception above `download_assets`:

```python
class AssetDownloadError(RuntimeError):
    def __init__(self, key: str, cause: Exception):
        super().__init__(str(cause))
        self.key = key
```

Wrap the existing per-key body without changing selection behavior:

```python
for key in selected:
    print(f"[{key}] starting", file=output)
    try:
        if key == "model":
            result = ensure_model_snapshot(MODEL_SPEC, models_dir, snapshot_fetch)
        elif key == "dataset":
            result = prepare(Path(project_root) / "data")
        elif key == "checkpoint":
            result = ensure_model_snapshot(CHECKPOINT_SPEC, models_dir, snapshot_fetch)
        else:
            raise ValueError(f"unknown asset selection: {key}")
    except Exception as exc:
        raise AssetDownloadError(key, exc) from exc
    results[key] = result
    print(f"[{key}] ready: {result}", file=output)
```

Run the Step 6 command. Expected: PASS.

- [ ] **Step 8: Add a failing executable-CLI retry test**

Add `import io` and `import sys` to the test module imports, then add:

```python
def test_main_reports_single_asset_retry_after_failure(self):
    from scripts import download_assets as command

    error = io.StringIO()
    failure = command.AssetDownloadError(
        "checkpoint", OSError("connection timed out")
    )
    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False), patch.object(
        command, "download_assets", side_effect=failure
    ), patch.object(sys, "stderr", error):
        status = command.main(["--only", "checkpoint"])
        self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    self.assertEqual(status, 1)
    self.assertIn("checkpoint download failed: connection timed out", error.getvalue())
    self.assertIn(
        "./venv/ilvr-venv/bin/python -m scripts.download_assets --only checkpoint",
        error.getvalue(),
    )
```

The patched function is the already-tested orchestration boundary; assertions remain on the real CLI's exit status and stderr behavior, not on mock calls.

- [ ] **Step 9: Run the CLI test and verify RED**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest \
  tests.test_model_assets.UnifiedAssetDownloadTests.test_main_reports_single_asset_retry_after_failure -v
```

Expected: FAIL with `AttributeError` because `main` does not exist.

- [ ] **Step 10: Implement the executable entry point and retry diagnostics**

Append:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_online_environment(os.environ)
    try:
        download_assets(args.only, PROJECT_ROOT)
    except AssetDownloadError as exc:
        print(f"{exc.key} download failed: {exc}", file=sys.stderr)
        print(
            "Retry: ./venv/ilvr-venv/bin/python -m scripts.download_assets "
            f"--only {exc.key}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run the Step 9 command. Expected: PASS.

- [ ] **Step 11: Run focused and full tests, then commit**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest tests.test_model_assets -v
./venv/ilvr-venv/bin/python -m unittest discover -s tests -v
./venv/ilvr-venv/bin/python -m py_compile scripts/model_assets.py scripts/download_assets.py
```

Expected: all tests PASS and compilation exits 0.

Commit:

```bash
git add scripts/download_assets.py tests/test_model_assets.py
git commit -m "feat: download all ILVR assets"
```

---

### Task 4: Local 3B Training and Published-Checkpoint Evaluation Defaults

**Files:**
- Modify: `tests/test_launcher_defaults.py`
- Modify: `run_training.sh`
- Modify: `eval.sh`

**Interfaces:**
- `run_training.sh` consumes optional `MODEL_NAME`; default is `${REPO_ROOT}/models/Qwen2.5-VL-3B-Instruct`.
- `eval.sh` consumes optional `MODEL_DIR`; default is `${REPO_ROOT}/models/comt_ckpt`.
- Both launchers continue to execute tools from `${REPO_ROOT}/venv/ilvr-venv/bin` and export `HF_HUB_OFFLINE=1` only at training/evaluation runtime.

- [ ] **Step 1: Replace source-text launcher checks with failing behavioral harness tests**

In `tests/test_launcher_defaults.py`, retain `unittest` and create a helper that copies one launcher into a temporary repository, creates fake executable commands under `venv/ilvr-venv/bin`, and records their arguments through `CAPTURE_FILE`:

```python
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LauncherDefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temporary.name)
        self.bin_dir = self.sandbox / "venv/ilvr-venv/bin"
        self.bin_dir.mkdir(parents=True)
        (self.sandbox / "data/images_comt").mkdir(parents=True)
        (self.sandbox / "data/TRAIN.jsonl").write_text("{}\n", encoding="utf-8")
        (self.sandbox / "data/TEST.jsonl").write_text("{}\n", encoding="utf-8")
        (self.sandbox / "models/Qwen2.5-VL-3B-Instruct").mkdir(parents=True)
        (self.sandbox / "models/comt_ckpt").mkdir(parents=True)
        self.capture = self.sandbox / "arguments.txt"

    def tearDown(self):
        self.temporary.cleanup()

    def write_executable(self, path: Path, body: str) -> None:
        path.write_text("#!/bin/bash\nset -eu\n" + body, encoding="utf-8")
        path.chmod(0o755)

    def run_launcher(
        self, name: str, overrides: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        shutil.copy2(self.repo / name, self.sandbox / name)
        environment = os.environ.copy()
        environment.pop("MODEL_NAME", None)
        environment.pop("MODEL_DIR", None)
        environment["CAPTURE_FILE"] = str(self.capture)
        environment.update(overrides or {})
        return subprocess.run(
            ["bash", str(self.sandbox / name)],
            cwd=self.sandbox,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_training_launcher_passes_local_3b_model_and_data(self):
        self.write_executable(self.bin_dir / "python", "echo 1\n")
        self.write_executable(
            self.bin_dir / "accelerate",
            'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
        )

        result = self.run_launcher("run_training.sh")

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.capture.read_text(encoding="utf-8").splitlines()
        model_index = arguments.index("--model")
        data_index = arguments.index("--data_path")
        image_index = arguments.index("--image-root")
        self.assertEqual(
            arguments[model_index + 1],
            str(self.sandbox / "models/Qwen2.5-VL-3B-Instruct"),
        )
        self.assertEqual(arguments[data_index + 1], str(self.sandbox / "data/TRAIN.jsonl"))
        self.assertEqual(arguments[image_index + 1], str(self.sandbox / "data/images_comt"))

    def test_evaluation_launcher_passes_published_checkpoint_and_test_data(self):
        self.write_executable(
            self.bin_dir / "python",
            'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
        )

        result = self.run_launcher("eval.sh")

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.capture.read_text(encoding="utf-8").splitlines()
        model_index = arguments.index("--model_dir")
        data_index = arguments.index("--test_data_path")
        self.assertEqual(arguments[model_index + 1], str(self.sandbox / "models/comt_ckpt"))
        self.assertEqual(arguments[data_index + 1], str(self.sandbox / "data/TEST.jsonl"))
```

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest tests.test_launcher_defaults -v
```

Expected: training test FAIL because the launcher passes the 7B Hub ID; evaluation test FAIL because empty `MODEL_DIR` is rejected.

- [ ] **Step 3: Implement environment-overridable local model defaults**

In `run_training.sh`, replace the fixed model line with:

```bash
MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
```

In `eval.sh`, replace the empty default with:

```bash
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/comt_ckpt}"
```

Retain each launcher's existing directory/file checks, repository-local virtual environment, explicit image root, and offline runtime export.

- [ ] **Step 4: Add override behavior tests**

The Step 1 helper already accepts an optional environment mapping. Add these exact tests:

```python
def test_training_launcher_honors_model_override(self):
    self.write_executable(self.bin_dir / "python", "echo 1\n")
    self.write_executable(
        self.bin_dir / "accelerate",
        'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
    )

    result = self.run_launcher(
        "run_training.sh", {"MODEL_NAME": "/custom/three-billion"}
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    arguments = self.capture.read_text(encoding="utf-8").splitlines()
    self.assertEqual(arguments[arguments.index("--model") + 1], "/custom/three-billion")

def test_evaluation_launcher_honors_model_override(self):
    custom_model = self.sandbox / "custom-checkpoint"
    custom_model.mkdir()
    self.write_executable(
        self.bin_dir / "python",
        'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
    )

    result = self.run_launcher("eval.sh", {"MODEL_DIR": str(custom_model)})

    self.assertEqual(result.returncode, 0, result.stderr)
    arguments = self.capture.read_text(encoding="utf-8").splitlines()
    self.assertEqual(arguments[arguments.index("--model_dir") + 1], str(custom_model))
```

- [ ] **Step 5: Run launcher tests, syntax checks, and full suite**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest tests.test_launcher_defaults -v
bash -n run_training.sh eval.sh
./venv/ilvr-venv/bin/python -m unittest discover -s tests -v
```

Expected: all tests PASS and Bash syntax checks exit 0.

- [ ] **Step 6: Commit launcher defaults**

```bash
git add tests/test_launcher_defaults.py run_training.sh eval.sh
git commit -m "feat: use downloaded model assets by default"
```

---

### Task 5: Ignore Rules, Documentation, and Whole-Feature Verification

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Test: all `tests/` modules

**Interfaces:**
- Documents: `./venv/ilvr-venv/bin/python -m scripts.download_assets` and all four explicit `--only` values.
- Documents: `MODEL_NAME`, `MODEL_DIR`, `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `HF_TOKEN`, and `HF_ENDPOINT` overrides.
- Documents: the 3B base model and 7B-derived published checkpoint as distinct artifacts.

- [ ] **Step 1: Add local model artifacts to ignore rules**

Append this exact repository-local directory rule to `.gitignore`:

```gitignore
/models/
```

Keep the existing `/data/`, `/venv/`, `/ilvr-venv/`, and `/.worktrees/` rules.

- [ ] **Step 2: Replace manual data download instructions with the unified command**

Replace the manual download subsection with this exact content:

````markdown
### Download All Required Assets

Run the unified downloader from the repository root. It uses the mandated
virtual environment, resumes interrupted Hugging Face transfers, and validates
each artifact before reporting it ready.

```bash
./venv/ilvr-venv/bin/python -m scripts.download_assets
```

| Artifact | Local path | Approximate size |
| --- | --- | ---: |
| Qwen2.5-VL 3B experiment base | `models/Qwen2.5-VL-3B-Instruct/` | 7.52 GB |
| Complete prepared CoMT dataset | `data/` | 11.5 GB |
| Published ILVR comparison checkpoint | `models/comt_ckpt/` | 31.8 GB |

Allow at least 51 GB for final payloads plus additional temporary/resume
metadata. `models/comt_ckpt` is the project's published 7B-derived comparison
checkpoint. It is not an adapter and must not be combined with the 3B base
model.
````

- [ ] **Step 3: Document selective retry and connectivity configuration**

Immediately after Step 2's table, add:

````markdown
Download or retry one asset without touching completed assets:

```bash
./venv/ilvr-venv/bin/python -m scripts.download_assets --only model
./venv/ilvr-venv/bin/python -m scripts.download_assets --only dataset
./venv/ilvr-venv/bin/python -m scripts.download_assets --only checkpoint
```

The downloader removes `HF_HUB_OFFLINE` only inside its own process. It honors
caller-provided `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `HF_TOKEN`, and
`HF_ENDPOINT` values; without `HF_ENDPOINT`, it uses the official Hugging Face
service. It never selects a third-party mirror automatically. Partial Hub
downloads and `data/.staging/` are retained so the same command can resume.
````

- [ ] **Step 4: Document training and evaluation defaults**

Replace the README's 7B default statement and current evaluation setup with:

````markdown
### Run the 3B Experiment

Training defaults to the downloaded `models/Qwen2.5-VL-3B-Instruct` base,
`data/TRAIN.jsonl`, and `data/images_comt`:

```bash
bash run_training.sh
```

Override the base model without editing the launcher:

```bash
MODEL_NAME=/path/to/another-base-model bash run_training.sh
```

### Evaluate

Evaluation defaults to the published `models/comt_ckpt` comparison checkpoint
and `data/TEST.jsonl`:

```bash
bash eval.sh
```

Evaluate a newly trained 3B checkpoint with:

```bash
MODEL_DIR=/path/to/trained-3b-checkpoint bash eval.sh
```
````

- [ ] **Step 5: Run whole-feature verification from the mandated environment**

Run:

```bash
./venv/ilvr-venv/bin/python -m unittest discover -s tests -v
./venv/ilvr-venv/bin/python -m py_compile \
  scripts/comt_dataset.py scripts/prepare_comt.py \
  scripts/model_assets.py scripts/download_assets.py
bash -n run_training.sh eval.sh
./venv/ilvr-venv/bin/python -m scripts.download_assets --help
git diff --check
git status --short
```

Expected: all tests PASS; compilation, Bash syntax, help, and whitespace checks exit 0; status contains only the files intended by this task.

- [ ] **Step 6: Commit documentation and ignore rules**

```bash
git add .gitignore README.md
git commit -m "docs: explain unified asset downloads"
```

- [ ] **Step 7: Perform final requirement and mutation review**

Manually map each design requirement to a passing test or documented runtime behavior. Mentally mutate the model repository ID, selection order, local directory, shard count, offline-variable removal, launcher default, and retry asset key; confirm at least one automated test fails for each mutation. Confirm `git diff main...HEAD --check` exits 0 and inspect `git diff --stat main...HEAD` for unrelated files.

---

### Task 6: Authorized Live Download Attempt

**Files:**
- Runtime artifacts only: `models/`, `data/`, and their resumable metadata.
- No source files are created or committed by this task.

**Interfaces:**
- Consumes: the completed unified command from Task 3.
- Produces when connectivity permits: verified 3B model, validated CoMT data, and verified published checkpoint at the documented paths.

- [ ] **Step 1: Confirm sufficient repository filesystem capacity**

Run:

```bash
df -h .
```

Expected: substantially more than 51 GB is available, with additional room for dataset staging and resume metadata. Stop without downloading if free capacity is below 80 GB.

- [ ] **Step 2: Run the complete downloader from the repository root**

Run:

```bash
./venv/ilvr-venv/bin/python -m scripts.download_assets
```

Allow Hugging Face resume metadata to remain if the command is interrupted or the endpoint is unreachable. Do not delete partial files.

- [ ] **Step 3: Verify all artifacts without loading weights into GPU memory**

Run:

```bash
./venv/ilvr-venv/bin/python - <<'PY'
from pathlib import Path

from scripts.comt_dataset import validate_dataset
from scripts.model_assets import CHECKPOINT_SPEC, MODEL_SPEC, verify_model_snapshot

root = Path.cwd()
print(verify_model_snapshot(root / "models" / MODEL_SPEC.directory_name, MODEL_SPEC))
print(validate_dataset(root / "data/TRAIN.jsonl", root / "data/TEST.jsonl", root / "data/images_comt"))
print(verify_model_snapshot(root / "models" / CHECKPOINT_SPEC.directory_name, CHECKPOINT_SPEC))
PY
```

Expected when the endpoint is reachable: two verified model paths and a dataset report with zero missing/corrupt images. If connectivity still fails, report the exact failed asset and retry command from the downloader; the source implementation remains complete, while runtime acquisition remains externally blocked.

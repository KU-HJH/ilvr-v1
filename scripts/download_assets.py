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


class AssetDownloadError(RuntimeError):
    def __init__(self, key: str, cause: Exception):
        super().__init__(str(cause))
        self.key = key


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
        try:
            if key == "model":
                result = ensure_model_snapshot(MODEL_SPEC, models_dir, snapshot_fetch)
            elif key == "dataset":
                result = prepare(Path(project_root) / "data")
            elif key == "checkpoint":
                result = ensure_model_snapshot(
                    CHECKPOINT_SPEC, models_dir, snapshot_fetch
                )
            else:
                raise ValueError(f"unknown asset selection: {key}")
        except Exception as exc:
            raise AssetDownloadError(key, exc) from exc
        results[key] = result
        print(f"[{key}] ready: {result}", file=output)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=(*ASSET_ORDER, "all"), default="all")
    return parser


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

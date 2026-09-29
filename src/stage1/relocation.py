"""Prepare one historical Stage 1 adapter for an explicit base relocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from artifact_paths import require_canonical_artifact_path
from stage1.checkpointing import (
    decode_stage1_peft_metadata_snapshot,
    read_stage1_peft_metadata_snapshot,
)


_COPIED_FILES = (
    "adapter_model.safetensors",
    "ilvr_token_rows.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
)


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing or linked: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_file_hashes(source_base: Path, local_base: Path) -> dict[str, str]:
    def files(root: Path) -> dict[str, Path]:
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"base model directory is missing or linked: {root}")
        found = {}
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"base model contains a non-regular file: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"base model contains a non-regular file: {path}")
            found[str(path.relative_to(root))] = path
        return found

    original = files(source_base)
    local = files(local_base)
    if (
        original.keys() != local.keys()
        or "config.json" not in original
        or not any(name.endswith(".safetensors") for name in original)
    ):
        raise ValueError("base model contents differ: required files or file sets")
    hashes = {}
    for name in sorted(original):
        if original[name].stat().st_size != local[name].stat().st_size:
            raise ValueError(f"base model contents differ: {name}")
        original_hash = _sha256(original[name])
        if original_hash != _sha256(local[name]):
            raise ValueError(f"base model contents differ: {name}")
        hashes[name] = original_hash
    return hashes


def prepare_original_stage1_initializer(
    source_artifact: Path,
    source_base: Path,
    local_base: Path,
    destination: Path,
) -> dict[str, object]:
    """Copy validated initializer files after proving base trees byte-identical."""
    source_artifact = Path(source_artifact).resolve(strict=True)
    source_base = Path(source_base).resolve(strict=True)
    local_base = Path(local_base).resolve(strict=True)
    destination = require_canonical_artifact_path(destination, kind="checkpoint")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    snapshot = read_stage1_peft_metadata_snapshot(source_artifact)
    metadata = decode_stage1_peft_metadata_snapshot(snapshot)
    if metadata.schema_version != 3:
        raise ValueError("historical Stage 1 initializer requires schema 3 metadata")
    if Path(metadata.base_model_reference).name != source_base.name:
        raise ValueError("source base name disagrees with historical metadata")
    config_path = source_artifact / "adapter_config.json"
    _regular_file(config_path)
    config = json.loads(config_path.read_bytes())
    if config.get("base_model_name_or_path") != metadata.base_model_reference:
        raise ValueError("Stage 1 adapter base reference mismatch")
    for name in ("adapter_model.safetensors", "ilvr_token_rows.safetensors"):
        _regular_file(source_artifact / name)
    base_hashes = _base_file_hashes(source_base, local_base)
    local_reference = str(local_base)
    relocated_metadata = metadata.to_dict()
    relocated_metadata["base_model_reference"] = local_reference
    relocated_config = dict(config)
    relocated_config["base_model_name_or_path"] = local_reference
    manifest = {
        "source_artifact": str(source_artifact),
        "source_base": str(source_base),
        "local_base": local_reference,
        "original_base_model_reference": metadata.base_model_reference,
        "source_metadata_sha256": hashlib.sha256(snapshot.metadata_json).hexdigest(),
        "original_stage1_metadata": metadata.to_dict(),
        "base_file_sha256": base_hashes,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".stage1-import-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary)
        for name in _COPIED_FILES:
            source = source_artifact / name
            if not source.exists():
                continue
            _regular_file(source)
            shutil.copyfile(source, staged / name)
            if _sha256(source) != _sha256(staged / name):
                raise ValueError(f"copied Stage 1 file changed: {name}")
        for name, payload in (
            ("stage1_peft_metadata.json", relocated_metadata),
            ("adapter_config.json", relocated_config),
            ("original_stage1_provenance.json", manifest),
        ):
            (staged / name).write_text(json.dumps(payload, sort_keys=True) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        staged.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--source-base", type=Path, required=True)
    parser.add_argument("--local-base", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_original_stage1_initializer(
        args.source_artifact, args.source_base, args.local_base, args.destination
    )
    print(
        json.dumps(
            {
                "destination": str(args.destination),
                "source_metadata_sha256": manifest["source_metadata_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Copy selected physical JSONL rows in caller-requested order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def select_jsonl_rows(
    input_path: Path, output_path: Path, indices: tuple[int, ...]
) -> None:
    """Publish selected line bytes exclusively, normalizing terminators to one LF."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError(f"output path already exists: {output_path}")
    if not isinstance(indices, tuple) or not indices:
        raise ValueError("indices must be a nonempty tuple")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("indices must be nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must not contain duplicates")

    lines = input_path.read_bytes().splitlines()
    if any(index >= len(lines) for index in indices):
        raise ValueError(f"row index is out of range for {len(lines)} input lines")
    payload = b"".join(lines[index] + b"\n" for index in indices)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise ValueError(f"output path already exists: {output_path}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--index", type=int, action="append", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        select_jsonl_rows(args.input_path, args.output_path, tuple(args.index))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

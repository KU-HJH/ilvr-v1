"""Command-line preparation and validation for the CoMT dataset."""

import argparse
import dataclasses
import json
from pathlib import Path

from scripts.comt_dataset import SOURCE_REPO, prepare_dataset, validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare", help="download, validate, and promote CoMT data")
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare.add_argument("--repo-id", default=SOURCE_REPO)
    prepare.add_argument("--revision")
    prepare.add_argument("--force", action="store_true")

    validate = subcommands.add_parser("validate", help="validate an existing CoMT dataset")
    validate.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        print(prepare_dataset(args.data_dir, args.repo_id, args.revision, args.force))
        return 0
    report = validate_dataset(
        args.data_dir / "TRAIN.jsonl",
        args.data_dir / "TEST.jsonl",
        args.data_dir / "images_comt",
    )
    print(json.dumps(dataclasses.asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

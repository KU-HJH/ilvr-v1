"""Offline evaluation scoring command."""

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import sys

import yaml

from evaluation.io import parse_config, read_inputs
from evaluation.registry import get_scorer
from evaluation.report import evaluate
from evaluation.schema import ContractError, Score, canonical_json


def write_reports(output_dir: Path, scores: list[Score], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "scores.jsonl").open("xb") as handle:
        for score in scores:
            handle.write(canonical_json(asdict(score)) + b"\n")
    with (output_dir / "summary.json").open("xb") as handle:
        handle.write(canonical_json(summary) + b"\n")


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    parser = argparse.ArgumentParser()
    for option in ("manifest", "predictions", "config", "output-dir"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)

    root = (Path.cwd() if repo_root is None else Path(repo_root)).resolve()
    resolve = lambda value: (root / value).resolve()
    out = resolve(args.output_dir)
    output_parent = (root / "results/evaluation").resolve()
    log_parent = (root / "logs/evaluation").resolve()
    log_dir = (log_parent / out.name).resolve()
    if (
        not out.name
        or out.parent != output_parent
        or log_dir.parent != log_parent
        or out.exists()
        or log_dir.exists()
    ):
        print("invalid or existing output/log directory", file=sys.stderr)
        return 2

    try:
        log_dir.mkdir(parents=True, exist_ok=False)
        handler = logging.FileHandler(log_dir / "score.log", mode="x", encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = logging.getLogger("evaluation.offline." + out.name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        try:
            raw = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
            config = parse_config(raw, root)
            get_scorer(config.benchmark, config.profile)
            prepared = read_inputs(
                resolve(args.manifest), resolve(args.predictions), config, root
            )
        except (ContractError, OSError, UnicodeError, yaml.YAMLError) as exc:
            logger.error("input error: %s", exc)
            print(str(exc), file=sys.stderr)
            return 2
        try:
            scores, summary = evaluate(prepared, config)
            write_reports(out, scores, summary)
        except Exception:
            logger.exception("scoring/output infrastructure failure")
            return 1
        logger.info(
            "complete=%s expected=%s observed=%s",
            summary["complete"], summary["N_expected"], summary["observed"],
        )
        print(str(out / "summary.json"))
        return 0 if summary["complete"] else 1
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())

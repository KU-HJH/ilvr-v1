"""Prepare a pinned evaluation dataset."""

import argparse
import logging
from pathlib import Path
import sys

from evaluation.registry import get_scorer
from evaluation.schema import ContractError, canonical_json
from evaluation.vsp_prepare import prepare_vsp, verify_source_reference


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    parser = argparse.ArgumentParser()
    for option in ("benchmark", "profile", "source-root", "output-dir"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--selection-manifest")
    parser.add_argument("--image-root")
    args = parser.parse_args(argv)

    root = (Path.cwd() if repo_root is None else Path(repo_root)).resolve()
    source_root = (root / args.source_root).resolve()
    output = (root / args.output_dir).resolve()
    data_parent = (root / "data/evaluation").resolve()
    result_parent = (root / "results/evaluation").resolve()
    log_parent = (root / "logs/evaluation").resolve()
    result = (result_parent / output.name).resolve()
    log = (log_parent / output.name).resolve()
    if (
        not output.name
        or output.parent != data_parent
        or result.parent != result_parent
        or log.parent != log_parent
        or output.exists()
        or result.exists()
        or log.exists()
    ):
        print("invalid or existing dataset/result/log directory", file=sys.stderr)
        return 2
    try:
        get_scorer(args.benchmark, args.profile)
    except ContractError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.benchmark == "comt":
        if args.selection_manifest is None or args.image_root is None:
            print(
                "CoMT preparation requires --selection-manifest and --image-root",
                file=sys.stderr,
            )
            return 2
    elif args.selection_manifest is not None or args.image_root is not None:
        kind = {"vsp": "VSP", "visulogic": "VisuLogic", "emma": "EMMA", "zebra_2d": "Zebra"}.get(args.benchmark, args.benchmark)
        print(f"{kind} preparation does not accept CoMT-only paths", file=sys.stderr)
        return 2

    try:
        log.mkdir(parents=True, exist_ok=False)
        handler = logging.FileHandler(log / "prepare.log", mode="x", encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    logger = logging.getLogger("evaluation.prepare." + output.name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        try:
            if args.benchmark == "comt":
                from evaluation.comt_prepare import prepare_comt

                summary = prepare_comt(
                    source_root,
                    (root / args.selection_manifest).resolve(),
                    (root / args.image_root).resolve(),
                    output,
                    repo_root=root,
                )
            elif args.benchmark == "visulogic":
                from evaluation.visulogic_prepare import prepare_visulogic

                summary = prepare_visulogic(source_root, output, repo_root=root)
            elif args.benchmark == "emma":
                from evaluation.emma_prepare import prepare_emma

                summary = prepare_emma(source_root, output, repo_root=root)
            elif args.benchmark == "zebra_2d":
                from evaluation.zebra_2d_prepare import prepare_zebra_2d

                summary = prepare_zebra_2d(source_root, output, repo_root=root)
            else:
                identity = verify_source_reference(source_root, root)
                summary = prepare_vsp(source_root, output, source_identity=identity)
        except ContractError as exc:
            logger.error("input error: %s", exc)
            print(str(exc), file=sys.stderr)
            return 2
        except Exception:
            logger.exception("preparation infrastructure failure")
            return 1
        try:
            result.mkdir(parents=True, exist_ok=False)
            (result / "prepare.json").write_bytes(canonical_json(summary) + b"\n")
        except Exception:
            logger.exception("preparation output infrastructure failure")
            return 1
        logger.info("complete expected=%s", summary["N_expected"])
        print(str(result / "prepare.json"))
        return 0
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())

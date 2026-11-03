"""Sequential VSP generation command with lightweight input preflight."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import sys

import yaml

from evaluation.inference_bridge import (
    generate_one,
    load_runtime,
    runtime_metadata,
    validate_generation_config,
    validate_runtime_paths,
    validate_vsp_inputs,
)
from evaluation.io import parse_config, read_manifest
from evaluation.registry import get_scorer
from evaluation.schema import ContractError, Prediction, canonical_json


def _absolute_without_resolving(value: str | Path, root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.absolute()


def _append_record(handle, record: dict) -> None:
    handle.write(canonical_json(record) + b"\n")
    handle.flush()


def _write_run(path: Path, record: dict) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json(record) + b"\n")
        handle.flush()


def _error_text(exc: BaseException) -> str:
    text = str(exc)
    return text if text.strip() else type(exc).__name__


def _mark_failed(run_path: Path, record: dict, exc: BaseException) -> None:
    record["status"] = "failed"
    record["error"] = _error_text(exc)
    _write_run(run_path, record)


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    parser = argparse.ArgumentParser()
    for option in ("manifest", "config", "model-dir", "model-identity", "output-dir"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--base-model-dir")
    parser.add_argument("--cache-dir", default="data/cache/evaluation")
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
        handler = logging.FileHandler(log_dir / "run.log", mode="x", encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = logging.getLogger("evaluation.generation." + out.name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        try:
            raw = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
            config = parse_config(raw, root)
            get_scorer(config.benchmark, config.profile)
            if args.model_identity != config.run_metadata.model_identity:
                raise ContractError("model_identity does not match config")
            validate_generation_config(config.run_metadata.generation_config)
            prepared = read_manifest(resolve(args.manifest), config, root)
            validate_vsp_inputs(prepared, config)
            model_dir, base_model_dir, cache_dir = validate_runtime_paths(
                Path(args.model_dir),
                Path(args.base_model_dir) if args.base_model_dir is not None else None,
                Path(args.cache_dir),
                root,
            )
            loader_base_model_dir = (
                _absolute_without_resolving(args.base_model_dir, root)
                if args.base_model_dir is not None
                else None
            )
        except (ContractError, OSError, UnicodeError, yaml.YAMLError) as exc:
            logger.error("input error: %s", exc)
            print(str(exc), file=sys.stderr)
            return 2

        predictions_path = out / "predictions.jsonl"
        details_path = out / "generation-details.jsonl"
        run_path = out / "run.json"
        run_record = {
            "schema_version": 1,
            "status": "running",
            "protocol_kind": "modified",
            "manifest_sha256": prepared.manifest_sha256,
            "run_fingerprint": prepared.run_fingerprint,
            "N_expected": len(prepared.samples),
            "processed": 0,
            "run_metadata": asdict(config.run_metadata),
            "paths": {
                "manifest": str(resolve(args.manifest)),
                "config": str(resolve(args.config)),
                "model_dir": str(model_dir),
                "base_model_dir": str(base_model_dir) if base_model_dir is not None else None,
                "output_dir": str(out),
                "log_dir": str(log_dir),
                "predictions": str(predictions_path),
                "generation_details": str(details_path),
            },
            "cache_dir": str(cache_dir),
        }
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            out.mkdir(parents=True, exist_ok=False)
            predictions_path.open("xb").close()
            details_path.open("xb").close()
            _write_run(run_path, run_record)
        except Exception as exc:
            logger.exception("generation output initialization failure")
            print(_error_text(exc), file=sys.stderr)
            return 1

        try:
            runtime = load_runtime()
            model, processor = runtime.load_evaluation_artifacts(
                model_dir=str(model_dir),
                base_model_dir=(
                    str(loader_base_model_dir)
                    if loader_base_model_dir is not None
                    else None
                ),
                cache_dir=str(cache_dir),
            )
            loaded_metadata = runtime_metadata(
                model,
                processor,
                runtime,
                model_dir,
                base_model_dir,
                config.run_metadata.generation_config,
            )
            run_record["runtime_metadata"] = loaded_metadata
        except Exception as exc:
            logger.exception("runtime load failure")
            try:
                _mark_failed(run_path, run_record, exc)
            except Exception:
                logger.exception("failed to update run record after runtime load failure")
            return 1

        try:
            with details_path.open("ab") as details_handle, predictions_path.open(
                "ab"
            ) as predictions_handle:
                for sample in prepared.samples:
                    try:
                        output = generate_one(
                            model,
                            processor,
                            sample.model_input,
                            config.dataset_root,
                            config.run_metadata.generation_config,
                            run_one_example=runtime.run_one_example,
                        )
                    except Exception as exc:
                        error = _error_text(exc)
                        details = {
                            "schema_version": 1,
                            "id": sample.id,
                            "manifest_sha256": prepared.manifest_sha256,
                            "run_fingerprint": prepared.run_fingerprint,
                            "status": "infrastructure_error",
                            "error": error,
                            "generated_suffix_token_ids": None,
                            "prompt_identity": None,
                            "prompt_token_length": None,
                            "stop_reason": None,
                            "latent_start_count": None,
                            "latent_block_count": None,
                            "inference_time": None,
                            "raw_output": None,
                        }
                        prediction = Prediction(
                            sample.id,
                            prepared.manifest_sha256,
                            prepared.run_fingerprint,
                            None,
                            "infrastructure_error",
                            error=error,
                        )
                        _append_record(details_handle, details)
                        _append_record(predictions_handle, asdict(prediction))
                        run_record["processed"] += 1
                        _mark_failed(run_path, run_record, exc)
                        logger.exception("sample inference failure: %s", sample.id)
                        return 1

                    details = {
                        "schema_version": 1,
                        "id": sample.id,
                        "manifest_sha256": prepared.manifest_sha256,
                        "run_fingerprint": prepared.run_fingerprint,
                        "status": "ok",
                        "error": None,
                        "generated_suffix_token_ids": output[
                            "generated_suffix_token_ids"
                        ],
                        "prompt_identity": output["prompt_identity"],
                        "prompt_token_length": output["prompt_token_length"],
                        "stop_reason": output["stop_reason"],
                        "latent_start_count": output["latent_start_count"],
                        "latent_block_count": output["latent_block_count"],
                        "inference_time": output["inference_time"],
                        "raw_output": output["raw_output"],
                    }
                    prediction = Prediction(
                        sample.id,
                        prepared.manifest_sha256,
                        prepared.run_fingerprint,
                        output["assistant_only"],
                        "ok",
                        termination_reason=output["stop_reason"],
                        prompt_identity=output["prompt_identity"],
                    )
                    _append_record(details_handle, details)
                    _append_record(predictions_handle, asdict(prediction))
                    run_record["processed"] += 1
                    _write_run(run_path, run_record)
        except Exception as exc:
            logger.exception("generation output failure")
            try:
                _mark_failed(run_path, run_record, exc)
            except Exception:
                logger.exception("failed to update run record after output failure")
            return 1

        run_record["status"] = "complete"
        run_record.pop("error", None)
        try:
            _write_run(run_path, run_record)
        except Exception:
            logger.exception("generation completion record failure")
            return 1
        logger.info(
            "complete expected=%s processed=%s",
            run_record["N_expected"],
            run_record["processed"],
        )
        print(str(predictions_path))
        return 0
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())

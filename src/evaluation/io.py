from dataclasses import fields
from pathlib import Path
import hashlib
import json
import re

from evaluation.benchmarks.vsp import normalize_map
from evaluation.schema import (
    ContractError,
    EvalConfig,
    EmmaTarget,
    ModelInput,
    McqTarget,
    Prediction,
    PreparedEvaluation,
    Provenance,
    RunMetadata,
    Sample,
    VspTarget,
    ZebraTarget,
    canonical_json,
    run_fingerprint,
)


def exact_keys(raw, required, optional=()):
    if (
        not isinstance(raw, dict)
        or not set(required) <= set(raw)
        or set(raw) - set(required) - set(optional)
    ):
        raise ContractError("unexpected or missing fields")


def nonempty(value):
    if not isinstance(value, str) or not value.strip():
        raise ContractError("expected nonempty string")
    return value


def sha256(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ContractError("expected SHA-256")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ContractError("non-finite JSON number: " + value)


def read_jsonl_bytes(payload: bytes) -> list[dict]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ContractError("JSONL is not valid UTF-8") from exc
    result = []
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except ContractError:
            raise
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid JSON at row {number}") from exc
        if not isinstance(row, dict):
            raise ContractError(f"row {number} is not an object")
        result.append(row)
    return result


def _field_names(cls) -> tuple[str, ...]:
    return tuple(field.name for field in fields(cls))


def _exact_int(value, message: str) -> int:
    if type(value) is not int:
        raise ContractError(message)
    return value


def _sequence(value, name: str) -> list:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be a list")
    return value


def _resolve_dataset_root(value, repo_root: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ContractError("dataset_root must be a path")
    try:
        repository = Path(repo_root).resolve()
        data_root = (repository / "data").resolve()
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = repository / candidate
        candidate = candidate.resolve()
        candidate.relative_to(data_root)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError("dataset_root must resolve inside repo_root/data") from exc
    if not candidate.is_dir():
        raise ContractError("dataset_root must be an existing directory")
    return candidate


def _resolve_input_path(value, repo_root: Path, storage: str, kind: str) -> Path:
    try:
        repository = Path(repo_root).resolve()
        storage_root = (repository / storage).resolve()
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = repository / candidate
        candidate = candidate.resolve()
        candidate.relative_to(storage_root)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"{kind} path must resolve inside repo_root/{storage}") from exc
    return candidate


def parse_config(raw: dict, repo_root: Path) -> EvalConfig:
    try:
        config_fields = _field_names(EvalConfig)
        exact_keys(
            raw,
            tuple(name for name in config_fields if name != "max_steps"),
            ("max_steps",),
        )
        version = _exact_int(raw["schema_version"], "schema_version must be integer 1")
        if version != 1:
            raise ContractError("schema_version must be integer 1")
        benchmark = nonempty(raw["benchmark"])
        profile = nonempty(raw["profile"])
        categories_raw = _sequence(raw["categories"], "categories")
        categories = tuple(nonempty(value) for value in categories_raw)
        if not categories or len(set(categories)) != len(categories):
            raise ContractError("categories must be nonempty and unique")
        if benchmark == "comt":
            from evaluation.benchmarks.comt import CATEGORIES as COMT_CATEGORIES

            if not set(categories) <= set(COMT_CATEGORIES):
                raise ContractError("CoMT categories must be a canonical nonempty subset")
            if raw.get("max_steps") is not None:
                raise ContractError("CoMT max_steps must be null or omitted")
            max_steps = None
        elif benchmark == "visulogic":
            from evaluation.benchmarks.visulogic import CATEGORIES as VISULOGIC_CATEGORIES

            if not set(categories) <= set(VISULOGIC_CATEGORIES):
                raise ContractError("VisuLogic categories must be a canonical nonempty subset")
            if raw.get("max_steps") is not None:
                raise ContractError("VisuLogic max_steps must be null or omitted")
            max_steps = None
        elif benchmark == "emma":
            from evaluation.benchmarks.emma import CATEGORIES as EMMA_CATEGORIES

            if not set(categories) <= set(EMMA_CATEGORIES):
                raise ContractError("EMMA categories must be a canonical nonempty subset")
            if raw.get("max_steps") is not None:
                raise ContractError("EMMA max_steps must be null or omitted")
            max_steps = None
        elif benchmark == "zebra_2d":
            from evaluation.benchmarks.zebra_2d import CATEGORIES as ZEBRA_CATEGORIES

            if not set(categories) <= set(ZEBRA_CATEGORIES):
                raise ContractError("Zebra categories must be a canonical nonempty subset")
            if raw.get("max_steps") is not None:
                raise ContractError("Zebra max_steps must be null or omitted")
            max_steps = None
        else:
            if "max_steps" not in raw:
                raise ContractError("max_steps must be a positive integer")
            max_steps = _exact_int(
                raw["max_steps"], "max_steps must be a positive integer"
            )
            if max_steps <= 0:
                raise ContractError("max_steps must be a positive integer")
        dataset_root = _resolve_dataset_root(raw["dataset_root"], repo_root)

        run_raw = raw["run_metadata"]
        exact_keys(run_raw, _field_names(RunMetadata))
        model_identity = nonempty(run_raw["model_identity"])
        prompt_identity = nonempty(run_raw["prompt_template_identity"])
        generation_config = run_raw["generation_config"]
        if not isinstance(generation_config, dict):
            raise ContractError("generation_config must be an object")
        canonical_json(generation_config)
        run = RunMetadata(model_identity, prompt_identity, generation_config)
        return EvalConfig(
            version,
            benchmark,
            profile,
            categories,
            dataset_root,
            max_steps,
            run,
        )
    except ContractError:
        raise
    except (TypeError, ValueError, OSError, UnicodeError) as exc:
        raise ContractError("invalid evaluation config") from exc


def parse_sample(raw: dict) -> Sample:
    try:
        exact_keys(raw, _field_names(Sample))
        version = _exact_int(raw["schema_version"], "schema_version must be integer 1")
        if version != 1:
            raise ContractError("schema_version must be integer 1")
        sample_id = nonempty(raw["id"])
        benchmark = nonempty(raw["benchmark"])
        subtask = nonempty(raw["subtask"])

        input_raw = raw["model_input"]
        exact_keys(input_raw, _field_names(ModelInput))
        text = input_raw["text"]
        if not isinstance(text, str):
            raise ContractError("model input text must be a string")
        images_raw = _sequence(input_raw["images"], "images")
        images = []
        for image in images_raw:
            if not isinstance(image, str) or not image.strip() or Path(image).is_absolute():
                raise ContractError("image paths must be nonempty relative strings")
            images.append(image)
        choices_raw = _sequence(input_raw["choices"], "choices")
        if any(not isinstance(choice, str) for choice in choices_raw):
            raise ContractError("choices must contain only strings")
        model_input = ModelInput(text, tuple(images), tuple(choices_raw))

        target_raw = raw["target"]
        if benchmark == "vsp":
            exact_keys(target_raw, _field_names(VspTarget))
            map_raw = _sequence(target_raw["map_desc"], "map_desc")
            map_desc = tuple(tuple(_sequence(row, "map row")) for row in map_raw)
            max_steps = _exact_int(
                target_raw["max_steps"], "max_steps must be a positive integer"
            )
            target = VspTarget(map_desc, max_steps)
            normalize_map(target)
        elif benchmark == "comt":
            from evaluation.benchmarks.comt import validate_target

            exact_keys(target_raw, _field_names(McqTarget))
            target = McqTarget(
                _exact_int(target_raw["index"], "CoMT target index must be an integer"),
                target_raw["label"],
                target_raw["content"],
            )
            validate_target(target, model_input)
        elif benchmark == "visulogic":
            from evaluation.benchmarks.visulogic import validate_target

            exact_keys(target_raw, _field_names(McqTarget))
            target = McqTarget(
                _exact_int(target_raw["index"], "VisuLogic target index must be an integer"),
                target_raw["label"], target_raw["content"],
            )
            validate_target(target, model_input)
        elif benchmark == "emma":
            from evaluation.benchmarks.emma import validate_target

            exact_keys(target_raw, _field_names(EmmaTarget))
            index = target_raw["index"]
            if index is not None:
                index = _exact_int(index, "EMMA target index must be an integer or null")
            target = EmmaTarget(
                target_raw["question_type"], target_raw["answer"], index,
                target_raw["content"], target_raw["category"], target_raw["task"],
            )
            validate_target(target, model_input)
        elif benchmark == "zebra_2d":
            from evaluation.benchmarks.zebra_2d import validate_target

            exact_keys(target_raw, _field_names(ZebraTarget))
            target = ZebraTarget(target_raw["answer"])
            validate_target(target, model_input, subtask)
        else:
            raise ContractError("unsupported sample benchmark")

        provenance_raw = raw["provenance"]
        exact_keys(provenance_raw, _field_names(Provenance))
        asset_hashes = provenance_raw["asset_sha256"]
        if not isinstance(asset_hashes, dict):
            raise ContractError("asset_sha256 must be an object")
        if set(asset_hashes) != set(images):
            raise ContractError("asset hashes must exactly match model input images")
        checked_hashes = {}
        for name, digest in asset_hashes.items():
            if not isinstance(name, str):
                raise ContractError("asset names must be strings")
            checked_hashes[name] = sha256(digest)
        provenance = Provenance(
            nonempty(provenance_raw["dataset_id"]),
            nonempty(provenance_raw["revision"]),
            nonempty(provenance_raw["config"]),
            nonempty(provenance_raw["split"]),
            nonempty(provenance_raw["source_id"]),
            checked_hashes,
        )
        return Sample(
            version,
            sample_id,
            benchmark,
            subtask,
            model_input,
            target,
            provenance,
        )
    except ContractError:
        raise
    except (TypeError, ValueError, OSError, UnicodeError) as exc:
        raise ContractError("invalid sample") from exc


def parse_prediction(raw: dict) -> Prediction:
    try:
        required = ("id", "manifest_sha256", "run_fingerprint", "response", "status")
        optional = ("error", "termination_reason", "prompt_identity")
        exact_keys(raw, required, optional)
        prediction_id = nonempty(raw["id"])
        manifest_hash = sha256(raw["manifest_sha256"])
        fingerprint = sha256(raw["run_fingerprint"])
        status = raw["status"]
        response = raw["response"]
        error = raw.get("error")
        termination_reason = raw.get("termination_reason")
        prompt_identity = raw.get("prompt_identity")
        if status not in ("ok", "infrastructure_error"):
            raise ContractError("unknown prediction status")
        if status == "ok":
            if not isinstance(response, str) or error is not None:
                raise ContractError("ok prediction requires response and null error")
        elif (response is not None and not isinstance(response, str)) or not isinstance(
            error, str
        ) or not error.strip():
            raise ContractError("infrastructure error requires nonempty error")
        for value in (termination_reason, prompt_identity):
            if value is not None and not isinstance(value, str):
                raise ContractError("optional prediction metadata must be strings or null")
        return Prediction(
            prediction_id,
            manifest_hash,
            fingerprint,
            response,
            status,
            error,
            termination_reason,
            prompt_identity,
        )
    except ContractError:
        raise
    except (TypeError, ValueError, OSError, UnicodeError) as exc:
        raise ContractError("invalid prediction") from exc


def _read_bytes(path: Path, kind: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read {kind}") from exc


def _validate_assets(sample: Sample, dataset_root: Path) -> None:
    for name in sample.model_input.images:
        relative = Path(name)
        if relative.is_absolute():
            raise ContractError("asset path must stay inside dataset_root")
        try:
            asset = (dataset_root / relative).resolve()
            asset.relative_to(dataset_root)
        except (OSError, ValueError) as exc:
            raise ContractError("asset path must stay inside dataset_root") from exc
        if not asset.is_file():
            raise ContractError("asset must be a readable regular file")
        payload = _read_bytes(asset, "asset")
        if hashlib.sha256(payload).hexdigest() != sample.provenance.asset_sha256[name]:
            raise ContractError("asset SHA-256 mismatch")


def read_manifest(
    manifest_path: Path,
    config: EvalConfig,
    repo_root: Path,
) -> PreparedEvaluation:
    dataset_root = _resolve_dataset_root(config.dataset_root, repo_root)
    manifest_path = _resolve_input_path(manifest_path, repo_root, "data", "manifest")
    manifest_payload = _read_bytes(manifest_path, "manifest")
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    manifest_rows = read_jsonl_bytes(manifest_payload)
    if not manifest_rows:
        raise ContractError("manifest must contain at least one row")

    samples = []
    sample_ids = set()
    for raw in manifest_rows:
        sample = parse_sample(raw)
        expected_id = (sample.provenance.source_id if sample.benchmark == "emma" else
                       f"{sample.benchmark}/{sample.subtask}/{sample.provenance.source_id}")
        if sample.id != expected_id:
            raise ContractError("sample id does not match provenance identity")
        if sample.benchmark != config.benchmark:
            raise ContractError("sample benchmark does not match config")
        if sample.subtask not in config.categories:
            raise ContractError("sample subtask is not configured")
        if sample.benchmark == "vsp" and sample.target.max_steps != config.max_steps:
            raise ContractError("sample max_steps does not match config")
        if sample.id in sample_ids:
            raise ContractError("duplicate sample id")
        _validate_assets(sample, dataset_root)
        sample_ids.add(sample.id)
        samples.append(sample)

    return PreparedEvaluation(
        tuple(samples), {}, manifest_hash, run_fingerprint(config.run_metadata)
    )


def read_inputs(
    manifest_path: Path,
    predictions_path: Path,
    config: EvalConfig,
    repo_root: Path,
) -> PreparedEvaluation:
    if config.benchmark == "emma":
        from evaluation.emma_math import require_emma_math_runtime

        require_emma_math_runtime()
    _resolve_dataset_root(config.dataset_root, repo_root)
    manifest_path = _resolve_input_path(manifest_path, repo_root, "data", "manifest")
    predictions_path = _resolve_input_path(
        predictions_path, repo_root, "results", "predictions"
    )
    prepared = read_manifest(manifest_path, config, repo_root)
    prediction_payload = _read_bytes(predictions_path, "predictions")
    prediction_rows = read_jsonl_bytes(prediction_payload)
    sample_ids = {sample.id for sample in prepared.samples}
    predictions = {}
    for raw in prediction_rows:
        prediction = parse_prediction(raw)
        if prediction.id in predictions:
            raise ContractError("duplicate prediction id")
        if prediction.id not in sample_ids:
            raise ContractError("prediction id is not in manifest")
        if prediction.manifest_sha256 != prepared.manifest_sha256:
            raise ContractError("prediction manifest SHA-256 mismatch")
        if prediction.run_fingerprint != prepared.run_fingerprint:
            raise ContractError("prediction run fingerprint mismatch")
        predictions[prediction.id] = prediction

    return PreparedEvaluation(
        prepared.samples,
        predictions,
        prepared.manifest_sha256,
        prepared.run_fingerprint,
    )

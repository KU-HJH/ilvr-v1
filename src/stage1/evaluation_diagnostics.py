"""Bounded Stage 1 evaluation evidence and recurrence diagnostics.

The pure reducers in this module deliberately have no model-loading side effects.
The two model probes load their artifacts only after their CLI contract and output
location have been validated.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

_platform_python_version = platform.python_version
if _platform_python_version().endswith("+"):
    platform.python_version = lambda: _platform_python_version().removesuffix("+")

try:
    import torch
    import torch.nn.functional as F

    from evaluation.continuous_latent import (
        ContinuousLatentController,
        ContinuousLatentGenerationConfig,
    )
    from evaluation.runtime import prepare_evaluation_input
    from evaluation.scoring import EvaluationScoreSummary, score_answer
    from stage2.checkpointing import resolve_latent_token_ids
finally:
    platform.python_version = _platform_python_version


RELATIVE_L2_LIMIT = 1e-6
COSINE_LIMIT = 0.999999
BOUNDARY_SOURCE_INDICES = (0, 1, 2, 7)
BOUNDARY_BUDGETS = (4096, 8192)


def _require_exact_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_finite_tensor(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(value) and not torch.is_complex(value):
        raise TypeError(f"{name} must be floating-point")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _require_discrete_tensor(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
        raise ValueError(f"{name} must contain exact integer values")
    return value


def _finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def symmetric_relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return ``2 ||left-right|| / (||left|| + ||right||)`` without NaN/Inf."""
    left = _require_finite_tensor(left, name="left")
    right = _require_finite_tensor(right, name="right")
    if left.shape != right.shape:
        raise ValueError("symmetric relative L2 requires matching tensor shapes")
    left64 = left.detach().to(dtype=torch.float64).reshape(-1)
    right64 = right.detach().to(dtype=torch.float64).reshape(-1)
    denominator = torch.linalg.vector_norm(left64) + torch.linalg.vector_norm(right64)
    if denominator.item() == 0.0:
        return 0.0
    return _finite_float(
        2.0 * torch.linalg.vector_norm(left64 - right64).item() / denominator.item(),
        name="symmetric relative L2",
    )


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return a finite cosine similarity, with equal all-zero vectors as one."""
    left = _require_finite_tensor(left, name="left")
    right = _require_finite_tensor(right, name="right")
    if left.shape != right.shape:
        raise ValueError("cosine similarity requires matching tensor shapes")
    left64 = left.detach().to(dtype=torch.float64).reshape(-1)
    right64 = right.detach().to(dtype=torch.float64).reshape(-1)
    left_norm = torch.linalg.vector_norm(left64).item()
    right_norm = torch.linalg.vector_norm(right64).item()
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for one zero vector")
    return _finite_float(
        torch.dot(left64, right64).item() / (left_norm * right_norm),
        name="cosine similarity",
    )


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    """Return the one-based rank of an exact integer token ID (largest is rank 1)."""
    logits = _require_finite_tensor(logits, name="logits")
    token_id = _require_exact_int(token_id, name="token_id")
    if logits.ndim != 1:
        raise ValueError("token rank requires one-dimensional logits")
    if not 0 <= token_id < logits.numel():
        raise ValueError("token_id is out of range for logits")
    return int((logits > logits[token_id]).sum().item()) + 1


def _json_safe(value: object) -> object:
    """Reject NaN/Inf and convert torch scalar/tensor evidence to JSON values."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise ValueError("diagnostic evidence must contain only finite values")
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return _finite_float(value, name="diagnostic evidence")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"diagnostic evidence is not JSON-safe: {type(value).__name__}")


def _summary_mapping(summary: EvaluationScoreSummary) -> dict[str, object]:
    return _json_safe(asdict(summary))  # type: ignore[return-value]


def rescore_historical_rows(
    historical_rows: Sequence[Mapping[str, object]], dataset_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Rescore legacy assistant-only rows after a strict integer-index join."""
    if not dataset_rows:
        raise ValueError("dataset rows must not be empty")
    by_index: dict[int, Mapping[str, object]] = {}
    for row in historical_rows:
        if not isinstance(row, Mapping):
            raise TypeError("historical rows must be mappings")
        index = _require_exact_int(row.get("index"), name="historical index")
        if index in by_index:
            raise ValueError(f"duplicate historical index: {index}")
        if not 0 <= index < len(dataset_rows):
            raise ValueError(f"historical index out of range: {index}")
        by_index[index] = row
    missing = sorted(set(range(len(dataset_rows))) - set(by_index))
    if missing:
        raise ValueError(f"missing historical indices: {missing}")

    rescored_rows: list[dict[str, object]] = []
    changes: list[dict[str, object]] = []
    for index in range(len(dataset_rows)):
        old = by_index[index]
        sample = dataset_rows[index]
        if not isinstance(sample, Mapping):
            raise TypeError(f"dataset row {index} must be a mapping")
        prediction = old.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"historical row {index} requires string prediction")
        old_match = old.get("match")
        if not isinstance(old_match, bool):
            raise ValueError(f"historical row {index} requires boolean match")
        score = score_answer(prediction, sample.get("original_final_answer"))
        new_match = bool(score.match)
        row = {
            "index": index,
            "prediction": prediction,
            "old_match": old_match,
            "new_match": new_match,
            "evidence_status": "legacy_assistant_only",
            "generated_suffix_token_ids": None,
            "stop_reason": None,
            "latent_evidence": None,
            "prediction_parse_status": score.prediction.status,
            "gold_parse_status": score.gold.status,
            "invalid_gold_reason": score.invalid_gold_reason,
        }
        rescored_rows.append(row)
        if old_match != new_match:
            changes.append({"index": index, "old_match": old_match, "new_match": new_match})

    new_summary_rows = [
        {
            "index": row["index"],
            "match": row["new_match"],
            "invalid_gold_reason": row["invalid_gold_reason"],
        }
        for row in rescored_rows
    ]
    old_summary_rows = [
        {
            "index": row["index"],
            "match": row["old_match"],
            "invalid_gold_reason": row["invalid_gold_reason"],
        }
        for row in rescored_rows
    ]
    old_summary = EvaluationScoreSummary.from_rows(old_summary_rows, len(dataset_rows))
    new_summary = EvaluationScoreSummary.from_rows(new_summary_rows, len(dataset_rows))
    return {
        "source_count": len(dataset_rows),
        "old_correct": old_summary.correct,
        "new_correct": new_summary.correct,
        "old_score": _summary_mapping(old_summary),
        "new_score": _summary_mapping(new_summary),
        "legacy_raw_id_limitation": "unavailable",
        "rows": rescored_rows,
        "match_changes": changes,
    }


def _evidence_tensor(evidence: Mapping[str, object], name: str, *, discrete: bool) -> torch.Tensor:
    if name not in evidence:
        raise ValueError(f"forced-prefix evidence is missing {name}")
    return (
        _require_discrete_tensor(evidence[name], name=name)
        if discrete
        else _require_finite_tensor(evidence[name], name=name)
    )


def _evidence_states(evidence: Mapping[str, object], name: str) -> tuple[torch.Tensor, ...]:
    states = evidence.get(name)
    if not isinstance(states, (list, tuple)):
        raise ValueError(f"forced-prefix evidence {name} must be a sequence")
    return tuple(_require_finite_tensor(state, name=f"{name}[{index}]") for index, state in enumerate(states))


def compare_forced_prefix(
    reference: Mapping[str, object], candidate: Mapping[str, object], *, require_pass: bool = False
) -> dict[str, object]:
    """Compare recurrence evidence using exact discrete and pinned numeric gates."""
    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        raise TypeError("forced-prefix evidence must be mappings")
    discrete_match = True
    for name in ("sequences", "attention_mask", "position_ids"):
        expected = _evidence_tensor(reference, name, discrete=True)
        observed = _evidence_tensor(candidate, name, discrete=True)
        discrete_match = discrete_match and expected.shape == observed.shape and bool(torch.equal(expected, observed))
    reference_states = _evidence_states(reference, "latent_states")
    candidate_states = _evidence_states(candidate, "latent_states")
    reference_logits = _evidence_tensor(reference, "post_end_logits", discrete=False)
    candidate_logits = _evidence_tensor(candidate, "post_end_logits", discrete=False)
    if len(reference_states) != len(candidate_states):
        discrete_match = False

    l2_values: list[float] = []
    cosine_values: list[float] = []
    for expected, observed in zip(reference_states, candidate_states, strict=False):
        if expected.shape != observed.shape:
            discrete_match = False
            continue
        l2_values.append(symmetric_relative_l2(expected, observed))
        cosine_values.append(cosine_similarity(expected, observed))
    if reference_logits.shape != candidate_logits.shape:
        discrete_match = False
    else:
        l2_values.append(symmetric_relative_l2(reference_logits, candidate_logits))
        cosine_values.append(cosine_similarity(reference_logits, candidate_logits))
    max_l2 = max(l2_values, default=0.0)
    min_cosine = min(cosine_values, default=1.0)
    passed = discrete_match and max_l2 <= RELATIVE_L2_LIMIT and min_cosine >= COSINE_LIMIT
    result = {
        "pass": bool(passed),
        "discrete_match": bool(discrete_match),
        "latent_state_count": len(reference_states),
        "candidate_latent_state_count": len(candidate_states),
        "max_symmetric_relative_l2": _finite_float(max_l2, name="maximum L2"),
        "min_cosine_similarity": _finite_float(min_cosine, name="minimum cosine"),
        "relative_l2_limit": RELATIVE_L2_LIMIT,
        "cosine_limit": COSINE_LIMIT,
    }
    if require_pass and not passed:
        raise ValueError("forced-prefix recurrence failed discrete or threshold gate")
    return result


def parameter_delta_summary(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    """Summarize finite post-step deltas for exactly the observed parameters."""
    if set(before) != set(after):
        raise ValueError("parameter snapshots must contain identical names")
    parameters: dict[str, dict[str, object]] = {}
    for name in sorted(before):
        initial = _require_finite_tensor(before[name], name=f"before[{name}]")
        final = _require_finite_tensor(after[name], name=f"after[{name}]")
        if initial.shape != final.shape:
            raise ValueError(f"parameter snapshot shape changed: {name}")
        delta = final.detach().to(torch.float64) - initial.detach().to(torch.float64)
        parameters[name] = {
            "changed_elements": int(torch.count_nonzero(delta).item()),
            "l2_delta": _finite_float(torch.linalg.vector_norm(delta.reshape(-1)).item(), name=f"l2 delta {name}"),
            "symmetric_relative_l2": symmetric_relative_l2(initial, final),
        }
    return {"parameter_count": len(parameters), "parameters": parameters}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            if row.get("summary") is True:
                continue
            rows.append(row)
    return rows


def _require_results_output_path(value: str) -> Path:
    output = Path(value).expanduser().resolve()
    results_root = (Path.cwd() / "results").resolve()
    try:
        relative = output.relative_to(results_root)
    except ValueError as error:
        raise ValueError("output-json must be beneath results/<run-id>/") from error
    if len(relative.parts) < 2 or not relative.parts[0] or output.suffix.lower() != ".json":
        raise ValueError("output-json must be a JSON file beneath results/<run-id>/")
    return output


def derive_execution_log_path(output_json: str | Path) -> Path:
    """Derive the only permitted log destination from a results run directory."""
    output = Path(output_json).expanduser().resolve()
    results_root = (Path.cwd() / "results").resolve()
    try:
        relative = output.relative_to(results_root)
    except ValueError as error:
        raise ValueError("output-json must be beneath results/<run-id>/") from error
    if len(relative.parts) < 2:
        raise ValueError("output-json must be beneath results/<run-id>/")
    return (
        Path.cwd() / "logs" / relative.parent / f"{output.stem}.log"
    ).resolve()


def _write_json(output: Path, evidence: Mapping[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(evidence), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_execution_log(path: Path, *, mode: str, output: Path, status: str, detail: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join((
            f"mode={mode}",
            f"output_json={output}",
            f"status={status}",
            f"detail={detail}",
            "",
        )),
        encoding="utf-8",
    )


def _load_probe_runtime(model_dir: str, base_model_dir: str, cache_dir: str):
    """Load the existing Stage 1 artifact through its established evaluator loader."""
    from eval import load_evaluation_artifacts

    model, processor = load_evaluation_artifacts(
        model_dir=model_dir, base_model_dir=base_model_dir, cache_dir=cache_dir
    )
    qwen = getattr(model, "get_base_model", lambda: model)()
    # The established loader returns a PEFT wrapper for Stage 1; its base Qwen
    # is what owns the recurrence interface.
    if not hasattr(qwen, "config") or getattr(qwen.config, "stage", None) != "stage1":
        from peft_common import resolve_qwen2_5_vl_model

        qwen = resolve_qwen2_5_vl_model(model)
    if getattr(qwen.config, "stage", None) != "stage1":
        raise ValueError("diagnostics require a Stage 1 artifact")
    qwen.eval()
    return qwen, processor


def _images_for_sample(sample: Mapping[str, object], dataset_root: Path):
    from eval import open_images
    from multinode_evaluation import resolve_evaluation_image_paths

    paths = resolve_evaluation_image_paths(sample.get("image_input", []), dataset_root)
    return open_images(paths) if paths else None


def _model_device(model) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    raise ValueError("diagnostic model has no parameters")


def _ordinary_token_id(tokenizer, token_ids) -> int:
    for text in (" the", "a", "."):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            candidate = _require_exact_int(int(ids[0]), name="ordinary token id")
            if candidate not in {token_ids.pad_token_id, token_ids.start_token_id, token_ids.end_token_id}:
                return candidate
    raise ValueError("tokenizer did not provide an ordinary text token")


class _ForcedPrefixHead(torch.nn.Module):
    """Use the real head while forcing only the requested text-prefix IDs."""

    def __init__(self, original, forced_ids: Sequence[int]) -> None:
        super().__init__()
        self.original = original
        self.forced_ids = tuple(_require_exact_int(token_id, name="forced token id") for token_id in forced_ids)
        self.calls = 0

    def forward(self, hidden):
        logits = self.original(hidden)
        if self.calls < len(self.forced_ids):
            token_id = self.forced_ids[self.calls]
            if not 0 <= token_id < logits.shape[-1]:
                raise ValueError("forced token id is outside the language-model vocabulary")
            forced = torch.full_like(logits, -torch.inf)
            forced[..., token_id] = 0.0
            logits = forced
        self.calls += 1
        return logits


def _run_production_forced_prefix(model, prepared, token_ids, forced_ids):
    """Run the production controller once; the reference implementation never calls it."""
    original_head = model.lm_head
    model.lm_head = _ForcedPrefixHead(original_head, forced_ids)
    try:
        generated = ContinuousLatentController().generate(
            model=model,
            input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"],
            token_ids=token_ids,
            generation_config=ContinuousLatentGenerationConfig(
                max_new_tokens=len(forced_ids) + model.config.latent_size + 1,
                eos_token_id=None,
            ),
            pixel_values=prepared.get("pixel_values"),
            image_grid_thw=prepared.get("image_grid_thw"),
            return_latent_states=True,
            return_post_end_logits=True,
        )
    finally:
        model.lm_head = original_head
    mask = torch.cat(
        (prepared["attention_mask"], torch.ones_like(generated.sequences[:, prepared["input_ids"].shape[1] :])), dim=1
    )
    positions, _ = model.get_rope_index(generated.sequences, image_grid_thw=prepared.get("image_grid_thw"), attention_mask=mask)
    return {
        "sequences": generated.sequences,
        "attention_mask": mask,
        "position_ids": positions,
        "latent_states": tuple(state for block in generated.latent_states_by_block for state in block),
        "post_end_logits": generated.post_end_logits_by_block[-1],
    }


@torch.no_grad()
def independent_no_cache_forced_prefix(model, prepared, token_ids, forced_ids, *, static_feedback: bool = False):
    """Independent literal recurrence; it intentionally does not call the controller."""
    sequences = prepared["input_ids"].clone()
    attention_mask = prepared["attention_mask"].clone()
    inputs_embeds = model.prepare_stage2_base_inputs_embeds(
        input_ids=sequences,
        pixel_values=prepared.get("pixel_values"),
        image_grid_thw=prepared.get("image_grid_thw"),
    )
    latent_states: list[torch.Tensor] = []
    post_end_logits = None
    pad_embedding = model.get_input_embeddings()(
        torch.tensor([token_ids.pad_token_id], device=sequences.device)
    ).detach()
    for forced_id in forced_ids:
        forced_id = _require_exact_int(forced_id, name="forced token id")
        append = torch.tensor([[forced_id]], dtype=sequences.dtype, device=sequences.device)
        sequences = torch.cat((sequences, append), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(append)), dim=1)
        inputs_embeds = torch.cat((inputs_embeds, model.get_input_embeddings()(append)), dim=1)
        if forced_id != token_ids.start_token_id:
            continue
        suffix = torch.tensor(
            [[token_ids.pad_token_id] * model.config.latent_size + [token_ids.end_token_id]],
            dtype=sequences.dtype,
            device=sequences.device,
        )
        sequences = torch.cat((sequences, suffix), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(suffix)), dim=1)
        inputs_embeds = torch.cat((inputs_embeds, model.get_input_embeddings()(suffix)), dim=1)
        positions, _ = model.get_rope_index(sequences, image_grid_thw=prepared.get("image_grid_thw"), attention_mask=attention_mask)
        first_latent = sequences.shape[1] - model.config.latent_size - 1
        for position in range(first_latent, first_latent + model.config.latent_size):
            prefix_outputs = model.model(
                inputs_embeds=inputs_embeds[:, :position],
                attention_mask=attention_mask[:, :position],
                position_ids=positions[:, :, :position],
                use_cache=False,
                return_dict=True,
                generate_mode=True,
            )
            state = prefix_outputs.last_hidden_state[:, -1, :]
            latent_states.append(state)
            feedback = pad_embedding if static_feedback else state
            inputs_embeds = torch.cat((inputs_embeds[:, :position], feedback.unsqueeze(1), inputs_embeds[:, position + 1 :]), dim=1)
        full_outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
            generate_mode=True,
        )
        post_end_logits = model.lm_head(full_outputs.last_hidden_state[:, -1, :])
    if post_end_logits is None:
        raise ValueError("forced prefix must include latent_start")
    final_positions, _ = model.get_rope_index(sequences, image_grid_thw=prepared.get("image_grid_thw"), attention_mask=attention_mask)
    return {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "position_ids": final_positions,
        "latent_states": tuple(latent_states),
        "post_end_logits": post_end_logits,
    }


def _prepared_for_sample(model, processor, sample, dataset_root: Path):
    images = _images_for_sample(sample, dataset_root)
    prepared = prepare_evaluation_input(processor, sample, images)
    device = _model_device(model)
    tensors = {name: tensor.to(device) for name, tensor in prepared.tensors.items()}
    tensors.setdefault("attention_mask", torch.ones_like(tensors["input_ids"]))
    return tensors


def _prepared_for_index(model, processor, dataset_rows, dataset_root: Path, index: int):
    return _prepared_for_sample(model, processor, dataset_rows[index], dataset_root)


def run_recurrence_probe(*, model_dir: str, base_model_dir: str, dataset_jsonl: str, cache_dir: str) -> dict[str, object]:
    model, processor = _load_probe_runtime(model_dir, base_model_dir, cache_dir)
    dataset_path = Path(dataset_jsonl).resolve(strict=True)
    dataset = _read_jsonl(dataset_path)
    if not dataset:
        raise ValueError("dataset-jsonl must not be empty")
    prepared = _prepared_for_index(model, processor, dataset, dataset_path.parent, 0)
    token_ids = resolve_latent_token_ids(processor.tokenizer, model.config)
    ordinary = _ordinary_token_id(processor.tokenizer, token_ids)
    cases = {
        "latent_immediately": (token_ids.start_token_id,),
        "text_then_latent": (ordinary, token_ids.start_token_id),
    }
    reports: dict[str, object] = {}
    for name, forced_ids in cases.items():
        production = _run_production_forced_prefix(model, prepared, token_ids, forced_ids)
        reference = independent_no_cache_forced_prefix(model, prepared, token_ids, forced_ids)
        control = independent_no_cache_forced_prefix(
            model, prepared, token_ids, forced_ids, static_feedback=True
        )
        reports[name] = {
            "forced_prefix_ids": list(forced_ids),
            "production_vs_reference": compare_forced_prefix(reference, production, require_pass=True),
            "static_latent_pad_control": {
                "threshold_enforced": False,
                "comparison": compare_forced_prefix(reference, control),
            },
        }
    return {"diagnostic": "stage1-forced-prefix-recurrence", "cases": reports}


def validated_test_source_rows(
    dataset_path: Path, dataset_rows: Sequence[Mapping[str, object]], source_indices: Sequence[int]
) -> tuple[dict[str, object], ...]:
    """Bind the fixed boundary IDs to the actual TEST ordering, never offsets alone."""
    canonical_test_path = (Path.cwd() / "data" / "TEST.jsonl").resolve()
    if dataset_path.resolve() != canonical_test_path:
        raise ValueError("boundary-probe requires the canonical data/TEST.jsonl split")
    indexed: dict[int, Mapping[str, object]] = {}
    for source_index, row in enumerate(dataset_rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"TEST row {source_index} must be a mapping")
        declared_index = row.get("index", source_index)
        if _require_exact_int(declared_index, name="TEST row source index") != source_index:
            raise ValueError(
                f"TEST row source index mismatch at row {source_index}: {declared_index!r}"
            )
        indexed[source_index] = row
    requested = tuple(source_indices)
    if len(requested) != len(set(requested)):
        raise ValueError("boundary source indices must be unique")
    missing = [index for index in requested if index not in indexed]
    if missing:
        raise ValueError(f"boundary TEST source indices are missing: {missing}")
    return tuple(
        {"source_index": index, "sample": indexed[index]}
        for index in requested
    )


def _assistant_cue_patterns(tokenizer) -> tuple[tuple[int, ...], ...]:
    patterns = []
    for cue in ("<|im_start|>assistant\n", "<|im_start|>assistant"):
        encoded = tokenizer(cue, add_special_tokens=False)["input_ids"]
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        pattern = tuple(_require_exact_int(int(token_id), name="assistant cue token") for token_id in encoded)
        if pattern and pattern not in patterns:
            patterns.append(pattern)
    if not patterns:
        raise ValueError("tokenizer did not produce an assistant cue")
    return tuple(patterns)


def validate_assistant_cue_scaffold(prepared: Mapping[str, torch.Tensor], tokenizer) -> None:
    """Require the prompt to terminate at the tokenized assistant cue before forcing latent start."""
    input_ids = prepared.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("assistant cue scaffold requires one input_ids row")
    ids = tuple(_require_exact_int(int(token_id), name="prompt token") for token_id in input_ids[0].tolist())
    if not any(len(ids) >= len(pattern) and ids[-len(pattern):] == pattern for pattern in _assistant_cue_patterns(tokenizer)):
        raise ValueError("prompt does not end with the tokenized assistant cue before latent_start")


def enable_boundary_measurement_parameters(model) -> None:
    """Make only the in-memory boundary measurement targets differentiable."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.endswith("lm_head.output_delta")
            or name == "lm_head.output_delta"
            or _is_dora_measurement_parameter(name)
        )


def _is_dora_measurement_parameter(name: str) -> bool:
    return any(
        f".{component}." in f".{name}."
        for component in ("lora_A", "lora_B", "lora_magnitude_vector")
    )


def boundary_target_parameters(model, token_ids) -> dict[str, object]:
    """Find the exact Stage 1 sparse latent rows and every trainable DoRA tensor."""
    named = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    latent_matches = [
        (name, parameter) for name, parameter in named.items()
        if name.endswith("lm_head.output_delta") or name == "lm_head.output_delta"
    ]
    if len(latent_matches) != 1:
        raise ValueError("boundary-probe requires exactly one trainable latent rows parameter")
    latent_name, latent_parameter = latent_matches[0]
    latent_ids = (
        _require_exact_int(token_ids.pad_token_id, name="latent pad token id"),
        _require_exact_int(token_ids.start_token_id, name="latent start token id"),
        _require_exact_int(token_ids.end_token_id, name="latent end token id"),
    )
    if latent_parameter.ndim != 2 or latent_parameter.shape[0] != len(latent_ids):
        raise ValueError("trainable latent rows must be the exact three sparse output rows")
    components = ("lora_A", "lora_B", "lora_magnitude_vector")
    dora_parameters = {
        name: parameter for name, parameter in named.items()
        if _is_dora_measurement_parameter(name)
    }
    missing_components = [
        component for component in components
        if not any(component in name for name in dora_parameters)
    ]
    if missing_components:
        raise ValueError(
            "boundary-probe requires all trainable DoRA tensors: "
            + ", ".join(missing_components)
        )
    rows = [
        {
            "token_id": token_id,
            "row_index": row_index,
            "parameter_name": latent_name,
            "parameter": latent_parameter,
        }
        for row_index, token_id in enumerate(latent_ids)
    ]
    return {
        "latent_rows": rows,
        "latent_parameter": latent_parameter,
        "latent_parameter_name": latent_name,
        "dora_parameters": dora_parameters,
        "parameters": {latent_name: latent_parameter, **dora_parameters},
    }


def _manual_boundary_loss(model, prepared, token_ids, forced_ids):
    """Build a differentiable forced scaffold and return boundary logits and CE."""
    sequences = prepared["input_ids"]
    mask = prepared["attention_mask"]
    embeds = model.prepare_stage2_base_inputs_embeds(
        input_ids=sequences, pixel_values=prepared.get("pixel_values"), image_grid_thw=prepared.get("image_grid_thw")
    )
    for token_id in forced_ids:
        append = torch.tensor([[token_id]], dtype=sequences.dtype, device=sequences.device)
        sequences = torch.cat((sequences, append), dim=1)
        mask = torch.cat((mask, torch.ones_like(append)), dim=1)
        embeds = torch.cat((embeds, model.get_input_embeddings()(append)), dim=1)
        if token_id != token_ids.start_token_id:
            continue
        suffix = torch.tensor([[token_ids.pad_token_id] * model.config.latent_size + [token_ids.end_token_id]], dtype=sequences.dtype, device=sequences.device)
        sequences = torch.cat((sequences, suffix), dim=1)
        mask = torch.cat((mask, torch.ones_like(suffix)), dim=1)
        embeds = torch.cat((embeds, model.get_input_embeddings()(suffix)), dim=1)
        positions, _ = model.get_rope_index(sequences, image_grid_thw=prepared.get("image_grid_thw"), attention_mask=mask)
        first_latent = sequences.shape[1] - model.config.latent_size - 1
        for position in range(first_latent, first_latent + model.config.latent_size):
            prefix = model.model(inputs_embeds=embeds[:, :position], attention_mask=mask[:, :position], position_ids=positions[:, :, :position], use_cache=False, return_dict=True, generate_mode=True)
            state = prefix.last_hidden_state[:, -1, :]
            embeds = torch.cat((embeds[:, :position], state.unsqueeze(1), embeds[:, position + 1 :]), dim=1)
    positions, _ = model.get_rope_index(sequences, image_grid_thw=prepared.get("image_grid_thw"), attention_mask=mask)
    output = model.model(inputs_embeds=embeds, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True, generate_mode=True)
    logits = model.lm_head(output.last_hidden_state)
    prompt_length = prepared["input_ids"].shape[1]
    target_positions = list(range(prompt_length, sequences.shape[1]))
    per_position_ce = {
        int(position): _finite_float(F.cross_entropy(logits[:, position - 1, :], sequences[:, position]).item(), name="per-position CE")
        for position in target_positions
    }
    loss = F.cross_entropy(
        logits[:, prompt_length - 1 : -1, :].reshape(-1, logits.shape[-1]),
        sequences[:, prompt_length:].reshape(-1),
    )
    start_position = next(index for index, value in enumerate(sequences[0].tolist()) if index >= prompt_length and value == token_ids.start_token_id)
    end_position = start_position + model.config.latent_size + 1
    return loss, {
        "latent_start": logits[:, start_position - 1, :],
        "eos": logits[:, end_position - 1, :],
        "per_position_ce": per_position_ce,
    }


def _gradient_summary(parameters: Mapping[str, torch.nn.Parameter]) -> dict[str, object]:
    result = {}
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None:
            result[name] = None
            continue
        finite = _require_finite_tensor(gradient, name=f"gradient {name}")
        result[name] = {
            "l2": _finite_float(torch.linalg.vector_norm(finite.detach().float().reshape(-1)).item(), name=f"gradient {name}"),
            "nonzero_elements": int(torch.count_nonzero(finite).item()),
        }
    return result


def _tensor_gradient_evidence(gradient: torch.Tensor | None, *, name: str) -> dict[str, object] | None:
    if gradient is None:
        return None
    finite = _require_finite_tensor(gradient, name=f"gradient {name}")
    return {
        "l2": _finite_float(torch.linalg.vector_norm(finite.detach().float().reshape(-1)).item(), name=f"gradient {name}"),
        "nonzero_elements": int(torch.count_nonzero(finite).item()),
    }


def _boundary_gradient_evidence(targets: Mapping[str, object]) -> dict[str, object]:
    rows = []
    for row in targets["latent_rows"]:
        parameter = row["parameter"]
        assert isinstance(parameter, torch.nn.Parameter)
        row_index = _require_exact_int(row["row_index"], name="latent row index")
        gradient = None if parameter.grad is None else parameter.grad[row_index]
        rows.append({
            "token_id": _require_exact_int(row["token_id"], name="latent token id"),
            "row_index": row_index,
            "parameter_name": row["parameter_name"],
            "gradient": _tensor_gradient_evidence(gradient, name=str(row["parameter_name"])),
        })
    dora = targets["dora_parameters"]
    assert isinstance(dora, Mapping)
    return {"latent_rows": rows, "dora_parameters": _gradient_summary(dora)}


def _boundary_delta_evidence(
    targets: Mapping[str, object], before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    rows = []
    for row in targets["latent_rows"]:
        name = row["parameter_name"]
        assert isinstance(name, str)
        row_index = _require_exact_int(row["row_index"], name="latent row index")
        initial = before[name][row_index]
        final = after[name][row_index]
        delta = final.detach().to(torch.float64) - initial.detach().to(torch.float64)
        rows.append({
            "token_id": _require_exact_int(row["token_id"], name="latent token id"),
            "row_index": row_index,
            "parameter_name": name,
            "changed_elements": int(torch.count_nonzero(delta).item()),
            "l2_delta": _finite_float(torch.linalg.vector_norm(delta.reshape(-1)).item(), name="latent row delta"),
            "symmetric_relative_l2": symmetric_relative_l2(initial, final),
        })
    dora = targets["dora_parameters"]
    assert isinstance(dora, Mapping)
    return {
        "latent_rows": rows,
        "dora_parameters": parameter_delta_summary(
            {name: before[name] for name in dora},
            {name: after[name] for name in dora},
        ),
    }


def _parse_boundary_indices(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(_require_exact_int(int(part), name="source index") for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise ValueError("source-indices must be comma-separated integers 0,1,2,7") from error
    if parsed != BOUNDARY_SOURCE_INDICES:
        raise ValueError("boundary-probe source-indices must be exactly 0,1,2,7")
    return parsed


def run_boundary_probe(*, model_dir: str, base_model_dir: str, dataset_jsonl: str, cache_dir: str, source_indices: Sequence[int]) -> dict[str, object]:
    if tuple(source_indices) != BOUNDARY_SOURCE_INDICES:
        raise ValueError("boundary-probe source indices must be exactly 0,1,2,7")
    model, processor = _load_probe_runtime(model_dir, base_model_dir, cache_dir)
    dataset_path = Path(dataset_jsonl).resolve(strict=True)
    dataset = _read_jsonl(dataset_path)
    source_rows = validated_test_source_rows(dataset_path, dataset, source_indices)
    token_ids = resolve_latent_token_ids(processor.tokenizer, model.config)
    ordinary = _ordinary_token_id(processor.tokenizer, token_ids)
    all_parameters = dict(model.named_parameters())
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in all_parameters.items()
    }
    original_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in all_parameters.items()
    }
    original_training = model.training
    cases: list[dict[str, object]] = []
    all_trainable: dict[str, torch.nn.Parameter] = {}
    restore: dict[str, torch.Tensor] = {}

    def restore_values() -> None:
        with torch.no_grad():
            for name, parameter in all_trainable.items():
                parameter.copy_(restore[name])

    def restore_gradients() -> None:
        for name, parameter in all_parameters.items():
            gradient = original_gradients[name]
            parameter.grad = None if gradient is None else gradient.detach().clone()

    try:
        enable_boundary_measurement_parameters(model)
        targets = boundary_target_parameters(model, token_ids)
        all_trainable = {
            name: parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not all_trainable:
            raise ValueError("boundary-probe requires trainable Stage 1 parameters")
        restore = {
            name: parameter.detach().clone() for name, parameter in all_trainable.items()
        }
        model.train()
        for source_row in source_rows:
            index = _require_exact_int(source_row["source_index"], name="source index")
            sample = source_row["sample"]
            assert isinstance(sample, Mapping)
            prepared = _prepared_for_sample(model, processor, sample, dataset_path.parent)
            validate_assistant_cue_scaffold(prepared, processor.tokenizer)
            for case_name, forced_ids in (
                ("assistant_cue_then_latent", (token_ids.start_token_id,)),
                ("assistant_cue_text_then_latent", (ordinary, token_ids.start_token_id)),
            ):
                # Each case is a one-step measurement, not a short training run.
                restore_values()
                restore_gradients()
                optimizer = torch.optim.AdamW(
                    tuple(all_trainable.values()),
                    lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                )
                try:
                    optimizer.zero_grad(set_to_none=True)
                    loss, boundary = _manual_boundary_loss(model, prepared, token_ids, forced_ids)
                    loss.backward()
                    gradients_before_clipping = _boundary_gradient_evidence(targets)
                    selected_parameters = targets["parameters"]
                    assert isinstance(selected_parameters, Mapping)
                    before = {
                        name: parameter.detach().clone()
                        for name, parameter in selected_parameters.items()
                    }
                    global_norm_before_clipping = torch.nn.utils.clip_grad_norm_(
                        tuple(all_trainable.values()), 1.0
                    )
                    if not math.isfinite(float(global_norm_before_clipping)):
                        raise ValueError("global gradient norm must be finite")
                    optimizer.step()
                    after = {
                        name: parameter.detach().clone()
                        for name, parameter in selected_parameters.items()
                    }
                    cases.append({
                        "source_index": index,
                        "case": case_name,
                        "forced_prefix_ids": list(forced_ids),
                        "loss": _finite_float(loss.item(), name="boundary loss"),
                        "latent_start": {
                            "token_id": token_ids.start_token_id,
                            "rank": token_rank(boundary["latent_start"][0], token_ids.start_token_id),
                            "logit": _finite_float(boundary["latent_start"][0, token_ids.start_token_id].item(), name="latent-start logit"),
                        },
                        "eos": {
                            "token_id": token_ids.end_token_id,
                            "rank": token_rank(boundary["eos"][0], token_ids.end_token_id),
                            "logit": _finite_float(boundary["eos"][0, token_ids.end_token_id].item(), name="EOS logit"),
                        },
                        "per_position_ce": boundary["per_position_ce"],
                        "gradients_before_clipping": gradients_before_clipping,
                        "global_norm_before_clipping": _finite_float(float(global_norm_before_clipping), name="global gradient norm"),
                        "parameter_deltas": _boundary_delta_evidence(targets, before, after),
                    })
                finally:
                    restore_values()
                    restore_gradients()
    finally:
        restore_values()
        restore_gradients()
        for name, parameter in all_parameters.items():
            parameter.requires_grad_(original_requires_grad[name])
        model.train(original_training)

    # These are intentionally real bounded generations. They are invoked only
    # by the explicit diagnostic CLI, never during import or reducer tests.
    stop_observations = []
    for source_row in source_rows:
        index = _require_exact_int(source_row["source_index"], name="source index")
        sample = source_row["sample"]
        assert isinstance(sample, Mapping)
        prepared = _prepared_for_sample(model, processor, sample, dataset_path.parent)
        validate_assistant_cue_scaffold(prepared, processor.tokenizer)
        for budget in BOUNDARY_BUDGETS:
            original_head = model.lm_head
            model.lm_head = _ForcedPrefixHead(original_head, (token_ids.start_token_id,))
            try:
                generated = ContinuousLatentController().generate(
                    model=model,
                    input_ids=prepared["input_ids"], attention_mask=prepared["attention_mask"], token_ids=token_ids,
                    generation_config=ContinuousLatentGenerationConfig(max_new_tokens=budget, eos_token_id=getattr(processor.tokenizer, "eos_token_id", None)),
                    pixel_values=prepared.get("pixel_values"), image_grid_thw=prepared.get("image_grid_thw"),
                )
            finally:
                model.lm_head = original_head
            stop_observations.append({"source_index": index, "budget": budget, "stop_reason": generated.stop_reason})
    return {
        "diagnostic": "stage1-boundary-probe",
        "source_indices": list(source_indices),
        "optimizer": {"name": "AdamW", "lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "global_norm_clip": 1.0, "scheduler": None},
        "cases": cases,
        "budget_stop_observations": stop_observations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    rescore = modes.add_parser("rescore")
    rescore.add_argument("--input-jsonl", required=True)
    rescore.add_argument("--dataset-jsonl", required=True)
    rescore.add_argument("--output-json", required=True)
    for name in ("recurrence-probe", "boundary-probe"):
        probe = modes.add_parser(name)
        probe.add_argument("--model-dir", required=True)
        probe.add_argument("--base-model-dir", required=True)
        probe.add_argument("--dataset-jsonl", required=True)
        probe.add_argument("--output-json", required=True)
        probe.add_argument("--cache-dir", required=True)
        if name == "boundary-probe":
            probe.add_argument("--source-indices", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = _require_results_output_path(args.output_json)
    log_path = derive_execution_log_path(output)
    try:
        dataset_path = Path(args.dataset_jsonl).resolve(strict=True)
        if args.mode == "rescore":
            evidence = rescore_historical_rows(_read_jsonl(Path(args.input_jsonl).resolve(strict=True)), _read_jsonl(dataset_path))
        elif args.mode == "recurrence-probe":
            evidence = run_recurrence_probe(model_dir=args.model_dir, base_model_dir=args.base_model_dir, dataset_jsonl=str(dataset_path), cache_dir=args.cache_dir)
        else:
            evidence = run_boundary_probe(model_dir=args.model_dir, base_model_dir=args.base_model_dir, dataset_jsonl=str(dataset_path), cache_dir=args.cache_dir, source_indices=_parse_boundary_indices(args.source_indices))
        _write_json(output, evidence)
    except Exception as error:
        _write_execution_log(log_path, mode=args.mode, output=output, status="failure", detail=f"{type(error).__name__}: {error}")
        raise
    _write_execution_log(log_path, mode=args.mode, output=output, status="success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

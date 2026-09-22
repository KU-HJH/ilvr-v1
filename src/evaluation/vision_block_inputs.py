"""Fixed CE-analysis input preparation and replay identity validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import torch

from stage2.data import Stage2DataCollator, prepare_stage2_example
from step_preference.spans import expand_qwen_image_placeholders, map_reasoning_step_spans
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


EXPECTED_DATA_SHA256 = "6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618"
EXPECTED_SELECTION_SHA256 = "fb1f15fef5339abde6ba74d140e5f9a1f8823cedafe8aef1d4bb245a8e0286aa"
EXPECTED_CATEGORY_COUNTS = {
    "creation": 25,
    "deletion": 30,
    "update": 15,
    "selection": 30,
}


@dataclass(frozen=True)
class Site:
    source_index: int
    plan_index: int
    start: int
    body_start: int
    body_end: int
    end: int
    target_positions: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreparedSample:
    source_index: int
    model_inputs: dict
    sites: tuple[Site, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latent_token_ids(tokenizer) -> LatentTokenIds:
    values = tuple(
        int(tokenizer.convert_tokens_to_ids(token))
        for token in ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>")
    )
    if any(value < 0 or value == tokenizer.unk_token_id for value in values):
        raise ValueError("processor tokenizer lacks the Stage 2 latent tokens")
    return LatentTokenIds(*values)


def _exclusion(
    source_index: int,
    plan_index: int | None,
    reason: str,
    explanation: str,
) -> dict:
    return {
        "source_index": source_index,
        "plan_index": plan_index,
        "reason": reason,
        "explanation": explanation,
    }


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sample_identity(sample: PreparedSample) -> dict:
    input_hashes = {
        key: _tensor_hash(value)
        for key, value in sorted(sample.model_inputs.items())
        if isinstance(value, torch.Tensor)
    }
    sites = [asdict(site) for site in sample.sites]
    payload = json.dumps(
        {"source_index": sample.source_index, "inputs": input_hashes, "sites": sites},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "input_hashes": input_hashes,
        "sites_sha256": hashlib.sha256(
            json.dumps(sites, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def prepare_sample(
    row, source_index, processor, image_root
) -> tuple[PreparedSample | None, list[dict]]:
    """Prepare one original-image CE sample and independently map each valid site."""
    prepared = prepare_stage2_example(
        row,
        image_root=Path(image_root),
        latent_size=8,
        require_reasoning_units=False,
        assistant_plan_rendering="concatenated-v1",
    )
    token_ids = _latent_token_ids(processor.tokenizer)
    batch = Stage2DataCollator(
        processor=processor,
        token_ids=token_ids,
        latent_size=8,
        supervise_latent_boundaries=True,
        step_preference_enabled=False,
    )([prepared])

    model_inputs = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    }
    forbidden = {
        "labels",
        "pixel_values_latent",
        "image_grid_thw_latent",
    } & model_inputs.keys()
    if forbidden:
        raise ValueError(f"non-inference tensors leaked into model inputs: {sorted(forbidden)}")

    rendered_text = processor.apply_chat_template(
        prepared.conversation, tokenize=False, add_generation_prompt=False
    )
    expanded_text = expand_qwen_image_placeholders(
        rendered_text,
        image_grid_thw=batch["image_grid_thw"],
        image_token=processor.image_token,
        merge_size=processor.image_processor.merge_size,
    )
    offset_batch = processor.tokenizer(
        [expanded_text],
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = offset_batch.pop("offset_mapping")
    for key, value in offset_batch.items():
        if key not in batch or not torch.equal(value, batch[key]):
            return None, [
                _exclusion(
                    source_index,
                    None,
                    "sample_tokenization_mismatch",
                    f"offset tokenization does not match processor tensor: {key}",
                )
            ]

    mapping_kwargs = {
        "rendered_text": expanded_text,
        "assistant_text": prepared.conversation[1]["content"][0]["text"],
        "plan_char_spans": prepared.plan_char_spans,
        "offset_mapping": offsets,
        "input_ids": batch["input_ids"],
        "latent_pad_token_id": token_ids.pad_token_id,
        "preference_target_scope": "immediate_text",
    }
    try:
        map_reasoning_step_spans(reasoning_steps=(), **mapping_kwargs)
    except ValueError as error:
        return None, [
            _exclusion(
                source_index,
                None,
                "sample_span_mapping_invalid",
                str(error),
            )
        ]

    eligible = {step.latent.plan_index: step for step in prepared.reasoning_steps}
    exclusions = []
    sites = []
    for span in prepared.plan_char_spans:
        if span.step_type != "latent":
            continue
        step = eligible.get(span.plan_index)
        if step is None:
            exclusions.append(
                _exclusion(
                    source_index,
                    span.plan_index,
                    "ineligible_reasoning_unit",
                    "latent step lacks immediate non-empty text neighbors",
                )
            )
            continue
        try:
            mapped = map_reasoning_step_spans(
                reasoning_steps=(step,),
                **mapping_kwargs,
            )[0]
        except ValueError as error:
            exclusions.append(
                _exclusion(source_index, span.plan_index, "unmappable_site", str(error))
            )
            continue
        positions = tuple(range(mapped.target_text_start, mapped.target_text_end))
        ids = tuple(int(batch["input_ids"][0, position]) for position in positions)
        special_ids = set(processor.tokenizer.all_special_ids)
        if not positions or any(token_id in special_ids for token_id in ids):
            exclusions.append(
                _exclusion(
                    source_index,
                    span.plan_index,
                    "invalid_target_tokens",
                    "target is empty or contains a special token",
                )
            )
            continue
        site = Site(
            source_index=source_index,
            plan_index=span.plan_index,
            start=mapped.latent_start - 1,
            body_start=mapped.latent_start,
            body_end=mapped.latent_end,
            end=mapped.latent_end,
            target_positions=positions,
            target_ids=ids,
        )
        if site.end + 1 != site.target_positions[0] or site.body_end - site.body_start != 8:
            exclusions.append(
                _exclusion(
                    source_index,
                    span.plan_index,
                    "invalid_site_geometry",
                    "site token geometry is not contiguous",
                )
            )
            continue
        sites.append(site)

    if not sites:
        return None, exclusions
    return PreparedSample(source_index, model_inputs, tuple(sites)), exclusions


def _processor_identity(processor) -> dict:
    tokenizer = processor.tokenizer
    payload = {
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "name_or_path": str(getattr(processor, "name_or_path", "")),
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(
            str(getattr(tokenizer, "chat_template", "")).encode()
        ).hexdigest(),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _validate_selection(indices, row_count: int) -> tuple[int, ...]:
    valid = (
        isinstance(indices, list)
        and len(indices) == 100
        and all(type(index) is int and 0 <= index < row_count for index in indices)
        and len(set(indices)) == 100
    )
    if not valid:
        raise ValueError("selection must contain 100 unique in-range zero-based indices")
    return tuple(indices)


def _category(row: dict) -> str:
    images = row.get("image_input")
    path = images[0] if isinstance(images, list) and images else images
    if not isinstance(path, str):
        raise ValueError("selected row has no category-bearing image path")
    matches = [name for name in EXPECTED_CATEGORY_COUNTS if f"/{name}/" in f"/{path}"]
    if len(matches) != 1:
        raise ValueError(f"selected row has unknown image category: {path}")
    return matches[0]


def _load_site_record(record: dict) -> Site:
    scalar_fields = (
        "source_index",
        "plan_index",
        "start",
        "body_start",
        "body_end",
        "end",
    )
    sequence_fields = ("target_positions", "target_ids")
    if (
        type(record) is not dict
        or set(record) != set(scalar_fields + sequence_fields)
        or any(type(record[field]) is not int for field in scalar_fields)
        or any(type(record[field]) not in (list, tuple) for field in sequence_fields)
        or any(
            type(value) is not int
            for field in sequence_fields
            for value in record[field]
        )
    ):
        raise ValueError("persisted site has invalid fields or types")
    return Site(
        **{field: record[field] for field in scalar_fields},
        **{field: tuple(record[field]) for field in sequence_fields},
    )


def prepare_manifest(data_path, selection_path, processor, image_root, output_path) -> dict:
    """Prepare the immutable selected cohort and write its replay manifest."""
    data_path, selection_path, output_path = map(
        Path, (data_path, selection_path, output_path)
    )
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selection_document = json.loads(selection_path.read_text(encoding="utf-8"))
    indices = _validate_selection(
        selection_document.get("selection", {}).get("requested_indices"), len(rows)
    )
    data_hash, selection_hash = _sha256(data_path), _sha256(selection_path)
    if data_hash != EXPECTED_DATA_SHA256:
        raise ValueError(f"dataset SHA-256 mismatch: {data_hash}")
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError(f"selection SHA-256 mismatch: {selection_hash}")
    counts = {name: 0 for name in EXPECTED_CATEGORY_COUNTS}
    for index in indices:
        counts[_category(rows[index])] += 1
    if counts != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"selection category counts mismatch: {counts}")

    samples, exclusions = [], []
    for index in indices:
        sample, sample_exclusions = prepare_sample(rows[index], index, processor, image_root)
        exclusions.extend(sample_exclusions)
        if sample is not None:
            samples.append(
                {
                    "source_index": index,
                    "sites": [asdict(site) for site in sample.sites],
                    "identity": _sample_identity(sample),
                }
            )
    manifest = {
        "schema_version": 1,
        "source": {"path": str(data_path.resolve()), "sha256": data_hash},
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": selection_hash,
            "requested_indices": list(indices),
            "category_counts": counts,
        },
        "processor": _processor_identity(processor),
        "samples": samples,
        "exclusions": exclusions,
        "S": sum(len(sample["sites"]) for sample in samples),
        "N": sum(len(site["target_positions"]) for sample in samples for site in sample["sites"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_prepared_samples(manifest, processor, image_root) -> Iterator[PreparedSample]:
    """Re-prepare manifest samples and reject any changed input or target identity."""
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported vision-block manifest schema")
    if "rows" in manifest:
        rows = manifest["rows"]
    else:
        source = manifest.get("source", {})
        path = Path(source.get("path", ""))
        if _sha256(path) != source.get("sha256"):
            raise ValueError("source identity changed")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    for record in manifest.get("samples", []):
        index = record["source_index"]
        sample, _ = prepare_sample(rows[index], index, processor, image_root)
        actual = _sample_identity(sample) if sample is not None else None
        expected = record.get("identity")
        try:
            persisted_sites = tuple(_load_site_record(site) for site in record["sites"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"input/span/target identity changed for source index {index}"
            ) from error
        if (
            not isinstance(expected, dict)
            or actual != expected
            or sample is None
            or persisted_sites != sample.sites
        ):
            raise ValueError(f"input/span/target identity changed for source index {index}")
        yield sample

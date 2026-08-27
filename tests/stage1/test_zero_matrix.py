from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from src.stage1.benchmark import Stage1BenchmarkEvidence, Stage1ProducerIdentity


def audit_digest(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def producer(**overrides) -> Stage1BenchmarkEvidence:
    values = {
        "status": "success",
        "method": "full",
        "zero_stage": 3,
        "deepspeed_config_digest": "d" * 64,
        "micro_batch": 4,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_global_batch": 16,
        "optimizer_steps": 3,
        "samples_seen": 48,
        "sample_stream_digest": "a" * 64,
        "model_id": "model",
        "seed": 42,
        "model_load_seconds": 1.0,
        "trainer_seconds": 2.0,
        "total_seconds": 3.0,
        "final_save_seconds": 0.0,
        "samples_per_second": 8.0,
        "steps_per_second": 1.0,
        "rank_peak_allocated_bytes": (10, 11),
        "rank_peak_reserved_bytes": (12, 13),
        "trainable_parameters": 20,
        "total_parameters": 100,
        "checkpoint_bytes": 0,
        "final_loss": 0.5,
        "representative_gradient_norm": 0.25,
        "git_commit": "abc1234",
        "dependency_versions": {"torch": "2.7.1"},
        "gpu_names": ("B200", "B200"),
        "config_digest": "c" * 64,
        "probe_row_indices": (7, 3),
    }
    values.update(overrides)
    return Stage1BenchmarkEvidence(**values)


def attempt(*, micro: int, speed: float | None, status: str = "success"):
    from src.stage1.zero_matrix import ZeroMatrixAttempt

    accumulation = 8 // micro
    identity = Stage1ProducerIdentity(
        method="full",
        zero_stage=3,
        deepspeed_config_digest="d" * 64,
        world_size=2,
        micro_batch=micro,
        gradient_accumulation_steps=accumulation,
        effective_global_batch=16,
        sample_stream_digest="a" * 64,
        model_id="model",
        seed=42,
    )
    return ZeroMatrixAttempt(
        identity=identity,
        status=status,
        samples_per_second=speed,
        evidence_path=Path(f"/results/producer-{micro}.json"),
    )


def test_candidates_are_fixed_egb16_in_reverse_microbatch_order():
    """Catches accidental matrix expansion, reordering, or batch drift."""
    from src.stage1.zero_matrix import ZERO_MATRIX_CANDIDATES

    assert [
        (candidate.micro_batch, candidate.gradient_accumulation_steps)
        for candidate in ZERO_MATRIX_CANDIDATES
    ] == [(8, 1), (4, 2), (2, 4), (1, 8)]
    assert all(candidate.effective_global_batch == 16 for candidate in ZERO_MATRIX_CANDIDATES)


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    [
        (1, "torch.OutOfMemoryError: CUDA out of memory", "capacity_oom"),
        (1, "DeepSpeed CUDAMalloc failed allocating device memory", "capacity_oom"),
        (1, "NCCL watchdog collective timeout", "integration_failure"),
        (1, "loss became nonfinite", "integration_failure"),
        (1, "CPU out of memory", "integration_failure"),
        (0, "", "integration_failure"),
    ],
)
def test_only_cuda_allocator_oom_is_capacity_failure(returncode, stderr, expected):
    """Catches integration failures being treated as permission to shrink batch."""
    from src.stage1.zero_matrix import classify_zero_matrix_attempt

    assert classify_zero_matrix_attempt(returncode=returncode, stderr=stderr) == expected


def test_zero_exit_requires_complete_successful_producer():
    """Catches a zero process exit being accepted without producer evidence."""
    from src.stage1.zero_matrix import classify_zero_matrix_attempt

    assert classify_zero_matrix_attempt(returncode=0, stderr="", producer=producer()) == "success"
    failed = replace(producer(), status="integration_failure", optimizer_steps=0, samples_seen=0)
    assert (
        classify_zero_matrix_attempt(returncode=0, stderr="", producer=failed)
        == "integration_failure"
    )


def test_fastest_success_wins_and_failure_is_never_selected():
    """Catches capacity failures or slower successful rows winning selection."""
    from src.stage1.zero_matrix import select_fastest_success

    rows = [
        attempt(micro=4, speed=3.2),
        attempt(micro=2, speed=3.8),
        attempt(micro=8, speed=None, status="capacity_oom"),
    ]

    assert select_fastest_success(rows).micro_batch == 2


def test_selection_requires_a_complete_four_candidate_identity_matrix():
    """Catches partial or cross-method evidence being published as a winner."""
    from src.stage1.zero_matrix import select_zero_matrix

    rows = [attempt(micro=8, speed=3.0), attempt(micro=4, speed=3.2)]

    with pytest.raises(ValueError, match="exact four-candidate matrix"):
        select_zero_matrix(rows)


def test_runtime_profiles_are_loaded_once_per_stage(monkeypatch, tmp_path):
    """Catches the config loader moving into the method/candidate loop."""
    import src.stage1.zero_matrix as matrix

    calls = []
    profiles = {}

    def fake_loader(path, *, world_size):
        calls.append((Path(path), world_size))
        stage = int(Path(path).stem[-1])
        profile = object()
        profiles[stage] = profile
        return profile

    monkeypatch.setattr(matrix, "load_deepspeed_runtime_profile", fake_loader)
    paths = {stage: tmp_path / f"deepspeed_zero{stage}.json" for stage in (3, 2, 1)}

    loaded = matrix.load_zero_matrix_profiles(paths, world_size=2)

    assert calls == [(paths[3], 2), (paths[2], 2), (paths[1], 2)]
    assert loaded == {3: profiles[3], 2: profiles[2], 1: profiles[1]}


def test_prepare_reuses_profile_witness_without_rehashing_config(monkeypatch, tmp_path):
    """Catches each candidate reopening the already-loaded DeepSpeed config."""
    import src.stage1.zero_matrix as matrix

    config = tmp_path / "deepspeed_zero2.json"

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("DeepSpeed config was loaded again")

    monkeypatch.setattr(matrix, "load_deepspeed_runtime_profile", forbidden_loader)

    profile = matrix.runtime_profile_from_witness(
        config_path=config,
        config_digest="d" * 64,
        zero_stage=2,
        world_size=2,
    )

    assert profile.config_path == config.resolve()
    assert profile.config_digest == "d" * 64
    assert profile.zero_stage == 2


def test_audit_status_selects_first_exact_stat_witness_without_reading_dataset(
    monkeypatch, tmp_path
):
    """Catches legacy provenance scanning dataset contents or preferring it over canonical."""
    import src.stage1.zero_matrix as matrix

    data = tmp_path / "train.jsonl"
    data.write_text("contents must remain unread\n", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    witness = matrix.path_stat_witness(data)
    canonical = tmp_path / "canonical-audit.json"
    legacy = tmp_path / "legacy-audit.json"
    base_without_digest = {
        "schema_version": 1,
        "kind": "stage1-high-cost-row-audit",
        "data_path": str(data.resolve()),
        "image_root": str(images.resolve()),
        "requested_count": 64,
        "row_count": 64,
        "input_stat": witness,
        "selected_rows": [{"source_row_index": 7}],
    }
    canonical_payload = base_without_digest | {"input_stat": witness | {"size": 0}}
    canonical_payload["audit_digest"] = audit_digest(canonical_payload)
    legacy_payload = dict(base_without_digest)
    legacy_payload["audit_digest"] = audit_digest(legacy_payload)
    canonical.write_text(json.dumps(canonical_payload))
    legacy.write_text(json.dumps(legacy_payload))
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        if Path(path) == data:
            raise AssertionError("dataset contents were read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    match = matrix.resolve_sample_audit(
        [canonical, legacy], data_path=data, image_root=images, count=64
    )

    assert match.path == legacy.resolve()
    assert match.audit_digest == legacy_payload["audit_digest"]


def test_audit_status_rejects_tampered_small_witness(tmp_path):
    """Catches reuse of a path/stat/count payload whose audit seal no longer matches."""
    import src.stage1.zero_matrix as matrix

    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    payload = {
        "schema_version": 1,
        "kind": "stage1-high-cost-row-audit",
        "data_path": str(data.resolve()),
        "image_root": str(images.resolve()),
        "requested_count": 64,
        "row_count": 64,
        "input_stat": matrix.path_stat_witness(data),
        "selected_rows": [{"source_row_index": 7}],
    }
    payload["audit_digest"] = audit_digest(payload)
    payload["selected_rows"] = [{"source_row_index": 8}]
    tampered = tmp_path / "tampered-audit.json"
    tampered.write_text(json.dumps(payload))

    with pytest.raises(FileNotFoundError, match="exact path/stat/count"):
        matrix.resolve_sample_audit(
            [tampered], data_path=data, image_root=images, count=64
        )


def test_prepare_writes_zero_matrix_run_config_bound_to_profile_and_audit(tmp_path):
    """Catches prepare rehashing inputs or omitting the exact three-step identity."""
    from src.deepspeed_runtime import load_deepspeed_runtime_profile
    from src.stage1.zero_matrix import SampleAuditWitness, prepare_zero_matrix_run

    data = tmp_path / "probe.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    profile = load_deepspeed_runtime_profile(
        Path(__file__).resolve().parents[2] / "configs/deepspeed_zero2.json",
        world_size=2,
    )
    audit = SampleAuditWitness(
        path=(tmp_path / "sample-audit.json").resolve(),
        audit_digest="a" * 64,
        input_stat={
            "device": data.stat().st_dev,
            "inode": data.stat().st_ino,
            "size": data.stat().st_size,
            "mtime_ns": data.stat().st_mtime_ns,
        },
        row_indices=(7, 3),
    )
    result_dir = tmp_path / "result"
    run_config = result_dir / "run-config.json"

    prepared = prepare_zero_matrix_run(
        run_config_path=run_config,
        method="dora",
        model="model",
        source_data_path=data,
        data_path=data,
        image_root=images,
        output_dir=tmp_path / "checkpoint",
        candidate_micro_batch=4,
        profile=profile,
        sample_audit=audit,
        prerequisite_chain_path=tmp_path / "prerequisite.json",
        prerequisite_chain_digest="f" * 64,
        final_save=False,
    )

    assert prepared["schema_version"] == 2
    assert prepared["producer_kind"] == "zero_matrix"
    assert prepared["optimizer_steps"] == prepared["max_steps"] == 3
    assert prepared["effective_global_batch"] == 16
    assert prepared["deepspeed_config_digest"] == profile.config_digest
    assert prepared["audit_digest"] == "a" * 64
    assert json.loads(run_config.read_text()) == prepared


def test_select_atomically_publishes_report_with_producer_witnesses(tmp_path):
    """Catches selection dropping evidence provenance or writing outside the matrix root."""
    from src.stage1.zero_matrix import write_zero_matrix_report

    rows = [
        attempt(micro=8, speed=None, status="capacity_oom"),
        attempt(micro=4, speed=4.0),
        attempt(micro=2, speed=3.9),
        attempt(micro=1, speed=3.0),
    ]
    digests = {str(row.evidence_path): str(row.micro_batch) * 64 for row in rows}

    report_path = write_zero_matrix_report(tmp_path, rows, evidence_digests=digests)
    report = json.loads(report_path.read_text())

    assert report_path == tmp_path / "zero-matrix-report.json"
    assert report["selected"]["identity"]["micro_batch"] == 4
    assert [row["identity"]["micro_batch"] for row in report["rows"]] == [8, 4, 2, 1]
    assert report["selected"]["evidence_digest"] == "4" * 64


def test_report_preserves_prior_stage_method_matrix(tmp_path):
    """Catches each pair selection replacing the aggregate matrix report."""
    from src.stage1.zero_matrix import write_zero_matrix_report

    full_rows = [
        attempt(micro=8, speed=3.0),
        attempt(micro=4, speed=4.0),
        attempt(micro=2, speed=3.5),
        attempt(micro=1, speed=2.0),
    ]
    dora_rows = [
        replace(
            row,
            identity=replace(row.identity, method="dora", zero_stage=2),
            evidence_path=Path(str(row.evidence_path).replace("producer", "dora")),
        )
        for row in full_rows
    ]
    full_digests = {str(row.evidence_path): "a" * 64 for row in full_rows}
    dora_digests = {str(row.evidence_path): "b" * 64 for row in dora_rows}

    write_zero_matrix_report(tmp_path, full_rows, evidence_digests=full_digests)
    write_zero_matrix_report(tmp_path, dora_rows, evidence_digests=dora_digests)
    report = json.loads((tmp_path / "zero-matrix-report.json").read_text())

    assert set(report["matrices"]) == {"zero3-full", "zero2-dora"}
    assert len(report["rows"]) == 8
    assert [(row["identity"]["zero_stage"], row["identity"]["method"]) for row in report["rows"]] == [
        *((3, "full"),) * 4,
        *((2, "dora"),) * 4,
    ]


def test_selected_save_and_resume_witnesses_are_independently_reusable(tmp_path):
    """Catches selected resume reuse being conflated with selected save reuse."""
    from src.stage1.zero_matrix import (
        load_reusable_selected_run,
        write_selected_run_witness,
    )

    identity = attempt(micro=4, speed=4.0).identity
    roots = {}
    for run_kind in ("selected-save", "selected-resume"):
        experiment_id = f"stage1-zero3-full-mb4-ga2-{run_kind}-unit"
        directories = [tmp_path / root / experiment_id for root in ("checkpoints", "results", "logs")]
        for directory in directories:
            directory.mkdir(parents=True)
        roots[run_kind] = directories
        write_selected_run_witness(
            directories[1] / "selected-run-witness.json",
            run_kind=run_kind,
            identity=identity,
            experiment_id=experiment_id,
            checkpoint_dir=directories[0],
            result_dir=directories[1],
            log_dir=directories[2],
            resume_from=(
                None if run_kind == "selected-save" else roots["selected-save"][0]
            ),
        )

    save = load_reusable_selected_run(
        roots["selected-save"][1] / "selected-run-witness.json",
        run_kind="selected-save",
        identity=identity,
    )
    resume = load_reusable_selected_run(
        roots["selected-resume"][1] / "selected-run-witness.json",
        run_kind="selected-resume",
        identity=identity,
    )

    assert save is not None and save["experiment_id"] != resume["experiment_id"]
    assert resume is not None and resume["resume_from"] == str(roots["selected-save"][0])
    assert (
        load_reusable_selected_run(
            roots["selected-resume"][1] / "selected-run-witness.json",
            run_kind="selected-save",
            identity=identity,
        )
        is None
    )

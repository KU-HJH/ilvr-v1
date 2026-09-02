import json
import math
import os
import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import stage2.efficiency_probe as efficiency_probe_module
from stage2.efficiency_probe import (
    SEGMENTED_KV_GATE,
    Stage2EfficiencySample,
    authorize_segmented_kv,
    benchmark_worker_counts,
    build_parser,
    build_representative_dataloader,
    measure_cuda_training_step,
    read_efficiency_result,
    select_fastest_worker_count,
    select_representative_examples,
    summarize_samples,
    write_efficiency_result,
)


def manifest(*, lengths, block_counts, retained_indices):
    return SimpleNamespace(
        lengths=tuple(lengths),
        latent_block_counts=tuple(block_counts),
        retained_indices=tuple(retained_indices),
    )


def sample(*, step_ms=100.0, prefix_decoder_ms=30.0, peak_bytes=10, **kwargs):
    if not any(
        name in kwargs
        for name in (
            "collation_ms",
            "forward_loss_ms",
            "backward_ms",
            "optimizer_ms",
            "step_overhead_ms",
        )
    ):
        kwargs["step_overhead_ms"] = step_ms
    return Stage2EfficiencySample(
        step_ms=step_ms,
        prefix_decoder_ms=prefix_decoder_ms,
        peak_bytes=peak_bytes,
        **kwargs,
    )


def test_representative_selection_uses_retained_joint_quantiles_and_raw_index_ties():
    """Catches selection using filtered rows, marginal examples, or unstable ties."""
    selected = select_representative_examples(
        manifest(
            lengths=(1, 10, 10, 20, 30, 1000),
            block_counts=(99, 1, 1, 2, 4, 0),
            retained_indices=(1, 2, 3, 4),
        )
    )

    assert [(item.quantile, item.raw_index) for item in selected] == [
        ("p50", 1),
        ("p95", 4),
    ]
    assert selected[0].target_length == 15.0
    assert selected[0].target_block_count == 1.5
    assert selected[1].target_length == pytest.approx(28.5)
    assert selected[1].target_block_count == pytest.approx(3.7)


def test_representative_selection_rejects_empty_or_inconsistent_manifest():
    """Catches a probe silently selecting from absent or misaligned manifest rows."""
    with pytest.raises(ValueError, match="retained example"):
        select_representative_examples(
            manifest(lengths=(10,), block_counts=(1,), retained_indices=())
        )
    with pytest.raises(ValueError, match="cardinality"):
        select_representative_examples(
            manifest(lengths=(10, 20), block_counts=(1,), retained_indices=(0,))
        )


def test_aggregation_uses_component_medians_then_computes_prefix_share():
    """Catches averaging samples or averaging per-sample ratios at the hard gate."""
    evidence = summarize_samples(
        [
            sample(
                step_ms=100.0,
                prefix_decoder_ms=1.0,
                peak_bytes=100,
                collation_ms=3.0,
                final_decoder_ms=4.0,
                visual_ms=5.0,
                lm_head_ms=6.0,
                shifted_ce_loss_ms=2.0,
                forward_loss_ms=20.0,
                backward_ms=30.0,
                optimizer_ms=40.0,
                step_overhead_ms=7.0,
                forward_other_ms=3.0,
                loss=7.0,
                gradient_norm=8.0,
                prefix_input_length_sum=9,
            ),
            sample(
                step_ms=150.0,
                prefix_decoder_ms=60.0,
                peak_bytes=200,
                collation_ms=13.0,
                final_decoder_ms=14.0,
                visual_ms=15.0,
                lm_head_ms=16.0,
                shifted_ce_loss_ms=12.0,
                forward_loss_ms=30.0,
                backward_ms=40.0,
                optimizer_ms=50.0,
                step_overhead_ms=17.0,
                forward_other_ms=13.0,
                loss=17.0,
                gradient_norm=18.0,
                prefix_input_length_sum=19,
            ),
            sample(
                step_ms=200.0,
                prefix_decoder_ms=180.0,
                peak_bytes=300,
                collation_ms=23.0,
                final_decoder_ms=24.0,
                visual_ms=25.0,
                lm_head_ms=26.0,
                shifted_ce_loss_ms=22.0,
                forward_loss_ms=40.0,
                backward_ms=50.0,
                optimizer_ms=60.0,
                step_overhead_ms=27.0,
                forward_other_ms=23.0,
                loss=27.0,
                gradient_norm=28.0,
                prefix_input_length_sum=29,
            ),
        ]
    )

    assert evidence.sample_count == 3
    assert evidence.step_ms == 150.0
    assert evidence.prefix_decoder_ms == 60.0
    assert evidence.prefix_decoder_share == 0.4
    assert evidence.peak_bytes == 200
    assert evidence.collation_ms == 13.0
    assert evidence.final_decoder_ms == 14.0
    assert evidence.visual_ms == 15.0
    assert evidence.lm_head_ms == 16.0
    assert evidence.shifted_ce_loss_ms == 12.0
    assert evidence.forward_loss_ms == 30.0
    assert evidence.backward_ms == 40.0
    assert evidence.optimizer_ms == 50.0
    assert evidence.step_overhead_ms == 17.0
    assert evidence.forward_other_ms == 13.0
    assert evidence.loss == 17.0
    assert evidence.gradient_norm == 18.0
    assert evidence.prefix_input_length_sum == 19
    assert evidence.prefix_threshold_met is True
    assert evidence.segmented_kv_authorized is False


def test_aggregation_preserves_each_samples_same_clock_wall_partition():
    """Catches independent medians breaking valid per-sample wall partitions."""
    evidence = summarize_samples(
        [
            sample(step_ms=10.0, step_overhead_ms=10.0),
            sample(step_ms=20.0, optimizer_ms=20.0),
            sample(step_ms=30.0, backward_ms=30.0),
        ]
    )

    assert evidence.step_ms == 20.0
    assert (
        evidence.collation_ms
        + evidence.forward_loss_ms
        + evidence.backward_ms
        + evidence.optimizer_ms
        + evidence.step_overhead_ms
    ) == pytest.approx(evidence.step_ms, abs=1e-9)


def test_aggregation_rejects_mixed_clock_or_clamped_wall_partition():
    """Catches CUDA-event regions being mixed with wall step time then clamped."""
    with pytest.raises(ValueError, match="same-clock wall partition"):
        summarize_samples(
            [
                sample(
                    step_ms=10.0,
                    forward_loss_ms=8.0,
                    backward_ms=4.0,
                    step_overhead_ms=0.0,
                )
            ]
        )


def test_prefix_gate_is_fixed_at_thirty_percent():
    """Catches the numeric threshold drifting or authorizing without B200 evidence."""
    assert SEGMENTED_KV_GATE == 0.30
    below = summarize_samples(
        [sample(step_ms=100.0, prefix_decoder_ms=29.9, peak_bytes=10)]
    )
    boundary = summarize_samples(
        [sample(step_ms=100.0, prefix_decoder_ms=30.0, peak_bytes=10)]
    )

    assert below.segmented_kv_authorized is False
    assert below.prefix_threshold_met is False
    assert boundary.prefix_threshold_met is True
    assert boundary.segmented_kv_authorized is False


def representative_b200_metadata(**overrides):
    metadata = {
        "probe_kind": "stage2_no_cache",
        "device_name": "NVIDIA B200",
        "manifest_validated": True,
        "representative_quantiles": ["p50", "p95"],
        "warmup_count_per_representative": 3,
        "measured_count_per_representative": 10,
        "profile_uses_fixed_worker_count": True,
    }
    metadata.update(overrides)
    return metadata


def test_project_authorization_rejects_nonrepresentative_metadata():
    """Catches H100, self-test, or incomplete metadata authorizing KV work."""
    threshold_met = summarize_samples(
        [sample(step_ms=100.0, prefix_decoder_ms=40.0)]
    )

    assert authorize_segmented_kv(
        threshold_met, representative_b200_metadata(device_name="NVIDIA H100 NVL")
    ).segmented_kv_authorized is False
    assert authorize_segmented_kv(
        threshold_met, representative_b200_metadata(probe_kind="tiny_cuda_self_test")
    ).segmented_kv_authorized is False
    assert authorize_segmented_kv(
        threshold_met, representative_b200_metadata(manifest_validated=False)
    ).segmented_kv_authorized is False
    assert authorize_segmented_kv(
        summarize_samples([sample(step_ms=100.0, prefix_decoder_ms=29.9)]),
        representative_b200_metadata(),
    ).segmented_kv_authorized is False


def test_project_authorization_is_fail_closed_for_qualifying_b200_metadata():
    """Catches non-production probe metadata reopening the segmented-KV gate."""
    threshold_met = summarize_samples(
        [sample(step_ms=100.0, prefix_decoder_ms=40.0)]
    )

    assert threshold_met.prefix_threshold_met is True
    assert authorize_segmented_kv(
        threshold_met, representative_b200_metadata()
    ).segmented_kv_authorized is False


@pytest.mark.parametrize(
    "bad_sample",
    [
        sample(step_ms=math.nan),
        sample(prefix_decoder_ms=math.inf),
        sample(prefix_decoder_ms=-0.1),
        sample(step_ms=0.0),
        sample(peak_bytes=-1),
        sample(loss=math.nan),
        sample(gradient_norm=-0.1),
    ],
)
def test_aggregation_rejects_non_finite_or_negative_samples(bad_sample):
    """Catches corrupt timings, loss, gradients, or memory reaching gate evidence."""
    with pytest.raises(ValueError):
        summarize_samples([bad_sample])


def test_result_writer_is_strict_canonical_and_immutable(tmp_path):
    """Catches schema drift, traversal, overwrite, or non-canonical JSON output."""
    evidence = summarize_samples([sample()])
    metadata = {
        "device_name": "NVIDIA test GPU",
        "representative_quantile": "p50",
        "raw_index": 7,
    }

    path = write_efficiency_result(
        result_root=tmp_path,
        experiment_id="probe-1",
        evidence=evidence,
        metadata=metadata,
    )

    assert path == tmp_path / "probe-1" / "stage2-efficiency.json"
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert '": ' not in path.read_text(encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["experiment_id"] == "probe-1"
    assert payload["prefix_threshold_met"] is True
    assert payload["segmented_kv_authorized"] is False
    assert payload["metadata"] == metadata
    assert read_efficiency_result(path) == payload

    with pytest.raises(FileExistsError):
        write_efficiency_result(
            result_root=tmp_path,
            experiment_id="probe-1",
            evidence=evidence,
            metadata=metadata,
        )
    with pytest.raises(ValueError, match="one path component"):
        write_efficiency_result(
            result_root=tmp_path,
            experiment_id="../escape",
            evidence=evidence,
            metadata=metadata,
        )

    invalid_evidence = summarize_samples([sample()])
    object.__setattr__(invalid_evidence, "step_overhead_ms", 0.0)
    with pytest.raises(ValueError, match="same-clock wall partition"):
        write_efficiency_result(
            result_root=tmp_path,
            experiment_id="invalid-partition",
            evidence=invalid_evidence,
            metadata=metadata,
        )


def test_result_writer_rejects_true_authorization_even_with_qualifying_metadata(
    tmp_path,
):
    """Catches a forged evidence object persisting authorization as true."""
    evidence = summarize_samples(
        [sample(step_ms=100.0, prefix_decoder_ms=40.0)]
    )
    object.__setattr__(evidence, "segmented_kv_authorized", True)

    with pytest.raises(ValueError, match="segmented_kv_authorized must be false"):
        write_efficiency_result(
            result_root=tmp_path,
            experiment_id="forged-writer",
            evidence=evidence,
            metadata=representative_b200_metadata(),
        )

    assert not (tmp_path / "forged-writer" / "stage2-efficiency.json").exists()


def test_result_reader_rejects_true_authorization_even_with_qualifying_metadata(
    tmp_path,
):
    """Catches a forged schema-v1 payload being accepted as authorization."""
    path = tmp_path / "stage2-efficiency.json"
    evidence = summarize_samples(
        [sample(step_ms=100.0, prefix_decoder_ms=40.0)]
    )
    evidence_payload = vars(evidence).copy()
    evidence_payload["segmented_kv_authorized"] = True
    payload = {
        "schema_version": 1,
        "experiment_id": "forged-reader",
        "prefix_threshold_met": True,
        "segmented_kv_authorized": True,
        "evidence": evidence_payload,
        "metadata": representative_b200_metadata(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="segmented_kv_authorized must be false"):
        read_efficiency_result(path)


def test_result_reader_rejects_unknown_fields_and_non_finite_numbers(tmp_path):
    """Catches permissive evidence consumption that could misclassify KV authority."""
    path = tmp_path / "stage2-efficiency.json"
    path.write_text(
        '{"schema_version":1,"experiment_id":"x","segmented_kv_authorized":false,'
        '"evidence":{},"metadata":{},"unknown":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        read_efficiency_result(path)

    path.write_text(
        '{"schema_version":1,"experiment_id":"x","segmented_kv_authorized":false,'
        '"evidence":{"step_ms":NaN},"metadata":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid JSON constant"):
        read_efficiency_result(path)

    valid_path = write_efficiency_result(
        result_root=tmp_path,
        experiment_id="reader-partition",
        evidence=summarize_samples([sample()]),
        metadata={},
    )
    payload = json.loads(valid_path.read_text(encoding="utf-8"))
    payload["evidence"]["step_overhead_ms"] = 0.0
    valid_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="same-clock wall partition"):
        read_efficiency_result(valid_path)


def test_worker_benchmark_uses_two_warmups_then_one_short_retained_epoch():
    """Catches short datasets being recycled to twenty or warmups entering timing."""
    collated = []

    def collate(items):
        collated.append(items[0])
        return items[0]

    evidence = benchmark_worker_counts(
        dataset=[10, 20, 30],
        collate_fn=collate,
        worker_counts=(0,),
        warmup_batches=2,
        measured_batches=20,
    )

    assert collated == [10, 20, 10, 20, 30]
    assert len(evidence) == 1
    assert evidence[0].worker_count == 0
    assert evidence[0].measured_batches == 3
    assert math.isfinite(evidence[0].median_collation_ms)
    assert evidence[0].median_collation_ms >= 0.0
    assert evidence[0].within_host_capacity is True


def test_worker_benchmark_keeps_one_iterator_across_warmup_and_measurement():
    """Catches worker startup being repeated between warmup and measurement."""

    class CountingDataLoader(DataLoader):
        iterator_count = 0

        def __iter__(self):
            type(self).iterator_count += 1
            return super().__iter__()

    evidence = benchmark_worker_counts(
        dataset=[10, 20, 30],
        collate_fn=lambda items: items[0],
        worker_counts=(0,),
        warmup_batches=2,
        measured_batches=3,
        loader_factory=CountingDataLoader,
    )

    assert CountingDataLoader.iterator_count == 1
    assert evidence[0].measured_batches == 3


def test_worker_benchmark_captures_live_worker_processes_in_rss():
    """Catches host capacity sampling only the parent after workers exit."""
    evidence = benchmark_worker_counts(
        dataset=list(range(8)),
        collate_fn=lambda items: items[0],
        worker_counts=(2,),
        warmup_batches=2,
        measured_batches=2,
    )[0]

    assert len(evidence.worker_pids) == 2
    assert all(pid != os.getpid() for pid in evidence.worker_pids)
    assert evidence.peak_host_rss_bytes > 0


def test_worker_selection_uses_fastest_capacity_safe_count_and_stable_ties():
    """Catches an OOM-risk worker count winning or ties changing worker topology."""
    evidence_type = type(
        benchmark_worker_counts(
            dataset=[1],
            collate_fn=lambda items: items[0],
            worker_counts=(0,),
            warmup_batches=0,
            measured_batches=1,
        )[0]
    )
    values = (
        evidence_type(0, 1, 2.0, 100, True),
        evidence_type(2, 1, 1.0, 100, False),
        evidence_type(4, 1, 2.0, 100, True),
    )

    assert select_fastest_worker_count(values) == 0


class _TinyCudaRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(4, 4, bias=False)
        self.decoder = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 2, bias=False)

    def forward(self, values):
        encoded = self.visual(values)
        self.decoder(encoded[:, :1])
        self.decoder(encoded[:, :2])
        final = self.decoder(encoded)
        logits = self.lm_head(final)
        labels = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits.flatten(0, -2), labels.flatten())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
def test_cuda_step_classifies_last_decoder_call_and_always_removes_hooks():
    """Catches final-full misclassification or persistent instrumentation hooks."""
    model = _TinyCudaRuntime().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    baseline_hook_counts = tuple(
        (len(module._forward_pre_hooks), len(module._forward_hooks))
        for module in (model.decoder, model.visual, model.lm_head)
    )

    measured = measure_cuda_training_step(
        decoder=model.decoder,
        visual=model.visual,
        lm_head=model.lm_head,
        collate_fn=lambda: torch.ones(1, 3, 4, device="cuda"),
        loss_fn=model,
        optimizer=optimizer,
    )

    assert measured.decoder_input_lengths == (1, 2, 3)
    assert measured.prefix_input_length_sum == 3
    assert measured.step_ms > 0.0
    assert measured.prefix_decoder_ms >= 0.0
    assert measured.final_decoder_ms >= 0.0
    assert measured.shifted_ce_loss_ms > 0.0
    assert measured.forward_loss_ms > 0.0
    assert measured.backward_ms > 0.0
    assert measured.optimizer_ms >= 0.0
    assert measured.forward_other_ms >= 0.0
    assert measured.step_overhead_ms >= 0.0
    assert measured.step_ms >= measured.collation_ms
    assert measured.step_ms >= (
        measured.collation_ms
        + measured.forward_loss_ms
        + measured.backward_ms
        + measured.optimizer_ms
    )
    assert measured.peak_bytes > 0
    assert measured.loss >= 0.0
    assert measured.gradient_norm > 0.0
    assert tuple(
        (len(module._forward_pre_hooks), len(module._forward_hooks))
        for module in (model.decoder, model.visual, model.lm_head)
    ) == baseline_hook_counts

    with pytest.raises(RuntimeError, match="intentional"):
        measure_cuda_training_step(
            decoder=model.decoder,
            visual=model.visual,
            lm_head=model.lm_head,
            collate_fn=lambda: torch.ones(1, 3, 4, device="cuda"),
            loss_fn=lambda _batch: (_ for _ in ()).throw(RuntimeError("intentional")),
            optimizer=optimizer,
        )
    assert tuple(
        (len(module._forward_pre_hooks), len(module._forward_hooks))
        for module in (model.decoder, model.visual, model.lm_head)
    ) == baseline_hook_counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
def test_end_to_end_denominator_excludes_validation_only_gradient_sweep(monkeypatch):
    """Catches the validation-only FP32 gradient sweep entering step_ms."""
    model = _TinyCudaRuntime().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    original = efficiency_probe_module._gradient_norm_tensor

    def delayed_gradient_norm(optimizer):
        time.sleep(0.2)
        return original(optimizer)

    monkeypatch.setattr(
        efficiency_probe_module, "_gradient_norm_tensor", delayed_gradient_norm
    )
    measured = measure_cuda_training_step(
        decoder=model.decoder,
        visual=model.visual,
        lm_head=model.lm_head,
        collate_fn=lambda: torch.ones(1, 3, 4, device="cuda"),
        loss_fn=model,
        optimizer=optimizer,
    )

    assert measured.step_ms < 100.0


def test_representative_loader_uses_explicit_fixed_worker_count_and_real_collator():
    """Catches GPU profiling bypassing its configured DataLoader/collator workers."""
    collated = []
    loader = build_representative_dataloader(
        example=7,
        collate_fn=lambda items: collated.append(items[0]) or items[0],
        worker_count=0,
        batch_count=3,
    )

    assert list(loader) == [7, 7, 7]
    assert collated == [7, 7, 7]
    assert loader.num_workers == 0


def test_cli_rejects_custom_roots_and_requires_fixed_profile_worker_count():
    """Catches production artifacts escaping canonical roots or runtime autotuning."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--experiment-id", "x", "--result-root", "/tmp/results", "--self-test"]
        )
    args = parser.parse_args(["--experiment-id", "x", "--self-test"])
    assert not hasattr(args, "result_root")
    assert not hasattr(args, "log_root")
    assert args.dataloader_num_workers is None

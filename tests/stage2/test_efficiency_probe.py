import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stage2.efficiency_probe import (
    SEGMENTED_KV_GATE,
    Stage2EfficiencySample,
    benchmark_worker_counts,
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
                step_ms=10.0,
                prefix_decoder_ms=1.0,
                peak_bytes=100,
                collation_ms=3.0,
                final_decoder_ms=4.0,
                visual_ms=5.0,
                lm_head_ms=6.0,
                loss=7.0,
                gradient_norm=8.0,
                prefix_input_length_sum=9,
            ),
            sample(
                step_ms=100.0,
                prefix_decoder_ms=40.0,
                peak_bytes=200,
                collation_ms=13.0,
                final_decoder_ms=14.0,
                visual_ms=15.0,
                lm_head_ms=16.0,
                loss=17.0,
                gradient_norm=18.0,
                prefix_input_length_sum=19,
            ),
            sample(
                step_ms=1000.0,
                prefix_decoder_ms=900.0,
                peak_bytes=300,
                collation_ms=23.0,
                final_decoder_ms=24.0,
                visual_ms=25.0,
                lm_head_ms=26.0,
                loss=27.0,
                gradient_norm=28.0,
                prefix_input_length_sum=29,
            ),
        ]
    )

    assert evidence.sample_count == 3
    assert evidence.step_ms == 100.0
    assert evidence.prefix_decoder_ms == 40.0
    assert evidence.prefix_decoder_share == 0.4
    assert evidence.peak_bytes == 200
    assert evidence.collation_ms == 13.0
    assert evidence.final_decoder_ms == 14.0
    assert evidence.visual_ms == 15.0
    assert evidence.lm_head_ms == 16.0
    assert evidence.loss == 17.0
    assert evidence.gradient_norm == 18.0
    assert evidence.prefix_input_length_sum == 19
    assert evidence.segmented_kv_authorized is True


def test_prefix_gate_is_fixed_at_thirty_percent():
    """Catches the authorization threshold drifting below the approved 0.30 share."""
    assert SEGMENTED_KV_GATE == 0.30
    below = summarize_samples(
        [sample(step_ms=100.0, prefix_decoder_ms=29.9, peak_bytes=10)]
    )
    boundary = summarize_samples(
        [sample(step_ms=100.0, prefix_decoder_ms=30.0, peak_bytes=10)]
    )

    assert below.segmented_kv_authorized is False
    assert boundary.segmented_kv_authorized is True


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
    assert payload["segmented_kv_authorized"] is True
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
        return self.lm_head(final).square().mean()


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

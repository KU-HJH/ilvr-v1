import json
import numpy as np
from evaluation.latent_image_report import build_report
from tests.evaluation.latent_image_fixtures import write_synthetic_run


def test_report_sections_and_not_run(tmp_path):
    write_synthetic_run(tmp_path, questions=3, blocks=2, with_freegen=False)
    summary = build_report(tmp_path)
    assert set(summary) >= {"attention", "knockout", "patching", "content", "freegen", "gates"}
    assert summary["freegen"] == {"status": "NOT RUN"}
    assert "NOT RUN" in (tmp_path / "report.md").read_text()
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert (tmp_path / "figures" / "attention_layers.png").exists()


def test_attention_detects_image_preference_and_identification(tmp_path):
    write_synthetic_run(tmp_path, questions=12, blocks=2, image_boost=0.5)
    summary = build_report(tmp_path)
    assert summary["attention"]["significant_layers"] == [0, 1, 2, 3]
    assert summary["content"]["identification"]["mid"]["mean"] == 1.0


def test_freegen_mcnemar(tmp_path):
    write_synthetic_run(tmp_path, questions=4, with_freegen=True)
    fg = build_report(tmp_path)["freegen"]
    assert fg["correct"] == {"causal_mask": 4, "latent_mask": 1}
    assert fg["discordant"] == {"causal_only": 3, "latent_only": 0}


def test_attention_handles_empty_post_control(tmp_path):
    # Review Focus edge case: no text after the last block in any question, so the
    # post-text control group is empty for every (question, block) unit. build_report
    # must not crash; the pad-post diff is reported as no-data and significant_layers
    # is [] since it can no longer be computed from both controls.
    write_synthetic_run(tmp_path, questions=3, blocks=2, omit_post=True)
    summary = build_report(tmp_path)
    assert summary["attention"]["status"] == "RUN"
    assert summary["attention"]["differences"]["pad_minus_post"] == {
        "status": "NO DATA", "reason": "no post-text units in any block"}
    assert summary["attention"]["significant_layers"] == []
    assert summary["attention"]["significant_layers_missing_controls"] == ["post"]
    # pad-pre is unaffected and still a per-layer list.
    assert isinstance(summary["attention"]["differences"]["pad_minus_pre"], list)


def test_knockout_handles_empty_k_random(tmp_path):
    # No question has a k_random result (e.g. every eligible key set was empty), so the
    # k_all - k_random paired difference has an empty intersection.
    write_synthetic_run(tmp_path, questions=3, blocks=2, omit_k_random=True)
    summary = build_report(tmp_path)
    assert summary["knockout"]["status"] == "RUN"
    diff = summary["knockout"]["k_all_minus_k_random_latent_cos"]
    assert {k: diff[k] for k in ("status", "reason")} == {
        "status": "NO DATA", "reason": "no question has both k_all and k_random latent_cos"}
    assert diff["scope"] == "size-matched blocks only"


# ---------------------------------------------------------------------------
# Final-review fix wave (R13, R14, R16, R17, R18, R19)
# ---------------------------------------------------------------------------

def _edit_interventions(run_dir, q, fn):
    path = run_dir / "interventions" / f"q{q}.json"
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def _edit_attention_meta(run_dir, q, **fields):
    path = run_dir / "attention" / f"q{q}.json"
    doc = json.loads(path.read_text())
    doc.update(fields)
    path.write_text(json.dumps(doc))


def _set_metric(run_dir, q, metric, name, value):
    _edit_interventions(run_dir, q, lambda d: d[metric].__setitem__(name, value))


def test_k_random_is_size_matched_blocks_only(tmp_path):
    # R13: q0 block 1 is not size-matched (fewer eligible keys than image tokens), so
    # its huge k_random value must not enter the size-matched row or the k_all diff.
    write_synthetic_run(tmp_path, questions=3, blocks=2)
    _edit_interventions(tmp_path, 0, lambda d: d.__setitem__("random_matched", {"0": 1.0, "1": 0.5}))
    for q, values in enumerate(([0.2, 100.0], [0.4, 0.6], [0.8, 1.0])):
        _set_metric(tmp_path, q, "latent_cos", "k_random", values)
        _set_metric(tmp_path, q, "latent_cos", "k_all", [1.0, 1.0])
    ko = build_report(tmp_path)["knockout"]
    row = ko["table"]["k_random"]
    assert row["scope"] == "size-matched blocks only"
    assert (row["matched_blocks"], row["total_blocks"]) == (5, 6)
    assert np.isclose(row["latent_cos"]["mean"], np.mean([0.2, 0.5, 0.9]))
    assert np.isclose(row["latent_cos_first_block"]["mean"], np.mean([0.2, 0.4, 0.8]))
    # answer-level (trajectory) metrics only from questions whose blocks are all matched
    assert row["answer_dnll"]["n_clusters"] == 2
    unmatched = ko["table"]["k_random_all_blocks_not_size_matched"]
    assert unmatched["scope"] == "all blocks, not size-matched"
    assert np.isclose(unmatched["latent_cos"]["mean"], np.mean([50.1, 0.5, 0.9]))
    diff = ko["k_all_minus_k_random_latent_cos"]
    assert diff["scope"] == "size-matched blocks only"
    assert (diff["matched_blocks"], diff["total_blocks"]) == (5, 6)
    assert np.isclose(diff["mean"], 1.0 - np.mean([0.2, 0.5, 0.9]))


def test_pooled_statistics_weight_questions_equally(tmp_path):
    # R14: q2 has 6 blocks, q0/q1 have 1. Block-weighted pooling would give 6/8; the
    # question-equal mean is the mean of per-question means (1/3).
    write_synthetic_run(tmp_path, questions=3, blocks=[1, 1, 6])
    for q, value in ((0, 0.0), (1, 0.0), (2, 1.0)):
        n = 6 if q == 2 else 1
        _set_metric(tmp_path, q, "latent_cos", "k_all", [value] * n)
        _set_metric(tmp_path, q, "latent_cos", "blur_total", [value] * n)
        path = tmp_path / "attention" / f"q{q}.npz"
        meta = json.loads((tmp_path / "attention" / f"q{q}.json").read_text())
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        share = arrays["vw_share"].astype(np.float32)
        pad_rows = [i for i, s in enumerate(meta["specs"]) if s["slot"].startswith("pad")]
        share[pad_rows, :, :, 2] = 0.1 if q < 2 else 0.7
        arrays["vw_share"] = share.astype(np.float16)
        np.savez(path, **arrays)
    summary = build_report(tmp_path)
    k_all = summary["knockout"]["table"]["k_all"]["latent_cos"]
    assert np.isclose(k_all["mean"], 1 / 3) and k_all["n_clusters"] == 3
    assert np.isclose(summary["patching"]["by_corruption"]["blur"]["table"]["blur_total"]["latent_cos"]["mean"], 1 / 3)
    pad_curve = summary["attention"]["vw_image_share"]["pad"][0]
    assert np.isclose(pad_curve["mean"], (0.1 + 0.1 + 0.7) / 3, atol=1e-3)


def test_first_block_direct_ratio_is_primary(tmp_path):
    # R16: first-block ratio isolates block 0; all-block ratio is labelled as propagated.
    write_synthetic_run(tmp_path, questions=3, blocks=2)
    for q in range(3):
        _set_metric(tmp_path, q, "latent_cos", "blur_direct", [0.2, 5.0])
        _set_metric(tmp_path, q, "latent_cos", "blur_total", [0.4, 5.0])
        _set_metric(tmp_path, q, "latent_cos", "mean_direct", [0.3, 1.0])
        _set_metric(tmp_path, q, "latent_cos", "mean_total", [0.0, 1.0])
    by = build_report(tmp_path)["patching"]["by_corruption"]
    assert np.isclose(by["blur"]["direct_ratio_latent_cos_first_block"]["mean"], 0.5)
    assert np.isclose(by["blur"]["direct_ratio_latent_cos"]["mean"], 5.2 / 5.4)
    assert by["blur"]["labels"]["direct_ratio_latent_cos"] == "all blocks, includes propagation"
    assert "primary" in by["blur"]["labels"]["direct_ratio_latent_cos_first_block"]
    # R11: a zero first-block denominator is reported as unstable, not inf/nan.
    assert by["mean"]["direct_ratio_latent_cos_first_block"]["reason"] == "unstable denominator"
    assert by["mean"]["direct_ratio_latent_cos_first_block"]["mean"] is None


def test_sensitivity_excluding_truncated_questions(tmp_path):
    write_synthetic_run(tmp_path, questions=3, blocks=1, stop_reasons=["eos", "eos", "max_new_tokens"])
    for q, value in ((0, 0.2), (1, 0.4), (2, 9.0)):
        _set_metric(tmp_path, q, "latent_cos", "k_all", [value])
        _set_metric(tmp_path, q, "answer_dnll", "swap_direct", value)
    sens = build_report(tmp_path)["sensitivity_excluding_truncated"]
    assert sens["excluded_truncated"] == [2]
    assert np.isclose(sens["table"]["k_all"]["latent_cos"]["mean"], 0.3)
    assert np.isclose(sens["table"]["swap_direct"]["answer_dnll"]["mean"], 0.3)
    assert set(sens["table"]) == {"k_all"} | {f"{k}_{p}" for k in ("blur", "mean", "swap")
                                              for p in ("direct", "total")}


def test_attention_gate_failure_flags_section(tmp_path):
    write_synthetic_run(tmp_path, questions=3, blocks=2)
    assert build_report(tmp_path)["attention"]["status"] == "RUN"
    _edit_attention_meta(tmp_path, 1, g2=5e-3)
    summary = build_report(tmp_path)
    assert summary["attention"]["status"] == "GATE FAILED"
    assert summary["attention"]["gate_failures"] == ["G2"]
    assert "GATE FAILED" in (tmp_path / "report.md").read_text()
    _edit_attention_meta(tmp_path, 1, g2=1e-4, g1=0.05, g1b=0.02)
    assert build_report(tmp_path)["attention"]["gate_failures"] == ["G1", "G1b"]


def test_report_records_report_provenance(tmp_path):
    write_synthetic_run(tmp_path, questions=3, blocks=1)
    summary = build_report(tmp_path, provenance={"report_commit": "abc123", "report_dirty": False})
    assert summary["report_commit"] == "abc123" and summary["report_dirty"] is False


def test_eval_baseline_counts_freegen_question_set(tmp_path):
    # R19: the baseline is counted over the freegen set (all selected questions), not the
    # attention metas (which omit G5/length-budget exclusions).
    write_synthetic_run(tmp_path, questions=4, with_freegen=True, eval_matches=[True, True, True, False])
    (tmp_path / "attention" / "q3.json").unlink()
    (tmp_path / "attention" / "q3.npz").unlink()
    fg = build_report(tmp_path)["freegen"]
    assert fg["eval_baseline_correct"] == 3 and fg["eval_baseline_questions"] == 4

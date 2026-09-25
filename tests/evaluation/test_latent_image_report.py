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
    assert summary["knockout"]["k_all_minus_k_random_latent_cos"] == {
        "status": "NO DATA", "reason": "no question has both k_all and k_random latent_cos"}

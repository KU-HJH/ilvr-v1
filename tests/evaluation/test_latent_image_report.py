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

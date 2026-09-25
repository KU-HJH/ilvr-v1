import json
import pytest
import torch
from evaluation import latent_image_cli as cli
from evaluation.latent_image_interventions import ConditionResult
from evaluation.latent_image_layout import LayoutTokenIds, build_layout

T = LayoutTokenIds(80, 81, 83, 84, 91, 90, 92, (70, 71))
PROMPT = [1, 2, 81, 80, 80, 83, 5, 6, 84, 7, 8]


# ---------------------------------------------------------------------------
# Step 1 / helper tests
# ---------------------------------------------------------------------------

def test_shard_assignment_is_disjoint_and_complete():
    items = tuple(range(10))
    shards = [cli.shard_items(items, i, 3) for i in range(3)]
    assert sorted(x for s in shards for x in s) == list(items)


def test_manifest_mismatch_aborts(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"inputs": {"data": "aaa"}, "commit": "c1"}))
    with pytest.raises(ValueError, match="manifest"):
        cli.require_manifest(tmp_path, {"data": "bbb"}, "c1")
    with pytest.raises(ValueError, match="manifest"):
        cli.require_manifest(tmp_path, {"data": "aaa"}, "c2")
    cli.require_manifest(tmp_path, {"data": "aaa"}, "c1")


def test_g5_exclusion_rules():
    ok = cli.g5_verdict(matches=[True] * 200, first_block_offset=5)
    assert ok["excluded"] is False
    early = cli.g5_verdict(matches=[True, False] + [True] * 198, first_block_offset=5)
    assert early["excluded"] and early["reason"] == "mismatch_before_first_block"
    low = cli.g5_verdict(matches=[True] * 10 + [False] * 5 + [True] * 85, first_block_offset=5)
    assert low["excluded"] and low["reason"] == "match_rate_below_0.99"


def test_too_many_exclusions_abort():
    with pytest.raises(RuntimeError, match="exclusions"):
        cli.require_exclusion_budget(11, limit=10)
    cli.require_exclusion_budget(10, limit=10)


def test_head_selection_top_fraction():
    import numpy as np
    share = np.zeros((36, 16)); share[3, 4] = 1; share[10, 2] = .5
    heads = cli.select_heads(share, fraction=0.05)
    assert len(heads) == 29 and heads[0] == (3, 4) and heads[1] == (10, 2)


# ---------------------------------------------------------------------------
# Step 5 / per-question logic tests
# ---------------------------------------------------------------------------

def fake(name, shift=0.0, flip_at=None):
    lay = build_layout(PROMPT + [10, 91, 90, 90, 92, 13, 14, 70, 71, 5, 84], len(PROMPT), T, latent_size=2)
    scores = {p: ((1 if p == flip_at else 0), 1.0 + shift) for p in lay.scored_positions}
    lat = torch.ones(1, 2, 4) + shift * torch.tensor([1.0, -1, 0, 0])
    hidden = {p: torch.ones(4) + shift for p in lay.pre_text[0] + lay.post_text[0]}
    return lay, ConditionResult(name, lat, torch.ones(1, 4), torch.ones(1, 4), torch.ones(4), torch.ones(4),
                                hidden, scores)


def test_compare_runs_gate():
    _, a = fake("noop")
    _, b = fake("reference")
    assert cli.compare_runs(a, b)["pass"]
    _, c = fake("reference", flip_at=16)
    assert not cli.compare_runs(c, a)["pass"]


def test_summarize_question_deltas():
    lay, base = fake("noop")
    _, other = fake("k_all", shift=0.5, flip_at=16)
    out = cli.summarize_question({"noop": base, "k_all": other}, lay, {0: 1.0})
    assert out["post_dnll"]["k_all"] == [0.5] and out["post_flip"]["k_all"][0] == pytest.approx(0.5)
    assert out["answer_dnll"]["k_all"] == pytest.approx(0.5)
    assert out["latent_cos"]["k_all"][0] > 0 and "noop" not in out["latent_cos"]


# ---------------------------------------------------------------------------
# Step 8 / CLI driver tests: argument parsing, resume/skip, heatmap selection,
# and a full `prepare` run against a tiny synthetic dataset in tmp_path.
# ---------------------------------------------------------------------------

def test_help_lists_stages(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for stage in cli.STAGES:
        assert stage in out


def test_stage_subcommand_requires_selection_and_eval_rows():
    with pytest.raises(SystemExit):
        cli.main(["prepare", "--run-dir", "/tmp/does-not-matter"])


def test_heatmap_questions_covers_all_four_categories_and_pads_to_six():
    selection = (10, 20, 30, 40, 50, 60, 70)
    categories = {10: "creation", 20: "deletion", 30: "selection", 40: "update",
                  50: "creation", 60: "deletion", 70: "selection"}
    chosen = cli.heatmap_questions(selection, categories)
    assert chosen[:4] == (10, 20, 30, 40)
    assert len(chosen) == 6
    assert all(i in selection for i in chosen[4:])
    assert len(set(chosen)) == len(chosen)


def test_skip_existing_detects_any_existing_file(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert not cli.skip_existing(a, b)
    a.write_text("{}")
    assert cli.skip_existing(a, b)


def test_write_json_atomic_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "out" / "q1.json"
    cli.write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert not list(target.parent.glob("*.tmp"))


def test_savez_atomic_leaves_no_tmp_file(tmp_path):
    import numpy as np
    target = tmp_path / "out" / "q1.npz"
    cli.savez_atomic(target, x=np.array([1, 2, 3]))
    with np.load(target) as data:
        assert list(data["x"]) == [1, 2, 3]
    assert not list(target.parent.glob("*.tmp*"))


def _write_jsonl(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_prepare_fixture(tmp_path):
    data_root = tmp_path / "data"
    categories = ("creation", "deletion", "selection", "update")
    rows, indices = [], []
    idx = 0
    for category in categories:
        (data_root / "images_comt" / category).mkdir(parents=True)
        for i in range(5):
            (data_root / "images_comt" / category / f"{i}.png").write_bytes(b"\x89PNG\r\n")
            rows.append({"text_input": "q", "image_input": [f"images_comt/{category}/{i}.png"],
                        "original_final_answer": "1"})
            indices.append(idx)
            idx += 1

    data_path = tmp_path / "TEST.jsonl"
    _write_jsonl(data_path, rows)

    selection_path = tmp_path / "metadata.json"
    selection_path.write_text(json.dumps({"selection": {"requested_indices": indices, "requested_count": len(indices)}}))

    eval_rows_path = tmp_path / "merged.jsonl"
    _write_jsonl(eval_rows_path, [{"index": i} for i in indices])

    model_dir = tmp_path / "model" / "prepared-checkpoint-1"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "model" / "prepared-checkpoint-1.manifest.json").write_text("{}")
    base_model_dir = tmp_path / "base"
    base_model_dir.mkdir()

    return {
        "data": data_path, "data_root": data_root, "selection": selection_path, "eval_rows": eval_rows_path,
        "model_dir": model_dir, "base_model_dir": base_model_dir, "indices": tuple(indices), "categories": categories,
    }


def test_prepare_end_to_end(tmp_path, monkeypatch):
    fixture = _build_prepare_fixture(tmp_path)
    monkeypatch.setattr(cli, "EXPECTED_SHA256", {
        "data": cli.sha256(fixture["data"]), "selection": cli.sha256(fixture["selection"]),
        "eval_rows": cli.sha256(fixture["eval_rows"])})
    run_dir = tmp_path / "run"

    code = cli.main([
        "prepare", "--model-dir", str(fixture["model_dir"]), "--base-model-dir", str(fixture["base_model_dir"]),
        "--data", str(fixture["data"]), "--selection", str(fixture["selection"]),
        "--eval-rows", str(fixture["eval_rows"]), "--data-root", str(fixture["data_root"]),
        "--run-dir", str(run_dir)])
    assert code == 0

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["inputs"] == {"data": cli.sha256(fixture["data"]), "selection": cli.sha256(fixture["selection"]),
                                  "eval_rows": cli.sha256(fixture["eval_rows"])}
    assert manifest["commit"] == cli.git_commit()
    assert manifest["adapter_sha256"] == cli.sha256(fixture["model_dir"] / "adapter_model.safetensors")
    assert manifest["prepared_manifest_sha256"] == cli.sha256(
        tmp_path / "model" / "prepared-checkpoint-1.manifest.json")

    swaps = json.loads((run_dir / "swaps.json").read_text())
    assert set(swaps["questions"]) == {str(i) for i in fixture["indices"]}
    for idx in fixture["indices"]:
        entry = swaps["questions"][str(idx)]
        assert entry["category"] in fixture["categories"]
        assert len(entry["swaps"]) == 4
        assert idx not in entry["swaps"]
        assert all(swaps["questions"][str(j)]["category"] == entry["category"] for j in entry["swaps"])

    # require_manifest, used by every later stage, accepts the recorded state and
    # rejects a changed commit or input hash.
    cli.require_manifest(run_dir, manifest["inputs"], manifest["commit"])
    with pytest.raises(ValueError, match="manifest"):
        cli.require_manifest(run_dir, manifest["inputs"], "not-the-real-commit")

    # A fresh run directory is required; prepare refuses to overwrite a manifest.
    with pytest.raises(ValueError, match="manifest already exists"):
        cli.stage_prepare(cli._parser().parse_args([
            "prepare", "--model-dir", str(fixture["model_dir"]), "--base-model-dir", str(fixture["base_model_dir"]),
            "--data", str(fixture["data"]), "--selection", str(fixture["selection"]),
            "--eval-rows", str(fixture["eval_rows"]), "--data-root", str(fixture["data_root"]),
            "--run-dir", str(run_dir)]))


def test_prepare_rejects_hash_mismatch(tmp_path, monkeypatch):
    fixture = _build_prepare_fixture(tmp_path)
    monkeypatch.setattr(cli, "EXPECTED_SHA256", {
        "data": "0" * 64, "selection": cli.sha256(fixture["selection"]), "eval_rows": cli.sha256(fixture["eval_rows"])})
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="sha256"):
        cli.main([
            "prepare", "--model-dir", str(fixture["model_dir"]), "--base-model-dir", str(fixture["base_model_dir"]),
            "--data", str(fixture["data"]), "--selection", str(fixture["selection"]),
            "--eval-rows", str(fixture["eval_rows"]), "--data-root", str(fixture["data_root"]),
            "--run-dir", str(run_dir)])
    assert not (run_dir / "manifest.json").exists()

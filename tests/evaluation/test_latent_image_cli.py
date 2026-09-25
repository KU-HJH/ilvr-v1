import argparse
import json
from pathlib import Path

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
    monkeypatch.setattr(cli, "git_status_porcelain", lambda: "")   # tests run from a working tree
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
    monkeypatch.setattr(cli, "git_status_porcelain", lambda: "")
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="sha256"):
        cli.main([
            "prepare", "--model-dir", str(fixture["model_dir"]), "--base-model-dir", str(fixture["base_model_dir"]),
            "--data", str(fixture["data"]), "--selection", str(fixture["selection"]),
            "--eval-rows", str(fixture["eval_rows"]), "--data-root", str(fixture["data_root"]),
            "--run-dir", str(run_dir)])
    assert not (run_dir / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Fix round 1: stage-ordering guards (smoke gate, exclusion budget, attention
# completeness, model identity on resume).
# ---------------------------------------------------------------------------

def _model_dirs(tmp_path):
    model_dir = tmp_path / "model" / "prepared-checkpoint-1"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "model" / "prepared-checkpoint-1.manifest.json").write_text("{}")
    base_model_dir = tmp_path / "base"
    base_model_dir.mkdir()
    return model_dir, base_model_dir


def _write_manifest(run_dir, data_path, selection_path, eval_rows_path, model_dir=None, base_model_dir=None):
    inputs = {"data": cli.sha256(data_path), "selection": cli.sha256(selection_path),
             "eval_rows": cli.sha256(eval_rows_path)}
    manifest = {"inputs": inputs, "commit": cli.git_commit()}
    if model_dir is not None:
        manifest.update(
            model_dir=str(model_dir), base_model_dir=str(base_model_dir),
            prepared_manifest_sha256=cli.sha256(Path(str(model_dir) + ".manifest.json")),
            adapter_sha256=cli.sha256(model_dir / "adapter_model.safetensors"))
    cli.write_json_atomic(run_dir / "manifest.json", manifest)
    return manifest


def _minimal_inputs(tmp_path, indices):
    data_path = tmp_path / "TEST.jsonl"
    data_path.write_text("{}\n")
    selection_path = tmp_path / "metadata.json"
    selection_path.write_text(json.dumps({"selection": {"requested_indices": indices}}))
    eval_rows_path = tmp_path / "merged.jsonl"
    eval_rows_path.write_text("")
    return data_path, selection_path, eval_rows_path


def _stage_args(stage, data_path, selection_path, eval_rows_path, run_dir, model_dir=None, base_model_dir=None):
    argv = [stage, "--data", str(data_path), "--selection", str(selection_path),
           "--eval-rows", str(eval_rows_path), "--run-dir", str(run_dir)]
    if model_dir is not None:
        argv += ["--model-dir", str(model_dir), "--base-model-dir", str(base_model_dir)]
    return cli._parser().parse_args(argv)


# --- fix 4: model identity -------------------------------------------------

def test_require_model_identity_accepts_matching_state(tmp_path):
    model_dir, base_model_dir = _model_dirs(tmp_path)
    manifest = {
        "model_dir": str(model_dir), "base_model_dir": str(base_model_dir),
        "adapter_sha256": cli.sha256(model_dir / "adapter_model.safetensors"),
        "prepared_manifest_sha256": cli.sha256(Path(str(model_dir) + ".manifest.json")),
    }
    args = argparse.Namespace(model_dir=model_dir, base_model_dir=base_model_dir)
    cli.require_model_identity(manifest, args)  # does not raise


def test_require_model_identity_rejects_different_model_dir(tmp_path):
    model_dir, base_model_dir = _model_dirs(tmp_path)
    manifest = {
        "model_dir": str(model_dir), "base_model_dir": str(base_model_dir),
        "adapter_sha256": cli.sha256(model_dir / "adapter_model.safetensors"),
        "prepared_manifest_sha256": cli.sha256(Path(str(model_dir) + ".manifest.json")),
    }
    other_dir = tmp_path / "model" / "prepared-checkpoint-2"
    other_dir.mkdir(parents=True)
    (other_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "model" / "prepared-checkpoint-2.manifest.json").write_text("{}")
    args = argparse.Namespace(model_dir=other_dir, base_model_dir=base_model_dir)
    with pytest.raises(ValueError, match="manifest"):
        cli.require_model_identity(manifest, args)


def test_require_model_identity_rejects_changed_adapter_hash(tmp_path):
    model_dir, base_model_dir = _model_dirs(tmp_path)
    manifest = {
        "model_dir": str(model_dir), "base_model_dir": str(base_model_dir),
        "adapter_sha256": cli.sha256(model_dir / "adapter_model.safetensors"),
        "prepared_manifest_sha256": cli.sha256(Path(str(model_dir) + ".manifest.json")),
    }
    args = argparse.Namespace(model_dir=model_dir, base_model_dir=base_model_dir)
    cli.require_model_identity(manifest, args)  # matches before the edit

    (model_dir / "adapter_model.safetensors").write_bytes(b"a-different-checkpoint-entirely")
    with pytest.raises(ValueError, match="manifest"):
        cli.require_model_identity(manifest, args)


# --- fix 1: smoke gate required before attention ---------------------------

def test_require_smoke_pass(tmp_path):
    with pytest.raises(RuntimeError, match="smoke"):
        cli.require_smoke_pass(tmp_path)
    cli.write_json_atomic(tmp_path / "smoke" / "gates.json", {"pass": False})
    with pytest.raises(RuntimeError, match="smoke"):
        cli.require_smoke_pass(tmp_path)
    cli.write_json_atomic(tmp_path / "smoke" / "gates.json", {"pass": True})
    cli.require_smoke_pass(tmp_path)  # does not raise


def test_stage_attention_refuses_without_passing_smoke_gate(tmp_path):
    data_path, selection_path, eval_rows_path = _minimal_inputs(tmp_path, [0])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_dir, base_model_dir = _model_dirs(tmp_path)
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path, model_dir, base_model_dir)
    args = _stage_args("attention", data_path, selection_path, eval_rows_path, run_dir, model_dir, base_model_dir)

    with pytest.raises(RuntimeError, match="smoke"):
        cli.stage_attention(args)

    cli.write_json_atomic(run_dir / "smoke" / "gates.json", {"pass": False})
    with pytest.raises(RuntimeError, match="smoke"):
        cli.stage_attention(args)


# --- fix 3: attention completeness -----------------------------------------

def test_require_attention_complete(tmp_path):
    cli.write_json_atomic(tmp_path / "attention" / "q0.json", {})
    cli.write_json_atomic(tmp_path / "attention" / "excluded" / "q1.json", {})
    cli.require_attention_complete(tmp_path, [0, 1])  # does not raise
    with pytest.raises(RuntimeError, match="incomplete"):
        cli.require_attention_complete(tmp_path, [0, 1, 2])


def test_excluded_indices(tmp_path):
    cli.write_json_atomic(tmp_path / "attention" / "excluded" / "q3.json", {})
    cli.write_json_atomic(tmp_path / "attention" / "excluded" / "q7.json", {})
    assert cli.excluded_indices(tmp_path) == {3, 7}


def test_stage_heads_refuses_incomplete_attention(tmp_path):
    data_path, selection_path, eval_rows_path = _minimal_inputs(tmp_path, [0, 1, 2])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path)
    # only question 0 is present; 1 and 2 are neither attended nor excluded.
    cli.write_json_atomic(run_dir / "attention" / "q0.json",
                          {"specs": [], "g1": 0.0, "g1b": 0.0, "g2": 0.0, "g5": {},
                           "category": "x", "n_blocks": 1, "eval_match": True})
    args = _stage_args("heads", data_path, selection_path, eval_rows_path, run_dir)
    with pytest.raises(RuntimeError, match="incomplete"):
        cli.stage_heads(args)


def test_stage_interventions_refuses_incomplete_attention(tmp_path):
    data_path, selection_path, eval_rows_path = _minimal_inputs(tmp_path, [0, 1, 2])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_dir, base_model_dir = _model_dirs(tmp_path)
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path, model_dir, base_model_dir)
    cli.write_json_atomic(run_dir / "attention" / "excluded" / "q0.json", {"reason": "match_rate_below_0.99"})
    # 1 and 2 are missing entirely.
    args = _stage_args("interventions", data_path, selection_path, eval_rows_path, run_dir,
                       model_dir, base_model_dir)
    with pytest.raises(RuntimeError, match="incomplete"):
        cli.stage_interventions(args)


# --- fix 2: exclusion budget enforced downstream too ------------------------

def test_stage_heads_refuses_over_exclusion_budget(tmp_path):
    indices = list(range(11))
    data_path, selection_path, eval_rows_path = _minimal_inputs(tmp_path, indices)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path)
    for idx in indices:
        cli.write_json_atomic(run_dir / "attention" / "excluded" / f"q{idx}.json", {"reason": "match_rate_below_0.99"})
    args = _stage_args("heads", data_path, selection_path, eval_rows_path, run_dir)
    with pytest.raises(RuntimeError, match="exclusions"):
        cli.stage_heads(args)


def test_stage_interventions_refuses_over_exclusion_budget(tmp_path):
    indices = list(range(11))
    data_path, selection_path, eval_rows_path = _minimal_inputs(tmp_path, indices)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_dir, base_model_dir = _model_dirs(tmp_path)
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path, model_dir, base_model_dir)
    for idx in indices:
        cli.write_json_atomic(run_dir / "attention" / "excluded" / f"q{idx}.json", {"reason": "match_rate_below_0.99"})
    args = _stage_args("interventions", data_path, selection_path, eval_rows_path, run_dir,
                       model_dir, base_model_dir)
    with pytest.raises(RuntimeError, match="exclusions"):
        cli.stage_interventions(args)


# --- bonus: stage_heads pooling arithmetic on fabricated npz/json files ----

def test_stage_heads_pools_pad_rows_across_questions(tmp_path):
    import numpy as np
    data_path, selection_path, eval_rows_path = _minimal_inputs(tmp_path, [0, 1, 2, 3, 4])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path)

    attention_dir = run_dir / "attention"
    layers, heads_n, regions = 2, 2, 7
    specs = [{"position": 0, "group": "latent", "block": 0, "slot": "start"},
            {"position": 1, "group": "latent", "block": 0, "slot": "pad1"},
            {"position": 2, "group": "post_text", "block": 0, "slot": "post"}]

    def write_question(idx, pad_value):
        vw = np.zeros((3, layers, heads_n, regions), dtype=np.float32)
        vw[1, ..., 2] = pad_value  # pad1 row's IMAGE-region (index 2) share
        cli.savez_atomic(attention_dir / f"q{idx}.npz", mass=vw, vw_share=vw,
                         enrichment=np.zeros((3, layers, heads_n), dtype=np.float32))
        cli.write_json_atomic(attention_dir / f"q{idx}.json",
                              {"specs": specs, "g1": 0.0, "g1b": 0.0, "g2": 0.0, "g5": {},
                               "category": "creation", "n_blocks": 1, "eval_match": True})

    write_question(0, 0.9)
    write_question(1, 0.3)
    for idx in (2, 3, 4):
        cli.write_json_atomic(attention_dir / "excluded" / f"q{idx}.json", {"reason": "match_rate_below_0.99"})

    args = _stage_args("heads", data_path, selection_path, eval_rows_path, run_dir)
    assert cli.stage_heads(args) == 0

    heads_doc = json.loads((run_dir / "heads.json").read_text())
    share = np.array(heads_doc["share"])
    assert share.shape == (layers, heads_n)
    assert np.allclose(share, 0.6)  # (0.9 + 0.3) / 2, uniform across layers/heads


# ---------------------------------------------------------------------------
# Final-review fix wave: length budget (R12), question-equal heads (R14),
# smoke selection (R15), report commit (R17), freegen eval_match (R19).
# ---------------------------------------------------------------------------

def test_dual_length_and_length_budget_exclusion():
    lay = build_layout(PROMPT + [10, 91, 90, 90, 92, 13, 84], len(PROMPT), T, latent_size=2)
    assert cli.dual_length(lay) == lay.length + 2          # two image tokens in PROMPT
    assert cli.length_budget_exclusion(lay, budget=cli.dual_length(lay)) is None
    assert cli.length_budget_exclusion(lay, budget=cli.dual_length(lay) - 1) == {
        "excluded": True, "reason": "length_budget", "dual_length": cli.dual_length(lay)}
    assert cli.LENGTH_BUDGET == 8192


def test_length_budget_exclusions_do_not_count_toward_g5_budget(tmp_path):
    for idx in range(11):
        cli.write_json_atomic(tmp_path / "attention" / "excluded" / f"q{idx}.json",
                              {"excluded": True, "reason": "length_budget", "dual_length": 9000})
    cli.write_json_atomic(tmp_path / "attention" / "excluded" / "q20.json", {"reason": "match_rate_below_0.99"})
    assert cli.g5_exclusion_count(tmp_path) == 1
    assert cli.excluded_indices(tmp_path) == set(range(11)) | {20}
    cli.require_exclusion_budget(cli.g5_exclusion_count(tmp_path))  # does not raise
    cli.require_attention_complete(tmp_path, list(range(11)) + [20])  # excluded files count as present


def test_stage_attention_excludes_over_budget_question_without_running_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import evaluation.latent_image_interventions as li
    import evaluation.latent_image_trajectory as lt
    indices = list(range(11))
    data_path = tmp_path / "TEST.jsonl"
    _write_jsonl(data_path, [{"image_input": ["x"]} for _ in indices])
    selection_path = tmp_path / "metadata.json"
    selection_path.write_text(json.dumps({"selection": {"requested_indices": indices}}))
    eval_rows_path = tmp_path / "merged.jsonl"
    _write_jsonl(eval_rows_path, [{"index": i, "stop_reason": "eos", "match": True} for i in indices])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_dir, base_model_dir = _model_dirs(tmp_path)
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path, model_dir, base_model_dir)
    cli.write_json_atomic(run_dir / "smoke" / "gates.json", {"pass": True})
    cli.write_json_atomic(run_dir / "swaps.json", {"questions": {str(i): {"category": c, "swaps": []} for i, c in
                                                                zip(indices, ["creation", "deletion", "selection",
                                                                              "update"] * 3)}})
    monkeypatch.setattr(cli, "_teacher_forced_setup", lambda args: (None, None, T))
    layout = SimpleNamespace(length=8000, image_positions=tuple(range(500)))
    monkeypatch.setattr(lt, "build_trajectory", lambda *a, **k: SimpleNamespace(layout=layout))

    def boom(*a, **k):
        raise AssertionError("the model must not run for an over-budget question")
    monkeypatch.setattr(li, "run_condition", boom)
    args = _stage_args("attention", data_path, selection_path, eval_rows_path, run_dir, model_dir, base_model_dir)
    assert cli.stage_attention(args) == 0         # 11 exclusions, none counted toward the G5 budget
    for idx in indices:
        doc = json.loads((run_dir / "attention" / "excluded" / f"q{idx}.json").read_text())
        assert doc == {"excluded": True, "reason": "length_budget", "dual_length": 8500}


def test_attention_meta_records_eval_stop_reason():
    from types import SimpleNamespace
    capture = SimpleNamespace(specs=(), g1_max_rel=1e-4, g1b_linear_rel=1e-5, g2_max_abs=1e-6)
    lay = build_layout(PROMPT + [10, 91, 90, 90, 92, 13, 84], len(PROMPT), T, latent_size=2)
    trajectory = SimpleNamespace(category="deletion", layout=lay, eval_match=True)
    meta = cli.attention_meta(capture, trajectory, {"excluded": False}, {"stop_reason": "max_new_tokens"})
    assert meta["stop_reason"] == "max_new_tokens" and meta["n_blocks"] == 1
    assert meta["dual_length"] == cli.dual_length(lay)


def test_select_smoke_questions():
    def c(idx, n_blocks, stop="eos", length=1000):
        return {"idx": idx, "n_blocks": n_blocks, "stop_reason": stop, "dual_length": length}
    candidates = [c(1, 1, stop="max_new_tokens"), c(2, 1, length=9000), c(3, 6), c(4, 2), c(5, 1), c(6, 3), c(7, 4)]
    chosen = cli.select_smoke_questions(candidates)
    assert chosen["single"]["idx"] == 5 and chosen["multi"]["idx"] == 6
    consumed = []

    def lazy():
        for cand in candidates:
            consumed.append(cand["idx"])
            yield cand
    cli.select_smoke_questions(lazy())
    assert consumed == [1, 2, 3, 4, 5, 6]       # stops once both are found
    with pytest.raises(RuntimeError, match="smoke"):
        cli.select_smoke_questions([c(1, 1), c(3, 6), c(4, 3, stop="max_new_tokens"), c(8, 5, length=8193)])


def test_stage_heads_weights_questions_equally(tmp_path):
    # R14: q0 has three pad rows (0.9), q1 one pad row (0.3). Question-equal mean = 0.6,
    # row-weighted would be 0.75.
    import numpy as np
    data_path, selection_path, eval_rows_path = _minimal_inputs(tmp_path, [0, 1])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path)
    for idx, (n_pads, value) in enumerate(((3, 0.9), (1, 0.3))):
        specs = [{"position": i, "group": "latent", "block": 0, "slot": f"pad{i + 1}"} for i in range(n_pads)]
        vw = np.zeros((n_pads, 2, 2, 7), dtype=np.float32)
        vw[..., 2] = value
        cli.savez_atomic(run_dir / "attention" / f"q{idx}.npz", mass=vw, vw_share=vw,
                         enrichment=np.zeros((n_pads, 2, 2), dtype=np.float32))
        cli.write_json_atomic(run_dir / "attention" / f"q{idx}.json", {"specs": specs})
    assert cli.stage_heads(_stage_args("heads", data_path, selection_path, eval_rows_path, run_dir)) == 0
    share = np.array(json.loads((run_dir / "heads.json").read_text())["share"])
    assert np.allclose(share, 0.6)


def test_stage_report_allows_different_commit_but_checks_inputs(tmp_path):
    from tests.evaluation.latent_image_fixtures import write_synthetic_run
    data_path, selection_path, eval_rows_path = _minimal_inputs(tmp_path, [0])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_synthetic_run(run_dir, questions=1, blocks=1)
    manifest = _write_manifest(run_dir, data_path, selection_path, eval_rows_path)
    manifest["commit"] = "an-older-prepare-commit"
    cli.write_json_atomic(run_dir / "manifest.json", manifest)
    args = _stage_args("report", data_path, selection_path, eval_rows_path, run_dir)
    with pytest.raises(ValueError, match="manifest"):
        cli.stage_heads(_stage_args("heads", data_path, selection_path, eval_rows_path, run_dir))
    assert cli.stage_report(args) == 0
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["report_commit"] == cli.git_commit()
    assert summary["report_dirty"] == bool(cli.git_status_porcelain())
    with pytest.raises(ValueError, match="manifest"):
        cli.require_manifest(run_dir, {**manifest["inputs"], "data": "other"}, "x", allow_commit_change=True)


def test_stage_freegen_records_eval_match(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    data_path = tmp_path / "TEST.jsonl"
    _write_jsonl(data_path, [{"image_input": ["x"]}, {"image_input": ["y"]}])
    selection_path = tmp_path / "metadata.json"
    selection_path.write_text(json.dumps({"selection": {"requested_indices": [0, 1]}}))
    eval_rows_path = tmp_path / "merged.jsonl"
    _write_jsonl(eval_rows_path, [{"index": 0, "match": True}, {"index": 1, "match": False}])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_dir, base_model_dir = _model_dirs(tmp_path)
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path, model_dir, base_model_dir)
    fake_eval = ModuleType("eval")
    fake_eval.load_evaluation_artifacts = lambda **kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "eval", fake_eval)
    monkeypatch.setattr(cli, "question_freegen", lambda *a: {"match": True, "stop_reason": "eos"})
    args = _stage_args("freegen", data_path, selection_path, eval_rows_path, run_dir, model_dir, base_model_dir)
    assert cli.stage_freegen(args) == 0
    for mode in ("causal_mask", "latent_mask"):
        assert json.loads((run_dir / "freegen" / mode / "q0.json").read_text())["eval_match"] is True
        assert json.loads((run_dir / "freegen" / mode / "q1.json").read_text())["eval_match"] is False


# ---------------------------------------------------------------------------
# R20 minor fixes: dirty-tree prepare, manifest-last prepare, temp-file globs
# ---------------------------------------------------------------------------

def _prepare_argv(fixture, run_dir):
    return ["prepare", "--model-dir", str(fixture["model_dir"]), "--base-model-dir", str(fixture["base_model_dir"]),
            "--data", str(fixture["data"]), "--selection", str(fixture["selection"]),
            "--eval-rows", str(fixture["eval_rows"]), "--data-root", str(fixture["data_root"]),
            "--run-dir", str(run_dir)]


def _expect_fixture_hashes(monkeypatch, fixture):
    monkeypatch.setattr(cli, "EXPECTED_SHA256", {
        "data": cli.sha256(fixture["data"]), "selection": cli.sha256(fixture["selection"]),
        "eval_rows": cli.sha256(fixture["eval_rows"])})


def test_prepare_refuses_dirty_tracked_tree(tmp_path, monkeypatch):
    fixture = _build_prepare_fixture(tmp_path)
    _expect_fixture_hashes(monkeypatch, fixture)
    monkeypatch.setattr(cli, "git_status_porcelain", lambda: " M src/evaluation/latent_image_cli.py")
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="dirty"):
        cli.main(_prepare_argv(fixture, run_dir))
    assert not (run_dir / "manifest.json").exists()


def test_prepare_records_clean_status(tmp_path, monkeypatch):
    fixture = _build_prepare_fixture(tmp_path)
    _expect_fixture_hashes(monkeypatch, fixture)
    monkeypatch.setattr(cli, "git_status_porcelain", lambda: "")
    run_dir = tmp_path / "run"
    assert cli.main(_prepare_argv(fixture, run_dir)) == 0
    assert json.loads((run_dir / "manifest.json").read_text())["git_status"] == ""


def test_failed_swap_computation_leaves_no_manifest(tmp_path, monkeypatch):
    fixture = _build_prepare_fixture(tmp_path)
    _expect_fixture_hashes(monkeypatch, fixture)
    monkeypatch.setattr(cli, "git_status_porcelain", lambda: "")

    def boom(*a, **k):
        raise KeyError("category lookup failed")
    monkeypatch.setattr(cli, "question_category", boom)
    run_dir = tmp_path / "run"
    with pytest.raises(KeyError):
        cli.main(_prepare_argv(fixture, run_dir))
    assert not (run_dir / "manifest.json").exists()


def test_stage_heads_ignores_leftover_temp_npz(tmp_path):
    import numpy as np
    data_path, selection_path, eval_rows_path = _minimal_inputs(tmp_path, [0])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir, data_path, selection_path, eval_rows_path)
    specs = [{"position": 0, "group": "latent", "block": 0, "slot": "pad1"}]
    vw = np.full((1, 2, 2, 7), 0.5, dtype=np.float32)
    cli.savez_atomic(run_dir / "attention" / "q0.npz", mass=vw, vw_share=vw, enrichment=np.zeros((1, 2, 2)))
    cli.write_json_atomic(run_dir / "attention" / "q0.json", {"specs": specs})
    (run_dir / "attention" / "q5.tmp.npz").write_bytes(b"partial")
    (run_dir / "attention" / "excluded").mkdir()
    (run_dir / "attention" / "excluded" / "q6.json.tmp").write_text("{")
    assert cli.question_files(run_dir / "attention", "npz") == {0: run_dir / "attention" / "q0.npz"}
    assert cli.excluded_indices(run_dir) == set()
    assert cli.stage_heads(_stage_args("heads", data_path, selection_path, eval_rows_path, run_dir)) == 0

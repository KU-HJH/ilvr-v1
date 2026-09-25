"""CPU tests for fp32 trajectory regeneration (`trajectories`, `merge-trajectories`)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from evaluation import latent_image_cli as cli
from evaluation import latent_image_trajectories_gen as gen


def _write_jsonl(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _inputs(tmp_path, indices, eval_rows=None):
    data_path = tmp_path / "TEST.jsonl"
    _write_jsonl(data_path, [{"image_input": [f"img{i}"], "original_final_answer": str(i)}
                             for i in range(max(indices) + 1)])
    selection_path = tmp_path / "metadata.json"
    selection_path.write_text(json.dumps({"selection": {"requested_indices": indices}}))
    model_dir = tmp_path / "model" / "prepared-checkpoint-1"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "model" / "prepared-checkpoint-1.manifest.json").write_text("{}")
    base_model_dir = tmp_path / "base"
    base_model_dir.mkdir()
    argv = ["--data", str(data_path), "--selection", str(selection_path), "--model-dir", str(model_dir),
            "--base-model-dir", str(base_model_dir), "--out-dir", str(tmp_path / "traj")]
    if eval_rows is not None:
        eval_path = tmp_path / "merged.jsonl"
        _write_jsonl(eval_path, eval_rows)
        argv += ["--eval-rows", str(eval_path)]
    return argv


def _args(stage, argv, *extra):
    return cli._parser().parse_args([stage, *argv, *extra])


def _fake_row(idx, match=True):
    return {"prompt_identity": f"pid{idx}", "prompt_token_length": 10, "generated_suffix_token_ids": [1, 2, idx],
            "assistant_only": "The final answer is: 1", "extracted_final_answer": "1", "match": match,
            "latent_block_count": 1, "stop_reason": "eos"}


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = {"generated": [], "precision": [], "policy": 0}
    monkeypatch.setattr(cli, "_teacher_forced_setup", lambda args: ("model", "processor", None))

    def apply(model, precision):
        calls["precision"].append(precision)
        return {"language": "torch.float32", "visual": "torch.bfloat16"}
    monkeypatch.setattr(cli, "apply_precision", apply)
    monkeypatch.setattr(cli, "git_commit", lambda: "c" * 40)
    monkeypatch.setattr(cli, "git_status_porcelain", lambda: "")

    def generate(model, processor, test_row, data_root, *, mode=None, expected_prompt_identity=None):
        idx = int(test_row["image_input"][0][3:])
        calls["generated"].append((idx, mode, expected_prompt_identity))
        if idx == 3:
            return {"skipped": "length_budget", "prompt_token_length": 9000, "image_tokens": 10,
                    "prompt_dual_length": 9010}
        return _fake_row(idx, match=idx != 2)
    monkeypatch.setattr(gen, "generate_question", generate)
    return calls


# ---------------------------------------------------------------------------
# trajectories stage
# ---------------------------------------------------------------------------

def test_trajectories_stage_is_listed_and_needs_no_run_dir_or_eval_rows(tmp_path):
    args = _args("trajectories", _inputs(tmp_path, [0]))
    assert args.eval_rows is None and not hasattr(args, "run_dir") and args.precision == "fp32"
    assert "trajectories" in cli.STAGE_HANDLERS and "merge-trajectories" in cli.STAGE_HANDLERS


def test_trajectories_writes_owned_rows_and_separate_skips(tmp_path, fake_runtime):
    argv = _inputs(tmp_path, [0, 1, 2, 3, 4, 5])
    assert cli.main(["trajectories", *argv, "--shard", "1", "--num-shards", "2"]) == 0
    out = tmp_path / "traj"
    rows = _read_jsonl(out / "shard-1.jsonl")
    assert [r["index"] for r in rows] == [1, 5]                     # owned: 1, 3, 5; 3 skipped
    assert [c[0] for c in fake_runtime["generated"]] == [1, 3, 5]
    assert all(c[1] is None for c in fake_runtime["generated"])     # no intervention
    assert fake_runtime["precision"] == ["fp32"]
    skipped = _read_jsonl(out / "skipped-shard-1.jsonl")
    assert skipped == [{"index": 3, "skipped": "length_budget", "prompt_token_length": 9000, "image_tokens": 10,
                        "prompt_dual_length": 9010}]
    row = rows[0]
    assert set(row) >= {"index", "task_name", "prompt_identity", "prompt_token_length", "generated_suffix_token_ids",
                        "assistant_only", "extracted_final_answer", "match", "latent_block_count", "stop_reason",
                        "precision"}
    assert row["task_name"] == "zebra-cot" and row["precision"] == "fp32"
    assert row["precision_dtypes"] == {"language": "torch.float32", "visual": "torch.bfloat16"}
    meta = json.loads((out / "shard-1.meta.json").read_text())
    assert meta["commit"] == "c" * 40 and meta["precision"] == "fp32" and meta["num_shards"] == 2
    assert meta["generation_config"] == gen.GENERATION_KWARGS


def test_trajectories_resume_skips_done_and_trims_partial_line(tmp_path, fake_runtime):
    argv = _inputs(tmp_path, [0, 1, 2, 3])
    assert cli.main(["trajectories", *argv]) == 0
    out = tmp_path / "traj" / "shard-0.jsonl"
    lines = out.read_text().splitlines(keepends=True)
    assert len(lines) == 3
    out.write_text(lines[0] + lines[1] + lines[2][:15])             # crash mid-write of index 2
    fake_runtime["generated"].clear()
    assert cli.main(["trajectories", *argv]) == 0
    assert [c[0] for c in fake_runtime["generated"]] == [2]         # 0, 1 done; 3 already skipped
    assert [r["index"] for r in _read_jsonl(out)] == [0, 1, 2]


def test_trajectories_resume_refuses_changed_meta(tmp_path, fake_runtime, monkeypatch):
    argv = _inputs(tmp_path, [0])
    assert cli.main(["trajectories", *argv]) == 0
    monkeypatch.setattr(cli, "git_commit", lambda: "d" * 40)
    with pytest.raises(ValueError, match="shard meta"):
        cli.main(["trajectories", *argv])


def test_trajectories_passes_eval_row_prompt_identity(tmp_path, fake_runtime):
    argv = _inputs(tmp_path, [0, 1], eval_rows=[{"index": 0, "prompt_identity": "orig0"}])
    assert cli.main(["trajectories", *argv]) == 0
    assert fake_runtime["generated"] == [(0, None, "orig0"), (1, None, None)]


def test_trajectories_does_not_load_model_when_shard_complete(tmp_path, fake_runtime, monkeypatch):
    argv = _inputs(tmp_path, [0])
    assert cli.main(["trajectories", *argv]) == 0

    def boom(args):
        raise AssertionError("model must not load when nothing is pending")
    monkeypatch.setattr(cli, "_teacher_forced_setup", boom)
    assert cli.main(["trajectories", *argv]) == 0


# ---------------------------------------------------------------------------
# generate_question: length budget and row construction
# ---------------------------------------------------------------------------

def _prepared(prompt_len, n_images):
    ids = torch.tensor([[1] * (prompt_len - n_images) + [80] * n_images])
    return SimpleNamespace(tensors={"input_ids": ids}, prompt_token_length=prompt_len, prompt_identity="pid")


def test_generate_question_skips_over_budget_prompt_without_generating(monkeypatch):
    from step_preference.equivalence import _tiny_model
    model = _tiny_model()                                     # image_token_id 80
    monkeypatch.setattr(gen, "prepare_question", lambda processor, row, root: _prepared(100, 40))
    monkeypatch.setattr(gen, "LENGTH_BUDGET", 139)

    def boom(*a, **k):
        raise AssertionError("generation must not run for an over-budget prompt")
    monkeypatch.setattr(gen, "_generate", boom)
    out = gen.generate_question(model, None, {}, "data")
    assert out == {"skipped": "length_budget", "prompt_token_length": 100, "image_tokens": 40,
                   "prompt_dual_length": 140}
    monkeypatch.setattr(gen, "LENGTH_BUDGET", 140)
    monkeypatch.setattr(gen, "_generate", lambda qwen, processor, prepared, mode: "result")
    monkeypatch.setattr(gen, "row_from_result", lambda processor, prepared, row, result: {"ok": result})
    assert gen.generate_question(model, None, {}, "data") == {"ok": "result"}


def test_generate_question_refuses_prompt_identity_mismatch(monkeypatch):
    from step_preference.equivalence import _tiny_model
    monkeypatch.setattr(gen, "prepare_question", lambda processor, row, root: _prepared(10, 2))
    with pytest.raises(ValueError, match="prompt identity"):
        gen.generate_question(_tiny_model(), None, {}, "data", expected_prompt_identity="other")


def test_row_from_result_scores_like_eval():
    class Processor:
        def batch_decode(self, ids, skip_special_tokens=False):
            assert ids.tolist() == [[7, 8, 9]]
            return ["reasoning. The final answer is: 42<|im_end|>"]
    prepared = _prepared(3, 1)
    result = SimpleNamespace(sequences=torch.tensor([[1, 1, 80, 7, 8, 9]]), latent_block_count=2, stop_reason="eos")
    row = gen.row_from_result(Processor(), prepared, {"original_final_answer": "42"}, result)
    assert row["generated_suffix_token_ids"] == [7, 8, 9]
    assert row["match"] is True and row["extracted_final_answer"] == "42"
    assert row["prompt_identity"] == "pid" and row["prompt_token_length"] == 3
    assert row["latent_block_count"] == 2 and row["stop_reason"] == "eos"
    assert "<|im_end|>" not in row["assistant_only"]


# ---------------------------------------------------------------------------
# merge-trajectories
# ---------------------------------------------------------------------------

def _generate_all(tmp_path, indices, num_shards=2):
    argv = _inputs(tmp_path, indices)
    for shard in range(num_shards):
        assert cli.main(["trajectories", *argv, "--shard", str(shard), "--num-shards", str(num_shards)]) == 0
    return argv


def test_merge_writes_sorted_rows_and_manifest(tmp_path, fake_runtime):
    argv = _generate_all(tmp_path, [0, 1, 2, 3, 4])
    assert cli.main(["merge-trajectories", *argv]) == 0
    out = tmp_path / "traj"
    merged = out / "merged.jsonl"
    assert [r["index"] for r in _read_jsonl(merged)] == [0, 1, 2, 4]
    assert [r["index"] for r in _read_jsonl(out / "skipped.jsonl")] == [3]
    manifest = json.loads((out / "trajectories-manifest.json").read_text())
    args = _args("merge-trajectories", argv)
    assert manifest["merged_sha256"] == cli.sha256(merged)
    assert {k: manifest[k] for k in cli.model_identity(args)} == cli.model_identity(args)
    assert manifest["selection_sha256"] == cli.sha256(args.selection)
    assert manifest["data_sha256"] == cli.sha256(args.data)
    assert manifest["precision"] == "fp32" and manifest["commit"] == "c" * 40 and manifest["git_status"] == ""
    assert manifest["generation_config"] == gen.GENERATION_KWARGS
    assert manifest["numerics"] == "numerical_policy (SDPA MATH, TF32 off)"
    assert [s["index"] for s in manifest["skipped"]] == [3]
    assert manifest["count"] == 4 and manifest["correct"] == 3


def test_merge_refuses_missing_index(tmp_path, fake_runtime):
    argv = _inputs(tmp_path, [0, 1, 2])
    assert cli.main(["trajectories", *argv, "--shard", "0", "--num-shards", "2"]) == 0
    with pytest.raises(ValueError, match="shard meta"):
        cli.main(["merge-trajectories", *argv])                      # shard 1 never ran
    out = tmp_path / "traj" / "shard-0.jsonl"
    cli.main(["trajectories", *argv, "--shard", "1", "--num-shards", "2"])
    lines = out.read_text().splitlines(keepends=True)
    out.write_text(lines[0])                                         # drop index 2
    with pytest.raises(ValueError, match="missing"):
        cli.main(["merge-trajectories", *argv])
    assert not (tmp_path / "traj" / "merged.jsonl").exists()


def test_merge_refuses_duplicate_index(tmp_path, fake_runtime):
    argv = _generate_all(tmp_path, [0, 1, 2])
    out = tmp_path / "traj" / "shard-1.jsonl"
    with open(out, "a") as handle:
        handle.write(json.dumps({"index": 0, **_fake_row(0)}) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        cli.main(["merge-trajectories", *argv])


def test_merge_refuses_model_identity_change(tmp_path, fake_runtime):
    argv = _generate_all(tmp_path, [0, 1])
    with pytest.raises(ValueError, match="shard meta"):
        cli.main(["merge-trajectories", *argv, "--precision", "bf16"])

"""Staged CLI, gates and per-question drivers for latent image attention analysis.

Seven stages, run in order against a shared `--run-dir`:
`prepare`, `smoke`, `attention`, `heads`, `interventions`, `freegen`, `report`.
`prepare` records input hashes and the git commit in `manifest.json`; every
later stage aborts if the current inputs or commit differ (`require_manifest`),
except `report`, which only re-checks the inputs and records its own commit.

Two standalone stages, `trajectories` and `merge-trajectories` (R24), run before
`prepare` and regenerate the analysed greedy trajectories under the analysis
numerics into `--out-dir` (see `evaluation.latent_image_trajectories_gen`).

Questions whose dual-image sequence exceeds `LENGTH_BUDGET` are excluded by the
attention stage without running the model (reason "length_budget"); such
exclusions do not count toward the G5 exclusion budget.
"""
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from evaluation.latent_image_stats import cosine_distance

STAGES = ("prepare", "smoke", "attention", "heads", "interventions", "freegen", "report")
# R24: standalone stages that regenerate the analysed trajectories before `prepare`
# (see evaluation.latent_image_trajectories_gen).
TRAJECTORY_STAGES = ("trajectories", "merge-trajectories")

# Global Constraints values (compared against `prepare`'s computed hashes). Tests
# monkeypatch this dict since synthetic tmp_path inputs never match these hashes.
EXPECTED_SHA256 = {
    "data": "6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618",
    "selection": "fb1f15fef5339abde6ba74d140e5f9a1f8823cedafe8aef1d4bb245a8e0286aa",
    "eval_rows": "76380b6e01bc6cf7e6bd44c551cf99a8ed18b0745dbd1314c8378fe2af9396d5",
}

# R12: SDPA MATH materialises fp32 [heads, L, L] scores; dual-image interventions add one
# image span, so the dual length (sequence + image tokens) is capped for the 94 GB GPUs.
LENGTH_BUDGET = 8192

DEFAULT_MODEL_DIR = Path(
    "checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212")
DEFAULT_BASE_MODEL_DIR = Path("checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct")
DEFAULT_DATA = Path("data/TEST.jsonl")
DEFAULT_DATA_ROOT = Path("data")


# ---------------------------------------------------------------------------
# Helpers (Step 3)
# ---------------------------------------------------------------------------

def shard_items(items, shard, num_shards):
    return tuple(x for i, x in enumerate(items) if i % num_shards == shard)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_status_porcelain():
    """Tracked-file changes only (untracked files are ignored)."""
    return subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()


def require_manifest(run_dir, inputs, commit, *, allow_commit_change=False):
    """Input hashes must always match; the commit must match unless `allow_commit_change`
    (R17: only the CPU-only `report` stage, which records its own commit instead)."""
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    if manifest["inputs"] != inputs or (manifest["commit"] != commit and not allow_commit_change):
        raise ValueError("manifest inputs or commit differ from the current execution")
    return manifest


def model_identity(args):
    """The model `args` selects: directories, on-disk adapter/prepared-manifest content and precision."""
    return {
        "model_dir": str(args.model_dir), "base_model_dir": str(args.base_model_dir),
        "prepared_manifest_sha256": sha256(Path(str(args.model_dir) + ".manifest.json")),
        "adapter_sha256": sha256(args.model_dir / "adapter_model.safetensors"),
        "precision": getattr(args, "precision", None),
    }


def require_model_identity(manifest, args):
    """Every model-loading stage (smoke/attention/interventions/freegen) must resume
    against the exact model `prepare` recorded: same directories, same on-disk
    content (an adapter or prepared-manifest edit in place would go undetected by
    path comparison alone), and the same `--precision` (R22)."""
    current = model_identity(args)
    expected = {key: manifest.get(key) for key in current}
    if current != expected:
        raise ValueError("manifest model identity (model_dir, base_model_dir, adapter_sha256, "
                         "prepared_manifest_sha256, precision) differs from the current execution")


def apply_precision(model, precision):
    """R22: teacher-forced stages load a BF16 checkpoint via `_load_runtime`; when
    `precision == "fp32"`, cast the language model (decoder stack + lm_head) up to
    fp32 to avoid length-dependent BF16 rounding in the dual-image gate. `qwen.visual`
    is left untouched (FlashAttention2 requires BF16). Returns the effective
    {"language", "visual"} dtypes for provenance recording."""
    from stage2.peft import resolve_stage2_qwen_model
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"unknown precision {precision!r}; expected 'fp32' or 'bf16'")
    qwen = resolve_stage2_qwen_model(model)
    if precision == "fp32":
        qwen.model.to(torch.float32)
        qwen.lm_head.to(torch.float32)
    return {"language": str(next(qwen.model.parameters()).dtype),
            "visual": str(next(qwen.visual.parameters()).dtype)}


def dual_length(layout):
    """Sequence length of the dual-image interventions: the trajectory plus one extra image span."""
    return layout.length + len(layout.image_positions)


def length_budget_exclusion(layout, budget=LENGTH_BUDGET):
    n = dual_length(layout)
    return {"excluded": True, "reason": "length_budget", "dual_length": n} if n > budget else None


def question_files(directory, suffix):
    """{question index: path} for q{idx}.{suffix}, ignoring leftover temp files."""
    return {int(p.name[1:-len(suffix) - 1]): p for p in sorted(Path(directory).glob(f"q*.{suffix}"))
            if ".tmp" not in p.name}


def _exclusions(run_dir):
    return {idx: json.loads(p.read_text())
            for idx, p in question_files(Path(run_dir) / "attention" / "excluded", "json").items()}


def excluded_indices(run_dir):
    return set(_exclusions(run_dir))


def g5_exclusion_count(run_dir):
    """Exclusions that count toward the G5 budget (length-budget exclusions do not)."""
    return sum(1 for doc in _exclusions(run_dir).values() if doc.get("reason") != "length_budget")


def trajectory_skipped(manifest):
    """R24: questions the trajectory regeneration skipped (length budget); absent from eval rows."""
    return set(manifest.get("trajectory_skipped", ()))


def require_smoke_pass(run_dir):
    path = Path(run_dir) / "smoke" / "gates.json"
    if not path.exists():
        raise RuntimeError("attention requires a passing smoke/gates.json; run the smoke stage first")
    gates = json.loads(path.read_text())
    if not gates.get("pass"):
        raise RuntimeError("smoke gate did not pass; resolve the smoke failures before running attention")
    return gates


def require_attention_complete(run_dir, selection):
    run_dir = Path(run_dir)
    missing = [idx for idx in selection
              if not ((run_dir / "attention" / f"q{idx}.json").exists()
                      or (run_dir / "attention" / "excluded" / f"q{idx}.json").exists())]
    if missing:
        raise RuntimeError(f"attention stage incomplete; missing questions: {sorted(missing)}")


def g5_verdict(matches, first_block_offset):
    rate = sum(matches) / len(matches)
    first_bad = next((i for i, ok in enumerate(matches) if not ok), None)
    if first_bad is not None and first_bad < first_block_offset:
        return {"excluded": True, "reason": "mismatch_before_first_block", "rate": rate, "first_mismatch": first_bad}
    if rate < 0.99:
        return {"excluded": True, "reason": "match_rate_below_0.99", "rate": rate, "first_mismatch": first_bad}
    return {"excluded": False, "reason": None, "rate": rate, "first_mismatch": first_bad}


def require_exclusion_budget(count, limit=10):
    if count > limit:
        raise RuntimeError(f"{count} trajectory exclusions exceed the limit of {limit}")


def select_smoke_questions(candidates, budget=LENGTH_BUDGET):
    """R15: first candidate with exactly 1 block and first with 3-5 blocks, both with eval
    stop_reason "eos" and within the length budget. `candidates` yields dicts with idx,
    n_blocks, stop_reason and dual_length; it is consumed lazily and only until both are found."""
    chosen = {}
    for cand in candidates:
        if cand["stop_reason"] != "eos" or cand["dual_length"] > budget:
            continue
        if cand["n_blocks"] == 1:
            chosen.setdefault("single", cand)
        elif 3 <= cand["n_blocks"] <= 5:
            chosen.setdefault("multi", cand)
        if len(chosen) == 2:
            return chosen
    raise RuntimeError("selection lacks an eos, within-budget 1-block question and 3-5-block question for smoke")


def select_heads(share, fraction=0.05):
    count = int(round(share.size * fraction))          # 576 * 0.05 = 28.8 -> 29
    order = np.argsort(-share, axis=None, kind="stable")[:count]
    return [tuple(int(v) for v in np.unravel_index(i, share.shape)) for i in order]


# ---------------------------------------------------------------------------
# Per-question logic (Step 6)
# ---------------------------------------------------------------------------

def compare_runs(a, b, tol=1e-2):
    tokens = sorted(b.scores)
    mismatches = sum(a.scores[p][0] != b.scores[p][0] for p in tokens)
    max_dnll = max(abs(a.scores[p][1] - b.scores[p][1]) for p in tokens)
    rel = float(((a.latents - b.latents).norm(dim=-1) / b.latents.norm(dim=-1).clamp_min(1e-6)).max())
    return {"argmax_mismatches": int(mismatches), "max_abs_dnll": float(max_dnll), "max_latent_rel": rel,
            "pass": mismatches == 0 and max_dnll <= tol and rel <= tol}


def g5_for(reference, trajectory):
    ids = trajectory.input_ids[0].tolist()
    scored = trajectory.layout.scored_positions
    matches = [reference.scores[p][0] == ids[p] for p in scored]
    return g5_verdict(matches, scored.index(trajectory.layout.blocks[0].start))


def summarize_question(results, layout, matched):
    base = results["noop"]
    out = {key: {} for key in ("latent_cos", "latent_cos_pads", "post_dnll", "post_flip",
                               "answer_dnll", "answer_flip", "control_cos")}
    out.update(random_matched=matched, answer_status=layout.answer_status)
    controls = sorted(base.last_hidden)

    def delta(result, positions):
        if not positions:
            return None, None
        d = [result.scores[p][1] - base.scores[p][1] for p in positions]
        f = [result.scores[p][0] != base.scores[p][0] for p in positions]
        return float(np.mean(d)), float(np.mean(f))

    for name, result in results.items():
        if name == "noop":
            continue
        pads = cosine_distance(result.latents.numpy(), base.latents.numpy())
        out["latent_cos_pads"][name] = pads.tolist()
        out["latent_cos"][name] = pads.mean(1).tolist()
        post = [delta(result, span) for span in layout.post_span]
        out["post_dnll"][name] = [p[0] for p in post]
        out["post_flip"][name] = [p[1] for p in post]
        answer = delta(result, layout.answer_positions) if layout.answer_status == "ok" else (None, None)
        out["answer_dnll"][name], out["answer_flip"][name] = answer
        out["control_cos"][name] = (float(np.mean([cosine_distance(result.last_hidden[p].numpy(),
                                                                   base.last_hidden[p].numpy()) for p in controls]))
                                    if controls else None)
    return out


def question_interventions(model, processor, trajectory, heads, swap_images):
    from evaluation.latent_image_corruptions import (blur_image, mean_image, pixel_inputs,
                                                     require_same_grid, resized_swap)
    from evaluation.latent_image_interventions import Condition, random_key_sets, run_condition
    original = trajectory.images[0]

    def pix(image, label):
        values, grid = pixel_inputs(processor, image)
        require_same_grid(trajectory.image_grid_thw, grid, label)
        return values, grid

    ip = processor.image_processor
    alts = {"blur": pix(blur_image(original), "blur"), "mean": pix(mean_image(original), "mean"),
            "swap": pix(resized_swap(swap_images[0], original, ip), "swap")}
    candidates = [pix(resized_swap(image, original, ip), f"cand{i + 1}") for i, image in enumerate(swap_images)]
    random_sets, matched = random_key_sets(trajectory.layout, source_index=trajectory.source_index)
    results = {}

    def run(condition, **kwargs):
        results[condition.name] = run_condition(model, trajectory, condition, **kwargs)

    run(Condition("reference", native=True))
    run(Condition("noop"))
    run(Condition("noop_repeat"))
    run(Condition("k_all", latent_source="none"))
    run(Condition("k_heads", head_subset=tuple(tuple(h) for h in heads)))
    run(Condition("k_random", random_keys=True), random_blocked=random_sets)
    for kind, (values, grid) in alts.items():
        for name, text, latent in (("dual_clean", "clean", "clean"), ("total", "alt", "alt"),
                                   ("direct", "clean", "alt"), ("indirect", "alt", "clean")):
            run(Condition(f"{kind}_{name}", text, latent), alt_pixel_values=values, alt_grid=grid)
    for i, (values, grid) in enumerate(candidates):
        run(Condition(f"cand{i + 1}"), pixel_override=(values, grid))
    gates = {"G3": compare_runs(results["noop"], results["reference"]),
             "G4": compare_runs(results["blur_dual_clean"], results["noop"]),
             "G5": g5_for(results["reference"], trajectory)}
    return results, matched, gates


def content_arrays(results):
    names = ["noop", "cand1", "cand2", "cand3", "cand4"]
    return {key: np.stack([getattr(results[n], key).numpy() for n in names]).astype(np.float32)
            for key in ("mid_body", "final_body", "image_mid", "image_final")}


def attention_meta(capture, trajectory, g5, eval_row):
    return {"specs": [dict(position=s.position, group=s.group, block=s.block, slot=s.slot) for s in capture.specs],
            "g1": capture.g1_max_rel, "g1b": capture.g1b_linear_rel, "g2": capture.g2_max_abs, "g5": g5,
            "category": trajectory.category, "n_blocks": len(trajectory.layout.blocks),
            "eval_match": trajectory.eval_match, "stop_reason": eval_row.get("stop_reason"),
            "dual_length": dual_length(trajectory.layout)}


def heatmap_questions(selection, categories):
    chosen = [next(i for i in selection if categories[i] == c) for c in ("creation", "deletion", "selection", "update")]
    return tuple(chosen + [i for i in selection if i not in chosen][:2])


# ---------------------------------------------------------------------------
# I/O helpers used by the stage drivers
# ---------------------------------------------------------------------------

def load_test_rows(path):
    """TEST.jsonl rows indexed by 0-based line number (the `index` used everywhere else)."""
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def savez_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def skip_existing(*paths):
    return any(Path(p).exists() for p in paths)


def question_category(image_input, data_root):
    from multinode_evaluation import resolve_evaluation_image_paths
    root = Path(data_root).resolve()
    resolved = resolve_evaluation_image_paths(image_input, root)
    return Path(resolved[0]).resolve().relative_to(root).parts[1]


def open_swap_images(swap_indices, test_rows, data_root):
    from eval import open_images
    from multinode_evaluation import resolve_evaluation_image_paths
    root = Path(data_root).resolve()
    images = []
    for j in swap_indices:
        paths = resolve_evaluation_image_paths(test_rows[j]["image_input"], root)
        images.append(open_images(paths[:1])[0])
    return images


def _current_inputs(args):
    return {"data": sha256(args.data), "selection": sha256(args.selection), "eval_rows": sha256(args.eval_rows)}


# ---------------------------------------------------------------------------
# Stage drivers (Step 8)
# ---------------------------------------------------------------------------

def trajectory_provenance(args, inputs):
    """R24: `--eval-rows` is either the original eval rows (Global Constraints hash) or a
    regenerated `merged.jsonl` whose sibling `trajectories-manifest.json` vouches for
    exactly this file, model, precision, selection and data."""
    if inputs["eval_rows"] == EXPECTED_SHA256["eval_rows"]:
        return {"trajectory_source": "original", "trajectory_skipped": []}
    path = Path(args.eval_rows).parent / "trajectories-manifest.json"
    if not path.exists():
        raise ValueError(f"eval_rows sha256 {inputs['eval_rows']} is not the original eval rows "
                         f"({EXPECTED_SHA256['eval_rows']}) and no trajectories manifest exists at {path}")
    recorded = json.loads(path.read_text())
    expected = {**model_identity(args), "merged_sha256": inputs["eval_rows"],
                "selection_sha256": inputs["selection"], "data_sha256": inputs["data"]}
    mismatched = sorted(key for key in expected if recorded.get(key) != expected[key])
    if mismatched:
        raise ValueError(f"trajectories manifest {path} does not match the current inputs on {mismatched}")
    return {"trajectory_source": "regenerated", "trajectories_manifest_sha256": sha256(path),
            "trajectory_skipped": sorted(int(r["index"]) for r in recorded["skipped"])}


def stage_prepare(args):
    from evaluation.latent_image_corruptions import select_swaps
    from evaluation.latent_image_trajectory import load_selection
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError(f"manifest already exists at {manifest_path}; use a fresh run directory")
    status = git_status_porcelain()
    if status:
        raise RuntimeError(f"refusing to prepare from a dirty tracked tree; commit or revert first:\n{status}")
    inputs = _current_inputs(args)
    for key, path in (("data", args.data), ("selection", args.selection)):
        if inputs[key] != EXPECTED_SHA256[key]:
            raise ValueError(f"{key} sha256 {inputs[key]} does not match the expected Global Constraints "
                             f"value {EXPECTED_SHA256[key]}; refusing to prepare with unverified inputs ({path})")
    manifest = {
        "inputs": inputs, "commit": git_commit(), "git_status": status, **model_identity(args),
        **trajectory_provenance(args, inputs), "created": datetime.now(timezone.utc).isoformat(),
    }

    # Swaps are computed before the manifest is written, so a failed prepare leaves no manifest.
    test_rows = load_test_rows(args.data)
    selection = load_selection(args.selection)
    categories = {idx: question_category(test_rows[idx]["image_input"], args.data_root) for idx in selection}
    pool = {}
    for idx in selection:
        pool.setdefault(categories[idx], []).append(idx)
    questions = {}
    for idx in selection:
        swaps = select_swaps(idx, categories[idx], pool)
        questions[str(idx)] = {"category": categories[idx], "swaps": list(swaps)}
    write_json_atomic(args.run_dir / "swaps.json", {"pool": pool, "questions": questions})
    write_json_atomic(manifest_path, manifest)
    return 0


def _teacher_forced_setup(args):
    """Load the model/processor once under the analysis numerical policy."""
    from evaluation.vision_block_analysis import _load_runtime
    from evaluation.vision_block_runtime import numerical_policy
    from evaluation.latent_image_trajectory import layout_token_ids
    from stage2.peft import resolve_stage2_qwen_model
    with numerical_policy():
        model, processor, runtime = _load_runtime(args)
    token_ids = layout_token_ids(processor, resolve_stage2_qwen_model(model).config)
    return model, processor, token_ids


def stage_smoke(args):
    from evaluation.latent_image_attention import capture_attention
    from evaluation.latent_image_trajectory import build_trajectory, load_eval_rows, load_selection
    from evaluation.vision_block_runtime import numerical_policy
    manifest = require_manifest(args.run_dir, _current_inputs(args), git_commit())
    require_model_identity(manifest, args)
    test_rows = load_test_rows(args.data)
    eval_rows = load_eval_rows(args.eval_rows)
    selection = load_selection(args.selection)
    swaps = json.loads((args.run_dir / "swaps.json").read_text())
    model, processor, token_ids = _teacher_forced_setup(args)
    precision_record = apply_precision(model, args.precision)

    skipped = trajectory_skipped(manifest)

    def candidates():
        for idx in selection:
            if idx in skipped or eval_rows[idx].get("stop_reason") != "eos":
                continue
            trajectory = build_trajectory(processor, test_rows[idx], eval_rows[idx], idx, args.data_root, token_ids)
            yield {"idx": idx, "n_blocks": len(trajectory.layout.blocks), "stop_reason": "eos",
                   "dual_length": dual_length(trajectory.layout), "trajectory": trajectory}

    with numerical_policy():
        chosen = select_smoke_questions(candidates())
        gates, all_pass = {}, True
        for idx, trajectory in ((c["idx"], c["trajectory"]) for c in chosen.values()):
            capture = capture_attention(model, trajectory)
            swap_entry = swaps["questions"][str(idx)]
            swap_images = open_swap_images(swap_entry["swaps"], test_rows, args.data_root)
            _, _, cond_gates = question_interventions(model, processor, trajectory, [], swap_images)
            entry = {
                "G1": capture.g1_max_rel <= 1e-2, "G1b": capture.g1b_linear_rel <= 1e-2,
                "G2": capture.g2_max_abs <= 1e-3, "G3": cond_gates["G3"]["pass"], "G4": cond_gates["G4"]["pass"],
                "G5": not cond_gates["G5"]["excluded"],
                "detail": {"g1_max_rel": capture.g1_max_rel, "g1b_linear_rel": capture.g1b_linear_rel,
                          "g2_max_abs": capture.g2_max_abs, "G3": cond_gates["G3"], "G4": cond_gates["G4"],
                          "G5": cond_gates["G5"], "precision": precision_record},
            }
            gates[str(idx)] = entry
            all_pass = all_pass and all(entry[k] for k in ("G1", "G1b", "G2", "G3", "G4", "G5"))
        gates["pass"] = all_pass
    write_json_atomic(args.run_dir / "smoke" / "gates.json", gates)
    return 0 if all_pass else 1


SMOKE_FROM_KEYS = ("inputs", "model_dir", "base_model_dir", "adapter_sha256", "prepared_manifest_sha256",
                   "precision", "trajectory_source", "trajectories_manifest_sha256")


def apply_smoke_from(run_dir, source_run, manifest):
    """Reuse a prior run's passing smoke gate instead of re-running the smoke stage (avoids a
    redundant smoke run when nothing that smoke exercises has changed). Requires the prior
    run's inputs, model identity (model_dir/base_model_dir/adapter_sha256/
    prepared_manifest_sha256), precision and trajectory provenance (source and, when recorded,
    its manifest sha256) to equal the current run's manifest exactly; only the commit may
    differ. Writes, each atomically and in order: smoke/smoke-from.pending (an explicit
    in-progress marker), smoke/provenance.json ({"source_run", "source_commit", "copied_at"}),
    smoke/gates.json (a verbatim copy of the prior gate), then removes the pending marker.
    stage_attention's guard redoes the copy whenever gates.json is missing or the pending
    marker is present, so an interruption anywhere in this sequence self-heals. A gates.json
    present with no pending marker is trusted and never touched -- whether it is a genuine
    local `stage_smoke` result (which never creates the marker) or an already-completed
    --smoke-from copy -- so this never silently overwrites a real local smoke result."""
    source_run = Path(source_run)
    source_gates_path = source_run / "smoke" / "gates.json"
    if not source_gates_path.exists():
        raise ValueError(f"--smoke-from {source_run}: no smoke/gates.json found")
    source_gates = json.loads(source_gates_path.read_text())
    if not source_gates.get("pass"):
        raise ValueError(f"--smoke-from {source_run}: prior smoke/gates.json did not pass")
    source_manifest_path = source_run / "manifest.json"
    if not source_manifest_path.exists():
        raise ValueError(f"--smoke-from {source_run}: no manifest.json found")
    source_manifest = json.loads(source_manifest_path.read_text())
    mismatched = [key for key in SMOKE_FROM_KEYS if source_manifest.get(key) != manifest.get(key)]
    if mismatched:
        raise ValueError(f"--smoke-from {source_run}: prior manifest differs from the current run on {mismatched}")
    pending_path = Path(run_dir) / "smoke" / "smoke-from.pending"
    write_json_atomic(pending_path, {"source_run": str(source_run)})
    write_json_atomic(Path(run_dir) / "smoke" / "provenance.json", {
        "source_run": str(source_run), "source_commit": source_manifest.get("commit"),
        "copied_at": datetime.now(timezone.utc).isoformat()})
    write_json_atomic(Path(run_dir) / "smoke" / "gates.json", source_gates)
    pending_path.unlink()


def stage_attention(args):
    from evaluation.latent_image_attention import capture_attention
    from evaluation.latent_image_interventions import Condition, run_condition
    from evaluation.latent_image_trajectory import build_trajectory, load_eval_rows, load_selection
    from evaluation.vision_block_runtime import numerical_policy
    manifest = require_manifest(args.run_dir, _current_inputs(args), git_commit())
    require_model_identity(manifest, args)
    if getattr(args, "smoke_from", None):
        gates_path = args.run_dir / "smoke" / "gates.json"
        pending_path = args.run_dir / "smoke" / "smoke-from.pending"
        # Redo only when gates.json is missing, or the pending marker shows a prior copy was
        # interrupted mid-sequence. A gates.json present with no pending marker is trusted as
        # complete -- whether it is a genuine local `stage_smoke` result (which never creates
        # the marker) or an already-finished --smoke-from copy -- and is never overwritten.
        if not gates_path.exists() or pending_path.exists():
            apply_smoke_from(args.run_dir, args.smoke_from, manifest)
    require_smoke_pass(args.run_dir)
    test_rows = load_test_rows(args.data)
    eval_rows = load_eval_rows(args.eval_rows)
    selection = load_selection(args.selection)
    swaps = json.loads((args.run_dir / "swaps.json").read_text())
    categories = {int(i): entry["category"] for i, entry in swaps["questions"].items()}
    heatmap_set = set(heatmap_questions(selection, categories))
    owned = shard_items(selection, args.shard, args.num_shards)
    skipped = trajectory_skipped(manifest)
    model, processor, token_ids = _teacher_forced_setup(args)
    precision_record = apply_precision(model, args.precision)
    with numerical_policy():
        for idx in owned:
            out_json = args.run_dir / "attention" / f"q{idx}.json"
            excl_json = args.run_dir / "attention" / "excluded" / f"q{idx}.json"
            if skip_existing(out_json, excl_json):
                continue
            if idx in skipped:      # R24: never generated (prompt over budget), so no eval row exists
                write_json_atomic(excl_json, {"excluded": True, "reason": "length_budget",
                                              "source": "trajectories_skipped"})
                continue
            trajectory = build_trajectory(processor, test_rows[idx], eval_rows[idx], idx, args.data_root, token_ids)
            over_budget = length_budget_exclusion(trajectory.layout)
            if over_budget:
                write_json_atomic(excl_json, over_budget)
                continue
            reference = run_condition(model, trajectory, Condition("reference", native=True))
            g5 = g5_for(reference, trajectory)
            if g5["excluded"]:
                write_json_atomic(excl_json, {"reason": g5["reason"], "g5": g5})
                continue
            capture = capture_attention(model, trajectory, keep_image_probs=idx in heatmap_set)
            arrays = {"mass": capture.mass, "vw_share": capture.vw_share, "enrichment": capture.enrichment,
                     "image_grid_thw": trajectory.image_grid_thw.numpy(),
                     "top1_region": capture.top1_region, "top1_region_nosink": capture.top1_region_nosink,
                     "top1_vw_region": capture.top1_vw_region, "top5_image": capture.top5_image,
                     "top5_image_nosink": capture.top5_image_nosink, "top5_image_count": capture.top5_image_count,
                     "top5_vw_image": capture.top5_vw_image,
                     # R26: current-block sub-classification (SELF/CUR_START/CUR_PREV/OUTSIDE).
                     "cb_top1_raw": capture.cb_top1_raw, "cb_top1_nosink": capture.cb_top1_nosink,
                     "cb_top1_vw": capture.cb_top1_vw, "cb_top5_raw": capture.cb_top5_raw,
                     "cb_top5_nosink": capture.cb_top5_nosink, "cb_top5_vw": capture.cb_top5_vw,
                     "cb_mass": capture.cb_mass, "cb_vw_share": capture.cb_vw_share}
            if capture.image_probs is not None:
                arrays["image_probs"] = capture.image_probs
            savez_atomic(args.run_dir / "attention" / f"q{idx}.npz", **arrays)
            meta = attention_meta(capture, trajectory, g5, eval_rows[idx])
            meta["precision"] = precision_record
            write_json_atomic(out_json, meta)
    require_exclusion_budget(g5_exclusion_count(args.run_dir))
    return 0


def stage_heads(args):
    from evaluation.latent_image_layout import IMAGE
    from evaluation.latent_image_trajectory import load_selection
    require_manifest(args.run_dir, _current_inputs(args), git_commit())
    require_exclusion_budget(g5_exclusion_count(args.run_dir))
    require_attention_complete(args.run_dir, load_selection(args.selection))
    per_question = []
    for idx, npz_path in question_files(args.run_dir / "attention", "npz").items():
        meta = json.loads((args.run_dir / "attention" / f"q{idx}.json").read_text())
        pad_rows = [i for i, spec in enumerate(meta["specs"]) if spec["slot"].startswith("pad")]
        if not pad_rows:
            continue
        with np.load(npz_path) as data:
            share = data["vw_share"][pad_rows][..., IMAGE].astype(np.float64)
        per_question.append(share.mean(axis=0))            # R14: each question weighs equally
    if not per_question:
        raise RuntimeError("no attention output available; run the attention stage first")
    mean_share = np.mean(per_question, axis=0)
    heads = select_heads(mean_share)
    write_json_atomic(args.run_dir / "heads.json", {"heads": [list(h) for h in heads], "share": mean_share.tolist()})
    return 0


def stage_interventions(args):
    from evaluation.latent_image_trajectory import build_trajectory, load_eval_rows, load_selection
    from evaluation.vision_block_runtime import numerical_policy
    manifest = require_manifest(args.run_dir, _current_inputs(args), git_commit())
    require_model_identity(manifest, args)
    selection = load_selection(args.selection)
    require_exclusion_budget(g5_exclusion_count(args.run_dir))
    require_attention_complete(args.run_dir, selection)
    heads_path = args.run_dir / "heads.json"
    if not heads_path.exists():
        raise RuntimeError("interventions requires heads.json; run the heads stage first")
    heads = [tuple(h) for h in json.loads(heads_path.read_text())["heads"]]
    swaps = json.loads((args.run_dir / "swaps.json").read_text())
    test_rows = load_test_rows(args.data)
    eval_rows = load_eval_rows(args.eval_rows)
    owned = shard_items(selection, args.shard, args.num_shards)
    excluded = excluded_indices(args.run_dir)
    model, processor, token_ids = _teacher_forced_setup(args)
    apply_precision(model, args.precision)
    with numerical_policy():
        for idx in owned:
            if idx in excluded:
                continue
            out_json = args.run_dir / "interventions" / f"q{idx}.json"
            out_npz = args.run_dir / "interventions" / f"q{idx}.npz"
            if skip_existing(out_json):
                continue
            trajectory = build_trajectory(processor, test_rows[idx], eval_rows[idx], idx, args.data_root, token_ids)
            swap_entry = swaps["questions"][str(idx)]
            swap_images = open_swap_images(swap_entry["swaps"], test_rows, args.data_root)
            results, matched, gates = question_interventions(model, processor, trajectory, heads, swap_images)
            summary = summarize_question(results, trajectory.layout, matched)
            savez_atomic(out_npz, **content_arrays(results))
            write_json_atomic(out_json, {**summary, "gates": gates})
    return 0


def stage_freegen(args):
    """Free generation under both image masks with the analysis numerics (R24):
    teacher-forced loading, `apply_precision` and `numerical_policy()`. Prompts over
    the length budget, and questions the trajectory regeneration skipped, get a
    record under `freegen/<mode>/skipped/` instead of a generation."""
    from evaluation import latent_image_trajectories_gen as gen
    from evaluation.latent_image_trajectory import load_eval_rows, load_selection
    from evaluation.vision_block_runtime import numerical_policy
    manifest = require_manifest(args.run_dir, _current_inputs(args), git_commit())
    require_model_identity(manifest, args)
    test_rows = load_test_rows(args.data)
    eval_rows = load_eval_rows(args.eval_rows)
    owned = shard_items(load_selection(args.selection), args.shard, args.num_shards)
    skipped = trajectory_skipped(manifest)
    pending = []
    for mode in ("causal_mask", "latent_mask"):
        out_dir = args.run_dir / "freegen" / mode
        for idx in owned:
            if skip_existing(out_dir / f"q{idx}.json", out_dir / "skipped" / f"q{idx}.json"):
                continue
            if idx in skipped:
                write_json_atomic(out_dir / "skipped" / f"q{idx}.json",
                                  {"skipped": "length_budget", "source": "trajectories_skipped"})
                continue
            pending.append((mode, idx))
    if not pending:
        return 0
    model, processor, _ = _teacher_forced_setup(args)
    precision_record = apply_precision(model, args.precision)
    with numerical_policy():
        for mode, idx in pending:
            out_dir = args.run_dir / "freegen" / mode
            result = gen.generate_question(model, processor, test_rows[idx], args.data_root, mode=mode,
                                           expected_prompt_identity=eval_rows.get(idx, {}).get("prompt_identity"))
            if "skipped" in result:
                write_json_atomic(out_dir / "skipped" / f"q{idx}.json", result)
                continue
            result["eval_match"] = bool(eval_rows[idx]["match"])      # R19: eval baseline over this set
            result["precision"] = precision_record
            write_json_atomic(out_dir / f"q{idx}.json", result)
    return 0


def stage_report(args):
    commit = git_commit()
    require_manifest(args.run_dir, _current_inputs(args), commit, allow_commit_change=True)
    from evaluation.latent_image_report import build_report
    build_report(args.run_dir, provenance={"report_commit": commit, "report_dirty": bool(git_status_porcelain())})
    return 0


def stage_trajectories(args):
    from evaluation.latent_image_trajectories_gen import stage_trajectories as run
    return run(args)


def stage_merge_trajectories(args):
    from evaluation.latent_image_trajectories_gen import stage_merge_trajectories as run
    return run(args)


STAGE_HANDLERS = {
    "prepare": stage_prepare, "smoke": stage_smoke, "attention": stage_attention, "heads": stage_heads,
    "interventions": stage_interventions, "freegen": stage_freegen, "report": stage_report,
    "trajectories": stage_trajectories, "merge-trajectories": stage_merge_trajectories,
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _model_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser


def _parser():
    parser = argparse.ArgumentParser(
        description="Staged CLI for the latent image attention analysis (see task-8-brief.md).")
    commands = parser.add_subparsers(dest="stage", required=True)
    model = _model_parser()
    for name in STAGES:
        stage = commands.add_parser(name, parents=[model])
        stage.add_argument("--eval-rows", type=Path, required=True)
        stage.add_argument("--run-dir", type=Path, required=True)
        if name == "attention":
            stage.add_argument("--smoke-from", type=Path, default=None,
                               help="reuse a prior run's passing smoke/gates.json instead of "
                                    "re-running the smoke stage (requires matching inputs, "
                                    "model identity, precision and trajectory provenance)")
    for name in TRAJECTORY_STAGES:          # standalone: run before `prepare`, no run dir
        stage = commands.add_parser(name, parents=[model])
        stage.add_argument("--out-dir", type=Path, required=True)
        stage.add_argument("--eval-rows", type=Path, default=None,
                           help="optional original eval rows; only their prompt_identity is checked")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        raise ValueError("--shard must be within [0, --num-shards)")
    if args.stage in STAGES:
        args.run_dir.mkdir(parents=True, exist_ok=True)
    code = STAGE_HANDLERS[args.stage](args)
    return int(code) if code else 0


if __name__ == "__main__":
    raise SystemExit(main())

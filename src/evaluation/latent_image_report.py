"""Evidence-level summary for latent image analysis runs."""
import json
from pathlib import Path

import numpy as np

from evaluation import latent_image_stats as st
from evaluation.latent_image_layout import IMAGE

GROUPS = ("pad", "start", "end", "pre", "post")
PALETTE = {"pad": "#2a78d6", "start": "#eb6834", "end": "#1baf7a", "pre": "#eda100", "post": "#e87ba4"}
PATCH_PALETTE = {"total": "#2a78d6", "direct": "#eb6834", "indirect": "#1baf7a"}
MATCHED_SCOPE = "size-matched blocks only"
RATIO_LABELS = {"direct_ratio_latent_cos_first_block": "primary (first perturbed block only, no propagation)",
                "direct_ratio_latent_cos": "all blocks, includes propagation",
                "direct_ratio_answer_dnll": "answer span (trajectory level, includes propagation)"}
SENSITIVITY_NAMES = ("k_all",) + tuple(f"{k}_{p}" for k in ("blur", "mean", "swap") for p in ("direct", "total"))
ALL_BLOCK_NOTE = ("all blocks, includes propagation: patching \"direct\"/\"indirect\" route edges over the whole "
                  "trajectory, so later-block \"direct\" effects include propagation through earlier perturbed "
                  "latents; `*_first_block` variants isolate the first perturbed block only")


def _files(run_dir, sub, suffix):
    """{question index: path} for q{idx}.{suffix}, ignoring leftover temp files (".tmp")."""
    return {int(p.name[1:-len(suffix) - 1]): p for p in sorted((Path(run_dir) / sub).glob(f"q*.{suffix}"))
            if ".tmp" not in p.name}


def _safe_paired(a_map, b_map, reason="empty paired set"):
    """paired_cluster_bootstrap over the (a_map & b_map) key intersection, guarded against
    an empty intersection (e.g. a control group with no text in every block — see Review
    Focus). Reused by every question-keyed paired bootstrap so the empty check lives in
    one place instead of being repeated at each call site."""
    qs = sorted(set(a_map) & set(b_map))
    if not qs:
        return {"status": "NO DATA", "reason": reason}
    return st.paired_cluster_bootstrap([a_map[q] for q in qs], [b_map[q] for q in qs], np.array(qs))


def _by_question(unit_dict, keys=None):
    """R14: collapse (question, block) units to one value per question (mean over the
    question's blocks; each block is already a mean over its rows)."""
    grouped = {}
    for key in (sorted(unit_dict) if keys is None else keys):
        grouped.setdefault(key[0], []).append(unit_dict[key])
    return {q: np.mean(v, 0) for q, v in grouped.items()}


def _paired_layer_diff(a_dict, b_dict, layers, reason="empty paired set"):
    """Per-layer paired_cluster_bootstrap over the (a_dict & b_dict) key intersection
    (keys are (question, block) units, values are per-layer arrays), guarded the same way
    as `_safe_paired`. Units are averaged within question first (R14) so every question
    carries equal weight. Returns a per-layer list normally, or a single
    {"status": "NO DATA"} dict when the intersection is empty (e.g. no post-text after the
    last block in any question)."""
    keys = sorted(set(a_dict) & set(b_dict))
    if not keys:
        return {"status": "NO DATA", "reason": reason}
    a_q, b_q = _by_question(a_dict, keys), _by_question(b_dict, keys)
    qs = sorted(a_q)
    a, b = np.stack([a_q[q] for q in qs]), np.stack([b_q[q] for q in qs])
    return [st.paired_cluster_bootstrap(a[:, l], b[:, l], np.array(qs)) for l in range(layers)]


def _units(npz, meta, field, region_axis=True):
    """(question, block) units: per-group mean over rows of `field`, averaged over heads.

    `mass`/`vw_share` carry a trailing region axis (index into it with IMAGE); `enrichment`
    is already an image-specific ratio with no region axis, so it is only averaged over heads.
    """
    units = {g: {} for g in GROUPS}
    for q, path in npz.items():
        with np.load(path) as data:
            arr = data[field].astype(np.float32)
        values = arr[..., IMAGE].mean(-1) if region_axis else arr.mean(-1)      # [rows, layers]
        for row, spec in enumerate(json.loads(meta[q].read_text())["specs"]):
            group = "pad" if spec["slot"].startswith("pad") else spec["slot"]
            units[group].setdefault((q, spec["block"]), []).append(values[row])
    return {g: {k: np.mean(v, 0) for k, v in d.items()} for g, d in units.items()}


def attention_section(run_dir):
    npz, meta = _files(run_dir, "attention", "npz"), _files(run_dir, "attention", "json")
    if not npz:
        return {"status": "NOT RUN"}
    units = _units(npz, meta, "vw_share")
    layers = next(iter(units["pad"].values())).shape[0]
    curves = {}
    for group, data in units.items():
        per_q = _by_question(data)
        if not per_q:
            continue
        qs = sorted(per_q)
        values, clusters = np.stack([per_q[q] for q in qs]), np.array(qs)
        curves[group] = [st.cluster_bootstrap(values[:, l], clusters) for l in range(layers)]
    diffs = {f"pad_minus_{control}": _paired_layer_diff(
                 units["pad"], units[control], layers, reason=f"no {control}-text units in any block")
             for control in ("pre", "post")}
    missing_controls = [c for c in ("pre", "post") if isinstance(diffs[f"pad_minus_{c}"], dict)]
    significant = []
    if not missing_controls:
        for l in range(layers):
            if all(diffs[f"pad_minus_{c}"][l]["lo"] > 0 for c in ("pre", "post")):
                significant.append(l)
    mass = _units(npz, meta, "mass")
    enrichment = _units(npz, meta, "enrichment", region_axis=False)
    heads = json.loads((run_dir / "heads.json").read_text()) if (run_dir / "heads.json").exists() else None
    return {"status": "RUN", "evidence": "descriptive", "vw_image_share": curves, "differences": diffs,
            "significant_layers": significant, "significant_layers_missing_controls": missing_controls,
            "mass_image_mean": {g: float(np.mean([v.mean() for v in _by_question(d).values()]))
                                for g, d in mass.items() if d},
            "enrichment_mean": {g: float(np.mean([v.mean() for v in _by_question(d).values()]))
                                for g, d in enrichment.items() if d},
            "heads": heads["heads"] if heads else None}


def _question_means(docs, name, metric, first_block=False, keep=None):
    """R14: one value per question. Per-block (list) metrics are averaged over the
    question's blocks (only block 0 when `first_block`); trajectory-level (scalar) metrics
    are taken as is. `keep` ({question: {block: bool}}) restricts per-block metrics to the
    kept blocks and admits a scalar metric only when every block of the question is kept
    (used for the R13 size-matched k_random rows)."""
    out = {}
    for q, doc in docs.items():
        v = doc[metric].get(name)
        if v is None:
            continue
        if isinstance(v, list):
            items = list(enumerate(v))[:1] if first_block else list(enumerate(v))
            if keep is not None:
                items = [(b, x) for b, x in items if keep.get(q, {}).get(b, False)]
            xs = [x for _, x in items if x is not None]
        else:
            if keep is not None and not (keep.get(q) and all(keep[q].values())):
                continue
            xs = [v]
        if xs:
            out[q] = float(np.mean(xs))
    return out


def _boot(per_question):
    if not per_question:
        return None
    qs = sorted(per_question)
    return st.cluster_bootstrap([per_question[q] for q in qs], np.array(qs))


def _table(docs, names, keep=None):
    """Knockout/patching table, one unit per question (R14). R5: lead each row with
    first-block metrics (isolate the first perturbed block); all-block metrics follow and
    are labeled in report.md."""
    table = {}
    for name in names:
        row = {}
        for metric in ("latent_cos", "post_dnll"):
            row[f"{metric}_first_block"] = _boot(_question_means(docs, name, metric, True, keep))
        for metric in ("latent_cos", "post_dnll", "post_flip", "answer_dnll", "answer_flip"):
            row[metric] = _boot(_question_means(docs, name, metric, False, keep))
        table[name] = row
    return table


def _size_matched(docs):
    """R13: {question: {block: random_matched == 1.0}} for questions with a k_random result,
    plus matched/total block counts."""
    keep = {q: {int(b): float(v) == 1.0 for b, v in d.get("random_matched", {}).items()}
            for q, d in docs.items() if d["latent_cos"].get("k_random") is not None}
    counts = {"matched_blocks": sum(sum(m.values()) for m in keep.values()),
              "total_blocks": sum(len(m) for m in keep.values())}
    return keep, counts


def _safe_ratio(num_map, den_map):
    """R11: an unstable (non-finite or zero) denominator, or a non-finite bootstrap
    result, is reported as null with a reason instead of inf/nan."""
    qs = sorted(set(num_map) & set(den_map))
    if not qs:
        return None
    den = np.array([den_map[q] for q in qs], dtype=np.float64)
    den_sum = float(np.nansum(den))
    if not np.isfinite(den_sum) or den_sum == 0:
        return {"mean": None, "lo": None, "hi": None, "n_units": len(qs), "n_clusters": len(qs),
                "reason": "unstable denominator"}
    result = st.ratio_cluster_bootstrap([num_map[q] for q in qs], [den_map[q] for q in qs], np.array(qs))
    if not all(np.isfinite(result[k]) for k in ("mean", "lo", "hi")):
        return {"mean": None, "lo": None, "hi": None, "n_units": result["n_units"],
                "n_clusters": result["n_clusters"], "reason": "unstable denominator"}
    return result


def intervention_sections(run_dir):
    paths = _files(run_dir, "interventions", "json")
    if not paths:
        return ({"status": "NOT RUN"},) * 3
    docs = {q: json.loads(p.read_text()) for q, p in paths.items()}
    keep, counts = _size_matched(docs)
    table = _table(docs, ("noop_repeat", "k_all", "k_heads"))
    table["k_random"] = {"scope": MATCHED_SCOPE, **counts, **_table(docs, ("k_random",), keep)["k_random"]}
    table["k_random_all_blocks_not_size_matched"] = {"scope": "all blocks, not size-matched",
                                                     **_table(docs, ("k_random",))["k_random"]}
    knockout = {"status": "RUN", "evidence": "causal", "note": ALL_BLOCK_NOTE, "table": table}
    a = _question_means(docs, "k_all", "latent_cos", keep=keep)
    b = _question_means(docs, "k_random", "latent_cos", keep=keep)
    knockout["k_all_minus_k_random_latent_cos"] = {"scope": MATCHED_SCOPE, **counts, **_safe_paired(
        a, b, reason="no question has both k_all and k_random latent_cos")}
    patching = {"status": "RUN", "evidence": "causal", "note": ALL_BLOCK_NOTE, "by_corruption": {}}
    for kind in ("blur", "mean", "swap"):
        names = tuple(f"{kind}_{n}" for n in ("total", "direct", "indirect"))
        entry = {"table": _table(docs, names)}
        d = _question_means(docs, f"{kind}_direct", "latent_cos", first_block=True)
        t = _question_means(docs, f"{kind}_total", "latent_cos", first_block=True)
        entry["direct_ratio_latent_cos_first_block"] = _safe_ratio(d, t)
        for metric in ("latent_cos", "answer_dnll"):
            d, t = _question_means(docs, f"{kind}_direct", metric), _question_means(docs, f"{kind}_total", metric)
            entry[f"direct_ratio_{metric}"] = _safe_ratio(d, t)
        entry["labels"] = RATIO_LABELS
        patching["by_corruption"][kind] = entry
    content = {"status": "RUN", "evidence": "information content", "sensitivity": {}, "identification": {}}
    for kind in ("blur", "mean", "swap"):
        lat = _question_means(docs, f"{kind}_total", "latent_cos")
        ctl = {q: d["control_cos"][f"{kind}_total"] for q, d in docs.items()
               if d["control_cos"].get(f"{kind}_total") is not None}
        content["sensitivity"][kind] = _safe_paired(
            lat, ctl, reason=f"no question has both {kind}_total latent_cos and control_cos")
    arrays = {q: np.load(p) for q, p in _files(run_dir, "interventions", "npz").items()}
    for label, body, image in (("mid", "mid_body", "image_mid"), ("final", "final_body", "image_final")):
        if not arrays:
            content["identification"][label] = {"status": "NO DATA", "reason": "no interventions/q*.npz arrays",
                                                "chance": 0.2}
            continue
        per_q = [st.identification_accuracy(a[body].mean(1)[None], a[image][None]) for a in arrays.values()]
        content["identification"][label] = st.cluster_bootstrap(np.array(per_q), np.array(list(arrays)))
        content["identification"][label]["chance"] = 0.2
    return knockout, patching, content


def truncation_sensitivity_section(run_dir):
    """R14: latent_cos / answer_dnll for k_all and each *_direct/*_total recomputed without
    questions whose eval stop_reason was "max_new_tokens" (recorded in attention/q*.json)."""
    paths = _files(run_dir, "interventions", "json")
    if not paths:
        return {"status": "NOT RUN"}
    docs = {q: json.loads(p.read_text()) for q, p in paths.items()}
    meta = {q: json.loads(p.read_text()) for q, p in _files(run_dir, "attention", "json").items()}
    truncated = sorted(q for q in docs if meta.get(q, {}).get("stop_reason") == "max_new_tokens")
    missing = sorted(q for q in docs if "stop_reason" not in meta.get(q, {}))
    kept = {q: d for q, d in docs.items() if q not in truncated}
    return {"status": "RUN", "evidence": "sensitivity", "excluded_truncated": truncated,
            "stop_reason_missing": missing, "questions": len(kept),
            "table": {name: {m: _boot(_question_means(kept, name, m)) for m in ("latent_cos", "answer_dnll")}
                      for name in SENSITIVITY_NAMES}}


def freegen_section(run_dir):
    modes = ("causal_mask", "latent_mask")
    if not all((run_dir / "freegen" / m).is_dir() for m in modes):
        return {"status": "NOT RUN"}
    results = {m: {q: json.loads(p.read_text())["match"] for q, p in _files(run_dir, f"freegen/{m}", "json").items()}
               for m in modes}
    qs = sorted(set(results["causal_mask"]) & set(results["latent_mask"]))
    causal_only = sum(results["causal_mask"][q] and not results["latent_mask"][q] for q in qs)
    latent_only = sum(results["latent_mask"][q] and not results["causal_mask"][q] for q in qs)
    # R10: free generation covers all selected questions regardless of G5 exclusions; report
    # the question count per mode as well as the paired (intersection) count used for McNemar.
    # R19: the eval baseline is counted over the same freegen question set (causal_mask files).
    baseline = [json.loads(p.read_text()).get("eval_match") for p in _files(run_dir, "freegen/causal_mask", "json").values()]
    baseline = [m for m in baseline if m is not None]
    return {"status": "RUN", "evidence": "accuracy", "questions": len(qs),
            "questions_per_mode": {m: len(results[m]) for m in modes},
            "correct": {m: int(sum(results[m][q] for q in qs)) for m in modes},
            "discordant": {"causal_only": causal_only, "latent_only": latent_only},
            "mcnemar_p": st.exact_mcnemar(causal_only, latent_only),
            "eval_baseline_correct": int(sum(baseline)), "eval_baseline_questions": len(baseline)}


def gates_section(run_dir):
    attention = {q: json.loads(p.read_text()) for q, p in _files(run_dir, "attention", "json").items()}
    interventions = {q: json.loads(p.read_text()) for q, p in _files(run_dir, "interventions", "json").items()}
    gates_only = {q: d.get("gates", {}) for q, d in interventions.items()}
    excluded = {q: json.loads(p.read_text()) for q, p in _files(run_dir, "attention/excluded", "json").items()}
    smoke = run_dir / "smoke" / "gates.json"
    matched_values = [v for d in interventions.values() for v in d.get("random_matched", {}).values()]
    random_matched = ({"n": len(matched_values), "mean": float(np.mean(matched_values)),
                       "min": float(np.min(matched_values)), "max": float(np.max(matched_values)),
                       "median": float(np.median(matched_values))} if matched_values else None)
    g5_rates = [a["g5"]["rate"] for a in attention.values() if "g5" in a]
    return {"smoke": json.loads(smoke.read_text()) if smoke.exists() else "NOT RUN",
            "g1_max": max((a["g1"] for a in attention.values()), default=None),
            "g1b_max": max((a["g1b"] for a in attention.values()), default=None),
            "g2_max": max((a["g2"] for a in attention.values()), default=None),
            "g3_fail": [q for q, g in gates_only.items() if not g.get("G3", {}).get("pass", False)],
            "g4_fail": [q for q, g in gates_only.items() if not g.get("G4", {}).get("pass", False)],
            "g5_rate_min": min(g5_rates) if g5_rates else None,
            "excluded": excluded, "excluded_count": len(excluded),
            # R12: length-budget exclusions are reported separately and are not G5 exclusions.
            "excluded_length_budget": sorted(q for q, d in excluded.items() if d.get("reason") == "length_budget"),
            "excluded_g5_count": sum(1 for d in excluded.values() if d.get("reason") != "length_budget"),
            "random_matched": random_matched}


ATTENTION_GATES = (("G1", "g1_max", 1e-2), ("G1b", "g1b_max", 1e-2), ("G2", "g2_max", 1e-3))


def attention_gate_failures(gates):
    """R18: attention gates over the analysed questions."""
    return [name for name, key, limit in ATTENTION_GATES if gates.get(key) is not None and gates[key] > limit]


def _markdown(summary):
    lines = ["# Latent image attention analysis", ""]
    if "report_commit" in summary:
        lines += [f"Report commit: `{summary['report_commit']}` (dirty tracked tree: {summary.get('report_dirty')})", ""]
    titles = {
        "attention": "③ Attention (evidence level: technical/descriptive)",
        "knockout": "① Knockout (evidence level: causal)",
        "patching": "② Image patching (evidence level: causal)",
        "content": "B. Latent information content (evidence level: information content)",
        "sensitivity_excluding_truncated": "Sensitivity excluding max_new_tokens-truncated questions",
        "freegen": "Free generation (evidence level: answer accuracy)",
        "gates": "Gates and exclusions",
    }
    notes = {
        "knockout": "`noop_repeat` is the noise floor: under the analysis numerical policy (SDPA MATH, TF32 "
                    "off) teacher-forced runs are deterministic, so its effects are expected 0. Metrics below cover all perturbed blocks unless suffixed `_first_block` "
                    "(first perturbed block only). Whole-trajectory numbers are " + ALL_BLOCK_NOTE + ".",
        "patching": "Direct/indirect are edge-routing effects over the whole trajectory: for later "
                    "blocks, \"direct\" includes propagation through earlier perturbed latents. Metrics "
                    "below cover all perturbed blocks unless suffixed `_first_block` (first perturbed "
                    "block only, i.e. no propagation).",
    }
    for key, title in titles.items():
        section = summary[key]
        status = section.get("status", "RUN") if isinstance(section, dict) else "RUN"
        lines += [f"## {title}", "", f"Status: **{status}**", ""]
        if status == "GATE FAILED" and key == "attention":
            lines += [f"{'/'.join(section.get('gate_failures', []))} exceeded its tolerance for at least one "
                      "analysed question; this section's numbers are **GATE FAILED** and should not be "
                      "trusted as descriptive evidence.", ""]
        elif status == "GATE FAILED":
            lines += ["G3/G4 failed for at least one question; this section's numbers are **GATE FAILED** "
                      "and should not be trusted as causal evidence.", ""]
        if key in notes and status not in ("NOT RUN",):
            lines += [notes[key], ""]
        if status != "NOT RUN":
            # Full JSON, never cut mid-object (summary.json holds the same content).
            lines += ["```json", json.dumps(section, indent=1), "```", ""]
    return "\n".join(lines)


def _ci_arrays(entries):
    mean = [e["mean"] for e in entries]
    lo = [e["lo"] for e in entries]
    hi = [e["hi"] for e in entries]
    return mean, lo, hi


def _patch_bars(patching):
    """Bars for the patching figure: one per available (corruption, part) latent_cos stat;
    missing stats get no bar (never a zero bar)."""
    bars = []
    by = patching.get("by_corruption", {})
    for part in ("total", "direct", "indirect"):
        for kind in by:
            row = by[kind]["table"].get(f"{kind}_{part}")
            stat = row.get("latent_cos") if row else None
            if stat and stat.get("mean") is not None:
                bars.append({"kind": kind, "part": part, "mean": stat["mean"], "lo": stat["lo"], "hi": stat["hi"]})
    return bars


def write_figures(run_dir, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)

    attention = summary.get("attention", {})
    if isinstance(attention, dict) and attention.get("status") == "RUN":
        curves = attention["vw_image_share"]
        n_layers = max((len(v) for v in curves.values()), default=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        for group in GROUPS:
            entries = curves.get(group)
            if not entries:
                continue
            mean, lo, hi = _ci_arrays(entries)
            x = list(range(len(entries)))
            ax.plot(x, mean, color=PALETTE[group], linewidth=2, label=group)
            ax.fill_between(x, lo, hi, color=PALETTE[group], alpha=0.2, linewidth=0)
        for layer in attention.get("significant_layers", []):
            ax.axvline(layer, color="#9a9a92", linewidth=1, linestyle=":", zorder=0)
        ax.set_xlabel("layer")
        ax.set_ylabel("value-weighted image share")
        ax.set_title("Value-weighted image attention share by layer")
        if n_layers:
            step = max(1, n_layers // 12)
            ax.set_xticks(list(range(0, n_layers, step)))
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures / "attention_layers.png", dpi=150)
        plt.close(fig)

    patching = summary.get("patching", {})
    if isinstance(patching, dict) and patching.get("status") == "RUN":
        kinds = list(patching.get("by_corruption", {}))
        if kinds:
            fig, ax = plt.subplots(figsize=(8, 5))
            width = 0.25
            xs = np.arange(len(kinds))
            bars = _patch_bars(patching)
            for i, part in enumerate(("total", "direct", "indirect")):
                mine = [b for b in bars if b["part"] == part]
                if not mine:
                    continue
                offset = [xs[kinds.index(b["kind"])] + (i - 1) * width for b in mine]
                ax.bar(offset, [b["mean"] for b in mine], width=width, color=PATCH_PALETTE[part], label=part,
                       yerr=[[b["mean"] - b["lo"] for b in mine], [b["hi"] - b["mean"] for b in mine]], capsize=3)
            ax.set_xticks(xs)
            ax.set_xticklabels(kinds)
            ax.set_ylabel("latent_cos mean (all blocks, includes propagation)")
            ax.set_title("Image patching effect by corruption")
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(figures / "patching.png", dpi=150)
            plt.close(fig)

    if isinstance(attention, dict) and attention.get("status") == "RUN":
        for q, path in _files(run_dir, "attention", "npz").items():
            with np.load(path) as data:
                if "image_probs" not in data:
                    continue
                probs = data["image_probs"].astype(np.float32)   # [rows, layers, n_image]
                grid = data["image_grid_thw"][0]
            meta = json.loads((run_dir / "attention" / f"q{q}.json").read_text())
            pad_rows = [i for i, s in enumerate(meta["specs"]) if s["slot"].startswith("pad")]
            if not pad_rows:
                continue
            layer_mean = probs[pad_rows].mean(axis=(0, 1))   # [n_image]
            h, w = int(grid[1]) // 2, int(grid[2]) // 2
            if h * w != layer_mean.size:
                continue
            heat = layer_mean.reshape(h, w)
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(heat, cmap="Blues")
            ax.set_title(f"q{q}: pad-row image attention")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(figures / f"heatmap_q{q}.png", dpi=150)
            plt.close(fig)


def build_report(run_dir, provenance=None):
    """`provenance` (e.g. {"report_commit", "report_dirty"}) is merged into summary.json."""
    run_dir = Path(run_dir)
    knockout, patching, content = intervention_sections(run_dir)
    summary = {"attention": attention_section(run_dir), "knockout": knockout, "patching": patching,
               "content": content, "sensitivity_excluding_truncated": truncation_sensitivity_section(run_dir),
               "freegen": freegen_section(run_dir), "gates": gates_section(run_dir)}
    summary.update(provenance or {})
    if summary["gates"]["g3_fail"] or summary["gates"]["g4_fail"]:
        for key in ("knockout", "patching", "content", "sensitivity_excluding_truncated"):
            if summary[key].get("status") == "RUN":
                summary[key]["status"] = "GATE FAILED"
    failures = attention_gate_failures(summary["gates"])
    if failures and summary["attention"].get("status") == "RUN":
        summary["attention"].update(status="GATE FAILED", gate_failures=failures)
    summary = json.loads(json.dumps(summary, default=float))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "report.md").write_text(_markdown(summary))
    write_figures(run_dir, summary)
    return summary

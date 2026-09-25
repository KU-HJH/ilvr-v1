"""Evidence-level summary for latent image analysis runs."""
import json
from pathlib import Path

import numpy as np

from evaluation import latent_image_stats as st
from evaluation.latent_image_layout import IMAGE

GROUPS = ("pad", "start", "end", "pre", "post")
PALETTE = {"pad": "#2a78d6", "start": "#eb6834", "end": "#1baf7a", "pre": "#eda100", "post": "#e87ba4"}
PATCH_PALETTE = {"total": "#2a78d6", "direct": "#eb6834", "indirect": "#1baf7a"}
ALL_BLOCK_NOTE = ("all blocks, includes propagation: patching \"direct\"/\"indirect\" route edges over the whole "
                  "trajectory, so later-block \"direct\" effects include propagation through earlier perturbed "
                  "latents; `*_first_block` variants isolate the first perturbed block only")


def _files(run_dir, sub, suffix):
    return {int(p.stem[1:]): p for p in sorted((run_dir / sub).glob(f"q*.{suffix}"))}


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
        keys = sorted(data)
        if not keys:
            continue
        values, clusters = np.stack([data[k] for k in keys]), np.array([k[0] for k in keys])
        curves[group] = [st.cluster_bootstrap(values[:, l], clusters) for l in range(layers)]
    diffs, significant = {}, []
    for control in ("pre", "post"):
        keys = sorted(set(units["pad"]) & set(units[control]))
        a = np.stack([units["pad"][k] for k in keys])
        b = np.stack([units[control][k] for k in keys])
        clusters = np.array([k[0] for k in keys])
        diffs[f"pad_minus_{control}"] = [st.paired_cluster_bootstrap(a[:, l], b[:, l], clusters) for l in range(layers)]
    for l in range(layers):
        if all(diffs[f"pad_minus_{c}"][l]["lo"] > 0 for c in ("pre", "post")):
            significant.append(l)
    mass = _units(npz, meta, "mass")
    enrichment = _units(npz, meta, "enrichment", region_axis=False)
    heads = json.loads((run_dir / "heads.json").read_text()) if (run_dir / "heads.json").exists() else None
    return {"status": "RUN", "evidence": "descriptive", "vw_image_share": curves, "differences": diffs,
            "significant_layers": significant,
            "mass_image_mean": {g: float(np.mean([v.mean() for v in d.values()])) for g, d in mass.items() if d},
            "enrichment_mean": {g: float(np.mean([v.mean() for v in d.values()])) for g, d in enrichment.items() if d},
            "heads": heads["heads"] if heads else None}


def _metric_units(docs, name, metric, first_block=False):
    values, clusters = [], []
    for q, doc in docs.items():
        v = doc[metric].get(name)
        if v is None:
            continue
        v = v[:1] if first_block and isinstance(v, list) else v
        for x in (v if isinstance(v, list) else [v]):
            if x is not None:
                values.append(x)
                clusters.append(q)
    return np.array(values, dtype=np.float64), np.array(clusters)


def _table(docs, names):
    """Knockout/patching table. R5: lead each row with first-block metrics (isolate the
    first perturbed block); all-block metrics follow and are labeled in report.md."""
    table = {}
    for name in names:
        row = {}
        for metric in ("latent_cos", "post_dnll"):
            v, c = _metric_units(docs, name, metric, first_block=True)
            row[f"{metric}_first_block"] = st.cluster_bootstrap(v, c) if len(v) else None
        for metric in ("latent_cos", "post_dnll", "post_flip", "answer_dnll", "answer_flip"):
            v, c = _metric_units(docs, name, metric)
            row[metric] = st.cluster_bootstrap(v, c) if len(v) else None
        table[name] = row
    return table


def _question_means(docs, name, metric):
    return {q: float(np.mean([x for x in (d[metric][name] if isinstance(d[metric][name], list) else [d[metric][name]])
                              if x is not None]))
            for q, d in docs.items() if d[metric].get(name) is not None}


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
    knockout = {"status": "RUN", "evidence": "causal", "note": ALL_BLOCK_NOTE,
                "table": _table(docs, ("noop_repeat", "k_all", "k_heads", "k_random"))}
    a, b = _question_means(docs, "k_all", "latent_cos"), _question_means(docs, "k_random", "latent_cos")
    qs = sorted(set(a) & set(b))
    knockout["k_all_minus_k_random_latent_cos"] = st.paired_cluster_bootstrap(
        [a[q] for q in qs], [b[q] for q in qs], np.array(qs))
    patching = {"status": "RUN", "evidence": "causal", "note": ALL_BLOCK_NOTE, "by_corruption": {}}
    for kind in ("blur", "mean", "swap"):
        names = tuple(f"{kind}_{n}" for n in ("total", "direct", "indirect"))
        entry = {"table": _table(docs, names)}
        for metric in ("latent_cos", "answer_dnll"):
            d, t = _question_means(docs, f"{kind}_direct", metric), _question_means(docs, f"{kind}_total", metric)
            entry[f"direct_ratio_{metric}"] = _safe_ratio(d, t)
        patching["by_corruption"][kind] = entry
    content = {"status": "RUN", "evidence": "information content", "sensitivity": {}, "identification": {}}
    for kind in ("blur", "mean", "swap"):
        lat = _question_means(docs, f"{kind}_total", "latent_cos")
        ctl = {q: d["control_cos"][f"{kind}_total"] for q, d in docs.items()
               if d["control_cos"].get(f"{kind}_total") is not None}
        qs = sorted(set(lat) & set(ctl))
        content["sensitivity"][kind] = st.paired_cluster_bootstrap(
            [lat[q] for q in qs], [ctl[q] for q in qs], np.array(qs)) if qs else None
    arrays = {q: np.load(p) for q, p in _files(run_dir, "interventions", "npz").items()}
    for label, body, image in (("mid", "mid_body", "image_mid"), ("final", "final_body", "image_final")):
        per_q = [st.identification_accuracy(a[body].mean(1)[None], a[image][None]) for a in arrays.values()]
        content["identification"][label] = st.cluster_bootstrap(np.array(per_q), np.array(list(arrays)))
        content["identification"][label]["chance"] = 0.2
    return knockout, patching, content


def freegen_section(run_dir):
    modes = ("causal_mask", "latent_mask")
    if not all((run_dir / "freegen" / m).is_dir() for m in modes):
        return {"status": "NOT RUN"}
    results = {m: {int(p.stem[1:]): json.loads(p.read_text())["match"]
                   for p in (run_dir / "freegen" / m).glob("q*.json")} for m in modes}
    qs = sorted(set(results["causal_mask"]) & set(results["latent_mask"]))
    causal_only = sum(results["causal_mask"][q] and not results["latent_mask"][q] for q in qs)
    latent_only = sum(results["latent_mask"][q] and not results["causal_mask"][q] for q in qs)
    # R10: free generation covers all selected questions regardless of G5 exclusions; report
    # the question count per mode as well as the paired (intersection) count used for McNemar.
    attn_meta = {q: json.loads(p.read_text()) for q, p in _files(run_dir, "attention", "json").items()}
    eval_correct = sum(1 for m in attn_meta.values() if m.get("eval_match"))
    return {"status": "RUN", "evidence": "accuracy", "questions": len(qs),
            "questions_per_mode": {m: len(results[m]) for m in modes},
            "correct": {m: int(sum(results[m][q] for q in qs)) for m in modes},
            "discordant": {"causal_only": causal_only, "latent_only": latent_only},
            "mcnemar_p": st.exact_mcnemar(causal_only, latent_only),
            "eval_baseline_correct": eval_correct, "eval_baseline_questions": len(attn_meta)}


def gates_section(run_dir):
    attention = {q: json.loads(p.read_text()) for q, p in _files(run_dir, "attention", "json").items()}
    interventions = {q: json.loads(p.read_text()) for q, p in _files(run_dir, "interventions", "json").items()}
    gates_only = {q: d.get("gates", {}) for q, d in interventions.items()}
    excluded = {int(p.stem[1:]): json.loads(p.read_text())
                for p in sorted((run_dir / "attention" / "excluded").glob("q*.json"))}
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
            "random_matched": random_matched}


def _markdown(summary):
    lines = ["# Latent image attention analysis", ""]
    titles = {
        "attention": "③ Attention (evidence level: technical/descriptive)",
        "knockout": "① Knockout (evidence level: causal)",
        "patching": "② Image patching (evidence level: causal)",
        "content": "B. Latent information content (evidence level: information content)",
        "freegen": "Free generation (evidence level: answer accuracy)",
        "gates": "Gates and exclusions",
    }
    notes = {
        "knockout": "Metrics below cover all perturbed blocks unless suffixed `_first_block` "
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
        if status == "GATE FAILED":
            lines += ["G3/G4 failed for at least one question; this section's numbers are **GATE FAILED** "
                      "and should not be trusted as causal evidence.", ""]
        if key in notes and status not in ("NOT RUN",):
            lines += [notes[key], ""]
        if status != "NOT RUN":
            lines += ["```json", json.dumps(section, indent=1)[:20000], "```", ""]
    return "\n".join(lines)


def _ci_arrays(entries):
    mean = [e["mean"] for e in entries]
    lo = [e["lo"] for e in entries]
    hi = [e["hi"] for e in entries]
    return mean, lo, hi


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
            for i, part in enumerate(("total", "direct", "indirect")):
                means, err_lo, err_hi = [], [], []
                for kind in kinds:
                    row = patching["by_corruption"][kind]["table"].get(f"{kind}_{part}")
                    stat = row.get("latent_cos") if row else None
                    mean = stat["mean"] if stat else 0.0
                    means.append(mean)
                    err_lo.append(mean - stat["lo"] if stat else 0.0)
                    err_hi.append(stat["hi"] - mean if stat else 0.0)
                offset = xs + (i - 1) * width
                ax.bar(offset, means, width=width, color=PATCH_PALETTE[part], label=part,
                       yerr=[err_lo, err_hi], capsize=3)
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


def build_report(run_dir):
    run_dir = Path(run_dir)
    knockout, patching, content = intervention_sections(run_dir)
    summary = {"attention": attention_section(run_dir), "knockout": knockout, "patching": patching,
               "content": content, "freegen": freegen_section(run_dir), "gates": gates_section(run_dir)}
    if summary["gates"]["g3_fail"] or summary["gates"]["g4_fail"]:
        for key in ("knockout", "patching", "content"):
            if summary[key].get("status") == "RUN":
                summary[key]["status"] = "GATE FAILED"
    summary = json.loads(json.dumps(summary, default=float))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "report.md").write_text(_markdown(summary))
    write_figures(run_dir, summary)
    return summary

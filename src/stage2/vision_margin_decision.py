"""Pre-registered verdict rules for the Stage 2 Vision-margin experiment.

Thresholds (documents/stage2-vision-margin-preregistration.md), Δ = probe − step0:
(a) ΔS_F ≥ +0.05 and ΔS_F(arm) ≥ ΔS_F(CE) − 0.02
(b) ΔS_F ≥ 0.4·ΔG and ΔS_M ≥ −0.10
(c) G_gt_0 ≥ 0.60 and ΔG ≥ +0.05
(d) TEST400 accuracy ≥ 0.51 and exact one-sided McNemar p ≥ 0.05 versus CE-only
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mcnemar_one_sided_decrease(ce_correct, arm_correct):
    """Exact one-sided test of H1: the arm is worse than CE on paired items."""
    if len(ce_correct) != len(arm_correct):
        raise ValueError("paired outcomes must have equal length")
    b = sum(1 for ce, arm in zip(ce_correct, arm_correct) if ce and not arm)
    c = sum(1 for ce, arm in zip(ce_correct, arm_correct) if arm and not ce)
    n = b + c
    p = 1.0 if n == 0 else sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n
    return {"b_ce_only_correct": b, "c_arm_only_correct": c, "b": b, "c": c,
            "discordant": n, "p_value": min(1.0, p)}


def _verdict(ok):
    return "PASS" if ok else "FAIL"


def decide_arm(probe, *, step0, ce, accuracy, p_value):
    d_sf = probe["S_F"] - step0["S_F"]
    d_sm = probe["S_M"] - step0["S_M"]
    d_g = probe["G_mean"] - step0["G_mean"]
    d_sf_ce = ce["S_F"] - step0["S_F"]
    criteria = {
        "a": {"verdict": _verdict(d_sf >= 0.05 and d_sf >= d_sf_ce - 0.02),
              "delta_S_F": d_sf, "delta_S_F_ce": d_sf_ce},
        "b": {"verdict": _verdict(d_sf >= 0.4 * d_g and d_sm >= -0.10),
              "delta_S_F": d_sf, "delta_G": d_g, "delta_S_M": d_sm},
        "c": {"verdict": _verdict(probe["G_gt_0"] >= 0.60 and d_g >= 0.05),
              "G_gt_0": probe["G_gt_0"], "delta_G": d_g},
    }
    if accuracy is None or p_value is None:
        criteria["d"] = {"verdict": "undecided", "accuracy": accuracy, "p_value": p_value}
    else:
        criteria["d"] = {"verdict": _verdict(accuracy >= 0.51 and p_value >= 0.05),
                         "accuracy": accuracy, "p_value": p_value}
    mechanism = "PASS" if all(criteria[k]["verdict"] == "PASS" for k in "abc") else "FAIL"
    return {"criteria": criteria, "mechanism": mechanism,
            "no_regression": criteria["d"]["verdict"], "provisional": True}


def _correct_by_index(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {row["index"]: bool(row["match"]) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--ce-eval", type=Path, required=True)
    parser.add_argument("--arm-eval", action="append", default=[], metavar="ARM=MERGED_JSONL")
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    probes = {name: json.loads((args.probe_dir / f"probe_{name}.json").read_text())
              for name in ["step0", "ce", *args.arms]}
    evals = dict(item.split("=", 1) for item in args.arm_eval)
    ce_correct = _correct_by_index(args.ce_eval)
    mcnemar = {"test": "exact", "alternative": "one-sided-decrease", "alpha": 0.05,
               "control": str(args.ce_eval), "control_accuracy": sum(ce_correct.values()) / len(ce_correct),
               "arms": {}}
    table = {}
    for arm in args.arms:
        accuracy = p_value = None
        if arm in evals:
            arm_correct = _correct_by_index(evals[arm])
            if sorted(arm_correct) != sorted(ce_correct):
                raise ValueError(f"{arm}: evaluation indices do not match the CE control")
            keys = sorted(ce_correct)
            test = mcnemar_one_sided_decrease([ce_correct[k] for k in keys], [arm_correct[k] for k in keys])
            accuracy = sum(arm_correct.values()) / len(arm_correct)
            p_value = test["p_value"]
            mcnemar["arms"][arm] = {"eval": evals[arm], "accuracy": accuracy, **test}
        table[arm] = decide_arm(probes[arm], step0=probes["step0"], ce=probes["ce"],
                                accuracy=accuracy, p_value=p_value)
    (args.probe_dir / "test400_mcnemar.json").write_text(json.dumps(mcnemar, indent=2) + "\n")
    (args.probe_dir / "decision_table.json").write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps({arm: {k: v["verdict"] for k, v in row["criteria"].items()}
                      | {"mechanism": row["mechanism"], "no_regression": row["no_regression"]}
                      for arm, row in table.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

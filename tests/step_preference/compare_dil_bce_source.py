"""CPU-only, hash-gated comparison of the pinned source and port pure losses."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from step_preference import dil_bce_loss, dil_topk_loss, entropy_selection
from step_preference.config import StepPreferenceConfig
from step_preference.dil_reduction import reduce_dil_context_loss


FILES = {name: f"src/step_preference/{name}.py" for name in
         ("dil_bce_loss", "dil_topk_loss", "entropy_selection")}
FIELDS = ("negative_loss", "mass_loss", "entropy", "clean_mass", "off_mass",
          "mass_abs_gap", "gt_coverage")


class IdentityError(ValueError):
    pass


def _load_verified(source_root, manifest):
    identity = json.loads(manifest.read_text())
    if Path(identity["source"]).resolve() != source_root.resolve():
        raise IdentityError("source root differs from identity manifest")
    for relative in FILES.values():
        expected = identity["sha256"][relative]
        actual = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise IdentityError(f"source identity mismatch: {relative}: {actual} != {expected}")
    modules = {}
    for name, relative in FILES.items():
        module_name = f"_verified_dil_source_{name}"
        spec = importlib.util.spec_from_file_location(module_name, source_root / relative)
        module = importlib.util.module_from_spec(spec)
        # dataclass resolves its annotations through sys.modules at definition time.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules, {relative: identity["sha256"][relative] for relative in FILES.values()}


def _tensor_error(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max())


def _case(name, inputs, values, *, gradient_fields=()):
    errors = {field: _tensor_error(*pair) for field, pair in values.items()}
    passed = all(torch.allclose(*pair, atol=1e-5 if field in gradient_fields else 1e-6,
                                rtol=1e-4 if field in gradient_fields else 1e-5)
                 for field, pair in values.items())
    return {"name": name, "input": inputs, "errors": errors,
            "max_error": max(errors.values(), default=0.0), "pass": passed}


def _bce_cases(source):
    cases = []
    for variant, clean_values, off_values in (
        ("ordinary", [-1.3, 0.2, 1.9], [0.7, -0.9, 0.4]),
        ("unequal_steps", [-2.0, 0.5], [1.1, -0.4]),
    ):
        clean = torch.tensor(clean_values, requires_grad=True)
        off = torch.tensor(off_values, requires_grad=True)
        source_result = source.dil_bce_loss(clean, off, 0.5)
        port_result = dil_bce_loss.dil_bce_loss(clean, off, 0.5)
        sg = torch.autograd.grad(source_result.loss, (clean, off), retain_graph=True,
                                 allow_unused=True)
        sg = tuple(torch.zeros_like(x) if g is None else g for x, g in zip((clean, off), sg))
        pg = torch.autograd.grad(port_result.loss, (clean, off))
        values = {field: (getattr(source_result, field), getattr(port_result, field))
                  for field in ("loss", "pair_losses", "positive_losses", "negative_losses", "policy_margins")}
        values.update(clean_gradient=(sg[0], pg[0]), off_gradient=(sg[1], pg[1]))
        cases.append(_case(f"bce_{variant}", {"kind": "bce", "beta": 0.5,
                     "clean_scores": clean_values, "off_scores": off_values}, values,
                     gradient_fields=("clean_gradient", "off_gradient")))
    return cases


def _topk_cases(source):
    cases = []
    for k in (5, 8, 50, 1516, 15166):
        vocab, tokens = k + 7, 3 if k <= 50 else 2
        generator = torch.Generator().manual_seed(42 + k)
        base_clean = torch.randn(tokens, vocab, generator=generator) * 0.8
        base_off = torch.randn(tokens, vocab, generator=generator) * 0.9
        ids = torch.arange(tokens, dtype=torch.long) + 1
        for mass_coefficient in (0.0, 1.0):
            clean = base_clean.detach().clone().requires_grad_()
            off = base_off.detach().clone().requires_grad_()
            source_result = source.topk_negative_and_mass(clean, off, ids, beta=0.5,
                                                          k=k, compute_mass=True)
            port_result = dil_topk_loss.topk_negative_and_mass(clean, off, ids, beta=0.5,
                                                               k=k, compute_mass=True)
            source_loss = source_result.negative_loss + mass_coefficient * source_result.mass_loss
            port_loss = port_result.negative_loss + mass_coefficient * port_result.mass_loss
            sg = torch.autograd.grad(source_loss, (clean, off), retain_graph=True,
                                     allow_unused=True)
            pg = torch.autograd.grad(port_loss, (clean, off), allow_unused=True)
            sg = tuple(torch.zeros_like(x) if g is None else g for x, g in zip((clean, off), sg))
            pg = tuple(torch.zeros_like(x) if g is None else g for x, g in zip((clean, off), pg))
            values = {field: (getattr(source_result, field), getattr(port_result, field))
                      for field in FIELDS}
            values.update(clean_gradient=(sg[0], pg[0]), off_gradient=(sg[1], pg[1]))
            cases.append(_case(f"topk_k{k}_mass{mass_coefficient:g}",
                         {"kind": "topk", "seed": 42 + k, "k": k, "vocab": vocab,
                          "tokens": tokens, "mass_coefficient": mass_coefficient}, values,
                         gradient_fields=("clean_gradient", "off_gradient")))
    return cases


def _entropy_cases(source):
    cases = []
    for entropy_source in ("clean", "voff"):
        for percent in (20.0, 100.0, 28.0):
            # 25 tied tokens at 28% select exactly seven, in original token order.
            lengths = (2, 3, 1) if percent != 28.0 else (12, 13)
            generator = torch.Generator().manual_seed(42)
            clean = [torch.randn(n, 9, generator=generator) for n in lengths]
            off = [torch.randn(n, 9, generator=generator) for n in lengths]
            selected_logits = clean if entropy_source == "clean" else off
            if percent == 28.0:
                selected_logits = [torch.zeros_like(x) for x in selected_logits]
            items = [SimpleNamespace(sample_id="sample-42",
                     span=SimpleNamespace(step_id=i, target_text_start=sum(lengths[:i])),
                     chosen_target_ids=torch.arange(n)) for i, n in enumerate(lengths)]
            logits_by_item = {("sample-42", i): x for i, x in enumerate(selected_logits)}
            source_selection = source.select_sample_entropy_masks(items, logits_by_item, percent)
            entropies = []
            for logits in selected_logits:
                logp = logits.float().log_softmax(-1)
                entropies.append(-(logp.exp() * logp).sum(-1))
            port_masks = entropy_selection.select_entropy_masks(entropies, percent)
            source_mask = torch.cat([source_selection.masks[("sample-42", i)] for i in range(len(lengths))])
            port_mask = torch.cat(port_masks)
            values = {"mask": (source_mask.float(), port_mask.float())}
            cases.append(_case(f"entropy_{entropy_source}_{percent:g}",
                         {"kind": "entropy", "source": entropy_source, "percent": percent,
                          "lengths": lengths, "selected": int(source_mask.sum())}, values))
    return cases


def _reduction_cases(source_topk, source_bce, source_entropy):
    cases = []
    generator = torch.Generator().manual_seed(42)
    for mode in ("bce", "topk"):
        for percent in ((100.0,) if mode == "bce" else (20.0, 100.0)):
            lengths = (1, 4, 2)
            clean_logits = [torch.randn(n, 11, generator=generator).requires_grad_() for n in lengths]
            off_logits = [torch.randn(n, 11, generator=generator).requires_grad_() for n in lengths]
            ids = [torch.arange(n, dtype=torch.long) + 1 for n in lengths]
            clean_logps = [x.log_softmax(-1).gather(-1, y[:, None]).squeeze(-1) for x, y in zip(clean_logits, ids)]
            off_logps = [x.log_softmax(-1).gather(-1, y[:, None]).squeeze(-1) for x, y in zip(off_logits, ids)]
            items = [SimpleNamespace(sample_id="sample-42",
                     span=SimpleNamespace(step_id=i, target_text_start=sum(lengths[:i])),
                     chosen_target_ids=y) for i, y in enumerate(ids)]
            # Source trainer applies these masks before each pure loss call.
            source_masks = (source_entropy.select_sample_entropy_masks(
                items, {("sample-42", i): x for i, x in enumerate(clean_logits)}, percent).masks
                if mode == "topk" else {("sample-42", i): torch.ones(n, dtype=torch.bool)
                                        for i, n in enumerate(lengths)})
            entropy = [-(x.softmax(-1) * x.log_softmax(-1)).sum(-1) for x in clean_logits]
            port_masks = (entropy_selection.select_entropy_masks(entropy, percent)
                          if mode == "topk" else tuple(source_masks[("sample-42", i)]
                                                       for i in range(len(lengths))))
            source_steps = []
            for i, (clean, off, target) in enumerate(zip(clean_logits, off_logits, ids)):
                key = ("sample-42", i)
                mask = source_masks[key]
                if not bool(mask.any()):
                    source_steps.append(clean.sum() * 0.0 + off.sum() * 0.0)
                elif mode == "bce":
                    source_steps.append(source_bce.dil_bce_loss(clean_logps[i][mask].mean(),
                                        off_logps[i][mask].mean(), 0.5).loss)
                else:
                    topk = source_topk.topk_negative_and_mass(clean[mask], off[mask], target[mask],
                                                               beta=0.5, k=5)
                    positive = source_bce.dil_bce_positive(clean_logps[i][mask].mean(), 0.5)
                    source_steps.append(positive + topk.negative_loss + 0.3 * topk.mass_loss)
            source_loss = torch.stack(source_steps).mean()
            topk_terms = ([dil_topk_loss.topk_token_terms(clean, off, target, beta=0.5, k=5)
                           for clean, off, target in zip(clean_logits, off_logits, ids)]
                          if mode == "topk" else None)
            config = StepPreferenceConfig(
                enabled=True, mode="stage2b", objective="context_preference",
                preference_loss_type="dil_bce_rf_topk" if mode == "topk" else "dil_bce_rf",
                preference_topk=5 if mode == "topk" else None,
                lambda_topk_mass=0.3 if mode == "topk" else 0.0,
                preference_entropy_top_percent=percent if mode == "topk" else None, beta=0.5,
                lambda_preference=1.0, lambda_ce=1.0, lambda_entropy=0.0,
                preference_margin=0.0, shared_prefix_enabled=True,
                visual_off_suppress_latent_boundaries=False, direct_from_stage1=True)
            port_loss, _ = reduce_dil_context_loss(
                clean_logits[0].new_zeros(()), clean_logps, off_logps,
                config=config, topk_terms=topk_terms, selection_entropies=entropy)
            sg = torch.autograd.grad(source_loss, clean_logits + off_logits,
                                     retain_graph=True, allow_unused=True)
            sg = tuple(torch.zeros_like(x) if g is None else g
                       for x, g in zip(clean_logits + off_logits, sg))
            pg = torch.autograd.grad(port_loss, clean_logits + off_logits)
            values = {"selected_step_loss": (source_loss, port_loss),
                      "clean_gradient": (torch.cat([x.flatten() for x in sg[:3]]),
                                         torch.cat([x.flatten() for x in pg[:3]])),
                      "off_gradient": (torch.cat([x.flatten() for x in sg[3:]]),
                                       torch.cat([x.flatten() for x in pg[3:]]))}
            cases.append(_case(f"reduction_{mode}_{percent:g}",
                         {"kind": "reduction", "mode": mode, "percent": percent,
                          "lengths": lengths, "selected": int(sum(m.sum() for m in port_masks))},
                         values, gradient_fields=("clean_gradient", "off_gradient")))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"verdict": "error", "source": str(args.source_root), "cases": []}
    try:
        modules, hashes = _load_verified(args.source_root, args.identity_manifest)
        report["sha256"] = hashes
        report["cases"] = (_bce_cases(modules["dil_bce_loss"])
                           + _topk_cases(modules["dil_topk_loss"])
                           + _entropy_cases(modules["entropy_selection"])
                           + _reduction_cases(modules["dil_topk_loss"], modules["dil_bce_loss"],
                                              modules["entropy_selection"]))
        report["verdict"] = "pass" if all(case["pass"] for case in report["cases"]) else "fail"
    except (IdentityError, KeyError, FileNotFoundError) as exc:
        report["verdict"] = "identity_mismatch"
        report["error"] = str(exc)
    except Exception as exc:
        import traceback
        report["error"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['verdict']}: {len(report['cases'])} cases", file=sys.stderr)
    if "error" in report:
        print(report["error"], file=sys.stderr)
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

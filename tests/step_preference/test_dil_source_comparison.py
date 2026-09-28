"""Source parity witness uses only hash-verified pure files and CPU tensors."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).with_name("compare_dil_bce_source.py")
PURE = ("dil_bce_loss.py", "dil_topk_loss.py", "entropy_selection.py")


def _fixture(tmp_path):
    source = tmp_path / "source"
    folder = source / "src" / "step_preference"
    folder.mkdir(parents=True)
    current = Path(__file__).resolve().parents[2] / "src" / "step_preference"
    for name in PURE:
        (folder / name).write_bytes((current / name).read_bytes())
    # The external source's entropy entry point has a different calling shape.
    (folder / "entropy_selection.py").write_text('''import math
from fractions import Fraction
import torch
from types import SimpleNamespace

def select_sample_entropy_masks(items, logits_by_item, percent):
    vectors = []
    for item in items:
        logits = logits_by_item[(item.sample_id, item.span.step_id)]
        logp = logits.log_softmax(-1)
        vectors.append(-(logp.exp() * logp).sum(-1))
    values = torch.cat(vectors)
    keep = math.ceil(len(values) * Fraction(str(percent)) / 100)
    selected = torch.zeros(len(values), dtype=torch.bool)
    selected[torch.argsort(values, descending=True, stable=True)[:keep]] = True
    masks, cursor = {}, 0
    for item, vector in zip(items, vectors):
        masks[(item.sample_id, item.span.step_id)] = selected[cursor:cursor + len(vector)]
        cursor += len(vector)
    return SimpleNamespace(masks=masks)
''')
    manifest = tmp_path / "identity.json"
    manifest.write_text(json.dumps({"source": str(source), "sha256": {
        f"src/step_preference/{name}": hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in PURE
    }}))
    return source, folder, manifest


def _run(source, manifest, output):
    return subprocess.run([sys.executable, str(SCRIPT), "--source-root", str(source),
                           "--identity-manifest", str(manifest), "--output", str(output)],
                          capture_output=True, text=True, check=False)


def test_changed_source_hash_fails_before_comparison(tmp_path):
    """Catches loading a changed source file under an earlier trusted identity."""
    source, folder, manifest = _fixture(tmp_path)
    (folder / PURE[0]).write_text((folder / PURE[0]).read_text() + "\n# changed\n")
    output = tmp_path / "comparison.json"
    result = _run(source, manifest, output)
    assert result.returncode != 0
    report = json.loads(output.read_text())
    assert report["verdict"] == "identity_mismatch"
    assert report["cases"] == []


def test_changed_gradient_produces_failure_and_case_error(tmp_path):
    """Catches a comparator that checks scalar values but misses source gradients."""
    source, folder, manifest = _fixture(tmp_path)
    target = folder / "dil_bce_loss.py"
    code = target.read_text().replace("F.softplus(-beta * clean)",
                                      "F.softplus(-beta * clean.detach())")
    assert code != target.read_text()
    target.write_text(code)
    identity = json.loads(manifest.read_text())
    identity["sha256"]["src/step_preference/dil_bce_loss.py"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(identity))
    output = tmp_path / "comparison.json"
    result = _run(source, manifest, output)
    assert result.returncode != 0, result.stderr
    report = json.loads(output.read_text())
    assert report["verdict"] == "fail"
    assert any(case["input"]["kind"] == "bce" and case["errors"]["clean_gradient"] > 0
               for case in report["cases"])

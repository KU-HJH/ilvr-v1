"""Live full-vocabulary gap selection: CUDA objectives and host entry contracts."""
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from stage2.trainer import _selected_entropies
from step_preference.dil_identity import attach_dil_identity, build_dil_identity, read_dil_identity, validate_dil_resume
from step_preference.dil_topk_loss import topk_token_terms
from step_preference.equivalence import _trainer
from tests.step_preference.test_dil_both_lower import LOWER, lower_cli
from tests.step_preference.test_dil_config import _bce_config
from utils_deepseed import parse_args


def gap_config(**updates):
    return _bce_config(preference_loss_type=LOWER, preference_topk=2,
        preference_entropy_top_percent=20., preference_entropy_source="entropy_gap",
        lambda_ce=.1, **updates)


def literal_loss(ce, clean, off, ids, selected, *, k):
    # Independent fixed-mask oracle: means precede softplus within each step.
    total = .1 * ce
    for c, o, target, positions in zip(clean, off, ids, selected):
        if not positions:
            continue
        lp = c.double().log_softmax(-1)
        op = o.double().log_softmax(-1)
        gt = lp[positions].gather(-1, target[positions, None]).mean()
        support = o[positions].detach().topk(k, -1, sorted=False).indices
        q = o[positions].double().gather(-1, support).softmax(-1)
        mc = lp[positions].detach().exp().gather(-1, support).sum(-1)
        mo = op[positions].exp().gather(-1, support).sum(-1)
        mass = (mc * (mc/mo).log() + (1-mc) * ((1-mc)/(1-mo)).log()).mean()
        total = total + (F.softplus(-gt) + F.softplus((q*q.log()).sum(-1).mean()) + mass)/len(clean)
    return total


@pytest.mark.parametrize("source,selected", [("entropy_gap", [[0, 1], []]),
    ("clean", [[0], [0]]), ("voff", [[1], [0]])])
@pytest.mark.parametrize("zero_gap", [False, True])
def test_cuda_trainer_source_ranks_absolute_full_vocab_gap_across_original_steps(source, selected, zero_gap, monkeypatch):
    # Catches clean/OFF/signed/top-k entropy substitution, step-local selection,
    # unstable ties, discarded-step renormalization and gradient detachment errors.
    import step_preference.entropy_selection as selection
    uniform, peaked = [.25]*4, [.97, .01, .01, .01]
    narrow, sharp = [.49995, .49995, .00005, .00005], [.99985, .00005, .00005, .00005]
    c = torch.tensor([uniform, peaked, narrow, sharp] + [uniform]*6, device="cuda").log().requires_grad_()
    o = torch.tensor([peaked, uniform, sharp, narrow] + [uniform]*6, device="cuda").log().requires_grad_()
    if zero_gap:
        o = c.detach().clone().requires_grad_()
        if source == "entropy_gap":
            selected = [[0, 1], []]  # all gaps zero: retain original token order
        else:
            selected = [[0], [0]]
    clean, off = [c[:4], c[4:]], [o[:4], o[4:]]
    ids = [torch.tensor([0, 1, 2, 3], device="cuda"), torch.tensor([0, 1, 2, 3, 0, 1], device="cuda")]
    clean_e, off_e = [_selected_entropies(x) for x in clean], [_selected_entropies(x) for x in off]
    terms = [topk_token_terms(a, b, i, beta=1., k=2) for a, b, i in zip(clean, off, ids)]
    clean_gt = [a.log_softmax(-1).gather(-1, i[:, None]).squeeze(-1) for a, i in zip(clean, ids)]
    off_gt = [a.log_softmax(-1).gather(-1, i[:, None]).squeeze(-1) for a, i in zip(off, ids)]
    spans = [SimpleNamespace(step_id=i) for i in range(2)]
    results = {(i, "visual"): dict(entropies=off_e[i], dil_tokens=terms[i], target_logps=off_gt[i]) for i in range(2)}
    observed = []
    original = selection.select_entropy_masks
    def observe(scores, percent):
        observed.append(tuple(scores))
        return original(scores, percent)
    monkeypatch.setattr(selection, "select_entropy_masks", observe)
    ce = torch.tensor(.8, device="cuda", requires_grad=True)
    config = gap_config() if source == "entropy_gap" else _bce_config(preference_loss_type=LOWER,
        preference_topk=2, preference_entropy_top_percent=20., preference_entropy_source=source, lambda_ce=.1)
    actual, metrics = _trainer(config)._reduce_context_objective(ce, [], clean_e, results,
        clean_token_logps=clean_gt, spans=spans)
    expected = literal_loss(ce, clean, off, ids, selected, k=2)
    torch.testing.assert_close(actual, expected.float(), atol=1e-6, rtol=1e-5)
    assert metrics["stats/selected_tokens"].item() == 2
    assert metrics["stats/selected_steps"].item() == sum(bool(p) for p in selected)
    assert metrics["stats/eligible_steps"].item() == 2
    got = torch.autograd.grad(actual, [ce, c, o], retain_graph=True)
    want = torch.autograd.grad(expected, [ce, c, o], retain_graph=True)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    if source == "entropy_gap":
        assert all(not s.requires_grad and s.dtype == torch.float32 and s.is_cuda for s in observed[0])
        for score, a, b in zip(observed[0], clean_e, off_e):
            torch.testing.assert_close(score, (b.detach()-a.detach()).abs(), atol=0, rtol=0)
        # Explicit contrary rankings must disagree with the hand-derived gap mask.
        if not zero_gap:
            signed = torch.cat(off_e)-torch.cat(clean_e)
            topk_gap = (_selected_entropies(o.topk(2, -1).values)-_selected_entropies(c.topk(2, -1).values)).abs()
            assert signed.argsort(descending=True, stable=True)[:2].tolist() != [0, 1]
            assert topk_gap.argsort(descending=True, stable=True)[:2].tolist() == [2, 3]
            assert torch.count_nonzero(got[1][4:]) == 0
            assert torch.count_nonzero(got[2][4:]) == 0


@pytest.mark.parametrize("backend", ["dense", "flex"])
def test_cuda_native_compute_loss_sidecar_metrics_gradients_and_live_recomputation(backend, monkeypatch):
    # Real tiny CUDA model exercises parent metrics and checkpointed sidecar;
    # two forwards with changed trainable weights catch frozen initializer scores.
    import step_preference.entropy_selection as selection
    from step_preference.shared_prefix_measurement import tiny_measurement_case
    from tests.step_preference.test_dil_trainer import compare_grads
    config = gap_config(sidecar_attention_backend=backend, target_scope="until_next_latent")
    with torch.device("cuda:0"):
        model, batch = tiny_measurement_case(steps=2, total_target_tokens=10, device="cuda:0")
    model.config.supervise_latent_boundaries = True
    model.train()
    spans = list(batch["reasoning_step_spans"])
    spans[0] = replace(spans[0], target_text_end=spans[0].target_text_start+3)
    spans[1] = replace(spans[1], prev_text_end=spans[0].target_text_end)
    batch["reasoning_step_spans"] = tuple(spans)
    batch["step_preference"] = {"steps": tuple({"span": s} for s in spans)}
    trainer, scores = _trainer(config), []
    original = selection.select_entropy_masks
    def observe(values, percent):
        assert all(not v.requires_grad for v in values)
        scores.append(torch.cat(values).clone())
        return original(values, percent)
    monkeypatch.setattr(selection, "select_entropy_masks", observe)
    for iteration in range(2):
        projections, modes = [], []
        head = model.lm_head.register_forward_hook(lambda _m, _a, output: projections.append(output))
        mode = model.register_forward_pre_hook(lambda m, _a: modes.append((m.training, torch.is_grad_enabled())))
        try:
            actual, output = trainer.compute_loss(model, batch, return_outputs=True)
        finally:
            head.remove(); mode.remove()
        assert modes == [(True, True)] and len(projections) == 2
        c, o = output.stage2_selected_logits[0].float(), projections[-1][0].float()
        cp, op = c.log_softmax(-1), o.log_softmax(-1)
        gap = (-(op.exp()*op).sum(-1) + (cp.exp()*cp).sum(-1)).abs().detach()
        torch.testing.assert_close(scores[-1], gap, atol=0, rtol=0)
        selected = sorted(range(len(c)), key=lambda i: (-float(gap[i]), i))[:math.ceil(len(c)*.2)]
        clean, off, ids, masks, cursor = [], [], [], [], 0
        for span in spans:
            size = span.target_text_end-span.target_text_start
            clean.append(c[cursor:cursor+size]); off.append(o[cursor:cursor+size])
            ids.append(batch["input_ids"][0,span.target_text_start:span.target_text_end])
            masks.append([i-cursor for i in selected if cursor <= i < cursor+size])
            cursor += size
        expected = literal_loss(output.loss, clean, off, ids, masks, k=2)
        torch.testing.assert_close(actual, expected.float(), atol=1e-6, rtol=1e-5)
        grads = compare_grads(actual, expected, model)
        assert any(g is not None and g.is_cuda and g.abs().sum() > 0 for g in grads)
        assert trainer._step_preference_metric_batches == iteration+1
        assert trainer._step_preference_metric_totals["stats/selected_tokens"] == 2*(iteration+1)
        assert trainer._step_preference_metric_totals["stats/eligible_steps"] == 2*(iteration+1)
        if iteration == 0:
            with torch.no_grad():
                model.lm_head.weight.mul_(1.3)
    assert not torch.equal(scores[0], scores[1])


@pytest.mark.parametrize("saved_source", ["clean", "voff", "entropy_gap"])
def test_host_cli_attach_serialization_and_resume_preserve_source_identity(tmp_path, saved_source):
    # Dropped CLI/identity admission or source comparison could resume a changed objective.
    fresh = parse_args(lower_cli()+["--step_preference_entropy_source", saved_source]).step_preference_config
    model = SimpleNamespace(config=SimpleNamespace(), model=SimpleNamespace(), lm_head=SimpleNamespace())
    attach_dil_identity(model, fresh)
    (tmp_path/"config.json").write_text(json.dumps(vars(model.config)))
    assert read_dil_identity(json.loads((tmp_path/"config.json").read_text()))["entropy_source"] == saved_source
    same = parse_args(lower_cli(tmp_path)+["--step_preference_entropy_source", saved_source]).step_preference_config
    validate_dil_resume(tmp_path, same)
    for requested_source in {"clean", "voff", "entropy_gap"}-{saved_source}:
        with pytest.raises(ValueError, match="identity"):
            validate_dil_resume(tmp_path, replace(same, preference_entropy_source=requested_source))


@pytest.mark.parametrize("updates", [dict(preference_entropy_top_percent=None),
    dict(preference_loss_type="dpo"), dict(preference_loss_type="dil_bce_rf", preference_topk=None, lambda_topk_mass=0.),
    dict(shared_prefix_enabled=False), dict(block_original_image=True)])
def test_host_gap_rejects_unsupported_combinations(updates):
    values = dict(preference_loss_type=LOWER, preference_topk=2, preference_entropy_top_percent=20.,
        preference_entropy_source="entropy_gap", lambda_ce=.1)
    with pytest.raises(ValueError):
        _bce_config(**(values|updates))


@pytest.mark.parametrize("resume", [False, True])
def test_host_gap_survives_both_real_worker_shells(tmp_path, resume):
    from tests.multinode.test_dil_forwarding import worker_commands
    settings = dict(STEP_PREFERENCE_LOSS_TYPE=LOWER, STEP_PREFERENCE_TOPK="1516",
        STEP_PREFERENCE_LAMBDA_TOPK_MASS="1", STEP_PREFERENCE_ENTROPY_TOP_PERCENT="20",
        STEP_PREFERENCE_ENTROPY_SOURCE="entropy_gap", STEP_PREFERENCE_BETA="", CE_WEIGHT="",
        STEP_PREFERENCE_TARGET_SCOPE="until_next_latent", STAGE2_SAMPLER_POLICY="historical_balanced_2rank")
    if resume:
        saved = parse_args(lower_cli(tmp_path)+["--step_preference_entropy_source", "entropy_gap"]).step_preference_config
        (tmp_path/"config.json").write_text(json.dumps({"ilvr_context_preference": build_dil_identity(saved)}))
        settings.update(STAGE2B_DIRECT_FROM_STAGE1="false", STAGE1_ADAPTER_PATH="", RESUME_CHECKPOINT=str(tmp_path))
    for rank, _, result in worker_commands(settings, ("--max_steps", "212")):
        assert result.returncode == 0, result.stderr
        argv = json.loads(result.stdout)
        assert argv[argv.index("--step_preference_entropy_source")+1] == "entropy_gap"
        assert argv[argv.index("--machine_rank")+1] == str(rank)
        entry = next(i for i, a in enumerate(argv) if a.endswith("/main.py"))
        parsed = parse_args(argv[entry+1:]).step_preference_config
        assert parsed.preference_entropy_source == "entropy_gap"
        assert parsed.is_dil_resume == resume

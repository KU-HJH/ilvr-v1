# ILVR causal mask control implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** approved for execution on 2026-09-11 by user `진행해`, bound to commitment v3. Subagent-Driven execution is selected. E3/E4 launch still requires the specified runtime gates; existing E0/E1/E2/E5 work remains authorized.

**Goal:** Separate E3/E4 image-edge removal from the observed 2D-to-4D SDPA numerical-path confound.

**Architecture:** Add one mode to the existing generation-local policy and CLI. Reuse current mask construction, controller, scoring and paired comparison functions; add a neutral4D control panel and explicit representation bridge.

**Tech Stack:** repository Python3.11 environment, PyTorch2.7.1, vendored Qwen2.5-VL, SDPA decoder / FA2 vision, bf16, pytest.

**Spec:** [commitment v3](../specs/2026-09-10-ilvr-visual-causal-mask-control-design.md), supplementing [v1](../specs/2026-09-10-ilvr-visual-causal-analysis-design.md). Read both; v3 takes precedence for E3/E4 controls and amended gates.

## Global constraints

- Work in `/root/workspace/ILVR/.worktrees/ours1-visual-causal`. Base implementation `a78ea0576348885326c2708185081a2c6a2395b3`; reuse the isolated worktree. Do not rewrite the source of running jobs or merge/push/remove worktrees.
- Python exclusively `./venv/ilvr-venv/bin/python -B`, with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:.`. No package installation.
- Model-bearing artifacts remain beneath `checkpoints/`; data under `data/`, generated reports under `results/`, execution scripts/logs under `logs/`. Preserve existing output, failed probes and hold logic.
- Use `--attention-backend sdpa`, temperature0, top_p1, repetition_penalty1.05, max_new_tokens4096, max_sequence_length32768, batch1, bf16, latent_size8 for full panels. Short witnesses use max_new_tokens64 only.
- Source `data/TEST.jsonl`, SHA256 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`; main indices `data/evaluation/visual-causal-20260910/main-indices.json` (100). No supplementary E5 rows in the new control panel. Keep the 100 coverage / 99-valid-gold distinction.
- No training, controller, vendored-model, metric-formula, existing-evaluator or global kernel-selector changes. No generalized retry/orchestration layer.
- New full run ID `visual-causal-mask-control-20260910-r1`; verify its result directories do not already exist before launch. If a real run fails, preserve its evidence and use a fresh explicit run ID after diagnosis.
- E1/E2/E5 keep their existing controls. E3/E4 compare against `causal_mask_control`; report existing SDPA E0 versus neutral4D as a separate representation bridge. Do not require bridge equality.

Checkpoint map for all commands, with base `/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`:

```python
CHECKPOINTS = {
    "s1": "checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga4-e1-lr5e5-egb16-20260902-024225/prepared-checkpoint-215",
    "ce": "checkpoints/evaluation-staging/ours1-stage2a-s1matched-egb16-lr5e5-dora-z2-20260908t022000z-r3/prepared-checkpoint-212",
    "pref": "checkpoints/evaluation-staging/ours1-ctx-pref-vonly-direct-s1-dora-egb16-lr5e5-default-s212-r2-20260910t022900z/prepared-checkpoint-212",
}
```

### Task 1: Neutral mode, focused regression tests and runbook

**Files:** modify `src/evaluation/visual_interventions.py`, `scripts/analyze_comt_visual.py`, `tests/evaluation/test_visual_interventions.py`, `tests/evaluation/test_visual_analysis_runner.py`, and `docs/evaluation/visual-causal-analysis.md`.

**Interfaces:** existing `VisualIntervention(mode, image_token_id, sliding_window=None)` accepts `causal_mask_control`; existing `build_image_attention_mask(...)` accepts that mode. The CLI `--visual-ablation-mode causal_mask_control --attention-backend sdpa` emits the existing schema with its distinct condition name. No new function signature or output schema is necessary.

- [ ] Write the independent edge regression using existing test helpers. This test must fail before the new mode exists:

```python
def test_causal_mask_control_retains_image_edges_and_logits():
    decoder = _CaptureDecoder()
    policy = VisualIntervention("causal_mask_control", image_token_id=80)
    policy.begin(torch.tensor([[7, 80, 8]]), TOKENS)
    policy.register_latent_span(
        SimpleNamespace(start_position=2, latent_positions=(3,), end_position=4)
    )
    kw = _decoder_kwargs(5)
    policy.wrap_decoder(decoder)(**kw)
    expected = [[k <= q for k in range(5)] for q in range(5)]
    assert _allowed(decoder.calls[0]["attention_mask"]) == expected
    assert decoder.calls[0]["position_ids"] is kw["position_ids"]
    logits = torch.zeros(1, 128)
    assert policy.process_logits(
        torch.tensor([[7, 80, 8]]), logits, logits, TOKENS
    ) is logits
```

- [ ] Extend the independently enumerated padding/window test to cover the new mode. Extend the real controller routing witness to assert neutral causal visibility at ordinary, START/PAD, post-END and following-text calls; retain its no-cache and exact-position checks. Keep the old baseline identity and E3/E4 expected-edge assertions.
- [ ] Add runner regressions using `_run`, `_successful_generator` and fake models already in its test file. Capture `context['intervention'].mode`, prepared image tensors and `allow_latent_generation`; require `causal_mask_control`, original pixels/grid, True, and serialized condition equality. Test argparse acceptance and `_require_mask_compatible` rejection for `backend='native'` or a concrete Flash decoder class. Do not infer actual controller execution from the injected runner fake.

```python
def test_neutral_control_runner_keeps_images_and_fresh_policies(tmp_path):
    seen = []
    def generate(**context):
        policy = context["intervention"]
        assert policy.mode == "causal_mask_control"
        assert {"pixel_values", "image_grid_thw"} <= set(context["inputs"])
        assert context["allow_latent_generation"] is True
        seen.append(policy)
        return _successful_generator(**context)
    rows = [
        {"text_input": "question A", "image_input": ["images_comt/deletion/a.png"], "original_final_answer": "10"},
        {"text_input": "question B", "image_input": ["images_comt/deletion/b.png"], "original_final_answer": "10"},
    ]
    result, _, _ = _run(
        tmp_path, rows, visual_ablation_mode="causal_mask_control",
        generation_runner=generate,
    )
    assert result["metadata"]["status"] == "completed"
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert {row["condition"] for row in result["rows"]} == {"causal_mask_control"}

@pytest.mark.parametrize("backend,attention", [
    ("native", "Qwen2_5_VLSdpaAttention"),
    ("sdpa", "Qwen2_5_VLFlashAttention2"),
])
def test_neutral_control_rejects_incompatible_backend(backend, attention):
    with pytest.raises(ValueError, match="requires"):
        runner._require_mask_compatible(
            {"decoder_attention_classes": [attention]}, "causal_mask_control", backend
        )

def test_cli_accepts_neutral_control():
    args = runner.build_parser().parse_args([
        "run", "--model-dir", "checkpoints/fake", "--test-data-path", "data/TEST.jsonl",
        "--source-data-path", "data/TEST.jsonl", "--output-dir", "results/fake",
        "--visual-ablation-mode", "causal_mask_control", "--attention-backend", "sdpa",
    ])
    assert args.visual_ablation_mode == "causal_mask_control"
```
- [ ] Run RED and retain the actual failure output:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B -m pytest -q tests/evaluation/test_visual_interventions.py tests/evaluation/test_visual_analysis_runner.py > logs/visual-causal-implementation-20260910/mask-control-red.log 2>&1
```

- [ ] Implement only the new alias/effect and wrapper membership, then register it in CLI/compatibility choices:

```python
# Add to _MODE_ALIASES in visual_interventions.py:
"causal_mask_control": "causal_mask",
# Include causal_mask in wrap_decoder's existing 4D composition condition:
if self._effect in {"causal_mask", "latent_mask", "global_mask"}:
    kwargs["attention_mask"] = build_image_attention_mask(
        attention_mask,
        self._image_positions,
        tuple(self._latent_query_positions),
        self.mode,
        inputs_embeds.dtype,
        self.sliding_window,
    )
# Add "causal_mask_control" to runner VISUAL_ABLATION_MODES and MASKED_MODES.
```

The builder's latent/global branches remain the only branches that remove image edges; the newly resolved effect therefore returns base causal/padding/window visibility.

- [ ] Update the runbook with the v3 control table and neutral/bridge/E3/E4 commands. Explicitly state that baseline2D-versus-neutral equality is not a gate and that incoming 2D shape logs are not actual delivered-mask evidence.
- [ ] Run GREEN once for the affected tests and controller tests, then CLI help; read complete outputs:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B -m pytest -q tests/evaluation/test_visual_interventions.py tests/evaluation/test_visual_analysis_runner.py tests/evaluation/test_continuous_latent.py > logs/visual-causal-implementation-20260910/mask-control-green.log 2>&1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transformers/src:src:. ./venv/ilvr-venv/bin/python -B scripts/analyze_comt_visual.py run --help
```

- [ ] Commit the five scoped files and request task-scoped review. Resolve Important/Critical findings before Task2. Record the resulting immutable commit in probe and campaign metadata; do not repeat the broad repository suite at this task boundary.

### Task 2: Runtime first nail, amended detached panels and report

**Owner:** root coordinates actual resources and runtime evidence. **Files:** short witness `logs/visual-causal-mask-control-witness-20260910-r1/probe.py`; explicit queue/validation scripts in `logs/visual-causal-mask-control-20260910-r1/`; outputs under corresponding `results/` directories; append observed commands/evidence to runbook.

**Consumes:** Task1 CLI, v1 checkpoint/data artifacts and existing paired comparison functions. **Produces:** per-model witness JSON, neutral100/E3-100/E4-100 JSONLs, representation-bridge/E3/E4 summaries, final cross-model analysis. Neither a witness output nor an aggregate table may be labeled complete before its assertions pass.

- [ ] Authenticate the selected local/remote checkpoint paths beneath canonical checkpoints storage and the exact data hash. Check GPU/process availability and wait for the old E2 queues to stop at their hold before reusing their GPUs. The currently free remote GPU1 may run the bounded witness; do not interrupt unrelated processes. Sync the reviewed commit to the existing remote analysis checkout only when its active Python jobs have finished, preserving outputs and symlinks.
- [ ] Adapt the existing `real_policy_probe.py`/`real_cli_smoke.py` diagnostic pattern into the specified probe file after approval. Reuse the current loader and processor; loop all three CHECKPOINTS and modes `baseline`, `causal_mask_control` twice, E3, E4. Build fresh prepared inputs, policy and controller state each time. Use original index3/max64. If neutral/E3/E4 lacks a completed block or post-END text, rerun all five generations once at the first other ascending main index with a completed existing SDPA E0 block; if still uncovered, stop that checkpoint's amended panels. Do not search indefinitely or force START. Save raw tokens, prompt identities, scores and actual delivered-mask/position/cache observations.
- [ ] Independently construct each expected allowed-edge matrix from query/key indices in the decoder pre-hook, without calling the production builder. Start with `key <= query` and valid-key positions, apply the configured sliding window, then only E3 START/PAD→image or E4 non-image→image exclusions. Require exact zero/finite-min additive values, explicit positions exactly equal to the clone captured before that same policy call converts the mask, use_cacheFalse and absent past K/V. Never use another condition's generated trajectory as the position reference. For neutral, require every base-visible image edge remains present. Record ordinary/latent/post-END roles from the actual invocations.
- [ ] Profile the short CUDA generations with CPU operator tracing, without global monkeypatching or output_attentions fallback. The concrete dispatch assertion is:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU]
) as profile:
    result = controller.generate(**generation_arguments)
counts = {event.key: event.count for event in profile.key_averages()}
concrete = {
    key: count for key, count in counts.items()
    if key.startswith("aten::_scaled_dot_product_")
}
assert concrete == {
    "aten::_scaled_dot_product_efficient_attention": decoder_call_count * decoder_layer_count
}
```

Here `controller` is a fresh `ContinuousLatentController`, `generation_arguments` are the current prepared tensors/token IDs/config/policy from the preceding step, `decoder_call_count` is the pre-hook count and `decoder_layer_count` is `len(qwen.model.layers)`. Apply the assertion to neutral/E3/E4 only, not baseline2D. Persist exact commands, classes/dtypes, counts, source commit and GPU identity.
- [ ] Verify two independent neutral continuations have identical prompt identity/raw tokens/scoring. For the inactive-E3 check, replay captured first-call pre-policy kwargs through fresh neutral/E3 wrappers with empty span lists into the same eval decoder. Compare masks and last-position logits exactly; discard outputs and exclude these diagnostic calls from generation-profile counts. Do not demand equality once E3 actually removes edges. Run one changed-image sanity check only if the existing numerical witnesses fail or a new contradiction calls them into question.
- [ ] Inspect all three witness outputs. Any mask/position/cache/dispatch/equality failure blocks full panels and returns to systematic debugging; preserve logs and report the root cause. No automatic kernel fallback or speculative fix. This is the not-yet-run first nail, not a pre-approved success claim.
- [ ] After passing witnesses and review, persist three explicit queue scripts, one model per available GPU. Use `set -euo pipefail`, detached tmux, and fresh result directories. For each model execute neutral100, E3-100 and E4-100 with the reviewed CLI, then validate each output once. Use the existing queue pattern; leave the old E3/E4 hold intact.
- [ ] Commands for each model use the following exact argument construction; CHECKPOINTS is the map above, `label` is its current key, and `mode` loops the three listed strings. Materialize the shell commands into the queue logs before launch, then inspect them:

```python
import json
import shlex
from pathlib import Path
indices = json.loads(Path("data/evaluation/visual-causal-20260910/main-indices.json").read_text())
assert len(indices) == len(set(indices)) == 100
run_id = "visual-causal-mask-control-20260910-r1"
for label, checkpoint in CHECKPOINTS.items():
    for mode in ("causal_mask_control", "latent_image_attention_block", "global_image_attention_block"):
        command = [
            "./venv/ilvr-venv/bin/python", "-B", "scripts/analyze_comt_visual.py", "run",
            "--model-dir", checkpoint,
            "--base-model-dir", "/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct",
            "--test-data-path", "data/TEST.jsonl", "--source-data-path", "data/TEST.jsonl",
            "--indices", ",".join(map(str, indices)),
            "--output-dir", f"results/{run_id}/{label}/{mode}",
            "--visual-ablation-mode", mode, "--attention-backend", "sdpa",
            "--max-new-tokens", "4096", "--max-sequence-length", "32768",
            "--repetition-penalty", "1.05", "--temperature", "0", "--top-p", "1",
        ]
        print(shlex.join(command))
```

- [ ] Reuse the current `check_and_summarize.py` validation logic in the new run's direct script, retaining count100, zero-error, unique-index, raw-position and source-rescoring checks. Replace its comparison mapping explicitly: bridge baseline is `results/visual-causal-panel-20260910t115000z/{label}/e0-main.jsonl`; E3/E4 baseline is `results/visual-causal-mask-control-20260910-r1/{label}/causal_mask_control/predictions.jsonl`. Resolve these paths locally per node; copy remote outputs into the same current-worktree results tree for root aggregation. Do not edit the old held queue script or reuse its sdpa/union114 branch for the new100 control.
- [ ] Write paired CSV/JSON/Markdown reports using `compare_prediction_files`; retain its baseline/condition paths and hashes. State that full-panel concrete CUDA operator dispatch is not profiled; only the bounded witnesses directly authenticate it. The new control row's block/timing/accuracy distribution is itself a measured result. Assemble the original E1/E2/E5 results with the new control and E3/E4 results, explicitly naming every paired baseline. Interpret Cases A–E only within the verified pathways and uncertainty scope in the spec. Do not pool the two control families.
- [ ] Request final whole-change review after the runtime task and reports are ready. Follow the existing SDD ledger and project full-suite/verification/branch-finish gates once at actual completion; no merge/push/cleanup without user decision. Record the full `git diff --stat ff53706..HEAD`, per-file explanation, exact commands, remaining limitations and execution status. Use partial status if a required panel or gate remains pending.

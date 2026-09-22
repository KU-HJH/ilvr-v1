# CE Vision Block Full-Length v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. The user already selected sub-agent execution; retain it. Steps use checkbox syntax for tracking.

**Goal:** Replace the failed short-suffix execution with the approved full-length method, then obtain the four-condition, five-metric analysis of the fixed 100 CE-only samples.

**Architecture:** Reuse the reviewed input, metric and CLI contracts. Recompute one complete decoder sequence from frozen clean embeddings for each independent site/condition, enforcing unchanged prefix K/V and one numerical policy. This amendment replaces the runtime strategy in original Tasks 3–4; completed original Tasks 1–2 and unrelated reviewed behavior are not reimplemented.

**Tech Stack:** repository Python 3.11 venv, patched Transformers, PyTorch 2.7.1, PEFT, SDPA MATH language attention, FlashAttention2 vision; BF16 base plus saved FP32 adapters and FP32 log-softmax.

**Spec:** [design commitment v2](../specs/2026-09-22-ce-vision-block-full-length-amendment.md), approved by the user's exact reply `v2 설계 승인 (권장) — 전체 길이 계산으로 변경`. Approval binds the document at commit `75d6ceba51d6e1becd664f4155d0edf9e873cff4`, SHA256 `9739f29b5e6f2f4cbe53c27bc83434397cc150d6b892bcb53c7d46f1e3b1ad1c`. Its pending banners describe publication before that reply, not the current approval state. This implementation amendment awaits the writing-plans handoff review.

## Global Constraints

- Execute in `/root/workspace/ILVR/.worktrees/ce-block-analysis-20260921`; starting HEAD is `75d6ceba51d6e1becd664f4155d0edf9e873cff4`, with product code at `31c91b30f21802f984f71640fb6411873f8cca16`. Preserve unrelated/untracked files and the old failed run.
- Model: prepared CE-only `stage2-ce-matched-gb16-s212-20260919-155759/checkpoint-212`; original Qwen2.5-VL-3B base; preference off; no training or decoder changes.
- Keep fixed main100 selection, original image access, clean Body, independent sites, `concatenated-v1`, `immediate_text`, original three-axis positions, target IDs and logical sequence length. Existing measured cohort: 100 samples, 327 sites, 12,843 target tokens, zero exclusions. Revalidate rather than force those counts.
- Conditions in order: `clean`, `body_mask`, `boundary_mask`, `full_mask`. For queries at/after the current END, block direct attention to Body for body/full and START for boundary/full. Zero only current END input for boundary/full. Queries before END retain ordinary causal visibility. Later unscored rows remain causal.
- Numerical policy: SDPA MATH throughout ordinary rollout, capture, condition passes and repeats; both effective TF32 flags false. Preserve actual parameter dtypes and vision FlashAttention2; no claim of equality to historical auto-kernel clean scores.
- Require bit-identical layerwise K/V strictly before current END against clean for every site/condition; mismatch fails the run. Clean gates remain argmax mismatches=0, max absolute NLL difference <=0.01 nat and mean <=0.001 nat, with repeated clean in smoke. No tolerance relaxation or dropped failed sample.
- Metrics remain clean-relative first_changed/S, any_changed/S, tokens_changed/N, token-weighted mean NLL and signed delta NLL. NLL uses annotated next-text targets; this is teacher forcing, not generated reasoning or an accuracy/semantic-causality claim.
- Python command prefix from this worktree: `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python`. No dependencies installed. Data/checkpoints symlinks retained; generated data under `data/`, results under `results/`, all logs under `logs/`.
- Root owns control artifacts and execution; fresh implementers own product changes. Independent spec-and-quality task reviews, scoped fixes, whole-branch review and fresh verification precede actual smoke/main100. Agents do not invoke paperthin hooks.

## Review Focus

1. Nested numerical contexts or exceptions restore stale TF32/kernel settings incorrectly: test effective settings inside ordinary and condition calls, and restoration on exception (Task 1).
2. Full recomputation silently changes clean prefix/Body: compare every layer K/V and inject a single prefix corruption to prove rejection (Task 1).
3. Old captured-KV perturbation tests become disconnected from execution: replace with actual decoder edge observations and a live visible-Body positive control (Task 1).
4. CLI accepts an old suffix smoke or incomplete new witness: require explicit strategy/policy identity and exact site-condition-layer K/V witness coverage (Task 2).
5. Full-length future rows, distinct mRoPE axes or later sites affect scored earlier text: verify original positions, causal edges, selected predictor indexing and independent-site order (Task 1), then use fixed complete coverage in Task 2.

## Files and interfaces

Original plan: `docs/superpowers/plans/2026-09-21-ce-vision-block-local-analysis.md`. Its Tasks 1/2 are metrics/inputs; Tasks 3/4 are runtime/CLI. This amendment's Task numbering is local, not a request to reopen the old Tasks 1/2. Controller ledger: `.superpowers/sdd/2026-09-21-ce-vision-block-local-analysis/progress.md`; adjacent task/review reports and approval receipts are backed up in `results/ce-block-development-20260921/workflow-evidence/`. Record v2 task reports there with `v2-` names so old evidence is preserved. Implementation is blocked until the controller records the user's plan-handoff confirmation.

Preserved failed manifest: `data/ce-vision-block-analysis/ce-block-main100-20260922-012440/manifest.json`. Failed smoke and diagnostic reports: `results/ce-block-main100-20260922-012440/smoke/` and `results/ce-block-main100-20260922-012440/diagnosis/diagnosis.md`; logs: `logs/ce-block-main100-20260922-012440/`. Prior focused evidence at product revision31c91b3: 137 passed, 2 skipped, `logs/ce-block-development-20260921/final-focused.log`. No full-suite baseline has been established; any failure must be diagnosed and reported rather than presumed unrelated or waived.

Task 1 changes `src/evaluation/vision_block_runtime.py`, `tests/evaluation/test_vision_block_runtime.py` and, only if necessary, test oracle `tests/evaluation/vision_block_reference.py`. Task 2 changes `src/evaluation/vision_block_analysis.py` and `tests/evaluation/test_vision_block_analysis.py`. Existing input/metric modules, historical entrypoints and patched decoder remain untouched.

Retain `CleanState`, `capture_clean(model, sample)`, `build_replay_inputs(state, site, condition)`, `replay_site(model, state, site, condition)` and `analyze_sample(model, sample, repeat_clean=False)` contracts. Add optional keyword `prefix_witness=None` to `replay_site`; when a list is supplied append one successful record per pass. Runtime always validates prefix identity, even when no list is supplied. Add reusable context manager `numerical_policy()` and JSON-serializable helper `numerical_policy_identity()` reporting actual active flags and enabled SDPA backends. Context restores prior flags/backends in `finally` and supports nesting.

Each prefix record contains `plan_index`, `condition`, `prefix_length`, `layers`, `kv_comparisons` (=2*layers), and `exact=True`; do not store tensor copies in JSON. `analyze_sample` returns existing rows and witnesses plus `execution_strategy='full_length_v2'`, `numerical_policy`, `prefix_identity` (four conditions for every site), and `repeat_prefix_identity` (one clean record per site when repeated). Existing fidelity comparison names stay compatible. CLI validator receives expected sites and actual model layer count, so missing/duplicate records or fabricated counts cannot qualify as complete coverage.

Derive layer count from `len(resolve_stage2_qwen_model(model).model.layers)`, cross-check with the captured tuple count and config count. Capture stores detached full-sequence K/V shaped `[batch, kv_heads, sequence, head_dim]`; explicitly validate ranks/shapes rather than silently assuming this layout. `normal_scores` come from the unmodified ordinary final Stage 2 decoder call, `captured_scores` from a separate full decoder capture on its frozen embeddings/positions. Both map `(plan_index, target_position)` to `(argmax_id, nll)`. Existing comparison keys are `normal_vs_captured`, `normal_vs_replay`, `captured_vs_replay`, `repeat_clean`, `repeat_normal`, `repeat_captured`. Preserve their meanings under the new execution strategy.

### Task 1: Full-length runtime and causal evidence

**Consumes:** existing `PreparedSample`/`Site` and loaded CE model. **Produces:** runtime APIs and witness fields above; independently testable without CLI.

- [ ] Add the full-length regression below to the existing runtime tests (using existing `runtime` fixture); run the file with `-q` and preserve the intended failure showing the old short shape. The four parameterizations pin changed query scope and original axes.

```python
@pytest.mark.parametrize('condition', ['clean', 'body_mask', 'boundary_mask', 'full_mask'])
def test_full_length_v2_geometry(runtime, condition):
    p = torch.arange(14)[None, None, :]
    state = runtime.CleanState(torch.ones(1, 14, 8), torch.cat((p, 2*p, 3*p)), (), {}, {})
    site = Site(0, 1, 2, 3, 11, 11, (12, 13), (7, 8))
    embeds, positions, mask = runtime.build_replay_inputs(state, site, condition)
    assert embeds.shape == state.embeds.shape
    assert torch.equal(positions, state.positions)
    blocked = set(range(3, 11)) if condition in ('body_mask', 'full_mask') else set()
    if condition in ('boundary_mask', 'full_mask'):
        blocked.add(2)
    expected = torch.tensor([[k <= q and (q < 11 or k not in blocked)
                              for k in range(14)] for q in range(14)])
    assert torch.equal(torch.isfinite(mask[0, 0]), expected)
    expected_embeds = state.embeds.clone()
    if condition in ('boundary_mask', 'full_mask'):
        expected_embeds[:, 11] = 0
    assert torch.equal(embeds, expected_embeds)
    assert torch.equal(state.embeds, torch.ones_like(state.embeds))
```

- [ ] Implement the full-length branch using the following core; retain existing geometry/type/finite validation and `target-1` scoring. Do not feed `stage2_prefix_key_values` to the new branch. Add the policy context around direct public capture/replay/analyze calls, including ordinary latent rollout and all scoring; use `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` and save/restore both TF32 flags.

```python
embeds = state.embeds.clone()
positions = state.positions
length = embeds.shape[1]
visible = torch.ones(length, length, dtype=torch.bool, device=embeds.device).tril()
if condition in ('body_mask', 'full_mask'):
    visible[site.end:, site.body_start:site.body_end] = False
if condition in ('boundary_mask', 'full_mask'):
    visible[site.end:, site.start] = False
    embeds[:, site.end] = 0
mask = torch.zeros((1, 1, length, length), dtype=embeds.dtype, device=embeds.device)
mask.masked_fill_(~visible[None, None], float('-inf'))
# qwen is resolved with the existing PEFT helper.
out = qwen.model(inputs_embeds=embeds, position_ids=positions,
                 attention_mask=mask, stage2_capture_kv=True,
                 generate_mode=True, use_cache=False, return_dict=True)
assert len(out.stage2_key_values) == len(state.prefix_kv)
for actual_pair, clean_pair in zip(out.stage2_key_values, state.prefix_kv):
    for actual, clean in zip(actual_pair, clean_pair):
        if not torch.equal(actual[..., :site.end, :], clean[..., :site.end, :]):
            raise ValueError('full-length prefix K/V changed')
# Production uses explicit ValueError for shape/pair-count mismatches as well.
scores = _scores(qwen, out.last_hidden_state,
                 [p - 1 for p in site.target_positions], site.target_ids)
```

- [ ] Replace suffix-shape expectations and disconnected cache perturbations. Preserve the independent layer-by-layer cached-prefix FP32 oracle: compare its suffix logits to the matching full-path predictor rows with `atol=1e-5, rtol=1e-5`. Its cached-prefix strategy is intentionally different; never claim that its tests prove BF16 equality. For one/two-site image fixtures, capture actual decoder/attention inputs to verify full L, original axes, causal prefix and blocked direct edges. Mutate a live visible Body value at an actual layer boundary and observe a downstream score change in boundary mode; intervene after prefix validation or inspect direct decoder logits so the positive control is not swallowed by the prefix-fidelity guard. Do not perturb unused `state.prefix_kv` and label it a causal influence test.
- [ ] Add a distinct prefix-guard causal regression: alter one stored clean prefix K/V element and assert `replay_site` raises rather than returning scores. This mutation tests reference checking only. Run every condition with two sites in forward/reverse order, compare rows keyed by site/condition, and verify shared clean tensors unchanged. Exercise policy nesting and exception restoration with decoder hooks reading actual enabled SDPA flags. Preserve active-DoRA, hook-cleanup, chunk/tie and fidelity-threshold tests.
- [ ] Run `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_vision_block_runtime.py -q`; read full output. Commit only assigned Task 1 files and record RED/GREEN logs, call-path evidence and any limitations. Independent reviewer checks implementation against this plan/spec before Task 2 begins.

### Task 2: CLI policy, smoke coverage and identity

**Consumes:** Task 1 APIs/witness fields. **Produces:** unchanged prepare/smoke/run/summarize commands, with enforced v2 execution identity and complete runtime witnesses.

- [ ] Update CLI test fixtures to include exact prefix records, expected sites/layers and effective policy. Add rejection tests before implementing the validator: missing one prefix record, duplicate site/condition, wrong comparison count, `exact=False`, missing repeat-clean record, old suffix strategy and any enabled non-MATH backend/TF32 flag. Existing fixture supplies two samples and three sites; construct expected records directly from those sites.

```python
# New runtime policy identity contract, read while context is active:
policy = dict(sdpa_math=True, sdpa_flash=False, sdpa_mem_efficient=False,
              sdpa_cudnn=False, tf32_matmul=False, tf32_cudnn=False)
records = [dict(plan_index=s.plan_index, condition=c, prefix_length=s.end,
                layers=2, kv_comparisons=4, exact=True)
           for c in ('clean', 'body_mask', 'boundary_mask', 'full_mask')
           for s in sample.sites]
# Validator signature includes the exact expected sites and loaded layer count:
cli._validate_witness(witness, sites=sample.sites, layers=2, repeat=True)
# Each listed mutation must raise ValueError, including removal of records[-1].
```

- [ ] Wrap the real inference portion of CLI execution in a locally imported `numerical_policy()` context, starting before `_load_runtime` and identity capture and lasting through sample analysis. Keep stdlib-only `summarize` lazy. Record actual policy, full-length strategy and decoder layer count in runtime identity; these participate in smoke/run identity comparison. Update `_analyze` literal smoke assertions to all L positions, LxL visibility and END at original index. Update all validator call sites (smoke, stored gate and main100) with the actual expected sites and layer count. Compare exact record keys/counts; do not accept a bare successful boolean.

```python
expected = {(s.plan_index, c): s.end for c in CONDITIONS for s in sites}
actual = {(r['plan_index'], r['condition']): r for r in witness['prefix_identity']}
if len(actual) != len(witness['prefix_identity']) or actual.keys() != expected.keys():
    raise ValueError('incomplete prefix identity coverage')
for key, end in expected.items():
    r = actual[key]
    if (r['prefix_length'] != end or r['layers'] != layers
            or r['kv_comparisons'] != 2 * layers or r['exact'] is not True):
        raise ValueError('failed prefix identity witness')
```

- [ ] Verify actual `_analyze(repeat_clean=True)` with the existing tiny image-aware single/multi fixtures, without mocking runtime, and assert new full-length witnesses. Retain failure-state, stale identity, exact coverage, summary weighting and offline-no-torch tests. Run `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_vision_block_analysis.py tests/evaluation/test_vision_block_runtime.py -q`. Commit assigned files and preserve evidence; obtain independent spec/quality review, then required whole-branch review against original base `f8f80be29182a5d96764f1717a50382bf3b2704c`, distinguishing reused reviews from new v2 claims.

## Execution and reporting after review

- [ ] Root runs one focused integration set covering `tests/evaluation/test_vision_block_{metrics,inputs,runtime,analysis}.py`. Read all output; no repeated broad loops without a new failure/change. Recheck GPU availability and actual model/input identities before launch. No production model job starts before required code review.
- [ ] Choose a fresh UTC run ID `ce-block-main100-v2-YYYYMMDD-HHMMSS`. Reuse CLI `prepare` with the exact paths below, saving `data/ce-vision-block-analysis/<run-id>/manifest.json`; validate 100/327/12843 and original selection/hash/targets against the preserved failed manifest. Any unexplained cohort difference blocks launch, not permission to force counts.

```text
model-dir: /root/workspace/ILVR/.worktrees/voff-reproduction-20260917/checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212
base-model-dir: /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct
data: ./data/TEST.jsonl
selection: /root/workspace/ILVR/results/server-comparison-20260912/analysis_completed/visual-causal-panel-20260910t115000z/ce/no_latent_execution/metadata.json
image-root: ./data
TEST SHA256: 6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618
selection SHA256: fb1f15fef5339abde6ba74d140e5f9a1f8823cedafe8aef1d4bb245a8e0286aa
```

Concrete CLI entrypoint is `scripts/analyze_vision_block.py`. After checking that the chosen run paths are unused and GPU0 is free, the existing execution path is below (choose a different single free GPU if necessary). These are execution commands after all review gates, not commands to run while planning:

```bash
export PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD"
ce_run_id="ce-block-main100-v2-$(date -u +%Y%m%d-%H%M%S)"
ce_model='/root/workspace/ILVR/.worktrees/voff-reproduction-20260917/checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212'
ce_base='/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct'
ce_selection='/root/workspace/ILVR/results/server-comparison-20260912/analysis_completed/visual-causal-panel-20260910t115000z/ce/no_latent_execution/metadata.json'
ce_manifest="./data/ce-vision-block-analysis/$ce_run_id/manifest.json"
mkdir -p "./logs/$ce_run_id"
./venv/ilvr-venv/bin/python scripts/analyze_vision_block.py prepare --model-dir "$ce_model" --base-model-dir "$ce_base" --image-root ./data --data ./data/TEST.jsonl --selection "$ce_selection" --manifest "$ce_manifest" > "./logs/$ce_run_id/prepare-console.log" 2>&1
# Check prepare exit status and manifest identity before the next command.
CUDA_VISIBLE_DEVICES=0 ./venv/ilvr-venv/bin/python scripts/analyze_vision_block.py smoke --model-dir "$ce_model" --base-model-dir "$ce_base" --image-root ./data --manifest "$ce_manifest" --output-dir "./results/$ce_run_id/smoke" > "./logs/$ce_run_id/smoke-console.log" 2>&1
# Check smoke status, identity and all runtime witnesses before launch.
CUDA_VISIBLE_DEVICES=0 nohup ./venv/ilvr-venv/bin/python scripts/analyze_vision_block.py run --model-dir "$ce_model" --base-model-dir "$ce_base" --image-root ./data --manifest "$ce_manifest" --smoke-report "./results/$ce_run_id/smoke/smoke.json" --output-dir "./results/$ce_run_id/main100" > "./logs/$ce_run_id/main100-console.log" 2>&1 &
ce_pid=$!
printf '%s\n' "$ce_pid" > "./logs/$ce_run_id/main100.pid"
# Observe this PID and logs/$ce_run_id/main100/execution.log; do not signal foreign processes.
# After exit and completed status, validate output and summarize:
./venv/ilvr-venv/bin/python scripts/analyze_vision_block.py summarize --manifest "$ce_manifest" --output-dir "./results/$ce_run_id/main100" > "./logs/$ce_run_id/summarize-console.log" 2>&1
```

- [ ] Run revised actual `smoke` on one available exposed GPU; automatic source3/source101 cases must remain single/multi, totaling 7 sites and 315 tokens. Save `results/<run-id>/smoke/` and console `logs/<run-id>/smoke-console.log` outside the CLI-owned empty log leaf. Validate repeat clean, prefix identity and effective MATH/TF32-off witnesses. Diagnostic exact agreement is precedent, not a substitute for this production gate.
- [ ] After smoke passes, launch unchanged `run --smoke-report .../smoke/smoke.json --output-dir ./results/<run-id>/main100` detached with the same model/base/manifest/image arguments and one exposed GPU. Save console/PID under `logs/<run-id>/`; poll actual progress and report failures immediately. All 100 samples must satisfy fixed gates; partial output cannot become a completed smaller cohort.
- [ ] Run offline `summarize` once on completed output. Verify four conditions, identical S/N, 51,372 total four-condition target rows, finite NLL and clean-relative counts; report the five-metric table promptly with artifact links and teacher-forcing scope. Full run and CPU tests are separate claims.
- [ ] Before branch completion, invoke verification-before-completion and finishing-a-development-branch together as applicable; run the project's full suite once: `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest -q`. Read complete output. Persist exact reviewed revision, reviews, approvals, commands and results in existing workflow evidence before any skill scratch cleanup. Human retains merge/PR/keep-as-is choice; no automatic integration or deletion.

## Planning evidence

Controller self-review covers the approved v2 scope, unchanged metric/input contracts, full-length causal geometry, live-prefix evidence, policy lifetime, gate identity and execution order. Task 1 produces the exact interfaces consumed by Task 2. Design approval is recorded separately from this plan's pending handoff review. Context-free shower reader `/root/ce_block_v2_plan_cold_read` (gpt-5.6-sol, medium) read the entire draft inline: verdict minor gaps, intended behavior understood. This revision adds the original-plan mapping, exact failed-manifest/evidence paths, inherited clean-state/score definitions, layer-count/layout checks, baseline limits and CLI commands. The live-Body intervention point remains an explicit implementer test choice, to be justified with actual call-path evidence and checked independently; it is not a new scientific-design choice. The cold read is advisory and does not replace code review or runtime evidence.

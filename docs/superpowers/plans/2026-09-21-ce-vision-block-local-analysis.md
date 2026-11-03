# CE Vision Block Local Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking. The standing ILVR policy requires implementer delegation and independent specification-and-quality review; root/native implementation is not an alternative.

**Goal:** Evaluate clean/body_mask/boundary_mask/full_mask on the approved CE-only model and main100, reporting the user's five next-text metrics with common denominators.

**Architecture:** Add an analysis-only input adapter, replay engine, CLI and standard-library aggregator. Reuse existing Stage 2 preprocessing, model loading, rollout and decoder K/V interfaces; preserve existing training APIs. Each sample provides one immutable clean state; every site/condition replays from END independently.

**Tech Stack:** repository Python 3.11 venv, patched Transformers, PyTorch, PEFT, SDPA decoder, FlashAttention2 vision; standard-library JSON/math for aggregation.

**Spec:** [approved design commitment v1](../specs/2026-09-21-ce-vision-block-local-analysis-design.md). Approved content at commit `b0a665bbddeee78a7bfd31c6a4804a9f711670b7`, SHA-256 `93af8e1f669b461924592145065f8af4393d65cb6f21dd3c876f86b628054588`; user subsequently replied `continue` to the written-spec approval request. This plan awaits its own approval.

## Global Constraints

- Model run `stage2-ce-matched-gb16-s212-20260919-155759`, prepared checkpoint-212 and original Qwen2.5-VL-3B base, exactly as spec §2. Preference remains off; no training.
- Base code commit `f8f80be29182a5d96764f1717a50382bf3b2704c` in `.worktrees/voff-reproduction-20260917`; it is a descendant of approved `1a63667bc9db3f54c5cfc350e874797e790e952a`. Verify that ancestry before creating the implementation worktree.
- `eval()`/gradient off, BF16 model, FP32 log-softmax, latent-size 8, batch sample 1. Analysis replay uses dense additive rectangular masks and the existing SDPA-prefix decoder for all four conditions; no Flex/dense variation across conditions.
- Preserve original 3-axis position columns, logical input length and GT token IDs. Use `concatenated-v1`, `immediate_text`, clean Body, per-site independent branches. Never feed helper-image pixels.
- Main100 comes from spec §3's exact metadata selection and hashes. Structural candidates=327; post-tokenization S/N are measured, not forced. A blank final-answer field is not itself a site exclusion.
- Input identity: TEST SHA-256 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`; selection metadata SHA-256 `fb1f15fef5339abde6ba74d140e5f9a1f8823cedafe8aef1d4bb245a8e0286aa`. Read the metadata JSON's `selection.requested_indices`, require 100 unique in-range zero-based integers and category counts 25/30/15/30. Dataset or selection hash mismatch is a preflight error, not permission to select replacements.
- GPU fidelity gate: zero selected-argmax mismatches, token NLL absolute difference max ≤0.01 nat and mean ≤0.001 nat; repeated clean argmax changes=0. Tiny FP32 logits use `atol=1e-5, rtol=1e-5`. Nonfinite data fails.
- Paths: manifests under worktree `data/`, outputs under `results/<run-id>/`, logs under `logs/<run-id>/`; keep checkpoint/data links. Never overwrite another run or create model files in results.
- All Python commands use `./venv/ilvr-venv/bin/python` with `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD"` from the execution worktree. No system Python, reference imports or dependency installation.
- Sequence: implementation via implementer → independent spec/quality reviews → focused verification → real-model smoke → main100. Resolve required review findings through a fixer. Preserve the skill's whole-branch review and finishing gates.

## Review Focus

1. mRoPE axes differ: preserve all three original columns, never rebuild scalar positions (Task 3).
2. One annotation entry cannot map exactly: record the affected site/sample exclusion before inference, never choose a later token as the first (Task 2).
3. Empty final answer but usable reasoning: keep its sites; do not reuse final-answer scorer eligibility (Task 2).
4. Capture/replay changes attention dispatch: compare ordinary clean, captured clean and replay clean independently; clean agreement must hold (Task 3).
5. A condition fails after partial output: summary remains incomplete with the fixed expected site/token set; never report a smaller successful denominator (Tasks 1 and 4).

## Files and boundaries

Create `src/evaluation/vision_block_metrics.py` (stdlib contracts/aggregation), `vision_block_inputs.py` (processor/token mapping/manifest), `vision_block_runtime.py` (capture/replay/scoring), `vision_block_analysis.py` (CLI orchestration), and `scripts/analyze_vision_block.py` (thin entrypoint). Create matching tests `tests/evaluation/test_vision_block_{metrics,inputs,runtime,analysis}.py` plus `tests/evaluation/vision_block_reference.py` (test-only independent oracle). Do not modify the historical `eval.py`, training planner/validator, or patched decoder merely to accept boundary-only; the low-level decoder already accepts explicit prefix K/V and rectangular visibility.

### Task 1: Result contract and independently testable metrics

**Files:** Create `src/evaluation/vision_block_metrics.py`, `tests/evaluation/test_vision_block_metrics.py`.

**Interfaces:** `CONDITIONS=('clean','body_mask','boundary_mask','full_mask')`. Token rows are dictionaries with `source_index:int`, `plan_index:int`, `target_position:int`, `target_id:int`, `condition:str`, `argmax_id:int`, `nll:float`. `summarize(rows:list[dict], expected:dict[tuple[int,int], tuple[tuple[int,int],...]])->dict` consumes expected `(position,target_id)` tuples per site. Return `status`, `sites`, `tokens`, and `conditions` keyed by condition, each containing `first_changed`, `any_changed`, `tokens_changed`, `mean_nll`, `delta_nll`. Only a complete matching four-condition set returns `status='completed'`; malformed inputs raise `ValueError`.

- [ ] Set up the isolated worktree via `using-git-worktrees`, based on the exact code commit above, at `.worktrees/ce-block-analysis-20260921`. Preserve shared venv/data/checkpoint links and copy the approved control documents byte-for-byte into it. Record baseline identity and existing unrelated changes. Do not implement inside the reproduction worktree.
- [ ] Write the failing metric test (all imports below are part of this task):

```python
from evaluation.vision_block_metrics import CONDITIONS, summarize

def test_common_denominators_and_token_weighted_nll():
    expected = {(0, 1): ((5, 7),), (1, 1): ((5, 7), (6, 8), (7, 9))}
    rows = []
    for (sample, plan), targets in expected.items():
        for pos, target in targets:
            for condition in CONDITIONS:
                changed = condition == 'body_mask' and (sample, pos) == (1, 6)
                rows.append(dict(source_index=sample, plan_index=plan,
                    target_position=pos, target_id=target, condition=condition,
                    argmax_id=99 if changed else target,
                    nll=1.0 if condition == 'clean' else 0.5))
    out = summarize(rows, expected)
    assert (out['sites'], out['tokens']) == (2, 4)
    got = out['conditions']['body_mask']
    assert (got['first_changed'], got['any_changed'], got['tokens_changed']) == (0, 1, 1)
    assert got['mean_nll'] == 0.5 and got['delta_nll'] == -0.5
```

- [ ] Run `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_vision_block_metrics.py -q`; observe the intended missing-module/function failure.
- [ ] Implement indexed rows keyed by `(site, position, condition)`, validate exact expected keys/GT agreement/finite nonnegative NLL, then reduce each condition. Use `math.fsum(nll_values)/N`; first position is the first expected tuple, `any_changed=any(...)`, and clean-relative subtraction is signed. Missing/duplicate rows, unknown condition, empty expected set, wrong target IDs and NaN/Inf are errors; do not import torch.

```python
# Core reduction after exact key-set validation:
mean_nll = math.fsum(condition_nlls) / token_count
first_changed = sum(site_changes[site][0] for site in expected)
any_changed = sum(any(site_changes[site]) for site in expected)
tokens_changed = sum(sum(changes) for changes in site_changes.values())
```

- [ ] Add tests by mutating the fixture: remove one row, duplicate one row, remove an entire site's four conditions while keeping expected, mismatch one target, replace one NLL with NaN, and use unequal per-site NLL to distinguish token mean from site mean. Confirm each error and weighting result explicitly. Run the same file, then commit only the two Task 1 files.

### Task 2: Fixed input selection and aligned site manifest

**Files:** Create `src/evaluation/vision_block_inputs.py`, `tests/evaluation/test_vision_block_inputs.py`.

**Interfaces:** frozen `Site(source_index:int, plan_index:int, start:int, body_start:int, body_end:int, end:int, target_positions:tuple[int,...], target_ids:tuple[int,...])`, where `end` is the END token column and Body is `[body_start,body_end)`. `PreparedSample(source_index:int, model_inputs:dict, sites:tuple[Site,...])`. `prepare_sample(row, source_index, processor, image_root)->tuple[PreparedSample|None,list[dict]]`; exclusions have source/plan indices, stable reason and text explanation. `prepare_manifest(data_path, selection_path, processor, image_root, output_path)->dict` produces schema_version=1, source/selection/processor identity, original 100 IDs, per-sample site records, exclusions, S/N. `load_prepared_samples(manifest, processor, image_root)->iterator[PreparedSample]` re-prepares and rejects a changed input/span/target identity.

- [ ] Write failures using a processor fixture and existing `Stage2PreparedExample` helpers: a valid first site plus an unmappable later site; blank final answer plus valid reasoning; separate final-answer text; a crossing text/text token; a missing/duplicate selection index. Assert exact included positions and exclusion reasons, not just counts.

```python
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from evaluation.vision_block_inputs import prepare_sample

def test_blank_final_answer_keeps_reasoning_site(tmp_path):
    artifact = Path('/root/workspace/ILVR/.worktrees/voff-reproduction-20260917/checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212')
    processor = AutoProcessor.from_pretrained(artifact, local_files_only=True)
    Image.new('RGB', (56, 56)).save(tmp_path / 'original.png')
    row = dict(text_input='Question?', image_input=['original.png'],
        original_final_answer='', sequence_plan=[
            dict(type='text', content='Before.'),
            dict(type='latent', helper_image='must-not-be-loaded.png'),
            dict(type='text', content='Reason one.'),
            dict(type='text', content='The final answer is: 1')])
    sample, exclusions = prepare_sample(row, 57, processor, tmp_path)
    assert sample is not None and len(sample.sites) == 1 and not exclusions
    assert sample.sites[0].target_positions[0] == sample.sites[0].end + 1
    assert 'final answer' not in processor.tokenizer.decode(sample.sites[0].target_ids)
    assert 'pixel_values_latent' not in sample.model_inputs
```
- [ ] Run `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_vision_block_inputs.py -q`; confirm failures are for the new contract.
- [ ] Implement preparation using `stage2.data.prepare_stage2_example(..., latent_size=8, assistant_plan_rendering='concatenated-v1')` and `Stage2DataCollator(..., step_preference_enabled=False)`. Keep original processor outputs; independently obtain offsets as existing collator code does, verify IDs/masks agree, and map each available reasoning step with existing `map_reasoning_step_spans(..., reasoning_steps=(step,), preference_target_scope='immediate_text')`. Supply the complete original rendered text and plan spans, not a shortened conversation. If rendering/tokenization invalidates the whole sample, log a sample-level exclusion rather than pretending to isolate a site. Missing image/file/device failures are operational failures, not discretionary sample exclusions.

```python
# Once a validated span is returned; actual plan_index comes from matching
# step.latent to the original prepared.plan_char_spans entry.
positions = tuple(range(span.target_text_start, span.target_text_end))
site = Site(source_index, plan_index, span.latent_start - 1,
            span.latent_start, span.latent_end, span.latent_end,
            positions, tuple(int(ids[p]) for p in positions))
assert site.end + 1 == site.target_positions[0]
assert site.body_end - site.body_start == 8
```

- [ ] Verify site target IDs omit special tokens, original input retains excluded crossing tokens, the actual first token remains evaluated, and inference kwargs exclude metadata/labels/helper tensors. Record hashes for processor-rendered IDs, attention mask, pixels/grid, sites and target identities; tokenization changes must fail revalidation. Select only source metadata's 100 indices, never sample replacements after seeing model predictions.
- [ ] Run the new input tests plus `tests/step_preference/test_collator_spans.py` and `tests/step_preference/test_target_scope.py`; commit only Task 2 files.

### Task 3: Capture, independent four-condition replay and runtime witnesses

**Files:** Create `src/evaluation/vision_block_runtime.py`, `tests/evaluation/test_vision_block_runtime.py`, `tests/evaluation/vision_block_reference.py`.

**Interfaces:** `CleanState(embeds, positions, prefix_kv, normal_scores, captured_scores)` contains detached device tensors and `(argmax_id,nll)` results keyed by `(plan_index,target_position)`; `capture_clean(model, sample:PreparedSample)->CleanState`; `build_replay_inputs(state,site:Site,condition:str)->tuple[Tensor,Tensor,Tensor]` returns suffix embeddings, original 3-axis columns and additive mask; `replay_site(model,state,site:Site,condition:str)->list[dict]`; `analyze_sample(model,sample,repeat_clean=False)->tuple[list[dict],dict]`. The witness dict contains normal-vs-captured-vs-replay argmax/NLL differences, repeat-clean differences, tensor positions/shapes, dtype/backends, and finite checks. Every sample enforces the spec fidelity threshold, not only smoke samples.

- [ ] Write a failing visibility test with a tiny image-aware Qwen fixture and differing mRoPE axes. At site START=2, Body=3..10, END=11, targets=12,13, require root visible positions: clean=0..10, body_mask=0..2, boundary_mask=0,1,3..10, full_mask=0,1. No condition sees clean root column 11 or later. Every suffix sees itself and earlier rows of that suffix only. END input is zero exactly for boundary_mask/full_mask; actual position columns must match the fixture's three distinct axes.

```python
import torch
from evaluation.vision_block_inputs import Site
from evaluation.vision_block_runtime import CleanState, build_replay_inputs

def test_boundary_mask_preserves_body_and_original_three_axis_positions():
    p = torch.arange(14)[None, None, :]
    state = CleanState(torch.ones(1,14,8), torch.cat((p,2*p,3*p)), (), {}, {})
    site = Site(0,1,2,3,11,11,(12,13),(7,8))
    embeds, positions, mask = build_replay_inputs(state,site,'boundary_mask')
    assert torch.equal(positions, state.positions[:,:,11:13])
    assert torch.count_nonzero(embeds[:,0]) == 0
    visible = mask[0,0] == 0
    assert visible[0,:14].nonzero().flatten().tolist() == [0,1,3,4,5,6,7,8,9,10]
    assert visible[:,14:].tolist() == [[True,False],[True,True]]
    assert torch.equal(state.embeds, torch.ones_like(state.embeds))
```
- [ ] Run `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_vision_block_runtime.py -q`; confirm the intended missing runtime failure.
- [ ] Implement capture without changing production training code. Resolve the loaded PEFT Qwen via `stage2.peft.resolve_stage2_qwen_model`. Temporarily register removable decoder pre/post hooks only to **observe** the final full-length clean decoder call (rollout recurrence calls have shorter lengths). Require exactly one matching final call, collect its input embeddings, original positions and output hidden states, then remove both hooks in `finally`, also on error. Normal call is the loaded model's ordinary Stage 2 forward with `stage2_shared_prefix_inputs=None`, no labels, `stage2_selective_logits_only=True` and a single valid logit position to avoid full-vocabulary projection for every input token. Score the saved normal hidden states at selected target−1 positions in fixed 128-row vocabulary chunks.
- [ ] From those frozen clean embeddings/positions, make one separate existing decoder call with `generate_mode=True`, `use_cache=False`, `return_dict=True`, `stage2_capture_kv=True`. Read `stage2_key_values` from its output. Score the same selected hidden rows and compare captured clean with ordinary clean; record the capture path's dispatch rather than assuming capture is passive. The hook must not enable capture or alter the original normal call.
- [ ] Implement replay one site at a time; suffix is the full original interval `[END, last_evaluated_target)`, preserving any unscored input rows. Build explicit additive rectangular visibility without invoking the training `SidecarPlan` validator. Use original positions via `state.positions.index_select(2, columns)`. All branches use `qwen.model(..., stage2_prefix_key_values=state.prefix_kv, attention_mask=mask, generate_mode=True, use_cache=False, return_dict=True)`.

```python
# Root has L columns; q is suffix input-row count. site.end is not visible.
root_visible = torch.arange(L, device=device) < site.end
if condition in ('body_mask', 'full_mask'):
    root_visible[site.body_start:site.body_end] = False
if condition in ('boundary_mask', 'full_mask'):
    root_visible[site.start] = False
visibility = torch.cat((root_visible.expand(q, L),
                        torch.ones(q, q, dtype=torch.bool, device=device).tril()), dim=1)
mask = torch.zeros((1, 1, q, L + q), device=device, dtype=embeds.dtype)
mask.masked_fill_(~visibility[None, None], float('-inf'))
if condition in ('boundary_mask', 'full_mask'):
    embeds = embeds.clone()
    embeds[:, 0] = 0
```

- [ ] Select predictor rows `target_position - 1 - site.end`, project with the existing LM head in 128-row chunks, FP32 log-softmax, deterministic first-index argmax, and gather GT NLL. Return Task 1 rows. Run all sites of each condition in clean→body_mask→boundary_mask→full_mask order for that sample; do not mutate shared clean tensors. Compare replay clean to ordinary/captured clean using all evaluated rows. Repeat clean for smoke and reject nonzero repeated argmax changes.
- [ ] Implement the independent test oracle in `vision_block_reference.py`: run one site at a time, build dense permitted edges with Python index predicates directly from START/Body/END, then loop over decoder layers using the same fixed captured layer K/V and model layer operations. It must not call production planner, visibility builder, replay helper or Task 1 aggregator. Sharing decoder math/weights is intentional and recorded. The following oracle skeleton specifies the layer path; form `oracle_mask` from the explicit permitted-edge sets in the first test, not the production helper:

```python
hidden = suffix_embeds.clone()
rotary = qwen.model.rotary_emb(hidden, original_position_columns)
for layer_index, layer in enumerate(qwen.model.layers):
    hidden = layer(hidden, attention_mask=oracle_mask,
        position_ids=original_position_columns,
        position_embeddings=rotary, past_key_value=None,
        use_cache=False, output_attentions=False, generate_mode=True,
        stage2_prefix_key_value=frozen_prefix_kv[layer_index])[0]
oracle_logits = qwen.lm_head(qwen.model.norm(hidden)).float()
```

Compare all four conditions' logits against production at `atol=1e-5, rtol=1e-5` on one- and two-site tiny cases. Compare body-mask logits with the existing `build_context_sidecar_plan(visual_only=True, suppress_latent_boundaries=False)` execution using the same frozen inputs, so 'current body mask' is validated against the existing path as well as the independent oracle.
- [ ] Add decisive perturbation tests: mutate captured blocked Body for body_mask/full_mask; mutate captured START for boundary_mask/full_mask; mutate captured original END and later root keys for all conditions. Corresponding off logits must remain unchanged. In a deliberately nondegenerate tiny fixture, visible Body perturbation must change boundary_mask logits; otherwise the negative controls could pass on a disconnected model. Check cleanup after thrown decoder errors, all three position axes, normal→capture→replay comparison and cross-site isolation. These tests are the first-nail witness, not generic shape checks.
- [ ] Run the runtime file plus `tests/step_preference/test_context_sidecars.py` and `tests/step_preference/test_clean_first_order.py` under the same venv/PYTHONPATH. Preserve GPU-marker skips as skips, not passes. Commit only Task 3 files and obtain independent task review before integration.

### Task 4: CLI, smoke gate, main100 and offline reporting

**Files:** Create `src/evaluation/vision_block_analysis.py`, `scripts/analyze_vision_block.py`, `tests/evaluation/test_vision_block_analysis.py`.

**Interfaces:** `main(argv=None)->int`, subcommands `prepare`, `smoke`, `run`, `summarize`. Common flags: `--model-dir`, `--base-model-dir`, `--data`, `--selection`, `--image-root`, `--manifest`; output-producing flags `--output-dir`. `run` also requires `--smoke-report`. Unknown modes/options fail. Loading is lazy so `summarize` imports only stdlib metrics and never torch/model/Reference.

- [ ] Write failing CLI tests with fake prepare/model/runtime functions: incomplete condition output cannot yield completed summary; changed manifest/model/code identity rejects the smoke report; failed gate blocks main100; logs/output directories must belong to the current repository storage trees; preexisting nonempty run output fails without overwrite; unknown command exits nonzero. No actual model is loaded in these tests.

```python
import pytest
from evaluation.vision_block_analysis import main

def test_unknown_command_fails_before_loading_model():
    with pytest.raises(SystemExit) as error:
        main(['unknown-analysis'])
    assert error.value.code == 2
```
- [ ] Run `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_vision_block_analysis.py -q`; confirm failures, then implement dispatch and a thin `if __name__ == '__main__': raise SystemExit(main())` entrypoint.
- [ ] `prepare` uses the prepared artifact's processor, Task 2 and exact spec hashes, writes a fresh data manifest, and records S/N/exclusions. `smoke`/`run` reuse `eval.load_evaluation_artifacts(model_dir=...,base_model_dir=...,cache_dir=...)` with only one GPU exposed, ensure stage2 configuration and SDPA/FA2, call `eval()` and `torch.inference_mode()`. Record actual parameter dtypes, TF32 settings, software/code/model/processor identities; keep these fixed between smoke and run. Existing loader may set TF32; record and preserve that setting rather than silently claiming it is off. Do not use a Stage 1 fresh-training loader for the Stage 2 artifact.
- [ ] `smoke` chooses first valid single-site and first multi-site sample by source index. Counts refer to post-tokenization valid sites. If no sample has exactly one valid site, use only the first valid site of the first multi-site sample as the single-branch case, then all valid sites of that same sample as the multi-branch case. Do not truncate its original prefix or regenerate Body. If no multi-site sample exists, fail the gate rather than claim that case passed. It runs Task 3 including repeat clean. The JSON gate is passed only if thresholds, visibility/position/finite assertions and both cases succeed; record tested source IDs and exact execution identities. `run` validates that report, writes `status=incomplete` before inference, runs every manifest site, writes one `tokens.jsonl` (four conditions) plus per-site and exclusion outputs, invokes Task 1 to validate complete coverage, and only then sets completed.
- [ ] `summarize --manifest ... --output-dir ...` rereads `tokens.jsonl`, uses the manifest's entire expected set and Task 1, then writes `summary.json`, `summary.csv`, and Korean `report.md` under output-dir. Report four rows of five metrics with fractions/percentages, natural-log token mean, signed clean-relative ΔNLL, S/N/exclusions, independent-site teacher forcing and residual image/Body-history caveats. A failed/partial run cannot be relabeled completed by the offline command.
- [ ] Test the whole fake-data CLI sequence and error cases; compare offline summary with known expected counts and NLL independently. Run one focused combined set after all fixes:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_vision_block_metrics.py tests/evaluation/test_vision_block_inputs.py tests/evaluation/test_vision_block_runtime.py tests/evaluation/test_vision_block_analysis.py tests/step_preference/test_collator_spans.py tests/step_preference/test_target_scope.py tests/step_preference/test_context_sidecars.py tests/step_preference/test_clean_first_order.py -q
```

- [ ] Commit only Task 4 files. Obtain final independent specification/quality review; send deliverable fixes to a fixer, and record reviewed commit identity. Resolve gate-blocking findings before real-model execution.
- [ ] After code review and plan approval, create a unique run ID and verify checkpoint/data paths plus available GPU resources. Execute `prepare`, then `smoke`, then detached `run` using the following flag mapping. Preserve stdout/stderr in worktree logs. There is no automatic retry or resume framework; after a failure fix the cause and use a fresh run ID.

```text
entrypoint = ./venv/ilvr-venv/bin/python scripts/analyze_vision_block.py
model-dir = /root/workspace/ILVR/.worktrees/voff-reproduction-20260917/checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212
base-model-dir = /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct
data = ./data/TEST.jsonl
selection = /root/workspace/ILVR/results/server-comparison-20260912/analysis_completed/visual-causal-panel-20260910t115000z/ce/no_latent_execution/metadata.json
image-root = ./data
manifest = ./data/ce-vision-block-analysis/<run-id>/manifest.json
smoke output-dir = ./results/<run-id>/smoke
run output-dir = ./results/<run-id>/main100
smoke-report = ./results/<run-id>/smoke/smoke.json
```

`<run-id>` is the unique runtime identifier, not a missing protocol choice. Choose one once from UTC timestamp and keep it in all paths. Apply `CUDA_VISIBLE_DEVICES` only after checking ownership/resource availability; never signal another process. Use `nohup` with a saved PID only for this run's main evaluation, and report progress as completed samples/sites against the frozen manifest.

- [ ] Run offline `summarize` once after completed inference, check four-condition coverage and exact common S/N, report the requested table immediately. Do not wait for optional hardening.
- [ ] For branch completion, invoke both verification-before-completion and finishing-a-development-branch. Their shared full-suite witness is `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest -q`, run once against the final reviewed revision. This full suite does not replace the distinct smoke/output witnesses. If failures exist, report them and resolve applicable blockers; no clean completion claim. Human retains merge/PR/keep-as-is choice; do not auto-merge or delete worktrees.

## Plan self-review and handoff

Spec coverage: §2→Tasks 3/4; §3→Task 2; §4→Task 3; §5→Tasks 1/4; §6→file map and Tasks 2–4; §7→all tests, smoke and final gates; §8–9→workflow/delegation/approval constraints. Review Focus entries each have owning tests. Task 1's row contract is used unchanged by runtime and CLI; Task 2's END/body ranges are used by replay and oracle. Ordinary clean stays uninstrumented except observation hooks; capture's separate attention dispatch is explicitly compared, not presumed equivalent.

Implementation approval is pending. After root self-review and a context-free handoff read, present this plan for user review. Execution method is already fixed by standing policy: subagent-driven development, independent review, root coordination. Plan approval authorizes its implementation, tests, smoke and 100-sample evaluation within the stated scope; it does not waive runtime gates.

# Image and Current-Body Local Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. The user already selected sub-agent implementation with independent review; retain that selection. Steps use checkbox (`- [ ]`) syntax for tracking. The user's standing workflow prohibits an inline fallback.

**Goal:** Measure the approved three-condition source-image/current-Body intervention on the fixed beta0.5 model and 100 samples, with the same five clean-relative metrics.

**Architecture:** Extend the existing full-length analysis with two explicit profiles and a validated original-image position set. Preserve ordinary clean rollout and pre-END states; apply extra key masks only to queries at/after the evaluated END. Reuse preparation, scoring, detached execution and output validation rather than adding orchestration.

**Tech Stack:** Repository Python 3.11 venv, patched Transformers, PyTorch 2.7.1, BF16/SDPA MATH, PEFT DoRA, stdlib JSON/CSV, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-image-current-body-local-analysis-design.md`, approved by user `ok` at commit `8fa4c2bacc7ce0f6c307406766ca2d515ee72e01`, SHA256 `0b8ba453dcad681fde4e49f336cc3f269481cd54f73ae16215a747f1a1955d66`. This plan still requires handoff review before implementation.

## Global Constraints

- "모든 입력 embedding을 그대로 유지"; new conditions never zero END or regenerate latent Body without the original image.
- "`q < e`는 세 조건 모두 원래 causal visibility 그대로다." For q ≥ e, `image_mask` blocks original image payload keys I; `image_body_mask` blocks I ∪ current Body8 only.
- "START/END, 다른 Body와 이전 텍스트에 대한 기존 causal 접근은 유지한다." Each site independently branches from the same clean state.
- "모든 layer의 `position < e` K/V가 clean과 bit-identical인지 검사한다." No cached short-suffix execution or stored-K/V mutation masquerading as a live causal test.
- "기본값은 기존 네 조건의 `block-local-v2`로 유지"; `image-local-v1` explicitly selects `clean,image_mask,image_body_mask`. Reject unknown/mismatched profiles; records without a profile are legacy only.
- "정답 y는 TEST annotation에서 독립적으로 얻으며 clean 예측을 정답으로 쓰지 않는다." Same100/S327/N12843; image profile requires exactly38,529 token rows; neither exclusions nor denominator drift may silently shrink this run.
- "두 mask 조건 모두 같은 완전한 clean과 비교한다." FP32 log-softmax, token-weighted NLL, signed ΔNLL; no generation accuracy or complete-image-information-erasure claim.
- "SDPA MATH only, matmul/cudnn TF32 모두 off, vision FlashAttention2"; BF16 base and saved FP32 adapters unchanged, including observed policy after loader mutations.
- Expected decoder layer count is 36. Each site/condition compares K and V in every layer: `sites × conditions × 36 × 2`. Smoke: `7 × 3 × 36 × 2 = 1512`, repeated clean: `7 × 1 × 36 × 2 = 504`; main: `327 × 3 × 36 × 2 = 70632`. Fail on an unexpected model layer count rather than silently changing these totals.
- Metrics are `first_changed/S`, `any_changed/S`, `tokens_changed/N`, token-weighted mean NLL (natural log), and signed condition mean NLL minus fully clean mean NLL. Store integer change counts and denominators; report count/denominator and percentage.
- Clean gates: argmax mismatch0; max target NLL abs difference ≤0.01 nat, mean ≤0.001 nat, separately per sample/comparison. Source3/source101 smoke:7sites315targets,1,512 condition-prefix K/V comparisons plus504 repeated-clean comparisons. Main:70,632 condition-prefix comparisons.
- Use `/root/workspace/ILVR/.worktrees/ce-block-analysis-20260921` and `./venv/ilvr-venv/bin/python`; `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD"`. Preserve data/checkpoints/venv symlinks and existing untracked artifacts. No dependency reinstall or training/decoder changes.
- Product changes belong to implementer/fixer agents; independent task spec/quality review and final review precede GPU execution. Paperthin hooks are root-only. Root owns execution and control artifacts.
- Logs under `logs/`, results under `results/`, manifests under `data/`; model files only under `checkpoints/`. Fresh run ID after failure; no retry/recovery framework.

## Review Focus

1. Image payload confused with marker/latent IDs or malformed grid: reject before inference, preserving marker access. Task1 geometry tests and Task2 model-vs-processor validation.
2. Image masking starts after END rather than at END: first predicted text token must already use the intervention. Task2 literal matrix and live attention tests include row e.
3. Global/cumulative intervention changes prior clean computation or other Body: Task2 multi-site, reversed-site order, exact prefix/embedding checks.
4. Legacy/cross-profile manifests or smoke results authorize wrong conditions: Task3 parser, stored profile/order, wrong-gate and legacy-summary tests; Task1 exact coverage tests.
5. Metric/profile integration hides missing conditions or wrong clean denominator: Task1 unequal-length hand calculation and Task3 offline subprocess tests, plus final raw-row audit.

---

## File map and task dependencies

| File | Responsibility | Owner |
|---|---|---|
| `src/evaluation/vision_block_metrics.py` | stdlib profile lookup and profile-specific exact coverage, unchanged metric formulas | Task1 |
| `src/evaluation/vision_block_inputs.py` | validate original payload geometry, persist/recheck image positions and profile | Task1 |
| `tests/evaluation/test_vision_block_metrics.py`, `test_vision_block_inputs.py` | contracts above | Task1 |
| `src/evaluation/vision_block_runtime.py` | original-image extraction from actual model IDs, full-length masks, actual attention witnesses | Task2 |
| `tests/evaluation/test_vision_block_runtime.py`, `tests/evaluation/vision_block_reference.py` | causal controls, independent oracle, legacy runtime regression | Task2 |
| `src/evaluation/vision_block_analysis.py` | CLI profile plumbing, identity/smoke validation, actual-model report labels | Task3 |
| `tests/evaluation/test_vision_block_analysis.py` | CLI contracts and actual tiny-model integration | Task3 |

Execute1→review1→2→review2→3→review3. These tasks have producer/consumer dependencies and must not be run concurrently. Existing `scripts/analyze_vision_block.py` is a thin entrypoint and needs no behavior change. If an unforeseen implementation change affects the approved intervention, stop at design revision rather than silently widening scope.

### Task 1: Profiles, image geometry and exact metric contracts

**Files:** Modify the two Task1 modules and their existing tests listed above. No new framework/module.

**Interfaces produced:**

```python
# vision_block_metrics.py; stdlib only
DEFAULT_PROFILE = 'block-local-v2'
IMAGE_PROFILE = 'image-local-v1'
CONDITIONS = ('clean', 'body_mask', 'boundary_mask', 'full_mask')  # preserved
IMAGE_CONDITIONS = ('clean', 'image_mask', 'image_body_mask')

def conditions_for_profile(analysis_profile: str) -> tuple[str, ...]:
    if analysis_profile == DEFAULT_PROFILE:
        return CONDITIONS
    if analysis_profile == IMAGE_PROFILE:
        return IMAGE_CONDITIONS
    raise ValueError(f'unknown analysis profile: {analysis_profile!r}')

# Existing summarize gains keyword-only analysis_profile=DEFAULT_PROFILE.
# Return the existing sites/tokens/conditions metric structure; additive
# analysis_profile/condition_order metadata is allowed, never metric renaming.

# vision_block_inputs.py
# processor_image_contract(processor) -> dict[str, int]
#   exact keys: image_token_id, vision_start_token_id, vision_end_token_id,
#               merge_size
# original_image_positions(sample: PreparedSample, *, image_token_id: int,
#     vision_start_token_id: int, vision_end_token_id: int,
#     merge_size: int) -> tuple[int, ...]
# prepare_manifest(data_path, selection_path, processor, image_root, output_path,
#                  *, analysis_profile=DEFAULT_PROFILE) -> dict
# load_prepared_samples(manifest, processor, image_root) retains its signature.
```

`processor_image_contract` resolves IDs from the processor's image token and `<|vision_start|>`/`<|vision_end|>` via its tokenizer, and `processor.image_processor.merge_size`. Reject missing/unknown, negative, bool or equal IDs and nonpositive/bool merge size. These IDs must not be hard-coded to production values.

`original_image_positions` validates unpadded batch1 input IDs, exactly one image grid row of three positive integers, h/w divisible by merge_size, one ordered original marker pair, contiguous payload exactly between markers, matching grid token count `t*h*w//merge_size**2`, all payload positions before the earliest evaluated latent START, and no stray image payload IDs outside that span. Return a sorted tuple; raise ValueError for malformed geometry rather than excluding the sample. Preserve PreparedSample and its identity hash.

Every newly prepared manifest records root `analysis_profile` and `condition_order`. Image profile also records root `image_token_contract` and each sample's `original_image_positions` list. Keep schema_version1 and existing source/selection/processor/sites/identity fields. Legacy manifests missing the new fields remain readable under the default profile. Re-preparation of an image profile must recompute and compare contract and exact position list before yielding samples; missing, duplicate, moved or non-int persisted positions fail.

- [ ] **Step1: Add metric RED cases using the existing module.** This explicit unequal-length fixture catches wrong denominators and image-mask-as-baseline mistakes:

```python
def test_image_profile_metrics_use_full_clean():
    expected = {(0, 1): ((5, 7),), (1, 1): ((9, 8), (10, 9))}
    rows = []
    for condition in ('clean', 'image_mask', 'image_body_mask'):
        for (source, plan), targets in expected.items():
            for position, target in targets:
                changed = condition != 'clean' and source == 1 and position == 10
                rows.append(dict(source_index=source, plan_index=plan,
                    target_position=position, target_id=target, condition=condition,
                    argmax_id=99 if changed else target,
                    nll={'clean': 2., 'image_mask': 1., 'image_body_mask': 3.}[condition]))
    out = summarize(rows, expected, analysis_profile='image-local-v1')
    assert (out['sites'], out['tokens']) == (2, 3)
    assert out['conditions']['image_mask'] == dict(first_changed=0, any_changed=1,
        tokens_changed=1, mean_nll=1., delta_nll=-1.)
    assert out['conditions']['image_body_mask']['delta_nll'] == 1.
    with pytest.raises(ValueError):
        summarize(rows[:-1], expected, analysis_profile='image-local-v1')
    with pytest.raises(ValueError):
        summarize(rows, expected)  # image rows cannot pass as legacy
    with pytest.raises(ValueError):
        summarize(rows, expected, analysis_profile='typo')
```

- [ ] **Step2: Add geometry RED cases without loading a model.** Add imports for torch, PreparedSample, Site and the new helper to the existing inputs test module.

```python
def test_image_payload_positions_exclude_boundaries():
    ids = torch.tensor([[10, 20, 21, 21, 22, 30, 91, *([90]*8), 92, 40]])
    sample = PreparedSample(0, dict(input_ids=ids,
        attention_mask=torch.ones_like(ids), image_grid_thw=torch.tensor([[1, 1, 2]])),
        (Site(0, 1, 6, 7, 15, 15, (16,), (40,)),))
    contract = dict(image_token_id=21, vision_start_token_id=20,
                    vision_end_token_id=22, merge_size=1)
    assert original_image_positions(sample, **contract) == (2, 3)
    bad_grid = dict(sample.model_inputs, image_grid_thw=torch.tensor([[1, 1, 3]]))
    with pytest.raises(ValueError):
        original_image_positions(PreparedSample(0, bad_grid, sample.sites), **contract)
    for column, value in [(1, 10), (4, 21), (5, 21)]:
        bad = ids.clone(); bad[0, column] = value
        with pytest.raises(ValueError):
            original_image_positions(PreparedSample(0,
                dict(sample.model_inputs, input_ids=bad), sample.sites), **contract)
```

Extend with parameterized zero/negative/non-integral grids, multiple images/markers, missing payload, misplaced image after latent START, invalid merge and malformed persisted image lists. Use the existing real processor fixture for prepare/load round-trip and verify original `_sample_identity` and Site tuples unchanged. Add an image-profile manifest re-load test whose saved positions are mutated; it must fail even when the old input hash stays valid. Keep all existing legacy tests.

- [ ] **Step3: Run focused RED once and inspect causal failures.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_vision_block_metrics.py tests/evaluation/test_vision_block_inputs.py
```

Expected failures concern missing profile/helper interfaces. Distinguish fixture/import mistakes from intended behavioral failures in the report.

- [ ] **Step4: Implement the minimal contracts.** Use `conditions_for_profile` instead of global CONDITIONS inside summarize; leave its clean-relative comparison and math.fsum unchanged. Validate geometry with tokenizer-resolved IDs and explicit ranges:

```python
ids = sample.model_inputs["input_ids"]
payload = tuple(torch.nonzero(ids[0] == image_token_id, as_tuple=False).flatten().tolist())
# After checking one start/end pair, positive integral grid and merge size:
expected = tuple(range(start + 1, end))
if not payload or payload != expected or end >= min(s.start for s in sample.sites):
    raise ValueError('invalid original image span')
if len(payload) != t * h * w // (merge_size ** 2):
    raise ValueError('original image payload/grid mismatch')
```

Integrate manifest metadata into the existing sample preparation loop; do not tokenize images twice for a new cache, replace image files, or alter target identity. Load-time validation must use actual regenerated input tensors, not merely trust persisted positions.

- [ ] **Step5: Run the same focused tests after the change, inspect GREEN, self-review, commit.** Do not repeat unchanged GREEN merely because of the commit. Report exact commands/results, changed files, interfaces and remaining real-model gates. Root obtains an independent spec/quality review of this task's recorded base→head diff before Task2.

```bash
git add src/evaluation/vision_block_metrics.py src/evaluation/vision_block_inputs.py tests/evaluation/test_vision_block_metrics.py tests/evaluation/test_vision_block_inputs.py
git commit -m "feat: add image analysis profile and payload contracts"
```

### Task 2: Full-length image/current-Body replay and live witnesses

**Files:** Runtime module, existing runtime tests and `tests/evaluation/vision_block_reference.py` only.

**Consumes:** Task1 `DEFAULT_PROFILE`, `IMAGE_PROFILE`, `conditions_for_profile`, `original_image_positions`. Derive the image contract from `qwen.config.image_token_id`, `vision_start_token_id`, `vision_end_token_id`, and `qwen.config.vision_config.spatial_merge_size`; these actual model values must agree with the processor contract carried by the manifest at CLI execution.

**Interfaces produced (existing positional arguments stay compatible):**

```python
# Add after existing CleanState.dispatch to preserve old positional construction:
# image_positions: tuple[int, ...] = ()
# image_token_contract: dict = field(default_factory=dict)

# capture_clean(model, sample, *, analysis_profile=DEFAULT_PROFILE) -> CleanState
# build_replay_inputs(state, site, condition, *, analysis_profile=DEFAULT_PROFILE)
#     -> (embeds, original_positions, additive_4d_mask)
# replay_site(model, state, site, condition, *, prefix_witness=None,
#             visibility_witness=None, analysis_profile=DEFAULT_PROFILE) -> list[dict]
# analyze_sample(model, sample, repeat_clean=False, *,
#                analysis_profile=DEFAULT_PROFILE) -> (rows, witness)
```

Image-profile witness adds `analysis_profile`, `condition_order`, `image_token_contract`, `original_image_positions` and `visibility_identity`. Keep existing fidelity/policy/prefix fields. Each successful site/condition visibility record has exactly meaningful values:

```python
dict(plan_index=site.plan_index, condition=condition,
     query_start=site.end, query_stop=site.target_positions[-1],
     blocked_positions=sorted(blocked), layers_observed=len(qwen.model.layers),
     exact=True, embeddings_unchanged=True)
```

`query_stop` is exclusive: the last scored predictor is last_target_position−1. `blocked_positions` is empty for clean, I for image_mask, I∪current Body for image_body_mask. Records are appended only after actual execution and checks succeed; duplicate/missing layer calls fail. Legacy witness schema remains accepted for legacy execution.

- [ ] **Step1: Add RED full-geometry and actual tiny-model cases.** Reuse existing `image_case(two_sites=True)` (two layers, original payload positions2,3, latent_size8) and existing distinct three-axis positions. This example must fail on the old unsupported image profile:

```python
def test_image_profile_keeps_every_embedding_and_prefix(runtime):
    model, sample = image_case(two_sites=True)
    state = runtime.capture_clean(model, sample, analysis_profile='image-local-v1')
    assert state.image_positions == (2, 3)
    saved = state.embeds.clone()
    for site in reversed(sample.sites):
        for condition in ('clean', 'image_mask', 'image_body_mask'):
            embeds, positions, mask = runtime.build_replay_inputs(
                state, site, condition, analysis_profile='image-local-v1')
            blocked = set() if condition == 'clean' else {2, 3}
            if condition == 'image_body_mask':
                blocked.update(range(site.body_start, site.body_end))
            length = embeds.shape[1]
            expected = torch.tensor([[k <= q and (q < site.end or k not in blocked)
                                      for k in range(length)] for q in range(length)])
            assert torch.equal(torch.isfinite(mask[0, 0]), expected)
            assert torch.equal(embeds, saved)
            assert torch.equal(positions, state.positions)
            prefixes, visibility = [], []
            rows = runtime.replay_site(model, state, site, condition,
                analysis_profile='image-local-v1', prefix_witness=prefixes,
                visibility_witness=visibility)
            assert len(rows) == len(site.target_ids)
            assert prefixes[0]['exact'] and visibility[0]['layers_observed'] == 2
            assert visibility[0]['blocked_positions'] == sorted(blocked)
    assert torch.equal(state.embeds, saved)
```

Also compare forward versus reversed site results; verify every unblocked START/END/other Body is visible under original causality. Reject missing/duplicate/out-of-range image positions, foreign-profile condition strings, and model-ID/grid disagreement before producing accepted rows.

- [ ] **Step2: Add the live causal falsification test.** Intercept the **actual** SDPA call in the tiny model. Preserve the original prefix outputs and perturb only the V contribution used for suffix query rows. Do not change pixel inputs or stored clean K/V. The core probe is:

```python
def suffix_value_probe(original_sdpa, query, key, value, *, mask, end,
                       chosen_key, replacement, **kwargs):
    ordinary = original_sdpa(query, key, value, attn_mask=mask, **kwargs)
    changed_value = value.clone()
    changed_value[..., chosen_key, :] = replacement
    changed = original_sdpa(query, key, changed_value, attn_mask=mask, **kwargs)
    return torch.cat((ordinary[..., :end, :], changed[..., end:, :]), dim=-2)
```

Attach this only during actual replay, normalize positional/keyword arguments of the installed SDPA API in the test, and count observed calls. Capture outputs with the existing test helper `production_logits(runtime, model, state, site, condition, *, analysis_profile=DEFAULT_PROFILE)`, forwarding that keyword into `runtime.replay_site`. Use deterministic nonuniform replacement vectors. The same image-key intervention must change selected clean logits but leave both image-mask conditions invariant; a current-Body key intervention must change image_mask logits but leave image_body_mask invariant. Require all prefix K/V checks to continue passing and at least one visible-control target logit to differ; do not count an unexercised monkeypatch as success. Cover both single and two-site samples and remove hooks/patches on failures.

- [ ] **Step3: Extend the independent FP32 oracle.** Add keyword-only `analysis_profile=DEFAULT_PROFILE` to existing `reference_logits`; for the image profile build the explicit visibility in that test helper without calling production planners. Reuse clean layerwise K/V for the reference's prefix and retain full original positions at the corresponding suffix positions. Image and current Body blocking must be derived from original integer geometry, not production `build_replay_inputs`. Compare corresponding selected production logits at atol1e−5/rtol1e−5 and check target−1 scoring. The existing default oracle behavior stays unchanged.

```python
actual = production_logits(runtime, model, state, site, 'image_mask',
                           analysis_profile='image-local-v1')
expected = reference_logits(model, state, site, 'image_mask',
                            analysis_profile='image-local-v1')
torch.testing.assert_close(actual[:, site.end:site.target_positions[-1]],
                           expected, atol=1e-5, rtol=1e-5)
```

- [ ] **Step4: Run runtime RED, then implement capture/profile plumbing and masks.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_vision_block_runtime.py
```

For image profile capture, validate the original image geometry with actual model IDs before ordinary rollout and store positions/contract in CleanState. Keep ordinary generation otherwise unchanged. The new mask branch is:

```python
if analysis_profile == IMAGE_PROFILE:
    if condition in ('image_mask', 'image_body_mask'):
        visibility[site.end:, list(state.image_positions)] = False
    if condition == 'image_body_mask':
        visibility[site.end:, site.body_start:site.body_end] = False
    # No embedding mutation, including END.
else:
    if condition in ("body_mask", "full_mask"):
        visibility[site.end:, site.body_start:site.body_end] = False
    if condition in ("boundary_mask", "full_mask"):
        visibility[site.end:, site.start] = False
        embeds[:, site.end] = 0
```

Reject a condition not in `conditions_for_profile` before building tensors.

For actual image-profile visibility witnesses, use temporary `self_attn` forward-pre-hooks on each decoder layer during replay. Observe the effective additive mask after the decoder's mask preparation. Validate finite visibility for rows `[e:last_target_position)` against a literal causal-and-blocked expression; this includes every predictor and any intervening unscored row. Check full mask dimensions and layer coverage, all input embeddings/positions, and retain exact pre-END K/V checks. Tiny tests additionally exercise all full-L matrix entries and the actual SDPA path. Remove hooks in finally; do not patch shared decoder or globally wrap SDPA in production. A boolean set by the mask builder alone is not a runtime witness.

For the image profile, implement the approved repeat pair explicitly: existing `repeat_captured` currently compares the second ordinary/captured scores, so simply retaining that expression would not test two captured runs. Keep legacy witness semantics unchanged; for the new profile use:

```python
if analysis_profile == IMAGE_PROFILE:
    witness['repeat_captured'] = _comparison(state.captured_scores,
                                             repeated.captured_scores)
```

Retain second ordinary↔captured fidelity as an additional check, not a replacement for the cross-run captured comparison. Add a causal regression with independently perturbed capture score dictionaries: spy on `_comparison` operands and assert that original captured and repeated captured are compared for the new profile. Use distinct NLL values below the tolerance so passing/failing thresholds cannot disguise a wrong pair; keep all argmax IDs fixed. The selected-token coverage and per-sample aggregation stay unchanged.

- [ ] **Step5: Run runtime GREEN once after all edits, inspect output, self-review and commit.** Include actual single/multi execution, causal controls, independent oracle, repeated-clean, policy restoration and existing legacy tests. Root dispatches independent spec/quality review on the recorded full task diff. Task3 and real-model execution wait for that approval.

```bash
git add src/evaluation/vision_block_runtime.py tests/evaluation/test_vision_block_runtime.py tests/evaluation/vision_block_reference.py
git commit -m "feat: replay local image masks with runtime witnesses"
```

### Task 3: Profile-bound CLI, gates and model-correct reporting

**Files:** Existing CLI module and its tests only. Reuse Task1/2 interfaces; do not rename them silently.

**Produces:** All four subcommands accept `--analysis-profile`, default DEFAULT_PROFILE. Internal `_analyze`, `_validate_witness` and `_validate_gate` gain keyword-only `analysis_profile=DEFAULT_PROFILE`. `_validate_witness` additionally takes `image_positions=None` and `image_token_contract=None` so persisted witnesses can be checked against manifest truth, not themselves. `_smoke`, the `main` run branch and `_report` use the chosen condition set. A private stdlib helper `_require_profile(document, analysis_profile)` rejects a profile or condition_order mismatch; missing fields are accepted only for legacy documents.

- [ ] **Step1: Write RED parser and persistence tests.**

```python
@pytest.mark.parametrize('command', ['prepare', 'smoke', 'run', 'summarize'])
def test_all_commands_parse_image_profile(command):
    argv = [command, '--manifest', 'data/test/manifest.json',
            '--analysis-profile', 'image-local-v1']
    if command != 'summarize':
        argv += ['--model-dir', 'checkpoints/model', '--base-model-dir',
                 'checkpoints/base', '--image-root', 'data']
    if command == 'prepare':
        argv += ['--data', 'data/TEST.jsonl', '--selection', 'results/selection.json']
    else:
        argv += ['--output-dir', 'results/test']
    if command == 'run':
        argv += ['--smoke-report', 'results/test-smoke/smoke.json']
    assert cli._parser().parse_args(argv).analysis_profile == 'image-local-v1'

def test_profile_binding_never_upgrades_legacy_document():
    cli._require_profile({}, 'block-local-v2')
    with pytest.raises(ValueError):
        cli._require_profile({}, 'image-local-v1')
    good = dict(analysis_profile='image-local-v1',
                condition_order=['clean', 'image_mask', 'image_body_mask'])
    cli._require_profile(good, 'image-local-v1')
    with pytest.raises(ValueError):
        cli._require_profile(dict(good, condition_order=['clean']), 'image-local-v1')
    with pytest.raises(ValueError):
        cli._require_profile(good, 'block-local-v2')
```

Extend existing cohort mocks with profile keyword support rather than weakening legacy assertions. Image fixtures must carry nonempty validated positions and contract; generate all3 condition rows and actual expected visibility records for their sites. Mutation tests must reject absent/duplicated visibility records, wrong first query e+1, wrong blocked positions/current Body, missing layer observations, changed embeddings, false policy, or mismatched stored contract. Test smoke from legacy/model/manifest/profile cannot authorize image run even if its status says passed.

- [ ] **Step2: Add actual tiny-model integration before implementation.** Extend existing real tiny `_analyze` tests (no runtime mocking) using Task2's single/multi fixtures and new profile. Check 3×target_count rows, profile/positions/actual per-layer visibility coverage, repeated prefix witnesses and zero clean argmax differences. Add a real loader-boundary regression that retains the loader-mutated TF32 test. Add an offline subprocess `summarize` test with a shadow `torch.py` that raises on import; feed a completed three-condition manifest/status/token fixture, require correct profile/model label and all five metrics. Keep legacy no-profile fixture summarize passing.

- [ ] **Step3: Run CLI RED and implement profile plumbing.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_vision_block_analysis.py
```

The stdlib binding helper is deliberately small:

```python
def _require_profile(document, analysis_profile):
    wanted = conditions_for_profile(analysis_profile)
    stored = document.get('analysis_profile', DEFAULT_PROFILE)
    if stored != analysis_profile:
        raise ValueError('analysis profile mismatch')
    order = document.get('condition_order')
    if order is None and analysis_profile == DEFAULT_PROFILE:
        return
    if order != list(wanted):
        raise ValueError('analysis condition order mismatch')
```

Call binding before any model work on an existing manifest and before consuming smoke/status/identity. For new records persist profile/order in manifest, identity, smoke, status and summary; per-token condition strings stay explicit. Pass analysis_profile through every `summarize`, capture/replay/analyze and validation call. Use named metric columns in CSV output, not `metrics.values()` if metadata changes are involved.

Before the first ordinary clean rollout in both smoke and run, resolve the actual loaded Qwen config contract and compare it with the prepared processor contract in the manifest; reject disagreement before `_analyze`. Also check runtime `image_token_contract` against manifest and every sample witness, and exact original image positions against that sample's manifest record. For each site validate visibility records against positions ∪ current Body with query_start=e/query_stop=last_target; require3 conditions×sites and exact layer count, not record count alone. Repeated clean prefix validation remains separately required. Never accept legacy visibility markers as image-profile evidence.

Model-correct report metadata comes from the actual execution identity, not the CLI description or a guessed beta. Use a generic vision-analysis title, resolved model path/profile, the unchanged five-column formula, and image-profile wording that direct original-image access is blocked while clean history still carries indirect information. Offline reporting remains stdlib-only and supports legacy files without identity metadata with an explicit unavailable label rather than inventing a model.

- [ ] **Step4: Run CLI GREEN plus the smallest critical integration set once.**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_vision_block_analysis.py tests/evaluation/test_vision_block_runtime.py tests/evaluation/test_vision_block_inputs.py tests/evaluation/test_vision_block_metrics.py
```

This single command supplies amended CLI and four-module integration evidence; do not also repeat the same green pair/suite before and after committing. Inspect complete output, self-review and commit. Root obtains Task3 independent review, then one whole-change final review from this plan's product base; exact prior unchanged reviews can be reused, but new diff claims need independent review. Fixes belong to a fixer and receive scoped re-review; retain any unresolved findings.

```bash
git add src/evaluation/vision_block_analysis.py tests/evaluation/test_vision_block_analysis.py
git commit -m "feat: bind image analysis CLI to profile and smoke evidence"
```

## Controller execution after all implementation/review gates

The following are approved-scope execution instructions after plan approval and successful development gates; they are not authorizations to bypass those gates.

Prior b05 comparison inputs (relative to the worktree):

- Manifest: `data/ce-vision-block-analysis/b05-block-main100-v2-20260922-044315/manifest.json`.
- Completed results: `results/b05-block-main100-v2-20260922-044315/main100/` (including `summary.json` and `controller-validation.json`).
- Design/approval evidence: `results/image-block-analysis-design-20260922/`.
- New durable implementation evidence: `results/image-block-analysis-development-20260922/workflow-evidence/`; record task and review commands/results with exact revisions here before removing any scratch records.
- At approved SDD setup, run `bash /root/.codex/plugins/cache/openai-curated-remote/superpowers/6.4.1/skills/subagent-driven-development/scripts/sdd-workspace docs/superpowers/plans/2026-09-22-image-current-body-local-analysis.md`. Its printed directory owns `progress.md`, briefs and review packages. Expected uncollided location is `.superpowers/sdd/2026-09-22-image-current-body-local-analysis/progress.md`; trust the resolver if a name collision requires another path. First ledger line must identify this exact plan. Do not read or replace another plan's ledger.

- [ ] Verify reviewed HEAD and code files, source/model/selection identity, symlinks, and one available GPU. Preserve the previous b05 results as comparison evidence. Choose fresh run ID `image-body-main100-YYYYMMDD-HHMMSS`. Record actual authorization, task/reviewer IDs and commands in the plan's SDD ledger and durable `results/` evidence.
- [ ] Prepare using the unchanged model/base/selection below. Compare every source/site/target/input hash with the prior b05 manifest; allow only profile/image-position metadata additions. Require100/S327/N12843/exclusions0 and one original image per sample.

```bash
export PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD"
image_run_id="image-body-main100-$(date -u +%Y%m%d-%H%M%S)"
image_model='checkpoints/evaluation-staging/voff-immediate-mask-b05-fixed-20260919-021026/prepared-checkpoint-212'
image_base='checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct'
image_selection='/root/workspace/ILVR/results/server-comparison-20260912/analysis_completed/visual-causal-panel-20260910t115000z/ce/no_latent_execution/metadata.json'
image_manifest="data/ce-vision-block-analysis/$image_run_id/manifest.json"
mkdir -p "logs/$image_run_id"
./venv/ilvr-venv/bin/python scripts/analyze_vision_block.py prepare --analysis-profile image-local-v1 --model-dir "$image_model" --base-model-dir "$image_base" --image-root data --data data/TEST.jsonl --selection "$image_selection" --manifest "$image_manifest" > "logs/$image_run_id/prepare-console.log" 2>&1
```

Persist these resolved values with the run evidence; restore them from that record if the shell session changes.

- [ ] Check prepare exit status before launching actual smoke. Execute the detached snippet below with `IMAGE_ANALYSIS_PHASE=smoke` first, then inspect its completed output before using `IMAGE_ANALYSIS_PHASE=run`. Expose exactly one GPU verified free; the example sets GPU0 only if that preflight passes. Read status, identity, case selection, actual masks and1,512+504 prefix comparisons. New first-nail runtime evidence must be green before main100. No tolerance relaxation or sample substitution.
- [ ] Launch main `run` with the same arguments plus `--smoke-report results/$image_run_id/smoke/smoke.json` and `--output-dir results/$image_run_id/main100`. Use the already-proven stdlib detached process approach; no shell nohup lifecycle assumption:

```bash
# Use these exact recorded values from prepare. GPU0 must have passed preflight.
export IMAGE_ANALYSIS_RUN_ID="$image_run_id"
export IMAGE_ANALYSIS_MODEL="$image_model"
export IMAGE_ANALYSIS_BASE="$image_base"
export CUDA_VISIBLE_DEVICES=0
export IMAGE_ANALYSIS_PHASE=smoke
./venv/ilvr-venv/bin/python - <<'PY'
import os, subprocess
from pathlib import Path
root = Path.cwd()
rid = os.environ['IMAGE_ANALYSIS_RUN_ID']
phase = os.environ['IMAGE_ANALYSIS_PHASE']
assert phase in ('smoke', 'run')
output = 'smoke' if phase == 'smoke' else 'main100'
argv = [str(root/'venv/ilvr-venv/bin/python'), 'scripts/analyze_vision_block.py',
    phase, '--analysis-profile', 'image-local-v1',
    '--model-dir', os.environ['IMAGE_ANALYSIS_MODEL'],
    '--base-model-dir', os.environ['IMAGE_ANALYSIS_BASE'], '--image-root', 'data',
    '--manifest', f'data/ce-vision-block-analysis/{rid}/manifest.json',
    '--output-dir', f'results/{rid}/{output}', '--log-dir', f'logs/{rid}/{output}']
if phase == 'run':
    argv += ['--smoke-report', f'results/{rid}/smoke/smoke.json']
with (root/'logs'/rid/f'{output}-console.log').open('xb') as log:
    process = subprocess.Popen(argv, cwd=root, env=os.environ.copy(),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
(root/'logs'/rid/f'{output}.pid').write_text(str(process.pid)+'\n')
print(process.pid)
PY
```

After smoke passes all gates, set `IMAGE_ANALYSIS_PHASE=run` and execute the same Python snippet without resetting that variable to smoke. Record environment/argv, observe the owned PID and progress without signaling foreign processes. A completed smaller cohort is not a successful substitute.

- [ ] After completed status, run offline summarize with the **actual recorded paths**:

```bash
./venv/ilvr-venv/bin/python scripts/analyze_vision_block.py summarize --analysis-profile image-local-v1 --manifest "$image_manifest" --output-dir "results/$image_run_id/main100" --log-dir "logs/$image_run_id/main100"
```

Audit38,529 unique rows, target IDs,100sources327sites12843positions, three conditions, finite NLL, profile/model/input/code identities,70,632 exact prefix comparisons and per-layer visibility records. Independently recompute all five metrics from clean argmax/NLL maps and raw rows. Report the table and raw artifact links immediately, noting preserved indirect image/history routes.
- [ ] Before branch completion apply verification-before-completion and finishing-a-development-branch together when applicable. Run the project's full suite once against the reviewed revision:

```bash
set -o pipefail
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" ./venv/ilvr-venv/bin/python -m pytest -q 2>&1 | tee "logs/$image_run_id/full-suite.log"
```

Read complete output/exit status, retain skipped-test limits, and do not present the branch menu before all required witnesses pass. Preserve approvals/reviews/commands/results/diff identities and every root Ruling before deleting only this plan's SDD scratch workspace. Human chooses merge/PR/keep; previous branch disposition is still pending and does not authorize automatic integration.

## Plan handoff status

Design v1 is approved; this plan is not yet approved. No product change or new model execution is authorized by merely writing this file. Controller self-review checks specification coverage, exact interface consistency, concrete test content and the five Review Focus classes. Root self-review corrected a stale reference-helper filename, executable command gaps and the repeated-captured comparison requirement. `/root/image_local_plan_coldread` read the complete standalone plan and reported minor gaps: prior comparison paths, metric presentation, witness arithmetic, pre-inference contract comparison and evidence locations. These are now explicit. This handoff read does not replace implementation reviews or runtime witnesses. After user plan review, use the preserved sub-agent method and execute all approved tasks/gates without per-task permission questions.

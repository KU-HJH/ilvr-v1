# Stage1-v5 History Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기본 legacy selector를 보존하면서 Stage1-v5에 `--query_mode adaptive`를 추가한다.

**Architecture:** 기존 target builder가 mode dispatch와 candidate 선택을 소유한다. 별도 query 수식/config, effective DoRA prefix adapter와 reduced-attention observer를 연결해 이전 target이 주입된 teacher prefix에서 현재 query를 만든다. 기존 v5 loss/reference/공통 target snapshot과 저장·재개 경계를 유지한다.

**Tech Stack:** Repository Python3.11 venv, custom Transformers/Qwen2.5-VL, PyTorch2.7.1, PEFT DoRA, pytest, 기존 Bash/multinode coordinator. 새 dependency는 없다.

**Spec:** `docs/superpowers/specs/2026-10-15-stage1-v5-history-query-design.md`, commitment v1, commit `9c402ff2daa49c1e293b4eb0829f3402389f2839`, SHA256 `9ad3acbf64a4fc17ea25332591dddbb351a47c3dc02f6bf2ecdac0f7e69aae40`. 사용자는 `불확실성 수용·명세 승인`으로 승인했다. 이 계획의 검토/구현 방법 선택은 아직 대기 중이다.

## Global Constraints

- 제품 코드 기준 `148b0e8ead14456d02cc595f0cc32a0752ee22e8`; worktree `/root/.codex/worktrees/1061/ILVR`, branch `stage1_v5`를 유지한다.
- CLI는 `--query_mode legacy|adaptive`, default legacy; `--history_text_gamma 0.9`; `--history_attention_layers last|last4|all`, default last다. `--history_query_mode` alias는 없다.
- Adaptive는 Stage1-v5/Qwen2.5-VL/DoRA/batch1만 지원한다. Frozen teacher tau1, decoder SDPA, vision FlashAttention2, BF16, 기존 checkpointing/ZeRO1을 보존한다.
- Adaptive text는 discounted segments, latent는 attention mass다. Query/key/head chunk64/2048/4는 내부 설정으로 기록한다. 추가 ablation, EMA runtime, generation 변경과 새 launcher는 없다.
- Candidate pooling/top-p/top-k/stable order/variable Kstar, packed `[N,H]` target과 `[B,T]` mask, alignment t 대 t-1, v5 loss/optimizer/reference와 student RNG를 보존한다.
- Prefix는 현재 START 직전까지만 포함한다. 과거 active pad에 선택한 raw target만 주입하며 inactive pad는 유지한다. Helper는 candidate extractor에만 사용한다.
- Effective DoRA projection, 실제 MRoPE/GQA/causal mask를 따른다. 확률은 전체 valid causal key 분모로 계산하고 mass를 재정규화하지 않는다. Teacher no-grad, hook/cache는 예외에서도 제거한다.
- Source/reference/resume 안전 검증을 완화하지 않는다. Query 설정 없는 과거 checkpoint는 legacy로만 해석하고 adaptive 전환 resume은 거부한다. Objective 변경은 fresh run이다.
- 모든 Python/test는 `./venv/ilvr-venv/bin/python`; 외부 checkout/Reference를 runtime dependency로 사용하지 않는다. Model artifact는 checkpoints, 결과는 results, 실행 로그는 logs에 둔다.
- Training smoke도 기존 coordinator의 2 nodes×2 GPUs/four ranks만 허용한다. 이 계획은 training/GPU premise probe/평가를 실행하는 승인이 아니다.
- 구현 방법은 사용자가 선택한다. SDD는 task별 implementer/fixer와 독립 리뷰, Native는 root 구현과 독립 최종 리뷰 등 선택한 skill의 gates를 따른다. Paperthin은 worker/reviewer가 호출하지 않는다.

## Review Focus

1. Multi-token assistant header, special/image token과 빈 text가 reasoning 평균에 섞이지 않는가 — Task1의 layout/empty 테스트로 고정한다.
2. Nonzero DoRA·image MRoPE·GQA에서 adapter 또는 observer가 plain-model 근사를 읽지 않는가 — Task2/3의 실제 native 비교로 고정한다.
3. Kstar<K, multiple-helper mapping과 이전 target 교체에서 위치/주입/causality가 맞는가 — Task4의 입력 capture와 baseline 비교로 고정한다.
4. 자동 생성된 `--textual_states previous`와 sourced stale 값이 adaptive를 막거나 resume 기본값이 저장된 query를 덮어쓰지 않는가 — Task5의 실제 shell→양 node argv→parser 테스트로 고정한다.
5. 마지막 축적 window, empty auxiliary block과 adaptive helper 재실행이 v5 loss 분모/reference/RNG/호출 관측에 섞이지 않는가 — Task6의 실제 parent Trainer 통합으로 고정한다.

---

## File structure and interfaces

Create `src/stage1/query.py`, `history_teacher.py`, `history_attention.py`와
`tests/stage1/history_fixtures.py`, `test_history_query.py`,
`test_history_teacher.py`, `test_history_attention.py`, `test_history_targets.py`,
`v5/test_history_integration.py`. 기존 CLI/composition/trainer/v5 resume/launcher
파일을 좁게 수정한다. Existing metadata/dataclass를 새 병렬 시스템으로 대체하지 않는다.

Task1의 config/math를 Task2/3이 사용하고, Task4가 세 모듈을 builder에 연결한다.
Task5는 공개 설정·저장·전달을 연결하며 Task6는 전체 경로를 검증/문서화한다.
아래에 정의한 이름/타입을 임의로 바꾸지 말고 변경이 필요하면 소비 task도 함께 조정한다.

### Task 1: Query config, text layout and mass-preserving math

**Files:** Create `src/stage1/query.py`, `tests/stage1/test_history_query.py`.

**Interfaces:**

- Produces frozen `Stage1QueryConfig(mode='legacy', text_gamma=.9, attention_layers='last', query_chunk_size=64, key_chunk_size=2048, head_chunk_size=4)`.
- Produces `resolve_query_config(explicit: Mapping[str, object], *, saved: Mapping[str, object] | None = None) -> Stage1QueryConfig`; explicit keys are the three public CLI destinations. A missing saved query config is handled as historical legacy by Task5, not guessed as adaptive here.
- Produces `resolve_query_layers(selector: str, layer_count: int) -> tuple[int, ...]` and `query_execution_config(config: Stage1QueryConfig, resolved_layers: tuple[int, ...]) -> dict[str, object]`.
- Produces frozen `DiscountedTextState(total: Tensor | None = None, weight: float = 0.)`, `HistoryQueryResult(query: Tensor, latent_mass: Tensor, text_present: bool)`.
- Produces `history_text_positions(ids: Tensor, attention_mask: Tensor, spans: tuple[Stage1LatentSpan, ...], assistant_content_start: int, special_ids: frozenset[int]) -> tuple[tuple[int, ...], ...]`.
- Produces `update_text_history(state: DiscountedTextState, current_mean: Tensor | None, *, gamma: float) -> tuple[DiscountedTextState, Tensor | None]`, `aggregate_history_latents(hidden: Tensor, scores: Tensor) -> tuple[Tensor, Tensor]`, and `compose_history_query(global_intent: Tensor, text_summary: Tensor | None, latent_sum: Tensor, latent_mass: Tensor) -> HistoryQueryResult`.
- Layout ids/mask는 sample row `[S]`이고 span은 기존 collator 좌표다. Mean/query/latent_sum은 `[1,H]`, latent hidden은 `[M,H]`, scores는 `[M]`, mass는 scalar다. 모든 aggregate 결과는 FP32다.

- [ ] **Step1: Write RED math/config/layout tests.** Test public CLI-field mapping without an old-name alias, invalid/nonfinite gamma, layer resolution, absent text, gamma0/.9/1, unequal segment lengths, empty-step aging and excluded special/header/padding positions. Include these numerical assertions:

```python
state, _ = update_text_history(DiscountedTextState(), torch.tensor([[1., 0.]]), gamma=.9)
_, summary = update_text_history(state, torch.tensor([[0., 1.]]), gamma=.9)
torch.testing.assert_close(summary, torch.tensor([[.9 / 1.9, 1. / 1.9]]))
b, mass = aggregate_history_latents(torch.eye(2), torch.tensor([.001, .002]))
torch.testing.assert_close(b, torch.tensor([[.001, .002]]))
torch.testing.assert_close(mass, torch.tensor(.003))
```

- [ ] **Step2: Observe RED.** Run `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_history_query.py`. Missing APIs/required behavior must cause failure; unrelated dependency/import/environment failures are not the intended RED.
- [ ] **Step3: Implement the interfaces in query.py.** FP32 math; q=(u+present*d+b)/(1+present+mass), omit absent d, fail nonfinite or norm<=1e-8. Use explicit errors for invalid layout/config. `query_execution_config` returns schema_version1 and formula_id: legacy `stage1_legacy_query_v1`; adaptive `stage1_injected_prefix_discounted_segments_attention_mass_v1`. Adaptive records text_mode/latent_mode, gamma, layer selector/resolved indices, chunks and `mask_mode='full_causal_2d_v1'`; legacy records only schema/mode/formula.
- [ ] **Step4: Observe GREEN.** Run Step2's command. All assertions must pass; gamma1 must remain a segment mean, and small latent mass must remain small.
- [ ] **Step5: Commit/review the task under the selected execution skill.** Stage only the two named files; commit `feat: define Stage1 history query math and config`. Record RED/GREEN and commit identity in the workflow ledger.

### Task 2: Effective DoRA teacher prefix adapter

**Files:** Create `src/stage1/history_teacher.py`, `tests/stage1/history_fixtures.py`, `tests/stage1/test_history_teacher.py`.

**Interfaces:**

- Consumes existing teacher, sample tensors/metadata and Task1 layout/config.
- Produces frozen `HistorySelection(block_index: int, positions: tuple[int, ...], values: Tensor, candidate_indices: tuple[int, ...])`, `PreparedHistoryPrefix(input_ids: Tensor, inputs_embeds: Tensor, attention_mask: Tensor, position_ids: Tensor)` and `HistoryPrefixObservation(last_hidden_state: Tensor)`.
- Produces `HistoryPrefixAdapter(teacher, *, input_ids: Tensor, attention_mask: Tensor, original_pixels: Tensor, original_grid: Tensor)` with `.backbone`, `.prepare(stop: int, selections: tuple[HistorySelection, ...]) -> PreparedHistoryPrefix` and `.forward(prefix: PreparedHistoryPrefix) -> HistoryPrefixObservation`.
- Produces test `HistoryCase(teacher, tokenizer, inputs, metadata)` and `make_history_case(*, layers: int = 1, dtype=torch.float32, attention_backend: str = 'sdpa', nonzero_dora: bool = True) -> HistoryCase`.

- [ ] **Step1: Write RED native-prefix tests.** Reuse `tiny_original_qwen`/`stage1_tuning` conventions from `tests/stage1/test_checkpointing.py`; build random tiny weights with actual nonzero DoRA, valid original image grid, GQA and at least two latent blocks. Assert effective projection differs from its plain base. Capture native wrapper inputs and compare adapter embeddings/positions/hidden for past injected targets. Assert original/helper separation, no LM-head call, input immutability, frozen teacher/no-grad and unmodified student RNG. Parameterize no prior target, partial Kstar and multiple original images.
- [ ] **Step2: Observe RED.** Run `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_history_teacher.py`.
- [ ] **Step3: Implement the adapter and fixture.** Retain effective PEFT modules through `resolve_qwen2_5_vl_model`; encode original images once per adapter, validate token/feature counts, clone prefix embeddings and scatter only earlier selections. Compute original-grid prefix MRoPE through the existing model API without mutating rope/cache state. Call bare backbone with the spec's no-cache/no-all-hidden/no-attention kwargs. Reject selections outside the prefix, future-image placement and unsupported mask/trajectory rather than silently realigning them.
- [ ] **Step4: Observe GREEN.** Repeat Step2. Native FP32 comparisons use atol1e-6/rtol1e-5. This proves only the fixture path; it does not establish production throughput or multi-node support.
- [ ] **Step5: Commit/review.** Commit the three named files as `feat: prepare injected Stage1 teacher prefixes`; preserve observed native comparison output in logs/results and ledger.

### Task 3: Reduced native-attention observer and the numerical first nail

**Files:** Create `src/stage1/history_attention.py`, `tests/stage1/test_history_attention.py`; extend `tests/stage1/history_fixtures.py` only for required layer configurations.

**Interfaces:**

- Consumes Task1 config/layer IDs, effective language backbone and prepared prefix mask; independent of candidate selection.
- Produces context manager `ReducedHistoryAttentionObserver(backbone, *, layer_indices: tuple[int, ...], query_positions: tuple[int, ...], latent_positions: tuple[int, ...], attention_mask: Tensor, config: Stage1QueryConfig)` and `.scores() -> Tensor` of FP32 shape `[len(latent_positions)]`.

- [ ] **Step1: Write RED attention tests.** Compare native eager probability rows on the tiny nonzero-DoRA/GQA/image-MRoPE prefix with observer scores. Cover last/last4/all, causal/padding conditions, unequal query/key/head chunks, zero-query/no-latent cases, single capture per layer, observer noninterference and cleanup after a forced forward exception. Dense equal-logit fixture with four valid causal keys/two latent keys must give scores [.25,.25], mass .5. Missing/duplicate captures must error, not return uniform weights.
- [ ] **Step2: Observe RED.** Run `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_history_attention.py`.
- [ ] **Step3: Implement the observer.** Layer pre-hook reads actual position embeddings; q_proj/k_proj hooks retain effective outputs. Apply native MRoPE/GQA mapping and scaled causal logits; use chunked streaming full-key log-sum-exp and average probabilities across queries/heads/layers. Keep O(prefix hidden) projections for only the active layer, never full S×S probability storage. Do not change attention backend or use the old global-intent approximate helper. Empty query/latent sets install no hooks and return zeros/empty scores as appropriate.
- [ ] **Step4: Observe GREEN and decide the numerical gate.** Run Step2 plus Task2 tests together once. Native FP32 attention/prefix comparisons must pass their exact tolerances before Task4 integration. Failure requires a targeted correction through the selected method; do not mask it with looser tolerances or uniform pooling. This is ordinary approved implementation TDD after plan approval, not authorization to run a separate premise experiment now. BF16/GPU support stays separately unverified.
- [ ] **Step5: Commit/review.** Commit as `feat: observe reduced causal history attention`. Record native comparison, hook counts and residual unsupported cases.

### Task 4: Sequential adaptive target construction with legacy parity

**Files:** Modify `src/stage1/targets.py::Stage1TeacherTargetBuilder`; create `tests/stage1/test_history_targets.py`; extend `tests/stage1/history_fixtures.py` with a read-only baseline loader.

**Interfaces:**

- Consumes Tasks1–3 and existing `Stage1SampleMetadata`, `_user_query`, helper candidates and `_select`.
- Extends builder constructor with `query_config: Stage1QueryConfig | None = None`; absent means legacy. Existing `build(teacher, batch, metadata) -> Stage1TeacherTargets` is unchanged.
- `_build_sample_legacy`, `_build_sample_adaptive`와 dispatch `_build_sample`은 기존 keyword-only `teacher, sample, sample_ids, sample_attention, original_pixels, original_grid, helper_pixels, helper_grid, image_token_id` 인자를 그대로 받는다. 반환은 `(packed: list[Tensor], block_counts: tuple[int, ...], block_indices: tuple[tuple[int, ...], ...], block_margins: tuple[float | str, ...], empty_values: Tensor)`다.
- Produces `load_baseline_target_builder(base_commit: str) -> type` in test fixtures using inspected `git show <base>:src/stage1/targets.py` in an isolated test namespace, never a production dependency on another checkout.

- [ ] **Step1: Write RED comparative/causality tests.** Run old/new builders with cloned identical teachers/inputs/RNG for both textual_states values and compare target rows, mask, candidate indices and language/vision calls. Pin the baseline to the product-base commit above. For adaptive capture every actual prefix embedding; test first/partial-Kstar/multiple-helper/empty-text/no-latent cases and original source order. Change only previous selected targets to distinguish query dependence from teacher-state dependence; require downstream change in a known synthetic dependency fixture, not a large change on arbitrary real samples. Mutate future text/latent/helper and assert unchanged current query/selection; mutate current helper and assert unchanged query. Test malformed markers/missing candidates/mask mismatch as errors.
- [ ] **Step2: Observe RED.** Run `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_history_targets.py tests/stage1/test_targets.py`.
- [ ] **Step3: Implement dispatch and adaptive construction.** Keep legacy's old body/one teacher call. Adaptive validates single trajectory/header/special positions, reuses existing helper mapping/pooling, builds prompt-only u with the existing global helper outside observer scope, then selects each block using Task1 math and injected-prefix observations. Use a build-local text state/selection ledger. Convert to the existing packed `[N,H]` result/mask once; keep selected order and current diagnostic controls. Reject invariants even when v5 debug is off. No helper images in language context and no CE-only fallback for malformed latent samples.
- [ ] **Step4: Observe GREEN.** Repeat Step2. Compare actual helper call counts: legacy1; adaptive prompt1+M prefixes when M>0, with empty-case behavior separately recorded. Read target/position/RNG comparisons, not just test names.
- [ ] **Step5: Commit/review.** Commit as `feat: add adaptive Stage1 history target selection`. Review the old-body diff and runtime baseline comparisons before downstream wiring.

### Task 5: Public CLI, current-v5 contracts/resume and existing launch path

**Files:** Modify `src/utils_deepseed.py::parse_args/_resolve_stage1_v5_args`, `src/stage1/composition.py::build_stage1_training_components`, `src/stage1/trainer.py::__init__`, `src/stage1/v5_trainer.py` contract collection, `src/stage1/v5_resume.py`, `src/multinode_runtime.py::_STAGE1_VALUE_OPTIONS`, `run_stage1_training.sh`, `scripts/stage1_v5_train_eval.sh`; inspect `src/multinode.py` and change only if the real forwarded path requires it. Extend `tests/stage1/test_cli.py`, `test_composition.py`, `test_launcher.py`, `v5/test_resume.py`, `v5/test_launcher.py`, `v5/test_workflow.py`.

**Interfaces:**

- Consumes Task1 config/serialization and Task4 constructor. Parser exposes `args.stage1_query_config` and `args.stage1_query_explicit_values` separately from the existing v5 loss explicit values.
- Produces `resolve_v5_query_resume(saved_contract: Mapping[str, object], explicit_query: Mapping[str, object]) -> Stage1QueryConfig` in `v5_resume.py`, using strict Task1 semantics. Restores omitted mode/gamma/layers before fresh-only conflict/default validation.
- Public coordinator/worker/recipe spelling stays exactly `--query_mode`, `--history_text_gamma`, `--history_attention_layers`; no additional naming alias or environment API.

- [ ] **Step1: Write RED wiring/resume/real-shell tests.** Default legacy and adaptive .9/last; reject unknown mode/gamma/layer and unsupported adaptive Stage2/standard/full/batch profile before loading weights. Reject explicit adaptive+textual_states and legacy+history controls, including accepted argparse abbreviations/equal-value syntax through action.dest tracking. Save nested execution.query_config with independent query formula ID; inherit omitted settings, reject missing/tampered adaptive settings/formula/reference, preserve loss formula and source checks. Query-less historical contracts resolve only legacy and cannot be relabeled adaptive. Exercise actual coordinator→both mapped worker dry-runs→Python parser, including an adaptive resume with omitted query flags, `.55` saved gamma and `last4` saved layers. Assert topology still four ranks/two nodes and masks/paths are unchanged.
- [ ] **Step2: Observe RED.** Run `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_cli.py tests/stage1/test_composition.py tests/stage1/test_launcher.py tests/stage1/v5/test_resume.py tests/stage1/v5/test_launcher.py tests/stage1/v5/test_workflow.py`.
- [ ] **Step3: Implement query config plumbing/save/resume.** Pass config composition→trainer→builder; resolve layers from the effective model topology. Extend existing execution contract/source hashes with the new query modules; do not change loss semantics or create another EMA checkpoint system. Separate query fields from v5 loss resolution and preserve source/reference authentication. Normalize absent historical query metadata only to legacy; keep any existing incompatible-source rejection explicit. Restore saved query before diagnosing explicit textual/history conflicts.
- [ ] **Step4: Implement existing shell/coordinator forwarding.** Add only the three exact flags to the allowlist and shells. For fresh adaptive, omit the worker's generated textual_states fallback; preserve an explicit caller textual_states request so Python rejects it. Resume must omit query/history defaults unless requested. The fresh v5 recipe accepts the new flags, enforces their resolved values after settings sourcing, clears stale STAGE1_TEXTUAL_STATES for adaptive, and omits its current hardcoded textual_states argument in that mode. Default legacy recipe retains its current behavior. Add a sourced-stale-settings regression for this distinction; do not introduce a second launcher.
- [ ] **Step5: Observe GREEN and commit/review.** Repeat Step2, then run `bash -n run_stage1_training.sh scripts/stage1_v5_train_eval.sh run_multinode_stage1_training.sh`. Save actual emitted both-node argv evidence; no SSH/training is performed. Commit `feat: wire history query options and resume semantics`.

### Task 6: Actual v5 integration, operator documentation and completion gates

**Files:** Create `tests/stage1/v5/test_history_integration.py`; extend existing tiny v5 fixtures only as required. Modify `README.md` in the existing Stage1-v5 operator section, not unrelated evaluation docs. Product fixes follow their owning task/method.

**Interfaces:**

- Consumes the real parser/composition/builder/adapter/observer and `CustomTrainerStage1V5.training_step`, existing CPU tiny-Qwen/DoRA parent-Trainer fixtures from `v5/test_trainer.py`.
- Produces runtime witnesses for selected shared snapshot, gradients, immutable reference, query resume, RNG, actual call counts and unchanged loss reductions. No new orchestration or training harness.

- [ ] **Step1: Write RED integration witnesses.** Run one actual tiny CPU parent-Trainer step through adaptive preparation with two latent blocks; capture F/reference/packed-OFF target tensors and verify a shared snapshot, teacher no-grad/student trainable grads and unchanged frozen reference checksum. Cover empty auxiliary blocks and a final partial accumulation window against the existing loss denominator oracle. Check legacy call path unchanged and adaptive M+4 decoder calls only when all three downstream contexts exist. Include repeated builds proving no cross-sample ledger/text state or residual hooks and save/resume matching next target.
- [ ] **Step2: Observe RED.** Run `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/v5/test_history_integration.py`. If it immediately passes, demonstrate sensitivity by a scoped causal perturbation/removal of the relevant new path, then restore it; do not invent an implementation change just to create RED.
- [ ] **Step3: Resolve only causal integration defects.** Use the selected method's fixer/review flow. Keep loss definitions, reference lifecycle, parent dispatch and accumulation denominator unchanged; a required material design change returns to human approval rather than being slipped into a fix.
- [ ] **Step4: Observe GREEN and document actual scope.** Repeat Step2. README includes default legacy, an adaptive coordinator dry-run example using query_mode/gamma/layers, required DoRA/v5 profile, incompatible explicit textual_states, fresh-run/resume distinction, extra prompt+M helper cost and query identities. Describe CPU results separately from unrun BF16/GPU/four-rank/long-input/benchmark checks. Commit `test: verify Stage1-v5 history query integration` with the owned tests/docs/fixes.
- [ ] **Step5: Complete selected-method independent review and fresh verification.** Supply the approved spec, this plan, base-to-head diff, commit identities and focused witnesses to an independent reviewer distinct from implementer. Follow required task/final-fix/re-review gates; record unresolved findings as unresolved. Do not substitute these design critiques or self-review for code review.

## Final verification and execution handoff

Before choosing the final test command, determine whether branch finishing
applies next. If it does, invoke verification-before-completion and announce
finishing; use the same project full-suite result once for their overlapping
test-pass claim. Otherwise verify the exact implementation claim. The project
full-suite command is:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q
```

Run once against the final reviewed code, choose an idle visible local GPU
when its unit tests require one, and read complete output. Do not claim a
skipped/missing GPU or distributed witness passed. Re-run only distinct shell
syntax/dry-run/content checks needed for claims not covered by that suite.
Keep full execution logs under `logs/stage1-v5-history-query-implementation/`
and result metadata under the matching `results/` subdirectory.

This plan does not authorize production training or a training smoke. Once
implementation is concrete/reviewed, any requested real training check needs
the existing coordinator, a fresh run ID and explicit execution approval.
Prepare the exact commands and minimum preflight first. Re-read canonical
AGENTS/training-defaults/multinode guide, resolve the full current-v5 profile
on both nodes, derive update count from the actual retained sampler, and
verify both nodes/ranks0–3 at runtime before reporting multi-node support.
No single-node torchrun fallback is permitted. Without that evidence, report
CPU/reference implementation coverage and explicitly unverified runtime scope.

The user owns merge/PR/keep-as-is and cleanup decisions. Preserve the requested
worktree. Record approvals, selected skills/method, reviewer/worker IDs,
changes/commit hashes, review verdicts/fixes and exact verification commands
and results in the existing workflow record; no separate recovery framework.

## Plan self-review and next action

Spec coverage: §3/4→Tasks1/5; §5→Tasks1/2/4; §6→Task1;
§7→Tasks2/3; §8→Tasks4/6; §9→Task5; §10/11→Tasks1–6 and
the separately authorized runtime gate; §12/13 uncertainty is preserved.
Each Review Focus entry has a named owner/test. All cross-task public names
are defined above; the approved spec's source/shape/teacher adaptations remain.
No new durable subsystem or speculative retry/provenance infrastructure.
These checkboxes are future actions, not implementation/test completion claims.

Root read the complete plan against the approved spec. Self-review clarified
row/tensor shapes, preserved the existing sample-builder signature/return,
removed ambiguous alias wording, and added explicit-option abbreviation and
generated-fallback tests. Planned commands have not been executed; no product
implementation or runtime result is implied by this review.

Plan review and execution-method selection remain required. Recommend **Native**
for this plan because the six tasks share small, tightly coupled interfaces;
root can implement them sequentially with one context and obtain the required
independent whole-change review. **Subagent-driven** remains available for
separate implementer/reviewer gates per task. Do not choose on the user's behalf.

# Stage 1 Vision Gain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 Stage 1에 전역 vision-gain gradient projection과 별도 parameter displacement를 추가하고 DoRA + ZeRO-1/2에서 검증한다.

**Architecture:** teacher target을 한 번 만든 뒤 auxiliary clean, 블록별 masked, base forward/backward를 순차 실행한다. 별도 score-gradient buffer를 update 단위로 전역 집계하여 projection하고, base AdamW step 이후 FP32 master와 모델 weight에 보조 이동을 적용한 다음 EMA를 갱신한다. 최초 runtime probe의 증거가 있어야 이후 product task를 확정·실행한다.

**Tech Stack:** repository Python 3.11 venv; PyTorch 2.7.1+cu128; patched local Transformers; TRL 0.15.2; Accelerate 1.10.1; DeepSpeed 0.18.7; DoRA; pytest.

**Spec:** [2026-10-02-stage1-vision-gain-design.md](../specs/2026-10-02-stage1-vision-gain-design.md), 승인 기준 commit `ecfae64`, SHA256 `18ba04c084ff3f7435fbee799c8a80bc145ca5c948a79e37a7c0010e7e86becb`.

## Global Constraints

- Worktree `/root/workspace/ILVR/.worktrees/stage_1_vision_gain`, branch `stage_1_vision_gain`; product baseline `2f48c51985c395c16bea80a89847d7586728f9cd`.
- `alpha_vis` 기본값은 0이고, 활성화 실행에서는 양수 값을 명시한다. 검증된 효능 계수는 없다.
- 학습 CLI `--stage1_vision_gain_alpha`, coordinator `--vision-gain-alpha`, 환경 `STAGE1_VISION_GAIN_ALPHA`; resume의 생략은 saved 값 상속이다.
- 초기 활성화 경로는 **DoRA + ZeRO-1/2**다. BF16, decoder SDPA, 기존 vision backend와 non-reentrant checkpointing을 사용한다.
- 원본 token/CE/teacher feature·alignment·EMA/Stage 2 목적함수는 보존한다. latent-body mask는 `q >= r_b`, `l_b < k < r_b`다.
- FP32 전체 vocabulary log-softmax, target `i`는 predictor `i-1`; 바로 다음 text item만 점수화하고 text/text crossing token은 gain에서만 제외한다.
- FP32 gradient buffer, FP64 dot/norm 및 projection 중간값; `min_h_norm_sq=1e-30`; epsilon으로 denominator를 바꾸지 않는다.
- `theta_new = theta_old + Delta_S1 + alpha_vis*u`; base AdamW moments는 base gradient로만 갱신한다.
- 모든 GPU training/probe/smoke는 기존 coordinator의 **2노드·4rank**, remote rank 0 `10.1.102.41`, local rank 1, `bond1.102`를 사용한다. 직접 rank 실행은 금지다.
- fresh 실행은 microbatch 1, accumulation 4, global batch 16, LR 5e-5, seed 42, warmup 10, CE 0.1; DoRA r32/alpha64/dropout0.05, latent8 및 기존 Stage 1 설정을 명시한다. tiny probe도 이를 사용하고 기존 tiny model/data fixture의 크기만 작게 둔다.
- Python/tests는 `./venv/ilvr-venv/bin/python`, import 경로는 `PYTHONPATH=transformers/src:src:.`; CPU tests는 `CUDA_VISIBLE_DEVICES=''`를 명시한다. dependency 설치는 하지 않는다.
- 데이터는 `data/`, weights/state는 `checkpoints/`, 결과는 `results/`, 로그는 `logs/`; 기존 symlink를 유지한다. fixture도 저장 종류에 따라 배치한다.
- OOM은 실제 할당 원인에 맞춰 수정한다. block/token 축소, detach, global batch 변경, single-node fallback은 허용하지 않는다.
- 설계의 모든 tolerance를 테스트에 고정한다. float64 `rtol=1e-10, atol=1e-12`; FP32 projection `rtol=1e-5` 및 spec의 scale-aware absolute bound; chunk scoring `rtol=1e-5, atol=1e-6`; BF16 score `rtol=1e-3, atol=1e-3`, gradients `rtol=1e-2, atol=1e-5` 및 global relative L2 ≤ 1%; master delta `rtol=1e-6, atol=1e-7`.
- TDD와 독립 specification/quality review를 적용한다. 실행 방식은 사용자가 선택하며 아직 미선택이다. probe 코드를 쓴다는 이유로 lifecycle을 생략하지 않는다.
- 아래 공통 CPU test 명령은 `PYTHONPATH=transformers/src:src:. CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest`다. GREEN step의 파일 목록도 이 명령 뒤에 붙인다. 모든 commit은 해당 task의 명시된 파일만 `git add -- <files>`로 stage하고 제시한 메시지로 `git commit -m '<message>'`한다.

## Review Focus

1. 동일 text가 여러 source item에 반복되어도 문자열 첫 occurrence로 window를 잘못 붙이지 않는다 — Task 3 `test_repeated_text_and_processor_expansion_keep_item_identity`.
2. 특정 rank의 블록 수가 0이고 마지막 accumulation이 짧아도 collective 순서와 전역 denominator가 맞는다 — Task 4 `test_zero_rank_and_partial_group_match_global_reference`.
3. 이전 microbatch base gradient가 이미 쌓인 상태에서 auxiliary 미분해도 `.grad`와 ZeRO reducer state가 변하지 않는다 — Tasks 1/5 `test_auxiliary_preserves_existing_base_accumulation`.
4. BF16 replica만 바뀌어 다음 optimizer step이나 reload에서 보조 이동이 사라지지 않는다 — Tasks 1/5/7 `test_master_displacement_survives_next_step_and_resume`.
5. 활성 checkpoint resume에서 CLI 생략과 명시적 0을 혼동하지 않는다 — Tasks 2/6 `test_resume_omission_inherits_but_explicit_zero_rejects`.

---

## Execution gates and file map

이 문서는 **검토 가능한 조건부 계획**이다. 작성만으로 premise probe를 실행하거나 Task 2 이후를 시작하지 않는다. 사용자의 계획 검토·실행 방식 선택·아래 bounded probe의 별도 실행 승인을 기록한 후 Task 1을 수행한다. Task 1의 runtime 증거가 green이면 조건부 product task를 확정한다. 전제가 깨지면 영향받는 task를 재설계하여 다시 검토받는다; static 검토나 tiny 성공을 full-model 메모리 증거로 대체하지 않는다.

| 파일 | 책임 |
| --- | --- |
| `tests/stage1/vision_gain_zero_probe.py` (신규) | bounded 실제 ZeRO/DoRA 실행 전제와 rank별 witness |
| `src/multinode.py`, `src/multinode_runtime.py` | 기존 owned supervisor를 이용한 고정 probe route 및 alpha forwarding |
| `src/stage1/vision_gain_config.py` (신규) | 순수 설정·identity·resume resolution |
| `src/stage1/vision_gain_data.py` (신규) | source item에서 원래 token 좌표로 mapping |
| `src/stage1/data.py`, `types.py` | opt-in collator metadata 연결 |
| `src/stage1/vision_gain_math.py` (신규) | additive mask, GT score, accumulator, projection |
| `src/stage1/vision_gain_runtime.py` (신규) | context/RNG 수명, branch 미분, ZeRO master apply |
| `src/stage1/trainer.py` | 기존 lifecycle 복원 및 위 단위들의 연결 |
| `src/utils_deepseed.py`, `src/stage1/composition.py`, `run_stage1_training.sh` | CLI/env와 construction/save/resume 연결 |
| `tests/stage1/test_vision_gain_*.py`, `tests/multinode/test_vision_gain_*.py` (신규) | 각 구성의 contract 및 integration tests |

새 일반 orchestration, retry, archival 체계는 만들지 않는다. `checkpointing.py`의 strict PEFT schema를 확장하는 대신 이미 저장하는 Qwen `config.json`의 `stage1_vision_gain` 객체에 identity를 넣는다. 기존 artifact save/load 경로와 Stage 2 loader의 extra config 보존을 검증한다.

### Task 1: 기존 runtime 수정과 bounded ZeRO premise probe

**Files:** Modify `src/stage1/trainer.py` (`training_step`, `_SUPPORTED_DEEPSPEED_VERSION`); `src/multinode.py::_training_main`; `src/multinode_runtime.py` (owned harness builder 인접). Create `tests/stage1/vision_gain_zero_probe.py`, `tests/stage1/test_vision_gain_probe.py`, `tests/multinode/test_vision_gain_probe.py`. Extend `tests/stage1/test_trainer.py`.

**Interfaces:**
- Consumes: `tests.stage1.test_integration._tiny_dependencies`, 기존 `build_launch_plan`, owned supervisor, Stage1 stochastic context와 checkpoint save/load.
- Produces: `build_owned_vision_gain_probe_commands(plan: LaunchPlan, *, commit: str) -> OwnedTmuxCommands`; 고정 route `run_multinode_stage1_training.sh --vision-gain-runtime-probe ...`; `run_probe(args: argparse.Namespace) -> dict[str, object]` in the new test harness. Production scoring API는 만들지 않는다.
- Witness: schema 1, `kind="stage1-vision-gain-runtime-probe"`, commit/package identity, global rank, zero stage, steps, alpha, booleans `aux_grad_isolated`, `moments_base_only`, `master_delta_matches`, `second_step_retains_aux`, `reload_retains_aux`, 및 numeric residuals/peak memory. 모든 rank 0–3의 기록이 필요하다.

- [ ] **1. 기존 결함에 대한 RED tests를 작성한다.** `test_stage1_accepts_installed_ds0187_and_rejects_unknown_version`은 0.18.7을 허용하고 임의 버전을 거부한다. `test_training_step_keeps_context_through_checkpoint_backward`는 실제 inherited training_step dispatch에서 context가 backward까지 살아 있음을 검사한다. 기존 commit `2b4d06c`, `5644748`의 원인 tests를 읽고 재사용한다; version 숫자만 mocking한 성공으로 runtime를 주장하지 않는다.
- [ ] **2. RED를 실행한다.** `PYTHONPATH=transformers/src:src:. CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/stage1/test_trainer.py -k 'ds0187 or keeps_context' -q` — 기대: 위 두 누락으로 FAIL.
- [ ] **3. 필요한 기존 수정만 적용한다.** cherry-pick 전체 계보 대신 관련 diff와 tests를 현재 파일에 맞춰 이식한다. 기존 sample identity/seed와 unsupported-runtime rejection을 보존한다.
- [ ] **4. probe routing/witness RED를 작성한다.** `test_probe_is_fixed_dora_zero12_four_ranks`는 arbitrary entrypoint·ZeRO3·single node·잘못된 rank 집합을 거부하고 두 노드 명령을 검증한다. `test_probe_rejects_missing_runtime_evidence`는 booleans 일부 누락/false, NaN residual, 서로 다른 commit을 거부한다. 아래 `record`는 rank JSON을 decode한 dict다.

```python
assert record["global_rank"] in range(4)
assert record["zero_stage"] in (1, 2) and record["steps"] == 2
assert record["alpha"] == 0.001
assert all(record[k] for k in ("aux_grad_isolated", "moments_base_only", "master_delta_matches", "second_step_retains_aux", "reload_retains_aux"))
```

- [ ] **4a. probe RED 실행:** `PYTHONPATH=transformers/src:src:. CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest tests/stage1/test_vision_gain_probe.py tests/multinode/test_vision_gain_probe.py -q` — 기대: 새 API 부재로 FAIL.
- [ ] **5. 고정 probe를 구현한다.** 기존 tiny dependency fixture의 Qwen 구조를 사용하되 hidden width 64, decoder 2 layers, 실제 DoRA r32/alpha64/dropout0.05, latent8, SDPA/BF16/non-reentrant context를 사용한다. 동일 current state에서 깨끗한 score와 latent-body masked score를 직접 미분하고 FP32 수동 reference와 비교한다. probe 안의 작은 고정 계산만 사용하며 product 모듈의 선행 구현은 하지 않는다. 이전 microbatch base gradient가 남은 상황을 포함한다. 기존 owned supervisor의 preflight·종료·로그 수집을 재사용하는 고정 route만 추가한다; 일반 lifecycle-matrix 전체를 실행하지 않는다.
- [ ] **6. 위 CPU tests를 GREEN으로 확인한 뒤 검토·커밋한다.** 실행 방식에 따른 독립 review를 받고 기존 runtime 수정 및 probe 변경만 커밋한다. runtime launch 전에는 방법에 관계없이 독립 reviewer가 probe의 scope/topology/witness를 확인해야 한다. Commit: `test(stage1): verify vision gain feasibility on ZeRO-1/2`.
- [ ] **7. 승인된 경우에만 detached premise probe를 실행한다.** 승인 대상은 ZeRO1/2 각각 **initial 1 update + saved state reload 후 1 update**, 4rank, micro1/accum4, test-only `alpha_vis=0.001`, seed42/CE0.1/LR5e-5/warmup10이다. 이 alpha는 효능 기본값이 아니다. initial과 resume 모두 total `max_steps=2` 및 같은 schedule을 사용하고 probe 전용 callback이 첫 checkpoint 저장 뒤 initial을 종료한다; resume에서 budget/scheduler 설정을 바꾸지 않는다. 비교용 base-only AdamW 계산은 같은 현재 state/gradient의 CPU 또는 tensor reference로 수행하고 별도의 긴 training run을 만들지 않는다. 각 stage에 서로 다른 run ID·free port를 사용하고 양 노드 dry-run/실제 preflight를 확인한다. initial/reload child attempt는 coordinator가 소유한다. probe run 하나의 총 실제 optimizer updates는 2를 넘지 않는다.

```bash
./run_multinode_stage1_training.sh --vision-gain-runtime-probe \
  --run-id stage1-vg-probe-z1-<unique> --zero-stage 1 \
  --main-process-port <verified-free-port> --dry-run
# 승인·양 노드 preflight 후 같은 resolved 명령에서 --dry-run만 제거.
# ZeRO-2는 별도 run ID/port로 순차 실행.
```

- [ ] **8. Gate G0를 판정한다.** initial/reload 모두의 rank witness와 실제 base/masked gradients, master/BF16 sync, moments, residual을 읽는다. rank별 valid-block 패턴 `(0,1,2,3)` 및 두 번째 step의 partial group을 포함하되 모든 rank가 같은 collectives를 호출해야 한다. `B=0`/overflow는 CPU contract tests에서 먼저 검사한다. 실패하면 같은 code로 반복 launch하지 않는다. 증거를 `results/stage_1_vision_gain/workflow/premise.json`에 기록하고 Task 2–7 실행 가능 여부를 명시한다. GPU unavailable은 미검증이며 PASS가 아니다.

### Task 2: 설정과 resume identity 계약

**Files:** Create `src/stage1/vision_gain_config.py`, `tests/stage1/test_vision_gain_config.py`.

**Interfaces:**
- `VisionGainConfig(alpha_vis: float = 0.0)` frozen dataclass; properties `enabled: bool`, `min_h_norm_sq: float=1e-30`; `to_identity() -> dict[str, object]`.
- `resolve_vision_gain_config(requested_alpha: float | None, saved_identity: Mapping[str, object] | None, *, resume: bool) -> VisionGainConfig`.
- `validate_vision_gain_runtime(config: VisionGainConfig, *, method: str, zero_stage: int | None, bf16: bool, decoder_attention: str) -> None`.
- identity key `stage1_vision_gain`, exact version `stage1-vision-gain-projection-v2`, plus window/crossing/mask/reduction/precision policies from spec; unknown or changed semantic identity is rejected. Python boolean is not an accepted coefficient.

- [ ] **1. RED tests를 작성한다.** 아래 assertions에 negative, NaN, ±Inf, bool, malformed/missing identity fields 및 unknown formula를 추가한다.

```python
assert resolve_vision_gain_config(None, None, resume=False).alpha_vis == 0.0
saved = VisionGainConfig(0.001).to_identity()
assert resolve_vision_gain_config(None, saved, resume=True).alpha_vis == 0.001
with pytest.raises(ValueError):
    resolve_vision_gain_config(0.0, saved, resume=True)
with pytest.raises(ValueError):
    resolve_vision_gain_config(0.001, None, resume=True)
```

- [ ] **2. RED 실행:** 공통 CPU prefix로 `-m pytest tests/stage1/test_vision_gain_config.py -q` — 기대: module/API 부재로 FAIL.
- [ ] **3. 인터페이스를 구현한다.** fresh는 saved identity를 상속하지 않는다. resume 생략과 명시값을 분리한다. config OFF에서는 새로운 DoRA/ZeRO/attention 제한을 가하지 않는다. 활성화에서만 DoRA, ZeRO1/2, BF16, SDPA를 요구한다.
- [ ] **4. GREEN 실행:** 동일 test command — 기대: 모두 PASS; `test_resume_omission_inherits_but_explicit_zero_rejects`, `test_off_does_not_restrict_legacy_runtime` 포함.
- [ ] **5. 검토·커밋:** 선택한 방법의 review gate 후 `feat(stage1): define vision gain configuration identity`.

### Task 3: 원래 sequence를 보존하는 window, mask, scoring

**Files:** Create `src/stage1/vision_gain_data.py`, `src/stage1/vision_gain_math.py`, `tests/stage1/test_vision_gain_data.py`, `tests/stage1/test_vision_gain_scoring.py`. Modify `src/stage1/types.py::Stage1SampleMetadata`, `src/stage1/data.py::Stage1DataCollator`.

**Interfaces:**
- In `types.py`: frozen `Stage1GainBlock(block_index: int, start: int, end: int, target_positions: tuple[int, ...])`; `Stage1SampleMetadata.vision_gain_blocks: tuple[Stage1GainBlock, ...] = ()`.
- `Stage1DataCollator(..., *, vision_gain_enabled: bool = False)`; OFF는 offset 재구성·새 tokenization을 수행하지 않는다.
- In `vision_gain_data.py`: `build_gain_blocks(example: PreparedStage1Example, *, processor: object, rendered_text: str, full_inputs: Mapping[str, Tensor], final_input_ids: Tensor, final_labels: Tensor, latent_spans: tuple[Stage1LatentSpan, ...]) -> tuple[Stage1GainBlock, ...]`.
- In `vision_gain_math.py`: `build_gain_attention_mask(padding_mask: Tensor, *, row_index: int, start: int, end: int, dtype: torch.dtype) -> Tensor`; `score_gain_block(logits: Tensor, labels: Tensor, *, row_index: int, target_positions: tuple[int, ...], chunk_size: int = 1024) -> Tensor`.

- [ ] **1. window RED tests를 작성한다.** `test_immediate_item_excludes_next_text_and_no_pre_required`, `test_empty_adjacent_text_does_not_skip_forward`, `test_crossing_token_stays_in_ce`, `test_repeated_text_and_processor_expansion_keep_item_identity`, `test_latent_replacement_and_padding_coordinates`를 구현한다. 모든 fixture에서 input_ids/labels를 baseline collator와 `torch.equal`로 비교한다. original/helper 이미지 patch 수가 다르고 같은 text가 반복되는 fixture를 포함한다.
- [ ] **2. mask/scoring RED tests를 작성한다.** 다음 assertions와 causal/padding 보존, 다른 batch row/latent block 불변, 전체 vocabulary 분모·비균등 chunk 길이를 검사한다.

```python
mask = build_gain_attention_mask(torch.ones(1, 7), row_index=0, start=1, end=4, dtype=torch.float32)
assert torch.isneginf(mask[0, 0, 4:, 2:4]).all()
assert mask[0, 0, 4, 1] == 0 and mask[0, 0, 4, 4] == 0
score = score_gain_block(logits, labels, row_index=0, target_positions=(5, 6), chunk_size=1)
expected = logits[0, [4, 5]].float().log_softmax(-1).gather(1, labels[0, [5, 6], None]).mean()
torch.testing.assert_close(score, expected, rtol=1e-5, atol=1e-6)
```

- [ ] **3. RED 실행:** 공통 CPU prefix로 `-m pytest tests/stage1/test_vision_gain_data.py tests/stage1/test_vision_gain_scoring.py -q` — 기대: 새 API 부재로 FAIL.
- [ ] **4. 좌표 mapping을 구현한다.** metadata-only render에 source item별 충돌 없는 delimiter를 사용하고 제거한 결과가 원래 render와 동일함을 검증하여 char span을 얻는다. processor의 image-token 확장(`image_grid_thw.prod()/merge_size²`)을 span 이동에도 적용하고, 동일 tokenizer의 offsets를 얻어 processor full IDs와 일치 여부를 확인한다. 이 diagnostic tokenization은 입력을 대체하지 않는다. latent replacement 이전/이후 non-latent token 순서의 일대일 map으로 최종 좌표를 옮긴다. ambiguous delimiter/template/ID mismatch는 source ID를 포함해 거부한다; substring 첫 occurrence 검색으로 mapping하지 않는다.
- [ ] **5. mask/scoring을 구현한다.** mask는 `[B,1,T,T]` additive causal/padding에 지정 row의 `q>=end, start<k<end`를 `-inf`로 합친다. scoring은 nonempty valid positions를 요구하고 잘못된 좌표/label을 거부한다. empty window는 data 단계에서 scoring-invalid로 제외한다. FP32 full-vocabulary log-softmax를 chunk별 token 합계로 계산하고 전체 token 수로 나눈다.
- [ ] **6. GREEN 실행:** 위 새 tests와 `tests/stage1/test_data.py tests/stage1/test_targets.py` — 기대: 기존 입력과 teacher target tests까지 PASS.
- [ ] **7. 검토·커밋:** `feat(stage1): map immediate-text vision gain scores`.

### Task 4: FP32 누적과 전역 projection

**Files:** Extend `src/stage1/vision_gain_math.py`; Create `tests/stage1/test_vision_gain_math.py`.

**Interfaces:**
- `ProjectionResult(direction: tuple[Tensor, ...] | None, projected: bool, skip_reason: str | None, metrics: dict[str, float | None])`.
- `VisionGainUpdate(projection: ProjectionResult, block_count: int, clean_score: float | None, masked_score: float | None)`.
- `project_gain(g: tuple[Tensor, ...], h: tuple[Tensor, ...], *, min_h_norm_sq: float = 1e-30) -> ProjectionResult`.
- `VisionGainAccumulator(parameters: tuple[nn.Parameter, ...])`; `add_clean(gradients: tuple[Tensor | None, ...], *, score_sum: float, block_count: int) -> None`; `add_masked(gradients: tuple[Tensor | None, ...], *, score: float) -> None`; `finish(reduce_sum: Callable[[Tensor], Tensor]) -> VisionGainUpdate`; `clear() -> None`.
- reducer는 same-shape/device SUM Tensor를 반환한다. runtime adapter가 구현하며 accumulator는 통신 topology를 만들지 않는다. parameter별 unused gradient는 zero, masked count는 호출마다 1이다.

- [ ] **1. RED tests를 작성한다.** `g=(-1,2), h=(1,0)`에서 `u=(0,2)`와 세 directional identity를 확인한다. `h=0`, `g=0`, `g`가 `h`의 음수배, `h2=1e-32`, `h2=1e-30` 경계, nonfinite, shape mismatch를 검사한다. `test_zero_rank_and_partial_group_match_global_reference`는 four-rank microbatch fixture에서 block 수 `(0,1,2,3)`, 서로 다른 token 평균, partial tail을 수동 concatenated reference와 비교한다. mock reducer는 실제 rank별 SUM을 반환하며 call order와 count dtype을 기록한다.

```python
g, h = torch.tensor([-1., 2.]), torch.tensor([1., 0.])
result = project_gain((g,), (h,))
torch.testing.assert_close(result.direction[0], torch.tensor([0., 2.]))
assert result.projected and result.skip_reason is None
tiny = project_gain((g,), (torch.tensor([1e-16, 0.]),))
assert tiny.direction is None and tiny.skip_reason == "tiny_h_norm"
```

- [ ] **2. RED 실행:** 공통 CPU prefix로 `-m pytest tests/stage1/test_vision_gain_math.py -q` — 기대: 새 API 부재로 FAIL.
- [ ] **3. accumulator와 projection을 구현한다.** clean/masked sum과 count를 모으고 모든 rank의 count를 먼저 SUM한다. global B=0이면 공통 분기로 반환한다. 그 외 같은 순서로 모든 parameter buffer를 SUM하고 B로 나눈다. local masked count와 clean count 불일치는 오류다. projection은 spec §6의 FP64 scalar/chunk 중간값과 FP32 출력, tiny skip을 사용한다. 작은 scalar/temporary 외에 전체 FP64 parameter 복제본을 만들지 않는다.
- [ ] **4. GREEN 실행:** 동일 command — 기대: PASS, all-zero B에서 scores가 `None`, `clear()` 뒤 count와 buffer가 비어 있어 다음 update로 leakage 없음. float64 reference와 spec tolerance를 사용한다.
- [ ] **5. 검토·커밋:** `feat(stage1): accumulate and project global vision gradients`.

### Task 5: sequential auxiliary runtime와 실제 optimizer 경계 연결

**Files:** Create `src/stage1/vision_gain_runtime.py`, `tests/stage1/test_vision_gain_runtime.py`; Modify `src/stage1/trainer.py` (construction, training_step/compute_loss, `_DeepSpeedStepAdapter`, `_finish_optimizer_step`, log).

**Interfaces:**
- `Stage1VisionGainRuntime(config: VisionGainConfig, *, named_parameters: tuple[tuple[str, nn.Parameter], ...])`.
- `accumulate(model: nn.Module, *, model_inputs: Mapping[str, object], metadata: Stage1BatchMetadata, labels: Tensor, context_factory: Callable[[], ContextManager[object]]) -> None` — model_inputs에는 detached teacher latent와 original 2D mask가 포함된다.
- `prepare_update(reduce_sum: Callable[[Tensor], Tensor]) -> VisionGainUpdate`.
- `apply_after_base_step(engine: object, *, successful: bool) -> dict[str, object]` — pending update를 master에 적용 또는 skip하고 buffer를 비운다.
- `CustomTrainerStage1(..., vision_gain_config: VisionGainConfig | None = None)`; `_prepare_stage1_microbatch(model, inputs) -> tuple[dict[str, object], Stage1BatchMetadata, Stage1TeacherTargets]`를 기존 compute_loss에서 분리한다. cached prepared 입력은 한 training_step 안에서만 유효하며 finally로 지운다.

- [ ] **1. runtime RED tests를 작성한다.** `test_teacher_built_once_for_all_branches`, `test_end_marker_recomputed_with_original_positions`, `test_branch_context_rng_replay_matches_clean`, `test_auxiliary_preserves_existing_base_accumulation`, `test_alpha_zero_has_no_auxiliary_allocations`을 구성한다. tiny Qwen의 실제 training checkpoint dispatch와 dropout 호출을 기록하고 각 auxiliary graph가 다음 branch 전 해제되는지 weakref로 확인한다.
- [ ] **2. update RED tests를 작성한다.** engine spy는 `base-step -> apply-master -> sync-model -> EMA` 순서를 검사한다. base step 실패면 apply/EMA가 없고 buffer는 비워진다. master와 optimizer moment를 별도 tensor로 모델링해 model-only mutation을 탐지한다. `test_projection_ready_before_accelerate_backward_steps_engine`은 Accelerate wrapper를 경유한다. `test_local_zero_block_still_reduces`와 `test_validation_compute_loss_does_not_accumulate_aux`도 포함한다.

```python
# tests 내 engine spy가 아래 mutation event 이름과 전후 tensor snapshot을 기록한다.
assert events == ["base-step", "apply-master", "sync-model", "EMA"]
assert torch.equal(moment_after, base_only_moment_after)
torch.testing.assert_close(master_after, master_before + base_delta + 0.001 * u, rtol=1e-6, atol=1e-7)
assert torch.equal(base_grad_before_aux, base_grad_after_aux)
```

- [ ] **3. RED 실행:** 공통 CPU prefix로 `-m pytest tests/stage1/test_vision_gain_runtime.py -q` — 기대: runtime API 부재로 FAIL.
- [ ] **4. branch runtime을 구현한다.** 실제 unwrapped student forward를 사용하고 parameter identity가 prepared engine과 같은지 확인한다. explicit multimodal position IDs, identical sample seeds, 새 context 객체를 사용한다. clean score 합계 grad를 계산·해제하고 각 masked score grad를 순차 계산·해제한다. auxiliary forward에는 labels/use_cache를 각각 생략/false로 한다. original teacher input과 model state를 변경하지 않는다. 모델 global RNG는 fork/restore하며 각 branch에 같은 RNG 상태를 적용한다. hooks 우회를 위해 framework 내부를 임의 monkeypatch하지 않는다; Task 1에서 검증한 경로를 사용한다.
- [ ] **5. Trainer와 master apply를 연결한다.** auxiliary는 base forward/backward 전에 실행하고 `accelerator.sync_gradients` 경계에서 prepare_update한다. 원래 base loss 계산·스케일을 유지한다. 실제 successful step 이후 `safe_get_full_fp32_param`/`safe_set_full_fp32_param`과 기존 ZeRO1/2 `update_lp_params`로 반영한 뒤 EMA를 수행한다. getter가 None이거나 mapping 불일치면 model-only fallback 없이 오류다. fp32 master `.data`에 BF16 값을 다시 대입하지 않는다. NaN/Inf를 base step 전에 거부한다.
- [ ] **6. GREEN 실행:** 새 runtime tests, `tests/stage1/test_trainer.py tests/stage1/test_equivalence.py tests/stage1/test_ema.py` — 기대: PASS. 기존 pure base baseline과 OFF 결과·gradient·optimizer state를 비교한다. 로그는 update-level score/projection/skip/peak를 사용하고 B=0 score는 unavailable로 기록한다.
- [ ] **7. 검토·커밋:** `feat(stage1): apply projected vision gain outside AdamW`.

### Task 6: CLI·coordinator·checkpoint/resume 연결

**Files:** Modify `src/utils_deepseed.py::parse_args`, `src/stage1/composition.py`, `src/stage1/trainer.py::save_model/_save_checkpoint`, `run_stage1_training.sh`, `src/multinode_runtime.py::_STAGE1_VALUE_OPTIONS`, `src/multinode.py::_training_main`. Create `tests/stage1/test_vision_gain_wiring.py`, `tests/multinode/test_vision_gain_forwarding.py`. Extend `tests/stage1/test_cli.py`, `test_checkpointing.py`, `test_launcher.py`.

**Interfaces:**
- `read_saved_vision_gain_identity(checkpoint: Path) -> Mapping[str, object] | None` in `vision_gain_config.py`, reads `config.json["stage1_vision_gain"]` without model loading; absent key=legacy, malformed present key=error.
- argparse raw alpha default `None`; composition resolves after the actual resume checkpoint path is known, before loading weights/teacher or training. Fresh resolves to 0. `config.stage1_vision_gain = resolved.to_identity()` is saved through existing config serialization.
- worker env default is **unset**, not `0`; explicit `STAGE1_VISION_GAIN_ALPHA` or `--vision-gain-alpha` must be forwarded once. coordinator must resolve that env into an explicit forwarded option for both nodes; it cannot rely on SSH inheriting local env.

- [ ] **1. RED tests를 작성한다.** `test_resume_omission_inherits_but_explicit_zero_rejects` traverses parser → composition and rejects before model-load spy. Test fresh0, env positive, CLI overrides env, NaN/negative, same-value resume, legacy OFF, malformed identity, auto-latest checkpoint resolution, and identity mismatch. `test_active_config_saved_at_numbered_and_final_artifacts` verifies config.json in both artifact types; `test_stage2_loader_accepts_extra_stage1_identity` guards the unchanged handoff.
- [ ] **2. forwarding RED tests를 작성한다.** `test_alpha_env_is_explicit_in_both_owned_commands` checks two resolved command specs contain same alpha exactly once. Omission stays omitted through coordinator/worker on resume. Run fresh and resume shell dry-run tests without model/GPU side effects.

```python
# node_argv: 두 owned command에서 추출한 worker argv; coordinator env alpha=0.001.
for argv in node_argv:
    assert argv.count("--vision-gain-alpha") == 1
    assert argv[argv.index("--vision-gain-alpha") + 1] == "0.001"
assert numbered_config["stage1_vision_gain"] == final_config["stage1_vision_gain"]
assert numbered_config["stage1_vision_gain"] == VisionGainConfig(0.001).to_identity()
```

- [ ] **3. RED 실행:** 공통 CPU prefix로 `-m pytest tests/stage1/test_vision_gain_wiring.py tests/multinode/test_vision_gain_forwarding.py -q` — 기대: flag/wiring 부재로 FAIL.
- [ ] **4. 기존 저장 경로에 연결한다.** Task 2 resolver를 사용하여 collator enabled와 Trainer config를 전달한다. active checkpoint identity를 numbered/final config에 저장하고 post-auxiliary-save 순서를 유지한다. schema1/2/3 PEFT metadata의 field 집합은 변경하지 않는다. saved alpha와 formula를 env 기본값으로 덮어쓰지 않는다. worker/coordinator alpha parse error는 launch 전에 표시한다.
- [ ] **5. GREEN 실행:** 새 tests와 `tests/stage1/test_cli.py tests/stage1/test_composition.py tests/stage1/test_checkpointing.py tests/stage1/test_launcher.py tests/stage2/test_stage1_checkpoint_handoff.py` — 기대: PASS; `bash -n run_stage1_training.sh run_multinode_stage1_training.sh`도 PASS.
- [ ] **6. 검토·커밋:** `feat(stage1): persist and launch vision gain objectives`.

### Task 7: real-model 검증, 관측된 OOM 수정, 최종 인계

**Files:** Extend `tests/stage1/vision_gain_zero_probe.py`, `tests/stage1/test_vision_gain_runtime.py`, 필요 시 `tests/stage1/test_vision_gain_scoring.py`. Modify `docs/multinode-training.md` with new flags/supported scope after runtime evidence. Only if measured OOM requires it: `src/stage1/vision_gain_runtime.py`, `src/stage1/vision_gain_math.py`, `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` and `tests/stage1/test_qwen_vision_gain_logits.py`.

**Interfaces:** Task 1 probe의 witness schema/owned coordinator 경로를 유지하면서 final production runtime을 직접 사용하게 전환한다. optional Qwen API는 measured LM-head OOM일 때만 `logits_to_keep: Tensor | None = None`을 forward에 추가한다; 1D predictor indices, default None은 기존 전체 logits이며 auxiliary labels=None에서만 허용한다.

- [ ] **1. 최종 production 경로의 causal tests를 보강한다.** `test_master_displacement_survives_next_step_and_resume`는 Task 5/6을 직접 경유한다. G0의 별도 probe 구현만 통과하는 경우는 FAIL이다. alpha0/base equivalence, moving/frozen EMA, skipped optimizer, malformed mapping, final partial를 포함한다. 새 변경이 있으면 먼저 RED를 확인하고 최소 수정 후 GREEN을 확인한다.
- [ ] **2. 개발 방식에 따른 독립 review를 완료한다.** 전 task의 specification/quality findings 및 fixes를 닫고, 남은 findings는 unresolved로 기록한다. runtime 검증 전 필요한 코드 review gate를 마친 reviewed commit/hash를 양 노드에 준비한다. 실행 방식이 요구하는 fixer agent를 controller 구현으로 대체하지 않는다.
- [ ] **3. real-model smoke protocol을 구체화하고 승인 범위와 대조한다.** 실행 직전 canonical training 문서 재확인, 전체 baseline 설정 강제, fresh run IDs·free ports·양 노드 동일 code/package·자원 preflight를 수행한다. original Qwen2.5-VL-3B에서 fresh Stage 1로 시작한다. Stage 2용 checkpoint-431 initializer를 Stage 1에 잘못 적용하지 않는다. retained 입력의 실제 processed 길이와 블록 수로 긴 sequence 및 multi-block 사례를 골라 `data/<run-id>/`에 source IDs를 기록한다. full training budget을 만들지 말고 ZeRO별 2-update smoke로 제한한다. smoke coefficient는 실행 승인 시 명시한다; 별도 값이 없으면 test-only 0.001의 사용을 요청하며 자동으로 효능 기본값으로 승격하지 않는다.
- [ ] **4. 승인된 smoke를 기존 coordinator로 detached 실행하고 한 번 결과를 검증한다.** ZeRO-1/2 각각의 4개 global rank participation, clean/masked finite scores와 gradient, master delta/state/resume evidence, phase별 peak memory를 수집한다. tiny fixture와 real-model 증거를 구분한다. random small inputs 성공을 longest-case memory PASS로 쓰지 않는다.
- [ ] **5. OOM이 실제 발생한 경우에만 원인별 최소 수정한다.** 실패 로그/phase/shape를 보존하고 새 causal regression을 먼저 RED로 만든다. activation 문제면 checkpoint/recompute 수명만 조정한다. LM-head logits 문제면 필요한 predictor row 선택 후 기존 head와 sparse output row를 그대로 적용하고 전체 vocabulary log-softmax를 유지한다. token chunk도 autograd graph가 모든 FP32 log-softmax를 붙잡지 않도록 non-reentrant checkpoint 또는 즉시 gradient 수집으로 실제 peak를 줄이는지 확인한다. small-input score/gradient 비교 및 OFF/Stage 2 regressions를 GREEN으로 만든 뒤 필요한 독립 review를 받고 fresh run ID로 retry한다. prefix-cache 전환이나 목적함수 변경이 필요하면 설계를 재승인받는다.
- [ ] **5a. operator guide를 갱신한다.** 선택한 구현 방식으로 `docs/multinode-training.md`에 alpha flags, OFF/default/resume 구별, ZeRO-1/2 지원 범위, 검증된 실행과 OOM 조정 결과를 기록한다. 이번 실행값을 새로운 기본값으로 표현하지 않는다. 문서도 독립 검토 대상에 포함하고 final verification 전에 변경을 끝낸다.
- [ ] **6. 최종 review 후 한 번의 completion verification을 실행한다.** 먼저 `superpowers:requesting-code-review`, `verification-before-completion`, 이어지는 `finishing-a-development-branch`의 적용 요건을 읽는다. full-suite command는 아래 repository CPU suite이며 finishing Step 1과 같은 test-pass claim에 재사용한다. GPU 증거는 최종 변경 파일 identity가 검증 대상과 동일한지 확인하고, 관련 코드 변경이 있었다면 그 영향받는 bounded smoke만 재실행한다. CPU PASS만으로 GPU 완료를 주장하지 않는다.

```bash
PYTHONPATH=transformers/src:src:. CUDA_VISIBLE_DEVICES='' \
  ./venv/ilvr-venv/bin/python -m pytest tests -m 'not gpu and not distributed' -q
bash -n run_stage1_training.sh run_multinode_stage1_training.sh
git diff --check
```

- [ ] **7. 결과와 branch 선택지를 인계한다.** `results/stage_1_vision_gain/workflow/verification.json`에 승인 scope, method, agents, review verdicts, final commit/diff identity, 명령·exit code·CPU/GPU 별 결과 및 미검증 범위를 기록한다. 검증 후 product/test/operator-doc 변경이 생기면 해당 review/verification gate로 돌아간다. 사용자에게 merge/PR/keep 선택을 맡기며 worktree를 자동 삭제하지 않는다. 완료 final 직전에 ntfy summary hook을 실행한다.

## Coverage and handoff

| Spec section | Owning tasks |
| --- | --- |
| 1–2 scope, existing runtime | 1; global constraints |
| 3 config/support | 2, 6 |
| 4 data/scoring | 3 |
| 5 branches/RNG/memory | 3, 5, 7 |
| 6 global math | 4 |
| 7 actual update/EMA | 1, 5, 7 |
| 8 premise evidence | 1 / G0 |
| 9 resume | 2, 6, 7 |
| 10 OOM/logging | 5, 7 |
| 11 verification | all task tests; 7 final gates |
| 12–13 files/storage/workflow | file map, constraints, task gates |

현재 상태: 계획 작성만 완료, 모든 실행 checkbox는 미완료다. implementation method와 probe 실행 승인은 아직 없다. 사용자의 명시적 writing-plans 요청은 spec 검토 후 계획 단계로의 전환 승인으로 기록했으며, probe나 제품 구현 승인으로 확대하지 않았다.

권장 실행 방식은 **Subagent-driven**이다. 데이터 좌표, 전역 수학, framework step/state가 각각 독립적으로 잘못될 수 있어 task별 review가 유용하다. Native를 선택하면 root가 구현하고 해당 skill의 독립 최종 review를 수행하며, Task 1 probe의 사전 독립 review는 별도로 유지한다.

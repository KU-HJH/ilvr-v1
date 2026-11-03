# ILVR History Query Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` for inline execution, or `superpowers:subagent-driven-development` when an approved subagent execution environment is available. The checkboxes below are implementation work items, not claims of completed work.

**Goal:** 선택된 과거 visual target이 실제 주입된 teacher prefix에서, discounted text와 attention-weighted latent history로 Stage 1 target을 선택한다.

**Architecture:** 기존 selector를 legacy 분기로 유지한다. 새 `src/history_*` 모듈에서 위치 파싱, 순수 query 연산, Q/K probability 집계, prefix replay를 분리한다. EMA runtime 검증/수정은 알고리즘 비교군 모두에 공통으로 적용한다.

**Tech Stack:** Python 3.11, repository의 custom Transformers, PyTorch/BF16, TRL SFTTrainer, Accelerate, DeepSpeed ZeRO-3, pytest.

**Spec:** `ILVR_History_Query_Spec.md`와 함께 읽는다. 저장소 내 배치 시 `docs/superpowers/specs/2026-10-15-ilvr-history-query.md`를 참조한다.

**Base commit:** `13a67ed2ee975d6ca56b4e9eba18d68b352b166b`.

**Plan status:** 실행 전 계획. 아래 PASS/FAIL은 예상 결과이다. 이번 문서 작성 과정에서 해당 테스트를 실행하지 않았다.

## Global Constraints

- Adaptive mode는 Stage 1, full fine-tuning, 프로세스당 batch size 1만 지원한다.
- Default는 `history_query_mode=legacy`, `ema_runtime_mode=upstream`이다. Adaptive는 verified EMA를 요구한다.
- 새 방법의 첫 실험값은 gamma=.8, 마지막 decoder layer, query/key/head chunk=64/2048/4이다.
- Student architecture, `transformers/` 소스, `eval.py`, candidate pooling, top-p/top-k, alignment loss를 임의 변경하지 않는다.
- 현재 latent 시작 전 prefix만 teacher context로 사용한다. 과거 target을 실제 주입해야 한다.
- Probability 분모는 전체 valid causal context다. Latent만 softmax하지 않는다.
- 모든 hook은 예외에서도 해제한다. Teacher에 gradient를 만들지 않는다.
- 새 기능의 버그를 uniform weight나 CE-only 학습으로 숨기지 않는다.
- 신규 파일은 명세의 §8 경로를 따른다. 구현 시작 전에 승인된 isolated branch/worktree를 사용한다.

## Review Focus

1. Role/header와 padding이 text span에 끼거나 latent 위치가 collator 이전 좌표로 계산되는 문제 — Task 1에서 검증.
2. Flash special forward 때문에 self-attn module hook이 실행되지 않는 문제 — Task 3에서 projection-hook 검증.
3. Teacher가 과거 selected target 대신 placeholder를 읽는 문제와 미래 helper 누출 — Task 4에서 검증.
4. `Kstar<K`의 target↔mask↔next-token loss alignment 오류 — Task 5에서 검증.
5. EMA가 microstep마다 갱신되거나 resume/rank 간 달라지는 문제 — Task 2와 Task 6에서 검증.

---

## Task 1. 위치 파싱과 query 순수 함수

**Files**

Create: `src/history_types.py`, `src/history_layout.py`, `src/history_query.py`, `tests/test_history_layout.py`, `tests/test_history_query.py`, `tests/conftest.py`, `pytest.ini`, `requirements-dev.txt`.

**Interfaces**

Produces: 명세 §7의 `SegmentLayout`, `SampleLayout`, `LatentRecord`, `PrefixObservation`, `QueryResult`, `TargetBuildResult`, `HistoryQueryConfig`.

Produces: `parse_sample_layout(...)`, `summarize_text(...)`, `aggregate_latent(..., permutation=None)`, `compose_query(...)`.

오류 타입은 `HistoryLayoutError(ValueError)`, `HistoryQueryError(ValueError)`, `UnsupportedHistoryModeError(NotImplementedError)`로 고정한다. 모든 mode/값의 검증은 dataclass 검증 함수로 모은다.

- [ ] **1.1 실패 테스트 작성:** 두 구간 gamma=.5의 결과 `(1/3,2/3)`, 세 구간 결과 `(1/7,2/7,4/7)`, gamma=0/1, 빈 구간, 작은 latent mass를 검증한다.
- [ ] **1.2 Layout 실패 테스트 작성:** 실제 tokenizer ID pattern을 모사한 fixture로 assistant header, 특수 토큰, multiple latent segments, 양쪽 padding, malformed marker를 검증한다.
- [ ] **1.3 실패 확인:** `python -m pytest -q tests/test_history_layout.py tests/test_history_query.py`. 신규 함수 미구현으로 FAIL해야 한다.
- [ ] **1.4 최소 구현:** 명세의 FP32 연산, empty 규칙, 원본 좌표를 그대로 구현한다. Dataset/model forward 의존성을 이 모듈에 넣지 않는다.
- [ ] **1.5 성공 확인:** 같은 명령이 PASS. Shuffled mode의 입력 permutation을 동일하게 재사용하면 결정적으로 같은 결과여야 한다.
- [ ] **1.6 Commit:** `git add src/history_types.py src/history_layout.py src/history_query.py tests requirements-dev.txt pytest.ini && git commit -m "feat: define history query math and segment contracts"`.

## Task 2. EMA lifecycle을 먼저 검증하고 공통 runtime으로 분리

**Files**

Create: `src/ema_runtime.py`, `tests/test_ema_runtime.py`.

Modify: `src/trainer.py::_EMATeacher`, `CustomTrainerStage1.__init__`, EMA 연결부.

**Interfaces**

Consumes: 기존 model/teacher parameter, Trainer callback lifecycle.

Produces: `VerifiedEMARuntime(trainer, teacher, decay)`; `on_successful_step()`, `save_checkpoint(output_dir, global_step)`, `restore_checkpoint(checkpoint_dir)`.

Runtime은 callback에서 trainer의 실제 skip 상태를 확인하고 `on_successful_step()`을 호출한다. `update_count`는 성공 update만 센다. `upstream` mode에서는 새 callback을 등록하지 않는다.

- [ ] **2.1 실패 테스트 작성:** accumulation=2의 4 microstep에서 update=2, skip update=0, scalar EMA `1→1.2`를 확인한다.
- [ ] **2.2 현재 hook 동작 관측:** 작은 Trainer fixture를 실제 학습 루프로 돌려 기존 `optimizer_step` override 호출 수를 측정한다. 이 결과를 기록하고 정적 추측만으로 완료 처리하지 않는다.
- [ ] **2.3 실패 확인:** `python -m pytest -q tests/test_ema_runtime.py -m "not distributed"`.
- [ ] **2.4 구현:** 명시적 callback, 중복 갱신 방지, 이름/shape 기반 parameter mapping, 저장/복원을 구현한다. Teacher 초기화 시 storage 독립성을 검사한다.
- [ ] **2.5 Resume 테스트:** uninterrupted run과 checkpoint 이후 재개 run의 teacher state와 next-step 결과가 일치한다. Teacher 파일 누락 또는 config mismatch는 오류여야 한다.
- [ ] **2.6 성공 확인 및 commit:** 같은 테스트 PASS 후 `git commit -m "fix: add verified EMA lifecycle and resume state"`.

## Task 3. Q/K 기반 reduced attention collector

**Files**

Create: `src/history_attention.py`, `tests/test_history_attention.py`.

**Interfaces**

Consumes: teacher backbone, resolved layer indices, prefix attention mask, current text positions, past active latent positions.

Produces: context manager `ReducedAttentionObserver(backbone, layer_ids, query_positions, latent_positions, prefix_mask, config)`.

Observer의 `scores()`는 명세 §4.3의 순서대로 FP32 `[M]`를 반환한다. 진입 전/종료 후 hook 개수를 검사할 수 있어야 한다. Query/head/key chunk 크기는 config에서 받는다.

- [ ] **3.1 Dense reference 실패 테스트 작성:** 동일 logits와 valid key 4개/latent 2개에서 mass=.5. Future/padding mask가 denominator에 반영되는지 검사한다.
- [ ] **3.2 Qwen reference 실패 테스트 작성:** 작은 eager backbone의 native attention과 비교한다. GQA, MRoPE, last/last4/all을 포함한다.
- [ ] **3.3 실패 확인:** `python -m pytest -q tests/test_history_attention.py -m "not gpu"`.
- [ ] **3.4 구현:** Decoder pre-hook으로 position embeddings를 읽고 q_proj/k_proj output hook으로 실제 projection을 관측한다. Streaming log-sum-exp로 전체 key 분모를 계산한다. Full attention tensor를 저장하지 않는다.
- [ ] **3.5 부작용 검사:** Observer 유무로 backbone hidden state가 바뀌지 않는다. 예외 후 hook이 남지 않는다. Flash special forward에서 projection hook이 실행된다.
- [ ] **3.6 성공 확인 및 commit:** CPU reference PASS, 지원 환경에서 GPU reference PASS 후 `git commit -m "feat: add reduced causal attention observer"`. GPU 미실행은 별도 미검증으로 기록한다.

## Task 4. Teacher prefix adapter와 순차 target builder

**Files**

Create: `src/history_teacher.py`, `tests/test_history_teacher.py`.

**Interfaces**

Consumes: Task 1의 타입/함수, Task 3 observer, fixed teacher, 기존 global/candidate/selection/mask callback.

Produces: `TeacherPrefixAdapter.observe(...) -> PrefixObservation`, `HistoryTargetBuilder.build(inputs, k, p) -> TargetBuildResult`.

`HistoryTargetBuilder`는 새로운 선택 순서를 생성하지 않고 기존 selector가 반환한 순서를 보존한다.

- [ ] **4.1 Injection 실패 테스트 작성:** 두 helper 구간 fixture에서 첫 구간의 target이 두 번째 prefix의 실제 embedding에 주입되어 있는지 검사한다.
- [ ] **4.2 Causality 실패 테스트 작성:** 미래 text/helper 변경으로 현재 query가 바뀌지 않는다. 현재 helper 변경은 query가 아니라 candidate/selection에만 영향을 줄 수 있다.
- [ ] **4.3 실패 확인:** `python -m pytest -q tests/test_history_teacher.py`.
- [ ] **4.4 Adapter 구현:** Embedding 조립과 prefix MRoPE를 계산하고 bare `teacher.model`을 호출한다. LM head, full-layer hidden-state output, helper-image LM 입력을 사용하지 않는다.
- [ ] **4.5 Builder 구현:** i 순서로 selected-target ledger를 갱신하고 text mean 상태를 현재 build 안에만 보관한다. Empty text, 없는 latent, variable Kstar를 명세대로 처리한다.
- [ ] **4.6 성공 확인 및 commit:** 테스트 PASS와 input 불변성/no-grad 확인 후 `git commit -m "feat: replay selected latent prefixes for target construction"`.

## Task 5. 기존 trainer 및 CLI 연결

**Files**

Modify: `src/trainer.py`, `src/utils_deepseed.py`, `src/main.py`, `src/task_deepseed.py`.

Create: `run_training_history.sh`, `tests/test_history_config.py`, `tests/test_history_trainer.py`.

**Interfaces**

Consumes: Tasks 1–4의 config, runtime, builder.

Produces: 기존 `(teacher_latents, firstK_mask)` API를 유지하는 mode dispatch, 명세 §9의 CLI, run/config manifest.

Legacy 함수의 내부 알고리즘은 보존한다. CLI seed는 `args=get_args()` 이후 `seed_everything(args.seed)`에 연결하고 SFTConfig의 seed/data_seed에도 일치시킨다. Image root는 환경변수 우선, 기존 값 fallback으로 처리한다. `--resume_from_checkpoint`는 경로를 검증한 뒤 `trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)`에 전달한다.

- [ ] **5.1 실패 테스트 작성:** Default legacy/upstream, adaptive에 필요한 verified/stage1 제약, invalid gamma/chunk/layers, unsupported LoRA/batch 조합을 검사한다.
- [ ] **5.2 Legacy parity 테스트 작성:** 동일 fixed teacher/RNG에 대해 수정 전 reference와 legacy target/mask가 일치한다.
- [ ] **5.3 Alignment 실패 테스트 작성:** `teacher_latents.shape[1] == firstK_mask.sum()`, inactive pad 제외, target t 대 pred t-1 대응을 검사한다.
- [ ] **5.4 실패 확인:** `python -m pytest -q tests/test_history_config.py tests/test_history_trainer.py`.
- [ ] **5.5 구현:** 새 config를 trainer로 전달하고 adaptive 분기에서만 새 builder를 호출한다. Diagnostics는 model kwargs에서 분리한다. Mismatch의 silent CE-only fallback을 adaptive에서 차단한다.
- [ ] **5.6 실행 스크립트 검증:** 데이터 파일/image root/log/output 경로를 검증한다. `bash -n run_training_history.sh`가 성공해야 한다. 예시 새 인자가 `--help`에 표시되어야 한다.
- [ ] **5.7 성공 확인 및 commit:** 테스트 PASS 후 `git commit -m "feat: wire adaptive history target selection into ILVR training"`.

## Task 6. End-to-end, 분산, 재현성 검증

**Files**

Create: `tests/test_history_integration.py`, `tests/zero3_history_smoke.py`.

Modify: `src/ema_runtime.py`, `src/history_teacher.py`는 통합 테스트로 발견한 계약 위반에 한해서만 수정한다.

**Interfaces**

Consumes: 실제 trainer/collector/builder 전체 경로.

Produces: CPU/tiny model smoke 결과, GPU reference 결과, 2-GPU ZeRO-3 결과, checkpoint-restart 결과, 측정 JSON.

- [ ] **6.1 Integration 실패 테스트 작성:** 여러 latent 구간, variable Kstar, 빈 text, teacher no-grad/student grad, config restore를 검증한다.
- [ ] **6.2 2-rank 검사 작성:** EMA update count/checksum 동일, 모든 rank의 teacher shape 정상, target shape 정상, collective deadlock 없음.
- [ ] **6.3 Unit/reference 전체 실행:** `python -m pytest -q tests -m "not gpu and not distributed"`.
- [ ] **6.4 GPU 실행:** `python -m pytest -q tests/test_history_attention.py tests/test_history_integration.py -m gpu`.
- [ ] **6.5 분산 실행:** `torchrun --standalone --nproc_per_node=2 tests/zero3_history_smoke.py`.
- [ ] **6.6 Profile:** 8k fixture와 사용할 최대 길이에서 teacher/probe latency와 peak VRAM을 기록한다. 큰 attention matrix가 저장되지 않는지 확인한다. OOM 또는 미실행은 실패/미검증으로 구분한다.
- [ ] **6.7 Commit:** `git commit -m "test: verify history target causality gradients and distributed resume"`.

## Task 7. 실험 설정과 인수 문서

**Files**

Create: `docs/history_query_experiments.md`.

Modify: `README.md`의 별도 기능 설명 영역에만 새 실행/제약/테스트 절을 추가한다.

**Interfaces**

Consumes: 완료된 테스트와 실제 측정값.

Produces: R0/R1/C0…C4 비교 명령 표, config manifest 예, 지원/미지원 환경 표, 재현 절차.

- [ ] **7.1 설정 비교 검사:** 실험군 간 달라지는 field가 명세 §13과 일치하는지 자동 비교한다. Gamma/layer sweep에서 dataset/seed/K/budget가 의도치 않게 바뀌면 실패다.
- [ ] **7.2 문서 작성:** Runtime/context 정비 효과와 query weighting 효과를 분리한다. `coverage_p=.9`의 variable Kstar를 설명한다.
- [ ] **7.3 검증 결과 첨부:** 실제 실행 명령, 결과, GPU/라이브러리 버전, 미검증 항목을 기록한다. 구현 완료와 성능 우위를 분리한다.
- [ ] **7.4 최종 회귀 테스트:** Task 6의 unit/reference 명령을 다시 실행한다. Working tree와 변경 파일 범위를 확인한다.
- [ ] **7.5 Commit:** `git commit -m "docs: document adaptive history query and controlled experiments"`.

## Handoff

실행자는 먼저 spec과 이 plan의 경계 조건을 읽고, 실제 checkout/import 경로가 고정된 commit에 대응하는지 확인한다. 실행 환경이 v1 범위를 벗어나거나 reference attention과 불일치하면 그 원인을 해결하기 전 성능 실험으로 진행하지 않는다.

이 문서에는 구현 결과나 성능 개선 수치가 포함되어 있지 않다.

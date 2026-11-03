# design proposal v1 — ILVR 오프라인 평가 기반과 VSP

상태: 동결할 설계 제안. 아직 구현 승인이 아니다.
작성일: 2026-09-10. 기준 ILVR: `2497ee7bafad1b51ef6c2aba6f40f084830916c4`.
상위 요구: `documents/migration_plan_codex_reference_ko.md`.
로컬 근거: `docs/evaluation/audit.md`, `references.lock.json`.

## 문제와 범위

저장된 모델 응답을 명시적인 샘플 목록과 연결해 재채점하고, VSP의 답안 추출·경로 문법·환경 전이를 구별한다. 현재 CLI의 VSP 함수는 잘못된 문자를 삭제하고 목표 즉시 종료를 하지 않으며, MathRuler 실패 문자열이 fallback을 막는다. 이번 P1은 합성 입력으로 검증 가능한 오프라인 모듈/CLI만 만든다. 실제 모델 호출과 원본 데이터 manifest 준비는 P2다.

현재 `eval.py`, `src/evaluate_deepseed.py`, `src/task_deepseed.py`, patched Transformers, `src/evaluation/{scoring,runtime,continuous_latent}.py`의 동작은 유지한다. GPU, judge, 데이터 전체 다운로드, CoMT/VisuLogic/EMMA/Zebra adapter, 추론 bridge, resume/retry 시스템은 P1 범위 밖이다.

## 접근 비교

| 접근 | 판단 |
|---|---|
| 기존 evaluation 패키지 안에 독립 offline 모듈 추가 | 권장. 기존 normalization/분모 계약을 보존하며 공통 ID·오류 계약을 다음 benchmark가 재사용 |
| 기존 scoring.py에 benchmark 분기를 추가 | 기존 exact-match/Yes-No 정규화와 새 profile의 경계가 섞임. 이번에는 선택하지 않음 |
| VSP 전용 단일 스크립트만 추가 | 시작은 작지만 manifest join/분모/오류 계약을 후속 adapter마다 다시 만들게 됨. 이번에는 선택하지 않음 |

## 모듈 경계

`src/evaluation/schema.py`, `io.py`, `registry.py`, `report.py`, `benchmarks/__init__.py`, `benchmarks/vsp.py`와 `scripts/__init__.py`, `scripts/score_eval.py`, `configs/eval/vsp_validated_v1.yaml`을 추가한다. 이미 존재하는 `src/evaluation/__init__.py`는 경량 상태를 보존한다. 런타임 import는 현재 저장소 관례인 `evaluation.*`; `PYTHONPATH=src`를 명시한다. 파일별 구체적 함수와 실패 테스트 코드는 승인 뒤 상세 plan에서 확정한다.

새 scorer는 표준 라이브러리로 동작하고 CLI의 YAML 읽기에 이미 설치된 PyYAML만 사용한다. torch/Transformers/CUDA/MathRuler/Reference import를 하지 않는다. upstream 파일/함수를 런타임에 AST로 읽는 방식도 사용하지 않는다. 고정 참조 검사는 개발 감사/선택적 원본 대조용이며 CPU scorer 실행의 전제 조건이 아니다.

## 데이터 흐름과 계약

```text
Sample manifest + Prediction JSONL + 명시적 profile config
  -> 파일/ID/형식/fingerprint 검증
  -> response에서 답안 추출 (gold 접근 없음)
  -> 경로 전체 문법 검증
  -> target의 지도/환경 규칙으로 채점
  -> manifest 순서의 Score JSONL + summary JSON
```

- Sample: schema version, `benchmark/subtask/source_id`를 포함하는 namespaced id, benchmark/subtask, ModelInput, benchmark-specific Target, provenance.
- ModelInput: 원문 문제 텍스트, 순서가 보존되는 문제 이미지 상대 경로, 선택지. trace/해설/정답 label은 허용하지 않는다. 원문 문제에 포함된 지도 표현은 보존한다.
- Target: P1은 VSP 지도와 환경 규칙. 정답 경로는 필수가 아니다. 선택지 벤치마크 Target의 실제 타입 검증은 해당 adapter 착수 때 추가하며 미구현 benchmark는 거부한다.
- Provenance: dataset ID/revision/config/split/source_id, 입력 자산 hash. 합성 fixture는 합성 출처임을 명시한다. 원본 benchmark/논문 split인 것처럼 표기하지 않는다.
- Manifest hash: 완성된 JSONL bytes의 SHA-256. Sample 자체에 자기 파일 hash를 넣지 않고 Prediction/run metadata가 참조한다.
- Prediction: Sample id, manifest hash, run fingerprint, 원문 response, 생성 상태. P1 run fingerprint는 응답 생성 설정/모델·prompt identity를 명시한 metadata에서 계산하며 부재를 임의 상수로 채우지 않는다. synthetic 입력은 synthetic identity를 사용한다. P1이 실제 모델 identity의 진실성까지 입증하지는 않는다.
- Score: id, parsed answer, extraction method/status, profile/version, correctness와 환경 종료 정보 또는 infrastructure 오류.
- 이미지 상대 경로는 명시적 dataset root 기준으로 resolve하며 그 root는 현재 `data/` 아래여야 한다. 열거된 이미지의 존재와 hash를 preflight한다. Reference를 이미지 root로 삼지 않는다.
- manifest/prediction이 비어 있거나 JSONL 형식/ID가 잘못되면 명시적 오류. duplicate/unknown ID, fingerprint 불일치를 즉시 거부한다. 새로운 일반 fallback은 없다.

## VSP profile

`vsp_validated_v1`은 수정한 parser와 명시적 환경 계약의 이름이다. 이름 자체가 원본 환경 또는 논문 재현 검증 통과를 뜻하지 않는다. report에서 verification level을 별도 기록한다. `vsp_ilvr_legacy`는 P0 현재 소스 관찰을 가리키는 감사용 이름으로 유지한다. P1 production registry에는 검증된 새 계약만 등록하고, legacy 비교 사례는 tests/보고에 둔다. 과거 코드를 새 런타임 의존성으로 만들지 않는다.

추출 우선순위는 마지막 `\boxed{...}`의 balanced span, 그 표시가 없으면 마지막 줄 시작 final-answer/action-plan marker, 그런 marker도 없으면 전체 응답이다. 마지막 boxed가 미완성이면 fallback 없이 invalid. marker는 대소문자를 구별하지 않고 `final answer:`, `final answer is:`, `action plan:` 및 전각 colon을 허용한다. marker 이후의 나머지 문자열 전체를 후보로 사용해 뒤의 잘못된 문자를 숨기지 않는다.

후보의 문법은 대소문자 U/D/L/R, ASCII 공백·tab·CR·LF 및 comma만 허용한다. comma는 비어 있지 않은 action group 사이에만 허용하고 중복/선행/후행 comma는 거부한다. 공백 구분 또는 붙여쓴 action은 허용한다. 따옴표·마침표·영어 방향명·기타 문자는 삭제하지 않고 invalid로 처리한다. 빈 응답은 empty, 불완전 boxed/허용하지 않은 문법은 invalid_format으로 구분한다. 전체 후보 검증은 환경 전이보다 먼저 실행한다. 따라서 목표 뒤 X가 있는 `RX`도 invalid다.

지도는 numeric `0/1/2/-1` 또는 문자 `F/S/G/H`에서 공통 표현으로 변환한다. boolean을 숫자로 받아들이지 않는다. 직사각형, 양의 크기, 시작점/목표 각 1개, 알려진 셀 값만 허용한다. 통과 조건은 목표 도달이며 shortest-path 점수는 추가하지 않는다.

환경 계약: deterministic 이동, 경계 밖 이동은 제자리이며 step은 소비한다. hole 또는 goal 도달 즉시 terminated. 명시적 양의 정수 max_steps를 요구하고 전이 후 elapsed step이 limit에 도달하면 truncated. goal과 limit가 같은 step이면 reward=1, terminated=true, truncated=true이며 성공이다. 경로가 먼저 끝나면 위치를 유지한 채 실패/미종료로 보고한다. Gym 0.26.2 FrozenLake-v1 대응 config의 max_steps는 소스에서 확인한 100으로 명시한다. CPU scorer는 Gym을 import하지 않는다.

## 집계·실패·저장

- 분모는 전체 manifest의 N_expected이며 0개 manifest는 오류다. 빈/문법 실패 response는 모델 실패로 분모에 포함한다.
- 누락 prediction 또는 infrastructure error가 있으면 complete=false, 최종 accuracy=null. coverage는 사용 가능한 모델 응답 수 / N_expected; 빈/문법 실패 응답도 관측된 응답이다. 오류 응답과 미수신은 coverage에 포함하지 않는다.
- complete=true일 때 accuracy=correct/N_expected. P1에서는 provisional accuracy와 macro average를 추가하지 않는다. 필요하면 해당 benchmark 단계에서 명시적 계약으로 추가한다.
- 예상 category 목록은 config가 명시한다. 비어 있는 category는 count=0, accuracy=null; 예상하지 않은 category는 preflight 오류다. partial 결과를 최종 accuracy로 표시하지 않는다.
- 새 CLI는 완료 0, 입력/config 오류 2, 미완료 평가 1을 반환한다. 입력 오류에는 최종 점수 파일을 쓰지 않는다. 미완료이면 상태가 명시된 보고서를 남긴다. 기존 결과를 덮어쓰지 않고 고유 run output 디렉터리를 요구한다.
- 데이터/이미지/manifest/데이터 fixture는 `data/evaluation/`; 모델 응답과 점수는 `results/evaluation/<run-id>/`; 실행 로그는 `logs/evaluation/<run-id>/`. 기존 data symlink를 유지한다. 모델을 쓰지 않는 P1은 checkpoints를 생성하지 않는다.

## 검증과 남는 불확실성

P1 TDD는 지도만 있는 VSP, R/RD/XR/RX, MathRuler sentinel과 무관한 final-answer 추출, hole/경계/limit, 잘못된 지도/ID/image/fingerprint, empty/invalid response와 missing/infrastructure 구분, 고정 분모/빈 category를 다룬다. source tree와 Reference가 없는 별도 subprocess에서 새 경량 모듈/저장 응답 재채점 smoke를 실행한다. 새 모듈과 필요한 최소 fixture만 전달하고 `data/`와 `results/` 경계를 지킨다. 기존 scorer 11개 테스트를 호환 검증에 포함한다.

현재 Gym은 미설치이며 원본은 Python 3.12.2/Gym 0.26.2/NumPy 1.26.4를 명시한다. 원본 대조 first-nail은 원본 evaluator의 make/reset/step 경로를 작은 지도와 경계/종료/100-step 사례로 직접 관찰하는 것이다. dependencies 변경/별도 환경 준비는 현재 승인 범위가 아니므로 아직 실행하지 않았다. 새 구현과의 runtime 비교는 구현 후 별도 witness가 필요하다.

P1은 이러한 미실행 상태를 숨기지 않고 CPU 계약 구현까지 진행하는 제안이다. 승인 시에도 upstream_environment, GPU, full benchmark, paper protocol은 자동 통과되지 않는다. 원본 환경 증거 없이 과학적 재현 완료를 보고하지 않는다. 데이터 누출 검증은 독립 원본 label/map/manifest와 맞대조해야 하며, 같은 설계에서 만든 fixture가 통과한 사실만으로 benchmark 정확성을 입증하지 않는다.

## 다음 gate

이 제안을 동결해 hate와 필요한 macrothink/prism을 적용하고 design commitment를 만든다. 사용자가 최종 commitment를 승인한 뒤 writing-plans로 P1의 실제 실패 테스트·함수·명령·작업 경계를 작성한다. 그 전에는 구현 코드를 수정하지 않는다.

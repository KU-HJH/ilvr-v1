# design commitment v2 — ILVR 오프라인 평가 기반과 VSP

작성일: 2026-09-10. 상태: **설계·계획 승인 완료; P1 구현 및 CPU 계약 검증 완료 (.worktrees/evaluation-p1)**.
기준 ILVR: `2497ee7bafad1b51ef6c2aba6f40f084830916c4`.

이 문서는 다음 P1의 최종 검토 대상이다. 상위 요구는 `documents/migration_plan_codex_reference_ko.md`, 로컬 근거는 `references.lock.json`과 `docs/evaluation/audit.md`다. 상세 규칙을 동결한 `docs/evaluation/p1-design-proposal-v2.md`(SHA-256 `af47a62a48e6349d123d824bfd6f32da6636b4bf53269ea86ed1b407edd4eda9`)와 의미를 일치시켰다. 비평은 `docs/evaluation/p1-design-review.md`에 기록했다. 자동 commit/push하지 않는다.

## 1. 문제·범위·선택

저장 응답을 고정 Sample 목록과 연결해 재채점하는 CPU 기반 경로를 만든다. 첫 benchmark는 VSP다. 현재 VSP의 잘못된 문자 삭제, 목표 도달 뒤 계속 이동, MathRuler의 truthy `"None"` 문제를 새 명시적 profile에서 구분한다.

기존 `evaluation` 패키지에 독립 offline 모듈을 추가하는 접근을 선택한다. 기존 `scoring.py` 확장은 Yes/No 정규화와 기존 분모 계약을 섞고, VSP 단일 스크립트만 추가하면 후속 benchmark가 ID/오류/분모 처리를 중복하므로 선택하지 않는다.

P1은 공통 계약·VSP scorer·저장 응답 CLI·CPU 테스트만 구현한다. `eval.py`, `src/evaluate_deepseed.py`, `src/task_deepseed.py`, 기존 `evaluation/{scoring,runtime,continuous_latent}.py`, patched Transformers와 생성 동작은 유지한다. P2에서 실제 데이터 준비/추론 bridge를 다루며 이미 존재하는 suffix 반환을 재사용한다. GPU, judge, 전체 데이터 다운로드, 나머지 benchmark adapter, retry/resume 시스템은 P1에서 만들지 않는다.

## 2. 컴포넌트와 실행 경계

추가 경로: `src/evaluation/schema.py`, `io.py`, `registry.py`, `report.py`, `benchmarks/__init__.py`, `benchmarks/vsp.py`; `scripts/__init__.py`, `scripts/score_eval.py`; `configs/eval/vsp_validated_v1.yaml`. 기존 경량 `evaluation/__init__.py`를 보존한다. 실제 함수 signature와 실패 테스트 코드는 승인 뒤 P1 implementation plan에서 고정한다.

새 scorer는 표준 라이브러리만 사용하고 CLI YAML에는 이미 설치된 PyYAML을 사용한다. torch/Transformers/CUDA/MathRuler/Reference import 또는 원본 source의 런타임 AST 실행은 금지한다. reference lock은 개발 감사용이며 scorer 사용자가 Reference를 설치해야 하는 조건이 아니다.

Python은 `./venv/ilvr-venv/bin/python`, import는 `evaluation.*`, 실행 환경은 `PYTHONPATH=src`를 사용한다. P1에서는 dependency 설치/업그레이드나 별도 Python 환경을 만들지 않는다. 다른 기존 latent 테스트까지 모두 torch-free라고 주장하지 않는다.

## 3. 데이터 계약

흐름은 manifest/prediction/config 검증 → gold를 읽지 않는 추출 → 전체 경로 문법 검증 → target 기반 전이/채점 → Score/summary 저장이다.

Sample은 schema version, namespaced `benchmark/subtask/source_id`, benchmark/subtask, ModelInput, Target, provenance를 갖는다. ModelInput은 원문 텍스트·순서 있는 이미지 상대 경로·선택지만 포함한다. 원문 문제 속 지도는 유지하고 trace/해설/정답 label은 넣지 않는다. P1 Target은 지도와 환경 규칙이며 gold 경로는 없어도 된다. 미구현 benchmark를 일반 scorer로 fallback하지 않는다.

provenance는 dataset ID/revision/config/split/source_id와 자산 hash를 기록한다. 합성 fixture는 synthetic identity를 명시한다. 이미지 root는 현재 `data/` 아래의 명시적 dataset root이며 열거한 이미지의 존재/hash를 검증한다. 실제 source의 진위나 image-map 대응을 P1 schema 검사만으로 증명하지 않는다.

완성된 manifest JSONL bytes의 SHA-256은 Prediction/run metadata에 둔다. Sample에 자기 manifest hash를 넣는 순환 정의는 금지한다. Prediction에는 id, manifest hash, run fingerprint, 원문 response, 생성 상태가 필요하다. fingerprint는 모델·공통 prompt_template_identity와 생성 설정의 명시적 metadata에서 계산하며 미제공 정보를 임의 상수로 만들지 않는다. 기존 응답이 이 정보를 갖지 않으면 자동으로 준비 완료라고 처리하지 않는다. P1이 metadata 내용의 사실성까지 검증하지는 않는다. 샘플별 prompt hash는 별도 optional metadata이며 같은 run의 공통 fingerprint를 바꾸지 않는다.

생성 status는 `ok` 또는 `infrastructure_error`다. `ok`에는 빈 문자열/문법 실패/모델 token-limit 종료도 포함한다. `infrastructure_error`는 error 정보를 요구하며 response가 있어도 채점과 coverage에서 제외한다. 다른 status는 입력 오류다. missing은 레코드 부재에서 계산한다. 모델 종료 이유는 별도 metadata다.

Score는 id, 추출 답안, 추출 방법/status, profile/version, correctness와 환경 종료 정보 또는 infrastructure 오류를 기록한다. 빈 manifest, 존재하지 않는 입력 파일, 잘못된 JSONL/ID, duplicate/unknown ID, fingerprint 불일치는 입력 오류다. 존재하는 빈 Prediction JSONL은 모든 예측이 누락된 미완료 평가로 처리한다. partial prediction도 아래 미완료 규칙을 따른다.

## 4. VSP 문법과 전이

production registry에는 `vsp_validated_v1`을 등록한다. 이는 수정 profile의 이름이며 원본 환경 검증 통과 표시가 아니다. `vsp_ilvr_legacy`는 현재 코드의 감사/비교 사례 이름으로 남기고 runtime profile로 이식하지 않는다.

추출은 마지막 balanced `\boxed{...}`를 우선한다. 마지막 boxed가 미완성이면 invalid이며 fallback하지 않는다. boxed가 없으면 줄 시작의 마지막 final-answer/action-plan marker를 사용하고, marker도 없으면 전체 응답을 사용한다. marker는 대소문자 무관 `final answer:`, `final answer is:`, `action plan:` 및 전각 colon이다. marker 뒤 전체 나머지를 후보로 사용한다.

허용 문법은 대소문자 U/D/L/R, ASCII 공백/tab/CR/LF, 비어 있지 않은 action group 사이의 comma다. 붙여쓴 action과 공백 구분을 허용하되 선행/후행/중복 comma, 따옴표, 마침표, 영어 방향명, 기타 문자는 거부한다. 빈 응답은 empty, 잘못된 문법/boxed는 invalid_format이다. 전이 전에 전체 후보를 검증하므로 `XR`/`RX`는 둘 다 invalid다.

지도는 numeric `0/1/2/-1` 또는 문자 `F/S/G/H`를 공통 표현으로 변환한다. 대응은 `0→F`(safe), `1→S`(start), `2→G`(goal), `-1→H`(hole)다. boolean, 알려지지 않은 셀, 빈/비직사각형 지도, 시작/목표가 각각 1개가 아닌 지도를 거부한다. 성공 기준에 shortest-path 조건을 추가하지 않는다.

전이는 deterministic이다. 경계 밖 이동도 step을 소비하며 제자리에 머문다. hole/goal에서 즉시 terminated. max_steps는 명시적 양의 정수이며 전이 뒤 elapsed step이 limit에 도달하면 truncated. goal과 limit가 같은 step이면 reward=1, terminated=true, truncated=true로 성공이다. 경로가 먼저 끝나면 도달 위치에서 실패/미종료다. Gym 0.26.2 FrozenLake-v1 대응 config에는 소스로 확인한 max_steps=100을 명시한다. 실제 Gym runtime과의 대조는 별도다.

## 5. 집계·오류·경로

분모는 N_expected이고 0개 manifest는 오류다. 빈/invalid 응답은 모델 실패로 포함한다. 누락 prediction/infrastructure error가 있으면 complete=false, 최종 accuracy=null. coverage는 사용 가능한 모델 응답 수/N_expected이며 빈/invalid 응답은 관측값에 포함하고 미수신/infrastructure error는 제외한다. 완전한 평가만 accuracy=correct/N_expected를 제공한다. P1에 provisional accuracy나 macro average는 추가하지 않는다.

P1 category는 Sample.subtask이며 config가 허용 목록을 명시한다. 빈 category는 complete=true, count=0, accuracy=null이고 전역 complete에 영향을 주지 않는다. 미정의 category는 preflight 오류다. category 내부에 누락/infrastructure error가 있으면 그 category의 complete=false, accuracy=null이다. CLI는 완료 0, 입력/config 오류 2, 미완료 1이다. 입력 오류에는 최종 점수를 쓰지 않고 미완료에는 상태를 표시한 보고서를 쓴다. 고유 output 디렉터리를 요구하고 기존 결과를 덮어쓰지 않는다.

데이터·이미지·manifest·데이터 fixture/cache는 `data/evaluation/`; 응답·점수·표는 `results/evaluation/<run-id>/`; 실행 로그는 `logs/evaluation/<run-id>/`다. 기존 `data/` symlink를 유지한다. P1은 모델 파일을 만들지 않는다.

## 6. 근거와 검증

현재 로컬 함수 조각 14개 관찰에서 세 VSP simulator의 R 성공/RD 실패/XR·RX 성공과 두 extractor의 truthy `"None"`을 확인했다. 이는 전체 CLI나 원본 환경 실행이 아니다. 기존 scorer 11개 테스트는 명시적 `PYTHONPATH=src`에서 통과했다. 신규 evaluator 테스트는 아직 없다.

P1 TDD는 지도만 있는 입력, R/RD/XR/RX, final-answer/boxed/empty, hole/경계/limit, 지도/ID/image/fingerprint 오류, model failure와 infrastructure/missing 구분, 분모/빈 category를 포함한다. 원래 source tree/Reference 없이 새 모듈과 최소 fixture만 둔 subprocess에서 import/재채점 smoke를 수행한다. 기존 scorer 테스트도 포함한다. 이 테스트들은 구현 계약 검증이며 독립 benchmark ground truth가 아니다.

## 7. 비평 합성과 위험 수용 요청

**제안 대비 변경:** 동결 제안 v2와 규칙이 같다. v1 자기검토에서 빈 prediction 파일의 입력 오류 규칙과 모든 prediction 누락의 미완료 규칙이 충돌함을 발견해, v2는 빈 prediction을 미완료로 처리한다. 샘플별 prompt hash와 공통 run identity를 분리했고, cold read가 지적한 지도 mapping/category/생성 status/계획 승인 경계를 명확히 했다. 범위와 P1/P2 선후관계는 유지한다. map 복구 후보 관찰과 아래 미해결 이견/위험 상태를 함께 기록한다.

**hate root:** 새 simulator와 자체 fixture만의 상호 일치를 원본 정확성으로 오인할 수 있다. **first_nail:** pinned Gym 0.26.2의 make/reset/step을 작은 지도와 경계/hole/100-step 사례에서 관찰하고 구현 뒤 새 simulator와 비교한다. **상태: not_run.** 현재 Gym이 없고 원본은 Python 3.12.2/NumPy 1.26.4이며 환경 변경은 승인되지 않았다. 원본 소스 확인과 현재 14개 관찰은 이 실험을 대체하지 않는다.

**macrothink의 남은 이견:** 두 fresh read는 실제 VSP record/manifest 복구를 scorer보다 먼저 확정하라고 제안했다. 읽기 전용 후속 조사에서 원본 6개 level별 img/table/pure_text basename 집합(각 100개)이 일치하고 level5/0의 좌표 설명 한 쌍이 일치함을 확인했다. 안정적 `level/id`와 map 복구의 후보 경로는 있지만, 기존 prediction과의 완전한 join/전체 이미지 대응/논문 split은 입증하지 못했다. 이 사실로 착수 순서 이견이 해결되었다고 주장하지 않는다.

**prism:** 과학적 적용 가능성은 unclear, 기존 기능 호환성은 설계 수준에서 pass다. 판단을 가르는 질문은 P1을 합성 입력으로 검증한 계약/scorer까지로 승인하고 실제 benchmark 준비 완료는 P2 이후로 남길지 여부다.

**미검증 전제:** 새 전이 구현의 원본 환경 일치, 실제 manifest/prediction 연결, 원본 map-image 내용의 전수 대응, 논문 split/judge/집계 동일성, GPU 생성 동등성. source-known 규칙과 수정 parser 선택은 구현 후에도 구별한다. CoMT/EMMA는 라이선스 파일을 찾지 못해 후속 소스 복사를 보류한다.

**명시적 risk waiver: 2026-09-10 사용자 승인 완료.** 설계 v2 및 해당 한계를 유지한 상세 계획 진행 여부를 묻는 요청에 사용자가 `Ok`로 응답했다. 승인은 원본 환경·실데이터 검증을 완료했다고 인정하는 것이 아니라, 그 상태를 blocked/not_run으로 공개한 채 P1 CPU 계약/scorer 설계를 상세 계획으로 진행하는 범위에 한정한다. dependency 변경·유료 호출·논문 재현 인정은 포함하지 않는다.

## 8. 다음 단계

이 commitment의 인간 승인 뒤 `superpowers:writing-plans`로 P1의 정확한 파일·타입·함수·실패 테스트 코드·명령을 작성한다. 승인 전 구현 소스/새 scorer/CLI는 만들지 않는다. 계획 작성 후 산출물을 보고하고, 사용자의 계획 승인 뒤 구현한다.

구현 승인 기록: 2026-09-10 사용자가 상세 계획을 보고 `ㅇㅋ`로 승인했다. P1 Task 1–5 구현 및 계획에 명시한 검증을 진행한다.

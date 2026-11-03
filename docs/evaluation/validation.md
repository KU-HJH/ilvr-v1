# P6 Zebra-CoT 2D 최종 오프라인 검증

실제 400행 준비는 exit 0으로 완료됐다. 산출물은
`data/evaluation/zebra-2d400-v1`, manifest SHA-256은
`966453f6973e68a6a426f00c0fd83bd495a7390f9a9da01e9413e866646f4d5b`다.
source selection과 records digest는 각각
`92e776e6b7e7e646c7f3b23c05c87b0993af44821faecac789a0e29ef2737927`,
`1a533ce69f426cce9397251df976cf4331118f530b342713747f2ebdae3d4f25`다.

독립 전수 검사는 canonical ID로 원본과 manifest를 join해 400개의 Question,
target, config/split/source_id, 이미지 상대 경로, 원본·복사본 bytes와 SHA-256을
모두 비교했다. Jigsaw/Search 각 200개, 이미지 400개, 제외 0개였다.
[전수 결과](../../results/evaluation/p6-implementation-20260914-01/prepared-correspondence.json)와
[준비 요약](../../results/evaluation/zebra-2d400-v1/prepare.json)에 결과를 보존했다.
최종 수정 후 Zebra 집중 검사는 **83 passed in 0.80s**였다. 최종 통합 CPU
검사는 **633 passed in 8.35s, exit 0**이다. `tests/evaluation/test_*.py`를
파일명 순으로 선택하되 `test_continuous_latent.py`와
`test_inference_bridge_runtime.py`는 제외했다. 이 결과는 P1–P6의 오프라인
CPU/stub 계약이며 저장소 전체 테스트나 GPU 검증이 아니다.
[정확한 명령·종료 코드·검증 결과](../../results/evaluation/p6-final-20260914-02/verification.json),
[CPU 전체 출력](../../logs/evaluation/p6-final-20260914-02/cpu.log),
[최종 실행 스크립트](../../logs/evaluation/p6-final-20260914-02/verify.py)를 보존했다.

최종 리뷰의 JPEG 검증 누락은 실제 pixel load를 수행하도록 수정했다. 기존
성공 fixture가 header verify만 통과하고 pixel load에는 실패함을 재현했고,
정상 JPEG fixture와 digest를 갱신한 손상 JPEG 회귀 사례로 수정했다. 외부
경로를 삭제 대상으로 등록하던 테스트 정리 코드도 fixture UUID 경로로 제한했다.
두 원인 회귀 검사는 RED 후 GREEN이며, 네 리뷰 지적 모두 재검토에서 해결됐다.
[집중 검사](../../logs/evaluation/p6-final-fix-20260914-01/focused-zebra-tests.log)와
[원인 회귀 검사](../../logs/evaluation/p6-final-fix-20260914-01/green-regressions.log)는
모델 점수 증거가 아니다.

수정된 준비 CLI로 실제 400개를 별도
`data/evaluation/p6-final-prepared-20260914-01/`에 다시 준비했다. 전체 이미지
pixel decode를 포함한 preflight가 통과했고 manifest hash는 위 기존 산출물과
같았다. 기본 config는 보존한 `data/evaluation/zebra-2d400-v1/`를 계속 가리키며,
최종 공통 reader가 이 manifest 400행과 이미지 hash를 읽어 확인했다.
[재준비 요약](../../results/evaluation/p6-final-prepared-20260914-01/prepare.json).

보호 파일 5,055개가 불변이며 Reference 7개의 독립 root/origin/commit/clean
상태와 lock의 evidence/license hash 33개가 일치했다. 원본 두 digest와 동결
proposal digest, 기존 data/checkpoints symlink도 유지됐다. P6 변경의 whitespace
검사는 통과했다. 전체 index에는 기존 P5 pytest 기록의 trailing whitespace가
남아 있으며 보호 파일이므로 수정하지 않았다. 테스트 후 변경은 새 Zebra 테스트
파일 끝의 빈 줄 하나 제거와 완료 문서 갱신뿐이다.

첫 최종 실행 `p6-final-20260914-01`은 632 passed, 1 setup error였다. 검증
스크립트의 pytest basetemp 부모가 없어 발생했으며 제품 코드 실패가 아니었다.
[실패 로그](../../logs/evaluation/p6-final-20260914-01/cpu.log)를 보존하고,
기존 방식대로 data/evaluation 바로 아래의 새 basetemp를 사용했다. 실패한
단일 사례 1 passed를 확인한 뒤 새 ID `p6-final-20260914-02`로 위 633개를
재실행했다. 추가 broad review나 제품 코드 수정은 없었다.

이 검증은 고정 source projection에서 prepared manifest로의 대응 증거다. 원본
51,899행 전체 parquet bytes, gold의 독립 재풀이, 모델 응답·정확도, OOD identity,
논문 sample identity/judge 계약, GPU inference는 검증하지 않았다. 따라서 상태는
`protocol_kind=modified`, `ood_verified=unknown`,
`paper_protocol_verified=false`이며 실제처럼 보이는 모델 점수는 기록하지 않았다.


# P1 검증 결과

2026-09-10, 작업 위치 `.worktrees/evaluation-p1`, 기준 ILVR `2497ee7bafad1b51ef6c2aba6f40f084830916c4`.
공통 데이터 계약·VSP 오프라인 채점·입력 검증·집계·CLI를 구현했다. 데이터는 기존 `data/` symlink의 같은 저장 위치를 사용하고, 결과와 로그는 이 작업 트리의 `results/`, `logs/`에 저장한다.

## 최종 실행

아래 명령을 작업 트리 루트에서 한 번 실행했다. **105 passed in 0.23s, exit 0**.
새 P1 테스트 94개와 기존 scorer 호환성 테스트 11개다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest \
  tests/evaluation/test_offline_schema.py \
  tests/evaluation/test_vsp_offline.py \
  tests/evaluation/test_offline_io.py \
  tests/evaluation/test_offline_report.py \
  tests/evaluation/test_offline_cli.py \
  tests/evaluation/test_scoring.py -q -p no:cacheprovider
```

[전체 출력](../../logs/evaluation/p1-final-20260910-01/tests.log), [종료 코드](../../logs/evaluation/p1-final-20260910-01/tests.exit-code), [명령·환경·결과](../../results/evaluation/p1-final-20260910-01/tests.json).

| 검증 | 관찰 결과 |
|---|---|
| schema | 4 passed |
| VSP 추출·문법·전이 | 28 passed |
| JSONL·ID·hash·이미지·경로 입력 검증 | 43 passed |
| 분모·범주·누락·인프라 오류 집계 | 8 passed |
| CLI·저장·복사본 실행 | 11 passed |
| 기존 scorer 호환성 | 11 passed |

복사본 테스트는 새 Python 파일 9개의 정확한 목록을 확인하고 기존 추론 소스와 `Reference/` 없이 실행했다. child에서 금지된 heavy/legacy import를 차단하고, 로딩된 `evaluation` 관련 모듈과 `scripts.score_eval` 8개의 실제 경로가 복사본 안인지 확인했다. 합성 샘플 2개의 `N_expected=2`, `correct=2`, `accuracy=1.0`, `complete=true`도 검증했다. [실제 모듈 경로와 child 출력](../../logs/evaluation/p1-subprocess-93661c3cfa584eaabe7337182a62189c/copied-workspace.log). 이는 실행한 경로의 의존성 증거이며 모든 미실행 경로의 일반 증명은 아니다.

## 문서 CLI 예제

[프로토콜 문서](protocols.md)의 quick-start 명령을 그대로 실행해 exit 0과 합성 예시 2/2를 확인했다. 이 수치는 실제 벤치마크 성능이 아니다.

- [입력 manifest](../../data/evaluation/p1-test-3535bc552fe642dd94da0df518c78e29/manifest.jsonl), [저장 응답](../../results/evaluation/p1-test-3535bc552fe642dd94da0df518c78e29/predictions.jsonl), [실행 config](../../data/evaluation/p1-test-3535bc552fe642dd94da0df518c78e29/config.yaml)
- [채점 summary](../../results/evaluation/p1-quickstart-score-e07217fe2b2a4c38b16f7394f0e8ca40/summary.json), [실행 로그](../../logs/evaluation/p1-final-20260910-01/quickstart.log), [실행 기록](../../results/evaluation/p1-final-20260910-01/quickstart.json)

## 보존과 검토

기준 Python 파일 9개의 SHA-256이 유지됐고, 기존 tracked 파일 변경은 없었다. 7개 참조의 독립 root·origin·commit·clean 상태와 근거 파일 25개 및 라이선스 파일 5개의 hash가 lock과 일치했다. [보존 검증](../../results/evaluation/p1-final-20260910-01/preservation.json).

다섯 작업의 요구사항/품질 검토와 전체 변경 검토를 완료했다. 검토 중 발견한 누락 경로 검증·dataset root 확인·Unicode JSONL framing·정규화 후 내부에 남는 이미지 경로를 회귀 테스트로 수정했다. 전체 검토의 마지막 두 결함은 [RED 3 failed/40 passed](../../logs/evaluation/p1-final-fix-red-20260910-01/pytest.log) 후 [GREEN 43 passed](../../logs/evaluation/p1-final-fix-green-20260910-01/pytest.log)로 확인했다.

CLI 복사본 검사의 파일 목록·전체 모듈 경로 확인과 프로토콜 설명도 보완했다. 보완 중 디렉터리를 파일 목록에 포함한 테스트 오류는 [실패 로그](../../logs/evaluation/task-5-green-20260910-2/pytest.log)에 보존돼 있고, 수정 후 최종 105개 검사에 포함됐다. 초기 Task 5 RED는 agent tool transcript에서만 관찰됐으며 별도 파일 로그가 없다. 과거 로그를 새로 생성한 것으로 꾸미지 않는다.

## 확인하지 않은 범위

| 범위 | 상태 |
|---|---|
| 원본 Gym runtime과 직접 비교 | not_run: Gym/Gymnasium 미설치, 환경 변경 범위 밖 |
| 실제 VSP source→manifest→prediction 연결 | not_run: P1은 합성 계약 검증 |
| GPU 생성 동작 | not_run |
| 실제 전체 benchmark | not_run |
| paper protocol | false / 미검증 |
| 전체 프로젝트 학습·모델 테스트 | not_run: 이번 P1 범위의 통과와 구분 |

소스 hash 보존은 원본 CLI 전체 실행 동등성을 입증하지 않는다. 합성 fixture와 새 scorer의 일치는 독립 benchmark ground truth가 아니다. 실제 입력 연결과 원본 환경 대조는 후속 단계에서 별도 증거가 필요하다. 구현은 `codex/evaluation-p1` 작업 트리에 있으며 commit·merge·push는 수행하지 않았다.

# P2 VSP 추론 연결 검증

## 준비된 실제 소스 데이터

고정한 Visual-Spatial-Planning commit
`6d74d7bb46ae71bf41aa4d3d874a4f31fc5f0366`에서 준비 CLI가 exit 0으로
완료됐다. 이미 완성된 목적지를 덮어쓰거나 prepare를 반복하지 않았다.

- `data/evaluation/vsp-pinned-6d74d7bb46ae-v1/manifest.jsonl`: 600행,
  level3–level8 각 100행
- 같은 root의 `smoke-manifest.jsonl`: 6행, 각 level의 source ID 0 한 개
- full manifest SHA-256:
  `35d2396a51bc0e6592915452591c2270dc80c37dc7384ea8d7e99ecf61621cba`
- smoke manifest SHA-256:
  `d2fd2718228d63ddea8a40456c1a25af0e894a01b54c068681964118aaef4c96`
- prompt identity:
  `vsp-ilvr-three-images-v1:8034d008ea5ca5e693c73b5bc8a079e2aa55a1270bbe7e5a4e78c706037af03d`

[prepare 요약](../../results/evaluation/vsp-pinned-6d74d7bb46ae-v1/prepare.json)과
[prepared manifest 검증](../../results/evaluation/vsp-pinned-6d74d7bb46ae-v1/prepared-validation.json)은
`read_manifest`로 전체 순서 ID, 범주 수, 각 샘플의 세 asset hash, Reference
비의존성을 확인한 결과다. [전체 이미지 preflight](../../results/evaluation/vsp-pinned-6d74d7bb46ae-v1/full-image-preflight.json)는
600개 입력이 참조하는 602개 distinct PNG를 모두 decode했고 heavy import가
없었다고 기록한다. 이는 이미지와 map 의미가 600개 모두 맞다는 독립 검증은
아니다. 원본 이미지 의미의 직접 대조 범위는 level별 source ID 0, 즉 6/600이다.

## Reference-free 복사본 파이프라인

Task 5 GREEN은 다음 새 lightweight 테스트만 한 번 실행해 **105 passed in
1.74s, exit 0**을 관찰했다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. \
  ./venv/ilvr-venv/bin/python -m pytest \
  tests/evaluation/test_manifest_preflight.py \
  tests/evaluation/test_vsp_prepare.py \
  tests/evaluation/test_inference_bridge.py \
  tests/evaluation/test_run_eval.py \
  tests/evaluation/test_vsp_pipeline.py \
  -q -p no:cacheprovider
```

[GREEN 로그](../../logs/evaluation/task-5-green-20260910-01/pytest.log).
RED는 새 config가 없을 때 소비 테스트가 예상대로 실패하고 복사본 통합
테스트는 통과한 [1 failed, 1 passed 기록](../../logs/evaluation/task-5-red-20260910-02/pytest.log)이다.

복사본 테스트는 Python 소스 12개와 준비된 합성 asset만 `data/evaluation/`
아래의 격리 root에 복사했다. 그 root의 `data`, `results`, `logs`,
`checkpoints`는 활성 저장 root를 가리키는 symlink다. Reference, 기존
`eval.py`, torch, Transformers, weight 파일을 복사하거나 import하지 않았다.
subprocess에 주입한 test-only `eval`만 weight 경계에서 두 literal 응답과
suffix ID를 반환했고, 실제 `scripts.run_eval.main`과
`scripts.score_eval.main`이 preflight와 출력 처리를 수행했다.

[복사본 실행 근거](../../results/evaluation/vsp-pipeline-19a080b6704d4060aa4ebce28f6d6ce2-evidence/pipeline-evidence.json)는
로드한 12개 모듈의 실제 origin, 모델에 전달된 정확한 key
`image_input`/`text_input`, legend→worked example→target 순서의 절대 이미지
경로, generation config를 기록한다. 성공 경로는 generation/score exit 0,
`N_expected=2`, `observed=2`, `correct=2`, `accuracy=1.0`, 두 suffix
`[101,102]`/`[201,202]`, manifest hash와 run fingerprint 일치를 확인했다.
통제된 첫 generation 실패 경로는 generation/score exit 1,
`infrastructure_error=1`, `missing=1`, `complete=false`, `accuracy=null`을
확인했다. 이 2/2는 fixture 계약 증거이며 실제 VSP 모델 성능이 아니다.

## 실제 GPU 여섯 문제 smoke

root controller가 H100 NVL에서 고정된 Stage1 adapter/base와 범주별 1문제,
총 6개에 대해 별도 실행을 완료했다. 먼저 direct 기존 함수와 bridge 호출의
`prompt_identity`, `prompt_token_length`, `generated_suffix_token_ids`,
`assistant_only`, `stop_reason` 다섯 필드가 6쌍 모두 같았다.
[비교 결과](../../results/evaluation/p2-gpu-parity-20260910-02/comparison.json)는
greedy, seed 1, `max_new_tokens=128`과 실제 runtime/metadata hash를 기록한다.
관찰된 Qwen model forward call 1,460회는 모두 `use_cache=False`였고 전달된
`past_key_values`가 없었다. paired token/output은 일치했다. 이는 가능한 모든
모델 내부 상태가 호출 사이에 초기화된다는 일반 증명이 아니며, 한 Stage1
checkpoint와 고정 6입력에 한정된다.

같은 128-token data config로 실제 generation CLI와 score CLI도 각각 exit
0이었다. [CLI 검증](../../results/evaluation/p2-gpu-cli-20260910-01/validation.json)은
6 prediction, 6 detail, ID/hash/fingerprint 일치와 이전 bridge output 일치를
기록한다. [score summary](../../results/evaluation/p2-gpu-score-20260910-01/summary.json)는
`observed=6`, `correct=0`, `accuracy=0.0`, `invalid_format=6`이다. 여섯 응답
모두 128 token 제한에서 `max_new_tokens`로 종료했으며 정상적인 모델 관찰로
보존했다. 이는 6문제 smoke 결과이며 600문제 benchmark, 기본 4096-token
config, paper protocol 성능이 아니다.

P1 scorer의 raw summary 안 `verification.gpu_smoke`는 실행 provenance를
추론하지 않는 고정 `not_run` placeholder라 그대로 유지했다. 위
`validation.json`과 `comparison.json`이 실제 GPU 실행 근거다. 기본 config의
4096-token 실제 실행과 600개 full benchmark, 원본 Gym 대조, paper protocol
검증은 수행하지 않았다.

## P2 최종 검증과 완료 상태 — 2026-09-10

P2의 prepare → 기존 ILVR 추론 bridge → P1 score 구현과 승인된 6문제 GPU
smoke를 완료했다. 전체 변경 리뷰의 두 지적은 한 번의 수정과 범위 한정
재리뷰로 해소했다. 공백뿐인 예외 메시지는 예외 클래스명으로 기록하여 P1이
infrastructure_error로 읽을 수 있게 했고, 실제 recipe의 계약 테스트는
실제 준비 데이터의 존재 대신 합성 fixture를 사용하게 했다. 정상 생성 경로는
변경하지 않았으며 GPU 실험은 반복하지 않았다.

| 최종 증거 | 관찰 결과 |
|---|---|
| P1/P2 CPU 계약 테스트 | **212 passed in 1.98s**, exit 0 |
| 별도 torch runtime 스텁 및 기존 실행 경로 테스트 | **7 passed in 8.80s**, exit 0; 실제 weight/CUDA 실행 아님 |
| 구현 전 130개 파일 hash | 공통 IO 추출 대상 `src/evaluation/io.py`만 변경 |
| 기존 tracked 코드 및 실제 editable Transformers 3개 파일 | HEAD bytes와 일치 |
| Reference 7개 독립 root/origin/commit/clean/hash | evidence 28개, license 5개 모두 일치 |
| data/checkpoints 저장 링크 | 기존 `/raid/workspace/ILVR/` 목적지 보존 |

실행 명령은 아래 두 driver이며 전체 pytest 인자·종료 코드는
[최종 verification.json](../../results/evaluation/p2-final-20260910-02/verification.json)에 있다.

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv/ilvr-venv/bin/python logs/evaluation/p2-final-20260910-01/verify.py
PYTHONDONTWRITEBYTECODE=1 ./venv/ilvr-venv/bin/python logs/evaluation/p2-final-20260910-02/verify.py
```

첫 driver의 [CPU 로그](../../logs/evaluation/p2-final-20260910-01/cpu.log)는
212개 통과다. 이어진 스텁 실행은 검증 driver가 `--basetemp` 부모 디렉터리를
만들지 않아 4 passed, 3 setup errors로 종료했다. 제품 코드 실패가 아니며
[실패 로그](../../logs/evaluation/p2-final-20260910-01/runtime-stubs.log)를 보존했다.
부모 생성 한 줄과 [원인 재현 검증](../../logs/evaluation/p2-final-20260910-02/path-regression.log)을
추가하고 새 run ID로 스텁 7개와 아직 실행되지 않은 보존 검증만 수행했다.
[스텁 로그](../../logs/evaluation/p2-final-20260910-02/runtime-stubs.log)와
[보존 검증 로그](../../logs/evaluation/p2-final-20260910-02/verification.log)가
그 결과다. 이미 통과한 CPU suite와 GPU 실험은 반복하지 않았다.

실제 GPU 결과는 앞 절의 6쌍 출력 일치 및 0/6 smoke 점수에 한정한다.
원본 Gym 실행 비교, 600문제 모델 평가, 기본 4096-token 실행, paper protocol,
전체 프로젝트 학습·분산 테스트는 미실행이다. 다음 migration 대상은 CoMT,
VisuLogic, EMMA와 Zebra-CoT 2D이며 이 P2 완료에 포함하지 않는다.
구현과 검증 기록은 `codex/evaluation-p1` worktree에 보존했고 commit·merge·push는
수행하지 않았다.

# P3 CoMT 오프라인 구현 검증 — 2026-09-11

P3는 `comt_reference_rules_v1` 채점 경로에 고정된 TEST 400개 준비 CLI와
오프라인 recipe를 연결했다. 아래 합성 fixture의 채점 수치는 실제 모델
성능이 아니다. 실제 `ilvr-test400-corrected-v1` 400행 준비와 별도 재검증은
완료했다. 최종 P1/P2/P3 회귀 및 보존 검증은 아래 후속 기록과 구분한다.

## 구현 단계에서 직접 관찰한 범위

Task 3의 RED는 새 prepare 인자가 인식되지 않고 recipe가 없어서 **7 failed,
1 passed**였다. 최소 dispatch/config 구현 뒤 focused 실행은 **8 passed**였다.
검사는 CoMT에서 VSP source verifier를 호출하지 않는 것, CoMT 전용 두 경로의
조건부 필수성, VSP에서 그 경로를 거부하는 것, 미등록 profile의 exit 2,
recipe parsing, VSP 전용 `run_eval`의 모델 loader 이전 CoMT exit 2를 확인했다.

Reference 없는 복사본 통합 검사는 합성 source/selection/PNG만 활성 `data/`
아래의 격리 저장소에 만들고 lightweight Python 파일만 복사했다. fixture hash는
subprocess 안에서만 고정값으로 대체했다. 실제 `prepare_eval.main`과 변경하지
않은 `score_eval.main`을 호출해 400행, 네 범주 각 100행, 두 보정 target,
manifest hash와 module origin을 검사했다. target에서 계산하지 않은 literal
응답은 일반 행 B, 두 보정 행은 명시적으로 B와 D를 사용했다. 완전한 fixture는
400/400과 exit 0, 한 Prediction을 제거한 두 번째 채점은 exit 1,
`complete=false`, `accuracy=null`을 확인한다. torch, Transformers, `eval.py`,
Reference, 모델 weight, GPU는 이 통합에 사용하지 않는다.

RED 전체 출력은
`logs/evaluation/task-3-red-20260911-01/pytest.log`, focused GREEN 출력은
`logs/evaluation/task-3-green-focused-20260911-02/pytest.log`에 보존했다. Task 3
최종 combined gate는 **149 passed in 4.33s**, exit 0이었다
([로그](../../logs/evaluation/task-3-combined-20260911-01/pytest.log)).
리뷰 뒤 통합 검사를 400개 전체의 ID·순서·정확한 해시·provenance·run metadata로
강화했고 focused **8 passed**를 확인했다
([로그](../../logs/evaluation/task-3-fix1-focused-20260911-01/pytest.log)).

## 검증 수준과 남은 범위

원본 CoMT 함수와의 최종 분리 비교는 [35종 응답 × 4개 target 기록](../../results/evaluation/p3-upstream-function-20260911-04/comparison.json)에서
새 함수·upstream 함수·literal 예상값 **140개**와 upstream integer-target **140개**가
일치했다. [실행 로그](../../logs/evaluation/p3-upstream-function-20260911-04/comparison.log)를 보존했다.
초기 24종 비교가 놓친 연속 공백/부분 토큰 및 Unicode `casefold` 차이를 실제로
재현한 뒤 exact-letter membership과 원본 `lower()` 의미로 수정했다.
실패 증거도 [토큰 경계](../../results/evaluation/p3-upstream-function-20260911-02/comparison.json)와
[Unicode 경계](../../results/evaluation/p3-unicode-boundary-20260911-01/comparison.json)에 남겼다.
이 비교는 분리된 함수의 열거 사례에 한정하며 전체 upstream 환경·CLI·모든 입력의
동등성이나 모델 성능을 증명하지 않는다.

| 검증 수준 | 상태 |
|---|---|
| CoMT 단위 계약과 합성 400행 prepare→score | 구현 단계에서 실행 |
| source 함수 35×4 비교 | 140개 + integer-target 140개 일치; 제한된 함수 증거 |
| 실제 corrected TEST 400 준비 및 재검증 | exit 0; 400개 ID/이미지 hash 일치, 제외 0 |
| 실제 모델 응답 / CoMT generation | 미실행, 이 recipe가 지원하지 않음 |
| 전체 400개 모델 성능 | 미실행 |
| paper protocol / upstream 전체 환경 | 미검증 |
| 전체 gold의 독립 재풀이 및 모든 학습 데이터 OOD | 미검증 |

공백/형식 실패를 관측된 오답으로 처리한 결과와 누락/인프라 오류로 accuracy가
null인 결과를 구분한다. 합성 400/400은 pipeline 배선 증거일 뿐 benchmark
accuracy가 아니다. 실제 응답을 채점할 때는 checked-in template의 모든
run metadata를 실제 생성 identity/settings로 바꾸고 P1 Prediction hash 계약을
일치시켜야 한다.

## 실제 TEST400 준비 증거

[준비 요약](../../results/evaluation/comt-ilvr-test400-corrected-v1/prepare.json)과
[별도 대조 결과](../../results/evaluation/comt-ilvr-test400-corrected-v1/prepared-validation.json)는
네 범주 각 100개, 원본 순서/ID 400개, pinned 원격 LFS 이미지 hash 400개의 일치를
기록한다. 398개 text는 동일하고 `creation-12125`(B/18), `creation-12164`(D/24)만
승인된 새 배열/text/target으로 보정했다. 원본 TEST/TRAIN/source hash는 불변이며
새 모델 응답 생성 수는 0이다. manifest SHA-256:
`64d2488befb7659144fb4fb4c7ad50d4d642245e45cc4e6afd2b91a5d6578de7`.

산출물은 `data/evaluation/comt-ilvr-test400-corrected-v1/`의 manifest와 출처·보정
기록 4개 파일이다. 실행은 [prepare 로그](../../logs/evaluation/comt-ilvr-test400-corrected-v1/prepare.log),
독립 대조는 [검증 로그](../../logs/evaluation/p3-prepared-validation-20260911-01/validation.log)에 있다.

## P3 최종 완료 검증

P3 구현·실제 TEST400 준비·제한된 원본 함수 대조를 완료했다. 변경 전체 리뷰에서
남은 수정 요구는 없었고, 아래 최종 검증을 한 번 수행했다.

| 증거 | 결과 |
|---|---|
| P1/P2/P3 경량 CPU 계약·CLI 통합 | **307 passed in 4.67s**, exit 0 |
| 별도 torch 실행 경로 스텁 | **7 passed in 8.91s**, exit 0; 모델/CUDA 실행 아님 |
| 보호 코드·원본 데이터 17개 / 기존 테스트 77개 hash | 모두 불변 |
| Reference 7개 root/origin/commit/clean, evidence 28개 + license 5개 hash | 모두 일치 |
| 동결 설계 proposal / data·checkpoints 링크 | 보존 |

전체 명령·종료 코드는 [verification.json](../../results/evaluation/p3-final-20260911-01/verification.json),
출력은 [CPU 로그](../../logs/evaluation/p3-final-20260911-01/cpu.log),
[스텁 로그](../../logs/evaluation/p3-final-20260911-01/runtime-stubs.log),
[보존 검증 로그](../../logs/evaluation/p3-final-20260911-01/verification.log)에 있다.

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv/ilvr-venv/bin/python logs/evaluation/p3-final-20260911-01/verify.py
```

실제 CoMT 모델 응답 생성·400개 모델 성능 평가·paper protocol 검증은 미실행이다.
전체 프로젝트 학습/분산 suite 또는 upstream 환경 전체 재현을 통과했다고 주장하지
않는다. 다음 남은 migration 구현은 P4 VisuLogic, P5 EMMA, P6 Zebra-CoT 2D다.
구현은 기존 `codex/evaluation-p1` 작업 트리에 보존하며 commit·merge·push는 수행하지 않았다.

# P4 VisuLogic 오프라인 평가 — 2026-09-13

사용자가 승인한 P4 설계 v1을 기존 MCQ/준비·채점 CLI에 추가했다. 새 모델 생성과
judge 호출은 포함하지 않는다. [사용 계약과 명령](protocols.md#p4-visulogic-rules-only-recipe)을 따른다.

실제 데이터는 [준비 요약](../../results/evaluation/visulogic-pqs579-v1-r02/prepare.json)과
[구현 전 조사에 대한 독립 대조](../../results/evaluation/visulogic-pqs579-v1-r02/prepared-validation.json)에서
579개 원본 ID/순서·질문·선택지/정답·이미지 해시 일치를 확인했다. Position 136,
Quantity 353, Style 90개이며 선택 범주 안의 제외·gold 보정은 0이다. 다른 세
범주의 421개는 이 평가 범위 밖이다. 전체 gold의 의미적 정확성이나 모든 학습
자료에 대한 OOD 검증은 수행하지 않았다.

최초 실제 준비는 PNG 확장자에서 PNG 인코딩을 가정하여 입력 오류로 중단됐다.
원인은 [PNG 541/JPEG 38개 인코딩 조사](../../results/evaluation/p4-implementation-20260913-01/source-image-formats.json)로
확인했고, JPEG 바이트를 `.png` 경로에 넣는 원인 회귀 테스트를 RED→GREEN으로
실행했다. 인코딩 제한만 제거한 후 새 `-r02` run ID로 성공했다. 원본 바이트는
변환하지 않았다. [실패 로그](../../logs/evaluation/visulogic-pqs579-v1/prepare.log)와
[성공 로그](../../logs/evaluation/visulogic-pqs579-v1-r02/prepare.log)를 보존했다.

[분리된 원본 함수 대조](../../results/evaluation/p4-implementation-20260913-01/source-comparison.json)는
51종 응답 × 4개 target, **204개 판정**에서 일치했다. 유효 답의 `box/lang`
method도 확인했다. 빈 응답의 진단 표현은 의도적으로 다르다. 비교 전 pinned
VLMEvalKit root/origin/commit/clean/hash를 검사했고 원본 모듈이나 GPT를 실행하지
않았다. 이 결과를 모든 입력의 동등성이나 upstream 전체 환경 실행으로 일반화하지 않는다.

합성 데이터 통합은 Reference/torch/Transformers/eval.py 없이 격리 복사본에서
실제 prepare/score CLI를 실행한다. 579개 전체 ID/순서, source question, 선택지,
정답, provenance/image hash, manifest hash/run fingerprint를 확인한다. 고정 B
응답으로 579/579가 되는 fixture는 소프트웨어 연결 증거이며 모델 성능이 아니다.
하나 누락 시 578개 관측, 전체 분모 579, accuracy=null과 exit 1을 확인한다.
유효 VisuLogic 입력의 VSP 전용 run CLI 거부와 모델 로더 호출 0회도 검증한다.

구현 단계 로그는 `logs/evaluation/p4-implementation-20260913-01/`의
`scorer-red.log`, `scorer-green.log`, `scorer-green-02.log`, `prepare-red.log`,
`prepare-green.log`, `pipeline-red.log`, `pipeline-green.log`,
`image-format-red.log`, `image-format-green.log`에 있다. 최초 scorer GREEN에서
테스트가 기존 summary의 nested verification 필드를 top-level로 읽은 오류는
테스트 접근만 고쳤으며 report 계약은 변경하지 않았다. 최종 리뷰 및 전체 경량
회귀·보존 검증은 아래 완료 기록으로 별도 남긴다.

## P4 최종 완료 기록

변경한 13개 파일의 독립 리뷰에서 수정 요구가 없었고,
[리뷰 기록](../../logs/evaluation/p4-implementation-20260913-01/review.md)을 보존했다.
그 뒤 P1/P2/P3/P4 경량 CPU 계약·CLI 통합 **365 passed in 6.39s**, 별도 실행
경로 torch 스텁 **7 passed in 11.33s**, 두 명령 exit 0을 확인했다.
[전체 명령·종료 코드 및 보존 검사](../../results/evaluation/p4-final-20260913-01/verification.json),
[CPU 출력](../../logs/evaluation/p4-final-20260913-01/cpu.log),
[스텁 출력](../../logs/evaluation/p4-final-20260913-01/runtime-stubs.log),
[보존 검사 출력](../../logs/evaluation/p4-final-20260913-01/verification.log)을 참조한다.

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv/ilvr-venv/bin/python logs/evaluation/p4-final-20260913-01/verify.py
```

허용한 변경 파일을 제외한 코드·테스트·설정과 기존 원본 데이터/실제 editable
Transformers를 포함하는 보호 목록 3,572개의 hash가 불변이다. Reference 7개의
독립 root/origin/commit/clean, evidence/license hash 33개, 기존 data/checkpoints
심볼릭 링크도 확인했다. 실제 579행 manifest SHA-256은
`4ae97316ce2b2d38ba5fccf0296c7398b9b1e81dcc39d8487dfe9de17713a203`이다.

이로써 P4 오프라인 채점·실데이터 준비·위 범위 검증은 완료했다. 실제 모델
응답 생성·579개 모델 성능·GPU·upstream 전체 환경·paper protocol 검증은
미실행이며, 전체 gold의 독립 재풀이와 universal OOD는 미검증이다. 다음 남은
migration 구현은 P5 EMMA와 P6 Zebra-CoT 2D다. 기존 격리 작업 트리에 변경을
보존했으며 commit·merge·push·작업 트리 정리는 수행하지 않았다.

# P5 EMMA 오프라인 검증 기록

## 실제 mini-400 준비와 전수 대응

고정한 `luckychao/EMMA-mini` revision
`6b9ae9a74733bb57f0b741213f8d0ec9ebae067a`에서 준비 CLI retry가 exit 0으로
완료됐다. 실제 산출물은 `data/evaluation/emma-mini400-v1-r02`이고 manifest
SHA-256은
`b41e7b210898417dfea6d48280737bc3e35c2133de183937fae1dc2ca805f961`이다.
[준비 요약](../../results/evaluation/emma-mini400-v1-r02/prepare.json)과
[전수 대응 결과](../../results/evaluation/p5-implementation-20260913-01/prepared-correspondence.json)를
확인했다.

독립 inventory와 준비 산출물의 400개 pid를 전수 대응했고 중복과 제외가 없다.
원본 answer(소문자 21개 포함), choices, subject, context+question 결합, 정규화한
target type, target index/content를 모두 비교했다. Chemistry/Coding/Math/Physics
각 100개, multiple_choice 341개, open_ended 59개다. 614개 asset 각각의 슬롯명,
manifest `asset_sha256`, 준비 파일 바이트 SHA-256을 inventory와 대조했다. 이미지
형식은 PNG 463, JPEG 149, WEBP 2이고 문항별 이미지 수 분포는 1개 313문항,
2개 36문항, 4개 26문항, 5개 25문항이다. 전체 manifest는 `read_manifest`로
config의 실제 dataset root에서 읽혔다. 이 증거는 데이터 대응이며 gold 정답의
독립 재풀이 또는 모델 성능 증거가 아니다.

첫 준비 시도는 source `solution`을 문자열로만 제한해, 실제 50개 null 중
`Math_54`에서 중단됐다. dataset/result directory는 생성되지 않았고
[실패 기록](../../results/evaluation/p5-implementation-20260913-01/actual-prepare-first-attempt.json)과
`logs/evaluation/emma-mini400-v1/prepare.log`를 그대로 보존했다. string 또는
null만 audit에 보존하고 model input에는 넣지 않는 인과 수정 뒤 집중 검사는
[33 passed](../../logs/evaluation/p5-t4-fix2-green-20260913-01/pytest.log)였다.

## 원본 helper의 제한 비교

Reference lock 확인 뒤 원본 commit
`47259952dd8e95bc1e3855a4164ccd3634f8eb1e`의 `is_number`,
`extract_full_boxed_content`, `fast_extract_answer` 세 pure helper만 AST로
격리해 `re`만으로 실행했다. 최종 extractor bytes에서 반복한
[최종 비교](../../results/evaluation/p5-extraction-comparison-final-20260913-03/extraction-comparison.json)와
[전체 로그](../../logs/evaluation/p5-extraction-comparison-final-20260913-03/comparison.log)는
23개 case 중 selected value 동일 12개, 의도한 차이 11개다.

동일 case는 raw decimal, fraction/nested box, 마지막 완성 box, 뒤의 미완성
box 앞 완성값, `nan`, `inf`, `B:` prefix, number-word marker, 8-choice `H`,
2-choice 범위 밖 raw `H`, 2-choice boxed text `A`를 포함한다. `nan`과 `inf`는
문자열 값만 같고 local status는 `invalid_format/non_finite_value`다. 차이 case는
marker decimal 3개(`0.5`, terminal-period `0.5`, `-2.75`), lowercase-colon raw,
일반 `A ...` prose, scientific decimal, 여러 marker가 있는 응답, 2-choice
lowercase label, choice 없는 `A:` raw, 대소문자를 보존한 marker `X`, balanced
nested text wrapper다. 이 차이는 소수/대소문자 보존, label 범위와 prefix 제한,
whole-raw fallback, balanced wrapper라는 modified contract다. 앞선 비교에서
발견한 prefix 결함의 결과는 진단 이력으로만 보존하며 최종 비교를 대체하지 않는다.
수정 뒤의 이전 r02 비교도 역사적 증거로 보존했고, 최종 r03의 23개 observation은
r02와 정확히 같다.

이 비교는 원본 전체 evaluator import, parser 동등성 stub, 원본 환경 실행,
scorer equivalence 또는 paper protocol 검증이 아니다.

## 계약 증거와 남은 검증

Task 1 최종 집중 수식 검사는 97 passed이며 실제 constructor 이전의 rational
component 한도 witness를 포함한다. Task 2/3/4의 세부 RED/GREEN과 review 수정은
각 SDD task report에 보존했다. 마지막 nullable-solution 수정 뒤 Task 4 집중
검사는 33 passed다. 그 전에 실행한 all-six-new-tests 167 passed는 역사적
통합 증거일 뿐 최종 bytes의 전체 P1–P5 결과로 표시하지 않는다.

| 범위 | 상태 |
| --- | --- |
| 실제 mini source→manifest 전수 대응 | 완료: 400행, 614이미지, 제외 0 |
| 제한된 원본 추출 helper 비교 | 완료: 23 cases, selected value 동일 12 / 의도한 차이 11 |
| 최종 reviewed-byte P1–P5 scoped CPU 계약 | 완료: 550 passed in 7.79s, exit 0 |
| 최종 controlled runtime integration stubs | 완료: 7 passed in 9.17s, exit 0; GPU/model inference 아님 |
| 실제 모델별 extraction first-nail | not_run: 실제 모델 응답 없음 |
| GPU/모델 inference | not_run |
| 실제 400개 응답 성능 | not_run |
| 원본 전체 환경 실행 | not_run |
| paper protocol | false / 미검증 |
| 전체 프로젝트 학습·모델 suite | not_run: scoped CPU/stub 검증과 구분 |

P5 결과는 `protocol_kind=modified`, `paper_protocol_verified=false`,
`ood_verified=unknown`이다. [최종 검증 기록](../../results/evaluation/p5-final-20260913-01/verification.json),
[CPU 전체 출력](../../logs/evaluation/p5-final-20260913-01/cpu.log),
[runtime stub 전체 출력](../../logs/evaluation/p5-final-20260913-01/runtime-stubs.log)에
명령과 exit code를 보존했다. CPU 명령은 P1–P5 관련 평가 계약 24개 파일만 지정한
scoped suite이며 전체 프로젝트 suite가 아니다. runtime 명령은 기존 bridge와
Stage2 통합의 controlled stub 7개이며 CUDA/GPU 또는 실제 모델 추론 증거가 아니다.

최종 보존 검사는 보호 파일 5,026개가 모두 불변임을 확인했다. Reference 7개의
독립 root/origin/commit/clean 상태와 evidence/license hash 33개가 lock과 일치했고,
동결한 design proposal v1/v2/v3 hash도 유지됐다. 기존 `data`와 `checkpoints`
심볼릭 링크를 보존했다. 최종 reader는 실제 manifest 400행과 SHA-256
`b41e7b210898417dfea6d48280737bc3e35c2133de183937fae1dc2ca805f961`을 다시
확인했다.

이로써 P5 오프라인 scorer 계약과 실제 mini-400 준비·대응은 완료했다. 실제 모델
응답, 모델별 extraction first-nail, GPU inference, 원본 전체 환경, 400개 모델
성능, paper protocol은 계속 `not_run` 또는 false이며 이 완료 범위에 포함되지 않는다.

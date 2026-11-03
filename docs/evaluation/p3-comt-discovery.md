# P3 CoMT 착수 조사 — 2026-09-11

상태: 읽기 전용 조사와 설계 방향 제안. 사용자가 TEST 400개 기준을 선택했다.
두 행의 보정 및 P3 설계 최종 승인은 아직 없다. 아래 초기 398개 제안은 마지막
절의 사용자 선택과 추가 근거로 대체됐으며 조사 이력으로 남긴다.

understood as: 사용자의 후속 평가 구현 요청은 P2 뒤 CoMT → VisuLogic →
EMMA → Zebra-CoT 2D를 진행하는 것이다. 각 단계에서 실제 데이터 계약을
확인한 뒤 오프라인 채점부터 구현한다. 논문 재현이나 VSP 추가 GPU 실행은
이번 착수의 필수 조건으로 삼지 않는다. 기존 P1/P2 작업 트리와 데이터는 보존한다.

## 확인한 근거

활성 worktree는 `.worktrees/evaluation-p1`이다. Reference 7개의 독립 Git root,
origin, commit, clean 상태와 evidence/license 33개 hash가 현재 lock과 일치했다.
[검증 기록](../../results/evaluation/p3-discovery-20260911-01/reference-verification.json).
CoMT evaluator 근거는 commit `01095663a1d1f3b4818d255e0a8f02ef754bddc4`의
`evaluate.py::judge_answer/run`과 README다. 이 조사에서 원본 함수를 실행하거나
소스를 복사하지 않았다.

코드 README가 연결한 [공식 데이터 저장소](https://huggingface.co/datasets/czh-up/comt)의
revision `bde314be94942a7b4ab66581e7f7c4801b84e686`에서 `comt/data.jsonl`과 README를
조회했다. 데이터는 `data/evaluation/comt-source-discovery-20260911-01/`에 저장했다.
JSONL은 5,307,217 bytes, SHA-256
`15935e366f3b9998d3fd18661f563d9cd9a248d33a534e38a73d9b5c65310133`이다.
데이터 card는 `cc-by-sa-4.0`을 선언한다. 이는 코드 저장소의 라이선스 파일 부재를
대신하지 않는다. 원본 코드 복사나 의존성 설치는 하지 않는다.

첫 recursive tree 응답은 1,000개 항목의 첫 페이지다. 전체 이미지 목록으로
취급하지 않으며, 이미지의 원격 hash 대조도 아직 수행하지 않았다.

## 실제 데이터 계약

[전체 행 inventory](../../results/evaluation/p3-discovery-20260911-01/comt-inventory.json)는
3,853개 원본 행과 고유 문자열 ID를 확인했다. 범주는 creation 1,020,
deletion 1,008, update 932, selection 893이다. 원본 label은 모두 A–D다.
선택지 원소는 문자열 또는 정수이며 3,849개 행은 선택지 4개, 3개 행은 6개,
1개 행은 8개다. 따라서 README의 단순 예시만으로 strict 계약을 정하면 안 된다.

모든 `image` 값은 dict다. creation 질문 1,020개에는 이미지 placeholder가 없고,
나머지 2,833개 질문은 `[IMAGE0]`만 참조한다. 모든 행에 IMAGE0가 있다.
IMAGE1 이후는 질문 입력으로 자동 채택하지 않는다. 원본 rationale/annotations와
중간 그림을 모델 입력에 넣지 않는다. 로컬 `data/images_comt/<type>/<IMAGE0>.*`
대응은 3,851개가 하나의 파일로 연결되고 update 2개는 없다. 이는 파일명 대응이며
원격 이미지 bytes나 이미지 의미 일치의 증거는 아니다.
[입력 inventory](../../results/evaluation/p3-discovery-20260911-01/comt-input-inventory.json).

## 기존 ILVR TEST/TRAIN 연결

원본 후보는 `(type, IMAGE0 token)`으로 찾았다. 정답을 이용해 후보를 선택하지
않았다. 이후 원본 질문에서 `[IMAGE0]`를 제거하고 양끝 공백을 trim한 뒤
`" option:" + str(option)`을 붙인 결과와 기존 `text_input`을 대조했다.
원본 라벨 위치의 선택지 내용과 기존 `original_final_answer`도 별도로 비교했다.

| 로컬 파일 | 행 수 | 고유 원본 ID 대응 | 질문·선택지 변환 일치 | 라벨→내용 변환 일치 |
|---|---:|---:|---:|---:|
| TEST.jsonl | 400 | 400 | 400 | 400 |
| TRAIN.jsonl | 3,450 | 3,450 | 3,450 | 3,450 |

TEST는 네 범주 각각 100개다. 확인한 두 파일 사이 원본 ID 교집합은 0이다.
이것은 모든 모델의 학습 이력, 다른 데이터의 중복 이미지, 논문 split 또는
OOD를 검증한 결과가 아니다.
[연결 검증](../../results/evaluation/p3-discovery-20260911-01/comt-legacy-join-validation.json).
최초 probe는 placeholder 제거 후 앞쪽 줄바꿈을 남겨 불일치를 보고했다.
실제 trim 규칙을 확인한 후 위 대조를 수행했다. 그 최초 기록도 data 아래 보존한다.

## 발견한 평가 차단 문제

TEST의 아래 두 행은 원본 label이 빈 선택지를 가리키며, 기존 gold 내용도 빈 문자열이다.

| TEST 행 index | 원본 ID | 선택지 수 | 원본 label |
|---|---|---:|---|
| 57 | creation-12125 | 8 | B |
| 93 | creation-12164 | 6 | D |

전체 원본의 추가 비정상 길이 행은 creation-11380, creation-11960이다. 빈칸 제거 후
라벨을 다시 해석하거나 질문을 풀어서 gold를 고치는 것은 현재 소스의 확정 사실이
아니다. 자동 수정하지 않는다. 원본의 creation-10610에는 같은 내용의 선택지 두 개도
있다. 라벨 채점과 내용 매핑의 차이를 테스트해야 한다.

upstream의 ALPHA_MAP은 A–D 네 개뿐이다. 괄호 문자 추출은 가장 마지막 항목,
내용 substring 매칭 및 독립 문자 매칭은 선택지 순서상 가장 뒤의 후보를 택한다.
`[Answer]`가 있으면 `ANSWER:`로 split하는 조건도 관찰했다. 비정상 선택지 수와
빈 문자열은 이 코드의 content fallback에서 위험하다. 이는 정적 관찰이며 실제
upstream 함수 대조는 승인된 구현 검증 단계에서 별도로 수행할 대상이다.

## 제안할 P3 설계 방향 — 최종 commitment 아님

공통 schema/io/registry와 score CLI를 재사용해 MCQ target을 추가한다. VSP의
기존 JSON/CLI·지도 검증·fingerprint·분모 계약은 보존한다. CoMT의 추출 함수는
응답과 선택지만 받고, gold 라벨과 내용 비교는 별도 단계가 담당한다.
명시적 `comt_reference_rules_v1` profile로 관찰한 upstream 추출 우선순위를
독립 구현하며, 유효한 4개 비어 있지 않은 선택지에서만 원본 비교를 주장한다.
분모와 missing/infra 처리, 범주별 빈 집계는 기존 P1 계약을 재사용한다.
원본 추출 규칙의 관대한 substring 및 복수 후보 우선순위를 strict 규칙으로
조용히 바꾸지 않는다.

권장 데이터 경로는 기존 TEST 400개의 ID를 복구한 뒤, 위 2개의 명시적 보류 ID를
따로 기록하고 398개를 새 이름의 검증 가능한 subset으로 준비하는 것이다.
creation은 98개, 나머지는 각각 100개다. 기존 TEST/TRAIN은 수정하지 않고,
398개 점수를 400개 또는 원본 전체 점수로 보고하지 않는다. 사용자가 400개
분모 유지를 요구하면 독립적으로 gold/선택지 수정 근거를 확보하기 전 전체
실데이터 평가 준비는 미완료로 남는다. 이 선택은 아직 승인되지 않았다.

대안은 원본 3,853개 전체를 대상으로 데이터 문제 4개와 로컬 누락 이미지 2개를
먼저 해결하는 것이다. 준비 범위가 크고 학습 데이터 포함 여부를 따로 다뤄야 한다.
또 다른 대안인 원본 라벨만으로 비정상 행을 그대로 채점하는 방식은 선택지/내용
대응의 검증을 제공하지 않으므로 권장하지 않는다.

P3 완료 조건은 원본 ID·입력·선택지·target join, 네 범주 집계, gold 비의존 추출,
P1/P2 CPU 회귀, Reference 없는 score CLI 통합, 분리된 upstream 함수 대조다.
GPU 생성·논문 재현은 별도다. VisuLogic은 rules/LLM profile, EMMA는 명시적
split·수치/기호 동등성, Zebra 2D는 전체 답안 형식 inventory를 각 단계에서
확정한다. 미확인 후속 벤치마크의 상세 구현 계획을 이번에 미리 고정하지 않는다.

Superpowers 설계 단계가 활성 상태다. 위 데이터 선택 및 설계 절을 확인한 뒤
written spec을 만들고, AGENTS.md의 hate/해당 critique 합성을 거쳐 정확한 최종
commitment의 승인을 받는다. 아직 구현 skill이나 production 코드 변경은 시작하지 않았다.

## 사용자 선택과 추가 근거

사용자는 후속 질문에 **기존 TEST 400개를 원본 ID·선택지·범주와 연결**하는
방향을 선택했다. 398개로 줄이는 초기 제안은 채택하지 않는다.

두 문제를 pinned 원본 JSONL에서 다시 읽었다. creation-12125의 rationale는
평행사변형 둘레가 18cm라고 명시하고, creation-12164는 마름모 둘레가 24cm라고
명시한다. 원본 문제 그림 두 장도 직접 열어 확인했다. 빈 항목만 제거한 두
명시적 배열은 각각 B→18, D→24로 원본 label과 해설 내용을 일치시킨다.
이는 답을 새로 추측한 근거가 아니라 원본에 있는 근거의 대조다. 일반적인 자동
빈칸 제거 규칙을 모든 데이터에 적용하는 근거로 확장하지 않는다.

추가 원격 조회로 TEST에 연결된 정확한 이미지 400개만 metadata를 받았다.
서버의 canonical dataset ID는 `czh-up/CoMT`이며 요청의 소문자 alias에서 307
redirect됐다. 첫 urllib POST는 redirect 처리에서 실패해 로그를 보존했고,
이미 설치된 requests의 기본 redirect 처리로 새 probe ID에서 읽기를 완료했다.
데이터 다운로드/복제 없이 기존 로컬 400개 파일의 SHA-256과 source LFS OID를
대조했고 **400/400 일치**, **400/400 decode**를 관찰했다.
[이미지 검증](../../results/evaluation/p3-discovery-20260911-01/comt-test-image-validation-02.json).
이는 전체 원격 파일 목록을 받은 것과 다르며 모든 이미지의 의미 검증도 아니다.

최종 승인 대상은 같은 400개를 유지하는
[design commitment v1](../superpowers/specs/2026-09-11-ilvr-comt-offline-design.md)이다.
기존 TEST/TRAIN/source를 보존하고 새 `ilvr-test400-corrected-v1` 데이터 버전에만
위 두 배열, 해당 target content 및 text의 option 부분을 보정한다. 나머지 398개
문자열·400개 순서·네 범주 100개씩·문제 이미지 bytes를 유지한다.
아직 데이터 보정이나 P3 구현을 실행하지 않았다.

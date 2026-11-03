# P3 CoMT design commitment v1

작성일: 2026-09-11. 상태: 2026-09-11 사용자 최종 승인 후 P3 구현·실제400 준비·최종 검증 완료.
TEST 400개 유지, 아래 두 행의 명시적 보정 및 구현 설계는 사용자의 `ㅇㅋ`로 승인됐다.

## 문제와 범위

사용자는 P2 뒤 후속 평가 구현을 요청했고, CoMT는 기존 `data/TEST.jsonl`
400개를 원본 ID·선택지·범주와 연결하도록 선택했다. 먼저 P3 CoMT 오프라인
채점과 실제 데이터 준비를 완료한다. 그 뒤 VisuLogic → EMMA → Zebra-CoT 2D의
실제 데이터 계약에 맞춘 후속 단계를 진행한다. 미확인 후속 adapter 세 개의
상세 구현을 미리 확정하지 않는다. GPU 생성, 논문 성능 재현, 외부 judge,
의존성 설치, 기존 학습·추론 코드 변경, commit/merge/push는 이 P3 범위가 아니다.

기존 작업 트리 `.worktrees/evaluation-p1`과 staged P1/P2를 재사용한다.
데이터는 `data/`, 결과는 `results/evaluation/<run-id>/`, 로그는
`logs/evaluation/<run-id>/`에 두며 기존 data/checkpoints 링크를 보존한다.

## 출처와 실제 데이터

코드 근거는 lock의 CoMT commit `01095663a1d1f3b4818d255e0a8f02ef754bddc4`다.
Reference 7개 root/origin/commit/clean과 33개 evidence/license hash를 재확인했다.
Reference는 runtime 의존성이 아니며 라이선스 파일이 없는 CoMT 소스는 복사하지
않고 관찰한 행동을 독립 구현한다.

공식 데이터는 `czh-up/CoMT`, revision
`bde314be94942a7b4ab66581e7f7c4801b84e686`의 `comt/data.jsonl`이다.
JSONL SHA-256은 `15935e366f3b9998d3fd18661f563d9cd9a248d33a534e38a73d9b5c65310133`.
기존 TEST SHA-256은 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`.
원본은 3,853행이나 선택 모집단은 TEST 순서의 400개, 네 범주 각 100개다.

400개 모두 `(type, IMAGE0)`로 원본 ID에 유일하게 연결됐다. 원본 질문의
`[IMAGE0]` 제거 및 양끝 공백 trim 뒤 `" option:" + str(option)`을 붙인 문자열이
기존 text_input과 400개 모두 일치한다. 원본 label 위치의 내용도 기존 gold와
일치하지만 그중 아래 두 항목은 빈 문자열이라는 데이터 결함이 있다.
TEST 문제 이미지 400개는 고정 revision의 LFS SHA-256과 모두 일치하며 decode됐다.
확인한 TRAIN 3,450개와 TEST의 원본 ID 교집합은 0이다. 다른 학습 데이터·이미지
중복·모든 모델의 학습 이력에 대한 OOD 확인은 아니다.

## 두 행의 명시적 보정

| source_id | 원본 선택지 | 새 선택지 | 보존 label / 새 content |
|---|---|---|---|
| creation-12125 | `['17','','18','','22','','34','']` | `['17','18','22','34']` | B / `18` |
| creation-12164 | `['10cm','12cm','16','','24','']` | `['10cm','12cm','16','24']` | D / `24` |

근거는 같은 pinned 원본의 rationale가 각각 18cm와 24cm를 명시한다는 점이다.
두 문제 그림도 직접 확인했으나 이 검토를 400개 gold의 독립 재풀이 검증으로
일반화하지 않는다. 일반적인 빈 선택지 제거 규칙은 만들지 않는다. 정확한
두 ID, 원본 배열, 원본 label, source JSONL hash가 모두 맞을 때만 위 보정을
적용한다. 다른 비정상 행은 입력 오류다. 현재 TEST/TRAIN와 원본 JSONL은 수정하지 않는다.

새 데이터 profile은 `ilvr-test400-corrected-v1`이다. 같은 TEST 400개를 유지하고
모든 제외 개수는 0이다. 두 행의 새 ModelInput.text의 option 부분도 새 배열로
일치시킨다. 나머지 398개 text는 기존 TEST bytes의 문자열과 동일하다. 변경
목록과 old/new 값, 근거, hash는 data 아래 별도 corrections JSON에 보존한다.
이는 원본 그대로의 split/prompt라고 주장하지 않는 modified 데이터 프로토콜이다.

## 채점 계약

`comt` + `comt_reference_rules_v1` 조합을 명시적으로 등록한다. 이름의 reference는
유효한 4개 선택지에서 관찰한 원본 추출 규칙을 뜻하며, 수정 데이터·분모 처리나
전체 upstream 환경까지 같다는 의미는 아니다. 다른 benchmark/profile의 묵시적
fallback은 없다.

추출은 raw response와 ModelInput의 choices만 받는다. gold는 추출 인터페이스에
전달하지 않는다. 비교 단계에서만 target의 A–D label/index/content를 읽는다.
선택지 정수는 데이터 준비에서 `str`로 변환하고 순서를 유지한다. 유효 target은
정확히 4개 비어 있지 않은 문자열 choices, 0–3의 정확한 int index, 그 위치와
일치하는 label 및 content다. bool, 빈 gold, 불일치 label/content는 preflight 오류다.
동일 내용 선택지가 있을 때 label/index를 합치거나 의미상 같은 보기라고 재정의하지 않는다.

원본 관찰 규칙의 우선순위는 다음과 같다.

1. 대소문자를 구분한 literal `[Answer]`가 raw response에 있으면
   `response.split("ANSWER:")[-1]`의 의미로 마지막 `ANSWER:` 뒤를 사용한다.
   `[Answer]`만 있고 `ANSWER:`가 없으면 응답 전체가 그대로 남는다.
   `ANSWER:`만 있고 `[Answer]`가 없을 때도 이 전처리를 적용하지 않는다.
2. 정규식 `\(([A-Za-z])\)`에 맞는 literal `(A)` 형태가 있으면 응답에서
   가장 마지막 일치의 문자를 대문자로 추출한다. `( A )`, `[A]`, `{A}`는
   이 단계의 괄호 일치가 아니다.
3. 없으면 대소문자를 무시한 choice content substring 매칭을 한다. 복수 일치 시
   선택지 순서상 가장 뒤의 후보를 택한다.
4. 없으면 줄바꿈·마침표·쉼표·느낌표·물음표를 공백으로 바꾸고, literal space로
   나눈 token에서 A–D 대문자를 찾는다. 복수 일치 시 선택지 순서상 가장 뒤다.
5. 없으면 같은 소문자 규칙을 사용하고, 그래도 없으면 unparseable이다.

괄호로 추출된 E–Z는 유효한 정답 label이 아니므로 오답이다. 의미를 추측하거나
새 boxed/marker fallback을 추가하지 않는다. 이 관대한 규칙과 더 엄격한 추출의
차이는 문서와 고정 regression 사례로 드러낸다. 빈 응답·형식 실패는 관측된
모델 실패이며 missing/인프라 오류와 구분한다. 여기서 형식 실패는 유효한
Prediction 레코드 안 응답 문자열의 추출 실패다. Prediction의 필수 필드·타입이
잘못된 경우는 별도 입력 오류이며 모델 오답으로 변환하지 않는다. 최종 correctness는
추출된 label과 target label의 동등 비교다. target content는 그 label/index의
정합성을 검증하는 값이며 별도의 content OR 조건으로 정답을 추가하지 않는다.

분모는 manifest 400개다. ID 중복·미등록·hash 불일치는 입력 오류다. 누락/인프라
오류가 있으면 complete=false, accuracy=null이다. 네 범주 집계는 기존 P1
report 흐름을 쓰고, 빈 범주도 count=0, accuracy=null로 처리한다. scorer 출력의
protocol_kind는 modified이며 paper_protocol_verified=false다.

## 기존 코드에 연결하는 경계

`schema.py`에 MCQ target을 추가하고 Sample.target을 benchmark로 구분한다.
`io.py`는 VSP의 기존 target/max_steps 검증을 그대로 수행하며, CoMT에는 위
MCQ target 검증을 수행한다. VSP의 기존 config 필드·동작은 유지한다. CoMT
config에는 VSP 전용 max_steps를 요구하지 않고 내부 값은 None으로 표현한다.
구체적인 타입과 함수 signature는 승인 후 implementation plan에서 한 번 고정한다.

`benchmarks/comt.py`와 `comt_prepare.py`를 추가하고 기존 registry, prepare CLI,
score CLI를 재사용한다. 새로운 공통 plugin/judge/retry 계층은 만들지 않는다.
`scripts/run_eval.py`와 inference bridge는 VSP용으로 유지하며 CoMT의 모델
생성 지원으로 표시하지 않는다. 공유 schema/registry 확장 뒤에도 VSP 전용
run CLI가 CoMT 입력을 모델 로딩 전에 명시적으로 거부하는지 검증한다.

prepare는 source metadata와 TEST selection, image root를 명시적으로 받는다.
기존 source root와 fresh output 인자 계약에 CoMT에서만 필요한 인자를 추가하고
기존 VSP 사용법은 유지한다. 선택 파일의 원본 ID가 없으면 filename/category로
유일한 후보를 찾되 질문·선택지의 정확한 관찰 변환을 반드시 대조한다. gold를
join key로 쓰지 않는다. 400개 순서와 각 범주 100개를 검증하며 위 두 보정은
join 후 적용한다. 원본 rationale/annotations/IMAGE1 이후는 ModelInput에 넣지 않는다.

prepare output은 새 `data/evaluation/<dataset-id>/`의 manifest, source-selection
연결 기록, corrections/source metadata다. 기존 `data/images_comt/`의 검증한
문제 그림 400개를 재사용하며 복제하지 않는다. dataset_root는 활성 repo의
`data/`이고 manifest의 이미지 경로는 `images_comt/<type>/<file>`이다.
Reference 없이 이 data tree와 저장 응답만 있으면 score가 동작한다.
준비 요약과 로그는 동일 dataset-id의 results/logs 아래 둔다. source/selection/
입력 자산의 실제 hash를 기록하고, 알 수 없는 source revision이나 예상과 다른
두 보정 대상은 거부한다. 출력 디렉터리는 fresh만 허용하며 자동 복구는 하지 않는다.

기존 응답을 순서로 붙이거나 새 manifest hash로 재포장하지 않는다. 새 P1 Prediction
계약의 ID·manifest hash·run fingerprint를 가진 저장 응답을 score CLI가 받는다.
실제 모델 응답을 확보하기 전 합성 fixture 점수를 실제 CoMT 성능으로 보고하지 않는다.

## 검증과 완료 기준

CPU TDD는 문자↔내용 매핑, substring 중복 우선순위, 괄호/marker 특이 규칙,
범위 밖 label, 빈 응답, malformed target, gold mutation 독립성, 네 범주/빈 범주,
분모·ID/hash·오류 계약을 고정한다. 준비 테스트는 exact join, 문제 이미지 한 장,
trace 미유입, 두 보정의 exact match 및 예상 밖 원본 거부, 원본 파일 불변을 검사한다.
unit fixture는 실제 데이터/Reference/torch/CUDA/model을 요구하지 않는다.

실데이터 완료 조건은 보정된 400행을 다시 읽어 원본 ID·순서·네 범주·target·입력
hash와 변경 범위(2개)를 확인하는 것이다. Reference 없는 복사본에서 실제 prepare/
score CLI 통합을 검사하고, P1/P2 CPU 계약과 필요한 기존 stub 회귀를 한 번 실행한다.
변경 허용 대상은 공유 schema/io/registry, prepare CLI와 신규 CoMT 모듈·config·
테스트·문서다. 이 파일들은 동작 회귀로 검증한다. 바이트 보존 대상은 기존
`eval.py`, `src/evaluate_deepseed.py`, `src/task_deepseed.py`, 기존 scoring/runtime/
continuous_latent, P2 inference bridge/run CLI와 실제 editable Transformers의
기존 세 Qwen 파일이다. Reference lock과 저장 링크 보존도 별도 확인한다.

원본 CoMT의 judge 함수는 네트워크/Fire 실행 없이 dependency를 검토한 제한된
function-only 비교로 고정 사례에 대조한다. 이 제한된 비교는 P3 완료의 필수
검증이며 실행할 수 없으면 그 검증과 전체 P3 완료를 미완료로 보고한다. upstream 전체 실행 환경
재현과 구분한다. 이 비교는 승인 뒤 수행하며 현재 실행한 것처럼 기록하지 않는다.

P3 구현/CPU 계약, 실제 400개 준비, 함수 대조, GPU, full benchmark, paper protocol은
각각 보고한다. GPU와 400개 모델 추론, 논문 프로토콜은 이 단계에서는 not_run이다.
다음 VisuLogic/EMMA/Zebra 착수는 이 구현 완료와 구분한다.

## 비평 합성과 승인 범위

**합의된 문제와 범위:** 사용자가 선택한 기존 TEST 400개를 원본과 연결해
CoMT 전용 오프라인 평가를 구현한다. 위 모든 절이 구현 대상 계약이며, 나머지
세 벤치마크와 모델 실행은 그 단계의 검증 조건에 따라 별도로 진행한다.

**최종 설계 및 제안 대비 변경:** 동결 proposal v1과 같은 행동·인터페이스·범위다.
두 개의 source-grounded 보정을 새 데이터 버전에 적용하고, 기존 P1/P2 및 raw
데이터를 보존하며, source 규칙과 수정 프로토콜을 구분한다.
context-free cold read 뒤 literal 추출 문법, 입력 오류와 응답 형식 실패,
바이트/동작 보존 대상, 함수 대조의 필수 여부를 명문화했다. 원본에서 관찰한
규칙과 제안의 검증 범위를 해석 없이 실행할 수 있게 한 설명이며 새 fallback이나
새 모델/데이터 범위는 추가하지 않는다. `comt_reference_rules_v1`은 scorer profile,
`ilvr-test400-corrected-v1`은 데이터 준비 profile이며 역할이 다르다.

**hate root / first_nail:** 임의 보정한 gold를 scorer가 되풀이 확인하는 위험을
검토했다. 저비용 first_nail은 두 원본 행의 option/label/rationale와 문제 이미지
대조, 400개 ID 및 source image bytes 대응 확인이다. 읽기 전용 조사로 수행했고
위 두 mapping에 반례를 찾지 못했다. 이는 원본 해설이라는 외부 근거이며 모든
gold의 독립 풀이 증명은 아니다. 상태는 관찰 완료; 코드 보정 실험은 미실행이다.

**다른 관점과 남은 이견:** macrothink는 사용자가 작업 순서와 400개 모집단을
명시했고 별도 모델/훈련 방향 선택이 없어 추가 fresh read 비용을 정당화하지
않는다고 판단해 생략했다. prism의 평가 타당성 관점은 정확한 두 보정의 승인
조건부 수용, 호환성 관점은 기존 경로 보존 설계 수용이다. 설계를 뒤집는 기술적
충돌은 없지만 실제 회귀/원본 함수 일치는 구현 뒤에만 판정한다. mandela 점검은
이 제한된 주장 안에서 새 확정 leakage를 찾지 못했으며 gold 비의존 추출, helper
image 제외와 합성 점수/실성능 구분을 검증 조건으로 유지한다.

**미검증 전제:** 구현 정확성, 원본 함수와의 실측 비교, 전체 모델 평가, 398개
나머지 gold의 독립적인 의미 검증, 모든 학습 자료에 대한 OOD, paper protocol이다.
계획된 테스트 또는 읽기 전용 비평을 이미 통과한 runtime 증거로 보고하지 않는다.

**요청하는 명시적 승인:** 같은 TEST 400개를 유지하고 source rationale로
뒷받침되는 위 두 배열/target/text 보정을 새 `ilvr-test400-corrected-v1`에만
적용한다. 이 수정 데이터 및 검증 한계를 공개한 채 implementation plan과
순차 TDD/review를 수행한다. source/test 파일 수정, dependencies, 외부 judge,
GPU 실행, commit/merge/push 권한은 포함하지 않는다.

**자기검토:** 범위·데이터 선택·기존 CLI 보존·gold 분리·실행 증거 구분을
교차 확인했다. detailed 함수 signature와 fixture 코드는 승인 후 plan에서
고정할 사항이며 실행 가능한 코드가 이미 존재한다는 뜻이 아니다.

**근거 문서:** [조사](../../evaluation/p3-comt-discovery.md),
[동결 제안](../../evaluation/p3-design-proposal-v1.md),
[비평 기록](../../evaluation/p3-design-review.md). 결과 파일은 조사 문서에 연결했다.

승인 후 다음 단계는 Superpowers writing-plans다. 기존 AGENTS의 구현 경로
선택에 따라 승인된 plan 안의 작업은 SDD로 순차 수행하며 task마다 계속할지
다시 묻지 않는다.

승인 기록: 두 보정을 포함한 P3 설계대로 구현할지 묻는 최종 요청에 사용자가 2026-09-11 `ㅇㅋ`로 응답했다. 계획 작성과 순차 구현·검증을 진행한다.

실행 완료 기록: 실제400개(범주별100, 두 보정, 제외0), 원본 함수35×4 판정 일치, 최종CPU307개 및 별도스텁7개 통과. 상세 증거와 미실행 범위는 [검증 문서](../../evaluation/validation.md)의 P3 절을 참조한다. 기존 설계 본문의 미래형은 승인 당시 계약이며 이 완료 기록과 구분한다.

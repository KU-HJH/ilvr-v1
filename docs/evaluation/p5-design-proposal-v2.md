# design proposal v2 — P5 EMMA 오프라인 평가

2026-09-13. 상태: **세부 계약 승인; self-review 완료; 비평용 동결 제안**.

understood as: 사용자는 EMMA-mini 400개와 기존 환경의 규칙 기반 접근을 선택했고, 이어 제시한 세부 계약에 `ㅇㅋ`로 응답했다. 해당 계약을 최종 설계 검토 대상으로 정리한다. 최종 commitment 승인과 구현 계획은 아직 남아 있다. 자동 commit/push하지 않는다.

## 1. 범위와 기존 경로

고정 EMMA-mini test 400개를 준비하고 저장된 prediction을 오프라인 채점한다. `benchmark=emma`, `profile=emma_safe_rules_v1`을 명시하며 기본 benchmark/profile을 추가하지 않는다. mini는 dataset ID와 revision으로 명시한다. 이번 준비 경로는 full이나 임의 subset을 지원하지 않고, 전체 400개를 요구한다.

기존 `scripts/prepare_eval.py`, `scripts/score_eval.py`, Sample/Prediction, manifest hash, run fingerprint, 보고서와 경로 검사를 재사용한다. EMMA target/scorer/preparer와 과목 외 집계만 추가한다. 기존 VSP/CoMT/VisuLogic target·profile·결과 필드는 바꾸지 않는다. P2 추론 bridge와 `run_eval.py`의 VSP 범위도 유지한다.

추가 후보 파일은 `src/evaluation/benchmarks/emma.py`, `src/evaluation/emma_math.py`, `src/evaluation/emma_prepare.py`, `configs/eval/emma_safe_rules_v1.yaml`과 EMMA CPU 테스트다. 공통 `schema.py`, `io.py`, `registry.py`, `report.py` 및 prepare CLI에 명시적 EMMA 분기를 추가한다. 기존 runtime, patched Transformers, prompts, 생성 설정은 변경하지 않는다.

## 2. 원본과 데이터 계약

- dataset: `luckychao/EMMA-mini`, revision `6b9ae9a74733bb57f0b741213f8d0ec9ebae067a`, 각 과목 config의 `test` split. 과목은 Chemistry/Coding/Math/Physics 각 100개다.
- 입력 parquet의 크기·SHA-256을 확보한 repository tree와 대조한다. 고유 ID 400개, 객관식 341개, 자유응답 59개, 이미지 614개를 준비 전 검증한다. 추가·제외·정답 수정은 하지 않는다.
- `Sample.id`는 원본 pid, `subtask`는 subject다. `ModelInput.text`는 context가 있으면 `context + "\n\n" + question`, 없으면 question이다. 해설·정답·judge prompt를 넣지 않는다. 원본 텍스트와 선택지는 재작성하지 않는다.
- `ModelInput.images`는 존재하는 원본 슬롯을 번호 순으로 담는다. 파일명에 `image_2` 등의 슬롯 번호를 유지하고 질문의 `<image_n>`도 유지한다. `phy_6`의 슬롯 `[2, 3]`을 `[1, 2]`로 바꾸지 않는다. 이미지 bytes는 재인코딩 없이 저장한다. 이 규칙은 데이터 보존 계약이며 아직 EMMA 추론 연결의 증거가 아니다.
- EMMA 전용 target은 `question_type`, 원본 `answer`, `index`, `content`, `category`, `task`를 가진다. 답과 평가용 분류를 함께 보관하되, 추출 함수에는 이 target 전체를 전달하지 않는다. 공통 Sample의 필드나 다른 benchmark의 JSON 모양을 확장하지 않는다.
- 객관식 `question_type=multiple_choice`: 선택지는 2~8개의 비어 있지 않은 문자열이다. 원본 answer의 A–H/a–h 한 글자로 index를 구하고 범위를 검사한다. answer 자체의 대소문자를 보존하며 content는 해당 선택지와 정확히 같아야 한다.
- 자유응답 `question_type=open_ended`: choices는 빈 tuple, index는 null, content는 원본 answer다. 원본의 `[]`/null options를 이 표현으로 매핑한다. 실제 mini의 자유응답 정답은 모두 정수 문자열이다. scorer 계약 테스트에는 분수·수식 target도 사용할 수 있다.
- category/task는 원본 문자열이다. 이 필드들은 집계에서만 사용한다. 준비 시 원본 question/context/type/answer와 슬롯 대응을 별도 source inventory에 보관해 합성된 model input과 대조할 수 있게 한다.
- 모든 source/archive/cache/image/manifest/inventory는 `data/` 아래에, 생성 보고서는 `results/`, 실행 로그는 `logs/` 아래에 둔다. 기존 출력 디렉터리를 덮어쓰지 않는다.

## 3. 정답을 보지 않는 추출

`extract_answer(response, model_input)`은 target이나 정답 index/content를 받지 않는다. 빈 응답은 empty다. 후보 선택은 다음 순서로 고정한다.

1. 전체 응답이 해당 문항의 유효한 선택지 라벨이거나 `A:`/`A.` 같은 대문자 라벨 접두사이면 라벨을 추출한다. 접두사는 객관식에서만 적용한다.
2. 전체 응답이 유한 decimal/scientific 숫자이면 그대로 추출한다.
3. 원본처럼 중첩 중괄호를 세어 마지막 완성된 `\boxed{...}`의 내용을 추출한다. 뒤의 미완성 box 때문에 앞의 완성된 box를 버리지 않는다. 완성된 box가 없으면 다음 단계로 간다. 후보 전체를 감싸는 `\text{...}`는 한 겹 벗긴다.
4. 원본 우선순위의 marker `final answer is`, `correct answer is`, `answer should be`, `answer is`, `answer:`를 대소문자 무관하게 찾는다. 가장 먼저 해당하는 marker 종류의 마지막 출현 뒤 첫 줄을 취한다. 앞뒤 공백과 끝의 문장 마침표 한 개만 제거하고 소수점·변수 대소문자를 보존한다. 첫 마침표로 나누지 않는다.
5. 그 외 단독 응답 전체를 후보로 남긴다. 문자열 정확 일치와 지원 수식 비교를 허용하지만, 문장 중간에서 정답으로 보이는 부분을 추가로 찾지 않는다. 이 fallback은 원본과 다른 명시적 규칙이다.

빈 추출 후보는 invalid_format이며, 추출 실패와 오답을 구별해 기록한다. raw `nan`, `inf`와 동등한 비유한 숫자 표기는 invalid_format으로 처리한다. 후보 전체를 감싸는 단일 수학 구분자 `$...$`, `\(...\)`는 비교용 표현에서만 벗기고 기록된 추출값은 유지한다. 빈 문자열끼리 같다고 정답으로 처리하지 않는다.

## 4. 비교 규칙과 지원 범위

객관식은 유효한 라벨이면 정답 index와 비교한다. 유효한 라벨이 오답일 때 다른 선택지 내용으로 재해석하지 않는다. 그 외 후보는 정답 선택지 content와 아래 규칙으로 비교한다. 동일 내용의 선택지가 여러 개일 때 내용 응답을 인정하는 것은 원본의 label/content 비교에 따른 제한이다. 자유응답은 answer와 비교한다.

비교의 우선순위는 비어 있지 않은 문자열의 앞뒤 공백만 제거한 후 **대소문자를 보존한 정확 일치** (내부 공백·줄바꿈은 유지), 지원 수식의 동등성 순이다. 라벨 대소문자만 무시한다. 따라서 일반 텍스트와 변수의 대소문자를 모두 무시하는 원본과 다르다. 영어 숫자 단어 변환은 이번 profile에서 지원하지 않는다. `twelve`와 `12`를 자동으로 같다고 처리하지 않는다.

수식은 현재 설치된 SymPy 1.14.0을 EMMA 비교 모듈에서만 지연 사용한다. 다른 benchmark CPU 경로에는 SymPy를 요구하지 않는다. 버전 불일치나 의존성 부재는 실행 전 입력/환경 오류이며 모델 오답이 아니다. 새 패키지는 설치하지 않는다.

지원 문법은 유한 정수·소수·과학 표기, 단항 +/-와 사칙연산, 괄호, 정수 지수 `^`/`**`, `sqrt(...)`, `pi`, LaTeX `\frac{...}{...}`, `\sqrt{...}`, `\pi`, `\cdot`, `\times`다. 곱셈은 명시해야 한다. `x+x`와 `2*x`는 지원하고 `2x`, 임의 함수 호출, 삼각함수, 적분, 행렬, 등식, 단위 환산은 지원하지 않는다. 임의의 LaTeX 명령을 삭제해 수식처럼 만들지 않는다.

문법을 모두 소비하는 제한된 parser를 사용해 허용된 SymPy 숫자·연산 생성자로만 식을 만든다. 문자열을 `eval`, `exec`, `sympify`, `parse_expr` 또는 동적으로 선택한 callable에 넘기지 않는다. 수식 파서 입력 512자, 연산 노드 64개, 괄호 깊이 16, 숫자 토큰 64자리, 과학 표기 지수 절댓값 100, 정수 거듭제곱 지수 절댓값 8로 제한한다. 정확한 문자열 비교는 이 파서 제한 전에 수행한다. 따라서 512자가 넘는 원본 코드 선택지도 정확히 같으면 정답이며 파서에 넣지 않는다. 숫자 계산의 중간 값은 유한 실수이고 절댓값 1e100 이하여야 한다. 이는 입력 실행·계산 자원 안전을 위한 제한이며 retry나 복구 시스템을 추가하지 않는다.

변수가 없는 두 식은 각각 유한 실수로 평가하여 Python `round(float(value), 2)`가 같으면 정답이다. 원본의 소수 둘째 자리 반올림을 유지하며 별도의 상대/절대 tolerance를 도입하지 않는다. 예를 들어 `1/2`와 `0.5`는 같다. 반올림 경계와 이진 부동소수점 영향은 테스트·문서에 드러낸다.

변수 식은 대소문자를 구분하는 ASCII 한 글자 변수의 유리수 계수 다항식만 지원한다. 변수는 최대 8개, 총 차수는 8, 전개 항 수의 사전 상한은 128이다. 나눗셈의 분모는 0이 아닌 유리수 상수만 허용한다. 음의 변수 지수와 변수 아래 제곱근은 지원하지 않는다. 두 다항식의 계수 차이가 모두 0이면 정답이며 변수에 임의 숫자를 대입해 추정하지 않는다. `x/x`와 `1`은 이 기호 비교 범위에 포함하지 않는다.

지원하지 않는 문법은 정확한 문자열 일치만 인정하고, 일치하지 않으면 `correctness=false`와 비교 진단을 함께 기록한다. 일반 오답과 unsupported expression, 비유한 값, 계산 제한을 구분한다. 예상하지 못한 내부 오류는 오답으로 삼키지 않고 실행 실패로 처리한다. 위 제한은 EMMA 전체의 수식 의미를 판정한다는 주장이 아니다.

## 5. 집계·실패·검증

전체 micro accuracy와 과목별 집계는 기존 report를 재사용한다. EMMA에 한해 문제 유형, 과목별 category, `subject + '_' + task` 집계를 추가한다. Coding category는 세미콜론으로 나누고 공백을 제거하며 같은 문항의 중복 tag는 한 번만 센다. 빈 category/task는 해당 세부 집계에서 제외하되 전체 분모에서 제외하지 않는다. 기존 summary categories는 과목을 계속 의미하므로 원본의 category 집계는 별도의 EMMA 필드에 둔다.

모든 집계에서 빈/형식 오류/지원하지 않는 응답은 관측된 오답이다. prediction 누락·infrastructure error는 미완료이며 해당 집계 accuracy는 null이다. 빈 집계도 accuracy=null이다. Coding tag의 합이 전체 문항 수보다 커질 수 있으며 tag 정확도를 전체 점수로 평균하지 않는다. 진단 횟수는 EMMA 보고서에 별도로 집계한다.

검증은 (a) 추출·비교·IO·집계 CPU 계약, (b) 설치한 원본 환경 없이 수행하는 명시적 함수 관찰, (c) 실제 mini 원본과 준비 산출물 전수 대응, (d) 기존 경로 회귀를 구분한다. gold 변화가 추출 결과를 바꾸지 않는 검사를 넣는다. 소수점/과학 표기, nested boxed, 2~8개 선택지, 소문자 원본 라벨, 59개 open-ended, `phy_6` 슬롯, 다항식 정오답, 반올림 경계, 허용되지 않는 호출·수식, 누락/환경 오류를 다룬다.

원본 직접 비교는 실제 실행한 helper와 입력 범위에만 한정한다. 공개 모델의 `true_false`를 정답 근거로 사용하지 않는다. 모델 로딩·추론·GPU·judge·full benchmark 성능·논문 프로토콜은 이번 검증에 포함하지 않는다. `protocol_kind=modified`, `paper_protocol_verified=false`를 유지한다.

## 6. 현재 근거와 다음 gate

데이터 근거는 [조사 기록](p5-emma-discovery.md) 및 [mini 검사 결과](../../results/evaluation/p5-mini-discovery-20260913-01/verification.json)다. 공개 Qwen_Direct 응답 400개를 원본의 순수 추출 helper로 조사했을 때 라벨 341개, decimal literal 59개였다([구문 관찰](../../results/evaluation/p5-design-discovery-20260913-01/published-response-syntax.json)). Gold나 공개 `true_false`는 이 구문 관찰에 사용하지 않았다. 이 파일에 한정된 관찰이며 다른 모델 응답의 지원 범위를 보장하지 않는다.

세부 계약 승인은 2026-09-13 사용자가 제시된 계약을 인용해 `ㅇㅋ`라고 응답한 것으로 기록한다. 아래는 self-review에서 고정한 세부 해석이다. 이번 v2는 이 내용을 포함해 동결하며 비평 중 덮어쓰지 않는다.

- 수식 512자 제한과 문자열 정확 일치를 구별했다. 원본 선택지 178개가 512자를 넘고 최대 길이는 3,504자다. 수식 안전 제한으로 정상적인 긴 코드 선택지의 정확 일치를 막지 않는다.
- 앞뒤 공백만 strip하며 문자열 내부의 공백·줄바꿈을 지우지 않는다. 단독 라벨은 A–H/a–h 중 현재 choices 범위에 속하는 한 글자만 인정한다. 접두사는 대문자와 `:` 또는 `.`를 즉시 이어 쓴 형태다. 소문자 접두사는 raw 후보로 남는다. 추출 1번의 순수 라벨 인식도 choices가 있는 객관식에 한정한다.
- 비유한 숫자 전용 표기는 후보 선택과 wrapper 제거 후에도 검사한다. 선택지 범위 밖 한 글자를 정답 선택지 내용으로 비교하는 것은 가능하지만, 유효한 라벨의 오답을 content로 재해석하지 않는다.
- `^`와 `**`의 연속 사용(`2^3^2`)은 거부한다. 거듭제곱은 단항 부호보다 우선하며 괄호가 이를 바꾼다. 지수 문법은 부호 있는 정수 literal로 한정하고, 그 literal을 괄호 또는 LaTeX 중괄호 한 겹으로 감싼 것도 허용한다. `(2^3)^2`는 허용하지만 수식으로 계산한 지수 `2^(3^2)`는 거부한다. `sqrt`와 `pi`만 이름 전체로 허용하고, 나머지 변수는 ASCII 한 글자다. 공백을 삭제해 여러 토큰을 연결하지 않는다. 숫자 소수점/지수부를 포함한 lexer는 전체 입력을 소비해야 한다.
- LaTeX frac/sqrt의 인자는 균형 잡힌 중괄호다. `x^{2}`처럼 literal 지수의 중괄호도 허용한다. 임의 `\text`, `\left`, `\right` 등 다른 명령은 수식 parser에서 허용하지 않는다. 추출에서 후보 전체를 감싼 `\text{...}` 한 겹을 제거하는 것과 구별한다.
- 문자 변수를 포함한 수식이 연산으로 상수가 되더라도 먼저 다항식 허용 조건을 확인한다. 변수 분모나 비유리수 계수를 상쇄한 뒤 허용 범위라고 추정하지 않는다. 상수 다항식과 숫자는 정규화 후 수치 비교를 적용하고, 자유변수가 남은 식끼리는 유리수 계수 다항식의 정확 일치를 사용한다. 변수 집합이 달라도 차이의 계수가 모두 0인지로 판단한다.
- 파서의 길이/토큰/깊이/지수/차수/항 수 제한을 위험한 생성자 호출과 전개 전에 확인한다. `sqrt`를 포함한 숫자 연산은 허용된 노드만 사용하며 검사 전에 광범위한 simplify/expand를 호출하지 않는다. 각 노드의 실수·유한성·크기 제한을 검사한다. 이는 아직 구현의 안전성 증거가 아니다.
- 새 EMMA target은 schema_version=1의 benchmark별 분기로 추가하고 다른 target과 Sample/Score 필드는 유지한다. category/task는 문자열(빈 문자열 허용), answer/content는 비어 있지 않은 문자열, index는 정확한 int 또는 null이어야 한다. 질문 유형·선택지·index·content의 대응 오류는 preflight 입력 오류다.
- config categories는 Chemistry/Coding/Math/Physics의 비어 있지 않은 고유 subset이며 실제 mini config는 네 과목 전부다. categories는 허용 목록 검사에만 사용하며 문항을 자동으로 필터링하지 않는다. manifest의 과목이 허용 목록 밖이면 실행 전 입력 오류다. 따라서 원본 400개 manifest를 채점하려면 네 과목 전부를 지정해야 한다. 합성 CPU 계약 fixture에서는 manifest와 일치하는 과목 subset을 사용할 수 있다. max_steps는 null/생략만 허용한다. profile이 유효한지 registry로 확인하고 SymPy 1.14.0의 존재/버전을 EMMA score 실행 전에 검사한다. 준비의 parquet/Pillow 의존성과 score 의존성을 분리한다.
- EMMA 추가 집계는 summary의 `emma` 객체 아래 `question_types`, `categories`, `tasks`, `comparison_diagnostics`에 둔다. `categories`는 subject 아래 원본 category별 통계를 둔다. `question_types`에는 두 canonical 유형을 항상 표시하며 없는 유형의 count=0, accuracy=null이다. category/task는 manifest에서 관측한 그룹만 만든다. 기존 `category_stats`의 count/observed/correct/complete/accuracy 계약을 재사용한다.
- 문항별 비교 진단은 기존 `Score.error`에 `unsupported_expression`, `non_finite_value`, `math_limit_exceeded` 등 고정 코드로 기록한다. 정확 일치 성공은 파서를 호출하지 않아 진단이 없다. 일반적인 비교 불일치의 error는 null이다. 지원하지 않는 표현은 parsed/false이며 추출 자체가 빈 후보이거나 비유한 숫자 전용 표기면 invalid_format/false다. 실제 코드의 예상치 못한 오류는 CLI 실행 실패로 전파한다.
- 준비는 검증된 원본에서 새 `data/evaluation/emma-mini400-v1/`에 manifest, SOURCE, source-selection 및 이미지 파일을 만든다. 파일 이름과 출력 JSON의 정확한 키·함수 signature·테스트 명령은 승인 후 구현 계획에서 확정한다. fixture는 data/, 결과는 results/, 로그는 logs/ 아래의 고유 run ID로 분리한다.

## 7. 동결 시 미검증 전제

현재 입증된 것은 원본 데이터의 로컬 검사와 제한된 원본 함수/응답 구문 관찰이다. 새 parser/scorer의 실행, 다항식 및 자원 제한의 구현 정확성, 실제 준비 manifest의 원본 대응은 구현 후 검증해야 한다. 공개 Qwen 한 모델의 단독 라벨/숫자 출력 분포는 다른 모델에서 지원하지 않는 표현이 적다는 증거가 아니다. 이 수정 profile은 원본 EMMA 전체 parser·judge·논문 점수와의 동등성을 약속하지 않는다.

## 8. v1 독해 검토에서 반영한 변경

독립된 context-free reviewer가 v1 최종 문서 사본의 모든 section을 읽고 minor gaps를 보고했다. 두 관찰은 (1) config subject subset의 필터 여부, (2) 연속 거듭제곱과 literal-only 지수의 충돌이다. v2는 각각 자동 필터 금지/manifest 허용 목록 검사와 연속 거듭제곱 거부로 고정했다. 이 동작 구체화가 있으므로 v1은 수정하지 않고 v2를 새로 동결한다. 채점 모집단·지원 연산 종류·숫자 반올림 기준·외부 실행 범위는 유지한다.

후속 first-nail의 대상은 **새로 준비할 고정 400개 manifest에 대해 처음 실제로 확보하는 prediction run**이다. 기존 manifest SHA-256, run fingerprint와 응답 파일 hash로 입력을 고정하고 추출기에 target을 넘기지 않는다. 의미상 정답인데 지원 제한/규칙 변경 때문에 거부된 사례를 하나라도 확인하면, cross-model 성능 비교 전에 그 사례와 정책 유지/수정 결정을 명시적으로 검토한다. 임계 비율을 사후 선택하지 않는다. 현재 그러한 ILVR 응답 run이 없으므로 not_run이며, 공개 Qwen의 다른 prompt/선택지 이력에 대한 구문 관찰로 대체하지 않는다. 이 절차 자체가 실제 응답 생성의 승인은 아니다.

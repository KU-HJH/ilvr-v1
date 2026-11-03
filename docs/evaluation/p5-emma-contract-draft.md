# P5 EMMA 세부 계약 초안

2026-09-13. 상태: **규칙 기반 접근·아래 세부 계약 승인; 최종 설계는 별도 문서**.

understood as: 사용자의 `ㅇㅋ`는 기존 환경에서 명시적인 규칙 기반 EMMA 평가 설계를 구체화하는 방향을 승인했다. 최종 design commitment, 상세 구현 계획 또는 구현을 승인한 것으로 확대하지 않는다. 이 문서는 brainstorming의 세부 계약 검토 자료이며 아직 동결 proposal이나 최종 spec이 아니다.

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

비교의 우선순위는 비어 있지 않은 문자열의 공백 제거 후 **대소문자를 보존한 정확 일치**, 지원 수식의 동등성 순이다. 라벨 대소문자만 무시한다. 따라서 일반 텍스트와 변수의 대소문자를 모두 무시하는 원본과 다르다. 영어 숫자 단어 변환은 이번 profile에서 지원하지 않는다. `twelve`와 `12`를 자동으로 같다고 처리하지 않는다.

수식은 현재 설치된 SymPy 1.14.0을 EMMA 비교 모듈에서만 지연 사용한다. 다른 benchmark CPU 경로에는 SymPy를 요구하지 않는다. 버전 불일치나 의존성 부재는 실행 전 입력/환경 오류이며 모델 오답이 아니다. 새 패키지는 설치하지 않는다.

지원 문법은 유한 정수·소수·과학 표기, 단항 +/-와 사칙연산, 괄호, 정수 지수 `^`/`**`, `sqrt(...)`, `pi`, LaTeX `\frac{...}{...}`, `\sqrt{...}`, `\pi`, `\cdot`, `\times`다. 곱셈은 명시해야 한다. `x+x`와 `2*x`는 지원하고 `2x`, 임의 함수 호출, 삼각함수, 적분, 행렬, 등식, 단위 환산은 지원하지 않는다. 임의의 LaTeX 명령을 삭제해 수식처럼 만들지 않는다.

문법을 모두 소비하는 제한된 parser를 사용해 허용된 SymPy 숫자·연산 생성자로만 식을 만든다. 문자열을 `eval`, `exec`, `sympify`, `parse_expr` 또는 동적으로 선택한 callable에 넘기지 않는다. 후보 512자, 연산 노드 64개, 괄호 깊이 16, 숫자 토큰 64자리, 과학 표기 지수 절댓값 100, 정수 거듭제곱 지수 절댓값 8로 제한한다. 숫자 계산의 중간 값은 유한 실수이고 절댓값 1e100 이하여야 한다. 이는 입력 실행·계산 자원 안전을 위한 제한이며 retry나 복구 시스템을 추가하지 않는다.

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

세부 계약의 인간 확인 후 내용을 반영한 written spec을 self-review하고 design proposal vN으로 동결한다. 그때 AGENTS.md의 hate 및 필요한 critique를 같은 동결 제안에 적용하고 design commitment vN을 합성한다. 이후 최종 written-spec 승인과 writing-plans로 진행한다. 이 초안에는 구현 소스, 실행 계획, 최종 승인 또는 비평 완료 주장이 없다.

2026-09-13 승인 기록: 사용자가 제시한 세부 계약을 인용해 `ㅇㅋ`로 확인했다. 이 초안은 당시 계약을 보존한다. 최종 검토 대상은 [design commitment v2](../superpowers/specs/2026-09-13-ilvr-emma-offline-design.md)다.

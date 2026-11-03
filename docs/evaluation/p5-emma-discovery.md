# P5 EMMA 조사 — 2026-09-13

상태: **EMMA-mini 원본 400개·이미지 614개 확보 및 전수 검사; v2 승인 후 계획 독해 수정 반영; v3와 상세 구현 계획 결합 승인 완료·P5 오프라인 구현 및 데이터 준비 완료**.

understood as: P4 완료 뒤 평가 이식을 P5 EMMA로 이어간다. 기존 오프라인 경로를 재사용하며, 데이터 구성과 자유응답 비교 계약을 확인한 뒤 설계를 검토받는다. 이번 `계속`은 아직 제시하지 않은 P5 상세 설계의 승인이 아니다.

## 확인한 근거

- `references.lock.json`의 7개 독립 저장소 root/origin/commit/clean 상태와 기록된 33개 파일 hash 확인. EMMA commit은 `47259952dd8e95bc1e3855a4164ccd3634f8eb1e`다.
- [`reference-verification.json`](../../results/evaluation/p5-discovery-20260913-01/reference-verification.json), 추가로 읽은 generation/config/공개 결과 파일의 [`supplementary-source-hashes.json`](../../results/evaluation/p5-discovery-20260913-01/supplementary-source-hashes.json).
- 원본 `evaluation/evaluate.py`, `evaluation/utils.py`, `evaluation/calculate_acc.py`, `data_utils.py`, `generate_response.py`, `configs/gpt.yaml`을 읽었다. 원본을 런타임 의존성으로 연결하지 않았다.
- EMMA의 tracked license 파일 부재는 기존 P0 관찰을 유지한다. 소스를 복사하는 구현은 진행하지 않았다.

## 실제 평가 대상 후보

공식 Hugging Face API metadata와 repository tree를 `data/evaluation/emma-source-discovery-20260913-01/`에 저장했다. 아래 개수는 metadata 관찰이며, parquet 전수 확인 결과가 아니다.

| 구성 | dataset revision | Chemistry | Coding | Math | Physics | 합계 | parquet bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| [full](https://huggingface.co/datasets/luckychao/EMMA/tree/6c87aec9048c489108170088e50b768646b5bcf9) | `6c87aec9048c489108170088e50b768646b5bcf9` | 1,176 | 564 | 892 | 156 | 2,788 | 258,204,107 |
| [mini](https://huggingface.co/datasets/luckychao/EMMA-mini/tree/6b9ae9a74733bb57f0b741213f8d0ec9ebae067a) | `6b9ae9a74733bb57f0b741213f8d0ec9ebae067a` | 100 | 100 | 100 | 100 | 400 | 50,613,971 |

현재 후속 구현 검증 목적에는 과목별 균형과 실행 규모를 이유로 mini를 권고했다. 이후 사용자가 실수 입력을 취소하고 `잘못누름. 계속`이라고 하여, 권장한 mini 기준으로 계속한다는 해석을 알리고 원본 확보·검사를 진행했다. 이는 상세 P5 설계의 승인이 아니다. mini parquet 4개를 내려받았고 이미지 bytes는 원본 parquet 안에 보관했다. 평가 manifest는 아직 만들지 않았다. 어느 쪽도 논문 split으로 간주하지 않는다.

원본에 포함된 `results/EMMA-mini/open-source/Qwen_Direct.json`은 400개이며, 객관식 341개와 자유응답 59개다. 자유응답 정답 59개는 해당 파일에서 모두 정수 문자열이었다. 이는 공식 dataset revision과의 대조 전 관찰이며, 모델 점수나 전체 EMMA의 응답 범위를 증명하지 않는다. 답안이 포함된 목록은 `data/evaluation/emma-source-discovery-20260913-01/published-result-inventory.json`에 저장했다. 원본 `random_choice.json`은 객관식 341개만 포함하므로 400개 manifest의 대체물이 될 수 없다.

## 추출·비교·집계 관찰

원본 순수 함수 3개만 AST로 분리해 16개 응답을 관찰했다. [`source-extraction-observations.json`](../../results/evaluation/p5-discovery-20260913-01/source-extraction-observations.json)에 입력과 출력을 기록했다. 전체 원본 환경 또는 모델 평가는 아니다.

- `The final answer is 0.5`는 `0`, `The final answer is -2.75`는 `-2`, `answer: 1.5e-3`은 `1`로 잘린다. 원인은 marker 뒤 첫 `.`에서 나누는 코드다. 숫자만 있는 `0.5`는 유지된다.
- 중첩 괄호가 있는 `\boxed{\frac{1}{2}}`는 내부 수식을 유지한다. 완성된 box가 여러 개면 마지막 것을 사용하고, 뒤에 미완성 box가 있으면 앞의 완성된 것을 반환한다.
- raw `nan`과 `inf`도 원본 숫자 추출 경로에서 반환한다. 새 비교 계약에서 유한성 정책을 명시해야 한다.
- 원본 비교는 소문자 문자열 일치, 영어 숫자 단어 변환, 수치 소수 둘째 자리 반올림, symbolic simplify 순서다. MCQ의 정답 label 또는 정답 선택지 내용 중 하나와 일치하면 참으로 처리한다. `gt_content`는 생성 준비 시 정답 index의 선택지에서 만들어진다.
- 원본 집계는 전체 문항 기준 micro accuracy, 과목, 문제 유형, 과목별 category, 과목과 task 조합이다. Coding category는 세미콜론으로 여러 범주에 들어갈 수 있다. 원본의 빈 집계 0 처리와 기존 신규 경로의 `accuracy=null` 계약은 명시적으로 구분해야 한다.

현재 venv는 SymPy 1.14.0, datasets 4.2.0, pyarrow 21.0.0이다. `latex2sympy2`, `word2number`, `antlr4-python3-runtime`은 설치되어 있지 않다. 의존성 설치·업그레이드 없이 조사했다. 원본 전체 동등성 함수의 실행 비교는 하지 않았다.

## 다음 설계에서 해결할 사항

1. mini 원본의 ID, 질문, 선택지, 정답, context, image_1–image_5 슬롯과 답안 유형의 전수 검사는 완료했다. 아래 후속 관찰을 준비 경로의 계약에 반영한다.
2. 기존 prepare/score 경로를 유지하면서 자유응답 target과 유형별 집계를 어떻게 추가할지 정한다. 이는 공통 계약 변경이므로 architectural brainstorming 경로를 적용한다.
3. 소수점 수정, 유한 숫자, 영어 숫자 단어, 안전한 수식 문법, 반올림·기호 동등성 정책과 지원하지 않는 입력 처리를 명시한다. 원본과 다른 규칙은 수정 profile로 구분한다. 모델 출력에 안전하지 않은 `eval`을 사용하지 않는다.
4. 원본 호환 의존성을 별도로 도입하는 접근, 현재 의존성으로 제한된 안전 비교를 구현하는 접근, judge 기반 접근의 비용·지원 범위를 비교한다. 규칙 기반 접근을 선택했으며 구현은 아직 시작하지 않았다. Judge는 기본 off이고 별도 승인 없이는 호출하지 않는다.
5. 설계 section 검토와 최종 commitment 승인 뒤 상세 계획·구현으로 진행한다. P1–P4 승인으로 새 P5 계약을 승인받았다고 간주하지 않는다.

## 평가 독립성 점검

`mandela` 읽기 전용 점검에서 주의할 패턴은 **Shared hallucination / Tautology**다. 자체 scorer와 자체 fixture의 일치를 EMMA 정답 검증으로 보고하면 같은 설계가 양쪽 근거가 된다. 수정은 dataset 원본에서 정답을 독립적으로 가져오고, 추출에는 gold를 전달하지 않으며, 의도된 규칙 변경 사례와 원본 함수 비교를 따로 보고하는 것이다. 공개 모델 결과의 `true_false`는 독립 정답으로 사용하지 않는다.

이번 조사에는 새 scorer가 없고 모델 성능 결과도 없다. 위 점검은 이후 검증 설계의 한계를 식별한 것이며 독립적인 정확성 인증이 아니다. 현재 실행 근거는 [`discovery-summary.json`](../../results/evaluation/p5-discovery-20260913-01/discovery-summary.json), 로그는 `logs/evaluation/p5-discovery-20260913-01/discovery.log`에 있다.

## mini 원본 전수 검사

실행 ID: `p5-mini-discovery-20260913-01`. 원본 저장 경로는 `data/evaluation/emma-source-discovery-20260913-01/EMMA-mini/<subject>/test-00000-of-00001.parquet`다. 4개 파일의 크기와 SHA-256은 고정 revision repository tree의 LFS 정보와 모두 일치했다. 추가 Reference 비교 전에 7개 저장소의 root/origin/commit/clean/hash를 다시 확인했다.

| 과목 | 객관식 | 자유응답 | 합계 |
|---|---:|---:|---:|
| Chemistry | 78 | 22 | 100 |
| Coding | 100 | 0 | 100 |
| Math | 63 | 37 | 100 |
| Physics | 100 | 0 | 100 |
| 전체 | 341 | 59 | 400 |

- ID 400개가 고유하다. 질문은 모두 비어 있지 않고 문항별 이미지가 최소 1개 있다. 자유응답 정답 59개는 모두 정수 문자열이며, 빈 options는 `[]` 45개 / `null` 14개다.
- 객관식 선택지 수 분포는 2개: 11문항, 3개: 2문항, 4개: 250문항, 5개: 66문항, 7개: 1문항, 8개: 11문항이다. 선택지는 모두 비어 있지 않다. Physics의 소문자 정답 라벨은 21개다. 원본 `build_query`가 `.upper()`로 index를 구하는 대소문자 처리에 맞추면 모두 유효하다. 원본 답안을 수정할 필요는 없다.
- 이미지 614개를 디코딩했다: PNG 463, JPEG 149, WEBP 2. 문항별 이미지 수는 1개: 313문항, 2개: 36문항, 4개: 26문항, 5개: 25문항이다. 재인코딩하지 않았다.
- `phy_6`은 `image_1`이 없고 `image_2`, `image_3`가 있다. 질문도 `<image_2><image_3>`를 참조한다. 번호를 압축하거나 question token을 바꾸면 안 된다. 모든 문항에서 질문·context·선택지의 이미지 참조에 대응하는 슬롯이 있다.
- context는 368개가 빈 문자열이고 32개가 텍스트다. 공개 결과의 `chem_111` options 2개에 있는 깨진 문자는 고정 dataset 원본에서는 LaTeX alpha 표현이다. 원본을 권위 있는 입력으로 유지한다. 나머지 399문항은 비교한 질문·선택지·정답·분류·context 필드가 공개 결과와 일치했다.
- 최초 임시 검사는 대문자·연속 이미지 슬롯을 가정해 22개 관찰을 남겼다. 위 원본 의미를 적용한 재검사에서 미해결 입력 오류는 0개이며, 제외 문항·원본 정답 수정을 하지 않았다. 이것은 새 preparation CLI의 테스트 결과가 아니다.

[`data-validation.json`](../../results/evaluation/p5-mini-discovery-20260913-01/data-validation.json)에 집계와 차이를 기록했다. 문항·정답·이미지별 목록은 모두 `data/evaluation/emma-source-discovery-20260913-01/`의 `mini-record-inventory.json`, `mini-image-inventory.json`, `mini-initial-contract-observations.json`에 저장했다. 다운로드 로그는 `logs/evaluation/p5-mini-discovery-20260913-01/download.log`다.

## 수식 비교 경계와 접근 후보

원본 `is_equal`의 실제 제어 흐름을, parser 출력만 통제한 작은 관찰로 확인했다. `x+x`와 `2*x`를 각각 정상 SymPy 식으로 공급해도 숫자 변환에서 실패해 `simplify`가 호출되지 않고 false를 반환했다. `0.5`와 `1/2`는 true였다. [`comparator-control-flow-probe.json`](../../results/evaluation/p5-mini-discovery-20260913-01/comparator-control-flow-probe.json)은 **parser와 숫자 단어 변환을 대체한 함수 관찰**이며, 원본 parser 일치나 전체 원본 환경 실행 증거가 아니다. 따라서 자유변수 수식 동등성을 추가하면 원본 동작 수정으로 기록해야 한다.

접근 후보는 다음과 같다. 2026-09-13 사용자에게 첫 번째 방향을 권고했고 사용자의 `ㅇㅋ`로 해당 방향의 설계 구체화를 승인받았다. [세부 계약](p5-emma-contract-draft.md)도 승인받았고, [최종 design commitment v2](../superpowers/specs/2026-09-13-ilvr-emma-offline-design.md)의 written-spec 검토를 요청한다. 2026-09-13 최종 설계 v2 승인도 받았다.

1. **현재 환경의 명시적 규칙 기반 평가(권고):** 기존 prepare/score 경로와 SymPy 1.14.0을 사용하고, 소수점·분수·명시적으로 지원하는 수식 문법과 비교 정책을 고정한다. 원본과 다른 추출·비교를 수정 profile로 기록한다. 지원하지 않는 문법을 임의로 정답 처리하지 않는다. 구체적인 문법·오류 정책은 상세 설계에서 검토한다.
2. **원본 의존성 추가:** `latex2sympy2==1.9.1`, `antlr4-python3-runtime==4.7.2`, `word2number==1.1`을 별도로 검토해 도입하면 원본 함수 대조 범위가 넓어질 수 있다. 원본 EMMA가 이 버전을 고정했다는 증거는 없으며, 도입 자체가 원본 환경 재현은 아니다.
3. **judge 포함:** 의미 판단을 별도 모델에 맡길 수 있으나, 모델·프롬프트·비용·실패 정책과 독립 검증 범위가 추가된다. 현재는 judge off를 권고한다.

두 PyPI package의 metadata와 배포 archive만 `data/evaluation/emma-source-discovery-20260913-01/dependency-inspection/`에 받아 읽었다. 설치하지 않았다. `latex2sympy2`에서 저장된 변수값에 `parse_expr`를 호출하는 경로를 확인했다. [SymPy 공식 문서](https://docs.sympy.org/latest/modules/parsing.html#sympy.parsing.sympy_parser.parse_expr)는 `parse_expr`가 `eval`을 사용한다고 명시한다. 이 부분 검색만으로 parser 전체의 안전성을 판정하지 않으며, 모델 출력의 문자열을 해당 함수에 직접 넘기는 설계는 채택하지 않는다.

상세 실행 검토 대상: [P5 구현 계획](../superpowers/plans/2026-09-13-ilvr-emma-offline-plan.md). 계획 작성은 승인됐으며 구현 실행은 아직 시작 전이다.

계획 독해 후 수정: 분자/분모 1e100 제한은 잘못된 해석으로 철회했다. 작은 수를 허용하면서 계산 자원을 제한하는 별도 4,096비트 한도를 [design commitment v3](../superpowers/specs/2026-09-13-ilvr-emma-offline-design.md)에 명시했고, 실행 계획과 함께 새 승인을 요청한다. 구현은 아직 시작 전이다.

## 실행 승인 기록 — 2026-09-13

최신 사용자 `진행해`로 제시된 design commitment v3와 상세 실행 계획을 함께 승인받았다. 위 승인 대기 문구는 당시 검토 이력이다. 현재 상태는 **승인 완료·P5 오프라인 구현 및 데이터 준비 완료**이며 task별 재승인을 요청하지 않는다.

## 실행 완료 기록 — 2026-09-13

P5 오프라인 구현과 고정 mini 준비를 완료했다. 실제 출력은 `data/evaluation/emma-mini400-v1-r02/`이며 400문항·614이미지·제외 0개를 구현 전 원본 목록과 전수 대조했다. 최종 P1–P5 CPU 계약 550개와 별도 런타임 스텁 7개가 통과했다. 실제 모델 응답 평가·모델별 first-nail·원본 전체 환경·GPU·논문 프로토콜은 미실행이며 P6는 남아 있다. 검증 기록: `docs/evaluation/validation.md`, `results/evaluation/p5-final-20260913-01/verification.json`. 앞선 승인/실행 대기 문장은 당시 이력으로 보존한다.

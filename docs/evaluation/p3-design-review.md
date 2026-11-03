# P3 design commitment v1 비평 기록

대상은 동결한 [proposal v1](p3-design-proposal-v1.md) 하나이며 SHA-256은
`b20cfc5994868dd18d4a75eebaeeecda25951901236f90a90c1a4d663cc76ede`다.
읽기 전용 hook은 root controller가 수행했다. 구현이나 phase 전환을 위임하지 않았다.

## hate

**root:** 400개를 유지한다는 이유로 빈 gold를 추측 보정하면 원본과 독립적으로
고정되지 않은 정답을 새 scorer가 정답이라고 확인하는 평가가 된다.

**first_nail:** 문제 두 개의 원본 JSONL에서 option/label/rationale를 함께 읽고,
문제 이미지와 대조해 제안한 label→content mapping에 반례가 있는지 확인한다.
기존 TEST 400개 전체의 source ID join과 문제 이미지 bytes도 고정 원본과 대조한다.

**상태:** 저비용 읽기 전용 조사로 이미 수행한 증거가 있다. 원본 rationale는
creation-12125의 결과를 18cm, creation-12164를 24cm로 명시한다. 비어 있지 않은
선택지를 순서대로 둔 두 명시적 배열에서 B→18, D→24가 일치한다. 문제 이미지도
두 장 직접 확인했다. TEST 400개 source ID와 질문·선택지 변환이 유일하게 대응하며
문제 이미지 400개의 bytes는 pinned LFS SHA-256과 일치하고 decode됐다.
근거는 [조사 기록](p3-comt-discovery.md) 및 그 결과 파일이다.

이 관찰은 제안한 두 보정에 대한 외부 source 근거다. 원본이 항상 옳다는 수학적
보증, 나머지 398개 gold의 독립 풀이, 새로운 scorer의 구현 정확성은 아니다.
실제 파일 보정과 구현은 아직 수행하지 않았으며 인간 승인 대상이다.

## macrothink 적용 판단

추가 fresh reads는 생략했다. 기존 multi-benchmark migration의 P3 순서와
TEST 400개 모집단을 사용자가 명시적으로 정했고, 이 단계는 기존 오프라인
구조에 CoMT 계약을 더하는 범위다. 새 모델/학습/대규모 실행 방향을 선택하지
않는다. 주요 위험은 방향 추측이 아니라 위 두 보정의 외부 근거이며 직접
조사했다. 이 판단을 독립적인 방향 검증이나 에이전트 합의로 표시하지 않는다.

## prism

두 실패 모드를 구분해 같은 동결 문서를 읽었다. 별도의 agent 투표는 수행하지 않았다.

- 평가 타당성 관점: **조건부 수용**. 두 보정이 원본 rationale와 label에 근거하고
  변경된 데이터 프로토콜 및 400개 분모를 공개하지만, 기존 raw TEST와 동일한
  입력/정답 계약은 아니므로 정확한 보정에 대한 인간 승인이 필요하다.
- 기존 동작 보존 관점: **설계 수용**. target dispatch를 추가하고 VSP 경로와
  generation 소스를 보존하는 경계가 명시돼 있다. 실제 회귀 여부는 구현 뒤
  P1/P2 테스트 및 source/runtime witness 전까지 미검증이다.

두 관점은 제안의 구현 범위에 충돌하지 않는다. 승인을 가르는 한 질문은
**원본 파일을 보존한 채 새 데이터 profile에 정확히 두 보정을 적용하여 동일한
400개를 준비하는가**다. 회귀 테스트 계획을 실행 결과로 격상하지 않는다.

## mandela — 근거 독립성

구성 요소는 기존 ILVR 모델(이번에는 미실행), 새 CoMT scorer(구현 전), root
설계자, pinned 외부 source와 기존 TEST selection이다. 8개 pattern을 점검했다.
제안이 명시한 제한된 주장 안에서 새로운 확정 leakage는 발견하지 않았다.
다만 다음 경계가 검증에 필수다.

원본 image dict 전체와 rationale를 넘기는 경로는 정답 단서를 입력에 주므로
IMAGE0/질문/선택지만 허용한다. identity join은 image token/category와 질문으로
구성하고 gold는 사후 일치 검사에만 쓴다. 합성 fixture 점수는 모델 성능이나
독립적인 benchmark gold 검증이 아니다. 제한된 함수 대조는 실제 실행 후에만
보고하며 원본 전체 환경 재현이라고 하지 않는다. TRAIN/TEST ID 교집합 0을
다른 데이터에 대한 OOD나 학습 비노출로 확장하지 않는다.

이 hook의 자체 판단도 독립적인 성능 실험이 아니다. 독자는 pinned 원본의
두 rationale, source/TEST hash와 join/image 대조 결과로 근거 범위를 확인할 수 있다.

## Superpowers 합성

proposal v1의 문제·범위·인터페이스·행동 변경은 없다. 데이터 400개 유지와 두
명시적 보정, modified profile, source 불변, gold 비의존 추출을 그대로 commitment
v1로 전달한다. 확인한 근거와 아직 실행하지 않은 코드 검증을 구분한다.
설계 수용을 뒤집는 미해결 기술 이견은 없으며, 두 보정의 승인은 아직 대기다.
승인 이전 구현, 모델 실행, 의존성 설치 또는 commit 권한을 이 기록에서 추론하지 않는다.

## Context-free shower 결과와 반영

`/root/p3_design_cold_read`는 대화 이력 없이 설계 본문만 읽고 **minor gaps**로
판정했다. 두 보정·400개 유지·오프라인 범위·미승인 상태를 의도대로 이해했다.
지적한 literal `[Answer]`/`ANSWER:` 조건, `(A)` 괄호 범위, 변경 허용 코드와
바이트 보존 코드, 제한된 함수 대조의 필수 여부, 모델 응답 형식 실패와 잘못된
Prediction 레코드의 구분을 written commitment에 명문화했다. scorer profile과
데이터 준비 profile도 구분했다. 이 설명은 같은 제안의 원본 관찰 행동과 검증
범위를 명료하게 하며 새로운 동작이나 범위를 추가하지 않는다. 동결 제안은 그대로다.

리뷰어가 받은 inline 사본은 참조 파일명만 포함했으므로 실제 상대 링크를 열어
근거를 확인하지 못했다고 보고했다. written spec에는 상대 링크가 있으며 root가
최종 문서 검증에서 확인한다. 이 cold read는 데이터 정확성이나 구현 테스트의
대체 증거가 아니다. 별도 broad review를 반복하지 않는다.

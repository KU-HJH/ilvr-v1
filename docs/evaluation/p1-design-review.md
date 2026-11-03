# P1 설계 비평 기록 — 2026-09-10

현재 최종 검토 대상은 **design commitment v2**다. 아래 v1 기록은 보존하고 마지막 절에 v2 재검토를 기록한다.

대상은 [design proposal v1](p1-design-proposal-v1.md)이며 동결 SHA-256은 `050b2d054c1f15f09c44aebf2956a2dd6f62dc35fd25c0de617ad691bf1d286b`다. 아래 비평 사이에서 제안 본문을 수정하지 않았다. 모든 hook은 root가 호출했으며, 에이전트는 읽기 전용 자문만 수행했다.

## hate

```text
{root: "원본 환경 전이를 독립적으로 관찰하지 못한 채 새 simulator와 같은 설계에서 나온 fixture가 서로를 검증하면, 틀린 평가를 일관되게 만드는 데 그칠 수 있다.",
 first_nail: "고정된 Gym 0.26.2의 원본 make/reset/step 경로에서 작은 지도 R/RD, hole, 경계, limit 직전/당일/이후의 state/reward/terminated/truncated를 관찰하고, 구현 후 새 simulator와 대조한다."}
```

first_nail은 제안된 실험이며 실행 결과가 아니다. Gym 미설치와 Python/NumPy 차이가 있고 환경 변경은 승인되지 않았으므로 **not_run**이다. 현재 함수 14개 관찰과 원본 소스 열람을 이 실험의 통과 증거로 대체하지 않는다. P1을 CPU 계약 구현에 한정하고 upstream/paper 검증을 보류하는 위험 수용 여부를 최종 설계 gate에 남긴다.

## macrothink

서로 다른 benchmark에 퍼질 공통 계약의 방향을 잘못 잡는 비용이 두 번의 짧은 fresh read보다 크다고 판단해 실행했다. 동일 모델을 상속한 context-free 자문 두 개(`/root/p1_fresh_read_a`, `/root/p1_fresh_read_b`)를 사용했다. 선택한 모듈/접근 이름과 구체적인 추천안을 빼고 목표, 기존 코드 관찰, 데이터 metadata 부재, 변경 금지/저장 경로/검증 제약만 전달했다. 두 입력은 같은 동결 제안에서 도출한 동일한 사실 목록이었다. 이 자문은 전체 제안의 코드 리뷰가 아니다.

**divergent-incompatible — 착수 순서:** 두 fresh read 모두 scorer보다 VSP의 재현 가능한 record/manifest 복구 경로를 먼저 확정하라고 제안했다. B는 기존 prediction 하나를 완전한 scoring record로 복구하지 못하면 다음 구현을 manifest/provenance부터 하라고 했다. 제안 v1은 실제 데이터 준비를 P2에 두고 P1 synthetic 계약/scorer 구현을 허용하므로, 실제 데이터 준비와 scorer 구현의 선후관계에 차이가 남는다.

**divergent-compatible — 근거 구분:** A는 source-established / locally-observed / unresolved를 계약의 각 규칙에 구분하라고 했다. B는 map·ID·generation provenance를 연결하지 못하면 정확해 보이는 숫자도 재현 가능하지 않다고 지적했다. 두 지적은 같은 근본 문제인 **채점할 대상의 식별 가능성**으로 묶인다.

**convergent — reassurance only:** 추출과 gold 비교 분리, 원본/GPU/논문 검증 구분, 기존 suffix 반환 재사용, reviewable VSP 계약 우선에는 공통 의견이 있었다. 이를 정확성의 독립 증거로 취급하지 않는다.

자문 후 읽기 전용으로 upstream VSP의 6개 level별 img/table/pure_text ID 집합과 level5/0의 지도 설명을 확인했다. 복구 후보는 확인했지만 기존 prediction과의 join은 확인하지 못했다. 따라서 선후관계 차이가 모두 해소되었다고 주장하지 않는다. 최종 commitment는 기존 P1/P2 경계를 유지한 채 이 이견을 명시적으로 사용자에게 넘긴다.

## prism

동결 제안 전체를 root가 두 failure mode로 검토했다. 별도 agent 실행이나 runtime 증거를 주장하지 않는다.

- 과학적 적용 가능성 — **unclear**: 실제 manifest/prediction join과 원본 환경 witness가 없어 P1 완료를 benchmark 준비 완료로 해석할 수 없다.
- 기존 기능 호환성 — **pass at design level**: 추가 offline 모듈로 한정하고 기존 parser/scorer/추론/CLI의 변경을 금지해, 새 strict 규칙이 기존 사용자에게 적용되는 변경을 피한다. 실제 테스트 통과를 의미하지 않는다.

단일 다음 질문: **P1을 합성 입력으로 검증한 오프라인 계약과 VSP scorer까지로 승인하고, 실제 benchmark 사용 가능성과 원본 환경 일치는 P2 이후의 별도 검증 조건으로 유지할 것인가?** 이 질문이 최종 인간 승인 gate에 남는다.

## mandela / factchk 적용 범위

검증 설계에서 Shared hallucination / Tautology 위험은 새 simulator와 hand-written fixture만으로 원본 일치를 주장할 때 발생한다. 독립성 보완은 pinned 원본 환경 runtime witness와 원본 map/label/manifest 대조이며 아직 실행되지 않았다. 외부 사실 확인은 pinned reference 소스와 Gym 0.26.2의 등록/TimeLimit 소스를 사용했다. 코드가 선언한 의도만으로 실행 동등성을 판정하지 않았다. audit 문서는 관찰/정적 분석/미검증을 구별한다.

## synthesis

최초 commitment v1은 제안 v1의 구현 범위·인터페이스·문법·환경·집계 동작을 바꾸지 않고 map 복구 후보, 남은 이견, 위험 수용 대기 상태를 추가했다. 이후 자기검토와 cold read에서 명확화가 필요해 아래 v2로 갱신했다. 동결된 제안 v1 파일은 수정하지 않았다.

## shower와 자기검토

별도 context-free reviewer `/root/p1_document_cold_read`에게 commitment v1의 전체 본문을 inline으로 전달했다. 인접 파일/대화 의도에 접근하지 않게 했고, 8개 절 전체를 읽은 결과 **minor-gaps** 판정을 받았다. 설계 승인 문서이며 구현 명세가 아니라는 의도와, 원본 환경/실데이터 검증의 한계를 올바르게 읽었다.

지적된 공백은 숫자→문자 지도 의미, category 배정/부분 미완료, 생성 status/errored response 처리, 상세 계획 이후 승인 경계였다. root가 이를 v2에 명시했다. 추가 자기검토에서 빈 prediction 파일을 입력 오류로 규정한 v1이 전체 prediction 누락의 incomplete 계약과 충돌함을 발견해 수정했다. 공통 run의 template identity와 샘플별 prompt hash도 분리했다. v2에 대해 별도 shower를 다시 실행했다고 주장하지 않는다.

## design proposal v2의 재검토

[동결 제안 v2](p1-design-proposal-v2.md)의 SHA-256은 `af47a62a48e6349d123d824bfd6f32da6636b4bf53269ea86ed1b407edd4eda9`다. 이 버전 전체를 기준으로 hate→macrothink→prism을 다시 적용했다.

**hate:** root는 여전히 독립 원본 환경 증거 없이 simulator/fixture 상호 일치를 원본 정확성으로 오인하는 문제다. first_nail도 pinned Gym의 make/reset/step 경로 대조로 동일하며, 환경 변경 미승인으로 **not_run**이다. v2의 JSONL/identity 명확화가 이 위험을 해소하지 않는다.

**macrothink:** 새 context-free 두 read(`/root/p1_v2_fresh_read_a`, `/root/p1_v2_fresh_read_b`)에 v2에서 추천 접근/모듈명을 제거한 동일 사실 목록을 전달했다. 지도 파일 집합과 한 쌍의 내용 확인, 빈 Prediction JSONL 처리, 공통 run identity 분리를 사실/요구로 추가했다.

- divergent-incompatible: A는 기존 출력의 source-map 연결을 복구하거나 새 prediction 생성이 필요한지 먼저 결정하라고 했다. B도 scorer 충실도보다 identity 연결 경로의 확인을 우선했다. 실제 준비를 P2에 두는 제안과의 착수 순서 차이는 유지된다.
- divergent-compatible: 두 read는 최소 산출물을 source→manifest→prediction join 계약과 검증 공백을 표시한 worked example로 좁혔다. 이는 P1 계약을 실제 데이터에 연결할 경로를 설명해야 한다는 요구다. 상세 계획에서 synthetic fixture만으로 실제 평가가 준비되었다고 표기하지 않는다.
- convergent (reassurance only): read-only audit/design, 고정 분모, 오류 분리와 프로토콜 검증 공백 공개에는 공통 의견이었다. 둘의 일치는 독립 source/image/prediction 검증을 대신하지 않는다.

**prism:** 원본 과학적 적용 가능성 unclear / 기존 기능 호환성 pass at design level로 유지된다. 단일 다음 질문은 실제 benchmark 준비는 P2 이후로 남기는 P1 CPU 계약/scorer 경계를 승인할지 여부다.

**최종 합성:** [design commitment v2](../superpowers/specs/2026-09-10-ilvr-eval-foundation-vsp-design.md)는 v2 제안과 같은 동작을 담는다. 이전 이견을 숨기지 않으며 `risk waiver: 아직 없음`을 명시한다. 아직 작성하지 않은 상세 implementation plan이나 구현 완료를 주장하지 않는다.

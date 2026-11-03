# P2 설계 비평과 근거

작성일: 2026-09-10. Superpowers brainstorming 단계의 읽기 전용 비평을 root가 합성한 기록이다. 비평은 코드 수정이나 구현 승인을 대신하지 않는다.

대상 동결 제안: [p2-design-proposal-v1.md](p2-design-proposal-v1.md), SHA-256 `b312332c5864f64d09fe6d6be622e48c8a6fe4bc80b224bd56fb05f0a1cecfcf`.

## 자기검토와 cold read

동결 전에 scope, P1 호환 경계, 데이터 위치, source와 runtime 분리, 오류 상태, 가능한 근거와 미실행 검증을 대조했다. `shower`의 context-free reviewer에게 문서 사본만 제공하고 저장소·이웃 문서 접근을 금지했다. 전체를 읽은 verdict는 **minor gaps**였다. 리뷰어는 입력 순서·gold 분리·저장 위치·중단 정책·승인 상태·근거의 한계를 추측 없이 이해했다고 보고했다.

초안에 대해 나온 세부 문제와 root의 반영:

1. GPU 비교의 sampling/RNG/상태 통제: greedy 설정, 난수 상태 정렬, 기존 controller/cache 초기화 확인과 불명확할 때 미완료 처리를 명시했다.
2. 샘플 inference 실패의 위치: generation-details와 P1 Prediction 양쪽에 infrastructure_error를 쓰며 미수신 필드는 null이라고 명시했다.
3. run fingerprint 구성: P1의 세 metadata 필드 hash와 별도 manifest hash를 명시하고 weight content hash가 아님을 분명히 했다.
4. 기존 schema/loader/인자 세부 범위는 지정한 코드와 선행 spec이 계약 원문이다. 상세 함수 signature와 generation 값 검사는 승인 후 implementation plan에서 고정한다.

이 수정들은 동결 전에 이뤄졌다. reviewer는 production 구현이나 GPU 결과를 검증하지 않았다. cold read 이후 반복 광범위 review는 하지 않았다.

## hate

```json
{
  "root": "원본 그림과 target 지도 사이의 대응이 틀리면, bridge가 기존 함수를 완벽히 재사용해도 실제로 보지 않은 지도로 VSP를 채점한다. table과 pure_text가 서로 일치한다는 사실만으로는 이 핵심 입력 대응을 입증할 수 없다.",
  "first_nail": "각 지도 크기에서 원본 PNG 한 장씩을 열어 player/goal/hole의 1-based 좌표를 직접 읽고, 같은 basename의 table/pure_text와 대조한다. 한 장이라도 다르면 prepare 설계를 확정하지 말고 원본 매핑부터 다시 조사한다."
}
```

상태: **실행한 읽기 전용 source inspection에서 반례 없음**. 고정된 level3–level8의 0번 총 6장이 대응했다. level5/0은 이전 turn에서, 나머지 5장은 이번 turn에서 직접 view_image로 열었다. 이는 모델/GPU 실행이나 새 prepare 구현을 사용하는 실험이 아니며, 승인된 원본 조사 범위의 파일 읽기다. 이 관찰을 제안 뒤 새로 얻은 모델 결과처럼 취급하지 않는다.

관찰한 1-based 좌표:

| ID | player | goal | holes |
|---|---|---|---|
| level3/0 | (3,2) | (1,2) | 없음 |
| level4/0 | (2,2) | (3,2) | (2,4), (3,3), (4,1), (4,4) |
| level5/0 | (4,3) | (5,1) | (2,2), (3,5) |
| level6/0 | (5,6) | (4,3) | (1,2), (2,2), (2,3), (4,4), (4,6), (5,4), (6,4) |
| level7/0 | (1,4) | (1,3) | (1,5), (1,7), (2,2), (3,5), (4,5), (6,1), (6,6), (7,2) |
| level8/0 | (4,4) | (4,5) | (1,2), (1,8), (2,4), (2,5), (5,2), (5,3), (5,4), (6,4), (7,3), (7,5), (7,6), (7,8), (8,7) |

남은 불확실성: 선택은 무작위 표본이 아니며 594개 나머지 PNG의 의미 대응은 미검증이다. 600개 table/pure_text 좌표와 파일 집합 대조는 이미 수행했지만 두 형식이 독립적으로 만들어졌다고 가정하지 않는다. first_nail은 작은 반증 시도이지 전체 정당성 증명이 아니다.

## macrothink 적용 판단

이번에는 실행하지 않았다. 다음 단계의 문제는 P1에서 분리·승인된 VSP 실데이터/추론 연결이고, 이번 입력 방향도 사용자가 선택했다. 기존 source와 기존 반환값을 재사용하는 좁은 경계이며 신규 모델·훈련·평가 방법 선택을 하지 않는다. 현 단계에서 다른 문제를 풀 위험이 2–5개의 추가 fresh read 비용을 정당화할 만큼 크다고 판단하지 않았다. 이 생략을 방향 검증 통과로 기록하지 않는다.

## prism

동일한 동결 제안을 root가 서로 다른 두 실패 축으로 검토했다. 별도의 모델들 간 합의나 구현 review로 설명하지 않는다.

| Lens | Verdict | Load-bearing reason |
|---|---|---|
| 기존 추론 경로 호환성 | pass, 설계 범위 | 기존 함수 호출로 generation을 위임하고 ModelInput만 넘기며 직접 호출과 bridge의 입력/토큰 비교를 요구한다. 실행 동등성은 아직 검증하지 않았다. |
| 실제 benchmark의 해석 가능성 | unclear | 600행의 source metadata 대응은 있지만 PNG 의미의 전수 대응, 실제 GPU 비교와 논문 프로토콜은 확보하지 못했다. 이를 하나의 완료 표시로 합치면 평가 주장이 근거를 넘는다. |

판단을 가르는 단일 질문: **P2를 이 modified 프로토콜의 데이터 준비·추론 연결로 구현하되, 이미지 의미 대조 6/600, GPU/원본 환경/전체 모델 평가의 별도 상태와 한계를 유지하는 범위로 승인하는가?**

합성: 인터페이스 구현 진행에는 동의할 수 있으나 실제 benchmark 준비 완료라는 더 넓은 주장은 현재 근거로 승인할 수 없다. 이 차이를 없애기 위해 pass 수를 평균하거나 미실행 결과를 통과로 처리하지 않는다. 최종 commitment에 그대로 남겨 사용자에게 묻는다.

## mandela

검증 대상의 구성은 VSP 원본 PNG/좌표 metadata(데이터), 기존 ILVR과 새 bridge(모델 경로), P1 simulator/scorer(채점), 이 연결 설계와 fixture(설계자)다.

발화하는 패턴은 **Shared hallucination (#3), 근거 독립성 위험**이다. 같은 원본의 table/pure_text 일치를 그림 의미가 정확하다는 독립 검증으로 확대할 수 없다. 독립성 보완은 raw PNG 직접 대조이며 현재 6장까지 수행했다. 나머지는 미검증으로 남긴다. 합성 fixture 점수를 실제 모델 성능으로 취급하지 않는 규칙과 gold를 ModelInput에서 제외하는 경계는 유지한다.

이 audit도 root의 해석이며 source 전체의 무결성 인증이 아니다. 표와 원본 PNG를 직접 대조하면 같은 관찰을 확인할 수 있다. 별도 train/test 오염 조사는 하지 않았으며 그 부재를 clean 판정으로 바꾸지 않는다.

## 합성 결과와 승인 상태

동결 제안 v1의 동작·인터페이스·범위 변경은 없다. 최종 [design commitment v1](../superpowers/specs/2026-09-10-ilvr-vsp-inference-design.md)에 같은 규칙과 위 근거/이견/미검증 전제를 옮긴다.

입력 방향의 사전 승인 뒤, 최종 commitment와 그 한계를 제시한 질문에 사용자가 2026-09-10 `ㅇㅋ`로 응답했다. 이에 따라 설계 v1/risk waiver 승인과 계획 단계 진입을 기록했다. 비평이나 source inspection 자체를 구현 승인으로 해석하지 않는다.

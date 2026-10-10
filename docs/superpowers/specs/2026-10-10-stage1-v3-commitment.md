# Stage1-v3 — design commitment v1

상태: **서면 설계 최종 승인 전, first-nail 전제 실험의 별도 승인 대기**.
기준은 [동결한 design proposal v1](2026-10-10-stage1-v3-design.md) 전체다.
Git `690b1bc169f4c55e4fc0f82bc4373db452269c43`, SHA256
`07069c90f11297f7c5408ff280ac9a182b317b540fbd88095e53332dedb1d47a`.
세 독립 read가 모두 이 버전을 읽었으며 비판 사이에 proposal을 수정하지 않았다.

## 합의된 문제와 최종 설계 대상

기존 Stage1 안에 opt-in v3를 구현한다. 동일 helper target을 사용하여 선행 assistant text
접근을 제한한 R과 현재 latent body 접근까지 제한한 RV의 GT 예측 차이를 Gain으로 학습하고,
clean student F와 R의 Text-KL을 일반 optimizer loss에 더한다. direct attention-edge 제한의
효과를 측정하는 계약이며, 간접 정보까지 제거한 인과적 intervention이라는 주장은 하지 않는다.

사용자가 확인한 저장소 프로필과 A/B/C를 그대로 유지한다: CE0.1, frozen EMA1.0, 한 epoch
상당량, 2 nodes/4 ranks/global16; 현재 Stage1 entrypoint/coordinator; exact next-item span과
query별 mask; independent packed R/RV, shared-prefix gradient와 sample-stable dropout;
joint VJP를 통한 Text-KL gradient 전달; 성공한 ZeRO1 AdamW step 뒤 FP32 master에 Gain 추가;
완료 boundary의 checkpoint/resume와 실제 tail 기준 budget. 공식·계수·경계·오류·검증의
상세 승인 대상은 연결된 proposal의 3–9절이다.

범위는 위 알고리즘과 실행 계약을 위한 한정된 Stage1/Qwen 통합이다. 추가 framework나
학습 orchestration을 만들지 않는다. 원본 base로 새로 시작하며, 기존 목적함수와 saved resume의
의미를 보존하거나 미지원이면 실행 전에 거부한다. 기계적 정확성, 분산 호환성, 성능을 분리해
입증한다. 본 문서는 학습/평가 실행 또는 merge 승인이 아니다.

## 비판 결과 — divergence 우선

### divergent-incompatible: 정확성과 실행 횟수 최적화를 동시에 확정하는 순서

fresh read A/B/C가 제시한 공통 root는, 명시한 수학적 목적함수와 “추가 forward/세 번째 score
traversal 없음”을 함께 확정하면 gradient가 맞는 더 단순한 구현도 탈락할 수 있다는 점이다.
대안은 먼저 독립 계산으로 목적함수를 입증하고 실행 횟수 최적화를 별도 단계로 두는 것이다.
이 대안은 이미 확인받은 packed/joint-VJP 실행 계약을 바꾸므로 현 설계와 양립하지 않는
부분이다. 서로 같은 결론을 냈다는 사실을 기술적 입증이나 다수결로 사용하지 않는다.

검토별 근거:

- `/root/v3_fresh_read_a`: 원본 token 좌표와 두 block의 서로 다른 suffix 길이에서 gradient뿐
  아니라 dropout mask·invocation을 비교해야 한다. 단순한 독립 gradient 경로는 대안이며 자동 fallback이 아니다.
- `/root/v3_fresh_read_b`: 고정 forward/VJP budget은 학습 목적의 필연적 결과가 아닌 구현 제약이다.
  sample/step/site seed와 reference native RNG 방식의 차이는 별도 bridge 증거를 요구한다.
- `/root/v3_fresh_read_c`: 현재 context는 한 site의 original/recompute 중복 호출을 거부한다.
  RNG 값만 맞아도 lifecycle이 깨지면 충분한 검증이 아니다.

Root가 재확인한 source 근거: 현재 `src/stage1/equivalence.py:702`의 invocation 검증,
`:847`의 `SampleStableStage1Dropout(nn.Module)`, `:884` 이후 per-sample generator;
reference `stage1_vision_gain_packed.py:205–206,238–240`의 `nn.Dropout` 선택 및
`vision_gain_packed_execution.py:102`의 checkpoint용 dropout 선택.
이 차이는 직접 이식할 수 없다는 정적 근거다. bridge가 불가능하다는 runtime 증거는 없다.

### divergent-compatible: 최적화 경로의 전제 실험을 먼저 승인받기

세 read 모두 현재 목표를 유지한다면 실제 dropout/checkpoint machinery와 작은 독립 oracle을
사용한 bounded probe가 다음 증거로 유용하다고 제안했다. 이 순서는 proposal 10절과 양립한다.
CPU probe가 통과하더라도 실제 ZeRO1 master mutation, partial tail, save/resume의 네-rank
runtime witness가 따로 필요하다는 한계도 유지한다.

### convergent 요소: 통합 경계와 성능 주장의 분리

기존 실행 경로의 opt-in 통합, 저장소 계수, legacy compatibility, 역사적 정확도와 구현 검증의
분리에는 독립 read들이 가까운 방향을 제시했다. 이는 관점 점검에서의 reassurance일 뿐,
동등성·실행 가능성·성능의 증거가 아니다. 각 read의 주 분류는 위 divergent-incompatible이며
이 항목은 그 read 안의 부수적 일치다.

## hate 결과

```json
{
  "root": "이 설계는 sample-stable dropout과 checkpointing 아래에서 packed alias/shared-prefix/joint-VJP가 독립 계산과 같은 실제 학습 gradient를 보존할 수 있다는 미검증 전제에 의존한다. 참고 구현의 nn.Dropout/native RNG 계약은 현재 SampleStableStage1Dropout 계약과 다르므로, 이를 해결하지 못하면 빠른 실행 경로가 의도한 목적함수의 구현이 될 수 없다.",
  "first_nail": "별도 승인 후 CPU의 작은 2-layer causal model과 현행 SampleStableStage1Dropout을 사용해, 원본 token별 mask를 고정한 독립 full computation과 최소 packed/shared-prefix/joint-VJP bridge를 비교한다. 2개 latent block과 서로 다른 suffix 길이, dropout0/0.05, checkpoint on/off에서 R/RV score·Text-KL·prefix를 포함한 모든 trainable gradient·원본/recompute dropout mask와 호출 규칙을 비교한다. FP64 reference와 FP32 실행의 사전 고정 허용오차를 넘거나 호출 계약이 깨지면 해당 architecture premise를 기각하고 설계를 다시 연다. 생산 코드 변경, optimizer step, checkpoint 저장, GPU/분산 학습 없이 수행하는 throwaway 전제 실험이며 현재는 미실행이다."
}
```

상태: **제안만 했으며 실험은 실행하지 않았다.** source 차이와 기존 baseline test는 이
전제를 검증하지 않는다. 참고 성능이 다른 학습 프로필에서도 유지된다는 증거도 없다.
기반 branch의 partial accumulation resume 실패는 별도로 남아 있고, proposal의 scoped
수정·분산 검증 범위에 포함된다.

## Superpowers 종합과 proposal 대비 변화

공식·계수·통합 방식·호출 예산·사용자 확인 범위는 바꾸지 않는다. 비판을 반영하여 다음
결정 대상을 전체 구현이 아니라 **해당 실행 계약의 실현 가능성을 묻는 제한된 CPU probe**로
구체화한다. 이는 proposal의 별도 전제 실험 gate를 실행하는 절차다.

해결되지 않은 이견은 “호출 예산을 먼저 완화할지”이다. 현 계약을 유지한 채 증거를 구하는
쪽을 권고한다. probe가 실패하면 원인을 보고하고 설계를 다시 연다. 호출 예산, dropout,
mask 또는 loss를 바꾸는 것은 새 proposal/commitment 버전과 재승인이 필요한 변경이다.
probe 실패는 모든 가능한 bridge의 수학적 불가능성을 증명하지 않는다.

## 별도 승인 요청 대상: first-nail CPU probe

질문: 현행 sample-stable dropout/checkpoint 호출 규칙을 유지하면서 packed/shared-prefix/
joint-VJP가 독립 계산의 R/RV·Text-KL gradient를 재현할 수 있는가?

제안 범위:

- CPU의 작은 two-layer causal model, 고정 weights/seed/선택 mask, 두 latent block과 서로 다른
  suffix 길이, 중복·재배열된 원본 token 좌표. 실제 저장소 dropout/context를 연결한다.
- 독립 full computation과 throwaway 최소 bridge를 비교한다. dropout0/0.05,
  checkpoint on/off에서 score·KL·전체 gradient·prefix derivative·projection direction을 확인한다.
- 원본 token별 mask는 정확히 일치해야 한다. invocation은 각 경로의 사전 정의한 logical
  context별 original/recompute 횟수·순서와 기존 검증 규칙을 만족해야 한다. 독립 경로와
  packed 경로의 전체 호출 수가 같아야 한다는 뜻은 아니다. 수치 비교는 결과를 보기 전에
  FP64 `atol=1e-9, rtol=1e-7`, FP32 `atol=1e-6, rtol=1e-4`로 고정한다.
  실패를 통과시키기 위해 실행 후 허용오차를 넓히지 않는다.
- production module 수정, optimizer step, 모델/optimizer checkpoint 저장, GPU/분산 training은
  포함하지 않는다. throwaway 코드는 별도 skill scratch, 실행 로그는 `logs/`, 수치 결과는
  `results/`에 둔다. 독립 검토자가 oracle·관측값·주장 범위를 검토해야 한다.
- 최종 산출물은 일치/불일치와 원인, 실제 실행 명령·결과·검토 증거다. 재사용할 제품 구현으로
  전환하는 권한은 포함하지 않는다. 실패 시 임의 수정 반복이나 대규모 통합으로 확장하지 않는다.

이 승인 요청은 **probe의 범위**에 대한 것이다. 사용자에게 선택받은 구현 방식이 아직 없으므로,
probe 코드 작성 전 그 방식을 명시적으로 선택받아야 한다. 추천할 수 있는 가장 작은 경로는
root의 native 실행과 별도 reviewer 검토다. probe 승인과 그 방식 선택을 한 응답에서 할 수 있으나,
해당 응답을 written-spec 최종 승인이나 제품 구현 계획 승인으로도 중복 해석하지 않는다.

probe를 실행하지 않고 설계를 확정하려면, 위 gradient/dropout 실행 가능성 불확실성을 감수한다는
명시적 risk waiver와 정확한 commitment/spec 승인 두 의사를 받아야 한다. 그 경우에도 이후
필수 구현 검증이 면제되지 않는다. 현재는 probe 승인·방식 선택·risk waiver 모두 없다.

## 이후 승인 경계

전제 실험 결과 또는 명시적 risk waiver를 commitment에 반영하고 그 정확한 버전을 최종 서면
설계 승인 대상으로 제시한다. 그 승인 후에만 `superpowers:writing-plans`를 실행한다.
구현 계획 검토와 사용자 방식 선택, TDD·독립 review·fresh verification·해당 branch finish는
그 다음 단계다. A/B/C의 `Ok`와 이번 문서 작성 승인은 이 gate들을 대체하지 않는다.
